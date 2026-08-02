"""Voice router — single LLM call that gates + classifies the
always-listening mic.

Replaces the address-classifier stack (regex Tier 1 + LLM Tier 3) on
the always-listening path with one structured call that answers four
questions at once:

  * **coherent** — is the transcript comprehensible English or STT
    garbage? Cuts the noisy floor before any downstream work.
  * **addressed** — is the user speaking TO the companion, or to
    someone else / themselves / reading aloud / background?
  * **goal** — if addressed, what should the companion DO?
    ``act`` (primitive), ``converse`` (chat reply), ``clarify``
    (follow-up question needed), ``idle`` (silent ack), ``drop``
    (noise).
  * **confidence** — 0.0-1.0 for the addressed judgment, used by the
    confidence-tier dispatch downstream.

Why this replaces the regex
---------------------------
The old Tier 1 was a verb list (``play``, ``find``, ``show``…). Every
new vocabulary variant the user tries ("throw on some jazz", "fire up
the playlist", "spin up a podcast") was a regex miss that fell to a
broken LLM tier. The maintenance pattern was "patch the verb list when
something fails," which doesn't generalize beyond the one user being
tested.

The action-verb regex still exists as a CONFIDENCE BOOSTER (see
``_action_verb_score``), not as a routing decision. If a known
primitive verb appears, we raise the confidence floor for
``goal=act`` to bias toward dispatching. Vocabulary the regex doesn't
know about still routes correctly because the LLM has the actual
decision.

Renameability
-------------
The companion name comes from ``settings.companion_name`` and is
substituted as ``<COMPANION_NAME>`` in the prompt at build-time. The
code path never hard-codes ``Becca``; rename the companion and this
classifier keeps working.

PTT and wake-word are untouched — those modes auto-address every
utterance after the gate fires.
"""

from __future__ import annotations

import asyncio
import json
import re
import time as _time
from dataclasses import dataclass
from typing import Any, Literal

from augmentum.config import settings
from augmentum.models.base import InternalChatRequest, Message
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


VoiceGoal = Literal["act", "converse", "clarify", "idle", "drop"]
_GOALS: frozenset[str] = frozenset(("act", "converse", "clarify", "idle", "drop"))


@dataclass(frozen=True)
class VoiceRouterDecision:
    """Structured result of one voice-router classification call.

    ``raw_json`` is the parsed model output so callers can log /
    inspect the reasoning during tuning. ``latency_ms`` is end-to-end
    wall time including backend resolution.
    """

    coherent: bool
    addressed: bool
    confidence: float
    goal: VoiceGoal
    reasoning: str
    latency_ms: int
    model: str
    parsed_from: str  # "content" | "thinking" | "fallback" | "error"


# Action-verb stems used by the confidence booster. These mirror the
# verb shape the architect's registered primitives accept (see
# ``augmentum/intent/builtin/`` and ``augmentum/architect/primitives/
# ``). Used as a *signal*, not a gate — when this fires AND goal=act,
# we boost confidence so the dispatcher is more inclined to act.
#
# Deliberately short. The point is "did the user use a verb the
# companion has a primitive for", not "does this look addressed."
# The LLM handles the addressing question.
_ACTION_VERB_STEMS = (
    "play", "pause", "stop", "resume", "skip", "next", "previous",
    "open", "show", "find", "search", "set", "create", "make",
    "generat", "draw", "remind", "save", "remember", "send",
    "queue", "start", "cancel", "read", "list",
)
_ACTION_VERB_RE = re.compile(
    rf"\b(?:{'|'.join(_ACTION_VERB_STEMS)})(?:e|es|ed|ing|s)?\b",
    re.IGNORECASE,
)

# Sentinel used in the prompt; resolved to the live companion name
# from settings at build time.
_NAME_PLACEHOLDER = "<COMPANION_NAME>"


# JSON Schema for the verdict. Attached to the classifier request via
# ``raw_options["json_schema"]`` so the backend constrains generation to
# this exact shape (llama-server / OpenAI ``response_format`` json_schema;
# the local engine forwards the same key through its passthrough
# allowlist). This is what makes a tiny non-reasoning model viable on this
# hop: it CANNOT emit prose, chain-of-thought, or a truncated object — the
# four decision fields are forced, in order, and generation stops there.
# Kills the ``parse_fallback`` / think-only failure class at the source
# rather than salvaging it downstream.
_VOICE_ROUTER_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "coherent": {"type": "boolean"},
        "addressed": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "goal": {"type": "string", "enum": sorted(_GOALS)},
        # No free-text ``reasoning`` field on purpose. It was telemetry-only
        # (emitted LAST, after the decision — so it never informed the
        # verdict) but a variable-length string is the one part that can
        # balloon generation on a CPU-served tiny model. Dropping it forces
        # the json_schema grammar to stop right after ``goal`` (~25 tokens),
        # keeping the hop inside the 2.5s budget. ``_normalize_decision``
        # already defaults reasoning to "" when absent.
    },
    "required": ["coherent", "addressed", "confidence", "goal"],
    "additionalProperties": False,
}


def _action_verb_score(text: str) -> float:
    """Cheap signal: does the utterance contain a known action verb?

    Returns 0.0 or 0.10 — small enough to never override the LLM's
    own confidence, large enough to nudge borderline cases over the
    dispatch threshold.
    """
    return 0.10 if _ACTION_VERB_RE.search(text or "") else 0.0


def _build_system_prompt(companion_name: str) -> str:
    """System prompt with the companion name substituted in.

    Kept short — small models lose accuracy when system prompts run
    long. The four questions are numbered so the model has a clear
    schema to fill in, and the JSON contract is shown explicitly with
    a worked example.
    """
    name = (companion_name or "the assistant").strip() or "the assistant"
    return (
        f"You classify a transcribed utterance from a voice AI's always-listening mic. "
        f"The assistant is named {name}.\n\n"
        "Answer four questions in strict JSON:\n\n"
        f"1. coherent: Is the utterance comprehensible English (not transcription garbage)?\n"
        f"2. addressed: Is the user speaking TO {name}, or to someone else / themselves / "
        "reading aloud / background?\n"
        f"3. goal: If addressed, what should {name} do?\n"
        "   - \"act\": explicit task (play, find, set, open, search, control a device)\n"
        "   - \"converse\": greeting, question, or any conversational exchange that "
        "warrants a spoken reply (\"good morning\", \"hey, you there?\", \"what do you think?\")\n"
        "   - \"clarify\": ambiguous request that would need a follow-up question\n"
        "   - \"idle\": bare acknowledgment of something just completed that needs no "
        "reply (\"thanks\", \"got it\", \"okay cool\")\n"
        "   - \"drop\": noise / not actionable / not for the assistant\n"
        f"   CONTINUATION RULE: when {name}'s last response asked the user a question "
        "and the utterance answers it, the utterance IS addressed and the goal "
        f"carries the pending request forward. \"I'll let you choose\" / \"the second "
        f"one\" / \"yeah do that\" after {name} offered options = \"act\", never "
        "\"idle\" — the user is delegating or selecting, not closing the topic.\n"
        f"   REPLY RULE: a coherent reply to something {name} just said or did — "
        "including dismissals, corrections, and commentary on what's playing or on "
        "screen (\"no\", \"not that one\", \"never mind\", \"this one's better\") — "
        "IS addressed; the goal is \"converse\" (or \"act\" when it redirects the "
        f"task). Reserve \"drop\" for speech aimed at another person, self-talk, "
        f"reading aloud, {name}'s own voice echoed back, or noise — never for "
        f"something said TO {name}.\n"
        "4. confidence: 0.0-1.0 for the addressed judgment. High when clearly directed at "
        f"{name}; low when ambiguous.\n\n"
        "Reply with ONLY this JSON object, no other text:\n"
        "{\"coherent\": true, \"addressed\": true, \"confidence\": 0.85, "
        "\"goal\": \"act\"}"
    )


def _build_user_prompt(
    utterance: str,
    *,
    companion_name: str,
    last_assistant_response: str = "",
    last_dispatch_summary: str = "",
    seconds_since_last_tts: float | None = None,
    active_surface: str = "",
) -> str:
    """Compose the per-call user prompt with context.

    Only the utterance is required; the other fields enrich the
    decision (continuation detection, anaphora resolution, surface
    context) but the classifier degrades gracefully when they're
    missing.
    """
    name = (companion_name or "the assistant").strip() or "the assistant"
    parts: list[str] = [f"Utterance: {utterance.strip()!r}"]
    if last_assistant_response:
        parts.append(f"{name}'s last response: {last_assistant_response[:200]!r}")
    if last_dispatch_summary:
        parts.append(f"Last action taken: {last_dispatch_summary[:160]}")
    if seconds_since_last_tts is not None and seconds_since_last_tts < 60:
        parts.append(
            f"Seconds since {name} spoke: {seconds_since_last_tts:.0f}"
        )
    if active_surface:
        parts.append(f"User is currently on surface: {active_surface}")
    parts.append("\nReply with ONLY the JSON object.")
    return "\n".join(parts)


def _strip_code_fence(text: str) -> str:
    """Pull JSON out of ``` fences if the model wrapped its output."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t[3:]
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _parse_decision_json(raw: str) -> dict | None:
    """Best-effort JSON extraction; finds the first object in the text.

    Models occasionally pad with a sentence before / after the JSON
    blob. We slice from the first ``{`` to the matching ``}`` and try
    to parse that. Returns None if no object can be recovered.
    """
    if not raw:
        return None
    body = _strip_code_fence(raw)
    start = body.find("{")
    if start < 0:
        return None
    # Greedy: take to the LAST } in the response — handles trailing
    # commentary after the JSON but breaks if model emits a stray }
    # mid-string. Acceptable trade-off; the prompt explicitly says
    # "JSON only".
    end = body.rfind("}")
    if end < start:
        return None
    try:
        parsed = json.loads(body[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        return None
    return None


def _normalize_decision(
    parsed: dict, *, text: str, parsed_from: str, model: str, latency_ms: int,
) -> VoiceRouterDecision:
    """Coerce the model's raw fields into a typed decision.

    The model can return surprising shapes — strings instead of bools,
    "0.85" instead of 0.85, "Act" instead of "act". We coerce safely
    and fall back to conservative defaults (addressed=False, goal=drop)
    when a field is missing or invalid. False silence is cheaper than
    false speech.
    """
    coherent = bool(parsed.get("coherent", True))

    addressed_raw = parsed.get("addressed", False)
    if isinstance(addressed_raw, str):
        addressed = addressed_raw.strip().lower() in ("true", "yes", "1")
    else:
        addressed = bool(addressed_raw)

    confidence_raw = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    goal_raw = str(parsed.get("goal", "drop") or "drop").strip().lower()
    goal: VoiceGoal = goal_raw if goal_raw in _GOALS else "drop"  # type: ignore[assignment]

    reasoning = str(parsed.get("reasoning", "") or "")[:200]

    # Contradiction guard: "drop" is defined as "not for the assistant",
    # which cannot coexist with addressed=True. Observed live 2026-06-11:
    # "No fire." → addressed=True coherent=True conf=0.9 goal=drop — a
    # dismissal said TO her, silently killed at the gate. When the model
    # contradicts itself, the addressed judgment wins: a coherent
    # utterance directed at her warrants at least a light reply.
    if goal == "drop" and addressed and coherent:
        log.info(
            "voice_router_drop_contradiction_promoted",
            text_preview=text[:80],
            confidence=round(confidence, 2),
        )
        goal = "converse"

    # Confidence booster: if the LLM said act AND a known action verb
    # appears, raise the floor. Doesn't override a confident LLM call
    # (already >= 0.10 boost ceiling); helps borderline cases.
    if goal == "act" and addressed:
        confidence = min(1.0, confidence + _action_verb_score(text))

    return VoiceRouterDecision(
        coherent=coherent,
        addressed=addressed,
        confidence=confidence,
        goal=goal,
        reasoning=reasoning,
        latency_ms=latency_ms,
        model=model,
        parsed_from=parsed_from,
    )


def _regex_fallback(
    text: str, model: str, latency_ms: int, parsed_from: str,
) -> VoiceRouterDecision:
    """Salvage a decision from the legacy regex when the LLM can't answer.

    Used on timeout / backend error / parse failure. The regex has
    vocabulary gaps (which is why we replaced it as the primary path)
    but it's vastly better than silent drop — it still catches the
    common imperative + WH-question shapes that make up most of the
    real addressed traffic.

    Mapping legacy ``AddressDecision`` → router shape:
      * ``imperative_start`` / ``direct_request`` → goal=act
      * ``second_person_question`` / ``wh_question_*`` → goal=converse
      * ``continuation`` → goal=act (resolves to the prior dispatch)
      * ``self_talk`` / ``third_person`` → goal=drop, addressed=False
      * ``no_signal`` + ``timeout_fallback`` → goal=converse, addressed=True

    The last rule is the load-bearing one. On a timeout (not a hard
    error) the LLM was actively processing — strong evidence the
    transcript was coherent enough to engage a model. If the regex
    then returns no_signal (its catch-all for "didn't recognize any
    canonical structure"), defaulting to silent-drop is the worst
    possible UX: user spoke clearly, our infra was just too slow, and
    they get nothing back. Instead we lean into a conversational
    reply — false speech of "I didn't quite catch that, can you say
    it again?" is vastly cheaper than total silence. Strong-ambient
    signals (self_talk / third_person) still drop normally.
    """
    # Import lazily — the fallback only runs on the error path.
    from augmentum.architect.address import is_addressed
    legacy = is_addressed(text)
    goal: VoiceGoal

    if legacy.addressed:
        if legacy.signal in ("imperative_start", "direct_request",
                              "continuation", "self_talk_with_delegation"):
            goal = "act"
        else:
            goal = "converse"
        return VoiceRouterDecision(
            coherent=True,
            addressed=True,
            confidence=legacy.confidence,
            goal=goal,
            reasoning=f"regex_fallback:{legacy.signal}",
            latency_ms=latency_ms,
            model=model,
            parsed_from=parsed_from,
        )

    # Strong-ambient drop — but PRECISION-TIERED, because the fallback's
    # mandate is the opposite of the general classifier's. When the LLM
    # has failed, a MISS (silently eating a real turn → "she ignored
    # me") costs far more than a false-accept ("sorry, say that again?").
    # So invert the burden of proof: drop only on a HIGH-PRECISION
    # negative; lean addressed on anything the regex guesses weakly.
    #
    #   empty        — no speech. Perfect precision. Drop.
    #   third_person — start-anchored narration ("he told her to…"). The
    #                  one negative this regex gets reliably right. Drop.
    #   self_talk    — LOW precision: the declarative prefix ("It was…",
    #                  "I think…", "that's…") fires on genuine musing AND
    #                  on real continuations/answers to her own question,
    #                  and cannot tell them apart. On a timeout/parse
    #                  failure (LLM was actively engaged on a coherent
    #                  transcript) that weak guess must NOT drop the turn
    #                  — "It was about the narrative mode chat" was eaten
    #                  exactly this way (2026-06-13). Fall through to the
    #                  generous branch below.
    _release_on_engaged = (
        legacy.signal == "self_talk"
        and parsed_from in ("timeout_fallback", "parse_fallback", "error_fallback")
    )
    if legacy.signal in ("self_talk", "third_person", "empty") and not _release_on_engaged:
        return VoiceRouterDecision(
            coherent=True,
            addressed=False,
            confidence=legacy.confidence,
            goal="drop",
            reasoning=f"regex_fallback:{legacy.signal}",
            latency_ms=latency_ms,
            model=model,
            parsed_from=parsed_from,
        )
    if _release_on_engaged:
        log.info(
            "voice_router_fallback_released_self_talk",
            parsed_from=parsed_from,
            note="low-precision self_talk on an engaged-LLM failure — not dropping",
        )

    # no_signal on a SALVAGEABLE failure (timeout / parse / error). All three
    # share one diagnostic posture: the backend was REACHABLE (we passed model
    # resolution and got here from an actively-attempted chat call), STT decoded
    # a coherent transcript, and only the regex — not the human — failed to
    # recognize one of its canonical imperative / question / self-talk shapes.
    # False silence under those conditions is the worst possible UX, so we
    # ENGAGE: lean conversational and let the reply path run. If the backend is
    # genuinely unhealthy, that path fails with its OWN honest error — it never
    # blames the user and never goes silent. We fail FORWARD.
    #
    #   - timeout_fallback: model spent the whole budget and we cut it off.
    #   - parse_fallback: an asymmetric-thinking model (DeepSeek V4.x) ate the
    #     budget thinking and emitted truncated JSON, or a small model returned
    #     a malformed shape — a failure at the LLM<->JSON contract, not human<->mic.
    #   - error_fallback: a transient exception during the router's chat() on an
    #     ALREADY-RESOLVED backend. Used to DROP here — the lone silent-on-
    #     coherent path, now folded in (Matt 2026-07-27, silent-gates invariant).
    if parsed_from in ("timeout_fallback", "parse_fallback", "error_fallback"):
        return VoiceRouterDecision(
            coherent=True,
            addressed=True,
            confidence=0.60,
            goal="converse",
            reasoning=f"regex_fallback:no_signal_{parsed_from}_lean_addressed",
            latency_ms=latency_ms,
            model=model,
            parsed_from=parsed_from,
        )

    # Defensive: an UNEXPECTED parsed_from reached the salvage path. Genuinely
    # unreachable-backend states (resolve_failed / no_backend) resolve to
    # _conservative_drop upstream and never arrive here — so if we're here with
    # something else, we have no evidence the backend can serve a reply. Drop
    # rather than route to a conversational path that would just fail too.
    return VoiceRouterDecision(
        coherent=True,
        addressed=False,
        confidence=legacy.confidence,
        goal="drop",
        reasoning=f"regex_fallback:{legacy.signal}",
        latency_ms=latency_ms,
        model=model,
        parsed_from=parsed_from,
    )


def _conservative_drop(model: str, latency_ms: int, parsed_from: str) -> VoiceRouterDecision:
    """Default decision for the empty/disabled paths. False silence by design.

    Distinct from ``_regex_fallback`` which tries to salvage a verdict
    — this is used only when there's literally no input or the tier
    is gated off, so dropping is the right answer.
    """
    return VoiceRouterDecision(
        coherent=True,
        addressed=False,
        confidence=0.0,
        goal="drop",
        reasoning="",
        latency_ms=latency_ms,
        model=model,
        parsed_from=parsed_from,
    )


async def classify_voice(
    utterance: str,
    *,
    app_state: Any,
    user_id: str = "",
    session_id: str = "",
    last_assistant_response: str = "",
    last_dispatch_summary: str = "",
    seconds_since_last_tts: float | None = None,
    active_surface: str = "",
) -> VoiceRouterDecision:
    """Run the structured voice-router classification.

    Defaults to the user's currently active chat model — same backend
    resolution as the existing ``classify_with_llm`` tier. A latency-
    sensitive deployment can override via the
    ``companion_address_llm_model`` setting (URL or model id) and
    point at a small dedicated classifier.

    Returns a conservative drop decision (addressed=False, goal=drop)
    on any backend error, timeout, or parse failure. False silence is
    cheaper than false speech.
    """
    started_at = _time.monotonic()

    if not utterance or not utterance.strip():
        return _conservative_drop("", 0, "empty")

    if not getattr(settings, "companion_address_llm_enabled", True):
        # Whole tier is disabled — caller will fall back to whatever
        # else the path supports (the architect can still try its own
        # match against the transcript).
        return _conservative_drop("", 0, "disabled")

    registry = getattr(app_state, "provider_registry", None)
    if registry is None:
        return _conservative_drop("", 0, "no_registry")

    override = (getattr(settings, "companion_address_llm_model", "") or "").strip()
    companion_name = (getattr(settings, "companion_name", "") or "").strip() or "Becca"
    timeout_s = max(
        0.1,
        float(getattr(settings, "companion_address_llm_timeout_ms", 2000)) / 1000.0,
    )

    # Resolution chain — explicit override > classifier_model >
    # utility_model > primary_chat_model > default. Routing the call
    # through the role resolver means a dedicated classifier model
    # (when configured) shields the voice path from whatever heavy
    # reasoning model the user has selected for chat. Empty values
    # at each tier fall through, so the legacy "use chat model"
    # behavior is preserved when nothing is configured.
    try:
        backend, resolved_model = await registry.resolve_model_for_role(
            "classifier",
            override=override,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001 — degrade to drop
        log.warning("voice_router_resolve_failed", error=str(exc)[:160])
        return _conservative_drop(override, 0, "resolve_failed")

    if backend is None:
        return _conservative_drop(override, 0, "no_backend")

    # Surface which model is doing classifier-tier work — the timeout
    # default assumes a small fabric-hosted model. If this logs the
    # primary chat model, the user hasn't pointed classifier_model at
    # a real classifier and every voice turn will hit timeout fallback.
    # Logged at info so it shows up in default ops; the "fell_back_to_primary"
    # bool makes it scriptable.
    _classifier_setting = (getattr(settings, "classifier_model", "") or "").strip()
    _primary_setting = (getattr(settings, "primary_chat_model", "") or "").strip()
    _fell_back_to_primary = (
        not _classifier_setting
        and resolved_model == _primary_setting
    )
    log.info(
        "voice_router_classifier_resolved",
        resolved_model=resolved_model,
        configured_classifier=_classifier_setting or "(unset)",
        fell_back_to_primary=_fell_back_to_primary,
        timeout_ms=int(timeout_s * 1000),
    )

    system_prompt = _build_system_prompt(companion_name)
    user_prompt = _build_user_prompt(
        utterance,
        companion_name=companion_name,
        last_assistant_response=last_assistant_response,
        last_dispatch_summary=last_dispatch_summary,
        seconds_since_last_tts=seconds_since_last_tts,
        active_surface=active_surface,
    )

    # Classifier sampling. Default greedy (temp 0) for SmolLM/Qwen2.5; the
    # Gemma 4 E2B GPU option needs temp=1.0/top_p=0.95/top_k=64 or it
    # degenerates. top_p=1.0 / top_k=0 mean "off" → sent as None so the
    # backend omits them.
    _cs_temp = float(getattr(settings, "classifier_sampling_temperature", 0.0) or 0.0)
    _cs_top_p = float(getattr(settings, "classifier_sampling_top_p", 1.0) or 1.0)
    _cs_top_k = int(getattr(settings, "classifier_sampling_top_k", 0) or 0)

    req = InternalChatRequest(
        model=resolved_model or override or "",
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ],
        stream=False,
        temperature=_cs_temp,
        top_p=(_cs_top_p if _cs_top_p < 1.0 else None),
        top_k=(_cs_top_k if _cs_top_k > 0 else None),
        # No-thinking on this hop. The classifier is a single-token
        # JSON shape; chain-of-thought burns the entire timeout budget
        # without improving accuracy. Honored by Qwen 3.x, GLM-4.x,
        # EXAONE 4.x, Nemotron via llama-server's chat_template_kwargs.
        # Asymmetric-thinking families (DeepSeek V3.2/V4, MiniMax M2.x)
        # silently ignore this — they think regardless, eat the budget,
        # then emit JSON in the content channel. Sized to absorb that.
        chat_template_kwargs={"enable_thinking": False},
        # Schema-constrain the output to the exact verdict shape. On a
        # backend that honors it (local engine, llama-server sidecar, or
        # OpenAI) the model CANNOT emit prose / chain-of-thought / a
        # truncated object — which is what makes a tiny non-reasoning
        # classifier (SmolLM-135M) usable here. Backends that ignore the
        # field fall back to the prompt's "JSON only" instruction, so this
        # is safe to always attach.
        raw_options={
            "json_schema": _VOICE_ROUTER_SCHEMA,
            "json_schema_name": "voice_router_verdict",
        },
        # 384 is empirically enough for: 512-char thinking trace + the
        # ~80-char JSON object. Original 128 was sized for the no-think
        # case and silently failed for asymmetric-thinking models — the
        # JSON was being emitted but truncated mid-string at "confidence:
        # 0." (see voice_router_parse_failed events in prod logs). With the
        # schema constraint a compliant backend needs far less, but we keep
        # the headroom so asymmetric-thinking models that ignore the schema
        # still complete their JSON.
        max_tokens=384,
    )

    try:
        resp = await asyncio.wait_for(backend.chat(req), timeout=timeout_s)
    except TimeoutError:
        elapsed = int((_time.monotonic() - started_at) * 1000)
        log.info("voice_router_timeout", ms=elapsed, model=resolved_model)
        # Salvage rather than silent-drop. The regex misses vocab drift
        # but a hit is infinitely better than nothing — the user-perceived
        # failure mode of "I asked something and Becca did nothing" is
        # the worst possible outcome.
        return _regex_fallback(utterance, resolved_model, elapsed, "timeout_fallback")
    except Exception as exc:  # noqa: BLE001 — degrade to regex fallback
        elapsed = int((_time.monotonic() - started_at) * 1000)
        log.warning(
            "voice_router_backend_error", ms=elapsed,
            model=resolved_model, error=str(exc)[:200],
        )
        return _regex_fallback(utterance, resolved_model, elapsed, "error_fallback")

    elapsed = int((_time.monotonic() - started_at) * 1000)
    message = getattr(resp, "message", None)
    content = (getattr(message, "content", "") or "").strip()
    thinking = (getattr(message, "thinking", "") or "").strip()

    parsed = _parse_decision_json(content)
    parsed_from = "content"
    # Reasoning models can run out of budget before emitting content;
    # the verdict often shows up at the end of the thinking trace as
    # a JSON object the model was about to emit. Try that as a
    # fallback. If content is non-empty but unparseable, we still try
    # thinking — sometimes the content channel ends mid-string.
    if parsed is None and thinking:
        parsed = _parse_decision_json(thinking)
        if parsed is not None:
            parsed_from = "thinking"

    if parsed is None:
        log.info(
            "voice_router_parse_failed",
            ms=elapsed, model=resolved_model,
            content_preview=content[:80], thinking_chars=len(thinking),
        )
        return _regex_fallback(utterance, resolved_model, elapsed, "parse_fallback")

    decision = _normalize_decision(
        parsed, text=utterance, parsed_from=parsed_from,
        model=resolved_model, latency_ms=elapsed,
    )

    log.info(
        "voice_router_decision",
        ms=elapsed,
        model=resolved_model,
        coherent=decision.coherent,
        addressed=decision.addressed,
        confidence=round(decision.confidence, 2),
        goal=decision.goal,
        parsed_from=parsed_from,
        text_preview=utterance[:80],
        reasoning=decision.reasoning[:120],
    )

    return decision
