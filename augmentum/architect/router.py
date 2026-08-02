"""Architect router — confidence-tier dispatch via LLM.

This module supersedes the dispatch tier of the architect described in
``docs/superpowers/specs/2026-05-28-companion-architect-design.md``. The
new design lives in
``docs/superpowers/specs/2026-05-28-confidence-tier-dispatch-design.md``.

The router receives an utterance plus a confidence stack (template hint
+ signals) and asks an LLM to decide:

  * ``intent_id`` — which architect primitive to invoke (or REJECT)
  * ``args``      — parsed args, overriding the template hint when the
                    LLM reads the transcript differently
  * ``tier``      — A (ack+act), B (confirm+act), C (yes/no), REJECT
  * ``response_text`` — the spoken acknowledgment, in the user's register
  * ``cancel_grace_ms`` — for Tier B, the staging window
  * ``reasoning`` — short, for telemetry

Phase 1 wiring honors only Tier A; Tier B/C decisions land in
``RouterDecision`` and are recorded for telemetry but acted on as Tier
A. This lets us bring the router online safely while the
staging/cancel-window UX is still being built (Phases 2-3).

Latency budget: ``architect_router_timeout_ms`` (default 2500ms). The
router returns a REJECT decision on timeout/error so the voice path can
fall back to the legacy dispatch.
"""

from __future__ import annotations

import asyncio
import json
import time as _time
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from augmentum.architect.dispatch import ArchitectResult
from augmentum.architect.inference import infer_args
from augmentum.config import settings
from augmentum.intent.action import ActionResult, IntentMatch, SessionContext
from augmentum.intent.dispatch import get_referent_cache
from augmentum.intent.registry import REGISTRY
from augmentum.models.base import InternalChatRequest, Message
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _voiced_result(result: ActionResult, response_text: str) -> ActionResult:
    """Choose the spoken line for a dispatched action.

    The router pre-composes ``response_text`` (tier-appropriate language:
    present-continuous for A, declarative-with-cancel for B, a question for
    C) BEFORE the handler runs, and normally voices it instead of the
    handler's default speak. But it must NOT override a handler that did not
    actuate the intent — a library miss, an empty recommend, a parked
    clarification (``ActionResult.fulfilled is False``). Voicing "Playing your
    favorites — say cancel if you'd rather not" over a miss is the companion
    lying: she confirms an action that never happened and nothing plays.

    On ``fulfilled`` the router's line wins; otherwise the handler's honest
    line is kept verbatim (and ``clarify``/``digest``/``surface_emit`` survive,
    which the old manual rebuild silently dropped).
    """
    if response_text and result.fulfilled:
        return replace(result, speak=response_text)
    return result


RouterTier = Literal["A", "B", "C", "REJECT"]


@dataclass(frozen=True)
class ConfidenceStack:
    """Inputs the router weighs when picking a tier.

    Each field is best-effort: callers populate what's available, leave
    the rest at defaults. The router prompt explains the semantics of
    each signal so the model can reason about them.
    """

    stt_confidence: float = 1.0           # 0.0-1.0; 1.0 when STT didn't report
    address_signal: str = ""              # Tier 1 classifier signal name
    address_confidence: float = 0.0       # 0.0-1.0
    speaker_verified: bool = False        # voiceprint match
    template_hint_id: str = ""            # matched primitive (if any)
    template_hint_args: dict[str, Any] = field(default_factory=dict)
    audio_tier_media: bool = False        # AudioBus reports active media
    audio_tier_speech_other: bool = False # AudioBus reports other-speaker
    last_dispatch_id: str = ""            # previous architect dispatch
    last_dispatch_args: dict[str, Any] = field(default_factory=dict)
    last_dispatch_age_s: float = 0.0      # seconds since last dispatch
    active_surface: str = ""              # surface the user is on
    # Parked intent — a clarify question is waiting for its answer.
    # When set, this utterance is FIRST evaluated as that answer: the
    # router fills the missing args from it and dispatches the parked
    # verb, rather than re-deriving the whole intent from scratch.
    pending_intent_id: str = ""
    pending_intent_args: dict[str, Any] = field(default_factory=dict)
    pending_intent_missing: list[str] = field(default_factory=list)
    pending_intent_question: str = ""
    # Offered picks — media.play's "offer" / media.recommend parked
    # candidate cards. When set, an utterance that accepts one (by
    # ordinal, title fragment, or bare assent toward the offer) should
    # resolve to media.play with that candidate's exact file_id —
    # closing the loop the dock UI promises ("Tap, or just tell her").
    # Shape per entry: {"file_id", "title", "subtitle"?, "kind"?}.
    offered_candidates: list[dict] = field(default_factory=list)
    # Which verb + arg an accepted pick resolves to. Empty = the legacy
    # media.play/file_id default (media/livetv/games offers). coder.delegate
    # sets ("coder.delegate", "workspace_id") so "the second one" builds in
    # the chosen workspace instead of trying to play it.
    offered_intent: str = ""
    offered_id_field: str = ""
    # Presence — what the user is engaged with right now (browse page,
    # playing media). Deixis resolution: "this page" means the open
    # page, not a search query for the literal words "this page".
    current_page_url: str = ""
    current_page_title: str = ""
    now_playing_label: str = ""


@dataclass(frozen=True)
class RouterDecision:
    """Router's structured verdict on an utterance.

    ``tier == "REJECT"`` means the router declined to dispatch — the
    voice path should treat this as ambient / fall back to the legacy
    dispatcher / surface a clarifying response, depending on context.
    """

    intent_id: str                # "" when tier == REJECT
    args: dict[str, Any]
    tier: RouterTier
    response_text: str            # what Becca says
    cancel_grace_ms: int          # Tier B only; 0 otherwise
    reasoning: str                # one-line, for telemetry
    confidence: float             # router's own 0.0-1.0 confidence
    latency_ms: int
    model: str


# ----------------------------------------------------------------------
# Prompt assembly
# ----------------------------------------------------------------------

_SYSTEM_PROMPT_HEADER = (
    "You are the dispatch router for Augmentum's companion ({{char}}). "
    "The user just spoke; you decide whether to invoke a primitive (and "
    "which), or to decline.\n"
    "\n"
    "Your response MUST be a single JSON object with EXACTLY these keys: "
    '{"intent_id", "args", "tier", "response_text", "cancel_grace_ms", '
    '"reasoning", "confidence"}. No prose around the JSON. Keep '
    '"reasoning" to 8 words or fewer — it is telemetry, not analysis; '
    "every extra token delays the user's answer.\n"
    "\n"
    "TIER POLICY — choose based on intent confidence × action stakes:\n"
    "  - \"A\" = high confidence + trivial/reversible action. The "
    "companion says the response_text and the primitive fires "
    "immediately.\n"
    "  - \"B\" = high confidence + disruptive/costly action OR medium "
    "confidence. The companion says the response_text and stages the "
    "action with a cancel_grace_ms window before commit.\n"
    "  - \"C\" = low confidence OR irrevocable/safety_critical action. "
    "The companion says the response_text as an explicit yes/no "
    "question; the primitive only fires if the user confirms in the "
    "next turn.\n"
    "  - \"REJECT\" = utterance is not a request (self-talk, ambient "
    "speech, question about the companion's state), the right primitive "
    "doesn't exist, or you cannot infer required args. Set intent_id to "
    "\"\" and explain in reasoning.\n"
    "\n"
    "Bias toward REJECT when uncertain. The user can always rephrase; "
    "they cannot undo a wrongly-fired irrevocable action.\n"
    "\n"
    "INFORMATION vs SCREEN ACTION (headless delivery, 2026-06-10): "
    "when the user wants INFORMATION spoken back (\"what's the latest "
    "news\", \"who is X\", \"tell me about Y\"), REJECT — the "
    "conversational layer gathers with its own tools and answers in "
    "words; opening a panel at the user is the wrong outcome. Dispatch "
    "a search/browse primitive ONLY when they want it on their screen "
    "(\"show me\", \"open\", \"pull up\", \"put it on screen\"). "
    "Playback, timers, notes, and navigation asks are screen actions — "
    "dispatch those normally.\n"
    "\n"
    "RESPONSE_TEXT RULES:\n"
    "  - Tier A: present continuous (\"Setting a 5 minute timer.\"). "
    "Do NOT use past tense (\"I set a timer\") — the action hasn't "
    "completed yet. Do NOT use future tense (\"I'll set a timer\") — "
    "it's firing now.\n"
    "  - Tier B: declarative + cancel hint (\"Playing jazz from your "
    "favorites — say cancel if you'd rather not.\"). Calm, not "
    "interrogative.\n"
    "  - Tier C: explicit yes/no question (\"Did you want me to send "
    "this to John? Say yes or cancel.\"). The repeat-back of the parsed "
    "intent is mandatory.\n"
    "  - REJECT: a brief clarifying response or empty string.\n"
    "  - Keep it short and TTS-friendly. No markdown.\n"
)


def _format_action_catalog() -> str:
    """Render the available primitives as the router's tool catalog.

    Each entry: id, summary, arg_schema keys, stakes class. The router
    needs enough to pick which primitive to fire AND what tier its
    stakes justify — not the full handler docstring.
    """
    lines: list[str] = ["AVAILABLE PRIMITIVES:"]
    for action in REGISTRY.all():
        # Architect-callable primitives reachable from Becca's surface.
        # Empty ``surfaces`` means available EVERYWHERE (registry
        # convention — see Action.surfaces_for), so it must be
        # included; the previous ``not action.surfaces`` skip hid
        # media.play / navigate.* / note.* from the router entirely.
        # Tier-0 control verbs (stop/pause-the-stream) opt out of LLM
        # exposure via fanout.tier3=False — that's the filter that
        # keeps them out, not the surface list.
        if action.surfaces and "becca" not in action.surfaces:
            continue
        if not action.fanout.tier3:
            continue
        args_summary = ", ".join((action.arg_schema or {}).keys())
        required = (
            f" req:{','.join(action.required_args)}"
            if action.required_args else ""
        )
        # First clause only, hard-capped — same trim the companion
        # roster applies. Full multi-sentence summaries pushed the
        # catalog to ~6.8KB ≈ 2400 prompt tokens ≈ 1.5s prefill, which
        # plus ~0.9s of JSON generation overran the 2500ms budget even
        # with thinking disabled (every dispatch timed out 2026-06-10).
        # Trimmed: ~2.5KB ≈ 1100 tokens ≈ 1.6s round-trip.
        short = (action.summary or "").split(".")[0].strip()[:90].rstrip()
        lines.append(
            f"  {action.id} | {action.stakes} | ({args_summary}){required} — {short}"
        )
    return "\n".join(lines)


def _format_signals(stack: ConfidenceStack) -> str:
    """Render the confidence stack as compact prompt context."""
    parts: list[str] = ["SIGNALS:"]
    parts.append(f"  stt_confidence: {stack.stt_confidence:.2f}")
    if stack.address_signal:
        parts.append(
            f"  address_classifier: {stack.address_signal} "
            f"(confidence {stack.address_confidence:.2f})"
        )
    parts.append(f"  speaker_verified: {stack.speaker_verified}")
    if stack.template_hint_id:
        parts.append(
            f"  template_hint: {stack.template_hint_id} "
            f"args={stack.template_hint_args!r}"
        )
        parts.append(
            "  (the template matched, but you may override args or "
            "reject if the transcript shape contradicts the match — "
            "e.g. past-tense or self-talk wrapping an imperative form)"
        )
    if stack.audio_tier_media:
        parts.append(
            "  audio_tier_media: TRUE (the mic is hearing playback — "
            "background speech may be triggering the matcher; require "
            "strong direct address to act)"
        )
    if stack.audio_tier_speech_other:
        parts.append(
            "  audio_tier_speech_other: TRUE (other speaker in the "
            "room; user may not be addressing Becca)"
        )
    if stack.last_dispatch_id:
        parts.append(
            f"  last_dispatch: {stack.last_dispatch_id} "
            f"({stack.last_dispatch_age_s:.0f}s ago) args={stack.last_dispatch_args!r}"
        )
    if stack.pending_intent_id:
        parts.append(
            f"  PENDING CLARIFICATION: {stack.pending_intent_id} is waiting "
            f"for {stack.pending_intent_missing!r}. The assistant just asked: "
            f"{stack.pending_intent_question!r}. If this utterance answers "
            f"that question, choose {stack.pending_intent_id}, keep the "
            f"already-known args {stack.pending_intent_args!r}, and fill the "
            f"missing arg(s) from the answer. Only abandon the pending "
            f"intent if the utterance is clearly a NEW unrelated request."
        )
    if stack.offered_candidates:
        # Generic accept resolution: the offering verb declares which action +
        # id-arg a pick resolves to. Empty falls back to the historical
        # media.play/file_id (media/livetv/games offers), so those are
        # unchanged; coder.delegate sets ("coder.delegate", "workspace_id").
        _intent = stack.offered_intent or "media.play"
        _id_field = stack.offered_id_field or "file_id"
        lines = []
        for i, c in enumerate(stack.offered_candidates[:4], start=1):
            title = str(c.get("title") or "?")[:70]
            sub = str(c.get("subtitle") or "")[:60]
            idv = str(c.get(_id_field) or c.get("file_id") or "")
            lines.append(
                f"    {i}. {title}"
                + (f" ({sub})" if sub else "")
                + f" [{_id_field}={idv}]"
            )
        parts.append(
            "  OFFERED PICKS (just shown as tappable cards):\n"
            + "\n".join(lines) + "\n"
            "    If this utterance ACCEPTS one — by number ('the second "
            "one'), by title fragment ('the vending machine one'), or "
            "bare assent when only one was offered ('yeah, go ahead') — "
            f"choose {_intent} with args {{\"{_id_field}\": \"<that exact "
            f"{_id_field}>\"}}. Declines ('nah', 'not now') are a REJECT, not "
            "a new request. Only ignore the offer when the utterance is "
            "clearly a NEW unrelated request."
        )
    if stack.current_page_url:
        parts.append(
            f"  CURRENT PAGE OPEN: {stack.current_page_title or ''} "
            f"<{stack.current_page_url}> — deictic references ('this "
            f"page', 'this article', 'what I'm reading') mean THIS. "
            f"Never turn the deictic words themselves into a search "
            f"query. If they're asking ABOUT its content and no "
            f"primitive fits, REJECT so the conversational path can "
            f"discuss it."
        )
    if stack.now_playing_label:
        parts.append(
            f"  NOW PLAYING: {stack.now_playing_label} — 'this song' / "
            f"'what's playing' refers to it."
        )
    if stack.active_surface:
        parts.append(f"  active_surface: {stack.active_surface}")
    return "\n".join(parts)


def _build_user_prompt(transcript: str, stack: ConfidenceStack) -> str:
    parts = [
        f"UTTERANCE: {transcript.strip()!r}",
        "",
        _format_signals(stack),
        "",
        _format_action_catalog(),
        "",
        "Reply with the JSON decision now.",
    ]
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Response parsing
# ----------------------------------------------------------------------

_DEFAULT_TIER_GRACE_MS = {
    "A": 0,
    "B": 2000,
    "C": 0,
    "REJECT": 0,
}


def _parse_json_decision(raw: str) -> dict[str, Any] | None:
    """Extract a JSON object from the model response.

    Models occasionally wrap output in code fences or add a "Here is
    the JSON:" prefix even when told not to. Strip those and parse.
    Returns None on unrecoverable parse failure — the caller treats
    that as a router REJECT with a parse-failed reason.
    """
    if not raw:
        return None
    text = raw.strip()
    # Strip code fences
    if text.startswith("```"):
        # Find the first newline after the fence, then the closing fence
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    # Find the first { and the matching last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def _coerce_tier(raw: Any) -> RouterTier:
    if not raw:
        return "REJECT"
    s = str(raw).strip().upper()
    if s in ("A", "B", "C", "REJECT"):
        return s  # type: ignore[return-value]
    # Common loose variants
    if s in ("ACK", "ACT", "TIER_A", "TIERA"):
        return "A"
    if s in ("CONFIRM", "STAGE", "TIER_B", "TIERB"):
        return "B"
    if s in ("YESNO", "EXPLICIT", "TIER_C", "TIERC"):
        return "C"
    return "REJECT"


def _decision_from_dict(
    d: dict[str, Any],
    *,
    elapsed_ms: int,
    model: str,
) -> RouterDecision:
    tier = _coerce_tier(d.get("tier"))
    intent_id = str(d.get("intent_id") or "").strip()
    if tier == "REJECT":
        intent_id = ""
    args_raw = d.get("args") or {}
    args = args_raw if isinstance(args_raw, dict) else {}
    response_text = str(d.get("response_text") or "").strip()
    grace = d.get("cancel_grace_ms")
    try:
        cancel_grace_ms = int(grace) if grace is not None else _DEFAULT_TIER_GRACE_MS[tier]
    except (TypeError, ValueError):
        cancel_grace_ms = _DEFAULT_TIER_GRACE_MS[tier]
    reasoning = str(d.get("reasoning") or "").strip()
    confidence_raw = d.get("confidence")
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw))) if confidence_raw is not None else 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    return RouterDecision(
        intent_id=intent_id,
        args=args,
        tier=tier,
        response_text=response_text,
        cancel_grace_ms=cancel_grace_ms,
        reasoning=reasoning,
        confidence=confidence,
        latency_ms=elapsed_ms,
        model=model,
    )


def _reject(reason: str, *, elapsed_ms: int = 0, model: str = "") -> RouterDecision:
    return RouterDecision(
        intent_id="",
        args={},
        tier="REJECT",
        response_text="",
        cancel_grace_ms=0,
        reasoning=reason,
        confidence=0.0,
        latency_ms=elapsed_ms,
        model=model,
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


async def route_utterance(
    transcript: str,
    *,
    app_state: Any,
    stack: ConfidenceStack,
    user_id: str = "",
    session_id: str = "",
) -> RouterDecision:
    """Run the architect router on a single utterance.

    Returns a ``RouterDecision``. The caller is responsible for honoring
    the tier (which in Phase 1 means: execute Tier A immediately,
    promote B/C to A with a telemetry warning, treat REJECT as a
    fallthrough to the legacy dispatcher).

    Soft-fails to REJECT on any backend / timeout / parse error so the
    voice path always has a defined outcome.
    """
    if not transcript or not transcript.strip():
        return _reject("empty_transcript")
    if not getattr(settings, "architect_router_enabled", False):
        return _reject("router_disabled")

    timeout_s = max(
        0.5,
        float(getattr(settings, "architect_router_timeout_ms", 2500)) / 1000.0,
    )

    registry = getattr(app_state, "provider_registry", None)
    if registry is None:
        return _reject("no_provider_registry")

    # Resolution chain — explicit architect_router_model override >
    # classifier_model > utility_model > primary_chat_model > default
    # (mirrors voice_router.py). Routing through the role resolver means a
    # dedicated small classifier shields this hot-path dispatch call from
    # whatever heavy reasoning model the user picked for chat. architect_
    # router_model has no UI control, so WITHOUT this an empty value
    # silently inherited the primary — e.g. a remote deepseek-v4-pro that
    # timed out (4-5s) on every single dispatch. The classifier tier is
    # exactly what voice_router uses, so the two routing calls now agree.
    override = (getattr(settings, "architect_router_model", "") or "").strip()
    started_at = _time.monotonic()

    try:
        backend, resolved_model = await registry.resolve_model_for_role(
            "classifier",
            override=override,
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001 — degrade to REJECT
        log.warning("architect_router_resolve_failed", error=str(exc)[:160])
        return _reject("resolve_failed")

    if backend is None:
        return _reject("no_backend", model=override)

    # Surface which model is doing router-tier work, and whether it fell
    # through to the primary chat model (the slow case the chain guards
    # against). Mirrors voice_router_classifier_resolved so ops can grep
    # both routing calls the same way.
    _classifier_setting = (getattr(settings, "classifier_model", "") or "").strip()
    _primary_setting = (getattr(settings, "primary_chat_model", "") or "").strip()
    log.info(
        "architect_router_resolved",
        resolved_model=resolved_model,
        configured_override=override or "(unset)",
        configured_classifier=_classifier_setting or "(unset)",
        fell_back_to_primary=(
            not override and not _classifier_setting
            and resolved_model == _primary_setting
        ),
    )

    user_prompt = _build_user_prompt(transcript, stack)

    # Resolve {{char}} in the system prompt with the running companion's
    # display name. In-memory read off runtime.identity; falls back to
    # "Becca" (the default identity) when the runtime isn't initialized.
    # This is hot-path code (every dispatch request); kept cheap.
    system_prompt_resolved = _SYSTEM_PROMPT_HEADER
    try:
        runtime = getattr(app_state, "companion_runtime", None)
        char_name = "Becca"
        if runtime is not None:
            identity = getattr(runtime, "identity", None)
            if identity is not None:
                char_name = (
                    getattr(identity, "display_name", "")
                    or getattr(identity, "companion_id", "")
                    or "Becca"
                )
        system_prompt_resolved = system_prompt_resolved.replace(
            "{{char}}", char_name,
        )
    except Exception:
        log.debug("architect_router_char_substitution_failed", exc_info=True)

    # Classifier sampling — shares the voice router's recipe (default greedy;
    # Gemma 4 E2B GPU option needs temp=1.0/top_p=0.95/top_k=64). top_p=1.0 /
    # top_k=0 mean "off" → None so the backend omits them.
    _cs_temp = float(getattr(settings, "classifier_sampling_temperature", 0.0) or 0.0)
    _cs_top_p = float(getattr(settings, "classifier_sampling_top_p", 1.0) or 1.0)
    _cs_top_k = int(getattr(settings, "classifier_sampling_top_k", 0) or 0)

    req = InternalChatRequest(
        model=resolved_model or override or "",
        messages=[
            Message(role="system", content=system_prompt_resolved),
            Message(role="user", content=user_prompt),
        ],
        stream=False,
        temperature=_cs_temp,
        top_p=(_cs_top_p if _cs_top_p < 1.0 else None),
        top_k=(_cs_top_k if _cs_top_k > 0 else None),
        # No-thinking on this hop — same contract as the voice address
        # classifier (voice_router.py). Without it, thinking-mode
        # families (Qwen 3.x) burn the ENTIRE timeout budget on
        # chain-of-thought before the JSON starts: observed 2026-06-10
        # as architect_router_timeout on every single dispatch, first
        # at 800ms and still at 2500ms. Honored by Qwen 3.x / GLM-4.x /
        # EXAONE 4.x via llama-server's chat_template_kwargs.
        chat_template_kwargs={"enable_thinking": False},
        # Sized for the JSON + a one-line reasoning field, with slack
        # for asymmetric-thinking families that ignore the kwarg.
        max_tokens=384,
    )

    try:
        resp = await asyncio.wait_for(backend.chat(req), timeout=timeout_s)
    except TimeoutError:
        elapsed = int((_time.monotonic() - started_at) * 1000)
        log.info("architect_router_timeout", ms=elapsed, model=resolved_model)
        return _reject("timeout", elapsed_ms=elapsed, model=resolved_model)
    except Exception as exc:  # noqa: BLE001 — backend errors degrade safely
        elapsed = int((_time.monotonic() - started_at) * 1000)
        log.warning(
            "architect_router_backend_error",
            ms=elapsed, model=resolved_model, error=str(exc)[:200],
        )
        return _reject("backend_error", elapsed_ms=elapsed, model=resolved_model)

    elapsed = int((_time.monotonic() - started_at) * 1000)
    raw_content = getattr(getattr(resp, "message", None), "content", "") or ""
    parsed = _parse_json_decision(raw_content)
    if parsed is None:
        log.warning(
            "architect_router_parse_failed",
            ms=elapsed, model=resolved_model, raw=raw_content[:200],
        )
        return _reject("parse_failed", elapsed_ms=elapsed, model=resolved_model)

    decision = _decision_from_dict(parsed, elapsed_ms=elapsed, model=resolved_model)

    # Sanity check: if the router named a primitive that isn't
    # registered, treat as REJECT. Better to fall through than to
    # surface-emit on a phantom channel.
    if decision.intent_id and REGISTRY.get(decision.intent_id) is None:
        log.warning(
            "architect_router_unknown_primitive",
            intent_id=decision.intent_id, model=resolved_model,
        )
        return _reject(
            f"unknown_primitive:{decision.intent_id}",
            elapsed_ms=elapsed, model=resolved_model,
        )

    # Telemetry — emitted for every router pass. Drives the Phase 4
    # analysis ("which template hints earned their keep") and the
    # eventual companion-as-game feedback loop (which tier choices
    # were silently force-promoted today). Includes the full confidence
    # stack summary so post-hoc tuning can correlate any signal with
    # the router's verdict.
    template_hit = bool(stack.template_hint_id)
    template_hint_used = (
        template_hit
        and decision.intent_id == stack.template_hint_id
    )
    log.info(
        "architect_router_decision",
        intent_id=decision.intent_id or "(reject)",
        tier=decision.tier,
        confidence=round(decision.confidence, 2),
        latency_ms=decision.latency_ms,
        model=decision.model,
        # Confidence stack summary
        address_signal=stack.address_signal or "(none)",
        address_confidence=round(stack.address_confidence, 2),
        speaker_verified=stack.speaker_verified,
        audio_tier_media=stack.audio_tier_media,
        audio_tier_speech_other=stack.audio_tier_speech_other,
        template_hint=stack.template_hint_id or "(none)",
        template_hint_used=template_hint_used,
        template_hit_but_rejected=(template_hit and decision.tier == "REJECT"),
        template_hit_but_overridden=(
            template_hit
            and decision.intent_id != ""
            and decision.intent_id != stack.template_hint_id
        ),
        last_dispatch=stack.last_dispatch_id or "(none)",
        last_dispatch_age_s=round(stack.last_dispatch_age_s, 1),
        active_surface=stack.active_surface or "(none)",
        # Decision payload
        reasoning=decision.reasoning[:160],
        response_preview=decision.response_text[:80],
    )

    return decision


# ----------------------------------------------------------------------
# Router → Action dispatch
# ----------------------------------------------------------------------


async def dispatch_router_decision(
    decision: RouterDecision,
    *,
    transcript: str,
    surface: str,
    session: SessionContext,
    app_state: Any,
) -> ArchitectResult | None:
    """Execute the router's decision via the existing handler pipeline.

    Mirrors ``dispatch.dispatch_architect_command`` for the Phase 1
    rollout: looks up the primitive, surface-filters, runs inference +
    translation, calls the handler, then overrides the handler's
    spoken line with the router-chosen ``response_text`` (which is
    register-matched and tier-aware where the handler's default may
    not be).

    Returns None when the decision can't be executed (REJECT, missing
    action, surface filter, handler error). Voice path falls back to
    the legacy dispatcher on None.

    Phase 1 ONLY: Tier B / Tier C decisions are recorded for telemetry
    but executed as Tier A (immediate). See the design doc for the
    Phase 2 staging protocol that lifts this restriction.
    """
    if decision.tier == "REJECT" or not decision.intent_id:
        return None

    action = REGISTRY.get(decision.intent_id)
    if action is None:
        log.warning("architect_router_dispatch_unknown", intent_id=decision.intent_id)
        return None

    if not action.surfaces_for(surface):
        log.info(
            "architect_router_dispatch_surface_filtered",
            intent_id=decision.intent_id,
            surface=surface,
            action_surfaces=action.surfaces,
        )
        return None

    # Phase 1 force-promotion: B/C → A. Telemetry retains the original
    # tier so we can evaluate how often the router picks each tier
    # before the staging UX ships.
    if decision.tier in ("B", "C"):
        log.info(
            "architect_router_tier_force_promoted",
            intent_id=decision.intent_id,
            requested_tier=decision.tier,
            executed_as="A",
        )

    # Bind persistent referent cache so anchors written by the handler
    # (last_played_track etc.) survive the call.
    if app_state is not None:
        session.referents = get_referent_cache(
            app_state, session.user_id, session.session_id,
        )

    # Synthetic IntentMatch so the existing emit path can serialize it
    # without special-casing the router origin. Tier 99 is the router-
    # generated marker — any non-{1,2,3} value signals "not template".
    match = IntentMatch(
        action_id=decision.intent_id,
        args=dict(decision.args),
        confidence=decision.confidence,
        tier=99,
    )

    runtime = getattr(app_state, "companion_runtime", None) if app_state else None

    # Inference fills missing args from observation history exactly
    # like the legacy path. The router may have left required args out
    # for an inferable primitive (e.g. media.resume without a title);
    # this catches that case.
    filled_args = await infer_args(action, dict(match.args), session, runtime)

    # Translation reshapes raw user-derived args. The image primitive
    # expands "a dog" into a scene-rich prompt here. The router's
    # response_text was generated BEFORE this expansion, so the
    # spoken line and the dispatched prompt may differ — that's
    # intentional. The user hears about "a dog"; the model paints the
    # expanded prompt.
    if action.arg_transformer is not None:
        try:
            transformed = await action.arg_transformer(
                dict(filled_args), session, runtime,
            )
            if isinstance(transformed, dict):
                filled_args = transformed
        except Exception as exc:  # noqa: BLE001 — degrade gracefully
            log.warning(
                "architect_router_arg_transformer_failed",
                action_id=action.id, error=str(exc)[:200],
            )

    # Required-args validation — if the router missed a required arg
    # and inference couldn't fill it, surface a clarifying response
    # rather than firing the handler with incomplete state.
    missing = [
        a for a in action.required_args
        if a not in filled_args or filled_args[a] in (None, "", [])
    ]
    if missing:
        log.info(
            "architect_router_missing_required",
            intent_id=decision.intent_id, missing=missing,
        )
        question = f"I need to know the {missing[0]} for that — can you tell me?"
        # Park so the answer fills the slot on the next turn.
        import time as _t
        refs = getattr(session, "referents", None)
        if refs is not None:
            refs.pending_intent = {
                "action_id": decision.intent_id,
                "args": dict(filled_args),
                "missing": list(missing),
                "question": question,
                "asked_at": _t.time(),
            }
        clarifier = ActionResult(
            short_circuit=True,
            speak=question,
        )
        return ArchitectResult(
            match=match,
            action_result=clarifier,
            surface=surface,
            inferred_args=filled_args,
        )

    _pi_before = getattr(getattr(session, "referents", None), "pending_intent", None)
    try:
        result = await action.handler(transcript, session, filled_args)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "architect_router_handler_error",
            intent_id=decision.intent_id, error=str(exc)[:200],
        )
        return None

    if result is None:
        return None

    # Override the handler's spoken line with the router's response.
    # The router authored language that matches the tier policy
    # (present continuous for A, declarative-with-cancel for B,
    # explicit-question for C). The handler's default speak is a
    # fallback for the template-as-gate path; under the router, the
    # router owns the verbal acknowledgment — BUT only when the handler
    # actually actuated the intent. On a miss / empty / parked result
    # the handler's honest line wins, so she never confirms ("Playing X,
    # say cancel") an action that never happened. See ActionResult.fulfilled.
    result = _voiced_result(result, decision.response_text)

    # Record the dispatch on the referent cache + companion bus, same
    # sinks as the legacy dispatcher so downstream observers see no
    # difference in origin.
    refs = getattr(session, "referents", None)
    if refs is not None:
        refs.last_dispatch_action = decision.intent_id
        refs.last_dispatch_args = dict(filled_args)
        refs.last_dispatch_summary = (result.speak or result.toast or "")[:200]
        refs.last_dispatch_ts = _time.time()
        # Successful dispatch resolves any parked clarification —
        # unless this handler just parked a fresh one (identity check).
        if refs.pending_intent is _pi_before:
            refs.pending_intent = None

    if runtime is not None:
        try:
            bus = getattr(runtime, "bus", None)
            if bus is not None:
                # Reuse the dispatch payload scrubber from the legacy
                # path so bus consumers see a consistent payload shape
                # whether the dispatch came from templates or router.
                from augmentum.architect.dispatch import _scrub_payload
                await bus.publish_topic(
                    "surface.companion.architect_dispatch",
                    {
                        "user_id": session.user_id,
                        "session_id": session.session_id,
                        "surface": surface,
                        "action_id": decision.intent_id,
                        "tier": match.tier,  # 99 = router origin
                        "router_tier": decision.tier,
                        "args": _scrub_payload(filled_args),
                        "outcome": "short_circuit" if result.short_circuit else "augmented",
                        "spoken": result.speak[:200] if result.speak else "",
                    },
                )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "architect_router_bus_emit_failed", error=str(exc)[:200],
            )

    return ArchitectResult(
        match=match,
        action_result=result,
        surface=surface,
        inferred_args=filled_args,
    )
