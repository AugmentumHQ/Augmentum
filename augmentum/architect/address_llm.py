"""Tier 3 LLM address classifier — disambiguate utterances Tier 1 missed.

Tier 1 (``augmentum/architect/address.py``) covers strong cues:
imperatives, WH-questions with "you", self-talk, 3rd-person. It hits
the obvious cases in sub-millisecond regex but misses the ambiguous
middle — indirect requests ("I'd love some music"), bare fragments
("got anything by Miles?"), and context-dependent confirmations
("yes do it" right after an offer).

This module runs a small LLM zero-shot when Tier 1 returns
``no_signal``. The classifier sees:

  - The utterance itself
  - Becca's last spoken response (for continuation context)
  - The active surface the user is looking at
  - The last architect dispatch summary (so confirmations resolve)

It returns ``ADDRESSED`` / ``AMBIENT`` / ``UNSURE``. ``UNSURE`` is
treated as AMBIENT by callers — false silence is far cheaper than
false speech, so the default is the conservative one.

Latency budget: 2000ms hard timeout, sized for reasoning models that
need to finish thinking before emitting the verdict token. Cerebras /
Groq routinely finish in <300ms; local 7B at 60 tok/s lands ~2s; truly
slow paths time out and the dispatcher drops the turn. The
``companion_address_llm_model`` setting lets latency-sensitive
installs point at a small fast classifier instead of paying for the
user's primary chat model on every ambient utterance.

Settings:
  * ``companion_address_llm_enabled`` — turn the tier on/off
  * ``companion_address_llm_model`` — explicit model override
    (empty = use the user's current chat model)
  * ``companion_address_llm_timeout_ms`` — hard cap, default 2000
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from augmentum.config import settings
from augmentum.models.base import InternalChatRequest, Message
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


LlmAddressVerdict = Literal["ADDRESSED", "AMBIENT", "UNSURE"]


@dataclass(frozen=True)
class LlmAddressDecision:
    """Result of the LLM tier classifier."""

    verdict: LlmAddressVerdict
    latency_ms: int
    model: str
    reason: str = ""


# Prompt is deliberately tight — the model needs zero instructions
# about Augmentum, just the role of distinguishing addressed speech.
# Keeping it short keeps inference fast.
#
# The {{char}} token is substituted at call time with the running
# companion's display name (default "Becca") so the classifier knows
# *which* assistant the user might be addressing. Without the name
# the model handles "Becca, play that song" as well as it handles
# "Alex, play that song" — name conditioning shows up most on
# ambiguous self-talk vs. addressed-to-assistant edge cases.
_SYSTEM_PROMPT = (
    "You decide whether a transcribed utterance is directed at an AI assistant "
    "(named {{char}}) or is the user talking to themselves / someone else in "
    "the room / reading aloud. Reply with one of these tokens only: ADDRESSED, "
    "AMBIENT, UNSURE.\n\n"
    "Addressed = user is asking the assistant for something, asking it a "
    "question, confirming/cancelling a previous offer, or making a request — "
    "even indirectly (\"I'd love some music\", \"got anything by Miles?\"). "
    "Ambient = self-talk, talking to someone else, narration, reading aloud, "
    "background remarks. Unsure = genuinely ambiguous.\n\n"
    "Use the context (assistant's last spoken response, active surface, last "
    "dispatched action) to break ties — confirmations only count as addressed "
    "when the assistant recently offered something."
)


def _build_user_prompt(
    utterance: str,
    *,
    last_tts: str = "",
    active_surface: str = "",
    last_dispatch_summary: str = "",
) -> str:
    """Compose the per-call prompt with the available context."""
    parts: list[str] = [f"Utterance: {utterance.strip()!r}"]
    if last_tts:
        parts.append(f"Assistant's last response: {last_tts[:200]!r}")
    if last_dispatch_summary:
        parts.append(f"Last action taken: {last_dispatch_summary[:160]}")
    if active_surface:
        parts.append(f"User is currently on surface: {active_surface}")
    parts.append("\nReply with one token: ADDRESSED, AMBIENT, or UNSURE.")
    return "\n".join(parts)


def _parse_verdict(text: str) -> LlmAddressVerdict:
    """Normalize the model's response to one of the three tokens.

    Models occasionally pad with whitespace or add a trailing period —
    we accept the first token-shaped substring. Anything we can't
    confidently map collapses to UNSURE.
    """
    if not text:
        return "UNSURE"
    upper = text.strip().upper().replace(".", "").replace(",", "")
    for token in upper.split():
        if token in ("ADDRESSED", "AMBIENT", "UNSURE"):
            return token  # type: ignore[return-value]
    # Common loose phrasings the model might emit
    if "ADDRESS" in upper:
        return "ADDRESSED"
    if "AMBIENT" in upper:
        return "AMBIENT"
    return "UNSURE"


async def classify_with_llm(
    utterance: str,
    *,
    app_state: Any,
    user_id: str = "",
    session_id: str = "",
    last_tts: str = "",
    active_surface: str = "",
    last_dispatch_summary: str = "",
) -> LlmAddressDecision:
    """Run the Tier 3 classifier.

    Returns ``LlmAddressDecision(verdict="UNSURE", ...)`` whenever the
    backend can't be resolved, the call times out, or the response
    can't be parsed. The dispatch layer treats UNSURE as AMBIENT.

    The model defaults to the user's selected chat model when
    ``companion_address_llm_model`` is unset. For latency-sensitive
    installs, override that setting with a small fast model
    (Cerebras / Groq / a tiny local model).
    """
    import time as _time

    if not utterance or not utterance.strip():
        return LlmAddressDecision("AMBIENT", 0, "", reason="empty")
    if not getattr(settings, "companion_address_llm_enabled", True):
        return LlmAddressDecision("UNSURE", 0, "", reason="disabled")

    timeout_s = max(
        0.1,
        float(getattr(settings, "companion_address_llm_timeout_ms", 2000)) / 1000.0,
    )

    registry = getattr(app_state, "provider_registry", None)
    if registry is None:
        return LlmAddressDecision("UNSURE", 0, "", reason="no_registry")

    # Model selection: explicit override > user's current chat model.
    # The architect inference cache + ReferentCache live on app.state,
    # but the "what model is this user currently using" lookup is
    # deferred to the registry (which picks up the active backend).
    override = (getattr(settings, "companion_address_llm_model", "") or "").strip()
    started_at = _time.monotonic()

    try:
        backend, resolved_model = await registry.resolve_backend_with_fabric(
            override,
            user_id=user_id,
            session_id=session_id,
        )
    except Exception as exc:  # noqa: BLE001 — degrade to UNSURE
        log.warning("address_llm_resolve_failed", error=str(exc)[:160])
        return LlmAddressDecision("UNSURE", 0, override, reason="resolve_failed")

    if backend is None:
        return LlmAddressDecision("UNSURE", 0, override, reason="no_backend")

    user_prompt = _build_user_prompt(
        utterance,
        last_tts=last_tts,
        active_surface=active_surface,
        last_dispatch_summary=last_dispatch_summary,
    )

    # Resolve {{char}} in the system prompt with the running companion's
    # display name. Reads from runtime.identity (in memory, no DB hit
    # on the hot path) when available; falls back to "Becca" — the
    # default identity — when the runtime isn't initialized. This is
    # called per-voice-utterance, so cheap-by-default matters.
    system_prompt_resolved = _SYSTEM_PROMPT
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
        # Inline substitution — no need to import the prompt_compose
        # helper here since the address_llm has no {{user}} surface and
        # the regex is trivial.
        system_prompt_resolved = system_prompt_resolved.replace(
            "{{char}}", char_name,
        )
    except Exception:
        log.debug("address_llm_char_substitution_failed", exc_info=True)

    req = InternalChatRequest(
        model=resolved_model or override or "",
        messages=[
            Message(role="system", content=system_prompt_resolved),
            Message(role="user", content=user_prompt),
        ],
        stream=False,
        temperature=0.0,
        # Sized for reasoning models (GLM-4.7 / DeepSeek / Qwen3.x):
        # they spend most of the budget inside <think> and only emit the
        # verdict token after </think>. Non-reasoning models still pay
        # one token; the cap is a ceiling not a target.
        max_tokens=128,
    )

    try:
        resp = await asyncio.wait_for(backend.chat(req), timeout=timeout_s)
    except asyncio.TimeoutError:
        elapsed = int((_time.monotonic() - started_at) * 1000)
        log.info("address_llm_timeout", ms=elapsed, model=resolved_model)
        return LlmAddressDecision("UNSURE", elapsed, resolved_model, reason="timeout")
    except Exception as exc:  # noqa: BLE001 — backend errors degrade safely
        elapsed = int((_time.monotonic() - started_at) * 1000)
        log.warning(
            "address_llm_backend_error", ms=elapsed,
            model=resolved_model, error=str(exc)[:200],
        )
        return LlmAddressDecision("UNSURE", elapsed, resolved_model, reason="error")

    elapsed = int((_time.monotonic() - started_at) * 1000)
    message = getattr(resp, "message", None)
    content = (getattr(message, "content", "") or "").strip()
    thinking = (getattr(message, "thinking", "") or "").strip()

    # Parse content first; if reasoning truncated before emitting a
    # post-</think> verdict, fall back to scanning the thinking trace —
    # reasoning models often state the conclusion mid-thought ("...so
    # this is ADDRESSED.") before they ever reach the content channel.
    verdict = _parse_verdict(content)
    parsed_from = "content"
    if verdict == "UNSURE" and not content and thinking:
        verdict = _parse_verdict(thinking)
        parsed_from = "thinking" if verdict != "UNSURE" else "empty"

    log.info(
        "address_llm_decision",
        verdict=verdict,
        ms=elapsed,
        model=resolved_model,
        raw=content[:40],
        parsed_from=parsed_from,
        thinking_chars=len(thinking),
    )

    return LlmAddressDecision(verdict, elapsed, resolved_model, reason="ok")
