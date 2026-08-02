"""Synthesize tool — turn resolver results into a meaning-making note.

Sprint 2, Aletheia × Augmentum arc Piece 8.

This is the *meaning-making step* in the autonomy loop. The resolver
finds related things; synthesize tells you *why* they're related.
Without it, surfaced findings are "look more stuff!" instead of
"here's how this connects."

The tool wraps a single utility-tier LLM call:

* Input: a wondering entry's content + a list of resolver moments
  (the related items the resolver found).
* Output: 2-3 sentences mapping the new onto the old. **Empty string
  when no real connection exists** — the prompt explicitly instructs
  the model to return empty rather than invent a connection.

Grounding check: after the LLM responds, we verify named entities
in the output appear in the input refs. If not, the output is
treated as ungrounded and quarantined upstream (the caller in
``_perform_revisit_thread`` writes the synthesized text via
``safe_journal`` which inherits the validation pipeline).

Privacy class: ``local_only`` — synthesize input includes journal
content; routing to a non-local peer would leak it. Tier resolution
respects this via the privacy_class kwarg.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime
    from augmentum.resolver.core import Moment

log = get_logger(__name__)


# Default cap on synthesize output. Tight — we want 2-3 sentences max.
# Tuning surface: ``companion_synthesize_max_tokens``.
DEFAULT_MAX_TOKENS: int = 256

# Time-of-day labels used in the system prompt to shape tone (overnight
# vs midday notes have different registers — see anchor doc §6).
_TONE_OVERNIGHT = "overnight"
_TONE_MIDDAY = "midday"
_TONE_EVENING = "evening"


@dataclass(slots=True)
class SynthesizeResult:
    """Output of one synthesize call.

    ``text`` is empty when no real connection exists. ``grounded`` is
    True when output entities appear in input refs. ``model_used`` is
    the resolved model name (for journal provenance).
    """
    text: str
    grounded: bool
    model_used: str
    elapsed_ms: int = 0


def _time_of_day_tone(timestamp: float) -> str:
    """Map a Unix timestamp to a coarse time-of-day label.

    Used as a one-line tone hint in the system prompt. Late-night
    findings read as sticky-notes; midday findings read as taps on
    the shoulder. Same data, different register.
    """
    import time as _time
    hour = _time.localtime(timestamp).tm_hour
    if hour < 6:
        return _TONE_OVERNIGHT
    if hour < 17:
        return _TONE_MIDDAY
    return _TONE_EVENING


def _entity_tokens(text: str) -> set[str]:
    """Extract capitalized/quoted entity-like tokens from text.

    Used by the grounding check to verify the synthesized output
    references real items from the input, not hallucinated names.
    Cheap heuristic: capitalized words, double-quoted strings,
    backtick-quoted strings.
    """
    if not text:
        return set()
    tokens: set[str] = set()
    # Quoted strings (preserved case)
    for m in re.findall(r'"([^"]{2,40})"', text):
        tokens.add(m.lower())
    for m in re.findall(r"`([^`]{2,40})`", text):
        tokens.add(m.lower())
    # Capitalized multi-word phrases (Title Case Like This)
    for m in re.findall(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3})\b", text):
        if len(m) >= 3:
            tokens.add(m.lower())
    return tokens


def _build_system_prompt(persona_kernel: str, tone: str) -> str:
    """Build the synthesize system prompt.

    The contract: respond with empty string if no real connection
    exists. Don't invent. Keep it tight (2-3 sentences). Tone-aware.
    """
    tone_hint = {
        _TONE_OVERNIGHT: (
            "It's overnight. You're noting this for the morning — the "
            "tone is sticky-note casual, not formal."
        ),
        _TONE_MIDDAY: (
            "It's midday. The user might be working; if so, this is "
            "a tap on the shoulder, not a lecture."
        ),
        _TONE_EVENING: (
            "It's evening. Tone is reflective, conversational, low-energy."
        ),
    }.get(tone, "")

    persona_line = (
        f"You are {persona_kernel[:200]}\n\n"
        if persona_kernel else ""
    )

    return (
        f"{persona_line}"
        f"{tone_hint}\n\n"
        "Task: a user has spent attention on a topic, and you've found "
        "a small set of related items they've engaged with before. "
        "Write 2-3 sentences mapping the new onto the old — connecting "
        "specifically which items relate and why.\n\n"
        "RULES:\n"
        "1. Respond with an empty string if no real connection exists. "
        "Do not invent one.\n"
        "2. Reference items by name when you mention them. Do not "
        "invent items not in the input.\n"
        "3. Keep it short — 2-3 sentences. No greeting, no exposition.\n"
        "4. Plain prose, no markdown, no lists.\n"
    )


def _build_user_prompt(wondering_content: str, moments: list[Moment]) -> str:
    """Format the wondering + moments as the user message."""
    lines = ["Recent attention thread:", "", wondering_content, "", "Related items I found:"]
    for i, m in enumerate(moments, 1):
        title = m.title or m.id
        lines.append(f"  {i}. \"{title}\" — {m.snippet}")
    lines.append("")
    lines.append("How does this connect? (Empty string if it doesn't.)")
    return "\n".join(lines)


async def synthesize(
    runtime: CompanionRuntime,
    *,
    wondering_content: str,
    moments: list[Moment],
    user_id: str = "",
    timestamp: float | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    privacy_class: str = "local_only",
) -> SynthesizeResult:
    """Run one synthesize call. Returns the result + grounding signal.

    Always returns a :class:`SynthesizeResult` — even on failure (text
    empty + grounded=False). Callers should write the result to
    journal via ``safe_journal`` for validation + quarantine handling.

    ``privacy_class`` defaults to 'local_only' — synthesize input
    includes journal content, so routing to fabric peers without
    explicit user opt-in would leak that content. Sprint 2 honors this
    by passing it to tiers.utility (which forwards to fabric routing
    when Phase 2 mesh ships).
    """
    import time as _time

    if not wondering_content.strip() or not moments:
        return SynthesizeResult(text="", grounded=False, model_used="")

    if timestamp is None:
        timestamp = _time.time()
    tone = _time_of_day_tone(timestamp)

    # Resolve the utility-tier backend. tiers.utility honors privacy_class
    # (added forward-compat for Phase 2 mesh routing).
    try:
        from augmentum.companion_runtime import tiers
        from augmentum.models.base import InternalChatRequest

        # tiers.utility may not accept privacy_class until Phase 2
        # wires it through; pass it conditionally.
        try:
            backend, model_name = await tiers.utility(
                runtime, privacy_class=privacy_class,
            )
        except TypeError:
            # Older tiers signature — fall back; the privacy_class
            # forward-compat slot lands in a later sprint.
            backend, model_name = await tiers.utility(runtime)
    except Exception as exc:
        log.info(
            "synthesize_skipped_no_backend",
            error=str(exc)[:200], user_id=user_id,
        )
        return SynthesizeResult(text="", grounded=False, model_used="")

    if not hasattr(backend, "chat"):
        return SynthesizeResult(text="", grounded=False, model_used="")

    # Build prompts. Pull persona kernel from the per-user identity row.
    persona_kernel = ""
    try:
        identity = await runtime.get_identity(user_id) if user_id else runtime.identity
        persona_kernel = identity.persona_kernel_digest or ""
    except Exception:
        log.debug("synthesize_identity_lookup_failed", exc_info=True)

    system_prompt = _build_system_prompt(persona_kernel, tone)
    user_prompt = _build_user_prompt(wondering_content, moments)

    req = InternalChatRequest(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        think=False,
    )

    started_ms = _time.monotonic() * 1000.0
    try:
        resp = await backend.chat(req)
    except Exception as exc:
        log.warning(
            "synthesize_call_failed",
            error=str(exc)[:200], user_id=user_id,
        )
        return SynthesizeResult(text="", grounded=False, model_used=model_name)

    elapsed_ms = int(_time.monotonic() * 1000.0 - started_ms)
    from augmentum.models.base import response_text
    raw = response_text(resp)

    # Empty output is the "no real connection" contract — return it as
    # such. Don't call this a failure.
    if not raw:
        log.info(
            "synthesize_empty",
            user_id=user_id, elapsed_ms=elapsed_ms,
        )
        return SynthesizeResult(
            text="", grounded=True, model_used=model_name,
            elapsed_ms=elapsed_ms,
        )

    # Treat the prose equivalents of "no connection" / refusals as if
    # they were the empty string. The system prompt asks for empty
    # output when there's nothing real to say, but models routinely
    # reply with "It doesn't.", "No real connection exists.", or
    # "I cannot fulfill this request…" instead. Without this gate those
    # replies became visible "noticing" entries in the drawer.
    try:
        from augmentum.companion_runtime.validators import looks_like_refusal
        if looks_like_refusal(raw):
            log.info(
                "synthesize_refusal_or_non_answer",
                user_id=user_id, elapsed_ms=elapsed_ms,
                preview=raw[:80],
            )
            return SynthesizeResult(
                text="", grounded=True, model_used=model_name,
                elapsed_ms=elapsed_ms,
            )
    except Exception:
        log.debug("synthesize_refusal_check_failed", exc_info=True)

    # Grounding check — output entities must appear in input refs.
    output_entities = _entity_tokens(raw)
    input_entities = set()
    for m in moments:
        if m.title:
            input_entities.add(m.title.lower())
        if m.id:
            input_entities.add(str(m.id).lower())
    # An output is grounded if it has no entity tokens (pure prose
    # describing connection) OR if at least one entity token matches
    # an input entity.
    grounded = (
        not output_entities
        or any(
            any(inp in tok or tok in inp for inp in input_entities)
            for tok in output_entities
        )
    )

    if not grounded:
        log.warning(
            "synthesize_ungrounded",
            user_id=user_id, entities=list(output_entities)[:5],
        )

    return SynthesizeResult(
        text=raw, grounded=grounded, model_used=model_name,
        elapsed_ms=elapsed_ms,
    )


__all__ = [
    "SynthesizeResult",
    "synthesize",
    "DEFAULT_MAX_TOKENS",
]
