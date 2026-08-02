"""Compaction synthesis — an LLM-written handoff note for the compacted block.

The mechanical compaction segment (``_build_segment_parts`` in the coder
handler) preserves grounded content — file previews, tool tallies, files
touched — but carries no narrative: WHY things were done, what was
learned, what's still open. Post-compaction, that narrative is exactly
what keeps the model behaving like the pre-compaction agent (the Claude
Code compaction insight: summarize state + decisions + next steps, not
just events). This module adds that layer as one cheap second-model call
per compaction pass.

Call shape is copied from ``goal_judge.py`` (the proven second-LLM-call
plumbing): reuse the turn's backend + request so provider/routing carry,
clear KV slot affinity so the one-shot never evicts the act loop's warm
slot, stream=False, temperature=0, think=False, bounded max_tokens,
never raises. Fail open, LOUDLY (log.warning) — a broken synthesis path
degrades to the mechanical segment, never blocks compaction.

The synthesis text is written only into the freshly-appended segment of
the append-only ``<compacted>`` block, so the byte-stable prefix
contract (2026-07-02 measurements) is untouched by construction.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace as _dataclass_replace
from typing import TYPE_CHECKING

from augmentum.models.base import InternalChatRequest, Message
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import Backend

log = get_logger(__name__)

# Hard wall-clock bound on the synthesis round-trip. Compaction sits on
# the act loop's critical path; a hung provider must degrade to the
# mechanical segment, not stall the iteration.
_SYNTHESIS_TIMEOUT_S = 25.0

# Input caps. The segment preview is already condensed (per-message
# lines are clipped by the mechanical pass), so 16k chars covers even
# large dropped regions without blowing up the one-shot's prefill.
_MAX_PREVIEW_CHARS = 16_000
_MAX_GOAL_CHARS = 1_500

_SYNTHESIS_PROMPT = """\
You are writing a handoff note for a coding agent whose older working \
history is being condensed to save context. The agent will keep working \
on the same task and will see YOUR NOTE plus the condensed event log \
below — your note is what preserves its working understanding.

Write a dense plain-text note (no markdown headings, no code fences) \
with exactly these four labeled lines, each 1-3 sentences:

State: <where the work stands right now — what has been completed and verified>
Decisions: <key choices made and WHY (approaches picked, approaches ruled out)>
Learnings: <gotchas, constraints, or facts discovered that would be expensive to rediscover>
Next: <what remains or was in flight when this note was written>

Be specific (file paths, function names, error strings) and never \
invent anything not present in the log. If a section has nothing, \
write "none"."""


async def synthesize_compaction_segment(
    backend: Backend,
    *,
    source_request: InternalChatRequest,
    segment_preview: str,
    user_goal: str,
    timeout_s: float = _SYNTHESIS_TIMEOUT_S,
) -> str | None:
    """One synthesis round-trip. Never raises; ``None`` on any failure."""
    preview = (segment_preview or "").strip()
    if not preview:
        return None

    goal = (user_goal or "").strip()[:_MAX_GOAL_CHARS]
    if len(preview) > _MAX_PREVIEW_CHARS:
        # Keep the newest half of the budget verbatim (recent events
        # matter most for the handoff) plus the oldest quarter for the
        # setup; note the elision so the model doesn't treat the seam
        # as contiguous history.
        head = preview[: _MAX_PREVIEW_CHARS // 4]
        tail = preview[-(_MAX_PREVIEW_CHARS // 2):]
        preview = f"{head}\n… [middle of the log elided] …\n{tail}"

    prompt = (
        f"<task>\n{goal or '(not recorded)'}\n</task>\n\n"
        f"<condensed_log>\n{preview}\n</condensed_log>\n\n"
        f"{_SYNTHESIS_PROMPT}"
    )

    synth_request = _dataclass_replace(
        source_request,
        messages=[Message(role="user", content=prompt)],
        # No KV slot affinity — same rationale as goal_judge.py: a
        # differently-framed one-shot on the loop's warm slot would
        # overwrite its KV and force a cold re-prefill next iteration.
        kv_session_key="",
        stream=False,
        tools=None,
        tool_choice=None,
        temperature=0.0,
        max_tokens=450,
        chat_template_kwargs=None,
        # Reasoning OFF: deterministic summarization; a reasoning model
        # would burn the budget in the thought channel and return empty.
        think=False,
        reasoning_effort=None,
    )

    try:
        response = await asyncio.wait_for(
            backend.chat(synth_request), timeout=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "coder_compaction_synthesis_backend_error",
            error=str(exc)[:200],
            model=synth_request.model,
        )
        return None

    from augmentum.models.base import response_text

    raw = response_text(response, thinking_fallback=True).strip()
    if not raw:
        log.warning("coder_compaction_synthesis_empty_response")
        return None
    # Defensive bound: the note rides inside the compacted block whose
    # cap logic assumes segments stay small.
    return raw[:4_000]


__all__ = ["synthesize_compaction_segment"]
