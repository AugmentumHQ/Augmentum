"""Consolidation-on-write — LLM merge of related memories.

When storing a new memory, if vector search finds entries with similarity
in [CONSOLIDATION_LOW, CONSOLIDATION_HIGH), the consolidator merges them
using an LLM call.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.memory.models import Memory
    from augmentum.models.base import ModelBackend

log = get_logger(__name__)

CONSOLIDATION_LOW = 0.60
CONSOLIDATION_HIGH = 0.78  # Below existing CONTRADICTION_THRESHOLD

_MERGE_SYSTEM = """\
You merge two related user memories into one enriched statement. Rules:
- Combine all information from both memories into a single, self-contained sentence.
- Preserve specifics (names, numbers, tools, preferences).
- Do NOT add information not present in either memory.
- Return valid JSON: {"merged": "...", "importance": 0.8}
"""

_MERGE_USER = """\
Memory 1: {mem1}
Memory 2: {mem2}

Merge these into one enriched memory statement."""


async def try_consolidate(
    new_content: str,
    candidates: list[tuple[Memory, float]],
    backend: ModelBackend | None,
    model: str | None,
) -> tuple[str, float] | None:
    """Try to consolidate the new memory with a similar existing one.

    Args:
        new_content: The new memory content being stored.
        candidates: List of (Memory, similarity) pairs in consolidation range.
        backend: LLM backend for merge call.
        model: Model name to use.

    Returns:
        (merged_content, importance) on success, None on failure or no candidates.
    """
    if not candidates or backend is None:
        return None

    # Take the single most similar candidate
    best_mem, best_sim = max(candidates, key=lambda x: x[1])

    # Only consolidate if in range
    if not (CONSOLIDATION_LOW <= best_sim < CONSOLIDATION_HIGH):
        return None

    from augmentum.models.base import InternalChatRequest, Message

    safe_new = new_content.replace("{", "{{").replace("}", "}}")
    safe_old = best_mem.content.replace("{", "{{").replace("}", "}}")
    user_prompt = _MERGE_USER.format(mem1=safe_old, mem2=safe_new)

    # Non-latency-sensitive merge → let the onboard reasoner THINK (Gemma 4
    # E2B honors enable_thinking). No-op on non-reasoning models. Bump the
    # token cap so the reasoning trace doesn't starve the merged statement.
    from augmentum.config import settings as _settings
    _think = bool(getattr(_settings, "onboard_reasoning_thinking", True))
    # When thinking, the trace can run thousands of tokens — use the generous
    # configured ceiling so it never starves the merged statement (which would
    # silently skip the merge).
    _max_tok = int(getattr(_settings, "onboard_reasoning_max_tokens", 8192)) if _think else 300
    request = InternalChatRequest(
        model=model or "",
        messages=[
            Message(role="system", content=_MERGE_SYSTEM),
            Message(role="user", content=user_prompt),
        ],
        stream=False,
        temperature=0.1,
        max_tokens=_max_tok,
        think=_think,
    )

    try:
        response = await backend.chat(request)
        return _parse_merge_response(response.message.content)
    except Exception:
        log.warning("consolidation_failed", exc_info=True)
        return None


def _parse_merge_response(raw: str) -> tuple[str, float] | None:
    """Parse the LLM merge response."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    merged = str(data.get("merged", "")).strip()
    if len(merged) < 5:
        return None

    try:
        importance = max(0.0, min(1.0, float(data.get("importance", 0.7))))
    except (TypeError, ValueError):
        importance = 0.7

    return merged, importance
