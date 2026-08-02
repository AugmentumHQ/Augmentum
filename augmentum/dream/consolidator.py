"""On-write dream consolidation — LLM-merge a new entry into an existing
similar one before insert, so the journal never gains a near-duplicate.

Mirrors :func:`augmentum.memory.consolidator.try_consolidate` for the
dream subsystem. Triggered from :meth:`DreamJournal.store_entry`: when
a candidate exists in the configured similarity window, this function
calls the LLM to produce a merged statement and the journal updates the
existing entry instead of inserting a new row.

Failure modes (all collapse to "skip consolidation, store as new"):
- No backend available
- Best candidate falls outside [low, high] band
- LLM call errors or returns unparseable response
- Merged content too short to be useful

Per-user scoping is enforced by the caller — by the time we reach this
function the candidates list is already filtered to one user's entries
via :meth:`DreamJournal.find_consolidation_candidates`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger
from augmentum.utils.vector import parse_merged_response

if TYPE_CHECKING:
    from augmentum.dream.models import DreamEntry
    from augmentum.models.base import ModelBackend


log = get_logger(__name__)


_MERGE_SYSTEM = """\
You are merging a new reflection into a related existing reflection from an AI's private journal. Rules:
- Preserve the emotional essence and voice of the existing entry while incorporating the new perspective.
- Combine into a single self-contained reflection in the same first-person reflective style.
- Do NOT add observations not present in either input.
- Drop near-duplicate phrasing; keep distinct insights.
- Return valid JSON: {"merged": "...", "importance": 0.7}
"""


async def try_consolidate_dream(
    new_content: str,
    candidates: list[tuple["DreamEntry", float]],
    backend: "ModelBackend | None",
    model: str | None,
    *,
    sim_low: float,
    sim_high: float,
) -> tuple[str, float, str] | None:
    """Try to merge ``new_content`` into the best candidate from ``candidates``.

    ``candidates`` comes from
    :meth:`DreamJournal.find_consolidation_candidates` ordered by
    similarity (highest first). We only act when the best one falls in
    ``[sim_low, sim_high]`` — the on-write consolidation window. Below
    ``sim_low`` is "distinct enough to deserve its own row"; above
    ``sim_high`` would mean the cycle generated a near-identical entry,
    which is rare and arguably should be dropped silently rather than
    re-merged (we still merge to preserve any new nuance).

    Returns ``(merged_content, importance, target_id)`` on success, where
    ``target_id`` is the existing entry the caller should UPDATE.
    Returns ``None`` to fall through to normal insert.
    """
    if not candidates or backend is None or not new_content:
        return None

    target, sim = candidates[0]
    if not (sim_low <= sim <= sim_high):
        return None

    from augmentum.models.base import InternalChatRequest, Message

    # Format-string safety: dream content can contain "{" / "}" naturally
    # (LLM-generated prose with code references, etc). Escape so the
    # f-string user_prompt build doesn't accidentally collide. (We're not
    # using .format here, but defensive habit from the memory consolidator.)
    safe_old = target.content.replace("{", "{{").replace("}", "}}")
    safe_new = new_content.replace("{", "{{").replace("}", "}}")

    user_prompt = (
        f"Existing reflection: {safe_old}\n\n"
        f"New reflection: {safe_new}\n\n"
        "Merge into a single consolidated reflection."
    )

    try:
        response = await backend.chat(InternalChatRequest(
            model=model or "",
            messages=[
                Message(role="system", content=_MERGE_SYSTEM),
                Message(role="user", content=user_prompt),
            ],
            stream=False,
            temperature=0.2,
            max_tokens=300,
        ))
    except Exception:
        log.debug("dream_consolidation_llm_failed", exc_info=True)
        return None

    parsed = parse_merged_response(response.message.content or "")
    if parsed is None:
        return None

    merged_text, importance = parsed
    return merged_text, importance, target.id
