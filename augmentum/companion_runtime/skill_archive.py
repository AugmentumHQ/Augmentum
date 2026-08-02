"""Skill archive — append-only memory of dispatch outcomes.

Sprint 4b. Writes one row per dispatch decision into
``companion_skill_archive`` (migration 159). The row pairs the
context (intent text + embedding + intent source) with the chosen
subagent and a derived outcome signal in ``[-1, +1]``.

Outcome signal derivation:
- ``+1.0`` : ``subagent.completed`` fired without an ``error`` event
  in the following 60s AND no corrective user utterance in 5 min.
- ``+0.5`` : completed cleanly but the user follow-up suggests partial
  satisfaction (e.g. "again but shorter").
- ``0.0`` : indeterminate (default if telemetry didn't capture either
  positive or negative signals).
- ``-0.5`` : completed but the user corrected the choice ("not that
  mode, try analytical").
- ``-1.0`` : raised an exception OR the user explicitly rejected.

Sprint 4b's :mod:`dpo_retrieval` reads these rows at dispatch time
and computes a preference-weighted utility delta. No model weights
change — this is retrieval-augmented preference.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.state.backends.sqlite import transactional_write
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime, Intent

log = get_logger(__name__)


@dataclass(slots=True)
class ArchiveRow:
    """An entry in the skill archive."""
    id: int | None
    companion_id: str
    ts: float
    intent_text: str
    intent_source: str
    chosen_subagent: str
    outcome_signal: float
    outcome_reason: str = ""
    decision_ms: float = 0.0
    used_tiebreaker: bool = False
    context_embedding: list[float] | None = None


def _pack_embedding(vec: list[float] | None) -> bytes | None:
    if not vec:
        return None
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes | None) -> list[float] | None:
    if not blob:
        return None
    n = len(blob) // 4
    if n <= 0:
        return None
    return list(struct.unpack(f"<{n}f", blob))


async def _embed_intent(intent_text: str) -> list[float] | None:
    """Compute the intent's embedding. Returns None on failure — the
    row still goes in, just without an embedding (DPO retrieval
    gracefully degrades to non-vector matching).

    EmbeddingService.embed_one is synchronous and may block briefly on
    first call (model load). We offload to a thread so the dispatch
    path's event loop isn't stalled by the cold-load case.
    """
    try:
        import asyncio
        from augmentum.memory.embeddings import EmbeddingService
        return await asyncio.to_thread(EmbeddingService.embed_one, intent_text)
    except Exception:
        log.debug("skill_archive_embedding_failed", exc_info=True)
        return None


async def record_outcome(
    runtime: CompanionRuntime,
    intent: Intent,
    *,
    chosen_subagent: str,
    outcome_signal: float,
    outcome_reason: str = "",
    decision_ms: float = 0.0,
    used_tiebreaker: bool = False,
) -> int | None:
    """Append one row. Returns the new row id or ``None`` on failure.

    Flag-gated by ``companion_skill_archive_enabled``. When off this
    is a no-op so Sprint 4b can ship dark.
    """
    from augmentum.config import settings
    if not getattr(settings, "companion_skill_archive_enabled", False):
        return None

    outcome_signal = max(-1.0, min(1.0, outcome_signal))
    embedding = await _embed_intent(intent.text)
    blob = _pack_embedding(embedding)

    # backend.connect() returns None (it's an initializer, not a context
    # manager) — the prior `async with backend.connect()` would have
    # crashed had this flag-gated path ever run (audit 2026-06-17). Use
    # the live connection under transactional_write (commit/rollback).
    try:
        async with transactional_write(runtime.backend.conn) as conn:
            cur = await conn.execute(
                "INSERT INTO companion_skill_archive "
                "(companion_id, user_id, ts, intent_text, intent_source, "
                "context_embedding, chosen_subagent, outcome_signal, "
                "outcome_reason, decision_ms, used_tiebreaker) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    runtime.companion_id, intent.user_id, time.time(),
                    intent.text[:4000], intent.source,
                    blob, chosen_subagent,
                    outcome_signal, outcome_reason[:300],
                    decision_ms, 1 if used_tiebreaker else 0,
                ),
            )
            row_id = cur.lastrowid
    except Exception:
        log.exception("skill_archive_write_failed")
        return None

    await runtime.bus.publish_topic(
        "skill_archive.recorded",
        {
            "row_id": row_id,
            "subagent": chosen_subagent,
            "outcome_signal": outcome_signal,
        },
        source_companion_id=runtime.companion_id,
    )
    return row_id


async def nearest(
    runtime: CompanionRuntime,
    query_embedding: list[float],
    *,
    k: int = 8,
    user_id: str | None = None,
) -> list[ArchiveRow]:
    """Return the ``k`` archive rows whose context_embedding is
    closest to ``query_embedding`` by cosine similarity.

    When ``user_id`` is provided the search is scoped to that owner so
    one user's dispatch preferences can't steer another's subagent
    selection (audit 2026-06-17). ``None`` preserves the legacy
    companion-wide behaviour.

    Pure-Python ranking — fine up to ~10k rows. Sprint 6+ may grow a
    sqlite-vss path if archive volume warrants it.
    """
    if not query_embedding:
        return []
    rows: list[tuple[float, ArchiveRow]] = []
    where = "WHERE companion_id = ? AND context_embedding IS NOT NULL"
    params: tuple = (runtime.companion_id,)
    if user_id is not None:
        where += " AND user_id IS ?"
        params = (runtime.companion_id, user_id)
    # Read path — use the live connection directly (no transaction
    # needed). backend.connect() returns None, so the prior
    # `async with backend.connect()` would crash (audit 2026-06-17).
    try:
        conn = runtime.backend.conn
        cur = await conn.execute(
            "SELECT id, companion_id, ts, intent_text, intent_source, "
            "context_embedding, chosen_subagent, outcome_signal, "
            "outcome_reason, decision_ms, used_tiebreaker "
            "FROM companion_skill_archive "
            f"{where} "
            "ORDER BY ts DESC LIMIT 2000",
            params,
        )
        db_rows = await cur.fetchall()
        await cur.close()
    except Exception:
        log.exception("skill_archive_nearest_failed")
        return []

    qa = query_embedding
    qa_norm = sum(x * x for x in qa) ** 0.5 or 1.0
    for r in db_rows:
        emb = _unpack_embedding(r[5])
        if not emb or len(emb) != len(qa):
            continue
        dot = sum(x * y for x, y in zip(emb, qa))
        emb_norm = sum(x * x for x in emb) ** 0.5 or 1.0
        cosine = dot / (qa_norm * emb_norm)
        rows.append((cosine, ArchiveRow(
            id=r[0], companion_id=r[1], ts=r[2],
            intent_text=r[3], intent_source=r[4],
            context_embedding=emb,
            chosen_subagent=r[6], outcome_signal=r[7],
            outcome_reason=r[8], decision_ms=r[9],
            used_tiebreaker=bool(r[10]),
        )))
    rows.sort(key=lambda kv: kv[0], reverse=True)
    return [r for _, r in rows[:k]]


__all__ = [
    "ArchiveRow",
    "nearest",
    "record_outcome",
]
