"""Durable per-turn archive for the coder mode.

Each completed turn writes a structured record to ``coder_turn_archive``
(migration 213). The archive sits beneath the in-prompt FIFO
(turn_summaries cap 10) and beneath compaction — its purpose is to
survive both, so the agent can recall earlier work in future
sessions via semantic search (Phase 2) and the inspector can render
a full timeline (Phase 3).

Design contract
---------------

* **Append-only at write time.** No updates, no deletes (except via
  the user-deletion cascade or workspace deletion). Stale-fact
  problem is handled in retrieval (Phase 2: confidence decay +
  ``superseded_by`` links).
* **Best-effort.** Persistence failures log and continue; never raise
  back into the agent loop. The archive is one of several memory
  layers — a missed write doesn't break the turn.
* **Workspace-scoped by default** with ``(user_id, workspace_id)``
  prefix. Cross-workspace recall is explicit (Phase 2 setting).
* **Bi-temporal timestamps.** ``event_time`` (when the turn finished)
  and ``recorded_at`` (when this row was committed) are persisted
  separately so future queries can distinguish "we knew this then"
  from "we recorded this row late."
* **Row cap enforced at write time.** Per ``coder_archive_max_turns_per_workspace``
  setting; oldest rows pruned when the cap is hit. Default high
  (10000 turns/workspace) so the cap is a safety net, not a feature.

This module owns reads/writes; the embedding pipeline (Phase 2) lives
in ``augmentum/coder/turn_archive_embed.py`` and is purely additive
over this store.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


# Default row cap when settings.coder_archive_max_turns_per_workspace
# is unset/0. Tuned high — at ~5KB/turn (excluding embedding), 10k
# turns = ~50MB per workspace. Dogfood usage is in the hundreds, not
# tens of thousands, so the cap rarely fires in practice.
_DEFAULT_MAX_TURNS_PER_WORKSPACE = 10_000


@dataclass(slots=True)
class TurnArchiveEntry:
    """One archived turn. Mirrors the SQL schema 1:1."""

    archive_id: str = ""
    user_id: str = ""
    workspace_id: str = ""
    run_id: str = ""
    turn_id: str = ""
    turn_index: int = 0
    user_goal: str = ""
    outcome: str = ""
    verdict_reason: str = ""
    blockers: str = ""
    files_read: list[str] = field(default_factory=list)
    files_edited: list[dict] = field(default_factory=list)
    shell_commands: list[str] = field(default_factory=list)
    edits: list[dict] = field(default_factory=list)
    summary: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    event_time: int = 0
    recorded_at: int = 0
    embedding_status: str = "pending"
    superseded_by: str = ""
    confidence: float = 1.0

    def to_inspector_dict(self) -> dict[str, Any]:
        """Compact dict for the inspector timeline (Phase 3 UI)."""
        return {
            "archive_id": self.archive_id,
            "turn_index": self.turn_index,
            "user_goal": self.user_goal,
            "outcome": self.outcome,
            "verdict_reason": self.verdict_reason,
            "files_read_count": len(self.files_read),
            "files_edited_count": len(self.files_edited),
            "shell_commands_count": len(self.shell_commands),
            "summary": self.summary,
            "event_time": self.event_time,
            "blockers": (self.blockers or "")[:240],
        }


def _resolve_row_cap() -> int:
    """Read the configured row cap, defaulting to the module constant."""
    try:
        from augmentum.config import settings
        v = int(getattr(settings, "coder_archive_max_turns_per_workspace", 0) or 0)
        return v if v > 0 else _DEFAULT_MAX_TURNS_PER_WORKSPACE
    except Exception:
        return _DEFAULT_MAX_TURNS_PER_WORKSPACE


async def _next_turn_index(
    conn: aiosqlite.Connection, *, user_id: str, workspace_id: str,
) -> int:
    """Monotonic per-workspace counter. 1-based; first turn = 1.

    Read-modify-write but the only writer for a given workspace is the
    handler — the broker enforces single-active-run per workspace —
    so no race.
    """
    try:
        cursor = await conn.execute(
            "SELECT COALESCE(MAX(turn_index), 0) FROM coder_turn_archive "
            "WHERE user_id = ? AND workspace_id = ?",
            (user_id, workspace_id),
        )
        row = await cursor.fetchone()
        return int((row[0] if row else 0) or 0) + 1
    except Exception as exc:
        log.debug("coder_turn_archive.next_index_failed", error=str(exc)[:160])
        return 1


async def _prune_to_cap(
    conn: aiosqlite.Connection, *, user_id: str, workspace_id: str, cap: int,
) -> int:
    """Drop oldest rows until total ≤ ``cap``. Returns pruned count.

    Called after every write so the cap is a hard ceiling. At default
    cap (10k turns) this fires once a workspace is years old; the
    select-min query is bounded by the indexed (user_id, workspace_id)
    prefix and runs in single-digit ms.
    """
    if cap <= 0:
        return 0
    try:
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM coder_turn_archive "
            "WHERE user_id = ? AND workspace_id = ?",
            (user_id, workspace_id),
        )
        row = await cursor.fetchone()
        total = int((row[0] if row else 0) or 0)
        if total <= cap:
            return 0
        overflow = total - cap
        # Delete the ``overflow`` oldest rows by event_time.
        await conn.execute(
            "DELETE FROM coder_turn_archive "
            "WHERE archive_id IN ("
            "  SELECT archive_id FROM coder_turn_archive "
            "  WHERE user_id = ? AND workspace_id = ? "
            "  ORDER BY event_time ASC, turn_index ASC "
            "  LIMIT ?"
            ")",
            (user_id, workspace_id, overflow),
        )
        await conn.commit()
        log.info(
            "coder_turn_archive.pruned",
            user_id=user_id, workspace_id=workspace_id,
            pruned=overflow, cap=cap,
        )
        return overflow
    except Exception as exc:
        log.debug("coder_turn_archive.prune_failed", error=str(exc)[:160])
        return 0


async def append_turn(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    workspace_id: str,
    run_id: str = "",
    turn_id: str = "",
    user_goal: str = "",
    outcome: str = "",
    verdict_reason: str = "",
    blockers: str = "",
    files_read: list[str] | None = None,
    files_edited: list[dict] | None = None,
    shell_commands: list[str] | None = None,
    edits: list[dict] | None = None,
    summary: str = "",
    tokens_in: int = 0,
    tokens_out: int = 0,
    event_time: int | None = None,
) -> str | None:
    """Persist one turn's archive row.

    Returns the new ``archive_id`` on success, ``None`` on any failure
    (logged at debug). Caller never raises into the agent loop.
    """
    if not user_id or not workspace_id:
        return None

    archive_id = uuid.uuid4().hex[:16]
    now = int(time.time())
    et = int(event_time) if event_time is not None else now

    try:
        turn_index = await _next_turn_index(
            conn, user_id=user_id, workspace_id=workspace_id,
        )
        await conn.execute(
            """
            INSERT INTO coder_turn_archive (
                archive_id, user_id, workspace_id, run_id, turn_id,
                turn_index, user_goal, outcome, verdict_reason, blockers,
                files_read, files_edited, shell_commands, edits, summary,
                tokens_in, tokens_out, event_time, recorded_at,
                embedding_status, superseded_by, confidence
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?,
                'pending', '', 1.0
            )
            """,
            (
                archive_id, user_id, workspace_id, run_id, turn_id,
                turn_index, user_goal[:2000], outcome[:64],
                verdict_reason[:128], (blockers or "")[:1000],
                json.dumps(files_read or [])[:20_000],
                json.dumps(files_edited or [])[:50_000],
                json.dumps(shell_commands or [])[:20_000],
                json.dumps(edits or [])[:50_000],
                (summary or "")[:8_000],
                int(tokens_in or 0), int(tokens_out or 0),
                et, now,
            ),
        )
        await conn.commit()
    except Exception as exc:
        log.debug(
            "coder_turn_archive.append_failed",
            user_id=user_id, workspace_id=workspace_id,
            error=str(exc)[:160],
        )
        return None

    cap = _resolve_row_cap()
    await _prune_to_cap(
        conn, user_id=user_id, workspace_id=workspace_id, cap=cap,
    )

    log.info(
        "coder_turn_archive.appended",
        archive_id=archive_id,
        workspace_id=workspace_id,
        turn_index=turn_index,
        outcome=outcome,
    )

    # Phase 2 — embed this row immediately so semantic recall is
    # available on the next turn. Best-effort: failures stay 'pending'
    # for a future backfill sweep. Embedding runs in the same event
    # loop but is fast (~50ms locally for one 768-dim vector) so we
    # don't background it.
    try:
        from augmentum.coder.turn_archive_embed import embed_one
        await embed_one(conn, archive_id=archive_id)
    except Exception as exc:
        log.debug(
            "coder_turn_archive.embed_hook_failed",
            archive_id=archive_id, error=str(exc)[:160],
        )

    return archive_id


def _parse_json_list(raw: str) -> list:
    try:
        v = json.loads(raw or "[]")
        return v if isinstance(v, list) else []
    except Exception:
        return []


async def list_recent_turns(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    workspace_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[TurnArchiveEntry]:
    """Most-recent-first listing for the inspector timeline.

    Pure read — no decay applied. Caller orders presentation.
    """
    limit = max(1, min(int(limit or 50), 500))
    offset = max(0, int(offset or 0))
    out: list[TurnArchiveEntry] = []
    try:
        cursor = await conn.execute(
            """
            SELECT archive_id, user_id, workspace_id, run_id, turn_id,
                   turn_index, user_goal, outcome, verdict_reason, blockers,
                   files_read, files_edited, shell_commands, edits, summary,
                   tokens_in, tokens_out, event_time, recorded_at,
                   embedding_status, superseded_by, confidence
            FROM coder_turn_archive
            WHERE user_id = ? AND workspace_id = ?
            ORDER BY turn_index DESC
            LIMIT ? OFFSET ?
            """,
            (user_id, workspace_id, limit, offset),
        )
        rows = await cursor.fetchall()
    except Exception as exc:
        log.debug("coder_turn_archive.list_failed", error=str(exc)[:160])
        return out

    for row in rows or []:
        out.append(TurnArchiveEntry(
            archive_id=row[0] or "",
            user_id=row[1] or "",
            workspace_id=row[2] or "",
            run_id=row[3] or "",
            turn_id=row[4] or "",
            turn_index=int(row[5] or 0),
            user_goal=row[6] or "",
            outcome=row[7] or "",
            verdict_reason=row[8] or "",
            blockers=row[9] or "",
            files_read=_parse_json_list(row[10]),
            files_edited=_parse_json_list(row[11]),
            shell_commands=_parse_json_list(row[12]),
            edits=_parse_json_list(row[13]),
            summary=row[14] or "",
            tokens_in=int(row[15] or 0),
            tokens_out=int(row[16] or 0),
            event_time=int(row[17] or 0),
            recorded_at=int(row[18] or 0),
            embedding_status=row[19] or "pending",
            superseded_by=row[20] or "",
            confidence=float(row[21] or 1.0),
        ))
    return out


async def get_turn_window(
    conn: aiosqlite.Connection,
    *,
    user_id: str,
    workspace_id: str,
    center_turn_index: int,
    before: int = 2,
    after: int = 2,
) -> list[TurnArchiveEntry]:
    """Return turns around ``center_turn_index`` for the recall_expand
    tool. Inclusive at both ends.
    """
    before = max(0, int(before or 0))
    after = max(0, int(after or 0))
    lo = max(1, center_turn_index - before)
    hi = center_turn_index + after
    try:
        cursor = await conn.execute(
            """
            SELECT archive_id, user_id, workspace_id, run_id, turn_id,
                   turn_index, user_goal, outcome, verdict_reason, blockers,
                   files_read, files_edited, shell_commands, edits, summary,
                   tokens_in, tokens_out, event_time, recorded_at,
                   embedding_status, superseded_by, confidence
            FROM coder_turn_archive
            WHERE user_id = ? AND workspace_id = ?
              AND turn_index >= ? AND turn_index <= ?
            ORDER BY turn_index ASC
            """,
            (user_id, workspace_id, lo, hi),
        )
        rows = await cursor.fetchall()
    except Exception as exc:
        log.debug("coder_turn_archive.window_failed", error=str(exc)[:160])
        return []

    out: list[TurnArchiveEntry] = []
    for row in rows or []:
        out.append(TurnArchiveEntry(
            archive_id=row[0] or "",
            user_id=row[1] or "",
            workspace_id=row[2] or "",
            run_id=row[3] or "",
            turn_id=row[4] or "",
            turn_index=int(row[5] or 0),
            user_goal=row[6] or "",
            outcome=row[7] or "",
            verdict_reason=row[8] or "",
            blockers=row[9] or "",
            files_read=_parse_json_list(row[10]),
            files_edited=_parse_json_list(row[11]),
            shell_commands=_parse_json_list(row[12]),
            edits=_parse_json_list(row[13]),
            summary=row[14] or "",
            tokens_in=int(row[15] or 0),
            tokens_out=int(row[16] or 0),
            event_time=int(row[17] or 0),
            recorded_at=int(row[18] or 0),
            embedding_status=row[19] or "pending",
            superseded_by=row[20] or "",
            confidence=float(row[21] or 1.0),
        ))
    return out


async def count_turns(
    conn: aiosqlite.Connection, *, user_id: str, workspace_id: str,
) -> int:
    """Total archived turn count for the inspector header chip."""
    try:
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM coder_turn_archive "
            "WHERE user_id = ? AND workspace_id = ?",
            (user_id, workspace_id),
        )
        row = await cursor.fetchone()
        return int((row[0] if row else 0) or 0)
    except Exception:
        return 0


__all__ = [
    "TurnArchiveEntry",
    "append_turn",
    "count_turns",
    "get_turn_window",
    "list_recent_turns",
]
