"""Commitment ledger v1 — journal-backed open loops.

Companion Agency spec (docs/superpowers/specs/2026-06-10-companion-
agency-design.md §3). An entity that works alongside the user is one
that KNOWS WHAT IT OWES them. This module gives Becca that knowledge
with zero new tables: commitments are ``companion_journal`` rows with
``entry_type='commitment'`` (open) → ``'commitment_closed'``.

The journal was chosen deliberately:
  * ``safe_journal`` already validates + quarantines autonomous writes
    (length / injection / quality gates) — commitments inherit that.
  * Embeddings come free for future semantic recall ("what did you
    say you'd do about my notes?").
  * The notes drawer + Today already read the journal — visibility is
    a rendering decision, not new plumbing.
  * Migration 178's revisit columns support future "due for follow-up"
    scans without schema work.

Prompt surfacing reuses the EXISTING Layer-5 ``open_threads`` slot in
``prompt_compose`` (header: "Things you've been sitting with…") — the
placeholder that has been empty since Sprint F was deferred. Open
commitments ARE her open threads.

v1 writers: the act-gap path (she was asked to act and nothing
dispatched — that unmet ask becomes a tracked debt that resurfaces in
her own prompt next turn). v1 closers: any successful tool dispatch
for the same user closes the most recent open entry (the retry-
succeeded assumption — correct in the common case, self-healing via
TTL when wrong).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

ENTRY_OPEN = "commitment"
ENTRY_CLOSED = "commitment_closed"

# Open commitments older than this stop surfacing in the prompt (and
# get auto-closed on next ledger read). A day-old "I'll check the
# news" is stale guilt, not a live thread.
OPEN_TTL_S = 24 * 3600.0

# Prompt layer budget — open_threads renders top 2; we fetch a couple
# extra for the close-matching logic.
_FETCH_LIMIT = 6


def _conn_from_runtime(runtime: Any):
    """Best-effort aiosqlite connection from the runtime (mirrors
    grove_match._conn_from_runtime)."""
    if runtime is None:
        return None
    sm = getattr(runtime, "state_manager", None)
    if sm is None:
        app_state = getattr(runtime, "_app_state", None)
        if app_state is not None:
            sm = getattr(app_state, "state_manager", None)
    if sm is None:
        return None
    backend = getattr(sm, "backend", None)
    return getattr(backend, "conn", None) if backend else None


async def record_unmet_ask(
    runtime: Any,
    *,
    user_id: str,
    asked_text: str,
    source: str = "act_gap",
) -> int | None:
    """Record "they asked me to do X and it didn't happen" as an open
    commitment. Routed through ``safe_journal`` so the validation
    pipeline applies. Returns the journal entry id or None.
    """
    if runtime is None or not user_id or not (asked_text or "").strip():
        return None
    memory = getattr(runtime, "memory", None)
    if memory is None or not hasattr(memory, "safe_journal"):
        return None
    content = (
        f"They asked me to: {asked_text.strip()[:240]} — "
        f"and it hasn't happened yet."
    )
    try:
        entry_id = await memory.safe_journal(
            content,
            source=source,
            user_id=user_id,
            entry_type=ENTRY_OPEN,
            embed=False,  # short operational rows; semantic recall later
            origin={"source": "commitment", "detail": "from our chat"},
        )
        log.info(
            "commitment_recorded",
            user_id=user_id, entry_id=entry_id, source=source,
            text_preview=asked_text[:80],
        )
        return entry_id
    except Exception:  # noqa: BLE001 — the ledger must never break a turn
        log.warning("commitment_record_failed", exc_info=True)
        return None


def _age_phrase(created_at: str) -> str:
    """Human age for the prompt line. created_at is the journal's
    ISO/SQLite timestamp; failures degrade to empty."""
    try:
        raw = (created_at or "").replace("Z", "+00:00")
        ts = datetime.fromisoformat(raw)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        secs = (datetime.now(UTC) - ts).total_seconds()
        if secs < 90:
            return "moments ago"
        if secs < 3600:
            return f"{int(secs // 60)} minutes ago"
        return f"{int(secs // 3600)} hours ago"
    except Exception:  # noqa: BLE001
        return ""


async def open_threads(
    runtime: Any,
    *,
    user_id: str,
    companion_id: str = "",
    limit: int = 2,
) -> list[str]:
    """Open commitments rendered as prompt-ready thread lines.

    This IS the data source for prompt_compose's Layer-5
    ``open_threads`` slot (empty placeholder since Sprint F). Stale
    entries past OPEN_TTL_S are auto-closed during the read so the
    ledger self-heals.
    """
    conn = _conn_from_runtime(runtime)
    if conn is None or not user_id:
        return []
    cid = companion_id or getattr(runtime, "companion_id", "") or "becca"
    try:
        cursor = await conn.execute(
            "SELECT id, content, created_at FROM companion_journal "
            "WHERE companion_id = ? AND user_id = ? AND entry_type = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (cid, user_id, ENTRY_OPEN, _FETCH_LIMIT),
        )
        rows = await cursor.fetchall()
    except Exception:  # noqa: BLE001
        log.debug("commitment_read_failed", exc_info=True)
        return []

    out: list[str] = []
    stale_ids: list[int] = []
    now = time.time()
    for row in rows:
        entry_id, content, created_at = row[0], row[1] or "", row[2] or ""
        try:
            raw = created_at.replace("Z", "+00:00")
            ts = datetime.fromisoformat(raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if now - ts.timestamp() > OPEN_TTL_S:
                stale_ids.append(entry_id)
                continue
        except Exception:  # noqa: BLE001 — unparseable age: keep, no phrase
            log.debug("commitment_age_unparseable", exc_info=True)
        if len(out) < limit:
            age = _age_phrase(created_at)
            line = content if not age else f"{content} ({age})"
            out.append(line)

    if stale_ids:
        try:
            placeholders = ",".join("?" for _ in stale_ids)
            await conn.execute(
                f"UPDATE companion_journal SET entry_type = ? "
                f"WHERE id IN ({placeholders})",
                (ENTRY_CLOSED, *stale_ids),
            )
            await conn.commit()
            log.info("commitments_auto_closed_stale", count=len(stale_ids))
        except Exception:  # noqa: BLE001
            log.debug("commitment_stale_close_failed", exc_info=True)
    return out


async def close_latest(
    runtime: Any,
    *,
    user_id: str,
    companion_id: str = "",
    reason: str = "dispatch_succeeded",
) -> bool:
    """Close the most recent open commitment for this user.

    v1 closure policy: a successful dispatch right after an unmet ask
    is, in the common case, the retry that satisfied it. Wrong-close
    risk is bounded by the TTL (stale entries vanish anyway) and the
    benefit — she stops apologizing for things she already did — is
    immediate.
    """
    conn = _conn_from_runtime(runtime)
    if conn is None or not user_id:
        return False
    cid = companion_id or getattr(runtime, "companion_id", "") or "becca"
    try:
        cursor = await conn.execute(
            "SELECT id FROM companion_journal "
            "WHERE companion_id = ? AND user_id = ? AND entry_type = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (cid, user_id, ENTRY_OPEN),
        )
        row = await cursor.fetchone()
        if row is None:
            return False
        await conn.execute(
            "UPDATE companion_journal SET entry_type = ? WHERE id = ?",
            (ENTRY_CLOSED, row[0]),
        )
        await conn.commit()
        log.info(
            "commitment_closed",
            user_id=user_id, entry_id=row[0], reason=reason,
        )
        return True
    except Exception:  # noqa: BLE001
        log.debug("commitment_close_failed", exc_info=True)
        return False
