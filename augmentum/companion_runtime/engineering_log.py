"""Engineering continuity — durable memory of significant collaborative
coding work, so Becca opens with it next session instead of forgetting.

The companion is the *persistent* layer; an external coding agent (Claude
Code / Codex via the planned ExternalCoderDriver) or the native coder is the
*stateless* muscle. When a meaningful piece of work completes, this records it
in Becca's own memory in the user's framing — so the NEXT session she can open
with "last week we had Claude refactor the media store — want to pick that back
up?" instead of the work evaporating. That recall loop is where the persistence
stops being a notes file and starts feeling like presence.

Built exactly like ``commitments.py`` — **zero new tables**: rows are
``companion_journal`` entries with ``entry_type='engineering_log'``, written
through ``memory.safe_journal`` so they inherit the validation / injection /
quality gates AND get embeddings for future semantic recall for free. The notes
drawer + Today already read the journal, so visibility is a rendering decision,
not new plumbing.

Writer: the future ``ExternalCoderDriver`` calls ``record_engineering_outcome``
when a delegated run completes (the native coder can call it too — it's
engine-agnostic by design). Reader: ``recent_engineering`` returns prompt-ready
lines for the companion's prompt (see ``prompt_compose`` Layer 5.7).

See ``docs/superpowers/specs/2026-06-21-companion-external-coder-drivers-design.md``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

ENTRY_TYPE = "engineering_log"

# Engineering threads stay relevant far longer than a "I'll check the news"
# commitment — a refactor we did last week is still worth opening with. Two
# weeks balances "still live" against "stale guilt".
RECENT_TTL_S = 14 * 24 * 3600.0

# Fetch a few extra beyond the prompt's top-N so TTL filtering still leaves
# enough to render.
_FETCH_LIMIT = 8


def _conn_from_runtime(runtime: Any):
    """Best-effort aiosqlite connection from the runtime (mirrors
    ``commitments._conn_from_runtime``)."""
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


def _compose_content(*, task: str, outcome: str, engine: str, framing: str) -> str:
    """Build the journal line. Engine-agnostic: ``engine`` is a friendly label
    ("Claude Code", "Codex", "" for the native coder)."""
    who = f"{engine} " if engine else ""
    line = f"We had {who}work on: {task.strip()[:200]}."
    if outcome.strip():
        line += f" Outcome: {outcome.strip()[:200]}."
    if framing.strip():
        # The user's own words about why it mattered — the bit that makes recall
        # feel personal rather than a log entry.
        line += f" ({framing.strip()[:160]})"
    return line


async def record_engineering_outcome(
    runtime: Any,
    *,
    user_id: str,
    task: str,
    outcome: str = "",
    engine: str = "",
    framing: str = "",
    resume_ref: str = "",
    companion_id: str = "",
) -> int | None:
    """Record a completed piece of collaborative engineering work as a durable,
    recallable journal entry. Returns the journal entry id or None.

    ``resume_ref`` (a workspace/session id) is stashed in ``origin`` so a future
    "pick it back up" can jump straight back to that run. Never raises — a
    memory write must not break the run that produced it.
    """
    if runtime is None or not user_id or not (task or "").strip():
        return None
    memory = getattr(runtime, "memory", None)
    if memory is None or not hasattr(memory, "safe_journal"):
        return None
    content = _compose_content(
        task=task, outcome=outcome, engine=engine, framing=framing,
    )
    try:
        entry_id = await memory.safe_journal(
            content,
            source="engineering",
            user_id=user_id,
            entry_type=ENTRY_TYPE,
            embed=True,  # enable semantic recall ("what did we do about the media store?")
            origin={
                "source": "engineering",
                "engine": engine or "native",
                "resume_ref": resume_ref,
            },
        )
        log.info(
            "engineering_outcome_recorded",
            user_id=user_id, entry_id=entry_id, engine=engine or "native",
            task_preview=task[:80],
        )
        return entry_id
    except Exception:  # noqa: BLE001 — the ledger must never break a turn
        log.warning("engineering_outcome_record_failed", exc_info=True)
        return None


def _age_phrase(created_at: str) -> str:
    """Human age for the prompt line. Mirrors ``commitments._age_phrase`` but
    extends to days/weeks since engineering threads live longer."""
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
        if secs < 24 * 3600:
            return f"{int(secs // 3600)} hours ago"
        days = int(secs // (24 * 3600))
        if days < 14:
            return "yesterday" if days == 1 else f"{days} days ago"
        return f"{days // 7} weeks ago"
    except Exception:  # noqa: BLE001
        return ""


async def recent_engineering(
    runtime: Any,
    *,
    user_id: str,
    companion_id: str = "",
    limit: int = 2,
) -> list[str]:
    """Recent collaborative engineering work, rendered as prompt-ready lines for
    the companion to (optionally) open with. Entries past ``RECENT_TTL_S`` are
    skipped. Never raises.
    """
    conn = _conn_from_runtime(runtime)
    if conn is None or not user_id:
        return []
    cid = companion_id or getattr(runtime, "companion_id", "") or "becca"
    try:
        cursor = await conn.execute(
            "SELECT content, created_at FROM companion_journal "
            "WHERE companion_id = ? AND user_id = ? AND entry_type = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (cid, user_id, ENTRY_TYPE, _FETCH_LIMIT),
        )
        rows = await cursor.fetchall()
    except Exception:  # noqa: BLE001
        log.debug("engineering_recent_read_failed", exc_info=True)
        return []

    out: list[str] = []
    now = datetime.now(UTC).timestamp()
    for row in rows:
        content, created_at = row[0] or "", row[1] or ""
        try:
            raw = created_at.replace("Z", "+00:00")
            ts = datetime.fromisoformat(raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
            if now - ts.timestamp() > RECENT_TTL_S:
                continue
        except Exception:  # noqa: BLE001 — unparseable age: keep, no phrase
            log.debug("engineering_age_unparseable", exc_info=True)
        if len(out) < limit:
            age = _age_phrase(created_at)
            out.append(content if not age else f"{content} ({age})")
    return out
