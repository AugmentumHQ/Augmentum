"""Signal aggregator — daily pass that ingests user-visible signals from
existing user-scoped tables into the unified ``signal_events`` table.

Goal: find out empirically whether a cross-source dedup'd inbox surfaces
patterns the operator doesn't already know about. The v1 substrate writes;
the "do I want a UI?" decision waits on the data we accumulate.

v1 sources (both pure SQL, no container IO):

1. ``bug_finder_runs`` — every completed run with ``findings_confirmed > 0``
   becomes one signal with category ``bug``. Fingerprint is the run id, so
   re-ingestion is a no-op via the UNIQUE constraint.

2. ``companion_journal`` — entries where ``entry_type='noticing'`` become
   signals with category ``gap`` (Becca noticed a missing-thing pattern);
   entries with ``affect_tag in ('not_okay','unsure')`` become signals with
   category ``drift``. Fingerprint is the entry id. Entries without
   ``user_id`` are skipped in v1 (resolving companion → user is deferred
   work; the journal entries with user_id attached are the high-signal
   ones anyway).

Deferred sources, documented so future-me doesn't re-derive them:

* **Coder observation ledgers** — live as JSONL inside docker volumes;
  reading them needs the container running. Wait until Phase 1 PR-1.2
  pushes ``.augmentum/observations.jsonl`` into the bare repo on the
  host filesystem; then ingest by walking ``{data_dir}/projects/*/*.git``.

* **structlog warning aggregation** — no aggregation layer exists today.
  A separate substrate (probably a log handler + rolling counts table)
  is its own design pass.

* **Audit script findings** — currently emits to stdout, not a queryable
  source. Pipe it into a table first, then ingest from there.

Dedup contract: every helper returns rows with ``fingerprint`` unique
within ``(user_id, source)``. The DB-level UNIQUE index + INSERT OR
IGNORE makes re-runs idempotent without needing a watermark.

Categories are free TEXT in the schema (no CHECK constraint) so the
vocabulary can grow without a migration. Today's vocabulary:
``bug`` | ``gap`` | ``drift`` | ``gotcha`` | ``constraint`` | ``polish`` | ``other``.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass

import aiosqlite

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Row shape — what each source helper returns, what the writer consumes.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SignalRow:
    """One candidate row for ``signal_events``.

    The aggregator manufactures ``id`` and timestamps at write time;
    source helpers only need to produce the semantic fields.
    """
    user_id: str
    source: str
    category: str
    fingerprint: str
    summary: str
    details: dict


# ---------------------------------------------------------------------------
# Writer — INSERT OR IGNORE keyed on the dedup UNIQUE index.
#
# Why INSERT OR IGNORE rather than UPSERT with occurrence_count++:
# v1 fingerprints are row-id-based, so each source row produces exactly one
# signal_events row regardless of how many aggregator passes run. Bumping
# occurrence_count on every pass would conflate "aggregator re-saw this"
# with "underlying event recurred", which the row-id fingerprint can't
# distinguish. A future iteration with content-hash fingerprints will be
# able to use occurrence_count meaningfully — and will need an UPSERT.
# ---------------------------------------------------------------------------


async def _write_rows(
    conn: aiosqlite.Connection,
    rows: list[SignalRow],
    *,
    now_ms: int,
) -> int:
    """Insert candidate rows. Returns the count of rows actually inserted.

    Skipped (already-present) rows count as 0 — same fingerprint means the
    signal is already known. Status is left untouched on existing rows so
    a previously dismissed/resolved signal does NOT reopen on a later pass.
    """
    if not rows:
        return 0

    inserted = 0
    for row in rows:
        try:
            cursor = await conn.execute(
                """
                INSERT OR IGNORE INTO signal_events (
                    id, user_id, source, category, fingerprint,
                    summary, details_json,
                    first_seen_at, last_seen_at, occurrence_count, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 'open')
                """,
                (
                    uuid.uuid4().hex,
                    row.user_id,
                    row.source,
                    row.category,
                    row.fingerprint,
                    row.summary,
                    json.dumps(row.details, separators=(",", ":")),
                    now_ms,
                    now_ms,
                ),
            )
            if cursor.rowcount > 0:
                inserted += 1
        except Exception:
            log.warning(
                "signals.write_failed",
                source=row.source,
                fingerprint=row.fingerprint,
                exc_info=True,
            )
    await conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Source 1 — bug_finder_runs
#
# Each run with at least one confirmed finding is one signal. The summary
# is a short human-readable line; ``details`` carries enough provenance
# (run_id + counts + stop_reason) for a future UI to deep-link back to
# the full report_json without re-querying.
# ---------------------------------------------------------------------------


async def _ingest_bug_finder(
    conn: aiosqlite.Connection,
    user_id: str,
) -> list[SignalRow]:
    rows: list[SignalRow] = []
    cursor = await conn.execute(
        """
        SELECT run_id, findings_confirmed, findings_fixed, findings_fix_failed,
               stop_reason, started_at, completed_at
        FROM bug_finder_runs
        WHERE user_id = ?
          AND findings_confirmed > 0
        """,
        (user_id,),
    )
    async for raw in cursor:
        run_id, confirmed, fixed, fix_failed, stop_reason, started_at, completed_at = raw
        outstanding = max(0, int(confirmed) - int(fixed or 0))
        if outstanding == 0:
            # All confirmed findings were auto-fixed — the run resolved itself;
            # nothing for the inbox to surface.
            continue
        summary = (
            f"Bug Finder run found {confirmed} confirmed issue(s); "
            f"{outstanding} still outstanding"
        )
        rows.append(SignalRow(
            user_id=user_id,
            source="bug_finder",
            category="bug",
            fingerprint=f"bug_finder:{run_id}",
            summary=summary,
            details={
                "run_id": run_id,
                "findings_confirmed": int(confirmed),
                "findings_fixed": int(fixed or 0),
                "findings_fix_failed": int(fix_failed or 0),
                "outstanding": outstanding,
                "stop_reason": stop_reason or "",
                "started_at": int(started_at or 0),
                "completed_at": int(completed_at or 0),
            },
        ))
    await cursor.close()
    return rows


# ---------------------------------------------------------------------------
# Source 2 — companion_journal
#
# Two flavors map to two categories:
#   * entry_type='noticing'           → category='gap'
#   * affect_tag in ('not_okay','unsure') → category='drift'
#
# Same entry can satisfy both conditions — produce two rows with distinct
# fingerprints (suffixed by the channel) so dedup doesn't collapse them.
# Entries without user_id are skipped: the high-signal entries already
# carry user_id; the rest need a companion→user resolver that's its own
# design pass.
#
# We DON'T window by time — the UNIQUE index makes re-ingestion free, so
# a full-history scan on the first pass backfills everything and later
# passes only INSERT the new rows.
# ---------------------------------------------------------------------------


_JOURNAL_TRIGGERING_AFFECTS: frozenset[str] = frozenset({"not_okay", "unsure"})

# Cap on summary text length — journals can be long-form prose, but the
# inbox row just wants the gist. Truncate to 160 chars, ellipsis if cut.
_SUMMARY_CAP = 160


def _truncate_for_summary(text: str) -> str:
    """One-line, capped summary derived from journal content."""
    flat = " ".join((text or "").split())
    if len(flat) <= _SUMMARY_CAP:
        return flat
    return flat[: _SUMMARY_CAP - 1].rstrip() + "…"


async def _ingest_companion_journal(
    conn: aiosqlite.Connection,
    user_id: str,
) -> list[SignalRow]:
    rows: list[SignalRow] = []
    cursor = await conn.execute(
        """
        SELECT id, companion_id, entry_type, content, affect_tag, created_at
        FROM companion_journal
        WHERE user_id = ?
          AND (entry_type = 'noticing' OR affect_tag IN ('not_okay', 'unsure'))
        """,
        (user_id,),
    )
    async for raw in cursor:
        entry_id, companion_id, entry_type, content, affect_tag, created_at = raw
        base_details = {
            "entry_id": int(entry_id),
            "companion_id": companion_id,
            "entry_type": entry_type or "",
            "affect_tag": affect_tag or "",
            "created_at": created_at or "",
        }
        summary_text = _truncate_for_summary(content or "")

        if entry_type == "noticing":
            rows.append(SignalRow(
                user_id=user_id,
                source="companion_journal",
                category="gap",
                fingerprint=f"journal:noticing:{entry_id}",
                summary=f"Becca noticed: {summary_text}" if summary_text else "Becca noticed something",
                details=base_details,
            ))

        if affect_tag in _JOURNAL_TRIGGERING_AFFECTS:
            rows.append(SignalRow(
                user_id=user_id,
                source="companion_journal",
                category="drift",
                fingerprint=f"journal:affect:{entry_id}",
                summary=(
                    f"Becca felt {affect_tag}: {summary_text}"
                    if summary_text
                    else f"Becca felt {affect_tag} (no content captured)"
                ),
                details=base_details,
            ))
    await cursor.close()
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def aggregate_for_user(
    conn: aiosqlite.Connection,
    user_id: str,
    *,
    now_ms: int | None = None,
) -> dict[str, int]:
    """Run every source against one user. Returns {source: inserted_count}."""
    if not user_id:
        return {}
    ts = now_ms if now_ms is not None else int(time.time() * 1000)

    counts: dict[str, int] = {}

    bug_rows = await _ingest_bug_finder(conn, user_id)
    counts["bug_finder"] = await _write_rows(conn, bug_rows, now_ms=ts)

    journal_rows = await _ingest_companion_journal(conn, user_id)
    counts["companion_journal"] = await _write_rows(conn, journal_rows, now_ms=ts)

    log.info(
        "signals.aggregated_for_user",
        user_id=user_id,
        bug_finder=counts["bug_finder"],
        companion_journal=counts["companion_journal"],
    )
    return counts


async def _list_user_ids(conn: aiosqlite.Connection) -> list[str]:
    """All real user ids. Excludes the anon sentinel — anon rows shouldn't
    accrue signals."""
    cursor = await conn.execute(
        "SELECT id FROM users WHERE id != '' AND id != 'anon'"
    )
    rows = await cursor.fetchall()
    await cursor.close()
    return [r[0] for r in rows if r and r[0]]


async def aggregate_all_users(
    conn: aiosqlite.Connection,
    *,
    now_ms: int | None = None,
) -> dict[str, dict[str, int]]:
    """Daily entry point. Runs every source against every user.

    Returns {user_id: {source: inserted_count}}. An empty dict on no users.
    """
    ts = now_ms if now_ms is not None else int(time.time() * 1000)
    user_ids = await _list_user_ids(conn)
    results: dict[str, dict[str, int]] = {}
    for uid in user_ids:
        try:
            results[uid] = await aggregate_for_user(conn, uid, now_ms=ts)
        except Exception:
            log.warning("signals.aggregate_user_failed", user_id=uid, exc_info=True)
            results[uid] = {}
    total_inserted = sum(c for u in results.values() for c in u.values())
    log.info(
        "signals.aggregated_all_users",
        users=len(user_ids),
        total_inserted=total_inserted,
    )
    return results
