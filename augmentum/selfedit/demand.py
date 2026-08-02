"""Demand-side debt — lived user friction as self-edit targets.

The self-edit loop has only ever been *supply-driven*: every objective comes
from the audit (what the scanners see) or from the human typing an ask. But the
platform already records what users *feel* — and aggregates it into exactly the
right shape. ``signal_events`` (migration 206, written by
``augmentum/signals/aggregator.py``) is the unified friction inbox: bug-finder
confirmed findings, companion-noticed gaps, affect drift — deduped by
fingerprint, categorized (bug/gap/drift/gotcha/constraint/polish), and
occurrence-counted. It was built, wired to write, and then **never read by
anyone**.

This module is the missing reader: open ``signal_events`` rows → structural
``DebtTarget``s that surface in the Debt lane's "needs you" section, beside the
audit's findings. The engine stops fixing orphaned CSS while a user fumbles the
same broken flow daily — because the flow now appears on the same list.

Discipline:
* **Structural, always.** Friction is a report, not a mechanically-confirmable
  fix — every demand target is ``KIND_STRUCTURAL`` / ``CONFIRM_HUMAN`` and is
  *surfaced*, never auto-attempted. The audit's mechanical auto-lane is
  untouched.
* **Strict user isolation.** ``signal_events`` is user-scoped; the reader filters
  ``WHERE user_id = ?`` with NO ``OR user_id IS NULL`` fallback (isolation is a
  ground rule, not a tunable). An empty user_id reads nothing.
* **Read-only.** Never mutates ``signal_events`` (status transitions belong to the
  future inbox UI). A DB hiccup returns an empty list — demand degrades to
  "audit-only," it never breaks the loop.
* **Data, not code.** Rows become ``DebtTarget``s and flow through the SAME
  triage/report/annotate path the audit's structural findings already use.
"""

from __future__ import annotations

import json
from typing import Any

from augmentum.selfedit.debt import CONFIRM_HUMAN, KIND_STRUCTURAL, DebtTarget
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Only OPEN signals are actionable debt. dismissed/resolved dropped out already.
_OPEN_STATUS = "open"

# category → a human title stem, so the Debt card reads naturally. Unknown
# categories fall through to the raw category (the vocabulary is free TEXT by
# design, so a new category auto-appears rather than being silently dropped).
_CATEGORY_TITLES = {
    "bug": "User-hit bug",
    "gap": "Capability gap the user noticed",
    "drift": "Something felt off to the user",
    "gotcha": "A gotcha the user tripped on",
    "constraint": "A constraint the user ran into",
    "polish": "Rough edge the user felt",
    "other": "User friction",
}


def _title_for(category: str, summary: str) -> str:
    stem = _CATEGORY_TITLES.get(category, category or "User friction")
    s = (summary or "").strip().replace("\n", " ")
    return f"{stem}: {s[:80]}" if s else stem


def _objective_for(row: dict) -> str:
    """The scoped instruction handed to the agent (or read by the human). It
    describes the FRICTION and its provenance — it never guesses the fix, the
    same contract the audit's structural objectives follow."""
    summary = (row.get("summary") or "").strip()
    source = row.get("source") or "unknown"
    count = int(row.get("occurrence_count") or 1)
    seen = f"seen {count}×" if count > 1 else "seen once"
    lines = [
        f"Users are hitting this ({seen}, reported by '{source}'): {summary}",
        "",
        "This is lived friction surfaced from the signal inbox, not a scanner "
        "finding — investigate what's actually breaking or missing for the user "
        "and decide how to address it. A judgment call: propose, don't auto-fix.",
    ]
    details = row.get("details_json")
    if details:
        try:
            parsed = json.loads(details) if isinstance(details, str) else details
            if isinstance(parsed, dict) and parsed:
                # Full context, never truncated — the agent/human sees everything
                # the source recorded (deep-link ids, affect tags, finding counts).
                kv = ", ".join(f"{k}={v}" for k, v in parsed.items())
                lines += ["", f"Source context: {kv}"]
        except (ValueError, TypeError):
            pass
    return "\n".join(lines)


def signal_to_target(row: dict) -> DebtTarget:
    """One open ``signal_events`` row → a structural demand DebtTarget.

    ``count`` carries occurrence_count (how often the friction recurred = its
    weight). ``metric`` is ``<category>:<fingerprint>`` — the category keeps the
    row readable and groupable, and the fingerprint makes each signal a UNIQUE
    card (two 'bug' signals must not collide on one ``scanner.metric`` key the
    way aggregated audit findings can't, since each audit class is one target
    but each signal is its own card)."""
    category = str(row.get("category") or "other").strip().lower()
    source = str(row.get("source") or "unknown")
    count = int(row.get("occurrence_count") or 1)
    # fingerprint is the dedup key within (user_id, source) — unique per signal;
    # fall back to the row id, then a summary hash, so the card key is never blank.
    fp = str(row.get("fingerprint") or row.get("id") or abs(hash(row.get("summary", "")))) [:16]
    return DebtTarget(
        scanner="demand", metric=f"{category}:{fp}", count=count,
        kind=KIND_STRUCTURAL, title=_title_for(category, row.get("summary") or ""),
        objective=_objective_for(row), confirms_via=CONFIRM_HUMAN,
        note=f"from the user (source: {source}) — surfaced, never auto-touched",
        origin="demand",
    )


async def read_open_signals(
    conn: Any, *, user_id: str, limit: int = 50,
) -> list[dict]:
    """Open ``signal_events`` for ONE user, most-recurring & most-recent first.

    Strict isolation: an empty ``user_id`` returns nothing (never reads another
    user's or the anon row's friction). Read-only. Best-effort — any DB error
    (e.g. the table absent on an old install) logs and returns []."""
    if not user_id:
        return []
    try:
        cur = await conn.execute(
            """
            SELECT id, user_id, source, category, fingerprint, summary,
                   details_json, first_seen_at, last_seen_at, occurrence_count,
                   status
            FROM signal_events
            WHERE user_id = ? AND status = ?
            ORDER BY occurrence_count DESC, last_seen_at DESC
            LIMIT ?
            """,
            (user_id, _OPEN_STATUS, int(limit)),
        )
        rows = await cur.fetchall()
        await cur.close()
    except Exception as exc:  # noqa: BLE001 — demand degrades to audit-only
        log.warning("selfedit_demand_read_failed", error=repr(exc))
        return []
    cols = ("id", "user_id", "source", "category", "fingerprint", "summary",
            "details_json", "first_seen_at", "last_seen_at", "occurrence_count",
            "status")
    return [dict(zip(cols, r, strict=False)) for r in rows]


async def demand_targets(
    conn: Any, *, user_id: str, limit: int = 50,
) -> list[DebtTarget]:
    """The demand-side worklist: open friction for ``user_id`` as structural
    DebtTargets, most-recurring first. ``conn`` is the MAIN DB connection
    (``signal_events`` lives there, not in the growth DB). Empty on no user, no
    signals, or any read failure — the loop simply stays audit-only."""
    rows = await read_open_signals(conn, user_id=user_id, limit=limit)
    targets = [signal_to_target(r) for r in rows]
    if targets:
        log.info("selfedit_demand_targets", user_id=user_id, count=len(targets))
    return targets
