"""Demand-side debt — lived user friction (signal_events) as structural targets.

Locks the load-bearing behaviors:
* strict user isolation (never reads another user's or the anon row's friction);
* only OPEN signals become targets (dismissed/resolved drop out);
* every demand target is STRUCTURAL / CONFIRM_HUMAN / origin=demand — never the
  mechanical auto-lane;
* two signals of the same category get UNIQUE card keys (no scanner.metric
  collision);
* occurrence_count rides as the weight; details_json context is included whole
  (never truncated);
* a read failure degrades to [] (audit-only), never raises;
* the loop merges demand into the needs-you lane ahead of audit structural.
"""

from __future__ import annotations

import json
import time

import aiosqlite

from augmentum.selfedit import demand
from augmentum.selfedit.debt import CONFIRM_HUMAN, KIND_MECHANICAL, KIND_STRUCTURAL


async def _db_with_signals(rows):
    """A minimal signal_events table + rows. Each row: dict of column→value."""
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        """
        CREATE TABLE signal_events (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL, source TEXT NOT NULL,
            category TEXT NOT NULL, fingerprint TEXT NOT NULL, summary TEXT NOT NULL,
            details_json TEXT NOT NULL DEFAULT '{}', first_seen_at INTEGER NOT NULL,
            last_seen_at INTEGER NOT NULL, occurrence_count INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'open', resolved_at INTEGER
        )
        """
    )
    now = int(time.time() * 1000)
    for i, r in enumerate(rows):
        await conn.execute(
            """INSERT INTO signal_events
               (id, user_id, source, category, fingerprint, summary, details_json,
                first_seen_at, last_seen_at, occurrence_count, status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (r.get("id", f"s{i}"), r["user_id"], r.get("source", "bug_finder"),
             r.get("category", "bug"), r.get("fingerprint", f"fp{i}"),
             r.get("summary", "something broke"), r.get("details_json", "{}"),
             now, now, r.get("occurrence_count", 1), r.get("status", "open")),
        )
    await conn.commit()
    return conn


# ---------------------------------------------------------------------------
# isolation — the ground rule
# ---------------------------------------------------------------------------

async def test_reads_only_this_user():
    conn = await _db_with_signals([
        {"user_id": "u1", "summary": "u1 friction"},
        {"user_id": "u2", "summary": "u2 friction"},
    ])
    try:
        rows = await demand.read_open_signals(conn, user_id="u1")
        assert len(rows) == 1 and rows[0]["summary"] == "u1 friction"
    finally:
        await conn.close()


async def test_empty_user_reads_nothing():
    conn = await _db_with_signals([{"user_id": "u1", "summary": "x"}])
    try:
        assert await demand.read_open_signals(conn, user_id="") == []
        assert await demand.demand_targets(conn, user_id="") == []
    finally:
        await conn.close()


async def test_only_open_signals():
    conn = await _db_with_signals([
        {"user_id": "u1", "summary": "open one", "status": "open"},
        {"user_id": "u1", "summary": "dismissed", "status": "dismissed"},
        {"user_id": "u1", "summary": "resolved", "status": "resolved"},
    ])
    try:
        targets = await demand.demand_targets(conn, user_id="u1")
        assert len(targets) == 1 and "open one" in targets[0].title
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# mapping — structural, unique, weighted, full-context
# ---------------------------------------------------------------------------

def test_target_is_structural_human_and_tagged():
    t = demand.signal_to_target(
        {"category": "bug", "source": "bug_finder", "summary": "crash on save",
         "fingerprint": "abc", "occurrence_count": 3})
    assert t.kind == KIND_STRUCTURAL
    assert t.kind != KIND_MECHANICAL
    assert t.confirms_via == CONFIRM_HUMAN
    assert t.origin == "demand"
    assert t.scanner == "demand"
    assert t.count == 3                       # occurrence_count = weight
    assert "crash on save" in t.title


def test_same_category_signals_get_unique_keys():
    a = demand.signal_to_target({"category": "bug", "fingerprint": "fp-a", "summary": "A"})
    b = demand.signal_to_target({"category": "bug", "fingerprint": "fp-b", "summary": "B"})
    # both are 'bug' but must NOT collide on scanner.metric (the card key)
    assert a.metric != b.metric
    assert f"{a.scanner}.{a.metric}" != f"{b.scanner}.{b.metric}"


def test_details_context_included_whole():
    details = {"run_id": "r1", "findings_confirmed": 7, "affect_tag": "not_okay"}
    t = demand.signal_to_target(
        {"category": "gap", "summary": "s", "fingerprint": "f",
         "details_json": json.dumps(details)})
    for k, v in details.items():
        assert str(k) in t.objective and str(v) in t.objective


def test_unknown_category_survives():
    # category vocabulary is free TEXT — a new one must appear, not be dropped
    t = demand.signal_to_target({"category": "sadness", "summary": "hmm", "fingerprint": "f"})
    assert t.metric.startswith("sadness:")
    assert t.origin == "demand"


# ---------------------------------------------------------------------------
# ordering + resilience
# ---------------------------------------------------------------------------

async def test_most_recurring_first():
    conn = await _db_with_signals([
        {"user_id": "u1", "summary": "rare", "fingerprint": "a", "occurrence_count": 1},
        {"user_id": "u1", "summary": "constant", "fingerprint": "b", "occurrence_count": 9},
    ])
    try:
        targets = await demand.demand_targets(conn, user_id="u1")
        assert [t.count for t in targets] == [9, 1]  # most-recurring first
    finally:
        await conn.close()


async def test_missing_table_degrades_to_empty():
    # An old install without the signal_events table → [] not a crash.
    conn = await aiosqlite.connect(":memory:")
    try:
        assert await demand.read_open_signals(conn, user_id="u1") == []
        assert await demand.demand_targets(conn, user_id="u1") == []
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# loop integration — demand joins needs-you, never the auto-lane
# ---------------------------------------------------------------------------

async def test_loop_merges_demand_into_structural_ahead_of_audit():
    from augmentum.selfedit import loop as _loop
    from augmentum.selfedit.debt import DebtTarget

    demand_t = DebtTarget(scanner="demand", metric="bug:fp1", count=5,
                          kind=KIND_STRUCTURAL, title="from the user: X",
                          objective="user friction", confirms_via=CONFIRM_HUMAN,
                          origin="demand")

    # a synthetic audit with one structural finding, no mechanical
    audit_json = json.dumps({
        "score": 80.0,
        "metrics": {"dead_code": {"orphaned_endpoints": 2}},
    })

    async def _audit(_dir):
        return audit_json

    report = await _loop.run_debt_loop(
        repo_dir=".", user_id="u1", conn=None, dry_run=True,
        live_audit_runner=_audit, candidate_audit_runner=_audit,
        demand=[demand_t],
    )
    d = report.to_dict()
    structural = d["structural"]
    assert structural[0]["origin"] == "demand"           # demand ranks first
    assert any(s["scanner"] == "dead_code" for s in structural)  # audit still present
    # demand never enters the mechanical auto-lane
    assert all(t["scanner"] != "demand" for t in d["targets"])
