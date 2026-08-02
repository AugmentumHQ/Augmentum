"""Verification-spine oracle telemetry (spec 2026-07-06, Phase 2).

Pure-classifier tests plus the ledger fold: oracle calls classified from
the tool_result stream, `no_oracle_done` computed against the last write,
one `oracle_summary` event at turn close, and the store-level rollup that
backs GET /api/coder/oracle-stats.
"""
from __future__ import annotations

import aiosqlite
import pytest

from augmentum.coder.oracle_telemetry import (
    classify_oracle_kind,
    classify_outcome,
    summarize,
)


# ---------------------------------------------------------------------------
# classify_oracle_kind
# ---------------------------------------------------------------------------

def test_kind_direct_oracle_tools():
    assert classify_oracle_kind("test_run", {}) == "test"
    assert classify_oracle_kind("browser_verify", {}) == "browser"
    assert classify_oracle_kind("browser_wait", {}) == "browser"
    assert classify_oracle_kind("service_probe", {}) == "probe"
    assert classify_oracle_kind("http_probe", {}) == "probe"


def test_kind_shell_check_commands():
    assert classify_oracle_kind("shell_exec", {"command": "python -m pytest -x"}) == "shell_check"
    assert classify_oracle_kind("shell_exec", {"command": "ruff check augmentum/"}) == "shell_check"
    assert classify_oracle_kind("shell_exec", {"command": "npm test"}) == "shell_check"
    assert classify_oracle_kind("shell_exec", {"command": "go test ./..."}) == "shell_check"


def test_kind_non_oracle_tools_and_commands():
    assert classify_oracle_kind("file_read", {}) is None
    assert classify_oracle_kind("code_edit", {}) is None
    assert classify_oracle_kind("shell_exec", {"command": "npm install"}) is None
    assert classify_oracle_kind("shell_exec", {"command": "ls -la"}) is None


def test_kind_word_boundary_no_substring_hits():
    # The "form" ∈ "transformers" bug class: substrings must not match.
    assert classify_oracle_kind("shell_exec", {"command": "pip install ruffle-py"}) is None
    assert classify_oracle_kind("shell_exec", {"command": "cat pytest_notes.md"}) is None


# ---------------------------------------------------------------------------
# classify_outcome
# ---------------------------------------------------------------------------

def test_outcome_success_false_is_red():
    assert classify_outcome(success=False, output_preview="whatever") == "red"


def test_outcome_failure_markers_beat_pass_markers():
    # pytest mixed summary must read red even though "passed" appears.
    assert classify_outcome(success=True, output_preview="1 failed, 3 passed in 0.2s") == "red"
    assert classify_outcome(success=True, output_preview="FAILED tests/test_x.py::t - boom") == "red"
    assert classify_outcome(success=True, output_preview="Traceback (most recent call last):") == "red"


def test_outcome_green_and_unknown():
    assert classify_outcome(success=True, output_preview="9 passed in 0.6s") == "green"
    assert classify_outcome(success=True, output_preview="All checks passed!") == "green"
    assert classify_outcome(success=True, output_preview="") == "unknown"
    assert classify_outcome(success=None, output_preview="wrote 3 files") == "unknown"


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------

def _call(seq: int, kind: str = "test", outcome: str = "green") -> dict:
    return {"seq": seq, "kind": kind, "tool": kind, "outcome": outcome}


def test_summarize_verified_after_write():
    s = summarize(wrote=True, last_write_seq=5, oracle_calls=[_call(2), _call(8)])
    assert s["verified_after_last_write"] is True
    assert s["no_oracle_done"] is False
    assert s["last_outcome"] == "green"
    assert s["kinds"] == ["test"]


def test_summarize_stale_oracle_is_no_oracle_done():
    # Oracle ran only BEFORE the final write — the proven claim is stale.
    s = summarize(wrote=True, last_write_seq=9, oracle_calls=[_call(2)])
    assert s["verified_after_last_write"] is False
    assert s["no_oracle_done"] is True


def test_summarize_read_only_turn_is_not_flagged():
    s = summarize(wrote=False, last_write_seq=-1, oracle_calls=[])
    assert s["no_oracle_done"] is False
    assert s["wrote"] is False


# ---------------------------------------------------------------------------
# Ledger fold + store rollup (in-memory SQLite, same schema style as
# tests/test_coder_turn_ledger.py post-migration-200)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE coder_turn_runs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT '',
    project_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',
    strategy TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    prompt_profile TEXT NOT NULL DEFAULT '',
    tooling_profile TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    started_at REAL NOT NULL,
    first_event_at REAL,
    first_useful_action_at REAL,
    completed_at REAL,
    updated_at REAL NOT NULL,
    iterations INTEGER NOT NULL DEFAULT 0,
    tool_calls INTEGER NOT NULL DEFAULT 0,
    parallel_waves INTEGER NOT NULL DEFAULT 0,
    retries INTEGER NOT NULL DEFAULT 0,
    no_response_events INTEGER NOT NULL DEFAULT 0,
    empty_native_content INTEGER NOT NULL DEFAULT 0,
    malformed_tool_calls INTEGER NOT NULL DEFAULT 0,
    commands_run TEXT NOT NULL DEFAULT '[]',
    files_touched TEXT NOT NULL DEFAULT '[]',
    tests_run TEXT NOT NULL DEFAULT '[]',
    browser_checks TEXT NOT NULL DEFAULT '[]',
    finish_reason TEXT NOT NULL DEFAULT '',
    fallback_reason TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL DEFAULT '',
    changed_files TEXT NOT NULL DEFAULT '[]',
    closeout_json TEXT NOT NULL DEFAULT '{}',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    priming_telemetry TEXT NOT NULL DEFAULT '{}',
    input_cost_usd REAL NOT NULL DEFAULT 0,
    output_cost_usd REAL NOT NULL DEFAULT 0,
    cost_model_id TEXT NOT NULL DEFAULT ''
);
CREATE TABLE coder_turn_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    type TEXT NOT NULL,
    phase TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT '',
    payload TEXT NOT NULL DEFAULT '{}',
    user_id TEXT NOT NULL DEFAULT '',
    UNIQUE(run_id, seq)
);
-- finish() bulk-persists the citation ledger here (migration 328).
CREATE TABLE coder_run_citations (
    id INTEGER PRIMARY KEY,
    turn_run_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    workspace_id TEXT NOT NULL DEFAULT '',
    run_id TEXT NOT NULL DEFAULT '',
    tool_call_seq INTEGER NOT NULL DEFAULT 0,
    file TEXT NOT NULL DEFAULT '',
    line_start INTEGER,
    line_end INTEGER,
    evidence_kind TEXT NOT NULL DEFAULT 'write',
    evidence_ref TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


async def _make_ledger(conn):
    from augmentum.coder.ledger import CoderTurnLedger, CoderTurnLedgerStore

    store = CoderTurnLedgerStore(conn)
    ledger = await CoderTurnLedger.start(
        store, user_id="u1", workspace_id="ws1", session_id="s1",
        model="model-a", strategy="native",
    )
    return store, ledger


def _chunk(payload: dict):
    from augmentum.models.base import InternalStreamChunk

    return InternalStreamChunk(augmentum={"phase": "executing", **payload})


async def _tool_roundtrip(ledger, *, tool, tool_input=None, success=True, preview=""):
    tid = f"t{ledger.seq}"
    await ledger.observe_chunk(_chunk({
        "status": "tool_call",
        "tool_call": {"id": tid, "tool": tool, "input": tool_input or {}},
    }))
    await ledger.observe_chunk(_chunk({
        "status": "tool_result",
        "tool_result": {"id": tid, "tool": tool, "success": success,
                        "output_preview": preview},
    }))


@pytest.mark.asyncio
async def test_ledger_folds_oracle_summary_and_emits_event():
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA)
    store, ledger = await _make_ledger(conn)

    # write → verify (green): the healthy shape.
    await _tool_roundtrip(ledger, tool="code_edit",
                          tool_input={"path": "app.py"})
    await _tool_roundtrip(ledger, tool="test_run",
                          tool_input={"command": "pytest -x"},
                          preview="4 passed in 0.3s")
    await ledger.finish(status="completed")

    run = await store.get_run(ledger.run_id, user_id="u1")
    oracle = run["metrics_json"]["oracle"]
    assert oracle["wrote"] is True
    assert oracle["verified_after_last_write"] is True
    assert oracle["no_oracle_done"] is False
    assert oracle["last_outcome"] == "green"
    assert oracle["kinds"] == ["test"]

    events = await store.list_events(ledger.run_id, user_id="u1")
    summaries = [e for e in events if e["type"] == "oracle_summary"]
    assert len(summaries) == 1
    assert summaries[0]["payload"]["no_oracle_done"] is False


@pytest.mark.asyncio
async def test_ledger_flags_write_after_verification_as_no_oracle_done():
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA)
    store, ledger = await _make_ledger(conn)

    # verify → THEN write: the proven claim is stale.
    await _tool_roundtrip(ledger, tool="test_run", preview="4 passed")
    await _tool_roundtrip(ledger, tool="code_edit",
                          tool_input={"path": "app.py"})
    await ledger.finish(status="completed")

    run = await store.get_run(ledger.run_id, user_id="u1")
    oracle = run["metrics_json"]["oracle"]
    assert oracle["wrote"] is True
    assert oracle["oracle_calls"] == 1
    assert oracle["no_oracle_done"] is True


@pytest.mark.asyncio
async def test_oracle_stats_rollup():
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA)

    # Run 1: verified write. Run 2: unverified write. Run 3: pre-spine row
    # (no oracle block) — must count as runs_without_telemetry.
    store, l1 = await _make_ledger(conn)
    await _tool_roundtrip(l1, tool="code_edit", tool_input={"path": "a.py"})
    await _tool_roundtrip(l1, tool="test_run", preview="2 passed")
    await l1.finish(status="completed")

    _, l2 = await _make_ledger(conn)
    await _tool_roundtrip(l2, tool="file_write", tool_input={"path": "b.py"})
    await l2.finish(status="completed")

    _, l3 = await _make_ledger(conn)
    await l3.finish(status="completed")
    await conn.execute(
        "UPDATE coder_turn_runs SET metrics_json = '{}' WHERE id = ?",
        (l3.run_id,),
    )
    await conn.commit()

    stats = await store.oracle_stats(user_id="u1")
    assert stats["runs"] == 2
    assert stats["runs_without_telemetry"] == 1
    assert stats["write_runs"] == 2
    assert stats["no_oracle_done"] == 1
    assert stats["no_oracle_done_rate"] == 0.5
    assert stats["kinds"] == {"test": 1}
    assert stats["per_model"]["model-a"]["no_oracle_done"] == 1

    # User isolation: another user sees nothing.
    other = await store.oracle_stats(user_id="u2")
    assert other["runs"] == 0 and other["write_runs"] == 0


@pytest.mark.asyncio
async def test_oracle_stats_ignores_running_rows():
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA)
    store, ledger = await _make_ledger(conn)
    await _tool_roundtrip(ledger, tool="code_edit", tool_input={"path": "a.py"})
    # No finish() — run stays status='running'.
    stats = await store.oracle_stats(user_id="u1")
    assert stats["runs"] == 0 and stats["runs_without_telemetry"] == 0
