from __future__ import annotations

import aiosqlite
import pytest

from augmentum.coder.ledger import CoderTurnLedger, CoderTurnLedgerStore
from augmentum.models.base import InternalStreamChunk

# Mirrors the live table post-migration 200 (workspace_id → project_id)
# plus the priming/cost columns finish_run writes. Keep in sync with
# tests/test_oracle_telemetry.py.
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
-- finish() bulk-persists the citation ledger here (migration 328). Hand-rolled
-- schema must mirror what finish_run writes or the persist errors.
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


@pytest.mark.asyncio
async def test_turn_ledger_records_tool_and_closeout():
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA)
    store = CoderTurnLedgerStore(conn)

    ledger = await CoderTurnLedger.start(
        store,
        user_id="u1",
        workspace_id="ws1",
        session_id="s1",
        model="model-a",
        provider="provider-a",
        strategy="hybrid",
    )
    await ledger.observe_chunk(InternalStreamChunk(augmentum={
        "phase": "executing",
        "status": "tool_call",
        "tool_call": {
            "id": "tc1",
            "tool": "shell_exec",
            "input": {"command": "npm test"},
        },
    }))
    await ledger.observe_chunk(InternalStreamChunk(augmentum={
        "phase": "executing",
        "status": "tool_result",
        "tool_result": {"id": "tc1", "tool": "shell_exec", "success": True},
    }))
    await ledger.observe_chunk(InternalStreamChunk(augmentum={
        "phase": "executing",
        "status": "complete",
        "iterations_used": 3,
        "termination_reason": "model_stop",
    }))
    await ledger.finish(status="completed")

    run = await store.get_run(ledger.run_id, user_id="u1")
    assert run is not None
    assert run["status"] == "completed"
    assert run["strategy"] == "hybrid"
    assert run["tool_calls"] == 1
    assert run["iterations"] == 3
    assert run["commands_run"] == ["npm test"]
    assert run["metrics_json"]["visible_answer"] is True

    events = await store.list_events(ledger.run_id, user_id="u1")
    assert [event["status"] for event in events] == [
        "tool_call",
        "tool_result",
        "complete",
        # Verification-spine closeout summary (spec 2026-07-06 Phase 2),
        # emitted by finish().
        "oracle_summary",
    ]


@pytest.mark.asyncio
async def test_turn_ledger_persists_citations_end_to_end():
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA)
    store = CoderTurnLedgerStore(conn)
    ledger = await CoderTurnLedger.start(
        store, user_id="u1", workspace_id="ws1", session_id="s1", model="m",
    )
    # A write (claim) + a test oracle (proof), through the real tool stream.
    await ledger.observe_chunk(InternalStreamChunk(augmentum={
        "phase": "executing", "status": "tool_call",
        "tool_call": {"id": "w1", "tool": "file_write", "input": {"path": "a.py"}},
    }))
    await ledger.observe_chunk(InternalStreamChunk(augmentum={
        "phase": "executing", "status": "tool_result",
        "tool_result": {"id": "w1", "tool": "file_write", "success": True, "checkpoint": "ck1"},
    }))
    await ledger.observe_chunk(InternalStreamChunk(augmentum={
        "phase": "executing", "status": "tool_call",
        "tool_call": {"id": "t1", "tool": "test_run", "input": {"command": "pytest -x"}},
    }))
    await ledger.observe_chunk(InternalStreamChunk(augmentum={
        "phase": "executing", "status": "tool_result",
        "tool_result": {"id": "t1", "tool": "test_run", "success": True,
                        "output_preview": "3 passed"},
    }))
    await ledger.finish(status="completed")

    from augmentum.coder.citations import load_citations
    rows = await load_citations(conn, turn_run_id=ledger.run_id, user_id="u1")
    kinds = sorted(r["evidence_kind"] for r in rows)
    assert kinds == ["test", "write"]
    write_row = next(r for r in rows if r["evidence_kind"] == "write")
    assert write_row["file"] == "a.py"
    assert write_row["evidence_ref"] == "ck1"
    test_row = next(r for r in rows if r["evidence_kind"] == "test")
    assert test_row["outcome"] == "green"
    # User-scope: another user cannot read them.
    assert await load_citations(conn, turn_run_id=ledger.run_id, user_id="u2") == []


@pytest.mark.asyncio
async def test_turn_ledger_user_scope_hides_other_users():
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA)
    store = CoderTurnLedgerStore(conn)
    ledger = await CoderTurnLedger.start(
        store,
        user_id="u1",
        workspace_id="ws1",
        session_id="s1",
        model="model-a",
    )

    assert await store.get_run(ledger.run_id, user_id="u2") is None
    assert await store.list_events(ledger.run_id, user_id="u2") == []


@pytest.mark.asyncio
async def test_turn_ledger_records_token_budget_and_compactions():
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA)
    store = CoderTurnLedgerStore(conn)
    ledger = await CoderTurnLedger.start(
        store,
        user_id="u1",
        workspace_id="ws1",
        session_id="s1",
        model="model-a",
        strategy="native",
    )

    await ledger.observe_chunk(InternalStreamChunk(augmentum={
        "phase": "executing",
        "status": "budget",
        "tokens": {
            "scope": "native_iteration",
            "tokens": 1200,
            "limit": 16000,
            "ratio": 0.075,
            "iteration": 1,
            "compacted": False,
        },
    }))
    await ledger.observe_chunk(InternalStreamChunk(augmentum={
        "phase": "executing",
        "status": "compaction",
        "tokens_before": 19000,
        "tokens_after": 9000,
    }))
    await ledger.observe_chunk(InternalStreamChunk(augmentum={
        "phase": "executing",
        "status": "budget",
        "tokens": {
            "scope": "native_iteration",
            "tokens": 9000,
            "limit": 16000,
            "ratio": 0.5625,
            "iteration": 2,
            "compacted": True,
        },
    }))
    await ledger.finish(status="completed")

    run = await store.get_run(ledger.run_id, user_id="u1")
    metrics = run["metrics_json"]
    assert metrics["last_prompt_tokens"] == 9000
    assert metrics["max_prompt_tokens"] == 9000
    assert metrics["compactions"] == 1
    assert metrics["token_snapshots"][-1]["compacted"] is True
