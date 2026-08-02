"""Tests for the citation ledger — the claim→proof provenance primitive.

Covers the pure classifier (:func:`citations_from_tool_result`) and the
durable store (save/load with user-scope isolation). The emit-from-ledger
path is exercised indirectly via the classifier since the ledger just
extends a list with its output.
"""
from __future__ import annotations

import pytest

from augmentum.coder.citations import (
    Citation,
    citations_from_tool_result,
    load_citations,
    save_citations,
)

# ── Pure classifier ─────────────────────────────────────────────────────────

def test_single_file_write_is_one_write_citation():
    cites = citations_from_tool_result(
        seq=5, tool="file_write", tool_input={"path": "a.py"},
        success=True, checkpoint="ckpt_1",
    )
    assert len(cites) == 1
    c = cites[0]
    assert c.evidence_kind == "write"
    assert c.file == "a.py"
    assert c.tool_call_seq == 5
    assert c.evidence_ref == "ckpt_1"
    assert c.outcome == ""
    assert c.line_start is None and c.line_end is None


def test_batch_edit_is_one_citation_per_file_deduped():
    cites = citations_from_tool_result(
        seq=7, tool="code_edit_batch",
        tool_input={"edits": [
            {"path": "a.py"}, {"path": "b.py"}, {"path": "a.py"},  # dup
        ]},
        success=True,
    )
    files = sorted(c.file for c in cites)
    assert files == ["a.py", "b.py"]  # deduped
    assert all(c.evidence_kind == "write" for c in cites)


def test_failed_write_is_not_a_claim():
    assert citations_from_tool_result(
        seq=3, tool="file_write", tool_input={"path": "a.py"}, success=False,
    ) == []


def test_oracle_tool_carries_outcome_and_ref():
    cites = citations_from_tool_result(
        seq=9, tool="test_run", tool_input={"command": "pytest -x"},
        success=True, oracle_kind="test", outcome="green",
    )
    assert len(cites) == 1
    c = cites[0]
    assert c.evidence_kind == "test"
    assert c.outcome == "green"
    assert c.evidence_ref == "pytest -x"
    assert c.file == ""


def test_non_evidence_tool_yields_nothing():
    assert citations_from_tool_result(
        seq=1, tool="file_read", tool_input={"path": "a.py"}, success=True,
    ) == []


def test_explicit_line_range_is_captured():
    cites = citations_from_tool_result(
        seq=2, tool="code_edit",
        tool_input={"path": "a.py", "start_line": 10, "end_line": 20},
        success=True,
    )
    assert len(cites) == 1
    assert cites[0].line_start == 10
    assert cites[0].line_end == 20


def test_bad_line_range_stays_null():
    # end < start, or non-int, must not fabricate a range.
    cites = citations_from_tool_result(
        seq=2, tool="code_edit",
        tool_input={"path": "a.py", "start_line": 20, "end_line": 5},
        success=True,
    )
    assert cites[0].line_start is None and cites[0].line_end is None


# ── Durable store ───────────────────────────────────────────────────────────

async def _conn():
    import aiosqlite
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE coder_run_citations ("
        " id INTEGER PRIMARY KEY, turn_run_id TEXT NOT NULL, user_id TEXT NOT NULL,"
        " workspace_id TEXT NOT NULL DEFAULT '', run_id TEXT NOT NULL DEFAULT '',"
        " tool_call_seq INTEGER NOT NULL DEFAULT 0, file TEXT NOT NULL DEFAULT '',"
        " line_start INTEGER, line_end INTEGER,"
        " evidence_kind TEXT NOT NULL DEFAULT 'write',"
        " evidence_ref TEXT NOT NULL DEFAULT '', outcome TEXT NOT NULL DEFAULT '',"
        " created_at TEXT NOT NULL DEFAULT (datetime('now')))",
    )
    await conn.commit()
    return conn


@pytest.mark.asyncio
async def test_save_load_roundtrip_ordered():
    conn = await _conn()
    try:
        await save_citations(
            conn, turn_run_id="ctr_1", user_id="u1", workspace_id="ws_1",
            citations=[
                Citation(tool_call_seq=9, evidence_kind="test", outcome="green", evidence_ref="pytest"),
                Citation(tool_call_seq=3, evidence_kind="write", file="a.py"),
            ],
        )
        rows = await load_citations(conn, turn_run_id="ctr_1", user_id="u1")
        assert [r["tool_call_seq"] for r in rows] == [3, 9]  # ordered by seq
        assert rows[0]["file"] == "a.py"
        assert rows[1]["evidence_kind"] == "test" and rows[1]["outcome"] == "green"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_user_scope_isolation():
    conn = await _conn()
    try:
        await save_citations(
            conn, turn_run_id="ctr_1", user_id="u1",
            citations=[Citation(tool_call_seq=1, evidence_kind="write", file="a.py")],
        )
        assert await load_citations(conn, turn_run_id="ctr_1", user_id="stranger") == []
        assert len(await load_citations(conn, turn_run_id="ctr_1", user_id="u1")) == 1
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_empty_citations_is_noop():
    conn = await _conn()
    try:
        await save_citations(conn, turn_run_id="ctr_1", user_id="u1", citations=[])
        assert await load_citations(conn, turn_run_id="ctr_1", user_id="u1") == []
    finally:
        await conn.close()
