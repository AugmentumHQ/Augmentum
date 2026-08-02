"""Durable permission-audit trail tests (migration 260).

Two layers:

- ``PermissionAuditStore`` round-trip / scoping / truncation against an
  in-memory SQLite with the REAL migration file applied (read from
  disk, so schema drift between test and migration is impossible).
- ``PermissionRegistry`` audit-sink wiring: every modal outcome
  (user-approved, user-denied, timeout) must emit exactly one audit
  event with the right ``decided_by``, and a failing sink must never
  break the approval flow itself.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import aiosqlite
import pytest

from augmentum.coder.permission_audit import PermissionAuditStore
from augmentum.coder.permissions import PermissionRegistry

_MIGRATION = (
    Path(__file__).resolve().parents[1]
    / "augmentum" / "state" / "migrations"
    / "260_coder_permission_audit_trail.sql"
)


async def _mkstore() -> tuple[PermissionAuditStore, aiosqlite.Connection]:
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_MIGRATION.read_text(encoding="utf-8"))
    return PermissionAuditStore(conn), conn


# ---------------------------------------------------------------------------
# Store round-trip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_and_list_roundtrip():
    store, conn = await _mkstore()
    try:
        await store.record(
            tool_name="shell_exec",
            decision="allowed",
            decided_by="user",
            user_id="u1",
            workspace_id="ws1",
            tool_input={"command": "ls"},
        )
        events = await store.list_events(user_id="u1")
        assert len(events) == 1
        ev = events[0]
        assert ev["tool_name"] == "shell_exec"
        assert ev["decision"] == "allowed"
        assert ev["decided_by"] == "user"
        assert ev["workspace_id"] == "ws1"
        assert '"command"' in ev["tool_input"]
        assert ev["created_at"] > 0
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_user_isolation():
    store, conn = await _mkstore()
    try:
        await store.record(
            tool_name="file_write", decision="allowed",
            decided_by="user", user_id="user-a",
        )
        await store.record(
            tool_name="shell_exec", decision="denied",
            decided_by="policy", user_id="user-b",
        )
        a = await store.list_events(user_id="user-a")
        assert [e["tool_name"] for e in a] == ["file_write"]
        b = await store.list_events(user_id="user-b")
        assert [e["tool_name"] for e in b] == ["shell_exec"]
        # Empty user_id = single-tenant dev convention: sees everything.
        assert len(await store.list_events()) == 2
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_workspace_filter():
    store, conn = await _mkstore()
    try:
        await store.record(
            tool_name="t1", decision="allowed", decided_by="user",
            user_id="u", workspace_id="ws-1",
        )
        await store.record(
            tool_name="t2", decision="allowed", decided_by="user",
            user_id="u", workspace_id="ws-2",
        )
        only = await store.list_events(user_id="u", workspace_id="ws-2")
        assert [e["tool_name"] for e in only] == ["t2"]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_oversized_input_truncated():
    store, conn = await _mkstore()
    try:
        await store.record(
            tool_name="file_write", decision="allowed", decided_by="user",
            user_id="u", tool_input={"content": "x" * 100_000},
        )
        ev = (await store.list_events(user_id="u"))[0]
        assert len(ev["tool_input"]) < 3000
        assert ev["tool_input"].endswith("…(truncated)")
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_record_never_raises_on_broken_conn():
    store, conn = await _mkstore()
    await conn.close()
    # Closed connection — record() must swallow + log, not propagate.
    await store.record(
        tool_name="shell_exec", decision="allowed", decided_by="user",
    )


# ---------------------------------------------------------------------------
# Registry → sink wiring
# ---------------------------------------------------------------------------


class _SinkRecorder:
    def __init__(self):
        self.events: list[dict] = []

    async def __call__(self, **kwargs):
        self.events.append(kwargs)


@pytest.mark.asyncio
async def test_user_approve_emits_audit_event():
    sink = _SinkRecorder()
    reg = PermissionRegistry(audit_sink=sink)

    async def approver():
        await asyncio.sleep(0.01)
        reg.resolve(reg.pending_for("u1")[0].id, approved=True)

    result, _ = await asyncio.gather(
        reg.request("u1", "shell_exec", {"command": "rm -rf build"},
                    workspace_id="ws-9"),
        approver(),
    )
    assert result is True
    assert len(sink.events) == 1
    ev = sink.events[0]
    assert ev["decision"] == "allowed"
    assert ev["decided_by"] == "user"
    assert ev["user_id"] == "u1"
    assert ev["workspace_id"] == "ws-9"
    assert ev["tool_name"] == "shell_exec"


@pytest.mark.asyncio
async def test_user_deny_emits_audit_event():
    sink = _SinkRecorder()
    reg = PermissionRegistry(audit_sink=sink)

    async def denier():
        await asyncio.sleep(0.01)
        reg.resolve(reg.pending_for("u1")[0].id, approved=False)

    result, _ = await asyncio.gather(
        reg.request("u1", "code_edit", {"path": "/workspace/a"}),
        denier(),
    )
    assert result is False
    assert sink.events[0]["decision"] == "denied"
    assert sink.events[0]["decided_by"] == "user"


@pytest.mark.asyncio
async def test_timeout_emits_denied_timeout_event():
    sink = _SinkRecorder()
    reg = PermissionRegistry(audit_sink=sink)

    result = await reg.request("u1", "shell_exec", {}, timeout=0.01)

    assert result is False
    assert sink.events[0]["decision"] == "denied"
    assert sink.events[0]["decided_by"] == "timeout"


@pytest.mark.asyncio
async def test_failing_sink_does_not_break_approval():
    async def broken_sink(**kwargs):
        raise RuntimeError("db offline")

    reg = PermissionRegistry(audit_sink=broken_sink)

    async def approver():
        await asyncio.sleep(0.01)
        reg.resolve(reg.pending_for("u1")[0].id, approved=True)

    result, _ = await asyncio.gather(
        reg.request("u1", "shell_exec", {}),
        approver(),
    )
    assert result is True  # approval still resolved despite sink failure


@pytest.mark.asyncio
async def test_no_sink_is_a_clean_noop():
    reg = PermissionRegistry()  # legacy construction — no sink

    result = await reg.request("u1", "shell_exec", {}, timeout=0.01)
    assert result is False
