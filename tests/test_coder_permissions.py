"""Tests for ``augmentum.coder.permissions.PermissionRegistry``.

Covers the registry's approve/deny/timeout flow used by the hybrid
coder loop's ``_check_tool_permission`` gate. Route-layer coverage is
exercised through the main coder handler tests; this file is
concurrency-focused unit tests on the registry itself.
"""
from __future__ import annotations

import asyncio

import pytest

from augmentum.coder.permissions import PermissionRegistry


@pytest.mark.asyncio
async def test_registry_approve_resolves_request_to_true():
    reg = PermissionRegistry()

    async def approver():
        # Let `request()` register its pending entry first.
        await asyncio.sleep(0.01)
        pending = reg.pending_for("user-1")
        assert len(pending) == 1
        reg.resolve(pending[0].id, approved=True)

    async def caller():
        return await reg.request("user-1", "shell_exec", {"command": "ls"})

    result, _ = await asyncio.gather(caller(), approver())
    assert result is True
    assert reg.size() == 0  # request cleaned up after resolution


@pytest.mark.asyncio
async def test_registry_deny_resolves_request_to_false():
    reg = PermissionRegistry()

    async def denier():
        await asyncio.sleep(0.01)
        pending = reg.pending_for("user-1")
        reg.resolve(pending[0].id, approved=False)

    async def caller():
        return await reg.request("user-1", "code_edit", {"path": "/a"})

    result, _ = await asyncio.gather(caller(), denier())
    assert result is False
    assert reg.size() == 0


@pytest.mark.asyncio
async def test_registry_timeout_returns_false():
    """An unresolved request times out to False so the loop can proceed."""
    reg = PermissionRegistry(default_timeout=0.05)
    result = await reg.request("user-1", "shell_exec", {"command": "ls"})
    assert result is False
    assert reg.size() == 0


@pytest.mark.asyncio
async def test_pending_for_scopes_by_user():
    reg = PermissionRegistry(default_timeout=0.1)

    async def _fire(uid):
        return await reg.request(uid, "code_edit", {"path": "/a"})

    # Fire two requests from different users in parallel; don't await
    # them so they stay pending.
    t1 = asyncio.create_task(_fire("alice"))
    t2 = asyncio.create_task(_fire("bob"))
    await asyncio.sleep(0.01)

    alice_pending = reg.pending_for("alice")
    bob_pending = reg.pending_for("bob")
    assert len(alice_pending) == 1
    assert len(bob_pending) == 1
    assert alice_pending[0].user_id == "alice"
    assert bob_pending[0].user_id == "bob"

    # Empty user_id sees all pending (single-tenant dev mode)
    all_pending = reg.pending_for("")
    assert len(all_pending) == 2

    # Let timeouts fire so the tasks complete cleanly.
    await asyncio.gather(t1, t2)


@pytest.mark.asyncio
async def test_resolve_unknown_returns_false():
    reg = PermissionRegistry()
    assert reg.resolve("no-such-id", approved=True) is False


@pytest.mark.asyncio
async def test_resolve_twice_is_noop():
    reg = PermissionRegistry(default_timeout=1.0)

    async def inner():
        return await reg.request("u", "shell_exec", {"command": "ls"})

    task = asyncio.create_task(inner())
    await asyncio.sleep(0.01)
    pending = reg.pending_for("u")
    req_id = pending[0].id

    # First resolve wins, second is a no-op
    assert reg.resolve(req_id, approved=True) is True
    assert reg.resolve(req_id, approved=False) is False

    result = await task
    assert result is True


@pytest.mark.asyncio
async def test_to_dict_shape_stable():
    """The /pending route relies on to_dict returning {id, tool_name,
    tool_input, created_at, age_seconds}."""
    reg = PermissionRegistry(default_timeout=0.2)

    async def inner():
        return await reg.request("u", "file_write", {"path": "/a"})

    task = asyncio.create_task(inner())
    await asyncio.sleep(0.01)
    pending = reg.pending_for("u")
    d = pending[0].to_dict()
    assert set(d.keys()) == {"id", "tool_name", "tool_input", "created_at", "age_seconds"}
    assert d["tool_name"] == "file_write"
    assert d["tool_input"] == {"path": "/a"}
    assert d["age_seconds"] >= 0.0

    # Clean up: resolve so the task completes
    reg.resolve(pending[0].id, approved=False)
    await task
