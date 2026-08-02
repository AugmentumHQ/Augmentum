"""Tests for dream-related memory pipeline extensions."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_mock_store(tier_val: str = "provisional") -> MagicMock:
    """Build a mock MemoryStore with a fake _conn and memory object."""
    mem = MagicMock()
    mem.tier = tier_val

    conn = MagicMock()
    conn.execute = AsyncMock()
    conn.commit = AsyncMock()

    store = MagicMock()
    store.get = AsyncMock(return_value=mem)
    store.update_tier = AsyncMock()
    store._conn = conn
    return store


def _make_request(store, dream_scheduler=None) -> MagicMock:
    """Build a mock Starlette Request with app.state wired up."""
    app_state = MagicMock()
    app_state.memory_store = store
    if dream_scheduler is not None:
        app_state.dream_scheduler = dream_scheduler
    else:
        # Simulate attribute missing (getattr returns None)
        del app_state.dream_scheduler

    app = MagicMock()
    app.state = app_state

    request = MagicMock()
    request.app = app
    return request


@pytest.mark.asyncio
async def test_approve_sets_user_approved():
    """The approve endpoint must set user_approved=1 after promoting tier."""
    from augmentum.proxy.memory_routes import approve_memory

    store = _make_mock_store(tier_val="provisional")
    request = _make_request(store)

    with patch("augmentum.memory.models.MemoryTier") as MockTier:
        MockTier.PROVISIONAL = "provisional"
        MockTier.ACTIVE = "active"

        response = await approve_memory("mem-123", request)

    assert response.status_code == 200

    # Collect all SQL statements executed on the connection
    executed_sqls = [
        call.args[0].strip()
        for call in store._conn.execute.call_args_list
    ]

    user_approved_updates = [
        sql for sql in executed_sqls
        if "user_approved" in sql and "mem-123" in str(
            store._conn.execute.call_args_list[executed_sqls.index(sql)].args
        )
    ]
    assert user_approved_updates, (
        "Expected an UPDATE setting user_approved=1, but none was executed. "
        f"All executed SQL: {executed_sqls}"
    )


@pytest.mark.asyncio
async def test_approve_notifies_dream_scheduler():
    """The approve endpoint must call dream_scheduler.notify_approval when available."""
    from augmentum.proxy.memory_routes import approve_memory

    store = _make_mock_store(tier_val="active")
    scheduler = MagicMock()
    scheduler.notify_approval = MagicMock()
    request = _make_request(store, dream_scheduler=scheduler)

    with patch("augmentum.memory.models.MemoryTier") as MockTier:
        MockTier.PROVISIONAL = "provisional"
        MockTier.ACTIVE = "active"

        response = await approve_memory("mem-456", request)

    assert response.status_code == 200
    # Route passes the caller's user_id through; the mock's .scope.get().id
    # is itself a MagicMock — we only care that the memory id was forwarded.
    assert scheduler.notify_approval.call_count == 1
    args, kwargs = scheduler.notify_approval.call_args
    assert args == ("mem-456",)
    assert "user_id" in kwargs


@pytest.mark.asyncio
async def test_approve_works_without_dream_scheduler():
    """The approve endpoint must not raise when dream_scheduler is absent."""
    from augmentum.proxy.memory_routes import approve_memory

    store = _make_mock_store(tier_val="active")
    # No dream_scheduler on app.state — getattr returns None
    request = _make_request(store, dream_scheduler=None)

    with patch("augmentum.memory.models.MemoryTier") as MockTier:
        MockTier.PROVISIONAL = "provisional"
        MockTier.ACTIVE = "active"

        # Should not raise
        response = await approve_memory("mem-789", request)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_approve_promotes_provisional_to_active():
    """Existing behaviour: PROVISIONAL tier is promoted to ACTIVE."""
    from augmentum.proxy.memory_routes import approve_memory

    store = _make_mock_store(tier_val="provisional")
    request = _make_request(store)

    with patch("augmentum.memory.models.MemoryTier") as MockTier:
        MockTier.PROVISIONAL = "provisional"
        MockTier.ACTIVE = "active"

        response = await approve_memory("mem-001", request)

    store.update_tier.assert_called_once()
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_approve_user_approved_set_for_active_tier():
    """user_approved=1 must be set even when the memory is already ACTIVE (not PROVISIONAL)."""
    from augmentum.proxy.memory_routes import approve_memory

    store = _make_mock_store(tier_val="active")
    request = _make_request(store)

    with patch("augmentum.memory.models.MemoryTier") as MockTier:
        MockTier.PROVISIONAL = "provisional"
        MockTier.ACTIVE = "active"

        response = await approve_memory("mem-002", request)

    # update_tier must NOT be called (already active)
    store.update_tier.assert_not_called()

    # But user_approved UPDATE still must have run
    executed_sqls = [
        call.args[0].strip()
        for call in store._conn.execute.call_args_list
    ]
    assert any("user_approved" in sql for sql in executed_sqls), (
        "Expected user_approved UPDATE even for already-active memories. "
        f"Got: {executed_sqls}"
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_approve_returns_404_when_memory_not_found():
    """Approve must return 404 when the memory_id does not exist."""
    from augmentum.proxy.memory_routes import approve_memory

    store = MagicMock()
    store.get = AsyncMock(return_value=None)

    request = _make_request(store)

    with patch("augmentum.memory.models.MemoryTier"):
        response = await approve_memory("nonexistent", request)

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_approve_db_error_does_not_crash():
    """A DB error in the user_approved UPDATE must not crash the endpoint."""
    from augmentum.proxy.memory_routes import approve_memory

    store = _make_mock_store(tier_val="active")
    # Make execute raise on ALL calls (covers both provisional TTL clear and user_approved)
    store._conn.execute = AsyncMock(side_effect=Exception("db gone"))
    request = _make_request(store)

    with patch("augmentum.memory.models.MemoryTier") as MockTier:
        MockTier.PROVISIONAL = "provisional"
        MockTier.ACTIVE = "active"

        # Should still return 200, not raise
        response = await approve_memory("mem-err", request)

    assert response.status_code == 200
