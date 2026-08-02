"""Live HTTP-level smoke tests for the dream system.

Skipped unless an Augmentum instance is running on localhost:6100. These
tests cover the wire-level contract that the in-process isolation tests
can't reach: route handlers extract user_id from the auth scope, the
PUT /api/config/personalization toggle dynamically boots/tears down the
dream system, and cross-user reads return 404 (not 200 with another
user's data).

Run manually with::

    pytest tests/live/test_live_dream.py --run-live -v
"""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.live


async def _create_user_and_login(client, suffix: str) -> str:
    """Register + log in a user, return the session token."""
    username = f"dream_live_{suffix}"
    password = "dream-test-password-123"
    # Register (idempotent — accept 409 on second run)
    await client.post("/api/auth/register", json={
        "username": username, "password": password,
    })
    resp = await client.post("/api/auth/login", json={
        "username": username, "password": password,
    })
    if resp.status_code != 200:
        pytest.skip(f"auth not configured (login returned {resp.status_code})")
    return resp.json()["token"]


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_dream_status_per_user(live_augmentum):
    """Each user sees their own scheduler counters via GET /api/dream/status."""
    token_a = await _create_user_and_login(live_augmentum, "a")
    token_b = await _create_user_and_login(live_augmentum, "b")

    # Enable dreams for user A. Other user is independent.
    enable = await live_augmentum.put(
        "/api/config/personalization",
        json={"dreamEnabled": "true"},
        headers=_auth_headers(token_a),
    )
    if enable.status_code == 503:
        pytest.skip("settings store unavailable")
    assert enable.status_code == 200, enable.text

    # status for user A — should not 503 anymore
    sa = await live_augmentum.get("/api/dream/status", headers=_auth_headers(token_a))
    assert sa.status_code == 200, sa.text
    body_a = sa.json()
    assert "messages_since_dream" in body_a
    assert "next_dream_eligible" in body_a

    # status for user B — same scheduler, but per-user counters: theirs are zeroed.
    sb = await live_augmentum.get("/api/dream/status", headers=_auth_headers(token_b))
    assert sb.status_code == 200, sb.text
    body_b = sb.json()
    assert body_b["messages_since_dream"] == 0
    assert body_b["approved_memories_since_dream"] == 0


@pytest.mark.asyncio
async def test_journal_entries_are_user_scoped(live_augmentum):
    """User B cannot fetch user A's journal entries by id."""
    token_a = await _create_user_and_login(live_augmentum, "a")
    token_b = await _create_user_and_login(live_augmentum, "b")

    # Enable dreams (idempotent) and trigger a manual cycle for A.
    await live_augmentum.put(
        "/api/config/personalization",
        json={"dreamEnabled": "true"},
        headers=_auth_headers(token_a),
    )
    trigger = await live_augmentum.post(
        "/api/dream/trigger", json={"persona_id": "default"},
        headers=_auth_headers(token_a),
    )
    if trigger.status_code == 503:
        pytest.skip("dream system disabled or backend unavailable")
    assert trigger.status_code == 200, trigger.text

    # Give the cycle a moment in case it's queued async; trigger_manual is
    # synchronous in the current implementation but stay defensive.
    await asyncio.sleep(0.5)

    # List for A — may have entries (depends on whether A has approved memories
    # to dream from on a fresh account; permissive assertion).
    list_a = await live_augmentum.get(
        "/api/dream/journal", headers=_auth_headers(token_a),
    )
    assert list_a.status_code == 200, list_a.text
    a_entries = list_a.json().get("entries", [])

    # List for B — should be empty (B never dreamed)
    list_b = await live_augmentum.get(
        "/api/dream/journal", headers=_auth_headers(token_b),
    )
    assert list_b.status_code == 200, list_b.text
    b_entries = list_b.json().get("entries", [])
    assert len(b_entries) == 0, f"user B leaked {len(b_entries)} entries"

    # If A has any entries, confirm B can't fetch one by id
    if a_entries:
        eid = a_entries[0]["id"]
        cross = await live_augmentum.get(
            f"/api/dream/journal/{eid}", headers=_auth_headers(token_b),
        )
        assert cross.status_code == 404, \
            f"user B fetched user A's entry: {cross.status_code} {cross.text}"


@pytest.mark.asyncio
async def test_dream_disable_tears_down_live(live_augmentum):
    """Toggling dreamEnabled off should switch /api/dream/status to 503."""
    token = await _create_user_and_login(live_augmentum, "a")

    await live_augmentum.put(
        "/api/config/personalization",
        json={"dreamEnabled": "true"},
        headers=_auth_headers(token),
    )
    on = await live_augmentum.get("/api/dream/status", headers=_auth_headers(token))
    if on.status_code == 503:
        pytest.skip("dream system not enabled")
    assert on.status_code == 200

    # Disable
    await live_augmentum.put(
        "/api/config/personalization",
        json={"dreamEnabled": "false"},
        headers=_auth_headers(token),
    )
    off = await live_augmentum.get("/api/dream/status", headers=_auth_headers(token))
    assert off.status_code == 503, \
        "scheduler should be torn down after dreamEnabled=false"
