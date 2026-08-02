"""Admin self-heal — every install needs an owner.

Pins the 2026-06-18 fix for "coder subagents keep getting disabled": the
coder-subagent controls (and other admin-only tool settings) are gated on
``role == 'admin'``, so an install whose users were created outside the
setup flow (``create_user`` defaults to ``role='user'``) had ZERO admins
— the controls greyed out and any save 403'd and reverted. ``SessionManager
.ensure_admin_exists`` promotes the longest-standing active user when no
admin exists, and is a strict no-op once one does.
"""

from __future__ import annotations

import pytest

from augmentum.auth.session_manager import SessionManager
from augmentum.state.backends.sqlite import SQLiteBackend


async def _sm():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    return SessionManager(backend._conn), backend


@pytest.mark.asyncio
async def test_promotes_sole_user_when_no_admin():
    sm, backend = await _sm()
    try:
        u = await sm.create_user("matt", "supersecret")
        assert u.role == "user"          # created outside the setup flow
        assert await sm.active_admin_count() == 0

        promoted = await sm.ensure_admin_exists()
        assert promoted == u.id
        refreshed = await sm.get_user_by_id(u.id)
        assert refreshed.role == "admin"
        assert await sm.active_admin_count() == 1
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_idempotent_once_admin_exists():
    sm, backend = await _sm()
    try:
        u = await sm.create_user("matt", "supersecret")
        await sm.ensure_admin_exists()        # first call promotes
        second = await sm.ensure_admin_exists()  # no-op now
        assert second is None
        assert await sm.active_admin_count() == 1
        assert (await sm.get_user_by_id(u.id)).role == "admin"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_noop_when_admin_already_present_does_not_touch_others():
    sm, backend = await _sm()
    try:
        admin = await sm.create_user("owner", "supersecret", role="admin")
        plain = await sm.create_user("teammate", "supersecret")
        assert await sm.ensure_admin_exists() is None
        # The plain user is NOT promoted; the admin is untouched.
        assert (await sm.get_user_by_id(plain.id)).role == "user"
        assert (await sm.get_user_by_id(admin.id)).role == "admin"
        assert await sm.active_admin_count() == 1
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_noop_on_fresh_install_with_no_users():
    sm, backend = await _sm()
    try:
        assert await sm.ensure_admin_exists() is None
        assert await sm.active_admin_count() == 0
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_promotes_longest_standing_user():
    sm, backend = await _sm()
    try:
        older = await sm.create_user("first", "supersecret")
        newer = await sm.create_user("second", "supersecret")
        # Force a deterministic age gap (same-second creation otherwise
        # ties created_at and the id tiebreak is random).
        await backend._conn.execute(
            "UPDATE users SET created_at = '2020-01-01T00:00:00' WHERE id = ?",
            (older.id,),
        )
        await backend._conn.execute(
            "UPDATE users SET created_at = '2025-01-01T00:00:00' WHERE id = ?",
            (newer.id,),
        )
        await backend._conn.commit()

        promoted = await sm.ensure_admin_exists()
        assert promoted == older.id
        assert (await sm.get_user_by_id(newer.id)).role == "user"
    finally:
        await backend.close()
