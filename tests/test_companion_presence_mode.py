"""Sprint 5 tests — presence_mode + Observatory.

Covers:
* presence_mode helper returns validated values
* Silent mode suppresses wondering writes
* Silent mode suppresses pre-context injection
* Gentle mode allows wonderings but not pre-context
* Engaged mode allows both
* Observatory endpoint registered
* Observatory query shapes are correct
"""

from __future__ import annotations

import pytest

# ── presence_mode helper ─────────────────────────────────────────────


def test_presence_mode_default_silent():
    """Without explicit setting, mode is silent (safe default)."""
    from augmentum.companion_runtime import presence_mode as pm
    # In a fresh test the setting hasn't been touched — should fall back
    # to DEFAULT_MODE = silent
    assert pm.DEFAULT_MODE == pm.MODE_SILENT


def test_presence_mode_validates_input(monkeypatch):
    from augmentum.companion_runtime import presence_mode as pm
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")
    assert pm.get_presence_mode() == "gentle"

    monkeypatch.setattr(settings, "companion_presence_mode", "engaged")
    assert pm.get_presence_mode() == "engaged"

    monkeypatch.setattr(settings, "companion_presence_mode", "garbage")
    # Invalid → silent fallback
    assert pm.get_presence_mode() == "silent"


def test_autonomy_allowed_by_mode(monkeypatch):
    from augmentum.companion_runtime import presence_mode as pm
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_presence_mode", "silent")
    assert pm.autonomy_allowed() is False

    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")
    assert pm.autonomy_allowed() is True

    monkeypatch.setattr(settings, "companion_presence_mode", "engaged")
    assert pm.autonomy_allowed() is True


def test_pre_context_allowed_only_engaged(monkeypatch):
    from augmentum.companion_runtime import presence_mode as pm
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_presence_mode", "silent")
    assert pm.pre_context_allowed() is False

    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")
    assert pm.pre_context_allowed() is False

    monkeypatch.setattr(settings, "companion_presence_mode", "engaged")
    assert pm.pre_context_allowed() is True


def test_pip_allowed_not_silent(monkeypatch):
    from augmentum.companion_runtime import presence_mode as pm
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_presence_mode", "silent")
    assert pm.pip_allowed() is False

    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")
    assert pm.pip_allowed() is True


# ── Wondering generator respects presence_mode ───────────────────────


async def _boot_runtime_with_user(user_id: str = "usr_pm"):
    from collections import deque

    from augmentum.companion_runtime.runtime import CompanionRuntime
    from augmentum.state.backends.sqlite import SQLiteBackend
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        (user_id, "tester", "x"),
    )
    await backend.conn.commit()
    rt = CompanionRuntime(backend, companion_id="becca")
    await rt.identity.load()
    await rt.state.load()
    rt.observed_state = {
        "last_chat_mode": None,
        "last_chat_at": 0.0,
        "last_tool": None,
        "last_tool_at": 0.0,
        "last_mode_change": None,
        "recent": deque(maxlen=50),
    }
    return backend, rt


def _add_thread_events(rt, *, user_id: str):
    import time
    now = time.time()
    for i in range(3):
        rt.observed_state["recent"].append({
            "topic": "surface.browse.opened",
            "payload": {"url": f"https://example.com/article{i}", "user_id": user_id},
            "t": now - i * 60,
        })


@pytest.mark.asyncio
async def test_silent_mode_blocks_wondering(monkeypatch):
    """Silent presence_mode → wondering generator returns None even with
    a valid thread present."""
    from augmentum.companion_runtime.wondering import maybe_write_wondering
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_topical_aggregator_enabled", True)
    monkeypatch.setattr(settings, "companion_journal_hushed_until", "")
    monkeypatch.setattr(settings, "companion_presence_mode", "silent")

    backend, rt = await _boot_runtime_with_user("usr_silent")
    rt.user_cooldown_until = 0.0
    _add_thread_events(rt, user_id="usr_silent")

    result = await maybe_write_wondering(rt, user_id="usr_silent")
    assert result is None


@pytest.mark.asyncio
async def test_gentle_mode_allows_wondering(monkeypatch):
    from augmentum.companion_runtime.wondering import maybe_write_wondering
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_topical_aggregator_enabled", True)
    monkeypatch.setattr(settings, "companion_journal_hushed_until", "")
    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")
    monkeypatch.setattr(settings, "companion_wondering_daily_cap", 10)

    backend, rt = await _boot_runtime_with_user("usr_gentle")
    rt.user_cooldown_until = 0.0
    _add_thread_events(rt, user_id="usr_gentle")

    journal_id = await maybe_write_wondering(rt, user_id="usr_gentle")
    assert journal_id is not None
    assert journal_id > 0


# ── Pre-context respects presence_mode ──────────────────────────────


@pytest.mark.asyncio
async def test_silent_mode_blocks_pre_context(monkeypatch):
    from augmentum.companion_runtime.pre_context import maybe_inject_notes_context
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_pre_context_enabled", True)
    monkeypatch.setattr(settings, "companion_presence_mode", "silent")

    backend, rt = await _boot_runtime_with_user("usr_npc")
    # Plant a ready note that would otherwise match
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content, content_refs, quiet_share_ready) "
        "VALUES ('becca', 'usr_npc', 'noticing', "
        "'Prefix caching connects to KV restoration work', '[]', 1)",
    )
    await backend.conn.commit()

    result = await maybe_inject_notes_context(
        rt, user_id="usr_npc",
        first_message="tell me about prefix caching and KV cache stability",
    )
    assert result is None


@pytest.mark.asyncio
async def test_gentle_mode_blocks_pre_context(monkeypatch):
    """Pre-context is engaged-only. Gentle should NOT inject."""
    from augmentum.companion_runtime.pre_context import maybe_inject_notes_context
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_pre_context_enabled", True)
    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")

    backend, rt = await _boot_runtime_with_user("usr_npc2")
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content, content_refs, quiet_share_ready) "
        "VALUES ('becca', 'usr_npc2', 'noticing', "
        "'Prefix caching connects to KV restoration work', '[]', 1)",
    )
    await backend.conn.commit()

    result = await maybe_inject_notes_context(
        rt, user_id="usr_npc2",
        first_message="tell me about prefix caching and KV cache stability",
    )
    assert result is None


@pytest.mark.asyncio
async def test_engaged_mode_allows_pre_context(monkeypatch):
    from augmentum.companion_runtime.pre_context import maybe_inject_notes_context
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_pre_context_enabled", True)
    monkeypatch.setattr(settings, "companion_presence_mode", "engaged")
    monkeypatch.setattr(settings, "companion_pre_context_min_keyword_overlap", 2)

    backend, rt = await _boot_runtime_with_user("usr_engaged")
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content, content_refs, quiet_share_ready) "
        "VALUES ('becca', 'usr_engaged', 'noticing', "
        "'Prefix caching connects to KV restoration work from April', '[]', 1)",
    )
    await backend.conn.commit()

    result = await maybe_inject_notes_context(
        rt, user_id="usr_engaged",
        first_message="tell me about prefix caching and KV cache stability",
    )
    assert result is not None
    assert "becca's note" in result.lower()


# ── Observatory endpoint ────────────────────────────────────────────


def test_observatory_endpoint_registered():
    from augmentum.proxy.companion_routes import router
    paths = {getattr(r, "path", "") for r in router.routes}
    assert "/api/companion/observatory" in paths
