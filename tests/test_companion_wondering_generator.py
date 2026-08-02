"""Sprint 2 tests — wondering generator (Piece 7).

Covers:
* Kill switch (master flag) blocks
* Hush gate blocks
* User-recent-activity gate blocks
* Daily cap respected
* Cross-layer corroboration: thread + curiosity → confidence='normal'
* Single signal (thread only, no curiosity) → confidence='early'
* Topic mute blocks when matching scope exists
* Empty observed_state → no-op (no crash)
"""

from __future__ import annotations

import time

import pytest


async def _boot_runtime_with_user(user_id: str = "usr_w"):
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
    # Initialize the recent deque so the wondering generator has
    # something to look at.
    from collections import deque
    rt.observed_state = {
        "last_chat_mode": None,
        "last_chat_at": 0.0,
        "last_tool": None,
        "last_tool_at": 0.0,
        "last_mode_change": None,
        "recent": deque(maxlen=50),
    }
    return backend, rt


def _add_thread_events(rt, *, user_id: str, count: int = 3, domain: str = "example.com"):
    """Push enough surface events to form a thread."""
    now = time.time()
    for i in range(count):
        rt.observed_state["recent"].append({
            "topic": "surface.browse.opened",
            "payload": {"url": f"https://{domain}/article{i}", "user_id": user_id},
            "t": now - i * 60,
        })


# ── Kill switch ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_disabled_master_flag_blocks(monkeypatch):
    from augmentum.companion_runtime.wondering import maybe_write_wondering
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_topical_aggregator_enabled", False)
    backend, rt = await _boot_runtime_with_user("usr_w1")
    _add_thread_events(rt, user_id="usr_w1")
    result = await maybe_write_wondering(rt, user_id="usr_w1")
    assert result is None


# ── Gates ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hush_blocks(monkeypatch):
    from augmentum.companion_runtime.wondering import maybe_write_wondering
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_topical_aggregator_enabled", True)
    # Hush window in the future → is_hushed_now returns True
    future = "2030-01-01T00:00:00"
    monkeypatch.setattr(settings, "companion_journal_hushed_until", future)

    backend, rt = await _boot_runtime_with_user("usr_w2")
    _add_thread_events(rt, user_id="usr_w2")
    result = await maybe_write_wondering(rt, user_id="usr_w2")
    assert result is None


@pytest.mark.asyncio
async def test_user_recently_active_blocks(monkeypatch):
    from augmentum.companion_runtime.wondering import maybe_write_wondering
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_topical_aggregator_enabled", True)
    monkeypatch.setattr(settings, "companion_journal_hushed_until", "")

    backend, rt = await _boot_runtime_with_user("usr_w3")
    # Set cooldown into the future → is_user_recently_active returns True
    rt.user_cooldown_until = time.time() + 60
    _add_thread_events(rt, user_id="usr_w3")
    result = await maybe_write_wondering(rt, user_id="usr_w3")
    assert result is None


# ── Daily cap ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_daily_cap_blocks_after_n(monkeypatch):
    from augmentum.companion_runtime.wondering import maybe_write_wondering
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_topical_aggregator_enabled", True)
    monkeypatch.setattr(settings, "companion_journal_hushed_until", "")
    monkeypatch.setattr(settings, "companion_wondering_daily_cap", 2)

    backend, rt = await _boot_runtime_with_user("usr_w4")
    rt.user_cooldown_until = 0.0
    # Insert 2 wonderings already today
    for i in range(2):
        await backend.conn.execute(
            "INSERT INTO companion_journal "
            "(companion_id, user_id, entry_type, content, source) "
            "VALUES (?, ?, 'wondering', ?, 'autonomous')",
            ("becca", "usr_w4", f"already wondered {i}"),
        )
    await backend.conn.commit()

    _add_thread_events(rt, user_id="usr_w4")
    result = await maybe_write_wondering(rt, user_id="usr_w4")
    assert result is None  # cap reached


# ── Happy path ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thread_only_writes_at_early_confidence(monkeypatch):
    """Thread detected but no curiosity facet activation → confidence='early'."""
    from augmentum.companion_runtime.wondering import maybe_write_wondering
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_topical_aggregator_enabled", True)
    monkeypatch.setattr(settings, "companion_journal_hushed_until", "")
    monkeypatch.setattr(settings, "companion_wondering_daily_cap", 10)
    monkeypatch.setattr(settings, "companion_topical_min_events", 3)
    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")

    backend, rt = await _boot_runtime_with_user("usr_w5")
    rt.user_cooldown_until = 0.0
    _add_thread_events(rt, user_id="usr_w5")
    journal_id = await maybe_write_wondering(rt, user_id="usr_w5")
    assert journal_id is not None
    assert journal_id > 0

    cur = await backend.conn.execute(
        "SELECT entry_type, confidence_numeric, source, content_refs "
        "FROM companion_journal WHERE id = ?",
        (journal_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == "wondering"
    # confidence='early' → numeric 0.3 (no curiosity signal)
    assert row[1] == pytest.approx(0.3)
    assert row[2] == "autonomous"
    # content_refs should reference the surface events
    import json
    refs = json.loads(row[3] or "[]")
    assert any(r["kind"] == "surface" for r in refs)


@pytest.mark.asyncio
async def test_thread_plus_curiosity_writes_at_normal(monkeypatch):
    """Thread + curiosity facet activation → confidence='normal'."""
    from augmentum.companion_runtime.wondering import maybe_write_wondering
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_topical_aggregator_enabled", True)
    monkeypatch.setattr(settings, "companion_journal_hushed_until", "")
    monkeypatch.setattr(settings, "companion_wondering_daily_cap", 10)
    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")

    backend, rt = await _boot_runtime_with_user("usr_w6")
    rt.user_cooldown_until = 0.0

    # Plant a recent 'curious' facet activation (within last hour)
    await backend.conn.execute(
        "INSERT INTO personality_facet_activations "
        "(user_id, companion_id, facet, intensity, source) "
        "VALUES (?, ?, 'curious', 1.0, 'manual')",
        ("usr_w6", "becca"),
    )
    await backend.conn.commit()

    _add_thread_events(rt, user_id="usr_w6")
    journal_id = await maybe_write_wondering(rt, user_id="usr_w6")
    assert journal_id is not None

    cur = await backend.conn.execute(
        "SELECT confidence_numeric FROM companion_journal WHERE id = ?",
        (journal_id,),
    )
    row = await cur.fetchone()
    await cur.close()
    # confidence='normal' → numeric 0.6
    assert row[0] == pytest.approx(0.6)


# ── Topic mutes ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_topic_mute_blocks_matching_domain(monkeypatch):
    """When a mute matching the thread's domain exists, no write happens."""
    from augmentum.companion_runtime.wondering import maybe_write_wondering
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_topical_aggregator_enabled", True)
    monkeypatch.setattr(settings, "companion_journal_hushed_until", "")

    backend, rt = await _boot_runtime_with_user("usr_w7")
    rt.user_cooldown_until = 0.0

    # companion_topic_mutes now exists from migration 180 (Sprint 3) —
    # no ad-hoc CREATE needed. Just insert the mute row.
    await backend.conn.execute(
        "INSERT INTO companion_topic_mutes "
        "(user_id, companion_id, scope_json, expires_at) "
        "VALUES ('usr_w7', 'becca', "
        "'{\"domains\": [\"example.com\"], \"keywords\": []}', "
        "datetime('now', '+30 days'))"
    )
    await backend.conn.commit()

    _add_thread_events(rt, user_id="usr_w7")  # default domain example.com
    result = await maybe_write_wondering(rt, user_id="usr_w7")
    assert result is None


# ── Empty / no-op ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_observed_state_returns_none(monkeypatch):
    """Missing observed_state shouldn't crash."""
    from augmentum.companion_runtime.wondering import maybe_write_wondering
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_topical_aggregator_enabled", True)
    monkeypatch.setattr(settings, "companion_journal_hushed_until", "")

    backend, rt = await _boot_runtime_with_user("usr_w8")
    rt.user_cooldown_until = 0.0
    # Remove observed_state
    rt.observed_state = None
    result = await maybe_write_wondering(rt, user_id="usr_w8")
    assert result is None


@pytest.mark.asyncio
async def test_empty_recent_deque_returns_none(monkeypatch):
    from augmentum.companion_runtime.wondering import maybe_write_wondering
    from augmentum.config import settings

    monkeypatch.setattr(settings, "companion_topical_aggregator_enabled", True)
    monkeypatch.setattr(settings, "companion_journal_hushed_until", "")

    backend, rt = await _boot_runtime_with_user("usr_w9")
    rt.user_cooldown_until = 0.0
    # No events at all
    result = await maybe_write_wondering(rt, user_id="usr_w9")
    assert result is None
