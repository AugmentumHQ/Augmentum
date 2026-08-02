"""Tests for the Today entry — daily in-her-voice reflection surface.

Covers:
* Migration 186 applies cleanly
* Local-date helper returns YYYY-MM-DD
* Mute filter: domain overlap, multi-keyword overlap, single distinctive keyword
* Read/write happy path
* Quarantine path: empty output / structural / injection / low quality
* Quarantined regen DOES NOT overwrite a prior good row
* Debounce: second call within window returns existing
* Force regen uses shorter debounce
* Settle marks settled_at; settled rows aren't overwritten
* forget_refs quarantines source rows + invalidates today
* Per-user isolation (two users don't see each other's reflections)
* presence_mode=silent suppresses generation
* Today gating off via companion_today_enabled
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


async def _boot_runtime_with_user(user_id: str = "usr_today"):
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
    return backend, rt


def _mock_tier_call(tiers_module, response_text: str, model_name: str = "fake-model"):
    """Monkey-patch tiers.utility to return a fake backend whose chat
    returns ``response_text``. Returns the original to restore later.
    """
    fake_response = MagicMock()
    fake_response.content = response_text
    fake_response.text = response_text
    fake_backend = MagicMock()
    fake_backend.chat = AsyncMock(return_value=fake_response)

    async def _fake_utility(rt, **kwargs):
        return fake_backend, model_name

    original = tiers_module.utility
    tiers_module.utility = _fake_utility
    return original, fake_backend


# ── Migration ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_migration_186_applies():
    backend, _rt = await _boot_runtime_with_user("usr_mig186")
    cur = await backend.conn.execute(
        "SELECT MAX(version) FROM schema_version"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] >= 186
    # Table exists with the expected columns.
    cur = await backend.conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='companion_today_reflections'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row is not None


# ── Helpers ──────────────────────────────────────────────────────────


def test_local_date_format():
    from augmentum.companion_runtime.today import _local_date
    d = _local_date()
    assert len(d) == 10
    assert d[4] == "-" and d[7] == "-"


def test_is_muted_domain_overlap():
    from augmentum.companion_runtime.today import _is_muted
    mutes = [{"domains": ["politics"], "keywords": []}]
    refs = [{"kind": "surface", "id": 1, "domain": "politics"}]
    assert _is_muted("any prose", refs, mutes) is True


def test_is_muted_keyword_threshold():
    from augmentum.companion_runtime.today import _is_muted
    mutes = [{"keywords": ["augmentum", "narrative", "engine"]}]
    # 2 of 3 muted keywords present → muted
    assert _is_muted(
        "Today I worked on augmentum narrative tests.",
        [], mutes,
    ) is True


def test_is_muted_distinctive_single_keyword():
    from augmentum.companion_runtime.today import _is_muted
    mutes = [{"keywords": ["abandonment"]}]
    # Single 5+ char keyword present → muted (conservative)
    assert _is_muted(
        "He felt abandonment after the move.",
        [], mutes,
    ) is True


def test_is_muted_no_overlap():
    from augmentum.companion_runtime.today import _is_muted
    mutes = [{"domains": ["unrelated"], "keywords": ["foo", "bar"]}]
    assert _is_muted("ordinary prose", [], mutes) is False


# ── Read/write happy path ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_and_read_today(monkeypatch):
    from augmentum.companion_runtime import tiers
    from augmentum.companion_runtime import today as _today
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_today_enabled", True)
    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")
    monkeypatch.setattr(settings, "companion_today_max_chars", 360)

    backend, rt = await _boot_runtime_with_user("usr_happy")

    # Seed one journal entry so there's signal to reflect on.
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(id, companion_id, user_id, content, entry_type, source, "
        " confidence_numeric, validation_score, quarantined, created_at) "
        "VALUES (42, ?, ?, 'I noticed the media setup thread', "
        "        'observation', 'autonomous', 0.7, 1.0, 0, "
        "        datetime('now'))",
        (rt.companion_id, "usr_happy"),
    )
    await backend.conn.commit()

    reflection_text = (
        "Mostly puttered today. Your media setup keeps returning "
        "[journal:42]. Quiet afternoon."
    )
    original, _fake = _mock_tier_call(tiers, reflection_text)
    try:
        result = await _today.maybe_regenerate(rt, user_id="usr_happy")
    finally:
        tiers.utility = original

    assert result is not None
    assert result.quarantined is False
    assert "media setup keeps returning" in result.content_text
    # Citation matched a real journal id → kept in source_refs
    assert {"kind": "journal", "id": 42} in result.source_refs


# ── Quarantine paths ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_output_quarantines(monkeypatch):
    from augmentum.companion_runtime import tiers
    from augmentum.companion_runtime import today as _today
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_today_enabled", True)
    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")

    backend, rt = await _boot_runtime_with_user("usr_empty")
    original, _ = _mock_tier_call(tiers, "")
    try:
        result = await _today.maybe_regenerate(rt, user_id="usr_empty")
    finally:
        tiers.utility = original

    # Row exists, quarantined with empty_output reason.
    assert result is not None
    assert result.quarantined is True
    assert result.quarantine_reason == "empty_output"


@pytest.mark.asyncio
async def test_quarantined_does_not_overwrite_prior_good_row(monkeypatch):
    """If a good reflection exists for the day and regen produces junk,
    the prior good row is preserved (we'd rather show stale-but-good
    than swap in quarantine garbage)."""
    from augmentum.companion_runtime import tiers
    from augmentum.companion_runtime import today as _today
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_today_enabled", True)
    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")

    backend, rt = await _boot_runtime_with_user("usr_preserve")

    # First call writes a good row.
    good_text = (
        "Today felt productive. We talked about the engine work in the "
        "morning and that thread carried into the afternoon."
    )
    original, _ = _mock_tier_call(tiers, good_text)
    try:
        first = await _today.maybe_regenerate(rt, user_id="usr_preserve")
    finally:
        tiers.utility = original
    assert first is not None and not first.quarantined

    # Force a regen with empty output. Reset debounce by reaching into
    # the module's internal map.
    _today._LAST_REGEN_AT.clear()
    original2, _ = _mock_tier_call(tiers, "")
    try:
        second = await _today.maybe_regenerate(
            rt, user_id="usr_preserve", force=True,
        )
    finally:
        tiers.utility = original2

    # The good row is still in place. Either we returned it as `second`
    # or the persist short-circuited (both acceptable).
    cur = await backend.conn.execute(
        "SELECT content_text, quarantined "
        "FROM companion_today_reflections "
        "WHERE user_id = ? AND companion_id = ?",
        ("usr_preserve", rt.companion_id),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row is not None
    assert good_text in row[0]
    assert row[1] == 0  # not quarantined


# ── Debounce ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_debounce_prevents_back_to_back_regen(monkeypatch):
    from augmentum.companion_runtime import tiers
    from augmentum.companion_runtime import today as _today
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_today_enabled", True)
    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")

    backend, rt = await _boot_runtime_with_user("usr_debounce")
    _today._LAST_REGEN_AT.clear()
    call_count = {"n": 0}
    fake_response = MagicMock()
    fake_response.content = "Reflection text version one."
    fake_response.text = "Reflection text version one."
    fake_backend = MagicMock()

    async def _chat(req):
        call_count["n"] += 1
        return fake_response
    fake_backend.chat = _chat

    async def _fake_utility(rt, **kwargs):
        return fake_backend, "fake-model"
    original = tiers.utility
    tiers.utility = _fake_utility
    try:
        await _today.maybe_regenerate(rt, user_id="usr_debounce")
        # Second call within the hourly window — should be debounced.
        await _today.maybe_regenerate(rt, user_id="usr_debounce")
    finally:
        tiers.utility = original
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_force_regen_uses_shorter_window(monkeypatch):
    """force=True honors the 10min floor rather than 1hr."""
    from augmentum.companion_runtime import tiers
    from augmentum.companion_runtime import today as _today
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_today_enabled", True)
    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")

    backend, rt = await _boot_runtime_with_user("usr_force")
    _today._LAST_REGEN_AT.clear()
    # Simulate a regen 20 min ago — force should bypass, plain should not.
    import time as _t
    key = _today._debounce_key("usr_force", rt.companion_id)
    _today._LAST_REGEN_AT[key] = _t.time() - 1200  # 20 min ago

    call_count = {"n": 0}
    async def _chat(req):
        call_count["n"] += 1
        return MagicMock(content="Refreshed reflection.", text="Refreshed reflection.")
    fake_backend = MagicMock()
    fake_backend.chat = _chat

    async def _fake_utility(rt, **kwargs):
        return fake_backend, "fake-model"
    original = tiers.utility
    tiers.utility = _fake_utility
    try:
        # Plain call: within 1hr window → debounced.
        await _today.maybe_regenerate(rt, user_id="usr_force")
        assert call_count["n"] == 0
        # Force call: outside 10min window → proceeds.
        await _today.maybe_regenerate(rt, user_id="usr_force", force=True)
        assert call_count["n"] == 1
    finally:
        tiers.utility = original


# ── Settle ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_settle_marks_row(monkeypatch):
    from augmentum.companion_runtime import tiers
    from augmentum.companion_runtime import today as _today
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_today_enabled", True)
    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")

    backend, rt = await _boot_runtime_with_user("usr_settle")
    _today._LAST_REGEN_AT.clear()
    original, _ = _mock_tier_call(
        tiers, "A short reflection for the settle test.",
    )
    try:
        await _today.maybe_regenerate(rt, user_id="usr_settle")
    finally:
        tiers.utility = original

    today_date = _today._local_date()
    await _today.settle_date(rt, user_id="usr_settle", date_local=today_date)

    row = await _today._read_row(rt, user_id="usr_settle", date_local=today_date)
    assert row is not None
    assert row.settled_at is not None


@pytest.mark.asyncio
async def test_settled_row_not_overwritten(monkeypatch):
    """Once settled, opportunistic regen should not modify content."""
    from augmentum.companion_runtime import tiers
    from augmentum.companion_runtime import today as _today
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_today_enabled", True)
    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")

    backend, rt = await _boot_runtime_with_user("usr_settled_immut")
    _today._LAST_REGEN_AT.clear()
    original, _ = _mock_tier_call(tiers, "Initial reflection prose v1.")
    try:
        await _today.maybe_regenerate(rt, user_id="usr_settled_immut")
    finally:
        tiers.utility = original

    today_date = _today._local_date()
    await _today.settle_date(rt, user_id="usr_settled_immut", date_local=today_date)

    # New call should observe the settled_at flag and return as-is.
    _today._LAST_REGEN_AT.clear()
    original2, _ = _mock_tier_call(tiers, "New v2 reflection that should NOT win.")
    try:
        result = await _today.maybe_regenerate(
            rt, user_id="usr_settled_immut", force=True,
        )
    finally:
        tiers.utility = original2

    assert result is not None
    assert "v1" in result.content_text
    assert "v2" not in result.content_text


# ── forget_refs ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_forget_quarantines_source_and_invalidates_today(monkeypatch):
    from augmentum.companion_runtime import tiers
    from augmentum.companion_runtime import today as _today
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_today_enabled", True)
    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")

    backend, rt = await _boot_runtime_with_user("usr_forget")

    # Seed a journal row to forget.
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(id, companion_id, user_id, content, entry_type, source, "
        " confidence_numeric, validation_score, quarantined, created_at) "
        "VALUES (99, ?, ?, 'A thread the user wants gone', "
        "        'observation', 'autonomous', 0.7, 1.0, 0, "
        "        datetime('now'))",
        (rt.companion_id, "usr_forget"),
    )
    await backend.conn.commit()

    # Generate today referencing that entry.
    _today._LAST_REGEN_AT.clear()
    original, _ = _mock_tier_call(
        tiers, "I mentioned the thread [journal:99] earlier today.",
    )
    try:
        await _today.maybe_regenerate(rt, user_id="usr_forget")
    finally:
        tiers.utility = original

    # Forget the journal row.
    count = await _today.forget_refs(
        rt, user_id="usr_forget", refs=[{"kind": "journal", "id": 99}],
    )
    assert count == 1

    # The journal row is now quarantined.
    cur = await backend.conn.execute(
        "SELECT quarantined, quarantine_reason FROM companion_journal "
        "WHERE id = 99",
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == 1
    assert row[1] == "user_correction"

    # Today's reflection should be flagged for rebuild (quarantined).
    today_row = await _today._read_row(
        rt, user_id="usr_forget", date_local=_today._local_date(),
    )
    assert today_row is not None
    assert today_row.quarantined is True
    assert today_row.quarantine_reason == "user_correction"


# ── Per-user isolation ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_users_do_not_see_each_others_reflections(monkeypatch):
    from augmentum.companion_runtime import tiers
    from augmentum.companion_runtime import today as _today
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_today_enabled", True)
    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")

    # Single backend, two users.
    from augmentum.companion_runtime.runtime import CompanionRuntime
    from augmentum.state.backends.sqlite import SQLiteBackend
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    for uid in ("usr_alice", "usr_bob"):
        await backend.conn.execute(
            "INSERT INTO users (id, username, password_hash, created_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (uid, uid, "x"),
        )
    await backend.conn.commit()
    rt = CompanionRuntime(backend, companion_id="becca")
    await rt.identity.load()
    await rt.state.load()

    _today._LAST_REGEN_AT.clear()
    original, _ = _mock_tier_call(tiers, "Alice's reflection text.")
    try:
        await _today.maybe_regenerate(rt, user_id="usr_alice")
    finally:
        tiers.utility = original
    _today._LAST_REGEN_AT.clear()
    original, _ = _mock_tier_call(tiers, "Bob's reflection text.")
    try:
        await _today.maybe_regenerate(rt, user_id="usr_bob")
    finally:
        tiers.utility = original

    alice_today = await _today.get_today(rt, user_id="usr_alice")
    bob_today = await _today.get_today(rt, user_id="usr_bob")
    assert alice_today is not None
    assert bob_today is not None
    assert "Alice" in alice_today.content_text
    assert "Bob" in bob_today.content_text
    assert alice_today.content_text != bob_today.content_text


# ── Gating ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_silent_presence_suppresses_generation(monkeypatch):
    from augmentum.companion_runtime import tiers
    from augmentum.companion_runtime import today as _today
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_today_enabled", True)
    monkeypatch.setattr(settings, "companion_presence_mode", "silent")

    backend, rt = await _boot_runtime_with_user("usr_silent")
    _today._LAST_REGEN_AT.clear()
    call_count = {"n": 0}
    async def _chat(req):
        call_count["n"] += 1
        return MagicMock(content="x", text="x")
    fake_backend = MagicMock()
    fake_backend.chat = _chat
    async def _fake_utility(rt, **kwargs):
        return fake_backend, "fake-model"
    original = tiers.utility
    tiers.utility = _fake_utility
    try:
        result = await _today.maybe_regenerate(rt, user_id="usr_silent")
    finally:
        tiers.utility = original
    assert result is None
    assert call_count["n"] == 0


@pytest.mark.asyncio
async def test_today_disabled_suppresses_generation(monkeypatch):
    from augmentum.companion_runtime import tiers
    from augmentum.companion_runtime import today as _today
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_today_enabled", False)
    monkeypatch.setattr(settings, "companion_presence_mode", "gentle")

    backend, rt = await _boot_runtime_with_user("usr_off")
    _today._LAST_REGEN_AT.clear()
    call_count = {"n": 0}
    async def _chat(req):
        call_count["n"] += 1
        return MagicMock(content="x", text="x")
    fake_backend = MagicMock()
    fake_backend.chat = _chat
    async def _fake_utility(rt, **kwargs):
        return fake_backend, "fake-model"
    original = tiers.utility
    tiers.utility = _fake_utility
    try:
        result = await _today.maybe_regenerate(rt, user_id="usr_off")
    finally:
        tiers.utility = original
    assert result is None
    assert call_count["n"] == 0
