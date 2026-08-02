"""Tests for the journal dedup + observer mode-change-noop + salience
machine-shaped guard. All driven by the production-observed pip spam.

Coverage:

- _compute_content_hash normalizes consistently
- journal() inserts on first write, hits dedup on repeat within window
- Repetition_count increments on dedup hit
- Dedup window is configurable; window=0 disables
- Salience scorer rejects tool-result / synthesis-hint user_text
- Salience scorer no longer flags "again" / "broken" as frustrated
- Observer's mode.changed handler updates state but doesn't journal
"""

from __future__ import annotations

import asyncio
import time

import pytest


async def _boot_backend_with_runtime(*, companion_id: str = "becca"):
    """Fresh :memory: backend with full migrations + a runtime + memory facade."""
    from augmentum.companion_runtime.runtime import CompanionRuntime
    from augmentum.memory.store import MemoryStore
    from augmentum.memory.core_profile import CoreProfileManager
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    runtime = CompanionRuntime(backend, companion_id=companion_id)
    await runtime.start(
        memory_store=MemoryStore(backend),
        core_profile=CoreProfileManager(backend),
    )
    return runtime, backend


# ── content_hash helper ──────────────────────────────────────────────


def test_compute_content_hash_normalizes_whitespace_and_case():
    from augmentum.companion_runtime.memory import _compute_content_hash

    a = _compute_content_hash("Hello   world.")
    b = _compute_content_hash("hello world.")
    c = _compute_content_hash("HELLO    WORLD.")
    assert a == b == c


def test_compute_content_hash_empty_returns_empty():
    from augmentum.companion_runtime.memory import _compute_content_hash

    assert _compute_content_hash("") == ""
    assert _compute_content_hash("   ") == ""


def test_compute_content_hash_truncates_to_200_chars():
    """Two strings with the same first 200 chars hash identically."""
    from augmentum.companion_runtime.memory import _compute_content_hash

    base = "x" * 200
    a = _compute_content_hash(base + " differs after 200")
    b = _compute_content_hash(base + " also differs after 200")
    assert a == b


# ── journal() dedup ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_journal_dedup_skips_within_window(monkeypatch):
    """Same content within the window → no new insert, returns existing id."""
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "companion_journal_dedup_window_minutes", 240)

    runtime, backend = await _boot_backend_with_runtime()
    try:
        first_id = await runtime.memory.journal(
            content="The light is hitting the page at a specific angle.",
            entry_type="noticing",
            affect_tag="alert",
            embed=False,
        )
        assert first_id > 0

        # Second write with the same content
        second_id = await runtime.memory.journal(
            content="The light is hitting the page at a specific angle.",
            entry_type="noticing",
            affect_tag="alert",
            embed=False,
        )
        # Returns the SAME id — dedup hit
        assert second_id == first_id

        # Verify only one row exists with that content
        cur = await backend.conn.execute(
            "SELECT COUNT(*) FROM companion_journal WHERE content = ?",
            ("The light is hitting the page at a specific angle.",),
        )
        count = (await cur.fetchone())[0]
        await cur.close()
        assert count == 1
    finally:
        await runtime.stop(grace_seconds=1.0)
        await backend.close()


@pytest.mark.asyncio
async def test_journal_dedup_bumps_repetition_count(monkeypatch):
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "companion_journal_dedup_window_minutes", 240)

    runtime, backend = await _boot_backend_with_runtime()
    try:
        await runtime.memory.journal(
            content="dust motes are still dancing", embed=False,
        )
        for _ in range(3):
            await runtime.memory.journal(
                content="dust motes are still dancing", embed=False,
            )
        cur = await backend.conn.execute(
            "SELECT repetition_count FROM companion_journal WHERE content = ?",
            ("dust motes are still dancing",),
        )
        rep = (await cur.fetchone())[0]
        await cur.close()
        assert rep == 4  # 1 initial + 3 dedup'd hits
    finally:
        await runtime.stop(grace_seconds=1.0)
        await backend.close()


@pytest.mark.asyncio
async def test_journal_dedup_zero_window_disables(monkeypatch):
    """When the window is 0, dedup is disabled — same content writes
    produce N rows."""
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "companion_journal_dedup_window_minutes", 0)

    runtime, backend = await _boot_backend_with_runtime()
    try:
        for _ in range(3):
            await runtime.memory.journal(content="same again", embed=False)
        cur = await backend.conn.execute(
            "SELECT COUNT(*) FROM companion_journal WHERE content = ?",
            ("same again",),
        )
        count = (await cur.fetchone())[0]
        await cur.close()
        assert count == 3
    finally:
        await runtime.stop(grace_seconds=1.0)
        await backend.close()


@pytest.mark.asyncio
async def test_journal_dedup_different_content_inserts_normally(monkeypatch):
    """Different content writes don't accidentally dedup against each other."""
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "companion_journal_dedup_window_minutes", 240)

    runtime, backend = await _boot_backend_with_runtime()
    try:
        await runtime.memory.journal(content="first thought", embed=False)
        await runtime.memory.journal(content="second thought", embed=False)
        await runtime.memory.journal(content="third thought", embed=False)
        cur = await backend.conn.execute(
            "SELECT COUNT(*) FROM companion_journal WHERE content LIKE '% thought'"
        )
        count = (await cur.fetchone())[0]
        await cur.close()
        assert count == 3
    finally:
        await runtime.stop(grace_seconds=1.0)
        await backend.close()


# ── Salience: machine-shaped guards ──────────────────────────────────


@pytest.mark.asyncio
async def test_salience_rejects_tool_result_user_text():
    """Tool-result blocks threaded as user-role messages must not score."""
    from augmentum.companion_runtime.salience import score

    m = await score(
        user_text="## Tool Result (image_generation)\nImage generated successfully and is now visible in the gallery. Do not call image_generation again.",
        assistant_text="It's lovely.",
        mode="passthrough",
    )
    assert m is None


@pytest.mark.asyncio
async def test_salience_rejects_synthesis_hint():
    from augmentum.companion_runtime.salience import score

    m = await score(
        user_text="Synthesize the results into a clear response. Cite sources [1], [2] where applicable.",
        assistant_text="Here's what I found.",
        mode="passthrough",
    )
    assert m is None


@pytest.mark.asyncio
async def test_salience_accepts_normal_user_text():
    """Real user prose must still score normally."""
    from augmentum.companion_runtime.salience import score

    m = await score(
        user_text="I've been thinking about what we discussed last week and it's been sitting with me.",
        assistant_text="Tell me.",
        mode="passthrough",
    )
    assert m is not None
    # Just confirms scoring happens — exact threshold depends on
    # lexicon/disclosure tuning. Anything above the floor is healthy.
    assert m.salience > 0.2
    # And the text wasn't rejected as machine-shaped
    assert "thinking" in m.text.lower()


# ── Salience: affect lexicon tightened ───────────────────────────────


@pytest.mark.asyncio
async def test_salience_does_not_flag_again_as_frustrated():
    """Pre-fix 'again' in any sentence flipped to frustrated affect.
    Post-fix only specific frustration markers trip the tag."""
    from augmentum.companion_runtime.salience import score

    m = await score(
        user_text="I'd like to try this again with a different approach.",
        assistant_text="OK.",
        mode="passthrough",
    )
    assert m is not None
    assert m.user_affect != "frustrated"


@pytest.mark.asyncio
async def test_salience_does_not_flag_broken_as_frustrated():
    """'broken' is too generic — could be code/objects, not feelings."""
    from augmentum.companion_runtime.salience import score

    m = await score(
        user_text="The build is broken — getting a missing dependency error.",
        assistant_text="Let me look.",
        mode="passthrough",
    )
    assert m is not None
    assert m.user_affect != "frustrated"


@pytest.mark.asyncio
async def test_salience_still_flags_genuine_frustration():
    """The tightened lexicon must still catch real frustration."""
    from augmentum.companion_runtime.salience import score

    m = await score(
        user_text="I'm so stuck on this — been trying for an hour and nothing works. Ugh.",
        assistant_text="OK, let's slow down.",
        mode="passthrough",
    )
    assert m is not None
    assert m.user_affect == "frustrated"


# ── Observer: mode.changed no longer journals ────────────────────────


@pytest.mark.asyncio
async def test_observer_mode_changed_does_not_journal():
    """The 165 'user shifted from X to Y' templated noticings were
    coming from BeccaObserver._handle. Verify they no longer write."""
    from augmentum.companion_runtime.bus import PresenceBus
    from augmentum.companion_runtime.observer import BeccaObserver

    journals: list = []

    class _FakeMemory:
        async def journal(self, content, **kwargs):
            journals.append({"content": content, **kwargs})
            return 1

    class _FakeRuntime:
        bus = PresenceBus()
        companion_id = "becca"
        memory = _FakeMemory()

    runtime = _FakeRuntime()
    observer = BeccaObserver(runtime)
    await observer.start()
    try:
        await runtime.bus.publish_topic(
            "mode.changed",
            {"from": "passthrough", "to": "narrative", "session_id": "s1"},
            source_companion_id="becca",
        )
        await asyncio.sleep(0.1)
        # No journal write should have happened
        assert journals == []
        # But the observed_state was updated
        assert runtime.observed_state["last_mode_change"] == {
            "from": "passthrough",
            "to": "narrative",
            "at": runtime.observed_state["last_mode_change"]["at"],
        }
    finally:
        await observer.stop()
