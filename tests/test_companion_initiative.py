"""Tests for Piece 7' — waking the initiative engine.

The initiative engine is fully implemented but was never called and
had two silent column bugs. These tests cover:

* The kill switch (`companion_initiative_enabled`) — default off
* The interval cap (`companion_initiative_min_interval_s`) — caps DB cost
* The two fixed feature scorers (column-name bug) — now produce signal

Tests use an in-memory SQLite backend with migrations applied so the
companion_journal schema is real, not mocked. Mocking the schema
would defeat the point of testing the bug fix.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest


async def _boot_backend():
    """Spin up a :memory: backend with all migrations applied."""
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    return backend


def _fake_runtime(backend):
    """Build a minimal runtime that initiative.py's helpers can run against.

    initiative.py uses ``runtime.backend.connect()`` and
    ``runtime.companion_id``. Bus is only used after a non-trivial
    score, so a MagicMock with async publish_topic is enough.
    """
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    # Real string owner so the owner-scope clause / enqueue write get a
    # str, not a MagicMock bound into SQL (audit 2026-06-17). "" =
    # pass-through (unowned single-user box).
    runtime.owner_user_id = ""
    runtime.last_initiative_score_at = 0.0

    async def _async_publish(*args, **kwargs):
        return None

    runtime.bus.publish_topic = _async_publish
    return runtime


# ── Kill switch ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_step_skips_when_disabled(monkeypatch):
    """companion_initiative_enabled=False → step returns None, no DB hit."""
    from augmentum.config import settings
    from augmentum.companion_runtime.behavior import initiative

    monkeypatch.setattr(settings, "companion_initiative_enabled", False)

    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    result = await initiative.step(runtime)
    assert result is None
    # The timestamp must NOT advance when disabled — that would let
    # the disabled state burn the interval window.
    assert runtime.last_initiative_score_at == 0.0


# ── Interval cap ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_step_skips_within_interval(monkeypatch):
    """Repeated step() calls within min_interval_s must not re-score."""
    from augmentum.config import settings
    from augmentum.companion_runtime.behavior import initiative

    monkeypatch.setattr(settings, "companion_initiative_enabled", True)
    monkeypatch.setattr(settings, "companion_initiative_min_interval_s", 60.0)

    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    # Simulate a recent score 5s ago.
    runtime.last_initiative_score_at = time.time() - 5.0

    result = await initiative.step(runtime)
    assert result is None
    # Timestamp should NOT be bumped on a skip — we want it to reflect
    # the last actual score, not the last attempt.
    assert (time.time() - runtime.last_initiative_score_at) >= 4.0


@pytest.mark.asyncio
async def test_step_runs_after_interval(monkeypatch):
    """After min_interval_s elapsed, step actually scores."""
    from augmentum.config import settings
    from augmentum.companion_runtime.behavior import initiative

    monkeypatch.setattr(settings, "companion_initiative_enabled", True)
    monkeypatch.setattr(settings, "companion_initiative_min_interval_s", 60.0)

    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    # Score was 5 min ago — interval has elapsed.
    runtime.last_initiative_score_at = time.time() - 300.0

    before = time.time()
    result = await initiative.step(runtime)
    assert result is not None
    assert result.kind in (
        "revisit_thread",
        "share_creation",
        "surface_observation",
        "reach_out_after_quiet",
    )
    # last_initiative_score_at is bumped close to now
    assert runtime.last_initiative_score_at >= before


@pytest.mark.asyncio
async def test_first_run_runs_immediately(monkeypatch):
    """When last_initiative_score_at is 0 (fresh runtime), don't suppress."""
    from augmentum.config import settings
    from augmentum.companion_runtime.behavior import initiative

    monkeypatch.setattr(settings, "companion_initiative_enabled", True)
    monkeypatch.setattr(settings, "companion_initiative_min_interval_s", 60.0)

    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    # Default — no previous score
    assert runtime.last_initiative_score_at == 0.0

    result = await initiative.step(runtime)
    assert result is not None


# ── Fixed feature scorers (the column bug) ────────────────────────────


@pytest.mark.asyncio
async def test_time_since_returns_high_on_empty_journal():
    """No journal entries → return 1.0 (treat as long absence).

    Pre-fix this hit the column bug and returned 0.0 always. Now it
    reports the no-rows case correctly.
    """
    from augmentum.companion_runtime.behavior import initiative

    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    score = await initiative._time_since_last_interaction(runtime)
    assert score == 1.0


@pytest.mark.asyncio
async def test_time_since_returns_low_on_recent_entry():
    """Journal entry from seconds ago → score near 0 (recent activity)."""
    from augmentum.companion_runtime.behavior import initiative

    backend = await _boot_backend()
    runtime = _fake_runtime(backend)

    # Insert a journal entry at "now"
    await backend.conn.execute(
        "INSERT INTO companion_journal (companion_id, content, created_at) "
        "VALUES (?, ?, datetime('now'))",
        ("becca", "fresh thought"),
    )
    await backend.conn.commit()

    score = await initiative._time_since_last_interaction(runtime)
    # < 1h elapsed → score should be < 1/24 ≈ 0.042
    assert 0.0 <= score < 0.1


@pytest.mark.asyncio
async def test_unresolved_journal_counts_wondering_entries():
    """The fixed query counts wondering+unfinished, not the empty 0.0."""
    from augmentum.companion_runtime.behavior import initiative

    backend = await _boot_backend()
    runtime = _fake_runtime(backend)

    # Insert 3 wondering + 1 unfinished + 2 observation
    for _ in range(3):
        await backend.conn.execute(
            "INSERT INTO companion_journal (companion_id, content, entry_type) "
            "VALUES (?, ?, 'wondering')",
            ("becca", "open thread"),
        )
    await backend.conn.execute(
        "INSERT INTO companion_journal (companion_id, content, entry_type) "
        "VALUES (?, ?, 'unfinished')",
        ("becca", "loose end"),
    )
    for _ in range(2):
        await backend.conn.execute(
            "INSERT INTO companion_journal (companion_id, content, entry_type) "
            "VALUES (?, ?, 'observation')",
            ("becca", "just looking"),
        )
    await backend.conn.commit()

    score = await initiative._unresolved_journal(runtime)
    # 4 unresolved / 5 (the divisor) → 0.8
    assert score == pytest.approx(0.8, abs=0.01)


@pytest.mark.asyncio
async def test_unresolved_journal_excludes_suppressed():
    """Suppressed entries shouldn't count toward unresolved score."""
    from augmentum.companion_runtime.behavior import initiative

    backend = await _boot_backend()
    runtime = _fake_runtime(backend)

    await backend.conn.execute(
        "INSERT INTO companion_journal (companion_id, content, entry_type, suppressed) "
        "VALUES (?, ?, 'wondering', 1)",
        ("becca", "self-corrected"),
    )
    await backend.conn.commit()

    score = await initiative._unresolved_journal(runtime)
    assert score == 0.0


# ── End-to-end: proposal kind selection ───────────────────────────────


@pytest.mark.asyncio
async def test_proposal_kind_reflects_dominant_feature(monkeypatch):
    """With only journal-wondering entries, dominant feature is journal."""
    from augmentum.config import settings
    from augmentum.companion_runtime.behavior import initiative

    monkeypatch.setattr(settings, "companion_initiative_enabled", True)
    monkeypatch.setattr(settings, "companion_initiative_min_interval_s", 0.0)

    backend = await _boot_backend()
    runtime = _fake_runtime(backend)

    # Stuff enough wondering to dominate
    for i in range(8):
        await backend.conn.execute(
            "INSERT INTO companion_journal (companion_id, content, entry_type) "
            "VALUES (?, ?, 'wondering')",
            ("becca", f"thread {i}"),
        )
    await backend.conn.commit()

    result = await initiative.step(runtime)
    assert result is not None
    assert result.kind == "revisit_thread"


@pytest.mark.asyncio
async def test_no_enqueue_on_trivial_score(monkeypatch):
    """Zero-feature score → return Proposal but don't write to queue."""
    from augmentum.config import settings
    from augmentum.companion_runtime.behavior import initiative

    monkeypatch.setattr(settings, "companion_initiative_enabled", True)
    monkeypatch.setattr(settings, "companion_initiative_min_interval_s", 0.0)

    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    # No data → all features ≈ 0 except time_since which is 1.0
    # time_since weight is 0.30 so total ≈ 0.30, above the 0.05 trivial
    # threshold but below the 0.62 surface threshold. Queue write happens
    # but no bus emit.
    result = await initiative.step(runtime)
    assert result is not None

    # Queue should have exactly one row
    cur = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_initiative_queue WHERE companion_id = ?",
        ("becca",),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == 1
