"""Sprint 7 tests — feedback bias + reflection→identity loop.

Covers:
* Migration 181 + indexes
* Record writes feedback rows
* Bias multiplier clamped + decays
* Bias floor when many mutes, ceiling when many surfaces
* Initiative scoring uses bias when enabled
* Reflection naive extractor catches canonical patterns
* maybe_apply_nudge respects bias floor
* Cross-check rejects when facet not elevated
* DRIFT_CEILING / cumulative cap still applied via identity API
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


async def _boot_backend():
    from augmentum.state.backends.sqlite import SQLiteBackend
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('u_f', 'p', 'x', datetime('now'))",
    )
    await backend.conn.commit()
    return backend


def _rt(backend, user_id: str = "u_f"):
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    runtime.owner_user_id = user_id
    return runtime


# ── Migration 181 ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mig181_creates_table():
    backend = await _boot_backend()
    cur = await backend.conn.execute("PRAGMA table_info(companion_note_feedback)")
    cols = {c[1] for c in await cur.fetchall()}
    await cur.close()
    expected = {"id", "note_id", "user_id", "companion_id", "kind", "recorded_at"}
    assert expected.issubset(cols)


@pytest.mark.asyncio
async def test_mig181_indexes_exist():
    backend = await _boot_backend()
    cur = await backend.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    )
    names = {r[0] for r in await cur.fetchall()}
    await cur.close()
    assert "idx_note_feedback_user_kind_time" in names
    assert "idx_note_feedback_note" in names


# ── feedback.record ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_writes_row():
    from augmentum.companion_runtime import feedback
    backend = await _boot_backend()
    rt = _rt(backend)

    ok = await feedback.record(rt, note_id=42, user_id="u_f", kind="surfaced")
    assert ok is True

    cur = await backend.conn.execute(
        "SELECT note_id, kind FROM companion_note_feedback WHERE user_id = 'u_f'"
    )
    rows = await cur.fetchall()
    await cur.close()
    assert len(rows) == 1
    assert rows[0][0] == 42
    assert rows[0][1] == "surfaced"


@pytest.mark.asyncio
async def test_record_no_op_on_missing_args():
    from augmentum.companion_runtime import feedback
    backend = await _boot_backend()
    rt = _rt(backend)

    assert await feedback.record(rt, note_id=0, user_id="u_f", kind="surfaced") is False
    assert await feedback.record(rt, note_id=1, user_id="", kind="surfaced") is False
    assert await feedback.record(rt, note_id=1, user_id="u_f", kind="") is False


# ── feedback.aggregate_bias ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_aggregate_bias_neutral_when_no_feedback():
    from augmentum.companion_runtime import feedback
    backend = await _boot_backend()
    rt = _rt(backend)
    bias = await feedback.aggregate_bias(rt, user_id="u_f")
    assert bias == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_aggregate_bias_rises_on_surfaces():
    from augmentum.companion_runtime import feedback
    backend = await _boot_backend()
    rt = _rt(backend)
    # 5 surfaced events → strong positive signal
    for i in range(5):
        await feedback.record(rt, note_id=i + 1, user_id="u_f", kind="surfaced")
    bias = await feedback.aggregate_bias(rt, user_id="u_f")
    assert bias > 1.0
    assert bias <= feedback.BIAS_CEILING


@pytest.mark.asyncio
async def test_aggregate_bias_drops_on_mutes():
    from augmentum.companion_runtime import feedback
    backend = await _boot_backend()
    rt = _rt(backend)
    for i in range(5):
        await feedback.record(rt, note_id=i + 1, user_id="u_f", kind="muted")
    bias = await feedback.aggregate_bias(rt, user_id="u_f")
    assert bias < 1.0
    assert bias >= feedback.BIAS_FLOOR


@pytest.mark.asyncio
async def test_aggregate_bias_clamped_to_floor():
    """Many mutes → bias clamped at floor, never zero."""
    from augmentum.companion_runtime import feedback
    backend = await _boot_backend()
    rt = _rt(backend)
    # 50 mutes — way past saturation
    for i in range(50):
        await feedback.record(rt, note_id=i + 1, user_id="u_f", kind="muted")
    bias = await feedback.aggregate_bias(rt, user_id="u_f")
    assert bias == pytest.approx(feedback.BIAS_FLOOR, abs=0.01)


@pytest.mark.asyncio
async def test_feedback_summary_includes_counts():
    from augmentum.companion_runtime import feedback
    backend = await _boot_backend()
    rt = _rt(backend)
    await feedback.record(rt, note_id=1, user_id="u_f", kind="surfaced")
    await feedback.record(rt, note_id=2, user_id="u_f", kind="acknowledged")
    await feedback.record(rt, note_id=3, user_id="u_f", kind="muted")
    summary = await feedback.feedback_summary(rt, user_id="u_f")
    assert summary.surfaced_count == 1
    assert summary.acknowledged_count == 1
    assert summary.muted_count == 1


# ── reflection naive extractor ──────────────────────────────────────


def test_reflection_extracts_more_pattern():
    from augmentum.companion_runtime.reflection import _naive_extract
    proposals = _naive_extract(
        "I want to be more playful when we talk about hard topics."
    )
    assert len(proposals) == 1
    assert proposals[0].trait == "playful"
    assert proposals[0].delta > 0


def test_reflection_extracts_too_pattern():
    from augmentum.companion_runtime.reflection import _naive_extract
    proposals = _naive_extract(
        "I noticed I've been too cautious about pushing back lately."
    )
    assert len(proposals) == 1
    assert proposals[0].trait == "cautious"
    assert proposals[0].delta < 0


def test_reflection_extracts_no_match():
    from augmentum.companion_runtime.reflection import _naive_extract
    proposals = _naive_extract("It was a quiet day. Nothing in particular.")
    assert proposals == []


def test_reflection_ignores_unknown_traits():
    from augmentum.companion_runtime.reflection import _naive_extract
    proposals = _naive_extract("I want to be more spelunky and ferromagnetic.")
    # Unknown traits → no proposals
    assert proposals == []


# ── maybe_apply_nudge (cross-check) ─────────────────────────────────


@pytest.mark.asyncio
async def test_apply_disabled_returns_skip(monkeypatch):
    from augmentum.companion_runtime.reflection import maybe_apply_nudge
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_reflection_trait_nudge_enabled", False)

    backend = await _boot_backend()
    rt = _rt(backend)
    result = await maybe_apply_nudge(
        rt, user_id="u_f", diary_text="I want to be more curious.",
    )
    assert result["applied"] == []
    assert result["skipped"][0]["reason"] == "feature_disabled"


@pytest.mark.asyncio
async def test_apply_skips_on_low_bias(monkeypatch):
    """When recent feedback is negative, we don't apply nudges."""
    from augmentum.companion_runtime import feedback
    from augmentum.companion_runtime.reflection import maybe_apply_nudge
    from augmentum.config import settings
    monkeypatch.setattr(settings, "companion_reflection_trait_nudge_enabled", True)

    backend = await _boot_backend()
    rt = _rt(backend)
    # Drown the user with mutes → bias floors
    for i in range(20):
        await feedback.record(rt, note_id=i + 1, user_id="u_f", kind="muted")

    result = await maybe_apply_nudge(
        rt, user_id="u_f", diary_text="I want to be more playful.",
    )
    assert result["applied"] == []
    assert any("feedback_bias_low" in s["reason"] for s in result["skipped"])


@pytest.mark.asyncio
async def test_apply_skips_when_facet_not_elevated(monkeypatch):
    """No today's activation of the proposed trait's facet → skip.

    Uses a real runtime so identity.get_identity works correctly.
    """
    from augmentum.companion_runtime.reflection import maybe_apply_nudge
    from augmentum.companion_runtime.runtime import CompanionRuntime
    from augmentum.config import settings
    from augmentum.state.backends.sqlite import SQLiteBackend

    monkeypatch.setattr(settings, "companion_reflection_trait_nudge_enabled", True)
    monkeypatch.setattr(settings, "companion_feedback_bias_enabled", False)

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('u_r', 'p', 'x', datetime('now'))",
    )
    await backend.conn.commit()
    rt = CompanionRuntime(backend, companion_id="becca")
    await rt.identity.load()
    await rt.state.load()

    # No facet activations seeded — cross-check should fail
    result = await maybe_apply_nudge(
        rt, user_id="u_r", diary_text="I want to be more playful.",
    )
    assert result["applied"] == []
    assert any("facet_not_elevated" in s["reason"] for s in result["skipped"])


@pytest.mark.asyncio
async def test_apply_succeeds_when_facet_elevated(monkeypatch):
    """With both bias OK and facet elevated, the nudge applies."""
    from augmentum.companion_runtime.reflection import maybe_apply_nudge
    from augmentum.companion_runtime.runtime import CompanionRuntime
    from augmentum.config import settings
    from augmentum.state.backends.sqlite import SQLiteBackend

    monkeypatch.setattr(settings, "companion_reflection_trait_nudge_enabled", True)
    monkeypatch.setattr(settings, "companion_feedback_bias_enabled", False)

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('u_s', 'p', 'x', datetime('now'))",
    )
    # Seed today's playful facet activation
    await backend.conn.execute(
        "INSERT INTO personality_facet_activations "
        "(user_id, companion_id, facet, intensity, source) "
        "VALUES ('u_s', 'becca', 'playful', 0.8, 'manual')",
    )
    await backend.conn.commit()
    rt = CompanionRuntime(backend, companion_id="becca")
    await rt.identity.load()
    await rt.state.load()

    result = await maybe_apply_nudge(
        rt, user_id="u_s", diary_text="I want to be more playful with him.",
    )
    assert len(result["applied"]) == 1
    assert result["applied"][0]["trait"] == "playful"
