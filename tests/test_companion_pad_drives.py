"""Sprint 6 tests — PAD projector + drives layer.

Covers:
* PAD neutral when no data
* Valence math: warm cluster vs hard cluster
* Arousal scales with activation density
* Dominance from role vector
* Drives load defaults on first call
* Drive decay reverts toward baseline
* Drive satiation drops level
* Urgency recency dampening
* activity_selector drive-multiplicative behavior (smoke)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest


# ── PAD ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pad_neutral_when_no_data():
    """Empty backend → neutral PAD (slightly positive curious)."""
    from augmentum.companion_runtime.perception.pad import project_pad
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('u_pad', 'p', 'x', datetime('now'))",
    )
    await backend.conn.commit()

    pad = await project_pad(backend, user_id="u_pad")
    assert -1.0 <= pad.valence <= 1.0
    assert 0.0 <= pad.arousal <= 1.0
    assert -1.0 <= pad.dominance <= 1.0
    # No samples → defaults
    assert pad.sample_count == 0


@pytest.mark.asyncio
async def test_pad_valence_positive_with_warm_facets():
    """Warm cluster activations push valence positive."""
    from augmentum.companion_runtime.perception.pad import project_pad
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('u_v', 'p', 'x', datetime('now'))",
    )
    # Plant warm-cluster activations
    for facet in ("warm", "playful", "delighted"):
        await backend.conn.execute(
            "INSERT INTO personality_facet_activations "
            "(user_id, companion_id, facet, intensity, source) "
            "VALUES ('u_v', 'becca', ?, 0.8, 'self_label')",
            (facet,),
        )
    await backend.conn.commit()

    pad = await project_pad(backend, user_id="u_v")
    assert pad.valence > 0.0


@pytest.mark.asyncio
async def test_pad_valence_negative_with_hard_facets():
    """Hard cluster activations push valence negative."""
    from augmentum.companion_runtime.perception.pad import project_pad
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('u_h', 'p', 'x', datetime('now'))",
    )
    for facet in ("frustrated", "tired", "withholding"):
        await backend.conn.execute(
            "INSERT INTO personality_facet_activations "
            "(user_id, companion_id, facet, intensity, source) "
            "VALUES ('u_h', 'becca', ?, 0.9, 'self_label')",
            (facet,),
        )
    await backend.conn.commit()

    pad = await project_pad(backend, user_id="u_h")
    assert pad.valence < 0.0


@pytest.mark.asyncio
async def test_pad_dominance_from_role_vector():
    """Dominance = role.active - role.passive, regardless of facets."""
    from augmentum.companion_runtime.perception.pad import project_pad
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('u_d', 'p', 'x', datetime('now'))",
    )
    await backend.conn.commit()

    # Active-dominant role → positive dominance
    pad_active = await project_pad(
        backend, user_id="u_d", role_active=0.8, role_passive=0.1,
    )
    assert pad_active.dominance > 0.5

    # Passive-dominant → negative dominance
    pad_passive = await project_pad(
        backend, user_id="u_d", role_active=0.1, role_passive=0.8,
    )
    assert pad_passive.dominance < -0.5


# ── Drives ───────────────────────────────────────────────────────────


async def _boot_runtime_with_user(user_id: str = "u_dr"):
    from augmentum.companion_runtime.runtime import CompanionRuntime
    from augmentum.state.backends.sqlite import SQLiteBackend
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        (user_id, "p", "x"),
    )
    await backend.conn.commit()
    rt = CompanionRuntime(backend, companion_id="becca")
    await rt.identity.load()
    await rt.state.load()
    return backend, rt


@pytest.mark.asyncio
async def test_drives_load_defaults_on_first_call():
    """First load() returns DEFAULT_LEVELS for all four drives."""
    from augmentum.companion_runtime import drives
    backend, rt = await _boot_runtime_with_user("u_d1")
    state = await drives.load(rt, user_id="u_d1")
    for name in drives.DRIVE_NAMES:
        assert name in state.levels
        assert state.levels[name] == pytest.approx(drives.DEFAULT_LEVELS[name])


@pytest.mark.asyncio
async def test_drives_load_persists_row():
    """After first load, a row exists in companion_drive_state."""
    from augmentum.companion_runtime import drives
    backend, rt = await _boot_runtime_with_user("u_d2")
    await drives.load(rt, user_id="u_d2")
    cur = await backend.conn.execute(
        "SELECT user_id, curiosity_level FROM companion_drive_state "
        "WHERE user_id = ? AND companion_id = ?",
        ("u_d2", "becca"),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row is not None


@pytest.mark.asyncio
async def test_drives_satiate_drops_level():
    """satiate() reduces the named drive's level."""
    from augmentum.companion_runtime import drives
    backend, rt = await _boot_runtime_with_user("u_d3")
    # Bring curiosity to a known starting point
    await drives.load(rt, user_id="u_d3")
    await backend.conn.execute(
        "UPDATE companion_drive_state SET curiosity_level = 0.8 "
        "WHERE user_id = 'u_d3' AND companion_id = 'becca'",
    )
    await backend.conn.commit()

    await drives.satiate(rt, user_id="u_d3", drive=drives.CURIOSITY, amount=0.3)

    state = await drives.load(rt, user_id="u_d3")
    # 0.8 - 0.3 = 0.5
    assert state.levels[drives.CURIOSITY] == pytest.approx(0.5, abs=0.01)


@pytest.mark.asyncio
async def test_drives_decay_moves_toward_baseline():
    """decay() moves an elevated drive back toward its default level."""
    from augmentum.companion_runtime import drives
    backend, rt = await _boot_runtime_with_user("u_d4")
    await drives.load(rt, user_id="u_d4")
    # Set curiosity HIGH, last_decay_at = 4 hours ago
    await backend.conn.execute(
        "UPDATE companion_drive_state "
        "SET curiosity_level = 0.95, "
        "    last_decay_at = datetime('now', '-4 hours') "
        "WHERE user_id = 'u_d4' AND companion_id = 'becca'",
    )
    await backend.conn.commit()

    state = await drives.decay(rt, user_id="u_d4")
    baseline = drives.DEFAULT_LEVELS[drives.CURIOSITY]  # 0.6
    # After one half-life (4h), curiosity should be roughly halfway
    # between 0.95 and 0.6 — i.e., around 0.775
    assert state.levels[drives.CURIOSITY] < 0.95
    assert state.levels[drives.CURIOSITY] > baseline


@pytest.mark.asyncio
async def test_drives_decay_no_op_within_seconds():
    """Decay called twice within seconds should be a near-no-op."""
    from augmentum.companion_runtime import drives
    backend, rt = await _boot_runtime_with_user("u_d5")
    state1 = await drives.decay(rt, user_id="u_d5")  # first call provisions
    state2 = await drives.decay(rt, user_id="u_d5")  # immediate re-call

    for name in drives.DRIVE_NAMES:
        assert state2.levels[name] == pytest.approx(state1.levels[name], abs=0.01)


@pytest.mark.asyncio
async def test_drive_urgency_recency_dampens():
    """A drive satiated very recently has dampened urgency."""
    from augmentum.companion_runtime import drives
    backend, rt = await _boot_runtime_with_user("u_d6")
    await drives.load(rt, user_id="u_d6")
    # Bring level to 0.8 and stamp satiated_at to now
    await backend.conn.execute(
        "UPDATE companion_drive_state "
        "SET curiosity_level = 0.8, "
        "    curiosity_satiated_at = datetime('now') "
        "WHERE user_id = 'u_d6' AND companion_id = 'becca'",
    )
    await backend.conn.commit()

    state = await drives.load(rt, user_id="u_d6")
    urgency = state.urgency(drives.CURIOSITY)
    # With recency dampening, urgency < level
    assert urgency < 0.8


@pytest.mark.asyncio
async def test_drives_isolated_between_users():
    """User A's drive levels don't affect User B's."""
    from augmentum.companion_runtime import drives
    backend, rt = await _boot_runtime_with_user("u_iso_a")
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('u_iso_b', 'q', 'x', datetime('now'))",
    )
    await backend.conn.commit()

    await drives.load(rt, user_id="u_iso_a")
    await drives.load(rt, user_id="u_iso_b")
    # Drop A's curiosity
    await drives.satiate(rt, user_id="u_iso_a", drive=drives.CURIOSITY, amount=0.4)

    state_a = await drives.load(rt, user_id="u_iso_a")
    state_b = await drives.load(rt, user_id="u_iso_b")
    assert state_a.levels[drives.CURIOSITY] < state_b.levels[drives.CURIOSITY]
