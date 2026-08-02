"""Tests for Piece 1 runtime code-side — lazy_provision + per-user access.

Verifies:
* get_identity(user_id) returns a per-user-scoped CompanionIdentity
* get_state(user_id) returns a per-user-scoped CompanionState
* lazy_provision creates rows for identity, state, scene, and genesis
* lazy_provision is idempotent
* Provisioned rows seed from the canonical seed (user_id='') when available
* Two users' identities don't share data (isolation invariant)
* Legacy self.identity / self.state still resolve to the seed singleton
"""

from __future__ import annotations

import pytest


async def _boot_runtime():
    """Build a CompanionRuntime against an in-memory backend with mig 179 applied."""
    from augmentum.companion_runtime.runtime import CompanionRuntime
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    rt = CompanionRuntime(backend, companion_id="becca")
    # Load the seed singleton (user_id='') so self.identity is hydrated.
    await rt.identity.load()
    await rt.state.load()
    return backend, rt


# ── lazy_provision ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lazy_provision_creates_rows_for_new_user():
    """First lazy_provision for a fresh user_id creates identity, state, scene rows."""
    backend, rt = await _boot_runtime()
    provisioned = await rt.lazy_provision("usr_new")
    assert provisioned is True

    for table in ("companion_identities", "companion_state", "companion_scene"):
        cur = await backend.conn.execute(
            f"SELECT 1 FROM {table} WHERE user_id = ? AND companion_id = ?",
            ("usr_new", "becca"),
        )
        row = await cur.fetchone()
        await cur.close()
        assert row is not None, f"{table} missing row for usr_new"


@pytest.mark.asyncio
async def test_lazy_provision_idempotent():
    """Calling twice doesn't double-insert."""
    backend, rt = await _boot_runtime()
    first = await rt.lazy_provision("usr_x")
    second = await rt.lazy_provision("usr_x")
    assert first is True
    assert second is False  # cached / already provisioned

    cur = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_identities WHERE user_id = ? AND companion_id = ?",
        ("usr_x", "becca"),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == 1


@pytest.mark.asyncio
async def test_lazy_provision_writes_genesis_when_seed_has_kernel():
    """If the seed row has a kernel digest, genesis snapshot is written."""
    backend, rt = await _boot_runtime()
    # Plant a kernel digest on the seed row to simulate a real install
    await backend.conn.execute(
        "UPDATE companion_identities SET persona_kernel_digest = ? "
        "WHERE user_id = '' AND companion_id = 'becca'",
        ("§1 Name: Becca. She notices.",),
    )
    await backend.conn.commit()

    await rt.lazy_provision("usr_g")
    cur = await backend.conn.execute(
        "SELECT seed_kernel_digest FROM companion_identities_genesis "
        "WHERE user_id = ? AND companion_id = ?",
        ("usr_g", "becca"),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row is not None
    assert "Becca" in row[0]


@pytest.mark.asyncio
async def test_lazy_provision_skips_genesis_when_seed_has_no_kernel():
    """Bare install (no kernel digest yet) doesn't write a placeholder genesis."""
    backend, rt = await _boot_runtime()
    # Seed has empty kernel digest (default from migration)
    await rt.lazy_provision("usr_bare")
    cur = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_identities_genesis "
        "WHERE user_id = ? AND companion_id = ?",
        ("usr_bare", "becca"),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == 0


@pytest.mark.asyncio
async def test_lazy_provision_seeds_from_canonical():
    """Per-user identity is seeded with display_name + kernel from canonical."""
    backend, rt = await _boot_runtime()
    await backend.conn.execute(
        "UPDATE companion_identities "
        "SET persona_kernel_digest = ?, personality_doc_version = 5 "
        "WHERE user_id = '' AND companion_id = 'becca'",
        ("canonical digest",),
    )
    await backend.conn.commit()

    await rt.lazy_provision("usr_seed")
    cur = await backend.conn.execute(
        "SELECT display_name, persona_kernel_digest, personality_doc_version "
        "FROM companion_identities WHERE user_id = ? AND companion_id = ?",
        ("usr_seed", "becca"),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == "Becca"  # display_name from seed
    assert row[1] == "canonical digest"
    assert row[2] == 5


@pytest.mark.asyncio
async def test_lazy_provision_empty_user_id_noop():
    """Empty user_id never provisions (it's the legacy seed sentinel)."""
    backend, rt = await _boot_runtime()
    result = await rt.lazy_provision("")
    assert result is False


# ── get_identity / get_state ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_identity_returns_per_user_instance():
    """get_identity('usr_a') and get_identity('usr_b') return distinct instances."""
    backend, rt = await _boot_runtime()
    id_a = await rt.get_identity("usr_a")
    id_b = await rt.get_identity("usr_b")
    assert id_a is not id_b
    assert id_a.user_id == "usr_a"
    assert id_b.user_id == "usr_b"


@pytest.mark.asyncio
async def test_get_identity_caches():
    """Repeat calls return the same cached instance."""
    backend, rt = await _boot_runtime()
    id_1 = await rt.get_identity("usr_c")
    id_2 = await rt.get_identity("usr_c")
    assert id_1 is id_2


@pytest.mark.asyncio
async def test_get_identity_empty_returns_legacy_singleton():
    """get_identity('') falls back to self.identity (legacy seed)."""
    backend, rt = await _boot_runtime()
    legacy = await rt.get_identity("")
    assert legacy is rt.identity
    assert legacy.user_id == ""


@pytest.mark.asyncio
async def test_get_state_returns_per_user_instance():
    backend, rt = await _boot_runtime()
    st_a = await rt.get_state("usr_a")
    st_b = await rt.get_state("usr_b")
    assert st_a is not st_b
    assert st_a.user_id == "usr_a"
    assert st_b.user_id == "usr_b"


@pytest.mark.asyncio
async def test_get_state_empty_returns_legacy_singleton():
    backend, rt = await _boot_runtime()
    legacy = await rt.get_state("")
    assert legacy is rt.state


# ── Isolation invariant ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_two_users_dont_share_state():
    """User A transitions state; User B's state is unaffected."""
    from augmentum.companion_runtime.state import AttentionState

    backend, rt = await _boot_runtime()
    state_a = await rt.get_state("usr_iso_a")
    state_b = await rt.get_state("usr_iso_b")

    # Move A to 'present', leave B alone. force=True bypasses the
    # 2s cooldown that would otherwise reject the immediate transition.
    await state_a.transition_state(AttentionState.PRESENT, reason="test", force=True)

    # Re-read B's state from DB to confirm no cross-contamination
    cur = await backend.conn.execute(
        "SELECT state FROM companion_state WHERE user_id = 'usr_iso_b' AND companion_id = 'becca'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == "dormant"  # default; not affected by A's transition

    cur = await backend.conn.execute(
        "SELECT state FROM companion_state WHERE user_id = 'usr_iso_a' AND companion_id = 'becca'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == "present"


@pytest.mark.asyncio
async def test_two_users_dont_share_identity_kernel():
    """Setting user A's kernel doesn't change user B's."""
    backend, rt = await _boot_runtime()
    id_a = await rt.get_identity("usr_k_a")
    id_b = await rt.get_identity("usr_k_b")

    # Update A's kernel digest directly via the DB
    await backend.conn.execute(
        "UPDATE companion_identities SET persona_kernel_digest = 'A only' "
        "WHERE user_id = ? AND companion_id = ?",
        ("usr_k_a", "becca"),
    )
    await backend.conn.commit()

    # Re-fetch B's row from DB
    cur = await backend.conn.execute(
        "SELECT persona_kernel_digest FROM companion_identities "
        "WHERE user_id = ? AND companion_id = ?",
        ("usr_k_b", "becca"),
    )
    row = await cur.fetchone()
    await cur.close()
    # B's digest unchanged (whatever it was seeded with — not 'A only')
    assert row[0] != "A only"


@pytest.mark.asyncio
async def test_state_log_writes_user_id():
    """State transitions log their user_id (per migration 179)."""
    from augmentum.companion_runtime.state import AttentionState

    backend, rt = await _boot_runtime()
    st = await rt.get_state("usr_log")
    await st.transition_state(AttentionState.PRESENT, reason="test_logged", force=True)

    cur = await backend.conn.execute(
        "SELECT user_id, axis, to_value FROM companion_state_log "
        "WHERE companion_id = 'becca' AND user_id = ? ORDER BY ts DESC LIMIT 1",
        ("usr_log",),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row is not None
    assert row[0] == "usr_log"
    assert row[1] == "state"
    assert row[2] == "present"
