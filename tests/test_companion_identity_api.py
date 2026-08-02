"""Sprint α tests — CompanionIdentity API surface (Piece 2).

Verifies the new API surface added to CompanionIdentity in Sprint α:
- read_overlay / apply_overlay
- read_trait_deltas / nudge_trait (with both caps)
- read_traits_derived / write_traits_derived
- read_relationship_state / update_relationship_state

Plus the cross-user invariant — User A's nudges/overlay never affect
User B's.
"""

from __future__ import annotations

import pytest


async def _boot_runtime_with_two_users():
    """Provision identities for two users so isolation tests have
    real per-user rows to compare."""
    from augmentum.companion_runtime.runtime import CompanionRuntime
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    rt = CompanionRuntime(backend, companion_id="becca")
    # Seed users so lazy_provision has FK targets (though identity
    # doesn't FK to users today, future migrations might).
    for uid in ("usr_alpha", "usr_beta"):
        await backend.conn.execute(
            "INSERT INTO users (id, username, password_hash, created_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (uid, uid, "x"),
        )
    await backend.conn.commit()
    return backend, rt


# ── Overlay round-trips ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_overlay_empty_on_fresh_identity():
    """A freshly-provisioned identity has no overlay text."""
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    text = await identity.read_overlay()
    assert text == ""


@pytest.mark.asyncio
async def test_apply_overlay_round_trip():
    """apply_overlay writes, read_overlay returns the same text."""
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    ok = await identity.apply_overlay(text="became more comfortable with silence")
    assert ok is True
    text = await identity.read_overlay()
    assert text == "became more comfortable with silence"


@pytest.mark.asyncio
async def test_apply_overlay_preserves_trait_deltas():
    """Updating the freeform notes shouldn't wipe accumulated trait nudges."""
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    await identity.nudge_trait(name="playfulness", delta=0.01)
    await identity.apply_overlay(text="diary line")
    deltas = await identity.read_trait_deltas()
    assert deltas.get("playfulness") == pytest.approx(0.01)


# ── Trait nudging ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_trait_deltas_empty_on_fresh_identity():
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    deltas = await identity.read_trait_deltas()
    assert deltas == {}


@pytest.mark.asyncio
async def test_nudge_trait_applies_within_per_call_cap():
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    ok = await identity.nudge_trait(name="curiosity", delta=0.01)
    assert ok is True
    deltas = await identity.read_trait_deltas()
    assert deltas["curiosity"] == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_nudge_trait_rejects_above_per_call_cap():
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    # 0.02 > PER_CALL_TRAIT_CAP (0.01) → rejected
    ok = await identity.nudge_trait(name="playfulness", delta=0.02)
    assert ok is False
    deltas = await identity.read_trait_deltas()
    assert "playfulness" not in deltas


@pytest.mark.asyncio
async def test_nudge_trait_respects_cumulative_cap():
    """Five +0.01 nudges = +0.05 (right at cap). Sixth should reject."""
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    for _ in range(5):
        ok = await identity.nudge_trait(name="candor", delta=0.01)
        assert ok is True
    # Now at +0.05, exactly the cumulative cap. Next +0.01 would
    # take it to 0.06, over the cap.
    deltas = await identity.read_trait_deltas()
    assert deltas["candor"] == pytest.approx(0.05)

    rejected = await identity.nudge_trait(name="candor", delta=0.01)
    assert rejected is False
    deltas = await identity.read_trait_deltas()
    assert deltas["candor"] == pytest.approx(0.05)  # unchanged


@pytest.mark.asyncio
async def test_nudge_trait_can_go_negative():
    """Negative deltas accumulate symmetrically; cumulative cap also
    bounds negative magnitudes."""
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    for _ in range(5):
        ok = await identity.nudge_trait(name="caution", delta=-0.01)
        assert ok is True
    deltas = await identity.read_trait_deltas()
    assert deltas["caution"] == pytest.approx(-0.05)

    rejected = await identity.nudge_trait(name="caution", delta=-0.01)
    assert rejected is False


@pytest.mark.asyncio
async def test_nudge_trait_cap_is_per_trait():
    """Cumulative cap applies independently to each trait — capping
    one shouldn't prevent another from accumulating."""
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    # Max out playfulness
    for _ in range(5):
        await identity.nudge_trait(name="playfulness", delta=0.01)
    # Curiosity should still accept nudges
    ok = await identity.nudge_trait(name="curiosity", delta=0.01)
    assert ok is True


# ── traits_derived ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_traits_derived_empty_on_fresh():
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    traits = await identity.read_traits_derived()
    assert traits == {}


@pytest.mark.asyncio
async def test_write_traits_derived_round_trip():
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    ok = await identity.write_traits_derived(
        {"curiosity": 0.7, "playfulness": 0.62, "caution": 0.55}
    )
    assert ok is True
    traits = await identity.read_traits_derived()
    assert traits["curiosity"] == pytest.approx(0.7)
    assert traits["playfulness"] == pytest.approx(0.62)
    assert traits["caution"] == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_write_traits_derived_coerces_to_float():
    """Non-numeric values silently drop (avoid persisting junk)."""
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    await identity.write_traits_derived(
        {"curiosity": "0.8", "playfulness": "not a number", "caution": 0.55}
    )
    traits = await identity.read_traits_derived()
    assert traits["curiosity"] == pytest.approx(0.8)
    assert "playfulness" not in traits
    assert traits["caution"] == pytest.approx(0.55)


# ── relationship_state ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_relationship_state_empty_on_fresh():
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    rs = await identity.read_relationship_state()
    assert rs == {}


@pytest.mark.asyncio
async def test_update_relationship_state_merges_keys():
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    await identity.update_relationship_state("trust_level", 0.78)
    await identity.update_relationship_state("nicknames_earned", ["partner"])
    rs = await identity.read_relationship_state()
    assert rs["trust_level"] == pytest.approx(0.78)
    assert rs["nicknames_earned"] == ["partner"]


@pytest.mark.asyncio
async def test_update_relationship_state_overwrites_same_key():
    """Re-updating a key replaces; doesn't append/merge sub-structures."""
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    await identity.update_relationship_state("trust_level", 0.5)
    await identity.update_relationship_state("trust_level", 0.8)
    rs = await identity.read_relationship_state()
    assert rs["trust_level"] == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_update_relationship_state_rejects_non_serializable():
    """Caller passing a non-JSON-serializable value gets False, no write."""
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")

    class _NotSerializable:
        pass

    ok = await identity.update_relationship_state("weird", _NotSerializable())
    assert ok is False
    rs = await identity.read_relationship_state()
    assert "weird" not in rs


# ── Cross-user isolation ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nudge_trait_isolated_between_users():
    """User A's nudges don't bleed into User B's identity row."""
    backend, rt = await _boot_runtime_with_two_users()
    id_a = await rt.get_identity("usr_alpha")
    id_b = await rt.get_identity("usr_beta")

    await id_a.nudge_trait(name="playfulness", delta=0.01)

    deltas_a = await id_a.read_trait_deltas()
    deltas_b = await id_b.read_trait_deltas()
    assert deltas_a.get("playfulness") == pytest.approx(0.01)
    assert "playfulness" not in deltas_b


@pytest.mark.asyncio
async def test_overlay_isolated_between_users():
    backend, rt = await _boot_runtime_with_two_users()
    id_a = await rt.get_identity("usr_alpha")
    id_b = await rt.get_identity("usr_beta")

    await id_a.apply_overlay(text="A's overlay")

    assert (await id_a.read_overlay()) == "A's overlay"
    assert (await id_b.read_overlay()) == ""


@pytest.mark.asyncio
async def test_relationship_state_isolated_between_users():
    backend, rt = await _boot_runtime_with_two_users()
    id_a = await rt.get_identity("usr_alpha")
    id_b = await rt.get_identity("usr_beta")

    await id_a.update_relationship_state("trust_level", 0.9)
    await id_b.update_relationship_state("trust_level", 0.2)

    rs_a = await id_a.read_relationship_state()
    rs_b = await id_b.read_relationship_state()
    assert rs_a["trust_level"] == pytest.approx(0.9)
    assert rs_b["trust_level"] == pytest.approx(0.2)


# ── Parse defensive ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_overlay_handles_garbage_json():
    """A row with malformed kernel_overlay JSON parses to defaults."""
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    # Write malformed JSON directly
    await backend.conn.execute(
        "UPDATE companion_identities SET kernel_overlay = ? "
        "WHERE user_id = ? AND companion_id = ?",
        ("{not valid json", "usr_alpha", "becca"),
    )
    await backend.conn.commit()
    # Re-load to pick up the garbage
    await identity.load()
    assert (await identity.read_overlay()) == ""
    assert (await identity.read_trait_deltas()) == {}


@pytest.mark.asyncio
async def test_traits_derived_handles_garbage_json():
    backend, rt = await _boot_runtime_with_two_users()
    identity = await rt.get_identity("usr_alpha")
    await backend.conn.execute(
        "UPDATE companion_identities SET traits_derived_json = ? "
        "WHERE user_id = ? AND companion_id = ?",
        ("not json", "usr_alpha", "becca"),
    )
    await backend.conn.commit()
    await identity.load()
    assert (await identity.read_traits_derived()) == {}
