"""Phase 1 tests for the slot-occupancy refactor.

Phase 1 replaces ``_active_session: str`` with the dual-index
``_slot_occupancy`` (slot_id → SlotOccupancy) plus ``_session_to_slot``
(session_key → slot_id). All current call sites still operate on slot 0;
the structural change unblocks Phase 2 multi-slot routing.

These tests cover the helper invariants:
  - Both indexes stay in sync under all mutations.
  - Stale entries get pruned eagerly when a slot or session is reassigned.
  - Per-slot locks are independent — different slot IDs return different
    lock instances (so concurrent operations on different slots don't
    serialize).

Spec: docs/superpowers/specs/2026-05-05-multi-slot-kv-design.md
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from augmentum.models.llama_cpp import LlamaCppBackend, SlotOccupancy


def _make_backend() -> LlamaCppBackend:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={})
    )
    client = httpx.AsyncClient(transport=transport)
    return LlamaCppBackend(client, "http://llamacpp:8080")


class TestSlotOccupancy:
    """The SlotOccupancy dataclass is the unit of slot tracking."""

    def test_dataclass_carries_required_fields(self):
        """Tests rely on these fields existing — locks the schema."""
        occ = SlotOccupancy(slot_id=0, session_key="sess", last_observed_mono=1.0)
        assert occ.slot_id == 0
        assert occ.session_key == "sess"
        assert occ.last_observed_mono == 1.0


class TestClaimSlot:
    """``_claim_slot`` is the single mutation point for occupancy.
    Every other helper assumes it keeps both indexes in sync.
    """

    def test_new_claim_creates_both_index_entries(self):
        backend = _make_backend()
        backend._claim_slot(0, "sess-a")

        assert 0 in backend._slot_occupancy
        assert backend._slot_occupancy[0].session_key == "sess-a"
        assert backend._slot_occupancy[0].slot_id == 0
        assert backend._session_to_slot["sess-a"] == 0

    def test_reclaim_same_slot_same_session_updates_timestamp(self):
        """Reclaiming the same (slot, session) pair updates
        last_observed_mono so LRU eviction sees the freshness.
        """
        backend = _make_backend()
        backend._claim_slot(0, "sess-a")
        first_ts = backend._slot_occupancy[0].last_observed_mono

        # Re-claim same slot/session.
        backend._claim_slot(0, "sess-a")
        second_ts = backend._slot_occupancy[0].last_observed_mono

        # Timestamp may be equal if the clock didn't tick (monotonic
        # clock resolution on Windows can be ~16ms), but it MUST never
        # decrease.
        assert second_ts >= first_ts
        # Both indexes still consistent.
        assert backend._session_to_slot == {"sess-a": 0}
        assert set(backend._slot_occupancy.keys()) == {0}

    def test_displacement_at_same_slot_prunes_old_session_inverse(self):
        """Slot 0 had session A; now session B claims it. The old
        session_to_slot["A"] must be removed — A is no longer in any
        slot from our POV.
        """
        backend = _make_backend()
        backend._claim_slot(0, "sess-a")
        backend._claim_slot(0, "sess-b")

        assert backend._slot_occupancy[0].session_key == "sess-b"
        assert "sess-a" not in backend._session_to_slot, (
            "displaced session must be pruned from inverse index"
        )
        assert backend._session_to_slot["sess-b"] == 0

    def test_session_moves_between_slots_prunes_old_slot_forward(self):
        """Session A was on slot 0; now it's claimed on slot 1. The old
        slot_occupancy[0] must be removed — slot 0 is now unoccupied
        from our POV (engine may still have the KV in --cache-ram).
        """
        backend = _make_backend()
        backend._claim_slot(0, "sess-a")
        backend._claim_slot(1, "sess-a")

        assert 0 not in backend._slot_occupancy, (
            "old slot must be pruned when session migrates"
        )
        assert backend._slot_occupancy[1].session_key == "sess-a"
        assert backend._session_to_slot == {"sess-a": 1}

    def test_concurrent_claims_different_slots_coexist(self):
        """Two sessions on two slots — both indexes hold both entries."""
        backend = _make_backend()
        backend._claim_slot(0, "sess-a")
        backend._claim_slot(1, "sess-b")

        assert set(backend._slot_occupancy.keys()) == {0, 1}
        assert backend._slot_occupancy[0].session_key == "sess-a"
        assert backend._slot_occupancy[1].session_key == "sess-b"
        assert backend._session_to_slot == {"sess-a": 0, "sess-b": 1}

    def test_empty_session_key_is_noop(self):
        """Opaque external requests (no session_key) shouldn't pollute
        the tracker. Empty string returns immediately.
        """
        backend = _make_backend()
        backend._claim_slot(0, "")

        assert backend._slot_occupancy == {}
        assert backend._session_to_slot == {}

    def test_displacement_chain_preserves_invariants(self):
        """Stress: A on slot 0, B on slot 1, then A claims slot 1
        (displacing B and freeing slot 0). Both indexes must reflect
        only A on slot 1.
        """
        backend = _make_backend()
        backend._claim_slot(0, "sess-a")
        backend._claim_slot(1, "sess-b")
        # A migrates to slot 1, displacing B.
        backend._claim_slot(1, "sess-a")

        # Slot 0 freed (A moved out), slot 1 has A (B was displaced).
        assert 0 not in backend._slot_occupancy
        assert backend._slot_occupancy[1].session_key == "sess-a"
        # B is gone from inverse (displaced from its only slot).
        assert "sess-b" not in backend._session_to_slot
        assert backend._session_to_slot == {"sess-a": 1}


class TestReleaseSlot:
    """``_release_slot`` evicts a slot's occupancy. Used on engine
    crash, model swap, or pre-prewarm clearing.
    """

    def test_release_drops_both_indexes(self):
        backend = _make_backend()
        backend._claim_slot(0, "sess-a")

        backend._release_slot(0)

        assert backend._slot_occupancy == {}
        assert backend._session_to_slot == {}

    def test_release_unoccupied_slot_is_noop(self):
        backend = _make_backend()
        # No prior claim. Release should not raise or mutate.
        backend._release_slot(0)
        backend._release_slot(99)

        assert backend._slot_occupancy == {}
        assert backend._session_to_slot == {}

    def test_release_preserves_other_slots(self):
        backend = _make_backend()
        backend._claim_slot(0, "sess-a")
        backend._claim_slot(1, "sess-b")

        backend._release_slot(0)

        assert 0 not in backend._slot_occupancy
        assert "sess-a" not in backend._session_to_slot
        # Slot 1 untouched.
        assert backend._slot_occupancy[1].session_key == "sess-b"
        assert backend._session_to_slot["sess-b"] == 1


class TestQueries:
    """Read-only helpers on the occupancy state."""

    def test_get_session_for_slot_unoccupied(self):
        backend = _make_backend()
        # Empty string for unoccupied slot — never None, so callers
        # don't have to defend against truthiness traps.
        assert backend._get_session_for_slot(0) == ""

    def test_get_session_for_slot_after_claim(self):
        backend = _make_backend()
        backend._claim_slot(2, "sess-x")
        assert backend._get_session_for_slot(2) == "sess-x"
        # Other slots still unoccupied.
        assert backend._get_session_for_slot(0) == ""

    def test_get_slot_for_session_returns_none_when_unknown(self):
        backend = _make_backend()
        assert backend._get_slot_for_session("unknown") is None
        assert backend._get_slot_for_session("") is None

    def test_get_slot_for_session_after_claim(self):
        backend = _make_backend()
        backend._claim_slot(3, "sess-y")
        assert backend._get_slot_for_session("sess-y") == 3


class TestPerSlotLocks:
    """Each slot_id gets its own lock instance. Phase 2 multi-slot
    relies on this so concurrent ops on different slots run in
    parallel.
    """

    @pytest.mark.asyncio
    async def test_different_slots_get_different_locks(self):
        backend = _make_backend()

        lock_0 = backend._get_slot_lock(0)
        lock_1 = backend._get_slot_lock(1)

        assert lock_0 is not lock_1, (
            "different slot IDs must get different lock instances; "
            "Phase 2 parallelism breaks if they share."
        )

    @pytest.mark.asyncio
    async def test_same_slot_returns_same_lock_instance(self):
        backend = _make_backend()

        first = backend._get_slot_lock(0)
        second = backend._get_slot_lock(0)

        assert first is second, "repeated calls must return the same lock"

    @pytest.mark.asyncio
    async def test_default_slot_is_zero(self):
        """Phase 1 callers that don't pass slot_id should get slot 0's
        lock — that's the backwards-compat behavior.
        """
        backend = _make_backend()

        explicit = backend._get_slot_lock(0)
        default = backend._get_slot_lock()

        assert explicit is default

    @pytest.mark.asyncio
    async def test_locks_actually_serialize_within_a_slot(self):
        """Smoke: two coroutines holding the SAME slot lock are
        serialized. Critical sanity check that we didn't accidentally
        return non-functional placeholders.
        """
        backend = _make_backend()
        lock = backend._get_slot_lock(0)
        order: list[str] = []

        async def hold(label: str, hold_for: float) -> None:
            async with lock:
                order.append(f"{label}-enter")
                await asyncio.sleep(hold_for)
                order.append(f"{label}-exit")

        await asyncio.gather(
            hold("a", 0.02),
            hold("b", 0.01),
        )

        # Whichever ran first must enter+exit before the other enters.
        assert order in (
            ["a-enter", "a-exit", "b-enter", "b-exit"],
            ["b-enter", "b-exit", "a-enter", "a-exit"],
        ), f"locks did not serialize: {order}"

    @pytest.mark.asyncio
    async def test_different_slot_locks_run_in_parallel(self):
        """Inverse: holding slot-0 lock must NOT block slot-1 lock.
        This is the whole point of per-slot locking.
        """
        backend = _make_backend()
        lock_0 = backend._get_slot_lock(0)
        lock_1 = backend._get_slot_lock(1)
        events: list[str] = []

        async def hold_0() -> None:
            async with lock_0:
                events.append("0-enter")
                await asyncio.sleep(0.05)
                events.append("0-exit")

        async def hold_1() -> None:
            # Tiny delay so 0 enters first deterministically.
            await asyncio.sleep(0.005)
            async with lock_1:
                events.append("1-enter")
                events.append("1-exit")

        await asyncio.gather(hold_0(), hold_1())

        # Slot-1 should enter and exit while slot-0 is still holding
        # its lock — i.e. before "0-exit" appears.
        zero_exit_idx = events.index("0-exit")
        one_enter_idx = events.index("1-enter")
        assert one_enter_idx < zero_exit_idx, (
            f"slot 1 was blocked by slot 0's lock — per-slot isolation broken. "
            f"Order: {events}"
        )
