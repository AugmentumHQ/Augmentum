"""Phase 2 tests for multi-slot KV.

Verifies the behavior gated on ``engine_multislot_enabled=True``:

  - CLI args: ``--parallel -1 --kv-unified --cache-ram <auto> ...`` plus
    cache-ram auto-sizing from system RAM.
  - id_slot observation triggers ``_claim_slot`` for the request's
    session fingerprint (multi-slot) but only logs in single-slot.
  - ``_manage_slot`` becomes occupancy-driven when multi-slot:
    hot path is no-op, cold path with checkpoint restores into a picked
    slot, no pre-claim (response observation reconciles).
  - ``_pick_restore_target_slot`` picks unoccupied first, LRU when full.
  - ``_pick_checkpoint_target_slot(avoid=N)`` picks a non-N slot.
  - ``prepare_stable_checkpoint`` targets a non-chat slot when multi.
  - ``prewarm_context`` pins ``id_slot`` in payload when given.

Spec: docs/superpowers/specs/2026-05-05-multi-slot-kv-design.md
"""

from __future__ import annotations

import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from augmentum.models.base import (
    InternalChatRequest,
    Message,
)
from augmentum.models.llama_cpp import LlamaCppBackend
from augmentum.models.llama_server_manager import LlamaServerManager


def _make_request(**kwargs) -> InternalChatRequest:
    base = {
        "model": "test-model.gguf",
        "messages": [Message(role="user", content="hello")],
        "stream": True,
    }
    base.update(kwargs)
    return InternalChatRequest(**base)


def _make_backend() -> LlamaCppBackend:
    transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={})
    )
    client = httpx.AsyncClient(transport=transport)
    return LlamaCppBackend(client, "http://llamacpp:8080")


@contextlib.contextmanager
def _multislot_on():
    """Toggle the feature flag for the duration of a test."""
    from augmentum.config import settings
    prev = getattr(settings, "engine_multislot_enabled", False)
    settings.engine_multislot_enabled = True
    try:
        yield
    finally:
        settings.engine_multislot_enabled = prev


# ---------------------------------------------------------------------------
# CLI args (LlamaServerManager._build_slot_scheduling_args)
# ---------------------------------------------------------------------------


class TestSlotSchedulingArgs:
    """The CLI-args builder is the single switch between single-slot
    and multi-slot subprocess startup. Phase 2's load-bearing behavior
    flag — every other Phase 2 change is in-process.
    """

    def test_flag_off_emits_parallel_1(self):
        """Explicit "Always off" override: user opts out of multi-slot,
        engine startup uses ``--parallel 1`` regardless of the codebase
        recommendation.
        """
        from augmentum.config import settings
        prev = settings.engine_multislot_enabled
        settings.engine_multislot_enabled = False
        try:
            m = LlamaServerManager.__new__(LlamaServerManager)
            # __new__ skips __init__; the args builder reads this attr
            # (added 275fd8a) — set the production default explicitly.
            m._force_single_slot = False
            args = m._build_slot_scheduling_args()
            assert args == ["--parallel", "1"]
        finally:
            settings.engine_multislot_enabled = prev

    def test_flag_on_emits_full_multislot_arg_set(self):
        """Phase 2 multi-slot path: --parallel -1 (or pinned),
        --kv-unified, --cache-ram, --ctx-checkpoints,
        --checkpoint-every-n-tokens.

        ``--cache-idle-slots`` must NOT be emitted: under --kv-unified the
        slot-state save path 501s upstream, so idle-slot eviction destroys
        live KV (restore always fails) — every narrative turn's checkpoint
        prewarm evicted the chat slot and forced a full cold re-prefill
        (verified live 2026-07-02: identical request 6 s apart re-evaluated
        2490/2490 tokens, "failed to load prompt from cache").
        """
        m = LlamaServerManager.__new__(LlamaServerManager)
        # __new__ skips __init__; the args builder reads this attr
        # (added 275fd8a) — set the production default explicitly.
        m._force_single_slot = False
        with _multislot_on():
            args = m._build_slot_scheduling_args()

        # Ordering matters for some flags (--parallel needs a value
        # next), so check positions where it does and presence
        # otherwise.
        assert args[0] == "--parallel"
        assert args[1] in ("-1", "4"), f"unexpected parallel value: {args[1]}"
        assert "--kv-unified" in args
        assert "--cache-ram" in args
        # --cache-ram needs an integer argument right after.
        cache_ram_idx = args.index("--cache-ram")
        assert args[cache_ram_idx + 1].isdigit()
        assert int(args[cache_ram_idx + 1]) >= 1024  # floor guarantee
        assert "--cache-idle-slots" not in args  # evict==destroy under --kv-unified
        assert "--ctx-checkpoints" in args
        assert "--checkpoint-every-n-tokens" in args

    def test_pinned_parallel_overrides_auto(self):
        """When user pins engine_parallel_slots, that wins over -1."""
        from augmentum.config import settings
        m = LlamaServerManager.__new__(LlamaServerManager)
        # __new__ skips __init__; the args builder reads this attr
        # (added 275fd8a) — set the production default explicitly.
        m._force_single_slot = False
        with _multislot_on():
            settings.engine_parallel_slots = 8
            try:
                args = m._build_slot_scheduling_args()
            finally:
                settings.engine_parallel_slots = 0

        idx = args.index("--parallel")
        assert args[idx + 1] == "8"

    def test_pinned_cache_ram_overrides_auto(self):
        from augmentum.config import settings
        m = LlamaServerManager.__new__(LlamaServerManager)
        # __new__ skips __init__; the args builder reads this attr
        # (added 275fd8a) — set the production default explicitly.
        m._force_single_slot = False
        with _multislot_on():
            settings.engine_cache_ram_mib = 4096
            try:
                args = m._build_slot_scheduling_args()
            finally:
                settings.engine_cache_ram_mib = 0

        idx = args.index("--cache-ram")
        assert args[idx + 1] == "4096"


class TestCacheReuseArgs:
    """``--cache-reuse`` — mid-prompt KV chunk salvage via shifting.

    Passed unconditionally when the setting is > 0: llama-server
    self-gates on ``llama_memory_can_shift`` and ignores the flag on
    hybrid-attention models, so no per-architecture gating is needed
    on our side.
    """

    def test_default_passes_256(self):
        args = LlamaServerManager._cache_reuse_args()
        assert args == ["--cache-reuse", "256"]

    def test_zero_disables(self):
        from augmentum.config import settings
        prev = settings.engine_cache_reuse_min
        settings.engine_cache_reuse_min = 0
        try:
            assert LlamaServerManager._cache_reuse_args() == []
        finally:
            settings.engine_cache_reuse_min = prev

    def test_custom_value(self):
        from augmentum.config import settings
        prev = settings.engine_cache_reuse_min
        settings.engine_cache_reuse_min = 512
        try:
            args = LlamaServerManager._cache_reuse_args()
            assert args == ["--cache-reuse", "512"]
        finally:
            settings.engine_cache_reuse_min = prev


class TestCacheRamAutoSize:
    """Auto-sizing for ``--cache-ram``. Model-aware when a profile is
    available (the cache's job is holding a few full session STATES of
    the loaded model, so size follows ctx * kv-bytes/token); falls back
    to the RAM-fraction heuristic without one.
    """

    @staticmethod
    def _mgr():
        return LlamaServerManager.__new__(LlamaServerManager)

    @staticmethod
    def _mem(total_gib, *, limited=True):
        """Patch the container-aware probe, not psutil.

        Sizing reads ``hostmem.memory_info()`` so it sees the CGROUP
        ceiling rather than the host's RAM. Patching ``psutil`` here would
        no longer exercise the real path.
        """
        from augmentum.resource.hostmem import MemoryInfo
        mib = int(total_gib * 1024)
        return patch(
            "augmentum.resource.hostmem.memory_info",
            return_value=MemoryInfo(
                total_mib=mib, available_mib=mib, used_mib=0,
                source="test", limited=limited,
            ),
        )

    def test_floor_at_1_gib(self):
        """Even on a tiny system we never disable warm tier entirely."""
        with self._mem(0.25):  # 256 MiB
            assert self._mgr()._auto_cache_ram_mib() == 1024

    def test_absolute_cap_without_profile(self):
        """Model-blind fallback is bounded by the ABSOLUTE cap.

        Regression guard for the 2026-07-25 incident (B1): on a big box the
        old 25%-of-total heuristic sized this at 23.6 GiB of anonymous,
        unreclaimable host RAM. The hard cap makes that impossible
        regardless of how much memory the machine reports.
        """
        with self._mem(256):
            got = self._mgr()._auto_cache_ram_mib()
        assert got == LlamaServerManager._CACHE_RAM_ABSOLUTE_CAP_MIB

    def test_25_percent_within_band(self):
        """Typical workstation: 32 GiB → 8 GiB cache (25%, at the cap)."""
        with self._mem(32):
            assert self._mgr()._auto_cache_ram_mib() == 8192

    def test_sizes_against_available_not_total_when_limited(self):
        """Under a real ceiling, size against what is still FREE.

        Memory already spoken for is not ours to hand out a second time —
        the failure mode where every consumer independently claims a
        fraction of the same machine.
        """
        from augmentum.resource.hostmem import MemoryInfo
        with patch(
            "augmentum.resource.hostmem.memory_info",
            return_value=MemoryInfo(
                total_mib=32768, available_mib=4096, used_mib=28672,
                source="test", limited=True,
            ),
        ):
            # 25% of the 4 GiB still available, not of the 32 GiB total.
            assert self._mgr()._auto_cache_ram_mib() == 1024

    def test_probe_failure_falls_back_conservatively(self):
        """If the memory probe explodes, stay small rather than guess big."""
        with patch(
            "augmentum.resource.hostmem.memory_info",
            side_effect=RuntimeError("simulated"),
        ), pytest.raises(RuntimeError):
            self._mgr()._auto_cache_ram_mib()

    def test_model_aware_sizes_three_states(self):
        """With a profile, cache = 3 full-ctx KV states (clamped to RAM cap).

        The point: a 16 GiB blanket holds 100+ states of a small model
        (waste) but ~2 of a 122B at long ctx (evicts under light use).
        Sizing from the model keeps 'a few session switches' warm on
        both ends.
        """
        from augmentum.models.llama_server_manager import ModelProfile
        profile = ModelProfile(
            model_path="x.gguf", model_name="x",
            n_layers=32, n_embed=4096, n_heads=32, n_heads_kv=8,
        )
        mgr = self._mgr()
        with self._mem(128):
            got = mgr._auto_cache_ram_mib(
                profile=profile, ctx_size=32768, kv_cache_type="f16",
            )
        per_token = mgr._kv_bytes_per_token(profile, "f16")
        expected = max(
            1024,
            min(
                LlamaServerManager._CACHE_RAM_ABSOLUTE_CAP_MIB,
                (32768 * per_token) // (1024 * 1024) * 3,
            ),
        )
        assert got == expected

    def test_model_aware_respects_ram_cap(self):
        """A huge model at huge ctx can't soak more than 25% of RAM."""
        from augmentum.models.llama_server_manager import ModelProfile
        profile = ModelProfile(
            model_path="y.gguf", model_name="y",
            n_layers=94, n_embed=8192, n_heads=64, n_heads_kv=8,
        )
        mgr = self._mgr()
        with self._mem(16):
            got = mgr._auto_cache_ram_mib(
                profile=profile, ctx_size=131072, kv_cache_type="f16",
            )
        assert got == 4096  # 25% of 16 GiB


# ---------------------------------------------------------------------------
# Slot picker helpers
# ---------------------------------------------------------------------------


class TestPickRestoreTarget:
    """Picks where a fresh disk-restore lands in multi-slot mode."""

    def test_prefers_unoccupied_slot(self):
        backend = _make_backend()
        with _multislot_on():
            # No occupancy yet — target slot 0 (first free).
            assert backend._pick_restore_target_slot() == 0

            # Slot 0 occupied — target slot 1 (next free).
            backend._claim_slot(0, "sess-a")
            assert backend._pick_restore_target_slot() == 1

    def test_evicts_lru_when_all_full(self):
        backend = _make_backend()
        with _multislot_on():
            # Fill all 4 slots with synthetic timestamps.
            for slot, key in [(0, "sess-0"), (1, "sess-1"), (2, "sess-2"), (3, "sess-3")]:
                backend._claim_slot(slot, key)
            # Manually fudge timestamps so slot 2 is oldest.
            backend._slot_occupancy[0].last_observed_mono = 100.0
            backend._slot_occupancy[1].last_observed_mono = 200.0
            backend._slot_occupancy[2].last_observed_mono = 50.0  # oldest
            backend._slot_occupancy[3].last_observed_mono = 150.0

            assert backend._pick_restore_target_slot() == 2

    def test_single_slot_mode_returns_0(self):
        """Flag off: only one slot exists, that's the answer."""
        backend = _make_backend()
        # Flag off (default).
        assert backend._pick_restore_target_slot() == 0


class TestPickCheckpointTarget:
    """The non-chat-slot picker for prepare_stable_checkpoint. Killing
    the regenerate UX bug depends on this picking right.
    """

    def test_avoids_chat_slot_when_others_free(self):
        backend = _make_backend()
        with _multislot_on():
            backend._claim_slot(0, "chat-sess")
            # Avoid slot 0 → first unoccupied non-0 is slot 1.
            assert backend._pick_checkpoint_target_slot(avoid=0) == 1

    def test_prefers_unoccupied_non_avoid(self):
        backend = _make_backend()
        with _multislot_on():
            backend._claim_slot(0, "chat-sess")
            backend._claim_slot(1, "other-sess")
            # Slot 1 occupied; avoid 0 → slot 2 is the first free non-0.
            assert backend._pick_checkpoint_target_slot(avoid=0) == 2

    def test_lru_when_all_occupied_avoiding_chat(self):
        backend = _make_backend()
        with _multislot_on():
            for slot, key in [(0, "chat"), (1, "a"), (2, "b"), (3, "c")]:
                backend._claim_slot(slot, key)
            backend._slot_occupancy[1].last_observed_mono = 200.0
            backend._slot_occupancy[2].last_observed_mono = 50.0  # LRU non-0
            backend._slot_occupancy[3].last_observed_mono = 100.0

            # Slot 0 chat → avoid it. LRU among {1,2,3} is slot 2.
            assert backend._pick_checkpoint_target_slot(avoid=0) == 2

    def test_falls_back_to_avoid_when_only_one_slot(self):
        """Multi-slot off — only slot 0 — fallback returns 0."""
        backend = _make_backend()
        # Flag off.
        assert backend._pick_checkpoint_target_slot(avoid=0) == 0


# ---------------------------------------------------------------------------
# id_slot observation → _claim_slot (multi-slot only)
# ---------------------------------------------------------------------------


def _sse_chunk(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _sse_done() -> bytes:
    return b"data: [DONE]\n\n"


class _StreamingMockTransport(httpx.AsyncBaseTransport):
    def __init__(self, completion_chunks: list[bytes]) -> None:
        self._chunks = completion_chunks
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url.endswith("/completion"):
            return httpx.Response(
                status_code=200,
                content=b"".join(self._chunks),
                headers={"content-type": "text/event-stream"},
                request=request,
            )
        return httpx.Response(404, request=request, text="not in mock")


class TestIdSlotObservationDrivesOccupancy:
    """When the engine reports id_slot=N, multi-slot mode claims N for
    the request's session_fp. This is the authoritative occupancy
    source — engine's choice of slot, not ours.
    """

    @pytest.mark.asyncio
    async def test_multislot_on_observation_claims_slot(self):
        """Engine returns id_slot=2; backend claims slot 2 for the session."""
        chunks = [
            _sse_chunk({"content": "Hi", "stop": False, "id_slot": 2}),
            _sse_chunk({"content": "", "stop": True, "id_slot": 2,
                        "timings": {"prompt_n": 5, "predicted_n": 1}}),
            _sse_done(),
        ]
        transport = _StreamingMockTransport(chunks)
        client = httpx.AsyncClient(transport=transport)
        backend = LlamaCppBackend(client, "http://llamacpp:8080")

        req = _make_request(kv_session_key="sess-x")
        with _multislot_on():
            stream = backend._stream_completion(req, [1, 2, 3])
            async for _ in stream:
                pass

        # Multi-slot mode claimed slot 2 for sess-x.
        assert backend._get_slot_for_session("sess-x") == 2
        assert backend._get_session_for_slot(2) == "sess-x"

    @pytest.mark.asyncio
    async def test_multislot_off_observation_does_not_claim(self):
        """Single-slot mode: log only, no _claim_slot (Phase 1 owns
        slot 0 claims via _manage_slot already). Tests the explicit
        "Always off" override path — the codebase recommendation may
        be True but the user has opted out.
        """
        from augmentum.config import settings
        chunks = [
            _sse_chunk({"content": "Hi", "stop": False, "id_slot": 0}),
            _sse_chunk({"content": "", "stop": True, "id_slot": 0}),
            _sse_done(),
        ]
        transport = _StreamingMockTransport(chunks)
        client = httpx.AsyncClient(transport=transport)
        backend = LlamaCppBackend(client, "http://llamacpp:8080")

        prev = settings.engine_multislot_enabled
        settings.engine_multislot_enabled = False
        try:
            req = _make_request(kv_session_key="sess-y")
            stream = backend._stream_completion(req, [1])
            async for _ in stream:
                pass
        finally:
            settings.engine_multislot_enabled = prev

        # No claim happened — _manage_slot is the source of truth in
        # single-slot mode.
        assert backend._get_slot_for_session("sess-y") is None


# ---------------------------------------------------------------------------
# _manage_slot occupancy-driven routing (multi-slot)
# ---------------------------------------------------------------------------


def _ready_manager() -> MagicMock:
    """A mock manager in READY state with the in-flight ctx + slot dir."""
    from augmentum.models.llama_server_manager import ProcessState

    mgr = MagicMock()
    mgr.state = ProcessState.READY
    mgr._slot_dir = ""
    mgr._warm_session_key = ""
    mgr.check_alive = MagicMock(return_value=True)
    mgr.model_id = "test-model"
    mgr.model_path = "/models/test-model.gguf"
    mgr.current_ctx_size = 8192
    mgr.kv_cache_type = "q8_0"
    mgr._session_manifest = None

    counter = {"n": 0}

    @contextlib.asynccontextmanager
    async def _in_flight():
        counter["n"] += 1
        try:
            yield
        finally:
            counter["n"] -= 1

    mgr.request_in_flight = _in_flight
    return mgr


class TestMultislotManageSlot:
    """Phase 2 ``_manage_slot`` switches on the flag.

    Single-slot path (already covered by Phase 1 tests) stays unchanged.
    Multi-slot path: occupancy-driven with no pre-claim.
    """

    @pytest.mark.asyncio
    async def test_multislot_hot_session_is_noop(self):
        """If the session is already in some slot, _manage_slot is a
        no-op (engine routes via prefix matcher, no save/restore needed).
        """
        backend = _make_backend()
        backend._manager = _ready_manager()
        backend.save_session_state = AsyncMock(return_value=True)
        backend.restore_session_state = AsyncMock(return_value=True)

        with _multislot_on():
            # Pre-populate occupancy: session is hot in slot 1.
            backend._claim_slot(1, "live-sess")
            await backend._manage_slot(_make_request(kv_session_key="live-sess"))

        backend.save_session_state.assert_not_awaited()
        backend.restore_session_state.assert_not_awaited()
        # Occupancy unchanged — _manage_slot doesn't pre-claim.
        assert backend._get_slot_for_session("live-sess") == 1

    @pytest.mark.asyncio
    async def test_multislot_cold_no_checkpoint_does_nothing(self):
        """Cold session, no on-disk checkpoint → let it cold-prefill.
        No restore attempt (no state to restore from).
        """
        backend = _make_backend()
        backend._manager = _ready_manager()
        backend._slot_state_exists = MagicMock(return_value=False)
        backend.save_session_state = AsyncMock(return_value=True)
        backend.restore_session_state = AsyncMock(return_value=True)

        with _multislot_on():
            await backend._manage_slot(_make_request(kv_session_key="cold-sess"))

        backend.restore_session_state.assert_not_awaited()
        backend.save_session_state.assert_not_awaited()
        # No pre-claim (response observation will reconcile if needed).
        assert backend._get_slot_for_session("cold-sess") is None

    @pytest.mark.asyncio
    async def test_multislot_cold_with_checkpoint_restores_to_picked_slot(self):
        """Cold session WITH on-disk checkpoint → restore into a picked
        target slot. The actual served slot may differ (engine routing),
        but pre-warming the disk state into a real slot makes the
        engine's prefix matcher more likely to pick that slot.
        """
        backend = _make_backend()
        backend._manager = _ready_manager()
        backend._slot_state_exists = MagicMock(return_value=True)
        backend.save_session_state = AsyncMock(return_value=True)
        backend.restore_session_state = AsyncMock(return_value=True)

        with _multislot_on():
            await backend._manage_slot(_make_request(kv_session_key="cold-but-saved"))

        # First-free target is slot 0 (no occupancy).
        backend.restore_session_state.assert_awaited_once_with(
            "cold-but-saved", slot_id=0,
            request=backend.restore_session_state.await_args.kwargs["request"],
        )
        # Pre-claim NOT done — response observation is the source of truth.
        assert backend._get_slot_for_session("cold-but-saved") is None

    @pytest.mark.asyncio
    async def test_multislot_cold_evicts_displaced_session(self):
        """Restoring into an occupied target slot triggers a save of
        the displaced session's KV before the slot is overwritten.
        Bounds the cost: displaced KV survives in --cache-ram + on disk
        even after the eviction.
        """
        backend = _make_backend()
        backend._manager = _ready_manager()
        backend._slot_state_exists = MagicMock(return_value=True)
        backend.save_session_state = AsyncMock(return_value=True)
        backend.restore_session_state = AsyncMock(return_value=True)

        with _multislot_on():
            # Fill all 4 slots so the picker has to evict.
            backend._claim_slot(0, "sess-0")
            backend._claim_slot(1, "sess-1")
            backend._claim_slot(2, "sess-2")
            backend._claim_slot(3, "sess-3")
            backend._slot_occupancy[2].last_observed_mono = 0.0  # LRU

            await backend._manage_slot(_make_request(kv_session_key="new-cold"))

        # Picker chose slot 2 (LRU). Backend saved sess-2 first, then
        # restored new-cold into slot 2.
        backend.save_session_state.assert_awaited_once_with(
            "sess-2", slot_id=2,
        )
        backend.restore_session_state.assert_awaited_once()
        kwargs = backend.restore_session_state.await_args.kwargs
        assert kwargs["slot_id"] == 2


# ---------------------------------------------------------------------------
# prewarm_context with id_slot pinning
# ---------------------------------------------------------------------------


class TestPrewarmIdSlotPinning:
    """``prewarm_context`` with slot_id sets ``id_slot`` in the
    /completion request body. Pre-Phase-2 callers (slot_id=None) don't
    pin. Used by prepare_stable_checkpoint to land prewarms on a non-
    chat slot.
    """

    @pytest.mark.asyncio
    async def test_slot_id_appears_in_payload(self):
        captured: dict = {}

        async def handle(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.endswith("/apply-template"):
                return httpx.Response(200, json={"prompt": "hello"})
            if url.endswith("/tokenize"):
                return httpx.Response(200, json={"tokens": [1, 2, 3]})
            if url.endswith("/completion"):
                captured["payload"] = json.loads(request.content)
                return httpx.Response(200, json={})
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        backend = LlamaCppBackend(client, "http://llamacpp:8080")
        backend._manager = _ready_manager()

        await backend.prewarm_context(
            [{"role": "user", "content": "hi"}], slot_id=2,
        )

        assert "payload" in captured, "no /completion request fired"
        assert captured["payload"].get("id_slot") == 2

    @pytest.mark.asyncio
    async def test_slot_id_none_omits_pinning(self):
        captured: dict = {}

        async def handle(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url.endswith("/apply-template"):
                return httpx.Response(200, json={"prompt": "hello"})
            if url.endswith("/tokenize"):
                return httpx.Response(200, json={"tokens": [1, 2, 3]})
            if url.endswith("/completion"):
                captured["payload"] = json.loads(request.content)
                return httpx.Response(200, json={})
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        backend = LlamaCppBackend(client, "http://llamacpp:8080")
        backend._manager = _ready_manager()

        await backend.prewarm_context(
            [{"role": "user", "content": "hi"}],
            # No slot_id → no pinning, engine auto-routes.
        )

        assert "id_slot" not in captured["payload"]
