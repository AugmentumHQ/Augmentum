"""Phase 4 tests — settings UI exposure + kv_tier ContextVar correlation.

Phase 4 wires:
  - ``engine_multislot_enabled``, ``engine_parallel_slots``,
    ``engine_cache_ram_mib`` into ``_TOOL_SETTINGS`` so they're
    user-tunable via the admin Settings UI.
  - ``kv_tier_var`` ContextVar + ``bind_kv_tier`` helper, set by
    ``_manage_slot`` and read by ``_log_performance`` to stamp the
    tier on ``engine_perf`` — no log JOIN needed for distribution
    analysis.

Spec: docs/superpowers/specs/2026-05-05-multi-slot-kv-design.md
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from augmentum.models.base import (
    InternalChatRequest,
    Message,
)
from augmentum.models.llama_cpp import LlamaCppBackend
from augmentum.proxy.config_routes import _TOOL_SETTINGS


@contextlib.contextmanager
def _multislot_on():
    from augmentum.config import settings
    prev = getattr(settings, "engine_multislot_enabled", False)
    settings.engine_multislot_enabled = True
    try:
        yield
    finally:
        settings.engine_multislot_enabled = prev


def _make_request(**kwargs) -> InternalChatRequest:
    base = {
        "model": "test-model.gguf",
        "messages": [Message(role="user", content="hi")],
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


def _ready_manager() -> MagicMock:
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
    return mgr


# ---------------------------------------------------------------------------
# Settings exposure
# ---------------------------------------------------------------------------


class TestSettingsExposure:
    """The three multi-slot settings must appear in ``_TOOL_SETTINGS``
    so the admin Settings UI can show + persist them. Without this,
    the user can't tune the ramp.
    """

    def test_multislot_enabled_is_a_bool_setting(self):
        assert "engine_multislot_enabled" in _TOOL_SETTINGS
        cast_fn, lo, hi = _TOOL_SETTINGS["engine_multislot_enabled"]
        assert cast_fn is bool
        assert lo == 0 and hi == 1

    def test_parallel_slots_int_with_auto_zero(self):
        assert "engine_parallel_slots" in _TOOL_SETTINGS
        cast_fn, lo, hi = _TOOL_SETTINGS["engine_parallel_slots"]
        assert cast_fn is int
        assert lo == 0  # 0 = auto sentinel
        assert hi >= 8  # household-deployer override range

    def test_cache_ram_mib_int_with_auto_zero(self):
        assert "engine_cache_ram_mib" in _TOOL_SETTINGS
        cast_fn, lo, hi = _TOOL_SETTINGS["engine_cache_ram_mib"]
        assert cast_fn is int
        assert lo == 0  # 0 = auto sentinel
        assert hi >= 16384  # at least the soft cap

    def test_settings_have_safe_defaults(self):
        """Defaults for the multi-slot tri-state set:

        - ``engine_multislot_enabled = None`` means "auto" — follow
          the codebase's recommended default. The default cohort
          (users who have never touched the toggle) tracks the
          recommendation.
        - ``engine_parallel_slots = 0`` means "auto" (delegate to
          ``--parallel -1`` which b8935 resolves to 4).
        - ``engine_cache_ram_mib = 0`` means "auto-size from RAM".

        Phase 4 originally shipped with multi-slot OFF by default
        (``False``); 2026-05-06 flipped to tri-state ``None`` so the
        default cohort follows ``MULTISLOT_DEFAULT_ENABLED`` without
        clobbering explicit user choices.
        """
        from augmentum.config import settings
        assert settings.engine_multislot_enabled is None
        assert settings.engine_parallel_slots == 0
        assert settings.engine_cache_ram_mib == 0


# ---------------------------------------------------------------------------
# kv_tier ContextVar wiring
# ---------------------------------------------------------------------------


class TestKvTierContextVar:
    """``kv_tier_var`` is set by ``_manage_slot`` at decision time and
    read by ``_log_performance`` so ``engine_perf`` self-contains the
    tier without needing a request_id JOIN. The single-source-of-truth
    is the ContextVar — any consumer reads it directly.
    """

    def test_default_is_empty_string(self):
        from augmentum.proxy.status_bus import kv_tier_var
        # Default is "" (no tier decided yet / opaque request).
        assert kv_tier_var.get() == ""

    def test_bind_kv_tier_sets_value(self):
        from augmentum.proxy.status_bus import bind_kv_tier, kv_tier_var
        token = bind_kv_tier("hot")
        try:
            assert kv_tier_var.get() == "hot"
        finally:
            kv_tier_var.reset(token)
        assert kv_tier_var.get() == ""

    @pytest.mark.asyncio
    async def test_manage_slot_binds_hot_tier(self):
        from augmentum.proxy.status_bus import kv_tier_var

        backend = _make_backend()
        backend._manager = _ready_manager()
        backend.save_session_state = AsyncMock(return_value=True)
        backend.restore_session_state = AsyncMock(return_value=True)

        with _multislot_on():
            backend._claim_slot(2, "live-sess")
            await backend._manage_slot(_make_request(kv_session_key="live-sess"))

        # ContextVar has the tier the decision landed on.
        assert kv_tier_var.get() == "hot"

    @pytest.mark.asyncio
    async def test_manage_slot_binds_cold_with_checkpoint_tier(self):
        from augmentum.proxy.status_bus import kv_tier_var

        backend = _make_backend()
        backend._manager = _ready_manager()
        backend._slot_state_exists = MagicMock(return_value=True)
        backend.save_session_state = AsyncMock(return_value=True)
        backend.restore_session_state = AsyncMock(return_value=True)

        with _multislot_on():
            await backend._manage_slot(_make_request(kv_session_key="cold-saved"))

        assert kv_tier_var.get() == "cold_with_checkpoint"

    @pytest.mark.asyncio
    async def test_manage_slot_binds_cold_no_checkpoint_tier(self):
        from augmentum.proxy.status_bus import kv_tier_var

        backend = _make_backend()
        backend._manager = _ready_manager()
        backend._slot_state_exists = MagicMock(return_value=False)
        backend.save_session_state = AsyncMock(return_value=True)
        backend.restore_session_state = AsyncMock(return_value=True)

        with _multislot_on():
            await backend._manage_slot(_make_request(kv_session_key="brand-new"))

        assert kv_tier_var.get() == "cold_no_checkpoint"

    @pytest.mark.asyncio
    async def test_engine_perf_log_includes_kv_tier(self, capfd):
        """Integration: tier set by _manage_slot reaches the engine_perf
        log line via ContextVar. This is the load-bearing correlation
        — distribution analysis reads this field directly.
        """
        from augmentum.proxy.status_bus import bind_kv_tier
        backend = _make_backend()
        backend._manager = _ready_manager()

        token = bind_kv_tier("hot")
        try:
            backend._log_performance(
                {
                    "prompt_n": 100, "predicted_n": 50,
                    "prompt_ms": 200.0, "predicted_ms": 1000.0,
                },
                t_start=0.0,
                t_first_token=0.5,
            )
        finally:
            from augmentum.proxy.status_bus import kv_tier_var
            kv_tier_var.reset(token)

        out, err = capfd.readouterr()
        combined = out + err
        engine_perf_lines = [
            line for line in combined.splitlines()
            if "engine_perf" in line
        ]
        assert len(engine_perf_lines) == 1
        assert "kv_tier=hot" in engine_perf_lines[0]

    @pytest.mark.asyncio
    async def test_kv_tier_isolated_across_concurrent_tasks(self):
        """ContextVar isolation: two concurrent tasks each set their
        own tier; neither sees the other's value. Required for multi-
        slot mode where multiple sessions process simultaneously.
        """
        import asyncio
        from augmentum.proxy.status_bus import bind_kv_tier, kv_tier_var

        observed: dict[str, str] = {}

        async def stamp(name: str, tier: str) -> None:
            tok = bind_kv_tier(tier)
            try:
                await asyncio.sleep(0)  # yield so tasks interleave
                observed[name] = kv_tier_var.get()
            finally:
                kv_tier_var.reset(tok)

        await asyncio.gather(
            stamp("a", "hot"),
            stamp("b", "cold_with_checkpoint"),
        )

        assert observed == {
            "a": "hot",
            "b": "cold_with_checkpoint",
        }
