"""Phase 3 tests — tier-hit telemetry.

Phase 3's revised scope: emit ``kv_tier_decided`` from ``_manage_slot``
at the moment we decide which tier to use for a request, and stamp
``request_id`` on ``engine_perf`` so post-hoc correlation across the
full request timeline (status_bus stage_start/complete, slot_observation,
kv_tier_decided, engine_perf) is greppable from one id.

Originally Phase 3 was "smart prefetch — query /slots, compute LCP,
skip disk restore when warm tier is hot." That heuristic optimization
is deferred until Phase 4 telemetry shows it's actually needed for
real workloads — premature otherwise.

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


def _grep(out_err: str, key: str) -> list[str]:
    return [line for line in out_err.splitlines() if key in line]


class TestTierTelemetryMultislot:
    """The tier classification under multi-slot mode."""

    @pytest.mark.asyncio
    async def test_hot_tier_logged_when_session_in_occupancy(self, capfd):
        backend = _make_backend()
        backend._manager = _ready_manager()
        backend.save_session_state = AsyncMock(return_value=True)
        backend.restore_session_state = AsyncMock(return_value=True)

        with _multislot_on():
            backend._claim_slot(2, "live-sess")
            await backend._manage_slot(_make_request(kv_session_key="live-sess"))

        out, err = capfd.readouterr()
        events = _grep(out + err, "kv_tier_decided")
        assert len(events) == 1
        assert "tier=hot" in events[0]
        assert "slot=2" in events[0]
        assert "multislot=True" in events[0]
        assert "session=live-sess" in events[0]

    @pytest.mark.asyncio
    async def test_cold_with_checkpoint_logged(self, capfd):
        backend = _make_backend()
        backend._manager = _ready_manager()
        backend._slot_state_exists = MagicMock(return_value=True)
        backend.save_session_state = AsyncMock(return_value=True)
        backend.restore_session_state = AsyncMock(return_value=True)

        with _multislot_on():
            await backend._manage_slot(_make_request(kv_session_key="cold-saved"))

        out, err = capfd.readouterr()
        events = _grep(out + err, "kv_tier_decided")
        assert len(events) == 1
        assert "tier=cold_with_checkpoint" in events[0]
        assert "slot=0" in events[0]  # picker picks slot 0 first
        assert "restored=True" in events[0]

    @pytest.mark.asyncio
    async def test_cold_no_checkpoint_logged(self, capfd):
        backend = _make_backend()
        backend._manager = _ready_manager()
        backend._slot_state_exists = MagicMock(return_value=False)
        backend.save_session_state = AsyncMock(return_value=True)
        backend.restore_session_state = AsyncMock(return_value=True)

        with _multislot_on():
            await backend._manage_slot(_make_request(kv_session_key="brand-new"))

        out, err = capfd.readouterr()
        events = _grep(out + err, "kv_tier_decided")
        assert len(events) == 1
        assert "tier=cold_no_checkpoint" in events[0]


class TestTierTelemetrySingleSlot:
    """Single-slot mode also emits kv_tier_decided so we can compare
    distributions across the flag-on/off ramp.
    """

    @pytest.mark.asyncio
    async def test_hot_tier_in_single_slot_mode(self, capfd):
        from augmentum.config import settings
        backend = _make_backend()
        backend._manager = _ready_manager()

        prev = settings.engine_multislot_enabled
        settings.engine_multislot_enabled = False  # explicit "Always off"
        try:
            backend._claim_slot(0, "same-sess")
            await backend._manage_slot(_make_request(kv_session_key="same-sess"))
        finally:
            settings.engine_multislot_enabled = prev

        out, err = capfd.readouterr()
        events = _grep(out + err, "kv_tier_decided")
        assert len(events) == 1
        assert "tier=hot" in events[0]
        assert "multislot=False" in events[0]

    @pytest.mark.asyncio
    async def test_cold_with_checkpoint_single_slot(self, capfd):
        from augmentum.config import settings
        backend = _make_backend()
        backend._manager = _ready_manager()
        backend._slot_state_exists = MagicMock(return_value=True)
        backend.save_session_state = AsyncMock(return_value=True)
        backend.restore_session_state = AsyncMock(return_value=True)

        prev = settings.engine_multislot_enabled
        settings.engine_multislot_enabled = False  # explicit "Always off"
        try:
            await backend._manage_slot(_make_request(kv_session_key="single-cold"))
        finally:
            settings.engine_multislot_enabled = prev

        out, err = capfd.readouterr()
        events = _grep(out + err, "kv_tier_decided")
        assert len(events) == 1
        assert "tier=cold_with_checkpoint" in events[0]
        assert "slot=0" in events[0]
        assert "multislot=False" in events[0]


class TestRequestIdCorrelation:
    """request_id propagates into kv_tier_decided so a full request's
    timeline is greppable from the same id (status_bus stage events,
    slot_observation, kv_tier_decided, engine_perf all stamp it).
    """

    @pytest.mark.asyncio
    async def test_request_id_appears_in_tier_event(self, capfd):
        from augmentum.proxy.status_bus import bind_request_id, reset_request_id

        backend = _make_backend()
        backend._manager = _ready_manager()
        backend._slot_state_exists = MagicMock(return_value=False)
        backend.save_session_state = AsyncMock(return_value=True)
        backend.restore_session_state = AsyncMock(return_value=True)

        token = bind_request_id("trace-abc")
        try:
            with _multislot_on():
                await backend._manage_slot(_make_request(kv_session_key="some-sess"))
        finally:
            reset_request_id(token)

        out, err = capfd.readouterr()
        events = _grep(out + err, "kv_tier_decided")
        assert len(events) == 1
        assert "request_id=trace-abc" in events[0]

    @pytest.mark.asyncio
    async def test_no_request_id_safe_default(self, capfd):
        """When no request_id is bound (e.g. tests, direct API calls),
        the field is empty string. Doesn't crash, just empty.
        """
        backend = _make_backend()
        backend._manager = _ready_manager()
        backend._slot_state_exists = MagicMock(return_value=False)

        with _multislot_on():
            await backend._manage_slot(_make_request(kv_session_key="some-sess"))

        out, err = capfd.readouterr()
        events = _grep(out + err, "kv_tier_decided")
        assert len(events) == 1
        # Empty request_id is rendered as request_id= with nothing after.
        assert "request_id=" in events[0]
