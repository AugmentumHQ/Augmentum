"""Tests for the background resource sampler (spec §4.5/§4.6, Phase 0)."""

from __future__ import annotations

import asyncio
import time
import types
from unittest.mock import AsyncMock

import pytest

from augmentum.resource import sampler


def _state(**kw):
    s = types.SimpleNamespace()
    for k, v in kw.items():
        setattr(s, k, v)
    return s


class TestInterval:
    def test_active_when_recently_polled(self):
        st = _state(resource_panel_last_access=time.monotonic())
        assert sampler._interval(st) == sampler._ACTIVE_INTERVAL_S

    def test_idle_when_never_polled(self):
        assert sampler._interval(_state()) == sampler._IDLE_INTERVAL_S

    def test_idle_after_window(self):
        st = _state(resource_panel_last_access=time.monotonic() - 999)
        assert sampler._interval(st) == sampler._IDLE_INTERVAL_S


class TestSampleOnce:
    def test_warms_all_caches(self, monkeypatch):
        ledger = types.SimpleNamespace(collect=AsyncMock())
        called = {"container": 0, "host": 0}

        async def _fake_container(app_state):
            called["container"] += 1
            return []

        async def _fake_host(http):
            called["host"] += 1
            return None

        monkeypatch.setattr(sampler, "probe_sidecar_containers", _fake_container)
        monkeypatch.setattr(sampler, "probe_host_stats", _fake_host)
        st = _state(resource_ledger=ledger, http_client=object())

        spent = asyncio.run(sampler._sample_once(st))
        ledger.collect.assert_awaited_once()
        assert called["container"] == 1
        assert called["host"] == 1
        assert spent >= 0

    def test_never_raises_when_a_probe_is_wedged(self, monkeypatch):
        # A wedged probe must degrade to a stale cache, not crash the loop.
        ledger = types.SimpleNamespace(collect=AsyncMock(side_effect=RuntimeError("nvidia-smi hung")))

        async def _boom_container(app_state):
            raise RuntimeError("docker daemon wedged")

        monkeypatch.setattr(sampler, "probe_sidecar_containers", _boom_container)
        monkeypatch.setattr(sampler, "probe_host_stats", AsyncMock(side_effect=RuntimeError("agent down")))
        st = _state(resource_ledger=ledger, http_client=object())

        # Must complete without raising.
        spent = asyncio.run(sampler._sample_once(st))
        assert spent >= 0

    def test_no_ledger_no_http_is_safe(self):
        spent = asyncio.run(sampler._sample_once(_state()))
        assert spent >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
