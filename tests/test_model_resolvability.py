"""`ProviderRegistry.model_is_resolvable` — the guard that lets callers
detect a STALE user-configured model reference and surface the choice to the
user instead of silently riding the default-backend fallback (never
auto-select). Used by the narrative memory-model alert path.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from augmentum.models.provider_registry import ProviderRegistry


def _registry(*, model_map=None, backends=None, pins=None) -> ProviderRegistry:
    reg = ProviderRegistry.__new__(ProviderRegistry)
    reg._backends = backends or {"engine": SimpleNamespace(name="engine")}
    reg._model_map = model_map or {}
    reg._model_pins = pins or {}
    reg._default = "engine"

    async def _noop_refresh():
        return None

    reg.refresh_model_map = _noop_refresh  # type: ignore[method-assign]
    return reg


def _run(coro):
    return asyncio.run(coro)


def test_empty_is_not_resolvable():
    reg = _registry()
    assert _run(reg.model_is_resolvable("")) is False
    assert _run(reg.model_is_resolvable("   ")) is False


def test_model_in_map_resolves():
    reg = _registry(model_map={"qwen3.6-27b": "engine"})
    assert _run(reg.model_is_resolvable("qwen3.6-27b")) is True


def test_missing_model_is_not_resolvable():
    # The whole point: a removed/renamed model must report False so the caller
    # raises the alert instead of falling through to the default backend.
    reg = _registry(model_map={"qwen3.6-27b": "engine"})
    assert _run(reg.model_is_resolvable("deepseek-v4-pro")) is False


def test_explicit_backend_suffix_resolves_when_backend_exists():
    reg = _registry(backends={"engine": SimpleNamespace(), "vim": SimpleNamespace()})
    assert _run(reg.model_is_resolvable("some-model@vim")) is True
    assert _run(reg.model_is_resolvable("some-model@ghost")) is False


def test_routing_pin_resolves():
    reg = _registry(pins={"pinned-model": "engine"})
    assert _run(reg.model_is_resolvable("pinned-model")) is True


def test_mode_prefix_is_stripped_before_check():
    # Direct-passthrough ids ("d/<model>") and other mode prefixes must be
    # stripped so the same reference the chat path uses is checked here.
    reg = _registry(model_map={"qwen3.6-27b": "engine"})
    assert _run(reg.model_is_resolvable("d/qwen3.6-27b")) is True
