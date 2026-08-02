"""Routing tests for the secondary local engine slot ("Slot B").

The secondary slot is a second resident llama-server process. It shares
its GGUF model_dirs with the primary engine, so naive catalog probing
would collide on every model name. The registry instead routes the
slot's loaded model via an explicit pin and excludes the slot's backend
from catalog probing. These tests pin that contract:

- pin_model routes a model to the slot even when it's not in the slot's
  advertised catalog;
- a pin wins over the primary's catalog entry for the same model (no
  cold-swap, no ``name@backend`` collision);
- the pinned model appears exactly once in the rebuilt map;
- an excluded backend never contributes its catalog to the map;
- unpin restores normal catalog routing.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from augmentum.models.base import ModelInfo


class _FakeBackend:
    """Minimal backend exposing the two surfaces the registry probes."""

    def __init__(self, model_names: list[str], loaded: str = "") -> None:
        self._names = list(model_names)
        # ``_local_backend_has_loaded`` reads ``backend._manager.model_id``.
        self._manager = MagicMock()
        self._manager.model_id = loaded

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name=n, model=n) for n in self._names]


def _registry():
    with patch("augmentum.models.provider_registry.settings") as mock_settings:
        mock_settings.ollama_base_url = "http://ollama:11434"
        mock_settings.openai_api_key = None
        mock_settings.llamacpp_base_url = None
        mock_settings.default_backend = "ollama"
        from augmentum.models.provider_registry import ProviderRegistry

        return ProviderRegistry(MagicMock())


@pytest.mark.asyncio
async def test_pin_routes_to_slot_backend():
    """A pinned model resolves to the slot backend even before any probe."""
    reg = _registry()
    slot = _FakeBackend([], loaded="ModelX")
    reg.register_backend("engine_secondary", slot)
    reg.pin_model("ModelX", "engine_secondary")

    backend, name = await reg.resolve_backend_for_model("ModelX")
    assert backend is slot
    assert name == "ModelX"


@pytest.mark.asyncio
async def test_pin_wins_over_primary_catalog():
    """When the same GGUF is in the primary catalog AND pinned to the
    slot, the pin wins — the request hits the resident slot, not a swap
    on the primary."""
    reg = _registry()
    primary = _FakeBackend(["ModelX", "ModelY"])
    slot = _FakeBackend([], loaded="ModelX")
    reg.register_backend("engine", primary)
    reg.register_backend("engine_secondary", slot)
    reg.exclude_backend_from_map("engine_secondary")
    reg.pin_model("ModelX", "engine_secondary")

    backend, name = await reg.resolve_backend_for_model("ModelX")
    assert backend is slot
    assert name == "ModelX"

    # The non-pinned sibling still routes to the primary.
    backend_y, _ = await reg.resolve_backend_for_model("ModelY")
    assert backend_y is primary


@pytest.mark.asyncio
async def test_pinned_model_appears_once_in_map():
    """The pin is injected into the rebuilt map as a single unique entry —
    no ``ModelX@engine`` / ``ModelX@engine_secondary`` collision pair."""
    reg = _registry()
    primary = _FakeBackend(["ModelX"])
    slot = _FakeBackend(["ModelX"], loaded="ModelX")  # would collide if probed
    reg.register_backend("engine", primary)
    reg.register_backend("engine_secondary", slot)
    reg.exclude_backend_from_map("engine_secondary")
    reg.pin_model("ModelX", "engine_secondary")

    new_map = await reg.refresh_model_map(force=True)
    assert new_map.get("ModelX") == "engine_secondary"
    assert "ModelX@engine" not in new_map
    assert "ModelX@engine_secondary" not in new_map


@pytest.mark.asyncio
async def test_excluded_backend_not_probed():
    """A model that exists ONLY on the excluded slot backend (not pinned)
    does not leak into the map — the slot never advertises its catalog."""
    reg = _registry()
    slot = _FakeBackend(["SlotOnly"])
    reg.register_backend("engine_secondary", slot)
    reg.exclude_backend_from_map("engine_secondary")

    new_map = await reg.refresh_model_map(force=True)
    assert "SlotOnly" not in new_map


@pytest.mark.asyncio
async def test_unpin_restores_catalog_routing():
    """After unpin, the model falls back to the primary catalog entry
    (swap-on-demand) instead of the slot."""
    reg = _registry()
    primary = _FakeBackend(["ModelX"])
    slot = _FakeBackend([], loaded="ModelX")
    reg.register_backend("engine", primary)
    reg.register_backend("engine_secondary", slot)
    reg.exclude_backend_from_map("engine_secondary")
    reg.pin_model("ModelX", "engine_secondary")

    backend, _ = await reg.resolve_backend_for_model("ModelX")
    assert backend is slot

    reg.unpin_model("ModelX")
    backend2, _ = await reg.resolve_backend_for_model("ModelX")
    assert backend2 is primary
    assert reg.pinned_backend_for("ModelX") == ""


@pytest.mark.asyncio
async def test_unpin_all_clears_every_pin():
    reg = _registry()
    slot = _FakeBackend([], loaded="A")
    reg.register_backend("engine_secondary", slot)
    reg.pin_model("A", "engine_secondary")
    reg.pin_model("B", "engine_secondary")

    reg.unpin_model()  # no arg → clear all
    assert reg.pinned_backend_for("A") == ""
    assert reg.pinned_backend_for("B") == ""


def test_pin_ignores_empty_inputs():
    reg = _registry()
    reg.pin_model("", "engine_secondary")
    reg.pin_model("ModelX", "")
    assert reg.pinned_backend_for("ModelX") == ""


def test_is_listing_excluded_predicate():
    """The exclusion predicate is the single source of truth consulted by
    every catalog-listing path (/api/tags, /v1/models, list_all_models)."""
    reg = _registry()
    assert reg.is_listing_excluded("engine_secondary") is False
    reg.exclude_backend_from_map("engine_secondary")
    assert reg.is_listing_excluded("engine_secondary") is True
    assert reg.is_listing_excluded("engine") is False


@pytest.mark.asyncio
async def test_excluded_slot_does_not_collide_in_list_all_models():
    """Regression: the secondary slot shares the primary's GGUF catalog, so
    listing it would tag every model ``name@engine`` / ``name (engine)`` in
    the model manager. Excluded → primary's catalog lists each model ONCE
    with no backend suffix (the reported '@engine on every model' bug)."""
    from augmentum.models.model_manager import ModelManager

    reg = _registry()
    reg._backends.clear()  # isolate to just the two engine backends
    primary = _FakeBackend(["ModelX", "ModelY"])
    slot = _FakeBackend(["ModelX", "ModelY"], loaded="ModelX")  # same catalog
    reg.register_backend("engine", primary)
    reg.register_backend("engine_secondary", slot)
    reg.exclude_backend_from_map("engine_secondary")

    mm = ModelManager(reg)
    models = await mm.list_all_models()
    names = sorted(m.name for m in models)
    # Single listable backend after exclusion → no suffix, no duplicates.
    assert names == ["ModelX", "ModelY"]
    assert not any("@" in n or "(" in n for n in names)
