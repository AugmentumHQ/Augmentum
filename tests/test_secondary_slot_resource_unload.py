"""The system resource panel's unload (X) must hit the RIGHT engine slot.

Both the primary engine and the secondary slot ("Slot B") surface in the
panel with backend="engine". The /api/resources/unload route therefore has
to disambiguate by model id — otherwise clicking X on the Slot B model stops
the primary engine and leaves Slot B resident, holding its VRAM (the
"not getting reclaimed" symptom).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.proxy import resource_routes


def _request(state, body):
    req = MagicMock()
    req.app.state = state
    req.json = AsyncMock(return_value=body)
    return req


def _state(*, secondary_model="", primary=None, secondary=None, registry=None, store=None):
    return SimpleNamespace(
        secondary_slot=secondary,
        llama_manager=primary,
        provider_registry=registry or MagicMock(),
        settings_store=store,
    )


@pytest.fixture(autouse=True)
def _no_ledger(monkeypatch):
    monkeypatch.setattr("augmentum.resource.ledger.invalidate", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(resource_routes.settings, "engine_secondary_model", "SlotB", raising=False)
    yield


@pytest.mark.asyncio
async def test_unload_slot_b_model_unloads_secondary_not_primary():
    secondary = MagicMock()
    secondary.manager = SimpleNamespace(model_id="SlotB")
    secondary.unload = AsyncMock()
    primary = MagicMock()
    primary.stop = AsyncMock()
    registry = MagicMock()
    store = MagicMock(); store.set = AsyncMock()
    state = _state(primary=primary, secondary=secondary, registry=registry, store=store)

    resp = await resource_routes.unload_model(
        _request(state, {"name": "SlotB", "backend": "engine"}),
    )

    assert resp.status_code == 200
    secondary.unload.assert_awaited_once()
    primary.stop.assert_not_awaited()           # primary untouched
    registry.unpin_model.assert_called_once_with("SlotB")
    store.set.assert_awaited_once_with("engine_secondary_model", "")
    assert resource_routes.settings.engine_secondary_model == ""


@pytest.mark.asyncio
async def test_unload_primary_model_does_not_touch_slot_b():
    secondary = MagicMock()
    secondary.manager = SimpleNamespace(model_id="SlotB")
    secondary.unload = AsyncMock()
    primary = MagicMock()
    primary.stop = AsyncMock()
    registry = MagicMock()
    state = _state(primary=primary, secondary=secondary, registry=registry)

    resp = await resource_routes.unload_model(
        _request(state, {"name": "PrimaryModel", "backend": "engine"}),
    )

    assert resp.status_code == 200
    primary.stop.assert_awaited_once()          # primary stopped
    secondary.unload.assert_not_awaited()       # Slot B untouched


@pytest.mark.asyncio
async def test_unload_engine_no_secondary_configured():
    """No secondary slot → behaves exactly as before (primary unload)."""
    primary = MagicMock()
    primary.stop = AsyncMock()
    state = _state(primary=primary, secondary=None, registry=MagicMock())

    resp = await resource_routes.unload_model(
        _request(state, {"name": "PrimaryModel", "backend": "engine"}),
    )

    assert resp.status_code == 200
    primary.stop.assert_awaited_once()
