"""Route tests for the secondary slot (Slot B) load/unload/status.

Exercises the route functions directly with stubbed app.state so the
contract is pinned without standing up the full app:

- load resolves the model, starts the slot, pins it, persists the
  sticky setting, and reports the model id;
- a missing model is 404 and never touches the slot;
- a load failure surfaces as 502/504 (not a silent 200) and does NOT
  pin or persist;
- unload stops the slot, drops the pin, clears the sticky setting;
- status returns both slots when the feature is enabled, and
  ``{enabled: False}`` when it isn't.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from augmentum.proxy import model_routes


def _request(state: SimpleNamespace, body: dict | None = None):
    """Minimal Request stand-in: ``.app.state`` + awaitable ``.json()``."""
    req = MagicMock()
    req.app.state = state
    req.json = AsyncMock(return_value=body or {})
    return req


def _state(*, slot=None, registry=None, store=None, primary=None, ledger=None):
    return SimpleNamespace(
        secondary_slot=slot,
        provider_registry=registry or MagicMock(),
        settings_store=store,
        llama_manager=primary,
        resource_ledger=ledger,
    )


@pytest.fixture(autouse=True)
def _reset_setting(monkeypatch):
    # Keep the module-level settings object from leaking across tests.
    monkeypatch.setattr(model_routes.settings, "engine_secondary_model", "", raising=False)
    yield


@pytest.mark.asyncio
async def test_load_success_pins_and_persists(monkeypatch):
    slot = MagicMock()
    slot.resolve_model_path.return_value = "/models/Qwen.gguf"
    slot.manager = MagicMock()
    slot.load = AsyncMock(return_value="Qwen3.6-35B")
    registry = MagicMock()
    store = MagicMock()
    store.set = AsyncMock()
    state = _state(slot=slot, registry=registry, store=store)

    # No explicit load options in the body → saved per-model config applies.
    monkeypatch.setattr(model_routes, "_extract_engine_load_options", lambda b, m: {})
    monkeypatch.setattr(
        "augmentum.resource.ledger.invalidate", lambda *a, **k: None, raising=False,
    )

    resp = await model_routes.engine_secondary_load(
        _request(state, {"model": "Qwen"}),
    )

    assert resp.status_code == 200
    slot.load.assert_awaited_once_with("/models/Qwen.gguf", load_options=None)
    registry.pin_model.assert_called_once_with("Qwen3.6-35B", "engine_secondary")
    store.set.assert_awaited_once_with("engine_secondary_model", "Qwen3.6-35B")
    assert model_routes.settings.engine_secondary_model == "Qwen3.6-35B"


@pytest.mark.asyncio
async def test_load_missing_model_is_404_and_no_load():
    slot = MagicMock()
    slot.resolve_model_path.return_value = None
    slot.load = AsyncMock()
    state = _state(slot=slot)

    with pytest.raises(HTTPException) as ei:
        await model_routes.engine_secondary_load(_request(state, {"model": "ghost"}))
    assert ei.value.status_code == 404
    slot.load.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_runtime_failure_is_502_and_no_pin(monkeypatch):
    slot = MagicMock()
    slot.resolve_model_path.return_value = "/models/X.gguf"
    slot.manager = MagicMock()
    slot.load = AsyncMock(side_effect=RuntimeError("CUDA OOM"))
    registry = MagicMock()
    state = _state(slot=slot, registry=registry)
    monkeypatch.setattr(model_routes, "_extract_engine_load_options", lambda b, m: {})

    with pytest.raises(HTTPException) as ei:
        await model_routes.engine_secondary_load(_request(state, {"model": "X"}))
    assert ei.value.status_code == 502
    registry.pin_model.assert_not_called()
    assert model_routes.settings.engine_secondary_model == ""


@pytest.mark.asyncio
async def test_load_blocked_by_admission_is_507_and_no_load(monkeypatch):
    slot = MagicMock()
    slot.resolve_model_path.return_value = "/models/Big.gguf"
    slot.manager = MagicMock()
    slot.load = AsyncMock()
    ledger = MagicMock()
    ledger.check_engine_fit = AsyncMock(return_value=(False, "needs ~18.0 GB VRAM but only ~6.0 GB is free", 18000, 6000))
    state = _state(slot=slot, ledger=ledger)
    monkeypatch.setattr(model_routes, "_extract_engine_load_options", lambda b, m: {})

    resp = await model_routes.engine_secondary_load(_request(state, {"model": "Big"}))

    assert resp.status_code == 507
    import json
    payload = json.loads(resp.body)
    assert payload["error"] == "insufficient_vram"
    assert payload["free_mb"] == 6000
    slot.load.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_force_bypasses_admission(monkeypatch):
    slot = MagicMock()
    slot.resolve_model_path.return_value = "/models/Big.gguf"
    slot.manager = MagicMock()
    slot.load = AsyncMock(return_value="Big")
    ledger = MagicMock()
    ledger.check_engine_fit = AsyncMock(return_value=(False, "too big", 18000, 6000))
    registry = MagicMock()
    store = MagicMock(); store.set = AsyncMock()
    state = _state(slot=slot, registry=registry, store=store, ledger=ledger)
    monkeypatch.setattr(model_routes, "_extract_engine_load_options", lambda b, m: {})
    monkeypatch.setattr("augmentum.resource.ledger.invalidate", lambda *a, **k: None, raising=False)

    resp = await model_routes.engine_secondary_load(
        _request(state, {"model": "Big", "force": True}),
    )

    assert resp.status_code == 200
    ledger.check_engine_fit.assert_not_awaited()  # force skips the gate
    slot.load.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_admission_pass_proceeds(monkeypatch):
    slot = MagicMock()
    slot.resolve_model_path.return_value = "/models/Small.gguf"
    slot.manager = MagicMock()
    slot.load = AsyncMock(return_value="Small")
    ledger = MagicMock()
    ledger.check_engine_fit = AsyncMock(return_value=(True, "", 4000, 12000))
    registry = MagicMock()
    store = MagicMock(); store.set = AsyncMock()
    state = _state(slot=slot, registry=registry, store=store, ledger=ledger)
    monkeypatch.setattr(model_routes, "_extract_engine_load_options", lambda b, m: {})
    monkeypatch.setattr("augmentum.resource.ledger.invalidate", lambda *a, **k: None, raising=False)

    resp = await model_routes.engine_secondary_load(_request(state, {"model": "Small"}))

    assert resp.status_code == 200
    ledger.check_engine_fit.assert_awaited_once()
    slot.load.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_disabled_slot_is_404():
    state = _state(slot=None)
    with pytest.raises(HTTPException) as ei:
        await model_routes.engine_secondary_load(_request(state, {"model": "X"}))
    assert ei.value.status_code == 404


@pytest.mark.asyncio
async def test_unload_stops_drops_pin_and_clears_setting(monkeypatch):
    slot = MagicMock()
    slot.manager = SimpleNamespace(model_id="Qwen3.6-35B")
    slot.unload = AsyncMock()
    registry = MagicMock()
    store = MagicMock()
    store.set = AsyncMock()
    state = _state(slot=slot, registry=registry, store=store)
    monkeypatch.setattr(model_routes.settings, "engine_secondary_model", "Qwen3.6-35B")
    monkeypatch.setattr(
        "augmentum.resource.ledger.invalidate", lambda *a, **k: None, raising=False,
    )

    resp = await model_routes.engine_secondary_unload(_request(state))

    assert resp.status_code == 200
    slot.unload.assert_awaited_once()
    registry.unpin_model.assert_called_once_with("Qwen3.6-35B")
    store.set.assert_awaited_once_with("engine_secondary_model", "")
    assert model_routes.settings.engine_secondary_model == ""


@pytest.mark.asyncio
async def test_status_enabled_returns_both_slots():
    slot = MagicMock()
    slot.status.return_value = {"loaded": True, "model_id": "B"}
    primary = MagicMock()
    primary.status.return_value = {"model_id": "A", "gpu": {"vram_free_mib": 4096}}
    state = _state(slot=slot, primary=primary)

    resp = await model_routes.engine_secondary_status(_request(state))
    import json

    payload = json.loads(resp.body)
    assert payload["enabled"] is True
    assert payload["secondary"]["model_id"] == "B"
    assert payload["primary"]["gpu"]["vram_free_mib"] == 4096


@pytest.mark.asyncio
async def test_status_disabled():
    state = _state(slot=None, primary=None)
    resp = await model_routes.engine_secondary_status(_request(state))
    import json

    payload = json.loads(resp.body)
    assert payload == {"enabled": False}


# ---------------------------------------------------------------------------
# Progress endpoints resolve to the right engine (#1)
# ---------------------------------------------------------------------------

def _progress_state(*, primary=None, secondary=None, pinned=None):
    registry = MagicMock()
    registry.pinned_backend_for = MagicMock(return_value=pinned or "")
    return SimpleNamespace(
        llama_manager=primary,
        secondary_slot=secondary,
        provider_registry=registry,
    )


def test_resolve_empty_model_returns_primary():
    primary = MagicMock()
    state = _progress_state(primary=primary, secondary=MagicMock())
    assert model_routes._resolve_engine_manager_for_model(_request(state), "") is primary


def test_resolve_pinned_model_returns_secondary():
    primary = MagicMock()
    sec = MagicMock(); sec.manager = MagicMock(); sec.manager.model_id = ""
    state = _progress_state(primary=primary, secondary=sec, pinned="engine_secondary")
    mgr = model_routes._resolve_engine_manager_for_model(_request(state), "SlotBModel")
    assert mgr is sec.manager


def test_resolve_loading_model_returns_secondary():
    """During a Slot B cold load the model isn't pinned yet but its load
    snapshot carries the model id — still resolve to the secondary."""
    primary = MagicMock()
    sec = MagicMock()
    sec.manager = MagicMock()
    sec.manager.model_id = ""
    sec.manager._load_progress = {"model_id": "Loading-B"}
    state = _progress_state(primary=primary, secondary=sec, pinned="")
    mgr = model_routes._resolve_engine_manager_for_model(_request(state), "Loading-B")
    assert mgr is sec.manager


def test_resolve_primary_model_returns_primary():
    primary = MagicMock()
    sec = MagicMock(); sec.manager = MagicMock(); sec.manager.model_id = "OtherB"
    sec.manager._load_progress = None
    state = _progress_state(primary=primary, secondary=sec, pinned="")
    mgr = model_routes._resolve_engine_manager_for_model(_request(state), "PrimaryModel")
    assert mgr is primary
