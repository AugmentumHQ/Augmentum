"""Route tests for classifier-slot ("Slot C") takeover of the external
Docker classifier sidecar.

On external-sidecar installs (compose.classifier.yaml) the "classifier"
backend key is held by an env-frozen container — historically the model
manager could not change the classifier's model/ctx/mmproj at all. The
load route now supports an explicit user takeover:

- loading while the external holds the key WITHOUT take_over → 409 with
  ``take_over_required`` (surface the choice, never silently reroute);
- WITH take_over=true → the external backend is stashed and the slot's
  backend takes the key;
- unload restores the stashed external registration (and says so).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.models.classifier_slot import CLASSIFIER_BACKEND_KEY
from augmentum.proxy import model_routes


def _request(state: SimpleNamespace, body: dict | None = None):
    req = MagicMock()
    req.app.state = state
    req.json = AsyncMock(return_value=body or {})
    return req


def _slot():
    slot = MagicMock()
    slot.resolve_model_path.return_value = "/models/gemma-4-e2b.gguf"
    slot.manager = MagicMock()
    slot.load = AsyncMock(return_value="gemma-4-e2b")
    slot.unload = AsyncMock()
    slot.is_vision_capable.return_value = False
    slot.backend = MagicMock(name="slot_backend")
    return slot


def _state(slot, registry):
    return SimpleNamespace(
        classifier_slot=slot,
        classifier_external_backend=None,
        provider_registry=registry,
        settings_store=None,
        resource_ledger=None,
    )


def _registry(holder):
    registry = MagicMock()
    registry._backends = {CLASSIFIER_BACKEND_KEY: holder} if holder else {}
    registry.refresh_model_map = AsyncMock()
    return registry


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(model_routes, "_extract_engine_load_options", lambda b, m: {})
    monkeypatch.setattr(
        "augmentum.resource.ledger.invalidate", lambda *a, **k: None, raising=False,
    )
    # Stub the Docker container helpers so tests never touch a real daemon
    # (both the takeover-stop and unload-resume paths call these). Inline
    # imports in the routes resolve to the source module at call time, so
    # patching there takes effect.
    monkeypatch.setattr(
        "augmentum.resource.container_probe.find_sidecar_container",
        AsyncMock(return_value=None), raising=False,
    )
    monkeypatch.setattr(
        "augmentum.resource.container_probe.set_container_paused",
        AsyncMock(return_value=(True, "")), raising=False,
    )
    yield


def _reachable_backend(name: str = "gemma-4-e2b-ext"):
    """A backend mock whose list_models() reports a served model (so the
    unload health-probe treats it as reachable)."""
    b = MagicMock(name="external_backend")
    b.list_models = AsyncMock(return_value=[SimpleNamespace(name=name)])
    return b


@pytest.mark.asyncio
async def test_load_over_external_requires_take_over():
    slot = _slot()
    external = MagicMock(name="external_backend")
    registry = _registry(external)
    state = _state(slot, registry)

    resp = await model_routes.engine_classifier_load(
        _request(state, {"model": "gemma-4-e2b"}),
    )

    assert resp.status_code == 409
    slot.load.assert_not_awaited()
    assert registry._backends[CLASSIFIER_BACKEND_KEY] is external  # untouched


@pytest.mark.asyncio
async def test_take_over_stashes_external_and_registers_slot():
    slot = _slot()
    external = MagicMock(name="external_backend")
    registry = _registry(external)
    registry.register_backend = MagicMock(
        side_effect=lambda k, b: registry._backends.__setitem__(k, b),
    )
    state = _state(slot, registry)

    resp = await model_routes.engine_classifier_load(
        _request(state, {"model": "gemma-4-e2b", "take_over": True}),
    )

    assert resp.status_code == 200
    slot.load.assert_awaited_once()
    assert state.classifier_external_backend is external
    assert registry._backends[CLASSIFIER_BACKEND_KEY] is slot.backend


@pytest.mark.asyncio
async def test_unload_restores_reachable_external_and_syncs_name():
    slot = _slot()
    external = _reachable_backend("gemma-4-e2b-ext")
    registry = _registry(slot.backend)
    registry.register_backend = MagicMock(
        side_effect=lambda k, b: registry._backends.__setitem__(k, b),
    )
    state = _state(slot, registry)
    state.classifier_external_backend = external
    state.classifier_external_container = "augmentum-classifier-1"
    store = MagicMock(); store.set = AsyncMock()
    state.settings_store = store

    resp = await model_routes.engine_classifier_unload(_request(state))

    assert resp.status_code == 200
    slot.unload.assert_awaited_once()
    assert registry._backends[CLASSIFIER_BACKEND_KEY] is external
    assert state.classifier_external_backend is None
    import json
    assert json.loads(resp.body)["restored_external"] is True
    # Bug A: classifier_sidecar_model is re-synced to the EXTERNAL's served
    # model, never left as the just-unloaded slot's model.
    store.set.assert_any_await("classifier_sidecar_model", "gemma-4-e2b-ext")
    store.set.assert_any_await("classifier_slot_model", "")


@pytest.mark.asyncio
async def test_unload_external_unreachable_fails_open():
    """Bug B: if the external body is a corpse, unload must NOT route the role
    to it — drop the key so classify fails open to primary, and don't report a
    restore."""
    slot = _slot()
    dead = MagicMock(name="dead_external")
    dead.list_models = AsyncMock(side_effect=OSError("Name or service not known"))
    registry = _registry(slot.backend)
    registry.register_backend = MagicMock(
        side_effect=lambda k, b: registry._backends.__setitem__(k, b),
    )
    state = _state(slot, registry)
    state.classifier_external_backend = dead
    store = MagicMock(); store.set = AsyncMock()
    state.settings_store = store

    resp = await model_routes.engine_classifier_unload(_request(state))

    assert resp.status_code == 200
    # Key dropped (not pointed at the corpse); role falls open to primary tier.
    assert CLASSIFIER_BACKEND_KEY not in registry._backends
    import json
    assert json.loads(resp.body)["restored_external"] is False
    # Stale slot model name cleared either way.
    store.set.assert_any_await("classifier_sidecar_model", "")


@pytest.mark.asyncio
async def test_unload_without_stash_pops_key():
    slot = _slot()
    registry = _registry(slot.backend)
    state = _state(slot, registry)

    resp = await model_routes.engine_classifier_unload(_request(state))

    assert resp.status_code == 200
    assert CLASSIFIER_BACKEND_KEY not in registry._backends
    import json
    assert json.loads(resp.body)["restored_external"] is False


@pytest.mark.asyncio
async def test_status_reports_serving_source():
    slot = _slot()
    slot.status.return_value = {"state": "READY"}
    external = MagicMock(name="external_backend")

    for holder, expected in (
        (slot.backend, "managed"),
        (external, "external"),
        (None, "none"),
    ):
        registry = _registry(holder)
        state = _state(slot, registry)
        resp = await model_routes.engine_classifier_status(_request(state))
        import json
        assert json.loads(resp.body)["serving"] == expected
