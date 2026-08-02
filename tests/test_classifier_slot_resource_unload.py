"""The system resource panel's unload (X) must hit the RIGHT engine slot for
the managed classifier slot ("Slot C") too.

Slot C is RESIDENT and now shows up in the resource snapshot with
backend="engine" alongside the primary engine and Slot B. The
/api/resources/unload route therefore has to disambiguate by model id —
otherwise clicking X on the Slot C model stops the primary engine and leaves
the classifier resident, holding its VRAM (the "not getting reclaimed"
symptom). Parity with test_secondary_slot_resource_unload.py.
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


def _state(*, primary=None, secondary=None, classifier=None, registry=None):
    return SimpleNamespace(
        secondary_slot=secondary,
        classifier_slot=classifier,
        llama_manager=primary,
        provider_registry=registry or MagicMock(),
        settings_store=None,
    )


@pytest.fixture(autouse=True)
def _no_ledger(monkeypatch):
    monkeypatch.setattr(
        "augmentum.resource.ledger.invalidate", lambda *a, **k: None, raising=False
    )
    yield


@pytest.mark.asyncio
async def test_unload_slot_c_model_unloads_classifier_not_primary(monkeypatch):
    classifier = MagicMock()
    classifier.manager = SimpleNamespace(model_id="SlotC")
    primary = MagicMock()
    primary.stop = AsyncMock()
    state = _state(primary=primary, secondary=None, classifier=classifier)

    # The route delegates to the canonical handler — patch it so we assert the
    # disambiguation (Slot C chosen, primary untouched) without exercising the
    # registry/settings internals covered by the classifier route's own tests.
    fake_unload = AsyncMock()
    monkeypatch.setattr(
        "augmentum.proxy.model_routes.engine_classifier_unload", fake_unload
    )

    resp = await resource_routes.unload_model(
        _request(state, {"name": "SlotC", "backend": "engine"}),
    )

    assert resp.status_code == 200
    fake_unload.assert_awaited_once()
    primary.stop.assert_not_awaited()           # primary untouched


@pytest.mark.asyncio
async def test_unload_primary_model_does_not_touch_slot_c(monkeypatch):
    classifier = MagicMock()
    classifier.manager = SimpleNamespace(model_id="SlotC")
    primary = MagicMock()
    primary.stop = AsyncMock()
    state = _state(primary=primary, secondary=None, classifier=classifier)

    fake_unload = AsyncMock()
    monkeypatch.setattr(
        "augmentum.proxy.model_routes.engine_classifier_unload", fake_unload
    )

    resp = await resource_routes.unload_model(
        _request(state, {"name": "PrimaryModel", "backend": "engine"}),
    )

    assert resp.status_code == 200
    primary.stop.assert_awaited_once()          # primary stopped
    fake_unload.assert_not_awaited()            # Slot C untouched


@pytest.mark.asyncio
async def test_unload_engine_no_classifier_configured():
    """No classifier slot → behaves exactly as before (primary unload)."""
    primary = MagicMock()
    primary.stop = AsyncMock()
    state = _state(primary=primary, secondary=None, classifier=None)

    resp = await resource_routes.unload_model(
        _request(state, {"name": "PrimaryModel", "backend": "engine"}),
    )

    assert resp.status_code == 200
    primary.stop.assert_awaited_once()
