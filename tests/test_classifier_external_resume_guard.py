"""Resuming the EXTERNAL classifier container while the managed Slot C holds
the classifier role must be refused — else it loads a duplicate model the
registry never routes to (the "third model after unpause" bug Matt hit).

Also covers the negative cases: when Slot C does NOT hold the key (external
still serving), and when the container isn't the classifier, resume proceeds.
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


@pytest.fixture
def _patched_set_paused(monkeypatch):
    fake = AsyncMock(return_value=(True, ""))
    monkeypatch.setattr(
        "augmentum.resource.container_probe.set_container_paused", fake
    )
    return fake


@pytest.mark.asyncio
async def test_resume_refused_when_slot_c_holds_classifier(_patched_set_paused):
    slot_backend = object()
    slot = SimpleNamespace(_backend=slot_backend)
    registry = SimpleNamespace(_backends={"classifier": slot_backend})
    state = SimpleNamespace(classifier_slot=slot, provider_registry=registry)

    resp = await resource_routes.resume_container(
        _request(state, {"container": "augmentum-classifier-1"}),
    )

    assert resp.status_code == 409
    _patched_set_paused.assert_not_awaited()      # never touched the container


@pytest.mark.asyncio
async def test_resume_allowed_when_external_still_holds(_patched_set_paused):
    slot_backend = object()
    external_backend = object()
    slot = SimpleNamespace(_backend=slot_backend)
    registry = SimpleNamespace(_backends={"classifier": external_backend})
    state = SimpleNamespace(classifier_slot=slot, provider_registry=registry)

    resp = await resource_routes.resume_container(
        _request(state, {"container": "augmentum-classifier-1"}),
    )

    assert resp.status_code == 200
    _patched_set_paused.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_non_classifier_container_unguarded(_patched_set_paused):
    # A TTS sidecar has nothing to do with the classifier role — never guarded,
    # even if Slot C happens to hold the classifier key.
    slot_backend = object()
    slot = SimpleNamespace(_backend=slot_backend)
    registry = SimpleNamespace(_backends={"classifier": slot_backend})
    state = SimpleNamespace(classifier_slot=slot, provider_registry=registry)

    resp = await resource_routes.resume_container(
        _request(state, {"container": "augmentum-kokoro-1"}),
    )

    assert resp.status_code == 200
    _patched_set_paused.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_no_slot_c_configured(_patched_set_paused):
    # Pure external install — no managed slot at all; resume proceeds.
    state = SimpleNamespace(classifier_slot=None, provider_registry=SimpleNamespace(_backends={}))

    resp = await resource_routes.resume_container(
        _request(state, {"container": "augmentum-classifier-1"}),
    )

    assert resp.status_code == 200
    _patched_set_paused.assert_awaited_once()
