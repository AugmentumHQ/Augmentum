"""Authorization paths for the subagent cancel route (P0 hardening).

The cancel endpoint must:
  - allow the documented race (no persisted row yet) — dispatcher is authoritative
  - refuse a cross-user cancel (row owned by someone else) with 403
  - fail CLOSED (503) when the ownership check itself errors, rather than
    fall through to an unverified cancel (a DB outage must not become a
    cancel-anyone primitive).
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from augmentum.proxy import coder_subagents_routes as sr


class _FakeDispatcher:
    def __init__(self) -> None:
        self.cancelled = []

    def cancel(self, subagent_id, reason=""):
        self.cancelled.append((subagent_id, reason))
        return True


def _make_request(store, user_id="alice"):
    user = SimpleNamespace(id=user_id)
    app = SimpleNamespace(state=SimpleNamespace(coder_subagent_store=store))
    req = SimpleNamespace(
        scope={"user": user},
        app=app,
        json=AsyncMock(return_value={"reason": "stop"}),
    )
    return req


def _body(resp):
    return json.loads(bytes(resp.body))


@pytest.mark.asyncio
async def test_cancel_fails_closed_when_store_errors(monkeypatch):
    dispatcher = _FakeDispatcher()
    monkeypatch.setattr(sr, "find_subagent_owner", lambda _id: dispatcher, raising=False)
    monkeypatch.setattr(
        "augmentum.agents.dispatch.find_subagent_owner",
        lambda _id: dispatcher,
        raising=False,
    )

    store = SimpleNamespace(get_run=AsyncMock(side_effect=RuntimeError("db down")))
    resp = await sr.cancel_subagent_run(_make_request(store), "sa_1")

    assert resp.status_code == 503
    assert _body(resp)["error"] == "ownership check failed"
    # Must NOT have reached the dispatcher.
    assert dispatcher.cancelled == []


@pytest.mark.asyncio
async def test_cancel_refuses_cross_user(monkeypatch):
    dispatcher = _FakeDispatcher()
    monkeypatch.setattr(
        "augmentum.agents.dispatch.find_subagent_owner",
        lambda _id: dispatcher,
        raising=False,
    )

    store = SimpleNamespace(
        get_run=AsyncMock(return_value=None),
        get_run_any=AsyncMock(return_value={"id": "sa_1", "user_id": "bob"}),
    )
    resp = await sr.cancel_subagent_run(_make_request(store), "sa_1")

    assert resp.status_code == 403
    assert dispatcher.cancelled == []


@pytest.mark.asyncio
async def test_cancel_allows_uncommitted_race(monkeypatch):
    dispatcher = _FakeDispatcher()
    monkeypatch.setattr(
        "augmentum.agents.dispatch.find_subagent_owner",
        lambda _id: dispatcher,
        raising=False,
    )

    # No row by user, and no row at all → genuine race, allow.
    store = SimpleNamespace(
        get_run=AsyncMock(return_value=None),
        get_run_any=AsyncMock(return_value=None),
    )
    resp = await sr.cancel_subagent_run(_make_request(store), "sa_1")

    assert resp.status_code == 200
    assert dispatcher.cancelled and dispatcher.cancelled[0][0] == "sa_1"
