"""Trail substrate + take_me_there verb (headless-agency P2)."""

from __future__ import annotations

import pytest

import augmentum.intent  # noqa: F401 — registers builtins (incl. trail)
from augmentum.companion_runtime.native_loop import (
    CORE_TOOL_NAMES,
    TRAIL_CAP,
    _append_trail,
)
from augmentum.intent.action import ReferentCache, SessionContext
from augmentum.intent.registry import list_actions


def get_action(action_id: str):
    for a in list_actions():
        if a.id == action_id:
            return a
    return None


class _FakeAppState:
    pass


@pytest.fixture()
def session():
    return SessionContext(
        user_id="u1", session_id="s1", referents=ReferentCache(),
    )


def _refs_via_get(monkeypatch, refs):
    import importlib
    dispatch_mod = importlib.import_module("augmentum.intent.dispatch")
    monkeypatch.setattr(
        dispatch_mod, "get_referent_cache", lambda *a, **k: refs,
    )


def test_core_tools_include_headless_page_eye():
    assert "web_fetch" in CORE_TOOL_NAMES
    assert "web_search" in CORE_TOOL_NAMES


def test_append_trail_caps_and_shapes(monkeypatch):
    refs = ReferentCache()
    _refs_via_get(monkeypatch, refs)
    for i in range(TRAIL_CAP + 5):
        _append_trail(
            _FakeAppState(), "u1", "s1",
            kind="page", label=f"page {i}", ref=f"https://x.test/{i}",
        )
    assert len(refs.trail) == TRAIL_CAP
    head = refs.trail[-1]
    assert head["label"] == f"page {TRAIL_CAP + 4}"
    assert head["kind"] == "page"
    assert "ts" in head


def test_append_trail_never_raises_without_cache(monkeypatch):
    import importlib
    dispatch_mod = importlib.import_module("augmentum.intent.dispatch")
    def _boom(*a, **k):
        raise RuntimeError("no cache")
    monkeypatch.setattr(dispatch_mod, "get_referent_cache", _boom)
    _append_trail(_FakeAppState(), "u1", "s1", kind="page", label="x")


@pytest.mark.asyncio
async def test_take_me_there_empty_trail(session):
    action = get_action("companion.take_me_there")
    assert action is not None
    result = await action.handler("take me there", session, {})
    assert result.short_circuit
    assert result.surface_emit is None
    assert "haven't gone anywhere" in result.speak


@pytest.mark.asyncio
async def test_take_me_there_page_head(session):
    session.referents.trail.append({
        "kind": "page", "label": "AP article",
        "ref": "https://apnews.com/a", "ts": 0,
    })
    action = get_action("companion.take_me_there")
    result = await action.handler("take me there", session, {})
    assert result.surface_emit["channel"] == "browse.open_url"
    assert result.surface_emit["payload"]["url"] == "https://apnews.com/a"


@pytest.mark.asyncio
async def test_take_me_there_search_head(session):
    session.referents.trail.append({
        "kind": "search", "label": "latest ai news", "ref": "", "ts": 0,
    })
    action = get_action("companion.take_me_there")
    result = await action.handler("show me", session, {})
    assert result.surface_emit["channel"] == "browse.search"
    assert result.surface_emit["payload"]["query"] == "latest ai news"
