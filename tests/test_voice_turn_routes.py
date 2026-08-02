"""Tests for the HTTPS companion turn endpoint (voice_turn_routes).

Protects the cert-free, model-driven agency seam:
  * companion unavailable (runtime off / no façade / signed out) →
    handled=false so the client falls through to chat
  * empty transcript → clean no-op
  * a turn that produces reply text + surface events → handled=true with
    reply + reshaped {channel, payload} surfaces the client can route
  * a loop error never 500s — always 200, handled=false

The endpoint reuses BeccaDirectHandler (the web app's companion chat
path) which composes her prompt and consumes native_loop_events; the
tests stub the handler + backend so they stay fast and offline.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from augmentum.proxy.voice_turn_routes import VoiceTurnRequest, voice_turn


class _Companion:
    def __init__(self, started=True):
        self.started = started
        self.runtime = SimpleNamespace()


def _req(user_id="u1", companion="default"):
    if companion == "default":
        companion = _Companion()
    companions = {"becca": companion} if companion is not None else {}
    app_state = SimpleNamespace(
        companions=companions,
        state_manager=None,
        intent_referents={},
    )
    scope = {}
    if user_id:
        scope["user"] = SimpleNamespace(id=user_id)
    return SimpleNamespace(app=SimpleNamespace(state=app_state), scope=scope)


def _body(resp):
    return json.loads(resp.body)


def _enable_runtime(monkeypatch):
    from augmentum.proxy import voice_turn_routes as mod
    monkeypatch.setattr(mod.settings, "companion_runtime_enabled", True, raising=False)


def _patch_tiers(monkeypatch):
    import augmentum.companion_runtime.tiers  # noqa: F401
    tmod = sys.modules["augmentum.companion_runtime.tiers"]

    async def _primary(runtime):
        return (object(), "test-model")

    monkeypatch.setattr(tmod, "primary", _primary)


def _patch_handler(monkeypatch, chunks):
    """Replace BeccaDirectHandler with a stub yielding the given chunks."""
    import augmentum.modes.becca_direct.handler as hmod

    class _Stub:
        def __init__(self, *a, **k):
            pass

        async def handle_stream(self, request):
            for c in chunks:
                yield c

    monkeypatch.setattr(hmod, "BeccaDirectHandler", _Stub)


@pytest.mark.asyncio
async def test_empty_transcript_noop():
    resp = await voice_turn(_req(), VoiceTurnRequest(transcript="   "))
    data = _body(resp)
    assert data["handled"] is False
    assert data.get("empty") is True


@pytest.mark.asyncio
async def test_runtime_off_falls_through():
    # companion_runtime_enabled defaults off → handled false
    resp = await voice_turn(_req(), VoiceTurnRequest(transcript="open browse"))
    assert _body(resp)["handled"] is False


@pytest.mark.asyncio
async def test_no_user_falls_through(monkeypatch):
    _enable_runtime(monkeypatch)
    resp = await voice_turn(_req(user_id=""), VoiceTurnRequest(transcript="open browse"))
    assert _body(resp)["handled"] is False


@pytest.mark.asyncio
async def test_companion_not_started_falls_through(monkeypatch):
    _enable_runtime(monkeypatch)
    resp = await voice_turn(
        _req(companion=_Companion(started=False)),
        VoiceTurnRequest(transcript="open browse"),
    )
    assert _body(resp)["handled"] is False


@pytest.mark.asyncio
async def test_reply_and_surfaces(monkeypatch):
    _enable_runtime(monkeypatch)
    _patch_tiers(monkeypatch)
    chunks = [
        SimpleNamespace(content_delta="Opening ", augmentum=None),
        SimpleNamespace(content_delta="browse.", augmentum=None),
        SimpleNamespace(
            content_delta="",
            augmentum={"becca_tool_result": {"ui_effects": [
                {"kind": "navigate.open_surface", "target": "_inline",
                 "payload": {"surface": "browse"}},
            ]}},
        ),
        SimpleNamespace(content_delta="", augmentum=None, done=True),
    ]
    _patch_handler(monkeypatch, chunks)

    resp = await voice_turn(_req(), VoiceTurnRequest(transcript="open browse"))
    data = _body(resp)
    assert data["handled"] is True
    assert data["reply"] == "Opening browse."
    assert data["speak"] == "Opening browse."
    assert data["surfaces"][0]["channel"] == "navigate.open_surface"
    assert data["surfaces"][0]["payload"]["surface"] == "browse"


@pytest.mark.asyncio
async def test_reply_only_no_surfaces(monkeypatch):
    _enable_runtime(monkeypatch)
    _patch_tiers(monkeypatch)
    chunks = [SimpleNamespace(content_delta="I'm doing well, thanks.", augmentum=None)]
    _patch_handler(monkeypatch, chunks)

    resp = await voice_turn(_req(), VoiceTurnRequest(transcript="how are you"))
    data = _body(resp)
    assert data["handled"] is True
    assert data["reply"] == "I'm doing well, thanks."
    assert data["surfaces"] == []


@pytest.mark.asyncio
async def test_empty_reply_no_surfaces_falls_through(monkeypatch):
    _enable_runtime(monkeypatch)
    _patch_tiers(monkeypatch)
    _patch_handler(monkeypatch, [SimpleNamespace(content_delta="", augmentum=None)])

    resp = await voice_turn(_req(), VoiceTurnRequest(transcript="hmm"))
    assert _body(resp)["handled"] is False


@pytest.mark.asyncio
async def test_loop_error_never_500(monkeypatch):
    _enable_runtime(monkeypatch)
    _patch_tiers(monkeypatch)
    import augmentum.modes.becca_direct.handler as hmod

    class _Boom:
        def __init__(self, *a, **k):
            pass

        async def handle_stream(self, request):
            raise RuntimeError("kaboom")
            yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(hmod, "BeccaDirectHandler", _Boom)
    resp = await voice_turn(_req(), VoiceTurnRequest(transcript="open browse"))
    assert resp.status_code == 200
    assert _body(resp)["handled"] is False
