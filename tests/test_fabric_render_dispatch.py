"""Tests for the render dispatch + peer protocol.

Three layers:

  - execute_local_render: stub executor produces a RenderResult.
  - dispatch_render: orchestrates routing + execution. No-director
    (solo) path, local route, peer route, no-capable-node, missing
    http_client, missing peer_state. Each branch verified.
  - render_via_peer: signed POST to peer's /api/cast/render. Happy
    path, transport error, status >= 400, non-JSON. Never raises.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import httpx
import pytest

from augmentum.cast.dispatcher import dispatch_render
from augmentum.cast.executors import execute_local_render
from augmentum.cast.render import (
    RENDER_HTML,
    RENDER_VRM,
    RenderJob,
    RenderResult,
    RenderRoute,
)
from augmentum.fabric.identity import FabricIdentity
from augmentum.fabric.render_client import render_via_peer, serialise_result
from augmentum.state.settings_store import SettingsStore

# ── execute_local_render (stub) ───────────────────────────────────


@pytest.mark.asyncio
async def test_local_render_stub_returns_ok_result():
    """No renderer / store available → executor returns stub result.
    Single-machine no-Chrome installs land here and keep working.
    """
    job = RenderJob(kind=RENDER_HTML, target_device_id="dev_xxx",
                    payload={"html": "<h1>hi</h1>"})
    result = await execute_local_render(job, node_id="node_local")
    assert result.ok is True
    assert result.location == "local"
    assert result.node_id == "node_local"
    assert result.output_url.startswith("stub://html/")
    assert result.metadata.get("stub") is True
    assert "html" in result.metadata.get("payload_keys", [])


@pytest.mark.asyncio
async def test_local_render_html_with_renderer_returns_real_output_url():
    """When both renderer + output store are supplied, RENDER_HTML
    returns a token-gated /api/cast/render-output/{token} URL."""
    from augmentum.cast.output_store import RenderOutputStore

    class FakeRenderer:
        async def render_html_to_image(self, html, **_kwargs):
            assert "<h1>hi</h1>" in html
            return b"\x89PNG\r\n\x1a\nfake-rendered"

    store = RenderOutputStore()
    job = RenderJob(kind=RENDER_HTML, target_device_id="dev_xxx",
                    payload={"html": "<h1>hi</h1>"})
    result = await execute_local_render(
        job, node_id="node_local", user_id="u-42",
        html_renderer=FakeRenderer(), output_store=store,
    )

    assert result.ok is True
    assert result.location == "local"
    assert result.output_url.startswith("/api/cast/render-output/ro_")
    assert result.metadata["content_type"] == "image/png"
    assert result.metadata["bytes"] > 0
    assert result.metadata.get("stub") is None

    # The bytes are retrievable via the store.
    token = result.output_url.rsplit("/", 1)[-1]
    fetched = store.fetch(token)
    assert fetched is not None
    assert fetched.body.startswith(b"\x89PNG")


@pytest.mark.asyncio
async def test_local_render_html_with_renderer_failure_returns_failure():
    """A renderer that raises during render_html_to_image surfaces
    as ok=False with code='render_failed' — never propagates."""
    from augmentum.cast.output_store import RenderOutputStore

    class CrashRenderer:
        async def render_html_to_image(self, html, **_kwargs):
            raise RuntimeError("simulated chrome crash")

    store = RenderOutputStore()
    job = RenderJob(kind=RENDER_HTML, payload={"html": "<p>x</p>"})
    result = await execute_local_render(
        job, html_renderer=CrashRenderer(), output_store=store,
    )
    assert result.ok is False
    assert result.code == "render_failed"
    assert "simulated chrome crash" in result.message


@pytest.mark.asyncio
async def test_local_render_html_without_html_payload_fails_cleanly():
    """RENDER_HTML missing the 'html' payload field returns a clean
    failure code instead of crashing the renderer."""
    from augmentum.cast.output_store import RenderOutputStore

    store = RenderOutputStore()
    job = RenderJob(kind=RENDER_HTML, payload={})
    result = await execute_local_render(
        job, html_renderer=object(), output_store=store,
    )
    assert result.ok is False
    assert result.code == "payload_missing_html"


# ── dispatch_render ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_with_no_director_runs_local():
    """Single-machine grace: no fabric, no director → always run local
    without forcing the caller to construct fabric machinery."""
    job = RenderJob(kind=RENDER_HTML, target_device_id="dev_abc")
    result = await dispatch_render(job, user_id="u1")
    assert result.ok is True
    assert result.location == "local"


@pytest.mark.asyncio
async def test_dispatch_local_route_runs_local_executor():
    """Director returns RenderRoute('local', ...) → executor fires
    locally, not over fabric."""
    job = RenderJob(kind=RENDER_HTML)
    fake_director = MagicMock()
    fake_director.maybe_route_render = AsyncMock(return_value=RenderRoute(
        location="local", node_id="node_local", tier="standard",
    ))

    result = await dispatch_render(job, user_id="u1", director=fake_director)
    assert result.ok is True
    assert result.location == "local"
    assert result.node_id == "node_local"


@pytest.mark.asyncio
async def test_dispatch_none_route_returns_no_capable_node():
    """Router returns None → dispatcher returns the failure result
    with code='no_capable_node'."""
    job = RenderJob(kind=RENDER_VRM)
    fake_director = MagicMock()
    fake_director.maybe_route_render = AsyncMock(return_value=None)

    result = await dispatch_render(job, user_id="u1", director=fake_director)
    assert result.ok is False
    assert result.code == "no_capable_node"
    assert "vrm" in result.message


@pytest.mark.asyncio
async def test_dispatch_peer_route_without_http_client_fails_cleanly():
    """Peer route requires an http_client to dispatch. Missing one
    is a programming error but we return a clean RenderResult code
    rather than raise — preserves the dispatcher's no-exception
    contract."""
    job = RenderJob(kind=RENDER_HTML)
    fake_director = MagicMock()
    fake_director.maybe_route_render = AsyncMock(return_value=RenderRoute(
        location="peer", node_id="peer-x", tier="heavy",
    ))

    result = await dispatch_render(job, user_id="u1", director=fake_director)
    assert result.ok is False
    assert result.code == "missing_http_client"
    assert result.location == "peer"


@pytest.mark.asyncio
async def test_dispatch_peer_route_with_missing_peer_state_fails():
    """If the peer state is gone between routing and dispatch (rare
    race), the dispatcher returns 'peer_state_missing' instead of
    crashing trying to read None.paired.addr."""
    job = RenderJob(kind=RENDER_HTML)
    fake_director = MagicMock()
    fake_director.maybe_route_render = AsyncMock(return_value=RenderRoute(
        location="peer", node_id="peer-gone", tier="standard",
    ))
    fake_coord = MagicMock()
    fake_coord.peer_state = MagicMock(return_value=None)
    fake_director._coordinator = fake_coord
    fake_http = MagicMock()

    result = await dispatch_render(
        job, user_id="u1",
        director=fake_director, http_client=fake_http,
    )
    assert result.ok is False
    assert result.code == "peer_state_missing"


@pytest.mark.asyncio
async def test_dispatch_peer_route_calls_render_via_peer(monkeypatch):
    """Peer route + valid state → render_via_peer is invoked with
    the right args."""
    job = RenderJob(kind=RENDER_HTML, target_device_id="dev_xxx")

    fake_director = MagicMock()
    fake_director.maybe_route_render = AsyncMock(return_value=RenderRoute(
        location="peer", node_id="peer-x", tier="heavy",
    ))
    fake_paired = MagicMock(addr="192.168.1.50:6443")
    fake_state = MagicMock(paired=fake_paired)
    fake_director._coordinator = MagicMock()
    fake_director._coordinator.peer_state = MagicMock(return_value=fake_state)
    fake_director._coordinator._identity = MagicMock()

    captured = {}

    async def fake_render_via_peer(**kwargs):
        captured.update(kwargs)
        return RenderResult(
            ok=True, location="peer", node_id="peer-x",
            output_url="stub://html/dev_xxx",
        )

    monkeypatch.setattr(
        "augmentum.fabric.render_client.render_via_peer",
        fake_render_via_peer,
    )

    fake_http = MagicMock()
    result = await dispatch_render(
        job, user_id="u-42",
        director=fake_director, http_client=fake_http,
    )

    assert result.ok is True
    assert result.location == "peer"
    assert captured["peer_node_id"] == "peer-x"
    assert captured["peer_addr"] == "192.168.1.50:6443"
    assert captured["user_id"] == "u-42"
    assert captured["job"] is job


# ── render_via_peer ───────────────────────────────────────────────


async def _make_identity():
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.commit()
    return conn, await FabricIdentity.from_settings_store(SettingsStore(conn))


def _peer_response(body: dict, status: int = 200) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.json.return_value = body
    resp.text = json.dumps(body)
    return resp


@pytest.mark.asyncio
async def test_render_via_peer_happy_path():
    """Peer returns ok=True with output_url → render_via_peer parses
    the JSON into a RenderResult matching the peer's reply."""
    conn, identity = await _make_identity()
    try:
        fake_http = MagicMock()
        fake_http.post = AsyncMock(return_value=_peer_response({
            "ok": True, "location": "local", "node_id": "peer-x",
            "output_url": "stub://html/dev_xxx",
            "metadata": {"stub": True},
        }))

        result = await render_via_peer(
            http_client=fake_http, identity=identity, user_id="u",
            peer_node_id="peer-x", peer_addr="192.168.1.30:6443",
            job=RenderJob(kind=RENDER_HTML, target_device_id="dev_xxx"),
        )
        assert result.ok is True
        assert result.output_url == "stub://html/dev_xxx"

        # Body was sent as content= (raw bytes we signed), not json=.
        kwargs = fake_http.post.call_args[1]
        assert isinstance(kwargs["content"], bytes)
        assert "json" not in kwargs
        assert json.loads(kwargs["content"])["kind"] == RENDER_HTML
        # Signed envelope headers present.
        assert "X-Fabric-Sender" in kwargs["headers"]
        assert "X-Fabric-Signature" in kwargs["headers"]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_render_via_peer_unreachable_returns_failure_not_raise():
    """Connection error returns ok=False with code='peer_unreachable'
    rather than letting httpx exceptions propagate to the caller."""
    conn, identity = await _make_identity()
    try:
        fake_http = MagicMock()
        fake_http.post = AsyncMock(side_effect=httpx.ConnectError("LAN partition"))

        result = await render_via_peer(
            http_client=fake_http, identity=identity, user_id="u",
            peer_node_id="peer-x", peer_addr="192.168.1.30:6443",
            job=RenderJob(kind=RENDER_HTML),
        )
        assert result.ok is False
        assert result.code == "peer_unreachable"
        assert "LAN partition" in result.message
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_render_via_peer_status_5xx_returns_failure():
    """Peer responding 500 returns ok=False with the status in the
    code, so callers can distinguish from transport failures."""
    conn, identity = await _make_identity()
    try:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 503
        resp.text = "service degraded"
        fake_http = MagicMock()
        fake_http.post = AsyncMock(return_value=resp)

        result = await render_via_peer(
            http_client=fake_http, identity=identity, user_id="u",
            peer_node_id="peer-x", peer_addr="192.168.1.30:6443",
            job=RenderJob(kind=RENDER_HTML),
        )
        assert result.ok is False
        assert result.code == "peer_status_503"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_render_via_peer_non_json_returns_failure():
    """A peer returning HTML instead of JSON (e.g. mis-routed by a
    reverse proxy) is handled gracefully."""
    conn, identity = await _make_identity()
    try:
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.text = "<html>oops</html>"
        resp.json.side_effect = ValueError("not json")
        fake_http = MagicMock()
        fake_http.post = AsyncMock(return_value=resp)

        result = await render_via_peer(
            http_client=fake_http, identity=identity, user_id="u",
            peer_node_id="peer-x", peer_addr="192.168.1.30:6443",
            job=RenderJob(kind=RENDER_HTML),
        )
        assert result.ok is False
        assert result.code == "peer_non_json"
    finally:
        await conn.close()


# ── serialise_result ──────────────────────────────────────────────


def test_serialise_result_roundtrips_through_dict():
    """The wire-form of RenderResult uses asdict() so the receiver
    can reconstruct via dataclass kwargs without bespoke serializer."""
    original = RenderResult(
        ok=True, location="peer", node_id="peer-x",
        output_url="stub://html/dev_xxx",
        metadata={"stub": True, "extra": [1, 2, 3]},
    )
    raw = serialise_result(original)
    assert raw["ok"] is True
    assert raw["metadata"]["extra"] == [1, 2, 3]

    rebuilt = RenderResult(**raw)
    assert rebuilt == original
