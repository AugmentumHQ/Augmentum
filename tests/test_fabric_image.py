"""Tests for Phase 7 image-generation routing.

Two layers:

  - generate_image_via_peer: two-step round-trip — POST generate
    then GET bytes — both signed envelopes. Covers happy path,
    body integrity coverage, unreachable, mid-flight disconnect.
  - RoutingDirector.maybe_route_image: local-first invariant, peer
    picking, empty when nobody advertises.

The route handler hook in image_routes.py is deferred (that file
has too many concurrent agent edits to commit safely); tests here
pin the contract so wiring it is a tiny mechanical step later.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import httpx
import pytest

from augmentum.fabric.coordinator import FabricCoordinator
from augmentum.fabric.director import RoutingDirector
from augmentum.fabric.identity import FabricIdentity
from augmentum.fabric.image_client import (
    RemoteImageError,
    generate_image_via_peer,
)
from augmentum.fabric.peer_auth import PairedPeer
from augmentum.state.settings_store import SettingsStore

# ── Helpers ───────────────────────────────────────────────────────


async def _make_identity_db():
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings (key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        " updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.execute(
        """CREATE TABLE fabric_nodes (
            id TEXT PRIMARY KEY, hostname TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL DEFAULT 'peer',
            pubkey_ed25519 TEXT NOT NULL, pubkey_fingerprint TEXT NOT NULL,
            addr TEXT NOT NULL DEFAULT '', tier TEXT NOT NULL DEFAULT 'local',
            fabric_share_enabled INTEGER NOT NULL DEFAULT 1,
            paired_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_seen_at TEXT, icon TEXT NOT NULL DEFAULT '')"""
    )
    await conn.commit()
    return conn


def _peer(node_id: str, addr: str = "192.168.1.30:6443") -> PairedPeer:
    return PairedPeer(
        node_id=node_id, hostname=f"h-{node_id[:4]}", role="peer",
        pubkey_b64="dGVzdA==", fingerprint=f"SHA256:{node_id[:8]}",
        addr=addr, tier="local", fabric_share_enabled=True,
        paired_at="2026-05-16 00:00:00", last_seen_at=None,
    )


def _gen_response(image_id: str) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "image_id": image_id, "job_id": "j-1",
        "url": f"/api/image/{image_id}",
        "status": "completed",
    }
    return resp


def _bytes_response(payload: bytes) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = 200
    resp.content = payload
    return resp


# ── generate_image_via_peer ───────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_via_peer_two_step_round_trip():
    """Happy path: POST returns image_id, GET returns bytes."""
    conn = await _make_identity_db()
    try:
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))

        async def fake_post(url, **kwargs):
            assert url.endswith("/api/image/generate")
            return _gen_response("img-abc")

        async def fake_get(url, **kwargs):
            assert url.endswith("/api/image/img-abc")
            return _bytes_response(b"\x89PNG\r\n\x1a\n" + b"fake-png-data")

        fake_client = MagicMock()
        fake_client.post = AsyncMock(side_effect=fake_post)
        fake_client.get = AsyncMock(side_effect=fake_get)

        image_bytes, metadata = await generate_image_via_peer(
            http_client=fake_client, identity=identity,
            user_id="user-42", peer_addr="192.168.1.30:6443",
            generate_request_payload={"prompt": "a cat", "width": 512, "height": 512},
        )

        assert image_bytes.startswith(b"\x89PNG")
        # The peer-relative URL is stripped from metadata so the caller
        # doesn't accidentally surface it to a local consumer.
        assert "url" not in metadata
        assert metadata["image_id"] == "img-abc"
        # Both requests were signed.
        post_headers = fake_client.post.call_args[1]["headers"]
        get_headers = fake_client.get.call_args[1]["headers"]
        assert post_headers["X-Fabric-Sender"] == identity.node_id
        assert get_headers["X-Fabric-Sender"] == identity.node_id
        assert "X-Fabric-Signature" in post_headers
        assert "X-Fabric-Signature" in get_headers
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_generate_body_is_what_was_signed():
    """The POST body MUST be the exact bytes covered by the signed
    sha256, not a re-serialised version (which httpx's json= would
    produce). Without this guarantee the receiver would reject the
    body-integrity check.
    """
    conn = await _make_identity_db()
    try:
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))

        fake_client = MagicMock()
        fake_client.post = AsyncMock(return_value=_gen_response("img-x"))
        fake_client.get = AsyncMock(return_value=_bytes_response(b"x"))

        payload = {"prompt": "a dog", "model": "flux-dev", "seed": 42}
        await generate_image_via_peer(
            http_client=fake_client, identity=identity,
            user_id="u", peer_addr="192.168.1.30:6443",
            generate_request_payload=payload,
        )

        # The post call used content= (raw bytes), not json= (which
        # would mean httpx serialises it for us and we couldn't have
        # signed the actual bytes).
        kwargs = fake_client.post.call_args[1]
        body = kwargs["content"]
        assert isinstance(body, bytes)
        # The bytes match what we'd produce ourselves.
        assert json.loads(body) == payload
        assert "json" not in kwargs
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_generate_unreachable_becomes_remote_image_error():
    """A connection error → RemoteImageError so the caller can
    surface it as "image gen failed" instead of letting a raw
    httpx exception bubble up.
    """
    conn = await _make_identity_db()
    try:
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
        fake_client = MagicMock()
        fake_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(RemoteImageError) as excinfo:
            await generate_image_via_peer(
                http_client=fake_client, identity=identity, user_id="u",
                peer_addr="192.168.1.99:6443",
                generate_request_payload={"prompt": "anything"},
            )
        assert "unreachable" in str(excinfo.value).lower()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_generate_disconnect_mid_fetch_raises():
    """Peer returned image_id then died before serving the bytes —
    we MUST NOT report success with no bytes. The user would see
    nothing happen and have no idea why.
    """
    conn = await _make_identity_db()
    try:
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
        fake_client = MagicMock()
        fake_client.post = AsyncMock(return_value=_gen_response("img-y"))
        fake_client.get = AsyncMock(
            side_effect=httpx.ReadError("connection reset"),
        )

        with pytest.raises(RemoteImageError) as excinfo:
            await generate_image_via_peer(
                http_client=fake_client, identity=identity, user_id="u",
                peer_addr="192.168.1.30:6443",
                generate_request_payload={"prompt": "x"},
            )
        assert "before image fetch" in str(excinfo.value).lower()
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_generate_empty_bytes_treated_as_error():
    """Receiver returned 200 with zero bytes — that's not a real
    image. Treat as failure rather than silently producing an empty
    image in the user's library.
    """
    conn = await _make_identity_db()
    try:
        identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
        fake_client = MagicMock()
        fake_client.post = AsyncMock(return_value=_gen_response("img-z"))
        fake_client.get = AsyncMock(return_value=_bytes_response(b""))

        with pytest.raises(RemoteImageError) as excinfo:
            await generate_image_via_peer(
                http_client=fake_client, identity=identity, user_id="u",
                peer_addr="192.168.1.30:6443",
                generate_request_payload={"prompt": "x"},
            )
        assert "empty" in str(excinfo.value).lower()
    finally:
        await conn.close()


# ── RoutingDirector.maybe_route_image ─────────────────────────────


async def _make_director_with_image_peer(model_id: str):
    conn = await _make_identity_db()
    identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
    coord = FabricCoordinator(identity, conn)
    await coord.register_paired_peer(_peer("peer-img"))
    ws = MagicMock()
    ws.close = AsyncMock()
    await coord.attach_connection("peer-img", ws)
    coord.record_remote_capabilities("peer-img", [{
        "kind": "image.generation", "schema_version": 1,
        "model_id": model_id, "family": "flux", "loaded": True,
        "max_resolution": "1024x1024",
    }])
    director = RoutingDirector(coord, MagicMock())
    return conn, director


@pytest.mark.asyncio
async def test_maybe_route_image_returns_none_when_local_can_serve():
    """Local-first invariant: if local has the model, never route.
    This is the load-bearing assertion that protects users from
    silent latency degradation.
    """
    conn, director = await _make_director_with_image_peer("flux-dev")
    try:
        peer = await director.maybe_route_image(
            model_id="flux-dev", local_can_serve=True,
        )
        assert peer is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_maybe_route_image_picks_peer_when_local_missing():
    """Local can't serve + peer advertises it → route to peer."""
    conn, director = await _make_director_with_image_peer("flux-dev")
    try:
        peer = await director.maybe_route_image(
            model_id="flux-dev", local_can_serve=False,
        )
        assert peer is not None
        peer_node_id, peer_addr = peer
        assert peer_node_id == "peer-img"
        assert peer_addr == "192.168.1.30:6443"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_maybe_route_image_no_match_returns_none():
    """Local can't serve AND no peer advertises it → stay local.
    The local pipeline will fail with a clean "model not found"
    error rather than the director masking it.
    """
    conn, director = await _make_director_with_image_peer("flux-dev")
    try:
        peer = await director.maybe_route_image(
            model_id="dalle-3", local_can_serve=False,
        )
        assert peer is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_route_helper_treats_disk_only_models_as_local():
    """Regression for the inventory-drift bug.

    The dropdown reads ``ModelManager.list_local_models()`` (disk scan
    of model_dir + system_dir). The fabric routing helper had only been
    reading ``ImagePersistence.list_models()`` (SQLite ``image_models``
    rows, populated only by the download flow). Baked-in system models
    and hand-dropped folders are in the dropdown but missing from
    SQLite -- the helper called those "not local" and dispatched to a
    peer, which on host B then errored with "fabric image dispatch
    infrastructure not initialised" once the route picked a peer that
    the local host couldn't dial. Fix unions both sources.
    """
    from unittest.mock import MagicMock as _MM

    from fastapi import Request as _Req

    from augmentum.image.schemas import GenerateRequest
    from augmentum.proxy.image_routes import _maybe_route_image_to_peer

    conn, director = await _make_director_with_image_peer("dreamshaper_8")
    try:
        # SQLite reports empty; the disk scanner has the model.
        persistence = _MM()
        persistence.list_models = AsyncMock(return_value=[])
        model_mgr = _MM()
        model_mgr.list_local_models.return_value = [
            {"name": "dreamshaper_8", "pipeline_type": _MM(), "path": "/baked"},
        ]

        request = _MM(spec=_Req)
        request.app = _MM()
        request.app.state = _MM()
        request.app.state.fabric_director = director
        request.app.state.image_persistence = persistence
        request.app.state.image_model_manager = model_mgr
        request.app.state.fabric_coordinator = _MM()
        request.app.state.fabric_http_client = _MM()

        req = GenerateRequest(prompt="cat", model="dreamshaper_8")
        result = await _maybe_route_image_to_peer(req, request, "dreamshaper_8")
        # Stay local: helper returns None even though a peer advertises
        # this model, because the disk scan proved we have it.
        assert result is None
    finally:
        await conn.close()
