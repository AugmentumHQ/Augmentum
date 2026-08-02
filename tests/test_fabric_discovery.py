"""Tests for LAN-discovery primitives.

Three layers:

  - ``enumerate_hosts``: subnet validation. Non-RFC1918, loopback,
    and >/22 ranges all return empty. Valid /24 yields ~254 hosts.
  - ``_parse_hello_response``: tightness on the wire shape.
    Mis-typed service name / missing fields / bad fingerprint format
    all yield None rather than a half-built peer record.
  - ``sweep_subnet``: end-to-end behaviour with mocked HTTP.
    Self-filtering, already-paired filtering, error surfacing, and
    503-on-peer handling.

The hello + discover routes get their own section at the bottom --
admin guard, fabric_enabled gate, response shape.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import aiosqlite
import httpx
import pytest

from augmentum.fabric import discovery as disc
from augmentum.fabric.coordinator import FabricCoordinator
from augmentum.fabric.identity import FabricIdentity
from augmentum.proxy.fabric_routes import fabric_discover, fabric_hello
from augmentum.state.settings_store import SettingsStore


# ── enumerate_hosts ───────────────────────────────────────────────


def test_enumerate_hosts_rejects_public_ranges():
    """Augmentum must never sweep public IP space -- the operator
    typed a wrong CIDR, refuse rather than become a port scanner.
    """
    assert disc.enumerate_hosts("8.8.8.0/24") == []
    assert disc.enumerate_hosts("1.1.1.0/24") == []


def test_enumerate_hosts_rejects_loopback():
    """Loopback subnets would just discover this node -- skip them."""
    assert disc.enumerate_hosts("127.0.0.0/24") == []


def test_enumerate_hosts_caps_at_slash22():
    """Subnets wider than /22 (1024 addresses) are almost certainly
    mis-entered. Refuse rather than block the operator UI for minutes
    sweeping /16.
    """
    # /21 = 2048 addresses — over the cap.
    assert disc.enumerate_hosts("10.0.0.0/21") == []
    # /22 = 1024 addresses (boundary). 1024 > 1024 is false, so the
    # cap rejects ONLY when strictly over -- /22 should be allowed.
    # Our implementation gates on `num_addresses > 1024` so /22 passes.
    hosts_22 = disc.enumerate_hosts("10.0.0.0/22")
    assert len(hosts_22) > 0


def test_enumerate_hosts_returns_full_slash24():
    """A typical /24 returns 254 hosts (skip network + broadcast)."""
    hosts = disc.enumerate_hosts("192.168.1.0/24")
    assert len(hosts) == 254
    assert "192.168.1.1" in hosts
    assert "192.168.1.254" in hosts
    assert "192.168.1.0" not in hosts
    assert "192.168.1.255" not in hosts


def test_enumerate_hosts_handles_malformed_cidr():
    """Bad input returns [] rather than raising."""
    assert disc.enumerate_hosts("not-a-subnet") == []
    assert disc.enumerate_hosts("") == []
    assert disc.enumerate_hosts("192.168.1.0/99") == []


# ── _parse_hello_response ─────────────────────────────────────────


def _valid_hello_body() -> dict:
    return {
        "service": "augmentum-fabric",
        "node_id": "abc123",
        "fingerprint": "SHA256:" + "a" * 32,
        "public_key": "dGVzdA==",
        "hostname": "peer-host",
        "version": "0.1.0",
        "icon": "🚀",
    }


def test_parse_hello_rejects_wrong_service():
    """A random HTTP responder on 6443 might return JSON with no
    service field. Don't surface it as a peer.
    """
    body = _valid_hello_body()
    body["service"] = "some-other-thing"
    assert disc._parse_hello_response(body, scheme="http", host="x", port=1) is None


def test_parse_hello_rejects_missing_fields():
    """Even with the right service tag, missing identity fields
    means we can't safely use the responder.
    """
    body = _valid_hello_body()
    del body["fingerprint"]
    assert disc._parse_hello_response(body, scheme="http", host="x", port=1) is None


def test_parse_hello_rejects_bad_fingerprint_format():
    """Fingerprints MUST start with SHA256: — anything else is
    a wire-format mismatch and should be refused.
    """
    body = _valid_hello_body()
    body["fingerprint"] = "MD5:1234"
    assert disc._parse_hello_response(body, scheme="http", host="x", port=1) is None


def test_parse_hello_rejects_non_dict():
    """A responder returning a JSON array shouldn't blow up."""
    assert disc._parse_hello_response([], scheme="http", host="x", port=1) is None


def test_parse_hello_builds_peer_on_valid_input():
    peer = disc._parse_hello_response(
        _valid_hello_body(), scheme="https", host="192.168.1.42", port=6443,
    )
    assert peer is not None
    assert peer.url == "https://192.168.1.42:6443"
    assert peer.addr == "192.168.1.42:6443"
    assert peer.node_id == "abc123"
    assert peer.icon == "🚀"


# ── sweep_subnet with mocked HTTP ─────────────────────────────────


def _patch_httpx(monkeypatch, handler) -> None:
    """Redirect every httpx.AsyncClient construction in discovery.py
    through a MockTransport that runs ``handler``.
    """
    real_client_cls = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    def factory(**kwargs):
        kwargs.pop("transport", None)
        return real_client_cls(transport=transport, **kwargs)

    monkeypatch.setattr(disc.httpx, "AsyncClient", factory)


def _make_hello_handler(
    responders: dict[str, dict | int],
):
    """Build a MockTransport handler that maps `<host>:<port>` to
    either a JSON body (200) or an HTTP status code (anything else).
    Hosts not in the map raise ConnectError (port closed).
    """
    def handler(request: httpx.Request) -> httpx.Response:
        key = f"{request.url.host}:{request.url.port}"
        match = responders.get(key)
        if match is None:
            raise httpx.ConnectError("connection refused", request=request)
        if isinstance(match, int):
            return httpx.Response(match)
        return httpx.Response(200, json=match)
    return handler


@pytest.mark.asyncio
async def test_sweep_finds_responding_host(monkeypatch):
    """One host on the subnet answers /api/fabric/hello correctly;
    we surface it as a candidate peer.
    """
    body = _valid_hello_body()
    handler = _make_hello_handler({
        "192.168.1.42:6443": body,
    })
    _patch_httpx(monkeypatch, handler)

    result = await disc.sweep_subnet(subnet="192.168.1.40/30")
    # /30 yields 2 hosts (192.168.1.41, 192.168.1.42); .42 answers.
    assert len(result.peers) == 1
    assert result.peers[0].host == "192.168.1.42"
    assert result.peers[0].port == 6443


@pytest.mark.asyncio
async def test_sweep_self_filters_by_fingerprint(monkeypatch):
    """If we sweep our own LAN IP, the response carries OUR
    fingerprint — partition it into self_seen, not peers.
    """
    body = _valid_hello_body()
    handler = _make_hello_handler({"192.168.1.42:6443": body})
    _patch_httpx(monkeypatch, handler)

    result = await disc.sweep_subnet(
        subnet="192.168.1.40/30",
        own_fingerprint=body["fingerprint"],
    )
    assert result.peers == []
    assert len(result.self_seen) == 1


@pytest.mark.asyncio
async def test_sweep_already_paired_filtering(monkeypatch):
    """node_id present in known_node_ids -> classify as
    already_paired, not a new candidate.
    """
    body = _valid_hello_body()
    handler = _make_hello_handler({"192.168.1.42:6443": body})
    _patch_httpx(monkeypatch, handler)

    result = await disc.sweep_subnet(
        subnet="192.168.1.40/30",
        known_node_ids={body["node_id"]},
    )
    assert result.peers == []
    assert len(result.already_paired) == 1


@pytest.mark.asyncio
async def test_sweep_skips_responders_with_503(monkeypatch):
    """A peer with fabric_enabled=False returns 503 on /hello.
    That's a real augmentum but it opted out; don't surface as a
    candidate.
    """
    handler = _make_hello_handler({"192.168.1.42:6443": 503})
    _patch_httpx(monkeypatch, handler)

    result = await disc.sweep_subnet(subnet="192.168.1.40/30")
    assert result.peers == []


@pytest.mark.asyncio
async def test_sweep_skips_non_augmentum_responders(monkeypatch):
    """A non-augmentum HTTP service that happens to be on 6443
    (or 6100) won't match the service tag — drop it silently.
    """
    handler = _make_hello_handler({
        "192.168.1.42:6443": {"service": "some-other-thing", "ok": True},
    })
    _patch_httpx(monkeypatch, handler)

    result = await disc.sweep_subnet(subnet="192.168.1.40/30")
    assert result.peers == []


@pytest.mark.asyncio
async def test_sweep_falls_back_from_6443_to_6100(monkeypatch):
    """HTTPS port refused -> still try plain HTTP on 6100."""
    body = _valid_hello_body()
    handler = _make_hello_handler({"192.168.1.42:6100": body})
    _patch_httpx(monkeypatch, handler)

    result = await disc.sweep_subnet(subnet="192.168.1.40/30")
    assert len(result.peers) == 1
    assert result.peers[0].port == 6100
    assert result.peers[0].scheme == "http"


@pytest.mark.asyncio
async def test_sweep_invalid_subnet_returns_error():
    result = await disc.sweep_subnet(subnet="not-a-subnet")
    assert result.peers == []
    assert "_subnet" in result.errors


# ── /api/fabric/hello route ───────────────────────────────────────


async def _make_coord() -> tuple[aiosqlite.Connection, FabricCoordinator]:
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
    identity = await FabricIdentity.from_settings_store(SettingsStore(conn))
    return conn, FabricCoordinator(identity, conn)


def _request_with(coord=None, *, is_admin=True, body=None):
    """Hand-built Request mock matching what the handlers read."""
    req = MagicMock()
    req.scope = {"user": MagicMock(is_admin=is_admin)}
    req.app.state = MagicMock()
    req.app.state.fabric_coordinator = coord

    async def _json():
        if body is None:
            raise ValueError("no body")
        return body
    req.json = _json
    return req


@pytest.mark.asyncio
async def test_hello_503_when_fabric_disabled(monkeypatch):
    from augmentum.proxy import fabric_routes
    monkeypatch.setattr(fabric_routes.settings, "fabric_enabled", False)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as excinfo:
        await fabric_hello(_request_with())
    assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_hello_returns_identity_envelope(monkeypatch):
    """The shape is locked: discovery probes parse exactly these
    fields. Schema drift would break every existing client.
    """
    from augmentum.proxy import fabric_routes
    monkeypatch.setattr(fabric_routes.settings, "fabric_enabled", True)
    monkeypatch.setattr(fabric_routes.settings, "local_fabric_icon", "🚀")
    conn, coord = await _make_coord()
    try:
        result = await fabric_hello(_request_with(coord))
        assert result["service"] == "augmentum-fabric"
        assert result["node_id"] == coord._identity.node_id
        assert result["fingerprint"] == coord._identity.fingerprint
        assert result["public_key"] == coord._identity.public_key_b64
        assert result["fingerprint"].startswith("SHA256:")
        assert result["icon"] == "🚀"
        assert result["role"] == "peer"
        assert "version" in result
        assert "hostname" in result
    finally:
        await conn.close()


# ── POST /api/fabric/discover route ───────────────────────────────


@pytest.mark.asyncio
async def test_discover_admin_only(monkeypatch):
    """Non-admin requests are rejected -- discovery exposes peer
    fingerprints + addresses which we don't want to surface to
    non-admins (even though pairing also requires admin).
    """
    from augmentum.proxy import fabric_routes
    monkeypatch.setattr(fabric_routes.settings, "fabric_enabled", True)
    # The require_admin guard returns a JSONResponse-shaped reject;
    # the route returns it directly. We just confirm it's not a
    # success dict shape with `peers` in it.
    result = await fabric_discover(_request_with(is_admin=False))
    # `require_admin` returns a JSONResponse on rejection -- not a
    # dict with `peers`. Either an HTTPException raised, or a
    # JSONResponse-like return.
    assert not (isinstance(result, dict) and "peers" in result)


@pytest.mark.asyncio
async def test_discover_503_when_fabric_disabled(monkeypatch):
    from augmentum.proxy import fabric_routes
    monkeypatch.setattr(fabric_routes.settings, "fabric_enabled", False)
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as excinfo:
        await fabric_discover(_request_with())
    assert excinfo.value.status_code == 503


@pytest.mark.asyncio
async def test_discover_passes_subnet_through_to_sweep(monkeypatch):
    """The route delegates to discover_fabric_peers; verify the
    subnet arg + clamping behaviour reaches the discovery layer.
    """
    from augmentum.proxy import fabric_routes
    monkeypatch.setattr(fabric_routes.settings, "fabric_enabled", True)
    conn, coord = await _make_coord()

    captured: dict = {}

    async def fake_discover(**kwargs):
        captured.update(kwargs)
        return disc.DiscoveryResult()

    monkeypatch.setattr(fabric_routes, "discover_fabric_peers", fake_discover)

    try:
        result = await fabric_discover(_request_with(
            coord, body={"subnet": "192.168.1.0/24", "timeout_s": 999},
        ))
        assert captured["subnet"] == "192.168.1.0/24"
        # 999 > 60s cap, should clamp.
        assert captured["timeout_s"] == 60.0
        assert captured["own_fingerprint"] == coord._identity.fingerprint
        assert result["ok"] is True
        assert "peers" in result
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_discover_handles_empty_body(monkeypatch):
    """No subnet supplied -> the route should still call discovery
    (which will fall back to common subnets).
    """
    from augmentum.proxy import fabric_routes
    monkeypatch.setattr(fabric_routes.settings, "fabric_enabled", True)
    conn, coord = await _make_coord()

    captured: dict = {}

    async def fake_discover(**kwargs):
        captured.update(kwargs)
        return disc.DiscoveryResult()

    monkeypatch.setattr(fabric_routes, "discover_fabric_peers", fake_discover)

    try:
        result = await fabric_discover(_request_with(coord, body={}))
        assert captured["subnet"] is None
        assert result["ok"] is True
    finally:
        await conn.close()
