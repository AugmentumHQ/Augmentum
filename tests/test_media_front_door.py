"""Tests for media-server HTTPS front doors + credential recovery/change.

Covers the additive pieces from the front-door feature:
  - catalog https_port uniqueness + range (a collision would silently
    break the second listener)
  - caddy_front_door snippet templating, atomic write, safety guards,
    idempotent apply
  - per-provider change_password request shapes + returned token
  - store.update_token_for_provider (install-wide token refresh)

Docker + live Caddy are NOT exercised — apply_front_door is driven with
docker=None so it writes the snippet and skips reload.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from augmentum.providers import caddy_front_door as fd
from augmentum.providers.catalog import ProviderCatalog
from augmentum.providers.models import ServiceCategory


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _mock_response(status_code: int, json_body: Any = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json = MagicMock(return_value=json_body if json_body is not None else {})
    return resp


# --- Catalog https_port hygiene -----------------------------------------


class TestCatalogHttpsPorts:
    def test_media_entries_have_unique_in_range_ports(self):
        cat = ProviderCatalog()
        media = cat.list_by_category(ServiceCategory.MEDIA)
        ports = [sd.https_port for sd in media if sd.https_port]
        # Every media server we ship should have a front-door port.
        assert len(ports) == len(media), "every media entry needs an https_port"
        # Unique — a collision means the second Caddy listener never binds.
        assert len(ports) == len(set(ports)), f"duplicate https_port in {ports}"
        # In the range published on the caddy service in compose.yaml.
        for p in ports:
            assert fd.FRONT_DOOR_PORT_MIN <= p <= fd.FRONT_DOOR_PORT_MAX

    def test_non_media_services_have_no_front_door(self):
        cat = ProviderCatalog()
        for sd in cat.list_all():
            if sd.category != ServiceCategory.MEDIA:
                assert not sd.https_port, f"{sd.id} should not have an https_port"


# --- caddy_front_door snippet + safety ----------------------------------


class TestSnippet:
    def _tmp(self):
        fd.SITES_DIR = tempfile.mkdtemp()
        return fd.SITES_DIR

    def test_snippet_text_shape(self):
        text = fd._snippet_text("jellyfin", 6801, 8096)
        assert text.startswith(":6801 {")
        assert "tls /data/cert.pem /data/key.pem" in text
        assert "reverse_proxy augmentum-jellyfin:8096" in text
        assert text.rstrip().endswith("}")

    def test_snippet_has_starting_page_handler(self):
        # Every front-door block serves the service-aware "still starting" page
        # on a 502/503 (container up, not yet answering) instead of a raw Caddy
        # error or the misleading app boot page. The marker header lets the page
        # poll until the real service answers.
        text = fd._snippet_text("jellyfin", 6801, 8096)
        assert "handle_errors 502 503 {" in text
        assert "/service-starting.html" in text
        assert 'X-Augmentum-Service-Starting "1"' in text
        # Present on the gate block too (both blocks get the handler).
        gated = fd._snippet_text("n8n", 6800, 5678, "aug.lan", fd.GATE_MODE_ACCESS)
        assert gated.count("handle_errors 502 503 {") == 2

    def test_write_is_atomic_no_tmp_left(self):
        d = self._tmp()
        path = fd.write_snippet("jellyfin", 6801, 8096)
        assert path.exists()
        # No leftover temp file from the atomic replace.
        leftover = [f for f in os.listdir(d) if f.endswith(".tmp")]
        assert leftover == []
        assert path.read_text(encoding="utf-8") == fd._snippet_text("jellyfin", 6801, 8096)

    def test_remove_snippet(self):
        self._tmp()
        fd.write_snippet("komga", 6803, 25600)
        assert fd.snippet_exists("komga") is True
        assert fd.remove_snippet("komga") is True
        assert fd.snippet_exists("komga") is False
        # Removing a non-existent one is a no-op (False, not an error).
        assert fd.remove_snippet("komga") is False

    def test_unsafe_id_rejected(self):
        self._tmp()
        with pytest.raises(ValueError):
            fd.write_snippet("../evil", 6801, 8096)
        with pytest.raises(ValueError):
            fd.write_snippet("has space", 6801, 8096)

    def test_out_of_range_port_rejected(self):
        self._tmp()
        with pytest.raises(ValueError):
            fd.write_snippet("jellyfin", 9999, 8096)
        assert fd.front_door_port_ok(6800) is True
        assert fd.front_door_port_ok(6810) is False

    def test_apply_without_docker_writes_then_idempotent(self):
        self._tmp()

        async def go():
            # First apply with no docker → snippet written, not "live".
            live = await fd.apply_front_door(None, "jellyfin", 6801, 8096)
            assert live is False
            assert fd.snippet_exists("jellyfin") is True
            # Second apply with identical content → idempotent True (no reload).
            live2 = await fd.apply_front_door(None, "jellyfin", 6801, 8096)
            assert live2 is True
        _run(go())

    def test_apply_rejects_unsafe_id(self):
        self._tmp()

        async def go():
            live = await fd.apply_front_door(None, "../evil", 6801, 8096)
            assert live is False
            assert fd.snippet_exists("../evil") is False
        _run(go())


class TestGateSnippet:
    def test_no_gate_block_without_domain(self):
        text = fd._snippet_text("suwayomi", 6800, 4567, "", fd.GATE_MODE_BASIC)
        assert text.startswith(":6800 {")
        assert "forward_auth" not in text
        assert ".aug.lan" not in text

    def test_no_gate_block_when_mode_off(self):
        # A domain alone no longer triggers a gate block — the mode gates it.
        text = fd._snippet_text("suwayomi", 6800, 4567, "aug.lan", fd.GATE_MODE_OFF)
        assert text.startswith(":6800 {")
        assert "forward_auth" not in text
        assert ".aug.lan" not in text

    def test_basic_gate_block_appended(self):
        text = fd._snippet_text("suwayomi", 6800, 4567, "aug.lan", fd.GATE_MODE_BASIC)
        # Keeps the dedicated-port front door...
        assert ":6800 {" in text
        # ...and adds the gated subdomain block on the published HTTPS port.
        assert "suwayomi.aug.lan:6443 {" in text
        assert "forward_auth augmentum:6100 {" in text
        assert "uri /api/gate/verify?svc=suwayomi" in text
        assert "copy_headers X-Aug-Gate-Authz" in text
        assert "header_up Authorization {http.request.header.X-Aug-Gate-Authz}" in text
        assert "header_up -X-Aug-Gate-Authz" in text
        assert "header_up -Cookie" in text  # session never reaches upstream
        assert "reverse_proxy augmentum-suwayomi:4567" in text
        # Anti-spoof: client-supplied gate/auth headers stripped first, in a
        # route{} so order is enforced (strip → forward_auth → proxy).
        assert "route {" in text
        assert "request_header -X-Aug-Gate-Authz" in text
        assert "request_header -Authorization" in text

    def test_access_gate_block_appended(self):
        # Access mode: forward_auth gates access, then proxies straight through
        # — no credential injection, and the app's OWN cookies pass through so
        # its login persists (nothing stripped, nothing copied).
        text = fd._snippet_text("n8n", 6800, 5678, "aug.lan", fd.GATE_MODE_ACCESS)
        assert ":6800 {" in text                       # raw-port door kept
        assert "n8n.aug.lan:6443 {" in text
        assert "forward_auth augmentum:6100 {" in text
        assert "uri /api/gate/verify?svc=n8n" in text
        assert "reverse_proxy augmentum-n8n:5678" in text
        # NO credential injection / cookie stripping in access mode.
        assert "X-Aug-Gate-Authz" not in text
        assert "header_up Authorization" not in text
        assert "header_up -Cookie" not in text

    def test_unsafe_gate_domain_omits_block(self):
        text = fd._snippet_text("suwayomi", 6800, 4567, "bad domain!/x", fd.GATE_MODE_BASIC)
        assert "forward_auth" not in text  # rejected by _SAFE_DOMAIN

    def test_gate_only_when_port_pool_exhausted(self):
        # https_port=0 (no dedicated port left in 6800-6809) MUST still emit the
        # unbounded gate-subdomain door — otherwise the service falls through
        # Caddy's catch-all to the Augmentum app ("Ollama is running"). This is
        # the whole point of the gate: reachability past the 10-port cap.
        text = fd._snippet_text("n8n", 0, 5678, "aug.lan", fd.GATE_MODE_ACCESS)
        assert "n8n.aug.lan:6443 {" in text          # gate door present
        assert "forward_auth augmentum:6100 {" in text
        assert "reverse_proxy augmentum-n8n:5678" in text
        # ...but NO dedicated-port block (there's no port to bind).
        assert not text.lstrip().startswith(":0 {")
        assert ":0 {" not in text

    def test_no_door_at_all_when_no_port_and_no_gate(self):
        # No port AND no gate domain → nothing to write (write/apply reject it).
        text = fd._snippet_text("n8n", 0, 5678, "", fd.GATE_MODE_OFF)
        assert text == ""
        import pytest
        with pytest.raises(ValueError):
            fd.write_snippet("n8n", 0, 5678, "", fd.GATE_MODE_OFF)

    def test_gate_only_snippet_writes_with_zero_port(self, tmp_path, monkeypatch):
        # A gate domain makes write_snippet accept https_port=0 (gate-only).
        monkeypatch.setattr(fd, "SITES_DIR", str(tmp_path))
        path = fd.write_snippet("n8n", 0, 5678, "aug.lan", fd.GATE_MODE_ACCESS)
        body = path.read_text(encoding="utf-8")
        assert "n8n.aug.lan:6443 {" in body and ":0 {" not in body


# --- /api/gate/verify (forward_auth verdict) ----------------------------


class _FakeUser:
    id = "u_test"
    is_active = True


class _FakeSM:
    def __init__(self, valid_token: str):
        self._valid = valid_token

    async def validate_token(self, token):
        return _FakeUser() if token == self._valid else None


class _FakeServer:
    def __init__(self, provider):
        self.provider = provider


class _FakeStore:
    def __init__(self, providers):
        self._providers = providers

    async def list_visible(self, *, user_id):
        return [_FakeServer(p) for p in self._providers]


def _gate_app(monkeypatch, *, valid_token, visible_providers):
    """Minimal FastAPI app mounting the gate router with mocked state."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from augmentum.providers.models import ServiceCategory, ServiceDefinition
    from augmentum.proxy import gate_routes

    sd = ServiceDefinition(
        id="suwayomi", name="Suwayomi", description="", category=ServiceCategory.MEDIA,
        image="x", internal_port=4567, host_port=6480,
        features=["managed_auth", "self_hosted"],
    )

    class _FakeMgr:
        def get_definition(self, svc):
            return sd if svc == "suwayomi" else None

    monkeypatch.setattr(gate_routes, "_store",
                        lambda req: _FakeStore(visible_providers))
    monkeypatch.setattr(gate_routes, "_db_conn", lambda req: None)  # derived creds
    app = FastAPI()
    app.include_router(gate_routes.router)
    app.state.session_manager = _FakeSM(valid_token)
    app.state.service_manager = _FakeMgr()
    return TestClient(app, raise_server_exceptions=True)


class TestGateVerify:
    def test_no_session_redirects_to_login(self, monkeypatch):
        client = _gate_app(monkeypatch, valid_token="good", visible_providers=["suwayomi"])
        r = client.get("/api/gate/verify?svc=suwayomi", follow_redirects=False)
        assert r.status_code == 302
        # Targets the SPA at /ui/ (public), NOT /login (401s at auth
        # middleware) and NOT bare / (the Ollama-compat stub). See
        # _login_redirect.
        loc = r.headers.get("location", "")
        assert "/ui/?next=" in loc and "/login?next=" not in loc

    def test_valid_session_with_access_injects_basic(self, monkeypatch):
        client = _gate_app(monkeypatch, valid_token="good", visible_providers=["suwayomi"])
        r = client.get(
            "/api/gate/verify?svc=suwayomi",
            headers={"Cookie": "augmentum_session=good"},
            follow_redirects=False,
        )
        assert r.status_code == 200
        assert r.headers.get("X-Aug-Gate-Authz", "").startswith("Basic ")

    def test_valid_session_without_access_403(self, monkeypatch):
        # User is authed but has no Suwayomi connection → denied.
        client = _gate_app(monkeypatch, valid_token="good", visible_providers=["jellyfin"])
        r = client.get(
            "/api/gate/verify?svc=suwayomi",
            headers={"Cookie": "augmentum_session=good"},
            follow_redirects=False,
        )
        assert r.status_code == 403


# --- session cookie domain scoping (don't break bare-IP login) ----------


class TestCookieDomain:
    def test_none_when_no_gate(self, monkeypatch):
        from augmentum.config import settings
        from augmentum.proxy import auth_routes
        monkeypatch.setattr(settings, "gate_domain", "")
        assert auth_routes._cookie_domain(None) is None

    def test_widen_on_gate_host(self, monkeypatch):
        from augmentum.config import settings
        from augmentum.proxy import auth_routes
        monkeypatch.setattr(settings, "gate_domain", "aug.lan")
        apex = types.SimpleNamespace(headers={"host": "aug.lan"})
        sub = types.SimpleNamespace(headers={"host": "suwayomi.aug.lan:6443"})
        assert auth_routes._cookie_domain(apex) == "aug.lan"
        assert auth_routes._cookie_domain(sub) == "aug.lan"

    def test_host_only_on_bare_ip(self, monkeypatch):
        # The whole point: logging in via the IP must NOT widen the cookie,
        # so bare-IP access keeps working when a gate domain is configured.
        from augmentum.config import settings
        from augmentum.proxy import auth_routes
        monkeypatch.setattr(settings, "gate_domain", "aug.lan")
        ip = types.SimpleNamespace(headers={"host": "192.168.1.42:6443"})
        assert auth_routes._cookie_domain(ip) is None

    def test_forwarded_host_respected(self, monkeypatch):
        from augmentum.config import settings
        from augmentum.proxy import auth_routes
        monkeypatch.setattr(settings, "gate_domain", "aug.lan")
        req = types.SimpleNamespace(headers={
            "x-forwarded-host": "jellyfin.aug.lan", "host": "augmentum:6100",
        })
        assert auth_routes._cookie_domain(req) == "aug.lan"


# --- Per-provider change_password ---------------------------------------


class TestChangePassword:
    def test_jellyfin_change_password(self):
        from augmentum.media.providers.jellyfin import JellyfinProvider

        async def go():
            http = MagicMock()
            http.post = AsyncMock(side_effect=[
                _mock_response(200, {"AccessToken": "tok1", "User": {"Id": "u1"}}),
                _mock_response(204),  # /Users/u1/Password
                _mock_response(200, {"AccessToken": "tok2", "User": {"Id": "u1"}}),
            ])
            p = JellyfinProvider(http)
            new_token = await p.change_password(
                "http://jf:8096", "augmentum", "oldpass", "newpass1",
            )
            assert new_token == "tok2"
            pw_call = http.post.call_args_list[1]
            assert "/Users/u1/Password" in pw_call.args[0]
            assert pw_call.kwargs["json"] == {"CurrentPw": "oldpass", "NewPw": "newpass1"}
        _run(go())

    def test_audiobookshelf_change_password(self):
        from augmentum.media.providers.audiobookshelf import AudiobookshelfProvider

        async def go():
            http = MagicMock()
            http.post = AsyncMock(side_effect=[
                _mock_response(200, {"user": {"token": "t1"}}),
                _mock_response(200, {"user": {"token": "t2"}}),
            ])
            http.patch = AsyncMock(return_value=_mock_response(200))
            p = AudiobookshelfProvider(http)
            new_token = await p.change_password(
                "http://abs:13378", "augmentum", "oldpass", "newpass1",
            )
            assert new_token == "t2"
            call = http.patch.call_args
            assert "/api/me/password" in call.args[0]
            assert call.kwargs["json"] == {"password": "oldpass", "newPassword": "newpass1"}
            assert call.kwargs["headers"]["Authorization"] == "Bearer t1"
        _run(go())

    def test_komga_change_password_recomputes_basic_token(self):
        from augmentum.media.providers.komga import KomgaProvider, _encode_basic

        async def go():
            http = MagicMock()
            http.patch = AsyncMock(return_value=_mock_response(204))
            p = KomgaProvider(http)
            new_token = await p.change_password(
                "http://komga:25600", "augmentum@augmentum.local", "oldpass", "newpass1",
            )
            assert new_token == _encode_basic("augmentum@augmentum.local", "newpass1")
            call = http.patch.call_args
            assert "/users/me/password" in call.args[0]
            assert call.kwargs["json"] == {"password": "newpass1"}
            # Authenticated by the CURRENT credential.
            assert call.kwargs["headers"]["Authorization"] == \
                f"Basic {_encode_basic('augmentum@augmentum.local', 'oldpass')}"
        _run(go())

    def test_komga_change_password_bad_current_raises(self):
        from augmentum.media.providers.komga import KomgaProvider

        async def go():
            http = MagicMock()
            http.patch = AsyncMock(return_value=_mock_response(401))
            p = KomgaProvider(http)
            with pytest.raises(ValueError):
                await p.change_password("http://komga:25600", "u", "old", "newpass1")
        _run(go())


# --- store.update_token_for_provider ------------------------------------


_SCHEMA = """
CREATE TABLE users (id TEXT PRIMARY KEY);
CREATE TABLE user_media_servers (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    access_token TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'untested',
    status_detail TEXT NOT NULL DEFAULT '',
    last_sync_at TEXT,
    item_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    total_seen INTEGER NOT NULL DEFAULT 0,
    skipped_count INTEGER NOT NULL DEFAULT 0,
    scope TEXT NOT NULL DEFAULT 'private',
    last_sync_skipped TEXT NOT NULL DEFAULT '[]'
);
INSERT INTO users (id) VALUES ('u_a'), ('u_b');
"""


_MS_SCHEMA = """
CREATE TABLE managed_services (
    id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
INSERT INTO managed_services (id, config_json)
VALUES ('jellyfin', '{"augmentum_env": {}, "volume_overrides": {}}');
"""


class TestCredentialOverride:
    def test_resolve_falls_back_to_derived_without_override(self):
        from augmentum.providers.service_auth import (
            managed_service_credentials,
            resolve_managed_credentials,
        )

        async def go():
            conn = await aiosqlite.connect(":memory:")
            await conn.executescript(_MS_SCHEMA)
            user, derived = managed_service_credentials("jellyfin")
            ruser, rpw = await resolve_managed_credentials("jellyfin", conn)
            assert ruser == user
            assert rpw == derived  # no override set → derived value
            await conn.close()
        _run(go())

    def test_set_then_resolve_returns_override(self):
        from augmentum.providers.service_auth import (
            resolve_managed_credentials,
            set_credential_override,
        )

        async def go():
            conn = await aiosqlite.connect(":memory:")
            await conn.executescript(_MS_SCHEMA)
            ok = await set_credential_override("jellyfin", "my-memorable-pw", conn)
            assert ok is True
            _, rpw = await resolve_managed_credentials("jellyfin", conn)
            assert rpw == "my-memorable-pw"
            # Other config keys are preserved (not clobbered by the merge).
            cur = await conn.execute(
                "SELECT config_json FROM managed_services WHERE id='jellyfin'")
            data = json.loads((await cur.fetchone())[0])
            assert "augmentum_env" in data and "volume_overrides" in data
            await conn.close()
        _run(go())

    def test_set_override_no_row_returns_false(self):
        from augmentum.providers.service_auth import set_credential_override

        async def go():
            conn = await aiosqlite.connect(":memory:")
            await conn.executescript(_MS_SCHEMA)
            ok = await set_credential_override("nonexistent", "pw", conn)
            assert ok is False
            await conn.close()
        _run(go())


class TestUpdateTokenForProvider:
    def test_refreshes_all_rows_for_provider(self):
        from augmentum.media.store import MediaServerStore

        async def go():
            conn = await aiosqlite.connect(":memory:")
            await conn.executescript(_SCHEMA)
            store = MediaServerStore(conn)
            a = await store.create(
                user_id="u_a", provider="jellyfin", name="J",
                base_url="http://augmentum-jellyfin:8096", access_token="old",
            )
            b = await store.create(
                user_id="u_b", provider="jellyfin", name="J",
                base_url="http://augmentum-jellyfin:8096", access_token="old",
            )
            # A different provider must NOT be touched.
            c = await store.create(
                user_id="u_a", provider="komga", name="K",
                base_url="http://augmentum-komga:25600", access_token="keepme",
            )
            n = await store.update_token_for_provider("jellyfin", "fresh")
            assert n == 2
            ra = await store.get(a.id, user_id="u_a")
            rb = await store.get(b.id, user_id="u_b")
            rc = await store.get(c.id, user_id="u_a")
            assert ra.access_token == "fresh"
            assert rb.access_token == "fresh"
            assert rc.access_token == "keepme"
            await conn.close()
        _run(go())

    def test_base_url_scope_spares_external_rows(self):
        # The managed-credential refresh must not clobber the token on a
        # manually-connected EXTERNAL server of the same provider — it has
        # its own, unrelated credentials.
        from augmentum.media.store import MediaServerStore

        async def go():
            conn = await aiosqlite.connect(":memory:")
            await conn.executescript(_SCHEMA)
            store = MediaServerStore(conn)
            managed = await store.create(
                user_id="u_a", provider="jellyfin", name="J (managed)",
                base_url="http://augmentum-jellyfin:8096", access_token="old",
            )
            external = await store.create(
                user_id="u_a", provider="jellyfin", name="NAS Jellyfin",
                base_url="http://192.168.1.99:8096", access_token="nas-token",
            )
            n = await store.update_token_for_provider(
                "jellyfin", "fresh", base_url="http://augmentum-jellyfin:8096",
            )
            assert n == 1
            rm = await store.get(managed.id, user_id="u_a")
            re = await store.get(external.id, user_id="u_a")
            assert rm.access_token == "fresh"
            assert re.access_token == "nas-token"
            await conn.close()
        _run(go())


# --- managed-instance row discrimination ---------------------------------
#
# The class of bug these pin down: "is this server managed?" must be a
# per-ROW fact (does the base_url point at the Augmentum-provisioned
# container?), never a per-provider fact. A manually-connected external
# Jellyfin/ABS must not inherit the managed instance's gate URL, host
# ports, access panel, or credentials.


def _row(provider: str, base_url: str):
    return types.SimpleNamespace(provider=provider, base_url=base_url)


class TestIsManagedInstance:
    def test_container_name_matches(self):
        from augmentum.proxy.media_routes import _is_managed_instance
        assert _is_managed_instance(
            _row("jellyfin", "http://augmentum-jellyfin:8096")) is True

    def test_bare_network_alias_matches(self):
        from augmentum.proxy.media_routes import _is_managed_instance
        assert _is_managed_instance(_row("jellyfin", "http://jellyfin:8096")) is True

    def test_external_lan_host_is_not_managed(self):
        from augmentum.proxy.media_routes import _is_managed_instance
        assert _is_managed_instance(_row("jellyfin", "http://192.168.1.99:8096")) is False
        assert _is_managed_instance(_row("jellyfin", "https://media.example.com")) is False
        assert _is_managed_instance(_row("jellyfin", "http://host.docker.internal:8096")) is False

    def test_other_providers_container_is_not_this_providers_instance(self):
        from augmentum.proxy.media_routes import _is_managed_instance
        assert _is_managed_instance(_row("komga", "http://augmentum-jellyfin:8096")) is False

    def test_garbage_urls_are_not_managed(self):
        from augmentum.proxy.media_routes import _is_managed_instance
        assert _is_managed_instance(_row("jellyfin", "")) is False
        assert _is_managed_instance(_row("", "http://augmentum-:8096")) is False


class TestHealSkipsExternalRows:
    def test_heal_is_noop_for_manual_external_server(self):
        # An external row of a managed-credential provider must never be
        # "healed" with the managed login (or have its token overwritten).
        from augmentum.proxy import media_routes

        sd = types.SimpleNamespace(features=["managed_auth"])

        class _Mgr:
            def get_definition(self, provider):
                return sd

        state = types.SimpleNamespace(service_manager=_Mgr())
        request = types.SimpleNamespace(
            app=types.SimpleNamespace(state=state))
        server = _row("jellyfin", "http://192.168.1.99:8096")

        result = _run(media_routes._heal_managed_token(request, server, "u_a"))
        # Returned unchanged — no login attempt, no store write (we never
        # reached the http/store lookups, which this fake request lacks).
        assert result is server
