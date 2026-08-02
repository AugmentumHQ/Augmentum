"""Tests for the Discover media-server provision path (audio-OS slice B1).

Covers:
  - The MEDIA-category catalog entry (Suwayomi) is shaped correctly.
  - The media-server loader emits kind=media_server / install_via=media-server
    cards, and the providers loader SKIPS media-category services.
  - The _install_media_server dispatcher orchestration: provision →
    poll-until-reachable → auto-create the per-user connection, and the
    idempotent re-install path (find_match → update, no duplicate row).

The dispatcher test uses fakes for the service manager, provider client,
and media store so it exercises orchestration logic without Docker, real
encryption, or a live SQLite backend.
"""

from __future__ import annotations

import types

import aiosqlite
import pytest

from augmentum.providers.catalog import ProviderCatalog
from augmentum.providers.models import ServiceCategory

# ── Schema (mirrors marketplace_listings incl. migration-254 columns) ──

_SCHEMA_SQL = """
CREATE TABLE marketplace_listings (
    id                  TEXT PRIMARY KEY,
    publisher           TEXT NOT NULL DEFAULT 'augmentum',
    title               TEXT NOT NULL,
    kind                TEXT NOT NULL,
    runtime_preferred   TEXT NOT NULL DEFAULT '',
    runtime_alternates  TEXT NOT NULL DEFAULT '[]',
    tagline             TEXT NOT NULL DEFAULT '',
    description         TEXT NOT NULL DEFAULT '',
    thumbnail_url       TEXT NOT NULL DEFAULT '',
    source_url          TEXT NOT NULL DEFAULT '',
    embed_url           TEXT NOT NULL DEFAULT '',
    install_via         TEXT NOT NULL,
    install_payload     TEXT NOT NULL DEFAULT '{}',
    capabilities        TEXT NOT NULL DEFAULT '{}',
    metadata            TEXT NOT NULL DEFAULT '{}',
    rating              REAL,
    install_count       INTEGER NOT NULL DEFAULT 0,
    signature           TEXT NOT NULL DEFAULT '',
    listed_at           TEXT NOT NULL DEFAULT (datetime('now')),
    delisted_at         TEXT,
    category            TEXT NOT NULL DEFAULT '',
    tags                TEXT NOT NULL DEFAULT '[]',
    featured            INTEGER NOT NULL DEFAULT 0
);
"""


async def _mkstore():
    from augmentum.marketplace import MarketplaceStore
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_SQL)
    await conn.commit()
    return MarketplaceStore(conn), conn


# ── Catalog shape ──────────────────────────────────────────────────────

def test_suwayomi_is_a_media_category_service():
    cat = ProviderCatalog()
    sd = cat.get("suwayomi")
    assert sd is not None, "suwayomi must exist in the provider catalog"
    assert sd.category == ServiceCategory.MEDIA
    assert sd.internal_port == 4567
    # Provisioned Suwayomi must NOT run open — it's flagged for managed auth.
    assert "managed_auth" in sd.features
    assert "no_auth" not in sd.features


# ── Loaders ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_media_loader_emits_media_server_card():
    from augmentum.marketplace import load_media_servers_into_store
    store, conn = await _mkstore()
    try:
        stats = await load_media_servers_into_store(store)
        assert stats["loaded"] >= 1
        listings = await store.list_for_discover(category="media", limit=50)
        ids = {l.id for l in listings}
        assert "mkt:media:suwayomi" in ids
        suwa = next(l for l in listings if l.id == "mkt:media:suwayomi")
        assert suwa.kind == "media_server"
        assert suwa.install_via == "media-server"
        assert suwa.category == "media"
        assert suwa.publisher == "augmentum-media"
        assert suwa.install_payload.get("service_id") == "suwayomi"
        assert suwa.install_payload.get("provider") == "suwayomi"
        # Post-install setup card metadata: empty-content note + a setup
        # guide URL + the host port for the "open the console" link.
        assert "empty" in suwa.metadata.get("content_note", "").lower()
        assert suwa.metadata.get("setup_guide_url", "").startswith("http")
        assert suwa.capabilities.get("host_port") == 6480
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_providers_loader_skips_media_services():
    from augmentum.marketplace import load_providers_into_store
    store, conn = await _mkstore()
    try:
        await load_providers_into_store(store)
        listings = await store.list_for_discover(limit=200)
        kinds = {l.kind for l in listings}
        assert "provider_service" in kinds, "TTS/STT providers should still load"
        # No media-category service leaks in as a provider listing.
        assert all(l.kind != "media_server" for l in listings)
        assert "mkt:provider:suwayomi" not in {l.id for l in listings}
    finally:
        await conn.close()


# ── Dispatcher orchestration (fakes — no Docker / DB / encryption) ──────

class _FakeMgr:
    def __init__(self, *, service_features=("managed_auth",), port=4567, name="Suwayomi"):
        self.enabled = []
        self.last_volume_overrides = None
        self._features = list(service_features)
        self._port = port
        self._name = name

    def get_definition(self, service_id):
        return types.SimpleNamespace(
            id=service_id, name=self._name, internal_port=self._port,
            features=self._features,
        )

    async def enable_service(self, service_id, *, volume_overrides=None):
        self.enabled.append(service_id)
        self.last_volume_overrides = volume_overrides
        return types.SimpleNamespace(id=f"svc_{service_id}")


class _FakeClient:
    def __init__(self):
        self.login_args = None
        self.wizard_args = None

    async def first_run_setup(self, base_url, user, password):
        self.wizard_args = (base_url, user, password)

    async def login(self, base_url, user, password):
        self.login_args = (base_url, user, password)
        # Mimic the providers: with creds, login returns a usable token.
        return f"tok:{user}:{password}" if (user or password) else ""


class _FakeStore:
    def __init__(self, existing=None):
        self._existing = existing
        self.created = []
        self.updated = []

    async def find_match(self, *, user_id, provider, base_url):
        return self._existing

    async def create(self, *, user_id, provider, name, base_url, access_token):
        self.created.append(
            {"provider": provider, "base_url": base_url, "token": access_token},
        )
        return types.SimpleNamespace(id="ms_new")

    async def update(self, server_id, *, user_id, **kw):
        self.updated.append({"id": server_id, **kw})
        return None


def _fake_request(mgr=None):
    state = types.SimpleNamespace(
        service_manager=mgr if mgr is not None else _FakeMgr(),
        http_client=object(),
    )
    app = types.SimpleNamespace(state=state)
    return types.SimpleNamespace(app=app, scope={})


_TEST_KEY = b"k" * 44  # stable HMAC key for deterministic derive_secret


def _patch_common(monkeypatch, store):
    import augmentum.auth.guards as guards
    import augmentum.marketplace.install_dispatchers as disp
    import augmentum.proxy.media_routes as media_routes
    import augmentum.utils.secrets as secrets
    client = _FakeClient()
    monkeypatch.setattr(guards, "is_admin", lambda request: True)
    monkeypatch.setattr(media_routes, "_provider_client", lambda p, h: client)
    monkeypatch.setattr(disp, "_media_server_store", lambda request: store)
    monkeypatch.setattr(secrets, "_load_or_create_key", lambda: _TEST_KEY)
    return client


@pytest.mark.asyncio
async def test_install_media_server_provisions_and_autoconnects(monkeypatch):
    from augmentum.marketplace.install_dispatchers import _install_media_server
    store = _FakeStore(existing=None)
    client = _patch_common(monkeypatch, store)
    req = _fake_request()

    server_id = await _install_media_server(
        req, {"service_id": "suwayomi", "provider": "suwayomi"}, "u1",
    )

    assert server_id == "ms_new"
    assert req.app.state.service_manager.enabled == ["suwayomi"]
    assert len(store.created) == 1
    assert store.created[0]["provider"] == "suwayomi"
    # Auto-connect points at the container on the shared Docker network.
    assert store.created[0]["base_url"] == "http://augmentum-suwayomi:4567"
    # Managed auth: logged in with the derived credential, NOT empty creds,
    # and the resulting Basic token (not "") is what we persisted.
    assert client.login_args is not None
    _, login_user, login_pass = client.login_args
    assert login_user == "augmentum" and login_pass
    assert store.created[0]["token"].startswith("tok:augmentum:")
    # Reachable → status flipped to ok.
    assert store.updated and store.updated[0]["status"] == "ok"


@pytest.mark.asyncio
async def test_install_media_server_is_idempotent(monkeypatch):
    from augmentum.marketplace.install_dispatchers import _install_media_server
    existing = types.SimpleNamespace(id="ms_existing")
    store = _FakeStore(existing=existing)
    _patch_common(monkeypatch, store)
    req = _fake_request()

    server_id = await _install_media_server(
        req, {"service_id": "suwayomi", "provider": "suwayomi"}, "u1",
    )

    assert server_id == "ms_existing"
    assert store.created == []  # no duplicate row
    assert store.updated and store.updated[0]["id"] == "ms_existing"


@pytest.mark.asyncio
async def test_install_media_server_requires_admin(monkeypatch):
    from fastapi import HTTPException

    import augmentum.auth.guards as guards
    from augmentum.marketplace.install_dispatchers import _install_media_server
    monkeypatch.setattr(guards, "is_admin", lambda request: False)
    req = _fake_request()

    with pytest.raises(HTTPException) as ei:
        await _install_media_server(req, {"service_id": "suwayomi"}, "u1")
    assert ei.value.status_code == 403


# ── Managed auth (derived credential + env injection) ──────────────────

def test_derive_secret_is_deterministic_and_scoped(monkeypatch):
    import augmentum.utils.secrets as secrets
    monkeypatch.setattr(secrets, "_load_or_create_key", lambda: b"k" * 44)
    a = secrets.derive_secret("media-auth:suwayomi")
    b = secrets.derive_secret("media-auth:suwayomi")
    c = secrets.derive_secret("media-auth:other")
    assert a == b           # stable across calls (survives restart)
    assert a != c           # label-scoped
    assert len(a) == 32 and a.isalnum()


def test_managed_auth_env_for_suwayomi_only(monkeypatch):
    import augmentum.utils.secrets as secrets
    monkeypatch.setattr(secrets, "_load_or_create_key", lambda: b"k" * 44)
    from augmentum.providers.catalog import ProviderCatalog
    from augmentum.providers.service_auth import (
        managed_auth_env,
        managed_service_credentials,
        needs_managed_auth,
    )
    cat = ProviderCatalog()
    suwa = cat.get("suwayomi")
    assert needs_managed_auth(suwa)
    env = managed_auth_env(suwa)
    assert env["AUTH_MODE"] == "basic_auth"
    assert env["AUTH_USERNAME"] == "augmentum"
    user, pw = managed_service_credentials("suwayomi")
    assert env["AUTH_PASSWORD"] == pw and pw
    # A normal inference provider (TTS) is NOT managed-auth → no env.
    tts = cat.get("kokoro-server")
    assert tts is not None and managed_auth_env(tts) == {}


def test_build_container_config_merges_auth_env_overrides():
    from augmentum.providers.catalog import ProviderCatalog
    from augmentum.providers.manager import ServiceManager
    sd = ProviderCatalog().get("suwayomi")
    cfg = ServiceManager._build_container_config(
        sd, "augmentum-net",
        {"AUTH_MODE": "basic_auth", "AUTH_USERNAME": "augmentum", "AUTH_PASSWORD": "secret"},
    )
    env = cfg["Env"]
    assert "AUTH_MODE=basic_auth" in env
    assert "AUTH_PASSWORD=secret" in env
    # Overrides are additive over the catalog env (TZ stays).
    assert any(e.startswith("TZ=") for e in env)


# ── Console reverse proxy (auth injection + gating) ────────────────────

class _ConsoleMgr:
    def get_definition(self, service_id):
        return types.SimpleNamespace(
            internal_port=4567,
            category=types.SimpleNamespace(value="media"),
            features=["managed_auth"],
        )


class _CapturingHttp:
    def __init__(self):
        self.calls = []

    async def request(self, method, url, **kw):
        self.calls.append({"method": method, "url": url, **kw})
        return types.SimpleNamespace(
            content=b"<html>ok</html>", status_code=200,
            headers={"content-type": "text/html"},
        )


def _console_app(monkeypatch, fake_http, *, admin=True, mgr=None):
    from fastapi import FastAPI

    import augmentum.proxy.media_console_routes as mc
    import augmentum.utils.secrets as secrets
    monkeypatch.setattr(mc, "is_admin", lambda request: admin)
    monkeypatch.setattr(secrets, "_load_or_create_key", lambda: b"k" * 44)
    app = FastAPI()
    app.include_router(mc.router)
    app.state.service_manager = mgr if mgr is not None else _ConsoleMgr()
    app.state.http_client = fake_http
    return app


def test_console_proxy_injects_auth_and_forwards(monkeypatch):
    from fastapi.testclient import TestClient
    http = _CapturingHttp()
    app = _console_app(monkeypatch, http, admin=True)
    client = TestClient(app)

    r = client.get("/api/media/console/suwayomi/api/graphql?x=1")
    assert r.status_code == 200
    assert r.content == b"<html>ok</html>"
    call = http.calls[0]
    # Forwarded to the container on the internal Docker network.
    assert call["url"] == "http://augmentum-suwayomi:4567/api/graphql"
    # Managed credential injected SERVER-SIDE — never sent to the browser.
    assert call["headers"]["Authorization"].startswith("Basic ")
    assert call["params"] == {"x": "1"}


def test_console_proxy_admin_only(monkeypatch):
    from fastapi.testclient import TestClient
    app = _console_app(monkeypatch, _CapturingHttp(), admin=False)
    client = TestClient(app)
    assert client.get("/api/media/console/suwayomi/x").status_code == 403


def test_console_proxy_unknown_service_404(monkeypatch):
    from fastapi.testclient import TestClient

    class _NoneMgr:
        def get_definition(self, service_id):
            return None

    app = _console_app(monkeypatch, _CapturingHttp(), admin=True, mgr=_NoneMgr())
    client = TestClient(app)
    assert client.get("/api/media/console/nope/x").status_code == 404


# ── Jellyfin (first-run wizard bootstrap) ──────────────────────────────

def test_jellyfin_catalog_entry_is_first_run_wizard():
    cat = ProviderCatalog()
    sd = cat.get("jellyfin")
    assert sd is not None
    assert sd.category == ServiceCategory.MEDIA
    assert sd.internal_port == 8096
    # Account-based, not Basic — bootstraps via the first-run wizard.
    assert "first_run_wizard" in sd.features
    assert "managed_auth" not in sd.features


@pytest.mark.asyncio
async def test_media_loader_includes_jellyfin_with_managed_credentials():
    from augmentum.marketplace import load_media_servers_into_store
    store, conn = await _mkstore()
    try:
        await load_media_servers_into_store(store)
        listings = await store.list_for_discover(category="media", limit=50)
        jelly = next(
            (l for l in listings if l.id == "mkt:media:jellyfin"), None,
        )
        assert jelly is not None and jelly.kind == "media_server"
        # First-run servers still have an Augmentum-managed login → the card
        # should surface credentials.
        assert jelly.capabilities.get("managed_credentials") is True
        assert jelly.capabilities.get("first_run_wizard") is True
    finally:
        await conn.close()


class _FakeJellyHttp:
    """Records Jellyfin /Startup calls; reports wizard state on Public info."""

    def __init__(self, *, completed=False):
        self.calls = []
        self._completed = completed

    async def get(self, url, **kw):
        self.calls.append(("GET", url))
        if url.endswith("/System/Info/Public"):
            return types.SimpleNamespace(
                status_code=200,
                json=lambda: {"StartupWizardCompleted": self._completed},
            )
        return types.SimpleNamespace(status_code=200, json=lambda: {})

    async def post(self, url, **kw):
        self.calls.append(("POST", url, kw.get("json")))
        return types.SimpleNamespace(status_code=204, json=lambda: {})


@pytest.mark.asyncio
async def test_jellyfin_first_run_setup_creates_admin():
    from augmentum.media.providers.jellyfin import JellyfinProvider
    http = _FakeJellyHttp(completed=False)
    jf = JellyfinProvider(http)
    await jf.first_run_setup("http://augmentum-jellyfin:8096", "augmentum", "pw123")

    posts = [c for c in http.calls if c[0] == "POST"]
    urls = [c[1] for c in posts]
    assert any(u.endswith("/Startup/Configuration") for u in urls)
    assert any(u.endswith("/Startup/User") for u in urls)
    assert any(u.endswith("/Startup/Complete") for u in urls)
    # The admin account is created with the managed credential.
    user_post = next(c for c in posts if c[1].endswith("/Startup/User"))
    assert user_post[2] == {"Name": "augmentum", "Password": "pw123"}


@pytest.mark.asyncio
async def test_jellyfin_first_run_setup_is_idempotent():
    from augmentum.media.providers.jellyfin import JellyfinProvider
    http = _FakeJellyHttp(completed=True)  # wizard already done
    jf = JellyfinProvider(http)
    await jf.first_run_setup("http://x", "augmentum", "pw")
    # No wizard mutations when the server is already set up.
    assert not any(c[0] == "POST" for c in http.calls)


@pytest.mark.asyncio
async def test_install_media_server_runs_first_run_wizard(monkeypatch):
    from augmentum.marketplace.install_dispatchers import _install_media_server
    store = _FakeStore(existing=None)
    client = _patch_common(monkeypatch, store)
    req = _fake_request(
        mgr=_FakeMgr(service_features=["first_run_wizard"], port=8096, name="Jellyfin"),
    )

    server_id = await _install_media_server(
        req, {"service_id": "jellyfin", "provider": "jellyfin"}, "u1",
    )

    assert server_id == "ms_new"
    # Wizard bootstrapped with the derived admin credential, THEN logged in.
    assert client.wizard_args is not None
    _, wiz_user, wiz_pass = client.wizard_args
    assert wiz_user == "augmentum" and wiz_pass
    assert client.login_args is not None
    assert store.created[0]["base_url"] == "http://augmentum-jellyfin:8096"
    assert store.created[0]["token"].startswith("tok:augmentum:")


# ── Audiobookshelf + Komga (first-run increments) ──────────────────────

def test_abs_and_komga_catalog_entries():
    cat = ProviderCatalog()
    for sid, port in [("audiobookshelf", 80), ("komga", 25600)]:
        sd = cat.get(sid)
        assert sd is not None, f"{sid} must exist in the catalog"
        assert sd.category == ServiceCategory.MEDIA
        assert sd.internal_port == port
        assert "first_run_wizard" in sd.features


@pytest.mark.asyncio
async def test_all_four_media_servers_load():
    from augmentum.marketplace import load_media_servers_into_store
    store, conn = await _mkstore()
    try:
        await load_media_servers_into_store(store)
        ids = {l.id for l in await store.list_for_discover(category="media", limit=50)}
        assert {
            "mkt:media:suwayomi", "mkt:media:jellyfin",
            "mkt:media:audiobookshelf", "mkt:media:komga",
        } <= ids
    finally:
        await conn.close()


def test_komga_uses_email_username_others_plain(monkeypatch):
    import augmentum.utils.secrets as secrets
    monkeypatch.setattr(secrets, "_load_or_create_key", lambda: b"k" * 44)
    from augmentum.providers.service_auth import managed_service_credentials
    komga_user, _ = managed_service_credentials("komga")
    jelly_user, _ = managed_service_credentials("jellyfin")
    assert "@" in komga_user                # Komga's claim validates an email
    assert jelly_user == "augmentum"        # others keep the plain handle


class _FakeAbsHttp:
    def __init__(self, *, is_init=False):
        self.calls = []
        self._is_init = is_init

    async def get(self, url, **kw):
        self.calls.append(("GET", url))
        if url.endswith("/status"):
            return types.SimpleNamespace(
                status_code=200, json=lambda: {"isInit": self._is_init},
            )
        return types.SimpleNamespace(status_code=200, json=lambda: {})

    async def post(self, url, **kw):
        self.calls.append(("POST", url, kw.get("json")))
        return types.SimpleNamespace(status_code=200, json=lambda: {})


@pytest.mark.asyncio
async def test_abs_first_run_setup_creates_root():
    from augmentum.media.providers.audiobookshelf import AudiobookshelfProvider
    http = _FakeAbsHttp(is_init=False)
    await AudiobookshelfProvider(http).first_run_setup(
        "http://augmentum-audiobookshelf:80", "augmentum", "pw",
    )
    init = next(c for c in http.calls if c[0] == "POST" and c[1].endswith("/init"))
    assert init[2] == {"newRoot": {"username": "augmentum", "password": "pw"}}


@pytest.mark.asyncio
async def test_abs_first_run_setup_idempotent():
    from augmentum.media.providers.audiobookshelf import AudiobookshelfProvider
    http = _FakeAbsHttp(is_init=True)  # already initialized
    await AudiobookshelfProvider(http).first_run_setup("http://x", "u", "p")
    assert not any(c[0] == "POST" for c in http.calls)


class _FakeKomgaHttp:
    def __init__(self, *, claimed=False):
        self.calls = []
        self._claimed = claimed

    async def get(self, url, **kw):
        self.calls.append(("GET", url))
        if url.endswith("/api/v1/claim"):
            return types.SimpleNamespace(
                status_code=200, json=lambda: {"isClaimed": self._claimed},
            )
        return types.SimpleNamespace(status_code=200, json=lambda: {})

    async def post(self, url, **kw):
        self.calls.append(("POST", url, kw.get("headers")))
        return types.SimpleNamespace(status_code=204, json=lambda: {})


@pytest.mark.asyncio
async def test_komga_first_run_setup_claims_with_email_headers():
    from augmentum.media.providers.komga import KomgaProvider
    http = _FakeKomgaHttp(claimed=False)
    await KomgaProvider(http).first_run_setup(
        "http://augmentum-komga:25600", "augmentum@augmentum.local", "pw",
    )
    claim = next(
        c for c in http.calls if c[0] == "POST" and c[1].endswith("/api/v1/claim")
    )
    headers = claim[2]
    assert headers["X-Komga-Email"] == "augmentum@augmentum.local"
    assert headers["X-Komga-Password"] == "pw"


@pytest.mark.asyncio
async def test_komga_first_run_setup_idempotent():
    from augmentum.media.providers.komga import KomgaProvider
    http = _FakeKomgaHttp(claimed=True)  # already claimed
    await KomgaProvider(http).first_run_setup("http://x", "augmentum@augmentum.local", "p")
    assert not any(c[0] == "POST" for c in http.calls)


class _FakeKomgaMeHttp:
    """GET returns 200 only for the configured 'me' path, else 404."""

    def __init__(self, ok_path):
        self.ok_path = ok_path
        self.tried = []

    async def get(self, url, **kw):
        self.tried.append(url)
        ok = self.ok_path is not None and url.endswith(self.ok_path)
        return types.SimpleNamespace(status_code=200 if ok else 404, json=lambda: {})


@pytest.mark.asyncio
async def test_komga_login_prefers_v2_then_falls_back_to_v1():
    """Verified against the bundled OpenAPI 1.24.4 (no /api/v1/users*).

    Current Komga exposes /api/v2/users/me; older servers only v1. Login
    must succeed on both.
    """
    from augmentum.media.providers.komga import KomgaProvider
    http2 = _FakeKomgaMeHttp("/api/v2/users/me")
    tok = await KomgaProvider(http2).login("http://k", "a@b.c", "pw")
    assert tok and http2.tried[0].endswith("/api/v2/users/me")  # v2 tried first

    http1 = _FakeKomgaMeHttp("/api/v1/users/me")  # legacy server
    tok = await KomgaProvider(http1).login("http://k", "a@b.c", "pw")
    assert tok and any(u.endswith("/api/v1/users/me") for u in http1.tried)


@pytest.mark.asyncio
async def test_komga_login_bad_credentials_raises_valueerror():
    from augmentum.media.providers.komga import KomgaProvider

    class _Http401:
        async def get(self, url, **kw):
            return types.SimpleNamespace(status_code=401, json=lambda: {})

    with pytest.raises(ValueError):
        await KomgaProvider(_Http401()).login("http://k", "a@b.c", "bad")


# ── External media library (host bind mounts) ──────────────────────────

def test_looks_like_host_path():
    from augmentum.marketplace.install_dispatchers import _looks_like_host_path
    assert _looks_like_host_path("/mnt/media")
    assert _looks_like_host_path("C:\\Media")
    assert _looks_like_host_path("D:/stuff")
    assert not _looks_like_host_path("relative/path")
    assert not _looks_like_host_path("")
    assert not _looks_like_host_path("   ")


def test_build_container_config_bind_mounts_media_override():
    from augmentum.providers.catalog import ProviderCatalog
    from augmentum.providers.manager import ServiceManager
    sd = ProviderCatalog().get("jellyfin")
    cfg = ServiceManager._build_container_config(
        sd, "augmentum-net", None, {"/media": "/mnt/movies"},
    )
    binds = cfg["HostConfig"]["Binds"]
    assert "/mnt/movies:/media" in binds            # media → host bind
    assert "jellyfin_config:/config" in binds       # config stays a named volume


def test_media_loader_marks_media_path_per_service():
    # jellyfin/abs/komga have an external library mount; suwayomi doesn't
    # (it streams from online sources into its own data volume).
    from augmentum.marketplace.loaders.media_servers import _service_to_listing
    from augmentum.providers.catalog import ProviderCatalog
    cat = ProviderCatalog()
    wants = {"jellyfin": "/media", "audiobookshelf": "/audiobooks", "komga": "/data"}
    for sid, mount in wants.items():
        l = _service_to_listing(cat.get(sid))
        assert l.capabilities["needs_media_path"] is True
        assert l.capabilities["media_mount"] == mount
        assert l.install_payload["media_mount"] == mount
    suwa = _service_to_listing(cat.get("suwayomi"))
    assert suwa.capabilities["needs_media_path"] is False


@pytest.mark.asyncio
async def test_load_persisted_volume_overrides_roundtrip():
    import json as _json

    import aiosqlite

    from augmentum.providers.manager import ServiceManager
    db = await aiosqlite.connect(":memory:")
    await db.execute("CREATE TABLE managed_services (id TEXT PRIMARY KEY, config_json TEXT)")
    await db.execute(
        "INSERT INTO managed_services (id, config_json) VALUES (?, ?)",
        ("jellyfin", _json.dumps({
            "augmentum_env": {}, "volume_overrides": {"/media": "/mnt/movies"},
        })),
    )
    await db.commit()
    mgr = ServiceManager(None, db)
    assert await mgr._load_persisted_volume_overrides("jellyfin") == {"/media": "/mnt/movies"}
    assert await mgr._load_persisted_volume_overrides("missing") == {}
    await db.close()


@pytest.mark.asyncio
async def test_install_media_server_binds_external_host_path(monkeypatch):
    from augmentum.marketplace.install_dispatchers import _install_media_server
    store = _FakeStore(existing=None)
    _patch_common(monkeypatch, store)
    mgr = _FakeMgr(service_features=["first_run_wizard"], port=8096, name="Jellyfin")
    req = _fake_request(mgr=mgr)

    await _install_media_server(req, {
        "service_id": "jellyfin", "provider": "jellyfin", "media_mount": "/media",
        "_install_options": {"media_host_path": "/mnt/movies"},
    }, "u1")

    assert mgr.last_volume_overrides == {"/media": "/mnt/movies"}


@pytest.mark.asyncio
async def test_install_media_server_rejects_relative_host_path(monkeypatch):
    from fastapi import HTTPException

    from augmentum.marketplace.install_dispatchers import _install_media_server
    store = _FakeStore(existing=None)
    _patch_common(monkeypatch, store)
    req = _fake_request(mgr=_FakeMgr(service_features=["first_run_wizard"], port=8096))
    with pytest.raises(HTTPException) as ei:
        await _install_media_server(req, {
            "service_id": "jellyfin", "provider": "jellyfin", "media_mount": "/media",
            "_install_options": {"media_host_path": "relative/oops"},
        }, "u1")
    assert ei.value.status_code == 400


@pytest.mark.asyncio
async def test_install_media_server_blank_path_uses_named_volume(monkeypatch):
    from augmentum.marketplace.install_dispatchers import _install_media_server
    store = _FakeStore(existing=None)
    _patch_common(monkeypatch, store)
    mgr = _FakeMgr(service_features=["managed_auth"], port=4567)
    req = _fake_request(mgr=mgr)
    # Suwayomi: no media_mount → never binds a host path regardless of input.
    await _install_media_server(req, {
        "service_id": "suwayomi", "provider": "suwayomi", "media_mount": "",
        "_install_options": {},
    }, "u1")
    assert mgr.last_volume_overrides is None


# ── Uninstall: mark_uninstalled + teardown registry ────────────────────

_INSTALLS_SCHEMA = """
CREATE TABLE users (
    id TEXT PRIMARY KEY, username TEXT, password_hash TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE marketplace_installs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    listing_id TEXT NOT NULL,
    install_via TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT '',
    resource_id TEXT NOT NULL DEFAULT '',
    installed_at TEXT NOT NULL DEFAULT (datetime('now')),
    uninstalled_at TEXT
);
CREATE UNIQUE INDEX idx_mki_active
    ON marketplace_installs(user_id, listing_id) WHERE uninstalled_at IS NULL;
"""

_LISTING = "mkt:svc:suwayomi"


async def _mk_installs_store():
    from augmentum.marketplace import MarketplaceStore
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_INSTALLS_SCHEMA)
    for uid in ("u1", "u2"):
        await conn.execute(
            "INSERT INTO users(id, username, password_hash) VALUES (?,?,?)",
            (uid, uid, "x"),
        )
    await conn.commit()
    return MarketplaceStore(conn), conn


async def _record(store, uid):
    return await store.record_install(
        user_id=uid, listing_id=_LISTING, install_via="media-server",
        kind="media_server", resource_id=f"ms_{uid}",
    )


@pytest.mark.asyncio
async def test_mark_uninstalled_clears_only_that_user():
    store, conn = await _mk_installs_store()
    try:
        await _record(store, "u1")
        await _record(store, "u2")
        ids = [_LISTING]
        assert await store.installed_listing_ids_for_user("u1", ids) == {_LISTING}
        n = await store.mark_uninstalled(_LISTING, user_id="u1")
        assert n == 1
        # u1 cleared, u2 untouched.
        assert await store.installed_listing_ids_for_user("u1", ids) == set()
        assert await store.installed_listing_ids_for_user("u2", ids) == {_LISTING}
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_mark_uninstalled_install_wide_clears_everyone():
    store, conn = await _mk_installs_store()
    try:
        await _record(store, "u1")
        await _record(store, "u2")
        # Empty user_id = install-wide teardown (shared container stopped).
        n = await store.mark_uninstalled(_LISTING)
        assert n == 2
        ids = [_LISTING]
        assert await store.installed_listing_ids_for_user("u1", ids) == set()
        assert await store.installed_listing_ids_for_user("u2", ids) == set()
        # Active partial-unique index is freed → re-install inserts a fresh row.
        again = await _record(store, "u1")
        assert again, "re-install after uninstall must mint a new active row"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_mark_uninstalled_noop_when_nothing_active():
    store, conn = await _mk_installs_store()
    try:
        assert await store.mark_uninstalled(_LISTING) == 0
        assert await store.mark_uninstalled("") == 0
    finally:
        await conn.close()


def test_uninstall_dispatcher_registry():
    from augmentum.marketplace.install_dispatchers import (
        _uninstall_media_server,
        get_uninstall_dispatcher,
    )
    assert get_uninstall_dispatcher("media-server") is _uninstall_media_server
    # No teardown for resource-less kinds — the route just clears the record.
    assert get_uninstall_dispatcher("community-character") is None
    assert get_uninstall_dispatcher("provider-service") is None
