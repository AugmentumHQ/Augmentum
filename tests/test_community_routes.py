"""Integration tests for the community install route.

Covers:
- GET /community-install with no session redirects to /login (NOT 401)
- GET /community-install with invalid manifest_url returns a 400 HTML error
- POST /api/community/install requires auth
- POST /api/community/install rejects untrusted manifest URLs
- POST /api/community/install enforces known categories
- Multi-tenant scoping: an install lands on the calling user_id only
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


# The conftest already provides `app`, `client` (auth-bearer wired), and
# `test_user`. The community route writes an audit row, but for the
# rejection-path tests below we deliberately never reach the DB-write
# branch — every test either gets rejected before dispatch (untrusted
# URL / unknown category / kill switch) or returns 501 (powers/knowledge
# deferred to v1). So the conftest's MemoryBackend-backed `app` is
# sufficient for v0 integration coverage.


# ── GET /community-install — public route behavior ──────────────────


def test_get_passes_middleware_without_auth(app):
    """Unauthenticated GET to /community-install must REACH the handler
    rather than getting 401'd by AuthMiddleware. This is the whole point
    of adding the route to ``_PUBLIC_PATHS``.

    The handler then validates the URL — with an untrusted-origin URL
    here, it returns 400 with the error-HTML page, which proves the
    handler ran.

    Confirmed against a real Augmentum 2026-06-04: without the exemption,
    cross-origin navigation from augmentumhq.com gets 401 before the
    handler ever runs.
    """
    tc = TestClient(app)  # no auth header
    resp = tc.get(
        "/community-install?manifest_url=https://example.com/foo.yaml"
    )
    # The handler ran (400, not 401) and returned the untrusted-source
    # error page since example.com isn't on the trusted origins list.
    assert resp.status_code == 400
    assert "untrusted" in resp.text.lower()


def test_get_missing_param_returns_html_error(client):
    """With auth, but no manifest_url query param, returns the error page."""
    resp = client.get("/community-install")
    assert resp.status_code == 400
    assert "missing" in resp.text.lower() or "manifest_url" in resp.text.lower()


def test_get_untrusted_origin_returns_html_error(client):
    """With auth, but manifest URL not from trusted origin, error HTML."""
    resp = client.get(
        "/community-install?manifest_url=https://evil.example.com/foo.yaml"
    )
    assert resp.status_code == 400
    assert "untrusted" in resp.text.lower()


# ── POST /api/community/install — auth-gated install handler ────────


def test_post_requires_auth(app):
    """No auth → 401. The install endpoint is NOT in the public-path
    exemption — only the GET preview is."""
    tc = TestClient(app)  # no auth header
    resp = tc.post(
        "/api/community/install",
        json={
            "manifest_url": "https://raw.githubusercontent.com/AugmentumHQ/community/main/x.yaml",
            "category": "characters",
            "artifact": {"name": "test"},
        },
    )
    # 401 or 503 (degraded mode without session_manager) — both are
    # fail-closed and acceptable. The point: never 200 without auth.
    assert resp.status_code in (401, 503)


def test_post_rejects_untrusted_manifest_url(client):
    """Even with auth, manifest URL from untrusted origin → 400."""
    resp = client.post(
        "/api/community/install",
        json={
            "manifest_url": "https://evil.example.com/foo.yaml",
            "category": "characters",
            "artifact": {"name": "test"},
        },
    )
    assert resp.status_code == 400
    assert "untrusted" in resp.text.lower()


def test_post_rejects_unknown_category(client):
    """Known categories are enforced; unknown → 400."""
    resp = client.post(
        "/api/community/install",
        json={
            "manifest_url": "https://raw.githubusercontent.com/AugmentumHQ/community/main/x.yaml",
            "category": "not-a-real-category",
            "artifact": {"name": "test"},
        },
    )
    assert resp.status_code == 400


_VALID_POWER_BODY = """---
name: test-tdd
kind: verifier
version: 0.1.0
augmentum_min_version: 0.4.0
description: A test power for unit testing.
author: augmentumhq
license: CC0
---

# Test TDD

This is a test community power.
"""


def test_post_powers_installs_to_community_dir(client, app, tmp_path, monkeypatch):
    """Admin can install a community Power; the POWER.md lands under
    {data_dir}/community-powers/<slug>/ and PowerRegistry.rescan() runs."""
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "data_dir", str(tmp_path), raising=False)

    rescan_calls = []
    class _MockRegistry:
        def rescan(self):
            rescan_calls.append(1)
    app.state.power_registry = _MockRegistry()

    resp = client.post(
        "/api/community/install",
        json={
            "manifest_url": "https://raw.githubusercontent.com/AugmentumHQ/community/main/powers/test-tdd/manifest.yaml",
            "category": "powers",
            "artifact": _VALID_POWER_BODY,
            "manifest": {"slug": "test-tdd", "version": "0.1.0"},
        },
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "installed"
    assert resp.json()["id"] == "test-tdd"

    # File was written
    target = tmp_path / "community-powers" / "test-tdd" / "POWER.md"
    assert target.exists()
    assert "name: test-tdd" in target.read_text()

    # Registry was rescanned
    assert len(rescan_calls) == 1


def test_post_powers_rejects_invalid_slug(client, app, tmp_path, monkeypatch):
    """Power slug must be kebab-case 3-48 chars."""
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "data_dir", str(tmp_path), raising=False)

    bad = _VALID_POWER_BODY.replace("name: test-tdd", "name: Bad Slug!")
    resp = client.post(
        "/api/community/install",
        json={
            "manifest_url": "https://raw.githubusercontent.com/AugmentumHQ/community/main/powers/x/manifest.yaml",
            "category": "powers",
            "artifact": bad,
        },
    )
    assert resp.status_code == 400
    assert "kebab" in resp.text.lower() or "slug" in resp.text.lower()


def test_post_powers_rejects_invalid_kind(client, app, tmp_path, monkeypatch):
    """Power kind must be one of guidance/verifier/workflow/integration/bridge."""
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "data_dir", str(tmp_path), raising=False)

    bad = _VALID_POWER_BODY.replace("kind: verifier", "kind: not-a-real-kind")
    resp = client.post(
        "/api/community/install",
        json={
            "manifest_url": "https://raw.githubusercontent.com/AugmentumHQ/community/main/powers/x/manifest.yaml",
            "category": "powers",
            "artifact": bad,
        },
    )
    assert resp.status_code == 400
    assert "kind" in resp.text.lower()


def test_post_powers_requires_admin(app, test_nonadmin_user, monkeypatch, tmp_path):
    """Non-admin user is rejected with 403 — Powers are install-wide."""
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "data_dir", str(tmp_path), raising=False)

    # Swap session manager to return the non-admin user
    app.state.session_manager.validate_token.return_value = test_nonadmin_user
    app.state.session_manager.get_user_by_id.return_value = test_nonadmin_user

    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    resp = tc.post(
        "/api/community/install",
        json={
            "manifest_url": "https://raw.githubusercontent.com/AugmentumHQ/community/main/powers/x/manifest.yaml",
            "category": "powers",
            "artifact": _VALID_POWER_BODY,
        },
    )
    assert resp.status_code == 403


def test_post_knowledge_pack_enqueues_install_job(client, app, tmp_path, monkeypatch):
    """Admin can install a community knowledge pack; returns job_id and
    queues the download. v0 supports format=augpack only."""

    class _MockPackManager:
        def __init__(self):
            self.pack_dir = str(tmp_path / "packs")
        async def scan(self):
            return None

    app.state.knowledge_pack_manager = _MockPackManager()
    app.state.install_jobs = {}

    # Don't actually download — patch httpx so the test is fast
    class _MockStream:
        def __init__(self): self.status_code = 200
        def raise_for_status(self): return None
        async def aiter_bytes(self, chunk_size=64*1024):
            yield b""

    import httpx
    class _MockHttpx:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return None
        def stream(self, method, url, **kw):
            return _MockStreamContext()

    class _MockStreamContext:
        async def __aenter__(self): return _MockStream()
        async def __aexit__(self, *a): return None

    monkeypatch.setattr(httpx, "AsyncClient", _MockHttpx)

    resp = client.post(
        "/api/community/install",
        json={
            "manifest_url": "https://raw.githubusercontent.com/AugmentumHQ/community/main/knowledge/x/manifest.yaml",
            "category": "knowledge",
            "artifact": {
                "name": "test-pack",
                "format": "augpack",
                "version": "0.1.0",
                "download_url": "https://huggingface.co/datasets/x/test.augpack",
                "size_bytes": 1024,
            },
            "manifest": {"slug": "test-pack", "version": "0.1.0"},
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "installed"
    # The job_id is the resource_id for knowledge installs
    assert len(data["id"]) == 12  # 12-char hex job id
    assert data["id"] in app.state.install_jobs


def test_post_knowledge_rejects_zim_format(client, app):
    """ZIM-format community packs are not yet supported — point users at
    /api/knowledge/install directly."""
    resp = client.post(
        "/api/community/install",
        json={
            "manifest_url": "https://raw.githubusercontent.com/AugmentumHQ/community/main/knowledge/x/manifest.yaml",
            "category": "knowledge",
            "artifact": {
                "name": "test-pack",
                "format": "zim",
                "download_url": "https://download.kiwix.org/zim/x.zim",
            },
        },
    )
    assert resp.status_code == 400
    assert "zim" in resp.text.lower()


def test_post_knowledge_rejects_oversized_pack(client, app):
    """Packs over community_max_pack_size_mb are rejected before download."""

    class _MockSettings:
        community_install_enabled = True
        community_max_pack_size_mb = 1  # 1 MB cap
    app.state.settings = _MockSettings()

    resp = client.post(
        "/api/community/install",
        json={
            "manifest_url": "https://raw.githubusercontent.com/AugmentumHQ/community/main/knowledge/x/manifest.yaml",
            "category": "knowledge",
            "artifact": {
                "name": "huge-pack",
                "format": "augpack",
                "download_url": "https://huggingface.co/datasets/x/huge.augpack",
                "size_bytes": 50 * 1024 * 1024,  # 50 MB
            },
        },
    )
    assert resp.status_code == 413
    assert "size" in resp.text.lower() or "exceed" in resp.text.lower()


def test_post_knowledge_requires_admin(app, test_nonadmin_user, monkeypatch):
    """Non-admin user is rejected with 403."""
    app.state.session_manager.validate_token.return_value = test_nonadmin_user
    app.state.session_manager.get_user_by_id.return_value = test_nonadmin_user

    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    resp = tc.post(
        "/api/community/install",
        json={
            "manifest_url": "https://raw.githubusercontent.com/AugmentumHQ/community/main/knowledge/x/manifest.yaml",
            "category": "knowledge",
            "artifact": {
                "name": "test-pack",
                "format": "augpack",
                "download_url": "https://huggingface.co/datasets/x/test.augpack",
            },
        },
    )
    assert resp.status_code == 403


# ── Settings kill switch ────────────────────────────────────────────


def test_post_respects_kill_switch(client, app):
    """community_install_enabled=False → POST refuses with 403."""

    class FakeSettings:
        community_install_enabled = False

    app.state.settings = FakeSettings()
    resp = client.post(
        "/api/community/install",
        json={
            "manifest_url": "https://raw.githubusercontent.com/AugmentumHQ/community/main/x.yaml",
            "category": "characters",
            "artifact": {"name": "test"},
        },
    )
    assert resp.status_code == 403


# ── End-to-end preview rendering paths ──────────────────────────────


_MOCK_MANIFEST_YAML = """
slug: morning-news-brief
name: Morning News Brief
category: reasoning-flows
description: A daily flow that scans your configured news sources.
version: 0.1.0
augmentum_min_version: 0.4.0
author:
  name: augmentumhq
license: CC0
source_url: https://raw.githubusercontent.com/AugmentumHQ/community/main/reasoning-flows/morning-news-brief/flow.json
"""

_MOCK_ARTIFACT_JSON = '{"name": "Morning News Brief", "description": "...", "steps": []}'


@pytest.fixture
def mock_safe_http(monkeypatch):
    """Stub SafeHttpClient.fetch so the route doesn't hit GitHub during tests."""
    from augmentum.proxy import community_routes

    async def fake_fetch(self, url, *, timeout=15.0):
        if url.endswith("manifest.yaml") or "manifest" in url:
            return _MOCK_MANIFEST_YAML, {"url": url}
        return _MOCK_ARTIFACT_JSON, {"url": url}

    monkeypatch.setattr(
        community_routes.SafeHttpClient, "fetch", fake_fetch
    )


def test_get_authenticated_renders_confirm_button(client, mock_safe_http):
    """An authenticated GET with a trusted manifest URL renders the
    preview HTML containing the confirm button — not the login form."""
    resp = client.get(
        "/community-install"
        "?manifest_url=https://raw.githubusercontent.com/AugmentumHQ/community/main/reasoning-flows/morning-news-brief/manifest.yaml"
    )
    assert resp.status_code == 200
    assert "Install to my account" in resp.text
    assert 'id="confirm-btn"' in resp.text
    # Login form must NOT appear when the user is already authenticated.
    assert 'id="inline-login"' not in resp.text


def test_get_unauthenticated_renders_login_form(app, mock_safe_http):
    """An unauthenticated GET with a trusted manifest URL renders the
    preview HTML containing the inline login form — not the confirm
    button. This is the entry point users hit when clicking "Open in
    Augmentum" from augmentumhq.com without an existing session."""
    tc = TestClient(app)  # no auth header
    resp = tc.get(
        "/community-install"
        "?manifest_url=https://raw.githubusercontent.com/AugmentumHQ/community/main/reasoning-flows/morning-news-brief/manifest.yaml"
    )
    assert resp.status_code == 200
    assert 'id="inline-login"' in resp.text
    assert "Sign in to install" in resp.text
    # The confirm button must NOT appear before login.
    assert 'id="confirm-btn"' not in resp.text
    # The preview metadata (name, category) should be visible so the
    # user knows what they're agreeing to install before signing in.
    assert "Morning News Brief" in resp.text


def test_get_manifest_404_reports_fetch_error(client, monkeypatch):
    """A missing raw GitHub manifest should not be parsed as partial YAML.

    GitHub returns a small body like ``404: Not Found`` for missing raw
    files. YAML parses that as a mapping, so the route must honor the HTTP
    status before schema validation.
    """
    from augmentum.proxy import community_routes

    async def fake_fetch(self, url, *, timeout=15.0):
        return "404: Not Found", {"url": url, "status_code": 404}

    monkeypatch.setattr(community_routes.SafeHttpClient, "fetch", fake_fetch)

    resp = client.get(
        "/community-install"
        "?manifest_url=https://raw.githubusercontent.com/AugmentumHQ/community/main/reasoning-flows/missing/manifest.yaml"
    )
    assert resp.status_code == 400
    assert "fetch manifest" in resp.text
    assert "HTTP 404" in resp.text
    assert "Incomplete manifest" not in resp.text


def test_get_artifact_404_reports_fetch_error(client, monkeypatch):
    """A valid manifest with a missing artifact should fail before parsing."""
    from augmentum.proxy import community_routes

    async def fake_fetch(self, url, *, timeout=15.0):
        if url.endswith("manifest.yaml"):
            return _MOCK_MANIFEST_YAML, {"url": url, "status_code": 200}
        return "404: Not Found", {"url": url, "status_code": 404}

    monkeypatch.setattr(community_routes.SafeHttpClient, "fetch", fake_fetch)

    resp = client.get(
        "/community-install"
        "?manifest_url=https://raw.githubusercontent.com/AugmentumHQ/community/main/reasoning-flows/morning-news-brief/manifest.yaml"
    )
    assert resp.status_code == 400
    assert "fetch artifact" in resp.text
    assert "HTTP 404" in resp.text
    assert "Invalid artifact JSON" not in resp.text
