"""Behavior tests for /api/powers/* — the Augmentum Powers plugin surface.

Powers are filesystem-discovered capability packs (native under
.augmentum/powers/, compat under .claude/skills/) that can be enabled
and activated per-workspace per-user. The routes expose:

* Discovery (list / get / rescan)
* Per-user enable/disable state
* Per-user per-workspace activation state

These tests use the real PowerRegistry so manifests are read from the
shipped powers on disk (test-author, migration-safety, etc.), and a
real PowerStateStore backed by an in-memory SettingsStore so enable/
activate writes can be round-tripped.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


TEST_USER_ID = "usr_test"


@pytest.fixture
def power_client(app):
    """Client with a real PowerRegistry + SettingsStore.

    The registry reads the project's shipped .augmentum/powers/ directory
    so list_powers() returns real manifests. Test assertions that name
    specific powers are anchored to packs known to ship at the time of
    writing; if those are renamed/removed the test will fail loudly.
    """
    from augmentum.powers.registry import PowerRegistry
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.manager import StateManager
    from augmentum.state.settings_store import SettingsStore

    backend = SQLiteBackend(":memory:")
    _run(backend.connect())
    app.state.state_manager = StateManager(backend)
    app.state.settings_store = SettingsStore(backend._conn)
    app.state.power_registry = PowerRegistry()

    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc, app.state.power_registry, app.state.settings_store
    _run(backend.close())


def _first_power_id(registry) -> str:
    """Return the ID of the first manifest the registry discovered.

    Tests assert against *a* real manifest rather than pinning a specific
    name so they survive power-pack renames."""
    powers = registry.list_powers()
    assert powers, "expected the repo to ship at least one power manifest"
    return powers[0].id


# ===========================================================================
# GET /api/powers  — list + registry-disabled fallback
# ===========================================================================

class TestListPowers:
    def test_lists_registered_powers(self, power_client):
        client, registry, _ = power_client
        r = client.get("/api/powers")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["powers"], list)
        assert len(data["powers"]) >= 1
        # Every list item has the summary shape the UI expects
        for p in data["powers"]:
            assert {"id", "display_name", "status", "enabled", "active", "health"}.issubset(p.keys())

    def test_registry_unavailable_returns_empty(self, app):
        """When the power registry wasn't wired at boot (degraded startup),
        the route must return an empty-but-well-formed payload so the UI
        doesn't crash on `powers.map(...)`."""
        if hasattr(app.state, "power_registry"):
            delattr(app.state, "power_registry")
        tc = TestClient(app)
        tc.headers.update({"Authorization": "Bearer test-token"})
        r = tc.get("/api/powers")
        assert r.status_code == 200
        assert r.json() == {"powers": [], "active": None}

    def test_default_enabled_true(self, power_client):
        """Powers are enabled-by-default. A fresh user must see each power's
        `enabled=true` without first having to POST /enable, otherwise the
        UI would render every power as disabled on first use."""
        client, _, _ = power_client
        r = client.get("/api/powers")
        assert all(p["enabled"] is True for p in r.json()["powers"])

    def test_rescan_failure_still_returns_list_payload(self, power_client):
        class BrokenRescanRegistry:
            def rescan(self):
                raise RuntimeError("boom")

            def list_powers(self):
                return []

        client, _, _ = power_client
        client.app.state.power_registry = BrokenRescanRegistry()
        r = client.get("/api/powers")
        assert r.status_code == 200
        assert r.json() == {"powers": [], "active": None}


# ===========================================================================
# GET /api/powers/{id}
# ===========================================================================

class TestGetPower:
    def test_returns_manifest_detail(self, power_client):
        client, registry, _ = power_client
        pid = _first_power_id(registry)
        r = client.get(f"/api/powers/{pid}")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == pid
        # Detail shape adds body_markdown + manifest_path on top of summary
        assert "body_markdown" in data
        assert "manifest_path" in data

    def test_unknown_returns_404(self, power_client):
        client, _, _ = power_client
        r = client.get("/api/powers/does-not-exist")
        assert r.status_code == 404


# ===========================================================================
# Enable / disable round-trip
# ===========================================================================

class TestEnableDisable:
    def test_disable_then_enable_round_trip(self, power_client):
        client, registry, _ = power_client
        pid = _first_power_id(registry)

        # Disable
        r = client.post(f"/api/powers/{pid}/disable")
        assert r.status_code == 200
        assert r.json()["enabled"] is False
        # Reflected in list
        listed = {p["id"]: p for p in client.get("/api/powers").json()["powers"]}
        assert listed[pid]["enabled"] is False

        # Re-enable
        r = client.post(f"/api/powers/{pid}/enable")
        assert r.status_code == 200
        assert r.json()["enabled"] is True
        listed = {p["id"]: p for p in client.get("/api/powers").json()["powers"]}
        assert listed[pid]["enabled"] is True

    def test_enable_unknown_returns_404(self, power_client):
        client, _, _ = power_client
        r = client.post("/api/powers/ghost/enable")
        assert r.status_code == 404

    def test_disable_unknown_returns_404(self, power_client):
        client, _, _ = power_client
        r = client.post("/api/powers/ghost/disable")
        assert r.status_code == 404


# ===========================================================================
# Activate / clear
# ===========================================================================

class TestActivation:
    def test_activate_sets_active_power(self, power_client):
        client, registry, _ = power_client
        pid = _first_power_id(registry)

        r = client.post(f"/api/powers/{pid}/activate",
                        json={"workspace_id": "ws-1", "reason": "testing"})
        assert r.status_code == 200
        assert r.json()["active"]["power_id"] == pid
        assert r.json()["active"]["workspace_id"] == "ws-1"

        # GET /active returns the same record for the matching workspace
        r = client.get("/api/powers/active?workspace_id=ws-1")
        assert r.json()["active"]["power_id"] == pid

    def test_activate_disabled_power_returns_409(self, power_client):
        """Can't activate what's disabled — protects users from invoking
        a power they explicitly turned off."""
        client, registry, _ = power_client
        pid = _first_power_id(registry)
        client.post(f"/api/powers/{pid}/disable")

        r = client.post(f"/api/powers/{pid}/activate", json={"workspace_id": "ws-1"})
        assert r.status_code == 409

    def test_activate_unknown_returns_404(self, power_client):
        client, _, _ = power_client
        r = client.post("/api/powers/ghost/activate", json={"workspace_id": "ws-1"})
        assert r.status_code == 404

    def test_disabling_active_power_clears_it(self, power_client):
        """Disabling the currently-active power must also clear the
        activation — otherwise the next turn would still try to invoke it
        even though the user just said "turn this off"."""
        client, registry, _ = power_client
        pid = _first_power_id(registry)

        client.post(f"/api/powers/{pid}/activate", json={"workspace_id": "ws-1"})
        assert client.get("/api/powers/active?workspace_id=ws-1").json()["active"]["power_id"] == pid

        client.post(f"/api/powers/{pid}/disable", json={"workspace_id": "ws-1"})
        r = client.get("/api/powers/active?workspace_id=ws-1")
        assert r.json()["active"] is None

    def test_clear_activation_via_dedicated_endpoint(self, power_client):
        client, registry, _ = power_client
        pid = _first_power_id(registry)
        client.post(f"/api/powers/{pid}/activate", json={"workspace_id": "ws-2"})

        r = client.post("/api/powers/clear-activation", json={"workspace_id": "ws-2"})
        assert r.status_code == 200
        assert r.json()["cleared"] is True
        assert client.get("/api/powers/active?workspace_id=ws-2").json()["active"] is None

    def test_activation_scoped_per_workspace(self, power_client):
        """Activating in workspace A must not affect workspace B — powers
        are per-workspace selections, not global."""
        client, registry, _ = power_client
        pid = _first_power_id(registry)

        client.post(f"/api/powers/{pid}/activate", json={"workspace_id": "ws-A"})
        r_b = client.get("/api/powers/active?workspace_id=ws-B").json()
        assert r_b["active"] is None


# ===========================================================================
# POST /api/powers/rescan
# ===========================================================================

class TestRescan:
    def test_rescan_returns_count(self, power_client):
        client, registry, _ = power_client
        r = client.post("/api/powers/rescan")
        assert r.status_code == 200
        assert r.json()["rescanned"] is True
        assert r.json()["count"] == len(registry.list_powers())

    def test_rescan_without_registry_returns_503(self, app):
        if hasattr(app.state, "power_registry"):
            delattr(app.state, "power_registry")
        tc = TestClient(app)
        tc.headers.update({"Authorization": "Bearer test-token"})
        r = tc.post("/api/powers/rescan")
        assert r.status_code == 503


# ===========================================================================
# Router sanity
# ===========================================================================

class TestRouterShape:
    def test_prefix(self):
        from augmentum.proxy.power_routes import router
        assert router.prefix == "/api/powers"

    def test_expected_paths(self):
        from augmentum.proxy.power_routes import router
        paths = {r.path for r in router.routes}
        expected = {
            "/api/powers",
            "/api/powers/active",
            "/api/powers/rescan",
            "/api/powers/clear-activation",
            "/api/powers/{power_id}",
            "/api/powers/{power_id}/activate",
            "/api/powers/{power_id}/enable",
            "/api/powers/{power_id}/disable",
        }
        assert expected.issubset(paths)
