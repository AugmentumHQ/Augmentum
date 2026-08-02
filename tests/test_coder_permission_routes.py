"""Behavior tests for the coder permission-approval endpoints
(/v1/coder/permissions/*).

These are the HTTP surface of the registry that gates approval-required
coder tools. The routes determine whether user A can see/approve/deny
user B's requests, which is the trust boundary — if isolation slips
here, one user could auto-approve destructive shell commands that
another user's agent is waiting on.

Why the paired tests:

* pending isolation — a privacy leak showing another user's tool_input
  (which may contain file paths, shell commands, API tokens)
* approve/deny ownership — an authorization leak; A must not be able
  to unblock B's pending request by guessing its ID
* 404 vs 403 vs 409 split — the UI relies on these codes to render
  "request gone" vs "not yours" vs "already settled"
"""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi.testclient import TestClient

TEST_USER_ID = "usr_test"  # matches conftest.test_user


def _make_pending_request(registry, user_id: str, tool_name: str = "shell_exec",
                          tool_input: dict | None = None) -> str:
    """Manually insert a PermissionRequest with a controllable Future.

    Using ``registry.request()`` would block waiting on the future — fine
    for the round-trip test, too heavy for per-endpoint checks. This
    helper lets us set up a concrete pending state with one call.
    """
    import uuid

    from augmentum.coder.permissions import PermissionRequest

    loop = asyncio.get_event_loop()
    future = loop.create_future()
    req_id = str(uuid.uuid4())
    req = PermissionRequest(
        id=req_id,
        user_id=user_id,
        tool_name=tool_name,
        tool_input=tool_input or {"command": "rm -rf /tmp/x"},
        created_at=time.time(),
        future=future,
    )
    registry._pending[req_id] = req
    return req_id


@pytest.fixture
def perm_client(app):
    """Client with a real PermissionRegistry wired onto app.state.

    The conftest ``app`` authenticates "Bearer test-token" as usr_test.
    """
    from augmentum.coder.permissions import PermissionRegistry

    registry = PermissionRegistry()
    app.state.permission_registry = registry

    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    yield tc, registry


# ===========================================================================
# GET /v1/coder/permissions/pending
# ===========================================================================

class TestListPending:
    def test_registry_disabled_returns_enabled_false(self, app):
        """If permissions weren't wired on boot, the route must still
        respond cleanly so the UI can show "permissions disabled"."""
        if hasattr(app.state, "permission_registry"):
            delattr(app.state, "permission_registry")
        tc = TestClient(app)
        tc.headers.update({"Authorization": "Bearer test-token"})

        r = tc.get("/v1/coder/permissions/pending")
        assert r.status_code == 200
        data = r.json()
        assert data["enabled"] is False
        assert data["pending"] == []

    def test_empty_when_no_requests(self, perm_client):
        client, _ = perm_client
        r = client.get("/v1/coder/permissions/pending")
        assert r.status_code == 200
        data = r.json()
        assert data["enabled"] is True
        assert data["pending"] == []

    def test_returns_only_own_pending(self, perm_client):
        """Isolation: a user must not see another user's pending requests,
        because the tool_input payload can include file paths, shell
        commands, etc. that shouldn't cross tenants."""
        client, registry = perm_client
        mine = _make_pending_request(registry, TEST_USER_ID, "shell_exec",
                                     {"command": "ls /mine"})
        _make_pending_request(registry, "usr_other", "shell_exec",
                              {"command": "ls /SECRET"})

        r = client.get("/v1/coder/permissions/pending")
        pending = r.json()["pending"]
        assert len(pending) == 1
        assert pending[0]["id"] == mine
        # Other user's command must not leak
        dumped = str(pending)
        assert "SECRET" not in dumped

    def test_request_serialization_shape(self, perm_client):
        client, registry = perm_client
        _make_pending_request(registry, TEST_USER_ID, "file_write",
                              {"path": "/workspace/x.py", "content": "..."})
        r = client.get("/v1/coder/permissions/pending")
        pending = r.json()["pending"]
        assert len(pending) == 1
        entry = pending[0]
        assert set(entry.keys()) >= {"id", "tool_name", "tool_input", "created_at", "age_seconds"}
        assert entry["tool_name"] == "file_write"
        assert entry["tool_input"]["path"] == "/workspace/x.py"
        assert entry["age_seconds"] >= 0.0


# ===========================================================================
# POST /v1/coder/permissions/{id}/approve
# ===========================================================================

class TestApprove:
    def test_approve_resolves_pending_request(self, perm_client):
        """Happy path. After approve, the registry has one less pending."""
        client, registry = perm_client
        req_id = _make_pending_request(registry, TEST_USER_ID)
        assert registry.size() == 1

        r = client.post(f"/v1/coder/permissions/{req_id}/approve")
        assert r.status_code == 200
        assert r.json()["status"] == "approved"
        assert r.json()["id"] == req_id

    def test_approve_unknown_returns_404(self, perm_client):
        client, _ = perm_client
        r = client.post("/v1/coder/permissions/does-not-exist/approve")
        assert r.status_code == 404

    def test_approve_not_owner_returns_403(self, perm_client):
        """Authorization boundary: user A must not resolve user B's request.
        A 403 here (not 404) signals it exists but isn't yours — matches
        the UI expectation."""
        client, registry = perm_client
        req_id = _make_pending_request(registry, "usr_other")
        r = client.post(f"/v1/coder/permissions/{req_id}/approve")
        assert r.status_code == 403

    def test_approve_already_resolved_returns_409(self, perm_client):
        """After the first approve, the future is resolved. A second
        approve (e.g. a double-click race) must return 409, not 200, so
        the UI doesn't show success twice and resubmit."""
        client, registry = perm_client
        req_id = _make_pending_request(registry, TEST_USER_ID)
        # Resolve directly so the registry entry remains but is already-done
        registry.resolve(req_id, approved=True)

        r = client.post(f"/v1/coder/permissions/{req_id}/approve")
        assert r.status_code == 409

    def test_approve_when_registry_disabled_returns_400(self, app):
        if hasattr(app.state, "permission_registry"):
            delattr(app.state, "permission_registry")
        tc = TestClient(app)
        tc.headers.update({"Authorization": "Bearer test-token"})
        r = tc.post("/v1/coder/permissions/any-id/approve")
        assert r.status_code == 400


# ===========================================================================
# POST /v1/coder/permissions/{id}/deny
# ===========================================================================

class TestDeny:
    def test_deny_resolves_pending(self, perm_client):
        client, registry = perm_client
        req_id = _make_pending_request(registry, TEST_USER_ID)

        r = client.post(f"/v1/coder/permissions/{req_id}/deny")
        assert r.status_code == 200
        assert r.json()["status"] == "denied"

    def test_deny_unknown_returns_404(self, perm_client):
        client, _ = perm_client
        r = client.post("/v1/coder/permissions/does-not-exist/deny")
        assert r.status_code == 404

    def test_deny_not_owner_returns_403(self, perm_client):
        client, registry = perm_client
        req_id = _make_pending_request(registry, "usr_other")
        r = client.post(f"/v1/coder/permissions/{req_id}/deny")
        assert r.status_code == 403

    def test_deny_already_resolved_returns_409(self, perm_client):
        client, registry = perm_client
        req_id = _make_pending_request(registry, TEST_USER_ID)
        registry.resolve(req_id, approved=True)

        r = client.post(f"/v1/coder/permissions/{req_id}/deny")
        assert r.status_code == 409

    def test_deny_when_registry_disabled_returns_400(self, app):
        if hasattr(app.state, "permission_registry"):
            delattr(app.state, "permission_registry")
        tc = TestClient(app)
        tc.headers.update({"Authorization": "Bearer test-token"})
        r = tc.post("/v1/coder/permissions/any-id/deny")
        assert r.status_code == 400


# ===========================================================================
# End-to-end: awaiting callback receives the approval decision
# ===========================================================================

class TestRoundTrip:
    """These tests verify that the HTTP endpoints actually unblock a
    pending ``registry.request()`` await — i.e. the coder agent's
    callback returns the correct allow/deny value.

    Without these, the other tests only prove the HTTP contract; they
    don't prove the wiring from route → future → callback caller.
    """

    @pytest.mark.asyncio
    async def test_approve_unblocks_awaiting_coroutine_with_true(self, perm_client):
        client, registry = perm_client

        # Start the permission request in the background. It'll suspend on
        # the future until the HTTP approve fires.
        task = asyncio.create_task(
            registry.request(
                user_id=TEST_USER_ID, tool_name="shell_exec",
                tool_input={"command": "ls"}, timeout=5.0,
            )
        )
        # Wait briefly so the request is actually registered
        await asyncio.sleep(0.05)
        assert registry.size() == 1
        req_id = next(iter(registry._pending))

        # Approve via HTTP
        r = client.post(f"/v1/coder/permissions/{req_id}/approve")
        assert r.status_code == 200

        result = await asyncio.wait_for(task, timeout=2.0)
        assert result is True

    @pytest.mark.asyncio
    async def test_deny_unblocks_awaiting_coroutine_with_false(self, perm_client):
        client, registry = perm_client

        task = asyncio.create_task(
            registry.request(
                user_id=TEST_USER_ID, tool_name="shell_exec",
                tool_input={"command": "rm -rf /"}, timeout=5.0,
            )
        )
        await asyncio.sleep(0.05)
        req_id = next(iter(registry._pending))

        r = client.post(f"/v1/coder/permissions/{req_id}/deny")
        assert r.status_code == 200

        result = await asyncio.wait_for(task, timeout=2.0)
        assert result is False


# ===========================================================================
# Router sanity
# ===========================================================================

class TestRouterShape:
    def test_prefix(self):
        from augmentum.proxy.coder_permission_routes import router
        assert router.prefix == "/v1/coder/permissions"

    def test_registered_paths(self):
        from augmentum.proxy.coder_permission_routes import router
        paths = {r.path for r in router.routes}
        assert "/v1/coder/permissions/pending" in paths
        assert "/v1/coder/permissions/{request_id}/approve" in paths
        assert "/v1/coder/permissions/{request_id}/deny" in paths
