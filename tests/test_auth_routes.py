"""Behavior tests for the auth REST endpoints (/api/auth/*).

These exercise the full stack: AuthMiddleware → route handler → real
SessionManager → in-memory SQLite. Replaces the previous 56-line
import-smoke stub that gave false confidence in security-critical code.
"""

from __future__ import annotations

import asyncio
import re

import pytest
from fastapi.testclient import TestClient


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Fixture: fresh app with real SessionManager + in-memory SQLite per test
# ---------------------------------------------------------------------------

@pytest.fixture
def real_auth_app():
    """FastAPI app wired to a real SessionManager + in-memory SQLite.

    No auth mocks — the middleware validates tokens against real sessions
    created by SessionManager. Migrations run on backend.connect() so the
    users/auth_sessions/auth_audit_log tables exist.
    """
    from augmentum.auth.session_manager import SessionManager
    from augmentum.proxy.server import create_app
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.manager import StateManager

    backend = SQLiteBackend(":memory:")
    _run(backend.connect())

    app = create_app()
    app.state.session_manager = SessionManager(backend._conn)
    app.state.state_manager = StateManager(backend)

    yield app

    _run(backend.close())


@pytest.fixture
def unauth_client(real_auth_app):
    """TestClient with no Authorization header — for public/unauth paths."""
    return TestClient(real_auth_app)


# ---------------------------------------------------------------------------
# Seeding + auth helpers
# ---------------------------------------------------------------------------

def _seed_user(app, username="alice", password="supersecret", role="user"):
    sm = app.state.session_manager
    return _run(sm.create_user(username, password, role=role))


def _auth_client(app, *, user=None, username="alice", password="supersecret", role="user"):
    """Create a TestClient authenticated as `user` (seeding one if absent).

    Returns (client, user, token). Token is installed as Bearer header.
    """
    if user is None:
        user = _seed_user(app, username=username, password=password, role=role)
    sm = app.state.session_manager
    token = _run(sm.create_session(user.id))
    tc = TestClient(app)
    tc.headers.update({"Authorization": f"Bearer {token}"})
    return tc, user, token


# ===========================================================================
# GET /api/auth/status  (public)
# ===========================================================================

class TestAuthStatus:
    def test_reports_setup_required_when_zero_users(self, unauth_client):
        r = unauth_client.get("/api/auth/status")
        assert r.status_code == 200
        data = r.json()
        assert data["setup_required"] is True
        assert data["authenticated"] is False
        assert data["user"] is None

    def test_reports_setup_complete_once_a_user_exists(self, real_auth_app, unauth_client):
        _seed_user(real_auth_app)
        r = unauth_client.get("/api/auth/status")
        assert r.json()["setup_required"] is False

    def test_reports_authenticated_with_valid_bearer(self, real_auth_app):
        tc, user, _ = _auth_client(real_auth_app)
        r = tc.get("/api/auth/status")
        assert r.json()["authenticated"] is True
        assert r.json()["user"]["id"] == user.id

    def test_reports_authenticated_with_cookie(self, real_auth_app):
        sm = real_auth_app.state.session_manager
        user = _seed_user(real_auth_app)
        token = _run(sm.create_session(user.id))
        tc = TestClient(real_auth_app)
        tc.cookies.set("augmentum_session", token)
        r = tc.get("/api/auth/status")
        assert r.json()["authenticated"] is True

    def test_invalid_token_reports_unauthenticated(self, real_auth_app, unauth_client):
        _seed_user(real_auth_app)
        unauth_client.headers.update({"Authorization": "Bearer not-a-real-token"})
        r = unauth_client.get("/api/auth/status")
        assert r.json()["authenticated"] is False


# ===========================================================================
# POST /api/auth/setup  (public, first-run only)
# ===========================================================================

class TestAuthSetup:
    def test_creates_admin_first_user_wins(self, real_auth_app, unauth_client):
        body = {"username": "admin", "password": "supersecret"}
        r = unauth_client.post("/api/auth/setup", json=body)
        assert r.status_code == 200
        assert r.json()["user"]["username"] == "admin"
        assert r.json()["user"]["role"] == "admin"
        # Cookie is set on the response
        assert "augmentum_session" in r.cookies

    def test_rejects_when_user_already_exists(self, real_auth_app, unauth_client):
        _seed_user(real_auth_app)
        body = {"username": "admin", "password": "supersecret"}
        r = unauth_client.post("/api/auth/setup", json=body)
        assert r.status_code == 403

    def test_rejects_username_too_short(self, unauth_client):
        body = {"username": "ab", "password": "supersecret"}
        r = unauth_client.post("/api/auth/setup", json=body)
        assert r.status_code == 400

    def test_rejects_username_with_invalid_chars(self, unauth_client):
        body = {"username": "bad user", "password": "supersecret"}
        r = unauth_client.post("/api/auth/setup", json=body)
        assert r.status_code == 400

    def test_rejects_password_too_short(self, unauth_client):
        body = {"username": "admin", "password": "short"}
        r = unauth_client.post("/api/auth/setup", json=body)
        assert r.status_code == 400

    def test_rejects_after_setup_already_complete(self, real_auth_app, unauth_client):
        _seed_user(real_auth_app)
        body = {"token": "test-setup-token", "username": "admin", "password": "supersecret"}
        r = unauth_client.post("/api/auth/setup", json=body)
        assert r.status_code == 403

    def test_setup_token_cleared_after_use(self, real_auth_app, unauth_client):
        body = {"token": "test-setup-token", "username": "admin", "password": "supersecret"}
        unauth_client.post("/api/auth/setup", json=body)
        assert real_auth_app.state.setup_token is None


# ===========================================================================
# POST /api/auth/login  (public)
# ===========================================================================

class TestAuthLogin:
    def test_valid_credentials_returns_session_cookie(self, real_auth_app, unauth_client):
        _seed_user(real_auth_app, "alice", "supersecret")
        r = unauth_client.post("/api/auth/login", json={"username": "alice", "password": "supersecret"})
        assert r.status_code == 200
        assert "augmentum_session" in r.cookies
        assert r.json()["user"]["username"] == "alice"

    def test_wrong_password_rejected(self, real_auth_app, unauth_client):
        _seed_user(real_auth_app, "alice", "supersecret")
        r = unauth_client.post("/api/auth/login", json={"username": "alice", "password": "WRONG"})
        assert r.status_code == 401

    def test_unknown_user_rejected(self, unauth_client):
        r = unauth_client.post("/api/auth/login", json={"username": "nobody", "password": "anything"})
        assert r.status_code == 401

    def test_unknown_user_does_not_leak_timing(self, real_auth_app, unauth_client):
        """Both unknown-user and wrong-password paths call argon2 verify
        (via verify_dummy for unknown) so response timing doesn't leak
        account enumeration. Smoke-check the two error messages are identical.
        """
        _seed_user(real_auth_app, "alice", "supersecret")
        r1 = unauth_client.post("/api/auth/login", json={"username": "alice", "password": "WRONG"}).json()
        r2 = unauth_client.post("/api/auth/login", json={"username": "ghost", "password": "WRONG"}).json()
        assert r1["error"] == r2["error"]

    def test_lockout_after_threshold_failed_attempts(self, real_auth_app, unauth_client):
        """After N failed attempts (configured auth_lockout_threshold=5),
        further attempts get 429 regardless of password correctness. This
        is the brute-force defense — if it silently stops working, password
        spray attacks become viable.
        """
        _seed_user(real_auth_app, "alice", "correctpassword")
        for _ in range(5):
            unauth_client.post("/api/auth/login", json={"username": "alice", "password": "WRONG"})
        r = unauth_client.post("/api/auth/login", json={"username": "alice", "password": "correctpassword"})
        assert r.status_code == 429
        assert "retry_after" in r.json()
        assert r.headers.get("retry-after") is not None

    def test_clear_attempts_after_success(self, real_auth_app, unauth_client):
        """Successful login resets the failure counter so the user isn't
        locked out on their next typo."""
        _seed_user(real_auth_app, "alice", "correctpassword")
        for _ in range(3):
            unauth_client.post("/api/auth/login", json={"username": "alice", "password": "WRONG"})
        # Success resets
        r = unauth_client.post("/api/auth/login", json={"username": "alice", "password": "correctpassword"})
        assert r.status_code == 200
        # Can again fail 3 times without lockout
        for _ in range(3):
            assert unauth_client.post("/api/auth/login", json={"username": "alice", "password": "WRONG"}).status_code == 401

    def test_inactive_user_rejected(self, real_auth_app, unauth_client):
        user = _seed_user(real_auth_app, "alice", "supersecret")
        sm = real_auth_app.state.session_manager
        _run(sm.update_user(user.id, is_active=False))
        r = unauth_client.post("/api/auth/login", json={"username": "alice", "password": "supersecret"})
        assert r.status_code == 403


# ===========================================================================
# POST /api/auth/logout
# ===========================================================================

class TestAuthLogout:
    def test_revokes_session_token(self, real_auth_app):
        tc, _, token = _auth_client(real_auth_app)
        r = tc.post("/api/auth/logout")
        assert r.status_code == 200
        sm = real_auth_app.state.session_manager
        # Token no longer validates — critical for session-invalidation-on-logout
        assert _run(sm.validate_token(token)) is None

    def test_unauthenticated_request_401(self, unauth_client):
        r = unauth_client.post("/api/auth/logout")
        assert r.status_code == 401

    def test_clears_cookie(self, real_auth_app):
        tc, _, _ = _auth_client(real_auth_app)
        r = tc.post("/api/auth/logout")
        # Set-Cookie header should clear augmentum_session
        set_cookie = r.headers.get("set-cookie", "")
        assert "augmentum_session=" in set_cookie
        # Cleared cookies have Max-Age=0 or past expiry
        assert ("max-age=0" in set_cookie.lower() or "expires=" in set_cookie.lower())


# ===========================================================================
# POST /api/auth/ws-ticket
# ===========================================================================

class TestWsTicket:
    def test_issues_ticket_resolvable_to_user(self, real_auth_app):
        tc, user, _ = _auth_client(real_auth_app)
        r = tc.post("/api/auth/ws-ticket")
        assert r.status_code == 200
        ticket = r.json()["ticket"]
        sm = real_auth_app.state.session_manager
        assert sm.validate_ws_ticket(ticket) == user.id

    def test_requires_auth(self, unauth_client):
        r = unauth_client.post("/api/auth/ws-ticket")
        assert r.status_code == 401


# ===========================================================================
# GET /api/auth/me
# ===========================================================================

class TestAuthMe:
    def test_returns_current_user_profile(self, real_auth_app):
        tc, user, _ = _auth_client(real_auth_app)
        r = tc.get("/api/auth/me")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == user.id
        assert data["username"] == user.username
        assert data["role"] == user.role
        assert "password" not in str(data).lower()

    def test_requires_auth(self, unauth_client):
        r = unauth_client.get("/api/auth/me")
        assert r.status_code == 401


# ===========================================================================
# PUT /api/auth/me/password
# ===========================================================================

class TestChangeOwnPassword:
    def test_success_new_password_works_on_login(self, real_auth_app):
        tc, user, _ = _auth_client(real_auth_app)
        r = tc.put(
            "/api/auth/me/password",
            json={"current_password": "supersecret", "new_password": "new_password_xyz"},
        )
        assert r.status_code == 200
        # Independent client can login with new password
        unauth = TestClient(real_auth_app)
        r = unauth.post("/api/auth/login", json={"username": user.username, "password": "new_password_xyz"})
        assert r.status_code == 200

    def test_wrong_current_password_rejected(self, real_auth_app):
        tc, _, _ = _auth_client(real_auth_app)
        r = tc.put(
            "/api/auth/me/password",
            json={"current_password": "WRONG", "new_password": "new_password_xyz"},
        )
        assert r.status_code == 401

    def test_new_password_too_short_rejected(self, real_auth_app):
        tc, _, _ = _auth_client(real_auth_app)
        r = tc.put(
            "/api/auth/me/password",
            json={"current_password": "supersecret", "new_password": "short"},
        )
        assert r.status_code == 400

    def test_revokes_other_sessions_but_keeps_current(self, real_auth_app):
        """Changing password invalidates all OTHER sessions (stolen device
        protection) but keeps the current one so the user isn't kicked out."""
        tc, user, current_token = _auth_client(real_auth_app)
        sm = real_auth_app.state.session_manager
        other_token = _run(sm.create_session(user.id))

        r = tc.put(
            "/api/auth/me/password",
            json={"current_password": "supersecret", "new_password": "new_password_xyz"},
        )
        assert r.status_code == 200
        assert _run(sm.validate_token(current_token)) is not None
        assert _run(sm.validate_token(other_token)) is None


# ===========================================================================
# Admin: GET /api/auth/users
# ===========================================================================

class TestAdminListUsers:
    def test_admin_can_list(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        _seed_user(real_auth_app, "bob", "bobpass12345", role="user")
        tc, _, _ = _auth_client(real_auth_app, user=admin)
        r = tc.get("/api/auth/users")
        assert r.status_code == 200
        names = {u["username"] for u in r.json()["users"]}
        assert {"admin", "bob"}.issubset(names)

    def test_non_admin_gets_403(self, real_auth_app):
        # Seed an admin so we aren't in setup mode
        _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        user = _seed_user(real_auth_app, "rando", "userpass1234", role="user")
        tc, _, _ = _auth_client(real_auth_app, user=user)
        r = tc.get("/api/auth/users")
        assert r.status_code == 403

    def test_unauthenticated_rejected(self, real_auth_app, unauth_client):
        _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        r = unauth_client.get("/api/auth/users")
        assert r.status_code in (401, 403)


# ===========================================================================
# Admin: POST /api/auth/users
# ===========================================================================

class TestAdminCreateUser:
    def test_creates_user_with_201(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        tc, _, _ = _auth_client(real_auth_app, user=admin)
        r = tc.post("/api/auth/users", json={
            "username": "newuser", "password": "newpass12345", "role": "user",
        })
        assert r.status_code == 201
        assert r.json()["user"]["username"] == "newuser"
        assert r.json()["user"]["role"] == "user"

    def test_duplicate_username_conflict(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        _seed_user(real_auth_app, "taken", "whatever12345")
        tc, _, _ = _auth_client(real_auth_app, user=admin)
        r = tc.post("/api/auth/users", json={
            "username": "taken", "password": "newpass12345", "role": "user",
        })
        assert r.status_code == 409

    def test_invalid_role_rejected(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        tc, _, _ = _auth_client(real_auth_app, user=admin)
        r = tc.post("/api/auth/users", json={
            "username": "new", "password": "newpass12345", "role": "superuser",
        })
        assert r.status_code == 400

    def test_short_password_rejected(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        tc, _, _ = _auth_client(real_auth_app, user=admin)
        r = tc.post("/api/auth/users", json={
            "username": "new", "password": "short", "role": "user",
        })
        assert r.status_code == 400

    def test_non_admin_cannot_create(self, real_auth_app):
        _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        user = _seed_user(real_auth_app, "rando", "userpass1234", role="user")
        tc, _, _ = _auth_client(real_auth_app, user=user)
        r = tc.post("/api/auth/users", json={
            "username": "new", "password": "newpass12345", "role": "user",
        })
        assert r.status_code == 403


# ===========================================================================
# Admin: PUT /api/auth/users/{id}
# ===========================================================================

class TestAdminUpdateUser:
    def test_updates_display_name(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        target = _seed_user(real_auth_app, "bob", "bobpass12345")
        tc, _, _ = _auth_client(real_auth_app, user=admin)
        r = tc.put(f"/api/auth/users/{target.id}", json={"display_name": "Bob Smith"})
        assert r.status_code == 200
        sm = real_auth_app.state.session_manager
        updated = _run(sm.get_user_by_id(target.id))
        assert updated.display_name == "Bob Smith"

    def test_admin_cannot_demote_self(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        # Need a second admin so last-admin guard doesn't fire first
        _seed_user(real_auth_app, "admin2", "adminpass12", role="admin")
        tc, _, _ = _auth_client(real_auth_app, user=admin)
        r = tc.put(f"/api/auth/users/{admin.id}", json={"role": "user"})
        assert r.status_code == 400

    def test_admin_cannot_deactivate_self(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        _seed_user(real_auth_app, "admin2", "adminpass12", role="admin")
        tc, _, _ = _auth_client(real_auth_app, user=admin)
        r = tc.put(f"/api/auth/users/{admin.id}", json={"is_active": False})
        assert r.status_code == 400

    def test_cannot_remove_last_admin_via_demotion(self, real_auth_app):
        """System must always have at least one active admin. Promoting a
        second admin first, then demoting the original, should work. But
        demoting the sole admin must be refused.
        """
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        other = _seed_user(real_auth_app, "bob", "bobpass12345", role="user")
        tc, _, _ = _auth_client(real_auth_app, user=admin)
        # There's only one admin — can't demote another admin, but can we demote bob? Bob isn't admin.
        # Real test: try to demote the only admin (not self path since self block fires first).
        # Use `other` as actor to demote `admin`.
        tc_other, _, _ = _auth_client(real_auth_app, user=other)
        # `other` is not admin so this would be 403, not a last-admin test. Instead:
        # Promote other to admin, login as other, demote original admin (last-admin guard should still block since system would drop to 1 admin... no it won't — still 1 admin [other])
        # Simplest path: verify via direct method check.
        sm = real_auth_app.state.session_manager
        # System has 1 admin. Demoting admin would remove last admin.
        would_remove = _run(sm._would_remove_last_admin(admin.id, new_role="user"))
        assert would_remove is True
        # Route-level: try demoting admin via second admin's session
        _run(sm.update_user(other.id, role="admin"))
        tc_other, _, _ = _auth_client(real_auth_app, user=_run(sm.get_user_by_id(other.id)))
        # Now 2 admins. Demote original admin — should succeed.
        r = tc_other.put(f"/api/auth/users/{admin.id}", json={"role": "user"})
        assert r.status_code == 200
        # Now 1 admin (other). Try demoting other — but other is "self" for the tc_other client.
        # Self-block fires before last-admin block in this case. Verify the SessionManager-level
        # guard works on its own:
        would_remove = _run(sm._would_remove_last_admin(other.id, new_role="user"))
        assert would_remove is True

    def test_deactivating_user_revokes_their_sessions(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        target = _seed_user(real_auth_app, "bob", "bobpass12345")
        sm = real_auth_app.state.session_manager
        bob_token = _run(sm.create_session(target.id))
        tc, _, _ = _auth_client(real_auth_app, user=admin)

        r = tc.put(f"/api/auth/users/{target.id}", json={"is_active": False})
        assert r.status_code == 200
        # Bob's session is gone
        assert _run(sm.validate_token(bob_token)) is None


# ===========================================================================
# Admin: PUT /api/auth/users/{id}/password
# ===========================================================================

class TestAdminResetPassword:
    def test_admin_can_reset_and_target_can_login_with_new_password(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        target = _seed_user(real_auth_app, "bob", "oldpass123456")
        tc, _, _ = _auth_client(real_auth_app, user=admin)

        r = tc.put(f"/api/auth/users/{target.id}/password", json={"new_password": "resetpass123"})
        assert r.status_code == 200

        unauth = TestClient(real_auth_app)
        r = unauth.post("/api/auth/login", json={"username": "bob", "password": "resetpass123"})
        assert r.status_code == 200

    def test_reset_revokes_all_target_sessions(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        target = _seed_user(real_auth_app, "bob", "oldpass123456")
        sm = real_auth_app.state.session_manager
        bob_token = _run(sm.create_session(target.id))
        tc, _, _ = _auth_client(real_auth_app, user=admin)

        tc.put(f"/api/auth/users/{target.id}/password", json={"new_password": "resetpass123"})
        assert _run(sm.validate_token(bob_token)) is None

    def test_short_password_rejected(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        target = _seed_user(real_auth_app, "bob", "oldpass123456")
        tc, _, _ = _auth_client(real_auth_app, user=admin)
        r = tc.put(f"/api/auth/users/{target.id}/password", json={"new_password": "short"})
        assert r.status_code == 400

    def test_unknown_target_404(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        tc, _, _ = _auth_client(real_auth_app, user=admin)
        r = tc.put("/api/auth/users/usr_ghostghost/password", json={"new_password": "resetpass123"})
        assert r.status_code == 404


# ===========================================================================
# Admin: DELETE /api/auth/users/{id}
# ===========================================================================

class TestAdminDeleteUser:
    def test_requires_confirm_header(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        target = _seed_user(real_auth_app, "bob", "bobpass12345")
        tc, _, _ = _auth_client(real_auth_app, user=admin)

        r = tc.delete(f"/api/auth/users/{target.id}")
        assert r.status_code == 400

        r = tc.delete(f"/api/auth/users/{target.id}", headers={"X-Confirm-Delete": "true"})
        assert r.status_code == 200

    def test_admin_cannot_delete_self(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        _seed_user(real_auth_app, "admin2", "adminpass12", role="admin")  # second admin so last-admin block isn't the failure
        tc, _, _ = _auth_client(real_auth_app, user=admin)
        r = tc.delete(f"/api/auth/users/{admin.id}", headers={"X-Confirm-Delete": "true"})
        assert r.status_code == 400

    def test_unknown_target_404(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        tc, _, _ = _auth_client(real_auth_app, user=admin)
        r = tc.delete("/api/auth/users/usr_ghostghost", headers={"X-Confirm-Delete": "true"})
        assert r.status_code == 404

    def test_cannot_delete_last_admin(self, real_auth_app):
        """Delete-the-only-admin must be refused. We create two admins so
        the actor can delete the *other* admin down to one, then the final
        deletion attempt has to come from a non-admin surface — which is
        blocked by admin gating. So we verify the guard at the session-
        manager level and via a delete-of-a-third-party path."""
        admin1 = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        admin2 = _seed_user(real_auth_app, "admin2", "adminpass12", role="admin")
        sm = real_auth_app.state.session_manager

        tc, _, _ = _auth_client(real_auth_app, user=admin1)
        # OK — deleting admin2 leaves 1 admin (admin1)
        r = tc.delete(f"/api/auth/users/{admin2.id}", headers={"X-Confirm-Delete": "true"})
        assert r.status_code == 200
        # Now 1 admin. Guard at the store level confirms deletion would remove last admin.
        assert _run(sm._would_remove_last_admin(admin1.id, deleting=True)) is True

    def test_non_admin_cannot_delete(self, real_auth_app):
        _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        user = _seed_user(real_auth_app, "rando", "userpass1234", role="user")
        target = _seed_user(real_auth_app, "bob", "bobpass12345")
        tc, _, _ = _auth_client(real_auth_app, user=user)
        r = tc.delete(f"/api/auth/users/{target.id}", headers={"X-Confirm-Delete": "true"})
        assert r.status_code == 403

    def test_cascades_user_scoped_rows(self, real_auth_app):
        """Every user_id-scoped row must go, not just the user row.

        Only ~10 tables declare ON DELETE CASCADE; the other ~80 use
        plain ``REFERENCES users(id)`` with default NO ACTION. The
        explicit-discover-and-delete path in ``delete_user`` is what
        prevents those tables from leaving orphan rows whose user_id
        points at a non-existent users.id.
        """
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        target = _seed_user(real_auth_app, "victim", "victimpass1234")
        sm = real_auth_app.state.session_manager
        conn = sm._db

        # Cascade table: auth_sessions has ON DELETE CASCADE.
        _run(sm.create_session(target.id))

        # Non-cascade table: ui_sessions got user_id via ALTER (no cascade).
        # If the explicit-delete code path doesn't run, this row survives
        # as an orphan and the final assertion below fails.
        async def _seed_ui_session():
            await conn.execute(
                "INSERT INTO ui_sessions (id, user_id) VALUES (?, ?)",
                ("ses_cascade_test", target.id),
            )
            await conn.commit()
        _run(_seed_ui_session())

        # Pre-state sanity
        async def _count(table: str) -> int:
            async with conn.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE user_id = ?', (target.id,),
            ) as cur:
                row = await cur.fetchone()
                return row[0] if row else 0
        assert _run(_count("auth_sessions")) == 1
        assert _run(_count("ui_sessions")) == 1

        # Delete via the admin route (exercises the full stack).
        tc, _, _ = _auth_client(real_auth_app, user=admin)
        r = tc.delete(
            f"/api/auth/users/{target.id}",
            headers={"X-Confirm-Delete": "true"},
        )
        assert r.status_code == 200

        # User row gone
        async def _user_exists() -> bool:
            async with conn.execute(
                "SELECT 1 FROM users WHERE id = ?", (target.id,),
            ) as cur:
                return (await cur.fetchone()) is not None
        assert _run(_user_exists()) is False

        # Cascade table swept (via FK cascade)
        assert _run(_count("auth_sessions")) == 0
        # Non-cascade table swept (via explicit-delete code path)
        assert _run(_count("ui_sessions")) == 0

        # Strong invariant: no user-scoped table retains ANY row for
        # the deleted user. Mirrors the same discovery query
        # delete_user uses internally.
        async def _orphan_total() -> int:
            async with conn.execute(
                "SELECT m.name FROM sqlite_master m "
                "WHERE m.type = 'table' "
                "  AND m.name NOT LIKE 'sqlite_%' "
                "  AND m.name != 'users' "
                "  AND EXISTS ("
                "      SELECT 1 FROM pragma_table_info(m.name) p "
                "      WHERE p.name = 'user_id'"
                "  )"
            ) as cur:
                tables = [r[0] for r in await cur.fetchall()]
            total = 0
            for t in tables:
                async with conn.execute(
                    f'SELECT COUNT(*) FROM "{t}" WHERE user_id = ?', (target.id,),
                ) as cur:
                    row = await cur.fetchone()
                    total += row[0] if row else 0
            return total
        assert _run(_orphan_total()) == 0, (
            "delete_user must leave no orphans across user-scoped tables"
        )

    def test_cleans_up_on_disk_projects_dir(self, real_auth_app, tmp_path, monkeypatch):
        """The Project entity (Phase 1) stores bare git repos on disk at
        ``{data_dir}/projects/{user_id}/{project_id}.git/``. The DB
        cascade handles the rows; the on-disk dir must be removed
        separately. Spec risk register flags missing this as High."""
        from augmentum.config import settings as cfg

        monkeypatch.setattr(cfg, "data_dir", str(tmp_path))
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        target = _seed_user(real_auth_app, "victim", "victimpass1234")

        # Stage a fake bare-repo dir + a sibling marker so we can verify
        # the user dir was wiped (not just the project subdir).
        from pathlib import Path
        projects_dir = Path(tmp_path) / "projects" / target.id
        repo_dir = projects_dir / "prj_test.git"
        repo_dir.mkdir(parents=True)
        (repo_dir / "HEAD").write_text("ref: refs/heads/main\n")
        (projects_dir / "marker.txt").write_text("present")

        tc, _, _ = _auth_client(real_auth_app, user=admin)
        r = tc.delete(
            f"/api/auth/users/{target.id}",
            headers={"X-Confirm-Delete": "true"},
        )
        assert r.status_code == 200
        assert not projects_dir.exists(), (
            "delete_user must rmtree {data_dir}/projects/{user_id}/"
        )


# ===========================================================================
# Admin: GET /api/auth/audit
# ===========================================================================

class TestAuditLog:
    def test_admin_sees_audit_entries_after_changes(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        tc, _, _ = _auth_client(real_auth_app, user=admin)
        # Generate audit entries by creating a user via API
        r = tc.post("/api/auth/users", json={
            "username": "newuser", "password": "newpass12345", "role": "user",
        })
        assert r.status_code == 201

        r = tc.get("/api/auth/audit")
        assert r.status_code == 200
        entries = r.json()["entries"]
        assert len(entries) >= 1
        # Most recent audit entry should be user_create for "newuser"
        actions = [e.get("action") for e in entries]
        assert "user_create" in actions

    def test_non_admin_cannot_read_audit(self, real_auth_app):
        _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        user = _seed_user(real_auth_app, "rando", "userpass1234", role="user")
        tc, _, _ = _auth_client(real_auth_app, user=user)
        r = tc.get("/api/auth/audit")
        assert r.status_code == 403

    def test_audit_respects_limit(self, real_auth_app):
        admin = _seed_user(real_auth_app, "admin", "adminpass12", role="admin")
        tc, _, _ = _auth_client(real_auth_app, user=admin)
        # Create multiple users to generate audit rows
        for i in range(3):
            tc.post("/api/auth/users", json={
                "username": f"user{i}", "password": "password12345", "role": "user",
            })
        r = tc.get("/api/auth/audit?limit=2")
        assert r.status_code == 200
        assert len(r.json()["entries"]) <= 2


# ===========================================================================
# Module/router sanity (preserve what the old stub covered)
# ===========================================================================

class TestRouterShape:
    def test_router_prefix(self):
        from augmentum.proxy.auth_routes import router
        assert router.prefix == "/api/auth"

    def test_username_regex(self):
        from augmentum.proxy.auth_routes import _USERNAME_RE
        assert _USERNAME_RE.match("valid_user123")
        assert _USERNAME_RE.match("abc")
        assert not _USERNAME_RE.match("ab")
        assert not _USERNAME_RE.match("a" * 33)
        assert not _USERNAME_RE.match("bad user")
        assert not _USERNAME_RE.match("bad@user")

    def test_get_ip_forwarded(self):
        from augmentum.proxy.auth_routes import _get_ip
        from unittest.mock import MagicMock
        req = MagicMock()
        req.headers = {"x-forwarded-for": "1.2.3.4, 5.6.7.8"}
        req.client = None
        assert _get_ip(req) == "1.2.3.4"

    def test_get_ip_direct(self):
        from augmentum.proxy.auth_routes import _get_ip
        from unittest.mock import MagicMock
        req = MagicMock()
        req.headers = {}
        req.client.host = "10.0.0.1"
        assert _get_ip(req) == "10.0.0.1"
