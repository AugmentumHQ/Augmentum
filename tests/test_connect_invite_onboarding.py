"""Onboarding-form additions: live username check + invitee visibility choice.

Covers the two public surfaces the Connect join page now drives:
  * ``GET  /api/auth/invite/{token}/check-username`` — live availability/validity
    (gated behind a live invite token, returns ok/taken/reserved/invalid).
  * ``POST /api/auth/invite/{token}/claim`` with ``discoverable`` — persists the
    invitee's directory-visibility choice (private-to-inviter vs server-wide).

Full stack: AuthMiddleware → route → real SessionManager/SettingsStore → SQLite.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from augmentum.auth.invite_store import create_invite
from augmentum.auth.session_manager import SessionManager
from augmentum.config import settings
from augmentum.proxy.server import create_app
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.manager import StateManager
from augmentum.state.settings_store import SettingsStore


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def app_env():
    """App wired to real SessionManager + SettingsStore + an admin inviter."""
    backend = SQLiteBackend(":memory:")
    _run(backend.connect())

    app = create_app()
    app.state.session_manager = SessionManager(backend._conn)
    app.state.state_manager = StateManager(backend)
    app.state.settings_store = SettingsStore(backend.conn)

    inviter = _run(app.state.session_manager.create_user("operator", "supersecret", role="admin"))
    yield app, backend, inviter
    _run(backend.close())


def _mint(app, inviter, **kw):
    conn = app.state.state_manager.backend.conn
    return _run(create_invite(conn, inviter_user_id=inviter.id, **kw))


@pytest.fixture(autouse=True)
def _reset_handle(monkeypatch):
    monkeypatch.setattr(settings, "connect_instance_handle", "", raising=False)
    monkeypatch.setattr(settings, "augmentum_public_host", "", raising=False)


# --------------------------------------------------------------------------
# Live username check
# --------------------------------------------------------------------------

class TestUsernameCheck:
    def test_available_username_is_ok(self, app_env):
        app, _, inviter = app_env
        inv = _mint(app, inviter)
        tc = TestClient(app)
        r = tc.get(f"/api/auth/invite/{inv['token']}/check-username?u=freshname")
        assert r.status_code == 200
        assert r.json() == {"available": True, "reason": "ok"}

    def test_taken_username_reports_taken(self, app_env):
        app, _, inviter = app_env  # 'operator' already exists
        inv = _mint(app, inviter)
        tc = TestClient(app)
        r = tc.get(f"/api/auth/invite/{inv['token']}/check-username?u=operator")
        assert r.json() == {"available": False, "reason": "taken"}

    def test_reserved_username_reports_reserved(self, app_env):
        app, _, inviter = app_env
        inv = _mint(app, inviter)
        tc = TestClient(app)
        r = tc.get(f"/api/auth/invite/{inv['token']}/check-username?u=admin")
        assert r.json() == {"available": False, "reason": "reserved"}

    def test_malformed_username_reports_invalid(self, app_env):
        app, _, inviter = app_env
        inv = _mint(app, inviter)
        tc = TestClient(app)
        r = tc.get(f"/api/auth/invite/{inv['token']}/check-username?u=no")  # too short
        assert r.json() == {"available": False, "reason": "invalid"}

    def test_bad_token_is_404_not_a_lookup_oracle(self, app_env):
        # Without a live invite, the endpoint must not answer — no enumerating
        # the user table by probing usernames against a junk token.
        app, _, _ = app_env
        tc = TestClient(app)
        r = tc.get("/api/auth/invite/not-a-real-token/check-username?u=operator")
        assert r.status_code == 404

    def test_check_does_not_collide_with_token_preview_route(self, app_env):
        # ``/invite/{token}`` preview still resolves — the static check-username
        # segment didn't swallow the path param.
        app, _, inviter = app_env
        inv = _mint(app, inviter)
        tc = TestClient(app)
        r = tc.get(f"/api/auth/invite/{inv['token']}")
        assert r.status_code == 200
        assert r.json()["invite"]["status"] == "active"


# --------------------------------------------------------------------------
# Invitee visibility choice at claim
# --------------------------------------------------------------------------

class TestVisibilityChoice:
    def _claim(self, app, inviter, *, username, discoverable):
        inv = _mint(app, inviter)
        tc = TestClient(app)
        r = tc.post(
            f"/api/auth/invite/{inv['token']}/claim",
            json={"username": username, "password": "supersecret", "discoverable": discoverable},
        )
        assert r.status_code == 201, r.text
        return r.json()["user"]["id"]

    def test_private_choice_hides_from_directory(self, app_env):
        app, _, inviter = app_env
        uid = self._claim(app, inviter, username="quietuser", discoverable=False)
        val = _run(app.state.settings_store.get_user(uid, "ui.connectDiscoverableSameInstance"))
        assert val == "false"

    def test_public_choice_lists_in_directory(self, app_env):
        app, _, inviter = app_env
        uid = self._claim(app, inviter, username="louduser", discoverable=True)
        val = _run(app.state.settings_store.get_user(uid, "ui.connectDiscoverableSameInstance"))
        assert val == "true"

    def test_external_guest_claim_issues_call_capable_grant(self, app_env):
        # The "Invite someone" comms flow mints an external_guest invite; claiming
        # it yields a scoped guest (role='guest') with a durable grant that allows
        # BOTH text and call out of the box (Matt's accepted behaviour), plus the
        # one-time grant token the join page hands to the installable guest PWA.
        app, _, inviter = app_env
        inv = _mint(app, inviter, kind="external_guest", role="guest")
        tc = TestClient(app)
        r = tc.post(
            f"/api/auth/invite/{inv['token']}/claim",
            json={"username": "guestpal", "password": "supersecret"},
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body.get("guest_grant_token"), "guest claim must return a grant token for the PWA"
        guest_id = body["user"]["id"]

        conn = app.state.state_manager.backend.conn
        role_cur = _run(conn.execute("SELECT role FROM users WHERE id = ?", (guest_id,)))
        assert _run(role_cur.fetchone())[0] == "guest"

        scope_cur = _run(conn.execute(
            "SELECT scopes FROM connect_guest_grants WHERE guest_user_id = ?", (guest_id,),
        ))
        scopes = _run(scope_cur.fetchone())[0]
        assert "text" in scopes and "call" in scopes, scopes

    def test_private_invitee_still_excluded_from_search_but_inviter_keeps_contact(self, app_env):
        # The private invitee is hidden from the same-instance directory query
        # (opt-out), yet the inviter still has them as a mutual contact (reach).
        from augmentum.connect.contact_store import get_contact
        from augmentum.connect.contacts import local_did_for
        from augmentum.proxy.connect_routes import _query_discoverable_same_instance_peers

        app, _, inviter = app_env
        uid = self._claim(app, inviter, username="hidden1", discoverable=False)
        conn = app.state.state_manager.backend.conn

        # Some OTHER member browsing the directory does not see the hidden user.
        other = _run(app.state.session_manager.create_user("browser", "supersecret"))
        rows = _run(_query_discoverable_same_instance_peers(conn, other.id))
        assert uid not in {r[0] for r in rows}

        # But the inviter still reaches them — auto-added as a mutual contact.
        c = _run(get_contact(conn, user_id=inviter.id, peer_did=local_did_for(uid)))
        assert c is not None


class TestGuestSurfaceGate:
    """A role='guest' session is confined to the Connect comms surface."""

    def _guest_client(self, app, inviter):
        # Claim an external_guest invite, then drive the app as that guest via
        # its session cookie (the claim sets it; reuse the returned token).
        inv = _mint(app, inviter, kind="external_guest", role="guest")
        tc = TestClient(app)
        r = tc.post(
            f"/api/auth/invite/{inv['token']}/claim",
            json={"username": "gatedguest", "password": "supersecret"},
        )
        assert r.status_code == 201, r.text
        # Authenticate subsequent calls as the guest.
        guest_id = r.json()["user"]["id"]
        token = _run(app.state.session_manager.create_session(guest_id))
        gc = TestClient(app)
        gc.headers.update({"Authorization": f"Bearer {token}"})
        return gc

    def test_guest_allowed_on_connect_surface(self, app_env):
        app, _, inviter = app_env
        gc = self._guest_client(app, inviter)
        # A Connect comms endpoint is reachable (200/empty, NOT 403).
        r = gc.get("/api/connect/contacts")
        assert r.status_code != 403, r.text

    def test_guest_denied_full_app_apis(self, app_env):
        app, _, inviter = app_env
        gc = self._guest_client(app, inviter)
        for path in ("/api/coder/workspaces", "/api/config/ui", "/api/library/collections"):
            r = gc.get(path)
            assert r.status_code == 403, f"{path} should be 403 for a guest, got {r.status_code}"

    def test_guest_denied_member_directory_and_guest_management(self, app_env):
        app, _, inviter = app_env
        gc = self._guest_client(app, inviter)
        for path in ("/api/connect/directory", "/api/connect/search?q=a", "/api/connect/guests"):
            r = gc.get(path)
            assert r.status_code == 403, f"{path} should be 403 for a guest, got {r.status_code}"

    def test_guest_denied_credential_minting_and_pairing(self, app_env):
        # A guest must not mint a persistent API key or pair a device — both
        # would outlive / widen the revocable comms-only session.
        app, _, inviter = app_env
        gc = self._guest_client(app, inviter)
        assert gc.get("/api/auth/keys").status_code == 403
        assert gc.post("/api/auth/keys", json={"name": "x"}).status_code == 403
        assert gc.post("/api/auth/pair/start", json={}).status_code == 403

    def test_full_member_not_gated(self, app_env):
        # The gate is guest-only — a normal member still reaches the app.
        app, _, _ = app_env
        member = _run(app.state.session_manager.create_user("realmember", "supersecret"))
        token = _run(app.state.session_manager.create_session(member.id))
        mc = TestClient(app)
        mc.headers.update({"Authorization": f"Bearer {token}"})
        assert mc.get("/api/config/ui").status_code != 403


class TestPortalRegister:
    """POST /api/portal/register/{token} — external-guest portal onboarding.

    Regression guard: the register handler compared the invite status against
    "valid", a value ``invite_status`` never emits (it returns active|expired|
    used|revoked), so EVERY register attempt 410'd and portal onboarding was
    silently dead (same class bug as the portal.js frontend check).
    """

    def test_active_invite_registers_pending(self, app_env):
        app, _, inviter = app_env
        inv = _mint(app, inviter, kind="external_guest", role="guest")
        tc = TestClient(app)
        r = tc.post(
            f"/api/portal/register/{inv['token']}",
            json={"username": "portalguest", "password": "supersecret",
                  "device_id": "dev-abc", "device_public_key": ""},
        )
        assert r.status_code == 201, r.text
        assert r.json().get("status") == "pending"
        # A guest user + a pending registration now exist.
        conn = app.state.state_manager.backend.conn
        cur = _run(conn.execute(
            "SELECT role FROM users WHERE username = ?", ("portalguest",)))
        row = _run(cur.fetchone())
        assert row and row[0] == "guest"

    def test_unknown_token_is_410(self, app_env):
        app, _, _ = app_env
        tc = TestClient(app)
        r = tc.post(
            "/api/portal/register/definitely-not-a-real-token",
            json={"username": "nobody", "password": "supersecret"},
        )
        assert r.status_code == 410, r.text

    def test_used_invite_is_410_on_second_register(self, app_env):
        # Default max_uses=1: the first register consumes it, the second 410s
        # (the atomic consume_invite gate, independent of the status pre-check).
        app, _, inviter = app_env
        inv = _mint(app, inviter, kind="external_guest", role="guest")
        tc = TestClient(app)
        first = tc.post(
            f"/api/portal/register/{inv['token']}",
            json={"username": "guestone", "password": "supersecret"},
        )
        assert first.status_code == 201, first.text
        second = tc.post(
            f"/api/portal/register/{inv['token']}",
            json={"username": "guesttwo", "password": "supersecret"},
        )
        assert second.status_code == 410, second.text


class TestThreadFlags:
    """PATCH /api/connect/threads/{id} — pin / mute / archive persistence.

    The DB columns + ``set_thread_flag`` store fn already existed; only the
    route + UI wiring were missing, so these prefs were client-only and reset
    on reload. This exercises the new route end to end.
    """

    def _member_client(self, app):
        member = _run(app.state.session_manager.create_user("threadmember", "supersecret"))
        token = _run(app.state.session_manager.create_session(member.id))
        mc = TestClient(app)
        mc.headers.update({"Authorization": f"Bearer {token}"})
        return member, mc

    def _make_thread(self, app, user_id):
        from augmentum.connect.message_store import get_or_create_thread, new_thread_id
        conn = app.state.state_manager.backend.conn
        tid = new_thread_id()
        _run(get_or_create_thread(
            conn, thread_id=tid, user_id=user_id, peer_did="did:key:zPeerXYZ"))
        return tid

    def test_patch_persists_and_lists(self, app_env, monkeypatch):
        monkeypatch.setattr(settings, "connect_enabled", True, raising=False)
        app, _, _ = app_env
        member, mc = self._member_client(app)
        tid = self._make_thread(app, member.id)

        r = mc.patch(f"/api/connect/threads/{tid}", json={"pinned": True, "muted": True})
        assert r.status_code == 200, r.text
        assert r.json()["updated"] == {"pinned": True, "muted": True}

        # Survives a fresh read (i.e. persisted, not just optimistic UI).
        lr = mc.get("/api/connect/threads")
        assert lr.status_code == 200, lr.text
        t = next(t for t in lr.json()["threads"] if t["thread_id"] == tid)
        assert t["pinned"] is True and t["muted"] is True and t["archived"] is False

    def test_patch_unknown_thread_is_404(self, app_env, monkeypatch):
        monkeypatch.setattr(settings, "connect_enabled", True, raising=False)
        app, _, _ = app_env
        _, mc = self._member_client(app)
        r = mc.patch("/api/connect/threads/no-such-thread", json={"pinned": True})
        assert r.status_code == 404, r.text

    def test_patch_empty_body_is_400(self, app_env, monkeypatch):
        monkeypatch.setattr(settings, "connect_enabled", True, raising=False)
        app, _, _ = app_env
        member, mc = self._member_client(app)
        tid = self._make_thread(app, member.id)
        r = mc.patch(f"/api/connect/threads/{tid}", json={})
        assert r.status_code == 400, r.text

    def test_another_user_cannot_flag_my_thread(self, app_env, monkeypatch):
        # set_thread_flag is scoped by (thread_id, user_id): a different user
        # PATCHing the same thread_id updates zero rows → 404, and my copy is
        # untouched.
        monkeypatch.setattr(settings, "connect_enabled", True, raising=False)
        app, _, _ = app_env
        member, _ = self._member_client(app)
        tid = self._make_thread(app, member.id)

        other = _run(app.state.session_manager.create_user("otheruser", "supersecret"))
        otoken = _run(app.state.session_manager.create_session(other.id))
        oc = TestClient(app)
        oc.headers.update({"Authorization": f"Bearer {otoken}"})
        r = oc.patch(f"/api/connect/threads/{tid}", json={"pinned": True})
        assert r.status_code == 404, r.text
