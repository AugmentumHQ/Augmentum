"""HTTP routes for the Connect substrate.

Mirrors test_notifications_routes.py: uses the conftest ``sqlite_client``
fixture so migrations are applied automatically. The Connect feature
flag is forced on per-test (default off, so each test owns its own
visibility window).
"""

from __future__ import annotations

import pytest

from augmentum.config import settings


@pytest.fixture(autouse=True)
def _enable_connect():
    """Force connect_enabled=True for the duration of each test."""
    orig = settings.connect_enabled
    object.__setattr__(settings, "connect_enabled", True)
    yield
    object.__setattr__(settings, "connect_enabled", orig)


# ── Feature flag gating ───────────────────────────────────────────


class TestFeatureFlag:
    """Each HTTP route returns 503 when the substrate is disabled."""

    def _disable(self) -> None:
        object.__setattr__(settings, "connect_enabled", False)

    def test_turn_credentials_503(self, sqlite_client) -> None:
        self._disable()
        resp = sqlite_client.get("/api/connect/turn-credentials")
        assert resp.status_code == 503

    def test_presence_503(self, sqlite_client) -> None:
        self._disable()
        resp = sqlite_client.get("/api/connect/presence")
        assert resp.status_code == 503

    def test_threads_503(self, sqlite_client) -> None:
        self._disable()
        resp = sqlite_client.get("/api/connect/threads")
        assert resp.status_code == 503

    def test_messages_503(self, sqlite_client) -> None:
        self._disable()
        resp = sqlite_client.get("/api/connect/threads/t1/messages")
        assert resp.status_code == 503

    def test_send_503(self, sqlite_client) -> None:
        self._disable()
        resp = sqlite_client.post(
            "/api/connect/threads/t1/send",
            json={"peer_did": "bob@this-instance", "body": "hi"},
        )
        assert resp.status_code == 503

    def test_mark_read_503(self, sqlite_client) -> None:
        self._disable()
        resp = sqlite_client.post("/api/connect/threads/t1/mark-read")
        assert resp.status_code == 503

    def test_calls_503(self, sqlite_client) -> None:
        self._disable()
        resp = sqlite_client.get("/api/connect/calls")
        assert resp.status_code == 503

    def test_call_detail_503(self, sqlite_client) -> None:
        self._disable()
        resp = sqlite_client.get("/api/connect/calls/c1")
        assert resp.status_code == 503

    def test_contacts_503(self, sqlite_client) -> None:
        self._disable()
        resp = sqlite_client.get("/api/connect/contacts")
        assert resp.status_code == 503

    def test_add_contact_503(self, sqlite_client) -> None:
        self._disable()
        resp = sqlite_client.post(
            "/api/connect/contacts",
            json={"peer_did": "bob@this-instance"},
        )
        assert resp.status_code == 503


# ── TURN credentials ─────────────────────────────────────────────


class TestTurnCredentials:
    def test_returns_ice_servers_and_expiry(self, sqlite_client) -> None:
        resp = sqlite_client.get("/api/connect/turn-credentials")
        assert resp.status_code == 200
        data = resp.json()
        assert "ice_servers" in data
        assert isinstance(data["ice_servers"], list)
        assert len(data["ice_servers"]) == 1
        ice = data["ice_servers"][0]
        assert "username" in ice
        assert "credential" in ice
        assert "urls" in ice
        assert isinstance(ice["urls"], list)
        assert "expires_at" in data
        assert isinstance(data["expires_at"], int)


# ── Presence ────────────────────────────────────────────────────


class TestPresence:
    def test_returns_online_list_and_server_time(self, sqlite_client) -> None:
        resp = sqlite_client.get("/api/connect/presence")
        assert resp.status_code == 200
        data = resp.json()
        assert "online_user_ids" in data
        assert isinstance(data["online_user_ids"], list)
        assert "server_time" in data


# ── Threads / messages HTTP loop ─────────────────────────────────


class TestThreadsAndMessages:
    """Smoke-test the HTTP send → list-threads → list-messages loop."""

    def test_threads_empty_initially(self, sqlite_client) -> None:
        resp = sqlite_client.get("/api/connect/threads")
        assert resp.status_code == 200
        assert resp.json() == {"threads": []}

    def test_send_then_list_threads_and_messages(self, sqlite_client) -> None:
        # 1) Send a message via HTTP. The route mints a thread_id +
        # message_id when the caller doesn't supply them.
        resp = sqlite_client.post(
            "/api/connect/threads/auto/send",
            json={
                "peer_did": "bob@this-instance",
                "body": "hello from http",
                "format": "plain",
            },
        )
        assert resp.status_code == 200, resp.text
        sent = resp.json()
        assert sent["thread_id"]
        assert sent["message_id"]
        thread_id = sent["thread_id"]

        # 2) /threads returns the new row.
        resp = sqlite_client.get("/api/connect/threads")
        assert resp.status_code == 200
        threads = resp.json()["threads"]
        assert len(threads) == 1
        assert threads[0]["thread_id"] == thread_id
        assert threads[0]["peer_did"] == "bob@this-instance"
        assert "hello" in threads[0]["last_message_preview"]

        # 3) /threads/{id}/messages returns the message.
        resp = sqlite_client.get(
            f"/api/connect/threads/{thread_id}/messages",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["thread"]["thread_id"] == thread_id
        assert len(data["messages"]) == 1
        assert data["messages"][0]["body"] == "hello from http"

    def test_send_requires_peer_did(self, sqlite_client) -> None:
        resp = sqlite_client.post(
            "/api/connect/threads/auto/send",
            json={"body": "hi"},
        )
        assert resp.status_code == 400

    def test_send_empty_body_400(self, sqlite_client) -> None:
        resp = sqlite_client.post(
            "/api/connect/threads/auto/send",
            json={"peer_did": "bob@this-instance", "body": ""},
        )
        # Empty body without attachment_ref is a message_empty 400 from
        # the routing layer.
        assert resp.status_code == 400

    def test_messages_404_unknown_thread(self, sqlite_client) -> None:
        resp = sqlite_client.get(
            "/api/connect/threads/no-such-thread/messages",
        )
        assert resp.status_code == 404

    def test_messages_since_cursor_filters_to_newer_rows(self, sqlite_client) -> None:
        """Catch-up direction: ?since=<sent_at> returns only strictly
        newer messages. Used by the UI on reconnect to pull what it
        missed while the WS was down."""

        # Send three messages and capture their sent_at values from
        # the subsequent list call.
        thread_id = None
        for i in range(3):
            resp = sqlite_client.post(
                "/api/connect/threads/auto/send",
                json={
                    "peer_did": "bob@this-instance",
                    "thread_id": thread_id or "",
                    "body": f"msg-{i}",
                },
            )
            assert resp.status_code == 200, resp.text
            sent = resp.json()
            thread_id = thread_id or sent["thread_id"]

        # Pull everything to grab the timestamps (newest-first order).
        full = sqlite_client.get(
            f"/api/connect/threads/{thread_id}/messages",
        ).json()
        msgs = full["messages"]
        assert len(msgs) == 3
        # Cursor = oldest message's sent_at → expect the two newer back.
        oldest_sent_at = msgs[-1]["sent_at"]
        resp = sqlite_client.get(
            f"/api/connect/threads/{thread_id}/messages",
            params={"since": oldest_sent_at},
        )
        assert resp.status_code == 200
        catchup = resp.json()["messages"]
        assert len(catchup) == 2
        bodies = sorted(m["body"] for m in catchup)
        assert bodies == ["msg-1", "msg-2"]

    def test_send_fabric_peer_pending_503(self, sqlite_client) -> None:
        # @home.bob.dev resolves to kind=fabric; this test client has no
        # fabric coordinator wired, so the route returns 503 with
        # ``fabric_unavailable`` (post-Wedge-B error code; semantically
        # equivalent to the old ``fabric_routing_pending``).
        resp = sqlite_client.post(
            "/api/connect/threads/auto/send",
            json={
                "peer_did": "bob@home.bob.dev",
                "body": "cross-instance hi",
            },
        )
        assert resp.status_code == 503
        body = resp.json()
        assert body["detail"]["code"] in (
            "fabric_unavailable", "fabric_routing_pending",
        )


# ── Mark read ───────────────────────────────────────────────────


class TestMarkRead:
    def test_mark_read_returns_marked_count(self, sqlite_client) -> None:
        # Seed by sending a message first.
        sqlite_client.post(
            "/api/connect/threads/auto/send",
            json={"peer_did": "bob@this-instance", "body": "hi"},
        )
        threads = sqlite_client.get("/api/connect/threads").json()["threads"]
        thread_id = threads[0]["thread_id"]

        resp = sqlite_client.post(
            f"/api/connect/threads/{thread_id}/mark-read",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "marked" in data
        # Sender's own outgoing isn't bumped by the trigger, so the
        # mark count is 0 here — the API contract still returns the
        # number for the recipient-side case.
        assert isinstance(data["marked"], int)

    def test_mark_read_accepts_empty_body(self, sqlite_client) -> None:
        sqlite_client.post(
            "/api/connect/threads/auto/send",
            json={"peer_did": "bob@this-instance", "body": "hi"},
        )
        threads = sqlite_client.get("/api/connect/threads").json()["threads"]
        thread_id = threads[0]["thread_id"]

        # Empty body is fine — defaults to "all unread".
        resp = sqlite_client.post(
            f"/api/connect/threads/{thread_id}/mark-read",
        )
        assert resp.status_code == 200


# ── Calls ───────────────────────────────────────────────────────


class TestCallsEndpoints:
    """List + detail + rating endpoints for call_sessions."""

    def test_calls_empty_initially(self, sqlite_client) -> None:
        resp = sqlite_client.get("/api/connect/calls")
        assert resp.status_code == 200
        assert resp.json() == {"calls": []}

    def test_call_detail_404_for_unknown(self, sqlite_client) -> None:
        resp = sqlite_client.get("/api/connect/calls/no-such-id")
        assert resp.status_code == 404

    def test_rate_404_for_unknown_call(self, sqlite_client) -> None:
        resp = sqlite_client.post(
            "/api/connect/calls/no-such-id/rate",
            json={"rating": 1},
        )
        assert resp.status_code == 404

    def test_rate_validates_value(self, sqlite_client) -> None:
        resp = sqlite_client.post(
            "/api/connect/calls/no-such-id/rate",
            json={"rating": 5},
        )
        assert resp.status_code == 400


# ── Contacts ────────────────────────────────────────────────────


class TestContactsEndpoints:
    def test_contacts_empty_initially(self, sqlite_client) -> None:
        resp = sqlite_client.get("/api/connect/contacts")
        assert resp.status_code == 200
        assert resp.json() == {"contacts": []}

    def test_add_then_list(self, sqlite_client) -> None:
        resp = sqlite_client.post(
            "/api/connect/contacts",
            json={
                "peer_did": "bob@this-instance",
                "peer_display_name": "Bob",
                "tags": ["family"],
            },
        )
        assert resp.status_code == 200, resp.text
        added = resp.json()
        assert added["peer_did"] == "bob@this-instance"
        assert added["peer_display_name"] == "Bob"

        resp = sqlite_client.get("/api/connect/contacts")
        assert resp.status_code == 200
        contacts = resp.json()["contacts"]
        assert len(contacts) == 1
        assert contacts[0]["contact_id"] == added["contact_id"]

    def test_add_rejects_bad_peer_did(self, sqlite_client) -> None:
        resp = sqlite_client.post(
            "/api/connect/contacts",
            json={"peer_did": "no-at-sign-here"},
        )
        assert resp.status_code == 400

    def test_delete_then_list_empty(self, sqlite_client) -> None:
        contact_id = sqlite_client.post(
            "/api/connect/contacts",
            json={"peer_did": "bob@this-instance"},
        ).json()["contact_id"]

        resp = sqlite_client.delete(
            f"/api/connect/contacts/{contact_id}",
        )
        assert resp.status_code == 200

        contacts = sqlite_client.get("/api/connect/contacts").json()["contacts"]
        assert contacts == []

    def test_delete_404_for_unknown(self, sqlite_client) -> None:
        resp = sqlite_client.delete("/api/connect/contacts/no-such-id")
        assert resp.status_code == 404

    def test_patch_blocked_round_trip(self, sqlite_client) -> None:
        contact_id = sqlite_client.post(
            "/api/connect/contacts",
            json={"peer_did": "bob@this-instance"},
        ).json()["contact_id"]

        resp = sqlite_client.patch(
            f"/api/connect/contacts/{contact_id}",
            json={"blocked": True},
        )
        assert resp.status_code == 200
        assert resp.json()["blocked"] is True

        # Default list excludes blocked.
        visible = sqlite_client.get("/api/connect/contacts").json()["contacts"]
        assert visible == []
        all_ = sqlite_client.get(
            "/api/connect/contacts?include_blocked=true",
        ).json()["contacts"]
        assert len(all_) == 1

    def test_patch_tags_round_trip(self, sqlite_client) -> None:
        contact_id = sqlite_client.post(
            "/api/connect/contacts",
            json={"peer_did": "bob@this-instance"},
        ).json()["contact_id"]

        resp = sqlite_client.patch(
            f"/api/connect/contacts/{contact_id}",
            json={"tags": ["work", "team"]},
        )
        assert resp.status_code == 200
        assert resp.json()["tags"] == ["work", "team"]

    def test_patch_404_when_missing(self, sqlite_client) -> None:
        resp = sqlite_client.patch(
            "/api/connect/contacts/no-such-id",
            json={"blocked": True},
        )
        assert resp.status_code == 404
