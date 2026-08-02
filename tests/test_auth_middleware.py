"""Tests for auth middleware token extraction and public paths."""

from __future__ import annotations

import pytest

from augmentum.auth.middleware import AuthMiddleware


class TestPublicPaths:
    def test_root_is_public(self):
        mw = AuthMiddleware(None)
        assert mw._is_public("/") is True

    def test_login_is_public(self):
        mw = AuthMiddleware(None)
        assert mw._is_public("/api/auth/login") is True

    def test_setup_is_public(self):
        mw = AuthMiddleware(None)
        assert mw._is_public("/api/auth/setup") is True

    def test_static_is_public(self):
        mw = AuthMiddleware(None)
        assert mw._is_public("/ui/index.html") is True

    def test_api_chat_is_not_public(self):
        mw = AuthMiddleware(None)
        assert mw._is_public("/api/chat") is False

    def test_api_chats_is_not_public(self):
        mw = AuthMiddleware(None)
        assert mw._is_public("/api/chats/") is False

    def test_auth_status_exact_is_public(self):
        mw = AuthMiddleware(None)
        assert mw._is_public("/api/auth/status") is True

    def test_auth_status_prefix_lookalike_is_not_public(self):
        """Regression: /api/auth/status was a startswith prefix, which would
        also exempt a future /api/auth/status_secret route from auth. It's
        now an exact-match only, so look-alike paths stay gated."""
        mw = AuthMiddleware(None)
        assert mw._is_public("/api/auth/status_secret") is False
        assert mw._is_public("/api/auth/statuses") is False


class TestTokenExtraction:
    def test_bearer_header(self):
        mw = AuthMiddleware(None)
        scope = {"headers": [(b"authorization", b"Bearer abc123token")]}
        assert mw._extract_token(scope) == "abc123token"

    def test_cookie(self):
        mw = AuthMiddleware(None)
        scope = {"headers": [(b"cookie", b"other=x; augmentum_session=tok456; foo=bar")]}
        assert mw._extract_token(scope) == "tok456"

    def test_bearer_takes_priority_over_cookie(self):
        mw = AuthMiddleware(None)
        scope = {"headers": [
            (b"authorization", b"Bearer bearer_tok"),
            (b"cookie", b"augmentum_session=cookie_tok"),
        ]}
        assert mw._extract_token(scope) == "bearer_tok"

    def test_no_token(self):
        mw = AuthMiddleware(None)
        scope = {"headers": []}
        assert mw._extract_token(scope) is None

    def test_x_api_key_header(self):
        """Anthropic Messages API clients (Claude Code etc.) send the
        key as ``x-api-key`` rather than ``Authorization: Bearer``.
        Both must yield the same downstream user lookup."""
        mw = AuthMiddleware(None)
        scope = {"headers": [(b"x-api-key", b"sk-aug-abc123token")]}
        assert mw._extract_token(scope) == "sk-aug-abc123token"

    def test_x_api_key_sk_aug_beats_bearer(self):
        """When both are present AND x-api-key is sk-aug-shaped,
        x-api-key wins. Claude Code sends BOTH headers (cached OAuth
        Bearer + user-set x-api-key); without this priority every
        request 401s on the OAuth token before it ever sees the
        real key. CCR / LiteLLM hit the same trap."""
        mw = AuthMiddleware(None)
        scope = {"headers": [
            (b"authorization", b"Bearer sk-ant-cached-oauth-token"),
            (b"x-api-key", b"sk-aug-realkey"),
        ]}
        assert mw._extract_token(scope) == "sk-aug-realkey"

    def test_bearer_wins_when_x_api_key_is_not_sk_aug(self):
        """For non-Augmentum x-api-key values, fall through to Bearer.
        Keeps existing OpenWebUI / Cursor flows unchanged."""
        mw = AuthMiddleware(None)
        scope = {"headers": [
            (b"authorization", b"Bearer bearer_tok"),
            (b"x-api-key", b"not-an-aug-key"),
        ]}
        assert mw._extract_token(scope) == "bearer_tok"


class TestTicketExtraction:
    def test_ticket_in_query(self):
        mw = AuthMiddleware(None)
        assert mw._extract_ticket("ticket=abc123&session_id=s1") == "abc123"

    def test_no_ticket(self):
        mw = AuthMiddleware(None)
        assert mw._extract_ticket("session_id=s1&model=llama") is None


class TestFailClosedWhenNoSessionManager:
    """When SessionManager is None (DB down at startup), non-public paths
    must be denied — not passed through as if auth were disabled. Passing
    through would expose user-scoped endpoints with user_id="" and return
    unscoped rows across users.
    """

    async def _run(self, scope):
        sent: list[dict] = []
        called = {"inner": False}

        async def inner_app(s, r, sd):
            called["inner"] = True

        async def send(msg):
            sent.append(msg)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        mw = AuthMiddleware(inner_app, session_manager=None)
        await mw(scope, receive, send)
        return sent, called["inner"]

    @pytest.mark.asyncio
    async def test_http_non_public_returns_503(self):
        scope = {"type": "http", "path": "/api/chat", "headers": [], "app": None}
        sent, inner_called = await self._run(scope)
        assert inner_called is False
        assert sent[0]["type"] == "http.response.start"
        assert sent[0]["status"] == 503

    @pytest.mark.asyncio
    async def test_http_public_still_passes_through(self):
        scope = {"type": "http", "path": "/api/auth/status", "headers": [], "app": None}
        _, inner_called = await self._run(scope)
        assert inner_called is True

    @pytest.mark.asyncio
    async def test_websocket_non_public_closes_with_1011(self):
        scope = {
            "type": "websocket", "path": "/ws/voice",
            "headers": [], "query_string": b"", "app": None,
        }
        sent, inner_called = await self._run(scope)
        assert inner_called is False
        assert sent[0]["type"] == "websocket.close"
        assert sent[0]["code"] == 1011
