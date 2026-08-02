"""Tests for openai_routes.py — OpenAI-compatible API endpoints (/v1/*)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from augmentum.models.base import InternalChatResponse, Message, Usage

# ---------------------------------------------------------------------------
# POST /v1/chat/completions (non-streaming)
# ---------------------------------------------------------------------------


class TestOpenAIChatCompletions:
    def test_chat_non_streaming(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) >= 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "content" in data["choices"][0]["message"]

    def test_chat_response_shape(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
        })
        data = resp.json()
        # OpenAI format fields
        assert "id" in data
        assert "object" in data
        assert data["object"] == "chat.completion"
        assert "model" in data
        assert "usage" in data

    def test_chat_empty_model_rejected(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 422

    def test_chat_empty_messages_rejected(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "llama3.1:8b",
            "messages": [],
        })
        assert resp.status_code == 422

    def test_chat_with_temperature(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "temperature": 0.5,
        })
        assert resp.status_code == 200

    def test_chat_with_system_message(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "llama3.1:8b",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
            "stream": False,
        })
        assert resp.status_code == 200

    def test_chat_coder_workspace_uses_workspace_session_id(self, client):
        handler = MagicMock()
        handler.handle = AsyncMock(return_value=InternalChatResponse(
            message=Message(role="assistant", content="ok"),
            model="llama3.1:8b",
            usage=Usage(),
        ))
        with patch("augmentum.proxy.openai_routes.get_handler_for_mode", return_value=handler) as mock_get_handler:
            resp = client.post(
                "/v1/chat/completions",
                headers={
                    "X-Augmentum-Mode": "coder",
                    "X-Augmentum-Workspace": "ws-openai",
                },
                json={
                    "model": "llama3.1:8b",
                    "messages": [{"role": "user", "content": "Inspect this repo"}],
                    "stream": False,
                },
            )
        assert resp.status_code == 200
        assert mock_get_handler.call_args.args[2] == "ws-openai"


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


class TestOpenAIModels:
    def test_models_returns_list(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1

    def test_models_have_openai_fields(self, client):
        resp = client.get("/v1/models")
        data = resp.json()
        for model in data["data"]:
            assert "id" in model
            assert "object" in model
            assert model["object"] == "model"
            assert "owned_by" in model

    def test_models_include_mode_prefixes(self, client):
        resp = client.get("/v1/models")
        data = resp.json()
        ids = [m["id"] for m in data["data"]]
        has_a = any(i.startswith("a/") for i in ids)
        has_n = any(i.startswith("n/") for i in ids)
        has_p = any(i.startswith("p/") for i in ids)
        assert has_a
        assert has_n
        assert has_p


# ---------------------------------------------------------------------------
# Streaming (basic contract)
# ---------------------------------------------------------------------------


class TestOpenAIStreaming:
    def test_streaming_returns_event_stream(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        })
        assert resp.status_code == 200
        # Streaming responses use text/event-stream content type
        assert "text/event-stream" in resp.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# GET /v1/image-models
# ---------------------------------------------------------------------------


class TestOpenAIImageModels:
    def test_image_models_returns_list(self, client):
        resp = client.get("/v1/image-models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)


# ---------------------------------------------------------------------------
# _backend_runtime_error_to_response — translate openai_compat RuntimeErrors
# from upstream non-200s into structured JSONResponses instead of ASGI 500s.
# ---------------------------------------------------------------------------


class TestBackendRuntimeErrorToResponse:
    """Direct-mode dispatch deliberately skips the standard chat path's
    fallback-to-passthrough recovery (its contract is verbatim pass-through;
    see augmentum/modes/direct/handler.py). Backend RuntimeErrors there have
    no in-mode recovery and would otherwise bubble as ASGI 500. The helper
    translates the ``Backend returned NNN: <body>`` shape into a clean
    response the UI can render.
    """

    def _parse(self, response):
        import json
        return response.status_code, json.loads(response.body)

    def test_context_window_502_remaps_to_400(self):
        """A 5xx with context-window markers is semantically a client error
        (input too long), not a server fault. OpenAI's canonical code for
        this is ``context_length_exceeded`` at status 400."""
        from augmentum.proxy.openai_routes import _backend_runtime_error_to_response

        exc = RuntimeError(
            'Backend returned 502: {"error":{"message":"Codex API error '
            '(502): Your input exceeds the context window of this model. '
            'Please adjust your input and try again."}}'
        )
        resp = _backend_runtime_error_to_response(exc)
        assert resp is not None
        status, payload = self._parse(resp)
        assert status == 400
        assert payload["error"]["code"] == "context_length_exceeded"
        assert payload["error"]["type"] == "invalid_request_error"

    def test_429_passes_through_with_status(self):
        """Non-context-window upstream errors keep their status code so the
        UI can distinguish rate limits from validation from server issues."""
        from augmentum.proxy.openai_routes import _backend_runtime_error_to_response

        exc = RuntimeError("Backend returned 429: rate limit exceeded")
        resp = _backend_runtime_error_to_response(exc)
        assert resp is not None
        status, payload = self._parse(resp)
        assert status == 429
        assert payload["error"]["code"] == "backend_429"

    def test_generic_500_passes_through(self):
        """A 5xx without context-window markers stays a 5xx — we don't
        re-classify random server errors as client errors."""
        from augmentum.proxy.openai_routes import _backend_runtime_error_to_response

        exc = RuntimeError("Backend returned 500: something went wrong")
        resp = _backend_runtime_error_to_response(exc)
        assert resp is not None
        status, _ = self._parse(resp)
        assert status == 500

    def test_non_matching_returns_none(self):
        """A RuntimeError that isn't the openai_compat backend-error shape
        returns None so the caller re-raises, preserving the existing
        bubble-up behavior for truly unexpected failures."""
        from augmentum.proxy.openai_routes import _backend_runtime_error_to_response

        exc = RuntimeError("something else entirely")
        assert _backend_runtime_error_to_response(exc) is None
