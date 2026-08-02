"""Tests for ollama_routes.py — Ollama-compatible API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from augmentum.models.base import InternalChatResponse, Message, Usage

# ---------------------------------------------------------------------------
# GET /api/version
# ---------------------------------------------------------------------------


class TestOllamaVersion:
    def test_version_returns_200(self, client):
        resp = client.get("/api/version")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data

    def test_head_root_returns_200(self, client):
        """HEAD /api/ is wired alongside GET /api/version."""
        resp = client.head("/api/")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET / (health check — defined in server.py)
# ---------------------------------------------------------------------------


class TestOllamaHealthCheck:
    def test_root_returns_ollama_running(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Ollama is running" in resp.text


# ---------------------------------------------------------------------------
# GET /api/tags
# ---------------------------------------------------------------------------


class TestOllamaTags:
    def test_tags_returns_model_list(self, client):
        resp = client.get("/api/tags")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        # At least the base model from the mock backend
        assert len(data["models"]) >= 1

    def test_tags_includes_mode_prefixes(self, client):
        resp = client.get("/api/tags")
        data = resp.json()
        names = [m["name"] for m in data["models"]]
        # Should include prefixed variants
        has_analytical = any(n.startswith("a/") for n in names)
        has_narrative = any(n.startswith("n/") for n in names)
        has_passthrough = any(n.startswith("p/") for n in names)
        assert has_analytical
        assert has_narrative
        assert has_passthrough


# ---------------------------------------------------------------------------
# POST /api/chat (non-streaming)
# ---------------------------------------------------------------------------


class TestOllamaChat:
    def test_chat_non_streaming(self, client):
        resp = client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert data["message"]["role"] == "assistant"
        assert data["done"] is True

    def test_chat_empty_model_rejected(self, client):
        resp = client.post("/api/chat", json={
            "model": "",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        })
        assert resp.status_code == 422  # Validation error

    def test_chat_empty_messages_rejected(self, client):
        resp = client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [],
            "stream": False,
        })
        assert resp.status_code == 422

    def test_chat_response_has_usage(self, client):
        resp = client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        })
        data = resp.json()
        assert "prompt_eval_count" in data or "eval_count" in data

    def test_chat_coder_workspace_uses_workspace_session_id(self, client):
        handler = MagicMock()
        handler.handle = AsyncMock(return_value=InternalChatResponse(
            message=Message(role="assistant", content="ok"),
            model="llama3.1:8b",
            usage=Usage(),
        ))
        with patch("augmentum.proxy.ollama_routes.get_handler_for_mode", return_value=handler) as mock_get_handler:
            resp = client.post(
                "/api/chat",
                headers={
                    "X-Augmentum-Mode": "coder",
                    "X-Augmentum-Workspace": "ws-ollama",
                },
                json={
                    "model": "llama3.1:8b",
                    "messages": [{"role": "user", "content": "Inspect this repo"}],
                    "stream": False,
                },
            )
        assert resp.status_code == 200
        assert mock_get_handler.call_args.args[2] == "ws-ollama"


# ---------------------------------------------------------------------------
# POST /api/generate (non-streaming)
# ---------------------------------------------------------------------------


class TestOllamaGenerate:
    def test_generate_non_streaming(self, client):
        resp = client.post("/api/generate", json={
            "model": "llama3.1:8b",
            "prompt": "Once upon a time",
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert data["done"] is True

    def test_generate_with_system_prompt(self, client):
        resp = client.post("/api/generate", json={
            "model": "llama3.1:8b",
            "prompt": "Hello",
            "system": "You are a pirate.",
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True


# ---------------------------------------------------------------------------
# POST /api/show
# ---------------------------------------------------------------------------


class TestOllamaShow:
    def test_show_returns_model_details(self, client):
        # The mock has ollama in registry.backends
        resp = client.post("/api/show", json={"name": "llama3.1:8b"})
        assert resp.status_code == 200
        data = resp.json()
        assert "modelfile" in data
        assert "details" in data

    def test_show_no_ollama_returns_404(self, client):
        # Mock get_backend to return None for "ollama"
        original = client.app.state.provider_registry.get_backend
        client.app.state.provider_registry.get_backend = MagicMock(return_value=None)
        try:
            resp = client.post("/api/show", json={"name": "llama3.1:8b"})
            assert resp.status_code == 404
        finally:
            client.app.state.provider_registry.get_backend = original


# ---------------------------------------------------------------------------
# POST /api/embeddings
# ---------------------------------------------------------------------------


class TestOllamaEmbeddings:
    def test_embeddings_proxies_to_backend(self, client):
        # The http_client is a MagicMock, so we mock the post method
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
        mock_response.status_code = 200
        client.app.state.http_client.post = AsyncMock(return_value=mock_response)

        resp = client.post("/api/embeddings", json={"model": "llama3.1:8b", "input": "Hello"})
        assert resp.status_code == 200

    def test_embed_endpoint_also_works(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [[0.1, 0.2]]}
        mock_response.status_code = 200
        client.app.state.http_client.post = AsyncMock(return_value=mock_response)

        resp = client.post("/api/embed", json={"model": "llama3.1:8b", "input": "Hello"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# GET /api/ps
# ---------------------------------------------------------------------------


class TestOllamaPs:
    def test_ps_returns_models(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_response.status_code = 200
        client.app.state.http_client.get = AsyncMock(return_value=mock_response)

        resp = client.get("/api/ps")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data

    def test_ps_returns_empty_on_failure(self, client):
        client.app.state.http_client.get = AsyncMock(side_effect=Exception("conn refused"))
        resp = client.get("/api/ps")
        assert resp.status_code == 200
        assert resp.json() == {"models": []}
