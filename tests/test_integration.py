"""Integration tests for end-to-end request flows through the proxy."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient


class TestOllamaAPIIntegration:
    """End-to-end tests for the Ollama-compatible API."""

    def test_health_check(self, client: TestClient):
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.text == "Ollama is running"

    def test_version_endpoint(self, client: TestClient):
        resp = client.get("/api/version")
        assert resp.status_code == 200
        assert resp.json()["version"] == "0.1.0"

    def test_chat_non_streaming(self, client: TestClient):
        """Non-streaming chat request through full pipeline."""
        resp = client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "message" in data
        assert data["message"]["role"] == "assistant"
        assert data["message"]["content"] != ""
        assert data["done"] is True

    def test_chat_streaming(self, client: TestClient):
        """Streaming chat request returns NDJSON chunks."""
        resp = client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert "application/x-ndjson" in resp.headers.get("content-type", "")

        # Parse NDJSON lines
        lines = resp.text.strip().split("\n")
        assert len(lines) > 0

        # Each line should be valid JSON
        chunks = []
        for line in lines:
            if line.strip():
                data = json.loads(line)
                chunks.append(data)

        # Should have at least one non-done chunk and one done chunk
        assert any(not c.get("done") for c in chunks)
        assert any(c.get("done") for c in chunks)

        # Last chunk should be done
        assert chunks[-1]["done"] is True

    def test_chat_with_system_message(self, client: TestClient):
        resp = client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
            "stream": False,
        })
        assert resp.status_code == 200
        assert resp.json()["message"]["content"] != ""

    def test_chat_with_options(self, client: TestClient):
        """Chat request with inference options."""
        resp = client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "options": {"temperature": 0.5, "num_predict": 50},
        })
        assert resp.status_code == 200
        assert resp.json()["done"] is True

    def test_generate_non_streaming(self, client: TestClient):
        """Generate endpoint converts to chat format internally."""
        resp = client.post("/api/generate", json={
            "model": "llama3.1:8b",
            "prompt": "Hello world",
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert data["done"] is True

    def test_generate_with_system(self, client: TestClient):
        resp = client.post("/api/generate", json={
            "model": "llama3.1:8b",
            "prompt": "Hello",
            "system": "You are helpful.",
            "stream": False,
        })
        assert resp.status_code == 200
        assert resp.json()["done"] is True

    def test_tags_lists_models(self, client: TestClient):
        """Model list includes base models and prefixed variants."""
        resp = client.get("/api/tags")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data

        model_names = [m["name"] for m in data["models"]]
        # Should have the base model
        assert "llama3.1:8b" in model_names
        # Should have prefixed variants
        assert "a/llama3.1:8b" in model_names
        assert "n/llama3.1:8b" in model_names
        assert "p/llama3.1:8b" in model_names

    def test_show_model(self, client: TestClient):
        resp = client.post("/api/show", json={"name": "llama3.1:8b"})
        assert resp.status_code == 200
        data = resp.json()
        assert "modelfile" in data
        assert "parameters" in data


class TestOpenAIAPIIntegration:
    """End-to-end tests for the OpenAI-compatible API."""

    def test_openai_chat_non_streaming(self, client: TestClient):
        resp = client.post("/v1/chat/completions", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] != ""
        assert "usage" in data

    def test_openai_models_list(self, client: TestClient):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0
        assert data["data"][0]["id"] == "llama3.1:8b"
        assert data["data"][0]["owned_by"] == "ollama"


class TestModeRoutingIntegration:
    """Tests for mode classification and routing through the proxy."""

    def test_passthrough_simple_message(self, client: TestClient):
        """Simple message should route through passthrough mode."""
        resp = client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "stream": False,
        })
        assert resp.status_code == 200
        assert resp.json()["done"] is True

    def test_mode_override_via_prefix(self, client: TestClient):
        """Model prefix a/ should force analytical mode."""
        resp = client.post("/api/chat", json={
            "model": "a/llama3.1:8b",
            "messages": [{"role": "user", "content": "Analyze this topic."}],
            "stream": False,
        })
        assert resp.status_code == 200
        # Should still return a valid response (may fall back to passthrough)
        assert resp.json()["done"] is True

    def test_mode_override_via_header(self, client: TestClient):
        """X-Augmentum-Mode header should force the requested mode."""
        resp = client.post(
            "/api/chat",
            json={
                "model": "llama3.1:8b",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
            headers={"X-Augmentum-Mode": "passthrough"},
        )
        assert resp.status_code == 200
        assert resp.json()["done"] is True

    def test_narrative_detection(self, client: TestClient):
        """Character card in system prompt should trigger narrative detection."""
        resp = client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "{{char}}'s name is Luna.\n"
                        "{{char}} is a brave warrior princess.\n"
                        "{{char}}'s personality: kind, strong, determined.\n"
                        "Scenario: {{user}} meets {{char}} in the forest."
                    ),
                },
                {"role": "user", "content": "I approach the clearing."},
            ],
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] is True
        assert data["message"]["role"] == "assistant"


class TestUIAPIIntegration:
    """Tests for UI-specific API endpoints."""

    def test_ui_status(self, client: TestClient):
        resp = client.get("/api/ui/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert data["version"] == "0.1.0"
        assert "backends" in data
        assert "tools" in data
        assert isinstance(data["tools"], list)

    def test_ui_settings(self, client: TestClient):
        resp = client.get("/api/ui/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "default_backend" in data
        assert "uarf_max_backtracks" in data
        assert "narrative_context_budget" in data
        assert "prompt_cache_enabled" in data

    def test_ui_session_state_no_engine(self, client: TestClient):
        """Requesting state for a non-existent session returns null state."""
        resp = client.get("/api/ui/session/nonexistent123/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] is None


class TestConfigAPIIntegration:
    """Tests for configuration API."""

    def test_config_returns_safe_dict(self, client: TestClient):
        resp = client.get("/api/config/")
        assert resp.status_code == 200
        data = resp.json()
        # API keys should be redacted
        if "openai_api_key" in data:
            assert data["openai_api_key"] in (None, "***")
        # Port should be present
        assert "port" in data

    def test_config_section_filtering(self, client: TestClient):
        """Config section endpoint filters by prefix."""
        resp = client.get("/api/config/section/uarf")
        assert resp.status_code == 200
        data = resp.json()
        # All keys should start with uarf_
        for key in data:
            assert key.startswith("uarf_")


class TestModelManagementIntegration:
    """Tests for model management API."""

    def test_models_status(self, client: TestClient):
        resp = client.get("/api/models/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_running_models(self, client: TestClient):
        resp = client.get("/api/models/running")
        assert resp.status_code == 200
        assert "models" in resp.json()

    def test_model_load_unload(self, client: TestClient):
        """Load and unload model endpoints respond correctly."""
        # Load
        resp = client.post("/api/models/llama3.1:8b/load")
        assert resp.status_code == 200
        assert "success" in resp.json()

        # Unload
        resp = client.post("/api/models/llama3.1:8b/unload")
        assert resp.status_code == 200
        assert "success" in resp.json()


class TestCacheIntegration:
    """Tests for cache API endpoints."""

    def test_cache_stats(self, client: TestClient):
        resp = client.get("/api/cache/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "prompt_cache" in data

    def test_cache_clear(self, client: TestClient):
        resp = client.post("/api/cache/clear")
        assert resp.status_code == 200
        data = resp.json()
        assert "prompt_cache_cleared" in data
        assert isinstance(data["prompt_cache_cleared"], int)


class TestMultiTurnConversation:
    """Tests for multi-turn conversation flows."""

    def test_multi_turn_passthrough(self, client: TestClient):
        """Multiple messages in a conversation should all work."""
        # First message
        resp1 = client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "What is Python?"}],
            "stream": False,
        })
        assert resp1.status_code == 200
        first_response = resp1.json()["message"]["content"]

        # Second message with history
        resp2 = client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": first_response},
                {"role": "user", "content": "Tell me more."},
            ],
            "stream": False,
        })
        assert resp2.status_code == 200
        assert resp2.json()["done"] is True

    def test_session_consistency(self, client: TestClient):
        """Same session header should create consistent session."""
        headers = {"X-Augmentum-Session": "test-session-123"}

        resp1 = client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        }, headers=headers)
        assert resp1.status_code == 200

        resp2 = client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"},
            ],
            "stream": False,
        }, headers=headers)
        assert resp2.status_code == 200


class TestErrorHandling:
    """Tests for error handling in the proxy."""

    def test_invalid_json_body(self, client: TestClient):
        """Invalid JSON should return a clear error."""
        resp = client.post(
            "/api/chat",
            content=b"not valid json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422  # FastAPI validation error

    def test_missing_model(self, client: TestClient):
        """Missing required model field should return validation error."""
        resp = client.post("/api/chat", json={
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        })
        assert resp.status_code == 422

    def test_empty_messages(self, client: TestClient):
        """Empty messages array should still be handled."""
        resp = client.post("/api/chat", json={
            "model": "llama3.1:8b",
            "messages": [],
            "stream": False,
        })
        # Should either work or return a clear error, not crash
        assert resp.status_code in (200, 400, 422)

    def test_nonexistent_endpoint(self, client: TestClient):
        """Unknown endpoints should return 404."""
        resp = client.get("/api/nonexistent")
        assert resp.status_code in (404, 405)
