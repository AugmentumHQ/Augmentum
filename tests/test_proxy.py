"""Tests for proxy endpoints — both Ollama and OpenAI formats."""

from __future__ import annotations

from unittest.mock import patch


def test_health_check(client):
    """GET / returns Ollama-compatible health check."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.text == "Ollama is running"


def test_ollama_version(client):
    """GET /api/version returns version info."""
    resp = client.get("/api/version")
    assert resp.status_code == 200
    assert "version" in resp.json()


def test_ollama_tags(client):
    """GET /api/tags returns model list with prefixed variants."""
    resp = client.get("/api/tags")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    # 1 base model + 3 prefixed variants (a/, n/, p/)
    assert len(data["models"]) == 4
    names = [m["name"] for m in data["models"]]
    assert "llama3.1:8b" in names


def test_ollama_show(client):
    """POST /api/show returns model details."""
    resp = client.post("/api/show", json={"name": "llama3.1:8b"})
    assert resp.status_code == 200
    data = resp.json()
    assert "modelfile" in data
    assert "details" in data


def test_ollama_chat_non_streaming(client):
    """POST /api/chat with stream=false returns full response."""
    resp = client.post(
        "/api/chat",
        json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is True
    assert data["message"]["role"] == "assistant"
    assert data["message"]["content"] == "Hello from mock Ollama!"
    assert data["model"] == "llama3.1:8b"


def test_ollama_generate_non_streaming(client):
    """POST /api/generate with stream=false returns full response."""
    resp = client.post(
        "/api/generate",
        json={
            "model": "llama3.1:8b",
            "prompt": "Hello",
            "stream": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["done"] is True
    assert data["response"] == "Hello from mock Ollama!"


def test_openai_chat_non_streaming(client):
    """POST /v1/chat/completions without streaming."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["content"] == "Hello from mock Ollama!"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert "usage" in data


def test_openai_models(client):
    """GET /v1/models returns model list in OpenAI format."""
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == "llama3.1:8b"


def test_ollama_chat_with_options(client):
    """POST /api/chat with options passes through correctly."""
    resp = client.post(
        "/api/chat",
        json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "options": {"temperature": 0.5, "top_p": 0.9, "num_predict": 100},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["done"] is True


def test_openai_chat_with_params(client):
    """POST /v1/chat/completions with optional params."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "temperature": 0.5,
            "max_tokens": 100,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["finish_reason"] == "stop"


# ── CORS Configuration ──────────────────────────────────────────────────


def test_ollama_chat_empty_model_rejected(client):
    """POST /api/chat with empty model is rejected."""
    resp = client.post(
        "/api/chat",
        json={"model": "", "messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert resp.status_code == 422


def test_ollama_chat_empty_messages_rejected(client):
    """POST /api/chat with empty messages is rejected."""
    resp = client.post(
        "/api/chat",
        json={"model": "llama3.1:8b", "messages": [], "stream": False},
    )
    assert resp.status_code == 422


def test_openai_chat_empty_model_rejected(client):
    """POST /v1/chat/completions with empty model is rejected."""
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "", "messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    assert resp.status_code == 422


def test_openai_chat_empty_messages_rejected(client):
    """POST /v1/chat/completions with empty messages is rejected."""
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "llama3.1:8b", "messages": [], "stream": False},
    )
    assert resp.status_code == 422


def test_session_list(client):
    """GET /api/sessions returns session list."""
    resp = client.get("/api/sessions/")
    assert resp.status_code == 200
    data = resp.json()
    assert "sessions" in data


def test_session_export_not_found(client):
    """GET /api/sessions/{id}/export returns 404 for unknown session."""
    resp = client.get("/api/sessions/nonexistent/export")
    assert resp.status_code == 404


def test_deep_health_check(client):
    """GET /api/health returns backend status."""
    resp = client.get("/api/health")
    data = resp.json()
    assert "status" in data
    assert "backends" in data


def test_cors_default_restricts_unknown_origin(client):
    """Default config restricts CORS to localhost origins — rejects unknown."""
    resp = client.options(
        "/api/chat",
        headers={"Origin": "http://evil.com", "Access-Control-Request-Method": "POST"},
    )
    # Unknown origin should NOT get access-control-allow-origin
    assert resp.headers.get("access-control-allow-origin") != "http://evil.com"


def test_cors_allows_localhost(client):
    """Default config allows localhost:6100."""
    resp = client.options(
        "/api/chat",
        headers={"Origin": "http://localhost:6100", "Access-Control-Request-Method": "POST"},
    )
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:6100"


def test_cors_wildcard_credentials_safety():
    """Setting credentials=True with wildcard origins forces credentials off."""
    from augmentum.config import Settings

    with patch("augmentum.proxy.server.settings", Settings(
        cors_origins=["*"],
        cors_allow_credentials=True,
        ollama_base_url="http://localhost:11434",
    )):
        from augmentum.proxy.server import create_app
        app = create_app()
        # Inspect middleware — the safety guard should have disabled credentials
        for mw in app.user_middleware:
            if mw.cls.__name__ == "CORSMiddleware":
                assert mw.kwargs.get("allow_credentials") is False
                break
        else:
            raise AssertionError("CORSMiddleware not found")


def test_cors_custom_origins_with_credentials():
    """Specific origins + credentials=True is allowed."""
    from augmentum.config import Settings

    with patch("augmentum.proxy.server.settings", Settings(
        cors_origins=["https://myapp.com"],
        cors_allow_credentials=True,
        ollama_base_url="http://localhost:11434",
    )):
        from augmentum.proxy.server import create_app
        app = create_app()
        for mw in app.user_middleware:
            if mw.cls.__name__ == "CORSMiddleware":
                assert mw.kwargs.get("allow_credentials") is True
                assert mw.kwargs.get("allow_origins") == ["https://myapp.com"]
                break
        else:
            raise AssertionError("CORSMiddleware not found")


# --- Vision-Language (VL) model support ---


class TestVisionLanguageParsing:
    """Test OpenAI vision content format parsing and backend payload building."""

    def test_parse_string_content(self):
        """Plain string content returns text only, no images."""
        from augmentum.proxy.openai_routes import _parse_openai_content

        text, images = _parse_openai_content("Hello world")
        assert text == "Hello world"
        assert images is None

    def test_parse_content_array_text_only(self):
        """Content array with only text parts."""
        from augmentum.proxy.openai_routes import _parse_openai_content

        content = [{"type": "text", "text": "What is this?"}]
        text, images = _parse_openai_content(content)
        assert text == "What is this?"
        assert images is None

    def test_parse_content_array_with_image(self):
        """Content array with text + image_url parts."""
        from augmentum.proxy.openai_routes import _parse_openai_content

        content = [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
        ]
        text, images = _parse_openai_content(content)
        assert text == "Describe this image"
        assert images == ["data:image/png;base64,abc123"]

    def test_parse_multiple_images(self):
        """Content array with multiple images."""
        from augmentum.proxy.openai_routes import _parse_openai_content

        content = [
            {"type": "text", "text": "Compare these"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,img1"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,img2"}},
        ]
        text, images = _parse_openai_content(content)
        assert text == "Compare these"
        assert len(images) == 2

    def test_parse_image_only_no_text(self):
        """Content array with image but no text part."""
        from augmentum.proxy.openai_routes import _parse_openai_content

        content = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        text, images = _parse_openai_content(content)
        assert text == ""
        assert images == ["data:image/png;base64,abc"]

    def test_to_internal_preserves_images(self):
        """OpenAI vision request converts to internal Message with images."""
        from augmentum.proxy.openai_routes import OpenAIChatRequest, to_internal_chat_request

        req = OpenAIChatRequest(
            model="llava:13b",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "What do you see?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}},
                ]},
            ],
            stream=False,
        )
        internal = to_internal_chat_request(req)
        assert internal.messages[0].content == "What do you see?"
        assert internal.messages[0].images == ["data:image/png;base64,xyz"]

    def test_to_internal_plain_text_no_images(self):
        """Plain text messages have no images."""
        from augmentum.proxy.openai_routes import OpenAIChatRequest, to_internal_chat_request

        req = OpenAIChatRequest(
            model="llama3.1:8b",
            messages=[{"role": "user", "content": "Hello"}],
            stream=False,
        )
        internal = to_internal_chat_request(req)
        assert internal.messages[0].content == "Hello"
        assert internal.messages[0].images is None

    def test_openai_backend_builds_vision_content(self):
        """OpenAI backend converts images to vision array format."""
        from augmentum.models.base import Message
        from augmentum.models.openai_compat import OpenAIBackend

        msg = Message(role="user", content="Describe this", images=["data:image/png;base64,abc"])
        result = OpenAIBackend._build_vision_content(msg)
        assert isinstance(result, list)
        assert result[0] == {"type": "text", "text": "Describe this"}
        assert result[1]["type"] == "image_url"
        assert result[1]["image_url"]["url"] == "data:image/png;base64,abc"

    def test_openai_backend_plain_text_stays_string(self):
        """OpenAI backend returns plain string when no images."""
        from augmentum.models.base import Message
        from augmentum.models.openai_compat import OpenAIBackend

        msg = Message(role="user", content="Hello")
        result = OpenAIBackend._build_vision_content(msg)
        assert isinstance(result, str)
        assert result == "Hello"

    def test_llamacpp_payload_includes_vision(self):
        """llama.cpp backend includes vision content in payload."""
        from augmentum.models.base import InternalChatRequest, Message
        from augmentum.models.llama_cpp import LlamaCppBackend

        req = InternalChatRequest(
            model="llava",
            messages=[
                Message(role="user", content="What is this?", images=["data:image/png;base64,abc"]),
            ],
        )
        payload = LlamaCppBackend._to_openai_payload(req)
        content = payload["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "What is this?"}

    def test_llamacpp_payload_plain_text(self):
        """llama.cpp backend uses plain string when no images."""
        from augmentum.models.base import InternalChatRequest, Message
        from augmentum.models.llama_cpp import LlamaCppBackend

        req = InternalChatRequest(
            model="llama3.1:8b",
            messages=[Message(role="user", content="Hello")],
        )
        payload = LlamaCppBackend._to_openai_payload(req)
        assert payload["messages"][0]["content"] == "Hello"

    def test_openai_endpoint_accepts_vision_format(self, client):
        """POST /v1/chat/completions accepts vision content array."""
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "llava:13b",
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    ]},
                ],
                "stream": False,
            },
        )
        # Should not get a 422 validation error
        assert resp.status_code != 422
