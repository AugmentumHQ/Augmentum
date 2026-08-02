"""Tests for ``POST /api/ui/translate-card``.

Pins:

- Empty/whitespace-only fields → 400.
- No backend → 503.
- LLM returns non-JSON → 502.
- LLM returns non-dict JSON → 502.
- ``preview=true`` echoes source fields alongside ``translated``.
- ``preview=false`` returns only ``translated``.
- ``source_language`` flows into the system prompt.
- ``model`` body param overrides resolver default.
- ```json fences are stripped.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from augmentum.models.base import InternalChatResponse, Message


def _wire_translate_backend(app, mock_backend, raw_content: str):
    """Wire ``resolve_model_for_role`` so the route's backend.chat returns
    ``raw_content`` verbatim. Returns the resolver mock so callers can
    inspect call args (model override, etc.)."""

    async def _chat(request):
        return InternalChatResponse(
            message=Message(role="assistant", content=raw_content),
            model=request.model,
            finish_reason="stop",
        )

    mock_backend.chat = _chat
    resolver = AsyncMock(return_value=(mock_backend, "test-utility-model"))
    app.state.provider_registry.resolve_model_for_role = resolver
    return resolver


class TestEmptyOrInvalidInput:
    def test_empty_fields_returns_400(self, sqlite_client):
        resp = sqlite_client.post(
            "/api/ui/translate-card",
            json={"fields": {}, "target_language": "English"},
        )
        assert resp.status_code == 400
        assert "No fields" in resp.json()["error"]

    def test_whitespace_only_fields_returns_400(self, sqlite_client):
        resp = sqlite_client.post(
            "/api/ui/translate-card",
            json={"fields": {"name": "   ", "desc": "\n\t"}},
        )
        assert resp.status_code == 400

    def test_no_backend_returns_503(self, sqlite_client, app):
        # The shared app fixture wires a single ollama backend; collapse it
        # to empty so the route hits the no-backend branch.
        app.state.provider_registry.backends = {}
        resp = sqlite_client.post(
            "/api/ui/translate-card",
            json={"fields": {"name": "Alice"}},
        )
        assert resp.status_code == 503


class TestSuccessfulTranslation:
    def test_preview_true_echoes_source(self, sqlite_client, app, mock_backend):
        _wire_translate_backend(
            app, mock_backend, '{"name": "Alicia", "description": "Una guerrera"}',
        )
        resp = sqlite_client.post(
            "/api/ui/translate-card",
            json={
                "fields": {"name": "Alice", "description": "A warrior"},
                "target_language": "Spanish",
                "preview": True,
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["translated"] == {"name": "Alicia", "description": "Una guerrera"}
        assert body["source"] == {"name": "Alice", "description": "A warrior"}

    def test_preview_false_omits_source(self, sqlite_client, app, mock_backend):
        _wire_translate_backend(
            app, mock_backend, '{"name": "Alicia"}',
        )
        resp = sqlite_client.post(
            "/api/ui/translate-card",
            json={
                "fields": {"name": "Alice"},
                "target_language": "Spanish",
                "preview": False,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["translated"] == {"name": "Alicia"}
        assert "source" not in body

    def test_json_fence_stripped(self, sqlite_client, app, mock_backend):
        fenced = '```json\n{"name": "Wrapped"}\n```'
        _wire_translate_backend(app, mock_backend, fenced)
        resp = sqlite_client.post(
            "/api/ui/translate-card",
            json={"fields": {"name": "Original"}, "preview": False},
        )
        assert resp.status_code == 200
        assert resp.json()["translated"] == {"name": "Wrapped"}


class TestLanguageHints:
    def test_source_language_in_system_prompt(self, sqlite_client, app, mock_backend):
        captured: dict = {}

        async def _chat(request):
            captured["system"] = request.messages[0].content
            captured["user"] = request.messages[1].content
            return InternalChatResponse(
                message=Message(role="assistant", content='{"name": "Alicia"}'),
                model=request.model,
                finish_reason="stop",
            )

        mock_backend.chat = _chat
        app.state.provider_registry.resolve_model_for_role = AsyncMock(
            return_value=(mock_backend, "test-model"),
        )

        resp = sqlite_client.post(
            "/api/ui/translate-card",
            json={
                "fields": {"name": "Alice"},
                "target_language": "Spanish",
                "source_language": "English",
                "preview": False,
            },
        )
        assert resp.status_code == 200
        assert "from English into Spanish" in captured["system"]
        assert "from English to Spanish" in captured["user"]

    def test_no_source_language_uses_autodetect_prompt(
        self, sqlite_client, app, mock_backend,
    ):
        captured: dict = {}

        async def _chat(request):
            captured["system"] = request.messages[0].content
            return InternalChatResponse(
                message=Message(role="assistant", content='{"name": "Alice"}'),
                model=request.model,
                finish_reason="stop",
            )

        mock_backend.chat = _chat
        app.state.provider_registry.resolve_model_for_role = AsyncMock(
            return_value=(mock_backend, "test-model"),
        )

        resp = sqlite_client.post(
            "/api/ui/translate-card",
            json={"fields": {"name": "Alice"}, "preview": False},
        )
        assert resp.status_code == 200
        # Auto-detect template includes "Auto-detect"
        assert "Auto-detect" in captured["system"]
        assert "already in English" in captured["system"]


class TestModelOverride:
    def test_model_body_param_passes_to_resolver(
        self, sqlite_client, app, mock_backend,
    ):
        resolver = _wire_translate_backend(
            app, mock_backend, '{"name": "X"}',
        )
        resp = sqlite_client.post(
            "/api/ui/translate-card",
            json={
                "fields": {"name": "Alice"},
                "model": "user-picked-model",
                "preview": False,
            },
        )
        assert resp.status_code == 200
        resolver.assert_called_once()
        kwargs = resolver.call_args.kwargs
        assert kwargs.get("override") == "user-picked-model"


class TestLlmErrorPaths:
    def test_invalid_json_returns_502(self, sqlite_client, app, mock_backend):
        _wire_translate_backend(app, mock_backend, "not valid json at all")
        resp = sqlite_client.post(
            "/api/ui/translate-card",
            json={"fields": {"name": "Alice"}, "preview": False},
        )
        assert resp.status_code == 502
        assert "not valid JSON" in resp.json()["error"]

    def test_non_dict_json_returns_502(self, sqlite_client, app, mock_backend):
        _wire_translate_backend(app, mock_backend, '["a", "list"]')
        resp = sqlite_client.post(
            "/api/ui/translate-card",
            json={"fields": {"name": "Alice"}, "preview": False},
        )
        assert resp.status_code == 502
        assert "invalid format" in resp.json()["error"]

    def test_backend_exception_returns_502(self, sqlite_client, app, mock_backend):
        async def _boom(_req):
            raise RuntimeError("provider timeout")

        mock_backend.chat = _boom
        app.state.provider_registry.resolve_model_for_role = AsyncMock(
            return_value=(mock_backend, "test-model"),
        )
        resp = sqlite_client.post(
            "/api/ui/translate-card",
            json={"fields": {"name": "Alice"}, "preview": False},
        )
        assert resp.status_code == 502
        assert "Translation failed" in resp.json()["error"]
