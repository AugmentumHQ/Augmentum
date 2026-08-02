"""Tests for the OpenAI-shape /v1/embeddings endpoint.

The endpoint is a thin shim over the resolved backend's ``embeddings``
method. These tests pin the request validation, the response envelope
shape, and the two fallback paths (backend returns shaped data → pass
through; backend returns nothing → 501).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from augmentum.proxy.openai_routes import OpenAIEmbeddingsRequest

# ----------------------------------------------------------------------
# Request schema
# ----------------------------------------------------------------------


def test_request_accepts_string_input():
    r = OpenAIEmbeddingsRequest(model="m", input="hi")
    assert r.input == "hi"


def test_request_accepts_list_input():
    r = OpenAIEmbeddingsRequest(model="m", input=["a", "b"])
    assert r.input == ["a", "b"]


def test_request_accepts_sdk_compat_fields():
    # encoding_format / dimensions / user are OpenAI SDK fields we
    # accept-but-ignore. Schema must validate without error.
    r = OpenAIEmbeddingsRequest(
        model="m", input="hi",
        encoding_format="float", dimensions=512, user="alice",
    )
    assert r.dimensions == 512


def test_request_rejects_missing_model():
    with pytest.raises(Exception):
        OpenAIEmbeddingsRequest(input="hi")  # type: ignore[call-arg]


# ----------------------------------------------------------------------
# Endpoint behaviour
# ----------------------------------------------------------------------


def _build_app(backend):
    """Minimal FastAPI app with the openai_router + a fake registry."""
    from fastapi import FastAPI

    from augmentum.proxy.openai_routes import router

    app = FastAPI()
    app.include_router(router)

    # Fake provider registry that always resolves to the supplied backend.
    registry = MagicMock()
    registry.resolve_backend_with_fabric = AsyncMock(return_value=(backend, "test-model"))
    app.state.provider_registry = registry
    app.state.http_client = MagicMock()
    return app


def test_endpoint_passes_through_backend_openai_shape():
    """When the backend returns an OpenAI-shaped envelope, pass through."""
    backend = MagicMock()
    backend.embeddings = AsyncMock(return_value={
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
        "model": "test-model",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    })
    app = _build_app(backend)
    with TestClient(app) as client:
        r = client.post("/v1/embeddings", json={"model": "test-model", "input": "hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert body["usage"]["prompt_tokens"] == 5


def test_endpoint_handles_list_input():
    backend = MagicMock()
    backend.embeddings = AsyncMock(return_value={
        "object": "list",
        "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1]},
            {"object": "embedding", "index": 1, "embedding": [0.2]},
        ],
        "model": "test-model",
    })
    app = _build_app(backend)
    with TestClient(app) as client:
        r = client.post("/v1/embeddings",
                        json={"model": "test-model", "input": ["a", "b"]})
    assert r.status_code == 200
    assert len(r.json()["data"]) == 2


def _break_internal_embedder(monkeypatch):
    """Make the internal-embedder fallback unavailable so tests can pin
    the terminal error paths."""
    from augmentum.memory.embeddings import EmbeddingService
    monkeypatch.setattr(
        EmbeddingService, "embed",
        classmethod(lambda cls, texts: (_ for _ in ()).throw(RuntimeError("embedder down"))),
    )


def _stub_internal_embedder(monkeypatch):
    """Deterministic internal embedder — no model download in unit tests."""
    from augmentum.memory.embeddings import EmbeddingService
    monkeypatch.setattr(
        EmbeddingService, "embed",
        classmethod(lambda cls, texts: [[0.5, 0.25, 0.125] for _ in texts]),
    )


def test_endpoint_returns_501_when_no_backend_supports_embeddings(monkeypatch):
    """No backend embeddings, no Ollama, internal embedder down → 501."""
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "ollama_base_url", "")
    _break_internal_embedder(monkeypatch)

    backend = MagicMock(spec=[])  # no embeddings attribute
    app = _build_app(backend)
    with TestClient(app) as client:
        r = client.post("/v1/embeddings", json={"model": "test-model", "input": "hi"})
    assert r.status_code == 501
    err = r.json()["error"]
    assert err["type"] == "embeddings_unavailable"
    assert "test-model" in err["model"]


def test_endpoint_serves_internal_fallback_when_no_backend(monkeypatch):
    """No backend embeddings, no Ollama → the internal nomic embedder
    serves the request (the default-install path for Cursor/LangChain)."""
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "ollama_base_url", "")
    _stub_internal_embedder(monkeypatch)

    backend = MagicMock(spec=[])  # no embeddings attribute
    app = _build_app(backend)
    with TestClient(app) as client:
        r = client.post("/v1/embeddings", json={"model": "test-model", "input": ["a", "b"]})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 2
    assert body["data"][1]["index"] == 1
    assert body["data"][0]["embedding"] == [0.5, 0.25, 0.125]
    # Reports the TRUE producing model, not the requested alias.
    from augmentum.memory.embeddings import EmbeddingService
    assert body["model"] == EmbeddingService.MODEL_NAME


def test_endpoint_returns_400_on_unavailable_model(monkeypatch):
    """Unknown model AND internal embedder down → the original 400."""
    from augmentum.models.provider_registry import ModelUnavailableError
    _break_internal_embedder(monkeypatch)

    backend = MagicMock()
    app = _build_app(backend)
    app.state.provider_registry.resolve_backend_with_fabric = AsyncMock(
        side_effect=ModelUnavailableError(
            "model ghost not found",
            model="ghost", peer_diagnostic={"connected_peers": []},
        ),
    )
    with TestClient(app) as client:
        r = client.post("/v1/embeddings", json={"model": "ghost", "input": "hi"})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "model_unavailable"


def test_endpoint_serves_internal_fallback_on_unknown_model(monkeypatch):
    """SDK-default model names ("text-embedding-3-small") that resolve to
    nothing get served by the internal embedder instead of a 400."""
    from augmentum.models.provider_registry import ModelUnavailableError
    _stub_internal_embedder(monkeypatch)

    backend = MagicMock()
    app = _build_app(backend)
    app.state.provider_registry.resolve_backend_with_fabric = AsyncMock(
        side_effect=ModelUnavailableError(
            "model not found",
            model="text-embedding-3-small", peer_diagnostic={"connected_peers": []},
        ),
    )
    with TestClient(app) as client:
        r = client.post(
            "/v1/embeddings",
            json={"model": "text-embedding-3-small", "input": "hi"},
        )
    assert r.status_code == 200
    assert len(r.json()["data"]) == 1


def test_endpoint_falls_through_when_backend_raises(monkeypatch):
    """If backend.embeddings raises, we fall through Ollama (unset) to the
    internal embedder; with that down too → 501."""
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "ollama_base_url", "")
    _break_internal_embedder(monkeypatch)

    backend = MagicMock()
    backend.embeddings = AsyncMock(side_effect=RuntimeError("backend exploded"))
    app = _build_app(backend)
    with TestClient(app) as client:
        r = client.post("/v1/embeddings", json={"model": "test-model", "input": "hi"})
    assert r.status_code == 501


def test_internal_fallback_rejects_token_array_input(monkeypatch):
    """OpenAI allows token-array inputs; the internal fallback can't embed
    those — it must decline (→ 501) rather than crash."""
    from augmentum.config import settings as _settings
    monkeypatch.setattr(_settings, "ollama_base_url", "")
    _stub_internal_embedder(monkeypatch)

    backend = MagicMock(spec=[])
    app = _build_app(backend)
    with TestClient(app) as client:
        r = client.post(
            "/v1/embeddings",
            json={"model": "test-model", "input": [[1, 2, 3]]},
        )
    # Pydantic may reject the shape outright (422) or the fallback
    # declines (501) — either way, never a crash or fake vectors.
    assert r.status_code in (422, 501)
