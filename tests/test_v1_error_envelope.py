"""Tests for the /v1 OpenAI-compatible error-envelope normalizer.

Verifies that errors on a ``/v1/*`` path get the OpenAI ``{"error": {...}}``
shape (so OpenAI SDK clients parse them), while non-``/v1`` paths keep
FastAPI's default ``{"detail": ...}`` shape. Self-contained — uses a tiny
app + the real ``register_openai_compat_error_handlers`` from production code.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel

from augmentum.proxy.openai_errors import (
    openai_error_type,
    register_openai_compat_error_handlers,
)


def _build_app() -> FastAPI:
    app = FastAPI()
    register_openai_compat_error_handlers(app)

    @app.get("/v1/boom")
    async def v1_boom():  # noqa: ANN202
        raise HTTPException(503, "TTS is not enabled")

    @app.get("/v1/teapot")
    async def v1_teapot():  # noqa: ANN202
        raise HTTPException(429, "Too many image jobs in progress")

    @app.get("/api/boom")
    async def api_boom():  # noqa: ANN202
        raise HTTPException(503, "TTS is not enabled")

    @app.post("/v1/needs-body")
    async def v1_body(body: _Body):  # noqa: ANN202
        return {"ok": body.x}

    return app


class _Body(BaseModel):
    x: int


@pytest.fixture
def client() -> TestClient:
    return TestClient(_build_app())


def test_v1_http_exception_uses_openai_envelope(client: TestClient) -> None:
    r = client.get("/v1/boom")
    assert r.status_code == 503
    body = r.json()
    assert isinstance(body.get("error"), dict)
    assert body["error"]["message"] == "TTS is not enabled"
    assert body["error"]["type"] == "server_error"
    assert set(body["error"]) == {"message", "type", "param", "code"}
    assert "detail" not in body  # NOT the FastAPI default shape


def test_v1_429_maps_to_rate_limit_error(client: TestClient) -> None:
    r = client.get("/v1/teapot")
    assert r.status_code == 429
    assert r.json()["error"]["type"] == "rate_limit_error"


def test_non_v1_http_exception_unchanged(client: TestClient) -> None:
    r = client.get("/api/boom")
    assert r.status_code == 503
    body = r.json()
    assert body == {"detail": "TTS is not enabled"}  # default preserved
    assert "error" not in body


def test_v1_validation_error_reshaped_to_openai(client: TestClient) -> None:
    r = client.post("/v1/needs-body", json={"x": "not-an-int"})
    assert r.status_code == 400  # reshaped from FastAPI's 422
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "x" in body["error"]["message"]
    assert "detail" not in body


def test_error_type_mapping() -> None:
    assert openai_error_type(400) == "invalid_request_error"
    assert openai_error_type(401) == "authentication_error"
    assert openai_error_type(403) == "permission_error"
    assert openai_error_type(404) == "not_found_error"
    assert openai_error_type(413) == "invalid_request_error"
    assert openai_error_type(422) == "invalid_request_error"
    assert openai_error_type(429) == "rate_limit_error"
    assert openai_error_type(500) == "server_error"
    assert openai_error_type(503) == "server_error"
