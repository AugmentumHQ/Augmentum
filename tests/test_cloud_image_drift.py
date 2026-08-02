"""Cloud image-gen drift regression guards (2026-06-15).

Two HIGH-severity drift bugs found in the provider audit:
- Fal.ai URLs have NO ``/v1/`` segment (the model id carries the path) —
  the prior ``/v1/`` 404'd every fal request.
- gpt-image-* REJECTS the ``response_format`` param (400) — it always
  returns b64_json; only dall-e / generic OAI-compat accept the param.

These pin the corrected URL/payload construction so neither silently
regresses.
"""

from __future__ import annotations

import contextlib

import pytest

import augmentum.proxy.cloud_image_routes as cir
from augmentum.proxy.cloud_image_routes import (
    CloudGenerateRequest,
    _generate_fal,
    _generate_openai_compat,
)


class _FakeResp:
    status_code = 200

    def __init__(self, data):
        self._data = data
        self.content = b""

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _FakeClient:
    def __init__(self, data):
        self._data = data
        self.calls: list[dict] = []

    async def post(self, url, json=None, headers=None):
        self.calls.append({"url": url, "json": json})
        return _FakeResp(self._data)

    async def get(self, url, **kw):
        return _FakeResp({})  # unused (image urls are empty in these tests)


@pytest.fixture
def patched_client(monkeypatch):
    """Patch _cloud_client to a capturing fake; returns the captor factory."""
    holder = {}

    def install(data):
        client = _FakeClient(data)
        holder["client"] = client

        @contextlib.asynccontextmanager
        async def _cm(base_url):
            yield client

        monkeypatch.setattr(cir, "_cloud_client", _cm)
        return client

    return install


def _req(model: str) -> CloudGenerateRequest:
    return CloudGenerateRequest(prompt="a cat", model=model, width=1024, height=1024, n=1)


@pytest.mark.asyncio
async def test_fal_url_has_no_v1_segment(patched_client):
    client = patched_client({"images": [{"url": ""}]})
    await _generate_fal(
        "https://fal.run", {}, "fal-ai/flux/dev", _req("fal-ai/flux/dev")
    )
    url = client.calls[0]["url"]
    assert url == "https://fal.run/fal-ai/flux/dev"
    assert "/v1/" not in url


@pytest.mark.asyncio
async def test_gpt_image_omits_response_format(patched_client):
    client = patched_client({"data": [{"b64_json": ""}]})
    await _generate_openai_compat(
        "https://api.openai.com", {}, "gpt-image-1", _req("gpt-image-1"), "high"
    )
    assert "response_format" not in client.calls[0]["json"]


@pytest.mark.asyncio
async def test_dalle_still_sends_response_format(patched_client):
    client = patched_client({"data": [{"b64_json": ""}]})
    await _generate_openai_compat(
        "https://api.openai.com", {}, "dall-e-3", _req("dall-e-3"), "standard"
    )
    assert client.calls[0]["json"]["response_format"] == "b64_json"
