"""Azure OpenAI chat-completions URL: mandatory api-version query param.

CORRECTIONS #11 — Azure requires ``?api-version=YYYY-MM-DD`` on every call;
the deployment path comes from the user's base_url but the version param was
never injected → bare request 400s. Non-Azure profiles are untouched, and a
version already encoded in base_url is not overridden.
"""

from __future__ import annotations

import httpx

from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.openai_compat import OpenAIBackend
from augmentum.models.provider_profiles import PROFILES

_AZURE_BASE = "https://myres.openai.azure.com/openai/deployments/gpt-5"


def _backend(profile_id: str, base_url: str) -> OpenAIBackend:
    return OpenAIBackend(httpx.AsyncClient(), base_url, None, profile=PROFILES[profile_id])


def _req() -> InternalChatRequest:
    return InternalChatRequest(
        model="gpt-5", messages=[Message(role="user", content="hi")]
    )


def test_azure_appends_api_version():
    url = _backend("azure", _AZURE_BASE)._chat_url(_req())
    assert url.startswith(f"{_AZURE_BASE}/chat/completions?api-version=")


def test_azure_uses_default_version():
    url = _backend("azure", _AZURE_BASE)._chat_url(_req())
    assert "api-version=2024-02-01" in url


def test_azure_honors_version_already_in_base_url():
    # User encoded a per-deployment version in base_url → don't override/duplicate.
    b = _backend("azure", f"{_AZURE_BASE}?api-version=2025-01-01")
    url = b._chat_url(_req())
    assert url.count("api-version=") == 1
    assert "api-version=2025-01-01" in url


def test_non_azure_url_has_no_api_version():
    url = _backend("openai", "https://api.openai.com/v1")._chat_url(_req())
    assert url == "https://api.openai.com/v1/chat/completions"
    assert "api-version" not in url


def test_azure_query_separator_when_base_has_query():
    b = _backend("azure", f"{_AZURE_BASE}?foo=bar")
    url = b._chat_url(_req())
    assert "?foo=bar" in url
    assert "&api-version=" in url
