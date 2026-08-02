"""Tests for OpenAIBackend profile-aware header construction."""

from __future__ import annotations

from unittest.mock import MagicMock

from augmentum.models.openai_compat import OpenAIBackend
from augmentum.models.provider_profiles import ProviderProfile


def _make_backend(
    api_key: str | None = None,
    profile: ProviderProfile | None = None,
) -> OpenAIBackend:
    client = MagicMock()
    return OpenAIBackend(
        client=client,
        base_url="https://example.com/v1",
        api_key=api_key,
        profile=profile,
    )


class TestProfiledAuthHeaders:
    def test_bearer_auth(self) -> None:
        profile = ProviderProfile(
            id="test",
            name="Test",
            base_url="https://example.com/v1",
            auth_type="bearer",
        )
        backend = _make_backend(api_key="sk-test", profile=profile)
        headers = backend._headers()
        assert headers["Authorization"] == "Bearer sk-test"
        assert headers["Content-Type"] == "application/json"

    def test_api_key_auth(self) -> None:
        profile = ProviderProfile(
            id="azure",
            name="Azure",
            base_url="https://example.com/v1",
            auth_type="api-key",
            auth_header="api-key",
        )
        backend = _make_backend(api_key="key123", profile=profile)
        headers = backend._headers()
        assert headers["api-key"] == "key123"
        assert "Authorization" not in headers

    def test_x_api_key_auth(self) -> None:
        profile = ProviderProfile(
            id="nanogpt",
            name="NanoGPT",
            base_url="https://example.com/v1",
            auth_type="x-api-key",
            auth_header="x-api-key",
        )
        backend = _make_backend(api_key="nano-key", profile=profile)
        headers = backend._headers()
        assert headers["x-api-key"] == "nano-key"
        assert "Authorization" not in headers

    def test_extra_headers_merged(self) -> None:
        profile = ProviderProfile(
            id="openrouter",
            name="OpenRouter",
            base_url="https://example.com/v1",
            auth_type="bearer",
            extra_headers={
                "HTTP-Referer": "https://augmentum.dev",
                "X-Title": "Augmentum",
            },
        )
        backend = _make_backend(api_key="sk-or", profile=profile)
        headers = backend._headers()
        assert headers["Authorization"] == "Bearer sk-or"
        assert headers["HTTP-Referer"] == "https://augmentum.dev"
        assert headers["X-Title"] == "Augmentum"

    def test_no_profile_default_bearer(self) -> None:
        backend = _make_backend(api_key="sk-default")
        headers = backend._headers()
        assert headers["Authorization"] == "Bearer sk-default"
        assert headers["Content-Type"] == "application/json"

    def test_no_api_key_no_auth_header(self) -> None:
        backend = _make_backend()
        headers = backend._headers()
        assert "Authorization" not in headers
        assert headers["Content-Type"] == "application/json"

    def test_no_api_key_with_profile_no_auth_header(self) -> None:
        profile = ProviderProfile(
            id="test",
            name="Test",
            base_url="https://example.com/v1",
            auth_type="bearer",
        )
        backend = _make_backend(profile=profile)
        headers = backend._headers()
        assert "Authorization" not in headers

    def test_profile_stored(self) -> None:
        profile = ProviderProfile(
            id="test",
            name="Test",
            base_url="https://example.com/v1",
        )
        backend = _make_backend(profile=profile)
        assert backend._profile is profile

    def test_no_profile_stored_as_none(self) -> None:
        backend = _make_backend()
        assert backend._profile is None
