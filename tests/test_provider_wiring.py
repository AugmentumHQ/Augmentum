"""Tests for provider registry factory function and wiring."""

from __future__ import annotations


class TestCreateBackendFromProfile:
    def test_create_openai_from_profile(self):
        from augmentum.models.provider_profiles import get_profile
        from augmentum.models.provider_registry import create_backend_from_profile

        profile = get_profile("groq")
        backend = create_backend_from_profile(
            profile, api_key="test-key", http_client=None
        )
        assert backend is not None
        assert backend._base_url == "https://api.groq.com/openai/v1"

    def test_create_claude_backend(self):
        from augmentum.models.adapters.claude import ClaudeBackend
        from augmentum.models.provider_registry import create_backend_from_profile

        backend = create_backend_from_profile(
            None, api_key="test", http_client=None, provider_type="claude"
        )
        assert isinstance(backend, ClaudeBackend)

    def test_create_gemini_backend(self):
        from augmentum.models.adapters.gemini import GeminiBackend
        from augmentum.models.provider_registry import create_backend_from_profile

        backend = create_backend_from_profile(
            None, api_key="test", http_client=None, provider_type="gemini"
        )
        assert isinstance(backend, GeminiBackend)

    def test_profile_injected_into_openai_backend(self):
        from augmentum.models.provider_profiles import get_profile
        from augmentum.models.provider_registry import create_backend_from_profile

        profile = get_profile("openrouter")
        backend = create_backend_from_profile(
            profile, api_key="test", http_client=None
        )
        assert backend._profile is not None
        assert backend._profile.id == "openrouter"

    def test_custom_base_url_overrides_profile(self):
        from augmentum.models.provider_profiles import get_profile
        from augmentum.models.provider_registry import create_backend_from_profile

        profile = get_profile("groq")
        backend = create_backend_from_profile(
            profile,
            api_key="test",
            http_client=None,
            base_url="http://custom:8080",
        )
        assert backend._base_url == "http://custom:8080"

    def test_claude_default_base_url(self):
        from augmentum.models.provider_registry import create_backend_from_profile

        backend = create_backend_from_profile(
            None, api_key="test", http_client=None, provider_type="claude"
        )
        assert backend._base_url == "https://api.anthropic.com/v1"

    def test_claude_custom_base_url(self):
        from augmentum.models.provider_registry import create_backend_from_profile

        backend = create_backend_from_profile(
            None,
            api_key="test",
            http_client=None,
            provider_type="claude",
            base_url="http://localhost:9090",
        )
        assert backend._base_url == "http://localhost:9090"

    def test_gemini_default_base_url(self):
        from augmentum.models.provider_registry import create_backend_from_profile

        backend = create_backend_from_profile(
            None, api_key="test", http_client=None, provider_type="gemini"
        )
        assert backend._base_url == "https://generativelanguage.googleapis.com"

    def test_no_profile_no_type_returns_openai(self):
        from augmentum.models.openai_compat import OpenAIBackend
        from augmentum.models.provider_registry import create_backend_from_profile

        backend = create_backend_from_profile(
            None, api_key="test", http_client=None, base_url="http://localhost:1234"
        )
        assert isinstance(backend, OpenAIBackend)
        assert backend._base_url == "http://localhost:1234"
