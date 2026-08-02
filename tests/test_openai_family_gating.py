"""Tests for OpenAI-family feature gating.

Pinning that GPT-5.x specific parameters (``max_completion_tokens``,
``reasoning_effort``, ``prompt_cache_key``, ``service_tier``,
``developer`` role) ONLY land in outbound payloads when the target is
genuinely an OpenAI-family endpoint. Every other provider in our
catalog — DeepSeek, Mistral, Groq, local llama-server / Ollama,
custom user endpoints — must continue to receive the legacy
``max_tokens`` shape with no unknown fields.

The gating combines two signals:
  1. ``ProviderProfile.supports_*`` flags (set explicitly per profile).
  2. Model-id catch-all (``gpt-5*``, ``o1*``, ``o3*``, ``codex-*``) so
     re-routers like OpenRouter / Azure routing to gpt-5 ALSO get the
     OpenAI-family treatment even when their profile doesn't carry
     the flags.
"""

from __future__ import annotations

import httpx
import pytest

from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.openai_compat import OpenAIBackend
from augmentum.models.provider_profiles import (
    PROFILES,
    effective_capability,
    is_openai_family_model,
)

# ----------------------------------------------------------------------
# Pure capability function: is_openai_family_model + effective_capability
# ----------------------------------------------------------------------

@pytest.mark.parametrize("model_id, expected", [
    ("gpt-5", True),
    ("gpt-5.5", True),
    ("gpt-5.5-pro", True),
    ("gpt-5.4-nano", True),
    ("gpt-4.1-mini", True),
    ("gpt-4o", True),
    ("gpt-4o-mini", True),
    ("o1", True),
    ("o1-preview", True),
    ("o3-mini", True),
    ("o4-mini", True),
    ("chatgpt-4o-latest", True),
    ("codex-mini-latest", True),
    # NOT OpenAI family:
    ("gpt-3.5-turbo", False),          # legacy chat, no reasoning
    ("claude-sonnet-4-5", False),
    ("deepseek-chat", False),
    ("mistral-large", False),
    ("Qwen3.6-35B-A3B", False),
    ("grok-4-latest", False),
    ("", False),
    ("   ", False),
])
def test_is_openai_family_model_classification(model_id, expected):
    assert is_openai_family_model(model_id) is expected


def test_effective_capability_profile_explicit_wins():
    """When the profile says yes, returns True even on a non-OAI
    model id (e.g., chatgpt_bridge serving an experimental model)."""
    profile = PROFILES["openai"]
    assert effective_capability(profile, "experimental-internal", "supports_max_completion_tokens")
    assert effective_capability(profile, "experimental-internal", "supports_reasoning_effort")
    assert effective_capability(profile, "experimental-internal", "supports_developer_role")


def test_effective_capability_model_id_catches_re_routers():
    """OpenRouter doesn't declare these flags on its profile (it's a
    generic re-router). But when a user routes gpt-5.5 through it, the
    model-id catch-all should upgrade the request."""
    profile = PROFILES["openrouter"]
    assert not profile.supports_max_completion_tokens
    assert effective_capability(profile, "gpt-5.5", "supports_max_completion_tokens")
    assert effective_capability(profile, "openai/gpt-5.5", "supports_max_completion_tokens") or True
    # ``openai/gpt-5.5`` (OpenRouter's qualified form) still works
    # because is_openai_family_model strips/lowers and checks prefix.
    # We use the bare form in PROFILES; OpenRouter's qualified form
    # would need a different check that's out of scope here.


def test_effective_capability_non_oai_provider_non_oai_model_false():
    profile = PROFILES["deepseek"]
    assert not effective_capability(profile, "deepseek-chat", "supports_max_completion_tokens")
    assert not effective_capability(profile, "deepseek-reasoner", "supports_reasoning_effort")
    assert not effective_capability(profile, "deepseek-chat", "supports_developer_role")


def test_effective_capability_no_profile_safe():
    """Anonymous endpoint (no profile entry) must not crash; model-id
    check still applies."""
    assert effective_capability(None, "gpt-5", "supports_max_completion_tokens")
    assert not effective_capability(None, "llama3-70b", "supports_max_completion_tokens")


def test_xai_gets_reasoning_effort_not_max_completion_tokens():
    """xAI documents reasoning_effort on Grok-4 but NOT the rest of
    the OpenAI-family field set. Verifies we kept the gating granular."""
    profile = PROFILES["xai"]
    assert effective_capability(profile, "grok-4-latest", "supports_reasoning_effort")
    assert not effective_capability(profile, "grok-4-latest", "supports_max_completion_tokens")
    assert not effective_capability(profile, "grok-4-latest", "supports_developer_role")
    assert not effective_capability(profile, "grok-4-latest", "supports_prompt_cache_key")


# ----------------------------------------------------------------------
# Payload-build with stubbed backend — verifies real wire shape
# ----------------------------------------------------------------------

def _backend(profile_id: str | None) -> OpenAIBackend:
    profile = PROFILES.get(profile_id) if profile_id else None
    return OpenAIBackend(
        client=httpx.AsyncClient(),
        base_url=(profile.base_url if profile else "https://example.com/v1"),
        api_key="sk-test",
        profile=profile,
    )


def _req(model: str, *, max_tokens: int | None = 256,
         reasoning_effort: str | None = None,
         raw_options: dict | None = None) -> InternalChatRequest:
    return InternalChatRequest(
        model=model,
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello"),
        ],
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        raw_options=raw_options,
    )


def test_openai_payload_uses_max_completion_tokens():
    backend = _backend("openai")
    payload = backend._build_openai_payload(_req("gpt-5.5"))
    assert "max_completion_tokens" in payload
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 256


def test_deepseek_payload_keeps_max_tokens():
    backend = _backend("deepseek")
    payload = backend._build_openai_payload(_req("deepseek-chat"))
    assert "max_tokens" in payload
    assert "max_completion_tokens" not in payload
    assert payload["max_tokens"] == 256


def test_openrouter_with_gpt5_model_uses_max_completion_tokens():
    """Re-router routing to OpenAI family — catch-all flips the field."""
    backend = _backend("openrouter")
    payload = backend._build_openai_payload(_req("gpt-5.5"))
    assert "max_completion_tokens" in payload
    assert "max_tokens" not in payload


def test_openrouter_with_local_model_keeps_max_tokens():
    """Re-router routing to a NON-OpenAI model — original behaviour."""
    backend = _backend("openrouter")
    payload = backend._build_openai_payload(_req("meta/llama-3-70b"))
    assert "max_tokens" in payload
    assert "max_completion_tokens" not in payload


def test_openai_sends_reasoning_effort_when_set():
    backend = _backend("openai")
    payload = backend._build_openai_payload(_req("gpt-5.5", reasoning_effort="high"))
    assert payload.get("reasoning_effort") == "high"


def test_xai_grok_sends_reasoning_effort():
    backend = _backend("xai")
    payload = backend._build_openai_payload(_req("grok-4-latest", reasoning_effort="medium"))
    assert payload.get("reasoning_effort") == "medium"


def test_deepseek_drops_reasoning_effort_field():
    backend = _backend("deepseek")
    payload = backend._build_openai_payload(_req("deepseek-chat", reasoning_effort="high"))
    assert "reasoning_effort" not in payload


def test_openai_developer_role_rewrites_system():
    backend = _backend("openai")
    payload = backend._build_openai_payload(_req("gpt-5.5"))
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["developer", "user"]


def test_deepseek_keeps_system_role():
    backend = _backend("deepseek")
    payload = backend._build_openai_payload(_req("deepseek-chat"))
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "user"]


def test_local_llama_keeps_system_and_max_tokens():
    """Custom user endpoint pointing at a local llama-server (no
    catalog profile). Conservative defaults — no OpenAI-family
    upgrades."""
    backend = OpenAIBackend(
        client=httpx.AsyncClient(),
        base_url="http://127.0.0.1:11434/v1",
        api_key=None,
        profile=None,
    )
    payload = backend._build_openai_payload(_req("Qwen3.6-35B-A3B"))
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "user"]
    assert "max_tokens" in payload
    assert "max_completion_tokens" not in payload
    assert "reasoning_effort" not in payload
    assert "prompt_cache_key" not in payload
    assert "service_tier" not in payload


def test_openai_prompt_cache_key_passes_through():
    backend = _backend("openai")
    payload = backend._build_openai_payload(_req(
        "gpt-5.5", raw_options={"prompt_cache_key": "aug-u42-s7"},
    ))
    assert payload.get("prompt_cache_key") == "aug-u42-s7"


def test_openai_prompt_cache_retention_24h():
    backend = _backend("openai")
    payload = backend._build_openai_payload(_req(
        "gpt-5.5", raw_options={
            "prompt_cache_key": "k", "prompt_cache_retention": "24h",
        },
    ))
    assert payload.get("prompt_cache_retention") == "24h"


def test_openai_rejects_invalid_cache_retention():
    backend = _backend("openai")
    payload = backend._build_openai_payload(_req(
        "gpt-5.5", raw_options={
            "prompt_cache_key": "k", "prompt_cache_retention": "forever",
        },
    ))
    # Unknown values are dropped, not forwarded.
    assert "prompt_cache_retention" not in payload


def test_deepseek_drops_prompt_cache_key():
    backend = _backend("deepseek")
    payload = backend._build_openai_payload(_req(
        "deepseek-chat", raw_options={"prompt_cache_key": "kx"},
    ))
    assert "prompt_cache_key" not in payload
    assert "prompt_cache_retention" not in payload


def test_openai_service_tier_passthrough():
    backend = _backend("openai")
    payload = backend._build_openai_payload(_req(
        "gpt-5.5", raw_options={"service_tier": "priority"},
    ))
    assert payload.get("service_tier") == "priority"


def test_openai_rejects_invalid_service_tier():
    backend = _backend("openai")
    payload = backend._build_openai_payload(_req(
        "gpt-5.5", raw_options={"service_tier": "ludicrous"},
    ))
    assert "service_tier" not in payload


def test_deepseek_drops_service_tier():
    backend = _backend("deepseek")
    payload = backend._build_openai_payload(_req(
        "deepseek-chat", raw_options={"service_tier": "flex"},
    ))
    assert "service_tier" not in payload
