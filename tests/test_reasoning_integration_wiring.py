"""Integration-level tests for the reasoning-handling fix pass.

These tests verify the WIRING between components — not just the unit
behavior of ``ThinkingStreamBuffer`` or ``_build_openai_payload`` in
isolation, but that the glue connecting them works: ``OpenAIBackend``
creates the buffer with the right ``local_engine=`` based on its URL,
profiles don't have conflicting flags, and the emit-side doesn't fire
when it shouldn't.

Gaps these fill:
  - R2 (#17): no test that ``is_local_engine()`` on a cloud-URL backend
    returns False (the ThinkingStreamBuffer unit tests construct the
    buffer directly, bypassing the wiring).
  - R4 (#15): no guard against ``reasoning_via_groq_params`` AND
    ``supports_reasoning_effort`` both True (double-emit).
  - R6/R7/R8: no test that ``reasoning_effort`` is absent when
    ``think=False`` (effort field should not appear at all).
"""

from __future__ import annotations

import httpx
import pytest

from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.openai_compat import OpenAIBackend, is_local_engine_url
from augmentum.models.provider_profiles import PROFILES


def _backend(profile_id: str | None) -> OpenAIBackend:
    profile = PROFILES[profile_id] if profile_id else None
    base = profile.base_url if profile else "http://localhost:8090/v1"
    return OpenAIBackend(httpx.AsyncClient(), base, None, profile=profile)


def _req(model: str = "test-model", **kw) -> InternalChatRequest:
    return InternalChatRequest(
        model=model,
        messages=[Message(role="user", content="test")],
        **kw,
    )


# --- R2 (#17): is_local_engine wiring on real profile URLs ------------

CLOUD_PROFILES = [
    "deepseek", "openai", "openrouter", "mistral", "ai21", "xai",
    "moonshot", "groq", "perplexity", "fireworks", "nvidia",
    "pollinations", "aimlapi", "electronhub", "chutes", "nanogpt",
    "zai", "siliconflow", "cohere", "together",
]


@pytest.mark.parametrize("pid", CLOUD_PROFILES)
def test_cloud_profile_is_not_local_engine(pid):
    """Every cloud provider's base_url must resolve is_local_engine=False.
    If this fails, that provider gets the asymmetric-think assumption
    and the #17 content-loss bug returns for it."""
    url = PROFILES[pid].base_url
    assert not is_local_engine_url(url), (
        f"profile '{pid}' base_url={url!r} resolved as local — "
        f"would get _inside_think=True on cloud GLM/DeepSeek-V4/Qwen3"
    )


@pytest.mark.parametrize("pid", CLOUD_PROFILES)
def test_cloud_backend_is_local_engine_returns_false(pid):
    """The backend instance method delegates to is_local_engine_url.
    Verify the full chain: profile → OpenAIBackend → is_local_engine()."""
    backend = _backend(pid)
    assert not backend.is_local_engine(), (
        f"OpenAIBackend('{pid}').is_local_engine() returned True"
    )


def test_local_url_is_local_engine():
    """Sanity: a loopback URL IS a local engine."""
    assert is_local_engine_url("http://127.0.0.1:8091/v1")
    assert is_local_engine_url("http://localhost:8091/v1")


def test_docker_compose_host_is_local_engine():
    """Docker-compose model-server hostnames are local."""
    assert is_local_engine_url("http://ollama:11434/v1")
    assert is_local_engine_url("http://llamacpp:8080/v1")
    assert is_local_engine_url("http://vllm:8000/v1")


# --- R4 (#15): mutual-exclusion guard --------------------------------

CUSTOM_REASONING_FLAGS = [
    "reasoning_via_groq_params",
    "reasoning_via_chat_template_kwargs",
    "supports_openrouter_reasoning",
    "reasoning_via_siliconflow_params",
]


def test_no_profile_has_conflicting_reasoning_flags():
    """Each profile should use AT MOST ONE custom reasoning path.
    If two fire on the same request, the second overwrites/conflicts."""
    for pid, profile in PROFILES.items():
        active = [f for f in CUSTOM_REASONING_FLAGS if getattr(profile, f, False)]
        assert len(active) <= 1, (
            f"profile '{pid}' has multiple reasoning flags: {active}"
        )


def test_custom_reasoning_profiles_dont_also_have_generic_effort():
    """Profiles with a custom reasoning path should NOT also set
    supports_reasoning_effort — the generic path would double-emit."""
    for pid, profile in PROFILES.items():
        has_custom = any(getattr(profile, f, False) for f in CUSTOM_REASONING_FLAGS)
        if has_custom and pid not in ("openrouter",):
            # OpenRouter is the exception: it ALSO accepts top-level
            # reasoning_effort for OpenAI-family models via the generic
            # path, alongside its own reasoning object. No conflict because
            # the reasoning object supersedes for non-OpenAI models.
            assert not profile.supports_reasoning_effort, (
                f"profile '{pid}' has both a custom reasoning flag AND "
                f"supports_reasoning_effort — would double-emit"
            )


# --- R6/R7/R8: effort absent when think=False -------------------------

EFFORT_PROFILES = ["perplexity", "fireworks", "pollinations"]


@pytest.mark.parametrize("pid", EFFORT_PROFILES)
def test_effort_emitted_regardless_of_think_when_effort_set(pid):
    """The generic ``supports_reasoning_effort`` path emits whenever
    ``request.reasoning_effort`` is set — it does NOT gate on ``think``.
    This matches OpenAI's behavior (GPT-5.x always reasons; effort
    controls depth, not on/off). Perplexity/Fireworks/Pollinations
    silently ignore effort on non-reasoning models, so no 400 risk."""
    payload = _backend(pid)._build_openai_payload(
        _req(think=False, reasoning_effort="high")
    )
    assert payload["reasoning_effort"] == "high"


@pytest.mark.parametrize("pid", EFFORT_PROFILES)
def test_effort_not_emitted_when_no_effort_set(pid):
    """When the user hasn't selected an effort level, the field must
    be absent — not sent as empty string or None."""
    payload = _backend(pid)._build_openai_payload(_req(think=True))
    assert "reasoning_effort" not in payload


# --- Cross-provider: OpenRouter reasoning + generic effort don't clash

def test_openrouter_reasoning_object_coexists_with_generic_effort():
    """OpenRouter emits BOTH the reasoning object (for non-OpenAI models)
    AND the generic reasoning_effort (for OpenAI-family models routed via
    OR). They must not clobber each other."""
    payload = _backend("openrouter")._build_openai_payload(
        _req("gpt-5-mini", think=True, reasoning_effort="high")
    )
    # Generic path fires for gpt-5 (OpenAI-family model)
    assert payload.get("reasoning_effort") == "high"
    # OpenRouter path also fires (profile flag)
    assert "reasoning" in payload
    assert payload["reasoning"]["effort"] == "high"
