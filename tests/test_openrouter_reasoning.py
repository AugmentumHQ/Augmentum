"""OpenRouter unified ``reasoning`` control object (CORRECTIONS #4, 2026-06-25).

OpenRouter normalizes ONE ``reasoning`` object across every underlying
provider it proxies (Anthropic/DeepSeek/Qwen/GLM/Gemini/Grok). Without it,
Augmentum's reasoning control only reaches OpenAI-family model ids, so
everything else routed via OR was uncontrolled.

Schema (openrouter.ai/docs/.../reasoning-tokens): ``reasoning.effort`` is
only ``low``/``medium``/``high`` — the UI's ``minimal``/``xhigh``/``max``
must be clamped or OR rejects them. These tests pin the mapping, the
enable/disable shape, and isolation to the OpenRouter profile.
"""

from __future__ import annotations

import httpx
import pytest

from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.openai_compat import OpenAIBackend
from augmentum.models.provider_profiles import PROFILES


def _backend(profile_id: str | None) -> OpenAIBackend:
    profile = PROFILES[profile_id] if profile_id else None
    base = profile.base_url if profile else "http://localhost:8090/v1"
    return OpenAIBackend(httpx.AsyncClient(), base, None, profile=profile)


def _req(model: str, **kw) -> InternalChatRequest:
    return InternalChatRequest(
        model=model,
        messages=[Message(role="user", content="what is 2+2?")],
        **kw,
    )


def test_openrouter_profile_has_the_capability():
    assert PROFILES["openrouter"].supports_openrouter_reasoning


@pytest.mark.parametrize("ui,sent", [
    ("low", "low"), ("medium", "medium"), ("high", "high"),
    ("minimal", "low"),   # OR effort has no minimal → clamp to low
    ("xhigh", "high"),    # OR effort caps at high
    ("max", "high"),
])
def test_effort_mapped_and_clamped(ui, sent):
    payload = _backend("openrouter")._build_openai_payload(
        _req("anthropic/claude-sonnet-4-6", think=True, reasoning_effort=ui)
    )
    assert payload["reasoning"] == {"effort": sent}


def test_think_on_no_effort_enables_default():
    payload = _backend("openrouter")._build_openai_payload(
        _req("deepseek/deepseek-v4", think=True)
    )
    assert payload["reasoning"] == {"enabled": True}


def test_think_off_disables():
    payload = _backend("openrouter")._build_openai_payload(
        _req("z-ai/glm-4.7", think=False)
    )
    assert payload["reasoning"] == {"enabled": False}


def test_effort_and_max_tokens_never_both():
    # We only ever emit `effort` OR `enabled` — never alongside max_tokens
    # (OR 400s if both effort and max_tokens are present).
    payload = _backend("openrouter")._build_openai_payload(
        _req("qwen/qwen3-235b", think=True, reasoning_effort="high")
    )
    assert "max_tokens" not in payload["reasoning"]


# --- isolation: non-OpenRouter profiles never get the reasoning object ---

def test_non_openrouter_profile_emits_no_reasoning_object():
    payload = _backend("deepseek")._build_openai_payload(
        _req("deepseek-v4-flash", think=True, reasoning_effort="high")
    )
    assert "reasoning" not in payload


def test_no_profile_emits_no_reasoning_object():
    payload = _backend(None)._build_openai_payload(
        _req("anthropic/claude-sonnet-4-6", think=True)
    )
    assert "reasoning" not in payload
