"""Reasoning-effort emit for Perplexity (#18), Fireworks (#20), Pollinations (#22),
DeepSeek nested (#2), SiliconFlow (#24). 2026-06-25.

Perplexity/Fireworks/Pollinations use the generic ``supports_reasoning_effort``
path (top-level ``reasoning_effort``). DeepSeek nests effort inside
``thinking:{type, reasoning_effort}``. SiliconFlow uses
``enable_thinking`` + ``thinking_budget``.
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


def _req(model: str = "test-model", **kw) -> InternalChatRequest:
    return InternalChatRequest(
        model=model,
        messages=[Message(role="user", content="what is 2+2?")],
        **kw,
    )


# --- Flag flips: Perplexity / Fireworks / Pollinations ----------------

EFFORT_FLAG_PROFILES = ["perplexity", "fireworks", "pollinations"]


@pytest.mark.parametrize("pid", EFFORT_FLAG_PROFILES)
def test_effort_flag_on(pid):
    assert PROFILES[pid].supports_reasoning_effort


@pytest.mark.parametrize("pid", EFFORT_FLAG_PROFILES)
def test_effort_emitted(pid):
    payload = _backend(pid)._build_openai_payload(
        _req(think=True, reasoning_effort="high")
    )
    assert payload["reasoning_effort"] == "high"


@pytest.mark.parametrize("pid", EFFORT_FLAG_PROFILES)
def test_no_effort_no_field(pid):
    payload = _backend(pid)._build_openai_payload(_req(think=True))
    assert "reasoning_effort" not in payload


def test_fireworks_minimal_demoted_to_low():
    # Fireworks has no minimal → supports_reasoning_effort_minimal=False
    assert not PROFILES["fireworks"].supports_reasoning_effort_minimal
    payload = _backend("fireworks")._build_openai_payload(
        _req(think=True, reasoning_effort="minimal")
    )
    assert payload["reasoning_effort"] == "low"


def test_perplexity_accepts_minimal():
    assert PROFILES["perplexity"].supports_reasoning_effort_minimal
    payload = _backend("perplexity")._build_openai_payload(
        _req(think=True, reasoning_effort="minimal")
    )
    assert payload["reasoning_effort"] == "minimal"


def test_pollinations_accepts_minimal():
    assert PROFILES["pollinations"].supports_reasoning_effort_minimal
    payload = _backend("pollinations")._build_openai_payload(
        _req(think=True, reasoning_effort="minimal")
    )
    assert payload["reasoning_effort"] == "minimal"


# --- DeepSeek: effort inside thinking:{type, reasoning_effort} --------

def test_deepseek_effort_nested_in_thinking():
    payload = _backend("deepseek")._build_openai_payload(
        _req("deepseek-v4-flash", think=True, reasoning_effort="high")
    )
    assert payload["thinking"] == {"type": "enabled", "reasoning_effort": "high"}


def test_deepseek_effort_max_accepted():
    payload = _backend("deepseek")._build_openai_payload(
        _req("deepseek-v4-pro", think=True, reasoning_effort="max")
    )
    assert payload["thinking"]["reasoning_effort"] == "max"


def test_deepseek_effort_only_high_max():
    # medium/low/xhigh are NOT in DeepSeek's accepted set → NOT emitted
    payload = _backend("deepseek")._build_openai_payload(
        _req("deepseek-v4-flash", think=True, reasoning_effort="medium")
    )
    assert "reasoning_effort" not in payload["thinking"]


def test_deepseek_effort_not_emitted_when_think_off():
    payload = _backend("deepseek")._build_openai_payload(
        _req("deepseek-v4-flash", think=False, reasoning_effort="high")
    )
    assert payload["thinking"] == {"type": "disabled"}


def test_deepseek_effort_not_on_zai():
    # Z.AI also has supports_thinking_type_toggle but should NOT get
    # the DeepSeek-specific nested effort
    payload = _backend("zai")._build_openai_payload(
        _req("glm-4.7", think=True, reasoning_effort="high")
    )
    assert "reasoning_effort" not in payload["thinking"]


# --- SiliconFlow: enable_thinking + thinking_budget -------------------

def test_siliconflow_flag_on():
    assert PROFILES["siliconflow"].reasoning_via_siliconflow_params


def test_siliconflow_think_on_deepseek():
    payload = _backend("siliconflow")._build_openai_payload(
        _req("deepseek-ai/DeepSeek-R1", think=True)
    )
    assert payload["enable_thinking"] is True


def test_siliconflow_think_off_deepseek():
    payload = _backend("siliconflow")._build_openai_payload(
        _req("deepseek-ai/DeepSeek-R1", think=False)
    )
    assert payload["enable_thinking"] is False


def test_siliconflow_budget_from_effort():
    payload = _backend("siliconflow")._build_openai_payload(
        _req("Qwen/Qwen3-235B-A22B", think=True, reasoning_effort="high")
    )
    assert payload["enable_thinking"] is True
    assert payload["thinking_budget"] == 16384


def test_siliconflow_budget_max():
    payload = _backend("siliconflow")._build_openai_payload(
        _req("Qwen/Qwen3-235B-A22B", think=True, reasoning_effort="max")
    )
    assert payload["thinking_budget"] == 32768


def test_siliconflow_budget_minimal():
    payload = _backend("siliconflow")._build_openai_payload(
        _req("deepseek-ai/DeepSeek-V3", think=True, reasoning_effort="minimal")
    )
    assert payload["thinking_budget"] == 512


def test_siliconflow_no_budget_without_effort():
    payload = _backend("siliconflow")._build_openai_payload(
        _req("Qwen/Qwen3-32B", think=True)
    )
    assert payload["enable_thinking"] is True
    assert "thinking_budget" not in payload


def test_siliconflow_non_reasoning_emits_nothing():
    payload = _backend("siliconflow")._build_openai_payload(
        _req("meta-llama/Meta-Llama-3.1-8B-Instruct", think=True)
    )
    assert "enable_thinking" not in payload
    assert "thinking_budget" not in payload


def test_siliconflow_not_on_other_profile():
    payload = _backend("openai")._build_openai_payload(
        _req("deepseek-v4-flash", think=True, reasoning_effort="high")
    )
    assert "enable_thinking" not in payload
    assert "thinking_budget" not in payload
