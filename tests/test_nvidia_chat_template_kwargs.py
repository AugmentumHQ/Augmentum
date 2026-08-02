"""NVIDIA NIM reasoning via nested ``chat_template_kwargs`` (2026-06-25).

NIM (NVIDIA's hosted vLLM) gates reasoning through a NESTED
``chat_template_kwargs`` object — NOT the top-level ``enable_thinking``
that llama-server accepts (gated to local engines), and NOT
``reasoning_effort``. **Load-bearing**: NIM strictly requires
``chat_template_kwargs:{thinking:true}`` to stream reasoning for
DeepSeek-V4 — WITHOUT it ``deepseek-v4-flash`` / ``-pro`` HANG
indefinitely (CORRECTIONS #25, build.nvidia.com).

These tests pin: the kwarg is emitted for DeepSeek (always, both
think states — that's the hang fix) and for the enable_thinking
families (Qwen/Nemotron) mirroring ``request.think``; non-reasoning
models on NIM get nothing; and NON-NVIDIA cloud profiles never emit it
(DeepSeek-the-provider uses the top-level ``thinking:{type}`` toggle).
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


def test_nvidia_profile_has_the_capability():
    # Guard against the flag silently dropping in a future edit.
    assert PROFILES["nvidia"].reasoning_via_chat_template_kwargs


# --- DeepSeek-V4: the HANG case — kwarg ALWAYS emitted ----------------

def test_deepseek_v4_emits_thinking_true_when_think_on():
    payload = _backend("nvidia")._build_openai_payload(
        _req("deepseek-ai/deepseek-v4-flash", think=True)
    )
    assert payload["chat_template_kwargs"] == {"thinking": True}


def test_deepseek_v4_emits_thinking_false_when_think_off():
    # Still emitted with think off — NIM hangs without the key even to
    # DISABLE reasoning. This is the core of the #25 fix.
    payload = _backend("nvidia")._build_openai_payload(
        _req("deepseek-ai/deepseek-v4-pro", think=False)
    )
    assert payload["chat_template_kwargs"] == {"thinking": False}


# --- enable_thinking families: mirror request.think ------------------

def test_qwen3_emits_enable_thinking():
    payload = _backend("nvidia")._build_openai_payload(
        _req("qwen/qwen3-235b-a22b", think=True)
    )
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}


def test_nemotron_emits_enable_thinking():
    payload = _backend("nvidia")._build_openai_payload(
        _req("nvidia/llama-3.3-nemotron-super-49b", think=True)
    )
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}


# --- non-reasoning model on NIM: nothing emitted ---------------------

def test_non_reasoning_model_emits_nothing():
    payload = _backend("nvidia")._build_openai_payload(
        _req("meta/llama-3.1-8b-instruct", think=True)
    )
    assert "chat_template_kwargs" not in payload


def test_glm_excluded_uncertain_convention():
    # GLM is deliberately NOT guessed for NIM (only DeepSeek hangs).
    payload = _backend("nvidia")._build_openai_payload(
        _req("zai/glm-4.7", think=True)
    )
    assert "chat_template_kwargs" not in payload


# --- isolation: non-NVIDIA profiles never get the nested kwarg --------

def test_deepseek_provider_does_not_emit_chat_template_kwargs():
    # The DeepSeek *provider* (api.deepseek.com) uses the top-level
    # thinking:{type} toggle, NOT nested chat_template_kwargs.
    payload = _backend("deepseek")._build_openai_payload(
        _req("deepseek-v4-flash", think=True)
    )
    assert "chat_template_kwargs" not in payload
    assert payload["thinking"] == {"type": "enabled"}  # the right mechanism


def test_no_profile_never_emits_chat_template_kwargs():
    payload = _backend(None)._build_openai_payload(
        _req("deepseek-v4-flash", think=True)
    )
    assert "chat_template_kwargs" not in payload


@pytest.mark.parametrize("think", [True, False])
def test_deepseek_kwarg_value_tracks_think(think):
    payload = _backend("nvidia")._build_openai_payload(
        _req("deepseek-ai/deepseek-v4-flash", think=think)
    )
    assert payload["chat_template_kwargs"] == {"thinking": think}
