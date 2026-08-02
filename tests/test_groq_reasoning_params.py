"""Groq per-model ``reasoning_effort`` emission (CORRECTIONS #15, 2026-06-25).

Groq strict-validates the request body and 400s on out-of-set effort
values, and each reasoning model accepts a DIFFERENT enum
(console.groq.com/docs/reasoning):

  * GPT-OSS 20B/120B → low / medium / high (no disable).
  * Qwen3 → none (disable) / default (enable).

The bare Groq profile previously sent nothing, so the thinking toggle was
a no-op on qwen3 and effort was uncontrolled on gpt-oss. These tests pin
the per-model mapping, the UI-tier clamp (minimal/xhigh can never 400),
that ``reasoning_format`` is never sent (raw + JSON/tools → 400), that
non-reasoning Groq models emit nothing, and that the path is isolated to
the Groq profile.
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


def test_groq_profile_has_the_capability():
    assert PROFILES["groq"].reasoning_via_groq_params


# --- GPT-OSS: low/medium/high, UI tiers clamped ----------------------

@pytest.mark.parametrize("ui,sent", [
    ("low", "low"), ("medium", "medium"), ("high", "high"),
    ("minimal", "low"),   # clamp: gpt-oss has no minimal
    ("xhigh", "high"),    # clamp: gpt-oss caps at high
    ("max", "high"),
])
def test_gpt_oss_effort_mapped_and_clamped(ui, sent):
    payload = _backend("groq")._build_openai_payload(
        _req("openai/gpt-oss-120b", think=True, reasoning_effort=ui)
    )
    assert payload["reasoning_effort"] == sent


def test_gpt_oss_no_effort_leaves_groq_default():
    # No UI selection → don't force; Groq's own default (medium) stands.
    payload = _backend("groq")._build_openai_payload(
        _req("openai/gpt-oss-20b", think=True)
    )
    assert "reasoning_effort" not in payload


def test_gpt_oss_never_sends_reasoning_format():
    # raw + JSON/tools → 400; we never send the field.
    payload = _backend("groq")._build_openai_payload(
        _req("openai/gpt-oss-120b", think=True, reasoning_effort="high")
    )
    assert "reasoning_format" not in payload


# --- Qwen3: none/default driven by the think toggle ------------------

def test_qwen3_think_on_enables():
    payload = _backend("groq")._build_openai_payload(
        _req("qwen/qwen3-32b", think=True)
    )
    assert payload["reasoning_effort"] == "default"


def test_qwen3_think_off_disables():
    payload = _backend("groq")._build_openai_payload(
        _req("qwen/qwen3-32b", think=False)
    )
    assert payload["reasoning_effort"] == "none"


def test_qwen3_never_sends_reasoning_format():
    payload = _backend("groq")._build_openai_payload(
        _req("qwen/qwen3-32b", think=True)
    )
    assert "reasoning_format" not in payload


# --- non-reasoning Groq model: nothing ------------------------------

def test_llama_on_groq_emits_nothing():
    payload = _backend("groq")._build_openai_payload(
        _req("llama-3.3-70b-versatile", think=True, reasoning_effort="high")
    )
    assert "reasoning_effort" not in payload


# --- isolation: non-Groq profiles never get this path ---------------

def test_non_groq_profile_does_not_emit_none_default():
    # DeepSeek profile (no groq flag, supports_reasoning_effort=False) must
    # not emit the qwen none/default that only the Groq path produces.
    payload = _backend("deepseek")._build_openai_payload(
        _req("qwen/qwen3-32b", think=False)
    )
    assert "reasoning_effort" not in payload


def test_no_profile_emits_nothing():
    payload = _backend(None)._build_openai_payload(
        _req("openai/gpt-oss-120b", think=True, reasoning_effort="high")
    )
    assert "reasoning_effort" not in payload
