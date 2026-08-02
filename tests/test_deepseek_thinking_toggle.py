"""Structured ``thinking`` toggle wiring (2026-06-15).

DeepSeek V4 (flash/pro), Moonshot Kimi, and Z.AI GLM all gate reasoning
per-request via the same top-level ``thinking: {"type": "enabled" |
"disabled"}`` field, default ENABLED. Augmentum's local-engine
``enable_thinking`` path never reached these cloud APIs, so every call
reasoned regardless of ``request.think`` — timing out the voice
classifier hop and emptying act-turn content. The fix is a single
provider-neutral capability ``supports_thinking_type_toggle`` (the wire
shape, not the vendor, gates emission).

These tests pin: the toggle is emitted for every flagged profile and
mirrors ``request.think``; effort rides along only while thinking is on;
and strict non-toggle profiles never see the unknown key (they 400 on
it). Verified against each provider's docs (deepseek/kimi/z.ai).
"""

from __future__ import annotations

import httpx
import pytest

from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.openai_compat import OpenAIBackend
from augmentum.models.provider_profiles import PROFILES

# Profiles that share the structured thinking:{type} shape.
TOGGLE_PROFILES = ["deepseek", "moonshot", "zai"]


def _backend(profile_id: str | None) -> OpenAIBackend:
    profile = PROFILES[profile_id] if profile_id else None
    base = profile.base_url if profile else "http://localhost:8090/v1"
    return OpenAIBackend(httpx.AsyncClient(), base, None, profile=profile)


def _req(model: str = "test-model", **kw) -> InternalChatRequest:
    return InternalChatRequest(
        model=model,
        messages=[Message(role="user", content="play some jazz")],
        **kw,
    )


def test_all_toggle_profiles_have_the_capability():
    # Guard against a profile silently losing the flag in a future edit.
    for pid in TOGGLE_PROFILES:
        assert PROFILES[pid].supports_thinking_type_toggle, pid


@pytest.mark.parametrize("pid", TOGGLE_PROFILES)
def test_disables_thinking_when_think_false(pid):
    # The classifier hop leaves think at its False default → reasoning OFF.
    payload = _backend(pid)._build_openai_payload(_req())
    assert payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in payload  # no effort while thinking off


@pytest.mark.parametrize("pid", TOGGLE_PROFILES)
def test_enables_thinking_when_think_true(pid):
    payload = _backend(pid)._build_openai_payload(_req(think=True))
    assert payload["thinking"] == {"type": "enabled"}


@pytest.mark.parametrize("pid", TOGGLE_PROFILES)
def test_toggle_block_never_emits_reasoning_effort(pid):
    # Effort is owned by the supports_reasoning_effort path; none of the
    # toggle providers set that flag, so a reasoning_effort on the request
    # must NOT ride along on the thinking toggle (Moonshot/Z.AI would
    # strict-400 on an undocumented field).
    assert not PROFILES[pid].supports_reasoning_effort, pid
    payload = _backend(pid)._build_openai_payload(
        _req(think=True, reasoning_effort="high")
    )
    assert "reasoning_effort" not in payload


def test_non_toggle_profile_never_emits_thinking_key():
    # Strict cloud providers 400 on the unknown top-level key.
    payload = _backend("openai")._build_openai_payload(_req(think=False))
    assert "thinking" not in payload


def test_no_profile_never_emits_thinking_key():
    payload = _backend(None)._build_openai_payload(_req(think=False))
    assert "thinking" not in payload
