"""Claude frontier-model capability gating (2026-06-15).

The native Claude path was pinned to the 4.6 generation as the frontier.
On current models (Opus 4.7/4.8, Fable 5) three things 400: the
``thinking:{budget_tokens}`` shape (adaptive is the only on-mode),
``temperature``/``top_p`` (removed unconditionally), and assistant
prefill. These tests pin the corrected gating so a current Claude model
stops producing guaranteed-400 request bodies.

Verified against Anthropic's model-migration + error-code docs (June 2026).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from augmentum.models.adapters.claude import ClaudeBackend
from augmentum.models.base import InternalChatRequest, Message
from augmentum.models.converters.claude import (
    is_adaptive_model,
    is_no_prefill_model,
    is_no_sampling_model,
    is_thinking_model,
)

FRONTIER = ["claude-opus-4-6", "claude-opus-4-7", "claude-opus-4-8",
            "claude-sonnet-4-6", "claude-fable-5"]
ALWAYS_REJECT_SAMPLING = ["claude-opus-4-7", "claude-opus-4-8", "claude-fable-5"]


def _run(coro):
    # Run on a throwaway loop, then reinstall a fresh open current loop.
    # A sibling test (test_claude_backend.test_context_length) still uses
    # the deprecated asyncio.get_event_loop(), which raises if we leave the
    # thread with no current loop after a bare _run().
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(asyncio.new_event_loop())


def _backend(**kw) -> ClaudeBackend:
    return ClaudeBackend(client=MagicMock(), api_key="sk-test", cache_enabled=False, **kw)


def _req(model: str, **kw) -> InternalChatRequest:
    return InternalChatRequest(
        model=model,
        messages=[Message(role="user", content="Hi")],
        max_tokens=2048,
        **kw,
    )


# ---- capability detection -------------------------------------------------

@pytest.mark.parametrize("model", FRONTIER)
def test_frontier_is_adaptive_and_no_prefill(model):
    assert is_adaptive_model(model), model
    assert is_no_prefill_model(model), model
    assert is_thinking_model(model), model  # fable-5 was previously missed


@pytest.mark.parametrize("model", ALWAYS_REJECT_SAMPLING)
def test_47_plus_reject_sampling(model):
    assert is_no_sampling_model(model), model


@pytest.mark.parametrize("model", ["claude-opus-4-6", "claude-sonnet-4-6"])
def test_46_still_accepts_sampling(model):
    # 4.6 / Sonnet-4.6 keep sampling params when thinking is off.
    assert not is_no_sampling_model(model), model


def test_older_models_are_not_adaptive():
    assert not is_adaptive_model("claude-3-7-sonnet")
    assert not is_no_sampling_model("claude-sonnet-4")


# ---- request body ---------------------------------------------------------

@pytest.mark.parametrize("model", FRONTIER)
def test_adaptive_thinking_never_sends_budget_tokens(model):
    body = _backend()._build_request_body(_req(model, think=True))
    assert body["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in body.get("thinking", {})
    assert body.get("output_config", {}).get("effort")


def test_older_thinking_model_uses_budget_tokens():
    body = _backend()._build_request_body(_req("claude-3-7-sonnet", think=True))
    assert body["thinking"]["type"] == "enabled"
    assert body["thinking"]["budget_tokens"] >= 1024


@pytest.mark.parametrize("model", ALWAYS_REJECT_SAMPLING)
def test_47_plus_strip_sampling_even_without_thinking(model):
    body = _backend()._build_request_body(_req(model, temperature=0.7, top_p=0.9))
    assert "temperature" not in body
    assert "top_p" not in body


def test_46_keeps_sampling_when_thinking_off():
    body = _backend()._build_request_body(
        _req("claude-sonnet-4-6", temperature=0.7)
    )
    assert body["temperature"] == 0.7


# ---- catalog + context ----------------------------------------------------

def test_catalog_drops_retired_and_adds_current():
    backend = _backend()
    names = {m.name for m in _run(backend.list_models())}
    assert {"claude-opus-4-8", "claude-opus-4-7", "claude-fable-5"} <= names
    # Retired 3.5/3.7 ids (404) gone.
    assert not any("3-5" in n or "3-7" in n for n in names)


@pytest.mark.parametrize("model,ctx", [
    ("claude-opus-4-8-20260101", 1_000_000),  # dated suffix tolerated
    ("claude-fable-5", 1_000_000),
    ("claude-opus-4-5", 200_000),
    ("claude-haiku-4-5", 200_000),
    ("claude-3-opus-latest", 200_000),
])
def test_context_length_from_catalog(model, ctx):
    assert _run(_backend().get_context_length(model)) == ctx
