"""Sprint 2 tests — synthesize tool (Piece 8).

The synthesize tool wraps a utility-tier LLM call. These tests cover
the pure-function helpers + the contract behavior. Full LLM integration
tests are out of scope (no real model available in unit test env);
they're covered by the revisit_thread end-to-end tests.

Covered here:
* Time-of-day tone mapping
* Entity token extraction
* Prompt assembly
* Empty input short-circuit
* Backend resolution failure → empty result
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest


def test_time_of_day_tone_mapping():
    from augmentum.tools.synthesize import (
        _time_of_day_tone,
        _TONE_OVERNIGHT, _TONE_MIDDAY, _TONE_EVENING,
    )
    # Build timestamps in local time
    import datetime
    today = datetime.date.today()

    def _ts(hour: int) -> float:
        return time.mktime(datetime.datetime.combine(
            today, datetime.time(hour, 0)).timetuple())

    assert _time_of_day_tone(_ts(3)) == _TONE_OVERNIGHT
    assert _time_of_day_tone(_ts(10)) == _TONE_MIDDAY
    assert _time_of_day_tone(_ts(19)) == _TONE_EVENING


def test_entity_tokens_picks_quoted_and_capitalized():
    from augmentum.tools.synthesize import _entity_tokens
    text = (
        'The "Prefix Caching" paper relates to the April KV work and '
        "the new `MoE` benchmarks."
    )
    tokens = _entity_tokens(text)
    # Quoted strings preserved (lowercased)
    assert "prefix caching" in tokens
    # Backtick-quoted preserved
    assert "moe" in tokens


def test_entity_tokens_empty_input():
    from augmentum.tools.synthesize import _entity_tokens
    assert _entity_tokens("") == set()


def test_build_system_prompt_includes_rules():
    from augmentum.tools.synthesize import _build_system_prompt, _TONE_MIDDAY
    prompt = _build_system_prompt(persona_kernel="", tone=_TONE_MIDDAY)
    # Contract rules present
    assert "empty string if no real connection exists" in prompt.lower()
    assert "do not invent" in prompt.lower()
    assert "2-3 sentences" in prompt.lower()


def test_build_user_prompt_includes_moments():
    from augmentum.tools.synthesize import _build_user_prompt
    from augmentum.resolver.core import Moment

    moments = [
        Moment(id="m1", kind="file", score=0.5,
               snippet="A paper on prefix caching", title="Prefix Paper",
               created_at="", content_refs=[], legs=[], raw={}),
    ]
    out = _build_user_prompt("Alex's attention thread", moments)
    assert "Alex's attention thread" in out
    assert "Prefix Paper" in out
    assert "prefix caching" in out


@pytest.mark.asyncio
async def test_synthesize_empty_input_short_circuits():
    """Empty wondering or moments → returns empty result without LLM call."""
    from augmentum.tools.synthesize import synthesize

    runtime = MagicMock()
    result = await synthesize(
        runtime, wondering_content="", moments=[],
    )
    assert result.text == ""
    assert result.model_used == ""


@pytest.mark.asyncio
async def test_synthesize_backend_failure_returns_empty():
    """When tiers.utility raises (no provider configured), result is empty."""
    from augmentum.tools.synthesize import synthesize
    from augmentum.resolver.core import Moment

    runtime = MagicMock()
    runtime._app_state = MagicMock()
    runtime._app_state.provider_registry = None  # forces tiers.utility to raise

    moments = [
        Moment(id="m1", kind="file", score=0.5, snippet="anything",
               title="t", created_at="", content_refs=[], legs=[], raw={}),
    ]
    result = await synthesize(
        runtime, wondering_content="thread content", moments=moments,
    )
    assert result.text == ""
    assert result.grounded is False


@pytest.mark.asyncio
async def test_synthesize_empty_llm_output_is_contract_compliant():
    """Empty model output is the 'no real connection' contract, not a failure."""
    from augmentum.tools.synthesize import synthesize
    from augmentum.resolver.core import Moment

    # Build a fake backend whose chat returns empty content
    fake_response = MagicMock()
    fake_response.content = ""
    fake_response.text = ""

    fake_backend = MagicMock()
    fake_backend.chat = AsyncMock(return_value=fake_response)

    runtime = MagicMock()
    runtime.identity = MagicMock()
    runtime.identity.persona_kernel_digest = ""
    # Force tiers.utility to return our fake backend
    async def _fake_utility(rt, **kwargs):
        return fake_backend, "fake-model"

    from augmentum.companion_runtime import tiers
    original = tiers.utility
    tiers.utility = _fake_utility
    try:
        moments = [
            Moment(id="m1", kind="file", score=0.5, snippet="x",
                   title="t", created_at="", content_refs=[], legs=[], raw={}),
        ]
        result = await synthesize(
            runtime, wondering_content="thread", moments=moments,
        )
        assert result.text == ""
        assert result.grounded is True  # empty IS grounded — contract honored
        assert result.model_used == "fake-model"
    finally:
        tiers.utility = original


@pytest.mark.asyncio
async def test_synthesize_grounded_when_entities_match():
    """When LLM output references items from input, grounded=True."""
    from augmentum.tools.synthesize import synthesize
    from augmentum.resolver.core import Moment

    fake_response = MagicMock()
    fake_response.content = 'The "Prefix Paper" connects to your April KV work.'
    fake_response.text = None

    fake_backend = MagicMock()
    fake_backend.chat = AsyncMock(return_value=fake_response)

    runtime = MagicMock()
    runtime.identity = MagicMock()
    runtime.identity.persona_kernel_digest = ""
    async def _fake_utility(rt, **kwargs):
        return fake_backend, "fake-model"

    from augmentum.companion_runtime import tiers
    original = tiers.utility
    tiers.utility = _fake_utility
    try:
        moments = [
            Moment(id="m1", kind="file", score=0.5,
                   snippet="prefix caching paper",
                   title="Prefix Paper", created_at="",
                   content_refs=[], legs=[], raw={}),
        ]
        result = await synthesize(
            runtime, wondering_content="thread", moments=moments,
        )
        assert "Prefix Paper" in result.text
        assert result.grounded is True
    finally:
        tiers.utility = original


@pytest.mark.asyncio
async def test_synthesize_ungrounded_when_entities_invented():
    """When LLM output references items NOT in input, grounded=False."""
    from augmentum.tools.synthesize import synthesize
    from augmentum.resolver.core import Moment

    fake_response = MagicMock()
    fake_response.content = (
        'This connects to the "Quantum Whatever" paper which doesn\'t exist '
        'and the "Made Up Reference" you allegedly read.'
    )
    fake_response.text = None

    fake_backend = MagicMock()
    fake_backend.chat = AsyncMock(return_value=fake_response)

    runtime = MagicMock()
    runtime.identity = MagicMock()
    runtime.identity.persona_kernel_digest = ""
    async def _fake_utility(rt, **kwargs):
        return fake_backend, "fake-model"

    from augmentum.companion_runtime import tiers
    original = tiers.utility
    tiers.utility = _fake_utility
    try:
        moments = [
            Moment(id="m1", kind="file", score=0.5,
                   snippet="real input snippet",
                   title="Real Paper", created_at="",
                   content_refs=[], legs=[], raw={}),
        ]
        result = await synthesize(
            runtime, wondering_content="thread", moments=moments,
        )
        # Output mentions invented entities not in moments → ungrounded
        assert result.text != ""
        assert result.grounded is False
    finally:
        tiers.utility = original
