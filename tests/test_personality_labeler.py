"""Tests for augmentum.personality.labeler — prompt construction + JSON parsing.

The labeler module is mostly pure functions (prompt + parser) plus one
async entry point that wraps a caller-supplied LLM call. Tests cover all
three, with the LLM call mocked for the async path.
"""
from __future__ import annotations

import pytest

from augmentum.personality.labeler import (
    build_labeler_messages,
    label_response,
    parse_labeler_response,
)

# ----------------------------------------------------------------------
# Prompt construction
# ----------------------------------------------------------------------

def test_build_messages_returns_system_and_user_pair():
    msgs = build_labeler_messages("a response", "some context")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_build_messages_includes_vocabulary_listing():
    msgs = build_labeler_messages("a response", "some context")
    system = msgs[0]["content"]
    # Canonical first-class facets must appear in the prompt vocabulary.
    assert "tender" in system
    assert "unsure" in system
    assert "not_okay" in system


def test_build_messages_includes_response_text():
    msgs = build_labeler_messages("THE-RESPONSE-MARKER", "ctx")
    assert "THE-RESPONSE-MARKER" in msgs[1]["content"]


def test_build_messages_includes_context():
    msgs = build_labeler_messages("response", "CONTEXT-MARKER")
    assert "CONTEXT-MARKER" in msgs[1]["content"]


def test_build_messages_max_facets_respected_in_prompt():
    msgs = build_labeler_messages("response", "ctx", max_facets=3)
    # The integer should appear in the instructions.
    assert "3" in msgs[0]["content"]


# ----------------------------------------------------------------------
# Parser — valid JSON cases
# ----------------------------------------------------------------------

def test_parse_clean_json():
    text = '{"facets": [{"name": "warm", "intensity": 0.8}]}'
    result = parse_labeler_response(text)
    assert result == [("warm", 0.8)]


def test_parse_multiple_facets():
    text = (
        '{"facets": ['
        '{"name": "warm", "intensity": 0.8}, '
        '{"name": "patient", "intensity": 0.4}'
        "]}"
    )
    result = parse_labeler_response(text)
    assert result == [("warm", 0.8), ("patient", 0.4)]


def test_parse_strips_markdown_fence():
    text = '```json\n{"facets": [{"name": "tender", "intensity": 0.6}]}\n```'
    result = parse_labeler_response(text)
    assert result == [("tender", 0.6)]


def test_parse_with_surrounding_commentary():
    text = (
        "Here is my analysis:\n"
        '{"facets": [{"name": "playful", "intensity": 0.5}]}\n'
        "Hope that helps!"
    )
    result = parse_labeler_response(text)
    assert result == [("playful", 0.5)]


def test_parse_empty_facets_list():
    text = '{"facets": []}'
    result = parse_labeler_response(text)
    assert result == []


def test_parse_lowercases_facet_names():
    text = '{"facets": [{"name": "WARM", "intensity": 0.8}]}'
    result = parse_labeler_response(text)
    assert result == [("warm", 0.8)]


def test_parse_strips_whitespace_in_names():
    text = '{"facets": [{"name": "  warm  ", "intensity": 0.8}]}'
    result = parse_labeler_response(text)
    assert result == [("warm", 0.8)]


# ----------------------------------------------------------------------
# Parser — graceful degradation
# ----------------------------------------------------------------------

def test_parse_empty_input_returns_empty():
    assert parse_labeler_response("") == []
    assert parse_labeler_response("   ") == []


def test_parse_invalid_json_returns_empty():
    text = "this is not json at all"
    assert parse_labeler_response(text) == []


def test_parse_malformed_json_returns_empty():
    text = '{"facets": [{"name": "warm", "intensity":}]}'
    assert parse_labeler_response(text) == []


def test_parse_missing_facets_key_returns_empty():
    text = '{"other_field": "value"}'
    assert parse_labeler_response(text) == []


def test_parse_facets_not_a_list_returns_empty():
    text = '{"facets": "warm"}'
    assert parse_labeler_response(text) == []


def test_parse_handles_nested_object_in_facet_entry():
    """Some models add a metadata sub-object to each facet entry. The regex
    extractor can't handle deep nesting; the json.loads first-pass does."""
    text = (
        '{"facets": [{"name": "warm", "intensity": 0.8, '
        '"meta": {"reason": "openness", "evidence": {"line": 3}}}]}'
    )
    result = parse_labeler_response(text)
    assert result == [("warm", 0.8)]


# ----------------------------------------------------------------------
# Parser — value coercion / clamping
# ----------------------------------------------------------------------

def test_intensity_above_one_clamped_to_one():
    text = '{"facets": [{"name": "warm", "intensity": 2.5}]}'
    result = parse_labeler_response(text)
    assert result == [("warm", 1.0)]


def test_intensity_below_zero_clamped_to_zero():
    text = '{"facets": [{"name": "warm", "intensity": -0.5}]}'
    result = parse_labeler_response(text)
    assert result == [("warm", 0.0)]


def test_intensity_nonnumeric_coerced_to_default():
    text = '{"facets": [{"name": "warm", "intensity": "high"}]}'
    result = parse_labeler_response(text)
    assert result == [("warm", 1.0)]


def test_intensity_missing_defaults_to_one():
    text = '{"facets": [{"name": "warm"}]}'
    result = parse_labeler_response(text)
    assert result == [("warm", 1.0)]


def test_facet_entry_missing_name_dropped():
    text = '{"facets": [{"name": "warm", "intensity": 0.8}, {"intensity": 0.5}]}'
    result = parse_labeler_response(text)
    assert result == [("warm", 0.8)]


def test_facet_entry_not_dict_dropped():
    text = '{"facets": [{"name": "warm", "intensity": 0.8}, "not a dict"]}'
    result = parse_labeler_response(text)
    assert result == [("warm", 0.8)]


# ----------------------------------------------------------------------
# Async entry point — full label_response()
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_label_response_empty_text_skips_llm():
    """Empty response_text → empty result, no LLM call made."""
    calls = []

    async def fake_llm(messages):
        calls.append(messages)
        return ""

    result = await label_response("", "context", llm_call=fake_llm)
    assert result == []
    assert calls == []


@pytest.mark.asyncio
async def test_label_response_whitespace_only_skips_llm():
    calls = []

    async def fake_llm(messages):
        calls.append(messages)
        return ""

    result = await label_response("   \n  ", "context", llm_call=fake_llm)
    assert result == []
    assert calls == []


@pytest.mark.asyncio
async def test_label_response_passes_through_llm_output():
    async def fake_llm(messages):
        return '{"facets": [{"name": "warm", "intensity": 0.7}]}'

    result = await label_response("response", "context", llm_call=fake_llm)
    assert result == [("warm", 0.7)]


@pytest.mark.asyncio
async def test_label_response_returns_empty_on_llm_exception():
    """Network / backend failure → empty list, no exception propagated."""

    async def failing_llm(messages):
        raise RuntimeError("backend down")

    result = await label_response("response", "context", llm_call=failing_llm)
    assert result == []


@pytest.mark.asyncio
async def test_label_response_returns_empty_on_garbage_output():
    async def garbage_llm(messages):
        return "completely unparseable output"

    result = await label_response("response", "context", llm_call=garbage_llm)
    assert result == []
