"""Tests for the narrative internal-tool conduct contract.

Covers the behavioral half of the 2026-07-15 fix (the mechanical half —
the pre-call gate — is tested in test_narrative_recall_loop.py): the
silent-suffix must land on every schema without mutating the module-level
constants, and the conduct directive must append to the system prompt
exactly once.
"""

from __future__ import annotations

import copy

from augmentum.models.base import InternalChatRequest, Message
from augmentum.modes.narrative.internal_tools import (
    SILENT_SUFFIX,
    TOOL_CONDUCT_DIRECTIVE,
    append_conduct_directive,
    with_silent_suffix,
)


def _schemas() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "lorebook_check",
                "description": "Check the lorebook for an entry.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall_entity",
                "description": "Look up an entity.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def test_suffix_applied_to_every_description():
    out = with_silent_suffix(_schemas())
    assert all(SILENT_SUFFIX in s["function"]["description"] for s in out)


def test_suffix_does_not_mutate_originals():
    originals = _schemas()
    snapshot = copy.deepcopy(originals)
    with_silent_suffix(originals)
    assert originals == snapshot


def test_suffix_idempotent():
    once = with_silent_suffix(_schemas())
    twice = with_silent_suffix(once)
    for s in twice:
        assert s["function"]["description"].count(SILENT_SUFFIX) == 1


def test_directive_appends_to_existing_system():
    req = InternalChatRequest(
        model="m",
        messages=[
            Message(role="system", content="You narrate the story."),
            Message(role="user", content="continue"),
        ],
    )
    assert append_conduct_directive(req) is True
    sys_msg = req.messages[0]
    assert sys_msg.role == "system"
    assert sys_msg.content.startswith("You narrate the story.")
    assert TOOL_CONDUCT_DIRECTIVE in sys_msg.content


def test_directive_inserts_system_when_absent():
    req = InternalChatRequest(
        model="m",
        messages=[Message(role="user", content="continue")],
    )
    assert append_conduct_directive(req) is True
    assert req.messages[0].role == "system"
    assert req.messages[0].content == TOOL_CONDUCT_DIRECTIVE
    assert req.messages[1].role == "user"


def test_directive_idempotent():
    req = InternalChatRequest(
        model="m",
        messages=[
            Message(role="system", content="You narrate the story."),
            Message(role="user", content="continue"),
        ],
    )
    assert append_conduct_directive(req) is True
    assert append_conduct_directive(req) is False
    assert req.messages[0].content.count(TOOL_CONDUCT_DIRECTIVE) == 1
