"""Tests for the hydrated recency buffer (2026-07-03).

The coder UI's ``getMessagesForLLM`` collapses prior turns to
``{role, content}`` — stripping tool_calls, tool results, and thinking —
so on a follow-up turn the model never sees its own recent tool-using
behavior and lapses to prose-only answers (observed live: 9B DeepSeek
interleaved 12 tools on turn 1, answered directly on turn 2).

The recency buffer fixes this by replaying the last N completed turns'
FULL in-format chains at the head of the next turn's history. These tests
pin the seam detection, hydration, nudge exclusion, pairing completeness,
and the safe empty-buffer fallback.
"""
from __future__ import annotations

from augmentum.modes.coder.handler import CoderHandler, _RECENCY_BUFFER_TURNS
from augmentum.models.base import Message

from tests.test_coder_handler import (
    _ExtendedContainerManager,
    _FakeBackend,
)


def _handler() -> CoderHandler:
    return CoderHandler(
        _FakeBackend([]),
        session_id="sess-recency",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-recency",
    )


def _asst_tool(call_id: str, name: str) -> Message:
    return Message(
        role="assistant", content="",
        tool_calls=[{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }],
        thinking="I should inspect the workspace first.",
    )


# ---------------------------------------------------------------------------
# _fresh_turn_input — the seam between hydrated history and new input
# ---------------------------------------------------------------------------

def test_fresh_input_is_everything_after_last_assistant():
    h = _handler()
    msgs = [
        Message(role="user", content="turn1 ask"),
        Message(role="assistant", content="turn1 answer"),
        Message(role="user", content="turn2 ask"),
    ]
    fresh = h._fresh_turn_input(msgs)
    assert [m.content for m in fresh] == ["turn2 ask"]


def test_fresh_input_first_turn_is_whole_payload():
    h = _handler()
    msgs = [Message(role="user", content="only ask")]
    fresh = h._fresh_turn_input(msgs)
    assert [m.content for m in fresh] == ["only ask"]


# ---------------------------------------------------------------------------
# _apply_recency_buffer — hydration + safe fallback
# ---------------------------------------------------------------------------

def test_apply_buffer_empty_is_passthrough():
    h = _handler()
    msgs = [
        Message(role="user", content="u1"),
        Message(role="assistant", content="a1"),
        Message(role="user", content="u2"),
    ]
    # No buffer yet → unchanged (never drop the client payload).
    out = h._apply_recency_buffer(msgs, h._fresh_turn_input(msgs))
    assert out is msgs


def test_apply_buffer_swaps_collapsed_for_hydrated():
    h = _handler()
    # Buffer holds turn 1 as a FULL chain (tool_calls + result).
    turn1 = [
        Message(role="user", content="organize files"),
        _asst_tool("t1", "file_list"),
        Message(role="tool", content="Directory: /workspace ...", tool_call_id="t1"),
        Message(role="assistant", content="Done, organized."),
    ]
    h._recent_turn_chains = [turn1]
    # Client re-sends turn 1 COLLAPSED (tool stuff stripped) + new turn 2.
    collapsed = [
        Message(role="user", content="organize files"),
        Message(role="assistant", content="Done, organized."),
        Message(role="user", content="now what's left?"),
    ]
    out = h._apply_recency_buffer(collapsed, h._fresh_turn_input(collapsed))
    # Hydrated turn 1 (with tool_calls + result) precedes the fresh ask.
    assert out[:4] == turn1
    assert out[-1].content == "now what's left?"
    # The exemplar the model now sees includes a real tool_call + result.
    assert any(m.role == "assistant" and m.tool_calls for m in out)
    assert any(m.role == "tool" for m in out)


def test_apply_buffer_falls_back_when_no_fresh_user():
    h = _handler()
    h._recent_turn_chains = [[Message(role="user", content="x")]]
    # Payload ends on an assistant (no fresh user turn) — don't risk a swap.
    msgs = [
        Message(role="user", content="u1"),
        Message(role="assistant", content="a1"),
    ]
    out = h._apply_recency_buffer(msgs, h._fresh_turn_input(msgs))
    assert out is msgs


# ---------------------------------------------------------------------------
# _capture_recency_turn — clean exemplar, nudge exclusion, identity seam
# ---------------------------------------------------------------------------

def test_capture_grabs_full_chain_excluding_nudges_and_carrier():
    h = _handler()
    user_in = Message(role="user", content="fix the bug")
    h._pending_turn_input = [user_in]
    # A realistic post-loop `messages`: system + carrier + input + a chain
    # that includes a synthetic <nudge> user message mid-flight.
    messages = [
        Message(role="system", content="NATIVE_SYSTEM ..."),
        Message(role="user", content="[Augmentum runtime context — not user dialogue]\n..."),
        user_in,
        _asst_tool("t1", "file_read"),
        Message(role="tool", content="file contents", tool_call_id="t1"),
        Message(role="user", content="<nudge>call a tool</nudge>"),  # synthetic
        _asst_tool("t2", "code_edit"),
        Message(role="tool", content="edited", tool_call_id="t2"),
        Message(role="assistant", content="Fixed."),
    ]
    h._capture_recency_turn(messages)
    assert len(h._recent_turn_chains) == 1
    chain = h._recent_turn_chains[0]
    roles = [m.role for m in chain]
    # Starts with the real user input; NO synthetic nudge user message.
    assert chain[0] is user_in
    assert roles.count("user") == 1
    assert all("<nudge>" not in (m.content or "") for m in chain)
    # Both tool_calls and both results preserved, pairing-complete.
    call_ids = {tc["id"] for m in chain if m.tool_calls for tc in m.tool_calls}
    result_ids = {m.tool_call_id for m in chain if m.role == "tool"}
    assert call_ids == result_ids == {"t1", "t2"}


def test_capture_survives_compaction_rebuild_via_identity():
    h = _handler()
    user_in = Message(role="user", content="do work")
    h._pending_turn_input = [user_in]
    a1 = _asst_tool("t1", "shell_exec")
    r1 = Message(role="tool", content="ok", tool_call_id="t1")
    final = Message(role="assistant", content="done")
    # Simulate a compaction rebuild: the list is a NEW list object with a
    # condensed prefix, but the recent-tail Message OBJECTS are preserved
    # by reference (that's how _maybe_compact_messages keeps the tail).
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="<compacted>...older...</compacted>"),
        user_in, a1, r1, final,
    ]
    h._capture_recency_turn(messages)
    chain = h._recent_turn_chains[0]
    assert chain[0] is user_in
    assert final in chain


def test_capture_bounds_to_window():
    h = _handler()
    for i in range(_RECENCY_BUFFER_TURNS + 3):
        u = Message(role="user", content=f"ask {i}")
        h._pending_turn_input = [u]
        h._capture_recency_turn([
            u,
            _asst_tool(f"t{i}", "file_read"),
            Message(role="tool", content="r", tool_call_id=f"t{i}"),
            Message(role="assistant", content=f"done {i}"),
        ])
    assert len(h._recent_turn_chains) == _RECENCY_BUFFER_TURNS
    # Keeps the NEWEST turns.
    last = h._recent_turn_chains[-1][0].content
    assert last == f"ask {_RECENCY_BUFFER_TURNS + 2}"


def test_capture_skips_turn_with_no_assistant_reply():
    h = _handler()
    u = Message(role="user", content="ask")
    h._pending_turn_input = [u]
    # Only a stray tool message, no assistant — nothing worth buffering.
    h._capture_recency_turn([u, Message(role="tool", content="x", tool_call_id="t")])
    assert getattr(h, "_recent_turn_chains", None) in (None, [])


def test_capture_skips_prose_only_turn_no_tools():
    """A conversational turn (assistant reply, zero tool use) must NOT be
    buffered — a prose exemplar in the hydrated window pulls the model
    toward answering without tools, the exact regression the buffer fixes.
    """
    h = _handler()
    u = Message(role="user", content="what is this workspace?")
    h._pending_turn_input = [u]
    h._capture_recency_turn([
        Message(role="system", content="sys"),
        u,
        Message(role="assistant", content="It's a Python project for X."),
    ])
    assert getattr(h, "_recent_turn_chains", None) in (None, [])


def test_prose_turn_between_tool_turns_is_skipped():
    """Interleave a prose turn between two tool turns — only the tool
    turns land in the buffer, so the window stays all-exemplar."""
    h = _handler()

    def cap(user_text, *, tools):
        u = Message(role="user", content=user_text)
        h._pending_turn_input = [u]
        chain = [u]
        if tools:
            chain += [
                _asst_tool("tc", "file_read"),
                Message(role="tool", content="r", tool_call_id="tc"),
            ]
        chain.append(Message(role="assistant", content="reply"))
        h._capture_recency_turn(chain)

    cap("edit the file", tools=True)      # buffered
    cap("why did you do that?", tools=False)  # prose — skipped
    cap("now run the tests", tools=True)  # buffered

    assert len(h._recent_turn_chains) == 2
    # Both buffered turns are tool-using exemplars.
    for chain in h._recent_turn_chains:
        assert any(m.role == "tool" for m in chain)


def test_capture_noop_when_seam_missing():
    h = _handler()
    h._pending_turn_input = [Message(role="user", content="orphan")]
    # The pending anchor object is NOT in messages (identity miss).
    h._capture_recency_turn([
        Message(role="user", content="different object"),
        Message(role="assistant", content="a"),
    ])
    assert getattr(h, "_recent_turn_chains", None) in (None, [])


# ---------------------------------------------------------------------------
# End-to-end seam: capture then apply reproduces the interleaved exemplar
# ---------------------------------------------------------------------------

def test_capture_then_apply_round_trip():
    h = _handler()
    # Turn 1 runs and is captured.
    u1 = Message(role="user", content="turn1")
    h._pending_turn_input = [u1]
    h._capture_recency_turn([
        Message(role="system", content="sys"),
        u1,
        _asst_tool("t1", "file_list"),
        Message(role="tool", content="listing", tool_call_id="t1"),
        Message(role="assistant", content="turn1 done"),
    ])
    # Turn 2 arrives collapsed from the client.
    collapsed = [
        Message(role="user", content="turn1"),
        Message(role="assistant", content="turn1 done"),
        Message(role="user", content="turn2"),
    ]
    out = h._apply_recency_buffer(collapsed, h._fresh_turn_input(collapsed))
    # The model now sees turn1's real tool_call + result before turn2.
    assert any(m.role == "assistant" and m.tool_calls for m in out)
    assert any(m.role == "tool" and m.content == "listing" for m in out)
    assert out[-1].content == "turn2"
