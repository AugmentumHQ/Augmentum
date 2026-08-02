"""Tool-call/result pairing reconciler for the Coder send path.

``_reconcile_tool_pairing`` runs at the single send choke-point
(``_stream_and_parse``) and guarantees the OpenAI-compatible invariant
that every assistant ``tool_calls`` message is immediately followed by one
``role="tool"`` message per ``tool_call_id`` — with no dangling tool
messages. Strict providers (DeepSeek direct) 400 the whole turn otherwise;
the orphans come from non-pairing-aware context compaction or a persisted
trailing unanswered call (see the 2026-06-27 log dig).

These tests pin:
  1. happy path is a no-op (no synthesized stubs, equivalent list)
  2. a missing trailing result gets a stub (the observed DeepSeek 400)
  3. a dangling leading tool message (assistant compacted away) is dropped
  4. partial multi-call batches are completed in declared order
  5. the input list + Message objects are never mutated (purity)

Run: python -m pytest tests/test_coder_tool_pairing_reconcile.py -v
"""
from __future__ import annotations

from augmentum.models.base import Message
from augmentum.modes.coder.handler import (
    _PAIRING_STUB,
    _reconcile_tool_pairing,
)


def _assistant(*ids: str, content: str = "") -> Message:
    return Message(
        role="assistant",
        content=content,
        tool_calls=[
            {"id": i, "type": "function",
             "function": {"name": "file_read", "arguments": "{}"}}
            for i in ids
        ],
    )


def _tool(tcid: str, content: str = "ok") -> Message:
    return Message(role="tool", content=content, tool_call_id=tcid)


def _pairs_ok(messages: list) -> bool:
    """True if every assistant tool_call id is answered by an immediately
    following tool message, and no tool message is dangling."""
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        if m.role == "assistant" and (m.tool_calls or []):
            declared = [tc["id"] for tc in m.tool_calls]
            j = i + 1
            answered: list[str] = []
            while j < n and messages[j].role == "tool":
                answered.append(messages[j].tool_call_id)
                j += 1
            if sorted(answered) != sorted(declared):
                return False
            i = j
            continue
        if m.role == "tool":
            return False  # dangling — not preceded by an assistant group
        i += 1
    return True


def test_happy_path_is_noop():
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="task"),
        _assistant("a", "b"),
        _tool("a"), _tool("b"),
        Message(role="assistant", content="done"),
    ]
    out = _reconcile_tool_pairing(msgs)
    # Same length, same roles/ids in order, no stubs inserted.
    assert [(m.role, m.tool_call_id) for m in out] == [
        (m.role, m.tool_call_id) for m in msgs
    ]
    assert all(m.content != _PAIRING_STUB for m in out)
    assert _pairs_ok(out)


def test_missing_trailing_result_gets_stub():
    # The observed DeepSeek 400: assistant emitted a call, its result was
    # compacted away, leaving an unanswered tool_call_id.
    msgs = [
        Message(role="user", content="task"),
        _assistant("a"),
        # no tool message for "a"
        Message(role="user", content="next"),
    ]
    out = _reconcile_tool_pairing(msgs)
    assert _pairs_ok(out)
    stub = [m for m in out if m.role == "tool" and m.tool_call_id == "a"]
    assert len(stub) == 1 and stub[0].content == _PAIRING_STUB


def test_dangling_leading_tool_message_dropped():
    # Compaction summarized away the assistant call but kept its result —
    # a tool message with no preceding declaration.
    msgs = [
        Message(role="user", content="<compacted>…</compacted>"),
        _tool("ghost"),
        Message(role="assistant", content="continuing"),
    ]
    out = _reconcile_tool_pairing(msgs)
    assert _pairs_ok(out)
    assert all(m.role != "tool" for m in out)  # the orphan is gone


def test_partial_batch_completed_in_declared_order():
    # Two calls declared, only the second answered — stub fills the first,
    # and order matches the declared order (a before b).
    msgs = [
        Message(role="user", content="task"),
        _assistant("a", "b"),
        _tool("b", content="real-b"),
    ]
    out = _reconcile_tool_pairing(msgs)
    assert _pairs_ok(out)
    tool_msgs = [m for m in out if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["a", "b"]
    assert tool_msgs[0].content == _PAIRING_STUB
    assert tool_msgs[1].content == "real-b"


def test_input_is_not_mutated():
    msgs = [
        Message(role="user", content="task"),
        _assistant("a"),
        Message(role="user", content="next"),
    ]
    snapshot_len = len(msgs)
    snapshot_roles = [m.role for m in msgs]
    out = _reconcile_tool_pairing(msgs)
    # Original list untouched; the fix lives only in the returned copy.
    assert len(msgs) == snapshot_len
    assert [m.role for m in msgs] == snapshot_roles
    assert out is not msgs
    assert any(m.content == _PAIRING_STUB for m in out)


def test_duplicate_result_collapsed():
    # A doubled tool result (re-append after a retry) is collapsed to one.
    msgs = [
        Message(role="user", content="task"),
        _assistant("a"),
        _tool("a", content="real"),
        _tool("a", content="dupe"),
    ]
    out = _reconcile_tool_pairing(msgs)
    assert _pairs_ok(out)
    answered = [m for m in out if m.role == "tool" and m.tool_call_id == "a"]
    assert len(answered) == 1 and answered[0].content == "real"
