"""Tests for context-preservation fixes in the coder hybrid loop.

Covers two bugs found 2026-04-20 that caused the model to re-call the same
tools across iterations:

  1. Parallel-read fanout cap silently dropped excess calls — the assistant
     message committed N tool_calls but only K<N tool_results came back.
     Native-tier IDs dangled, and the model (seeing no signal for the
     missing slots) re-emitted the same batch the next iteration. Fix
     synthesizes a tool_result per dropped call with a clear "retry in a
     smaller batch" message.

  2. Mid-turn compaction truncated successful tool-result content to 160
     chars. A 3k-char ``file_read`` collapsed to three wrapped lines of
     whitespace-normalised prose — the model then thought it had never
     seen the file content and re-emitted ``file_read`` the next
     iteration, which pushed tokens back over the compaction threshold,
     which truncated the re-read, which... fix raises the successful-
     result cap to 1500 and keeps newlines so line-prefixed outputs stay
     readable.

Run: python -m pytest tests/test_coder_context_preservation.py -v
"""
from __future__ import annotations

import pytest

from augmentum.models.base import (
    InternalStreamChunk,
    Message,
)
from augmentum.modes.coder.handler import CoderHandler

# Reuse the existing test scaffolding rather than rebuild _FakeTool,
# _FakeChunk, etc. Keeps the two test modules in lockstep.
from tests.test_coder_handler import (
    _ExtendedContainerManager,
    _FakeBackend,
    _FakeChunk,
    _FakeTool,
    _force_native_tier,
    _make_request,
    _tc_delta,
)

# ---------------------------------------------------------------------------
# Fix #1 — fanout-drop synthesis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fanout_drop_synthesizes_tool_result_per_dropped_call(monkeypatch):
    """When the model emits > _HYBRID_READ_FANOUT read calls, each dropped
    call gets a synthetic tool_result so tool_use IDs don't dangle."""
    _force_native_tier(monkeypatch)
    # Tighten the cap so we can trigger it with a small batch.
    monkeypatch.setattr(
        "augmentum.modes.coder.handler._HYBRID_READ_FANOUT", 2,
    )
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_read", output="ok")],
    )

    class _BigFanout:
        """Emits 5 file_read calls in one batch, then stops."""
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(i, f"tc-{i}", "file_read",
                              {"path": f"/workspace/f{i}.py"})
                    for i in range(5)
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _BigFanout(), session_id="sess-fanout",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-fanout",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(_make_request(), workspace_context=""):
        chunks.append(c)

    # 5 tool_result meta chunks total (2 real + 3 synthesized)
    tool_results = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "tool_result"
    ]
    assert len(tool_results) == 5, (
        f"expected 5 tool_result chunks (2 real + 3 dropped); got {len(tool_results)}"
    )

    # Exactly 3 should be flagged fanout_dropped
    dropped = [
        c for c in tool_results
        if c.augmentum.get("tool_result", {}).get("fanout_dropped")
    ]
    assert len(dropped) == 3, (
        f"expected 3 fanout_dropped markers; got {len(dropped)}"
    )

    # The dropped IDs are the tail of the batch (tc-2, tc-3, tc-4)
    dropped_ids = {c.augmentum["tool_result"]["id"] for c in dropped}
    assert dropped_ids == {"tc-2", "tc-3", "tc-4"}

    # Dropped results surface with success=False (so the model understands
    # it has to retry) but NOT as validation_errors (it's not the model's
    # fault — we imposed the cap).
    for c in dropped:
        tr = c.augmentum["tool_result"]
        assert tr["success"] is False
        assert "parallel-read cap" in tr["output_preview"]


@pytest.mark.asyncio
async def test_fanout_drop_appends_tool_messages_to_history(monkeypatch):
    """Native-tier: every committed tool_call must have a matching tool
    message. Fanout-dropped calls go through _append_tool_result_to_history
    just like real tool results, which means the conversation schema stays
    valid for the next backend round-trip."""
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler._HYBRID_READ_FANOUT", 1,
    )

    # Track every request the backend sees so we can inspect the history
    # shape after the fanout iteration.
    seen_histories: list[list[Message]] = []

    class _Recorder:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            seen_histories.append(list(request.messages))
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(i, f"tc-{i}", "file_read",
                              {"path": f"/workspace/f{i}.py"})
                    for i in range(3)
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_read", output="ok")],
    )

    handler = CoderHandler(
        _Recorder(), session_id="sess-fanout-history",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-fanout-history",
    )

    async for _c in handler._act_hybrid(_make_request(), workspace_context=""):
        pass

    # On the SECOND iteration (after the fanout), the backend should see
    # exactly 3 tool messages for the 3 tool_calls from iteration 1.
    # Anything less means dangling tool_use IDs.
    assert len(seen_histories) >= 2
    second_turn = seen_histories[1]
    tool_msgs = [m for m in second_turn if m.role == "tool"]
    tc_ids_in_history = {m.tool_call_id for m in tool_msgs}
    assert tc_ids_in_history == {"tc-0", "tc-1", "tc-2"}, (
        f"Expected all 3 tool_call_ids present as tool messages; "
        f"got {tc_ids_in_history}. Dangling IDs would leave the model "
        f"in an inconsistent state and encourage re-emission."
    )


@pytest.mark.asyncio
async def test_fanout_under_cap_does_not_synthesize(monkeypatch):
    """When the batch is at or below the cap, no synthesis happens."""
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler._HYBRID_READ_FANOUT", 5,
    )
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_read", output="ok")],
    )

    class _SmallFanout:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(i, f"tc-{i}", "file_read",
                              {"path": f"/workspace/f{i}.py"})
                    for i in range(3)
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _SmallFanout(), session_id="sess-under-cap",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-under-cap",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(_make_request(), workspace_context=""):
        chunks.append(c)

    dropped = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "tool_result"
        and c.augmentum.get("tool_result", {}).get("fanout_dropped")
    ]
    assert dropped == [], "No drops should happen when batch is under cap"


# ---------------------------------------------------------------------------
# Fix #2 — compaction preserves tool-result content
# ---------------------------------------------------------------------------


def _make_handler_for_compact() -> CoderHandler:
    return CoderHandler(
        _FakeBackend([]),
        session_id="sess-ctx",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-ctx",
    )


def _trip_compaction_thresholds(monkeypatch, keep_recent: int = 2) -> None:
    monkeypatch.setattr(
        "augmentum.modes.coder.handler._COMPACT_AT_TOKENS", 50,
    )
    monkeypatch.setattr(
        "augmentum.modes.coder.handler._COMPACT_KEEP_RECENT", keep_recent,
    )


def test_compact_preserves_grounded_tool_result_content(monkeypatch):
    """A file_read's content must survive compaction. Pre-fix it was
    clipped to 160 chars, which broke the model's working memory and
    drove re-read loops."""
    _trip_compaction_thresholds(monkeypatch)
    h = _make_handler_for_compact()

    # Simulated file_read output — structured content the model needs
    # verbatim to reason about.
    file_content = "\n".join(
        f"{i:4d} | line {i}: def something_{i}():" for i in range(1, 60)
    )  # ~1400 chars, under the new 1500 cap

    messages = [
        Message(role="system", content="sys" + "x" * 200),
        Message(role="user", content="task: explain this file " + "w " * 1100),
        # Middle turn 1: assistant calls file_read
        Message(
            role="assistant", content="",
            tool_calls=[{
                "id": "tc-read",
                "type": "function",
                "function": {"name": "file_read", "arguments": "{}"},
            }],
        ),
        Message(role="tool", content=file_content, tool_call_id="tc-read"),
        # Middle turn 2: another round (anything, just to have a middle)
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc-grep",
            "type": "function",
            "function": {"name": "code_grep", "arguments": "{}"},
        }]),
        Message(role="tool", content="grep match line\n" * 400,
                tool_call_id="tc-grep"),
        # Tail (kept verbatim)
        Message(role="assistant", content="final thought"),
        Message(role="tool", content="final", tool_call_id="tc-final"),
        Message(role="assistant", content="final2"),
        Message(role="tool", content="final2", tool_call_id="tc-final2"),
    ]

    compacted, _before, _after = h._maybe_compact_messages(messages)
    assert compacted

    summary = messages[2].content
    # The exact file line content (e.g. "line 30:") must survive in
    # the compacted summary — this is the grounded fact the model
    # uses to reason. Pre-fix the summary would contain
    # "T:    1 | line 1: def something_1():   2 | line 2..." wrapped
    # to 160 chars with newlines collapsed.
    assert "line 30: def something_30" in summary, (
        "Fix #2 regression: successful tool-result content was clipped "
        "below the 1500-char cap. Content under the cap MUST survive."
    )
    # Newlines preserved for grounded content — line-prefixed output
    # has to stay readable.
    assert "\n" in summary.split("<compacted", 1)[-1]


def test_compact_marks_tool_result_truncation_with_byte_count(monkeypatch):
    """Content over the 1500-char cap is clipped but includes a truncation
    marker telling the model how much was cut, so it can decide whether
    to re-fetch (with paging) or work from what it has."""
    _trip_compaction_thresholds(monkeypatch)
    h = _make_handler_for_compact()

    huge_content = "X" * 5000  # well over the 1500-char cap

    messages = [
        Message(role="system", content="sys" + "x" * 200),
        Message(role="user", content="task " + "w " * 1100),
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc-huge",
            "type": "function",
            "function": {"name": "file_read", "arguments": "{}"},
        }]),
        Message(role="tool", content=huge_content, tool_call_id="tc-huge"),
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc-grep",
            "type": "function",
            "function": {"name": "code_grep", "arguments": "{}"},
        }]),
        Message(role="tool", content="grep", tool_call_id="tc-grep"),
        Message(role="assistant", content="tail"),
        Message(role="tool", content="tail-result", tool_call_id="tc-tail"),
        Message(role="assistant", content="tail2"),
        Message(role="tool", content="tail2", tool_call_id="tc-tail2"),
    ]

    compacted, _before, _after = h._maybe_compact_messages(messages)
    assert compacted

    summary = messages[2].content
    # Marker must say how much was cut and point the model at the
    # earlier history so it knows where the full content lives.
    assert "truncated" in summary
    assert "more chars" in summary
    # The 3500 chars cut (5000 - 1500) should appear in the marker.
    assert "3500" in summary


def test_compact_error_tool_result_uses_400_cap(monkeypatch):
    """Error results keep their 400-char cap — not 1500 — because the
    recovery hint (schema + example) fits in 400 chars and bloating errors
    would crowd out real tool-result content."""
    _trip_compaction_thresholds(monkeypatch)
    h = _make_handler_for_compact()

    # Error content 600 chars — well under 1500, well over 400
    error_content = "ERROR: " + ("z" * 600)

    messages = [
        Message(role="system", content="sys" + "x" * 200),
        Message(role="user", content="task " + "w " * 1100),
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc-err",
            "type": "function",
            "function": {"name": "file_read", "arguments": "{}"},
        }]),
        Message(role="tool", content=error_content, tool_call_id="tc-err"),
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc-2",
            "type": "function",
            "function": {"name": "code_grep", "arguments": "{}"},
        }]),
        Message(role="tool", content="grep match line\n" * 400, tool_call_id="tc-2"),
        Message(role="assistant", content="tail"),
        Message(role="tool", content="tail-result", tool_call_id="tc-tail"),
        Message(role="assistant", content="tail2"),
        Message(role="tool", content="tail2", tool_call_id="tc-tail2"),
    ]

    compacted, _before, _after = h._maybe_compact_messages(messages)
    assert compacted

    summary = messages[2].content
    # Find the "T: ERROR:..." line in the summary
    error_line = next(
        (line for line in summary.splitlines() if line.startswith("T: ERROR:")),
        None,
    )
    assert error_line is not None, "Expected an ERROR: tool line in summary"
    # The line is "T: ERROR: zzz...zzz" — capped at 400 content chars, so
    # the line is ~404 chars after "T: " prefix. Much less than 607 chars
    # (the full error would be).
    assert len(error_line) < 450, (
        f"Error cap should be 400, not 1500; got {len(error_line)}-char line"
    )


def test_compact_rolls_back_when_summary_would_grow_tokens(monkeypatch):
    """If the compaction summary would be bigger than what it replaced —
    e.g. a small middle with sub-cap tool_results — roll back and report
    compacted=False, messages untouched. Prevents the degenerate case
    where compaction adds overhead without clip savings."""
    _trip_compaction_thresholds(monkeypatch)
    h = _make_handler_for_compact()

    # Small middle — 3 tool results, each well under the 1500-char cap.
    # The per-line "T: " overhead and preserved newlines would add up to
    # more than the original messages; the guard should catch it.
    messages = [
        Message(role="system", content="sys" + "x" * 200),
        Message(role="user", content="task " + "w " * 1100),
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc-a", "type": "function",
            "function": {"name": "file_read", "arguments": "{}"},
        }]),
        Message(role="tool", content="small\nresult", tool_call_id="tc-a"),
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc-b", "type": "function",
            "function": {"name": "code_grep", "arguments": "{}"},
        }]),
        Message(role="tool", content="another\nsmall\none", tool_call_id="tc-b"),
        Message(role="assistant", content="tail"),
        Message(role="tool", content="tail-result", tool_call_id="tc-tail"),
        Message(role="assistant", content="tail2"),
        Message(role="tool", content="tail2", tool_call_id="tc-tail2"),
    ]
    before_snapshot = [(m.role, m.content) for m in messages]

    compacted, before, after = h._maybe_compact_messages(messages)

    # Guard fired — compaction rolled back.
    assert compacted is False
    # When rolled back, before == after (no change in token count).
    assert after == before
    # And messages list is untouched.
    after_snapshot = [(m.role, m.content) for m in messages]
    assert after_snapshot == before_snapshot


def test_compact_small_tool_result_kept_verbatim(monkeypatch):
    """Results under the cap are kept completely verbatim — no truncation
    marker, no newline collapse. Uses a corpus big enough that compaction
    is still profitable (the rollback guard would otherwise fire).

    The interesting result is the small one ``tc-small`` with three
    newline-separated lines; the other tools are oversized padding just
    to make compaction worth running.
    """
    _trip_compaction_thresholds(monkeypatch)
    h = _make_handler_for_compact()

    small_content = "line1\nline2\nline3"

    messages = [
        Message(role="system", content="sys" + "x" * 200),
        Message(role="user", content="task " + "w " * 1100),
        # The small result we want to verify survives intact
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc-small",
            "type": "function",
            "function": {"name": "file_read", "arguments": "{}"},
        }]),
        Message(role="tool", content=small_content, tool_call_id="tc-small"),
    ]
    # Oversized padding so compaction is profitable and doesn't roll back
    for i in range(4):
        messages.append(Message(
            role="assistant", content="", tool_calls=[{
                "id": f"tc-pad-{i}",
                "type": "function",
                "function": {"name": "file_read", "arguments": "{}"},
            }],
        ))
        messages.append(Message(
            role="tool", content="x" * 4000, tool_call_id=f"tc-pad-{i}",
        ))
    # Tail
    messages.append(Message(role="assistant", content="tail"))
    messages.append(Message(role="tool", content="tail-result", tool_call_id="tc-tail"))

    compacted, _before, _after = h._maybe_compact_messages(messages)
    assert compacted

    summary = messages[2].content
    # Whole content present
    assert "line1" in summary
    assert "line2" in summary
    assert "line3" in summary
    # And newlines preserved — not collapsed
    assert "line1\nline2" in summary
