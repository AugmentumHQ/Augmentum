"""Structured-header tests for _maybe_compact_messages.

Phase 14 (2026-04-21) added an opencode-inspired structured header to
the compacted block so the model can see "what happened" without
re-scanning every T: / A: line. The existing per-message lines still
render underneath the header, preserving the grounded-content
guarantees the prior tests check.

Header categorises the dropped region into:
- Files edited (file_write / code_edit / code_multi_edit)
- Files read (file_read)
- Tool call counts (top 5)
- Test pass/fail tally
"""
from __future__ import annotations

import json

import pytest

from augmentum.modes.coder.handler import CoderHandler
from augmentum.models.base import Message


# Reuse the scaffolding the other compaction tests use.
from tests.test_coder_context_preservation import (
    _make_handler_for_compact,
    _trip_compaction_thresholds,
)


def _assistant_with_tool(name: str, args: dict, tool_id: str) -> Message:
    return Message(
        role="assistant", content="",
        tool_calls=[{
            "id": tool_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    )


def _tool_result(tool_id: str, content: str) -> Message:
    return Message(role="tool", content=content, tool_call_id=tool_id)


def test_structured_header_lists_edited_files(monkeypatch):
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()

    # Tool results are padded large enough that the compacted block
    # (header + per-message lines) is smaller than the dropped region.
    # Without this the rollback guard (`after >= before`) fires.
    big = "x" * 3000

    messages = [
        Message(role="system", content="sys" + "x" * 200),
        Message(role="user", content="task " + "w " * 1100),
        _assistant_with_tool("file_write", {"path": "/workspace/a.py"}, "t1"),
        _tool_result("t1", big),
        _assistant_with_tool("code_edit", {"path": "/workspace/b.py"}, "t2"),
        _tool_result("t2", big),
        _assistant_with_tool("file_read", {"path": "/workspace/c.py"}, "t3"),
        _tool_result("t3", big),
        Message(role="assistant", content="tail"),
        Message(role="tool", content="tail", tool_call_id="t4"),
        Message(role="assistant", content="tail2"),
        Message(role="tool", content="tail2", tool_call_id="tail2"),
    ]

    compacted, _before, _after = h._maybe_compact_messages(messages)
    assert compacted

    summary = messages[2].content
    assert "## Summary" in summary
    assert "Edited: /workspace/a.py, /workspace/b.py" in summary
    assert "Read: /workspace/c.py" in summary


def test_structured_header_counts_tools(monkeypatch):
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()

    big = "x" * 3000
    messages = [
        Message(role="system", content="sys" + "x" * 200),
        Message(role="user", content="task " + "w " * 1100),
        _assistant_with_tool("file_read", {"path": "/a.py"}, "t1"),
        _tool_result("t1", big),
        _assistant_with_tool("file_read", {"path": "/b.py"}, "t2"),
        _tool_result("t2", big),
        _assistant_with_tool("code_grep", {"pattern": "foo"}, "t3"),
        _tool_result("t3", big),
        Message(role="assistant", content="tail"),
        Message(role="tool", content="tail", tool_call_id="t4"),
        Message(role="assistant", content="tail2"),
        Message(role="tool", content="tail2", tool_call_id="tail2"),
    ]

    compacted, _before, _after = h._maybe_compact_messages(messages)
    assert compacted

    summary = messages[2].content
    # Tools line shows aggregate counts
    assert "Tools:" in summary
    assert "file_read×2" in summary
    assert "code_grep×1" in summary


def test_structured_header_absent_when_no_tools(monkeypatch):
    """If the dropped region has no tool activity, the Summary block
    stays out of the compacted message (nothing useful to say)."""
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()

    # All prose, no tool calls.
    messages = [
        Message(role="system", content="sys" + "x" * 200),
        Message(role="user", content="task " + "w " * 1100),
        Message(role="assistant", content="thinking 1 " * 100),
        Message(role="user", content="reply 1 " * 100),
        Message(role="assistant", content="thinking 2 " * 100),
        Message(role="user", content="reply 2 " * 100),
        Message(role="assistant", content="tail"),
        Message(role="tool", content="x", tool_call_id="t"),
    ]

    compacted, _before, _after = h._maybe_compact_messages(messages)
    if not compacted:
        pytest.skip("content didn't trigger compaction threshold")
    summary = messages[2].content
    assert "## Summary" not in summary
    # Details section still renders so the model can see what got said.
    assert "## Details" in summary


def test_structured_header_counts_test_outcomes(monkeypatch):
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()

    pad = "x" * 3000
    messages = [
        Message(role="system", content="sys" + "x" * 200),
        Message(role="user", content="task " + "w " * 1100),
        _assistant_with_tool("test_run", {}, "t1"),
        _tool_result("t1", "tests passed: 14\n" + pad),
        _assistant_with_tool("test_run", {}, "t2"),
        _tool_result("t2", "test failed: AssertionError in test_foo\n" + pad),
        _assistant_with_tool("test_run", {}, "t3"),
        _tool_result("t3", "test failed: ImportError\n" + pad),
        Message(role="assistant", content="tail"),
        Message(role="tool", content="tail", tool_call_id="t4"),
        Message(role="assistant", content="tail2"),
        Message(role="tool", content="tail2", tool_call_id="tail2"),
    ]

    compacted, _before, _after = h._maybe_compact_messages(messages)
    assert compacted

    summary = messages[2].content
    assert "Tests:" in summary
    # One pass, two fails
    assert "1 passed" in summary
    assert "2 failed" in summary


def test_structured_header_preserves_grounded_content(monkeypatch):
    """Regression guard: adding the structured header must not evict
    the grounded T: content from the compacted message."""
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()

    grep_output = (
        "src/auth.py:42: def authenticate(user):\n"
        "src/auth.py:50:     return validate_password(user)\n"
    ) + "other content " * 400

    messages = [
        Message(role="system", content="sys" + "x" * 200),
        Message(role="user", content="task " + "w " * 1100),
        _assistant_with_tool("code_grep", {"pattern": "auth"}, "t1"),
        _tool_result("t1", grep_output),
        _assistant_with_tool("file_read", {"path": "/a.py"}, "t2"),
        _tool_result("t2", "content " * 400),
        Message(role="assistant", content="tail"),
        Message(role="tool", content="tail", tool_call_id="t3"),
        Message(role="assistant", content="tail2"),
        Message(role="tool", content="tail2", tool_call_id="tail2"),
    ]

    compacted, _before, _after = h._maybe_compact_messages(messages)
    assert compacted

    summary = messages[2].content
    # Structured header is present
    assert "## Summary" in summary
    # Grounded grep match survived — this is the invariant the existing
    # tests assert; we check it here too to protect against the new
    # header displacing it.
    assert "def authenticate(user)" in summary


# ---------------------------------------------------------------------------
# State-derived fields (PR2.2): active task + recent blocker
# ---------------------------------------------------------------------------


def test_structured_header_surfaces_active_task(monkeypatch):
    """When state.tasks has an in_progress entry, it appears in the
    Summary block — the model's own self-declared focus must survive
    compaction even if its declaration message is in the dropped
    region."""
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()
    h._state.set_tasks([
        {"content": "Implement auth middleware",
         "activeForm": "Implementing", "status": "in_progress"},
        {"content": "Write tests",
         "activeForm": "Testing", "status": "pending"},
    ])

    big = "x" * 3000
    messages = [
        Message(role="system", content="sys" + "x" * 200),
        Message(role="user", content="task " + "w " * 1100),
        _assistant_with_tool("file_read", {"path": "/a.py"}, "t1"),
        _tool_result("t1", big),
        _assistant_with_tool("file_read", {"path": "/b.py"}, "t2"),
        _tool_result("t2", big),
        Message(role="assistant", content="tail"),
        Message(role="tool", content="tail", tool_call_id="t3"),
        Message(role="assistant", content="tail2"),
        Message(role="tool", content="tail2", tool_call_id="tail2"),
    ]

    compacted, _before, _after = h._maybe_compact_messages(messages)
    assert compacted
    summary = messages[2].content
    assert "Active task: Implement auth middleware" in summary


def test_structured_header_surfaces_repeated_blocker(monkeypatch):
    """A recent_tool_failures entry with count >= 2 (real pattern, not
    a one-off) appears in the Summary block as 'Recent blocker: ...'."""
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()
    # Simulate two failed code_edit attempts on the same target.
    h._state.record_tool_failure(
        tool_name="code_edit", target="/snake.html",
        error="No match found in '/snake.html'",
    )
    h._state.record_tool_failure(
        tool_name="code_edit", target="/snake.html",
        error="No match found in '/snake.html'",
    )

    big = "x" * 3000
    messages = [
        Message(role="system", content="sys" + "x" * 200),
        Message(role="user", content="task " + "w " * 1100),
        _assistant_with_tool("file_read", {"path": "/a.py"}, "t1"),
        _tool_result("t1", big),
        _assistant_with_tool("file_read", {"path": "/b.py"}, "t2"),
        _tool_result("t2", big),
        Message(role="assistant", content="tail"),
        Message(role="tool", content="tail", tool_call_id="t3"),
        Message(role="assistant", content="tail2"),
        Message(role="tool", content="tail2", tool_call_id="tail2"),
    ]

    compacted, _before, _after = h._maybe_compact_messages(messages)
    assert compacted
    summary = messages[2].content
    assert "Recent blocker:" in summary
    assert "code_edit" in summary
    assert "/snake.html" in summary
    assert "×2" in summary


def test_structured_header_skips_one_off_failures(monkeypatch):
    """A single failure (count=1) is noise — not surfaced. Only repeated
    patterns (count>=2) deserve the Recent blocker line."""
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()
    h._state.record_tool_failure(
        tool_name="file_read", target="/missing.py",
        error="No such file",
    )

    big = "x" * 3000
    messages = [
        Message(role="system", content="sys" + "x" * 200),
        Message(role="user", content="task " + "w " * 1100),
        _assistant_with_tool("file_read", {"path": "/a.py"}, "t1"),
        _tool_result("t1", big),
        _assistant_with_tool("file_read", {"path": "/b.py"}, "t2"),
        _tool_result("t2", big),
        Message(role="assistant", content="tail"),
        Message(role="tool", content="tail", tool_call_id="t3"),
        Message(role="assistant", content="tail2"),
        Message(role="tool", content="tail2", tool_call_id="tail2"),
    ]

    compacted, _before, _after = h._maybe_compact_messages(messages)
    assert compacted
    summary = messages[2].content
    assert "Recent blocker:" not in summary


# ---------------------------------------------------------------------------
# Append-stable extension (2026-07-02) — re-compaction must EXTEND the
# existing <compacted> block, never re-render it. Re-rendering rewrote
# the head of history every pass (llama-server prefix-cache kill,
# measured stable_pct 0.13) and crushed the prior summary to one line.
# ---------------------------------------------------------------------------


def _mid_pair(i: int, big: str) -> list[Message]:
    return [
        _assistant_with_tool("file_read", {"path": f"/w/f{i}.py"}, f"m{i}"),
        _tool_result(f"m{i}", big),
    ]


def test_recompaction_extends_block_byte_prefix_stable(monkeypatch):
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()
    big = "x" * 3000

    messages = [
        Message(role="system", content="sys" + "x" * 200),
        Message(role="user", content="task " + "w " * 1100),
        *_mid_pair(1, big), *_mid_pair(2, big), *_mid_pair(3, big),
        Message(role="assistant", content="tail"),
        Message(role="tool", content="tail", tool_call_id="tt"),
    ]
    compacted, _b, _a = h._maybe_compact_messages(messages)
    assert compacted
    block_v1 = messages[2].content
    assert block_v1.lstrip().startswith("<compacted")

    # Loop keeps working — more middle accumulates after the block.
    grow_at = len(messages) - 2
    for i in range(4, 9):
        messages[grow_at:grow_at] = _mid_pair(i, big)
        grow_at += 2

    compacted, _b, _a = h._maybe_compact_messages(messages)
    assert compacted
    block_v2 = messages[2].content

    # Byte-prefix contract: v2 starts with all of v1 minus the closer.
    v1_prefix = block_v1[: -len("</compacted>")].rstrip("\n")
    assert block_v2.startswith(v1_prefix)
    # One wrapper, two segments — never nested, never re-rendered.
    assert block_v2.count("<compacted") == 1
    assert block_v2.count("## Condensed segment") == 2
    assert block_v2.endswith("</compacted>")


def test_compaction_anchor_skips_runtime_carrier(monkeypatch):
    """A runtime carrier ahead of the task must not steal the preserved
    first-user slot (the real task was getting condensed to 160 chars),
    and the stale carrier itself is never condensed into the block."""
    from augmentum.modes.coder.chat_egress import RUNTIME_CARRIER_HEADER

    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()
    big = "x" * 3000

    messages = [
        Message(role="system", content="sys" + "x" * 200),
        Message(role="user", content=f"{RUNTIME_CARRIER_HEADER}\nstale prior_turns"),
        Message(role="user", content="the real task " + "y" * 200),
        *_mid_pair(1, big), *_mid_pair(2, big), *_mid_pair(3, big),
        Message(role="assistant", content="tail"),
        Message(role="tool", content="tail", tool_call_id="tt"),
    ]
    compacted, _b, _a = h._maybe_compact_messages(messages)
    assert compacted
    # Carrier and task preserved verbatim, block right after the task.
    assert messages[1].content.startswith(RUNTIME_CARRIER_HEADER)
    assert messages[2].content.startswith("the real task")
    assert messages[3].content.lstrip().startswith("<compacted")
    assert "stale prior_turns" not in messages[3].content


def test_cap_compacted_block_drops_oldest_segments():
    seg = lambda i: (  # noqa: E731
        f"## Condensed segment ({i} messages)\n### Details\nT: " + "z" * 400
    )
    content = (
        "<compacted reason=\"context-pressure\">\nstable intro line.\n\n"
        + "\n\n".join(seg(i) for i in range(1, 40))
        + "\n</compacted>"
    )
    capped = CoderHandler._cap_compacted_block(content, 100)  # cap≈12k chars
    assert len(capped) < len(content)
    assert capped.count("<compacted") == 1
    assert capped.endswith("</compacted>")
    assert "oldest history dropped" in capped
    # Newest segment survives; the oldest is gone.
    assert "## Condensed segment (39 messages)" in capped
    assert "## Condensed segment (1 messages)" not in capped
