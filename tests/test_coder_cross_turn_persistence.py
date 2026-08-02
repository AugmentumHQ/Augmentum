"""Tests for cross-turn trace persistence in the coder hybrid loop.

When the user sends a second message, the old loop wiped everything —
model re-read the same files, re-ran the same greps. 2026-04-20 fix
persists a compact per-turn summary into ``CoderState.turn_summaries``
(FIFO, cap 10) which is re-injected into the system prompt on the next
turn as a ``<prior_turns>`` block.

Covers:
  1. ``_build_turn_summary`` extracts files_read / files_edited /
     outcome / blockers from the final in-turn ``messages`` list.
  2. ``CoderState.add_turn_summary`` FIFO-caps at 10.
  3. ``_reset_for_new_request`` leaves ``turn_summaries`` alone.
  4. ``_render_prior_turns`` produces the injection block only when
     summaries exist, handles missing fields gracefully, clips long
     goals, and caps file lists.
  5. Hybrid loop calls ``_build_turn_summary`` + ``add_turn_summary``
     at end-of-turn.
  6. On a second ``_act_hybrid`` run, the new system message contains
     the prior turn's summary — integration check.
  7. Turns with zero tool calls are NOT recorded (no useful signal).

Run: python -m pytest tests/test_coder_cross_turn_persistence.py -v
"""
from __future__ import annotations

import json

import pytest

from augmentum.coder.state import CoderState
from augmentum.models.base import Message
from augmentum.modes.coder.handler import CoderHandler
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.manager import StateManager
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
# CoderState.add_turn_summary
# ---------------------------------------------------------------------------


def test_add_turn_summary_fifo_cap_default_10():
    """Cap defaults to 10; older entries drop first."""
    state = CoderState(session_id="s", workspace_id="ws")
    for i in range(15):
        state.add_turn_summary({"turn_idx": i, "user_goal": f"turn-{i}"})
    assert len(state.turn_summaries) == 10
    # Oldest dropped; last kept should be turn_idx 14 (most recent).
    assert state.turn_summaries[0]["turn_idx"] == 5
    assert state.turn_summaries[-1]["turn_idx"] == 14


def test_add_turn_summary_custom_cap():
    state = CoderState(session_id="s", workspace_id="ws")
    for i in range(8):
        state.add_turn_summary({"turn_idx": i}, max_kept=3)
    assert len(state.turn_summaries) == 3
    assert [s["turn_idx"] for s in state.turn_summaries] == [5, 6, 7]


def test_add_turn_summary_updates_updated_at():
    state = CoderState(session_id="s", workspace_id="ws")
    before = state.updated_at
    state.add_turn_summary({"turn_idx": 1})
    assert state.updated_at >= before


def test_to_dict_roundtrip_preserves_turn_summaries():
    state = CoderState(session_id="s", workspace_id="ws")
    state.add_turn_summary({
        "turn_idx": 1, "user_goal": "test", "files_read": ["a.py"],
        "files_edited": [], "outcome": "done", "blockers": "",
    })
    d = state.to_dict()
    parsed = json.loads(d["turn_summaries"])
    assert len(parsed) == 1
    assert parsed[0]["user_goal"] == "test"

    restored = CoderState.from_row({
        **d,
        # from_row expects raw dict-ish row; to_dict already emits the
        # right shape.
    })
    assert len(restored.turn_summaries) == 1
    assert restored.turn_summaries[0]["files_read"] == ["a.py"]


def test_from_row_handles_missing_turn_summaries_column():
    """Backward compatibility — rows predating migration 099 must still
    deserialize cleanly."""
    row = {
        "session_id": "s",
        "workspace_id": "ws",
        "phase": "waiting",
        # turn_summaries deliberately missing
    }
    state = CoderState.from_row(row)
    assert state.turn_summaries == []


@pytest.mark.asyncio
async def test_handler_restores_persisted_state_across_fresh_instances(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "coder-live.db"))
    await backend.connect()
    try:
        await backend.conn.execute(
            "INSERT INTO users (id, username, display_name, password_hash, role) "
            "VALUES (?, ?, ?, ?, ?)",
            ("alice", "alice", "Alice", "pw", "user"),
        )
        await backend.conn.commit()

        sm = StateManager(backend)
        first = CoderHandler(
            _FakeBackend([]),
            session_id="ws-live",
            container_manager=_ExtendedContainerManager(),
            workspace_id="ws-live",
            state_manager=sm,
            user_id="alice",
        )
        first._state.plan = "Plan:\n1. Keep going"
        first._state.plan_steps = ["Keep going"]
        first._state.set_tasks([{
            "content": "Keep going",
            "activeForm": "Keep going",
            "status": "in_progress",
        }])
        first._state.set_pending_objective_contract({
            "kind": "operate_remote_access",
            "summary": "public access not proven",
        })
        first._state.add_turn_summary({
            "turn_idx": 1,
            "user_goal": "first turn",
            "files_read": ["a.py"],
            "files_edited": [],
            "outcome": "done",
            "blockers": "",
        })
        await first._persist_state()

        second = CoderHandler(
            _FakeBackend([]),
            session_id="ws-live",
            container_manager=_ExtendedContainerManager(),
            workspace_id="ws-live",
            state_manager=sm,
            user_id="alice",
        )
        await second._restore_state()

        assert second._state.plan == first._state.plan
        assert second._state.tasks == first._state.tasks
        assert (
            second._state.pending_objective_contract
            == first._state.pending_objective_contract
        )
        assert second._state.turn_summaries == first._state.turn_summaries
    finally:
        await backend.close()


# ---------------------------------------------------------------------------
# _reset_for_new_request preserves turn_summaries
# ---------------------------------------------------------------------------


def test_reset_for_new_request_preserves_turn_summaries():
    """Request reset nukes plan / tasks / blockers — but NOT turn_summaries
    (that IS the cross-turn memory; wiping it defeats the fix)."""
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="sess-reset",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-reset",
    )
    handler._state.plan = "old plan"
    handler._state.tasks = [{"content": "t", "activeForm": "a", "status": "pending"}]
    handler._state.add_turn_summary({
        "turn_idx": 1, "user_goal": "earlier", "files_read": ["a.py"],
        "files_edited": [], "outcome": "done", "blockers": "",
    })

    handler._reset_for_new_request()

    assert handler._state.plan == ""
    assert handler._state.tasks == []
    # But the summary must still be there
    assert len(handler._state.turn_summaries) == 1
    assert handler._state.turn_summaries[0]["user_goal"] == "earlier"


def test_render_prior_turns_reminder_contract_lives_in_sticky_reminder_not_history():
    """Pending objective contracts belong in the sticky reminder, not prior_turns."""
    h = _make_handler_for_render()
    h._state.set_pending_objective_contract({
        "kind": "operate_remote_access",
        "summary": "Remote/public access is not yet proven.",
        "required_next": "Verify the public URL or explain the blocker plainly.",
        "latest_signal": "your url is: https://bright-rice-sleep.loca.lt",
    })

    reminder = h._build_sticky_reminder(
        goal="expose the app remotely",
        iteration=2,
        max_iters=100,
        writes=1,
    )
    prior = h._render_prior_turns()

    assert "Pending objective contract:" in reminder
    assert "Remote/public access is not yet proven." in reminder
    assert "Verify the public URL or explain the blocker plainly." in reminder
    assert prior == ""


# ---------------------------------------------------------------------------
# _render_prior_turns
# ---------------------------------------------------------------------------


def _make_handler_for_render() -> CoderHandler:
    return CoderHandler(
        _FakeBackend([]),
        session_id="sess-render",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-render",
    )


def test_render_prior_turns_empty_when_no_summaries():
    h = _make_handler_for_render()
    assert h._render_prior_turns() == ""


def test_render_prior_turns_includes_all_fields():
    h = _make_handler_for_render()
    h._state.add_turn_summary({
        "turn_idx": 1,
        "user_goal": "fix the failing auth test",
        "files_read": ["augmentum/auth/session_manager.py", "tests/test_auth.py"],
        "files_edited": [
            {"path": "augmentum/auth/session_manager.py", "summary": "edited"},
        ],
        "outcome": "done",
        "blockers": "",
    })

    block = h._render_prior_turns()
    assert "<prior_turns" in block
    # No count="N" attribute — it mutated every turn at the head of the
    # runtime carrier and truncated the prefix-cache LCP (2026-07-02).
    assert 'count=' not in block
    assert "## Turn 1 (done)" in block
    assert 'Goal: "fix the failing auth test"' in block
    assert "Read: augmentum/auth/session_manager.py, tests/test_auth.py" in block
    assert "Edited: augmentum/auth/session_manager.py (edited)" in block


def test_render_prior_turns_clips_long_goals():
    h = _make_handler_for_render()
    long_goal = "x" * 500
    h._state.add_turn_summary({
        "turn_idx": 1,
        "user_goal": long_goal,
        "files_read": [],
        "files_edited": [],
        "outcome": "done",
        "blockers": "",
    })
    block = h._render_prior_turns()
    # Goal section should be clipped; a 500-char string would bloat the
    # block and burn prompt budget.
    assert "x" * 500 not in block
    assert "…" in block  # ellipsis marks truncation


def test_render_prior_turns_caps_file_lists():
    h = _make_handler_for_render()
    h._state.add_turn_summary({
        "turn_idx": 1,
        "user_goal": "many reads",
        "files_read": [f"f{i}.py" for i in range(20)],
        "files_edited": [],
        "outcome": "done",
        "blockers": "",
    })
    block = h._render_prior_turns()
    # Should show the first 12 only (the cap in _render_prior_turns)
    assert "f0.py" in block
    assert "f11.py" in block
    assert "f15.py" not in block


def test_render_prior_turns_omits_missing_fields():
    """Missing / empty blockers, reads, edits shouldn't produce empty
    lines like 'Read: ' — those waste tokens."""
    h = _make_handler_for_render()
    h._state.add_turn_summary({
        "turn_idx": 1,
        "user_goal": "just a question",
        "files_read": [],
        "files_edited": [],
        "outcome": "done",
        "blockers": "",
    })
    block = h._render_prior_turns()
    assert "Read:" not in block
    assert "Edited:" not in block
    assert "Blockers:" not in block
    # Goal still renders
    assert 'Goal: "just a question"' in block


# ---------------------------------------------------------------------------
# _build_turn_summary extraction
# ---------------------------------------------------------------------------


def test_build_turn_summary_extracts_successful_reads_and_edits():
    h = _make_handler_for_render()

    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="fix the bug"),
        # Successful file_read
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc-read",
            "type": "function",
            "function": {
                "name": "file_read",
                "arguments": json.dumps({"path": "/workspace/bug.py"}),
            },
        }]),
        Message(role="tool", content="def foo(): ...", tool_call_id="tc-read"),
        # Successful code_edit
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc-edit",
            "type": "function",
            "function": {
                "name": "code_edit",
                "arguments": json.dumps({"path": "/workspace/bug.py"}),
            },
        }]),
        Message(role="tool", content="edit applied", tool_call_id="tc-edit"),
    ]

    req = _make_request("fix the bug")
    summary = h._build_turn_summary(
        messages=messages, user_goal="fix the bug",
        termination_reason="model_stop",
    )

    assert summary["user_goal"] == "fix the bug"
    assert summary["files_read"] == ["/workspace/bug.py"]
    assert len(summary["files_edited"]) == 1
    assert summary["files_edited"][0]["path"] == "/workspace/bug.py"
    assert summary["files_edited"][0]["summary"] == "edited"
    assert summary["outcome"] == "done"
    assert summary["blockers"] == ""


def test_build_turn_summary_skips_failed_calls():
    """ERROR: tool results shouldn't be counted as reads or edits —
    only successful ones."""
    h = _make_handler_for_render()

    messages = [
        Message(role="user", content="do thing"),
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc-bad",
            "type": "function",
            "function": {
                "name": "file_read",
                "arguments": json.dumps({"path": "/workspace/missing.py"}),
            },
        }]),
        Message(role="tool",
                content="ERROR: file not found",
                tool_call_id="tc-bad"),
    ]

    summary = h._build_turn_summary(
        messages=messages, user_goal="do thing",
        termination_reason="model_stop",
    )

    assert summary["files_read"] == []
    # Blockers captured from the ERROR content
    assert "file not found" in summary["blockers"]


def test_build_turn_summary_dedupes_repeated_paths():
    h = _make_handler_for_render()
    messages = [
        Message(role="user", content="do it"),
    ]
    # Same file read three times
    for i in range(3):
        messages.append(Message(role="assistant", content="", tool_calls=[{
            "id": f"tc-{i}",
            "type": "function",
            "function": {
                "name": "file_read",
                "arguments": json.dumps({"path": "/workspace/same.py"}),
            },
        }]))
        messages.append(Message(role="tool", content="content", tool_call_id=f"tc-{i}"))

    summary = h._build_turn_summary(
        messages=messages, user_goal="do it",
        termination_reason="model_stop",
    )
    assert summary["files_read"] == ["/workspace/same.py"]


# ---------------------------------------------------------------------------
# Phase 2.1 — per-edit snippet annotations on the turn summary
# ---------------------------------------------------------------------------


def test_build_turn_summary_records_code_edit_snippets():
    """code_edit must populate the new ``edits`` field with snippets
    of search + replace plus the standardized shape."""
    h = _make_handler_for_render()
    messages = [
        Message(role="user", content="fix the bug"),
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc-edit",
            "type": "function",
            "function": {
                "name": "code_edit",
                "arguments": json.dumps({
                    "path": "/workspace/bug.py",
                    "search": "def foo():\n    return None",
                    "replace": "def foo():\n    return 0",
                }),
            },
        }]),
        Message(role="tool", content="edit applied", tool_call_id="tc-edit"),
    ]
    summary = h._build_turn_summary(
        messages=messages, user_goal="fix the bug",
        termination_reason="model_stop",
    )
    assert "edits" in summary
    assert len(summary["edits"]) == 1
    e = summary["edits"][0]
    assert e["path"] == "/workspace/bug.py"
    assert e["tool"] == "code_edit"
    assert e["block_count"] == 1
    assert e["lines_written"] == 0
    assert "def foo()" in e["search_snippet"]
    assert "return None" in e["search_snippet"]
    assert "return 0" in e["replace_snippet"]


def test_build_turn_summary_records_code_edit_batch_first_block():
    """code_edit_batch records block_count + first block's snippets."""
    h = _make_handler_for_render()
    messages = [
        Message(role="user", content="batch"),
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc-batch",
            "type": "function",
            "function": {
                "name": "code_edit_batch",
                "arguments": json.dumps({
                    "path": "/workspace/multi.py",
                    "edits": [
                        {"search": "AAA", "replace": "BBB"},
                        {"search": "CCC", "replace": "DDD"},
                        {"search": "EEE", "replace": "FFF"},
                    ],
                }),
            },
        }]),
        Message(role="tool", content="3 edits applied", tool_call_id="tc-batch"),
    ]
    summary = h._build_turn_summary(
        messages=messages, user_goal="batch",
        termination_reason="model_stop",
    )
    assert len(summary["edits"]) == 1
    e = summary["edits"][0]
    assert e["tool"] == "code_edit_batch"
    assert e["block_count"] == 3
    assert e["search_snippet"] == "AAA"
    assert e["replace_snippet"] == "BBB"


def test_build_turn_summary_records_file_write_with_content_head():
    """file_write captures the head of the new content + lines_written."""
    h = _make_handler_for_render()
    content = "import os\n\n\ndef main():\n    return 0\n"
    messages = [
        Message(role="user", content="create"),
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc-write",
            "type": "function",
            "function": {
                "name": "file_write",
                "arguments": json.dumps({
                    "path": "/workspace/new.py",
                    "content": content,
                }),
            },
        }]),
        Message(role="tool", content="ok", tool_call_id="tc-write"),
    ]
    summary = h._build_turn_summary(
        messages=messages, user_goal="create",
        termination_reason="model_stop",
    )
    assert len(summary["edits"]) == 1
    e = summary["edits"][0]
    assert e["tool"] == "file_write"
    assert e["search_snippet"] == ""
    assert "import os" in e["replace_snippet"]
    assert e["lines_written"] == content.count("\n")


def test_build_turn_summary_does_not_record_failed_edits():
    """ERROR: tool results must NOT add an edits entry — symmetric
    with the existing files_edited guard."""
    h = _make_handler_for_render()
    messages = [
        Message(role="user", content="try"),
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc-bad",
            "type": "function",
            "function": {
                "name": "code_edit",
                "arguments": json.dumps({
                    "path": "/workspace/x.py",
                    "search": "xxx",
                    "replace": "yyy",
                }),
            },
        }]),
        Message(role="tool",
                content="ERROR: search not found",
                tool_call_id="tc-bad"),
    ]
    summary = h._build_turn_summary(
        messages=messages, user_goal="try",
        termination_reason="model_stop",
    )
    assert summary["edits"] == []


def test_build_turn_summary_collapses_multiline_snippets():
    """Newlines + whitespace runs collapse so the rendered prior_turns
    block stays compact."""
    h = _make_handler_for_render()
    multiline_search = "line1\n    line2\n\n    line3"
    messages = [
        Message(role="user", content="x"),
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc",
            "type": "function",
            "function": {
                "name": "code_edit",
                "arguments": json.dumps({
                    "path": "/workspace/x.py",
                    "search": multiline_search,
                    "replace": "single",
                }),
            },
        }]),
        Message(role="tool", content="ok", tool_call_id="tc"),
    ]
    summary = h._build_turn_summary(
        messages=messages, user_goal="x",
        termination_reason="model_stop",
    )
    snippet = summary["edits"][0]["search_snippet"]
    assert "\n" not in snippet
    assert "    " not in snippet
    assert snippet == "line1 line2 line3"


def test_build_turn_summary_truncates_long_snippets():
    """Snippets over 80 chars get truncated with an ellipsis so each
    edit stays bounded in the prior_turns budget."""
    h = _make_handler_for_render()
    long_text = "x" * 200
    messages = [
        Message(role="user", content="x"),
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc",
            "type": "function",
            "function": {
                "name": "code_edit",
                "arguments": json.dumps({
                    "path": "/workspace/x.py",
                    "search": long_text,
                    "replace": "y",
                }),
            },
        }]),
        Message(role="tool", content="ok", tool_call_id="tc"),
    ]
    summary = h._build_turn_summary(
        messages=messages, user_goal="x",
        termination_reason="model_stop",
    )
    snippet = summary["edits"][0]["search_snippet"]
    assert len(snippet) <= 80
    assert snippet.endswith("…")


def test_build_turn_summary_persists_edits_round_trip():
    """The edits field must round-trip through CoderState's to_dict /
    from_row so cross-turn persistence isn't quietly dropping it."""
    state = CoderState(session_id="s", workspace_id="ws")
    state.add_turn_summary({
        "turn_idx": 1,
        "user_goal": "test",
        "files_read": [],
        "files_edited": [{"path": "/x.py", "summary": "edited"}],
        "edits": [{
            "path": "/x.py",
            "tool": "code_edit",
            "search_snippet": "old",
            "replace_snippet": "new",
            "block_count": 1,
            "lines_written": 0,
        }],
        "outcome": "done",
        "blockers": "",
    })
    serialized = state.to_dict()
    restored = CoderState.from_row({**serialized, "phase": "waiting"})
    assert len(restored.turn_summaries) == 1
    assert restored.turn_summaries[0]["edits"] == [{
        "path": "/x.py",
        "tool": "code_edit",
        "search_snippet": "old",
        "replace_snippet": "new",
        "block_count": 1,
        "lines_written": 0,
    }]


def test_build_turn_summary_file_write_note_counts_lines():
    h = _make_handler_for_render()
    content = "line1\nline2\nline3"
    messages = [
        Message(role="user", content="write"),
        Message(role="assistant", content="", tool_calls=[{
            "id": "tc-w",
            "type": "function",
            "function": {
                "name": "file_write",
                "arguments": json.dumps({
                    "path": "/workspace/new.py", "content": content,
                }),
            },
        }]),
        Message(role="tool", content="ok", tool_call_id="tc-w"),
    ]
    summary = h._build_turn_summary(
        messages=messages, user_goal="write",
        termination_reason="model_stop",
    )
    assert summary["files_edited"][0]["summary"] == "wrote 3 lines"


def test_build_turn_summary_outcome_mapping():
    h = _make_handler_for_render()
    cases = [
        ("model_stop", "done"),
        ("model_stop_after_nudge", "done"),
        ("max_iterations_reached", "incomplete"),
        ("validation_error_streak", "stopped (tool errors)"),
        ("backend_error", "stopped (backend error)"),
        ("something_weird", "something_weird"),
    ]
    for term, expected in cases:
        s = h._build_turn_summary(
            messages=[Message(role="user", content="x")],
            user_goal="x", termination_reason=term,
        )
        assert s["outcome"] == expected, f"term={term}"


# ---------------------------------------------------------------------------
# End-to-end: hybrid loop writes turn summary and next call sees it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_writes_turn_summary_on_completion(monkeypatch):
    """After ``_act_hybrid`` returns, ``state.turn_summaries`` has a
    new entry with the files read during the turn."""
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_read", output="body")],
    )

    class _ReadThenStop:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, "tc-1", "file_read",
                              {"path": "/workspace/x.py"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(content_delta="done!",
                                 done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _ReadThenStop(), session_id="sess-turn1",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-turn1",
    )

    assert handler._state.turn_summaries == []
    async for _ in handler._act_hybrid(
        _make_request("explain x.py"), workspace_context="",
    ):
        pass

    assert len(handler._state.turn_summaries) == 1
    s = handler._state.turn_summaries[0]
    assert "/workspace/x.py" in s["files_read"]
    assert s["outcome"] == "done"


@pytest.mark.asyncio
async def test_hybrid_skips_summary_when_no_tools_used(monkeypatch):
    """A turn that never called a tool produces no summary — a useless
    stub would just waste a slot in the 10-item ring."""
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_read")],
    )

    class _JustTalks:
        async def chat_stream(self, request):
            yield _FakeChunk(content_delta="just chatting",
                             done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _JustTalks(), session_id="sess-chat",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-chat",
    )
    async for _ in handler._act_hybrid(
        _make_request("hi"), workspace_context="",
    ):
        pass
    assert handler._state.turn_summaries == []


@pytest.mark.asyncio
async def test_prior_turn_appears_in_next_request_system_prompt(monkeypatch):
    """Integration: after turn 1 finishes, turn 2's request payload (seen
    by the backend) contains the <prior_turns> block mentioning the
    file the model read in turn 1. The whole point of the fix.

    Position note: prior_turns lives in the per-turn runtime carrier
    (a user-role message inserted before the latest user turn) rather
    than the leading system block, so the long system prefix stays
    cache-stable for llama-server slot reuse. The block still reaches
    the model — just one message later in the payload.
    """
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_read", output="body")],
    )

    seen_payloads: list[str] = []

    class _TwoTurn:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            # Capture the full payload (all messages concatenated) so we
            # catch prior_turns wherever it lives — system prefix OR the
            # user-role runtime carrier (its current home).
            seen_payloads.append(
                "\n".join(m.content or "" for m in request.messages)
            )

            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, "tc-a", "file_read",
                              {"path": "/workspace/auth.py"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            elif self.calls == 2:
                yield _FakeChunk(content_delta="turn 1 done",
                                 done=True, finish_reason="stop")
            else:
                # Turn 2: just stop immediately
                yield _FakeChunk(content_delta="turn 2 done",
                                 done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _TwoTurn(), session_id="sess-two-turn",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-two-turn",
    )

    # --- TURN 1 ---
    req1 = _make_request("inspect auth.py")
    async for _ in handler._act_hybrid(req1, workspace_context=""):
        pass
    assert len(handler._state.turn_summaries) == 1, (
        "Turn 1 should have written a summary"
    )

    # Turn 1's own payload should NOT contain a prior_turns block
    # (first turn — nothing to inject yet).
    assert "<prior_turns" not in seen_payloads[0]

    # --- TURN 2 ---
    handler._reset_for_new_request()
    req2 = _make_request("now fix the bug in it")
    async for _ in handler._act_hybrid(req2, workspace_context=""):
        pass

    # Turn 2's payload (seen by the backend) must contain the
    # prior_turns block with auth.py referenced.
    turn2_payloads = [p for p in seen_payloads[2:] if "<prior_turns" in p]
    assert turn2_payloads, (
        "Turn 2's payload should contain the <prior_turns> block with "
        "the file read in turn 1 — that's the whole fix."
    )
    assert "auth.py" in turn2_payloads[0]


@pytest.mark.asyncio
async def test_save_session_state_returns_true_on_write(tmp_path):
    """A normal upsert reports success."""
    from augmentum.coder.state import CoderState
    from augmentum.state.coder_persistence import CoderPersistence

    backend = SQLiteBackend(str(tmp_path / "coder-save.db"))
    await backend.connect()
    try:
        await backend.conn.execute(
            "INSERT INTO users (id, username, display_name, password_hash, role) "
            "VALUES (?, ?, ?, ?, ?)",
            ("alice", "alice", "Alice", "pw", "user"),
        )
        await backend.conn.commit()
        p = CoderPersistence(backend.conn)
        state = CoderState(session_id="ws-1", workspace_id="ws-1")
        ok = await p.save_session_state("ws-1", state, user_id="alice")
        assert ok is True
        # Idempotent re-save by the same owner still reports success.
        ok2 = await p.save_session_state("ws-1", state, user_id="alice")
        assert ok2 is True
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_save_session_state_blocked_by_other_owner(tmp_path):
    """A cross-user session_id collision must NOT silently succeed."""
    from augmentum.coder.state import CoderState
    from augmentum.state.coder_persistence import CoderPersistence

    backend = SQLiteBackend(str(tmp_path / "coder-block.db"))
    await backend.connect()
    try:
        for uid in ("alice", "bob"):
            await backend.conn.execute(
                "INSERT INTO users (id, username, display_name, password_hash, role) "
                "VALUES (?, ?, ?, ?, ?)",
                (uid, uid, uid.title(), "pw", "user"),
            )
        await backend.conn.commit()
        p = CoderPersistence(backend.conn)

        assert await p.save_session_state("ws-x", CoderState(session_id="ws-x", workspace_id="ws-x"), user_id="alice") is True
        # Bob tries to write the same session_id — blocked by the ON
        # CONFLICT ownership guard, must report False.
        assert await p.save_session_state("ws-x", CoderState(session_id="ws-x", workspace_id="ws-x"), user_id="bob") is False

        # Alice's row is intact (still hers).
        cur = await backend.conn.execute(
            "SELECT user_id FROM coder_sessions WHERE session_id = ?", ("ws-x",),
        )
        row = await cur.fetchone()
        assert row[0] == "alice"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_foreign_keys_enforced_after_migrations(tmp_path):
    """FK enforcement must survive a migration-applying boot.

    ``PRAGMA foreign_keys`` is silently ignored inside a transaction;
    migration 081 flips it OFF in autocommit and tries to restore it
    after its DML opened the implicit transaction — so before the
    post-migration re-apply, every fresh install ran its entire first
    session unenforced.
    """
    backend = SQLiteBackend(str(tmp_path / "coder-fk.db"))
    await backend.connect()
    try:
        cur = await backend.conn.execute("PRAGMA foreign_keys")
        assert (await cur.fetchone())[0] == 1
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_save_session_state_unscoped_project_persists(tmp_path):
    """A session with no project (project_id='') must persist.

    '' is a VALUE to the FK checker (unlike NULL) — before the fix,
    every non-project coder session failed FOREIGN KEY on every save
    and silently lost tasks/mission state.
    """
    from augmentum.coder.state import CoderState
    from augmentum.state.coder_persistence import CoderPersistence

    backend = SQLiteBackend(str(tmp_path / "coder-noproj.db"))
    await backend.connect()
    try:
        await backend.conn.execute(
            "INSERT INTO users (id, username, display_name, password_hash, role) "
            "VALUES ('alice', 'alice', 'Alice', 'pw', 'user')",
        )
        await backend.conn.commit()
        p = CoderPersistence(backend.conn)
        state = CoderState(session_id="ws-np", workspace_id="ws-np")
        assert state.project_id == ""
        assert await p.save_session_state("ws-np", state, user_id="alice") is True
        cur = await backend.conn.execute(
            "SELECT project_id FROM coder_sessions WHERE session_id = 'ws-np'",
        )
        assert (await cur.fetchone())[0] is None
        # Round-trip: NULL maps back to the legacy '' shape.
        loaded = await p.load_session_state("ws-np", user_id="alice")
        assert loaded is not None and loaded.project_id == ""
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_save_session_state_heals_stale_project_ref(tmp_path):
    """A project deleted mid-session must not kill persistence forever.

    SQLite SET-NULLs the stored row, but in-memory state re-asserts the
    stale id on every save — the retry drops the dangling ref, persists,
    and clears state.project_id so later saves are clean.
    """
    from augmentum.coder.state import CoderState
    from augmentum.state.coder_persistence import CoderPersistence

    backend = SQLiteBackend(str(tmp_path / "coder-staleproj.db"))
    await backend.connect()
    try:
        await backend.conn.execute(
            "INSERT INTO users (id, username, display_name, password_hash, role) "
            "VALUES ('alice', 'alice', 'Alice', 'pw', 'user')",
        )
        await backend.conn.commit()
        p = CoderPersistence(backend.conn)
        state = CoderState(
            session_id="ws-sp", workspace_id="ws-sp", project_id="proj-ghost",
        )
        # proj-ghost does not exist in projects — the deleted-project shape.
        assert await p.save_session_state("ws-sp", state, user_id="alice") is True
        assert state.project_id == ""
        cur = await backend.conn.execute(
            "SELECT project_id FROM coder_sessions WHERE session_id = 'ws-sp'",
        )
        assert (await cur.fetchone())[0] is None
    finally:
        await backend.close()
