"""Tests for the native-mode guards added 2026-05-28.

Two narrow patches that close failure modes observed in live logs without
diluting what makes native lean:

  1. Silent-success-fog detector. When every shell_exec in an iter
     returns "(exit 0, no stdout)" and the streak hits
     ``_SILENT_SUCCESS_NUDGE_AT``, append a one-shot nudge pointing
     the model at a diagnostic command. Hybrid had this; native
     didn't, so the model could spiral through 6+ silent shells with
     no grounding signal. Fires once per turn — does not loop.

  2. Continuation-after-cancellation guidance. When the user message
     is a continuation request AND the most recent turn_summary is
     ``cancelled=True``, append a restate-before-act directive to
     the native system prompt. Narrow trigger — addendum is absent
     from every other turn shape so the spartan native sys_text stays
     spartan when it doesn't need help.

The "harm check" cases verify neither guard fires in normal native
operation (clean shell output, fresh turn, non-continuation message).

Run: python -m pytest tests/test_coder_native_guards.py -v
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from augmentum.modes.coder.handler import CoderHandler
from augmentum.models.base import InternalChatRequest, InternalStreamChunk, Message

from tests.test_coder_handler import (
    _ExtendedContainerManager,
    _FakeChunk,
    _FakeTool,
    _make_request,
    _tc_delta,
)


# ---------------------------------------------------------------------------
# Silent-success-fog detector
# ---------------------------------------------------------------------------


class _SilentShellLoop:
    """Backend that calls shell_exec N times, each returning silent
    success, then stops with substantive prose."""

    def __init__(self, silent_iters: int = 4):
        self.requests = []
        self.silent_iters = silent_iters

    async def chat_stream(self, request):
        self.requests.append(request)
        if len(self.requests) <= self.silent_iters:
            yield _FakeChunk(augmentum={"tool_calls": [
                _tc_delta(0, f"tc-{len(self.requests)}", "shell_exec",
                          {"command": "true"}),
            ]})
            yield _FakeChunk(done=True, finish_reason="tool_calls")
            return
        # Final: substantive prose so the TQG accepts the stop.
        yield _FakeChunk(content_delta=(
            "Done. All checks completed without issues. "
            "Workspace looks healthy across the requested probes."
        ))
        yield _FakeChunk(done=True, finish_reason="stop")

    async def chat(self, request):
        return None


@pytest.mark.asyncio
async def test_silent_success_streak_fires_nudge(monkeypatch):
    """3 consecutive iters of silent shell_exec → silent_success_nudge fires.

    Without this guard the model loops on the silent-fog pattern observed
    in production 2026-05-28.
    """
    silent_tool = _FakeTool("shell_exec")
    # FakeTool default returns success with output="ok" — override to
    # the literal sentinel that real shell_exec emits on empty stdout.
    from augmentum.tools.base import ToolResult
    silent_tool.execute = lambda **_k: _async_result(ToolResult(
        success=True,
        output="(exit 0, command succeeded with no stdout)",
    ))

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [silent_tool],
    )
    backend = _SilentShellLoop(silent_iters=4)
    handler = CoderHandler(
        backend,
        session_id="s", container_manager=_ExtendedContainerManager(),
        workspace_id="ws",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(
        _make_request("debug the issue"), workspace_context="",
    ):
        chunks.append(c)

    # Loop-health arbitration (loop_health.py, 2026-07-07): this
    # scenario trips TWO detectors on the same underlying loop —
    # identical_result (higher priority) and silent_success. Exactly
    # ONE nudge lands; the other is reported as suppressed telemetry.
    fired = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") in
        {"silent_success_nudge", "identical_result_nudge"}
    ]
    assert len(fired) == 1, (
        "Expected exactly one arbitrated nudge — observed "
        f"{len(fired)} (arbitration or detector regression)."
    )
    assert fired[0].augmentum.get("status") == "identical_result_nudge"
    assert fired[0].augmentum.get("strategy") == "native"
    suppressed = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") == "loop_health_suppressed"
        and c.augmentum.get("kind") == "silent_success_nudge"
    ]
    assert len(suppressed) == 1


async def _async_result(result):
    return result


@pytest.mark.asyncio
async def test_silent_success_nudge_only_fires_once(monkeypatch):
    """Even with 10 silent iters, the nudge body is appended exactly once.

    Prevents native from nagging on every iter once the streak hits
    threshold — the original hybrid contract is "one shot per turn".
    """
    silent_tool = _FakeTool("shell_exec")
    from augmentum.tools.base import ToolResult
    silent_tool.execute = lambda **_k: _async_result(ToolResult(
        success=True,
        output="(exit 0, command succeeded with no stdout)",
    ))
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [silent_tool],
    )
    backend = _SilentShellLoop(silent_iters=10)
    handler = CoderHandler(
        backend,
        session_id="s", container_manager=_ExtendedContainerManager(),
        workspace_id="ws",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(
        _make_request("debug"), workspace_context="",
    ):
        chunks.append(c)

    # Under loop-health arbitration the higher-priority identical_result
    # nudge wins the iteration; silent_success is suppressed (one-shot
    # flag still flips, so neither ever repeats).
    fired = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") in
        {"silent_success_nudge", "identical_result_nudge"}
    ]
    assert len(fired) == 1

    # And at most ONE loop-health nudge per iteration lands in history;
    # across 10 silent iters the one-shot flags keep the total at one.
    last_req = backend.requests[-1]
    nudge_msgs = [
        m for m in last_req.messages
        if m.role == "user" and "<nudge>" in (m.content or "")
        and ("exit 0, no stdout" in (m.content or "")
             or "same arguments" in (m.content or ""))
    ]
    assert len(nudge_msgs) == 1


@pytest.mark.asyncio
async def test_non_silent_shell_resets_streak(monkeypatch):
    """A shell_exec with real output resets the streak — no premature nudge."""
    # Mixed tool: alternates silent and noisy outputs by call count.
    from augmentum.tools.base import ToolResult
    counter = {"n": 0}

    async def _exec(**_k):
        counter["n"] += 1
        if counter["n"] in (1, 3, 5):
            return ToolResult(success=True, output="(exit 0, command succeeded with no stdout)")
        return ToolResult(success=True, output="real output line\nanother line")

    mixed_tool = _FakeTool("shell_exec")
    mixed_tool.execute = _exec
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [mixed_tool],
    )

    class _AlternatingBackend:
        def __init__(self):
            self.requests = []

        async def chat_stream(self, request):
            self.requests.append(request)
            if len(self.requests) < 6:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, f"tc-{len(self.requests)}", "shell_exec",
                              {"command": "x"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
                return
            yield _FakeChunk(content_delta=(
                "Done. The workspace is in the expected state. "
                "Six probes completed successfully."
            ))
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _AlternatingBackend()
    handler = CoderHandler(
        backend,
        session_id="s", container_manager=_ExtendedContainerManager(),
        workspace_id="ws",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(
        _make_request("debug"), workspace_context="",
    ):
        chunks.append(c)

    nudges = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "silent_success_nudge"
    ]
    # Pattern was silent, noisy, silent, noisy, silent, noisy — max
    # streak of 1. Nudge must not fire.
    assert nudges == []


# ---------------------------------------------------------------------------
# Harm check: clean native turns get no nudge + no addendum
# ---------------------------------------------------------------------------


class _CleanNative:
    """Backend that runs one productive read tool then summarizes."""

    def __init__(self):
        self.requests = []

    async def chat_stream(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            yield _FakeChunk(augmentum={"tool_calls": [
                _tc_delta(0, "tc1", "file_list", {"path": "/workspace"}),
            ]})
            yield _FakeChunk(done=True, finish_reason="tool_calls")
            return
        yield _FakeChunk(content_delta=(
            "Found three files in the workspace. "
            "No further action needed."
        ))
        yield _FakeChunk(done=True, finish_reason="stop")

    async def chat(self, request):
        return None


@pytest.mark.asyncio
async def test_normal_native_turn_no_silent_nudge_no_cancel_addendum(monkeypatch):
    """The common case — clean tool use + non-continuation message — must
    NOT trigger either new guard. This is the "don't harm native's natural
    flow" pin.
    """
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_list", output="a.py\nb.py")],
    )
    backend = _CleanNative()
    handler = CoderHandler(
        backend,
        session_id="s", container_manager=_ExtendedContainerManager(),
        workspace_id="ws",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(
        _make_request("list the files"), workspace_context="",
    ):
        chunks.append(c)

    nudges = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "silent_success_nudge"
    ]
    assert nudges == [], "silent_success_nudge fired on a clean turn (regression)"

    # The system prompt must NOT contain the cancellation addendum on
    # a normal turn — that addendum is the most expensive token
    # injection of the two guards, so verifying it stays narrow is the
    # primary "don't bloat native" assertion.
    sys_msg = backend.requests[0].messages[0]
    assert sys_msg.role == "system"
    assert "continuation_after_cancel" not in sys_msg.content


# ---------------------------------------------------------------------------
# Continuation-after-cancellation guidance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_continue_after_cancel_injects_addendum(monkeypatch):
    """User says 'continue' + last turn_summary is cancelled → addendum
    appears in the native system prompt."""
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_list")],
    )

    class _AnySuccessBackend:
        def __init__(self):
            self.requests = []

        async def chat_stream(self, request):
            self.requests.append(request)
            yield _FakeChunk(content_delta=(
                "I understood the prior ask to be: list the files. "
                "Proceeding now to do that."
            ))
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _AnySuccessBackend()
    handler = CoderHandler(
        backend,
        session_id="s", container_manager=_ExtendedContainerManager(),
        workspace_id="ws",
    )
    # Seed a cancelled turn_summary so the guard's condition is met.
    handler._state.add_turn_summary({
        "turn_idx": 1,
        "user_goal": "set up credentials",
        "files_read": ["/workspace/.env"],
        "cancelled": True,
        "errored": False,
        "cancel_reason": "user_cancel",
        "outcome": "cancelled",
        "blockers": "",
        "interrupted_at": {
            "phase": "executing", "tool_calls_made": 2,
            "current_step": 0, "total_steps": 0,
            "last_tool": {"tool": "shell_exec", "target": "ls /workspace"},
            "active_task": {},
        },
    })

    async for _ in handler._act_native(
        _make_request("continue"), workspace_context="",
    ):
        pass

    sys_msg = backend.requests[0].messages[0]
    assert sys_msg.role == "system"
    assert "continuation_after_cancel" in sys_msg.content
    # The addendum must point the model at the INTERRUPTED stanza.
    assert "INTERRUPTED" in sys_msg.content
    # And include the "ask one targeted clarifying question" escape
    # hatch so the model doesn't have to fabricate context.
    assert "clarifying question" in sys_msg.content


@pytest.mark.asyncio
async def test_continue_after_clean_completion_no_addendum(monkeypatch):
    """User says 'continue' but the last turn finished cleanly → no
    addendum. The guard is narrow to cancellation, not to all
    continuation patterns."""
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_list")],
    )

    class _PlainBackend:
        def __init__(self):
            self.requests = []

        async def chat_stream(self, request):
            self.requests.append(request)
            yield _FakeChunk(content_delta=(
                "Continuing from the prior pass. Everything looks good "
                "and there is nothing further to do."
            ))
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _PlainBackend()
    handler = CoderHandler(
        backend,
        session_id="s", container_manager=_ExtendedContainerManager(),
        workspace_id="ws",
    )
    # Clean completion in the prior turn — not cancelled.
    handler._state.add_turn_summary({
        "turn_idx": 1, "user_goal": "list files",
        "files_read": ["a.py"], "files_edited": [],
        "outcome": "done", "blockers": "",
    })

    async for _ in handler._act_native(
        _make_request("continue"), workspace_context="",
    ):
        pass

    sys_msg = backend.requests[0].messages[0]
    assert "continuation_after_cancel" not in sys_msg.content


@pytest.mark.asyncio
async def test_fresh_ask_after_cancel_no_addendum(monkeypatch):
    """A fresh user request after a cancelled turn — addendum doesn't
    fire because the message isn't a continuation. Cancellation context
    still surfaces via prior_turns (already covered by the cancellation
    tests); this just verifies the guard stays narrow."""
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_list")],
    )

    class _PlainBackend:
        def __init__(self):
            self.requests = []

        async def chat_stream(self, request):
            self.requests.append(request)
            yield _FakeChunk(content_delta=(
                "Starting on the new task right away. "
                "Will report back when there's something to show."
            ))
            yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    backend = _PlainBackend()
    handler = CoderHandler(
        backend,
        session_id="s", container_manager=_ExtendedContainerManager(),
        workspace_id="ws",
    )
    handler._state.add_turn_summary({
        "turn_idx": 1, "user_goal": "credentials setup",
        "cancelled": True, "errored": False,
        "cancel_reason": "user_cancel", "outcome": "cancelled",
        "blockers": "", "files_read": [],
        "interrupted_at": {
            "phase": "executing", "tool_calls_made": 0,
            "current_step": 0, "total_steps": 0,
            "last_tool": {}, "active_task": {},
        },
    })

    async for _ in handler._act_native(
        _make_request("now build a new module for X"),  # totally fresh ask
        workspace_context="",
    ):
        pass

    sys_msg = backend.requests[0].messages[0]
    assert "continuation_after_cancel" not in sys_msg.content


# ---------------------------------------------------------------------------
# Same-validation-error repeat breaker
# ---------------------------------------------------------------------------


class _LoopingValidationFailBackend:
    """Backend that keeps emitting the SAME validation-failing tool call.

    Models the 2026-05-29 transcript pattern: file_write called 6× in a
    row, each time with `path` missing because the args got truncated
    mid-stream. Pre-fix, native let this run to max_iterations because
    only canonical/hybrid had _find_repeat_offender wired."""

    def __init__(self, *, fail_iters: int = 10):
        self.requests: list = []
        self.fail_iters = fail_iters

    async def chat_stream(self, request):
        self.requests.append(request)
        if len(self.requests) > self.fail_iters:
            yield _FakeChunk(content_delta=(
                "Stopping — could not complete the request after many "
                "tool failures. Please try again with smaller changes."
            ))
            yield _FakeChunk(done=True, finish_reason="stop")
            return
        # Emit a file_write call with empty path — same fingerprint
        # every iteration so repeat_count climbs predictably.
        yield _FakeChunk(augmentum={"tool_calls": [
            _tc_delta(0, f"tc-{len(self.requests)}", "file_write",
                      {"path": "", "content": "body"}),
        ]})
        yield _FakeChunk(done=True, finish_reason="tool_calls")

    async def chat(self, request):
        return None


@pytest.mark.asyncio
async def test_native_breaks_on_repeated_validation_error(monkeypatch):
    """Native must hard-break on 2+ identical validation errors, matching
    canonical/hybrid. Pre-2026-05-29 native only had buddy-model
    escalation here, which is a no-op when no buddy is configured —
    the model could then loop on the SAME error until max_iters. Live
    transcript: 6× "file_write without 'path'" with no break."""

    from augmentum.tools.base import ToolResult

    fail_tool = _FakeTool("file_write")

    async def _exec(**_kwargs):
        return ToolResult(
            success=False,
            validation_error=True,
            error=(
                "file_write called without a 'path' argument. "
                "Required: path (string) + content (string)."
            ),
        )

    fail_tool.execute = _exec
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [fail_tool],
    )

    backend = _LoopingValidationFailBackend(fail_iters=10)
    handler = CoderHandler(
        backend,
        session_id="s", container_manager=_ExtendedContainerManager(),
        workspace_id="ws",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(
        _make_request("write a new file"), workspace_context="",
    ):
        chunks.append(c)

    breaks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "validation_error_break"
    ]
    assert len(breaks) == 1, (
        "Native didn't break on repeated identical validation errors. "
        "Either _find_repeat_offender isn't wired in _act_native, or "
        "record_validation_error isn't being called from the native "
        "tool-result path."
    )
    assert breaks[0].augmentum.get("tool") == "file_write"
    assert int(breaks[0].augmentum.get("repeat_count", 0)) >= 2

    assert len(backend.requests) <= 4, (
        f"Native made {len(backend.requests)} model calls; expected ≤4 "
        "(breaker should fire after 2 same-signature failures)."
    )
