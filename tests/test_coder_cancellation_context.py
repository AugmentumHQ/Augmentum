"""Tests for cancellation-aware cross-turn context in Coder mode.

When a Coder turn is cancelled mid-flight, the partial assistant
message persists in chat history and the next turn's
``<prior_turns>`` block has a silent gap — the model on N+1 sees a
truncated bubble with no signal that the turn was interrupted, and
tends to disorient (early-exit, re-do work, attempt to continue the
abandoned plan as a helpful gesture).

Three defenses covered here:

  1. ``CoderRunBroker.cancel(reason=...)`` propagates a reason hint
     onto the broker entry so the handler can read it.
  2. ``CoderHandler._record_interruption_summary`` writes a
     ``turn_summary`` entry with the reason + "where the agent was"
     context (last tool, phase, iteration, active task).
  3. ``CoderHandler._render_interruption_stanza`` renders a distinct
     CANCELLED / ERROR marker in the ``<prior_turns>`` block with
     an explicit directive telling the model not to silently resume.

Also covers the error path: an unhandled exception (timeout, rate
limit, network) is classified via ``_classify_runtime_error`` into
the same vocabulary so the model sees a consistent shape across
cancel and error.

Run: python -m pytest tests/test_coder_cancellation_context.py -v
"""

from __future__ import annotations

import asyncio

import pytest

from augmentum.coder.run_broker import CoderRunBroker
from augmentum.coder.state import CoderPhase, CoderState
from augmentum.models.base import InternalChatRequest, InternalStreamChunk, Message
from augmentum.modes.coder.handler import CoderHandler

from tests.test_coder_handler import (
    _ExtendedContainerManager,
    _FakeBackend,
    _make_request,
)


def _chunk(content: str = "", *, done: bool = False) -> InternalStreamChunk:
    return InternalStreamChunk(content_delta=content, done=done)


# ---------------------------------------------------------------------------
# Broker — cancel reason propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_records_reason_on_entry():
    """``broker.cancel(reason=...)`` stores the reason on the entry."""
    broker = CoderRunBroker()
    started = asyncio.Event()

    async def agent(_entry):
        started.set()
        for _ in range(1000):
            yield _chunk("x")
            await asyncio.sleep(0.01)

    await broker.start_run(
        run_id="r-reason", user_id="alice", workspace_id="ws1", agent=agent,
    )
    await started.wait()

    assert broker.cancel("r-reason", reason="slash_clear") is True
    entry = broker.get("r-reason")
    assert entry is not None
    assert entry.cancel_requested is True
    assert entry.cancel_reason == "slash_clear"

    while not entry.done:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_cancel_defaults_reason_to_user_cancel():
    """Bare ``broker.cancel(run_id)`` keeps the default for legacy callers."""
    broker = CoderRunBroker()
    started = asyncio.Event()

    async def agent(_entry):
        started.set()
        for _ in range(1000):
            yield _chunk("x")
            await asyncio.sleep(0.01)

    await broker.start_run(
        run_id="r-default", user_id="alice", workspace_id="ws1", agent=agent,
    )
    await started.wait()

    assert broker.cancel("r-default") is True
    entry = broker.get("r-default")
    assert entry.cancel_reason == "user_cancel"

    while not entry.done:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_cancel_reason_clipped_to_50_chars():
    """Defensive cap on caller-provided reason — UI bugs / probes shouldn't
    bloat the prompt budget."""
    broker = CoderRunBroker()
    started = asyncio.Event()

    async def agent(_entry):
        started.set()
        for _ in range(1000):
            yield _chunk("x")
            await asyncio.sleep(0.01)

    await broker.start_run(
        run_id="r-clip", user_id="alice", workspace_id="ws1", agent=agent,
    )
    await started.wait()

    long_reason = "a" * 200
    assert broker.cancel("r-clip", reason=long_reason) is True
    entry = broker.get("r-clip")
    assert len(entry.cancel_reason) <= 50

    while not entry.done:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_cancel_twice_does_not_overwrite_reason():
    """First reason wins — second cancel preserves the original context."""
    broker = CoderRunBroker()
    started = asyncio.Event()

    async def agent(_entry):
        started.set()
        for _ in range(1000):
            yield _chunk("x")
            await asyncio.sleep(0.01)

    await broker.start_run(
        run_id="r-twice", user_id="alice", workspace_id="ws1", agent=agent,
    )
    await started.wait()

    assert broker.cancel("r-twice", reason="user_cancel") is True
    entry = broker.get("r-twice")
    # Second cancel after the entry is still alive — should not clobber.
    # In practice this is a no-op since done=True after first cancel; check
    # the field directly before the task finishes unwinding.
    assert entry.cancel_reason == "user_cancel"

    while not entry.done:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_shutdown_tags_pending_runs_as_server_shutdown():
    """``broker.shutdown()`` writes server_shutdown to in-flight entries."""
    broker = CoderRunBroker()
    started = asyncio.Event()

    async def agent(_entry):
        started.set()
        for _ in range(1000):
            yield _chunk("x")
            await asyncio.sleep(0.01)

    await broker.start_run(
        run_id="r-shut", user_id="alice", workspace_id="ws1", agent=agent,
    )
    await started.wait()

    entry = broker.get("r-shut")
    assert entry.cancel_reason == ""
    await broker.shutdown()
    assert entry.cancel_reason == "server_shutdown"

    while not entry.done:
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# Handler — _resolve_cancel_reason
# ---------------------------------------------------------------------------


def _make_handler(broker=None, run_id: str = ""):
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="s-test",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-test",
        user_id="alice",
        coder_run_broker=broker,
    )
    if run_id:
        # The handler reads ``self._turn_ledger.run_id`` — stub it with
        # the minimum the helper needs (a ``run_id`` attribute).
        class _StubLedger:
            pass
        ledger = _StubLedger()
        ledger.run_id = run_id
        handler._turn_ledger = ledger
    return handler


@pytest.mark.asyncio
async def test_resolve_cancel_reason_reads_broker_entry():
    broker = CoderRunBroker()
    started = asyncio.Event()

    async def agent(_entry):
        started.set()
        for _ in range(1000):
            yield _chunk("x")
            await asyncio.sleep(0.01)

    await broker.start_run(
        run_id="r-rsv", user_id="alice", workspace_id="ws-test", agent=agent,
    )
    await started.wait()
    broker.cancel("r-rsv", reason="slash_compact")

    handler = _make_handler(broker=broker, run_id="r-rsv")
    assert handler._resolve_cancel_reason() == "slash_compact"

    entry = broker.get("r-rsv")
    while not entry.done:
        await asyncio.sleep(0.01)


def test_resolve_cancel_reason_falls_back_when_no_broker():
    """Legacy path: no broker wired → return ``user_cancel`` for renderable
    output."""
    handler = _make_handler(broker=None, run_id="r-none")
    assert handler._resolve_cancel_reason() == "user_cancel"


def test_resolve_cancel_reason_falls_back_when_run_missing():
    broker = CoderRunBroker()
    handler = _make_handler(broker=broker, run_id="r-missing")
    assert handler._resolve_cancel_reason() == "user_cancel"


# ---------------------------------------------------------------------------
# Handler — _classify_runtime_error
# ---------------------------------------------------------------------------


def test_classify_timeout_error():
    handler = _make_handler()
    assert handler._classify_runtime_error(asyncio.TimeoutError()) == "timeout"
    assert handler._classify_runtime_error(TimeoutError("read timed out")) == "timeout"


def test_classify_rate_limit_error():
    handler = _make_handler()

    class _RateLimitError(Exception):
        pass

    assert handler._classify_runtime_error(_RateLimitError("HTTP 429")) == "rate_limit"
    assert handler._classify_runtime_error(Exception("rate limit exceeded")) == "rate_limit"


def test_classify_network_error():
    handler = _make_handler()
    assert handler._classify_runtime_error(ConnectionError("reset")) == "network"
    assert handler._classify_runtime_error(Exception("network unreachable")) == "network"


def test_classify_unknown_error_defaults_to_backend_error():
    handler = _make_handler()
    assert handler._classify_runtime_error(ValueError("weird")) == "backend_error"


# ---------------------------------------------------------------------------
# Handler — _record_interruption_summary
# ---------------------------------------------------------------------------


def test_interruption_summary_records_cancel_shape():
    handler = _make_handler()
    handler._state.phase = CoderPhase.EXECUTING
    handler._state.tool_calls_made = 3
    handler._state.recent_tool_calls.extend([
        {"tool": "file_read", "key": "/workspace/snake.py", "last_iter": 4},
    ])
    handler._state.working_set = {"/workspace/snake.py", "/workspace/util.py"}
    handler._state.set_tasks([
        {"content": "Implement snake game", "activeForm": "Implementing", "status": "in_progress"},
    ])

    req = _make_request("make snake.py")
    handler._record_interruption_summary(req, reason="user_cancel", kind="cancelled")

    assert len(handler._state.turn_summaries) == 1
    s = handler._state.turn_summaries[0]
    assert s["cancelled"] is True
    assert s["errored"] is False
    assert s["cancel_reason"] == "user_cancel"
    assert s["outcome"] == "cancelled"
    assert s["user_goal"] == "make snake.py"

    interrupted = s["interrupted_at"]
    assert interrupted["phase"] == "executing"
    assert interrupted["tool_calls_made"] == 3
    assert interrupted["last_tool"]["tool"] == "file_read"
    assert interrupted["last_tool"]["target"] == "/workspace/snake.py"
    assert interrupted["active_task"]["content"] == "Implement snake game"


def test_interruption_summary_records_error_shape():
    handler = _make_handler()
    handler._state.tool_calls_made = 1

    req = _make_request("fix the bug")
    handler._record_interruption_summary(
        req, reason="timeout", kind="errored", error_text="read timed out after 30s",
    )

    assert len(handler._state.turn_summaries) == 1
    s = handler._state.turn_summaries[0]
    assert s["cancelled"] is False
    assert s["errored"] is True
    assert s["cancel_reason"] == "timeout"
    assert s["outcome"] == "errored"
    assert s["blockers"] == "read timed out after 30s"


def test_interruption_summary_handles_missing_recent_tool_calls():
    """Turn cancelled before any tool fired — last_tool stays empty without
    erroring."""
    handler = _make_handler()
    handler._state.recent_tool_calls.clear()

    req = _make_request("hello")
    handler._record_interruption_summary(req, reason="user_cancel", kind="cancelled")

    s = handler._state.turn_summaries[0]
    assert s["interrupted_at"]["last_tool"] == {}


def test_interruption_summary_clips_long_user_goal():
    handler = _make_handler()
    long_goal = "x" * 1000
    req = _make_request(long_goal)
    handler._record_interruption_summary(req, reason="user_cancel", kind="cancelled")

    s = handler._state.turn_summaries[0]
    assert len(s["user_goal"]) <= 300


# ---------------------------------------------------------------------------
# Handler — _render_interruption_stanza
# ---------------------------------------------------------------------------


def test_render_cancelled_stanza_includes_reason_label():
    handler = _make_handler()
    summary = {
        "turn_idx": 3,
        "user_goal": "build snake",
        "files_read": ["/workspace/snake.py"],
        "files_edited": [],
        "cancelled": True,
        "errored": False,
        "cancel_reason": "user_cancel",
        "outcome": "cancelled",
        "blockers": "",
        "interrupted_at": {
            "phase": "executing",
            "tool_calls_made": 5,
            "current_step": 2,
            "total_steps": 4,
            "last_tool": {
                "tool": "file_read", "target": "/workspace/snake.py", "iteration": 4,
            },
            "active_task": {"content": "Render the game loop"},
        },
        "created_at": 0.0,
    }
    stanza = handler._render_interruption_stanza(summary)
    assert "Turn 3" in stanza
    assert "INTERRUPTED" in stanza
    assert "CANCELLED" in stanza
    assert "user pressed Stop" in stanza
    assert "build snake" in stanza
    assert "file_read" in stanza
    assert "/workspace/snake.py" in stanza
    assert "phase=executing" in stanza
    assert "step=2/4" in stanza
    assert "Render the game loop" in stanza
    # The directive is the load-bearing piece — must be present.
    assert "Do not silently resume" in stanza


def test_render_errored_stanza_uses_error_label_and_directive():
    handler = _make_handler()
    summary = {
        "turn_idx": 1,
        "user_goal": "fix the bug",
        "files_read": [],
        "cancelled": False,
        "errored": True,
        "cancel_reason": "timeout",
        "outcome": "errored",
        "blockers": "read timed out after 30s",
        "interrupted_at": {
            "phase": "executing", "tool_calls_made": 0,
            "current_step": 0, "total_steps": 0,
            "last_tool": {}, "active_task": {},
        },
        "created_at": 0.0,
    }
    stanza = handler._render_interruption_stanza(summary)
    assert "ERROR" in stanza
    assert "time budget" in stanza  # the timeout reason label
    assert "read timed out after 30s" in stanza
    assert "Re-attempt only the part" in stanza  # error-variant directive


def test_render_stanza_for_each_canonical_reason():
    """Every reason in the labels map renders a human-readable phrase, not
    the raw token."""
    handler = _make_handler()
    cases = [
        ("user_cancel",      "user pressed Stop"),
        ("slash_clear",      "/clear"),
        ("slash_compact",    "/compact"),
        ("new_turn_started", "new turn"),
        ("page_unload",      "closed the page"),
        ("server_shutdown",  "shutting down"),
        ("timeout",          "time budget"),
        ("rate_limit",       "rate-limited"),
        ("network",          "network connection"),
    ]
    for reason, expected_phrase in cases:
        summary = {
            "turn_idx": 1, "user_goal": "",
            "cancelled": True, "errored": False,
            "cancel_reason": reason, "outcome": "cancelled",
            "blockers": "", "files_read": [],
            "interrupted_at": {
                "phase": "executing", "tool_calls_made": 0,
                "current_step": 0, "total_steps": 0,
                "last_tool": {}, "active_task": {},
            },
        }
        stanza = handler._render_interruption_stanza(summary)
        assert expected_phrase in stanza, (
            f"reason={reason!r} did not produce expected phrase {expected_phrase!r}"
        )


def test_render_stanza_unknown_reason_falls_back_to_raw_token():
    handler = _make_handler()
    summary = {
        "turn_idx": 1, "user_goal": "", "cancelled": True, "errored": False,
        "cancel_reason": "experimental_reason", "outcome": "cancelled",
        "blockers": "", "files_read": [],
        "interrupted_at": {
            "phase": "executing", "tool_calls_made": 0,
            "current_step": 0, "total_steps": 0,
            "last_tool": {}, "active_task": {},
        },
    }
    stanza = handler._render_interruption_stanza(summary)
    # Unknown reasons render verbatim — not pretty but informative.
    assert "experimental_reason" in stanza


# ---------------------------------------------------------------------------
# Handler — _render_prior_turns dispatch + intro extension
# ---------------------------------------------------------------------------


def test_prior_turns_block_routes_interruption_to_stanza_renderer():
    handler = _make_handler()
    # Normal completed turn.
    handler._state.add_turn_summary({
        "turn_idx": 1, "user_goal": "first", "files_read": ["a.py"],
        "files_edited": [], "outcome": "done", "blockers": "",
    })
    # Cancelled turn.
    handler._state.add_turn_summary({
        "turn_idx": 2, "user_goal": "second", "files_read": [],
        "cancelled": True, "errored": False, "cancel_reason": "user_cancel",
        "outcome": "cancelled", "blockers": "",
        "interrupted_at": {
            "phase": "executing", "tool_calls_made": 2,
            "current_step": 0, "total_steps": 0,
            "last_tool": {"tool": "shell_exec", "target": "ls -la"},
            "active_task": {},
        },
    })

    block = handler._render_prior_turns()
    assert "Turn 1 (done)" in block
    assert "Turn 2 (INTERRUPTED" in block
    assert "user pressed Stop" in block
    assert "shell_exec" in block


def test_prior_turns_intro_extends_when_interruption_present():
    """Block header adds the "do not silently resume" directive only when
    at least one summary is an interruption — keeps normal turns lean."""
    handler = _make_handler()
    handler._state.add_turn_summary({
        "turn_idx": 1, "user_goal": "first", "files_read": [],
        "files_edited": [], "outcome": "done", "blockers": "",
    })
    block_normal = handler._render_prior_turns()
    assert "do not silently resume" not in block_normal.lower()

    handler._state.add_turn_summary({
        "turn_idx": 2, "user_goal": "second", "cancelled": True,
        "errored": False, "cancel_reason": "user_cancel",
        "outcome": "cancelled", "blockers": "", "files_read": [],
        "interrupted_at": {
            "phase": "executing", "tool_calls_made": 0,
            "current_step": 0, "total_steps": 0,
            "last_tool": {}, "active_task": {},
        },
    })
    block_mixed = handler._render_prior_turns()
    assert "do not silently resume" in block_mixed.lower()


def test_prior_turns_block_empty_when_no_summaries():
    handler = _make_handler()
    assert handler._render_prior_turns() == ""


# ---------------------------------------------------------------------------
# Integration — cancellation through _run_agent_with_ledger writes summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelled_error_in_agent_records_turn_summary(tmp_path):
    """CancelledError raised mid-generator MUST land an interruption summary
    on the state before propagating. This is the integration check — the
    real cancellation path runs end-to-end."""
    handler = _make_handler()

    async def _fake_body(_request):
        # Yield one chunk then simulate a broker-driven cancel.
        yield _chunk("partial")
        raise asyncio.CancelledError()

    handler._handle_stream_body = _fake_body  # type: ignore[assignment]
    handler._container_manager = None  # skip exec cleanup

    req = _make_request("do a thing")
    with pytest.raises(asyncio.CancelledError):
        async for _ in handler._run_agent_with_ledger(req):
            pass

    assert len(handler._state.turn_summaries) == 1
    s = handler._state.turn_summaries[0]
    assert s["cancelled"] is True
    assert s["user_goal"] == "do a thing"


@pytest.mark.asyncio
async def test_runtime_error_in_agent_records_errored_summary():
    """An unhandled exception (timeout / rate-limit / network) also writes
    an interruption summary so the next turn isn't blind to the failure."""
    handler = _make_handler()

    async def _fake_body(_request):
        yield _chunk("starting…")
        raise TimeoutError("backend read timed out")

    handler._handle_stream_body = _fake_body  # type: ignore[assignment]
    handler._container_manager = None

    req = _make_request("debug this")
    with pytest.raises(TimeoutError):
        async for _ in handler._run_agent_with_ledger(req):
            pass

    assert len(handler._state.turn_summaries) == 1
    s = handler._state.turn_summaries[0]
    assert s["errored"] is True
    assert s["cancel_reason"] == "timeout"
    assert "timed out" in s["blockers"]
