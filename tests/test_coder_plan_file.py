"""Tests for the persistent plan.md attention-anchor artifact.

Adopted 2026-04-20 from Manus's todo.md pattern. Plan phase writes
/workspace/.augmentum/plan.md; the sticky system-reminder reads it
every iteration so the plan content always lives in the context tail
(compaction-safe). The agent can edit plan.md via normal file tools,
and the reminder content refreshes.

Covers:
  1. ``_plan_phase`` persists the generated plan to plan.md.
  2. ``_read_plan_md`` returns content on success, "" on any failure.
  3. ``_build_sticky_reminder`` includes a ``Plan (plan.md):`` section
     when content is provided.
  4. Long plans get truncated to 2000 chars with a "full content at
     path" suffix.
  5. Empty plan_md → no Plan section in the reminder.
  6. End-to-end: the hybrid loop's sticky reminder contains plan.md
     contents read from the container.

Run: python -m pytest tests/test_coder_plan_file.py -v
"""
from __future__ import annotations

import pytest

from augmentum.modes.coder.handler import CoderHandler
from tests.test_coder_handler import (
    _FakeBackend,
    _FakeChunk,
    _FakeTool,
    _force_native_tier,
    _make_request,
    _make_turn_context,
    _tc_delta,
)

# ---------------------------------------------------------------------------
# Helper container — records file_write calls and canned file_read
# ---------------------------------------------------------------------------


class _PlanCM:
    """Minimal container stub: records writes, returns canned reads."""

    def __init__(self, *, plan_md_content: str = "") -> None:
        self.writes: list[tuple[str, str]] = []
        self.shell_cmds: list[str] = []
        self.plan_md_content = plan_md_content
        # Start with a benign file_list so the snapshot refresh doesn't
        # throw in act_phase.

    async def file_read(self, workspace_id, path):  # noqa: ARG002
        if path == "/workspace/.augmentum/plan.md":
            return self.plan_md_content
        return ""

    async def file_write(self, workspace_id, path, content):  # noqa: ARG002
        self.writes.append((path, content))
        if path == "/workspace/.augmentum/plan.md":
            # Subsequent reads reflect the write
            self.plan_md_content = content

    async def run_command(self, *args, **kwargs):
        return await self._run_command(*args, **kwargs)

    async def _run_command(self, workspace_id, cmd, timeout=None):  # noqa: ARG002
        cmd_str = cmd[-1] if isinstance(cmd, list) else str(cmd)
        self.shell_cmds.append(cmd_str)
        return ""

    async def file_list(self, workspace_id, path):  # noqa: ARG002
        return []


# ---------------------------------------------------------------------------
# Plan phase persists to plan.md
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_phase_writes_plan_md(monkeypatch):
    """Running _plan_phase with a non-empty plan result should write
    the plan text to /workspace/.augmentum/plan.md via file_write,
    after ensuring the parent directory exists."""
    plan_text = "Plan: do the thing\n\n1. Read x.py\n2. Edit x.py\n3. Test"
    plan_chunks = [
        _FakeChunk(content_delta=plan_text),
        _FakeChunk(done=True, finish_reason="stop"),
    ]
    cm = _PlanCM()
    handler = CoderHandler(
        _FakeBackend(plan_chunks),
        session_id="sess-plan",
        container_manager=cm,
        workspace_id="ws-plan",
    )

    async for _ in handler._plan_phase(
        _make_request("build the thing"),
        _make_turn_context(),
    ):
        pass

    # plan.md written with the generated plan text
    plan_writes = [w for w in cm.writes if w[0] == "/workspace/.augmentum/plan.md"]
    assert plan_writes, f"expected plan.md write; got {cm.writes}"
    assert plan_writes[0][1] == plan_text

    # mkdir -p /workspace/.augmentum should have fired before the write
    assert any("mkdir -p" in c and ".augmentum" in c for c in cm.shell_cmds)


@pytest.mark.asyncio
async def test_plan_phase_write_failure_does_not_break_flow(monkeypatch):
    """Container write failure during plan.md persistence should not
    raise — best-effort only; the in-memory state.tasks still seeds."""
    class _FailingCM(_PlanCM):
        async def file_write(self, workspace_id, path, content):
            raise RuntimeError("disk full")

    plan_text = "Plan: stuff\n\n1. Step one\n2. Step two"
    plan_chunks = [
        _FakeChunk(content_delta=plan_text),
        _FakeChunk(done=True, finish_reason="stop"),
    ]
    cm = _FailingCM()
    handler = CoderHandler(
        _FakeBackend(plan_chunks),
        session_id="sess-fail",
        container_manager=cm,
        workspace_id="ws-fail",
    )
    # Must not raise
    async for _ in handler._plan_phase(
        _make_request("x"),
        _make_turn_context(),
    ):
        pass
    # Task list still seeded despite write failure
    assert len(handler._state.tasks) == 2


# ---------------------------------------------------------------------------
# _read_plan_md
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_plan_md_returns_content():
    cm = _PlanCM(plan_md_content="## Plan\n\n- step 1\n- step 2")
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="s", container_manager=cm, workspace_id="w",
    )
    content = await handler._read_plan_md()
    assert content == "## Plan\n\n- step 1\n- step 2"


@pytest.mark.asyncio
async def test_read_plan_md_empty_when_missing():
    class _NoFile(_PlanCM):
        async def file_read(self, workspace_id, path):
            raise FileNotFoundError(path)

    handler = CoderHandler(
        _FakeBackend([]),
        session_id="s", container_manager=_NoFile(), workspace_id="w",
    )
    assert await handler._read_plan_md() == ""


@pytest.mark.asyncio
async def test_read_plan_md_empty_without_container():
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="s", container_manager=None,
    )
    assert await handler._read_plan_md() == ""


# ---------------------------------------------------------------------------
# Sticky reminder renders plan.md section
# ---------------------------------------------------------------------------


def test_sticky_reminder_includes_plan_md_section():
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="s", container_manager=_PlanCM(), workspace_id="w",
    )
    reminder = handler._build_sticky_reminder(
        goal="build X", iteration=5, max_iters=100, writes=0,
        plan_md="## Plan\n\n1. Do thing\n2. Do other thing",
    )
    assert "Plan (plan.md):" in reminder
    assert "1. Do thing" in reminder


def test_sticky_reminder_omits_plan_when_empty():
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="s", container_manager=_PlanCM(), workspace_id="w",
    )
    reminder = handler._build_sticky_reminder(
        goal="x", iteration=1, max_iters=100, writes=0, plan_md="",
    )
    assert "Plan (plan.md):" not in reminder


def test_sticky_reminder_clips_long_plan():
    handler = CoderHandler(
        _FakeBackend([]),
        session_id="s", container_manager=_PlanCM(), workspace_id="w",
    )
    long_plan = "step\n" * 1000  # ~5000 chars
    reminder = handler._build_sticky_reminder(
        goal="x", iteration=1, max_iters=100, writes=0,
        plan_md=long_plan,
    )
    # Clipped with a truncation hint — full content recoverable.
    assert "plan truncated" in reminder
    assert ".augmentum/plan.md" in reminder
    # The "step\n" body portion of the plan stanza must not exceed
    # the 2000-char cap, measured up to the truncation marker. Body
    # AFTER the marker includes other stanzas (Tasks, Iteration) which
    # have their own budgets.
    marker = "\n... (plan truncated"
    plan_body = reminder.split("Plan (plan.md):\n", 1)[1].split(marker, 1)[0]
    assert len(plan_body) <= 2000


# ---------------------------------------------------------------------------
# End-to-end: hybrid loop picks up plan.md into the sticky reminder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_loop_injects_plan_md_into_backend_system_prompt(monkeypatch):
    """The backend sees the sticky reminder at the end of messages each
    iteration. That reminder must include the plan.md content."""
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_read", output="ok")],
    )

    plan_md = "## Plan\n\n1. read file X\n2. make a change\n3. run tests"

    seen_messages: list[list] = []

    class _Recorder:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            seen_messages.append(list(request.messages))
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, "tc-1", "file_read", {"path": "/x.py"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    cm = _PlanCM(plan_md_content=plan_md)
    handler = CoderHandler(
        _Recorder(), session_id="sess-e2e",
        container_manager=cm, workspace_id="ws-e2e",
    )
    async for _ in handler._act_hybrid(
        _make_request("implement foo"), workspace_context="",
    ):
        pass

    # At least one iteration's messages should include a sticky
    # reminder with the plan content.
    found_plan_in_reminder = False
    for msgs in seen_messages:
        for m in msgs:
            if (
                m.role == "user"
                and isinstance(m.content, str)
                and m.content.startswith("<system-reminder>")
                and "Plan (plan.md):" in m.content
                and "run tests" in m.content
            ):
                found_plan_in_reminder = True
                break
    assert found_plan_in_reminder, (
        "Expected plan.md content to appear inside the sticky reminder "
        "on at least one iteration"
    )
