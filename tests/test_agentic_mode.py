"""Tests for agentic mode — handler, task state, planner, and integration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from augmentum.classifier.router import Mode, MODE_MAP, MODE_PREFIXES
from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
)


# ---------------------------------------------------------------------------
# Mode / Classifier
# ---------------------------------------------------------------------------

class TestAgenticModeEnum:
    def test_agentic_mode_exists(self):
        assert Mode.AGENTIC == "agentic"

    def test_mode_prefix(self):
        assert "g/" in MODE_PREFIXES
        assert MODE_PREFIXES["g/"] == Mode.AGENTIC

    def test_mode_map(self):
        assert "agentic" in MODE_MAP
        assert MODE_MAP["agentic"] == Mode.AGENTIC

    def test_classifier_strips_prefix(self):
        from augmentum.classifier.router import RequestClassifier
        classifier = RequestClassifier()
        request = InternalChatRequest(
            model="g/llama3.1:8b",
            messages=[Message(role="user", content="Create a report")],
        )
        result = classifier.classify(request)
        assert result.mode == Mode.AGENTIC
        assert request.model == "llama3.1:8b"

    def test_classifier_header_override(self):
        from augmentum.classifier.router import RequestClassifier
        classifier = RequestClassifier()
        request = InternalChatRequest(
            model="llama3.1:8b",
            messages=[Message(role="user", content="hello")],
        )
        result = classifier.classify(request, mode_override="agentic")
        assert result.mode == Mode.AGENTIC


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class TestPlanner:
    def test_parse_plan_with_checklist(self):
        from augmentum.modes.agentic.planner import parse_plan

        plan = """\
## Task: Climate Change Report

- [ ] 1. Research current data
- [ ] 2. Draft introduction
- [ ] 3. Create visualizations
- [ ] 4. Compile report

Notes:
- Focus on actionable solutions"""

        title, steps = parse_plan(plan)
        assert title == "Climate Change Report"
        assert len(steps) == 4
        assert steps[0] == "Research current data"
        assert steps[3] == "Compile report"

    def test_parse_plan_numbered_fallback(self):
        from augmentum.modes.agentic.planner import parse_plan

        plan = """\
1. Research topic
2. Write draft
3. Review and finalize"""

        title, steps = parse_plan(plan)
        assert len(steps) == 3
        assert steps[0] == "Research topic"

    def test_parse_plan_empty(self):
        from augmentum.modes.agentic.planner import parse_plan

        title, steps = parse_plan("")
        assert title == ""
        assert steps == []

    def test_update_plan_step(self):
        from augmentum.modes.agentic.planner import update_plan_step

        plan = "- [ ] 1. Step one\n- [ ] 2. Step two\n- [ ] 3. Step three"
        updated = update_plan_step(plan, 0, "done")
        assert "- [x] 1. Step one (done)" in updated
        assert "- [ ] 2. Step two" in updated

    def test_update_plan_step_second(self):
        from augmentum.modes.agentic.planner import update_plan_step

        plan = "- [x] 1. Step one\n- [ ] 2. Step two\n- [ ] 3. Step three"
        updated = update_plan_step(plan, 1)
        assert "- [x] 2. Step two" in updated

    def test_mark_current_step(self):
        from augmentum.modes.agentic.planner import mark_current_step

        plan = "- [x] 1. Done\n- [ ] 2. Active\n- [ ] 3. Future"
        marked = mark_current_step(plan, 1)
        assert "← CURRENT" in marked
        assert marked.count("← CURRENT") == 1
        assert "2. Active ← CURRENT" in marked

    def test_mark_current_removes_old_markers(self):
        from augmentum.modes.agentic.planner import mark_current_step

        plan = "- [ ] 1. First ← CURRENT\n- [ ] 2. Second"
        marked = mark_current_step(plan, 1)
        assert "1. First ← CURRENT" not in marked
        assert "2. Second ← CURRENT" in marked

    def test_plan_to_context(self):
        from augmentum.modes.agentic.planner import plan_to_context

        plan = "- [ ] 1. Do something"
        ctx = plan_to_context(plan)
        assert "## Current Task Plan" in ctx
        assert "Do something" in ctx

    def test_plan_to_context_empty(self):
        from augmentum.modes.agentic.planner import plan_to_context

        assert plan_to_context("") == ""
        assert plan_to_context("   ") == ""


# ---------------------------------------------------------------------------
# Task State
# ---------------------------------------------------------------------------

class TestTaskState:
    def test_initial_state(self):
        from augmentum.modes.agentic.task_state import TaskState, TaskStatus

        task = TaskState()
        assert task.status == TaskStatus.PLANNING
        assert task.current_step == 0
        assert not task.is_complete
        assert task.progress_pct == 0.0

    def test_advance_step(self):
        from augmentum.modes.agentic.task_state import TaskState

        task = TaskState(total_steps=5)
        task.advance_step()
        assert task.current_step == 1
        # Progress is based on completed step outputs, not current_step
        assert task.progress_pct == 0.0
        task.record_step_output(0, "done")
        assert task.progress_pct == 20.0

    def test_record_step_output(self):
        from augmentum.modes.agentic.task_state import TaskState

        task = TaskState()
        task.record_step_output(0, "output 1")
        task.record_step_output(1, "output 2")
        assert task.step_outputs[0] == "output 1"
        assert task.step_outputs[1] == "output 2"

    def test_is_complete(self):
        from augmentum.modes.agentic.task_state import TaskState, TaskStatus

        task = TaskState(status=TaskStatus.COMPLETED)
        assert task.is_complete

        task2 = TaskState(status=TaskStatus.FAILED)
        assert task2.is_complete

        task3 = TaskState(status=TaskStatus.RUNNING)
        assert not task3.is_complete


# ---------------------------------------------------------------------------
# TaskStore (SQLite persistence)
# ---------------------------------------------------------------------------

class TestTaskStore:
    @pytest.fixture
    def mock_db(self):
        db = AsyncMock()
        cursor = AsyncMock()
        cursor.description = [
            ("id",), ("session_id",), ("flow_id",), ("status",),
            ("autonomy_level",), ("title",), ("plan_md",),
            ("current_step",), ("total_steps",), ("step_outputs",),
            ("tool_calls_made",), ("created_at",), ("updated_at",),
            ("completed_at",), ("error",),
        ]
        cursor.fetchone = AsyncMock(return_value=None)
        cursor.fetchall = AsyncMock(return_value=[])
        db.execute = AsyncMock(return_value=cursor)
        db.commit = AsyncMock()
        return db

    @pytest.mark.asyncio
    async def test_create_task(self, mock_db):
        from augmentum.modes.agentic.task_state import TaskState, TaskStore

        store = TaskStore(mock_db)
        task = TaskState(session_id="ses_abc", title="Test Task")
        result = await store.create(task)
        assert result.id  # ID should be auto-generated
        mock_db.execute.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_task(self, mock_db):
        from augmentum.modes.agentic.task_state import TaskState, TaskStatus, TaskStore

        store = TaskStore(mock_db)
        task = TaskState(
            id="task_123",
            status=TaskStatus.RUNNING,
            current_step=2,
        )
        await store.update(task)
        mock_db.execute.assert_called_once()
        mock_db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_returns_none_when_missing(self, mock_db):
        from augmentum.modes.agentic.task_state import TaskStore

        store = TaskStore(mock_db)
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_task(self, mock_db):
        from augmentum.modes.agentic.task_state import TaskState, TaskStatus, TaskStore

        cursor = mock_db.execute.return_value
        cursor.fetchone = AsyncMock(return_value=(
            "task_1", "ses_1", "flow_1", "running", 2, "My Task",
            "- [ ] step", 1, 3, '{"0": "output"}', 5,
            "2026-01-01", "2026-01-01", None, None,
        ))

        store = TaskStore(mock_db)
        task = await store.get("task_1")
        assert task is not None
        assert task.id == "task_1"
        assert task.status == TaskStatus.RUNNING
        assert task.title == "My Task"
        assert task.step_outputs == {0: "output"}

    @pytest.mark.asyncio
    async def test_get_incomplete_for_session(self, mock_db):
        from augmentum.modes.agentic.task_state import TaskStore

        cursor = mock_db.execute.return_value
        cursor.fetchone = AsyncMock(return_value=None)

        store = TaskStore(mock_db)
        result = await store.get_incomplete_for_session("ses_abc")
        assert result is None


# ---------------------------------------------------------------------------
# Variables — {plan} support
# ---------------------------------------------------------------------------

class TestPlanVariable:
    def test_plan_variable_resolved(self):
        from augmentum.reasoning.variables import StepContext, resolve_variables

        ctx = StepContext(query="test")
        ctx.plan = "- [ ] 1. Step one"
        result = resolve_variables("Plan: {plan}", ctx)
        assert "Step one" in result

    def test_plan_variable_empty(self):
        from augmentum.reasoning.variables import StepContext, resolve_variables

        ctx = StepContext(query="test")
        result = resolve_variables("Plan: {plan}", ctx)
        assert result == "Plan: "


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

class TestAgenticHandler:
    @pytest.fixture
    def mock_backend(self):
        backend = AsyncMock()
        backend.chat = AsyncMock(return_value=InternalChatResponse(
            message=Message(
                role="assistant",
                content=(
                    "## Task: Test Task\n\n"
                    "- [ ] 1. Research topic\n"
                    "- [ ] 2. Write content\n"
                    "- [ ] 3. Deliver result"
                ),
            ),
            model="test-model",
        ))

        async def fake_stream(req):
            yield InternalStreamChunk(content_delta="Step output here.", model="test-model")
            yield InternalStreamChunk(content_delta="", model="test-model", done=True)

        backend.chat_stream = fake_stream
        return backend

    @pytest.fixture
    def handler(self, mock_backend):
        from augmentum.modes.agentic.handler import AgenticHandler

        return AgenticHandler(
            backend=mock_backend,
            session_id="ses_test",
        )

    @pytest.mark.asyncio
    async def test_handle_stream_yields_chunks(self, handler):
        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="Create a report on AI")],
            stream=True,
        )

        chunks = []
        async for chunk in handler.handle_stream(request):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Should have agentic metadata
        meta_chunks = [c for c in chunks if c.augmentum and c.augmentum.get("mode") == "agentic"]
        assert len(meta_chunks) > 0

    @pytest.mark.asyncio
    async def test_handle_non_streaming(self, handler):
        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="Create a report")],
        )

        response = await handler.handle(request)
        assert response.message.role == "assistant"
        assert len(response.message.content) > 0

    @pytest.mark.asyncio
    async def test_ad_hoc_plan_and_execute(self, handler, mock_backend):
        """Without a flow store, handler uses ad-hoc plan+execute."""
        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="Build something")],
            stream=True,
        )

        chunks = []
        async for chunk in handler.handle_stream(request):
            chunks.append(chunk)

        # Should have planning and step execution phases
        statuses = set()
        for c in chunks:
            if c.augmentum:
                s = c.augmentum.get("task_status")
                if s:
                    statuses.add(s)
        assert "planning" in statuses or "plan_ready" in statuses

    @pytest.mark.asyncio
    async def test_agentic_meta_includes_progress(self, handler):
        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="Create a document about cats")],
            stream=True,
        )

        chunks = []
        async for chunk in handler.handle_stream(request):
            chunks.append(chunk)

        # Check for progress in metadata
        progress_chunks = [
            c for c in chunks
            if c.augmentum and "progress" in c.augmentum
        ]
        assert len(progress_chunks) > 0


# ---------------------------------------------------------------------------
# Handler with Task Store (checkpoints)
# ---------------------------------------------------------------------------

class TestAgenticCheckpoints:
    @pytest.fixture
    def mock_backend(self):
        backend = AsyncMock()
        backend.chat = AsyncMock(return_value=InternalChatResponse(
            message=Message(
                role="assistant",
                content=(
                    "## Task: Quick Task\n\n"
                    "- [ ] 1. Do thing\n"
                    "- [ ] 2. Finish"
                ),
            ),
            model="test-model",
        ))

        async def fake_stream(req):
            yield InternalStreamChunk(content_delta="Done.", model="test-model")
            yield InternalStreamChunk(content_delta="", model="test-model", done=True)

        backend.chat_stream = fake_stream
        return backend

    @pytest.fixture
    def mock_task_store(self):
        from augmentum.modes.agentic.task_state import TaskState, TaskStatus

        store = AsyncMock()
        store.create = AsyncMock(side_effect=lambda t: t)
        store.update = AsyncMock()
        store.get_incomplete_for_session = AsyncMock(return_value=None)
        return store

    @pytest.mark.asyncio
    async def test_creates_task_in_store(self, mock_backend, mock_task_store):
        from augmentum.modes.agentic.handler import AgenticHandler

        handler = AgenticHandler(
            backend=mock_backend,
            session_id="ses_test",
            task_store=mock_task_store,
        )

        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="Create something")],
            stream=True,
        )

        async for _ in handler.handle_stream(request):
            pass

        mock_task_store.create.assert_called_once()
        assert mock_task_store.update.call_count >= 1

    @pytest.mark.asyncio
    async def test_resumes_incomplete_task(self, mock_backend, mock_task_store):
        from augmentum.modes.agentic.handler import AgenticHandler
        from augmentum.modes.agentic.task_state import TaskState, TaskStatus

        # Set up an incomplete task
        incomplete = TaskState(
            id="task_resume",
            session_id="ses_test",
            status=TaskStatus.RUNNING,
            title="Resumable Task",
            plan_md="- [x] 1. Done\n- [ ] 2. Pending",
            current_step=0,
            total_steps=2,
        )
        mock_task_store.get_incomplete_for_session = AsyncMock(
            return_value=incomplete
        )

        handler = AgenticHandler(
            backend=mock_backend,
            session_id="ses_test",
            task_store=mock_task_store,
        )

        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="continue")],
            stream=True,
        )

        chunks = []
        async for chunk in handler.handle_stream(request):
            chunks.append(chunk)

        # Should have a resuming status
        statuses = set()
        for c in chunks:
            if c.augmentum:
                s = c.augmentum.get("task_status")
                if s:
                    statuses.add(s)
        assert "resuming" in statuses

    @pytest.mark.asyncio
    async def test_approval_pending_approve(self, mock_backend, mock_task_store):
        from augmentum.modes.agentic.handler import AgenticHandler
        from augmentum.modes.agentic.task_state import TaskState, TaskStatus

        pending = TaskState(
            id="task_approve",
            session_id="ses_test",
            status=TaskStatus.APPROVAL_PENDING,
            title="Needs Approval",
            plan_md="- [ ] 1. Create file",
            current_step=0,
            total_steps=1,
        )
        mock_task_store.get_incomplete_for_session = AsyncMock(
            return_value=pending
        )

        handler = AgenticHandler(
            backend=mock_backend,
            session_id="ses_test",
            task_store=mock_task_store,
        )

        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="approve")],
            stream=True,
        )

        chunks = []
        async for chunk in handler.handle_stream(request):
            chunks.append(chunk)

        # Task should have been set to RUNNING (resume)
        update_calls = mock_task_store.update.call_args_list
        assert len(update_calls) >= 1

    @pytest.mark.asyncio
    async def test_approval_pending_cancel(self, mock_backend, mock_task_store):
        from augmentum.modes.agentic.handler import AgenticHandler
        from augmentum.modes.agentic.task_state import TaskState, TaskStatus

        pending = TaskState(
            id="task_cancel",
            session_id="ses_test",
            status=TaskStatus.APPROVAL_PENDING,
            title="Cancel This",
            plan_md="- [ ] 1. Something",
            current_step=0,
            total_steps=1,
        )
        mock_task_store.get_incomplete_for_session = AsyncMock(
            return_value=pending
        )

        handler = AgenticHandler(
            backend=mock_backend,
            session_id="ses_test",
            task_store=mock_task_store,
        )

        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="cancel")],
            stream=True,
        )

        chunks = []
        async for chunk in handler.handle_stream(request):
            chunks.append(chunk)

        # Cancelled status should be in metadata, not chat content
        statuses = {
            c.augmentum.get("task_status")
            for c in chunks
            if c.augmentum
        }
        assert "cancelled" in statuses


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------

class TestHandlerFactory:
    def test_agentic_mode_creates_handler(self):
        from augmentum.proxy.handler_factory import get_handler_for_mode
        from augmentum.modes.agentic.handler import AgenticHandler

        backend = AsyncMock()
        app_state = MagicMock()
        app_state.tool_registry = None
        app_state.flow_store = None
        app_state.artifact_store = None
        app_state.task_store = None
        app_state.image_queue = None

        handler = get_handler_for_mode(
            Mode.AGENTIC, backend, "ses_test", app_state,
        )
        assert isinstance(handler, AgenticHandler)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_extract_query(self):
        from augmentum.modes.agentic.handler import _extract_query

        request = InternalChatRequest(
            model="m",
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="hello"),
                Message(role="assistant", content="hi"),
                Message(role="user", content="create a report"),
            ],
        )
        assert _extract_query(request) == "create a report"

    def test_extract_query_empty(self):
        from augmentum.modes.agentic.handler import _extract_query

        request = InternalChatRequest(
            model="m",
            messages=[Message(role="system", content="sys")],
        )
        assert _extract_query(request) == ""

    def test_is_new_task_request(self):
        from augmentum.modes.agentic.handler import _is_new_task_request

        assert _is_new_task_request("Create a detailed report on AI trends")
        assert _is_new_task_request("Generate a presentation about climate change")
        assert not _is_new_task_request("yes")
        assert not _is_new_task_request("continue")
        assert not _is_new_task_request("ok")

    def test_build_tools_section(self):
        from augmentum.modes.agentic.handler import _build_tools_section

        tool = MagicMock()
        tool.name = "create_document"
        tool.description = "Creates documents"
        tool.input_schema = {
            "properties": {"title": {"type": "string"}, "format": {"type": "string"}},
        }

        section = _build_tools_section([tool])
        assert "create_document" in section
        assert "title, format" in section
        assert "TOOL_CALL" in section

    def test_build_tools_section_empty(self):
        from augmentum.modes.agentic.handler import _build_tools_section

        assert _build_tools_section([]) == ""

    def test_build_conversation(self):
        from augmentum.modes.agentic.handler import _build_conversation

        request = InternalChatRequest(
            model="m",
            messages=[
                Message(role="user", content="first"),
                Message(role="assistant", content="response"),
                Message(role="user", content="second"),
            ],
        )
        conv = _build_conversation(request)
        assert "first" in conv
        assert "response" in conv
        # Last message excluded
        assert "second" not in conv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestAgenticConfig:
    def test_config_defaults(self):
        from augmentum.config import settings

        assert settings.agentic_enabled is True
        assert settings.agentic_max_steps == 20
        assert settings.agentic_default_autonomy == 2
        assert settings.agentic_checkpoint_enabled is True
