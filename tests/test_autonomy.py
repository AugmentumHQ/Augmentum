"""Tests for autonomy dial and approval mechanics (Phase C)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
)

# ---------------------------------------------------------------------------
# Autonomy level logic
# ---------------------------------------------------------------------------

class TestAutonomyLevels:
    def test_needs_plan_approval_level_1(self):
        from augmentum.modes.agentic.autonomy import needs_plan_approval
        assert needs_plan_approval(1) is True

    def test_needs_plan_approval_level_2(self):
        from augmentum.modes.agentic.autonomy import needs_plan_approval
        assert needs_plan_approval(2) is False

    def test_needs_plan_approval_level_3(self):
        from augmentum.modes.agentic.autonomy import needs_plan_approval
        assert needs_plan_approval(3) is False

    def test_needs_plan_approval_level_4(self):
        from augmentum.modes.agentic.autonomy import needs_plan_approval
        assert needs_plan_approval(4) is False


class TestStepApproval:
    def test_level_1_always_needs_approval(self):
        from augmentum.modes.agentic.autonomy import needs_step_approval
        assert needs_step_approval(1, "analyze") is True
        assert needs_step_approval(1, "search") is True
        assert needs_step_approval(1, "create") is True
        assert needs_step_approval(1, "deliver") is True

    def test_level_2_high_impact_needs_approval(self):
        from augmentum.modes.agentic.autonomy import needs_step_approval
        assert needs_step_approval(2, "create") is True
        assert needs_step_approval(2, "illustrate") is True
        # Deliver only presents results — not high-impact
        assert needs_step_approval(2, "deliver") is False

    def test_level_2_low_impact_no_approval(self):
        from augmentum.modes.agentic.autonomy import needs_step_approval
        assert needs_step_approval(2, "analyze") is False
        assert needs_step_approval(2, "search") is False
        assert needs_step_approval(2, "draft") is False
        assert needs_step_approval(2, "plan") is False
        assert needs_step_approval(2, "review") is False

    def test_level_2_many_tool_calls_needs_approval(self):
        from augmentum.modes.agentic.autonomy import needs_step_approval
        assert needs_step_approval(2, "analyze", tool_calls_in_step=3) is True
        assert needs_step_approval(2, "analyze", tool_calls_in_step=2) is False

    def test_level_3_never_needs_approval(self):
        from augmentum.modes.agentic.autonomy import needs_step_approval
        assert needs_step_approval(3, "create") is False
        assert needs_step_approval(3, "deliver") is False
        assert needs_step_approval(3, "illustrate") is False

    def test_level_4_never_needs_approval(self):
        from augmentum.modes.agentic.autonomy import needs_step_approval
        assert needs_step_approval(4, "create") is False
        assert needs_step_approval(4, "deliver") is False


# ---------------------------------------------------------------------------
# Approval chunk building
# ---------------------------------------------------------------------------

class TestApprovalChunks:
    def test_build_approval_chunk(self):
        from augmentum.modes.agentic.autonomy import build_approval_chunk
        from augmentum.modes.agentic.task_state import TaskState

        task = TaskState(id="t1", title="My Task", current_step=2, total_steps=5)
        chunk = build_approval_chunk("model", task, "Create Report", "create", "Making a PDF")

        assert "Approval needed" in chunk.content_delta
        assert "Create Report" in chunk.content_delta
        assert chunk.augmentum["task_status"] == "approval_pending"
        assert chunk.augmentum["approval_request"]["step_name"] == "Create Report"
        assert chunk.augmentum["approval_request"]["step_role"] == "create"

    def test_build_plan_approval_chunk(self):
        from augmentum.modes.agentic.autonomy import build_plan_approval_chunk
        from augmentum.modes.agentic.task_state import TaskState

        task = TaskState(
            id="t2", title="Research Report",
            plan_md="- [ ] 1. Research\n- [ ] 2. Write",
            total_steps=2,
        )
        chunk = build_plan_approval_chunk("model", task)

        assert "Plan ready for review" in chunk.content_delta
        assert "Research Report" in chunk.content_delta
        assert chunk.augmentum["task_status"] == "approval_pending"
        assert chunk.augmentum["approval_request"]["step_role"] == "plan"
        assert chunk.augmentum["plan_md"] == task.plan_md

    def test_build_inform_chunk(self):
        from augmentum.modes.agentic.autonomy import build_inform_chunk
        from augmentum.modes.agentic.task_state import TaskState

        task = TaskState(id="t3", title="Auto Task")
        chunk = build_inform_chunk("model", task, "Create PDF", "Generated 5-page report")

        assert "Create PDF" in chunk.content_delta
        assert chunk.augmentum["task_status"] == "informed"
        assert chunk.augmentum["informed_action"]["step_name"] == "Create PDF"


# ---------------------------------------------------------------------------
# ReasoningFlow autonomy_level field
# ---------------------------------------------------------------------------

class TestFlowAutonomyField:
    def test_default_autonomy_level(self):
        from augmentum.reasoning.models import ReasoningFlow
        flow = ReasoningFlow()
        assert flow.autonomy_level == 2

    def test_custom_autonomy_level(self):
        from augmentum.reasoning.models import ReasoningFlow
        flow = ReasoningFlow(autonomy_level=4)
        assert flow.autonomy_level == 4

    def test_create_request_has_autonomy(self):
        from augmentum.reasoning.models import FlowCreateRequest
        req = FlowCreateRequest(name="Test", autonomy_level=1)
        assert req.autonomy_level == 1

    def test_update_request_has_autonomy(self):
        from augmentum.reasoning.models import FlowUpdateRequest
        req = FlowUpdateRequest(autonomy_level=3)
        assert req.autonomy_level == 3


# ---------------------------------------------------------------------------
# Handler integration with autonomy
# ---------------------------------------------------------------------------

class TestHandlerAutonomy:
    @pytest.fixture
    def mock_backend(self):
        backend = AsyncMock()
        backend.chat = AsyncMock(return_value=InternalChatResponse(
            message=Message(
                role="assistant",
                content=(
                    "## Task: Quick Task\n\n"
                    "- [ ] 1. Research\n"
                    "- [ ] 2. Create output"
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
        store = AsyncMock()
        store.create = AsyncMock(side_effect=lambda t: t)
        store.update = AsyncMock()
        store.get_incomplete_for_session = AsyncMock(return_value=None)
        return store

    @pytest.mark.asyncio
    async def test_level_1_pauses_for_plan_approval(self, mock_backend, mock_task_store):
        """Autonomy level 1 should pause after generating plan."""
        from augmentum.config import settings
        from augmentum.modes.agentic.handler import AgenticHandler

        # Override config for this test
        original = settings.agentic_default_autonomy
        object.__setattr__(settings, "agentic_default_autonomy", 1)

        try:
            handler = AgenticHandler(
                backend=mock_backend,
                session_id="ses_test",
                task_store=mock_task_store,
            )

            request = InternalChatRequest(
                model="test-model",
                messages=[Message(role="user", content="Create a detailed report on AI")],
                stream=True,
            )

            chunks = []
            async for chunk in handler.handle_stream(request):
                chunks.append(chunk)

            # Should have approval_pending status
            statuses = set()
            for c in chunks:
                if c.augmentum and "task_status" in c.augmentum:
                    statuses.add(c.augmentum["task_status"])

            assert "approval_pending" in statuses

            # Should have an approval_request in metadata
            approval_chunks = [
                c for c in chunks
                if c.augmentum and "approval_request" in c.augmentum
            ]
            assert len(approval_chunks) > 0
            assert approval_chunks[0].augmentum["approval_request"]["step_role"] == "plan"
        finally:
            object.__setattr__(settings, "agentic_default_autonomy", original)

    @pytest.mark.asyncio
    async def test_level_4_no_plan_approval(self, mock_backend, mock_task_store):
        """Autonomy level 4 should execute without pausing."""
        from augmentum.config import settings
        from augmentum.modes.agentic.handler import AgenticHandler

        original = settings.agentic_default_autonomy
        object.__setattr__(settings, "agentic_default_autonomy", 4)

        try:
            handler = AgenticHandler(
                backend=mock_backend,
                session_id="ses_test",
                task_store=mock_task_store,
            )

            request = InternalChatRequest(
                model="test-model",
                messages=[Message(role="user", content="Create a detailed report on cats")],
                stream=True,
            )

            chunks = []
            async for chunk in handler.handle_stream(request):
                chunks.append(chunk)

            # Should NOT have approval_pending
            statuses = set()
            for c in chunks:
                if c.augmentum and "task_status" in c.augmentum:
                    statuses.add(c.augmentum["task_status"])

            assert "approval_pending" not in statuses
            # Should complete
            assert "completed" in statuses
        finally:
            object.__setattr__(settings, "agentic_default_autonomy", original)

    @pytest.mark.asyncio
    async def test_approval_then_resume(self, mock_backend, mock_task_store):
        """After approval, task should resume execution."""
        from augmentum.modes.agentic.handler import AgenticHandler
        from augmentum.modes.agentic.task_state import TaskState, TaskStatus

        # Simulate a task that was paused for approval
        paused_task = TaskState(
            id="task_paused",
            session_id="ses_test",
            status=TaskStatus.APPROVAL_PENDING,
            autonomy_level=1,
            title="Paused Task",
            plan_md="- [ ] 1. Step one\n- [ ] 2. Step two",
            current_step=0,
            total_steps=2,
        )
        mock_task_store.get_incomplete_for_session = AsyncMock(return_value=paused_task)

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

        # Task status should have been updated to RUNNING
        update_calls = mock_task_store.update.call_args_list
        updated_statuses = [
            call.args[0].status for call in update_calls if call.args
        ]
        assert TaskStatus.RUNNING in updated_statuses

    @pytest.mark.asyncio
    async def test_approval_modify(self, mock_backend, mock_task_store):
        """User sends modification text instead of approve/cancel."""
        from augmentum.modes.agentic.handler import AgenticHandler
        from augmentum.modes.agentic.task_state import TaskState, TaskStatus

        paused_task = TaskState(
            id="task_mod",
            session_id="ses_test",
            status=TaskStatus.APPROVAL_PENDING,
            autonomy_level=2,
            title="Modify Task",
            plan_md="- [ ] 1. Original step",
            current_step=0,
            total_steps=1,
        )
        mock_task_store.get_incomplete_for_session = AsyncMock(return_value=paused_task)

        handler = AgenticHandler(
            backend=mock_backend,
            session_id="ses_test",
            task_store=mock_task_store,
        )

        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="Also add a section about ethics")],
            stream=True,
        )

        chunks = []
        async for chunk in handler.handle_stream(request):
            chunks.append(chunk)

        # Plan should have been updated with user modification
        update_calls = mock_task_store.update.call_args_list
        modified_plans = [
            call.args[0].plan_md for call in update_calls
            if call.args and "User modification" in call.args[0].plan_md
        ]
        assert len(modified_plans) > 0
        assert "ethics" in modified_plans[0]


# ---------------------------------------------------------------------------
# Flow-based autonomy with step roles
# ---------------------------------------------------------------------------

class TestFlowStepAutonomy:
    @pytest.fixture
    def mock_backend(self):
        backend = AsyncMock()
        backend.chat = AsyncMock(return_value=InternalChatResponse(
            message=Message(role="assistant", content="Step output"),
            model="test-model",
        ))

        async def fake_stream(req):
            yield InternalStreamChunk(content_delta="Streaming.", model="test-model")
            yield InternalStreamChunk(content_delta="", model="test-model", done=True)

        backend.chat_stream = fake_stream
        return backend

    @pytest.fixture
    def mock_task_store(self):
        store = AsyncMock()
        store.create = AsyncMock(side_effect=lambda t: t)
        store.update = AsyncMock()
        store.get_incomplete_for_session = AsyncMock(return_value=None)
        return store

    @pytest.fixture
    def mock_flow_store(self):
        from augmentum.reasoning.models import FlowStep, ReasoningFlow

        flow = ReasoningFlow(
            id="flow_agentic_1",
            name="Test Agentic",
            trigger_domains=["agentic"],
            trigger_keywords=["presentation", "create"],
            autonomy_level=2,
            steps=[
                FlowStep(name="Research", role="search", stream_to_user=False),
                FlowStep(name="Draft", role="draft", stream_to_user=False),
                FlowStep(name="Create File", role="create", stream_to_user=True),
                FlowStep(name="Deliver", role="deliver", stream_to_user=True),
            ],
        )

        store = AsyncMock()
        store.list_all = AsyncMock(return_value=[flow])
        store.get = AsyncMock(return_value=flow)
        return store

    @pytest.mark.asyncio
    async def test_level_2_pauses_at_create_step(
        self, mock_backend, mock_task_store, mock_flow_store,
    ):
        """Level 2 should pause before 'create' role step."""
        from augmentum.modes.agentic.handler import AgenticHandler

        handler = AgenticHandler(
            backend=mock_backend,
            session_id="ses_test",
            task_store=mock_task_store,
            flow_store=mock_flow_store,
        )

        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="Create a detailed presentation about space")],
            stream=True,
        )

        chunks = []
        async for chunk in handler.handle_stream(request):
            chunks.append(chunk)

        # Should hit approval at the "Create File" step
        approval_chunks = [
            c for c in chunks
            if c.augmentum and c.augmentum.get("task_status") == "approval_pending"
        ]
        # Either plan approval (if level<=1) or step approval
        # At level 2, plan should not need approval, but create step should
        has_step_approval = any(
            c.augmentum.get("approval_request", {}).get("step_role") == "create"
            for c in approval_chunks
        )
        assert has_step_approval


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestAutonomyConfig:
    def test_default_autonomy_in_config(self):
        from augmentum.config import settings
        assert settings.agentic_default_autonomy == 2

    def test_autonomy_range(self):
        """Autonomy levels should be 1-4."""
        from augmentum.modes.agentic.autonomy import needs_plan_approval, needs_step_approval

        # Level 0 (invalid but should behave like 1)
        assert needs_plan_approval(0) is True
        # Level 5 (invalid but should behave like 4)
        assert needs_plan_approval(5) is False
        assert needs_step_approval(5, "create") is False
