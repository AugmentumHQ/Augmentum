"""Tests for agentic mode safety limits and polish (Phase F)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
)


# ---------------------------------------------------------------------------
# Step timeout tests
# ---------------------------------------------------------------------------


class TestStepTimeout:
    @pytest.fixture
    def slow_backend(self):
        """Backend that takes too long to respond."""
        backend = AsyncMock()

        async def slow_chat(req):
            await asyncio.sleep(10)  # Deliberately slow
            return InternalChatResponse(
                message=Message(role="assistant", content="slow response"),
                model="test-model",
            )

        backend.chat = slow_chat

        async def fake_stream(req):
            yield InternalStreamChunk(content_delta="stream", model="test-model")
            yield InternalStreamChunk(content_delta="", model="test-model", done=True)

        backend.chat_stream = fake_stream
        return backend

    @pytest.fixture
    def fast_backend(self):
        backend = AsyncMock()
        backend.chat = AsyncMock(return_value=InternalChatResponse(
            message=Message(
                role="assistant",
                content="## Task: Quick\n\n- [ ] 1. Step one\n- [ ] 2. Step two",
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
    async def test_step_timeout_triggers_failure(self, slow_backend, mock_task_store):
        """Step should fail if it exceeds the configured timeout."""
        from augmentum.config import settings
        from augmentum.modes.agentic.handler import AgenticHandler
        from augmentum.reasoning.models import FlowStep, ReasoningFlow

        # Set a very short timeout for testing
        original_timeout = settings.agentic_step_timeout
        object.__setattr__(settings, "agentic_step_timeout", 0.1)

        try:
            flow = ReasoningFlow(
                id="flow_timeout",
                name="Timeout Test",
                trigger_domains=["agentic"],
                trigger_keywords=["slow"],
                steps=[
                    FlowStep(name="Slow Step", role="analyze", stream_to_user=True),
                ],
            )

            flow_store = AsyncMock()
            flow_store.list_all = AsyncMock(return_value=[flow])
            flow_store.get = AsyncMock(return_value=flow)

            handler = AgenticHandler(
                backend=slow_backend,
                session_id="ses_test",
                task_store=mock_task_store,
                flow_store=flow_store,
            )

            request = InternalChatRequest(
                model="test-model",
                messages=[Message(role="user", content="Do something slow")],
                stream=True,
            )

            chunks = []
            async for chunk in handler.handle_stream(request):
                chunks.append(chunk)

            # Should have a failure status
            statuses = {
                c.augmentum["task_status"]
                for c in chunks
                if c.augmentum and "task_status" in c.augmentum
            }
            assert "failed" in statuses

            # Error details should be in task state, not in chat content
            # (internal errors are no longer leaked to the user)
        finally:
            object.__setattr__(settings, "agentic_step_timeout", original_timeout)


# ---------------------------------------------------------------------------
# Max steps safety limit
# ---------------------------------------------------------------------------


class TestMaxStepsSafety:
    @pytest.fixture
    def mock_backend(self):
        backend = AsyncMock()
        backend.chat = AsyncMock(return_value=InternalChatResponse(
            message=Message(
                role="assistant",
                content="## Task: Many Steps\n\n" + "\n".join(
                    f"- [ ] {i+1}. Step {i+1}" for i in range(25)
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

    def test_max_steps_config_default(self):
        from augmentum.config import settings

        assert settings.agentic_max_steps == 20

    @pytest.mark.asyncio
    async def test_max_steps_enforced_in_flow(self, mock_backend, mock_task_store):
        """Flow with more steps than max should be stopped."""
        from augmentum.config import settings
        from augmentum.modes.agentic.handler import AgenticHandler
        from augmentum.reasoning.models import FlowStep, ReasoningFlow

        original = settings.agentic_max_steps
        object.__setattr__(settings, "agentic_max_steps", 2)

        try:
            # Create a flow with 5 steps — should stop at 2
            flow = ReasoningFlow(
                id="flow_many",
                name="Many Steps",
                trigger_domains=["agentic"],
                trigger_keywords=["many"],
                steps=[
                    FlowStep(name=f"Step {i}", role="analyze", stream_to_user=False)
                    for i in range(5)
                ],
            )

            flow_store = AsyncMock()
            flow_store.list_all = AsyncMock(return_value=[flow])
            flow_store.get = AsyncMock(return_value=flow)

            handler = AgenticHandler(
                backend=mock_backend,
                session_id="ses_test",
                task_store=mock_task_store,
                flow_store=flow_store,
            )

            request = InternalChatRequest(
                model="test-model",
                messages=[Message(role="user", content="Do many things")],
                stream=True,
            )

            chunks = []
            async for chunk in handler.handle_stream(request):
                chunks.append(chunk)

            statuses = {
                c.augmentum["task_status"]
                for c in chunks
                if c.augmentum and "task_status" in c.augmentum
            }
            assert "failed" in statuses

            # Error details in task state, not leaked to chat content
        finally:
            object.__setattr__(settings, "agentic_max_steps", original)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestAgenticConfig:
    def test_all_agentic_config_fields_exist(self):
        from augmentum.config import settings

        assert hasattr(settings, "agentic_enabled")
        assert hasattr(settings, "agentic_max_steps")
        assert hasattr(settings, "agentic_default_autonomy")
        assert hasattr(settings, "agentic_artifact_dir")
        assert hasattr(settings, "agentic_max_artifact_size_mb")
        assert hasattr(settings, "agentic_checkpoint_enabled")
        assert hasattr(settings, "agentic_step_timeout")

    def test_agentic_config_defaults(self):
        from augmentum.config import settings

        assert settings.agentic_enabled is True
        assert settings.agentic_max_steps == 20
        assert settings.agentic_default_autonomy == 2
        assert settings.agentic_step_timeout == 300.0
        assert settings.agentic_max_artifact_size_mb == 50
        assert settings.agentic_checkpoint_enabled is True

    def test_autonomy_default_in_range(self):
        from augmentum.config import settings

        assert 1 <= settings.agentic_default_autonomy <= 4

    def test_step_timeout_positive(self):
        from augmentum.config import settings

        assert settings.agentic_step_timeout > 0


# ---------------------------------------------------------------------------
# Tool call limit in step
# ---------------------------------------------------------------------------


class TestToolCallLimit:
    def test_tool_calls_tracked_on_task(self):
        from augmentum.modes.agentic.task_state import TaskState

        task = TaskState(id="t1")
        assert task.tool_calls_made == 0
        task.tool_calls_made += 1
        assert task.tool_calls_made == 1

    def test_step_output_recorded(self):
        from augmentum.modes.agentic.task_state import TaskState

        task = TaskState(id="t1")
        task.record_step_output(0, "Step 0 output")
        task.record_step_output(1, "Step 1 output")
        assert task.step_outputs[0] == "Step 0 output"
        assert task.step_outputs[1] == "Step 1 output"


# ---------------------------------------------------------------------------
# Agentic handler robustness
# ---------------------------------------------------------------------------


class TestHandlerRobustness:
    @pytest.fixture
    def mock_backend(self):
        backend = AsyncMock()
        backend.chat = AsyncMock(return_value=InternalChatResponse(
            message=Message(
                role="assistant",
                content="## Task: Test\n\n- [ ] 1. Do something",
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
    async def test_handler_without_flow_store(self, mock_backend, mock_task_store):
        """Handler should fall back to ad-hoc mode without a flow store."""
        from augmentum.modes.agentic.handler import AgenticHandler

        handler = AgenticHandler(
            backend=mock_backend,
            session_id="ses_test",
            task_store=mock_task_store,
        )

        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="Create a test report")],
            stream=True,
        )

        chunks = []
        async for chunk in handler.handle_stream(request):
            chunks.append(chunk)

        # Should complete without error
        statuses = {
            c.augmentum.get("task_status")
            for c in chunks
            if c.augmentum
        }
        assert "completed" in statuses

    @pytest.mark.asyncio
    async def test_handler_without_task_store(self, mock_backend):
        """Handler should work without task persistence."""
        from augmentum.modes.agentic.handler import AgenticHandler

        handler = AgenticHandler(
            backend=mock_backend,
            session_id="ses_test",
        )

        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="Create a test report")],
            stream=True,
        )

        chunks = []
        async for chunk in handler.handle_stream(request):
            chunks.append(chunk)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_empty_query_handled(self, mock_backend, mock_task_store):
        """Empty query should not crash the handler."""
        from augmentum.modes.agentic.handler import AgenticHandler

        handler = AgenticHandler(
            backend=mock_backend,
            session_id="ses_test",
            task_store=mock_task_store,
        )

        request = InternalChatRequest(
            model="test-model",
            messages=[Message(role="user", content="")],
            stream=True,
        )

        chunks = []
        async for chunk in handler.handle_stream(request):
            chunks.append(chunk)

        # Should handle gracefully
        assert isinstance(chunks, list)
