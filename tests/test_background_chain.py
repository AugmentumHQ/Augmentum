"""Tests for BackgroundChainManager — background chain execution, limits, notifications."""

from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from augmentum.tools.background_chain import BackgroundChainManager, BackgroundTask


class TestBackgroundTaskDefaults(unittest.TestCase):
    """Test BackgroundTask dataclass defaults."""

    def test_defaults(self):
        task = BackgroundTask(
            task_id="t1",
            flow_name="Research",
            flow_id="f1",
            session_id="s1",
            query="hello",
        )
        assert task.status == "running"
        assert task.completed_at is None
        assert task.result_summary == ""
        assert task.error == ""
        assert task.injected is False
        assert isinstance(task.created_at, float)


class TestBackgroundChainManagerLaunch(unittest.TestCase):
    """Test launch, limit enforcement, and task retrieval."""

    def _make_flow(self, name: str = "Test Flow", flow_id: str = "f1") -> dict:
        return {"name": name, "id": flow_id}

    @patch("augmentum.tools.background_chain.execute_chain", new_callable=AsyncMock)
    @patch("augmentum.tools.background_chain.flow_to_plan")
    @patch("augmentum.tools.background_chain.build_synthesis_prompt", return_value="synth")
    def test_launch_creates_task_and_returns_id(self, _synth, _plan, _exec):
        """launch() creates a task entry and returns a task_id string."""
        mgr = BackgroundChainManager()

        task_id = asyncio.run(
            mgr.launch(self._make_flow(), "test query", "sess-1")
        )

        assert isinstance(task_id, str)
        assert len(task_id) == 12
        task = mgr.get_task(task_id)
        assert task is not None
        assert task.flow_name == "Test Flow"
        assert task.session_id == "sess-1"
        assert task.query == "test query"

    @patch("augmentum.tools.background_chain.execute_chain", new_callable=AsyncMock)
    @patch("augmentum.tools.background_chain.flow_to_plan")
    @patch("augmentum.tools.background_chain.build_synthesis_prompt", return_value="synth")
    def test_per_session_limit_enforcement(self, _synth, _plan, _exec):
        """Exceeding per-session limit raises ValueError."""
        mgr = BackgroundChainManager(max_per_session=2, max_total=50)

        async def _fill():
            await mgr.launch(self._make_flow(), "q1", "sess-1")
            await mgr.launch(self._make_flow(), "q2", "sess-1")
            with self.assertRaises(ValueError) as ctx:
                await mgr.launch(self._make_flow(), "q3", "sess-1")
            assert "per session" in str(ctx.exception).lower()

        asyncio.run(_fill())

    @patch("augmentum.tools.background_chain.execute_chain", new_callable=AsyncMock)
    @patch("augmentum.tools.background_chain.flow_to_plan")
    @patch("augmentum.tools.background_chain.build_synthesis_prompt", return_value="synth")
    def test_total_limit_enforcement(self, _synth, _plan, _exec):
        """Exceeding total limit raises ValueError."""
        mgr = BackgroundChainManager(max_per_session=10, max_total=2)

        async def _fill():
            await mgr.launch(self._make_flow(), "q1", "sess-a")
            await mgr.launch(self._make_flow(), "q2", "sess-b")
            with self.assertRaises(ValueError) as ctx:
                await mgr.launch(self._make_flow(), "q3", "sess-c")
            assert "total" in str(ctx.exception).lower()

        asyncio.run(_fill())


class TestBackgroundChainManagerNotifications(unittest.TestCase):
    """Test subscribe/unsubscribe and notification dispatch."""

    def test_subscribe_returns_queue(self):
        mgr = BackgroundChainManager()
        queue = mgr.subscribe("sess-1")
        assert isinstance(queue, asyncio.Queue)

    def test_unsubscribe_removes_queue(self):
        mgr = BackgroundChainManager()
        queue = mgr.subscribe("sess-1")
        mgr.unsubscribe("sess-1", queue)
        # Internal queues list should be cleaned up
        assert "sess-1" not in mgr._notification_queues

    def test_unsubscribe_unknown_queue_no_error(self):
        mgr = BackgroundChainManager()
        random_queue: asyncio.Queue = asyncio.Queue()
        mgr.unsubscribe("no-session", random_queue)  # Should not raise

    def test_push_notification_to_all_queues(self):
        """_push_notification sends to all subscribed queues for a session."""
        mgr = BackgroundChainManager()
        q1 = mgr.subscribe("sess-1")
        q2 = mgr.subscribe("sess-1")

        event = {"type": "flow_complete", "task_id": "t1"}
        asyncio.run(mgr._push_notification("sess-1", event))

        assert q1.get_nowait() == event
        assert q2.get_nowait() == event

    def test_push_notification_ignores_other_sessions(self):
        mgr = BackgroundChainManager()
        q1 = mgr.subscribe("sess-1")
        q2 = mgr.subscribe("sess-2")

        event = {"type": "flow_complete", "task_id": "t1"}
        asyncio.run(mgr._push_notification("sess-1", event))

        assert not q2.empty() is False  # q2 should be empty
        assert q1.get_nowait() == event
        assert q2.empty()


class TestBackgroundChainManagerRetrieval(unittest.TestCase):
    """Test get_tasks, get_task, get_pending_results, mark_injected."""

    def _populate(self, mgr: BackgroundChainManager, session_id: str = "sess-1"):
        """Directly insert tasks for retrieval tests (bypass launch)."""
        t1 = BackgroundTask(
            task_id="t1", flow_name="Flow A", flow_id="fa",
            session_id=session_id, query="q1", status="completed",
            result_summary="Result A",
        )
        t2 = BackgroundTask(
            task_id="t2", flow_name="Flow B", flow_id="fb",
            session_id=session_id, query="q2", status="running",
        )
        t3 = BackgroundTask(
            task_id="t3", flow_name="Flow C", flow_id="fc",
            session_id=session_id, query="q3", status="completed",
            result_summary="Result C", injected=True,
        )
        mgr._tasks["t1"] = t1
        mgr._tasks["t2"] = t2
        mgr._tasks["t3"] = t3
        mgr._session_tasks[session_id] = ["t1", "t2", "t3"]

    def test_get_tasks_returns_all_session_tasks(self):
        mgr = BackgroundChainManager()
        self._populate(mgr)
        tasks = mgr.get_tasks("sess-1")
        assert len(tasks) == 3
        assert {t.task_id for t in tasks} == {"t1", "t2", "t3"}

    def test_get_tasks_empty_for_unknown_session(self):
        mgr = BackgroundChainManager()
        assert mgr.get_tasks("nonexistent") == []

    def test_get_task_by_id(self):
        mgr = BackgroundChainManager()
        self._populate(mgr)
        task = mgr.get_task("t2")
        assert task is not None
        assert task.flow_name == "Flow B"

    def test_get_task_returns_none_for_unknown(self):
        mgr = BackgroundChainManager()
        assert mgr.get_task("nope") is None

    def test_get_pending_results_filters_correctly(self):
        """Only completed + not-injected tasks are pending."""
        mgr = BackgroundChainManager()
        self._populate(mgr)
        pending = mgr.get_pending_results("sess-1")
        assert len(pending) == 1
        assert pending[0].task_id == "t1"

    def test_mark_injected_updates_flag(self):
        mgr = BackgroundChainManager()
        self._populate(mgr)
        assert mgr.get_task("t1").injected is False
        mgr.mark_injected("t1")
        assert mgr.get_task("t1").injected is True

    def test_mark_injected_unknown_id_no_error(self):
        mgr = BackgroundChainManager()
        mgr.mark_injected("nonexistent")  # Should not raise


class TestBackgroundChainManagerCleanup(unittest.TestCase):
    """Test cleanup_session cancellation and state removal."""

    def test_cleanup_removes_tasks_and_queues(self):
        mgr = BackgroundChainManager()
        # Insert tasks directly
        t1 = BackgroundTask(
            task_id="t1", flow_name="F", flow_id="f",
            session_id="sess-1", query="q",
        )
        mgr._tasks["t1"] = t1
        mgr._session_tasks["sess-1"] = ["t1"]
        mgr.subscribe("sess-1")

        # Create a mock asyncio.Task
        mock_atask = MagicMock()
        mock_atask.done.return_value = False
        mgr._async_tasks["t1"] = mock_atask

        mgr.cleanup_session("sess-1")

        assert mgr.get_task("t1") is None
        assert mgr.get_tasks("sess-1") == []
        assert "sess-1" not in mgr._notification_queues
        mock_atask.cancel.assert_called_once()

    def test_cleanup_skips_done_tasks(self):
        mgr = BackgroundChainManager()
        t1 = BackgroundTask(
            task_id="t1", flow_name="F", flow_id="f",
            session_id="sess-1", query="q", status="completed",
        )
        mgr._tasks["t1"] = t1
        mgr._session_tasks["sess-1"] = ["t1"]

        mock_atask = MagicMock()
        mock_atask.done.return_value = True
        mgr._async_tasks["t1"] = mock_atask

        mgr.cleanup_session("sess-1")

        mock_atask.cancel.assert_not_called()

    def test_cleanup_unknown_session_no_error(self):
        mgr = BackgroundChainManager()
        mgr.cleanup_session("nonexistent")  # Should not raise


class TestRunChain(unittest.TestCase):
    """Test _run_chain completion and failure paths."""

    @patch("augmentum.tools.background_chain.execute_chain", new_callable=AsyncMock)
    @patch("augmentum.tools.background_chain.flow_to_plan")
    @patch("augmentum.tools.background_chain.build_synthesis_prompt", return_value="synth prompt")
    def test_successful_completion(self, _synth, mock_plan, mock_exec):
        """Successful chain updates task to completed and pushes notification."""
        from augmentum.models.base import InternalChatRequest, Message

        mock_exec.return_value = {1: MagicMock(success=True, output="done")}
        mock_plan.return_value = MagicMock(steps=[])

        # Mock backend for synthesis call
        backend = MagicMock()
        synth_resp = MagicMock()
        synth_resp.message.content = "Final summary"
        backend.chat = AsyncMock(return_value=synth_resp)

        registry = MagicMock()
        ctx = InternalChatRequest(
            model="test", messages=[Message(role="user", content="q")], stream=False,
        )

        task = BackgroundTask(
            task_id="t1", flow_name="Flow", flow_id="f1",
            session_id="sess-1", query="test query",
        )
        flow = {"name": "Flow", "id": "f1", "steps": []}

        mgr = BackgroundChainManager()
        q = mgr.subscribe("sess-1")

        asyncio.run(mgr._run_chain(task, flow, backend, registry, ctx))

        assert task.status == "completed"
        assert task.completed_at is not None
        assert task.result_summary == "Final summary"
        assert task.error == ""

        # Notification pushed
        event = q.get_nowait()
        assert event["type"] == "flow_complete"
        assert event["task_id"] == "t1"
        assert event["status"] == "completed"

    def test_failure_updates_task_with_error(self):
        """When chain fails, task is marked failed and error notification pushed."""
        task = BackgroundTask(
            task_id="t1", flow_name="Flow", flow_id="f1",
            session_id="sess-1", query="q",
        )
        flow = {"name": "Flow", "id": "f1"}

        mgr = BackgroundChainManager()
        q = mgr.subscribe("sess-1")

        # Pass None backend/registry to trigger RuntimeError
        asyncio.run(mgr._run_chain(task, flow, None, None, None))

        assert task.status == "failed"
        assert task.completed_at is not None
        assert task.error != ""

        event = q.get_nowait()
        assert event["type"] == "flow_failed"
        assert event["task_id"] == "t1"
        assert event["status"] == "failed"
        assert "error" in event


if __name__ == "__main__":
    unittest.main()
