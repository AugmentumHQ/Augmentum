"""Tests for agentic_routes.py — task listing and retrieval."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def _mock_task_store():
    store = MagicMock()
    mock_task = MagicMock(
        id="task_1", session_id="sess_1", flow_id="flow_1",
        status=MagicMock(value="running"),
        autonomy_level=2, title="Research task",
        current_step=1, total_steps=3,
        tool_calls_made=5, error=None,
        plan_md="## Plan", step_outputs={0: "output0"},
    )
    store.list_for_session = AsyncMock(return_value=[mock_task])
    store.get = AsyncMock(return_value=mock_task)
    return store


class TestListTasks:
    def test_list_tasks_no_store(self, client):
        resp = client.get("/api/agentic/tasks?session_id=sess_1")
        assert resp.status_code == 200
        assert resp.json()["tasks"] == []

    def test_list_tasks_missing_session(self, app, client):
        app.state.task_store = _mock_task_store()
        resp = client.get("/api/agentic/tasks")
        assert resp.status_code == 400

    def test_list_tasks_success(self, app, client):
        app.state.task_store = _mock_task_store()
        resp = client.get("/api/agentic/tasks?session_id=sess_1")
        assert resp.status_code == 200
        tasks = resp.json()["tasks"]
        assert len(tasks) == 1
        assert tasks[0]["id"] == "task_1"
        assert tasks[0]["status"] == "running"


class TestGetTask:
    def test_get_task_no_store(self, client):
        resp = client.get("/api/agentic/tasks/task_1")
        assert resp.status_code == 503

    def test_get_task_not_found(self, app, client):
        store = _mock_task_store()
        store.get = AsyncMock(return_value=None)
        app.state.task_store = store
        resp = client.get("/api/agentic/tasks/nonexistent")
        assert resp.status_code == 404

    def test_get_task_success(self, app, client):
        app.state.task_store = _mock_task_store()
        resp = client.get("/api/agentic/tasks/task_1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "task_1"
        assert data["plan_md"] == "## Plan"
        assert "step_outputs" in data
