"""Tests for notification_routes.py — SSE notifications and background tasks.

NOTE: The notification_routes router is NOT mounted in server.py (intentionally
removed — SSE endpoints are not yet consumed by the frontend).  These tests
manually mount the router to verify its logic in isolation.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

from augmentum.proxy.notification_routes import router as notification_router


@pytest.fixture
def notification_client(app):
    """Client with notification router manually included."""
    app.include_router(notification_router)
    tc = TestClient(app)
    tc.headers.update({"Authorization": "Bearer test-token"})
    return tc


# ---------------------------------------------------------------------------
# GET /api/notifications/{session_id} — SSE stream
# ---------------------------------------------------------------------------


class TestNotificationStream:
    def test_no_manager_returns_503(self, notification_client):
        """Without a background_chain_manager, should return 503."""
        resp = notification_client.get("/api/notifications/test-session")
        assert resp.status_code == 503

    def test_stream_content_type(self, notification_client):
        """When manager is available, returns event-stream."""
        mock_queue = asyncio.Queue()
        # Put None to terminate the stream immediately
        mock_queue.put_nowait(None)

        manager = MagicMock()
        manager.subscribe = MagicMock(return_value=mock_queue)
        manager.unsubscribe = MagicMock()
        notification_client.app.state.background_chain_manager = manager

        resp = notification_client.get("/api/notifications/test-session")
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        del notification_client.app.state.background_chain_manager


# ---------------------------------------------------------------------------
# GET /api/background-tasks/{session_id}
# ---------------------------------------------------------------------------


class TestListBackgroundTasks:
    def test_no_manager_returns_empty(self, notification_client):
        resp = notification_client.get("/api/background-tasks/test-session")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        assert data["tasks"] == []

    def test_with_tasks(self, notification_client):
        task = MagicMock()
        task.task_id = "task-1"
        task.flow_name = "search"
        task.status = "completed"
        task.query = "test query"
        task.result_summary = "Found results"
        task.error = None
        task.injected = False

        manager = MagicMock()
        manager.get_tasks = MagicMock(return_value=[task])
        notification_client.app.state.background_chain_manager = manager

        resp = notification_client.get("/api/background-tasks/test-session")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["task_id"] == "task-1"
        assert data["tasks"][0]["status"] == "completed"

        del notification_client.app.state.background_chain_manager


# ---------------------------------------------------------------------------
# GET /api/background-tasks/{session_id}/{task_id}
# ---------------------------------------------------------------------------


class TestGetBackgroundTask:
    def test_no_manager_returns_503(self, notification_client):
        resp = notification_client.get("/api/background-tasks/sess/task-1")
        assert resp.status_code == 503

    def test_task_not_found_returns_404(self, notification_client):
        manager = MagicMock()
        manager.get_task = MagicMock(return_value=None)
        notification_client.app.state.background_chain_manager = manager

        resp = notification_client.get("/api/background-tasks/sess/nonexistent")
        assert resp.status_code == 404

        del notification_client.app.state.background_chain_manager

    def test_task_session_mismatch_returns_404(self, notification_client):
        task = MagicMock()
        task.session_id = "other-session"
        manager = MagicMock()
        manager.get_task = MagicMock(return_value=task)
        notification_client.app.state.background_chain_manager = manager

        resp = notification_client.get("/api/background-tasks/my-session/task-1")
        assert resp.status_code == 404

        del notification_client.app.state.background_chain_manager

    def test_task_found_returns_details(self, notification_client):
        task = MagicMock()
        task.task_id = "task-1"
        task.session_id = "sess"
        task.flow_name = "search"
        task.flow_id = "flow-1"
        task.status = "completed"
        task.query = "test"
        task.result_summary = "Results here"
        task.error = None
        task.injected = True

        manager = MagicMock()
        manager.get_task = MagicMock(return_value=task)
        notification_client.app.state.background_chain_manager = manager

        resp = notification_client.get("/api/background-tasks/sess/task-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"] == "task-1"
        assert data["injected"] is True

        del notification_client.app.state.background_chain_manager
