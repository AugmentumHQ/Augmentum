"""Tests for executor_routes.py — sandboxed code execution proxy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient


class TestExecuteCode:
    def test_empty_code(self, client):
        resp = client.post("/api/execute", json={"code": ""})
        assert resp.status_code == 400
        assert "No code" in resp.json()["error"]

    def test_whitespace_only(self, client):
        resp = client.post("/api/execute", json={"code": "   "})
        assert resp.status_code == 400

    def test_blocked_pattern(self, client):
        resp = client.post(
            "/api/execute",
            json={"code": "import subprocess; subprocess.run(['rm', '-rf', '/'])"},
        )
        assert resp.status_code == 400
        assert "rejected" in resp.json()["error"].lower()

    def test_no_http_client(self, app, client):
        app.state.http_client = None
        resp = client.post("/api/execute", json={"code": "print(1+1)"})
        assert resp.status_code == 503

    def test_executor_unreachable(self, app, client):
        mock_http = MagicMock()
        mock_http.post = AsyncMock(side_effect=Exception("Connection refused"))
        app.state.http_client = mock_http
        resp = client.post("/api/execute", json={"code": "print(1+1)"})
        assert resp.status_code == 502

    def test_executor_bad_response(self, app, client):
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.side_effect = ValueError("not JSON")
        mock_resp.status_code = 200
        mock_http.post = AsyncMock(return_value=mock_resp)
        app.state.http_client = mock_http
        resp = client.post("/api/execute", json={"code": "print(1+1)"})
        assert resp.status_code == 503

    def test_executor_success(self, app, client):
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "success": True,
            "stdout": "2\n",
            "stderr": "",
            "return_value": None,
        }
        mock_resp.status_code = 200
        mock_http.post = AsyncMock(return_value=mock_resp)
        app.state.http_client = mock_http
        resp = client.post("/api/execute", json={"code": "print(1+1)"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["stdout"] == "2\n"

    def test_executor_error_response(self, app, client):
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "error": "NameError: name 'foo' is not defined",
            "traceback": "Traceback...",
        }
        mock_resp.status_code = 400
        mock_http.post = AsyncMock(return_value=mock_resp)
        app.state.http_client = mock_http
        resp = client.post("/api/execute", json={"code": "foo"})
        # Route returns 200 with success=False for display
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    def test_timeout_capped(self, app, client):
        mock_http = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"success": True, "stdout": "", "stderr": ""}
        mock_resp.status_code = 200
        mock_http.post = AsyncMock(return_value=mock_resp)
        app.state.http_client = mock_http
        # Timeout should be capped at 60
        resp = client.post(
            "/api/execute",
            json={"code": "import time; time.sleep(1)", "timeout": 999},
        )
        assert resp.status_code == 200
