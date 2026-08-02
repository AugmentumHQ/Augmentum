"""Tests for flow_routes.py — custom flow CRUD, execution, import/export."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def _mock_flow_store():
    """Create a mock custom flow store."""
    store = MagicMock()
    sample_flow = {
        "id": "flow_1",
        "name": "Search and Summarize",
        "description": "Search then summarize",
        "trigger_pattern": "search and summarize",
        "steps": [
            {"id": "s1", "tool": "web_search", "input": {"query": "{{query}}"}},
            {"id": "s2", "tool": "summarize", "input": {"text": "{{s1.output}}"}},
        ],
    }
    store.list_flows = AsyncMock(return_value=[sample_flow])
    store.get_flow = AsyncMock(return_value=sample_flow)
    store.create_flow = AsyncMock(return_value=sample_flow)
    store.update_flow = AsyncMock(return_value=sample_flow)
    store.delete_flow = AsyncMock(return_value=True)
    store.match_query = AsyncMock(return_value=None)
    store.export_all = AsyncMock(return_value=[sample_flow])
    store.import_flows = AsyncMock(return_value=1)
    return store


class TestFlowList:
    def test_list_flows_no_store(self, client):
        resp = client.get("/api/flows")
        assert resp.status_code == 200
        assert resp.json()["flows"] == []

    def test_list_flows_success(self, app, client):
        app.state.custom_flow_store = _mock_flow_store()
        resp = client.get("/api/flows")
        assert resp.status_code == 200
        assert len(resp.json()["flows"]) == 1


class TestFlowCRUD:
    def test_create_flow_no_store(self, client):
        resp = client.post("/api/flows", json={"name": "Test"})
        assert resp.status_code == 503

    def test_create_flow_no_name(self, app, client):
        app.state.custom_flow_store = _mock_flow_store()
        resp = client.post("/api/flows", json={"name": "", "steps": []})
        assert resp.status_code == 400

    def test_create_flow_no_steps(self, app, client):
        app.state.custom_flow_store = _mock_flow_store()
        resp = client.post("/api/flows", json={"name": "Test", "steps": []})
        assert resp.status_code == 400

    def test_create_flow_bad_step(self, app, client):
        app.state.custom_flow_store = _mock_flow_store()
        resp = client.post(
            "/api/flows",
            json={"name": "Test", "steps": [{"missing_id": True}]},
        )
        assert resp.status_code == 400
        assert "Step 0" in resp.json()["error"]

    def test_create_flow_success(self, app, client):
        app.state.custom_flow_store = _mock_flow_store()
        resp = client.post(
            "/api/flows",
            json={
                "name": "My Flow",
                "steps": [{"id": "s1", "tool": "web_search"}],
            },
        )
        assert resp.status_code == 201
        assert "flow" in resp.json()

    def test_get_flow_not_found(self, app, client):
        store = _mock_flow_store()
        store.get_flow = AsyncMock(return_value=None)
        app.state.custom_flow_store = store
        resp = client.get("/api/flows/nonexistent")
        assert resp.status_code == 404

    def test_get_flow_success(self, app, client):
        app.state.custom_flow_store = _mock_flow_store()
        resp = client.get("/api/flows/flow_1")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Search and Summarize"

    def test_update_flow_not_found(self, app, client):
        store = _mock_flow_store()
        store.update_flow = AsyncMock(return_value=None)
        app.state.custom_flow_store = store
        resp = client.put("/api/flows/nonexistent", json={"name": "Updated"})
        assert resp.status_code == 404

    def test_delete_flow_not_found(self, app, client):
        store = _mock_flow_store()
        store.delete_flow = AsyncMock(return_value=False)
        app.state.custom_flow_store = store
        resp = client.delete("/api/flows/nonexistent")
        assert resp.status_code == 404

    def test_delete_flow_success(self, app, client):
        app.state.custom_flow_store = _mock_flow_store()
        resp = client.delete("/api/flows/flow_1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True


class TestFlowMatch:
    def test_match_no_store(self, client):
        resp = client.get("/api/flows/match?q=search")
        assert resp.status_code == 200
        assert resp.json()["match"] is None


class TestFlowExportImport:
    def test_export_no_store(self, client):
        resp = client.get("/api/flows/export")
        assert resp.status_code == 200
        assert resp.json()["flows"] == []

    def test_export_success(self, app, client):
        app.state.custom_flow_store = _mock_flow_store()
        resp = client.get("/api/flows/export")
        assert resp.status_code == 200
        assert len(resp.json()["flows"]) == 1

    def test_import_no_store(self, client):
        resp = client.post("/api/flows/import", json={"flows": []})
        assert resp.status_code == 503

    def test_import_success(self, app, client):
        app.state.custom_flow_store = _mock_flow_store()
        resp = client.post("/api/flows/import", json={"flows": [{}]})
        assert resp.status_code == 200
        assert resp.json()["imported"] == 1
