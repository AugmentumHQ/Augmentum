"""Tests for reasoning_routes.py — reasoning flow CRUD, templates, clone."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient


def _mock_flow_store():
    store = MagicMock()
    mock_flow = MagicMock()
    mock_flow.model_dump.return_value = {
        "id": "rf_1", "name": "Deep Research", "description": "Multi-step research",
        "steps": [], "is_builtin": False, "is_default": False,
    }
    store.list_flows = AsyncMock(return_value=[(mock_flow, 3)])
    store.get_flow = AsyncMock(return_value=mock_flow)
    store.create_flow = AsyncMock(return_value=mock_flow)
    store.update_flow = AsyncMock(return_value=mock_flow)
    store.delete_flow = AsyncMock(return_value=True)
    store.clone_flow = AsyncMock(return_value=mock_flow)
    store.set_default = AsyncMock(return_value=True)
    store.export_flow = AsyncMock(return_value={"id": "rf_1", "name": "Deep Research"})
    store.import_flow = AsyncMock(return_value=mock_flow)
    return store


class TestReasoningFlowList:
    def test_list_no_store(self, client):
        resp = client.get("/api/reasoning/flows")
        assert resp.status_code == 503

    def test_list_success(self, app, client):
        app.state.flow_store = _mock_flow_store()
        resp = client.get("/api/reasoning/flows")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["step_count"] == 3


class TestReasoningFlowGet:
    def test_get_not_found(self, app, client):
        store = _mock_flow_store()
        store.get_flow = AsyncMock(return_value=None)
        app.state.flow_store = store
        resp = client.get("/api/reasoning/flows/nonexistent")
        assert resp.status_code == 404

    def test_get_success(self, app, client):
        app.state.flow_store = _mock_flow_store()
        resp = client.get("/api/reasoning/flows/rf_1")
        assert resp.status_code == 200
        assert resp.json()["name"] == "Deep Research"


class TestReasoningFlowCRUD:
    def test_create_flow(self, app, client):
        app.state.flow_store = _mock_flow_store()
        resp = client.post(
            "/api/reasoning/flows",
            json={"name": "New Flow", "steps": []},
        )
        assert resp.status_code == 201

    def test_update_flow_no_updates(self, app, client):
        app.state.flow_store = _mock_flow_store()
        resp = client.put("/api/reasoning/flows/rf_1", json={})
        assert resp.status_code == 400

    def test_update_flow_not_found(self, app, client):
        store = _mock_flow_store()
        store.update_flow = AsyncMock(return_value=None)
        app.state.flow_store = store
        resp = client.put(
            "/api/reasoning/flows/nonexistent",
            json={"name": "Updated"},
        )
        assert resp.status_code == 404

    def test_delete_flow_not_found(self, app, client):
        store = _mock_flow_store()
        store.delete_flow = AsyncMock(return_value=False)
        app.state.flow_store = store
        resp = client.delete("/api/reasoning/flows/nonexistent")
        assert resp.status_code == 404

    def test_delete_flow_success(self, app, client):
        app.state.flow_store = _mock_flow_store()
        resp = client.delete("/api/reasoning/flows/rf_1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True


class TestReasoningFlowClone:
    def test_clone_not_found(self, app, client):
        store = _mock_flow_store()
        store.clone_flow = AsyncMock(return_value=None)
        app.state.flow_store = store
        resp = client.post("/api/reasoning/flows/nonexistent/clone")
        assert resp.status_code == 404

    def test_clone_success(self, app, client):
        app.state.flow_store = _mock_flow_store()
        resp = client.post("/api/reasoning/flows/rf_1/clone")
        assert resp.status_code == 201


class TestReasoningFlowDefault:
    def test_set_default_not_found(self, app, client):
        store = _mock_flow_store()
        store.set_default = AsyncMock(return_value=False)
        app.state.flow_store = store
        resp = client.put("/api/reasoning/flows/nonexistent/default")
        assert resp.status_code == 404

    def test_set_default_success(self, app, client):
        app.state.flow_store = _mock_flow_store()
        resp = client.put("/api/reasoning/flows/rf_1/default")
        assert resp.status_code == 200
        assert resp.json()["default"] is True


class TestReasoningExportImport:
    def test_export_not_found(self, app, client):
        store = _mock_flow_store()
        store.export_flow = AsyncMock(return_value=None)
        app.state.flow_store = store
        resp = client.get("/api/reasoning/flows/nonexistent/export")
        assert resp.status_code == 404

    def test_export_success(self, app, client):
        app.state.flow_store = _mock_flow_store()
        resp = client.get("/api/reasoning/flows/rf_1/export")
        assert resp.status_code == 200

    def test_import_success(self, app, client):
        app.state.flow_store = _mock_flow_store()
        resp = client.post(
            "/api/reasoning/flows/import",
            json={"name": "Imported", "steps": []},
        )
        assert resp.status_code == 201


class TestTemplates:
    def test_list_templates(self, client):
        resp = client.get("/api/reasoning/templates")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
