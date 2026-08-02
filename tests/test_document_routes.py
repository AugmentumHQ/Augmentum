"""Tests for document_routes.py — upload, list, delete, search, session bindings."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def _mock_doc_store():
    store = MagicMock()
    store.list_documents = AsyncMock(return_value=[
        {"id": "doc_1", "filename": "test.pdf", "chunk_count": 10},
    ])
    store.get_document = AsyncMock(return_value={
        "id": "doc_1", "filename": "test.pdf", "chunk_count": 10,
    })
    store.delete_document = AsyncMock(return_value=True)
    store.ingest = AsyncMock(return_value={"id": "doc_new", "filename": "test.txt", "chunks": 5})
    store.search = AsyncMock(return_value=[{"content": "match", "score": 0.9}])
    return store


class TestDocumentList:
    def test_list_no_store(self, client):
        resp = client.get("/api/documents")
        assert resp.status_code == 503

    def test_list_success(self, app, client):
        app.state.document_store = _mock_doc_store()
        resp = client.get("/api/documents")
        assert resp.status_code == 200
        assert len(resp.json()["documents"]) == 1


class TestDocumentUpload:
    def test_upload_no_store(self, client):
        resp = client.post(
            "/api/documents",
            files={"file": ("test.txt", b"hello world", "text/plain")},
        )
        assert resp.status_code == 503

    def test_upload_unsupported_type(self, app, client):
        app.state.document_store = _mock_doc_store()
        resp = client.post(
            "/api/documents",
            files={"file": ("test.exe", b"MZ\x90", "application/x-msdownload")},
        )
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["error"]

    def test_upload_empty_file(self, app, client):
        app.state.document_store = _mock_doc_store()
        resp = client.post(
            "/api/documents",
            files={"file": ("empty.txt", b"", "text/plain")},
        )
        assert resp.status_code == 400
        assert "Empty" in resp.json()["error"]

    def test_upload_success(self, app, client):
        app.state.document_store = _mock_doc_store()
        resp = client.post(
            "/api/documents",
            files={"file": ("test.txt", b"hello world content", "text/plain")},
        )
        assert resp.status_code == 201
        assert resp.json()["id"] == "doc_new"


class TestDocumentGet:
    def test_get_not_found(self, app, client):
        store = _mock_doc_store()
        store.get_document = AsyncMock(return_value=None)
        app.state.document_store = store
        resp = client.get("/api/documents/nonexistent")
        assert resp.status_code == 404

    def test_get_success(self, app, client):
        app.state.document_store = _mock_doc_store()
        resp = client.get("/api/documents/doc_1")
        assert resp.status_code == 200
        assert resp.json()["filename"] == "test.pdf"


class TestDocumentDelete:
    def test_delete_not_found(self, app, client):
        store = _mock_doc_store()
        store.delete_document = AsyncMock(return_value=False)
        app.state.document_store = store
        resp = client.delete("/api/documents/nonexistent")
        assert resp.status_code == 404

    def test_delete_success(self, app, client):
        app.state.document_store = _mock_doc_store()
        resp = client.delete("/api/documents/doc_1")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True


class TestDocumentSearch:
    def test_search_no_query(self, app, client):
        app.state.document_store = _mock_doc_store()
        resp = client.post("/api/documents/search", json={"query": ""})
        assert resp.status_code == 400

    def test_search_success(self, app, client):
        app.state.document_store = _mock_doc_store()
        resp = client.post("/api/documents/search", json={"query": "test"})
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 1


class TestSessionBindings:
    def test_get_session_documents_no_db(self, client):
        resp = client.get("/api/documents/session/sess_1")
        assert resp.status_code == 200
        assert resp.json()["bindings"] == []

    def test_bind_document_missing_id(self, sqlite_client):
        resp = sqlite_client.post(
            "/api/documents/session/sess_1",
            json={"document_id": ""},
        )
        assert resp.status_code == 400

    def test_bind_document_invalid_mode(self, sqlite_client):
        resp = sqlite_client.post(
            "/api/documents/session/sess_1",
            json={"document_id": "doc_1", "inject_mode": "invalid"},
        )
        assert resp.status_code == 400
