"""Tests for chat_image_routes.py — chat image upload and retrieval."""

from __future__ import annotations

import asyncio
import base64


def _setup_chat_images_db(app):
    """Create the chat_images table."""
    sm = app.state.state_manager
    backend = sm.backend
    conn = backend.conn

    async def _create():
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_images (
                id TEXT PRIMARY KEY,
                mime_type TEXT NOT NULL,
                data BLOB NOT NULL,
                session_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await conn.commit()

    asyncio.get_event_loop().run_until_complete(_create())
    return conn


class TestUploadChatImage:
    def test_upload_missing_data_url(self, sqlite_client, app):
        _setup_chat_images_db(app)
        resp = sqlite_client.post(
            "/api/chat-images",
            json={"data_url": ""},
        )
        assert resp.status_code == 400

    def test_upload_invalid_data_url(self, sqlite_client, app):
        _setup_chat_images_db(app)
        resp = sqlite_client.post(
            "/api/chat-images",
            json={"data_url": "not-a-data-url"},
        )
        assert resp.status_code == 400

    def test_upload_success(self, sqlite_client, app):
        _setup_chat_images_db(app)
        # Create a small valid PNG-like data URL
        pixel = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100).decode()
        data_url = f"data:image/png;base64,{pixel}"
        resp = sqlite_client.post(
            "/api/chat-images",
            json={"data_url": data_url, "session_id": "sess_1"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert data["url"].startswith("/api/chat-images/")

    def test_upload_too_large(self, sqlite_client, app):
        _setup_chat_images_db(app)
        # Create a >20MB payload
        big_data = base64.b64encode(b"\x00" * (21 * 1024 * 1024)).decode()
        data_url = f"data:image/png;base64,{big_data}"
        resp = sqlite_client.post(
            "/api/chat-images",
            json={"data_url": data_url},
        )
        assert resp.status_code == 413


class TestGetChatImage:
    def test_get_not_found(self, sqlite_client, app):
        _setup_chat_images_db(app)
        resp = sqlite_client.get("/api/chat-images/nonexistent")
        assert resp.status_code == 404

    def test_get_success(self, sqlite_client, app):
        _setup_chat_images_db(app)
        # Upload first
        pixel = base64.b64encode(b"\x89PNG" + b"\x00" * 50).decode()
        data_url = f"data:image/png;base64,{pixel}"
        upload_resp = sqlite_client.post(
            "/api/chat-images",
            json={"data_url": data_url},
        )
        image_id = upload_resp.json()["id"]
        resp = sqlite_client.get(f"/api/chat-images/{image_id}")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
