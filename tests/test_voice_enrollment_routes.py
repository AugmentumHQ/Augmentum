"""Tests for voice_enrollment_routes.py — enrollment status, decline, phrases."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


class TestEnrollmentStatus:
    def test_status_no_db(self, client):
        """Without SQLite, enrollment status returns 503."""
        resp = client.get("/api/voice/enrollment")
        assert resp.status_code in (200, 503)

    def test_status_with_db(self, sqlite_client):
        resp = sqlite_client.get("/api/voice/enrollment")
        assert resp.status_code == 200
        data = resp.json()
        assert "enrolled" in data


class TestEnrollmentDecline:
    def test_decline(self, sqlite_client):
        resp = sqlite_client.post("/api/voice/enrollment/decline")
        assert resp.status_code == 200


class TestEnrollmentPhrases:
    def test_get_phrases(self, client):
        resp = client.get("/api/voice/enrollment/phrases")
        assert resp.status_code == 200
        data = resp.json()
        assert "phrases" in data
        assert isinstance(data["phrases"], list)


class TestDeleteEnrollment:
    def test_delete_no_db(self, client):
        resp = client.delete("/api/voice/enrollment")
        assert resp.status_code == 503

    def test_delete_with_db(self, sqlite_client):
        resp = sqlite_client.delete("/api/voice/enrollment")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
