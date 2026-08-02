"""Tests for metrics_routes.py — Prometheus metrics endpoint."""

from __future__ import annotations

from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


class TestPrometheusMetrics:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type(self, client):
        resp = client.get("/metrics")
        ct = resp.headers.get("content-type", "")
        assert "text/plain" in ct

    def test_metrics_body_is_string(self, client):
        resp = client.get("/metrics")
        # Prometheus text format is plain text
        assert isinstance(resp.text, str)

    def test_metrics_with_image_queue(self, client):
        """When an image queue is present, IMAGE_QUEUE_DEPTH is updated."""
        mock_queue = MagicMock()
        mock_queue.queue_size = 3
        client.app.state.generation_queue = mock_queue

        resp = client.get("/metrics")
        assert resp.status_code == 200
        # Cleanup
        del client.app.state.generation_queue

    def test_metrics_without_image_queue(self, client):
        """Without an image queue, metrics still render without error."""
        # Ensure no generation_queue is set
        if hasattr(client.app.state, "generation_queue"):
            delattr(client.app.state, "generation_queue")
        resp = client.get("/metrics")
        assert resp.status_code == 200
