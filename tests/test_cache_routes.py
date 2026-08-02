"""Tests for cache management API routes (/api/cache/)."""

from __future__ import annotations


# ── GET /stats ────────────────────────────────────────────────────────────

def test_cache_stats(client):
    """GET /api/cache/stats returns stats for all cache subsystems."""
    resp = client.get("/api/cache/stats")
    assert resp.status_code == 200
    data = resp.json()
    # The conftest app sets up prompt_cache, prefix_cache, and request_deduplicator
    assert "prompt_cache" in data
    assert "prefix_cache" in data
    assert "deduplicator" in data


def test_cache_stats_prompt_cache_shape(client):
    """Prompt cache stats include hit/miss counters and size."""
    resp = client.get("/api/cache/stats")
    data = resp.json()
    pc = data["prompt_cache"]
    assert "size" in pc


# ── POST /clear ──────────────────────────────────────────────────────────

def test_cache_clear(client):
    """POST /api/cache/clear clears caches and returns counts."""
    resp = client.post("/api/cache/clear")
    assert resp.status_code == 200
    data = resp.json()
    assert "prompt_cache_cleared" in data
    assert "prefix_cache_cleared" in data


def test_cache_clear_returns_zero_when_empty(client):
    """POST /api/cache/clear on empty caches returns zero counts."""
    resp = client.post("/api/cache/clear")
    assert resp.status_code == 200
    data = resp.json()
    assert data["prompt_cache_cleared"] == 0
    assert data["prefix_cache_cleared"] == 0


# ── GET /entries ─────────────────────────────────────────────────────────

def test_cache_entries_empty(client):
    """GET /api/cache/entries returns empty entries when caches are fresh."""
    resp = client.get("/api/cache/entries")
    assert resp.status_code == 200
    data = resp.json()
    assert "prompt_cache" in data
    assert "prefix_cache" in data
