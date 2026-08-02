"""Earned Understanding P3: the Mirror — why she believes a thing.

The evidence endpoint makes every belief explainable: its origin, the
independent signals that converged on it, the convergence strength, and how
recently it was reinforced (the visible edge of decay).

See docs/superpowers/specs/2026-06-20-earned-understanding-design.md (P3).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from augmentum.memory.evidence import EvidenceStore
from augmentum.memory.models import MemoryType, SourceType
from augmentum.proxy.memory_routes import (
    _days_since,
    _origin_phrase,
    get_belief_evidence,
)
from augmentum.state.backends.sqlite import SQLiteBackend

# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #

def test_origin_phrase_by_source():
    assert "playlist" in _origin_phrase("extracted", {"source": "playlist"}).lower()
    assert "offer" in _origin_phrase("extracted", {"source": "offer"}).lower()
    assert "directly" in _origin_phrase("explicit", None).lower()
    assert "pattern" in _origin_phrase("system", None).lower()
    assert "conversation" in _origin_phrase("extracted", None).lower()


def test_origin_phrase_parses_json_string_context():
    """Memory stores source_context as a JSON string, not a dict."""
    ctx = json.dumps({"source": "playlist", "playlist": "Study Jazz"})
    assert "playlist" in _origin_phrase("extracted", ctx).lower()
    # garbage string degrades gracefully to the source_type phrasing
    assert "conversation" in _origin_phrase("extracted", "{not json").lower()


def test_days_since():
    from datetime import UTC, datetime, timedelta
    assert _days_since("") is None
    assert _days_since("garbage") is None
    five_ago = (datetime.now(UTC) - timedelta(days=5)).isoformat()
    assert _days_since(five_ago) == 5


# --------------------------------------------------------------------------- #
# The endpoint (real store + evidence, fake request)
# --------------------------------------------------------------------------- #

def _patch_embed():
    def _emb(text):
        import hashlib
        vec = []
        seed = text.encode()
        while len(vec) < 768:
            seed = hashlib.sha256(seed).digest()
            vec.extend(b / 255.0 for b in seed)
        return vec[:768]
    p = patch("augmentum.memory.store.EmbeddingService")
    svc = p.start()
    svc.embed_one = MagicMock(side_effect=_emb)
    svc.to_blob = MagicMock(side_effect=lambda v: __import__("struct").pack(f"<{len(v)}f", *v))
    return p


def _req(store, ev, uid="u1"):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(memory_store=store, evidence_store=ev)),
        scope={"user": SimpleNamespace(id=uid)},
    )


async def _make():
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    from augmentum.memory.store import MemoryStore
    return backend, MemoryStore(backend), EvidenceStore(backend)


@pytest.mark.asyncio
async def test_evidence_endpoint_shows_trail_and_convergence(monkeypatch):
    backend, store, ev = await _make()
    p = _patch_embed()
    try:
        mid = await store.store(
            content="User likes jazz", memory_type=MemoryType.FACT,
            user_id="u1", source_type=SourceType.EXTRACTED)
        # Two independent corroborations.
        await ev.corroborate_belief(store, user_id="u1", memory_id=mid,
                                    source="playlist", claim='made "Study Jazz"')
        await ev.corroborate_belief(store, user_id="u1", memory_id=mid,
                                    source="chat", claim="mentioned jazz")

        resp = await get_belief_evidence(mid, _req(store, ev))
        body = json.loads(resp.body)

        assert body["memory_id"] == mid
        assert "conversation" in body["origin"].lower()        # EXTRACTED origin
        assert body["convergence"]["distinct_sources"] == 2
        assert body["convergence"]["score"] > 0
        sources = {t["source"] for t in body["trail"]}
        assert sources == {"playlist", "chat"}
        assert body["days_since_reinforced"] is not None
    finally:
        p.stop()
        await backend.close()


@pytest.mark.asyncio
async def test_evidence_endpoint_belief_with_no_trail(monkeypatch):
    """A chat-only belief still explains itself (origin), with an empty trail."""
    backend, store, ev = await _make()
    p = _patch_embed()
    try:
        mid = await store.store(
            content="User's dentist is Friday", memory_type=MemoryType.FACT,
            user_id="u1", source_type=SourceType.EXPLICIT)
        resp = await get_belief_evidence(mid, _req(store, ev))
        body = json.loads(resp.body)
        assert "directly" in body["origin"].lower()
        assert body["trail"] == []
        assert body["convergence"]["distinct_sources"] == 0
        # last_reinforced falls back to the belief's own creation
        assert body["last_reinforced_at"] == body["created_at"]
    finally:
        p.stop()
        await backend.close()


@pytest.mark.asyncio
async def test_evidence_endpoint_404_for_missing(monkeypatch):
    backend, store, ev = await _make()
    try:
        resp = await get_belief_evidence("nope", _req(store, ev))
        assert resp.status_code == 404
    finally:
        await backend.close()
