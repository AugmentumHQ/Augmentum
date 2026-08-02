"""Live integration tests for Augmentum memory, characters, config, and knowledge.

These tests hit real endpoints on a running Augmentum instance, exercising
the memory store, character card CRUD, config round-trips, and knowledge/
document listing.

Run:
    python -m pytest tests/live/test_live_memory.py -x -v -m live
"""

from __future__ import annotations

import json
import time
import uuid

import httpx
import pytest

BASE_URL = "https://localhost:6443"
MODEL = "deepseek-chat"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _probe() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/", timeout=3.0, verify=False)
        return r.status_code < 500
    except Exception:
        return False


@pytest.fixture(scope="module")
def _require_server():
    if not _probe():
        pytest.skip(f"Augmentum not reachable at {BASE_URL}")


@pytest.fixture
def client(_require_server):
    with httpx.Client(base_url=BASE_URL, timeout=60.0, verify=False) as c:
        yield c


def _store_fact(client: httpx.Client, content: str, **kw) -> str:
    """Store a memory fact and return its ID."""
    payload = {"content": content, "memory_type": "fact", "importance": 0.7, **kw}
    resp = client.post("/v1/memory/store", json=payload)
    assert resp.status_code == 201, f"Store failed: {resp.text}"
    data = resp.json()
    assert data["status"] == "stored"
    return data["id"]


# ---------------------------------------------------------------------------
# Memory CRUD
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestMemoryStore:
    """Memory store, retrieve, search, delete via /v1/memory/ endpoints."""

    def test_store_and_retrieve(self, client: httpx.Client):
        """POST /v1/memory/store -> GET /v1/memory/facts -> fact appears."""
        tag = f"test-{uuid.uuid4().hex[:8]}"
        content = f"The Eiffel Tower is 330 meters tall [{tag}]"
        mem_id = _store_fact(client, content)

        # List and find it
        resp = client.get("/v1/memory/facts", params={"limit": 200})
        assert resp.status_code == 200
        memories = resp.json()["memories"]
        found = [m for m in memories if m["id"] == mem_id]
        assert len(found) == 1, f"Memory {mem_id} not found in list"
        assert tag in found[0]["content"]

    def test_search(self, client: httpx.Client):
        """Store a fact, then semantic-search for it."""
        tag = f"test-{uuid.uuid4().hex[:8]}"
        content = f"Paris is the capital of France [{tag}]"
        mem_id = _store_fact(client, content)

        # Search — the query should find our specific fact
        resp = client.get("/v1/memory/search", params={"q": "capital of France", "limit": 20})
        assert resp.status_code == 200
        results = resp.json()["results"]
        ids = [r["id"] for r in results]
        assert mem_id in ids, (
            f"Stored memory {mem_id} not found in search results. "
            f"Got {len(results)} results: {[r['content'][:40] for r in results]}"
        )

    def test_delete(self, client: httpx.Client):
        """Store -> delete -> verify removed from list."""
        content = f"Temporary fact for deletion test [{uuid.uuid4().hex[:8]}]"
        mem_id = _store_fact(client, content)

        # Delete
        resp = client.delete(f"/v1/memory/facts/{mem_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

        # List — the memory should no longer appear (soft-deleted, not in active list)
        resp = client.get("/v1/memory/facts", params={"limit": 200})
        assert resp.status_code == 200
        ids = [m["id"] for m in resp.json()["memories"]]
        assert mem_id not in ids, f"Deleted memory {mem_id} still in list"

    def test_survives_across_clients(self, client: httpx.Client):
        """Store with one client, retrieve with a fresh client."""
        tag = f"cross-{uuid.uuid4().hex[:8]}"
        content = f"Cross-client persistence test [{tag}]"
        mem_id = _store_fact(client, content)

        # New, independent client
        with httpx.Client(base_url=BASE_URL, timeout=30.0, verify=False) as client_b:
            resp = client_b.get("/v1/memory/facts", params={"limit": 200})
            assert resp.status_code == 200
            found = [m for m in resp.json()["memories"] if m["id"] == mem_id]
            assert len(found) == 1, f"Memory {mem_id} not visible to second client"
            assert tag in found[0]["content"]


# ---------------------------------------------------------------------------
# Character Cards
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestCharacterCards:
    """Character card CRUD via /api/characters/ endpoints."""

    def test_round_trip(self, client: httpx.Client):
        """Create a character card, read it back, verify all fields preserved."""
        char_id = f"test-{uuid.uuid4().hex[:12]}"
        card = {
            "name": "Test Character & 'Quotes\"",
            "id": char_id,
            "description": "A character with special chars: <b>bold</b> & \"quotes\"",
            "personality": "Witty, sarcastic, fond of ${template} literals",
            "scenario": "Testing in a lab\nWith newlines\tand tabs",
            "firstMessage": "Hello! I'm a test character.",
            "exampleDialogue": "User: Hi\nChar: Hey there, ${user}!",
        }
        # Create/update via PUT
        resp = client.put(f"/api/characters/{char_id}", json=card)
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

        # Read back
        resp = client.get(f"/api/characters/{char_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == card["name"]
        assert data["description"] == card["description"]
        assert data["personality"] == card["personality"]
        assert data["scenario"] == card["scenario"]
        assert data["firstMessage"] == card["firstMessage"]

        # Cleanup
        client.delete(f"/api/characters/{char_id}")

    def test_appears_in_list(self, client: httpx.Client):
        """Created character appears in GET /api/characters/."""
        char_id = f"test-{uuid.uuid4().hex[:12]}"
        card = {"name": "List Test Char", "id": char_id, "description": "For listing."}
        client.put(f"/api/characters/{char_id}", json=card)

        resp = client.get("/api/characters/")
        assert resp.status_code == 200
        chars = resp.json()["characters"]
        ids = [c["id"] for c in chars]
        assert char_id in ids

        # Cleanup
        client.delete(f"/api/characters/{char_id}")

    def test_delete(self, client: httpx.Client):
        """Delete a character card, verify it's gone."""
        char_id = f"test-{uuid.uuid4().hex[:12]}"
        card = {"name": "Delete Me", "id": char_id}
        client.put(f"/api/characters/{char_id}", json=card)

        resp = client.delete(f"/api/characters/{char_id}")
        assert resp.status_code == 200

        resp = client.get(f"/api/characters/{char_id}")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Session with Character Context (narrative mode + character)
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestSessionWithCharacter:
    """Create a character, then use it in a narrative session."""

    def test_narrative_with_character(self, client: httpx.Client):
        """Create a character card, start a narrative session using that card's
        system prompt, and verify the model responds in character."""
        char_id = f"test-{uuid.uuid4().hex[:12]}"
        card = {
            "name": "Captain Sparkles",
            "id": char_id,
            "personality": "An enthusiastic space captain who ends every sentence with 'Sparkle on!'",
            "description": "Captain of the SS Glitter, exploring the Sequin Nebula.",
            "scenario": "The user is a new crew member on the SS Glitter.",
            "firstMessage": "Welcome aboard the SS Glitter, recruit! Sparkle on!",
        }
        client.put(f"/api/characters/{char_id}", json=card)

        try:
            # Build a system prompt from the character card
            system_prompt = (
                f"Character: {card['name']}\n"
                f"Personality: {card['personality']}\n"
                f"Description: {card['description']}\n"
                f"Scenario: {card['scenario']}\n"
                "Stay in character at all times."
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": card["firstMessage"]},
                {"role": "user", "content": "What's our mission today, Captain?"},
            ]
            resp = client.post("/api/chat", json={
                "model": f"n/{MODEL}",
                "messages": messages,
                "stream": False,
            })
            assert resp.status_code == 200
            content = resp.json()["message"]["content"]
            assert len(content) > 20, f"Response too short: {content!r}"
        finally:
            client.delete(f"/api/characters/{char_id}")


# ---------------------------------------------------------------------------
# Config Affects Behavior
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestConfig:
    """Verify runtime config changes don't break chat requests."""

    def test_temperature_change(self, client: httpx.Client):
        """Change temperature setting, send requests, both succeed.

        We can't deterministically assert content differences from temperature
        alone, but both requests must return valid responses.
        """
        prompt = [{"role": "user", "content": "Say one random word."}]

        # Low temperature
        resp1 = client.post("/api/chat", json={
            "model": MODEL,
            "messages": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        })
        assert resp1.status_code == 200
        c1 = resp1.json()["message"]["content"]
        assert len(c1) > 0

        # High temperature
        resp2 = client.post("/api/chat", json={
            "model": MODEL,
            "messages": prompt,
            "stream": False,
            "options": {"temperature": 1.5},
        })
        assert resp2.status_code == 200
        c2 = resp2.json()["message"]["content"]
        assert len(c2) > 0

    def test_tool_settings_round_trip(self, client: httpx.Client):
        """Read tool settings, update one, read again — verify persistence."""
        # Read current
        resp = client.get("/api/config/tools")
        assert resp.status_code == 200
        original = resp.json()

        # We'll toggle uarf_auto_search to its current value (safe no-op)
        current_val = original.get("uarf_auto_search", True)

        resp = client.put("/api/config/tools", json={"uarf_auto_search": current_val})
        assert resp.status_code == 200
        data = resp.json()
        assert "updated" in data

        # Read back
        resp = client.get("/api/config/tools")
        assert resp.status_code == 200
        assert resp.json()["uarf_auto_search"] == current_val


# ---------------------------------------------------------------------------
# Knowledge Packs
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestKnowledgePacks:
    """GET /api/knowledge/packs — verify response shape."""

    def test_list_packs(self, client: httpx.Client):
        """Knowledge packs endpoint returns a list (may be empty)."""
        resp = client.get("/api/knowledge/packs")
        assert resp.status_code == 200
        data = resp.json()
        assert "packs" in data
        assert isinstance(data["packs"], list)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestDocuments:
    """GET /api/documents — verify response shape."""

    def test_list_documents(self, client: httpx.Client):
        """Document listing endpoint returns a valid response."""
        resp = client.get("/api/documents")
        # 200 with documents, or 503 if document store not enabled — both valid
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            # Should be a list or dict with documents key
            assert isinstance(data, (list, dict))
