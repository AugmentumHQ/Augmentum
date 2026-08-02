"""Live integration tests for state persistence and settings wiring.

Requires a running Augmentum instance at https://localhost:6443.
Run with: pytest tests/live/test_live_state_persistence.py --run-live -x -v
"""

from __future__ import annotations

import json
import uuid

import httpx
import pytest

BASE = "https://localhost:6443"

pytestmark = pytest.mark.live


def _probe() -> bool:
    try:
        r = httpx.get(f"{BASE}/api/config/", timeout=3.0, verify=False)
        return r.status_code < 500
    except Exception:
        return False


@pytest.fixture(scope="module")
def client():
    if not _probe():
        pytest.skip("Augmentum not available at localhost:6443")
    with httpx.Client(base_url=BASE, timeout=15.0, verify=False) as c:
        yield c


@pytest.fixture()
def fresh_client():
    """A second, independent client — simulates a new browser tab."""
    if not _probe():
        pytest.skip("Augmentum not available at localhost:6443")
    with httpx.Client(base_url=BASE, timeout=15.0, verify=False) as c:
        yield c


# ---------------------------------------------------------------------------
# 1. Tool setting 4-layer round-trip
# ---------------------------------------------------------------------------

class TestToolSettingRoundTrip:
    def test_put_get_restore(self, client: httpx.Client):
        # Read current value
        orig = client.get("/api/config/tools").json()
        orig_val = orig.get("uarf_auto_search_queries", 3)

        # Change to a known value
        r = client.put("/api/config/tools", json={"uarf_auto_search_queries": 7})
        assert r.status_code == 200
        body = r.json()
        assert body["updated"]["uarf_auto_search_queries"] == 7

        # Fresh GET must reflect the change
        after = client.get("/api/config/tools").json()
        assert after["uarf_auto_search_queries"] == 7

        # Restore original
        r2 = client.put("/api/config/tools", json={"uarf_auto_search_queries": orig_val})
        assert r2.status_code == 200

        restored = client.get("/api/config/tools").json()
        assert restored["uarf_auto_search_queries"] == orig_val

    def test_bool_setting_round_trip(self, client: httpx.Client):
        orig = client.get("/api/config/tools").json()
        orig_val = orig.get("ghost_text_enabled", False)

        target = not orig_val
        r = client.put("/api/config/tools", json={"ghost_text_enabled": target})
        assert r.status_code == 200

        after = client.get("/api/config/tools").json()
        assert after["ghost_text_enabled"] is target

        # Restore
        client.put("/api/config/tools", json={"ghost_text_enabled": orig_val})

    def test_float_setting_round_trip(self, client: httpx.Client):
        orig = client.get("/api/config/tools").json()
        orig_val = orig.get("search_relevance_min_score", 0.0)

        r = client.put("/api/config/tools", json={"search_relevance_min_score": 0.42})
        assert r.status_code == 200

        after = client.get("/api/config/tools").json()
        assert abs(after["search_relevance_min_score"] - 0.42) < 0.001

        client.put("/api/config/tools", json={"search_relevance_min_score": orig_val})


# ---------------------------------------------------------------------------
# 2. String setting round-trip
# ---------------------------------------------------------------------------

class TestStringSettingRoundTrip:
    def test_personality_override(self, client: httpx.Client):
        orig = client.get("/api/config/tools").json()
        orig_val = orig.get("narrative_memory_prompt", "")

        test_prompt = f"live-test-{uuid.uuid4().hex[:8]}"
        r = client.put("/api/config/tools", json={"narrative_memory_prompt": test_prompt})
        assert r.status_code == 200

        after = client.get("/api/config/tools").json()
        assert after["narrative_memory_prompt"] == test_prompt

        # Clear it
        r2 = client.put("/api/config/tools", json={"narrative_memory_prompt": ""})
        assert r2.status_code == 200

        cleared = client.get("/api/config/tools").json()
        assert cleared["narrative_memory_prompt"] == ""

        # Restore original
        if orig_val:
            client.put("/api/config/tools", json={"narrative_memory_prompt": orig_val})

    def test_ui_string_setting(self, client: httpx.Client):
        """UI settings (personalization layer) round-trip."""
        orig = client.get("/api/config/ui").json()
        orig_name = orig.get("aiName", "")

        test_name = f"TestBot-{uuid.uuid4().hex[:6]}"
        r = client.put("/api/config/ui", json={"aiName": test_name})
        assert r.status_code == 200
        assert r.json()["updated"]["aiName"] == test_name

        after = client.get("/api/config/ui").json()
        assert after["aiName"] == test_name

        # Restore
        client.put("/api/config/ui", json={"aiName": orig_name})


# ---------------------------------------------------------------------------
# 3. Session CRUD lifecycle
# ---------------------------------------------------------------------------

class TestSessionCRUD:
    def test_create_read_delete(self, client: httpx.Client):
        sid = f"live-test-{uuid.uuid4().hex[:12]}"
        tree = {
            "msg-root": {
                "id": "msg-root",
                "role": "user",
                "content": "Hello from live test",
                "children": ["msg-reply"],
            },
            "msg-reply": {
                "id": "msg-reply",
                "role": "assistant",
                "content": "Hello back!",
                "children": [],
                "parent": "msg-root",
            },
        }
        payload = {
            "title": "Live Test Session",
            "mode": "passthrough",
            "tree": tree,
            "rootMessageId": "msg-root",
            "activeLeafId": "msg-reply",
        }

        # Create
        r = client.put(f"/api/chats/{sid}", json=payload)
        assert r.status_code == 200
        assert r.json()["ok"] is True

        # Read back
        r2 = client.get(f"/api/chats/{sid}")
        assert r2.status_code == 200
        data = r2.json()
        assert data["title"] == "Live Test Session"
        assert data["mode"] == "passthrough"
        assert "msg-root" in data["tree"]
        assert data["tree"]["msg-root"]["content"] == "Hello from live test"
        assert data["tree"]["msg-reply"]["content"] == "Hello back!"

        # Delete
        r3 = client.delete(f"/api/chats/{sid}")
        assert r3.status_code == 200

        # Verify gone
        r4 = client.get(f"/api/chats/{sid}")
        assert r4.status_code == 404


# ---------------------------------------------------------------------------
# 4. Session sync with branched message tree
# ---------------------------------------------------------------------------

class TestSessionBranchedTree:
    def test_branched_tree_survives_round_trip(self, client: httpx.Client):
        sid = f"live-branch-{uuid.uuid4().hex[:12]}"
        # Build a branched tree:
        #   root -> reply-a (branch 1)
        #        -> reply-b (branch 2) -> follow-up
        tree = {
            "root": {
                "id": "root",
                "role": "user",
                "content": "Tell me a story",
                "children": ["reply-a", "reply-b"],
            },
            "reply-a": {
                "id": "reply-a",
                "role": "assistant",
                "content": "Once upon a time (branch A)...",
                "parent": "root",
                "children": [],
            },
            "reply-b": {
                "id": "reply-b",
                "role": "assistant",
                "content": "In a galaxy far away (branch B)...",
                "parent": "root",
                "children": ["follow-up"],
            },
            "follow-up": {
                "id": "follow-up",
                "role": "user",
                "content": "Tell me more about branch B",
                "parent": "reply-b",
                "children": [],
            },
        }
        payload = {
            "title": "Branched Test",
            "mode": "narrative",
            "tree": tree,
            "rootMessageId": "root",
            "activeLeafId": "follow-up",
        }

        r = client.put(f"/api/chats/{sid}", json=payload)
        assert r.status_code == 200

        # Reload and verify tree structure
        data = client.get(f"/api/chats/{sid}").json()
        t = data["tree"]

        # Root has two children (branched)
        assert set(t["root"]["children"]) == {"reply-a", "reply-b"}
        # reply-b has a child
        assert t["reply-b"]["children"] == ["follow-up"]
        # Parent links intact
        assert t["follow-up"]["parent"] == "reply-b"
        assert t["reply-a"]["parent"] == "root"
        # Content intact
        assert "branch A" in t["reply-a"]["content"]
        assert "branch B" in t["reply-b"]["content"]

        # Cleanup
        client.delete(f"/api/chats/{sid}")


# ---------------------------------------------------------------------------
# 5. Character card CRUD
# ---------------------------------------------------------------------------

class TestCharacterCRUD:
    def test_create_update_delete(self, client: httpx.Client):
        cid = f"live-char-{uuid.uuid4().hex[:12]}"

        card = {
            "name": "Test Character",
            "description": "A test character for live integration tests",
            "personality": "Helpful and precise",
            "first_mes": "Hello, I am a test character.",
            "scenario": "Testing environment",
        }

        # Create
        r = client.put(f"/api/characters/{cid}", json=card)
        assert r.status_code == 200
        assert r.json()["ok"] is True

        # Read back
        r2 = client.get(f"/api/characters/{cid}")
        assert r2.status_code == 200
        data = r2.json()
        assert data["name"] == "Test Character"
        assert data["personality"] == "Helpful and precise"

        # Update name
        card["name"] = "Updated Character"
        card["personality"] = "Bold and creative"
        r3 = client.put(f"/api/characters/{cid}", json=card)
        assert r3.status_code == 200

        # Verify update
        r4 = client.get(f"/api/characters/{cid}")
        assert r4.status_code == 200
        updated = r4.json()
        assert updated["name"] == "Updated Character"
        assert updated["personality"] == "Bold and creative"

        # Delete
        r5 = client.delete(f"/api/characters/{cid}")
        assert r5.status_code == 200

        # Verify gone
        r6 = client.get(f"/api/characters/{cid}")
        assert r6.status_code == 404


# ---------------------------------------------------------------------------
# 6. Narrative preset lifecycle
# ---------------------------------------------------------------------------

class TestNarrativePresetLifecycle:
    def test_create_list_delete(self, client: httpx.Client):
        preset_name = f"LiveTestPreset-{uuid.uuid4().hex[:8]}"
        preset_body = {
            "name": preset_name,
            "system_prompt": "You are a helpful narrator.",
            "jailbreak": "",
            "post_history": "Continue the story.",
            "author_note": "Keep it brief.",
            "author_note_depth": 3,
            "is_default": False,
        }

        # Create
        r = client.post("/api/narrative/presets", json=preset_body)
        assert r.status_code == 200
        preset_id = r.json()["preset"]["id"]
        assert preset_id  # non-empty

        # List and find it
        r2 = client.get("/api/narrative/presets")
        assert r2.status_code == 200
        presets = r2.json()["presets"]
        matching = [p for p in presets if p["id"] == preset_id]
        assert len(matching) == 1
        assert matching[0]["name"] == preset_name
        assert matching[0]["system_prompt"] == "You are a helpful narrator."
        assert matching[0]["author_note_depth"] == 3

        # Delete
        r3 = client.delete(f"/api/narrative/presets/{preset_id}")
        assert r3.status_code == 200
        assert r3.json()["deleted"] is True

        # Verify gone
        r4 = client.get("/api/narrative/presets")
        remaining = [p for p in r4.json()["presets"] if p["id"] == preset_id]
        assert len(remaining) == 0


# ---------------------------------------------------------------------------
# 7. Browse notes persistence
# ---------------------------------------------------------------------------

class TestBrowseNotes:
    def test_create_list_delete(self, client: httpx.Client):
        note_title = f"LiveTestNote-{uuid.uuid4().hex[:8]}"
        note_body = {
            "title": note_title,
            "content": "This is test note content for live integration testing.",
            "tags": ["test", "live"],
            "source_url": "https://example.com/test",
            "source_title": "Example Page",
        }

        # Create
        r = client.post("/api/browse/notes", json=note_body)
        assert r.status_code == 201
        note = r.json()
        note_id = note["id"]
        assert note["title"] == note_title
        assert note["tags"] == ["test", "live"]

        # Get single note
        r2 = client.get(f"/api/browse/notes/{note_id}")
        assert r2.status_code == 200
        fetched = r2.json()
        assert fetched["content"] == note_body["content"]
        assert fetched["source_url"] == "https://example.com/test"

        # List and find it
        r3 = client.get("/api/browse/notes")
        assert r3.status_code == 200
        stubs = r3.json()["notes"]
        matching = [n for n in stubs if n.get("id") == note_id]
        assert len(matching) == 1

        # Delete
        r4 = client.delete(f"/api/browse/notes/{note_id}")
        assert r4.status_code == 200

        # Verify gone
        r5 = client.get(f"/api/browse/notes/{note_id}")
        assert r5.status_code == 404


# ---------------------------------------------------------------------------
# 8. Config persistence across "restart" (fresh client)
# ---------------------------------------------------------------------------

class TestConfigPersistenceAcrossClients:
    def test_setting_visible_from_fresh_client(
        self, client: httpx.Client, fresh_client: httpx.Client
    ):
        """A value set by one client must be visible from a different client."""
        orig = client.get("/api/config/tools").json()
        orig_val = orig.get("narrative_extraction_interval", 5)

        # Set via first client
        unique_val = 13  # within range [1, 20]
        r = client.put(
            "/api/config/tools",
            json={"narrative_extraction_interval": unique_val},
        )
        assert r.status_code == 200

        # Read from FRESH client (simulates new browser tab / restart)
        after = fresh_client.get("/api/config/tools").json()
        assert after["narrative_extraction_interval"] == unique_val

        # Restore
        client.put(
            "/api/config/tools",
            json={"narrative_extraction_interval": orig_val},
        )

    def test_ui_setting_visible_from_fresh_client(
        self, client: httpx.Client, fresh_client: httpx.Client
    ):
        """UI settings also persist server-side across clients."""
        orig = client.get("/api/config/ui").json()
        orig_style = orig.get("responseStyle", "")

        tag = f"test-style-{uuid.uuid4().hex[:6]}"
        client.put("/api/config/ui", json={"responseStyle": tag})

        after = fresh_client.get("/api/config/ui").json()
        assert after.get("responseStyle") == tag

        # Restore
        client.put("/api/config/ui", json={"responseStyle": orig_style})


# ---------------------------------------------------------------------------
# 9. Bulk session sync
# ---------------------------------------------------------------------------

class TestBulkSessionSync:
    def test_sync_three_sessions(self, client: httpx.Client):
        ids = [f"live-sync-{uuid.uuid4().hex[:10]}" for _ in range(3)]
        sessions = {}
        for i, sid in enumerate(ids):
            sessions[sid] = {
                "title": f"Sync Session {i}",
                "mode": "passthrough",
                "tree": {
                    f"msg-{i}": {
                        "id": f"msg-{i}",
                        "role": "user",
                        "content": f"Message in session {i}",
                        "children": [],
                    }
                },
                "rootMessageId": f"msg-{i}",
                "activeLeafId": f"msg-{i}",
            }

        # Bulk sync
        r = client.post("/api/chats/sync", json={"sessions": sessions})
        assert r.status_code == 200
        assert r.json()["imported"] == 3

        # Verify each individually
        for i, sid in enumerate(ids):
            r2 = client.get(f"/api/chats/{sid}")
            assert r2.status_code == 200, f"Session {sid} not found after sync"
            data = r2.json()
            assert data["title"] == f"Sync Session {i}"
            assert f"msg-{i}" in data["tree"]

        # Cleanup
        for sid in ids:
            client.delete(f"/api/chats/{sid}")


# ---------------------------------------------------------------------------
# 10. Memory CRUD
# ---------------------------------------------------------------------------

class TestMemoryCRUD:
    def test_store_list_delete(self, client: httpx.Client):
        tag = uuid.uuid4().hex[:8]
        content = f"Live test memory fact: {tag}"

        # Store
        r = client.post(
            "/v1/memory/store",
            json={
                "content": content,
                "memory_type": "fact",
                "importance": 0.8,
                "user_id": "default",
            },
        )
        if r.status_code == 503:
            pytest.skip("Memory system not initialized on this instance")
        assert r.status_code == 201
        memory_id = r.json()["id"]
        assert memory_id

        # List and find it
        r2 = client.get("/v1/memory/facts", params={"user_id": "default", "limit": 100})
        assert r2.status_code == 200
        memories = r2.json()["memories"]
        matching = [m for m in memories if m["id"] == memory_id]
        assert len(matching) == 1
        assert matching[0]["content"] == content
        assert matching[0]["importance"] == pytest.approx(0.8, abs=0.01)
        assert matching[0]["memory_type"] == "fact"

        # Delete (soft-delete)
        r3 = client.delete(f"/v1/memory/facts/{memory_id}")
        assert r3.status_code == 200

        # Verify no longer in active list (soft-deleted, not returned by default)
        r4 = client.get("/v1/memory/facts", params={"user_id": "default", "limit": 100})
        remaining = [m for m in r4.json()["memories"] if m["id"] == memory_id]
        assert len(remaining) == 0


# ---------------------------------------------------------------------------
# 11. Flow CRUD
# ---------------------------------------------------------------------------

class TestFlowCRUD:
    def test_create_get_update_delete(self, client: httpx.Client):
        flow_name = f"LiveTestFlow-{uuid.uuid4().hex[:8]}"
        flow_body = {
            "name": flow_name,
            "description": "A test flow for live integration",
            "steps": [
                {
                    "id": "step-1",
                    "tool": "calculator",
                    "params": {"expression": "2+2"},
                },
            ],
            "trigger_pattern": "",
        }

        # Create
        r = client.post("/api/flows", json=flow_body)
        if r.status_code == 503:
            pytest.skip("Flow store not available on this instance")
        assert r.status_code == 201
        flow = r.json()["flow"]
        flow_id = flow["id"]
        assert flow["name"] == flow_name

        # List and find it
        r2 = client.get("/api/flows")
        assert r2.status_code == 200
        flows = r2.json()["flows"]
        matching = [f for f in flows if f["id"] == flow_id]
        assert len(matching) == 1

        # Get by ID
        r3 = client.get(f"/api/flows/{flow_id}")
        assert r3.status_code == 200
        fetched = r3.json()
        assert fetched["name"] == flow_name
        assert fetched["description"] == "A test flow for live integration"

        # Update
        r4 = client.put(
            f"/api/flows/{flow_id}",
            json={
                "name": f"{flow_name}-updated",
                "description": "Updated description",
                "steps": [
                    {
                        "id": "step-1",
                        "tool": "calculator",
                        "params": {"expression": "3+3"},
                    },
                    {
                        "id": "step-2",
                        "tool": "datetime",
                        "params": {},
                    },
                ],
            },
        )
        assert r4.status_code == 200
        updated_flow = r4.json()["flow"]
        assert updated_flow["name"] == f"{flow_name}-updated"
        assert updated_flow["description"] == "Updated description"
        # DB column is "steps_json" (raw from SQLite), not "steps"
        steps = updated_flow.get("steps") or updated_flow.get("steps_json")
        if isinstance(steps, str):
            steps = json.loads(steps)
        assert len(steps) == 2

        # Delete
        r5 = client.delete(f"/api/flows/{flow_id}")
        assert r5.status_code == 200
        assert r5.json()["deleted"] is True

        # Verify gone
        r6 = client.get(f"/api/flows/{flow_id}")
        assert r6.status_code == 404


# ---------------------------------------------------------------------------
# 12. Concurrent writes
# ---------------------------------------------------------------------------

class TestConcurrentWrites:
    def test_two_clients_write_different_sessions(
        self, client: httpx.Client, fresh_client: httpx.Client
    ):
        """Two clients simultaneously PUT different sessions — both must survive."""
        sid_a = f"live-concurrent-a-{uuid.uuid4().hex[:10]}"
        sid_b = f"live-concurrent-b-{uuid.uuid4().hex[:10]}"

        payload_a = {
            "title": "Concurrent A",
            "mode": "passthrough",
            "tree": {
                "a1": {
                    "id": "a1",
                    "role": "user",
                    "content": "Session A message",
                    "children": [],
                }
            },
            "rootMessageId": "a1",
            "activeLeafId": "a1",
        }
        payload_b = {
            "title": "Concurrent B",
            "mode": "analytical",
            "tree": {
                "b1": {
                    "id": "b1",
                    "role": "user",
                    "content": "Session B message",
                    "children": [],
                }
            },
            "rootMessageId": "b1",
            "activeLeafId": "b1",
        }

        # Fire both writes (sequentially from sync clients, but tests the
        # server handling two rapid writes from different "clients")
        r_a = client.put(f"/api/chats/{sid_a}", json=payload_a)
        r_b = fresh_client.put(f"/api/chats/{sid_b}", json=payload_b)
        assert r_a.status_code == 200
        assert r_b.status_code == 200

        # Both must be readable and correct
        data_a = client.get(f"/api/chats/{sid_a}").json()
        data_b = fresh_client.get(f"/api/chats/{sid_b}").json()

        assert data_a["title"] == "Concurrent A"
        assert data_a["tree"]["a1"]["content"] == "Session A message"

        assert data_b["title"] == "Concurrent B"
        assert data_b["tree"]["b1"]["content"] == "Session B message"

        # Cross-read: each client can see the other's session
        cross_a = fresh_client.get(f"/api/chats/{sid_a}").json()
        cross_b = client.get(f"/api/chats/{sid_b}").json()
        assert cross_a["title"] == "Concurrent A"
        assert cross_b["title"] == "Concurrent B"

        # Cleanup
        client.delete(f"/api/chats/{sid_a}")
        client.delete(f"/api/chats/{sid_b}")

    def test_concurrent_writes_to_same_session(
        self, client: httpx.Client, fresh_client: httpx.Client
    ):
        """Last-write-wins: two rapid updates to the same session."""
        sid = f"live-race-{uuid.uuid4().hex[:10]}"

        base = {
            "title": "Race Session",
            "mode": "passthrough",
            "tree": {
                "m1": {"id": "m1", "role": "user", "content": "original", "children": []},
            },
            "rootMessageId": "m1",
            "activeLeafId": "m1",
        }

        # Create
        client.put(f"/api/chats/{sid}", json=base)

        # Two rapid updates
        update_a = {**base, "title": "Updated by A"}
        update_b = {**base, "title": "Updated by B"}

        client.put(f"/api/chats/{sid}", json=update_a)
        fresh_client.put(f"/api/chats/{sid}", json=update_b)

        # Whichever wrote last wins — the point is no crash / corruption
        final = client.get(f"/api/chats/{sid}").json()
        assert final["title"] in ("Updated by A", "Updated by B")
        assert final["tree"]["m1"]["content"] == "original"

        client.delete(f"/api/chats/{sid}")
