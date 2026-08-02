"""Tests for the universe-continuation + drafts-list features.

These tests cover the post-2026-05 additions:
  - GET /api/characters/cardsmith/sessions returns non-finalized drafts
    for the caller, ordered by recency, with friendly labels.
  - POST /start accepts ``parent_session_id`` and copies wiki/scratchpad
    state from the parent to the child session.
  - Cross-tenant guards on both endpoints.

These rely on the SQLite backend because ``/sessions`` reads from the
cardsmith_sessions disk table and the chain path reads parent meta via
``_resolve_session`` (memory miss → disk fallback).
"""

from __future__ import annotations

import json
import time

import pytest


def _post_start(client, **body) -> dict:
    payload = {"card_type": "single", "source": "describe"}
    payload.update(body)
    resp = client.post("/api/characters/cardsmith/start", json=payload)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ── /sessions endpoint ────────────────────────────────────────────────────


class TestListSessions:
    def test_empty_when_no_drafts(self, sqlite_client):
        resp = sqlite_client.get("/api/characters/cardsmith/sessions")
        assert resp.status_code == 200
        assert resp.json() == {"sessions": []}

    def test_lists_user_drafts(self, sqlite_client):
        # Two fresh sessions
        sid1 = _post_start(sqlite_client)["session_id"]
        time.sleep(0.01)  # ensure last_active_at ordering is deterministic
        sid2 = _post_start(sqlite_client, card_type="ensemble")["session_id"]

        resp = sqlite_client.get("/api/characters/cardsmith/sessions")
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert len(sessions) == 2
        ids = {s["session_id"] for s in sessions}
        assert ids == {sid1, sid2}
        # Newest first
        assert sessions[0]["session_id"] == sid2

    def test_friendly_label_falls_back_to_draft(self, sqlite_client):
        sid = _post_start(sqlite_client)["session_id"]
        resp = sqlite_client.get("/api/characters/cardsmith/sessions")
        row = next(s for s in resp.json()["sessions"] if s["session_id"] == sid)
        # No name committed, no wiki — falls through to "Empty draft" since
        # no messages have landed either.
        assert row["friendly_label"] in ("Empty draft", "0-turn draft")
        assert row["has_universe"] is False

    def test_excludes_other_users_drafts(self, sqlite_client):
        # Create a session for the test user
        _post_start(sqlite_client)
        # Forge a row for a different user_id directly via the backend so
        # we don't have to log in twice.
        import asyncio
        be = sqlite_client.app.state.state_manager.backend
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(be.conn.execute(
                """INSERT INTO cardsmith_sessions
                   (session_id, user_id, card_type, source, created_at,
                    last_active_at, messages, fields, meta, finalized)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "cs_other_user", "usr_other", "single", "describe",
                    time.time(), time.time(),
                    "[]", "{}", "{}", 0,
                ),
            ))
            loop.run_until_complete(be.conn.commit())
        finally:
            loop.close()
        resp = sqlite_client.get("/api/characters/cardsmith/sessions")
        sessions = resp.json()["sessions"]
        # The forged row must not leak through to the authenticated test user
        assert all(s["session_id"] != "cs_other_user" for s in sessions)


# ── Universe continuation via parent_session_id ───────────────────────────


class TestParentSessionInheritance:
    def test_unknown_parent_returns_404(self, sqlite_client):
        resp = sqlite_client.post(
            "/api/characters/cardsmith/start",
            json={
                "card_type": "single",
                "source": "describe",
                "parent_session_id": "cs_does_not_exist",
            },
        )
        assert resp.status_code == 404

    def test_other_users_parent_is_invisible(self, sqlite_client):
        # Forge a parent row for a different user.
        import asyncio
        be = sqlite_client.app.state.state_manager.backend
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(be.conn.execute(
                """INSERT INTO cardsmith_sessions
                   (session_id, user_id, card_type, source, created_at,
                    last_active_at, messages, fields, meta, finalized)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "cs_alien_parent", "usr_other", "single", "wiki",
                    time.time(), time.time(),
                    "[]", "{}",
                    json.dumps({"wiki_host": "example.com"}),
                    1,
                ),
            ))
            loop.run_until_complete(be.conn.commit())
        finally:
            loop.close()
        resp = sqlite_client.post(
            "/api/characters/cardsmith/start",
            json={
                "card_type": "single",
                "source": "describe",
                "parent_session_id": "cs_alien_parent",
            },
        )
        # _resolve_session returns None for cross-tenant; route 404s.
        assert resp.status_code == 404

    def test_child_inherits_scratchpad_and_wiki(self, sqlite_client):
        # Forge a finalized parent owned by the test user with wiki context
        # and one scratchpad entry. Bypass the wiki HTTP path so the test
        # is self-contained.
        import asyncio
        be = sqlite_client.app.state.state_manager.backend
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(be.conn.execute(
                """INSERT INTO cardsmith_sessions
                   (session_id, user_id, card_type, source, created_at,
                    last_active_at, messages, fields, meta, finalized)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    "cs_parent_finalized", "usr_test", "single", "wiki",
                    time.time(), time.time(),
                    "[]", "{}",
                    json.dumps({
                        "wiki_host": "wiki.example.com",
                        "wiki_title": "Test World",
                        "wiki_url": "https://wiki.example.com/world",
                        "wiki_context_data": {"title": "Test World"},
                        "scratchpad": [{
                            "url": "https://wiki.example.com/world",
                            "path": "/wiki/world",
                            "title": "Test World",
                            "summary": "A world for tests.",
                            "sections": {},
                            "infobox": {},
                            "aliases": [],
                            "extracted_links": [],
                            "source_kind": "mediawiki",
                            "zone": "active",
                            "consumed_by": "",
                            "fetched_at": time.time(),
                        }],
                        "universe_saves": [
                            {"char_id": "ch_lyra", "name": "Lyra Vex"},
                        ],
                    }),
                    1,
                ),
            ))
            loop.run_until_complete(be.conn.commit())
        finally:
            loop.close()

        resp = sqlite_client.post(
            "/api/characters/cardsmith/start",
            json={
                "card_type": "single",
                "source": "describe",
                "parent_session_id": "cs_parent_finalized",
            },
        )
        assert resp.status_code == 200
        child_sid = resp.json()["session_id"]

        # Inspect the child's in-memory state to confirm inheritance.
        from augmentum.modes.narrative.cardsmith import get_session
        child = get_session(child_sid, user_id="usr_test")
        assert child is not None
        assert child.meta.get("wiki_host") == "wiki.example.com"
        assert child.meta.get("wiki_title") == "Test World"
        assert child.meta.get("chained_from") == "cs_parent_finalized"
        # Scratchpad copied (one entry).
        sp = child.meta.get("scratchpad")
        assert isinstance(sp, list) and len(sp) == 1
        assert sp[0]["title"] == "Test World"
        # universe_saves carried forward.
        saves = child.meta.get("universe_saves")
        assert isinstance(saves, list) and len(saves) == 1
        assert saves[0]["name"] == "Lyra Vex"


# ── Finalize emits has_universe correctly ─────────────────────────────────


class TestFinalizeUniverseFlag:
    def test_finalize_without_wiki_returns_false(self, sqlite_client, mock_backend):
        from unittest.mock import AsyncMock
        from augmentum.modes.narrative.cardsmith import get_or_create_session

        sess = get_or_create_session(
            user_id="usr_test", card_type="single", source="describe",
        )
        # Populate enough fields that _state_is_sparse → False, so recovery
        # doesn't run.
        sess.fields = {
            "name": "Test", "personality": "stoic",
            "greeting": "Hi.", "scenario": "S",
            "desc_physical": "tall", "examples": "(user) hi (char) hi",
        }

        # The /finalize route resolves the same model the streaming /turn
        # uses — wire it even though recovery shouldn't run here.
        sqlite_client.app.state.provider_registry.resolve_model_for_role = AsyncMock(
            return_value=(mock_backend, "test-model"),
        )

        resp = sqlite_client.post(
            "/api/characters/cardsmith/finalize",
            json={"session_id": sess.session_id},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is True
        assert body["has_universe"] is False
        assert body["session_id"] == sess.session_id

    def test_finalize_with_wiki_scratchpad_returns_true(
        self, sqlite_client, mock_backend,
    ):
        from unittest.mock import AsyncMock
        from augmentum.modes.narrative.cardsmith import get_or_create_session

        sess = get_or_create_session(
            user_id="usr_test", card_type="single", source="wiki",
        )
        sess.meta["wiki_host"] = "wiki.example.com"
        sess.meta["scratchpad"] = [{
            "url": "https://wiki.example.com/x",
            "path": "/wiki/x",
            "title": "X",
            "summary": "x",
            "sections": {}, "infobox": {}, "aliases": [],
            "extracted_links": [],
            "source_kind": "mediawiki",
            "zone": "active", "consumed_by": "", "fetched_at": time.time(),
        }]
        sess.fields = {
            "name": "Test", "personality": "stoic",
            "greeting": "Hi.", "scenario": "S",
            "desc_physical": "tall", "examples": "(user) hi (char) hi",
        }
        sqlite_client.app.state.provider_registry.resolve_model_for_role = AsyncMock(
            return_value=(mock_backend, "test-model"),
        )
        resp = sqlite_client.post(
            "/api/characters/cardsmith/finalize",
            json={"session_id": sess.session_id},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["has_universe"] is True
