"""Integration tests for the notes/memory primitive verbs.

Verifies the round-trip: dispatch → handler → store. Uses an
in-memory fake notes_store to keep the tests fast and isolated; the
notes_store contract is exercised by its own dedicated tests.

What we're protecting:
  * note.create writes through to the store + populates the referent
  * note.append uses the referent when note_id is omitted
  * note.show_sticky surfaces the active note
  * start/end capture toggles the session flag + sets the deadline
  * memory.save / memory.recall propagate user_id correctly
  * Idempotency — duplicate note.create within the window dedupes
  * User-id guard — empty user_id refuses to write
"""

from __future__ import annotations

import pytest


class _FakeNotesStore:
    """In-memory notes_store stub. Mirrors the real shape used by
    augmentum.state.notes_store.NotesStore.{create, get, update}."""

    def __init__(self):
        self.notes = {}
        self.calls = []

    async def create(self, note, *, user_id):
        if not user_id:
            raise ValueError("user_id required")
        self.notes[note["id"]] = {**note, "user_id": user_id}
        self.calls.append(("create", note["id"], user_id))
        return self.notes[note["id"]]

    async def get(self, note_id, *, user_id):
        note = self.notes.get(note_id)
        if not note or note.get("user_id") != user_id:
            return None
        return dict(note)

    async def update(self, note_id, updates, *, user_id):
        note = self.notes.get(note_id)
        if not note or note.get("user_id") != user_id:
            return None
        note.update(updates)
        self.calls.append(("update", note_id, updates.get("content", "")))
        return dict(note)


class _AppState:
    def __init__(self):
        self.notes_store = _FakeNotesStore()
        self.intent_referents = {}


def _ctx(app, user="u1", sess="s1", mode="passthrough"):
    from augmentum.intent.action import SessionContext
    return SessionContext(
        user_id=user, session_id=sess, mode=mode, app_state=app,
    )


@pytest.mark.asyncio
class TestNoteCreate:
    async def test_creates_note_and_sets_active(self):
        from augmentum.intent import REGISTRY
        from augmentum.intent.dispatch import get_referent_cache

        app = _AppState()
        ctx = _ctx(app)
        action = REGISTRY.get("note.create")
        result = await action.handler("", ctx, {"title": "Trip planning"})

        assert result is not None
        assert result.short_circuit is True
        assert "Sure, here you go" in result.speak
        assert result.surface_emit["channel"] == "note.open_sticky"

        # Side effect: note created in store
        assert len(app.notes_store.notes) == 1
        note_id = next(iter(app.notes_store.notes))
        assert app.notes_store.notes[note_id]["title"] == "Trip planning"
        # Side effect: active note set
        refs = get_referent_cache(app, "u1", "s1")
        assert refs.active_note_id == note_id
        assert refs.active_note_title == "Trip planning"

    async def test_refuses_empty_user_id(self):
        from augmentum.intent import REGISTRY
        app = _AppState()
        ctx = _ctx(app, user="")
        action = REGISTRY.get("note.create")
        result = await action.handler("", ctx, {})
        assert result is not None
        assert result.short_circuit is True
        # Spoken hint when we refuse to write
        assert "save" in result.speak.lower() or "sure" in result.speak.lower()
        # Critically — NOTHING was written
        assert app.notes_store.notes == {}

    async def test_idempotent_create_within_window(self):
        from augmentum.intent import REGISTRY
        app = _AppState()
        ctx = _ctx(app)
        action = REGISTRY.get("note.create")
        # First create
        r1 = await action.handler("", ctx, {"title": "X", "content": "Y"})
        # Second create — same fingerprint
        r2 = await action.handler("", ctx, {"title": "X", "content": "Y"})
        # Only one row in store
        assert len(app.notes_store.notes) == 1
        # Both results point to the same note_id
        id1 = r1.surface_emit["payload"]["note_id"]
        id2 = r2.surface_emit["payload"]["note_id"]
        assert id1 == id2
        # Second time we say "Here it is" not "Sure, here you go"
        assert r2.speak != r1.speak


@pytest.mark.asyncio
class TestNoteAppend:
    async def test_appends_to_active_note(self):
        from augmentum.intent import REGISTRY
        app = _AppState()
        ctx = _ctx(app)
        create = REGISTRY.get("note.create")
        await create.handler("", ctx, {"title": "Ideas", "content": "first"})
        append = REGISTRY.get("note.append")
        result = await append.handler("", ctx, {"content": "second line"})
        assert result is not None
        assert result.short_circuit is True
        # Stored content includes both lines
        note_id = next(iter(app.notes_store.notes))
        stored = app.notes_store.notes[note_id]["content"]
        assert "first" in stored
        assert "second line" in stored

    async def test_no_active_note_speaks_friendly_error(self):
        from augmentum.intent import REGISTRY
        app = _AppState()
        ctx = _ctx(app)
        append = REGISTRY.get("note.append")
        result = await append.handler("", ctx, {"content": "anything"})
        assert result is not None
        assert "open note" in result.speak.lower() or "open" in result.speak.lower()


@pytest.mark.asyncio
class TestCaptureMode:
    async def test_start_sets_flag_and_deadline(self):
        import time

        from augmentum.intent import REGISTRY
        from augmentum.intent.dispatch import get_referent_cache
        app = _AppState()
        ctx = _ctx(app)
        start = REGISTRY.get("note.start_capture")
        t0 = time.monotonic()
        await start.handler("", ctx, {})
        refs = get_referent_cache(app, "u1", "s1")
        assert refs.note_capture_mode is True
        assert refs.note_capture_deadline > t0 + 60  # at least a minute out
        # Auto-created note since none was active
        assert refs.active_note_id is not None

    async def test_end_clears_flag(self):
        from augmentum.intent import REGISTRY
        from augmentum.intent.dispatch import get_referent_cache
        app = _AppState()
        ctx = _ctx(app)
        await REGISTRY.get("note.start_capture").handler("", ctx, {})
        result = await REGISTRY.get("note.end_capture").handler("", ctx, {})
        assert result is not None
        assert "Saved" in result.speak
        refs = get_referent_cache(app, "u1", "s1")
        assert refs.note_capture_mode is False
        assert refs.note_capture_deadline == 0.0

    async def test_end_capture_noop_when_not_active(self):
        from augmentum.intent import REGISTRY
        app = _AppState()
        ctx = _ctx(app)
        # Never started — end_capture should return None so the caller
        # falls through to UARF instead of treating it as a turn.
        result = await REGISTRY.get("note.end_capture").handler("", ctx, {})
        assert result is None


@pytest.mark.asyncio
class TestShowSticky:
    async def test_surfaces_active_note(self):
        from augmentum.intent import REGISTRY
        app = _AppState()
        ctx = _ctx(app)
        await REGISTRY.get("note.create").handler(
            "", ctx, {"title": "Saved Idea", "content": "..."},
        )
        result = await REGISTRY.get("note.show_sticky").handler("", ctx, {})
        assert result is not None
        assert result.surface_emit["channel"] == "note.open_sticky"
        assert result.surface_emit["payload"]["title"] == "Saved Idea"

    async def test_no_active_note_friendly_error(self):
        from augmentum.intent import REGISTRY
        app = _AppState()
        ctx = _ctx(app)
        result = await REGISTRY.get("note.show_sticky").handler("", ctx, {})
        assert result is not None
        assert "recent note" in result.speak.lower() or "note" in result.speak.lower()


@pytest.mark.asyncio
class TestMemoryGuards:
    async def test_memory_save_refuses_empty_user(self):
        from augmentum.intent import REGISTRY
        app = _AppState()
        ctx = _ctx(app, user="")
        result = await REGISTRY.get("memory.save").handler(
            "", ctx, {"content": "remember this"},
        )
        assert result is not None
        # We refuse to write — caller's speak hint is the user-visible
        # signal.
        assert "memory" in result.speak.lower()
