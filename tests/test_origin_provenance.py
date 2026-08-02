"""Provenance, not silos — origin column round-trips (migration 259).

Companion-created notes and image generations live in the SAME stores
and surfaces as user-created ones, distinguished only by ``origin``.
These tests run the REAL migrations on :memory: so schema drift between
the migration and the store code fails here, not in prod.
"""

from __future__ import annotations

import pytest

from augmentum.state.backends.sqlite import SQLiteBackend

UID = "user-prov-test"


@pytest.fixture
async def backend():
    b = SQLiteBackend(":memory:")
    await b.connect()
    yield b
    await b.close()


@pytest.mark.asyncio
async def test_note_origin_roundtrip(backend):
    from augmentum.state.notes_store import NotesStore
    store = NotesStore(backend.conn)

    await store.create({
        "id": "n-hers", "title": "Her note", "content": "x",
        "origin": "companion",
        "created_at": "2026-06-12", "updated_at": "2026-06-12",
    }, user_id=UID)
    await store.create({
        "id": "n-mine", "title": "My note", "content": "y",
        "created_at": "2026-06-12", "updated_at": "2026-06-12",
    }, user_id=UID)

    hers = await store.get("n-hers", user_id=UID)
    mine = await store.get("n-mine", user_id=UID)
    assert hers["origin"] == "companion"
    assert mine["origin"] == ""

    # list_stubs carries origin so the notes-list chip can filter.
    stubs = {s["id"]: s for s in await store.list_stubs(user_id=UID)}
    assert stubs["n-hers"]["origin"] == "companion"
    assert stubs["n-mine"]["origin"] == ""

    # Origin survives an edit — update() never touches it.
    await store.update("n-hers", {"content": "x2"}, user_id=UID)
    assert (await store.get("n-hers", user_id=UID))["origin"] == "companion"


@pytest.mark.asyncio
async def test_note_create_verb_marks_companion(backend):
    """The note.create intent verb writes origin='companion'."""
    import augmentum.intent  # noqa: F401 — registers builtins
    from augmentum.intent.action import SessionContext
    from augmentum.intent.registry import REGISTRY
    from augmentum.state.notes_store import NotesStore

    store = NotesStore(backend.conn)

    class _State:
        notes_store = store

    session = SessionContext(
        session_id="s1", user_id=UID, app_state=_State(),
    )
    action = REGISTRY.get("note.create")
    result = await action.handler(
        "", session, {"title": "Groceries", "content": "eggs"},
    )
    assert result is not None
    stubs = await store.list_stubs(user_id=UID)
    assert len(stubs) == 1
    assert stubs[0]["origin"] == "companion"


@pytest.mark.asyncio
async def test_image_generation_origin_filter(backend):
    from augmentum.image.persistence import ImagePersistence
    p = ImagePersistence(backend.conn)

    async def _save(image_id, origin):
        await p.save_generation(
            image_id=image_id, session_id="", prompt="a cat",
            negative_prompt="", model="m", seed=1, width=8, height=8,
            steps=1, cfg_scale=1.0, preset="", loras=[],
            file_path="", user_id=UID, origin=origin,
        )

    await _save("img-hers", "companion")
    await _save("img-mine", "")

    all_rows = await p.list_generations(user_id=UID)
    assert {e.image_id for e in all_rows} == {"img-hers", "img-mine"}
    origins = {e.image_id: e.origin for e in all_rows}
    assert origins["img-hers"] == "companion"
    assert origins["img-mine"] == ""

    hers = await p.list_generations(user_id=UID, origin="companion")
    assert [e.image_id for e in hers] == ["img-hers"]
    assert await p.count_generations(user_id=UID, origin="companion") == 1
    assert await p.count_generations(user_id=UID) == 2
