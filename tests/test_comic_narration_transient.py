"""Comic-narration audio as transient cache + retention prune.

Per-page narration audio (~30 artifacts/chapter) is regenerable playback
cache, not a deliverable. Pins the class fix:

- save_from_path(transient=True) stamps transient=1 and does NOT register
  in the file index / VFS (no Files-surface clutter);
- transient rows are invisible to list_all() but still fetchable by id
  (the player streams by artifact id);
- ComicNarrationStore.list_done orders newest-first, and the synth
  handler's prune drops chapters beyond comic_narration_cache_max,
  deleting their page artifacts.
"""

from __future__ import annotations

import pytest

from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.comic_narration_store import ComicNarrationStore
from augmentum.tools.artifact_storage import ArtifactStore

UID = "usr_test"


@pytest.fixture
async def backend(tmp_path, monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "data_dir", str(tmp_path), raising=False)
    b = SQLiteBackend(":memory:")
    await b.connect()
    await b._conn.execute(
        "INSERT OR IGNORE INTO users (id, username, password_hash) VALUES (?, ?, ?)",
        [UID, "test", "x"],
    )
    await b._conn.commit()
    yield b
    await b.close()


def _wav(tmp_path, name):
    p = tmp_path / name
    p.write_bytes(b"RIFF0000WAVEfmt ")
    return str(p)


@pytest.mark.asyncio
async def test_save_from_path_transient_hidden_but_fetchable(backend, tmp_path):
    store = ArtifactStore(backend._conn)
    saved = await store.save_from_path(
        _wav(tmp_path, "p1.wav"), "Comic p1.wav", "wav",
        display_name="Comic — page 1", user_id=UID, transient=True,
        metadata={"comic_narration_for": "ref1", "page": 0},
    )
    # Stamped transient, filed under the cache dir.
    row = await store.get(saved["id"], user_id=UID)
    assert row is not None                      # fetchable by id (player path)
    assert saved["path"].startswith("_transient/")
    # Hidden from listings.
    assert all(a["id"] != saved["id"] for a in await store.list_all(user_id=UID))
    # NOT registered in the file index.
    cur = await backend._conn.execute(
        "SELECT COUNT(*) FROM file_index WHERE source='artifacts' AND source_id=?",
        [saved["id"]],
    )
    assert (await cur.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_save_from_path_default_still_registers(backend, tmp_path):
    store = ArtifactStore(backend._conn)
    saved = await store.save_from_path(
        _wav(tmp_path, "book.wav"), "Book narration.wav", "wav",
        display_name="Book (narration)", user_id=UID,
    )
    # Visible in listings (real deliverable). File-index registration goes
    # through the VFS module's own connection, so it isn't asserted here —
    # the transient test asserts the SKIP against this conn, which is the
    # new behavior; the default path is unchanged from HEAD.
    assert any(a["id"] == saved["id"] for a in await store.list_all(user_id=UID))


@pytest.mark.asyncio
async def test_prune_drops_oldest_beyond_cap(backend, tmp_path):
    from types import SimpleNamespace

    from augmentum.jobs.handlers.comic_narration_synth import _prune_old_narrations

    astore = ArtifactStore(backend._conn)
    nstore = ComicNarrationStore(backend._conn)

    refs = []
    for i in range(4):
        ref = f"chap{i}"
        saved = await astore.save_from_path(
            _wav(tmp_path, f"c{i}.wav"), f"c{i}.wav", "wav",
            user_id=UID, transient=True,
            metadata={"comic_narration_for": ref, "page": 0},
        )
        row_id = await nstore.begin("file", ref, "voice", f"job{i}", user_id=UID)
        await nstore.mark_done(
            row_id, pages=[{"page": 0, "artifact_id": saved["id"], "lines": []}],
        )
        # Distinct updated_at ordering (mark_done stamps datetime('now');
        # force deterministic order explicitly).
        await backend._conn.execute(
            "UPDATE comic_narrations SET updated_at = ? WHERE id = ?",
            [f"2026-07-0{i + 1} 00:00:00", row_id],
        )
        refs.append((ref, saved["id"]))
    await backend._conn.commit()

    done = await nstore.list_done(user_id=UID)
    assert [r["comic_ref"] for r in done] == ["chap3", "chap2", "chap1", "chap0"]

    await _prune_old_narrations(
        nstore, astore, UID, SimpleNamespace(comic_narration_cache_max=2),
    )

    remaining = {r["comic_ref"] for r in await nstore.list_done(user_id=UID)}
    assert remaining == {"chap3", "chap2"}
    # Pruned chapters' page artifacts are gone; kept ones remain.
    assert await astore.get(refs[0][1], user_id=UID) is None
    assert await astore.get(refs[3][1], user_id=UID) is not None

    # cap=0 disables pruning.
    await _prune_old_narrations(
        nstore, astore, UID, SimpleNamespace(comic_narration_cache_max=0),
    )
    assert len(await nstore.list_done(user_id=UID)) == 2
