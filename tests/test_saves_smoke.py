"""Smoke tests for the AXF Save service (Phase C1).

Covers SaveStore CRUD, slot upserts (overwrite releases old blob),
size cap enforcement, and cascade delete-all-for-title.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest


_SCHEMA_SQL = """
CREATE TABLE users (id TEXT PRIMARY KEY);
CREATE TABLE game_saves (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id),
    artifact_id     TEXT NOT NULL,
    core_id         TEXT NOT NULL DEFAULT '',
    kind            TEXT NOT NULL,
    slot            INTEGER NOT NULL DEFAULT 0,
    sha256          TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL,
    label           TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, artifact_id, kind, slot)
);
"""


class _FakeBlobStore:
    """In-memory blob store standin -- mirrors BlobStore's relevant API.

    write/get use sha256 keys; release decrements; we keep bytes in a
    dict so get_with_bytes can read 'real_path' that points at /tmp via
    a stable per-sha file. Tests focus on store semantics, not disk
    layout, so we monkey-path the read helper at the SaveStore layer.
    """
    def __init__(self):
        import hashlib, tempfile, os
        self._data: dict[str, bytes] = {}
        self._refs: dict[str, int] = {}
        self._dir = tempfile.mkdtemp()
        self._hashlib = hashlib
        self._os = os

    async def write(self, data: bytes, *, mime_type: str = ""):
        sha = self._hashlib.sha256(data).hexdigest()
        path = self._os.path.join(self._dir, sha)
        with open(path, "wb") as f:
            f.write(data)
        self._data[sha] = data
        self._refs[sha] = self._refs.get(sha, 0) + 1
        return {
            "sha256": sha,
            "size_bytes": len(data),
            "mime_type": mime_type,
            "real_path": path,
            "refcount": self._refs[sha],
        }

    async def get(self, sha: str):
        if sha not in self._data:
            return None
        return {
            "sha256": sha,
            "size_bytes": len(self._data[sha]),
            "mime_type": "",
            "real_path": self._os.path.join(self._dir, sha),
            "refcount": self._refs.get(sha, 0),
        }

    async def release(self, sha: str) -> bool:
        if sha not in self._refs or self._refs[sha] <= 0:
            return False
        self._refs[sha] -= 1
        if self._refs[sha] == 0:
            self._data.pop(sha, None)
            try:
                self._os.remove(self._os.path.join(self._dir, sha))
            except OSError:
                pass
        return True


async def _mkstore():
    from augmentum.saves import SaveStore

    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_SQL)
    await conn.execute("INSERT INTO users (id) VALUES ('u1')")
    await conn.execute("INSERT INTO users (id) VALUES ('u2')")
    await conn.commit()
    blobs = _FakeBlobStore()
    return SaveStore(conn, blobs), conn, blobs


# ── Public surface ─────────────────────────────────────────────────


def test_imports():
    from augmentum.saves import (  # noqa: F401
        SAVE_KINDS,
        SaveKind,
        SaveRecord,
        SaveServiceError,
        SaveStore,
        SaveTooLargeError,
    )


# ── Round trip ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_put_and_get_roundtrip():
    store, _, _ = await _mkstore()
    rec = await store.put(
        user_id="u1", artifact_id="t_1",
        kind="sram", slot=0, data=b"hello world",
    )
    assert rec.kind == "sram"
    assert rec.slot == 0
    assert rec.size_bytes == 11
    pair = await store.get_with_bytes(
        user_id="u1", artifact_id="t_1", kind="sram", slot=0,
    )
    assert pair is not None
    record, data = pair
    assert data == b"hello world"
    assert record.id == rec.id


@pytest.mark.asyncio
async def test_put_overwrite_releases_old_blob():
    """Writing the same slot with different bytes must release the old
    blob's refcount so storage doesn't accumulate stale saves."""
    store, _, blobs = await _mkstore()
    a = await store.put(
        user_id="u1", artifact_id="t_1",
        kind="state", slot=1, data=b"old", core_id="fceumm",
    )
    assert blobs._refs[a.sha256] == 1
    b = await store.put(
        user_id="u1", artifact_id="t_1",
        kind="state", slot=1, data=b"new", core_id="fceumm",
    )
    # Old blob refcount went to 0 (released + cleaned)
    assert a.sha256 not in blobs._refs or blobs._refs[a.sha256] == 0
    # New blob is at 1
    assert blobs._refs[b.sha256] == 1


@pytest.mark.asyncio
async def test_put_same_bytes_does_not_double_release():
    """Re-PUTing identical bytes shouldn't underflow the blob refcount.
    The blob store dedup-bumps then we don't release because old==new."""
    store, _, blobs = await _mkstore()
    a = await store.put(
        user_id="u1", artifact_id="t_1",
        kind="sram", slot=0, data=b"same",
    )
    sha = a.sha256
    await store.put(
        user_id="u1", artifact_id="t_1",
        kind="sram", slot=0, data=b"same",
    )
    # Refcount should be exactly 1 (one row, one reference)
    assert blobs._refs[sha] == 1


# ── Validation ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_put_rejects_unknown_kind():
    from augmentum.saves import SaveServiceError

    store, _, _ = await _mkstore()
    with pytest.raises(SaveServiceError):
        await store.put(
            user_id="u1", artifact_id="t_1",
            kind="bogus", slot=0, data=b"x",
        )


@pytest.mark.asyncio
async def test_put_rejects_empty_data():
    from augmentum.saves import SaveServiceError

    store, _, _ = await _mkstore()
    with pytest.raises(SaveServiceError):
        await store.put(
            user_id="u1", artifact_id="t_1",
            kind="sram", slot=0, data=b"",
        )


@pytest.mark.asyncio
async def test_put_enforces_size_cap():
    from augmentum.saves import SaveTooLargeError

    store, _, _ = await _mkstore()
    with pytest.raises(SaveTooLargeError):
        await store.put(
            user_id="u1", artifact_id="t_1",
            kind="sram", slot=0,
            data=b"x" * 100,
            max_per_slot_bytes=10,
        )


# ── Listing ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_for_title_user_scoped():
    store, _, _ = await _mkstore()
    await store.put(user_id="u1", artifact_id="t_1",
                    kind="sram", slot=0, data=b"u1-sram")
    await store.put(user_id="u2", artifact_id="t_1",
                    kind="sram", slot=0, data=b"u2-sram")
    own = await store.list_for_title(user_id="u1", artifact_id="t_1")
    assert len(own) == 1
    assert own[0].user_id == "u1"


@pytest.mark.asyncio
async def test_list_for_title_kind_filter():
    store, _, _ = await _mkstore()
    await store.put(user_id="u1", artifact_id="t_1",
                    kind="sram", slot=0, data=b"sram")
    await store.put(user_id="u1", artifact_id="t_1",
                    kind="state", slot=1, data=b"state",
                    core_id="fceumm")
    only_sram = await store.list_for_title(
        user_id="u1", artifact_id="t_1", kind="sram",
    )
    assert len(only_sram) == 1 and only_sram[0].kind == "sram"


# ── Delete ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_releases_blob():
    store, _, blobs = await _mkstore()
    rec = await store.put(
        user_id="u1", artifact_id="t_1",
        kind="sram", slot=0, data=b"bye",
    )
    assert blobs._refs[rec.sha256] == 1
    ok = await store.delete(
        user_id="u1", artifact_id="t_1", kind="sram", slot=0,
    )
    assert ok
    assert blobs._refs.get(rec.sha256, 0) == 0
    # Subsequent delete returns False
    assert await store.delete(
        user_id="u1", artifact_id="t_1", kind="sram", slot=0,
    ) is False


@pytest.mark.asyncio
async def test_delete_all_for_title_cascades():
    store, _, blobs = await _mkstore()
    a = await store.put(user_id="u1", artifact_id="t_1",
                        kind="sram", slot=0, data=b"a")
    b = await store.put(user_id="u1", artifact_id="t_1",
                        kind="state", slot=1, data=b"b",
                        core_id="fceumm")
    other = await store.put(user_id="u1", artifact_id="t_2",
                            kind="sram", slot=0, data=b"c")
    n = await store.delete_all_for_title(
        user_id="u1", artifact_id="t_1",
    )
    assert n == 2
    # t_1 saves gone; t_2 untouched
    remaining = await store.list_for_title(user_id="u1", artifact_id="t_2")
    assert len(remaining) == 1 and remaining[0].id == other.id
    # blobs released
    assert blobs._refs.get(a.sha256, 0) == 0
    assert blobs._refs.get(b.sha256, 0) == 0
    assert blobs._refs.get(other.sha256, 0) == 1


# ── Aggregates ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_total_bytes_for_user():
    store, _, _ = await _mkstore()
    await store.put(user_id="u1", artifact_id="t_1",
                    kind="sram", slot=0, data=b"a" * 1024)
    await store.put(user_id="u1", artifact_id="t_2",
                    kind="state", slot=1, data=b"b" * 2048,
                    core_id="fceumm")
    await store.put(user_id="u2", artifact_id="t_3",
                    kind="sram", slot=0, data=b"c" * 4096)
    assert await store.total_bytes_for_user(user_id="u1") == 3072
    assert await store.total_bytes_for_user(user_id="u2") == 4096
