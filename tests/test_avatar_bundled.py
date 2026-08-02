"""Tests for bundled avatar manifest and seeding."""
from __future__ import annotations

import aiosqlite
import pytest

from augmentum.avatar.bundled import BUNDLED_AVATARS, VALID_MANNERISM_KEYS, seed_bundled_avatars
from augmentum.avatar.store import AvatarStore


@pytest.fixture
async def store(tmp_path):
    db_path = tmp_path / "test.db"
    async with aiosqlite.connect(str(db_path)) as conn:
        await conn.executescript(open("augmentum/state/migrations/060_avatars.sql").read())
        for mig in ("061_avatar_type.sql",):
            try:
                await conn.executescript(open(f"augmentum/state/migrations/{mig}").read())
            except Exception:
                pass  # column may already exist
        await conn.commit()
        yield AvatarStore(conn)


# ── Manifest tests ─────────────────────────────────────────────────────────────

def test_bundled_manifest_has_project_owned_avatars():
    assert [avatar["vrm_filename"] for avatar in BUNDLED_AVATARS] == [
        "vance.vrm",
        "Becca.vrm",
        "Lise.vrm",
        "Danny.vrm",
        "Roxanne.vrm",
    ]


def test_bundled_avatars_have_required_fields():
    required = {"id", "name", "vrm_filename", "mannerisms"}
    for avatar in BUNDLED_AVATARS:
        missing = required - avatar.keys()
        assert not missing, f"Avatar {avatar.get('id')!r} missing fields: {missing}"


def test_bundled_mannerisms_have_valid_keys():
    for avatar in BUNDLED_AVATARS:
        unknown = set(avatar["mannerisms"].keys()) - VALID_MANNERISM_KEYS
        assert not unknown, (
            f"Avatar {avatar['id']!r} has unknown mannerism keys: {unknown}"
        )
        # All required keys must be present
        missing = VALID_MANNERISM_KEYS - avatar["mannerisms"].keys()
        assert not missing, (
            f"Avatar {avatar['id']!r} missing mannerism keys: {missing}"
        )


# ── Seeding tests ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_seed_creates_records(store, tmp_path):
    bundled_dir = tmp_path / "bundled-avatars"
    bundled_dir.mkdir()
    for avatar in BUNDLED_AVATARS:
        (bundled_dir / avatar["vrm_filename"]).write_bytes(b"")

    seeded = await seed_bundled_avatars(store, str(bundled_dir))
    assert seeded == 2

    records = await store.list_bundled()
    assert len(records) == 2
    for rec in records:
        assert rec["is_bundled"] == 1
        assert rec["type"] == "vrm"


@pytest.mark.asyncio
async def test_seed_is_idempotent(store, tmp_path):
    bundled_dir = tmp_path / "bundled-avatars"
    bundled_dir.mkdir()
    for avatar in BUNDLED_AVATARS:
        (bundled_dir / avatar["vrm_filename"]).write_bytes(b"")

    first = await seed_bundled_avatars(store, str(bundled_dir))
    assert first == 2

    second = await seed_bundled_avatars(store, str(bundled_dir))
    assert second == 0

    records = await store.list_bundled()
    assert len(records) == 2


@pytest.mark.asyncio
async def test_seed_prunes_removed_bundled_records(store, tmp_path):
    bundled_dir = tmp_path / "bundled-avatars"
    bundled_dir.mkdir()
    for avatar in BUNDLED_AVATARS:
        (bundled_dir / avatar["vrm_filename"]).write_bytes(b"")

    await store._conn.execute(
        """INSERT INTO avatars
           (id, vrm_path, mannerisms, is_bundled, type, created_at, updated_at)
           VALUES (?, ?, '{}', 1, 'vrm', datetime('now'), datetime('now'))""",
        ("bundled_f_gentle", str(bundled_dir / "gentle_f.vrm")),
    )
    await store._conn.commit()

    seeded = await seed_bundled_avatars(store, str(bundled_dir))
    assert seeded == 5

    records = await store.list_bundled()
    assert {rec["id"] for rec in records} == {
        "bundled_m_vance", "bundled_f_becca",
        "bundled_f_lise", "bundled_m_danny", "bundled_f_roxanne",
    }


@pytest.mark.asyncio
async def test_seed_skips_missing_files(store, tmp_path):
    bundled_dir = tmp_path / "bundled-avatars"
    bundled_dir.mkdir()
    # Only create files for the first 2 avatars
    for avatar in BUNDLED_AVATARS[:2]:
        (bundled_dir / avatar["vrm_filename"]).write_bytes(b"")

    seeded = await seed_bundled_avatars(store, str(bundled_dir))
    assert seeded == 2

    records = await store.list_bundled()
    assert len(records) == 2
