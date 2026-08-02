"""Tests for migration 179 — per-user companion pivot.

The migration:
1. Pivots companion_identities/state/scene PK from companion_id to
   (user_id, companion_id) — destructive table replacement
2. Adds user_id column to 5 append-only tables (state_log, init queue,
   creations, observations, skill_archive)
3. Creates companion_identities_genesis (immutable seed table)
4. Adds Aletheia identity fields (kernel_overlay, traits_derived_json,
   relationship_state_json)
5. Preserves existing data via owner_user_id backfill

These tests verify:
* Migration applies on a fresh :memory: DB without error
* Existing seed rows survive (Becca singleton from mig 151 preserved)
* New columns present with correct types and defaults
* Composite PK enforced (two users CAN have a 'becca' each)
* user_id backfill paths populated correctly for the 5 ALTER'd tables
* Genesis table exists and is empty until written
"""

from __future__ import annotations

import pytest


async def _boot_backend():
    """Spin up a :memory: backend with all migrations applied (179 included)."""
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    return backend


# ── Structural checks ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_migration_applies_clean():
    """Migration 179 reaches schema_version 179 without error."""
    backend = await _boot_backend()
    cur = await backend.conn.execute(
        "SELECT MAX(version) FROM schema_version"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row is not None
    assert row[0] >= 179


@pytest.mark.asyncio
async def test_identities_has_composite_pk():
    """companion_identities PRIMARY KEY is now (user_id, companion_id).

    Two users CAN have a row with companion_id='becca' each.
    """
    backend = await _boot_backend()

    # Two users + two seperate 'becca' identities
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('usr_a', 'a', 'x', datetime('now'))"
    )
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('usr_b', 'b', 'x', datetime('now'))"
    )
    await backend.conn.commit()

    await backend.conn.execute(
        "INSERT INTO companion_identities (user_id, companion_id, display_name) "
        "VALUES ('usr_a', 'becca', 'Becca-A')"
    )
    await backend.conn.execute(
        "INSERT INTO companion_identities (user_id, companion_id, display_name) "
        "VALUES ('usr_b', 'becca', 'Becca-B')"
    )
    await backend.conn.commit()

    # Filter to the two we inserted; the migration-seeded Becca
    # (with user_id='') is also in the table and would otherwise add
    # a third row.
    cur = await backend.conn.execute(
        "SELECT user_id, display_name FROM companion_identities "
        "WHERE companion_id = 'becca' AND user_id IN ('usr_a', 'usr_b') "
        "ORDER BY user_id"
    )
    rows = await cur.fetchall()
    await cur.close()
    assert len(rows) == 2
    assert rows[0][0] == "usr_a" and rows[0][1] == "Becca-A"
    assert rows[1][0] == "usr_b" and rows[1][1] == "Becca-B"


@pytest.mark.asyncio
async def test_identities_pk_prevents_same_user_dup():
    """Same user can't insert two rows for the same companion_id."""
    backend = await _boot_backend()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('usr_x', 'x', 'x', datetime('now'))"
    )
    await backend.conn.execute(
        "INSERT INTO companion_identities (user_id, companion_id, display_name) "
        "VALUES ('usr_x', 'becca', 'Becca-1')"
    )
    await backend.conn.commit()

    with pytest.raises(Exception):  # PK constraint
        await backend.conn.execute(
            "INSERT INTO companion_identities (user_id, companion_id, display_name) "
            "VALUES ('usr_x', 'becca', 'Becca-2')"
        )
        await backend.conn.commit()


@pytest.mark.asyncio
async def test_state_has_composite_pk():
    """companion_state can hold per-user rows for the same companion_id."""
    backend = await _boot_backend()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('usr_p', 'p', 'x', datetime('now'))"
    )
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('usr_q', 'q', 'x', datetime('now'))"
    )
    # Insert state for each user; one in 'present', other in 'dormant'
    await backend.conn.execute(
        "INSERT INTO companion_state (user_id, companion_id, state) "
        "VALUES ('usr_p', 'becca', 'present')"
    )
    await backend.conn.execute(
        "INSERT INTO companion_state (user_id, companion_id, state) "
        "VALUES ('usr_q', 'becca', 'dormant')"
    )
    await backend.conn.commit()

    cur = await backend.conn.execute(
        "SELECT state FROM companion_state WHERE user_id = 'usr_p' AND companion_id = 'becca'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == "present"

    cur = await backend.conn.execute(
        "SELECT state FROM companion_state WHERE user_id = 'usr_q' AND companion_id = 'becca'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == "dormant"


@pytest.mark.asyncio
async def test_scene_has_composite_pk():
    """companion_scene supports per-user rows for the same companion_id."""
    backend = await _boot_backend()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('usr_one', 'one', 'x', datetime('now'))"
    )
    await backend.conn.execute(
        "INSERT INTO companion_scene (user_id, companion_id, location) "
        "VALUES ('usr_one', 'becca', 'main_room')"
    )
    await backend.conn.commit()

    cur = await backend.conn.execute(
        "SELECT location FROM companion_scene "
        "WHERE user_id = 'usr_one' AND companion_id = 'becca'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == "main_room"


# ── New identity fields ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_identity_has_kernel_overlay_field():
    """kernel_overlay column exists with empty-string default."""
    backend = await _boot_backend()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('usr_v', 'v', 'x', datetime('now'))"
    )
    await backend.conn.execute(
        "INSERT INTO companion_identities (user_id, companion_id, display_name) "
        "VALUES ('usr_v', 'becca', 'Becca')"
    )
    await backend.conn.commit()

    cur = await backend.conn.execute(
        "SELECT kernel_overlay, traits_derived_json, relationship_state_json "
        "FROM companion_identities WHERE user_id = 'usr_v'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == ""
    assert row[1] == "{}"
    assert row[2] == "{}"


@pytest.mark.asyncio
async def test_identity_overlay_writable():
    """kernel_overlay accepts free text up to whatever SQLite allows."""
    backend = await _boot_backend()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('usr_w', 'w', 'x', datetime('now'))"
    )
    await backend.conn.execute(
        "INSERT INTO companion_identities (user_id, companion_id, display_name, kernel_overlay) "
        "VALUES ('usr_w', 'becca', 'Becca', 'playfulness: +0.03')"
    )
    await backend.conn.commit()

    cur = await backend.conn.execute(
        "SELECT kernel_overlay FROM companion_identities WHERE user_id = 'usr_w'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert "playfulness" in row[0]


# ── Genesis table ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_genesis_table_exists_and_empty():
    """companion_identities_genesis exists and starts empty."""
    backend = await _boot_backend()
    cur = await backend.conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type = 'table' AND name = 'companion_identities_genesis'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row is not None

    cur = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_identities_genesis"
    )
    row = await cur.fetchone()
    await cur.close()
    # Migration doesn't seed genesis; lazy_provision will write it.
    assert row[0] == 0


@pytest.mark.asyncio
async def test_genesis_table_accepts_seed_rows():
    """Genesis is writable with the expected shape."""
    backend = await _boot_backend()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('usr_g', 'g', 'x', datetime('now'))"
    )
    await backend.conn.execute(
        "INSERT INTO companion_identities_genesis "
        "(user_id, companion_id, seed_kernel_digest, seed_personality_doc_version) "
        "VALUES ('usr_g', 'becca', 'kernel-digest-text', 1)"
    )
    await backend.conn.commit()

    cur = await backend.conn.execute(
        "SELECT seed_kernel_digest, seed_personality_doc_version "
        "FROM companion_identities_genesis WHERE user_id = 'usr_g'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == "kernel-digest-text"
    assert row[1] == 1


# ── user_id columns on append-only tables ────────────────────────────


@pytest.mark.asyncio
async def test_append_only_tables_have_user_id():
    """All 5 append-only tables gained a user_id column."""
    backend = await _boot_backend()
    for table in (
        "companion_state_log",
        "companion_initiative_queue",
        "companion_creations",
        "companion_observations",
        "companion_skill_archive",
    ):
        # PRAGMA table_info returns rows: cid, name, type, notnull, dflt_value, pk
        cur = await backend.conn.execute(f"PRAGMA table_info({table})")
        cols = await cur.fetchall()
        await cur.close()
        col_names = [c[1] for c in cols]
        assert "user_id" in col_names, f"{table} missing user_id column"


@pytest.mark.asyncio
async def test_initiative_queue_user_scoped_isolation():
    """Per-user initiative queues stay isolated by user_id."""
    backend = await _boot_backend()
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('usr_a', 'a', 'x', datetime('now'))"
    )
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES ('usr_b', 'b', 'x', datetime('now'))"
    )
    # Insert proposal for each user
    await backend.conn.execute(
        "INSERT INTO companion_initiative_queue "
        "(companion_id, user_id, proposed_at, kind, payload, score, status) "
        "VALUES ('becca', 'usr_a', 1000.0, 'revisit_thread', '{}', 0.7, 'pending')"
    )
    await backend.conn.execute(
        "INSERT INTO companion_initiative_queue "
        "(companion_id, user_id, proposed_at, kind, payload, score, status) "
        "VALUES ('becca', 'usr_b', 1001.0, 'reach_out_after_quiet', '{}', 0.5, 'pending')"
    )
    await backend.conn.commit()

    # Each user sees only their own proposal
    cur = await backend.conn.execute(
        "SELECT kind FROM companion_initiative_queue WHERE user_id = 'usr_a'"
    )
    rows = await cur.fetchall()
    await cur.close()
    assert len(rows) == 1
    assert rows[0][0] == "revisit_thread"

    cur = await backend.conn.execute(
        "SELECT kind FROM companion_initiative_queue WHERE user_id = 'usr_b'"
    )
    rows = await cur.fetchall()
    await cur.close()
    assert len(rows) == 1
    assert rows[0][0] == "reach_out_after_quiet"


@pytest.mark.asyncio
async def test_skill_archive_user_scoped_isolation():
    """Skill archive entries are scoped to user_id."""
    backend = await _boot_backend()
    await backend.conn.execute(
        "INSERT INTO companion_skill_archive "
        "(companion_id, ts, intent_text, chosen_subagent, user_id) "
        "VALUES ('becca', 100.0, 'hello', 'passthrough', 'usr_a')"
    )
    await backend.conn.execute(
        "INSERT INTO companion_skill_archive "
        "(companion_id, ts, intent_text, chosen_subagent, user_id) "
        "VALUES ('becca', 200.0, 'world', 'narrative', 'usr_b')"
    )
    await backend.conn.commit()

    cur = await backend.conn.execute(
        "SELECT intent_text FROM companion_skill_archive WHERE user_id = 'usr_a'"
    )
    rows = await cur.fetchall()
    await cur.close()
    assert len(rows) == 1
    assert rows[0][0] == "hello"


# ── Backfill behavior ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_existing_becca_seed_survives():
    """Migration 151 seeded Becca with companion_id='becca'. After
    migration 179, that row exists with user_id='' (no owner resolved
    in a fresh test DB)."""
    backend = await _boot_backend()
    cur = await backend.conn.execute(
        "SELECT user_id, display_name FROM companion_identities WHERE companion_id = 'becca'"
    )
    rows = await cur.fetchall()
    await cur.close()
    # Should have exactly one row, the seeded Becca with empty user_id
    assert len(rows) == 1
    assert rows[0][0] == ""  # backfill default when owner_user_id was NULL
    assert rows[0][1] == "Becca"


@pytest.mark.asyncio
async def test_existing_state_survives_pivot():
    """Migration 152 seeded companion_state for 'becca'. After pivot,
    that row exists with the same defaults under user_id=''."""
    backend = await _boot_backend()
    cur = await backend.conn.execute(
        "SELECT state, role_passive FROM companion_state "
        "WHERE companion_id = 'becca' AND user_id = ''"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row is not None
    assert row[0] == "dormant"
    assert row[1] == 1.0


@pytest.mark.asyncio
async def test_existing_scene_survives_pivot():
    """Migration 157 seeded companion_scene for 'becca'. After pivot,
    that row exists with defaults."""
    backend = await _boot_backend()
    cur = await backend.conn.execute(
        "SELECT location, posture FROM companion_scene "
        "WHERE companion_id = 'becca' AND user_id = ''"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row is not None
    assert row[0] == "main_room"
    assert row[1] == "idle"


# ── Index verification ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_user_scoped_indexes_exist():
    """The new user-scoped indexes should be present."""
    backend = await _boot_backend()
    expected_indexes = [
        "idx_initiative_queue_user_time",
        "idx_initiative_queue_user_kind_status",
        "idx_skill_archive_user_time",
        "idx_skill_archive_user_subagent",
        "idx_creations_user_time",
        "idx_obs_user_time",
        "idx_cstate_log_user_ts",
        "idx_companion_identities_genesis_pair",
    ]
    cur = await backend.conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'index'"
    )
    rows = await cur.fetchall()
    await cur.close()
    names = {r[0] for r in rows}
    for idx in expected_indexes:
        assert idx in names, f"missing index: {idx}"
