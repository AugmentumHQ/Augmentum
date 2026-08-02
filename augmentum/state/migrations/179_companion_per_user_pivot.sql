-- 179_companion_per_user_pivot.sql
-- Aletheia × Augmentum arc, Piece 1.
--
-- Pivots the 8 server-level companion tables to user-scoped so each
-- user gets their own isolated companion silo. Per-user is the load-
-- bearing invariant of the design doc (2026-05-19-aletheia-augmentum-
-- design.md §3) — a coworker who installs Augmentum and creates an
-- account should get a fresh Becca with no data creep from any other
-- user.
--
-- Also: extends companion_identities with the Aletheia identity fields
-- (kernel_overlay, traits_derived_json, relationship_state_json) and
-- creates companion_identities_genesis as the immutable seed snapshot
-- used by reset gestures.
--
-- ── PK pivot mechanics + FK safety ────────────────────────────────────
--
-- SQLite cannot ALTER a PRIMARY KEY directly. Standard pattern: create
-- new table, copy data, drop old, rename new. BUT — PRAGMA
-- foreign_keys=ON (sqlite.py:280) means dropping a parent table while
-- a child still references it FAILS with a constraint error.
--
-- Safe ordering used here:
--   1. CREATE all 3 new tables (identities, state, scene)
--   2. INSERT data into all 3 new tables, backfilling user_id
--   3. DROP child tables first (companion_state, companion_scene) —
--      this removes their FK constraints
--   4. DROP companion_identities — now no FK pressure
--   5. ALTER ... RENAME for all 3 to their canonical names
--   6. CREATE all indexes on the renamed tables
--
-- The new tables intentionally drop FK constraints back to identities.
-- Application code (CompanionRuntime + lazy_provision) is the single
-- owner of (user_id, companion_id) lifecycle; loose FK constraints
-- across the pivot would block legitimate cross-user reads. The
-- composite PRIMARY KEY on each table is what enforces per-user
-- uniqueness; FK enforcement at this scope adds no real safety.
--
-- ── Backfill strategy ─────────────────────────────────────────────────
--
-- Existing rows attribute to the owner_user_id from mig 173. If
-- owner_user_id is NULL (pre-mig 173 install, never resolved), backfill
-- with empty string '' — runtime's lazy_provision() treats user_id=''
-- as "needs provisioning" and creates a fresh row on first interaction.
-- No data loss; just a one-time tag.
--
-- ── What gets pivoted ─────────────────────────────────────────────────
--
--   companion_identities       PK companion_id → (user_id, companion_id)
--                              + new identity columns + genesis seed
--   companion_state            PK companion_id → (user_id, companion_id)
--   companion_scene            PK companion_id → (user_id, companion_id)
--   companion_state_log        adds user_id column + index
--   companion_initiative_queue adds user_id column + index
--   companion_creations        adds user_id column + index
--   companion_observations     adds user_id (owner) column + index
--   companion_skill_archive    adds user_id column + index
--
-- NOT pivoted (intentional):
--
--   companion_safety_floor_audit — designed anonymized via HMAC
--                                   fingerprints (mig 162); adding
--                                   user_id would defeat the privacy
--                                   posture.

-- ────────────────────────────────────────────────────────────────────
-- Phase A — Create all new tables (identities, state, scene)
-- ────────────────────────────────────────────────────────────────────

CREATE TABLE companion_identities_new (
    user_id                    TEXT NOT NULL,
    companion_id               TEXT NOT NULL,
    display_name               TEXT NOT NULL,
    persona_kernel_digest      TEXT NOT NULL DEFAULT '',
    persona_kernel_embedding   BLOB,
    personality_doc_version    INTEGER NOT NULL DEFAULT 0,
    drift_score                REAL NOT NULL DEFAULT 0.0,
    created_at                 TEXT NOT NULL DEFAULT (datetime('now')),
    last_kernel_refresh_at     TEXT,
    owner_user_id              TEXT,                            -- kept for compat with mig 173 callers
    -- New Aletheia fields (anchor doc §5 Piece 2):
    kernel_overlay             TEXT NOT NULL DEFAULT '',        -- accumulated trait nudges; capped ±0.05 per trait
    traits_derived_json        TEXT NOT NULL DEFAULT '{}',      -- {trait_name: float} projected from facets+overlay
    relationship_state_json    TEXT NOT NULL DEFAULT '{}',      -- {trust_level, known_rhythms[], nicknames_earned[], ...}
    PRIMARY KEY (user_id, companion_id)
);

CREATE TABLE companion_state_new (
    user_id           TEXT NOT NULL,
    companion_id      TEXT NOT NULL,
    state             TEXT NOT NULL DEFAULT 'dormant',
    role_active       REAL NOT NULL DEFAULT 0.0,
    role_passive      REAL NOT NULL DEFAULT 1.0,
    role_reflective   REAL NOT NULL DEFAULT 0.0,
    focus             TEXT NOT NULL DEFAULT 'none',
    entered_state_at  TEXT NOT NULL DEFAULT (datetime('now')),
    entered_role_at   TEXT NOT NULL DEFAULT (datetime('now')),
    entered_focus_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, companion_id)
);

CREATE TABLE companion_scene_new (
    user_id           TEXT NOT NULL,
    companion_id      TEXT NOT NULL,
    location          TEXT NOT NULL DEFAULT 'main_room',
    posture           TEXT NOT NULL DEFAULT 'idle',
    scene_blob        TEXT NOT NULL DEFAULT '{}',
    last_seen_with    TEXT,
    last_changed_at   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, companion_id)
);

-- Genesis snapshot — stores the immutable seed each user starts from.
-- Reset gestures (recovery.py soft/hard_reset) restore from this when
-- present. Append-only after first provision.
CREATE TABLE IF NOT EXISTS companion_identities_genesis (
    user_id                      TEXT NOT NULL,
    companion_id                 TEXT NOT NULL,
    seed_kernel_digest           TEXT NOT NULL,
    seed_kernel_embedding        BLOB,
    seed_personality_doc_version INTEGER NOT NULL DEFAULT 0,
    seeded_at                    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, companion_id, seeded_at)
);

-- ────────────────────────────────────────────────────────────────────
-- Phase B — Copy data into the new tables, backfilling user_id
-- ────────────────────────────────────────────────────────────────────

INSERT INTO companion_identities_new (
    user_id, companion_id, display_name, persona_kernel_digest,
    persona_kernel_embedding, personality_doc_version, drift_score,
    created_at, last_kernel_refresh_at, owner_user_id
)
SELECT
    COALESCE(owner_user_id, ''),
    companion_id, display_name, persona_kernel_digest,
    persona_kernel_embedding, personality_doc_version, drift_score,
    created_at, last_kernel_refresh_at, owner_user_id
FROM companion_identities;

INSERT INTO companion_state_new (
    user_id, companion_id, state,
    role_active, role_passive, role_reflective,
    focus, entered_state_at, entered_role_at, entered_focus_at, updated_at
)
SELECT
    COALESCE((SELECT owner_user_id FROM companion_identities ci
              WHERE ci.companion_id = cs.companion_id LIMIT 1), ''),
    cs.companion_id, cs.state,
    cs.role_active, cs.role_passive, cs.role_reflective,
    cs.focus, cs.entered_state_at, cs.entered_role_at, cs.entered_focus_at, cs.updated_at
FROM companion_state cs;

INSERT INTO companion_scene_new (
    user_id, companion_id, location, posture, scene_blob, last_seen_with, last_changed_at
)
SELECT
    COALESCE((SELECT owner_user_id FROM companion_identities ci
              WHERE ci.companion_id = sc.companion_id LIMIT 1), ''),
    sc.companion_id, sc.location, sc.posture, sc.scene_blob, sc.last_seen_with, sc.last_changed_at
FROM companion_scene sc;

-- ────────────────────────────────────────────────────────────────────
-- Phase C — Drop children first (removes FK pressure), then parent
-- ────────────────────────────────────────────────────────────────────

DROP TABLE companion_state;
DROP TABLE companion_scene;
DROP TABLE companion_identities;

-- ────────────────────────────────────────────────────────────────────
-- Phase D — Rename new tables to canonical names
-- ────────────────────────────────────────────────────────────────────

ALTER TABLE companion_identities_new RENAME TO companion_identities;
ALTER TABLE companion_state_new      RENAME TO companion_state;
ALTER TABLE companion_scene_new      RENAME TO companion_scene;

-- ────────────────────────────────────────────────────────────────────
-- Phase E — Indexes on the renamed tables
-- ────────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_companion_identities_companion_id
    ON companion_identities(companion_id);
CREATE INDEX IF NOT EXISTS idx_companion_identities_owner
    ON companion_identities(owner_user_id) WHERE owner_user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_companion_identities_genesis_pair
    ON companion_identities_genesis(user_id, companion_id, seeded_at DESC);

-- ────────────────────────────────────────────────────────────────────
-- Phase F — Add user_id column to the 5 append-only tables
-- ────────────────────────────────────────────────────────────────────

ALTER TABLE companion_state_log        ADD COLUMN user_id TEXT NOT NULL DEFAULT '';
ALTER TABLE companion_initiative_queue ADD COLUMN user_id TEXT NOT NULL DEFAULT '';
ALTER TABLE companion_creations        ADD COLUMN user_id TEXT NOT NULL DEFAULT '';
ALTER TABLE companion_observations     ADD COLUMN user_id TEXT NOT NULL DEFAULT '';
ALTER TABLE companion_skill_archive    ADD COLUMN user_id TEXT NOT NULL DEFAULT '';

-- Backfill the user_id columns from companion_identities.owner_user_id.
-- For rows whose companion no longer exists (shouldn't happen post-
-- pivot, but defensive) the COALESCE leaves user_id=''.

UPDATE companion_state_log
SET user_id = COALESCE(
    (SELECT owner_user_id FROM companion_identities ci
     WHERE ci.companion_id = companion_state_log.companion_id LIMIT 1),
    ''
)
WHERE user_id = '';

UPDATE companion_initiative_queue
SET user_id = COALESCE(
    (SELECT owner_user_id FROM companion_identities ci
     WHERE ci.companion_id = companion_initiative_queue.companion_id LIMIT 1),
    ''
)
WHERE user_id = '';

UPDATE companion_creations
SET user_id = COALESCE(
    (SELECT owner_user_id FROM companion_identities ci
     WHERE ci.companion_id = companion_creations.companion_id LIMIT 1),
    ''
)
WHERE user_id = '';

UPDATE companion_observations
SET user_id = COALESCE(
    (SELECT owner_user_id FROM companion_identities ci
     WHERE ci.companion_id = companion_observations.companion_id LIMIT 1),
    ''
)
WHERE user_id = '';

UPDATE companion_skill_archive
SET user_id = COALESCE(
    (SELECT owner_user_id FROM companion_identities ci
     WHERE ci.companion_id = companion_skill_archive.companion_id LIMIT 1),
    ''
)
WHERE user_id = '';

-- Indexes for user-scoped reads.

CREATE INDEX IF NOT EXISTS idx_cstate_log_user_ts
    ON companion_state_log(user_id, companion_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_initiative_queue_user_time
    ON companion_initiative_queue(user_id, companion_id, proposed_at DESC);
CREATE INDEX IF NOT EXISTS idx_initiative_queue_user_kind_status
    ON companion_initiative_queue(user_id, companion_id, kind, status, proposed_at DESC);

CREATE INDEX IF NOT EXISTS idx_creations_user_time
    ON companion_creations(user_id, companion_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_obs_user_time
    ON companion_observations(user_id, companion_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_skill_archive_user_time
    ON companion_skill_archive(user_id, companion_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_skill_archive_user_subagent
    ON companion_skill_archive(user_id, companion_id, chosen_subagent, ts DESC);

-- ────────────────────────────────────────────────────────────────────
-- Schema version marker
-- ────────────────────────────────────────────────────────────────────

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (179, 'companion per-user pivot: 8 tables scoped to (user_id, companion_id) + identity extensions + genesis seed');
