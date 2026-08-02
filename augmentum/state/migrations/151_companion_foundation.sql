-- 151_companion_foundation.sql
-- CompanionRuntime foundation: scope existing tables + create companion_identities.
--
-- Adds companion_id to memories and dream_* tables so the existing memory
-- and dream subsystems become companion-aware. Backfills 'becca' on every
-- existing row (single-companion phase). Creates the companion_identities
-- table and seeds Becca as the first companion. Sprint 1 Unit A migration 1
-- of 8 (CompanionRuntime initiative — see
-- docs/superpowers/specs/2026-05-14-companion-runtime-README.md).
--
-- companion_id is NULLABLE here to preserve backwards compatibility with
-- the pre-companion corpus. Sprint 7's migration 187 flips it to NOT NULL
-- once a second companion exists in production data.

-- ── Scoping: add companion_id to existing companion-bearing tables ─────
ALTER TABLE memories ADD COLUMN companion_id TEXT;
ALTER TABLE dream_entries ADD COLUMN companion_id TEXT;
ALTER TABLE dream_portraits ADD COLUMN companion_id TEXT;
ALTER TABLE dream_cycles ADD COLUMN companion_id TEXT;
ALTER TABLE dream_memory_log ADD COLUMN companion_id TEXT;

-- Backfill: every existing row belongs to 'becca' in the single-companion phase
UPDATE memories          SET companion_id = 'becca' WHERE companion_id IS NULL;
UPDATE dream_entries     SET companion_id = 'becca' WHERE companion_id IS NULL;
UPDATE dream_portraits   SET companion_id = 'becca' WHERE companion_id IS NULL;
UPDATE dream_cycles      SET companion_id = 'becca' WHERE companion_id IS NULL;
UPDATE dream_memory_log  SET companion_id = 'becca' WHERE companion_id IS NULL;

-- Companion-scoped indexes alongside the existing user_id/persona_id indexes
CREATE INDEX IF NOT EXISTS idx_memories_companion        ON memories(user_id, companion_id);
CREATE INDEX IF NOT EXISTS idx_dream_entries_companion   ON dream_entries(persona_id, companion_id);
CREATE INDEX IF NOT EXISTS idx_dream_portraits_companion ON dream_portraits(persona_id, companion_id);
CREATE INDEX IF NOT EXISTS idx_dream_cycles_companion    ON dream_cycles(persona_id, companion_id);

-- ── companion_identities: one row per companion ──────────────────────
-- persona_kernel_digest is the ~400-token compressed identity prefix that
-- gets threaded into every dispatch via DispatchContext.persona_kernel.
-- It starts empty; Sprint 1 Unit B (CompanionIdentity module) populates
-- it by digesting the canonical personality doc.
CREATE TABLE IF NOT EXISTS companion_identities (
    companion_id              TEXT PRIMARY KEY,
    display_name              TEXT NOT NULL,
    persona_kernel_digest     TEXT NOT NULL DEFAULT '',
    persona_kernel_embedding  BLOB,
    personality_doc_version   INTEGER NOT NULL DEFAULT 0,
    drift_score               REAL NOT NULL DEFAULT 0.0,
    created_at                TEXT NOT NULL DEFAULT (datetime('now')),
    last_kernel_refresh_at    TEXT
);

-- Seed Becca. Subsequent companions get instantiated in Sprint 7+.
INSERT OR IGNORE INTO companion_identities (companion_id, display_name)
VALUES ('becca', 'Becca');

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (151, 'companion_foundation: scope existing tables + companion_identities');
