-- 325_client_edit_stamps.sql
-- Client edit-stamps for the stale-write guard (augmentum/state/write_guard.py).
--
-- WHY A NEW COLUMN RATHER THAN REUSING updated_at
--
-- These tables already have `updated_at TEXT DEFAULT (datetime('now'))`.
-- That is the SERVER write time and it cannot detect staleness: it is
-- stamped by whichever write landed last, including the very write we are
-- trying to reject. Detecting "the stored copy contains an edit this client
-- has not seen" requires the stamp to be set by the CLIENT at edit time.
-- Two different clocks, so two different columns.
--
-- `ui_sessions` and `ui_characters` are not touched here — they store their
-- payload as a JSON blob and already carry (or will carry) the stamp at
-- `$.updatedAt` inside it, which write_guard reads via json_extract.
--
-- Milliseconds since epoch, INTEGER, DEFAULT 0. Zero means "no stamp" and
-- is treated as unguarded, so existing rows and older clients keep saving
-- normally until a client sends a real stamp. The rollout is therefore
-- non-breaking in both directions.
--
-- lorebook_entries and voice_mixes had NO updated_at column of any kind
-- before this migration, so they additionally get one for ordering/display
-- parity with their sibling tables.

ALTER TABLE prompt_presets   ADD COLUMN client_updated_at INTEGER NOT NULL DEFAULT 0;
ALTER TABLE regex_scripts    ADD COLUMN client_updated_at INTEGER NOT NULL DEFAULT 0;
ALTER TABLE custom_flows     ADD COLUMN client_updated_at INTEGER NOT NULL DEFAULT 0;
ALTER TABLE voice_mixes      ADD COLUMN client_updated_at INTEGER NOT NULL DEFAULT 0;
ALTER TABLE lorebook_entries ADD COLUMN client_updated_at INTEGER NOT NULL DEFAULT 0;

-- Server-side write time for the two tables that never had one.
ALTER TABLE voice_mixes      ADD COLUMN updated_at TEXT NOT NULL DEFAULT '';
ALTER TABLE lorebook_entries ADD COLUMN updated_at TEXT NOT NULL DEFAULT '';

-- Seed the new server column from created_at so ORDER BY updated_at does
-- something sensible for pre-existing rows instead of sorting them all
-- into one empty-string bucket.
UPDATE voice_mixes      SET updated_at = created_at WHERE updated_at = '';
UPDATE lorebook_entries SET updated_at = created_at WHERE updated_at = '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (325, 'Client edit-stamps for the shared stale-write guard');
