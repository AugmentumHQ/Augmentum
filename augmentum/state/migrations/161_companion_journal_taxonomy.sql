-- 161_companion_journal_taxonomy.sql
-- Becca runtime, Lane 2 §2.1 — extends companion_journal (migration 154)
-- with the columns the noticing pipeline needs to age, repeat, graduate,
-- and suppress entries without rewriting them.
--
-- The base table from migration 154 declared entry_type as an open string
-- with four canonical kinds: observation, wondering, noticing, unfinished.
-- This migration extends the canonical set with three more (reflection,
-- creation_note, correction) but does NOT enforce a CHECK constraint —
-- per-codebase convention, vocabulary is runtime-enforced so the schema
-- doesn't have to be rewritten every time the vocabulary grows.
--
-- New columns:
--   confidence         'early' | 'normal' | 'firm' — gates graduation
--   repetition_count   bumped when consolidation finds a near-duplicate
--   suppressed         user-flagged or self-corrected — never re-surface
--   suppressed_reason  'user_requested' | 'self_correction' | 'staleness' | 'rebuild'
--   graduated_at       when this noticing graduated to the relationship doc
--
-- All ALTER TABLE adds are nullable / defaulted so existing rows continue
-- to read cleanly. The new partial index supports the hot-path query
-- "active noticings worth surfacing" without scanning the whole journal.

ALTER TABLE companion_journal ADD COLUMN confidence        TEXT NOT NULL DEFAULT 'normal';
ALTER TABLE companion_journal ADD COLUMN repetition_count  INTEGER NOT NULL DEFAULT 1;
ALTER TABLE companion_journal ADD COLUMN suppressed        INTEGER NOT NULL DEFAULT 0;
ALTER TABLE companion_journal ADD COLUMN suppressed_reason TEXT;
ALTER TABLE companion_journal ADD COLUMN graduated_at      TEXT;

CREATE INDEX IF NOT EXISTS idx_cj_noticing_active
    ON companion_journal(companion_id, user_id, entry_type)
    WHERE entry_type = 'noticing' AND suppressed = 0 AND graduated_at IS NULL;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (161, 'companion_journal taxonomy: confidence, repetition_count, suppression, graduation');
