-- 304_lorebook_entries_branch_id.sql
-- Branch-awareness for model-authored lorebook entries (F5 — Lorebook
-- Authoring from Narrative).
--
-- The `lorebook.create` tool lets the narrative model record newly
-- established world detail as session lore (source="narrative_established").
-- For branch-aware retrieval, each entry needs to know which branch it was
-- created on — mirroring how `facts` and `entities` carry `branch_id`.
--
-- Entries created before a branch point are visible on both branches;
-- entries created after a branch belong to that branch only. Existing
-- entries (character_book imports + any prior rows) default to 'main',
-- which is correct: they predate any branch and stay globally visible.

ALTER TABLE lorebook_entries ADD COLUMN branch_id TEXT NOT NULL DEFAULT 'main';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (304, 'lorebook_entries branch_id for narrative-authored lore');
