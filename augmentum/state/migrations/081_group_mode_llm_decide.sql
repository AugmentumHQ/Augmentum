-- Expand character_groups.generation_mode CHECK constraint to include 'llm_decide'.
-- SQLite can't ALTER a CHECK constraint in place, so we recreate the table
-- via the rename-copy-swap pattern. All existing rows preserve their values.

PRAGMA foreign_keys = OFF;

CREATE TABLE character_groups_new (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    member_names TEXT NOT NULL DEFAULT '[]',
    generation_mode TEXT NOT NULL DEFAULT 'round_robin'
        CHECK(generation_mode IN ('round_robin', 'random', 'manual', 'llm_decide')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    member_summaries TEXT NOT NULL DEFAULT '{}',
    avatar TEXT NOT NULL DEFAULT '',
    muted_names TEXT NOT NULL DEFAULT '[]'
);

INSERT INTO character_groups_new
    (id, name, description, member_names, generation_mode,
     created_at, updated_at, member_summaries, avatar, muted_names)
SELECT
    id, name, description, member_names, generation_mode,
    created_at, updated_at, member_summaries, avatar, muted_names
FROM character_groups;

DROP TABLE character_groups;
ALTER TABLE character_groups_new RENAME TO character_groups;

PRAGMA foreign_keys = ON;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (81, 'group_mode_llm_decide_constraint');
