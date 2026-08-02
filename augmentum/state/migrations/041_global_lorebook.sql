-- Global lorebook collections — named sets of lorebook entries that can be
-- imported as a whole into any character.  Entries are copied on import,
-- so edits on the character are independent from the global version.

CREATE TABLE IF NOT EXISTS global_lorebook_collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    entry_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS global_lorebook_entries (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL REFERENCES global_lorebook_collections(id) ON DELETE CASCADE,
    name TEXT NOT NULL DEFAULT '',
    keys TEXT NOT NULL DEFAULT '[]',
    content TEXT NOT NULL DEFAULT '',
    enabled INTEGER NOT NULL DEFAULT 1,
    priority INTEGER NOT NULL DEFAULT 100,
    position TEXT NOT NULL DEFAULT 'before_char',
    sticky_turns INTEGER NOT NULL DEFAULT 0,
    cooldown_turns INTEGER NOT NULL DEFAULT 0,
    constant INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_global_lore_collection ON global_lorebook_entries(collection_id);

-- Drop the old flat table if it exists (from earlier migration attempt)
DROP TABLE IF EXISTS global_lorebook;

INSERT OR IGNORE INTO schema_version (version, description) VALUES (41, 'Global lorebook collections');
