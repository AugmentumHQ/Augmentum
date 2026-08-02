-- Prompt presets, regex scripts, and character groups for narrative mode.

CREATE TABLE IF NOT EXISTS prompt_presets (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    system_prompt TEXT NOT NULL DEFAULT '',
    jailbreak TEXT NOT NULL DEFAULT '',
    post_history TEXT NOT NULL DEFAULT '',
    author_note TEXT NOT NULL DEFAULT '',
    author_note_depth INTEGER NOT NULL DEFAULT 4,
    is_default INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS regex_scripts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    find_regex TEXT NOT NULL,
    replace_string TEXT NOT NULL DEFAULT '',
    placement TEXT NOT NULL DEFAULT 'output' CHECK(placement IN ('input', 'output', 'both')),
    enabled INTEGER NOT NULL DEFAULT 1,
    order_num INTEGER NOT NULL DEFAULT 100,
    character_name TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_regex_scripts_order ON regex_scripts(order_num);

CREATE TABLE IF NOT EXISTS character_groups (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    member_names TEXT NOT NULL DEFAULT '[]',
    generation_mode TEXT NOT NULL DEFAULT 'round_robin' CHECK(generation_mode IN ('round_robin', 'random', 'manual')),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT INTO schema_version (version, description)
VALUES (18, 'Prompt presets, regex scripts, character groups');
