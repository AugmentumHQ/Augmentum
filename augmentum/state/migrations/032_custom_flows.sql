-- Custom tool chain flows: user-defined reusable multi-step tool chains.

CREATE TABLE IF NOT EXISTS custom_flows (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    trigger_pattern TEXT DEFAULT '',
    steps_json TEXT NOT NULL,         -- JSON array of step definitions
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    enabled INTEGER DEFAULT 1
);

INSERT OR IGNORE INTO schema_version (version, description) VALUES (32, 'Custom tool chain flows');
