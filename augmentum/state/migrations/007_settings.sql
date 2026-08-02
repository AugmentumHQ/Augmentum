-- 007_settings.sql: Persistent key-value settings store

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT INTO schema_version (version, description) VALUES (7, 'Persistent settings store');
