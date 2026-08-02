-- Voice enrollment: speaker voiceprints for voice verification
CREATE TABLE IF NOT EXISTS voice_enrollments (
    id          TEXT PRIMARY KEY,
    scope       TEXT NOT NULL DEFAULT '',     -- user/session scope key
    voiceprint  TEXT NOT NULL,                -- JSON: embedding + metadata
    enrolled_at REAL NOT NULL,                -- Unix timestamp
    quality     REAL NOT NULL DEFAULT 0.0,    -- Self-consistency score 0-1
    samples     INTEGER NOT NULL DEFAULT 0,   -- Number of enrollment samples
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_voice_enrollments_scope
    ON voice_enrollments(scope);

INSERT OR IGNORE INTO schema_version (version, description) VALUES (31, 'voice_enrollment');
