-- Embedded narrative archive: pair-aware exchange storage with vector search
CREATE TABLE IF NOT EXISTS narrative_archive (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    user_content    TEXT NOT NULL,
    assistant_content TEXT NOT NULL,
    summary     TEXT NOT NULL DEFAULT '',
    turn_number INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_narrative_archive_session
    ON narrative_archive(session_id);

CREATE INDEX IF NOT EXISTS idx_narrative_archive_session_turn
    ON narrative_archive(session_id, turn_number);

INSERT OR IGNORE INTO schema_version (version, description) VALUES (27, 'narrative_archive');
