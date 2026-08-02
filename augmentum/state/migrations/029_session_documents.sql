-- Per-session document binding with inject mode
-- NOTE: session_id has no FK constraint because sessions are managed
-- client-side (localStorage) and may not exist in the sessions table.
CREATE TABLE IF NOT EXISTS session_documents (
    session_id  TEXT NOT NULL,
    document_id TEXT NOT NULL,
    inject_mode TEXT NOT NULL DEFAULT 'search',  -- 'search' or 'full'
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, document_id),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

INSERT OR IGNORE INTO schema_version (version, description) VALUES (29, 'session_documents');
