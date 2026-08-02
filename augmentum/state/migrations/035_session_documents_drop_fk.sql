-- Recreate session_documents without the session_id FK constraint.
-- Sessions are managed client-side (localStorage) and don't exist in
-- the server's sessions table, so the FK causes INSERT failures.

CREATE TABLE IF NOT EXISTS session_documents_new (
    session_id  TEXT NOT NULL,
    document_id TEXT NOT NULL,
    inject_mode TEXT NOT NULL DEFAULT 'search',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, document_id),
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

INSERT OR IGNORE INTO session_documents_new (session_id, document_id, inject_mode, created_at)
    SELECT session_id, document_id, inject_mode, created_at FROM session_documents;

DROP TABLE IF EXISTS session_documents;

ALTER TABLE session_documents_new RENAME TO session_documents;

INSERT OR IGNORE INTO schema_version (version, description) VALUES (35, 'session_documents_drop_fk');
