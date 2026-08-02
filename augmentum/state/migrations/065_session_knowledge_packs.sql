-- Session-knowledge-pack bindings: per-session manual attachment of knowledge packs.
CREATE TABLE IF NOT EXISTS session_knowledge_packs (
    session_id TEXT NOT NULL,
    pack_id TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, pack_id)
);
