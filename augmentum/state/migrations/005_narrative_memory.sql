-- Narrative long-term memory — stores rolling LLM-generated summaries per session.
CREATE TABLE IF NOT EXISTS narrative_memory (
    session_id TEXT PRIMARY KEY,
    card_type TEXT NOT NULL DEFAULT 'character',
    memory_summary TEXT NOT NULL DEFAULT '',
    last_summary_at INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO schema_version (version, description) VALUES (5, 'Narrative long-term memory');
