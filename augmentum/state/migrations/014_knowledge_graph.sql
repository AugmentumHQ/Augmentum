-- Knowledge graph: schema-free nodes and edges for universal relationship tracking.
-- Nodes represent entities (people, places, concepts, events) discovered from conversation.
-- Edges represent typed, weighted, bi-temporal relationships between nodes.
-- chat_id = NULL means global knowledge (visible in all chats for that user).

CREATE TABLE IF NOT EXISTS kg_nodes (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'thing',
    properties TEXT DEFAULT '{}',
    chat_id TEXT,
    user_id TEXT NOT NULL DEFAULT 'default',
    memory_id TEXT,
    embedding BLOB,
    mentions INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS kg_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.5,
    evidence TEXT,
    chat_id TEXT,
    valid_from INTEGER,
    valid_until INTEGER,
    message_idx INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_id, target_id, relation, chat_id)
);

CREATE INDEX IF NOT EXISTS idx_kg_nodes_chat ON kg_nodes(chat_id, user_id);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_kind ON kg_nodes(kind, chat_id);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_label ON kg_nodes(label);
CREATE INDEX IF NOT EXISTS idx_kg_nodes_memory ON kg_nodes(memory_id);

CREATE INDEX IF NOT EXISTS idx_kg_edges_source ON kg_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_target ON kg_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_chat ON kg_edges(chat_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_relation ON kg_edges(relation, chat_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_temporal ON kg_edges(chat_id, valid_from);
CREATE INDEX IF NOT EXISTS idx_kg_edges_active ON kg_edges(source_id, valid_until);

INSERT OR IGNORE INTO schema_version (version, description) VALUES (14, 'knowledge_graph');
