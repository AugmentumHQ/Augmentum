-- 002_narrative_state.sql: Tables for narrative engine state tracking

-- Session messages — DAG-based message tracking with branch detection
CREATE TABLE IF NOT EXISTS session_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    parent_id INTEGER REFERENCES session_messages(id),
    role TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    content TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    branch_id TEXT NOT NULL DEFAULT 'main',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_sm_session ON session_messages(session_id, branch_id, message_index);
CREATE INDEX IF NOT EXISTS idx_sm_hash ON session_messages(session_id, content_hash);
CREATE INDEX IF NOT EXISTS idx_sm_parent ON session_messages(parent_id);

-- Facts — established truths within a session
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    content TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'extracted',
    confidence REAL NOT NULL DEFAULT 0.8,
    domain TEXT NOT NULL DEFAULT 'general',
    established_at INTEGER NOT NULL DEFAULT 0,
    superseded_by TEXT REFERENCES facts(id),
    branch_id TEXT NOT NULL DEFAULT 'main',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id, branch_id);
CREATE INDEX IF NOT EXISTS idx_facts_domain ON facts(session_id, domain);

-- Fact tags for categorization
CREATE TABLE IF NOT EXISTS fact_tags (
    fact_id TEXT NOT NULL REFERENCES facts(id),
    tag TEXT NOT NULL,
    PRIMARY KEY (fact_id, tag)
);

-- Entities — characters, locations, items tracked within a session
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    aliases TEXT DEFAULT '[]',
    state TEXT DEFAULT '{}',
    branch_id TEXT NOT NULL DEFAULT 'main',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_entities_session ON entities(session_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(session_id, name);

-- Entity state history — delta-compressed state changes per message
CREATE TABLE IF NOT EXISTS entity_state_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL REFERENCES entities(id),
    message_index INTEGER NOT NULL,
    delta TEXT NOT NULL DEFAULT '{}',
    branch_id TEXT NOT NULL DEFAULT 'main',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_esh_entity ON entity_state_history(entity_id, branch_id, message_index);

-- Plot threads — narrative arcs within a session
CREATE TABLE IF NOT EXISTS plot_threads (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    title TEXT NOT NULL,
    description TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    established_at INTEGER NOT NULL DEFAULT 0,
    resolved_at INTEGER,
    branch_id TEXT NOT NULL DEFAULT 'main',
    state TEXT DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_plots_session ON plot_threads(session_id, status);

-- Contradictions — detected inconsistencies
CREATE TABLE IF NOT EXISTS contradictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    message_index INTEGER NOT NULL,
    contradiction_type TEXT NOT NULL,
    description TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'minor',
    resolution TEXT,
    fact_ids TEXT DEFAULT '[]',
    branch_id TEXT NOT NULL DEFAULT 'main',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_contradictions_session ON contradictions(session_id, branch_id);

-- Lorebook entries — world info / lore for context injection
CREATE TABLE IF NOT EXISTS lorebook_entries (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    keywords TEXT NOT NULL DEFAULT '[]',
    content TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 100,
    source TEXT NOT NULL DEFAULT 'character_book',
    enabled INTEGER NOT NULL DEFAULT 1,
    constant INTEGER NOT NULL DEFAULT 0,
    position TEXT NOT NULL DEFAULT 'before_char',
    scan_depth INTEGER NOT NULL DEFAULT 5,
    case_sensitive INTEGER NOT NULL DEFAULT 0,
    sticky_turns INTEGER NOT NULL DEFAULT 0,
    cooldown_turns INTEGER NOT NULL DEFAULT 0,
    last_triggered_at INTEGER,
    trigger_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_lorebook_session ON lorebook_entries(session_id, enabled);

-- Assumptions — inferences made by the engine
CREATE TABLE IF NOT EXISTS assumptions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    content TEXT NOT NULL,
    made_at INTEGER NOT NULL DEFAULT 0,
    validated INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.5,
    branch_id TEXT NOT NULL DEFAULT 'main',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_assumptions_session ON assumptions(session_id, branch_id);

-- Character cards — parsed and stored per session
CREATE TABLE IF NOT EXISTS character_cards (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    name TEXT NOT NULL,
    data TEXT NOT NULL DEFAULT '{}',
    source_format TEXT NOT NULL DEFAULT 'unknown',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_cards_session ON character_cards(session_id);

INSERT INTO schema_version (version, description) VALUES (2, 'Narrative engine state tables');
