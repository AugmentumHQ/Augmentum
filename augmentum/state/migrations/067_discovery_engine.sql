-- 067_discovery_engine.sql
-- Discovery Engine Phase 1: signals, history, content library

-- Raw interaction signal log
CREATE TABLE IF NOT EXISTS interaction_signals (
    id TEXT PRIMARY KEY,
    signal_type TEXT NOT NULL,
    source_url TEXT NOT NULL DEFAULT '',
    source_title TEXT NOT NULL DEFAULT '',
    source_domain TEXT NOT NULL DEFAULT '',
    content_type TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    weight REAL NOT NULL DEFAULT 1.0,
    cluster_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_signals_type ON interaction_signals(signal_type);
CREATE INDEX IF NOT EXISTS idx_signals_url ON interaction_signals(source_url);
CREATE INDEX IF NOT EXISTS idx_signals_cluster ON interaction_signals(cluster_id);
CREATE INDEX IF NOT EXISTS idx_signals_created ON interaction_signals(created_at DESC);

-- Deduplicated browse/video visit history
CREATE TABLE IF NOT EXISTS browse_history (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    domain TEXT NOT NULL DEFAULT '',
    content_type TEXT NOT NULL DEFAULT 'article',
    thumbnail TEXT NOT NULL DEFAULT '',
    metadata TEXT NOT NULL DEFAULT '{}',
    cluster_id TEXT,
    visit_count INTEGER NOT NULL DEFAULT 1,
    first_visited TEXT NOT NULL DEFAULT (datetime('now')),
    last_visited TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_history_url ON browse_history(url);
CREATE INDEX IF NOT EXISTS idx_history_visited ON browse_history(last_visited DESC);
CREATE INDEX IF NOT EXISTS idx_history_cluster ON browse_history(cluster_id);

-- Distilled knowledge chunks with embeddings
CREATE TABLE IF NOT EXISTS content_library (
    chunk_id TEXT PRIMARY KEY,
    source_url TEXT NOT NULL,
    source_title TEXT NOT NULL DEFAULT '',
    source_type TEXT NOT NULL DEFAULT 'article',
    content TEXT NOT NULL,
    embedding BLOB,
    cluster_id TEXT,
    retrieved_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_library_source ON content_library(source_url);
CREATE INDEX IF NOT EXISTS idx_library_cluster ON content_library(cluster_id);
CREATE INDEX IF NOT EXISTS idx_library_created ON content_library(created_at DESC);

-- Vector index for content library similarity search
CREATE VIRTUAL TABLE IF NOT EXISTS content_library_vec USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding float[768]
);

-- Interest clusters (placeholder for Phase 2, needed for FK references)
CREATE TABLE IF NOT EXISTS interest_clusters (
    cluster_id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    centroid_embedding BLOB,
    frecency_short REAL NOT NULL DEFAULT 0.0,
    frecency_long REAL NOT NULL DEFAULT 0.0,
    depth_level INTEGER NOT NULL DEFAULT 1,
    signal_count INTEGER NOT NULL DEFAULT 0,
    narration TEXT,
    knowledge_gaps TEXT,
    adjacent_topics TEXT,
    dampened INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
