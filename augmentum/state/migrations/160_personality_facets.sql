-- 160_personality_facets.sql
-- Personality facet Hebbian cooccurrence + cross-table memory associations.
--
-- Implements the substrate underlying commitment #7 (companion-with-owner is
-- its own being) — drift in the relationship-specific personality emerges from
-- facet co-activation patterns specific to a given user. Modeled after
-- Mischel's CAPS (Cognitive-Affective Personality System, 1995): personality
-- is not a trait vector but a network of facets that activate together in
-- context-specific patterns, and what's stable across situations is the
-- *if-then signature* of which facets co-fire when.
--
-- Mirrors migration 050 (memory_cooccurrence) structurally; the cross-table
-- personality_memory_associations is the integration point — when a memory
-- cluster reliably co-fires with a facet pattern, that's the "you bring out
-- a certain side of me" mechanism in code.
--
-- Tables:
--   personality_facets               — vocabulary (server-shared)
--   personality_facet_cooccurrence   — Hebbian facet × facet graph (user-scoped)
--   personality_facet_activations    — per-turn audit log (user-scoped)
--   personality_memory_associations  — cross-table memory × facet (user-scoped)
--
-- The three user-scoped tables follow CLAUDE.md's multi-tenancy invariant:
-- every CRUD function accepts *, user_id: str = "" and appends AND user_id = ?
-- The vocabulary table is server-level (like image_providers, knowledge_packs)
-- because the affect/facet lexicon is shared across the install.
--
-- All tables are additive; no existing schema is touched.

CREATE TABLE IF NOT EXISTS personality_facets (
    name         TEXT PRIMARY KEY,
    description  TEXT NOT NULL,
    category     TEXT NOT NULL,   -- 'affect' | 'cognitive' | 'social' | 'stance' | 'energy'
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Hebbian facet × facet cooccurrence — mirrors memory_cooccurrence (migration 050).
-- Canonical ordering (facet_a < facet_b alphabetically) enforced by callers.
-- companion_id allows the household model: multiple companions per user have
-- independent facet graphs since they ARE different beings (commitments 3, 7).
CREATE TABLE IF NOT EXISTS personality_facet_cooccurrence (
    user_id       TEXT NOT NULL,
    companion_id  TEXT NOT NULL,
    facet_a       TEXT NOT NULL,
    facet_b       TEXT NOT NULL,
    count         INTEGER NOT NULL DEFAULT 1,
    last_updated  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, companion_id, facet_a, facet_b)
);

CREATE INDEX IF NOT EXISTS idx_pers_cooccur_a
    ON personality_facet_cooccurrence(user_id, companion_id, facet_a, count DESC);
CREATE INDEX IF NOT EXISTS idx_pers_cooccur_b
    ON personality_facet_cooccurrence(user_id, companion_id, facet_b, count DESC);

-- Per-turn activation log. The labeler writes one row per active facet per turn.
-- Source of truth for cooccurrence updates (which write in batches) and for
-- the recent-activation feed the persona-kernel digester reads at composition
-- time. Compacted by dream-cycle consolidation (newer than 90 days kept
-- verbatim; older aggregated into cooccurrence and rows dropped).
--
-- INTEGER PRIMARY KEY is the SQLite rowid (no AUTOINCREMENT — per CLAUDE.md
-- migration rules, AUTOINCREMENT is forbidden in this codebase).
CREATE TABLE IF NOT EXISTS personality_facet_activations (
    id            INTEGER PRIMARY KEY,
    user_id       TEXT NOT NULL,
    companion_id  TEXT NOT NULL,
    session_id    TEXT,
    turn_id       TEXT,
    facet         TEXT NOT NULL,
    intensity     REAL NOT NULL DEFAULT 1.0,
    source        TEXT NOT NULL DEFAULT 'self_label',  -- 'self_label' | 'classifier' | 'manual'
    activated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_pers_activ_user_time
    ON personality_facet_activations(user_id, companion_id, activated_at DESC);
CREATE INDEX IF NOT EXISTS idx_pers_activ_facet
    ON personality_facet_activations(user_id, companion_id, facet, activated_at DESC);
CREATE INDEX IF NOT EXISTS idx_pers_activ_session
    ON personality_facet_activations(user_id, companion_id, session_id, activated_at DESC);

-- Cross-table: which memories reliably co-fire with which facets.
-- This is the *interesting* graph — it captures context-dependent personality.
-- When memory cluster X gets activated in retrieval, the associated facets get
-- biased into the persona-kernel composition for that turn. "You bring out a
-- certain side of me when we talk about X" — that's this table.
CREATE TABLE IF NOT EXISTS personality_memory_associations (
    user_id       TEXT NOT NULL,
    companion_id  TEXT NOT NULL,
    memory_id     TEXT NOT NULL,
    facet         TEXT NOT NULL,
    count         INTEGER NOT NULL DEFAULT 1,
    last_updated  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, companion_id, memory_id, facet)
);

CREATE INDEX IF NOT EXISTS idx_pers_mem_assoc_memory
    ON personality_memory_associations(user_id, companion_id, memory_id, count DESC);
CREATE INDEX IF NOT EXISTS idx_pers_mem_assoc_facet
    ON personality_memory_associations(user_id, companion_id, facet, count DESC);

INSERT OR IGNORE INTO schema_version (version, description)
    VALUES (160, 'personality_facets: Hebbian facet cooccurrence + memory associations (CAPS)');
