-- L0 layer of the Observation Substrate (BOM in the lab name).
-- Stores exact-text fingerprint → continuation observations per user,
-- tokenizer-agnostic. Per-model lookup caches are derived lazily from
-- this table at runtime (see augmentum/observation/exporter.py).
--
-- Phase A scope: L0 only. L1 (token-type abstractions) and L2 (logit
-- fingerprints) land in later phases per the substrate spec.

CREATE TABLE IF NOT EXISTS bom_observations_exact (
    user_id TEXT NOT NULL,
    surface TEXT NOT NULL DEFAULT 'chat',
    mode TEXT NOT NULL DEFAULT '',
    -- Fingerprint of the (text_prefix, surface, mode) tuple. Stored as
    -- TEXT (hex) rather than BLOB so it round-trips cleanly through
    -- aiosqlite + JSON without manual encoding ceremony.
    fingerprint TEXT NOT NULL,
    -- Literal text prefix the fingerprint was derived from. Kept so the
    -- exporter can emit a text corpus for llama-lookup-create without
    -- needing to round-trip through the LLM tokenizer first.
    prefix_text TEXT NOT NULL,
    -- Observed continuation. Bounded; the seeder caps at ~12 words so
    -- the cache file stays sane.
    continuation TEXT NOT NULL,
    observation_count INTEGER NOT NULL DEFAULT 1,
    last_seen_ts INTEGER NOT NULL,
    decay_weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (user_id, fingerprint, continuation),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Hot-path query: rank user's observations by recency-weighted count
-- for the exporter's top-K selection.
CREATE INDEX IF NOT EXISTS idx_bom_obs_exact_user_rank
    ON bom_observations_exact (user_id, observation_count DESC, last_seen_ts DESC);

-- Surface/mode slice — for future per-surface autocomplete consumers.
CREATE INDEX IF NOT EXISTS idx_bom_obs_exact_user_surface_mode
    ON bom_observations_exact (user_id, surface, mode);

INSERT OR IGNORE INTO schema_version (version, description)
    VALUES (234, 'Observation Substrate L0 (exact-text fingerprints)');
