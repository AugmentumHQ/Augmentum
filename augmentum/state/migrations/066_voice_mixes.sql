-- Persistent storage for saved Kokoro voice mixes (blended voices).
-- Stores the friendly name and the blend spec string so mixes survive
-- server restarts and appear in the voice listing.

CREATE TABLE IF NOT EXISTS voice_mixes (
    name TEXT PRIMARY KEY,
    blend_spec TEXT NOT NULL,
    provider_id TEXT NOT NULL DEFAULT 'kokoro-builtin',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO schema_version (version, description) VALUES (66, 'voice_mixes');
