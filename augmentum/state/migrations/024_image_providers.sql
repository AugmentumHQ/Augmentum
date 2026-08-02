-- Cloud image generation provider configuration.
-- Stores API endpoints for cloud image services (OpenAI, Together AI, etc.)
-- that Augmentum can proxy image generation requests to.

CREATE TABLE IF NOT EXISTS image_providers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    api_key TEXT,
    default_model TEXT NOT NULL DEFAULT '',
    default_quality TEXT NOT NULL DEFAULT 'standard',
    is_enabled INTEGER NOT NULL DEFAULT 1,
    is_default INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (24, 'Cloud image generation provider configuration');
