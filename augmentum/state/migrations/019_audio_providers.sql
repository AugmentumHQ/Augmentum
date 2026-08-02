-- Audio provider configuration for TTS and STT services.
-- Stores OpenAI-compatible API endpoints that Augmentum proxies to.

CREATE TABLE IF NOT EXISTS audio_providers (
    id TEXT PRIMARY KEY,
    provider_type TEXT NOT NULL CHECK (provider_type IN ('tts', 'stt')),
    name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    api_key TEXT,
    default_model TEXT NOT NULL DEFAULT '',
    default_voice TEXT NOT NULL DEFAULT '',
    is_enabled INTEGER NOT NULL DEFAULT 1,
    is_default INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO schema_version (version, description) VALUES (19, 'audio_providers');
