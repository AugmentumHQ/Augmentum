-- Inbound API keys: let external OpenAI-compatible clients
-- (OpenWebUI, SillyTavern, Cursor, etc.) authenticate to /v1/* and
-- /api/* without a browser session.
--
-- Distinct from user_api_keys (071_users_auth.sql) which stores
-- OUTBOUND keys the user gives us for upstream providers.
--
-- Storage: only the SHA-256 of the raw key. The raw key is shown to
-- the user exactly once at creation time and never recoverable.

CREATE TABLE IF NOT EXISTS augmentum_api_keys (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name        TEXT NOT NULL DEFAULT '',
    key_hash    TEXT NOT NULL UNIQUE,
    key_prefix  TEXT NOT NULL,  -- first 12 chars (incl. 'sk-aug-') for UI display
    scope       TEXT NOT NULL DEFAULT 'chat',  -- 'chat' or 'admin'
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    last_used_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_augmentum_api_keys_user
    ON augmentum_api_keys(user_id);
