-- Stage D of the multi-tenancy rollout: per-user settings.
--
-- The pre-existing ``app_settings`` table is a single global key-value
-- store used for everything from rate limits (genuinely install-wide) to
-- UI preferences like ``ui.aiName`` and ``ui.typographyTextSize`` (clearly
-- per-tenant). The latter were leaking across accounts — user B's name
-- and typography overwrote user A's whenever the frontend saved settings.
--
-- This migration introduces a sibling ``user_settings`` table with a
-- composite primary key on (user_id, key) so two tenants can hold
-- different values for the same key. ``app_settings`` stays as-is for
-- genuinely install-wide settings (auth config, model routing roles,
-- server secrets, rate limits, etc.). Which keys belong where is decided
-- at the application layer in settings_store.py and config_routes.py.
--
-- No data is migrated from app_settings here. Existing keys stay global
-- until a tenant explicitly saves their own override, at which point the
-- UI writes land in user_settings. The read path falls back from
-- user_settings to app_settings, so the first-save moment is the only
-- point where a tenant diverges from the install default.

CREATE TABLE IF NOT EXISTS user_settings (
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key        TEXT NOT NULL,
    value      TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, key)
);

CREATE INDEX IF NOT EXISTS idx_user_settings_user ON user_settings(user_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (94, 'per-user settings table');
