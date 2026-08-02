-- 281_connect_profiles.sql
-- Connect profiles (comms platform, Phase 1 — discovery/profiles).
--
-- A user's Connect-facing profile: a short bio, a custom status line, and an
-- optional status emoji. Kept SEPARATE from the auth `users` row so the social
-- surface can evolve without touching the account model; display_name still
-- lives on `users` (the canonical label), this table adds the soft, editable
-- presentation layer surfaced in the contact-detail panel and directory.
--
-- User-scoped per CLAUDE.md: `user_id` is both the owner key and the PRIMARY
-- KEY (one profile per user). `avatar_ref` points into the existing `avatars`
-- store when set.
CREATE TABLE IF NOT EXISTS connect_profiles (
    user_id        TEXT PRIMARY KEY,
    bio            TEXT NOT NULL DEFAULT '',
    status_message TEXT NOT NULL DEFAULT '',
    status_emoji   TEXT NOT NULL DEFAULT '',
    avatar_ref     TEXT NOT NULL DEFAULT '',
    updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (281, 'connect_profiles: bio / status for Connect discovery');
