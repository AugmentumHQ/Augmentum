-- Media-server provider credentials (per user).
--
-- Each user configures their own Audiobookshelf / Emby / Jellyfin / etc.
-- instances. Unlike `providers` (server-level LLM inference backends,
-- admin-only), these point at the *user's* own media servers — so the
-- scoping is per-tenant, mirroring the pattern CLAUDE.md documents for
-- user-owned data.
--
-- Catalog items (books, audiobooks, shows, movies) are NOT stored here.
-- They live in `file_index` with source='audiobookshelf' / 'emby' / etc.
-- and `source_metadata` carrying { server_id, external_id, stream_path,
-- progress_pct, duration_ms, cover_url, chapters? }. That lets the
-- existing file browser (search, chips, tags, trash, progress) work on
-- media rows without any special-casing on the listing path — same
-- architectural decision as bookmarks.
--
-- access_token is stored as plaintext at rest, matching the existing
-- `providers.api_key` column. Encrypting credentials at rest is a
-- pre-launch hardening task tracked in project_release_blockers; media
-- server tokens should get the same treatment whenever that lands.

CREATE TABLE IF NOT EXISTS user_media_servers (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider      TEXT NOT NULL,               -- 'audiobookshelf' | 'emby' | 'jellyfin' | ...
    name          TEXT NOT NULL,               -- user-chosen label ("Home Audiobookshelf")
    base_url      TEXT NOT NULL,               -- normalized, no trailing slash
    access_token  TEXT NOT NULL DEFAULT '',    -- bearer / api-key; empty until connected
    status        TEXT NOT NULL DEFAULT 'untested',  -- 'untested' | 'ok' | 'error'
    status_detail TEXT NOT NULL DEFAULT '',    -- last error message, for UI
    last_sync_at  TEXT,                        -- ISO timestamp of last catalog pull
    item_count    INTEGER NOT NULL DEFAULT 0,  -- last known catalog size, for UI
    created_at    TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_user_media_servers_user
    ON user_media_servers(user_id);

-- One server per (user, provider, base_url) combo. A user with the same
-- ABS reachable at two URLs (LAN + Tailscale) still gets two rows, which
-- is intentional — the tokens differ and failover is a separate concern.
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_media_servers_unique
    ON user_media_servers(user_id, provider, base_url);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (95, 'user_media_servers table for Audiobookshelf/Emby/Jellyfin credentials');
