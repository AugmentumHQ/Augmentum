-- Per-user history of capability invocations on devices.
--
-- Drives two user-visible features:
--
-- 1. Smart-match for voice / LLM commands. When the user says
--    "play the lofi music on the living room TV", the LLM tool layer
--    needs to pick *which* lofi item — favorites first, then
--    most-recently-played that fits the query, then most-played, then
--    arbitrary match. This table is the MRU/most-played source.
--
-- 2. The "Recently played on..." pivots in the Connected Devices panel
--    and the chat composer's device picker.
--
-- We track the device + capability + the resolved content reference
-- (file_id when known, otherwise an opaque content_key like an Emby
-- external ID, archive.org identifier, or generated-image hash).
--
-- This is a log, not a snapshot — multiple plays of the same item make
-- multiple rows. Aggregation (count, last_played, favorite) happens in
-- query layer.

CREATE TABLE IF NOT EXISTS device_play_history (
    id             TEXT PRIMARY KEY,                                -- 'dph_<sha1-12>'
    user_id        TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_id      TEXT NOT NULL,                                   -- saved_devices.id (no FK so we keep history if device removed)
    capability_id  TEXT NOT NULL,                                   -- e.g. 'media.audio_play@1'
    action         TEXT NOT NULL DEFAULT 'play',                    -- which capability action was invoked
    file_id        TEXT NOT NULL DEFAULT '',                        -- file_index.id if known
    content_key    TEXT NOT NULL DEFAULT '',                        -- driver/provider-scoped opaque key
    content_label  TEXT NOT NULL DEFAULT '',                        -- 'Lofi Hip Hop Mix - 4 Hours'
    content_kind   TEXT NOT NULL DEFAULT '',                        -- 'audiobook' | 'movie' | 'music' | 'image' | ...
    is_favorite    INTEGER NOT NULL DEFAULT 0,                      -- 0/1; user-toggled
    success        INTEGER NOT NULL DEFAULT 1,                      -- did the cast actually start?
    extra          TEXT NOT NULL DEFAULT '{}',                      -- JSON: room, query, source provider, etc.
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_device_play_history_user
    ON device_play_history(user_id);

CREATE INDEX IF NOT EXISTS idx_device_play_history_device
    ON device_play_history(user_id, device_id);

CREATE INDEX IF NOT EXISTS idx_device_play_history_capability
    ON device_play_history(user_id, capability_id, created_at);

-- Recent-plays-by-content lookup for the smart-match heuristic
CREATE INDEX IF NOT EXISTS idx_device_play_history_content
    ON device_play_history(user_id, content_kind, created_at);

-- Favorite lookup
CREATE INDEX IF NOT EXISTS idx_device_play_history_favorite
    ON device_play_history(user_id, is_favorite, content_kind);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (131, 'device_play_history MRU/favorite tracking for voice + LLM smart-match');
