-- Audiobook / podcast bookmarks: a user-saved position in a title, with
-- an optional note. Brings the listening experience to Audiobookshelf /
-- Plappa parity (bookmarks were the marquee missing feature).
--
-- user-scoped (every row carries user_id, per the multi-tenant contract).
-- episode_id distinguishes per-episode bookmarks for podcasts; '' for
-- single-file audiobooks. position_s is book-level seconds (matches the
-- media-player's book-time contract). label is an auto-derived display
-- string (chapter + timestamp); note is free text the user can add.
CREATE TABLE IF NOT EXISTS audiobook_bookmarks (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL DEFAULT '',
    file_id     TEXT NOT NULL,
    episode_id  TEXT NOT NULL DEFAULT '',
    position_s  REAL NOT NULL DEFAULT 0,
    label       TEXT NOT NULL DEFAULT '',
    note        TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

-- Hot path: list a title's bookmarks for the current user, ordered by
-- position. The composite covers the WHERE (user_id, file_id, episode_id)
-- and the ORDER BY (position_s) in one index.
CREATE INDEX IF NOT EXISTS idx_audiobook_bookmarks_lookup
    ON audiobook_bookmarks (user_id, file_id, episode_id, position_s);
