-- 238_companion_tracked_topics.sql
-- Topics the user explicitly tracks. A row with feed_url=NULL is a plain
-- topic pin (curator chooses where to look for updates). A row with
-- feed_url set is a subscription to a specific feed (RSS / arxiv cat /
-- HN tag / etc). Unified shape so the UX has one concept ("things I
-- want her to watch") instead of two ("topics" vs "feeds").
--
-- Derived interest signal lives in interest_clusters (migration 067)
-- and is unchanged. The curator merges both when picking what to
-- surface.

CREATE TABLE IF NOT EXISTS companion_tracked_topics (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    companion_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    feed_url TEXT,
    feed_kind TEXT,                                   -- 'rss' | 'hn' | 'arxiv' | 'reddit' | NULL=auto-route
    weight REAL NOT NULL DEFAULT 1.0,
    last_polled_at TEXT,
    error_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, companion_id, topic)
);

CREATE INDEX IF NOT EXISTS idx_tracked_topics_user
    ON companion_tracked_topics(user_id, companion_id);

CREATE INDEX IF NOT EXISTS idx_tracked_topics_poll_due
    ON companion_tracked_topics(last_polled_at);
