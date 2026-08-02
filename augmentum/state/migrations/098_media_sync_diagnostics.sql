-- Sync diagnostics for user_media_servers so the Media Servers panel
-- can surface "X items indexed, Y skipped" inline with the server row.
-- Before this, the only signal that a sync silently dropped books was
-- a docker log line — users hit "Sync" and had no way to tell that
-- 141 of their 453 books never landed (incident: 2026-04-20).
--
-- Columns:
--
--   total_seen          — how many items the catalog fetch saw (indexed + skipped)
--   skipped_count       — count of items that couldn't derive a stream path
--   last_sync_skipped   — JSON array of {title, reason}, capped to ~30 in Python
--                         so the column stays under a KB even on bad syncs.
--
-- All three default to "nothing known yet" so pre-existing rows read
-- back as 0 / empty until their next sync writes fresh values.

ALTER TABLE user_media_servers ADD COLUMN total_seen        INTEGER NOT NULL DEFAULT 0;
ALTER TABLE user_media_servers ADD COLUMN skipped_count     INTEGER NOT NULL DEFAULT 0;
ALTER TABLE user_media_servers ADD COLUMN last_sync_skipped TEXT    NOT NULL DEFAULT '[]';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (98, 'user_media_servers sync diagnostics (total_seen, skipped_count, last_sync_skipped)');
