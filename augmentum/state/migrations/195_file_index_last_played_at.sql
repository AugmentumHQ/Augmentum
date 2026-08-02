-- 195_file_index_last_played_at.sql
-- Dedicated playback-recency column so the Continue rail's ordering
-- survives catalog sync.
--
-- Symptom this fixes (observed 2026-05-25): the Continue rail's
-- ``sort=progress`` aliases to ``updated_at DESC`` (vfs/index.py:84).
-- Catalog sync UPSERTs every synced row with ``updated_at =
-- datetime('now')`` (vfs/index.py:269) AND it rebuilds source_metadata
-- from scratch — wiping the local ``last_read_at`` JSON field set on
-- every play (media_routes.py:3799). Both together meant the rail's
-- ordering signal was destroyed by every catalog sync: rows the user
-- hadn't touched in weeks would surface alongside genuinely-recent
-- plays because their updated_at had been refreshed by the same sync
-- pass. Compounded by the fact that cast-driven playback DOES push
-- to /api/media/progress correctly — the data path was right, the
-- column choice was wrong.
--
-- This column is updated EXCLUSIVELY by the progress endpoint and
-- never by sync. Continue-rail sort switches to use it (falling back
-- to updated_at for never-played items so they still appear in a
-- sensible default order). Sync continues to bump updated_at as
-- before — that's the right behavior for "row was touched at all"
-- queries; it just isn't the right signal for "row was played most
-- recently."
--
-- See [[project_continue_rail_recency]] for the broader fix context.

ALTER TABLE file_index ADD COLUMN last_played_at TEXT;

CREATE INDEX IF NOT EXISTS idx_file_index_last_played_at
    ON file_index(user_id, last_played_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (195, 'file_index.last_played_at — Continue rail ordering');
