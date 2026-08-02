-- 323_file_index_detached.sql
-- Mark file_index rows whose media server is no longer reachable BY THIS
-- USER, so a revoked share tears down cleanly instead of leaving ghosts.
--
-- Symptom this fixes (observed 2026-07-25): an admin un-shared a media
-- server. `store.set_scope` flips `user_media_servers.scope` to 'private'
-- and does nothing else — no cascade of any kind. Every borrower's
-- file_index and media_library_views rows survived, still carrying
-- `source_metadata.server_id` pointing at a row `get_visible` now refuses
-- them. Every stream / image / playback route resolves through
-- `get_visible` and started returning 502, but the library LISTING reads
-- file_index directly, so the items kept rendering as normal playable
-- cards. Thumbnails died except whatever was already in the image cache.
--
-- Same class, two more doors:
--   - `delete_server` DOES cascade (purge_server_data) but only for
--     `user_id = caller`, so an owner deleting a SHARED server orphans
--     every borrower's rows identically.
--   - media_library_views has a real ON DELETE CASCADE (migration 109),
--     so on delete borrowers lose their VIEWS but keep their ITEMS — a
--     half-torn-down state worse than either extreme.
--
-- Why a separate column instead of a new visibility flag: `is_trashed = 0`
-- is ALREADY the universal invisibility filter (~30 queries across
-- files_routes, media_routes, cast_routes, vfs/index, vfs/enrichment,
-- companion_runtime/validators). Setting is_trashed = 1 makes a row
-- visually gone everywhere for free; a brand-new flag would mean editing
-- all ~30 and missing one. But is_trashed alone is WRONG here:
--
--   - `purge_all_old_trash` (vfs/index.py) hard-deletes any is_trashed
--     row older than 30 days, across all users, unconditionally. Reusing
--     is_trashed by itself would silently delete the preserved history a
--     month later — precisely the data the tombstone exists to save.
--   - The rows would appear in the user's Trash with a Restore button
--     that resurrects a row pointing at a server they still can't reach.
--
-- So: is_trashed carries the invisibility, `detached_at` carries the
-- REASON, and the four trash-semantics paths (list_trash, the trash
-- count, list_trashed_older_than, purge_all_old_trash) exclude
-- `detached_at IS NOT NULL`. Four targeted exclusions instead of thirty
-- additions. Detached rows are invisible, not user-deleted, not
-- auto-purged, not restorable into a broken state.
--
-- Re-attach falls out for free: `register()` upserts on
-- (user_id, source, source_id), which tombstoning does not change, so a
-- resync after the admin re-shares hits the SAME row. Clearing
-- is_trashed + detached_at there restores the item with its progress
-- intact — no separate restore path needed.
--
-- `detached_server_id` records which server did it, so a re-share can
-- reattach exactly that server's rows and leave other detached rows
-- alone.
--
-- Additive only. No existing row is touched: `detached_at` is NULL
-- everywhere, which makes the trash-path exclusions provable no-ops
-- until something is actually marked.

ALTER TABLE file_index ADD COLUMN detached_at TEXT;
ALTER TABLE file_index ADD COLUMN detached_server_id TEXT NOT NULL DEFAULT '';

-- Serves both directions of the lifecycle: finding a user's detached
-- rows to re-attach on re-share, and the trash-path exclusions.
CREATE INDEX IF NOT EXISTS idx_file_index_detached
    ON file_index(user_id, detached_server_id, detached_at);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (323, 'file_index.detached_at — clean teardown for revoked shares');
