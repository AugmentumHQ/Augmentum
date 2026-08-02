-- Add a sharing scope to user_media_servers so an admin can publish
-- their connection (Audiobookshelf/Emby/Jellyfin/Komga/Suwayomi) to
-- every other user on the box. The credential row is shared READ-ONLY:
-- non-owners see it, can sync to their own file_index, and can stream
-- through it, but can't edit URL/token/name or flip the share toggle.
--
-- scope values:
--   'private' — default; only the owning user_id sees / uses it.
--   'shared'  — admin-published; visible to every authenticated user.
--
-- Catalog rows (file_index, media_library_views) stay user-scoped, so
-- "watched history" and per-user library state are NOT shared. Only
-- the credential record is. Sharing the catalog would require a
-- separate cross-user view table and is intentionally out of scope.
--
-- CORRECTION (2026-07-25): the paragraph above was true of the ROWS but
-- false of their CONTENTS as originally shipped. The shared credential
-- is the OWNER's, so every provider call made with it read and wrote the
-- owner's account: the catalog fetch itself carried their UserData
-- (Emby/Jellyfin send it when ``EnableUserData=true``; Komga folds in
-- readProgress), fetch_progress returned their resume points, and the
-- /progress endpoint pushed the BORROWER's playback back into the
-- OWNER's Emby/ABS account. Per-user rows were isolated; the per-user
-- state inside them was not, in either direction.
--
-- The rule is now enforced in code by ``MediaServer.is_borrowed_by``:
-- for a borrowed server, provider-side per-user state is stripped on
-- read and no progress is pushed on write. Progress on a borrowed
-- library is Augmentum-side only. Any new provider field describing
-- what a USER did (as opposed to what an ITEM is) must be added to
-- ``_OWNER_USER_STATE_EXTRA_KEYS`` in augmentum/media/sync.py.
--
-- Default 'private' is critical: pre-existing rows must NOT silently
-- become visible to every other user when this migration applies.
-- Admins explicitly flip individual servers later via the UI.

ALTER TABLE user_media_servers ADD COLUMN scope TEXT NOT NULL DEFAULT 'private';

-- Index on scope so `list_visible(user_id)` — which selects
-- `WHERE user_id = ? OR scope = 'shared'` — doesn't scan the whole
-- table when a server has 50 users.
CREATE INDEX IF NOT EXISTS idx_user_media_servers_scope
    ON user_media_servers(scope);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (172, 'user_media_servers.scope for admin-shared credentials');
