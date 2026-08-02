-- Reassign rows stranded under the sentinel user_id='default' bucket to
-- a real user. Those rows were created by code paths that called the
-- document store (or similar) without extracting user_id from the auth
-- scope — the now-fixed browse_save endpoint was the largest source
-- (see augmentum/proxy/browse_routes.py browse_save prior to
-- 2026-04-22). They were invisible to every real user because listings
-- and content fetches filter by user_id, and "default" never matches a
-- real account id.
--
-- Target selection: the oldest active admin. If no admin exists yet
-- (fresh DB, or users table missing), all UPDATEs no-op harmlessly via
-- the EXISTS guard.
--
-- Scope: only the three tables that browse_save touched. Other
-- "default"-scoped rows (if any) are out of scope here and can be
-- hand-reassigned if they surface.

UPDATE documents
SET user_id = (
    SELECT id FROM users
    WHERE role = 'admin' AND is_active = 1
    ORDER BY created_at ASC LIMIT 1
)
WHERE user_id = 'default'
  AND EXISTS (
      SELECT 1 FROM users
      WHERE role = 'admin' AND is_active = 1
  );

UPDATE document_chunks
SET user_id = (
    SELECT id FROM users
    WHERE role = 'admin' AND is_active = 1
    ORDER BY created_at ASC LIMIT 1
)
WHERE user_id = 'default'
  AND EXISTS (
      SELECT 1 FROM users
      WHERE role = 'admin' AND is_active = 1
  );

UPDATE file_index
SET user_id = (
    SELECT id FROM users
    WHERE role = 'admin' AND is_active = 1
    ORDER BY created_at ASC LIMIT 1
)
WHERE user_id = 'default'
  AND source = 'documents'
  AND EXISTS (
      SELECT 1 FROM users
      WHERE role = 'admin' AND is_active = 1
  );

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (105, 'Backfill default-bucket document rows to oldest active admin');
