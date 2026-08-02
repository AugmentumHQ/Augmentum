-- 305_provider_ownership_and_sharing.sql
-- Per-user ownership + a sharing flag for runtime providers.
--
--   owner_user_id: '' (or NULL) = admin/server-global — visible to everyone.
--                  a user id     = privately owned by that user.
--   shared:        1 = visible + resolvable by every user on the instance.
--                  0 = private to the owner.
--
-- Policy (enforced in augmentum/proxy/provider_routes.py):
--   admin-added provider  -> owner_user_id='', shared=1  (shared infra)
--   user-added provider   -> owner_user_id=<uid>, shared=0 (private)
-- Only an admin can flip `shared` (PUT /api/providers/{id}/share).
--
-- Visibility is enforced at two boundaries because provider->backend
-- resolution is process-global (see provider_registry._provider_meta):
--   LIST    (/v1/models, /api/tags, /api/providers) filters by user.
--   RESOLVE (resolve_backend_with_fabric) refuses a private provider's
--           model for a non-owner.

ALTER TABLE providers ADD COLUMN owner_user_id TEXT DEFAULT '';
ALTER TABLE providers ADD COLUMN shared INTEGER NOT NULL DEFAULT 0;

-- Backfill: every provider that exists at migration time predates this
-- feature and was admin-added shared infrastructure — keep it visible.
UPDATE providers SET shared = 1, owner_user_id = '';

CREATE INDEX IF NOT EXISTS idx_providers_owner ON providers(owner_user_id);
