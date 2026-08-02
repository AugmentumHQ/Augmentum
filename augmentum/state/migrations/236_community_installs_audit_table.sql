-- 236_community_installs_audit_table.sql
-- Audit trail for items pulled in from the community directory.
-- Spec: docs/specs/community-install.md (in the augmentumhq-site repo).

CREATE TABLE IF NOT EXISTS community_installs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    manifest_url TEXT NOT NULL,
    category TEXT NOT NULL,
    slug TEXT NOT NULL,
    item_version TEXT NOT NULL DEFAULT '',
    installed_resource_id TEXT NOT NULL,
    installed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_community_installs_user
    ON community_installs(user_id);

CREATE INDEX IF NOT EXISTS idx_community_installs_category_slug
    ON community_installs(category, slug);
