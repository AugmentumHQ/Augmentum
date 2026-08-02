-- 255_marketplace_installs_audit_table.sql
--
-- Per-user install audit for the Discover surface. Lets the catalog
-- endpoint mark "installed" on listings the user already has, so the
-- card CTA changes from Install → Installed (or Re-install / Open).
--
-- One row per (user_id, listing_id, install_attempt). Soft uninstall
-- via ``uninstalled_at`` so the count + history persist for analytics
-- even after a removal — the unique partial index below keeps the
-- "currently installed" lookup O(1).
--
-- This complements but doesn't replace the existing per-category
-- audit tables:
--   - community_installs (migration 236) — community URL installs
--   - managed_services (migration 068) — provider Docker containers
--   - artifacts (game/title installs)
-- ...all of which track different aspects. marketplace_installs is
-- the unified per-user × per-listing fact used by Discover's UI
-- enrichment query.

CREATE TABLE IF NOT EXISTS marketplace_installs (
    id              TEXT PRIMARY KEY,                       -- mki_<hex>
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    listing_id      TEXT NOT NULL,                          -- mkt:* (no FK — listing may be delisted but install row persists)
    install_via     TEXT NOT NULL,                          -- denormalised for cheap stats
    kind            TEXT NOT NULL DEFAULT '',               -- denormalised for filter joins
    resource_id     TEXT NOT NULL DEFAULT '',               -- whatever the installer returned (artifact_id, char_id, etc.)
    installed_at    TEXT NOT NULL DEFAULT (datetime('now')),
    uninstalled_at  TEXT
);

-- "Currently installed by user" partial index — drives the JOIN in
-- /api/discover/catalog that enriches each listing with installed=true.
CREATE UNIQUE INDEX IF NOT EXISTS idx_marketplace_installs_active
    ON marketplace_installs(user_id, listing_id)
    WHERE uninstalled_at IS NULL;

-- Listing → installer-count rollup (future: install_count per row).
CREATE INDEX IF NOT EXISTS idx_marketplace_installs_listing
    ON marketplace_installs(listing_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (255, 'marketplace_installs per-user install audit');
