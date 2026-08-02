-- 254_marketplace_listings_discover_columns.sql
--
-- Additive columns on marketplace_listings to support the unified
-- Discover surface (see docs/superpowers/specs/2026-06-10-discover-
-- surface-design.md). No CHECK constraint on `kind` — the store-layer
-- allow-list owns the kind enum so new content types slot in without
-- a schema migration.
--
-- ``category`` is the top-level group the Discover UI renders rows by
-- (providers / games / characters / powers / reasoning-flows /
-- knowledge / other). Existing rows are backfilled from their kind.
-- ``tags`` is a flat JSON array for filter chips. ``featured`` is a
-- boolean flag for the homepage rail.

ALTER TABLE marketplace_listings
    ADD COLUMN category TEXT NOT NULL DEFAULT '';

ALTER TABLE marketplace_listings
    ADD COLUMN tags TEXT NOT NULL DEFAULT '[]';

ALTER TABLE marketplace_listings
    ADD COLUMN featured INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_marketplace_listings_category
    ON marketplace_listings(category, listed_at DESC)
    WHERE delisted_at IS NULL;

-- Partial featured-rail index. featured=1 is the rare-row case so the
-- index stays tiny; the predicate matches the Discover homepage query
-- exactly so the planner picks it.
CREATE INDEX IF NOT EXISTS idx_marketplace_listings_featured
    ON marketplace_listings(featured, listed_at DESC)
    WHERE featured = 1 AND delisted_at IS NULL;

-- Backfill category from existing kind. The current catalog only has
-- game-shaped entries; future inserts must supply category at upsert
-- time (the loader is the source of truth).
UPDATE marketplace_listings
   SET category = CASE
       WHEN kind IN ('streamed_game', 'js13k_game', 'web_app') THEN 'games'
       ELSE 'other'
   END
 WHERE category = '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (254, 'marketplace_listings category + tags + featured');
