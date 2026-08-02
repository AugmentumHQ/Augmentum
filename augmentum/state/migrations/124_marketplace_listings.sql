-- 124_marketplace_listings.sql
-- Curated catalog for the Augmentum Experience Framework (AXF)
-- marketplace surface. Listings are NOT user-scoped -- they're the
-- shared catalog every user can browse. Per-user state (installed,
-- rated, etc.) lives elsewhere (artifacts, future marketplace_reviews).
--
-- Listings ship as a JSON file checked into the repo
-- (data/marketplace/listings.json). On startup the loader UPSERTs each
-- entry into this table so subsequent reads are one cheap query and
-- mid-run hot-reloads (rare) don't require a server restart.
--
-- Signature column is reserved for future Ed25519 verification of
-- third-party publishers. v1 ships only Augmentum-curated listings;
-- the signature field stays empty until publisher keys land.

CREATE TABLE IF NOT EXISTS marketplace_listings (
    id              TEXT PRIMARY KEY,                       -- 'mkt:luanti-voxel-world' style
    publisher       TEXT NOT NULL DEFAULT 'augmentum',
    title           TEXT NOT NULL,
    kind            TEXT NOT NULL,                          -- one of TITLE_KINDS
    runtime_preferred TEXT NOT NULL DEFAULT '',
    runtime_alternates TEXT NOT NULL DEFAULT '[]',          -- JSON array
    tagline         TEXT NOT NULL DEFAULT '',
    description     TEXT NOT NULL DEFAULT '',
    thumbnail_url   TEXT NOT NULL DEFAULT '',
    source_url      TEXT NOT NULL DEFAULT '',
    embed_url       TEXT NOT NULL DEFAULT '',
    -- Identifies the underlying Source the install path delegates to
    -- (e.g. 'js13k' to pin a curated js13k entry, 'agsp-profile' to
    -- launch a bundled streamed game, 'internal' for hand-built
    -- manifests). Lets the marketplace be a thin curation layer on
    -- top of the actual Sources rather than a parallel install path.
    install_via     TEXT NOT NULL,                          -- 'js13k' | 'agsp-profile' | 'internal' | ...
    install_payload TEXT NOT NULL DEFAULT '{}',             -- JSON: full manifest_data forwarded to the underlying source
    capabilities    TEXT NOT NULL DEFAULT '{}',             -- JSON
    metadata        TEXT NOT NULL DEFAULT '{}',             -- JSON: tags, year, screenshots, ...
    rating          REAL,                                   -- aggregated 1.0-5.0 (NULL = unrated)
    install_count   INTEGER NOT NULL DEFAULT 0,             -- aggregate counter
    signature       TEXT NOT NULL DEFAULT '',               -- Ed25519 over canonicalised listing JSON; empty for unsigned
    listed_at       TEXT NOT NULL DEFAULT (datetime('now')),
    delisted_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_marketplace_listings_kind
    ON marketplace_listings(kind, listed_at DESC);

CREATE INDEX IF NOT EXISTS idx_marketplace_listings_publisher
    ON marketplace_listings(publisher, listed_at DESC);

CREATE INDEX IF NOT EXISTS idx_marketplace_listings_active
    ON marketplace_listings(delisted_at)
    WHERE delisted_at IS NULL;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (124, 'marketplace_listings table');
