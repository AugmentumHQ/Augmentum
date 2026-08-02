-- 235_library_collections_activity_and_tags.sql
-- Library substrate (Steam-style):
--   * library_collections        — user-defined groupings (manual + dynamic)
--   * library_collection_items   — N:M for manual collections
--   * library_activity           — per-item event timeline (open/cast/edit/pin)
--   * artifacts.tags             — inline-editable tag list, JSON array
--
-- Pin + last-opened already live on artifacts (mig 057); no duplicate substrate.

CREATE TABLE IF NOT EXISTS library_collections (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    slug            TEXT NOT NULL DEFAULT '',           -- url-safe, unique-per-user
    kind            TEXT NOT NULL DEFAULT 'manual',     -- 'manual' | 'dynamic'
    filter_json     TEXT NOT NULL DEFAULT '{}',         -- dynamic only: {tags, types, since, ...}
    cover_url       TEXT NOT NULL DEFAULT '',
    accent_color    TEXT NOT NULL DEFAULT '',           -- '#rrggbb' (atelier tints)
    view_mode       TEXT NOT NULL DEFAULT 'list',       -- 'list' | 'grid' | 'cover'
    sort_order      INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_library_collections_user
    ON library_collections(user_id, sort_order);

CREATE UNIQUE INDEX IF NOT EXISTS idx_library_collections_slug
    ON library_collections(user_id, slug);

CREATE TABLE IF NOT EXISTS library_collection_items (
    collection_id   TEXT NOT NULL REFERENCES library_collections(id) ON DELETE CASCADE,
    artifact_id     TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    sort_order      INTEGER NOT NULL DEFAULT 0,
    added_at        TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (collection_id, artifact_id)
);

CREATE INDEX IF NOT EXISTS idx_library_collection_items_artifact
    ON library_collection_items(artifact_id);

CREATE INDEX IF NOT EXISTS idx_library_collection_items_user
    ON library_collection_items(user_id);

CREATE TABLE IF NOT EXISTS library_activity (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    artifact_id     TEXT NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
    action          TEXT NOT NULL,                      -- 'open' | 'cast' | 'edit' | 'pin' | 'unpin'
    surface         TEXT NOT NULL DEFAULT '',           -- 'desktop' | 'mobile' | 'tv' | 'cast'
    payload         TEXT NOT NULL DEFAULT '{}',         -- JSON: receiver_id, target_url, etc.
    occurred_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_library_activity_user_time
    ON library_activity(user_id, occurred_at DESC);

CREATE INDEX IF NOT EXISTS idx_library_activity_artifact_time
    ON library_activity(artifact_id, occurred_at DESC);

-- Tags: JSON array of strings. Stored on artifacts so a single SELECT pulls
-- everything the list-row needs. Dynamic collections filter by tag via
-- JSON_EACH at query time -- fine for the cardinality the Library deals with
-- (low thousands of artifacts per user, tens of tags per artifact).
ALTER TABLE artifacts ADD COLUMN tags TEXT NOT NULL DEFAULT '[]';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (235, 'Library collections, activity timeline, and artifact tags');
