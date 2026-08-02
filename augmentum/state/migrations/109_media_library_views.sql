-- Discovered media-server libraries/views plus user overrides.
--
-- Emby and Jellyfin expose native top-level libraries ("Movies", "Anime",
-- "TV shows", custom user names, etc.) that need to be classified by
-- metadata and then optionally remapped or hidden by the user. This table
-- preserves the provider-native identity while giving Augmentum a stable
-- place to store generic presentation overrides.

CREATE TABLE IF NOT EXISTS media_library_views (
    id                       TEXT PRIMARY KEY,
    user_id                  TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    server_id                TEXT NOT NULL REFERENCES user_media_servers(id) ON DELETE CASCADE,
    provider                 TEXT NOT NULL,
    provider_library_id      TEXT NOT NULL,
    provider_name            TEXT NOT NULL,
    provider_view_type       TEXT NOT NULL DEFAULT '',
    provider_collection_type TEXT NOT NULL DEFAULT '',
    detected_group           TEXT NOT NULL DEFAULT '',
    detected_primary_entity  TEXT NOT NULL DEFAULT '',
    detection_confidence     REAL NOT NULL DEFAULT 0.0,
    sample_type_counts       TEXT NOT NULL DEFAULT '{}',
    sample_notes             TEXT NOT NULL DEFAULT '{}',
    display_name_override    TEXT NOT NULL DEFAULT '',
    surface_group_override   TEXT NOT NULL DEFAULT '',
    is_hidden                INTEGER NOT NULL DEFAULT 0,
    include_in_search        INTEGER NOT NULL DEFAULT 1,
    include_in_overview      INTEGER NOT NULL DEFAULT 1,
    sort_order               INTEGER NOT NULL DEFAULT 0,
    last_seen_at             TEXT,
    created_at               TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at               TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, server_id, provider_library_id)
);

CREATE INDEX IF NOT EXISTS idx_media_library_views_server
    ON media_library_views(user_id, server_id);

CREATE INDEX IF NOT EXISTS idx_media_library_views_surface
    ON media_library_views(user_id, server_id, detected_group);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (109, 'media_library_views table for discovered remote libraries and user overrides');
