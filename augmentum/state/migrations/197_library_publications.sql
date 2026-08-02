-- 197_library_publications.sql
-- Save-to-Library Phase 1 — coder-built artifacts that survive workspace
-- teardown. One-way snapshot from a running coder preview into a
-- per-user catalog row + frozen storage dir under
-- {data_dir}/library_published/{user_id}/{publication_id}/.
--
-- Title is the soft natural key: re-saving with the same title bumps
-- `version` and replaces the storage dir atomically (last-write-wins).
-- A different title creates a new row; the previous publication stays.
--
-- Columns:
--   id                — pub_<random>; stable launch handle
--   user_id           — tenant scope; FK cascade matches existing user
--                        deletion cascade pattern in delete_user()
--   workspace_id      — source workspace at save time. Advisory only;
--                        the workspace may be deleted later and the
--                        publication must remain playable.
--   kind              — game | app | doc | other; UI sorts by this and
--                        the launcher picks a renderer
--   title             — user-supplied; (user_id, title) is the soft
--                        natural key. UI prevents collisions via
--                        overwrite-or-rename prompt
--   description       — optional one-liner shown on Library cards
--   screenshot_path   — relative within storage dir (e.g. "screenshot.png");
--                        "" means no screenshot, UI shows a placeholder
--   entry_point       — relative path to the file the launcher opens
--                        (e.g. "index.html" or "main.html"); derived
--                        from preview state at save time
--   storage_path      — absolute path to the publication dir on host
--                        (e.g. /data/library_published/<uid>/<pid>/)
--   storage_kind      — "bundle" (multi-file dir) or "single" (one file)
--                        controls how the launch route serves content
--   size_bytes        — sum of content/ + screenshot for budget check
--   version           — bumps on overwrite (same title save). v1 UI
--                        does not surface version history; reserved
--                        for future "View previous versions" feature
--   shared            — reserved for v3 marketplace; always 0 in v1
--   created_at        — first save (epoch seconds, REAL — matches
--                        other coder/journal tables)
--   updated_at        — most recent overwrite
--   last_launched_at  — bumped by launch route; powers "recently played"
--   launch_count      — bumped by launch route; powers ranking surfaces
--
-- ON DELETE CASCADE on user_id: harmless redundancy with the runtime
-- sweep in delete_user(), but mandatory per [[project_user_deletion_strands_data]].

CREATE TABLE IF NOT EXISTS library_publications (
    id                  TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    workspace_id        TEXT NOT NULL DEFAULT '',
    kind                TEXT NOT NULL,
    title               TEXT NOT NULL,
    description         TEXT NOT NULL DEFAULT '',
    screenshot_path     TEXT NOT NULL DEFAULT '',
    entry_point         TEXT NOT NULL,
    storage_path        TEXT NOT NULL,
    storage_kind        TEXT NOT NULL,
    size_bytes          INTEGER NOT NULL DEFAULT 0,
    version             INTEGER NOT NULL DEFAULT 1,
    shared              INTEGER NOT NULL DEFAULT 0,
    created_at          REAL NOT NULL,
    updated_at          REAL NOT NULL,
    last_launched_at    REAL,
    launch_count        INTEGER NOT NULL DEFAULT 0
);

-- Listing path: per-user, grouped by kind, newest first.
CREATE INDEX IF NOT EXISTS idx_library_publications_user_kind
    ON library_publications(user_id, kind, updated_at DESC);

-- Title-collision path: lookup (user_id, title) -> existing row.
CREATE INDEX IF NOT EXISTS idx_library_publications_user_title
    ON library_publications(user_id, title, version DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (197, 'library_publications: Save-to-Library Phase 1 catalog');
