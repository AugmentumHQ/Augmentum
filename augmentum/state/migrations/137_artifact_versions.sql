-- 137_artifact_versions.sql
-- Version history for application-builder artifacts.
--
-- Each successful build (initial or iterate) snapshots the project's
-- file set into artifact_versions before the new content overwrites
-- the artifact row. The workspace UI exposes "view history" / "revert"
-- by reading from this table; iteration mode also reuses snapshots
-- to drive diff views.
--
-- File contents live in `files_json` (a JSON array of {path, role,
-- content}) rather than as zip blobs because:
--   1. The version is shown in the diff viewer, which needs text not
--      bytes,
--   2. Build artifacts are typically small (a few KB),
--   3. We want zero-cost revert without re-parsing a zip.
--
-- The user_id column lets us scope reads exactly like every other
-- artifact-adjacent table per the multi-tenant rule in CLAUDE.md.

CREATE TABLE IF NOT EXISTS artifact_versions (
    id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL,
    version_index INTEGER NOT NULL,
    label TEXT,
    files_json TEXT NOT NULL,
    file_count INTEGER NOT NULL DEFAULT 0,
    score REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    user_id TEXT NOT NULL,
    FOREIGN KEY (artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE
);

-- Cover the two read patterns: (a) list versions for an artifact in
-- order, (b) enforce per-artifact uniqueness on version_index.
CREATE UNIQUE INDEX IF NOT EXISTS idx_artifact_versions_artifact_index
    ON artifact_versions(artifact_id, version_index);
CREATE INDEX IF NOT EXISTS idx_artifact_versions_user
    ON artifact_versions(user_id, created_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (137, 'Version history for application-builder artifacts');
