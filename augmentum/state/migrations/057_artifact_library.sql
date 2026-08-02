-- 057_artifact_library.sql
-- Library: pinned flag and last-opened tracking for artifacts

ALTER TABLE artifacts ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;
ALTER TABLE artifacts ADD COLUMN last_opened_at TEXT DEFAULT NULL;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (57, 'Library: pinned flag and last-opened tracking for artifacts');
