-- Transient artifacts: ephemeral cache entries (e.g. image_search thumbnails)
-- that must NOT appear in the user's Files browser and SHOULD be evicted
-- periodically by age/size caps. See augmentum/jobs/handlers/evict_transient_artifacts.py.
--
-- Why a column on artifacts rather than a new table:
--   * download URL (/api/artifacts/{id}/download) keeps working unchanged
--   * ArtifactStore already handles dir layout + persistence
--   * VFS registration is the one thing we need to skip

ALTER TABLE artifacts ADD COLUMN transient INTEGER NOT NULL DEFAULT 0;

-- Eviction sweep scans by (transient, created_at); keep it cheap.
CREATE INDEX IF NOT EXISTS idx_artifacts_transient_created
    ON artifacts(transient, created_at)
    WHERE transient = 1;

INSERT OR IGNORE INTO schema_version (version, description)
    VALUES (103, 'Transient artifact flag for image_search cache');
