-- 251_repair_artifacts_corrupt_user_id.sql
--
-- Historical migration: previously contained a deployment-specific repair
-- for two artifact rows whose user_id columns were corrupted by an April
-- 2026 writer bug. The repair targeted user_id values present on the
-- maintainer's deployment and is not applicable to OSS installs.
--
-- Replaced with a no-op so the migration sequence stays contiguous;
-- the underlying writer bug is fixed upstream.

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (251, 'No-op (was: deployment-specific artifact user_id repair)');
