-- 054_artifact_source.sql
-- Store structured source data for artifact editing.
-- For documents: the sections array. For slides: the slides array.
-- For spreadsheets: the sheets array. For charts: the config object.
-- NULL for legacy artifacts created before this migration.

ALTER TABLE artifacts ADD COLUMN source_json TEXT DEFAULT NULL;
