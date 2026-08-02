-- Harness memory per-project scoping (see augmentum/proxy/harness.py).
--
-- The flat "harness" scope pooled all projects AND all harnesses per user,
-- so conventions learned in one project bled into every other. New layout:
--   harness                       — global seeds only (source_type='system')
--   harness:<harness>:<project>   — learned/harvested, per harness+project
--   harness:default               — shared fallback (no X-Augmentum-Project
--                                   header) AND home of the legacy pool
--
-- Move every legacy learned/harvested memory to the shared default project
-- scope. Nothing is deleted; the seeded defaults (source_type='system')
-- stay in the global scope so every project keeps the baseline conventions.
UPDATE memories
SET scope = 'harness:default'
WHERE scope = 'harness'
  AND (source_type IS NULL OR source_type != 'system');
