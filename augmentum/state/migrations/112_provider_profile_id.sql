-- Persist the profile selection a user makes when adding a provider.
--
-- Without this column, the runtime had no way to know that a provider
-- the user named "vim" or "my-nvidia" actually targets NVIDIA's NIM
-- API, so post-processing rules (e.g. NVIDIA's "system message must be
-- at the beginning") never fired and narrative-mode requests 400'd.
--
-- New rows store the profile_id picked at create time. Legacy rows
-- (NULL profile_id) are matched at load time by URL pattern, so this
-- migration needs no data backfill.

ALTER TABLE providers ADD COLUMN profile_id TEXT NOT NULL DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (112, 'providers.profile_id for post-processing rules');
