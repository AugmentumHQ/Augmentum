-- 140_coder_workspaces_tooling_profile.sql
-- Persist the workspace tooling profile selected at creation time.
--
-- Existing workspaces are Standard. New workspaces can opt into Power or
-- Browser/Test provisioning without mutating active containers in place.

ALTER TABLE coder_workspaces
    ADD COLUMN tooling_profile TEXT NOT NULL DEFAULT 'standard';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (140, 'Coder workspace tooling profile');
