-- 136_coder_profile.sql
-- Coder mode: cross-project preference store. Phase 2.3 of the
-- coder foundation (docs/superpowers/specs/2026-05-10-coder-foundation.md).
--
-- Records observations the agent makes about how the user works:
-- preferred conventions per language, per-project quirks, edit-pattern
-- accept/reject signals, tool preferences. Populated organically by
-- the retro loop (Phase 7); this migration ships only the schema +
-- CRUD surface so the data layer is ready when population lands.
--
-- workspace_id semantics:
--   - empty string ('') → global preference (applies to every workspace)
--   - non-empty         → workspace-local preference (overrides global)
--
-- Empty-string sentinel (instead of NULL) lets the UNIQUE constraint
-- treat (user_id, '', category, key) as a single row — SQLite's NULL
-- handling treats NULL as distinct, which would let multiple rows
-- collide on the same logical key.
--
-- Lookup pattern (workspace-aware):
--   SELECT ... FROM coder_profile
--   WHERE user_id = ? AND (workspace_id = ? OR workspace_id = '')
--   ORDER BY category, key,
--            CASE WHEN workspace_id = '' THEN 1 ELSE 0 END
-- Then dedupe in Python so workspace-local rows shadow global rows
-- per (category, key).

CREATE TABLE IF NOT EXISTS coder_profile (
    id                TEXT PRIMARY KEY,
    user_id           TEXT NOT NULL,
    workspace_id      TEXT NOT NULL DEFAULT '',
    category          TEXT NOT NULL,
    key               TEXT NOT NULL,
    value             TEXT NOT NULL DEFAULT '',
    confidence        REAL NOT NULL DEFAULT 0.5,
    observation_count INTEGER NOT NULL DEFAULT 1,
    last_observed_at  REAL NOT NULL,
    created_at        REAL NOT NULL,
    UNIQUE(user_id, workspace_id, category, key)
);

CREATE INDEX IF NOT EXISTS idx_coder_profile_user_workspace
    ON coder_profile(user_id, workspace_id);

CREATE INDEX IF NOT EXISTS idx_coder_profile_user_category
    ON coder_profile(user_id, category);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (136, 'coder_profile cross-project preference store');
