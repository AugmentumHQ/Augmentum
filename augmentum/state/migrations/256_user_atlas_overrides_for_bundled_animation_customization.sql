-- 256_user_atlas_overrides_for_bundled_animation_customization.sql
-- user_atlas_overrides — per-user customization of BUNDLED anim-atlas
-- entries (the curated VRMA/BVH registry that ships in code at
-- ui/scripts/anim-atlas.js).
--
-- User-UPLOADED animations already carry their own editable metadata
-- in user_animations (migration 203). Bundled entries had no per-user
-- story: they couldn't be removed from the selection pool or re-tagged
-- when a user disagreed with the curation. This table closes that gap
-- without forking the atlas — the client merges these rows over the
-- bundled entries at registry load.
--
-- Shape: one row per (user, atlas_id) the user has touched. Untouched
-- bundled entries have no row (zero storage for defaults).
--   disabled — 1 removes the entry from auto-selection AND pickers
--              (re-enable by setting back to 0 or deleting the row).
--   patch    — JSON object of atlas-field overrides (roles, emotion,
--              modes, cost, cooldown, loop, explicitOnly, notes, ...).
--              Merged shallowly over the bundled entry client-side.
--              NULL/'{}' = no metadata change (disable-only row).
--
-- atlas_id is TEXT and intentionally NOT a foreign key — the bundled
-- registry lives in JS, not in a table. Stale ids (a bundled entry
-- renamed/removed upstream) are harmless: the merge skips unknown ids.

CREATE TABLE IF NOT EXISTS user_atlas_overrides (
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    atlas_id   TEXT NOT NULL,
    disabled   INTEGER NOT NULL DEFAULT 0,
    patch      TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, atlas_id)
);

CREATE INDEX IF NOT EXISTS idx_user_atlas_overrides_user
    ON user_atlas_overrides(user_id);
