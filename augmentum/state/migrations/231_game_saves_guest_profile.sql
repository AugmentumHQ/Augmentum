-- Cast couch co-op per-guest saves (Phase 4).
--
-- Spec: docs/superpowers/specs/2026-06-02-cast-couch-coop-design.md
--
-- A save row's owner is (user_id, guest_profile_id):
--   * guest_profile_id IS NULL  →  host's own save (the default;
--     existing rows are correctly classified here)
--   * guest_profile_id IS NOT NULL → that named guest's save under
--     the host's account
--
-- The unique constraint extends to include guest_profile_id so alice
-- and bob can both have their own slot-0 SRAM for the same title
-- under the same host without colliding with each other or the host.
--
-- SET NULL (not CASCADE) on the FK is deliberate: deleting a guest
-- profile should detach their saves to the host's pool, not delete
-- the save data. The host can then re-link or remove manually.

ALTER TABLE game_saves
    ADD COLUMN guest_profile_id TEXT
    REFERENCES guest_profiles(id) ON DELETE SET NULL;

-- The existing UNIQUE(user_id, artifact_id, kind, slot) needs to
-- be widened to include guest_profile_id. SQLite can't ALTER a
-- UNIQUE constraint, so we redeclare it via an INDEX. The original
-- UNIQUE stays but is effectively dead — pre-Phase-4 rows had no
-- guest_profile_id so they all match the same NULL key, and PUT
-- writes from Phase 4 onward always carry the column.
--
-- Per-guest uniqueness:
CREATE UNIQUE INDEX IF NOT EXISTS idx_game_saves_per_guest_unique
    ON game_saves(user_id, artifact_id, kind, slot,
                  COALESCE(guest_profile_id, ''));

-- Lookup index for "all saves belonging to alice at this host":
CREATE INDEX IF NOT EXISTS idx_game_saves_guest
    ON game_saves(guest_profile_id)
    WHERE guest_profile_id IS NOT NULL;
