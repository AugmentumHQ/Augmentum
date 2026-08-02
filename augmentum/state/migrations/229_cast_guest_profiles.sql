-- Cast couch co-op guest profiles (Phase 2).
--
-- Spec: docs/superpowers/specs/2026-06-02-cast-couch-coop-design.md
--
-- Guests of a host's co-op session get named identities under the
-- host's account — alice and bob can rejoin and the system remembers
-- their colour, last visit time, and (Phase 4) their save data.
--
-- User-scoped on host_user_id per CLAUDE.md's multi-tenant pattern:
--   * a guest profile belongs to the host, not to alice herself
--   * the host's account-delete cascade tears down everyone's profile
--     via ON DELETE CASCADE
--   * cross-host portability is explicitly out of scope (see spec)
--
-- Phase 3 adds guest_devices (FK to this table) for fingerprint-based
-- auto-reconnect. Phase 4 adds game_saves.guest_profile_id.

CREATE TABLE IF NOT EXISTS guest_profiles (
    id              TEXT PRIMARY KEY,            -- gp_<token12>
    host_user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    display_name    TEXT NOT NULL,
    -- Colour for the on-TV player chip + cast-control roster strip.
    -- Empty = auto-assign at render time (deterministic by id hash).
    color           TEXT NOT NULL DEFAULT '',
    created_at      INTEGER NOT NULL,            -- unix seconds
    last_seen_at    INTEGER NOT NULL,
    play_count      INTEGER NOT NULL DEFAULT 0,
    -- Per-host name uniqueness (not global). "alice" at one host
    -- and "alice" at another host can be different people.
    UNIQUE (host_user_id, display_name)
);

CREATE INDEX IF NOT EXISTS idx_guest_profiles_host
    ON guest_profiles (host_user_id);
CREATE INDEX IF NOT EXISTS idx_guest_profiles_last_seen
    ON guest_profiles (host_user_id, last_seen_at DESC);
