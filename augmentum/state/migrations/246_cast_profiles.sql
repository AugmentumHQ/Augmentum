-- 246_cast_profiles.sql
--
-- Per-(user, title) cast profile. Records which CastStrategy + adapter
-- chain to use when casting a given title to a TV, plus any per-game
-- quirks the proxy needs to apply.
--
-- Lookup is always user-scoped: a host's override and a guest's auto-
-- bootstrapped profile must not collide. The (user_id, title_id) pair
-- is the natural PK.
--
-- ``strategy`` is one of:
--   ``shim``         — same-origin /ui/play/ + universal adapter
--   ``proxy``        — Phase-3 origin-proxy strategy
--   ``containerized``— Phase-5 headless-Chromium + AGSP stream
--
-- ``input_chain`` is a JSON array of adapter ids in fallback order, e.g.
--   ``["gamepad_api"]`` (default)
--   ``["keyboard"]``                    (keyboard-only game)
--   ``["gamepad_api", "keyboard"]``     (try gamepad first, fall back)
--
-- ``keymap_json`` is the serialised KeymapProfile blob — empty string
-- means "use the adapter's built-in default."
--
-- ``quirks_json`` is the open quirks bag (CSP allowances, asset rewrite
-- skips, service worker mode, etc). Forward-compat: readers ignore
-- unknown keys; writers can add new keys without a migration.
--
-- ``classified_by`` provenance:
--   ``default``  — never classified; fell back to defaults at cast time
--   ``probe``    — Phase-4 Playwright probe wrote this entry
--   ``manual``   — user override from library2 detail-pane
--   ``telemetry``— Phase-4 demotion loop rewrote after live failure
--
-- See spec: docs/superpowers/specs/2026-06-04-universal-cast-pipeline-design.md

CREATE TABLE IF NOT EXISTS cast_profiles (
    user_id              TEXT NOT NULL DEFAULT '',
    title_id             TEXT NOT NULL,
    strategy             TEXT NOT NULL DEFAULT 'shim',
    embed_url            TEXT NOT NULL DEFAULT '',
    container_profile_id TEXT NOT NULL DEFAULT '',
    input_chain          TEXT NOT NULL DEFAULT '["gamepad_api"]',
    keymap_json          TEXT NOT NULL DEFAULT '',
    quirks_json          TEXT NOT NULL DEFAULT '{}',
    classified_by        TEXT NOT NULL DEFAULT 'default',
    classified_at        REAL NOT NULL DEFAULT 0,
    failed_at            REAL NOT NULL DEFAULT 0,
    notes                TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (user_id, title_id)
) STRICT;

-- Reverse lookup: surface "which titles has this user customised" in
-- the library2 settings view. Cheap secondary index.
CREATE INDEX IF NOT EXISTS idx_cast_profiles_classified_at
    ON cast_profiles(user_id, classified_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (246, 'cast_profiles per-(user,title) strategy + input chain');
