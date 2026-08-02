-- 125_game_saves.sql
-- Per-title save data (SRAM, save states, screenshots) for the
-- Augmentum Experience Framework. Engine-agnostic -- the same table
-- serves browser-WASM emulators, server-streamed RetroArch, and any
-- future runtime that produces savable state.
--
-- Save data lives in the existing blob store (refcount-tracked,
-- dedup'd, sha256-addressed). This table is the *index* over those
-- blobs: which save belongs to which title/user/slot, with a
-- per-(user,title,kind,slot) UNIQUE so PUT semantics are clean.
--
-- core_id is the libretro core that produced a state save. SRAM is
-- core-agnostic (raw cartridge memory); states are not. We refuse to
-- load a state into a different core_id at the runtime layer.

CREATE TABLE IF NOT EXISTS game_saves (
    id              TEXT PRIMARY KEY,
    user_id         TEXT NOT NULL REFERENCES users(id),
    artifact_id     TEXT NOT NULL,                          -- the title's artifact id (loose ref)
    core_id         TEXT NOT NULL DEFAULT '',               -- 'fceumm' / 'snes9x' / '' for SRAM
    kind            TEXT NOT NULL,                          -- 'sram' | 'state' | 'screenshot'
    slot            INTEGER NOT NULL DEFAULT 0,             -- 0 = quicksave/SRAM, 1..N manual states
    sha256          TEXT NOT NULL,                          -- → blobs table
    size_bytes      INTEGER NOT NULL,
    label           TEXT NOT NULL DEFAULT '',               -- user-named slot
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, artifact_id, kind, slot)
);

CREATE INDEX IF NOT EXISTS idx_game_saves_user
    ON game_saves(user_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_game_saves_artifact
    ON game_saves(artifact_id, kind, slot);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (125, 'game_saves table');
