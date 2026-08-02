-- 157_companion_scene.sql
-- Persistent shared scene — commitment 2 (she lives somewhere).
--
-- One row per companion. The scene state is JSON for flexibility (room
-- layout, current_position, objects, ambient cues) but a few load-bearing
-- fields are denormalized as columns for cheap queries: location, posture,
-- last_seen_with. The XR scene client subscribes via the bus and
-- reconciles against this on session entry.
--
-- Ambient continuity (time-of-day drift, light shifts) is driven by the
-- runtime's tick scheduler on a slow cadence; updates write here and
-- emit scene_changed events on the bus.

CREATE TABLE IF NOT EXISTS companion_scene (
    companion_id      TEXT PRIMARY KEY REFERENCES companion_identities(companion_id),
    location          TEXT NOT NULL DEFAULT 'main_room',
    posture           TEXT NOT NULL DEFAULT 'idle',
    scene_blob        TEXT NOT NULL DEFAULT '{}',     -- JSON: room layout, objects, position
    last_seen_with    TEXT,                            -- last user_id who was co-present
    last_changed_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Seed Becca's scene so the runtime never boots into a missing row.
INSERT OR IGNORE INTO companion_scene (companion_id) VALUES ('becca');

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (157, 'companion_scene: persistent shared scene (commitment 2)');
