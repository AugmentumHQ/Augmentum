-- 126_controller_remaps.sql
-- Per-user controller layouts for the AXF controller framework.
-- One row per (user, system) pair stores the user's binding overrides
-- on top of the canonical system defaults shipped in
-- augmentum/controllers/defaults.py. Resolved layout = defaults merged
-- with any non-null override entries.
--
-- The bindings_json blob is a small dict: { logical_action: {keyboard,
-- gamepad_button, gamepad_axis, ...} }. Schema is enforced by the
-- service layer, not the database, so adding a new system or a new
-- input source doesn't need a migration.

CREATE TABLE IF NOT EXISTS controller_remaps (
    user_id         TEXT NOT NULL REFERENCES users(id),
    system_id       TEXT NOT NULL,                              -- 'nes' / 'snes' / 'gba' / ...
    bindings_json   TEXT NOT NULL DEFAULT '{}',                 -- partial override, JSON
    pad_routing     TEXT NOT NULL DEFAULT 'index',              -- 'index' | 'firstpress' (P1/P2 routing strategy)
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, system_id)
);

CREATE INDEX IF NOT EXISTS idx_controller_remaps_user
    ON controller_remaps(user_id, updated_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (126, 'controller_remaps table');
