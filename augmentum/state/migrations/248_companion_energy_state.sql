-- 248_companion_energy_state.sql
-- Companion verbs architecture, Phase 3b.
--
-- Per-(user, companion) "energy" axis — a single scalar in [0, 1]
-- that decays toward a baseline over time and gets spent by activity.
-- Distinct from the drive levels (companion_drive_state, mig 184):
-- drives are appetites (curiosity, competence, connection, rest);
-- energy is the moment-to-moment "how depleted am I" capacity that
-- gates how much any drive can pull.
--
-- Also distinct from the companion_economy berry/mana (mig 230s) —
-- economy is motivation/earned, energy is physiological/regenerated.
--
-- Minimal substrate for the Phase 3b worked example. A future
-- multi-axis version (cognitive/social/creative) is possible without
-- breaking this row layout — just add columns.
--
-- Baseline default 0.6: at rest, she's at 60% energy and decays
-- toward that floor whether high or low. activity_selector can read
-- ``level`` as an additional scoring multiplier in a future phase
-- (Phase 3c+); this migration just creates the substrate.

CREATE TABLE IF NOT EXISTS companion_energy_state (
    user_id        TEXT NOT NULL,
    companion_id   TEXT NOT NULL,
    energy_level   REAL NOT NULL DEFAULT 0.6,
    baseline_level REAL NOT NULL DEFAULT 0.6,
    last_decay_at  TEXT NOT NULL DEFAULT (datetime('now')),
    last_spend_at  TEXT,
    PRIMARY KEY (user_id, companion_id)
);

CREATE INDEX IF NOT EXISTS idx_companion_energy_state_user
    ON companion_energy_state(user_id, companion_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (248, 'companion_energy_state: per-user energy axis for tick_energy verb');
