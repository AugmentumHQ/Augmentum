-- 184_companion_drive_state.sql
-- Aletheia × Augmentum arc, Sprint 6 Piece 4.
--
-- Per-(user, companion) state for the four drives:
--   curiosity   — appetite for novelty / open questions
--   competence  — appetite for finishing things / mastery
--   connection  — appetite for user presence / shared attention
--   rest        — appetite for low-activity periods
--
-- Each drive has a ``level`` in [0, 1] (current intensity) and
-- ``last_satiated_at`` (when something in that drive's tendency was
-- last performed). The runtime decays levels back toward baseline at
-- a configurable half-life (default 4h). Drive urgency at scoring time
-- is ``level × (1 - predicted_satiation_from_last_action)``.
--
-- A single row per (user, companion) holds all four drives as columns.
-- Cheaper than a (user, companion, drive_name) shape and the drive set
-- is fixed (changing it requires a migration + code change anyway).

CREATE TABLE IF NOT EXISTS companion_drive_state (
    user_id                TEXT NOT NULL,
    companion_id           TEXT NOT NULL,
    curiosity_level        REAL NOT NULL DEFAULT 0.6,
    competence_level       REAL NOT NULL DEFAULT 0.5,
    connection_level       REAL NOT NULL DEFAULT 0.6,
    rest_level             REAL NOT NULL DEFAULT 0.4,
    curiosity_satiated_at  TEXT,
    competence_satiated_at TEXT,
    connection_satiated_at TEXT,
    rest_satiated_at       TEXT,
    last_decay_at          TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, companion_id)
);

CREATE INDEX IF NOT EXISTS idx_companion_drive_state_user
    ON companion_drive_state(user_id, companion_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (184, 'companion_drive_state: 4-drive per-user state (curiosity/competence/connection/rest)');
