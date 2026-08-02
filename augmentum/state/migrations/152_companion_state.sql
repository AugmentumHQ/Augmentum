-- 152_companion_state.sql
-- 3-axis state machine current state per companion.
--
-- The runtime's state machine has three orthogonal axes (per design spec
-- v2 section 5): state (asleep|dormant|present), role (3-vector of
-- active/passive/reflective summing to ~1.0), focus (none|personal:X|
-- shared:X|executive:X|social:X). One row per companion; updated in
-- place. Transition history lives in companion_state_log (153).
--
-- Default boot state matches the design spec: dormant + passive-dominant
-- + no focus. The runtime restores from the latest checkpoint within 5
-- minutes on cold start, else uses these defaults.

CREATE TABLE IF NOT EXISTS companion_state (
    companion_id      TEXT PRIMARY KEY REFERENCES companion_identities(companion_id),

    -- Axis 1: discrete state. Cooldown 2.0s.
    state             TEXT NOT NULL DEFAULT 'dormant',     -- asleep|dormant|present

    -- Axis 2: role as soft 3-vector (sum ≈ 1.0 ± epsilon). Cooldown 0.4s.
    -- Dominant role is argmax. Stored as three columns so callers can
    -- branch on dominant or use full vector for scaled behavior.
    role_active       REAL NOT NULL DEFAULT 0.0,
    role_passive      REAL NOT NULL DEFAULT 1.0,
    role_reflective   REAL NOT NULL DEFAULT 0.0,

    -- Axis 3: discrete focus with opaque payload. Cooldown 0.8s.
    -- Format: 'none' or '<kind>:<payload>' (e.g. 'shared:matt_writing').
    focus             TEXT NOT NULL DEFAULT 'none',

    -- Per-axis entry timestamps drive cooldown enforcement.
    entered_state_at  TEXT NOT NULL DEFAULT (datetime('now')),
    entered_role_at   TEXT NOT NULL DEFAULT (datetime('now')),
    entered_focus_at  TEXT NOT NULL DEFAULT (datetime('now')),

    updated_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Seed Becca's initial state explicitly so the runtime never boots into
-- a missing row.
INSERT OR IGNORE INTO companion_state (companion_id) VALUES ('becca');

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (152, 'companion_state: 3-axis state machine current state');
