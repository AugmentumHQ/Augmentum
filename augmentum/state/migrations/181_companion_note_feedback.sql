-- 181_companion_note_feedback.sql
-- Aletheia × Augmentum arc, Sprint 7 Piece 15.
--
-- Records user actions on companion notes so the bias function can
-- learn what kind of findings resonate vs. get dismissed.
--
-- Kinds (the action endpoints write these):
--   surfaced       — user clicked "Pull it together" or opened a ref
--                     (route /api/companion/notes/{id}/surfaced)
--   acknowledged   — user clicked "Good to know"
--                     (route /api/companion/notes/{id}/acknowledged)
--   muted          — user clicked "Mute this topic"
--                     (route /api/companion/notes/{id}/muted_topic)
--
-- Future polish (deferred): dismissed_quickly (<5s glance), dwelled (>30s
-- linger), opened_ref (clicked a content_ref chip). The substrate is
-- here; the action endpoints just need to record the variant.
--
-- Bias function (companion_runtime/feedback.py): aggregates recent
-- feedback over a 14-day window, computes a multiplier in [0.5, 2.0]
-- per topic signature. The multiplier biases initiative scoring so
-- topics the user has engaged with get a boost, topics muted get
-- damped. Floored at 0.5 — never zero out (mute is the hard switch).

CREATE TABLE IF NOT EXISTS companion_note_feedback (
    id            INTEGER PRIMARY KEY,
    note_id       INTEGER NOT NULL,
    user_id       TEXT NOT NULL,
    companion_id  TEXT NOT NULL,
    kind          TEXT NOT NULL,                          -- surfaced|acknowledged|muted|...
    recorded_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Hot-path index: bias function reads recent rows by (user, kind).
CREATE INDEX IF NOT EXISTS idx_note_feedback_user_kind_time
    ON companion_note_feedback(user_id, companion_id, kind, recorded_at DESC);

-- Per-note index: Observatory shows "how was this note received"
CREATE INDEX IF NOT EXISTS idx_note_feedback_note
    ON companion_note_feedback(note_id, recorded_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (181, 'companion_note_feedback: user action persistence for bias learning');
