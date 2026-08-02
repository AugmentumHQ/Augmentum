-- Migration 159 — Sprint 4b skill archive.
--
-- Append-only store of dispatch outcomes for DPO-style preference
-- retrieval. Each row captures:
--   * the *context* in which a dispatch decision was made (embedded
--     intent + state summary)
--   * which subagent was chosen
--   * the derived outcome signal in [-1, +1]
--   * timestamp
--
-- The signal is derived (not labeled by a human): did the bus emit
-- ``subagent.completed`` with no error event in the following 60s?
-- Did the user issue a corrective utterance in the following 5 min?
-- Sprint 4b's :func:`record_outcome` writes the row; Sprint 4b's
-- DPO retrieval reads it at dispatch time.
--
-- No fine-tuning, no model weights change — this is retrieval-augmented
-- preference (sprint plan §7).

CREATE TABLE IF NOT EXISTS companion_skill_archive (
    id INTEGER PRIMARY KEY,
    companion_id TEXT NOT NULL,
    ts REAL NOT NULL,
    intent_text TEXT NOT NULL,
    intent_source TEXT NOT NULL DEFAULT 'user_chat',
    context_embedding BLOB,
    chosen_subagent TEXT NOT NULL,
    outcome_signal REAL NOT NULL DEFAULT 0.0,
    outcome_reason TEXT NOT NULL DEFAULT '',
    decision_ms REAL NOT NULL DEFAULT 0.0,
    used_tiebreaker INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_skill_archive_companion_ts
    ON companion_skill_archive(companion_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_skill_archive_subagent
    ON companion_skill_archive(companion_id, chosen_subagent);
