-- 182_companion_journal_resilience.sql
-- Aletheia × Augmentum arc, Sprint 1 (R1).
--
-- Adds resilience columns to companion_journal so every subsequent
-- autonomous write carries provenance, validation, and quarantine
-- capability. This is the load-bearing sequencing decision in the
-- design doc (§5 R1 Piece + §6 resilience model): every entry written
-- from Sprint 2 onward goes through safe_journal which uses these
-- columns to flag bad writes without losing the forensic record.
--
-- Column-by-column rationale (anchor doc §6 healing principles):
--
--   source              — where the write came from. Values:
--                          'autonomous' | 'reflection' | 'user_direct'
--                          | 'observer' | 'synthesize' | 'imported'
--                          | 'test'. Lets healing jobs scope retroactive
--                          checks to specific paths (e.g. re-validate
--                          all 'synthesize' entries from a deprecated
--                          model).
--
--   model_used          — model name + version that produced the
--                          content, when applicable. NULL for non-LLM
--                          writes. The basis for model-swap forensics
--                          ("which entries were written by the
--                          old utility tier?").
--
--   confidence_numeric  — continuous confidence in [0, 1]. Discrete
--                          'confidence' column (mig 161) stays for
--                          back-compat. Map: early=0.3, normal=0.6,
--                          firm=0.9. Continuous lets healing apply
--                          forgetting curve (× 0.99/30d) precisely.
--
--   validation_score    — output of safe_journal's validator on write.
--                          1.0 = passed cleanly; lower = penalties for
--                          length / repetition / suspect patterns.
--                          Updated retroactively by daily heal.
--
--   quarantined         — flag. When 1, the row is excluded from every
--                          downstream loop (revisit_thread, pre-context
--                          injection, dream cycle input). Original
--                          content preserved for forensics.
--
--   quarantine_reason   — short tag. 'adversarial_pattern' |
--                          'bad_refs' | 'low_quality' |
--                          'self_contradiction' | 'failed_corroboration'
--                          | 'user_correction' | 'model_drift'.
--
--   crystallized        — flag. When 1, the row is immune to forgetting
--                          curves + auto-archive + quarantine. Used for
--                          pinned milestones + user-corrected entries.
--                          Crystallized is a one-way commitment;
--                          un-crystallizing is a separate gesture.
--
--   archived_at         — when this entry was consolidated into a
--                          window summary (companion_journal_archive,
--                          mig 183). NULL means still active.

ALTER TABLE companion_journal ADD COLUMN source             TEXT NOT NULL DEFAULT 'autonomous';
ALTER TABLE companion_journal ADD COLUMN model_used         TEXT;
ALTER TABLE companion_journal ADD COLUMN confidence_numeric REAL NOT NULL DEFAULT 0.6;
ALTER TABLE companion_journal ADD COLUMN validation_score   REAL NOT NULL DEFAULT 1.0;
ALTER TABLE companion_journal ADD COLUMN quarantined        INTEGER NOT NULL DEFAULT 0;
ALTER TABLE companion_journal ADD COLUMN quarantine_reason  TEXT;
ALTER TABLE companion_journal ADD COLUMN crystallized       INTEGER NOT NULL DEFAULT 0;
ALTER TABLE companion_journal ADD COLUMN archived_at        TEXT;

-- Indexes for the healing-job hot paths.

CREATE INDEX IF NOT EXISTS idx_cj_quarantined_user
    ON companion_journal(companion_id, user_id, quarantined, created_at DESC)
    WHERE quarantined = 1;

CREATE INDEX IF NOT EXISTS idx_cj_crystallized_user
    ON companion_journal(companion_id, user_id, crystallized, created_at DESC)
    WHERE crystallized = 1;

-- Active-entries index: the hot read path used by revisit_thread +
-- pre-context injection. Partial index over the dominant case (not
-- quarantined, not archived) so the scan is bounded.
CREATE INDEX IF NOT EXISTS idx_cj_active_user
    ON companion_journal(companion_id, user_id, created_at DESC)
    WHERE quarantined = 0 AND archived_at IS NULL;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (182, 'companion_journal resilience: provenance, validation, quarantine, crystallization');
