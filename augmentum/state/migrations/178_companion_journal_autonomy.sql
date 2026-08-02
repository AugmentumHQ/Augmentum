-- 178_companion_journal_autonomy.sql
-- Substrate for Piece 9' (revisit_thread performer) + Piece 10' (note pip).
-- All three columns extend companion_journal (mig 154) and are nullable
-- or zero-defaulted so existing rows read cleanly without backfill.
--
-- Resource posture for these columns:
--
--   revisited_at  — Piece 9' per-thread cooldown. Performer SELECTs
--                   the next thread WHERE revisited_at IS NULL OR
--                   revisited_at < datetime('now', '-6 hours'). Without
--                   this column the performer would re-resolve the
--                   same thread on every tick — an unbounded resolver
--                   call loop. The index on (companion_id, revisited_at)
--                   makes the "due for revisit" scan cheap.
--
--   quiet_share_ready — Piece 10' pip eligibility flag. The expression
--                   channel reads `WHERE quiet_share_ready = 1 AND
--                   surfaced_at IS NULL`. INTEGER DEFAULT 0 means
--                   existing rows are inert until explicitly marked.
--
--   surfaced_at   — Piece 10' user-encounter timestamp. NULL until
--                   the user actually opens the note pip; then the
--                   row stops appearing in the "ready" feed.
--
-- All three are idempotent additions (catch-able as "duplicate column"
-- by the runner, see sqlite.py:1261).

ALTER TABLE companion_journal ADD COLUMN revisited_at      TEXT;
ALTER TABLE companion_journal ADD COLUMN quiet_share_ready INTEGER NOT NULL DEFAULT 0;
ALTER TABLE companion_journal ADD COLUMN surfaced_at       TEXT;

-- Index for the Piece 9' "thread due for revisit" scan. Partial index
-- — we only care about rows that haven't been revisited recently, so
-- the index doesn't bloat with the entire journal.
CREATE INDEX IF NOT EXISTS idx_cj_revisit_due
    ON companion_journal(companion_id, created_at DESC)
    WHERE revisited_at IS NULL
      AND COALESCE(suppressed, 0) = 0
      AND entry_type IN ('wondering', 'unfinished');

-- Index for the Piece 10' "notes ready to surface" scan.
CREATE INDEX IF NOT EXISTS idx_cj_quiet_share_ready
    ON companion_journal(companion_id, created_at DESC)
    WHERE quiet_share_ready = 1
      AND surfaced_at IS NULL;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (178, 'companion_journal autonomy substrate: revisited_at + quiet_share_ready + surfaced_at');
