-- 149_epub_narration_progress_checkpoint.sql
-- Resumable narration synthesis: track how far a job got so a crash/restart
-- continues from the last finished chunk instead of from chapter one.
-- (ALTER ADD COLUMN is a no-op-on-exists in the migration runner.)

ALTER TABLE epub_narrations ADD COLUMN processed_chunks INTEGER NOT NULL DEFAULT 0;
ALTER TABLE epub_narrations ADD COLUMN total_chunks INTEGER NOT NULL DEFAULT 0;
