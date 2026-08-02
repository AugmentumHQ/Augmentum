-- 059_skipped.sql
--
-- Placeholder migration — number 059 was never used.
--
-- Why this file exists:
--   058_dream_system.sql and 060_avatars.sql were authored on parallel
--   feature branches and merged in close succession. Both authors
--   independently picked the next-available number from main; the
--   second-merged branch was renumbered to 060 to avoid collision,
--   leaving 059 unused. The migration runner executes files in
--   alphabetical order, so the gap is harmless at runtime — but the
--   wiring validator flags it as suspicious. This no-op file makes
--   the sequence contiguous so the warning goes away.
--
-- Do not remove. Adding `CREATE TABLE IF NOT EXISTS` etc. here would
-- re-run on every deployment, even though the runner only applies new
-- files. The empty file is intentional.

SELECT 1;  -- no-op
