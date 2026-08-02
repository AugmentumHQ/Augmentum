-- 223_game_stream_paused_at.sql
-- (renamed from 221 to resolve collision with 221_notification_substrate.sql —
--  both files were sharing version 221, which the runner reads from the
--  filename prefix in sqlite.py:1469. Filename + internal schema_version
--  INSERT now agree at 223.)
-- Tracks when a session transitioned into PAUSED so the sweep loop can
-- enforce paused_stop_seconds (auto-stop a session that's been frozen
-- too long). Cleared back to NULL on RESUME.
--
-- The PAUSED status value itself doesn't require a schema change — the
-- status column is already TEXT — but the timestamp does. Stays NULL
-- for any session not currently paused.

ALTER TABLE game_stream_sessions ADD COLUMN paused_at TEXT;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (223, 'paused_at on game_stream_sessions');
