-- 220_game_stream_system_id.sql
-- Per-session system_id (rom_systems id like 'gamecube' / 'wii' / 'ps2') so
-- the cast-input container-WS handler can resolve the user's controller
-- pad_routing strategy via ControllerService.resolve(user_id, system_id).
--
-- Without this column, CastInputRegistry always defaulted to pad_routing
-- "index" regardless of the user's per-system preference, so the
-- "firstpress" strategy was unreachable for streamed-emulator sessions.
-- Non-emulator profiles (luanti, browser-stream) leave this NULL — they
-- have no concept of a ROM system to look up.

ALTER TABLE game_stream_sessions ADD COLUMN system_id TEXT;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (220, 'system_id on game_stream_sessions');
