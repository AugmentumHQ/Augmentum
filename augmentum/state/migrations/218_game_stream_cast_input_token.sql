-- 218_game_stream_cast_input_token.sql
-- Per-session token used by the in-container cast-input-bridge.py daemon to
-- authenticate when it dials augmentum's /api/cast/input/container-ws/{id}
-- endpoint. Minted at start_session for any emulator-streaming-capable
-- profile; unused (NULL) for non-cast profiles. ~32 url-safe bytes (256
-- bits of entropy), generated via secrets.token_urlsafe(32) in the runtime
-- layer; pattern mirrors game_agent_routes.SessionRecord.bridge_token.
--
-- No new isolation surface: the token is consumed by the registry layer
-- (augmentum/cast/input_bridge.py) which keys on (session_id) and the
-- session row's user_id already enforces ownership for everything that
-- reads it back.

ALTER TABLE game_stream_sessions ADD COLUMN cast_input_token TEXT;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (218, 'cast_input_token on game_stream_sessions');
