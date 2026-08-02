-- 243_remove_kittentts_provider.sql
-- Remove the retired KittenTTS built-in provider from existing installs.
--
-- KittenTTS shipped as a built-in TTS engine alongside Kokoro and Pocket
-- TTS. It was direct-swapped for Pocket TTS — same architectural slot
-- (CPU-only, in-process, no sidecar), strictly newer model. The dispatch
-- branch and provider registration have been removed from code, but
-- existing installs may still have a row in `audio_providers` plus the
-- four `tts_kitten_*` keys in the app_settings / user_settings KV tables.
--
-- Without this cleanup, a user who had `kittentts-builtin` as their
-- default TTS provider would hit the external-HTTP fallback in
-- tts_speech() and get "Request URL is missing an 'http://' protocol".
-- We delete the row outright and clear the orphan settings keys.
--
-- Note: the server-level KV table is `app_settings`, not `settings`.
-- An earlier draft of this migration referenced `settings` and crashed
-- the bootstrap (no such table) → falls back to in-memory backend →
-- everything 503s with `auth_unavailable_denied`.

DELETE FROM audio_providers
WHERE id = 'kittentts-builtin';

DELETE FROM app_settings
WHERE key IN (
    'tts_kitten_builtin',
    'tts_kitten_model_dir',
    'tts_kitten_hbe',
    'tts_kitten_preprocessor'
);

DELETE FROM user_settings
WHERE key IN (
    'tts_kitten_builtin',
    'tts_kitten_model_dir',
    'tts_kitten_hbe',
    'tts_kitten_preprocessor'
);
