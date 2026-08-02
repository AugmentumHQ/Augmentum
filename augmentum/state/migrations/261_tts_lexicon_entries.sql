-- 261_tts_lexicon_entries.sql
-- Per-voice TTS pronunciation lexicon (successor surface to the global
-- voice_tts_lexicon JSON setting, which remains as the base layer).
--
-- One row = "when speaking with <voice>, say <term> as <phonetics>".
--   voice ''      → applies to every voice (global entry)
--   phonetics ''  → shield: never let built-in normalization touch the
--                   term (mirrors the setting's empty-value semantic)
-- Voice-specific entries win over '' entries on the same term.
--
-- Application point: proxy/audio_routes.py speech endpoints, before
-- clean_for_tts — the only layer that knows user + voice + text at
-- once. Compiled-pattern cache lives in augmentum/voice/lexicon_store.py.

CREATE TABLE IF NOT EXISTS tts_lexicon_entries (
    id INTEGER PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    voice TEXT NOT NULL DEFAULT '',
    term TEXT NOT NULL,
    phonetics TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, voice, term)
);

CREATE INDEX IF NOT EXISTS idx_tts_lexicon_user_voice
    ON tts_lexicon_entries(user_id, voice);
