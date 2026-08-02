-- 253_companion_audio_history_table_for_mimi_token_rolling_buffer.sql
-- companion_audio_history — rolling buffer of conversation turns
--
-- Phase 2 of the presence pipeline. Captures Becca's PocketTTS output
-- as Mimi tokens (no extra encode pass — PocketTTS produces them
-- internally before vocoder decode). User input is stored as transcript
-- only for v1; encoding user PCM through Mimi requires a separate
-- server-side encoder pass that lands later.
--
-- Why Mimi tokens at all: any future Kyutai-family model swap (CSM,
-- Moshi, Pocket-vNext) speaks Mimi. Storing conversation context in
-- the codec's token space means future swaps inherit history for free.
--
-- Storage shape: one row per turn, not one row per session. Rolling
-- window enforced by a sweep that drops rows older than the retention
-- horizon (default 30 days) at startup. Per-session ordering via
-- turn_index — append-only, never reordered.
--
-- Compression: mimi_tokens is a BLOB holding the gzipped serialized
-- np.ndarray (int16, shape [N_codebooks, N_frames]). Mimi at 12.5 Hz
-- with 8 codebooks runs ~100 int16/sec uncompressed; gzip typically
-- halves that. A 30s turn ≈ 1.5 KB compressed.

CREATE TABLE IF NOT EXISTS companion_audio_history (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    speaker TEXT NOT NULL,           -- 'becca' | 'user'
    transcript TEXT NOT NULL DEFAULT '',
    mimi_tokens BLOB,                -- nullable: user turns omit tokens in v1
    duration_ms INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Per-session ordered read pattern (recent_window) — newest-first scan
-- bounded by turn_index DESC.
CREATE UNIQUE INDEX IF NOT EXISTS idx_companion_audio_history_session_turn
    ON companion_audio_history(user_id, session_id, turn_index);

-- Retention sweep pattern — drop rows older than horizon.
CREATE INDEX IF NOT EXISTS idx_companion_audio_history_created
    ON companion_audio_history(created_at);
