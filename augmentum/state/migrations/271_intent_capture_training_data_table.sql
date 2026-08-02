-- 271_intent_capture_training_data_table.sql
-- intent_capture training data table
--
-- Opt-in capture of the voice intent-router's decisions, so the routing
-- verdict produced by the (server-side, large) classifier can be distilled
-- into a small on-device model later. Gated by `intent_capture_enabled`
-- (default OFF) — only a user who turns it on writes rows. User-scoped so
-- each user only ever sees/exports their own captures.
--
-- The columns mirror exactly what the on-device model will see at inference
-- (input_text + context features) plus the teacher's verdict (goal,
-- confidence, model) — i.e. a ready-to-export (X, y) training row.

CREATE TABLE IF NOT EXISTS intent_capture (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT '',
    session_id TEXT NOT NULL DEFAULT '',

    -- which routing choke-point produced this row (room for architect_router,
    -- mode classifier, etc. later — for now: voice_router).
    surface TEXT NOT NULL DEFAULT 'voice_router',

    -- === the input the on-device model would also receive ===
    input_text TEXT NOT NULL DEFAULT '',          -- the STT transcript (X)
    last_assistant_response TEXT NOT NULL DEFAULT '',
    last_dispatch_summary TEXT NOT NULL DEFAULT '',
    active_surface TEXT NOT NULL DEFAULT '',
    seconds_since_last_tts REAL,                   -- NULL when unknown
    media_active INTEGER NOT NULL DEFAULT 0,
    explicit_capture INTEGER NOT NULL DEFAULT 0,   -- PTT/wake = addressing given

    -- === the teacher's verdict (y) ===
    goal TEXT NOT NULL DEFAULT '',                 -- act|converse|clarify|idle|drop (raw)
    effective_goal TEXT NOT NULL DEFAULT '',       -- after idle-promotion etc.
    coherent INTEGER NOT NULL DEFAULT 1,
    addressed INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.0,
    teacher_model TEXT NOT NULL DEFAULT '',        -- which model decided (distillation teacher)
    parsed_from TEXT NOT NULL DEFAULT '',          -- content|thinking|fallback|error
    reasoning TEXT NOT NULL DEFAULT '',
    latency_ms INTEGER NOT NULL DEFAULT 0,

    -- === correction flywheel (future) — user-confirmed label overrides ===
    corrected_goal TEXT NOT NULL DEFAULT '',

    schema_version INTEGER NOT NULL DEFAULT 1,
    captured_at REAL NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_intent_capture_user_time
    ON intent_capture (user_id, captured_at);
