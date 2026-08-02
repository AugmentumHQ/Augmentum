-- Learning-games results — per-session record.
--
-- One row per round of any learning game (bubble_pop, echo_chamber,
-- whisper_race, etc.). Powers high-score persistence on the hub cards,
-- session history graphs, and adaptive difficulty (the game can pull
-- the user's recent results to decide whether to ramp up/down).
--
-- User-scoped per the multi-tenant contract. The game_id is a freeform
-- TEXT slug so adding a new game means no migration; metadata is a JSON
-- blob for per-game stats (combos, peak speed, etc.) that don't justify
-- their own columns.

CREATE TABLE IF NOT EXISTS game_results (
    id            INTEGER PRIMARY KEY,
    user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    game_id       TEXT NOT NULL,                          -- 'bubble_pop' | 'echo_chamber' | ...
    lang_code     TEXT NOT NULL,
    score         INTEGER NOT NULL DEFAULT 0,
    words_played  INTEGER NOT NULL DEFAULT 0,
    words_correct INTEGER NOT NULL DEFAULT 0,
    duration_sec  INTEGER NOT NULL DEFAULT 0,
    started_at    TEXT NOT NULL DEFAULT (datetime('now')),
    ended_at      TEXT,
    metadata      TEXT                                    -- JSON: {max_combo: 8, ...}
);

CREATE INDEX IF NOT EXISTS idx_game_results_user_game
    ON game_results(user_id, game_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_game_results_user_lang
    ON game_results(user_id, lang_code, started_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (150, 'game_results — per-session results for learning-games suite');
