-- Language learning system — Phase 1.
--
-- One row per (user, language, word) tracking that user's spaced-
-- repetition state for that vocabulary item. User-scoped: vocabulary
-- progress is intensely personal and must never bleed between users on a
-- shared install. The dictionary/sentence data itself lives in a static,
-- server-shared `.augpack` (pack_kind=language) — only the *learner
-- state* is here.
--
-- `word_id` is the JMdict <ent_seq> (a numeric ID JMdict guarantees
-- stable per entry across releases), so a pack rebuild/reinstall never
-- orphans progress. For non-JMdict languages it's whatever stable key the
-- corresponding pack builder assigns; the column is TEXT to stay agnostic.
--
-- See docs/superpowers/specs/2026-05-11-language-learning-system.md.

CREATE TABLE IF NOT EXISTS vocab_state (
    user_id          TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    lang_code        TEXT NOT NULL,                    -- 'ja', 'es', ...
    word_id          TEXT NOT NULL,                    -- pack vocab entry id (JMdict <ent_seq>)
    fsrs_difficulty  REAL NOT NULL DEFAULT 5.0,
    fsrs_stability   REAL NOT NULL DEFAULT 0.0,
    fsrs_due_at      TEXT NOT NULL,                    -- ISO8601; = now+1d on first add
    fsrs_reps        INTEGER NOT NULL DEFAULT 0,
    fsrs_lapses      INTEGER NOT NULL DEFAULT 0,
    fsrs_last_grade  INTEGER,                          -- 1=again 2=hard 3=good 4=easy
    mastery_state    TEXT NOT NULL DEFAULT 'new',      -- new|learning|reviewing|mature|leech
    first_seen_at    TEXT NOT NULL DEFAULT (datetime('now')),
    last_reviewed_at TEXT,
    source_surface   TEXT NOT NULL,                    -- browse|youtube|narrative|manual|seeded
    source_ref       TEXT,                             -- url / context of first encounter
    exposure_input   INTEGER NOT NULL DEFAULT 0,       -- times seen as comprehension input
    exposure_output  INTEGER NOT NULL DEFAULT 0,       -- reserved for phase 3+ (production)
    PRIMARY KEY (user_id, lang_code, word_id)
);

CREATE INDEX IF NOT EXISTS idx_vocab_state_due
    ON vocab_state(user_id, lang_code, fsrs_due_at);

CREATE INDEX IF NOT EXISTS idx_vocab_state_mastery
    ON vocab_state(user_id, lang_code, mastery_state);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (145, 'vocab_state — language learning spaced-repetition state (user-scoped)');
