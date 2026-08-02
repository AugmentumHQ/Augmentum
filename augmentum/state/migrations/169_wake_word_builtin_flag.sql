-- Wake-word builtins flag: distinguishes the curated set shipped with
-- the app from user-trained custom models. Required so an operator
-- "reset my custom wake words" action can DELETE WHERE NOT is_builtin
-- without nuking the curated 15 that the bake script produced.
--
-- The bake script that runs once during dev sets is_builtin=1 on each
-- of its outputs. User-triggered training via POST /api/wake_word/train
-- leaves the default 0.
--
-- ALTER TABLE ... ADD COLUMN is the canonical idempotent path for this
-- in SQLite: a re-run on a database that already has the column would
-- fail at the ALTER, but the migration runner records a one-shot apply
-- per file so re-runs don't reach this statement. See migrations/
-- 161_companion_journal_taxonomy.sql for the same pattern.

ALTER TABLE wake_word_models
    ADD COLUMN is_builtin INTEGER NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_wake_word_models_is_builtin
    ON wake_word_models(is_builtin);
