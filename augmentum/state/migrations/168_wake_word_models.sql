-- Wake-word models: one trained ONNX file per avatar that wants a wake
-- word. The model itself lives on disk at /data/wake_word_models/{avatar_id}/
-- model.onnx; this table records which avatar maps to which file, what
-- phrase the model was trained on, and the training metrics (validation
-- AUC, false-accept rate) so the operator can see model quality without
-- reading logs.
--
-- Server-level (not user-scoped) — wake-word models are an instance
-- resource the operator manages, not per-user data. Mirrors the fabric
-- and provider tables.
--
-- Re-training re-writes the row with an incremented version.

CREATE TABLE IF NOT EXISTS wake_word_models (
    avatar_id TEXT PRIMARY KEY,
    phrase TEXT NOT NULL,
    model_path TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    trained_at TEXT NOT NULL DEFAULT (datetime('now')),
    train_metrics TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_wake_word_models_avatar_id
    ON wake_word_models(avatar_id);
