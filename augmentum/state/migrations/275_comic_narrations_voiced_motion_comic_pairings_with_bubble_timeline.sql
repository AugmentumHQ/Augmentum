-- 275_comic_narrations_voiced_motion_comic_pairings_with_bubble_timeline.sql
-- Pairs a comic (file-index row of a Komga/Suwayomi chapter) with a
-- synthesized TTS narration + a per-bubble timeline that drives the cast
-- pan-and-scan playback. Mirror of epub_narrations (mig 148/149) with the
-- timeline JSON + per-page checkpoint added. User-scoped.

CREATE TABLE IF NOT EXISTS comic_narrations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    comic_kind TEXT NOT NULL,                       -- 'file' (file_index id of the chapter)
    comic_ref TEXT NOT NULL,                        -- file_index row id
    narration_artifact_id TEXT NOT NULL DEFAULT '', -- synthesized audio artifact (when status='done')
    timeline TEXT NOT NULL DEFAULT '[]',            -- JSON bubble timeline (page/order/bbox/text/kind/audio offsets)
    voice TEXT NOT NULL DEFAULT '',
    engine_id TEXT NOT NULL DEFAULT '',             -- built-in TTS engine used
    reading_direction TEXT NOT NULL DEFAULT 'ltr',  -- 'ltr' (western) | 'rtl' (manga)
    status TEXT NOT NULL DEFAULT 'pending',         -- pending | running | done | failed
    error TEXT NOT NULL DEFAULT '',
    job_id TEXT NOT NULL DEFAULT '',
    processed_pages INTEGER NOT NULL DEFAULT 0,     -- checkpoint for restart-resume
    total_pages INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_comic_narrations_unique
    ON comic_narrations(user_id, comic_kind, comic_ref);
