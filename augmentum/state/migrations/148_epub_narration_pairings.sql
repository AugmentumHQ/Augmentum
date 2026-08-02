-- 148_epub_narration_pairings.sql
-- Pairs an EPUB (artifact or file-index row) with a synthesized TTS
-- narration (a WAV artifact). Built either explicitly ("Record narration")
-- or passively as a side effect of reading aloud in the viewer.

CREATE TABLE IF NOT EXISTS epub_narrations (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id),
    epub_kind TEXT NOT NULL,                       -- 'artifact' | 'file'
    epub_ref TEXT NOT NULL,                        -- artifact id or file_index id
    narration_artifact_id TEXT NOT NULL DEFAULT '',-- the synthesized WAV artifact (when status='done')
    voice TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',        -- pending | running | done | failed
    error TEXT NOT NULL DEFAULT '',
    job_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_epub_narrations_unique
    ON epub_narrations(user_id, epub_kind, epub_ref);
