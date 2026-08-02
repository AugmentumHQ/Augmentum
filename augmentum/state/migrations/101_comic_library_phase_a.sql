-- Comic Library Phase A: scan infrastructure schema.
--
-- Adds the columns and tables required to track comic archive scan state,
-- series identity, and resumable scan checkpoints. The reader plan's schema
-- (comic_bookmarks) lives in a later migration; this one covers only the
-- library/scan foundation.
--
-- Notes on idempotency:
-- 1) ``is_favorite`` already exists from migration 075; not added here.
-- 2) The migration runner (``_run_migrations``) swallows ``already exists`` and
--    ``duplicate column`` errors, so re-runs are safe even after partial apply.

-- --- file_index extensions ---------------------------------------------------

-- Scan lifecycle: pending | scanning | ok | partial | error | orphaned.
ALTER TABLE file_index ADD COLUMN scan_status TEXT NOT NULL DEFAULT 'pending';

-- Filesystem mtime at last successful scan (unix epoch seconds). NULL when
-- unknown or when the source isn't a real filesystem path.
ALTER TABLE file_index ADD COLUMN mtime INTEGER;

-- JSON blob populated only when scan_status ∈ {error, partial}.
-- Shape: {"code": "bad_zip"|"permission_denied"|..., "message": str, "at": ts}.
ALTER TABLE file_index ADD COLUMN scan_error TEXT;

-- 0.0 (filename guess) to 1.0 (ComicInfo + external API + user-verified).
ALTER TABLE file_index ADD COLUMN metadata_confidence REAL NOT NULL DEFAULT 0.5;

-- FK to comic_series.id (not enforced — SQLite ALTER TABLE can't add REFERENCES).
-- NULL for archives with no detectable series.
ALTER TABLE file_index ADD COLUMN series_id TEXT;

CREATE INDEX IF NOT EXISTS idx_file_index_scan_status
    ON file_index(user_id, scan_status);
CREATE INDEX IF NOT EXISTS idx_file_index_series
    ON file_index(user_id, series_id);
CREATE INDEX IF NOT EXISTS idx_file_index_mtime
    ON file_index(user_id, mtime);

-- --- comic_series ------------------------------------------------------------

-- UUID-keyed stable series identity. Archives FK to this, not to the series
-- name. canonical_name / cover_file_id / description can change freely as
-- metadata improves; ``id`` is permanent from first-ingest, so favorites,
-- collections, and bookmarks that reference it never break.
CREATE TABLE IF NOT EXISTS comic_series (
    id                      TEXT PRIMARY KEY,
    user_id                 TEXT NOT NULL REFERENCES users(id),
    canonical_name          TEXT NOT NULL,
    sort_name               TEXT NOT NULL,
    alias_names             TEXT NOT NULL DEFAULT '[]',
    publisher               TEXT,
    author                  TEXT,
    description             TEXT,
    cover_file_id           TEXT,
    status                  TEXT,
    year_started            INTEGER,
    year_ended              INTEGER,
    genres                  TEXT NOT NULL DEFAULT '[]',
    language_iso            TEXT,
    age_rating              TEXT,
    metadata_source         TEXT,
    metadata_confidence     REAL NOT NULL DEFAULT 0.5,
    archive_count_reported  INTEGER,
    accent_color            TEXT,
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_comic_series_user_sort
    ON comic_series(user_id, sort_name);
CREATE INDEX IF NOT EXISTS idx_comic_series_user_updated
    ON comic_series(user_id, updated_at DESC);

-- --- comic_scan_checkpoint --------------------------------------------------

-- One row per active (or paused) scan per (user, library_root). Updated every
-- N completions by the scan orchestrator. On server restart, rows with
-- ``status='running'`` are resumed from ``last_path``.
CREATE TABLE IF NOT EXISTS comic_scan_checkpoint (
    user_id         TEXT NOT NULL REFERENCES users(id),
    library_root    TEXT NOT NULL,
    started_at      TEXT NOT NULL DEFAULT (datetime('now')),
    total_found     INTEGER NOT NULL DEFAULT 0,
    completed       INTEGER NOT NULL DEFAULT 0,
    failed          INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'running',
    last_path       TEXT,
    observed_rate   REAL,
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (user_id, library_root)
);

CREATE INDEX IF NOT EXISTS idx_comic_scan_checkpoint_status
    ON comic_scan_checkpoint(user_id, status);

-- --- schema_version bump ----------------------------------------------------

INSERT INTO schema_version (version, applied_at) VALUES (101, strftime('%s', 'now'));
