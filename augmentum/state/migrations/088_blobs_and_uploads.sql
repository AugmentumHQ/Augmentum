-- Content-addressed blob store + uploads source.
--
-- `blobs` stores bytes once per unique SHA-256. Multiple logical files
-- (different filenames, owners, source tables) sharing the same content
-- point at one blob — zero-cost dedup. Reference counting drives cleanup:
-- decrement on each logical-file delete, purge the physical blob when the
-- last reference drops.
--
-- `uploads` is the first source backed entirely by this pattern. Future
-- adapters (Dropbox sync, S3 sync, etc.) can follow the same shape: their
-- own lightweight metadata row pointing at a shared blob.

CREATE TABLE IF NOT EXISTS blobs (
    sha256      TEXT PRIMARY KEY,
    size_bytes  INTEGER NOT NULL,
    mime_type   TEXT NOT NULL DEFAULT '',
    real_path   TEXT NOT NULL,
    refcount    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Cleanup queries and invariant checks scan by refcount.
CREATE INDEX IF NOT EXISTS idx_blobs_refcount ON blobs(refcount);


CREATE TABLE IF NOT EXISTS uploads (
    id          TEXT PRIMARY KEY,                       -- "ul_<hex>"
    user_id     TEXT NOT NULL REFERENCES users(id),
    filename    TEXT NOT NULL,
    blob_sha    TEXT NOT NULL REFERENCES blobs(sha256),
    size_bytes  INTEGER NOT NULL,
    mime_type   TEXT NOT NULL DEFAULT '',
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_uploads_user ON uploads(user_id);
CREATE INDEX IF NOT EXISTS idx_uploads_blob ON uploads(blob_sha);
