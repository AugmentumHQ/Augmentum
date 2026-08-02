-- 132_user_bios_files.sql
-- Per-user BIOS file index. The bytes live in the existing blob
-- store (refcount-tracked, sha256-addressed, dedup'd across users);
-- this table records WHICH user installed WHICH canonical BIOS for
-- WHICH system, plus a pointer to the blob.
--
-- A single physical blob can satisfy many rows (two users dropping
-- the same scph5500.bin share one blob, one row each). The
-- ``UNIQUE(user_id, system_id, canonical_filename)`` ensures users
-- can't double-register the same BIOS slot; PUT semantics replace
-- the existing row and release the old blob.
--
-- ``original_filename`` preserves the name the user dropped (for
-- diagnostics: "you dropped 'PSX_BIOS_USA.BIN' but I matched it to
-- 'scph5501.bin'"). The launch path always reads via the canonical
-- name so emulator runtimes get what they expect.
--
-- ``sha1`` is the BIOS file SHA1 (matched against bios_catalog).
-- The blob is keyed by SHA256 (BlobStore native); both are kept so
-- the BIOS panel can show "matched by hash" vs "matched by name+size"
-- in the status checklist.

CREATE TABLE IF NOT EXISTS user_bios_files (
    id                  TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL REFERENCES users(id),
    system_id           TEXT NOT NULL,                          -- 'psx' / 'ps2' / 'saturn' / ...
    canonical_filename  TEXT NOT NULL,                          -- 'scph5500.bin' (from bios_catalog)
    blob_sha256         TEXT NOT NULL,                          -- → blobs table
    original_filename   TEXT NOT NULL DEFAULT '',               -- name the user dropped
    sha1                TEXT NOT NULL DEFAULT '',               -- BIOS file SHA1 (catalog match key)
    size_bytes          INTEGER NOT NULL,
    matched_by          TEXT NOT NULL DEFAULT 'sha1',           -- 'sha1' | 'name_size' | 'manual'
    installed_at        TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, system_id, canonical_filename)
);

CREATE INDEX IF NOT EXISTS idx_user_bios_files_user
    ON user_bios_files(user_id, system_id);

CREATE INDEX IF NOT EXISTS idx_user_bios_files_sha
    ON user_bios_files(blob_sha256);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (132, 'user_bios_files table');
