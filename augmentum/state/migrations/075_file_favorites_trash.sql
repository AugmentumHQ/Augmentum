-- 075_file_favorites_trash.sql
-- Add favorites and soft-delete (trash) support to file index.

ALTER TABLE file_index ADD COLUMN is_favorite INTEGER NOT NULL DEFAULT 0;
ALTER TABLE file_index ADD COLUMN is_trashed INTEGER NOT NULL DEFAULT 0;
ALTER TABLE file_index ADD COLUMN trashed_at TEXT;

CREATE INDEX IF NOT EXISTS idx_file_index_favorite
  ON file_index(user_id, is_favorite) WHERE is_favorite = 1;

CREATE INDEX IF NOT EXISTS idx_file_index_trashed
  ON file_index(user_id, is_trashed) WHERE is_trashed = 1;
