-- 010_image_editing.sql: Add img2img and inpainting support

-- SQLite ALTER TABLE ADD COLUMN fails if column exists, so we check first.
-- Using a CTE trick isn't possible, so we rely on the error being caught
-- by executescript's behavior. Instead, wrap in a no-op if column exists.

-- These will silently fail if the column already exists (handled by migration runner)
ALTER TABLE image_generations ADD COLUMN job_type TEXT NOT NULL DEFAULT 'txt2img';
ALTER TABLE image_generations ADD COLUMN strength REAL NOT NULL DEFAULT 1.0;
ALTER TABLE image_generations ADD COLUMN source_image_id TEXT DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description) VALUES (10, 'Image editing (img2img, inpaint)');
