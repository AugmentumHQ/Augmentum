-- Add background collection flag to image generations
ALTER TABLE image_generations ADD COLUMN is_background INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_image_generations_background ON image_generations(is_background) WHERE is_background = 1;

INSERT OR IGNORE INTO schema_version (version, description) VALUES (30, 'image_backgrounds');
