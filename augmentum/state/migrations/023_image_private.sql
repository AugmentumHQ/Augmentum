-- Add private flag to image generations for private gallery section
ALTER TABLE image_generations ADD COLUMN is_private INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_ig_private ON image_generations(is_private);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (23, 'Image generation private flag');
