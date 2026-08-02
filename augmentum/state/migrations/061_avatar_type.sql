-- Add type column: 'portrait' (2D animated) or 'vrm' (3D model)
ALTER TABLE avatars ADD COLUMN type TEXT NOT NULL DEFAULT 'vrm';
-- Add cached segmentation data (MediaPipe landmarks + region masks as JSON)
ALTER TABLE avatars ADD COLUMN segmentation_data TEXT;
