-- 259_origin_column_for_companion_created_content.sql
--
-- Provenance, not silos (companion wiring program, principle 2):
-- companion-created items live in the SAME tables and surfaces as
-- user-created ones, distinguished by an origin marker the UI can
-- filter on. '' / 'user' = user-created; 'companion' = hers.
--
-- This migration covers the two stores she ALREADY writes today
-- (note.create -> browse_notes; image_generation -> image_generations).
-- Later program phases add the column to playlists/artifacts/etc. in
-- the phase whose verb first writes them.

ALTER TABLE browse_notes ADD COLUMN origin TEXT NOT NULL DEFAULT '';
ALTER TABLE image_generations ADD COLUMN origin TEXT NOT NULL DEFAULT '';
