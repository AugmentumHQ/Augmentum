-- Wiring program Phase 5: provenance for companion-created playlists.
-- Same convention as migration 259 (browse_notes / image_generations):
-- '' = user-created, 'companion' = created by the companion. Surfaces
-- mark origin inline (playlist dropdown suffix); no separate area.
ALTER TABLE playlists ADD COLUMN origin TEXT NOT NULL DEFAULT '';
