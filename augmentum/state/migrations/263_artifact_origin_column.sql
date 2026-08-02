-- Wiring program Phase 7: provenance for companion-created artifacts.
-- Same convention as migrations 259/262: '' = user-created, 'companion'
-- = produced by a companion tool call. Stamped via the artifact-origin
-- contextvar set on the companion execution path (see
-- augmentum/tools/artifact_storage.py) so every artifact-producing
-- tool inherits it without per-tool wiring.
ALTER TABLE artifacts ADD COLUMN origin TEXT NOT NULL DEFAULT '';
