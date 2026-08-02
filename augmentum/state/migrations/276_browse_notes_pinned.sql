-- 276_browse_notes_pinned.sql
-- Pin notes to the top of the list (Apple-Notes-style organization).
--
-- Until now the notes list was sorted purely by updated_at DESC, so there
-- was no way to keep an important note above the churn of recently-edited
-- ones. This additive column lets a note be pinned; list_stubs sorts
-- pinned DESC, then updated_at DESC.
--
-- All additive, back-compat: existing rows default to 0 (unpinned).
ALTER TABLE browse_notes
    ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (276, 'browse_notes: pinned column for pin-to-top notes');
