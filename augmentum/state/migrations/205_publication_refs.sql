-- 205_publication_refs.sql
-- Phase 1, PR-1.3 of the Integrated Coding Nervous System.
-- See docs/superpowers/specs/2026-05-29-integrated-coding-nervous-system.md.
--
-- Wires library_publications onto project_refs (migration 199). Before
-- this change, library_publications.workspace_id is *advisory only*
-- (per its migration comment) — the publication is a frozen snapshot
-- that has no structural relationship to the work that produced it.
-- After PR-1.3, every publication points at a `project_refs` row of
-- kind 'publication', which is a git tag on the project's bare repo.
-- That gives us:
--
--   * "Open in Coder" on a publication: clone the bare repo at the
--     tag's sha into a fresh checkout. Closes the "Library Play is a
--     read-only dead end" gap.
--   * Re-publish with edits: create a new tag at a new sha. Old tags
--     are immutable so existing launches stay working.
--   * Source attribution: from any publication, walk back to the
--     project that produced it.
--
-- Nullable on purpose: legacy publications (saved before PR-1.3) have
-- no associated project_ref and Open-in-Coder is greyed out for them
-- in the UI. The save route will populate this column for new
-- publications.
--
-- ``workspace_id`` stays on the row for one release as a deprecated
-- backfill source. Phase 2 will drop it.


ALTER TABLE library_publications ADD COLUMN project_ref_id TEXT
    REFERENCES project_refs(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_library_publications_project_ref
    ON library_publications(project_ref_id)
    WHERE project_ref_id IS NOT NULL;


INSERT OR IGNORE INTO schema_version (version, description)
VALUES (201, 'library_publications.project_ref_id — Phase 1 PR-1.3');
