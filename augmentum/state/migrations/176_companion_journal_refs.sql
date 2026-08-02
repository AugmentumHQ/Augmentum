-- 176_companion_journal_refs.sql
-- Companion journal becomes the resolver index.
--
-- The Reference Resolver routes natural-language queries through the
-- journal first: "show me that voxel sketch I uploaded" maps to a
-- journal entry where Becca noted the upload, with the entry's
-- content_refs pointing at the actual file_index row. Resolution
-- returns the moment (the journal entry's perspective), not just the
-- file. See docs/superpowers/specs/2026-05-19-reference-resolver-*.md
-- once that lands.
--
-- Schema additions:
--   content_refs  JSON array of {"kind": "...", "id": "..."} references
--                 the entry is "about". Examples:
--                   [{"kind": "file_index", "id": "fi_abc"}]
--                   [{"kind": "session_message", "id": "sm_xyz"},
--                    {"kind": "chat_image",     "id": "ci_qrs"}]
--   place_ref     Where the entry was written, e.g. xr_session_id
--                 ("xrs_...") or device id. Enables "place-aware
--                 recall" — surfacing entries that match the user's
--                 current location.
--
-- The embedding column already exists from migration 154; the
-- existing journal() write path computes embeddings, so the resolver
-- already has a vec leg available the moment Slice 1's read endpoints
-- ship.

ALTER TABLE companion_journal ADD COLUMN content_refs TEXT NOT NULL DEFAULT '[]';
ALTER TABLE companion_journal ADD COLUMN place_ref    TEXT NOT NULL DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (176, 'companion_journal: content_refs + place_ref for resolver');
