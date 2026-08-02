-- 257_notes_v2_provenance_origin_and_resurface_columns.sql
-- Notes v2 Phase 1 (spec: docs/superpowers/specs/2026-06-10-notes-v2-useful-first-design.md)
--
-- origin_json: provenance for every journal note — which pipeline wrote
-- it, which client's signals fed it, how many, over what window. Lets
-- the drawer answer "why am I seeing this" in one tap. Shape:
--   {"source": "attention", "client": "web", "signal_count": 3,
--    "window": "2026-06-08T06:34/06:40", "detail": "browse: example.com x3"}
--
-- resurface_after: "Save for later" stamp — a nightly pass re-emits the
-- note as a follow_up once this datetime passes (Phase 3 consumes it).
--
-- brief_json: Today-as-brief slot payload on companion_today_reflections
-- (Phase 4 consumes it; column lands now so the schema is settled).
--
-- All columns nullable — old rows render without provenance/brief,
-- no backfill required.

ALTER TABLE companion_journal ADD COLUMN origin_json TEXT;
ALTER TABLE companion_journal ADD COLUMN resurface_after TEXT;
ALTER TABLE companion_today_reflections ADD COLUMN brief_json TEXT;

CREATE INDEX IF NOT EXISTS idx_journal_resurface
    ON companion_journal(user_id, resurface_after)
    WHERE resurface_after IS NOT NULL;
