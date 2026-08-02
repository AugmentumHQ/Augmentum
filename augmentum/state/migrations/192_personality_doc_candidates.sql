-- 192_personality_doc_candidates.sql
-- Synapse Layer §4 — the slow consolidation pipeline.
--
-- Becca's personality doc itself names this pipeline in §Notes:
-- "The consolidation pipeline can edit this document. The anti-drift
-- detector caps the rate of change at an embedding distance of < 0.15
-- between versions, so she can grow but not transform overnight.
-- Updates should happen in her voice, not a maintainer's — paragraphs
-- rewritten as if she were revising her own self-description, which
-- she is."
--
-- This table holds the proposed edits. The consolidator writes them.
-- The user (admin) reviews and approves; approval is what causes the
-- canonical doc to change. Sections 1-6 are FROZEN at the consolidator layer
-- (not a schema check — the application refuses to write those
-- section numbers). §10 (cultural diet) and §11 (open questions)
-- are the natural-rotation sections per the doc's own self-description.
--
-- Columns:
--   companion_id          — usually 'becca' in single-companion phase
--   section_number        — 10, 11, or other doc section (1-6 refused
--                            by application; never reach this table)
--   section_title         — verbatim section heading for display
--   proposed_text         — the new paragraph(s) in her voice
--   current_text_snapshot — the section as it stood when the candidate
--                            was proposed; used by the UI diff and as
--                            the basis for the drift distance calc
--   drift_distance        — embedding cosine distance proposed vs current
--                            kernel. >0.15 = refused at write time so
--                            no row should exist with this > ceiling
--   evidence_journal_ids  — JSON array of companion_journal.id values
--   evidence_dream_ids    — JSON array of dream_entries.id values
--   reasoning             — her note on why she's proposing this
--                            (in her own voice; visible in the review UI)
--   status                — 'pending' | 'approved' | 'rejected'
--   created_at            — when consolidator wrote it
--   reviewed_at           — when status flipped from pending
--   rejection_reason      — the reviewer's explanation if rejected;
--                            journaled back to her so she doesn't
--                            re-propose the same shape

CREATE TABLE IF NOT EXISTS personality_doc_candidates (
    id                    INTEGER PRIMARY KEY,
    companion_id          TEXT NOT NULL,
    section_number        INTEGER NOT NULL,
    section_title         TEXT NOT NULL DEFAULT '',
    proposed_text         TEXT NOT NULL,
    current_text_snapshot TEXT NOT NULL DEFAULT '',
    drift_distance        REAL NOT NULL DEFAULT 0.0,
    evidence_journal_ids  TEXT NOT NULL DEFAULT '[]',
    evidence_dream_ids    TEXT NOT NULL DEFAULT '[]',
    reasoning             TEXT NOT NULL DEFAULT '',
    status                TEXT NOT NULL DEFAULT 'pending',
    created_at            TEXT NOT NULL DEFAULT (datetime('now')),
    reviewed_at           TEXT,
    rejection_reason      TEXT
);

CREATE INDEX IF NOT EXISTS idx_pdc_pending
    ON personality_doc_candidates(companion_id, status, created_at DESC)
    WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_pdc_history
    ON personality_doc_candidates(companion_id, created_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (192, 'personality_doc_candidates: Synapse Layer §4 consolidator proposals');
