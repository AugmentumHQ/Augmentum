-- 306_self_edit_attempts_source.sql
-- Ingest-all-work: tag each archived self-edit attempt with its ORIGIN so the
-- archive can grow from every stream of real work — not only the engine's own
-- autonomous attempts — while consumers stay honest about provenance.
--
--   autonomous  the self-edit engine's own loop (default; all pre-existing rows)
--   git         an ingested commit from the live repo's history
--   coder       an applied coder-mode turn (flag-gated hook)
--
-- Consumers (activation fold, retrodiction benchmark, Workshop lineage) weight
-- or filter by this tag — see augmentum/selfedit/activation.py::_SOURCE_WEIGHT.
-- The live archive is growth.db (augmentum/selfedit/growth_db.py, which adds
-- this column via _ADDED_COLUMNS); this migration keeps the main-DB mirror of
-- migration 288 at schema parity.

ALTER TABLE self_edit_attempts ADD COLUMN source TEXT NOT NULL DEFAULT 'autonomous';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (306, 'self_edit_attempts.source: provenance tag (autonomous|git|coder) for ingest-all-work');
