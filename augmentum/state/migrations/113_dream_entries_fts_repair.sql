-- 113_dream_entries_fts_repair.sql: Version marker for the
-- dream_entries_fts phantom-repair work. The actual repair runs in
-- Python at connect time alongside memories_fts, file_index_fts, and
-- document_chunks_fts (see ``SQLiteBackend._repair_phantom_fts_if_needed``
-- in ``augmentum/state/backends/sqlite.py``), BEFORE migrations execute.
--
-- Same class of bug as 096/097/104: the FTS5 virtual table's shadow
-- tables (``_data``, ``_idx``, ``_config``, ``_docsize``) survived
-- while the virtual-table row in sqlite_master was lost. The sync
-- triggers from migration 058 remained, so every INSERT into
-- ``dream_entries`` fired ``dream_entries_ai`` and failed with
-- ``no such table: main.dream_entries_fts``.
--
-- Triggering incident: 2026-05-02 — manual /api/dream/trigger crashed
-- on the first journal write after the engine successfully selected
-- 20 dream-eligible memories. Symptom looked like a generation bug
-- but was actually the same phantom-FTS pattern that's bitten the
-- other three FTS tables. Net effect was zero dream entries ever
-- written for users whose dream_entries_fts had drifted phantom.

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (113, 'dream_entries_fts phantom repair (handled in Python: _repair_phantom_fts_if_needed)');
