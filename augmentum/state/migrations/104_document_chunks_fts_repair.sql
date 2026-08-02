-- 104_document_chunks_fts_repair.sql: Version marker for the
-- document_chunks_fts phantom-repair work. The actual repair runs in
-- Python at connect time alongside memories_fts and file_index_fts
-- (see ``SQLiteBackend._repair_phantom_fts_if_needed`` in
-- ``augmentum/state/backends/sqlite.py``), BEFORE migrations execute.
--
-- Same class of bug as 096/097: the FTS5 virtual table's shadow tables
-- (``_data``, ``_idx``, ``_config``, ``_docsize``) survived while the
-- virtual-table row in sqlite_master was lost. The sync triggers from
-- migration 025 remained, so every INSERT into ``document_chunks``
-- fired ``trg_doc_chunks_ai`` and failed with
-- ``no such table: main.document_chunks_fts``.
--
-- Triggering incident: on 2026-04-22 browse_save against
-- https://en.wikipedia.org/wiki/N8n raised OperationalError on the
-- first parent-chunk INSERT. Any document ingest path (browse save,
-- file upload, app-builder attachment) was affected.

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (104, 'document_chunks_fts phantom repair (handled in Python: _repair_phantom_fts_if_needed)');
