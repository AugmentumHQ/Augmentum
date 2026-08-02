-- Add parent_id column to document_chunks for parent-child chunk relationships.
-- Parent chunks are stored with negative chunk_index; child chunks reference
-- their parent via parent_id for context expansion during retrieval.

ALTER TABLE document_chunks ADD COLUMN parent_id TEXT REFERENCES document_chunks(id) ON DELETE SET NULL;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (28, 'Document chunks: add parent_id for parent-child retrieval');
