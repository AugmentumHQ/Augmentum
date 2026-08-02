-- Document RAG: file storage and chunk indexing for retrieval-augmented generation.

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    filename TEXT NOT NULL,
    mime_type TEXT NOT NULL DEFAULT 'text/plain',
    file_size INTEGER DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    scope TEXT,                  -- NULL = global, or session/mode-specific
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_scope ON documents(user_id, scope);

CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,  -- 0-based position in document
    content TEXT NOT NULL,
    page_num INTEGER,              -- NULL for non-paged formats
    char_offset INTEGER DEFAULT 0, -- character offset in original text
    token_count INTEGER DEFAULT 0,
    embedding BLOB,                -- float32 blob for vec storage
    parent_id TEXT REFERENCES document_chunks(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id, chunk_index);

-- FTS5 for keyword search on chunks
CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
    content,
    content=document_chunks,
    content_rowid=rowid
);

-- FTS sync triggers
CREATE TRIGGER IF NOT EXISTS trg_doc_chunks_ai AFTER INSERT ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS trg_doc_chunks_ad AFTER DELETE ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS trg_doc_chunks_au AFTER UPDATE ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
    INSERT INTO document_chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (25, 'Document RAG: documents and document_chunks tables with FTS5');
