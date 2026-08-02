-- Kind — user-facing category (image, document, audio, video, archive, code,
-- other). Derived from mime_type + filename extension at register time, so the
-- files panel can surface an image/document/audio/archive tab regardless of
-- which backing subsystem it came from. Source keeps tracking the source table
-- (used by cascading deletes), kind drives user-facing filters.
ALTER TABLE file_index ADD COLUMN kind TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_file_index_kind ON file_index(kind);

-- Coarse mime-based backfill. The Python classifier runs on startup to fill in
-- anything this pass leaves blank (empty mime / octet-stream / extension-only
-- detection).
UPDATE file_index SET kind = CASE
    WHEN mime_type LIKE 'image/%' THEN 'image'
    WHEN mime_type LIKE 'audio/%' THEN 'audio'
    WHEN mime_type LIKE 'video/%' THEN 'video'
    WHEN mime_type = 'application/pdf' THEN 'document'
    WHEN mime_type = 'application/epub+zip' THEN 'document'
    WHEN mime_type LIKE '%officedocument%' THEN 'document'
    WHEN mime_type LIKE '%opendocument%' THEN 'document'
    WHEN mime_type LIKE '%msword%' THEN 'document'
    WHEN mime_type LIKE '%ms-excel%' THEN 'document'
    WHEN mime_type LIKE '%ms-powerpoint%' THEN 'document'
    WHEN mime_type = 'text/markdown' THEN 'document'
    WHEN mime_type = 'text/csv' THEN 'document'
    WHEN mime_type = 'text/html' THEN 'document'
    WHEN mime_type LIKE 'text/%' THEN 'document'
    WHEN mime_type = 'application/json' THEN 'document'
    WHEN mime_type = 'application/zip' THEN 'archive'
    WHEN mime_type LIKE '%x-tar%' THEN 'archive'
    WHEN mime_type LIKE '%gzip%' THEN 'archive'
    WHEN mime_type LIKE '%7z%' THEN 'archive'
    WHEN mime_type LIKE '%x-rar%' THEN 'archive'
    ELSE ''
END
WHERE kind = '';
