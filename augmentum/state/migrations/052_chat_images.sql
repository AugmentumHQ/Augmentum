-- Chat image attachments (VL messages)
-- Stores user-uploaded images so they survive page reloads and server restarts.
CREATE TABLE IF NOT EXISTS chat_images (
    id          TEXT PRIMARY KEY,
    mime_type   TEXT NOT NULL DEFAULT 'image/jpeg',
    data        BLOB NOT NULL,
    session_id  TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chat_images_session ON chat_images(session_id);

INSERT OR IGNORE INTO schema_version (version, description) VALUES (52, 'Chat image attachments');
