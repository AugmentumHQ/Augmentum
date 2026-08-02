-- Cross-instance attachment fetch fields on connect_messages.
--
-- When a fabric peer sends a message with attachment_ref, the
-- receiving instance stores:
--   * attachment_fetch_url — full HTTPS URL to the sender's
--     ``/api/connect/fabric/attachments/{ref}`` endpoint. The
--     recipient's browser fetches the blob directly from the
--     sender's instance (no proxy).
--   * attachment_fetch_token — short-TTL fabric-signed token the
--     sender's instance verifies before serving bytes.
--
-- Both columns are nullable + only populated for fabric-delivered
-- messages. Local same-instance messages continue to resolve via
-- the existing /threads/{tid}/messages/{mid}/attachment route which
-- looks up the local uploads row directly.

ALTER TABLE connect_messages ADD COLUMN attachment_fetch_url TEXT;
ALTER TABLE connect_messages ADD COLUMN attachment_fetch_token TEXT;

-- schema_version may not exist in test harnesses that load migration
-- scripts directly without the full backend bootstrap. Tolerate that.
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    description TEXT,
    applied_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (242, 'connect fabric attachment fetch fields');
