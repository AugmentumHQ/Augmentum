-- Message reactions for the Connect text substrate.
--
-- Adds the emoji-reaction layer on top of connect_messages. Each
-- reaction is a (message_id, user_id, reactor_did, emoji) row:
-- per-user-scoped per the multi-tenant pattern, with the reactor
-- DID identifying WHO reacted (own DID for outgoing reactions,
-- peer's DID for incoming routed reactions). A reactor can attach
-- multiple emojis to the same message, but only one of each.
--
-- The composite PK guarantees idempotency on retries (a duplicate
-- react fires INSERT OR IGNORE → no double-count).
--
-- Cascading delete on connect_messages — if a message gets deleted,
-- its reactions go with it (the reactions point at a soft-deleted
-- body but the row itself is removed; we don't preserve reactions
-- on tombstones since the message body is no longer renderable).

CREATE TABLE IF NOT EXISTS connect_message_reactions (
    message_id   TEXT NOT NULL,
    user_id      TEXT NOT NULL,      -- owner of this copy (scoping)
    reactor_did  TEXT NOT NULL,      -- who reacted (own DID or peer DID)
    emoji        TEXT NOT NULL,      -- short string (👍 ❤️ 😂 etc.)
    reacted_at   TIMESTAMP NOT NULL,
    PRIMARY KEY (message_id, user_id, reactor_did, emoji)
);

CREATE INDEX IF NOT EXISTS idx_connect_message_reactions_by_msg
    ON connect_message_reactions (user_id, message_id);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (233, 'connect message reactions');
