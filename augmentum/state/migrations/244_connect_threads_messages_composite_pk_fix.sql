-- 244_connect_threads_messages_composite_pk_fix.sql
--
-- Schema drift fix for connect_threads + connect_messages.
--
-- Migration 219 currently declares ``PRIMARY KEY (thread_id, user_id)``
-- on connect_threads and ``PRIMARY KEY (message_id, user_id)`` on
-- connect_messages. Earlier revisions of that same migration used
-- single-column PKs. Live DBs created from the earlier revision still
-- carry the single-column PK because ``CREATE TABLE IF NOT EXISTS``
-- is a no-op once the table exists.
--
-- Consequence: the per-user mirror pattern silently breaks. When user
-- A sends to user B, A's sender row inserts fine, but B's recipient
-- mirror gets IGNOREd by the single-column PK conflict on the shared
-- thread_id. ``get_or_create_thread`` then raises
-- "insert/read race produced no row" → 500 on send.
--
-- Fix: rebuild both tables with the correct composite PK via the
-- standard SQLite rebuild dance (rename → create new → copy → drop).
--
-- Two gotchas worth noting in the migration body:
--
--   1. SQLite's ``ALTER TABLE RENAME TO`` rewrites references in
--      attached triggers since v3.25. The trigger
--      ``connect_messages_after_insert`` lives on connect_messages
--      but its body references connect_threads. After RENAME its
--      body would point at ``connect_threads_legacy``; after we
--      DROP that, the trigger is broken. Solution: explicitly
--      DROP TRIGGER before the rebuild and CREATE it again after.
--
--   2. A prior failed attempt of this migration may have left a
--      ``connect_threads_legacy`` table around. DROP IF EXISTS at the
--      top makes the migration re-runnable.
--
-- Strips ``tmp:`` prefixes from any thread_ids that leaked through
-- the UI placeholder path (``_openOrCreateThreadForPeer`` minted a
-- ``tmp:<peer-did>`` thread_id and the outbox flush persisted it
-- before the JS-side fix in sendMessage). The prefix is purely
-- client-side fiction.

-- Drop trigger + any stale legacy tables from a prior partial run.
DROP TRIGGER IF EXISTS connect_messages_after_insert;
DROP TABLE   IF EXISTS connect_threads_legacy;
DROP TABLE   IF EXISTS connect_messages_legacy;


-- ── connect_threads rebuild ──────────────────────────────────────

ALTER TABLE connect_threads RENAME TO connect_threads_legacy;

CREATE TABLE IF NOT EXISTS connect_threads (
    thread_id           TEXT NOT NULL,
    user_id             TEXT NOT NULL,
    peer_did            TEXT NOT NULL,
    peer_display_name   TEXT NOT NULL DEFAULT '',
    last_message_at     TIMESTAMP,
    last_message_preview TEXT NOT NULL DEFAULT '',
    unread_count        INTEGER NOT NULL DEFAULT 0,
    muted               INTEGER NOT NULL DEFAULT 0,
    pinned              INTEGER NOT NULL DEFAULT 0,
    archived            INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, user_id)
);

INSERT INTO connect_threads (
    thread_id, user_id, peer_did, peer_display_name,
    last_message_at, last_message_preview,
    unread_count, muted, pinned, archived, created_at
)
SELECT
    CASE WHEN thread_id LIKE 'tmp:%'
         THEN SUBSTR(thread_id, 5)
         ELSE thread_id END,
    user_id, peer_did, peer_display_name,
    last_message_at, last_message_preview,
    unread_count, muted, pinned, archived, created_at
FROM connect_threads_legacy;

DROP TABLE connect_threads_legacy;

CREATE INDEX IF NOT EXISTS idx_connect_threads_user
    ON connect_threads(user_id, last_message_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_connect_threads_pair
    ON connect_threads(user_id, peer_did);


-- ── connect_messages rebuild ─────────────────────────────────────

ALTER TABLE connect_messages RENAME TO connect_messages_legacy;

CREATE TABLE IF NOT EXISTS connect_messages (
    message_id          TEXT NOT NULL,
    thread_id           TEXT NOT NULL,
    user_id             TEXT NOT NULL,
    sender_did          TEXT NOT NULL,
    body                TEXT NOT NULL DEFAULT '',
    format              TEXT NOT NULL DEFAULT 'plain',
    attachment_ref      TEXT NOT NULL DEFAULT '',
    reply_to            TEXT NOT NULL DEFAULT '',
    sent_at             TIMESTAMP NOT NULL,
    received_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    delivered_at        TIMESTAMP,
    read_at             TIMESTAMP,
    edited_at           TIMESTAMP,
    deleted_at          TIMESTAMP,
    transcript          TEXT NOT NULL DEFAULT '',
    attachment_fetch_url TEXT,
    attachment_fetch_token TEXT,
    PRIMARY KEY (message_id, user_id)
);

INSERT INTO connect_messages (
    message_id, thread_id, user_id, sender_did, body, format,
    attachment_ref, reply_to, sent_at, received_at, delivered_at,
    read_at, edited_at, deleted_at, transcript,
    attachment_fetch_url, attachment_fetch_token
)
SELECT
    message_id,
    CASE WHEN thread_id LIKE 'tmp:%'
         THEN SUBSTR(thread_id, 5)
         ELSE thread_id END,
    user_id, sender_did, body, format,
    attachment_ref, reply_to, sent_at, received_at, delivered_at,
    read_at, edited_at, deleted_at, transcript,
    attachment_fetch_url, attachment_fetch_token
FROM connect_messages_legacy;

DROP TABLE connect_messages_legacy;

CREATE INDEX IF NOT EXISTS idx_connect_messages_thread
    ON connect_messages(thread_id, sent_at DESC);
CREATE INDEX IF NOT EXISTS idx_connect_messages_user
    ON connect_messages(user_id, sent_at DESC);


-- ── Recreate the trigger we dropped at the top ───────────────────

CREATE TRIGGER connect_messages_after_insert
    AFTER INSERT ON connect_messages
    WHEN NEW.deleted_at IS NULL
BEGIN
    UPDATE connect_threads
       SET last_message_at      = NEW.sent_at,
           last_message_preview = SUBSTR(NEW.body, 1, 200),
           unread_count         = CASE
               WHEN NEW.sender_did != (
                   SELECT peer_did
                     FROM connect_threads
                    WHERE thread_id = NEW.thread_id
                      AND user_id   = NEW.user_id
               )
               THEN unread_count
               ELSE unread_count + 1
           END
     WHERE thread_id = NEW.thread_id
       AND user_id   = NEW.user_id;
END;


INSERT OR IGNORE INTO schema_version (version, description)
VALUES (244, 'connect_threads + connect_messages composite PK fix');
