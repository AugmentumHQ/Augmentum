-- 245_call_sessions_composite_pk_fix.sql
--
-- Same drift fix as migration 244, this time for call_sessions.
-- Migration 219 declares ``PRIMARY KEY (call_id, user_id)`` but live
-- DBs created from an earlier revision have ``call_id TEXT PRIMARY
-- KEY`` (single column). ``CREATE TABLE IF NOT EXISTS`` is a no-op so
-- the schema change never re-applied.
--
-- Consequence on call flow: when user A places a call to user B,
-- A's "ringing" row inserts fine, but B's "invited" mirror row gets
-- silently IGNOREd by the single-column PK conflict on the shared
-- call_id. The EVENT_INVITE still routes to B's WS (so a banner
-- shows), but the receiver-side state machine has no call_sessions
-- row to anchor against. B's accept fires MSG_ACCEPT with what looks
-- like a valid call_id, but the server's _handle_accept can't find
-- a row to transition → call stalls in "connecting" forever; the
-- caller's invite times out as "missed".
--
-- Standard SQLite rebuild dance. See 244 for the equivalent on
-- connect_threads + connect_messages and the trigger-handling gotcha
-- (call_sessions has no triggers attached, so the dance is simpler
-- here).

DROP TABLE IF EXISTS call_sessions_legacy;

ALTER TABLE call_sessions RENAME TO call_sessions_legacy;

CREATE TABLE IF NOT EXISTS call_sessions (
    call_id             TEXT NOT NULL,
    user_id             TEXT NOT NULL,
    initiator_did       TEXT NOT NULL,
    receiver_did        TEXT NOT NULL,
    modalities          TEXT NOT NULL DEFAULT 'audio',
    becca_present       INTEGER NOT NULL DEFAULT 0,
    state               TEXT NOT NULL DEFAULT 'invited',
    end_reason          TEXT NOT NULL DEFAULT '',
    initiated_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    connected_at        TIMESTAMP,
    ended_at            TIMESTAMP,
    quality_rating      INTEGER,
    quality_notes       TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (call_id, user_id)
);

INSERT INTO call_sessions (
    call_id, user_id, initiator_did, receiver_did,
    modalities, becca_present, state, end_reason,
    initiated_at, connected_at, ended_at,
    quality_rating, quality_notes
)
SELECT
    call_id, user_id, initiator_did, receiver_did,
    modalities, becca_present, state, end_reason,
    initiated_at, connected_at, ended_at,
    quality_rating, quality_notes
FROM call_sessions_legacy;

DROP TABLE call_sessions_legacy;

CREATE INDEX IF NOT EXISTS idx_call_sessions_user
    ON call_sessions(user_id, initiated_at DESC);
CREATE INDEX IF NOT EXISTS idx_call_sessions_peer
    ON call_sessions(user_id, initiator_did, receiver_did);


INSERT OR IGNORE INTO schema_version (version, description)
VALUES (245, 'call_sessions composite PK fix');
