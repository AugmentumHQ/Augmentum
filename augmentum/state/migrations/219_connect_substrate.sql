-- Connect mode substrate — voice/video calls + text threads between users.
--
-- See docs/superpowers/specs/2026-06-01-connect-and-os-positioning-design.md
-- for the full design discussion. STATUS at migration-write time: design
-- is captured; UI and signaling endpoint are not yet implemented. This
-- migration establishes the schema foundation so subsequent work has
-- something concrete to write against.
--
-- All tables are strictly user-scoped per the multi-tenant pattern
-- (CLAUDE.md). Each user has their own copy of contacts, threads, and
-- messages. This matters: when a thread exists between user A on this
-- instance and user B on a paired fabric peer, both instances store
-- their own copy. Same-instance threads between two users on this box
-- also get stored once per user. The redundancy is the price of strict
-- per-user isolation; storage cost is trivial relative to its benefit.
--
-- Five tables:
--   * connect_contacts        — discovered/known contact list per user
--   * call_sessions           — per-call lifecycle record
--   * call_events             — per-event log within a call (state transitions)
--   * connect_threads         — text conversation thread per pair
--   * connect_messages        — individual messages within a thread


CREATE TABLE IF NOT EXISTS connect_contacts (
    contact_id          TEXT PRIMARY KEY,           -- short UUID
    user_id             TEXT NOT NULL,              -- owner of this copy
    -- Peer identity. For Phase 1: user@instance form, with the instance
    -- being either "this-instance" sentinel or a paired fabric peer host.
    -- Forward-compatible with did:augmentum:<keyfp> form when that
    -- minimum-viable DID layer lands.
    peer_did            TEXT NOT NULL,
    peer_display_name   TEXT NOT NULL DEFAULT '',
    peer_avatar_url     TEXT NOT NULL DEFAULT '',
    -- How this contact was discovered. Mutual-enablement-as-consent means
    -- contacts surface automatically when both parties opt in with
    -- matching discoverability scope. Source captures which scope made
    -- this contact visible — useful for diagnostics and for revoking
    -- visibility cleanly when scope changes.
    --   'same_instance'  — both users on this Augmentum, both opted in
    --   'fabric_peer'    — paired peer's user, both opted in via peer scope
    --   'handle_added'   — manually added by handle (future Phase 2)
    discovery_source    TEXT NOT NULL DEFAULT 'same_instance',
    -- Async-friendly presence cache. Authoritative presence comes from
    -- the signaling layer's live state; this is the last-seen snapshot
    -- for offline UI rendering.
    last_seen_status    TEXT NOT NULL DEFAULT 'offline',  -- online | away | dnd | offline
    last_seen_at        TIMESTAMP,
    -- Blocking is asymmetric. If user A blocks B, B doesn't see A even
    -- when mutual permissions otherwise hold. Distinct from removing
    -- (which deletes the row entirely).
    blocked             INTEGER NOT NULL DEFAULT 0,
    -- User-level tagging. JSON list of free-form tags ("family",
    -- "work", etc.) for future filtering / grouping.
    tags                TEXT NOT NULL DEFAULT '[]',
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_connect_contacts_user
    ON connect_contacts(user_id);
CREATE INDEX IF NOT EXISTS idx_connect_contacts_peer
    ON connect_contacts(user_id, peer_did);


CREATE TABLE IF NOT EXISTS call_sessions (
    -- (call_id, user_id) composite PK: call_id is shared between both
    -- participants (it's the call's stable id used in signaling), and
    -- each participant stores their own row from their perspective.
    -- A standalone PRIMARY KEY (call_id) would block the receiver's
    -- row from being inserted on same-instance calls.
    call_id             TEXT NOT NULL,              -- shared call id
    user_id             TEXT NOT NULL,              -- owner of this copy
    -- Both legs of the call. Initiator's perspective dominates the
    -- record on initiator's instance; receiver's perspective on theirs.
    -- Reconciliation happens at the protocol layer if needed.
    initiator_did       TEXT NOT NULL,
    receiver_did        TEXT NOT NULL,
    -- 'audio' or 'audio,video' — comma-separated for forward compat.
    -- 'screen' could join the list in Phase 2.
    modalities          TEXT NOT NULL DEFAULT 'audio',
    -- Per the design: companion participation deferred. Column is
    -- here for forward-compatibility but stays 0 in Phase 1.
    becca_present       INTEGER NOT NULL DEFAULT 0,
    -- State machine. See the design doc for the full transition graph.
    -- Terminal states: ENDED, DECLINED, MISSED, FAILED.
    state               TEXT NOT NULL DEFAULT 'invited',
    end_reason          TEXT NOT NULL DEFAULT '',   -- 'hangup_initiator' | 'hangup_receiver' | 'network_failure' | 'declined' | 'timeout'
    -- Lifecycle timestamps. NULL until the corresponding transition fires.
    initiated_at        TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    connected_at        TIMESTAMP,
    ended_at            TIMESTAMP,
    -- Quality metadata, opt-in from the post-call rating UX.
    quality_rating      INTEGER,                    -- 1=good, -1=issues, NULL=unrated
    quality_notes       TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (call_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_call_sessions_user
    ON call_sessions(user_id, initiated_at DESC);
CREATE INDEX IF NOT EXISTS idx_call_sessions_peer
    ON call_sessions(user_id, initiator_did, receiver_did);


CREATE TABLE IF NOT EXISTS call_events (
    -- Plain INTEGER PRIMARY KEY (alias for rowid). AUTOINCREMENT adds
    -- a sqlite_sequence row + write overhead and was implicated in
    -- past corruption (see db_safety scanner). Rowid reuse is fine
    -- here — event_id is referenced only by the row itself.
    event_id            INTEGER PRIMARY KEY,
    call_id             TEXT NOT NULL,              -- references call_sessions
    user_id             TEXT NOT NULL,              -- mirrors call_sessions.user_id for scope-locality
    -- Event taxonomy (forward-compat — unknown event_types are logged):
    --   'state_change'   — call moved between state-machine states (data: old, new)
    --   'mute_toggle'    — user muted/unmuted (data: actor, modality, muted)
    --   'video_toggle'   — video on/off mid-call (data: actor, on)
    --   'network_issue'  — transient quality degradation (data: metric, value)
    --   'reconnect'      — ICE restart fired
    event_type          TEXT NOT NULL,
    event_data          TEXT NOT NULL DEFAULT '{}', -- JSON
    occurred_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_call_events_call
    ON call_events(call_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_call_events_user
    ON call_events(user_id, occurred_at DESC);


CREATE TABLE IF NOT EXISTS connect_threads (
    -- (thread_id, user_id) composite PK: thread_id is shared between
    -- the two thread participants; each stores their own row.
    thread_id           TEXT NOT NULL,              -- shared thread id
    user_id             TEXT NOT NULL,              -- owner of this copy
    peer_did            TEXT NOT NULL,              -- the other party
    peer_display_name   TEXT NOT NULL DEFAULT '',
    -- Denormalized message-tail snapshot for fast list rendering.
    -- Source of truth still lives in connect_messages.
    last_message_at     TIMESTAMP,
    last_message_preview TEXT NOT NULL DEFAULT '',
    -- Unread tracking from this user's perspective. Bumped when peer
    -- sends a message; cleared when this user views the thread.
    unread_count        INTEGER NOT NULL DEFAULT 0,
    -- Per-thread user preferences.
    muted               INTEGER NOT NULL DEFAULT 0,
    pinned              INTEGER NOT NULL DEFAULT 0,
    archived            INTEGER NOT NULL DEFAULT 0,
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (thread_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_connect_threads_user
    ON connect_threads(user_id, last_message_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_connect_threads_pair
    ON connect_threads(user_id, peer_did);


CREATE TABLE IF NOT EXISTS connect_messages (
    -- (message_id, user_id) composite PK: message_id is shared
    -- between sender and recipient so the wire-level message id
    -- matches across both copies; each user stores their own row.
    message_id          TEXT NOT NULL,              -- shared message id
    thread_id           TEXT NOT NULL,              -- references connect_threads
    user_id             TEXT NOT NULL,              -- owner of this copy (scoping)
    -- Sender identity. Could be this user (outgoing) or the peer (incoming).
    sender_did          TEXT NOT NULL,
    -- Message body and metadata. format describes how to render:
    --   'plain'       — plain text
    --   'markdown'    — markdown formatting
    --   'voice_note'  — voice note (body holds optional transcript;
    --                   audio reference in attachment_ref)
    --   'embed'       — Augmentum-object embed (character/narrative/
    --                   coder/pack — body holds the embed JSON)
    body                TEXT NOT NULL DEFAULT '',
    format              TEXT NOT NULL DEFAULT 'plain',
    -- Attachments stored separately (audio files, embed targets) and
    -- referenced by stable ID. Phase 1: voice_note audio + Augmentum
    -- surface references. Future: images, files.
    attachment_ref      TEXT NOT NULL DEFAULT '',
    -- Reply-to forms a soft thread tree. Phase 1 renders one level
    -- of reply context; nested replies stay flat (no nested tree UI).
    reply_to            TEXT NOT NULL DEFAULT '',
    -- Lifecycle timestamps. Bi-temporal: sent_at = sender's wall-clock,
    -- received_at = our instance's clock when we wrote the row.
    -- See [[hybrid-logical-clocks]] in the 20-primitives brainstorm —
    -- when HLCs land, sent_at gets supplemented by an HLC.
    sent_at             TIMESTAMP NOT NULL,
    received_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    delivered_at        TIMESTAMP,
    read_at             TIMESTAMP,
    -- Edit / delete tracking. Soft-delete: keep the row, clear the body.
    edited_at           TIMESTAMP,
    deleted_at          TIMESTAMP,
    -- For voice notes specifically: transcript inline if auto-transcribe
    -- was enabled at send time. Empty otherwise.
    transcript          TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (message_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_connect_messages_thread
    ON connect_messages(thread_id, sent_at DESC);
CREATE INDEX IF NOT EXISTS idx_connect_messages_user
    ON connect_messages(user_id, sent_at DESC);


-- Trigger: keep connect_threads.last_message_at / preview / unread_count
-- in sync with the actual messages. Cheaper than recomputing on read.
-- Only fires for non-deleted inserts.
--
-- The "incoming vs outgoing" check looks up THIS user's peer_did
-- (NEW.user_id-scoped) because both participants share the
-- (thread_id) but each has their own peer_did from their perspective.
-- Without the user_id scope SQLite can pick either perspective's
-- row and the trigger silently mis-attributes ownership.
CREATE TRIGGER IF NOT EXISTS connect_messages_after_insert
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
               THEN unread_count   -- our own outgoing message, no unread bump
               ELSE unread_count + 1
           END
     WHERE thread_id = NEW.thread_id
       AND user_id   = NEW.user_id;
END;
