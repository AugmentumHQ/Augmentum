-- Connect fabric outbox — durable buffer for cross-instance envelopes
-- that couldn't be flushed immediately because the target peer's WS
-- was disconnected.
--
-- One row per pending envelope. The row is written BEFORE the first
-- send attempt; on successful send it's deleted (no "sent_at" column —
-- we only track pending state, and a sent envelope is gone). On
-- FabricClient reconnect, we drain in queued_at order.
--
-- ``target_node_id`` is the fabric node_id (e.g. ``peer-1``), not the
-- DID host part. The mapping DID→node_id happens at enqueue time via
-- FabricCoordinator lookup; the outbox stores the resolved node so a
-- subsequent rename of the hostname doesn't lose the queued envelope.
--
-- ``envelope_json`` is the full wire-form ConnectEnvelope (text /
-- call signaling frame) the inbound handler will replay on the
-- recipient instance. Bounded by the ConnectEnvelope 64 KB cap.
--
-- ``attempts`` lets the drain loop give up cleanly after a max
-- retry count, surfacing EVENT_ERROR("fabric_outbox_exhausted") to
-- the local sender so their UI stops showing "sending..." forever.

CREATE TABLE IF NOT EXISTS connect_fabric_outbox (
    id               TEXT PRIMARY KEY,
    target_node_id   TEXT NOT NULL,
    sender_user_id   TEXT NOT NULL,        -- for error fan-back on giveup
    envelope_json    TEXT NOT NULL,
    queued_at        INTEGER NOT NULL,     -- unix seconds
    attempts         INTEGER NOT NULL DEFAULT 0,
    last_attempt_at  INTEGER,
    last_error       TEXT
);

CREATE INDEX IF NOT EXISTS idx_connect_fabric_outbox_drain
    ON connect_fabric_outbox (target_node_id, queued_at);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (241, 'connect fabric outbox');
