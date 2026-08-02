-- 292_fabric_durable_guards.sql
-- Connect federated-PBX production hardening: durable anti-replay state.
--
-- The security review (SEC-7, SEC-8) flagged that the replay window
-- (relay_seal.ReplayWindow) and the PoW single-use guard
-- (pow.ConsumedNonces) were in-memory only — a process restart reopened
-- both windows. For production these MUST survive restarts, so the live
-- path uses the durable store (fabric/durable_guards.py) backed by these
-- tables instead of the in-memory classes.
--
-- SERVER-LEVEL infrastructure (like domain_reputation): keyed by an
-- ``owner_id`` scope (a user_id for per-user E2E streams, '' for
-- instance-level relay) rather than the multi-tenant user_id column.

-- Per-(owner, source) monotonic sequence high-water mark. A sealed frame
-- whose seq <= the stored high-water is a replay and is rejected.
CREATE TABLE IF NOT EXISTS fabric_replay_watermarks (
    owner_id    TEXT NOT NULL DEFAULT '',     -- recipient scope (user_id or '')
    source_did  TEXT NOT NULL,                -- authenticated sender did:key
    high_seq    INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (owner_id, source_did)
);

-- Single-use PoW nonces. A solved challenge's nonce is spent once; a
-- replay of the (challenge, solution) pair is rejected. ``expires_at`` is
-- an epoch second past which the row may be pruned (the challenge TTL has
-- elapsed, so the nonce can never be presented again anyway).
CREATE TABLE IF NOT EXISTS fabric_consumed_nonces (
    nonce       TEXT PRIMARY KEY,
    expires_at  INTEGER NOT NULL DEFAULT 0,
    consumed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_fabric_consumed_nonces_exp
    ON fabric_consumed_nonces(expires_at);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (292, 'fabric_replay_watermarks + fabric_consumed_nonces: durable anti-replay state (Connect federated-PBX production hardening, SEC-7/SEC-8)');
