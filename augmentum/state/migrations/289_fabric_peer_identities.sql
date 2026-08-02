-- 289_fabric_peer_identities.sql
-- Connect federated-PBX P1: the did:key pin + verified-state layer.
--
-- The TOFU (trust-on-first-use) record of "this user has seen peer
-- identity <did:key> and here is how much we trust it." Distinct from:
--   * fabric_nodes (server-level instance<->instance PAIRING for the
--     compute fabric) — that's the transport substrate.
--   * connect_contacts (user-scoped peer ROWS, peer_did = user@instance) —
--     that's the address book.
-- This table is the SECURITY layer between them: the byte-comparable
-- did:key the contact resolves to, and whether the human ever verified
-- it out-of-band (the SAS/QR ceremony). Every downstream trust decision
-- (caller-ID, relay, message-request) keys on the row here, NOT on the
-- attacker-controllable display name.
--
-- USER-SCOPED (user_id) on purpose: TOFU is per-user. If user A pins
-- peer X's key, that must NOT silently become trust for user B on the
-- same instance — each user verifies independently. UNIQUE(user_id,
-- peer_did_key) lets two users hold independent verified-state for the
-- same peer, and lets us detect a key CHANGE for a known handle (the
-- "safety number changed" signal) by querying handle + a new did_key.

CREATE TABLE IF NOT EXISTS fabric_peer_identities (
    id                TEXT PRIMARY KEY,                  -- uuid hex
    user_id           TEXT NOT NULL,
    -- The canonical, byte-comparable identity. did:key:z... (Ed25519).
    peer_did_key      TEXT NOT NULL,
    -- Display handle (user@instance). Attacker-controllable — NEVER a
    -- trust input; stored only to render + to detect handle/key splits.
    handle            TEXT NOT NULL DEFAULT '',
    -- Last-known reachable endpoint (mutable; the did:key is permanent).
    endpoint          TEXT NOT NULL DEFAULT '',
    -- Per-user authenticity key bound into the ceremony (P2 splits this
    -- from the instance key via device subkeys; P1 carries it through so
    -- the wire format + SAS are forward-compatible).
    author_did_key    TEXT NOT NULL DEFAULT '',
    -- Trust state. 0 = pinned-not-verified (TOFU), 1 = verified via an
    -- out-of-band ceremony. The UI MUST render the difference (D1-01).
    verified          INTEGER NOT NULL DEFAULT 0,
    verified_method   TEXT NOT NULL DEFAULT '',          -- 'sas' | 'qr' | ''
    verified_at       TEXT,
    -- How this identity first entered the table.
    source            TEXT NOT NULL DEFAULT 'card',       -- 'invite'|'card'|'knock'
    first_pinned_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at        TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(user_id, peer_did_key)
);

CREATE INDEX IF NOT EXISTS idx_fabric_peer_ident_user
    ON fabric_peer_identities(user_id, peer_did_key);
-- Handle lookup powers the "safety-number-changed" detection: same
-- handle, different did_key => surface a key-change warning.
CREATE INDEX IF NOT EXISTS idx_fabric_peer_ident_handle
    ON fabric_peer_identities(user_id, handle);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (289, 'fabric_peer_identities: per-user did:key TOFU pin + out-of-band verified-state (Connect federated-PBX P1)');
