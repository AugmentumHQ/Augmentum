-- Fabric identity table: durable record of paired peers in the user's
-- fabric. Server-level (not user-scoped) because peer trust is a
-- whole-instance concern -- a user deleting their account shouldn't
-- evict the operator's peer pairings. Mirrors the providers table
-- pattern (also server-level).
--
-- High-velocity peer state (heartbeats, capabilities, current load)
-- intentionally NOT stored here -- that lives in app.state.fabric_state
-- as in-memory RAM only. Persisting heartbeats would cause writer-lock
-- contention with the resource ledger (2026-05-15 incident). This
-- table is for durable identity ONLY: rows written at pair time,
-- updated rarely (rename, role change, share-toggle).
--
-- pubkey_ed25519 is the peer's ed25519 public key (32 bytes, base64-
-- encoded). pubkey_fingerprint is a short, human-readable
-- "SHA256:abcd...wxyz" form for verification UX (SSH-style).
-- This local instance's OWN identity lives in the settings_store as
-- fabric.node_id / fabric.node_private_key -- NOT in this table.

CREATE TABLE IF NOT EXISTS fabric_nodes (
    id TEXT PRIMARY KEY,
    hostname TEXT NOT NULL DEFAULT '',
    role TEXT NOT NULL DEFAULT 'peer',
    pubkey_ed25519 TEXT NOT NULL,
    pubkey_fingerprint TEXT NOT NULL,
    addr TEXT NOT NULL DEFAULT '',
    tier TEXT NOT NULL DEFAULT 'local',
    fabric_share_enabled INTEGER NOT NULL DEFAULT 1,
    paired_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_fabric_nodes_role ON fabric_nodes(role);
CREATE INDEX IF NOT EXISTS idx_fabric_nodes_tier ON fabric_nodes(tier);
