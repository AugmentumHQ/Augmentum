-- 291_fabric_revocations.sql
-- Connect federated-PBX P3: key revocation tombstones + abuse denylist.
--
-- SERVER-LEVEL (not user-scoped, like domain_reputation): a revoked key
-- is globally revoked, and an operator's abuse block applies instance-
-- wide. Both are keyed on the canonical did:key.
--
-- D4 severed the directory as a publication channel, so revocations are
-- delivered out-of-band: served from this instance's .well-known and
-- pushed to already-known peers (revocation.py). A pinned contact that
-- learns of a revocation must drop to unverified and refuse the key.

-- Signed tombstones. A revocation is self-signed by the revoked key
-- (proves the holder is retiring it) or by a pre-committed succession
-- key. We store the whole signed object so it can be re-served verbatim.
CREATE TABLE IF NOT EXISTS fabric_revocations (
    revoked_did_key   TEXT PRIMARY KEY,                  -- the retired identity
    reason            TEXT NOT NULL DEFAULT '',
    supersedes_to     TEXT NOT NULL DEFAULT '',          -- successor did:key, if any
    tombstone_json    TEXT NOT NULL,                     -- the full signed object
    issued_at         INTEGER NOT NULL DEFAULT 0,
    recorded_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Subscribable abuse denylist. Entries can be locally added by the
-- operator or imported from a denylist another instance publishes
-- (source = the publishing instance's did:key, for provenance/unsub).
CREATE TABLE IF NOT EXISTS fabric_denylist (
    did_key       TEXT NOT NULL,                         -- blocked identity
    reason        TEXT NOT NULL DEFAULT '',
    source        TEXT NOT NULL DEFAULT 'local',         -- 'local' | publisher did:key
    added_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (did_key, source)
);

CREATE INDEX IF NOT EXISTS idx_fabric_denylist_did ON fabric_denylist(did_key);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (291, 'fabric_revocations + fabric_denylist: signed key tombstones + subscribable abuse blocks (Connect federated-PBX P3)');
