-- 294_fabric_device_bundles.sql
-- Connect E2E P2: published device bundles (PUBLIC keys only).
--
-- For end-to-end encryption a sender must learn the recipient's device
-- sealing keys. Each user publishes a bundle: their master author key +
-- one entry per device (signing-subkey did, X25519 sealing pubkey, and
-- the master-signed binding proving the device is authorized). ONLY
-- public material lives here — private keys never leave the device.
--
-- The server validates every binding chains to the bundle's master_did
-- before storing (a malformed/forged bundle is rejected at the door).
-- The recipient-side trust decision (does this master == the one I
-- verified in the ceremony?) is enforced by the CLIENT against its pin.
--
-- USER-SCOPED: one bundle per local user.

CREATE TABLE IF NOT EXISTS fabric_device_bundles (
    user_id     TEXT PRIMARY KEY,
    master_did  TEXT NOT NULL,                  -- the user's master author key
    bundle_json TEXT NOT NULL,                  -- {"devices":[{subkey_did,sealing_pub_b64,binding,label}]}
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (294, 'fabric_device_bundles: published per-user device bundles (public keys) for E2E (Connect E2E P2)');
