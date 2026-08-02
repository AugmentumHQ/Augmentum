-- 293_fabric_connect_codes.sql
-- Connect federation UX: short, shareable "connect codes".
--
-- A signed contact card is a long base64 blob — fine for a QR, awful to
-- read or type. This table maps a short, human-friendly code (e.g.
-- "K7P2-9QX4") to the full card so sharing feels like a product: "scan
-- this, or enter code K7P2-9QX4." Codes expire so stale invites lapse.
--
-- SERVER-LEVEL infrastructure keyed by the code; ``user_id`` records who
-- minted it (for listing/revoking your own invites).

CREATE TABLE IF NOT EXISTS fabric_connect_codes (
    code        TEXT PRIMARY KEY,              -- short shareable code (no ambiguous chars)
    user_id     TEXT NOT NULL,                 -- who minted it
    card_json   TEXT NOT NULL,                 -- the full signed contact card
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    expires_at  INTEGER NOT NULL DEFAULT 0     -- epoch seconds; 0 = no expiry
);

CREATE INDEX IF NOT EXISTS idx_fabric_connect_codes_user
    ON fabric_connect_codes(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fabric_connect_codes_exp
    ON fabric_connect_codes(expires_at);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (293, 'fabric_connect_codes: short shareable connect codes for contact cards (Connect federation UX)');
