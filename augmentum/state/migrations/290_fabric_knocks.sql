-- 290_fabric_knocks.sql
-- Connect federated-PBX P2: the deny-by-default knock tier.
--
-- A "knock" is a stranger's request to reach a user who has NOT pinned
-- them. It is the ONLY way an unknown party gets through, and it is
-- deliberately weak: it does NOT ring, the intro text is WITHHELD until
-- the recipient accepts, it is rate-limited at the dispatcher, and it
-- carries the envelope-verified source did:key (caller_id.py) so it
-- can't be spoofed. This structurally kills disposable-identity spam —
-- the gate is receiver-side and admission is explicit.
--
-- Admission posture (per the recipient's `fabric_admission_posture`
-- setting): private (no knocks) | allowlist | knock (default) | open.
--
-- USER-SCOPED (to_user_id): a knock targets a specific local user.

CREATE TABLE IF NOT EXISTS fabric_knocks (
    id              TEXT PRIMARY KEY,                  -- uuid hex
    to_user_id      TEXT NOT NULL,                     -- recipient (local)
    -- Envelope-verified source identity (NEVER the body's claim).
    from_did_key    TEXT NOT NULL,
    from_handle     TEXT NOT NULL DEFAULT '',          -- display only, untrusted
    -- Intro is withheld pre-accept: stored but not delivered/surfaced
    -- until status flips to 'accepted'. Classifier flag lets the UI
    -- quarantine likely-abuse without showing content.
    intro_text      TEXT NOT NULL DEFAULT '',
    intro_flagged   INTEGER NOT NULL DEFAULT 0,        -- 1 = classifier-flagged
    -- Source IP at intake — a scarce axis for rate limiting (per the v2
    -- fix: limit on IP/WS, not on the free-to-mint did/Number).
    src_ip          TEXT NOT NULL DEFAULT '',
    status          TEXT NOT NULL DEFAULT 'pending',   -- pending|accepted|rejected|expired
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    decided_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_fabric_knocks_user_status
    ON fabric_knocks(to_user_id, status);
-- Rate-limit lookups count recent pending knocks by source key + by IP.
CREATE INDEX IF NOT EXISTS idx_fabric_knocks_from
    ON fabric_knocks(from_did_key, created_at);
CREATE INDEX IF NOT EXISTS idx_fabric_knocks_ip
    ON fabric_knocks(src_ip, created_at);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (290, 'fabric_knocks: deny-by-default stranger knock tier, intro withheld + rate-limited (Connect federated-PBX P2)');
