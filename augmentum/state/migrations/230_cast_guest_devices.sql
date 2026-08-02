-- Cast couch co-op guest device fingerprints (Phase 3).
--
-- Spec: docs/superpowers/specs/2026-06-02-cast-couch-coop-design.md
--
-- A guest profile (table 229) is the named identity; a guest_device
-- row is the link between a specific browser-on-phone and that
-- profile, so the same alice can rejoin from her phone without
-- re-typing her name.
--
-- Multiple devices per profile is intentional — alice may use her
-- phone AND her tablet, both should resolve to her profile and the
-- same save data (Phase 4).
--
-- Fingerprint shape: localStorage UUID (the primary signal) +
-- UA hash (defence against accidental UUID rotation). NO canvas /
-- audio / IP fingerprinting — opt-in only, privacy thesis.
--
-- User-scoped on host_user_id alongside guest_profile_id so the
-- audit + delete-cascade story stays clean even if a profile is
-- somehow reparented.

CREATE TABLE IF NOT EXISTS guest_devices (
    id                 TEXT PRIMARY KEY,         -- gd_<token12>
    guest_profile_id   TEXT NOT NULL REFERENCES guest_profiles(id) ON DELETE CASCADE,
    host_user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- localStorage UUID minted on the guest's first visit. Per-host
    -- UNIQUE so the same device at two different hosts maintains
    -- independent links (cross-host portability is a non-feature).
    device_uuid        TEXT NOT NULL,
    -- sha1ish of UA + viewport. Used to spot likely fingerprint
    -- rotation (cleared localStorage but same device); the welcome-
    -- back path requires BOTH device_uuid AND ua_hash to match for
    -- an auto-resume.
    ua_hash            TEXT NOT NULL DEFAULT '',
    -- Optional guest-volunteered label (Phase 3 follow-up). Empty
    -- until the guest taps "Label this device" — host never sees
    -- a label the guest didn't explicitly type.
    label              TEXT NOT NULL DEFAULT '',
    first_seen_at      INTEGER NOT NULL,
    last_seen_at       INTEGER NOT NULL,
    UNIQUE (host_user_id, device_uuid)
);

CREATE INDEX IF NOT EXISTS idx_guest_devices_host
    ON guest_devices (host_user_id);
CREATE INDEX IF NOT EXISTS idx_guest_devices_profile
    ON guest_devices (guest_profile_id);
