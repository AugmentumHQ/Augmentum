-- 297_guest_device_reconnect.sql
-- Guest portal reconnection: carry the guest's web-device identity from
-- registration through to admin-confirm, so confirming a guest also
-- registers a TRUSTED DEVICE (trusted_mobile_devices, platform='web').
--
-- The device-bound session that flows from that is IP-INDEPENDENT — it's
-- what lets the guest reconnect from home WiFi / cellular / anywhere
-- without the per-guest IP allowlist (which broke mobility). Reuses the
-- Android mobile-pairing device model wholesale; these two columns are the
-- only schema add needed to bridge it into the guest flow.

ALTER TABLE guest_registrations ADD COLUMN device_id TEXT NOT NULL DEFAULT '';
ALTER TABLE guest_registrations ADD COLUMN device_public_key TEXT NOT NULL DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (297, 'guest_registrations += device_id/device_public_key: bridge guest registration into trusted-device reconnection');
