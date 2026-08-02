-- 190_trusted_receivers_wol.sql
--
-- Wake-on-LAN support per trusted receiver. A TV that's powered off
-- can't connect to the WS so it isn't in the runtime registry — but
-- the row in trusted_receivers persists. Storing the MAC + the last
-- local IP we saw on this row lets the user "wake" the TV from cast-
-- control without having to remember addresses themselves.
--
--   mac_address           — canonical lowercase aa:bb:cc:dd:ee:ff
--                           ('' = unknown, user can fill in manually)
--   last_local_ip         — last LAN IP the receiver self-reported on
--                           its ready event (e.g. '192.168.1.42').
--                           Used to derive the default broadcast addr
--                           (.255) when wol_broadcast_override is empty.
--   wol_broadcast_override — opt-in per-receiver broadcast (e.g. a
--                           non-/24 subnet, or 255.255.255.255 when
--                           host networking is available). Empty =
--                           auto-derive from last_local_ip.
--
-- All three are NOT NULL DEFAULT '' so older rows keep working with
-- zero migration drama; the helper layer treats empty as "unknown".

ALTER TABLE trusted_receivers
    ADD COLUMN mac_address TEXT NOT NULL DEFAULT '';

ALTER TABLE trusted_receivers
    ADD COLUMN last_local_ip TEXT NOT NULL DEFAULT '';

ALTER TABLE trusted_receivers
    ADD COLUMN wol_broadcast_override TEXT NOT NULL DEFAULT '';

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (
    190,
    'trusted_receivers.mac_address / last_local_ip / wol_broadcast_override for Wake-on-LAN'
);
