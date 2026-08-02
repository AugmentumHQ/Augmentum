-- 313: invite bundle base persistence (guest gateway).
--
-- The mint-time reachability plan picks a public base (e.g. an ephemeral
-- cloudflared quick-tunnel URL). The QR endpoint and any later re-render must
-- reproduce the IDENTICAL bundle URL, so the chosen base is recorded on the
-- invite row instead of being re-derived from whatever Host header the QR
-- request happens to arrive under.
-- Spec: docs/superpowers/specs/2026-07-16-guest-gateway-anonymous-tunnel-e2e-design.md
ALTER TABLE auth_invites ADD COLUMN join_base TEXT NOT NULL DEFAULT '';
