-- 303_workspace_lan_accessible.sql
-- Per-workspace LAN accessibility toggle.
--
-- When enabled, published ports bind to 0.0.0.0 (LAN-reachable) instead
-- of 127.0.0.1 (loopback-only). Deliberate user action only — never
-- auto-enabled. Revertible: toggling off recreates the container with
-- loopback bindings; the workspace volume (data) is unaffected.
--
-- Also enables gate-subdomain integration: when gate_domain is configured,
-- a LAN-accessible workspace with listening ports gets a Caddy reverse-proxy
-- snippet at <workspace-slug>.<gate_domain> with HTTPS + Augmentum auth.

ALTER TABLE project_checkouts ADD COLUMN lan_accessible INTEGER NOT NULL DEFAULT 0;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (303, 'per-workspace lan_accessible toggle');
