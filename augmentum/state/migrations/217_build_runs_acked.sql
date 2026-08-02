-- 217_build_runs_acked.sql
-- Cross-device dismissal for terminal Build Mode runs.
--
-- The persistent build monitor surfaces the most recent build via
-- /api/artifacts/build-status, which falls through to
-- BuildRunStore.latest_for_session() when ACTIVE_BUILDS is cold
-- (every server restart). That query had no acked or TTL predicate,
-- so a single terminal build would resurface on every page load,
-- on every device, indefinitely. The client localStorage ack only
-- covered the originating browser.
--
-- This column records server-side dismissal. latest_for_session()
-- filters out terminal+acked rows, and an additional 24h TTL on
-- terminal recall guards the "never explicitly dismissed" case.

ALTER TABLE build_runs ADD COLUMN acked_at TEXT;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (217, 'build_runs.acked_at for cross-device dismissal of terminal builds');
