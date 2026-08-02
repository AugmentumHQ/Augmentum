-- Phase 8: per-peer visual identifier.
--
-- Each paired remote box gets an operator-chosen icon (emoji) shown
-- adjacent to its hostname, in capability-matrix column headers, and
-- as a small badge on chat turns the box served. The icon is picked
-- by the LOCAL operator at pair time (not self-advertised by the
-- remote), so each operator labels their own fleet from their own
-- perspective. Defaults to '' meaning "no icon assigned" — the UI
-- falls back to a generic 🔗 in that case.
--
-- Safe schema change: nullable→empty-string default; existing rows
-- get '' on the ALTER without rewriting. The pair form will start
-- populating it for new pairings; existing peers can be re-iconed
-- via a future "edit peer" UI (not in Phase 8 scope).

ALTER TABLE fabric_nodes ADD COLUMN icon TEXT NOT NULL DEFAULT '';
