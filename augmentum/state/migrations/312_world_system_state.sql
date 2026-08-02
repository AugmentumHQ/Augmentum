-- 312_world_system_state.sql
-- World-system tracker state for the card-declared world manifest
-- (docs/superpowers/specs/2026-07-15-world-system-manifest-design.md).
--
-- One JSON column on narrative_memory rather than a new table: tracker
-- state is small (a dict of current values + bounded per-tracker history),
-- strictly session-scoped, and must ride the exact same save/load/branch
-- lifecycle as the rest of NarrativeSessionState — a sibling column gets
-- branch rollback, restart survival, and user scoping for free. The
-- manifest itself is NOT persisted here; it re-parses from the character
-- card (single source of truth) on session load.

ALTER TABLE narrative_memory ADD COLUMN world_state TEXT DEFAULT '{}';
