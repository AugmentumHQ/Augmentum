-- 250_backfill_memory_events_default_user.sql
--
-- Backfill memory_events rows stranded under user_id='default'.
--
-- Background: log_event() defaulted user_id to the literal "default"
-- string, and five call sites (4 in memory/store.py, 1 in
-- dream/scheduler.py) all knew the real user but didn't pass it through.
-- 120 rows accumulated from 2026-04-11 through 2026-06-06 under a
-- non-existent "default" user. The writer is fixed in the same commit;
-- this migration repatriates the existing rows.
--
-- Recovery sources (verified against the live DB before authoring):
--   * dream_cycle (32/32): user_id is embedded in the detail JSON
--     ($.user_id) by the scheduler's own logging.
--   * promotion + tier_change (86/88): JOIN memory_id -> memories.user_id
--     recovers the owner. The 2 remaining rows reference memories that
--     no longer exist (deleted) — those events are unrecoverable and
--     get deleted.
--
-- Idempotent: every UPDATE filters on user_id='default', so a re-run
-- after the first pass affects 0 rows.

-- 1. Repatriate dream_cycle events from the embedded JSON user_id.
UPDATE memory_events
SET user_id = json_extract(detail, '$.user_id')
WHERE user_id = 'default'
  AND event_type = 'dream_cycle'
  AND json_extract(detail, '$.user_id') IS NOT NULL
  AND json_extract(detail, '$.user_id') != '';

-- 2. Repatriate promotion + tier_change events from the parent memory.
--    Use a correlated subquery so we update without a JOIN syntax
--    (SQLite UPDATE...FROM landed in 3.33 but staying portable).
UPDATE memory_events
SET user_id = (
    SELECT m.user_id FROM memories m WHERE m.id = memory_events.memory_id
)
WHERE user_id = 'default'
  AND event_type IN ('promotion', 'tier_change')
  AND memory_id IS NOT NULL
  AND EXISTS (
      SELECT 1 FROM memories m WHERE m.id = memory_events.memory_id
  );

-- 3. The remaining 'default' rows reference memories that have been
--    deleted — the events are dead references to gone memories. Delete
--    rather than synthesize a fake owner.
DELETE FROM memory_events WHERE user_id = 'default';
