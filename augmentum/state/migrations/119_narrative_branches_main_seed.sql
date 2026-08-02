-- 119_narrative_branches_main_seed.sql
-- Backfill the new branch-tagged tables (115/116/117) with existing narrative
-- data. Three independent INSERTs, each idempotent via INSERT OR IGNORE or
-- NOT EXISTS guards so partial-apply scenarios self-heal on retry.
--
-- Stage 1 (here): seed 'main' branch rows + copy current STATE snapshot + unpack
-- current LEDGER JSON array into rows. Handles the 95% case (every existing
-- session gets a properly-seeded main branch).
--
-- Stage 2 (separate): scripts/migrate_narrative_branches.py runs after this
-- migration to unpack alternate branches from narrative_memory.branch_states
-- into proper rows. Marker-based idempotency in app_settings.
--
-- DEFENSIVE GUARDS:
-- 1. EXISTS-against-sessions filter: narrative_memory may hold rows for
--    session_ids no longer present in `sessions` (legacy dev experimentation,
--    aborted imports). New tables have FK to sessions(id) ON DELETE CASCADE,
--    so seeding orphan rows would abort the whole migration via FK violation.
-- 2. round_num NOT NULL filter: legacy memory_ledger entries may be malformed.
--    json_extract returns NULL for missing fields; the new ledger_entries
--    table has round_num INTEGER NOT NULL.
-- 3. json_valid() on every JSON column: corrupt blobs are skipped, not aborted.

-- 1. Seed a 'main' branch row for every session that has narrative_memory
--    data AND a corresponding sessions row. Composite PK (session_id, 'main')
--    makes this idempotent on retry.
INSERT OR IGNORE INTO narrative_branches
    (branch_id, session_id, parent_branch_id, branch_point, status, user_id)
SELECT 'main', nm.session_id, NULL, 0, 'active', nm.user_id
  FROM narrative_memory nm
 WHERE EXISTS (SELECT 1 FROM sessions s WHERE s.id = nm.session_id);

-- 2. Copy current state_snapshot into a single snapshot history row at
--    message_index = message_count. Skip empty/null/default-{} snapshots and
--    skip sessions that already have a main-branch snapshot row (re-run guard).
INSERT INTO narrative_state_snapshots
    (id, session_id, branch_id, message_index, snapshot_data, user_id)
SELECT lower(hex(randomblob(16))),
       nm.session_id,
       'main',
       COALESCE(nm.message_count, 0),
       nm.state_snapshot,
       nm.user_id
  FROM narrative_memory nm
 WHERE nm.state_snapshot IS NOT NULL
   AND nm.state_snapshot != ''
   AND nm.state_snapshot != 'null'
   AND nm.state_snapshot != '{}'
   AND json_valid(nm.state_snapshot)
   AND EXISTS (SELECT 1 FROM sessions s WHERE s.id = nm.session_id)
   AND NOT EXISTS (
       SELECT 1 FROM narrative_state_snapshots ss
        WHERE ss.session_id = nm.session_id
          AND ss.branch_id = 'main'
   );

-- 3. Unpack memory_ledger JSON array into row-per-entry. Skip empty/null/default
--    arrays, skip invalid JSON, skip sessions without a sessions row, skip
--    entries with a missing/null round_num, and skip sessions that already
--    have main-branch ledger entries (re-run guard — coarse but safe for backfill).
INSERT INTO narrative_ledger_entries
    (id, session_id, branch_id, round_num, category, content, user_id)
SELECT lower(hex(randomblob(16))),
       nm.session_id,
       'main',
       CAST(json_extract(je.value, '$.round_num') AS INTEGER),
       COALESCE(json_extract(je.value, '$.category'), ''),
       COALESCE(json_extract(je.value, '$.content'), ''),
       nm.user_id
  FROM narrative_memory nm, json_each(nm.memory_ledger) je
 WHERE nm.memory_ledger IS NOT NULL
   AND nm.memory_ledger != ''
   AND nm.memory_ledger != 'null'
   AND nm.memory_ledger != '[]'
   AND json_valid(nm.memory_ledger)
   AND json_extract(je.value, '$.round_num') IS NOT NULL
   AND EXISTS (SELECT 1 FROM sessions s WHERE s.id = nm.session_id)
   AND NOT EXISTS (
       SELECT 1 FROM narrative_ledger_entries le
        WHERE le.session_id = nm.session_id
          AND le.branch_id = 'main'
   );

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (119, 'narrative_branches main-seed backfill');
