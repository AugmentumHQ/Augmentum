-- 247_companion_verb_log.sql
--
-- Append-only audit ledger for the management-verb dispatcher
-- (companion_runtime/event_bus.py). One row per invocation, regardless
-- of outcome (success, cooldown_skipped, autonomy_gated, error, etc).
--
-- WHY a new table rather than extending existing audit substrates:
-- The Phase 1 catalog (docs/superpowers/working/companion_management
-- _verb_catalog.md) audited every candidate (companion_state_log,
-- companion_skill_archive, companion_economy_tx, signal_events) and
-- found none shape-compatible — they're transaction logs at the wrong
-- abstraction layer. The verb_log lives ABOVE them: when a state
-- transition or skill dispatch happens, the verb that caused it logs
-- here AND the substrate log records its own row. `cited_substrate`
-- JSON carries the joint references.
--
-- Two load-bearing roles:
--
-- 1. **Cooldown enforcement** — every verb dispatch reads MAX(fired_at)
--    WHERE (user_id, companion_id, verb_name) before firing. Replaces
--    the eight scattered module-globals (today._LAST_REGEN_AT,
--    runtime._last_curator_at, runtime._last_pad, runtime._last_affect
--    _tag, runtime._last_journal_at, runtime.last_initiative_score_at,
--    curator constants, etc.) that today reset on every restart.
--
-- 2. **Chain-depth limit** — every invocation can record
--    cited_verb_log_id pointing at the row that emitted the event it's
--    responding to. The dispatcher walks this chain to enforce max
--    depth 2 (prevents cascading verb fan-outs).
--
-- Outcome enumeration (TEXT — SQLite STRICT has no real enums):
--   ok                       — verb ran cleanly
--   cooldown_skipped         — last_fired_at within cooldown_ms
--   budget_exceeded          — wallclock_ms or db_ops over envelope
--   autonomy_gated           — presence_mode disallowed
--   chain_depth_exceeded     — event was N-th in a verb fan-out chain
--   error                    — exception in verb body
--   auto_paused              — consecutive_error_count reached threshold
--
-- Provenance:
-- - cited_substrate: JSON array — which tables/rows the verb touched.
--   Example: '[{"table":"companion_affect_baselines","row_id":42}]'
-- - args_hash: SHA-256 hex of (sorted JSON) event payload. Lets the
--   dispatcher dedupe identical events fanned to the same verb within
--   a coalesce window.
-- - cited_verb_log_id: parent row in this same table. NULL when fired
--   from a non-verb source (time tick, user action, system signal).
--
-- See spec:
-- docs/superpowers/specs/2026-06-05-companion-verbs-architecture-design.md
-- Phase 2 substrate.

CREATE TABLE IF NOT EXISTS companion_verb_log (
    id                  INTEGER PRIMARY KEY,
    user_id             TEXT NOT NULL DEFAULT '',
    companion_id        TEXT NOT NULL DEFAULT 'becca',
    verb_name           TEXT NOT NULL,
    verb_class          TEXT NOT NULL,           -- 'management' | 'core'
    dispatch_class      TEXT NOT NULL DEFAULT '',-- 'IDLE_OK' | 'TICK_ALIGNED' | 'EVENT_DRIVEN'
    event_topic         TEXT NOT NULL DEFAULT '',
    event_id            TEXT NOT NULL DEFAULT '',
    args_hash           TEXT NOT NULL DEFAULT '',
    outcome             TEXT NOT NULL DEFAULT '',
    latency_ms          INTEGER NOT NULL DEFAULT 0,
    db_ops              INTEGER NOT NULL DEFAULT 0,
    error               TEXT NOT NULL DEFAULT '',
    cited_substrate     TEXT NOT NULL DEFAULT '',
    cited_verb_log_id   INTEGER,
    fired_at            INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER))
) STRICT;

-- Cooldown lookup: hot path. Dispatcher reads MAX(fired_at) here on
-- every event fanout to decide if the verb should fire. Per-user
-- because cooldowns are user-scoped (one user's morning briefing
-- doesn't lock another's).
CREATE INDEX IF NOT EXISTS idx_verb_log_cooldown_lookup
    ON companion_verb_log(user_id, companion_id, verb_name, fired_at DESC);

-- Observability surface (Phase 5): "Becca's day" panel filters by
-- outcome+recency to render what fired, what was skipped, and why.
CREATE INDEX IF NOT EXISTS idx_verb_log_outcome_recent
    ON companion_verb_log(user_id, outcome, fired_at DESC);

-- Chain-walk: when enforcing depth limit, dispatcher follows the
-- cited_verb_log_id pointer from the current event back to the root.
-- Rare query but cheap index since cited_verb_log_id is sparse.
CREATE INDEX IF NOT EXISTS idx_verb_log_chain
    ON companion_verb_log(cited_verb_log_id)
    WHERE cited_verb_log_id IS NOT NULL;

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (247, 'companion_verb_log — management-verb dispatcher audit ledger');
