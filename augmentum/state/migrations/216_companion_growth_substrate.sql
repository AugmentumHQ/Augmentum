-- 216_companion_growth_substrate.sql
-- Substrate for the companion growth loop —
-- docs/superpowers/specs/2026-05-31-companion-growth-loop-design.md
--
-- Four tables that make growth a first-class thing the companion does,
-- distinct from chat handling. They do NOT replace any spine substrate
-- (drives, activity_selector, initiative_queue, skill_archive) — they
-- sit on top.
--
-- Why four tables and not one event log:
--   * backlog  — the queue of growth tasks she might pursue
--   * log      — the per-session record of plan + act + verify + outcome
--   * economy  — current mana + berry balance per (user, agent)
--   * tx       — append-only audit trail of every mana/berry movement
-- Splitting the balance from the transaction log lets the balance be a
-- cheap point read while the audit log grows unbounded. Splitting the
-- backlog from the log keeps "what she might do" out of "what she did."
--
-- All four are user-scoped + agent-scoped per the multi-tenant rules
-- in CLAUDE.md. Default agent_id 'becca' because the only companion
-- today is Becca; future siblings (Sage, Librarian, Dreamer per the
-- spine spec) re-use the same tables under different agent_ids.
--
-- Berries / mana vocabulary: the spec's open question 6 calls for
-- public-facing names (standing / capacity / credits). The SCHEMA
-- uses the internal terms; UI layers translate.


CREATE TABLE IF NOT EXISTS companion_growth_backlog (
    id                              TEXT PRIMARY KEY,
    user_id                         TEXT NOT NULL DEFAULT '',
    agent_id                        TEXT NOT NULL DEFAULT 'becca',

    -- One of: skill_refine / calibration_test / memory_consolidate /
    -- anticipation_train / subagent_consult / creation / discovery /
    -- proactive_offer / recall_connect. Free TEXT (no CHECK) so the
    -- catalog can grow without a migration.
    item_type                       TEXT NOT NULL,

    -- Pointer to the thing the task acts on (skill_id, observation
    -- pattern key, memory id, …). Free-form; interpretation lives in
    -- the action handler.
    target_ref                      TEXT NOT NULL DEFAULT '',

    -- Optional human-readable context — what the trigger noticed,
    -- captured at backlog-seed time so the session can render it in
    -- the plan phase.
    rationale                       TEXT NOT NULL DEFAULT '',

    priority                        REAL NOT NULL DEFAULT 0.5,
    source_signal                   TEXT NOT NULL DEFAULT '',

    expected_berry_yield            REAL NOT NULL DEFAULT 0,
    expected_mana_cost              REAL NOT NULL DEFAULT 0,
    expected_berry_cost             REAL NOT NULL DEFAULT 0,

    success_count                   INTEGER NOT NULL DEFAULT 0,
    fail_count                      INTEGER NOT NULL DEFAULT 0,
    last_attempted_at               INTEGER,
    last_consult_inconclusive_at    INTEGER,

    -- pending / in_progress / done / shelved
    state                           TEXT NOT NULL DEFAULT 'pending',

    created_at                      INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);

-- Selector reads "give me the top-priority pending items for this
-- user-agent" — the index covers that filter + sort.
CREATE INDEX IF NOT EXISTS idx_growth_backlog_user_ranked
    ON companion_growth_backlog(user_id, agent_id, state, priority DESC);


CREATE TABLE IF NOT EXISTS companion_growth_log (
    id                  TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL DEFAULT '',
    agent_id            TEXT NOT NULL DEFAULT 'becca',

    -- Nullable: ad-hoc sessions are allowed (user-explicit "work on X
    -- right now"). Loose FK so a deleted backlog row doesn't cascade
    -- away the historical record of work done.
    backlog_id          TEXT,

    started_at          INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),
    ended_at            INTEGER,

    -- Plan envelope: target, expected_return, success_criteria,
    -- budget_mana, budget_berries. JSON so the plan shape can evolve.
    plan_json           TEXT NOT NULL DEFAULT '{}',

    -- Per-step action log — tool calls, outcomes, debits, observations.
    act_log_json        TEXT NOT NULL DEFAULT '[]',

    -- Per-consult records (Phase 4): domain, problem, transcript,
    -- distillation, cost. Empty until consults wire in.
    consult_records_json TEXT NOT NULL DEFAULT '[]',

    -- Before/after metrics — the verifier's input + output.
    ledger_delta_json   TEXT NOT NULL DEFAULT '{}',

    -- in_progress / completed / rolled_back / aborted / suspended
    outcome             TEXT NOT NULL DEFAULT 'in_progress',

    -- 0 silent / 1 auto / 2 user-approval / 3 defensive-forbidden
    tier                INTEGER NOT NULL DEFAULT 0,
    -- n/a / pending / approved / rejected (Tier 2 path)
    approval_state      TEXT NOT NULL DEFAULT 'n/a',

    -- Closed-out totals; the truth lives in companion_economy_tx.
    -- These columns are denormalised for cheap log reads.
    mana_spent          REAL NOT NULL DEFAULT 0,
    berries_spent       REAL NOT NULL DEFAULT 0,
    berries_earned      REAL NOT NULL DEFAULT 0,

    -- Rollback pointer — opaque snapshot id captured at session start
    -- before any tier-1+ change. Empty when session made no durable
    -- modifications.
    snapshot_ref        TEXT NOT NULL DEFAULT ''
);

-- Two common reads:
--   1. Recent sessions for a user (inspector) — (user_id, agent_id, started_at DESC)
--   2. Sessions linked to a specific backlog item — (backlog_id)
CREATE INDEX IF NOT EXISTS idx_growth_log_user_time
    ON companion_growth_log(user_id, agent_id, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_growth_log_backlog
    ON companion_growth_log(backlog_id);


CREATE TABLE IF NOT EXISTS companion_economy (
    user_id             TEXT NOT NULL DEFAULT '',
    agent_id            TEXT NOT NULL DEFAULT 'becca',

    -- Bounded — regenerates over time. The selector reads + lazily
    -- ticks regen before debiting.
    mana                REAL NOT NULL DEFAULT 100,
    mana_cap            REAL NOT NULL DEFAULT 100,
    mana_regen_per_hour REAL NOT NULL DEFAULT 10,

    -- Unbounded (slow decay applied in tx-read paths, not at write).
    -- berries is the spend-and-earn current balance; berries_lifetime
    -- never decays — used for "how much trust has this agent ever
    -- accumulated" reads.
    berries             REAL NOT NULL DEFAULT 0,
    berries_lifetime    REAL NOT NULL DEFAULT 0,

    -- Unix seconds. Set on every regen tick + every read that ticked.
    last_mana_tick      INTEGER NOT NULL DEFAULT (strftime('%s', 'now')),

    PRIMARY KEY (user_id, agent_id)
);


CREATE TABLE IF NOT EXISTS companion_economy_tx (
    id              INTEGER PRIMARY KEY,
    user_id         TEXT NOT NULL DEFAULT '',
    agent_id        TEXT NOT NULL DEFAULT 'becca',

    -- Nullable: out-of-loop earnings (user vouch, decay sweep) carry
    -- no growth_log_id.
    growth_log_id   TEXT,

    -- One of: mana_debit / mana_regen / berry_earn / berry_spend /
    -- berry_decay / vouch / veto / sponsor / restraint_credit
    tx_type         TEXT NOT NULL,

    amount          REAL NOT NULL,
    reason          TEXT NOT NULL DEFAULT '',

    -- One of: explicit / implicit / affect / counterfactual /
    -- restraint / user_action / system
    signal_kind     TEXT NOT NULL DEFAULT 'system',

    evidence_ref    TEXT NOT NULL DEFAULT '',

    ts              INTEGER NOT NULL DEFAULT (strftime('%s', 'now'))
);

-- Per-user transaction history (audit log + ledger reconstruction).
CREATE INDEX IF NOT EXISTS idx_economy_tx_user_time
    ON companion_economy_tx(user_id, agent_id, ts DESC);

-- Sessions can read their own transactions cheaply.
CREATE INDEX IF NOT EXISTS idx_economy_tx_growth_log
    ON companion_economy_tx(growth_log_id);


INSERT OR IGNORE INTO schema_version (version, description)
VALUES (216, 'Companion growth-loop substrate: backlog, log, economy, economy_tx');
