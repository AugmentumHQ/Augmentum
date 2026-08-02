-- 193_companion_skill_graph.sql
-- Accumulation thesis Step 3 — the capability-side substrate.
--
-- Identity accumulation (her exemplars, behavior contract, kernel
-- digest) makes her recognizably herself across years. Capability
-- accumulation makes her measurably better at the things she does
-- with this user, across years. Same shape — accumulation, curation,
-- abstraction, drift discipline — applied to a different axis.
--
-- Three tables (sibling structure to memories + companion_journal):
--
--   companion_skills            — the nodes: named approaches she
--                                  has taken to recurring problem
--                                  shapes. Each carries her own
--                                  description, embedding, current
--                                  confidence, accumulated counts.
--
--   companion_skill_instances   — every time a skill was applied to
--                                  a specific situation. Bound to a
--                                  skill_id; carries context +
--                                  approach + turn refs for the
--                                  resolver and the consolidator.
--
--   companion_skill_outcomes    — what happened. Signal per instance
--                                  (+1.0 worked / -1.0 failed) plus
--                                  evidence + detection metadata.
--                                  Most outcomes are unknown until
--                                  inferred from user response or
--                                  later observation; that's the
--                                  honest default.
--
-- The thesis discipline (no autonomous mutation without consent) is
-- honored at the application layer, not the schema. Migrations only
-- create the substrate; the runtime decides what's safe to write.

CREATE TABLE IF NOT EXISTS companion_skills (
    id                  INTEGER PRIMARY KEY,
    companion_id        TEXT NOT NULL,
    user_id             TEXT,                       -- NULL = cross-user / shared
    name                TEXT NOT NULL,              -- short identifier, e.g. "isolate_before_guessing"
    description         TEXT NOT NULL DEFAULT '',   -- her own description, in her voice
    problem_shape       TEXT NOT NULL DEFAULT '',   -- structural description of what this addresses
    embedding           BLOB,                       -- problem_shape embedding for similarity search
    confidence          REAL NOT NULL DEFAULT 0.5,  -- her confidence in this approach
    instances_count     INTEGER NOT NULL DEFAULT 0,
    successes_count     INTEGER NOT NULL DEFAULT 0,
    failures_count      INTEGER NOT NULL DEFAULT 0,
    abstracted_from_ids TEXT NOT NULL DEFAULT '[]', -- JSON array of skill_ids this was abstracted from
    status              TEXT NOT NULL DEFAULT 'active',  -- active|suppressed|retired
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_companion_skills_companion_user
    ON companion_skills(companion_id, user_id, status, confidence DESC);

CREATE INDEX IF NOT EXISTS idx_companion_skills_active
    ON companion_skills(companion_id, status, updated_at DESC)
    WHERE status = 'active';

-- Skill instances — each application of a skill to a situation.
-- Links to companion_journal entries via turn_ref when available so
-- the consolidator can rehydrate evidence later.
CREATE TABLE IF NOT EXISTS companion_skill_instances (
    id              INTEGER PRIMARY KEY,
    companion_id    TEXT NOT NULL,
    user_id         TEXT,
    skill_id        INTEGER NOT NULL,
    context         TEXT NOT NULL DEFAULT '',   -- what the situation was
    approach        TEXT NOT NULL DEFAULT '',   -- what she actually did
    session_id      TEXT NOT NULL DEFAULT '',
    invocation_id   TEXT NOT NULL DEFAULT '',
    turn_ref        TEXT NOT NULL DEFAULT '',   -- JSON {kind, id} reference for resolver
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (skill_id) REFERENCES companion_skills(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_companion_skill_instances_skill
    ON companion_skill_instances(skill_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_companion_skill_instances_user
    ON companion_skill_instances(user_id, companion_id, created_at DESC)
    WHERE user_id IS NOT NULL;

-- Outcomes — what happened after the skill was applied. Many will
-- arrive minutes/hours after the instance (the user reads the response,
-- then accepts/rejects); some won't be resolved at all (we record
-- 'unknown' rather than guess). The outcome signal is what feeds the
-- skill's confidence over time.
CREATE TABLE IF NOT EXISTS companion_skill_outcomes (
    id            INTEGER PRIMARY KEY,
    instance_id   INTEGER NOT NULL,
    outcome       TEXT NOT NULL DEFAULT 'unknown',
                    -- accepted|rejected|corrected|shipped|problematic|unknown
    signal        REAL NOT NULL DEFAULT 0.0,
                    -- in [-1.0, +1.0]; rolled into skill.confidence
    evidence      TEXT NOT NULL DEFAULT '',
                    -- short description of what made us know the outcome
    detected_at   TEXT NOT NULL DEFAULT (datetime('now')),
    detected_by   TEXT NOT NULL DEFAULT 'inferred',
                    -- user_explicit|inferred|autonomous_check
    FOREIGN KEY (instance_id) REFERENCES companion_skill_instances(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_companion_skill_outcomes_instance
    ON companion_skill_outcomes(instance_id, detected_at DESC);

INSERT OR IGNORE INTO schema_version (version, description)
VALUES (193, 'companion_skill_graph: capability accumulation substrate (thesis step 3)');
