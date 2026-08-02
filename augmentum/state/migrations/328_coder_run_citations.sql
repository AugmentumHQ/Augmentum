-- 328_coder_run_citations.sql
-- Structured claim→proof provenance for a coder turn (the "Citation ledger"
-- primitive, spec 2026-07-27-companion-brief-stage-manager-design.md §7.4 +
-- the P3/P4 build plan). Decoupled from the Promise tree on purpose:
-- Promise.evidence is an unstructured str, so overloading it would lose
-- provenance. The worker emits one row per evidence-bearing tool result as
-- it works (write tools + oracle tools); the brief's citation dropdowns and
-- the (extended) run_verifier gate both READ from here.
--
-- Keyed by (turn_run_id, tool_call_seq): turn_run_id is the coder-turn
-- ledger id (ctr_...), which equals the brief's review_turn_id and the id
-- mountReviewPanel renders the diff for — so a citation deep-links straight
-- into the changed lines the user is deciding on. run_id (the broker run) is
-- a nullable cross-reference for the background/delegated path.
--
-- line_start/line_end are nullable NOW (MVP emits file + evidence_kind + ref;
-- file_write has no range, code_edit/apply_patch do). The column exists from
-- day one so per-line fidelity lands later with no schema change.

CREATE TABLE IF NOT EXISTS coder_run_citations (
    id             INTEGER PRIMARY KEY,                 -- rowid alias; no AUTOINCREMENT (SQLite footgun)
    turn_run_id    TEXT NOT NULL,                       -- ctr_... (== review_turn_id)
    user_id        TEXT NOT NULL REFERENCES users(id),
    workspace_id   TEXT NOT NULL DEFAULT '',
    run_id         TEXT NOT NULL DEFAULT '',            -- broker run cross-ref (nullable via '')
    tool_call_seq  INTEGER NOT NULL DEFAULT 0,
    file           TEXT NOT NULL DEFAULT '',
    line_start     INTEGER,                             -- null until per-line fidelity lands
    line_end       INTEGER,
    evidence_kind  TEXT NOT NULL DEFAULT 'write',       -- write|test|probe|browser|shell_check
    evidence_ref   TEXT NOT NULL DEFAULT '',            -- checkpoint id / command / probe target
    outcome        TEXT NOT NULL DEFAULT '',            -- green|red|unknown (oracle kinds); '' for writes
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_coder_run_citations_turn
    ON coder_run_citations (turn_run_id);

CREATE INDEX IF NOT EXISTS idx_coder_run_citations_user
    ON coder_run_citations (user_id);
