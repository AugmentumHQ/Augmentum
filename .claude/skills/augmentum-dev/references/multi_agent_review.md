# Multi-Agent Code Review Playbook

A repeatable process for auditing any Augmentum subsystem (or eventually the
whole codebase) by partitioning it into non-overlapping zones, running one
read-only review agent per zone in parallel, and consolidating the reports
into a single prioritized roll-up.

Designed for code-quality audits at the bar a company with Anthropic-level
standards would apply at PR review. Not a scanner — the agents read every
file in full, weigh intent, and surface things static tools miss
(state-machine gaps, trust-boundary holes, verb-symmetry breaks, perfect-
negotiation glare, listener leaks, CSP violations, etc.).

## When to use this

- **Pre-ship of a substantive subsystem** (e.g., Connect before declaring it
  user-facing).
- **Post-incident review** (after a data-loss / corruption / cross-tenant
  leak event, audit the surrounding code).
- **Periodic correctness checks** on subsystems that have grown >5k LOC
  since their last review.
- **Pre-handoff** when a new contributor will own the area.

Don't use it for:
- One-file or one-bug questions — just Read the file directly.
- Scanner-detectable issues — `audit.py` is cheaper.
- Architecture design — use the `Plan` agent.

It works best on subsystems that are:
- 1k–15k LOC total
- Mixed front + back (so partition delivers parallelism value)
- Have at least one cross-cutting concern (auth, routing, lifecycle, trust)

## How to partition

A good partition has **balanced load, no overlap, sensible boundaries**.

1. **One zone per concern, not per file.** Files that share a contract
   belong together (a route file + its store + its service module is one
   zone). Splitting them produces shallow reviews that miss the contract.
2. **Target 2k–4k LOC per zone.** Single-file zones lose context;
   >5k LOC zones blow agent context budgets and produce summary-quality
   reviews.
3. **Front + back are different zones.** Different categories apply
   (XSS / accessibility / CSP frontend; SQL / async / locking backend).
4. **Routes / HTTP get their own zone** when there's >500 LOC. They
   concentrate auth, validation, status codes, and rate limits that are
   easy to miss when bundled with business logic.
5. **6 zones is the sweet spot** for medium subsystems. 4 for small,
   8–10 for very large (narrative, coder).

### Worked example: Connect (2026-06-05)

~25k LOC across 6 zones, parallel, ~2 min wall clock per agent:

| Zone | Files | LOC |
|------|-------|-----|
| 1. Backend storage + protocol | `message_store`, `call_store`, `contact_store`, `protocol`, migrations 219/233/241/242/244/245 | ~3k |
| 2. Backend routing + lifecycle | `message_routing`, `call_routing`, `call_lifecycle`, `hub` | ~4k |
| 3. Fabric integration | `fabric_transport`, `fabric_inbound`, `contacts` | ~2k |
| 4. HTTP / WS routes | `connect_routes.py` | ~1.5k |
| 5. Frontend signaling + calls | 11 files (`client`, `dialer`, `incoming`, `incoming-modal`, `ringback`, `ringtone`, `broadcast`, `rate-toast`, `calls-panel`, `ui`, `icons`) | ~5k |
| 6. Frontend messaging + outbox + CSS | `thread-panel`, `messages`, `outbox`, `connect.css` | ~9k |

## The prompt template

Customize the bracketed bits per zone. Keep the rest verbatim — the
categories list is what drives the reviewer's coverage.

```
You are a senior reviewer auditing the Augmentum {SUBSYSTEM} subsystem
({ONE_SENTENCE_DESCRIPTION}) for code-quality issues that would block
a PR at a company with Anthropic-level standards.

{ONE_PARAGRAPH_CONTEXT_FOR_THIS_ZONE}

**Your scope:**
- {FILE_PATH_1}
- {FILE_PATH_2}
...

Read every file in full. Then flag issues across these categories:
- **Correctness**: race conditions, off-by-one, missing user_id filter,
  FK gotchas, idempotency bugs, missing indexes
- **Security / data isolation**: cross-user leaks, missing input
  validation, SQL injection, unbounded queries, XSS in template
  literals, CSP violations
- **Concurrency**: dict/list mutation without locks across asyncio
  tasks, held connections across awaits, dropped exceptions in
  fire-and-forget tasks, perfect-negotiation glare (frontend WebRTC)
- **State machine correctness**: missing transitions, dead branches,
  cross-handler races, verb-symmetry gaps between outbound + inbound
- **Error handling**: silent excepts, broad excepts, missing rollback,
  swallowed failures invisible to user
- **API surface**: inconsistent signatures, missing docstrings on
  public functions, unclear ownership, status-code hygiene
- **Schema / migration hygiene** (backend only): composite PK gotchas,
  CREATE TABLE IF NOT EXISTS post-deploy, missing triggers, dead
  columns, AUTOINCREMENT footguns, missing indexes
- **DOM / CSS hygiene** (frontend only): listener leaks, rAF leaks,
  !important overuse, focus traps, prefers-reduced-motion, a11y
- **Dead code**: TODOs, "Phase N" stubs, unused imports, dead
  variables, "coming soon" toasts
- **Test coverage gaps**: invariants you spot that aren't covered

Output: prioritized markdown list. For each finding include:
- severity (P0 ship-blocker / P1 should-fix / P2 nice-to-have)
- file:line
- one sentence describing the issue
- one-sentence suggested fix

Maximum {N} findings (focus on the most consequential).
Keep total response under {WORD_CAP} words.
Do not write any code — review only.
```

Settings I used for Connect:
- `N` = 25 (frontend messaging got 30 because it's larger)
- `WORD_CAP` = 800 (large zones: 1000)
- `subagent_type` = `general-purpose`
- `run_in_background` = `true` (so 6 fire in parallel; you get notified
  per-completion)

## Severity rubric

- **P0 ship-blocker** — security, data isolation, data loss, crash on
  normal use, "the design says X but the code does Y" divergence.
  Anything that would block a PR at a company with strong code review.
- **P1 should-fix** — real bug under specific conditions, footgun that
  primes future bugs, missing error visibility on a user-facing path,
  performance cliff at moderate scale.
- **P2 nice-to-have** — dead code, naming, minor style, "would be
  better as", future-proofing.

Reviewers tend to over-classify P0. After consolidation:
- **Demote P0 → P1** when it requires an attacker model the system
  doesn't claim to defend against, or it's a doc/test gap framed as
  a bug, or it's a hypothetical race that hasn't been triggered.
- **Promote P1 → P0** when multiple reviewers independently flag the
  same issue from different scope boundaries — that's a real signal.

## Consolidation

After all agents return:

1. **Group by theme**, not by reviewer. The same finding will surface
   in 2–3 reports under different framings (e.g., "block enforcement"
   shows up in routing review AND fabric review AND backend storage).
2. **Dedupe.** Pick the report with the best file:line and the
   clearest one-line fix; drop the others.
3. **Apply the demote/promote rules above.**
4. **Pick top 5 to fix this week** — smallest code delta × largest
   correctness/security delta. Everything else becomes backlog.
5. **Save the top findings to memory** so the next review can verify
   what landed vs. what didn't.

The consolidated output structure that worked for Connect:

```
## P0 — Ship-blockers
  ### Theme 1 (e.g., Trust boundary)
  ### Theme 2 (e.g., Block enforcement)
  ### Theme 3 (e.g., Concurrency)
  ...
## P1 — Should-fix before merge
## P2 — Backlog
## Tests demanded by ≥2 reviewers
## My read (1 paragraph + top-5 picks)
```

## Outputs to keep

- The N raw reports (already in `~/.claude/.../tasks/<id>.output`).
- A consolidated markdown summary (the user-facing roll-up).
- A memory entry: `project_audit_{subsystem}_{date}.md` with the top
  findings + which were landed. The next review reads this so it can
  verify what was actually closed.

## Anti-patterns to avoid

- **Re-reading the same file across zones.** If a file appears in two
  zones, the second reviewer pays the context cost twice. Better to
  enlarge one zone than to duplicate.
- **Asking the reviewer for code.** Reviewers writing fixes wastes
  tokens — they're for surveying, not editing. Land fixes from the
  main session.
- **Zones smaller than 500 LOC.** Single-file or single-concern zones
  produce thin reports. Either widen the zone or skip it.
- **Mixing subsystem boundaries within a zone.** If a zone spans
  Connect + Narrative, the reviewer can't apply subsystem-specific
  conventions consistently.
- **No word cap.** Without a cap, reports balloon to 3–5k words and
  become unreadable; consolidation falls apart.
- **Serial launch.** Always launch in parallel (single message,
  multiple `Agent` tool uses, all with `run_in_background: true`).
  Serial costs 6× wall clock.

## Planning a new subsystem audit — quick recipe

1. Pick the subsystem (table in CLAUDE.md lists all 30).
2. List files: `Glob "augmentum/<subsystem>/**/*.py"` and
   `Glob "ui/scripts/<surface>/**/*.{js,css}"`.
3. Count LOC, group into zones per the partition rules above.
4. For each zone, customize the prompt template:
   - SUBSYSTEM name
   - ONE_SENTENCE_DESCRIPTION (what the subsystem does)
   - ONE_PARAGRAPH_CONTEXT (key abstractions, known fragile bits)
   - FILE_PATH_LIST (absolute paths)
   - N (25–30) and WORD_CAP (800–1000)
5. Launch all agents in a single message, all with
   `run_in_background: true`, `subagent_type: general-purpose`.
6. Wait for the notifications. Consolidate when all return.
7. Save the roll-up + memory entry.

## Connect audit outcome (2026-06-05) — reference

Top-5 ship-blockers identified:

1. Block enforcement on call signaling verbs (`call_routing.py:374-383`)
2. Fabric inbound `target_user_id` + DID host normalization
   (`fabric_inbound.py:158-189`)
3. Remove test-secret fallback in `_attachment_signing_secret`
   (`fabric_transport.py:351-352`)
4. `arm_invite_timer` connection lifetime
   (`call_lifecycle.py:67-101`)
5. Outbox flush break-on-retry → continue
   (`outbox.js:232-262`)

Cross-cutting themes:
- Fabric inbound trusts sender claims (host, target_user_id) more
  than it should.
- Silent-block was applied verb-by-verb as features landed; call
  signaling verbs got missed.
- Several "ok-on-failure" paths cache or accept the wrong way
  (404 cached as immutable, clear-thread returns ok on wrong-user).
- Schema drift footguns still latent in three more tables despite
  244/245 fix.

Full consolidated report is in the session transcript and the
`project_connect_review_2026_06_05` memory entry.
