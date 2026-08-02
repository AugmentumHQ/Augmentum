---
name: Observation Keeper
description: >
  Persist non-obvious discoveries to the workspace's durable fact
  ledger so future sessions don't have to re-discover them. The
  ledger lives at /workspace/.augmentum/observations.jsonl and gets
  rendered into the model's <workspace_facts> block at turn start.
kind: guidance
activation_policy: controller
activation_windows:
  - post_write
  - verify_failed
  - pre_finish
modes:
  - coder
triggers:
  - took a while to figure out
  - turned out that
  - not obvious
  - tricky
  - gotcha
  - had to discover
preferred_tools:
  - observe
  - profile_update
verification_recipe:
  - Before recording, ask "would a fresh session benefit from this?"
  - Categorize correctly so it surfaces in the right context block.
  - Use 'tentative' confidence for inferred facts; 'confirmed' once a
    tool result backs the claim.
memory_writes:
  - category: constraint
    key: discovered_constraints
  - category: gotcha
    key: discovered_gotchas
success_criteria:
  - Non-obvious findings from the turn are written to the ledger.
  - Future sessions get the discovery via <workspace_facts> instead of
    re-running the same investigation.
tags:
  - memory
  - observations
  - cross-session
  - knowledge
---

# Observation Keeper

The `observe` tool writes to a durable cross-session fact ledger at
`/workspace/.augmentum/observations.jsonl`. Recent constraints and
gotchas surface in your `<workspace_facts>` block at turn start, so
recording the right things makes future you (and future sessions)
faster and less error-prone.

## What to record

- **build** — "build runs via `npm run build`, takes ~90s; output in `dist/`"
- **test** — "test runner is pytest; tests under `tests/` not `test/`"
- **deploy** — "deploys via GitHub Actions on push to `main`"
- **api** — "the `/api/foo` endpoint requires X-API-Key header, not Bearer"
- **data** — "the `users` table soft-deletes via `deleted_at`, not a row delete"
- **env** — "auth tokens read from `/workspace/.env.local`, NOT `.env`"
- **constraint** — "Node 18 is locked; do NOT require Node 20+ features"
- **gotcha** — "the linter ignores `# noqa` only with a specific code; bare ignore doesn't work"
- **style** — "this codebase uses ruff format, not black; tabs are 2 spaces"
- **other** — anything not in the above categories

## When NOT to record

- Already-documented facts (in README, profile, or existing observations)
- Transient state (current branch, current PR number, what you just edited)
- Personal opinions or speculation
- Facts a fresh session could derive from reading 1-2 files

## Confidence semantics

- `confidence='tentative'` — you inferred it but haven't verified
- `confidence='confirmed'` — a tool result (test pass, build success,
  command output) backs the claim

Re-record the same fact later with bumped confidence — the ledger
dedupes by (category, fact-lowercase) and updates timestamp +
confidence in place.

## Workflow

1. End of turn: scan what you did this turn. Anything surprising or
   non-obvious?
2. For each candidate, ask: would a future agent benefit, or is this
   ephemeral?
3. Call `observe(category="...", fact="...", confidence="...")`.
4. If you discovered a stable convention (always use X library for Y),
   also consider `profile_update` to bake it into the workspace profile.

## Guardrails

- Don't spam — 1-3 observations per turn is normal; 10+ usually means
  you're recording ephemeral state.
- Don't record fixes — the patch is already in the code; only record
  the WHY or the CONSTRAINT that drove the fix.
- Don't paste large content into a fact; keep each under ~200 chars.

## Good outputs

- "Recorded gotcha: linter only honors `# noqa: E501` with code; bare
  `# noqa` is ignored. Spent 20min on this; future me won't."
- "Recorded constraint: Node 18 locked — the build matrix breaks on 20+
  due to a deprecated crypto API. Don't propose upgrade in this repo."
- "Skipped recording 'edited handler.py' — that's ephemeral; the diff
  is the source of truth."
