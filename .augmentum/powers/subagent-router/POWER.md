---
name: Subagent Router
description: >
  Steer the coder toward task_dispatch at the right moments — wide
  exploration, second-opinion review, grounded research, security audit.
  Keeps the lead model's context lean by handing focused subtasks to
  subagents with their own model + tool subset + budget.
kind: guidance
activation_policy: controller
activation_windows:
  - pre_plan
  - post_write
  - verify_failed
  - pre_finish
modes:
  - coder
triggers:
  - find every
  - find all
  - where does
  - second opinion
  - review my
  - audit
  - security
  - design decision
  - which approach
  - research
preferred_tools:
  - task_dispatch
verification_recipe:
  - Before fanning out, name the specific role the subagent should play.
  - After a spawn, read the subagent's structured output before continuing.
  - Don't spawn the same role twice for the same scope in one turn.
memory_writes: []
success_criteria:
  - Wide-scope subtasks ran in a subagent, not in the lead's own context.
  - Each spawn returned a usable structured result, not a vague status.
tags:
  - subagent
  - delegation
  - orchestration
  - context-budget
---

# Subagent Router

Use the `task_dispatch` tool to delegate focused subtasks to subagents
with their own context budget. The lead model is expensive context;
spending it on wide grep / open-ended research / security sweeps bloats
the conversation and crowds out the work that only the lead can do.

## When to spawn (by role)

- **`task_dispatch(role="explore", prompt=…)`** — "find every site
  that calls X", "where is Y handled", "list all auth checkpoints".
  Read-only fan-out across the repo. Spawn instead of running a dozen
  grep/file_read calls in your own context.
- **`task_dispatch(role="plan", prompt=…)`** — design questions with
  non-obvious tradeoffs ("how should we structure the new pipeline?").
  Returns a structured plan (Goal / Approach / Steps / Risks).
- **`task_dispatch(role="review", prompt=…)`** — second-opinion code
  review on your diff or a specific file. Use at `verify_failed` (don't
  loop on the same fix) and at `pre_finish` (last chance gate before
  signing off).
- **`task_dispatch(role="research", prompt=…)`** — grounded answers
  via doc_search + doc_fetch. Use when "what's the current state of X"
  needs real sources, not training data.
- **`task_dispatch(role="security_review", prompt=…)`** — disproof-
  oriented vulnerability audit on one file/diff. Spawn automatically
  after editing security-sensitive paths (auth, crypto, anything with
  raw SQL or `eval`/`exec` patterns).
- **`task_dispatch(role="threat_model", prompt=…)`** — once per
  unfamiliar codebase, produce a written threat model the user can
  paste into a bug_finder run (closes the #1 source of FPs).

## Activation windows

- **pre_plan** — if the user's ask is wide ("audit X", "find all Y",
  "research best practice for Z"), pick the right role and spawn FIRST.
  Then plan the rest of the turn around the subagent's result.
- **post_write** — if you just edited a security-sensitive path,
  spawn `security_review` on the diff before continuing.
- **verify_failed** — don't loop. Spawn `review` for a second opinion.
- **pre_finish** — for substantive changes, spawn `review` on the
  aggregate diff as the last gate.

## Multi-provider

`model` accepts `name@provider` or `name@fabric:peer_id`. Use a faster
or cheaper backend for read-only roles (explore, research) and the
expensive model only when you need top-tier judgment (review,
security_review). Example:
`task_dispatch(role="explore", model="claude-haiku-4-5@anthropic", …)`.

## Guardrails

- Don't spawn for single-file edits or small reads — the overhead
  isn't worth it.
- Don't recurse: subagents can't spawn their own subagents unless
  the role's `permissions.can_spawn_subagents` is true (default false).
- Treat the subagent's structured output as the source of truth; if
  it disagreed with your plan, update the plan.

## Good outputs

- "Spawned `explore` to find every `resolve_backend_for_model` site —
  18 matches across 5 files. Going to update them in one batch."
- "Tests failed twice. Spawned `review` on the failing handler — it
  flagged the assertion-order issue I'd been overlooking."
- "Edited auth/middleware. Spawning `security_review` on the diff
  before signing off."
