---
name: Workspace Onboarding
description: >
  First-pass orientation for a new or unfamiliar workspace: identify stack,
  commands, ports, conventions, and safe paths to avoid.
kind: guidance
activation_policy: controller
activation_windows:
  - pre_plan
modes:
  - coder
triggers:
  - new workspace
  - what is this project
  - onboard
  - explore repo
  - setup project
preferred_tools:
  - dir_tree
  - file_read
  - shell_read
  - profile_update
verification_recipe:
  - Read manifests and README-like files first.
  - Discover dev/test/build commands without running broad mutations.
  - Write only concise, stable facts to the workspace profile.
memory_writes:
  - category: project
    key: framework
  - category: command
    key: dev_command
  - category: command
    key: test_command
success_criteria:
  - The workspace profile has concise stack and command facts.
  - The final answer distinguishes observed facts from assumptions.
tags:
  - onboarding
  - workspace
  - profile
---

# Workspace Onboarding

Use this Power for orientation turns. The goal is to make future
turns cheaper by learning only facts that are stable and useful —
not to exhaustively map every file. The kernel facts block at turn
start surfaces what you record here; junk in = junk out.

## Workflow

1. **Survey the structure** — `dir_tree` at depth 2-3. Note: the
   top-level dirs + the obvious "this is a {python|node|rust|go}
   project" signal.
2. **Read the load-bearing files** in this order:
   - `README*` (project's own self-description)
   - `pyproject.toml` / `package.json` / `go.mod` / `Cargo.toml`
   - `Makefile` / `justfile` / npm scripts / `docker-compose.yaml`
   - Any `docs/` index
   - `.augmentum/` directory if it exists (existing observations,
     objective.md, etc. — don't duplicate them)
3. **Identify the commands** (test, build, dev, deploy). DON'T run
   destructive ones to discover what they do — read first.
4. **Spawn `task_dispatch(role="explore", …)`** if the repo is
   large and the structure isn't self-explanatory. Subagent fans
   out without blowing your turn budget.
5. **Write the facts** via `profile_update` (one-shot identity:
   language, package manager, test runner) and `observe` (smaller
   gotchas / commands / constraints).

## What's worth recording

- **profile_update** (project-level, stable for the life of the repo):
  - language / runtime / framework
  - package manager
  - test command, build command, dev command, lint command
  - default branch
- **observe** (cross-session ledger entries):
  - non-obvious constraints (node version pin, deprecated APIs)
  - gotchas (a tool that needs a specific env var to work)
  - conventions (naming, file layout, where new modules go)

## What NOT to record

- File listings — those drift constantly and the workspace snapshot
  surfaces them anyway.
- Current state ("the foo branch is checked out") — ephemeral.
- Opinion ("the codebase is messy") — not actionable.
- Anything you can re-derive in 30s from a single file_read.

## Spawn the threat_model subagent (security-sensitive repos)

If the codebase handles user data, auth, payments, or anything with
a sensitive blast radius, end onboarding with:
`task_dispatch(role="threat_model", prompt="produce the threat model for this repo")`.
The output pastes directly into a bug_finder run's `threat_model`
field later, and a written threat model is the #1 thing that makes
later security work productive.

## Guardrails

- Don't run package installs or builds during onboarding — read
  first, set up second.
- Don't fork sub-tasks; this is a single orientation pass.
- Don't recurse if `.augmentum/profile.toml` already has good facts —
  acknowledge what's there and stop.
- If the README is missing or stale, that's a finding worth
  recording, not a reason to fabricate one.

## Good outputs

- "FastAPI + Python 3.11, uv-managed. Test: `pytest -x`. Dev:
  `uvicorn main:app --reload`. Recorded profile + 3 gotchas.
  Spawned threat_model — output saved for the user's next bug_finder
  run."
- "Already onboarded — .augmentum/profile.toml has 11 entries and
  observations.jsonl has 23 entries. Read them; no new orientation
  needed."
- "Small repo (12 files); didn't bother with subagents. Profile
  + 1 observation suffices."
