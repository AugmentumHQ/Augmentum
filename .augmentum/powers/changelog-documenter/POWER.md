---
name: Changelog Documenter
description: >
  Pre-finish gate: produce a structured summary of what changed, why,
  and any handoff notes. Keeps commit messages, PR descriptions, and
  turn summaries consistent across sessions. Reaches into observe
  for any non-obvious discoveries that should be captured.
kind: guidance
activation_policy: controller
activation_windows:
  - pre_finish
modes:
  - coder
triggers:
  - summarize the changes
  - commit message
  - changelog
  - what did I change
  - PR description
preferred_tools:
  - git
  - file_read
  - observe
verification_recipe:
  - Run `git status` + `git diff --stat` to ground the summary in
    real changes, not memory.
  - Group changes by intent, not by file.
  - Flag anything the user needs to do post-merge (rerun migrations,
    update env vars, restart services).
memory_writes:
  - category: other
    key: changelog_entries
success_criteria:
  - Summary names every load-bearing file touched.
  - Why is one sentence, not a paragraph.
  - Migration / env / restart hints are surfaced (or explicitly "none").
tags:
  - documentation
  - changelog
  - handoff
  - pre-finish
---

# Changelog Documenter

When a substantive turn closes, leave the user a tight, accurate
summary of what changed. Models routinely write either too little
("done!") or too much (a wall of internal narration). The right
shape is closer to a commit message body: 3-7 grouped bullets, a
line for migration/restart concerns, and a line for handoff notes.

## Workflow

1. Run `git status` + `git diff --stat` to list what actually
   changed. Don't summarize from memory — diffs are authoritative.
2. Group changes by intent (not by file):
   - "Backend: added X endpoint" (lists 2-3 files)
   - "Frontend: wired toggle for Y" (lists 1-2 files)
   - "Migration NNN: added Z column" (single file, but call it out)
3. For each group, one sentence: WHAT changed + WHY.
4. Identify post-merge work:
   - **Migration**: does this require running migrations?
   - **Config**: did a setting key get added/renamed/removed?
   - **Restart**: do services need to be restarted to pick this up?
   - **External**: was an API key, env var, or external service
     touched?
5. If you discovered anything non-obvious during the turn (a gotcha,
   a constraint, a build-system quirk), call `observe` to persist
   it BEFORE the summary so it shows up in future `<workspace_facts>`.

## Output shape (paste-ready)

```
## Summary

- Backend: added /api/notebook/entries with user_id scoping (notebook_routes.py, notebook_store.py)
- Frontend: wired the new entry-creation modal (notebook.js, notebook.css)
- Migration 213: notebook_entries table with FK to users

## Post-merge

- Restart augmentum container so migration 213 applies.
- New setting `notebook_enabled` defaults False; flip via Settings →
  Automation → Notebook.

## Tests

- 5 new tests in test_notebook_isolation.py; all pass.
- Audit run: no new wiring errors.

## Handoff

- Spike: the entry sorting is alphabetical by title. If the user
  later wants chronological, see notebook_store.list_entries.
```

## Guardrails

- Don't include internal reasoning ("I considered X and decided Y") —
  that belongs in the chat transcript, not the summary.
- Don't restate the obvious ("I changed the file because the user
  asked"). Stick to WHAT and WHY-IT-MATTERS.
- Don't claim untested code works — if you didn't run tests, say so.
- Don't omit migration hints — silent migration requirements are the
  #1 cause of "I pulled and now nothing works" reports.

## Good outputs

- A 4-bullet summary with one-line migration hint and one-line test
  status. Reads like a clean PR description.
- Caught a constraint mid-turn (build flag X is required) and
  persisted it via observe so future sessions auto-pick-it-up.
