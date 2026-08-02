---
name: Failure Triage
description: >
  Triage bugs from logs, tracebacks, failing commands, and symptoms. Use when the problem
  is runtime behavior, an unclear error, or a weak reproduction that needs to be narrowed
  into one concrete root cause.
kind: verifier
activation_policy: controller
activation_windows:
  - pre_plan
  - verify_failed
modes:
  - coder
triggers:
  - regression bug
  - reproducer
  - root cause
  - debug this
  - investigate error
  - traceback
  - failing command
  - bug triage
  - runtime failure
preferred_tools:
  - file_read
  - code_search
  - shell_read
  - test_run
  - dir_tree
  - task_dispatch
  - observe
tags:
  - debugging
  - logs
  - runtime
  - triage
---

# Failure Triage

Use this Power when the user has a symptom, not yet a diagnosis.

## Goal

Turn noisy evidence into one concrete failure story: trigger, root cause, fix path, and proof.

## Workflow

1. Capture the exact symptom:
   - command
   - stack trace
   - log line
   - user-visible failure
2. Narrow the reproduction to the smallest repeatable path.
3. Identify the first meaningful failure signal, not the last downstream explosion.
4. Trace backwards to the state or assumption that made it possible.
5. Propose the smallest fix that changes the failure story.
6. Re-run the failing path or the nearest deterministic check.

## Design rules

- Prefer one verified root cause over three speculative theories.
- Use the earliest relevant stack frame or log boundary.
- Distinguish:
  - trigger
  - symptom
  - underlying cause
- If the reproduction is flaky, say that and state the confidence level.

## When to spawn a subagent

- **Stuck on the same hypothesis after 2-3 attempts** — spawn
  `task_dispatch(role="review", prompt="here's the failing case
  and what I've tried; what am I missing?")` for a second opinion
  before looping again.
- **Wide investigation across many files** — spawn
  `task_dispatch(role="explore", prompt="find every site that
  modifies state X")` instead of grepping in your own context.

## After resolution

Persist non-obvious findings via `observe`:
- Root cause + category (gotcha / constraint / etc.)
- The misleading symptom that distracted you
- The check that would have caught it earlier

Future sessions surface these in `<workspace_facts>` and don't repeat
the investigation.

## Guardrails

- Do not paper over a failure with retries unless the real issue is actually transient behavior.
- Do not stop at the first exception string if it is clearly a downstream effect.
- Do not claim root cause without a code path or state explanation.
- Keep the final explanation short and causal.

## Good outputs

- "The 500 was downstream; the real break was a missing enum import during request routing."
- "The failing shell command exposed a stale path assumption, not a permissions issue."
- "I can reproduce it only intermittently; current best hypothesis is cache invalidation on the second turn."
