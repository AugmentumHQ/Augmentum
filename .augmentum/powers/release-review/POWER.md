---
name: Release Review
description: >
  Final quality gate for code changes. Use when reviewing work before merge, release,
  handoff, or deployment and the goal is evidence, not reassurance.
kind: verifier
activation_policy: controller
activation_windows:
  - pre_finish
modes:
  - coder
triggers:
  - final review
  - ship review
  - release gate
  - quality gate
  - pre release check
preferred_tools:
  - file_read
  - code_search
  - shell_exec
  - test_run
tags:
  - review
  - qa
  - release
  - validation
---

# Release Review

Use this Power as the last pass before calling work done.

## Goal

Produce a grounded go / fix first / block recommendation backed by actual evidence from the changed surface.

## Review order

1. Confirm the intended behavior and changed files.
2. Run the smallest meaningful checks first:
   - focused tests
   - lint or type checks if relevant
   - build step if the change touches build-sensitive areas
3. Inspect risky deltas:
   - auth and permissions
   - persistence and migrations
   - error handling
   - user-visible regressions
   - unsafe defaults or secret handling
4. Compare the result against the requested acceptance criteria.
5. Summarize findings in severity order.

## Reporting style

- Findings first.
- File and line references when available.
- Be explicit about what was executed and what was not.
- If no findings remain, say that directly and note residual risk or untested areas.

## Guardrails

- Do not say "looks good" without concrete validation.
- Do not inflate scope into a full architecture review unless asked.
- Do not hide missing tests, skipped builds, or incomplete checks.
- Prefer one focused failing command over broad speculative warnings.

## Verdicts

- `SHIP`: no blocking issues found in the requested scope.
- `FIX FIRST`: concrete issues exist, but they are local and actionable.
- `BLOCK`: correctness, security, or release integrity is materially at risk.
