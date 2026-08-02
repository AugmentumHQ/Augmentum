---
name: Test Baseline Keeper
description: >
  Before non-trivial changes, capture a baseline (test counts, perf
  measurement, build time). After the change, re-run and compare.
  Make regressions explicit instead of silent. Pairs with
  test-author and performance-profiler.
kind: verifier
activation_policy: controller
activation_windows:
  - pre_plan
  - pre_finish
modes:
  - coder
triggers:
  - regression
  - performance
  - before and after
  - benchmark
  - timing
  - did this break
preferred_tools:
  - test_run
  - shell_exec
  - shell_read
  - observe
verification_recipe:
  - At pre_plan, capture baseline test pass count + key timing if
    perf-sensitive. Store snapshot in the chat or via observe.
  - At pre_finish, re-run the same target and compare.
  - State delta explicitly: "tests N→N (Δ0)", "p99 latency 120ms→135ms (+13%)".
memory_writes:
  - category: test
    key: baseline_pass_count
  - category: build
    key: baseline_build_time
success_criteria:
  - Baseline was captured BEFORE the first edit.
  - Final report compares baseline to post-change with concrete
    numbers, not "looks fine".
  - Any regression is named, not buried.
tags:
  - testing
  - regression
  - baseline
  - verification
  - performance
---

# Test Baseline Keeper

Models routinely declare "tests pass" without ever running them
before the change to establish a baseline. The result: silently
broken tests that were already broken get blamed on the diff, or
newly-broken tests get hidden in the noise.

## Workflow

1. **Before any edit** (pre_plan window):
   - Auto-detect the test runner from the profile (`pytest -x`,
     `npm test`, `go test ./...`, etc.).
   - Run it ONCE against the unchanged tree. Capture:
     - Pass count
     - Fail count + names
     - Skip count
     - Wall time
   - If you're changing perf-sensitive code, also capture the
     relevant timing (build duration, request latency, etc.).
   - Note baselines in your turn state; for important workspaces,
     persist via `observe(category="test", fact="pytest baseline:
     247 pass, 0 fail (2026-05-31)")`.

2. **During the change**: don't re-run continuously — the noise
   isn't worth it. Edit, then re-verify at the end.

3. **Before declaring done** (pre_finish window):
   - Re-run the SAME test target.
   - Compare to baseline:
     - **Tests N→N (Δ0)**: clean
     - **Tests N→M (-K)**: regression — name which tests broke
     - **Tests N→N+J**: new tests added — confirm they assert the
       intended behavior, not just smoke
   - For perf: re-measure with the same harness; report the delta
     as a percentage and an absolute number.

## What to actually run

- **pytest projects**: `pytest -x` for full suite, OR `pytest path/to/specific_test.py` if a focused change
- **node projects**: `npm test` or `npm run test:unit`
- **go**: `go test ./...`
- **multi-language**: prefer the narrowest test that exercises the
  changed code, expand only if the failure surface is unclear

## Pairing with other powers

- **test-author** writes new tests; this power proves they actually
  fail before the fix and pass after.
- **performance-profiler** investigates regressions; this power
  catches them by always measuring before/after.
- **observation-keeper** persists baselines so they survive sessions
  for long-running workspaces.

## Guardrails

- Never claim "tests pass" without showing the test_run output.
- Never silently skip a baseline because "it's a small change" —
  small changes break tests routinely.
- If you can't capture a baseline (test runner not configured, tests
  fail to even start), say so explicitly and don't proceed as if
  baseline was clean.

## Good outputs

- "Baseline: pytest 247 pass / 0 fail. Post-change: 248 pass / 0
  fail (+1 from the regression test test-author added). Clean."
- "Baseline: 247 pass. Post: 245 pass / 2 fail
  (test_isolation::test_user_b_cannot_see_user_a + test_isolation::
  test_cascade_delete). My patch broke isolation — reverting the
  store change and trying a different approach."
- "Couldn't run baseline — pytest fails to collect due to import
  error in test_unrelated.py. Recorded the broken-collection
  constraint and proceeded; will manually verify the changed handler
  via a curl probe instead."
