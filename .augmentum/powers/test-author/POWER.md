---
name: Test Author
description: >
  Design and implement focused tests that prove behavior instead of padding coverage.
  Use when adding regression tests, validating a fix, or choosing the smallest
  meaningful test surface for a change.
kind: verifier
activation_policy: controller
activation_windows:
  - post_write
  - verify_failed
modes:
  - coder
triggers:
  - add tests
  - write tests
  - regression test
  - test this fix
  - improve coverage
preferred_tools:
  - file_read
  - code_search
  - code_edit
  - code_edit_batch
  - test_run
  - task_dispatch
tags:
  - testing
  - regression
  - verification
  - quality
---

# Test Author

Use this Power when the work needs proof, not just code changes.

## Goal

Write the smallest test or test set that would fail before the fix and pass after it.

## Workflow

1. Identify the exact behavior under test.
2. Find the nearest existing test style in the repo and match it.
3. Prefer the narrowest scope that proves the behavior:
   - unit before integration
   - integration before end-to-end
   - one regression before broad coverage expansion
4. Make the failure mode explicit.
5. Run only the relevant test target first.
6. Expand only if the first test cannot actually cover the risk.

## Design rules

- Assert behavior, not implementation trivia.
- Use stable fixtures and deterministic inputs.
- Keep setup shorter than the assertion logic whenever possible.
- Reuse repo helpers instead of inventing new ad hoc harness code.
- If a bug came from a boundary condition, encode that exact boundary in the test.

## Guardrails

- Do not add tests that only restate current implementation details.
- Do not widen to slow end-to-end tests if a smaller layer proves the fix.
- Do not claim coverage improved meaningfully if the new test does not fail on the old bug.
- If the code is hard to test, say why and identify the seam that is missing.

## Subagent assist

- Before writing tests in an unfamiliar pattern, spawn
  `task_dispatch(role="research", prompt="how does this repo
  structure pytest fixtures? Find the canonical pattern.")` —
  costs less than getting the convention wrong.
- For test design tradeoffs ("unit-test the parser vs integration-
  test the route"), spawn `task_dispatch(role="plan", prompt="…")`
  rather than coin-flipping.

## Pair with test-baseline-keeper

Use this power alongside `test-baseline-keeper`: capture the
pre-change pass count before writing the test, then prove the test
fails on the old code and passes on the new code. A regression test
that doesn't fail on the bug isn't a regression test — it's noise.

## Good outputs

- "Added one regression test that fails on the stale cache path and passes with the fix."
- "Matched the existing fixture pattern instead of building a parallel harness."
- "Skipped end-to-end coverage because the handler-level test already proves the failure mode."
