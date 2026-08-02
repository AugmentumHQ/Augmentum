---
name: Browser Verification
description: >
  Frontend and browser-flow verification for local apps. Use when reproducing UI bugs,
  checking forms, validating fixes, or confirming regressions in web interfaces.
kind: verifier
activation_policy: controller
activation_windows:
  - post_write
  - verify_failed
  - pre_finish
modes:
  - coder
triggers:
  - ui bug
  - browser test
  - visual regression
  - frontend verification
  - reproduce in browser
preferred_tools:
  - browser_open
  - browser_wait
  - browser_extract
  - browser_fill_form
  - browser_click
  - browser_verify
  - browser_evaluate
  - browser_screenshot
  - service_start
  - service_probe
  - file_read
  - test_run
tags:
  - frontend
  - browser
  - testing
  - verification
---

# Browser Verification

Use this Power for browser-facing work: reproducing bugs, validating fixes, checking user flows, and confirming that a frontend change actually works.

## Operating posture

- Prefer deterministic verification over broad exploration.
- Reproduce the issue first, then state the exact expected behavior.
- Keep the test scope tight to the affected page or flow.
- Use the NATIVE browser toolset — it is always present:
  - `browser_open` loads the page and returns its snapshot (status, title,
    visible elements) — read it, don't re-snapshot.
  - `browser_wait` for anything async (selector / text / network idle).
    Never a setTimeout sleep inside `browser_evaluate`.
  - `browser_fill_form` for forms — all fields + submit in one call.
  - `browser_click` / `browser_type` for single controls; pass `wait_for`
    when the target mounts late.
  - `browser_extract` for structured evidence (text/links/table/list/meta)
    and `browser_verify` for assertions; `browser_evaluate` only for
    computed state the others can't reach.
- Workspaces without the `browser` tooling profile fall back to plain-HTTP
  automatically (static HTML only — JS-rendered content is invisible). If a
  check needs a real DOM there, say so instead of faking it with curl.

## Workflow

1. Establish the target:
   - page, route, component, or interaction
   - expected behavior
   - current observed behavior
2. Find the minimal reproduction path.
3. Inspect the relevant UI code and assets before changing anything.
4. Make the smallest fix that addresses the verified failure.
5. Re-run only the checks that prove the behavior changed.
6. Report:
   - reproduction
   - fix
   - verification evidence
   - remaining uncertainty

## Guardrails

- Do not claim a UI fix is complete without some execution evidence or an explicit statement that execution was not possible.
- Do not browse unrelated websites or use public internet targets unless the task explicitly requires it.
- Do not widen the scope into redesign unless the user asked for that.
- Prefer stable selectors, repeatable steps, and concrete assertions over subjective visual claims.

## Good outputs

- "Reproduced on /settings: save button stayed disabled because field validation never cleared."
- "Patched the stale state path and verified with the local test plus one manual browser flow."
- "Could not execute browser automation here; static fix is in place, but runtime verification is still pending."
