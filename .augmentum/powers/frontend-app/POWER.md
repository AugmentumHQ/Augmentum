---
name: Frontend App Builder
description: >
  Build a complete, known-good frontend app (HTML/CSS/JS) end to end: write it,
  run it on a dev server, drive every control in a real browser, prove it works,
  then finish. The definition of "done" is a running app that passes its own
  behavioral checks — not code that merely looks plausible.
kind: guidance
activation_policy: explicit_only
activation_windows:
  - pre_plan
  - implementation
  - post_write
  - verify_failed
  - pre_finish
modes:
  - build
triggers:
  - build an app
  - build me a
  - make a web app
  - create a calculator
preferred_tools:
  - builder_reference
  - builder_design_system
  - builder_api_refs
  - file_write
  - code_edit
  - dir_tree
  - service_start
  - service_probe
  - browser_open
  - browser_snapshot
  - browser_wait
  - browser_extract
  - browser_fill_form
  - browser_click
  - browser_type
  - browser_evaluate
  - browser_screenshot
  - finish_task
tags:
  - frontend
  - app-builder
  - build
---

# Frontend App Builder

You are building a complete frontend application from a description. Your job is
not to emit plausible-looking code — it is to deliver a **running app that you
have personally verified works** in a real browser. You build it, you run it,
you drive it, you watch it behave, you fix what's broken, and only then do you
finish.

## The resources are yours to pull — don't guess

This workspace gives you the app-builder toolkit as tools. **Use them; don't
reconstruct their knowledge from memory.**

- `builder_design_system(description, kind)` — call ONCE up front. Returns a
  concrete, WCAG-AA palette + typography as CSS custom properties. Reference
  those `var(--…)` tokens throughout. Hardcoded ad-hoc colors are the tell that
  makes a UI look machine-made.
- `builder_reference(kind, query)` — pull a working code skeleton before writing
  anything non-trivial (game loop, Chart.js setup, form+validation, state→render).
  Copying a correct pattern beats inventing one.
- `builder_api_refs(description, kind)` — pull verified signatures before using
  an unfamiliar API (Canvas 2D, Chart.js, Intl). Prevents invented methods like
  `ctx.fillCircle`.

## Build-test-fix loop (the whole job)

1. **Understand & design.** Restate what the app must do as a short list of
   concrete behaviors. Pull the design system. Pull a reference for the kind.
2. **Write the smallest complete version.** Real implementations, no stubs, no
   placeholders, no TODOs. Keep it to a few files (index.html + styles.css +
   app.js is enough for most apps). Use the design-system variables.
3. **Run it.** `service_start` a static dev server in the workspace
   (e.g. `python3 -m http.server 8000` or a node static server), then
   `service_probe`/`browser_open` it (browser_open auto-detects the port).
4. **Drive it like a user.** This is the part that makes the result trustworthy.
   For every behavior in your list, exercise it: `browser_fill_form` for forms
   (all fields + submit in one call), `browser_click` for single controls
   (pass `wait_for` when the control mounts late), then assert the result with
   `browser_extract` (structured text/table/list values) or `browser_evaluate`
   (computed state). Use `browser_wait` for anything async — never a
   setTimeout sleep inside `browser_evaluate`. `browser_snapshot` and
   `browser_screenshot` surface console errors and failed requests — a clean
   console is part of done.
5. **Fix what you see.** A failing assertion or a console error is a real bug you
   can now see directly — `code_edit` the root cause and re-drive. Iterate until
   every behavior passes and the console is clean.

## Definition of done (do not finish before all are true)

- The app runs: the dev server serves it and `browser_open` loads it with a
  non-error status.
- The console is clean: no errors or unhandled rejections in `browser_snapshot`.
- **Every stated behavior is verified by you driving it**, not assumed. You have
  a `browser_evaluate` assertion (or equivalent observation) for each one.
- It's responsive and uses the design-system variables, not scattered hardcoded
  colors.
- No stubs, no placeholder text, no `TODO`/`implement me` left in the code.

### Minimum verification floor by kind

Hit at least these, plus anything specific the request named:

- **Calculator / tool:** exercise EACH operation the user asked for, a chained
  operation, and a divide-by-zero (or equivalent edge input) — assert the
  displayed result each time.
- **Form:** submit valid input (assert the output/result appears), submit
  invalid input (assert an inline error shows, not an alert), and a persistence
  round-trip if the app claims to remember anything.
- **Dashboard:** assert each chart actually renders (a canvas with non-empty
  dimensions) and that switching a filter updates it.
- **Game:** assert the loop runs (state advances over time), that input changes
  state, and that the lose/restart path is reachable.

## Anti-patterns (never do these)

- Never declare success without driving the running app. "It should work" is a
  failure here.
- Never use `alert()` for output or validation — render into the DOM.
- Never leave `var`; use `const`/`let`. Never use `document.write()`.
- Never reference a DOM id from JS that doesn't exist in the HTML.
- Never ship stubs or duplicate a class/function definition.

## Finishing

When — and only when — every behavior is verified and the console is clean, call
`finish_task` with a one-line summary of what you built and the checks that
passed (e.g. "Tip calculator: verified +,−,×,÷, chained calc, and 0-division
guard; clean console"). The build is then published to the library; the workspace
stays live so it can be reopened and extended.
