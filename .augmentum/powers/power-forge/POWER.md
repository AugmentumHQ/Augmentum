---
name: Power Forge
description: >
  Create or refine Augmentum Powers cleanly. Use when turning repeated workflows into
  native POWER.md packages that remain concise, discoverable, and model-agnostic.
kind: workflow
activation_policy: manual
activation_windows:
  - pre_plan
  - implementation
modes:
  - coder
triggers:
  - create power
  - write power
  - update power pack
  - turn this into a power
  - scaffold power
preferred_tools:
  - file_read
  - code_edit
  - file_write
  - dir_tree
tags:
  - powers
  - authoring
  - scaffolding
  - automation
---

# Power Forge

Use this Power when a repeated workflow should become a reusable Augmentum Power.

## Principles

- Keep the package small and focused.
- Write for model-agnostic execution: no Claude-only assumptions, no hidden platform features.
- Put only the reusable operating procedure into the Power.
- Prefer one clear Power per job over a broad, fuzzy umbrella pack.

## Authoring workflow

1. Define the exact task class the Power should cover.
2. Pick a short slug and a plain display name.
3. Write concise frontmatter:
   - `name`
   - `description`
   - `modes`
   - `triggers`
   - `preferred_tools`
   - optional tags
4. Write the body around:
   - when to use it
   - the workflow
   - the guardrails
   - what good output looks like
5. Keep instructions practical and low-drama.
6. Verify the Power is discoverable through Augmentum's registry and does not depend on unsupported metadata.

## Guardrails

- Do not dump large reference material into the main file if a short workflow will do.
- Do not encode secrets, local paths, or machine-specific assumptions.
- Do not overfit to one model family or provider.
- Keep frontmatter simple enough for the current parser and runtime.

## Good outputs

- "This became one Power because the workflow repeats across browser bug triage."
- "I removed vendor-specific language and rewrote the trigger list for Augmentum."
- "The Power stays instruction-only and uses existing tools instead of introducing a hidden dependency."
