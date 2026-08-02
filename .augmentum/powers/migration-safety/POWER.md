---
name: Migration Safety
description: >
  Guide schema and data changes safely. Use when adding migrations, changing persistence,
  backfilling fields, or touching data models where correctness and rollback behavior matter.
kind: guidance
activation_policy: controller
activation_windows:
  - pre_plan
  - implementation
  - pre_finish
modes:
  - coder
triggers:
  - write migration
  - schema change
  - backfill data
  - alter table
  - persistence change
preferred_tools:
  - file_read
  - code_search
  - code_edit
  - shell_exec
  - test_run
tags:
  - database
  - migration
  - persistence
  - safety
---

# Migration Safety

Use this Power when changing stored data or the shape of persisted state.

## Goal

Preserve existing data, preserve application boot, and make failure modes obvious before the change ships.

## Workflow

1. Identify the current schema and all read/write paths.
2. Separate the change into:
   - schema addition
   - data backfill
   - application read/write adoption
3. Prefer additive migrations first.
4. Keep old rows readable until the application has fully transitioned.
5. Verify both:
   - fresh install path
   - upgrade path from existing rows
6. Call out rollback limits if the change is not easily reversible.

## Design rules

- Add columns before requiring them.
- Use safe defaults where possible.
- Treat nullability changes and unique constraints as high-risk.
- Avoid mixing unrelated schema work into the same migration.
- For multi-tenant data, check user isolation explicitly.

## Guardrails

- Do not rely on application code alone to "eventually" fix incompatible rows.
- Do not drop or rename fields in the same pass unless compatibility is already proven.
- Do not claim safety without checking startup, reads, and writes against migrated state.
- If a destructive cleanup is needed later, stage it for a separate pass.

## Good outputs

- "Added the column first, kept old rows readable, and only then switched the write path."
- "Verified both in-memory fresh state and SQLite upgrade behavior."
- "Deferred the cleanup/drop step because it would have increased migration risk in this pass."
