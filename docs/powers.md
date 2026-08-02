# Powers — using and creating your own

**Powers** are small capability packs that shape how **coder mode** approaches
work — a methodology, a checkpoint specialist, a packaged workflow. Think of them
as reusable "modes of working" you can pin to a task: a security-audit lens, a
test-authoring routine, a migration-safety checklist. Augmentum ships a set of
built-in Powers, and you can create your own — usually without writing a file by
hand.

## Using a Power

- **In the UI:** open the **Powers** panel, find a Power, and click **Use Now**
  (or **Turn Off**). Enable/disable controls which Powers are available at all.
- **By command:** pin one for the current workspace with `/power <id>`. A pinned
  Power is the durable manual override — it's surfaced as an activation event at
  the start of each turn so you can see it's shaping the run.

Some Powers activate automatically at the right moment (e.g. a *verifier* Power
engaging after a failed test) if their activation policy allows; others are
manual-only. See **Activation**, below.

## Creating a Power — three ways

### 1. Let the agent write it (easiest)

In coder mode, just describe the workflow and ask for a Power:

> *"Turn this into a power."*  ·  *"Create a power that runs our migration
> safety checklist."*  ·  *"Scaffold a power for release review."*

This invokes the built-in **Power Forge**, which authors a clean `POWER.md` for
you — concise, discoverable, and model-agnostic. This is the recommended path:
do a workflow once, then capture it as a reusable Power.

### 2. Write a `POWER.md` by hand

A Power is a folder with a single markdown file:

```
.augmentum/powers/<your-slug>/POWER.md
```

The file is YAML frontmatter + an instructions body:

```markdown
---
name: Migration Safety
description: >
  Guard SQLite migrations — check reversibility, FK constraints, and that new
  columns are wired end-to-end before finishing. Use when adding migrations.
kind: verifier
activation_policy: controller
activation_windows:
  - post_write
  - verify_failed
modes:
  - coder
triggers:
  - add migration
  - alter table
  - new column
preferred_tools:
  - file_read
  - code_grep
  - test_run
tags:
  - database
  - safety
---

# Migration Safety

When the task touches a migration:

1. Confirm the migration is reversible (or note why it isn't).
2. Verify FK constraints won't block inserts.
3. Trace any new column end-to-end (write path → read path → UI) before finishing.
4. Fail loudly if a column is added but never consumed.
```

### 3. Import a Claude Skill

Augmentum reads Claude `SKILL.md` packs as native Powers — drop one under:

```
.claude/skills/<slug>/SKILL.md
```

and it appears in the Powers panel alongside the native ones.

## The `POWER.md` fields

| Field | Meaning |
| --- | --- |
| `name` | Display name. |
| `description` | One or two sentences — *when* to use it. This is what the model reads to decide relevance. |
| `kind` | Taxonomy (see below) — controls how the runtime treats it. |
| `activation_policy` | `manual`, `controller`, `model_request`, or `explicit_only`. |
| `activation_windows` | When it may engage: `pre_plan`, `implementation`, `post_write`, `verify_failed`, `pre_finish`. |
| `modes` | Which modes it applies to (usually `coder`). |
| `triggers` | Phrases that hint the Power is relevant. |
| `preferred_tools` | Tools the Power expects to use. |
| `tags` | Free-form labels. |

### Kinds

- **`guidance`** — a methodology or domain bias that shapes planning and
  implementation across a whole run.
- **`verifier`** — a checkpoint specialist that engages after writes, after a
  failed verification, or just before finishing.
- **`workflow`** — packaged authoring/procedure logic; usually manual, because
  it's about *process* more than a persistent reasoning bias.
- **`integration`** — external-system shaping (e.g. MCP/server design); usually
  manual, since you know when you're building an integration.
- **`bridge`** — packs that cross isolation boundaries or touch sensitive local
  state. Explicit-only by default.

### Activation policies

- **`manual`** — only when you pin it (`/power <id>` or **Use Now**).
- **`controller`** — the runtime may engage it in its allowed windows.
- **`model_request`** — the model may request it when it judges it relevant.
- **`explicit_only`** — never automatic; must be pinned. Used for `bridge` Powers.

The current runtime wiring is intentionally conservative: **user-pinned Powers
are always the primary strategy for the turn**, and automatic activation stays
within each Power's declared windows.

## Tips

- Keep the description sharp — it's the model's only signal for *when* to reach
  for the Power. "Use when adding a migration" beats "database stuff."
- Prefer **Power Forge** for a first draft, then trim.
- A good Power is concise and model-agnostic — it biases *how* work is done, it
  doesn't hard-code answers.
