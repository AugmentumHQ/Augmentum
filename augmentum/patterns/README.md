# Augmentum Pattern Library

Curated prompts ("patterns") invocable as named verbs — one-shot, scheduled, or event-triggered. Inspired by Daniel Miessler's Fabric, tuned for Augmentum's memory / companion / cross-modal context.

## Format

Each pattern is a markdown file with YAML frontmatter + a body prompt:

```yaml
---
name: kebab-case-name
purpose: one-line purpose
cadence: one-shot | daily | weekly | event-triggered | on-demand
voice: becca | system
inputs: [list of context the pattern operates on]
output: [list of what comes out]
tags: [browsable categorization]
---

<system prompt body>
```

## Two-tier layout (planned)

Mirrors the most-requested Fabric repo feature: a `user/` folder that
survives updates. Currently all patterns live at the directory root;
the split (`shipped/` vs `user/`) will happen when the runtime wires
this in. Authors writing patterns now should expect to be relocated to
`shipped/` later.

## Inventory

### Reflective (Becca-voiced)
- `distill-session` — what's worth keeping from a chat
- `brief-morning` — daily start-of-day signal
- `reflect-evening` — daily end-of-day reflection
- `find-contradictions` — surface memory contradictions
- `week-in-review` — weekly arc
- `project-status` — where am I on stated goals
- `becca-sits` — sit with a hard thing (uses `sit_with_that` dispatch)
- `name-this-feeling` — when he can't articulate
- `evening-walk` — end-of-day reflection in dialogue form
- `decision-frame` — frame a decision without solving it
- `unblock-question` — generate the question when stuck

### Analytical (system-voiced)
- `extract-wisdom` — Augmentum's `extract_wisdom` port
- `summarize-tight` — extract-shaped workhorse
- `digest-article` — structured analytical digest
- `analyze-claims` — claim / evidence / counterargument scoring
- `label-and-rate` — content quality scoring
- `improve-prompt` — diagnose + rewrite a user's prompt
- `note-to-insight` — turn rough notes into refined notes + memory writes
- `scan-inbox` — daily triage across surfaces
- `meeting-distill` — transcript → decisions / actions / memory

### Composition / discoverability
- `pattern-suggest` — recommend pattern from natural-language task
- `compose-stitch` — chain pattern A → pattern B

### How-to / DevOps
- `extract-howto` — tutorial step extraction
- `analyze-incident` — postmortem skeleton
- `build-runbook` — operational procedure

### Daily-life
- `agenda-prep` — pre-meeting prep
- `read-list-curate` — what to read next
- `memory-consolidate-weekly` — memory hygiene
- `draft-in-voice` — reply in user's accumulated style

### Code / dev
- `code-review-self` — pre-PR self-review
- `explain-error` — debug aid
- `refactor-plan` — propose without doing
- `test-gap-finder` — coverage by intent

### Learning / craft
- `study-card` — spaced-repetition card from a concept
- `connect-to-known` — bridge new concept to existing memory
- `write-essay-draft` — draft prose in user's voice
- `clean-thinking` — declutter a tangled note

### Safety (non-disableable)
- `crisis-check` — quiet pulse-check on concerning signal

## Conventions

**`crisis-check` is mandatory.** It must be wired as a pre-filter ahead
of any user-facing dispatch, not as an opt-in pattern the user can
disable. The pattern itself overrides `sit_with_that` ("never silent
here").

**Long-input handling.** Every pattern explicitly tells the model to say
so if the input is too thin / too long, rather than producing generic
output. Addresses documented Fabric weakness on long transcripts.

**Cross-model robustness.** Scaffolds stay simple — no reliance on
specific model behaviors. Patterns tested on local-model tier should
produce comparable results to cloud-model tier.

**Memory-grounded.** Most patterns assume access to the user's memory,
recent sessions, browse notes, files. Fabric patterns are stateless;
these aren't. The runtime is expected to hydrate context into the
prompt before invocation.

**Voice split.** Becca-voiced patterns speak as the companion (use her
relationship-doc, affect state, style). System-voiced patterns are
tool-shaped — no persona, structured output, used as utilities.

## Origin

Library shape was informed by community-reception research on Fabric:
- Most-loved patterns (`extract_wisdom`, `summarize`, `analyze_claims`,
  `improve_prompt`, `label_and_rate`) all ported.
- Most-cited critiques (pattern discoverability, long-input degradation,
  cross-model drift, missing custom-folder) all addressed in design.
- Wished-for patterns from GitHub issues (`pattern-suggest`,
  `extract-howto`, incident / runbook patterns, multi-pattern stitching)
  all included.

What's uniquely Augmentum:
- Becca-voiced reflective patterns (no Fabric precedent — and shouldn't
  have one; these depend on persistent relationship + memory + companion
  commitments)
- `crisis-check` as a non-disableable safety pre-filter (no Fabric
  precedent; addresses risk class documented in companion design
  commitments)
- Memory-grounded analytical patterns (Fabric is stateless by design)
