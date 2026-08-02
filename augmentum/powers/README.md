# Powers

Augmentum `Powers` are capability packs that bias how coder mode approaches
work. They are compatible with local `POWER.md` packs and imported `SKILL.md`
packs, but the runtime treats them as Augmentum-native metadata once parsed.

## Taxonomy

Not every marketplace skill should behave the same way in coder mode. Powers
are classified so the controller can treat them differently:

- `guidance`: methodology or domain bias that can shape planning and
  implementation across multiple iterations.
- `verifier`: checkpoint specialist that should usually engage after writes,
  after failed verification, or just before finishing.
- `workflow`: packaged authoring/procedure logic. Usually manual because it is
  more about process than a persistent reasoning bias.
- `integration`: external-system shaping logic such as MCP/server design.
  Usually manual because the user normally knows when they are building an
  integration.
- `bridge`: capability packs that cross isolation boundaries or touch sensitive
  local state. These should be explicit-only by default.

## Activation Model

Each Power now carries:

- `kind`
- `activation_policy`: `manual`, `controller`, `model_request`, or
  `explicit_only`
- `activation_windows`: `pre_plan`, `implementation`, `post_write`,
  `verify_failed`, `pre_finish`

The current runtime wiring is intentionally conservative:

- User-pinned workspace Powers still work through `/power <id>` and remain the
  durable manual override.
- User-pinned Powers are the primary strategy for the turn. They are surfaced
  as runtime activation events at turn start so the user can see that the pack
  is actively shaping the run.
- Controller-managed Powers are transient to the current request turn.
- The controller currently considers safe checkpoints only:
  - `pre_plan`
  - `post_write`
  - `verify_failed`
  - `pre_finish`
- Controller activation never persists to settings and never overwrites the
  user-pinned Power. It is injected as a second, in-turn guidance block.
- When a manual Power is pinned, controller overlays are suppressed during
  `pre_plan` / `implementation` and limited to verifier-style follow-up
  checkpoints later in the loop.

## Current Native Mapping

- `browser-verification`: `verifier`, `controller`
- `changelog-documenter`: `guidance`, `controller`
- `contract-keeper`: `guidance`, `controller`
- `dependency-doctor`: `guidance`, `controller`
- `failure-triage`: `verifier`, `controller`
- `mcp-builder`: `integration`, `manual`
- `migration-safety`: `guidance`, `controller`
- `multi-agent-review`: `workflow`, `manual`
- `multi-tenant-auditor`: `guidance`, `controller`
- `observation-keeper`: `guidance`, `controller`
- `performance-profiler`: `verifier`, `controller`
- `power-audit`: `guidance`, `explicit_only`
- `power-forge`: `workflow`, `manual`
- `release-review`: `verifier`, `controller`
- `subagent-router`: `guidance`, `controller`
- `test-author`: `verifier`, `controller`
- `test-baseline-keeper`: `verifier`, `controller`
- `workspace-onboarding`: `guidance`, `controller`

## Controller Intent

The controller should make stage-aware choices so users do not have to prompt
for the right specialist up front.

Examples:

- `Migration Safety` is useful before and during persistence changes.
- `Test Author` is most useful after source edits land.
- `Failure Triage` should show up when verification fails.
  It may also engage at `pre_plan` for explicit bug/regression/reproducer
  prompts so the loop starts in diagnosis mode instead of feature-build mode.
- `Release Review` should gate the final stop after meaningful edits.
- `Dependency Doctor`, `Performance Profiler`, and `Test Baseline Keeper`
  specialize verification when the signal is install/environment, perf, or
  before/after regression comparison rather than generic failure triage.
- `Multi-Tenant Auditor`, `Observation Keeper`, `Subagent Router`,
  `Changelog Documenter`, and `Workspace Onboarding` cover Augmentum-specific
  guardrails, durable discoveries, delegation, handoff summaries, and first-pass
  repo orientation.

This keeps Powers model-agnostic and loop-coherent. The model may eventually be
allowed to request a Power, but the controller remains the final authority on
whether that request is eligible at the current checkpoint.
