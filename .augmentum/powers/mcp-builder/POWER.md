---
name: MCP Builder
description: >
  Design and scaffold MCP-backed integrations cleanly. Use when turning APIs, services,
  or internal systems into tool-shaped interfaces for Augmentum or other agents.
kind: integration
activation_policy: manual
activation_windows:
  - pre_plan
  - implementation
modes:
  - coder
triggers:
  - build mcp
  - create mcp server
  - integrate api
  - tool wrapper
  - model context protocol
preferred_tools:
  - file_read
  - code_search
  - code_edit
  - shell_exec
tags:
  - mcp
  - integration
  - api
  - scaffolding
---

# MCP Builder

Use this Power when the job is to expose an external capability through a clean tool interface instead of scattering one-off calls through the app.

## Core goals

- Translate messy external systems into a small, stable tool surface.
- Keep authentication, transport, error handling, and schemas explicit.
- Prefer a thin server with sharp tool contracts over a giant integration layer.

## Workflow

1. Clarify the integration target:
   - service or API
   - authentication model
   - core operations
   - rate limits or failure modes
2. Define the tool boundary:
   - what should become a tool
   - what should stay internal
   - what inputs and outputs should look like
3. Start read-only where possible.
4. Normalize errors so callers see useful failure reasons.
5. Avoid hidden side effects.
6. Add a narrow smoke path before expanding feature coverage.

## Design rules

- Prefer a few well-shaped tools over dozens of thin endpoint mirrors.
- Make secrets configurable; never hardcode tokens or base URLs.
- Keep schemas predictable and human-readable.
- If an OpenAPI spec exists, use it as source material, but do not blindly generate every endpoint.
- Separate transport concerns from business/tool semantics.

## Guardrails

- Do not assume Claude-only behavior, subagents, or proprietary hooks.
- Do not add remote installs or example commands that mutate the host unless the user explicitly asked for them.
- Do not claim an integration is production-ready without at least one executed happy-path check or a clear note that verification is still pending.

## Good outputs

- "Exposed three tools: list projects, get project, create ticket. Left admin operations out of scope."
- "Wrapped 429s and auth failures into structured tool errors instead of leaking raw provider responses."
- "Used the existing Augmentum route/store patterns instead of inventing a parallel integration framework."
