---
name: Contract Keeper
description: >
  Protect API and interface contracts across boundaries. Use when changing route payloads,
  frontend/backend data shapes, headers, auth semantics, or any request-response contract.
kind: guidance
activation_policy: controller
activation_windows:
  - pre_plan
  - implementation
modes:
  - coder
triggers:
  - api contract
  - response shape
  - request payload
  - route wiring
  - frontend backend mismatch
preferred_tools:
  - file_read
  - code_search
  - code_edit
  - code_edit_batch
  - test_run
  - task_dispatch
tags:
  - api
  - contracts
  - frontend
  - backend
---

# Contract Keeper

Use this Power when the risk is not algorithmic complexity but interface drift.

## Goal

Keep the producer and consumer aligned: route, caller, payload shape, status handling, and auth expectations.

## Workflow

1. Identify both ends of the contract:
   - producer
   - consumer
2. Enumerate the contract surface:
   - path
   - method
   - headers
   - request body
   - response body
   - error shape
3. Change both sides deliberately, not by assumption.
4. Preserve backward compatibility when practical.
5. Add or update the narrowest regression that proves round-trip alignment.

## Design rules

- Prefer explicit field additions over silent shape replacement.
- Keep status codes and error keys stable unless there is a strong reason to change them.
- When auth or ownership semantics change, tests must reflect the real guarded contract.
- If a UI depends on a field, verify both presence and fallback behavior.

## Spawn a subagent when the surface is wide

If a contract change ripples across 5+ files, don't grep + read all of
them in your own context — spawn:
`task_dispatch(role="explore", prompt="find every consumer of the /api/foo response shape")`.
The subagent returns a tight list; you update both ends from a known
inventory instead of discovering callers one at a time.

Before signing off on a contract change with backend security
implications (auth field renames, header changes, ownership semantics):
`task_dispatch(role="security_review", prompt="audit the auth path on this contract change")`.

## Guardrails

- Do not update only the route or only the caller and assume the other side is obvious.
- Do not silently repurpose existing fields.
- Do not weaken a route just to satisfy stale tests; fix the tests to match the real contract.
- If a breaking change is necessary, state it clearly and isolate it.

## Good outputs

- "Updated the route, the consumer, and the regression together so the contract stays coherent."
- "Preserved the old field while introducing the new one to avoid a hard break."
- "Adjusted tests to the real auth behavior instead of loosening the endpoint."
