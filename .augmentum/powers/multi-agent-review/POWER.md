---
name: Multi-Agent Review
description: >
  Parallel multi-reviewer code audit. Partition a subsystem into 4-10
  non-overlapping zones, spawn one read-only audit_zone subagent per
  zone in parallel, consolidate by theme into a single prioritized
  P0/P1/P2 report. Use for substantive subsystem audits at the bar a
  company with Anthropic-level standards would apply.
kind: workflow
activation_policy: manual
modes:
  - coder
triggers:
  - multi-agent review
  - multi agent review
  - subsystem audit
  - codebase audit
  - audit X for shipping
  - have subagents review
  - parallel code review
  - ship-quality review
  - anthropic-bar review
preferred_tools:
  - file_read
  - code_grep
  - code_search
  - task_dispatch
tags:
  - review
  - audit
  - quality
  - subagent
  - parallel
---

# Multi-Agent Review

Run a parallel multi-reviewer audit of a subsystem. Each reviewer is a
read-only `audit_zone` subagent that covers a non-overlapping slice;
when all return you consolidate by theme into one prioritized report.

This is **the** play for "audit X for shipping quality" — don't reach
for one-shot `review` or `security_review` when the user wants
substantive subsystem coverage. They scale to one file or one diff;
this scales to a whole subsystem.

## When to use

- User asks for a code review of a subsystem ("review Connect", "audit
  the narrative mode", "have subagents review this").
- Pre-ship of a substantive surface.
- Post-incident review of the area around the incident.
- Periodic check on subsystems that have grown >5k LOC since the last
  audit.

Don't use for:
- One-file or one-bug questions — read the file or call `review`.
- Issues a static scanner finds (`audit.py` is cheaper).
- Architecture design — use the `plan` role.

## How to run it

### Step 1 — Plan the partition

A good partition has **balanced load, no overlap, sensible boundaries**.

1. **One zone per concern, not per file.** Files sharing a contract
   belong together (a route file + its store + service is one zone).
2. **Target 2k-4k LOC per zone.** Single-file zones lose context;
   >5k LOC zones blow the agent's context budget.
3. **Front + back are different zones.** Different categories apply
   (XSS / accessibility / CSP on frontend; SQL / async / locking on
   backend).
4. **Routes / HTTP get their own zone** when >500 LOC. They
   concentrate auth, validation, status-code, and rate-limit issues.
5. **6 zones is the sweet spot** for medium subsystems. 4 for small,
   8-10 for very large (narrative, coder, modes).

Use `code_search` / `code_grep` to enumerate files. Count LOC. Group
into zones matching the rules above.

State the partition to the user before spawning so they can redirect.

### Step 2 — Spawn N audit_zone subagents in parallel

For each zone, call `task_dispatch` with `role="audit_zone"` and a
prompt of this shape:

```
You are reviewing the {SUBSYSTEM} subsystem ({ONE_SENTENCE_DESCRIPTION}).

{ONE_PARAGRAPH_CONTEXT_FOR_THIS_ZONE — key abstractions, fragile bits,
what the lead has flagged as touchy.}

Your zone:
- {ABSOLUTE_FILE_PATH_1}
- {ABSOLUTE_FILE_PATH_2}
...

Maximum findings: {N}        # typically 25; up to 30 for large zones
Word cap: {WORD_CAP}         # typically 800; up to 1000 for large zones
```

The role's system prompt already contains the categories list, the
severity rubric, and the output format. Don't restate them. Don't ask
the reviewer to write code — the role enforces read-only.

**Spawn all zones in parallel** — make multiple `task_dispatch` calls
in a single tool-use batch. Serial spawning costs Nx the wall clock
for zero benefit.

### Step 3 — Consolidate by theme, not by reviewer

When all reviews return:

1. **Group findings by theme**, not by which reviewer flagged them.
   The same issue surfaces in 2-3 reports under different framings
   (e.g. "block enforcement" might appear in routing, fabric, and
   storage zones independently).
2. **Dedupe.** Pick the report with the best file:line and clearest
   fix; drop the duplicates.
3. **Apply the promote / demote rules:**
   - **Promote P1 → P0** when ≥2 reviewers independently flag the same
     issue from different zones. That cross-zone overlap is the
     strongest signal.
   - **Demote P0 → P1** when a finding requires an attacker model the
     system doesn't claim to defend against, or it's a documentation/
     test gap framed as a bug, or it's a hypothetical race that hasn't
     been triggered in practice.
4. **Pick a top-5-this-week list** — five items with the smallest code
   delta × largest correctness/security delta. Everything else becomes
   backlog.

## Output structure

Use this shape for the consolidated report:

```
# {Subsystem} — consolidated review across {N} reviewers

## P0 — Ship-blockers
### Theme 1 (e.g., Trust boundary)
### Theme 2 (e.g., Block enforcement)
...

## P1 — Should-fix before merge
{grouped by sub-theme}

## P2 — Backlog
{terse bullet list}

## Tests demanded by ≥2 reviewers
{cross-referenced gaps}

## My read (1 paragraph + top-5 picks)
```

## Anti-patterns

- **Re-reading the same file across zones.** If a file appears in two
  zones, the second reviewer pays the context cost twice. Better to
  enlarge one zone than duplicate.
- **Zones smaller than 500 LOC.** Single-file zones produce thin
  reports. Either widen the zone or drop it.
- **Mixing subsystem boundaries within a zone.** If a zone spans
  Connect + Narrative, the reviewer can't apply subsystem-specific
  conventions consistently.
- **No word cap.** Without it, reports balloon to 3-5k words and
  consolidation falls apart. Always pass the cap explicitly.
- **Serial launches.** Always batch the `task_dispatch` calls so they
  fan out in parallel.
- **Asking the reviewer for code.** The role is read-only — asking
  for fixes wastes tokens. Land fixes yourself after consolidation.

## Worked example — Connect (2026-06-05)

6 zones, ~25k LOC, ~2 min wall clock per agent:

| Zone | Files | LOC |
|------|-------|-----|
| Backend storage + protocol | message_store, call_store, contact_store, protocol, 6 migrations | ~3k |
| Backend routing + lifecycle | message_routing, call_routing, call_lifecycle, hub | ~4k |
| Fabric integration | fabric_transport, fabric_inbound, contacts | ~2k |
| HTTP/WS routes | connect_routes.py | ~1.5k |
| Frontend signaling + calls | 11 files (client, dialer, incoming, ringback, …) | ~5k |
| Frontend messaging + outbox + CSS | thread-panel, messages, outbox, connect.css | ~9k |

Top-5 ship-blockers identified after consolidation:
1. Block enforcement gap on call signaling verbs
2. Fabric inbound trusts sender claims (target_user_id + DID host)
3. Test-secret fallback reachable in prod
4. Request-scoped DB conn captured by 60s timer
5. Outbox flush break-on-retry aborts whole burst

Cross-zone theme that emerged: "the substrate is feature-complete; what
needs hardening is the trust boundary + adversarial-input checks that
come after MVP." That synthesis is what consolidation buys you over a
single-agent review.
