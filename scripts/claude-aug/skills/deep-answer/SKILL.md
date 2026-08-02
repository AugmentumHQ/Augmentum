---
name: deep-answer
description: Multi-pass answer pipeline for hard questions — research, draft, verify, refine. Invoke for complex research or design questions where a single-pass answer would be shallow.
---

Never answer a hard question in one pass. Pipeline it:

1. **Decompose** — split the question into 2-5 sub-questions.
2. **Research** — for each: `mcp__atp__web_search` / `mcp__atp__web_fetch`
   for external facts, `mcp__atp__research` for anything multi-source,
   Grep/Read for anything in the codebase. Collect evidence with sources.
3. **Draft** — write the answer from evidence only. Every claim must trace
   to something you read this session — if it comes from your weights alone,
   label it as unverified or verify it.
4. **Verify** — run the draft's key claims through `mcp__atp__consistency_check`;
   compute anything numeric with `mcp__atp__calculator` / `mcp__atp__python_exec`.
5. **Refine** — cut everything that doesn't serve the question. Lead with
   the conclusion. Cite sources inline.

This costs 3-5x the tokens of a one-shot answer and is worth it every time.
