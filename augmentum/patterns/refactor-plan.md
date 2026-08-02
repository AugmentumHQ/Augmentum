---
name: refactor-plan
purpose: Propose a refactor (don't execute) with phasing and load-bearing risks.
cadence: on-demand
voice: system
inputs: [target_code_or_module]
output: [what-changes, what-stays-and-why, 1h/half-day/multi-day phasing, risks]
tags: [code, planning]
---

Given target code: identify the refactor — what changes, what stays, why. **Phase the work**: what's a 1-hour change, what's a half-day, what's a multi-day. **Risks**: what's load-bearing in the current shape that the refactor disturbs. Don't write the refactor. Don't make it more elegant than needed. If "leave it alone" is the right call, say so.
