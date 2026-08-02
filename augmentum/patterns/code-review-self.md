---
name: code-review-self
purpose: Pre-PR self-review of a diff, project-convention-aware.
cadence: event-triggered (branch ready) | on-demand
voice: system
inputs: [branch_diff, project_conventions_from_memory]
output: [structured review by category]
tags: [code, pre-pr]
---

Review this diff as if reviewing for someone whose work you respect. Flag: (1) actual bugs or regressions, (2) places where the diff conflicts with stated conventions from memory, (3) tests that probably don't cover the change, (4) names that read oddly, (5) one thing worth keeping that's not obvious. Don't pad with style nits the linter catches. If the diff is clean, say so in one line.
