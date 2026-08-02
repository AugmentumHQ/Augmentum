---
name: pattern-suggest
purpose: Recommend the right pattern for a natural-language task; flag gaps when nothing fits.
cadence: on-demand
voice: system
inputs: [user_task_description, pattern_library_inventory]
output: [best-fit pattern, 0-2 backups, suggested args, optional chain-into-next]
tags: [meta, discoverability]
---

User described a task in natural language. Return: (1) the single best-fit pattern name, (2) up to 2 backups if the first doesn't land, (3) what arguments / input to pass it, (4) what to chain it into next if useful. If no pattern fits, say so plainly and describe the gap — that's a signal to author a new pattern, not to force-fit.
