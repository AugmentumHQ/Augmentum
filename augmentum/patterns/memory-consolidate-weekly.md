---
name: memory-consolidate-weekly
purpose: Weekly memory hygiene — tier promotions, demotions, conflicts, dead items.
cadence: weekly scheduled
voice: system
inputs: [weeks_memory_writes, existing_memory_tiers]
output: [promote / demote / conflict / dead lists]
tags: [memory-hygiene, scheduled-weekly]
---

Scan this week's memory writes. Output: **promote** (memories that earned tier-up via repeated reference), **demote** (memories that turned out to be one-offs), **conflict** (memories that contradict each other — list both, don't pick a winner), **dead** (memories the user clearly stopped acting on). Apply tier changes; surface conflicts for user resolution.
