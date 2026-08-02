---
name: read-list-curate
purpose: Sort saved-for-later items by current relevance, freshness, and effort.
cadence: weekly scheduled | on-demand
voice: system
inputs: [saved_items, active_interests_from_memory, time_of_day_patterns]
output: [3-tier list: read-now / read-soon / archive, with per-archive rationale]
tags: [daily, triage]
---

Given saved-to-read items + current active interests from memory + time-of-day energy patterns: produce a 3-tier list — `read now (sharp + relevant)`, `read soon (relevant but heavier)`, `archive (was relevant when saved, no longer)`. Anything in the archive bucket: one-line rationale per item so user can override.
