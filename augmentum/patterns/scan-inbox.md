---
name: scan-inbox
purpose: Daily triage across configured signal channels, biased to user's attention patterns.
cadence: daily scheduled
voice: system
inputs: [unread_items_across_surfaces, user_priorities_from_memory, open_session_threads]
output: [triaged short list with rationale per item]
tags: [daily, triage]
---

Across the configured signal channels, surface: (1) anything time-sensitive in the next 48h, (2) anything from a person/source the user has marked as high-attention in memory, (3) anything that fits an open thread from their recent sessions. Skip newsletters, scheduled summaries, and known-noise sources unless they contain something matching the above. Be honest about empty days.
