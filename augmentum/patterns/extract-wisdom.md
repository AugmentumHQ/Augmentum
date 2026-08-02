---
name: extract-wisdom
purpose: Augmentum-native port of Fabric's extract_wisdom — extract signal, but memory-grounded.
cadence: one-shot | browse-note-triggered
voice: system
inputs: [source_text, user_interests_from_core_profile, existing_memory]
output: [structured extraction with named buckets]
tags: [analytical, fabric-port]
---

Extract what's signal from the input. Surface (1) ideas the user would care about given what you know about them, (2) ideas that *challenge* their existing positions in memory, (3) quotable lines worth keeping, (4) references worth following. Mark items that contradict something in their memory with `~CONTRADICTS:` and a pointer. If most of this input is filler, give a short extraction and say so; don't pad.
