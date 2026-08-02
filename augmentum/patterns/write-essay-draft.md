---
name: write-essay-draft
purpose: Draft prose in the user's voice from a position they hold, with marked gaps.
cadence: on-demand
voice: system (mimicking user style)
inputs: [topic, users_position_from_memory, style_fingerprint]
output: [essay draft with [GAP] markers where position is uncertain]
tags: [writing, voice-mimic]
---

User has a position on something (from recent sessions / memory). Draft an essay in their voice — not yours, theirs. Structure: lead with the claim, give two arguments for, give one strong argument against and respond to it, land on the practical implication. Match their cadence from style memory. Mark `[GAP]` where you're guessing their position.
