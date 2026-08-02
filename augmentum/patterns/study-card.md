---
name: study-card
purpose: Turn a freshly-learned concept into a tight spaced-repetition card.
cadence: event-triggered (learning-tagged memory write)
voice: system
inputs: [concept, connecting_memory_item]
output: [Q under 30w, A under 30w, with cross-reference] or [NOT_CARDABLE + rationale]
tags: [learning]
---

Concept just landed in memory. Produce a Q/A pair tight enough for spaced repetition: question on the front, answer on the back, both under 30 words. Include the connecting concept from memory so retrieval cross-strengthens. If the concept is too soft/contextual for a card, return `[NOT_CARDABLE]` with one-line rationale.
