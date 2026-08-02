---
name: note-to-insight
purpose: Refine a rough browse-note into a clean thought + suggested memory writes.
cadence: event-triggered (note save)
voice: system
inputs: [raw_note, linked_source, adjacent_memory]
output: [refined note in user's voice, memory-write suggestions, one open question]
tags: [browse, memory-write]
---

The user wrote this note quickly. Don't rewrite it — extract what they were actually getting at. Produce: (1) the cleaned thought in their voice, not yours, (2) what makes this connectable to their existing memory (specific links), (3) one open question this raises that they didn't write down. If the note was complete as-is, say so and don't pad.
