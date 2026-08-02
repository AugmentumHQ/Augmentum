---
name: draft-in-voice
purpose: Draft a reply that sounds like the user, using accumulated style memory.
cadence: on-demand
voice: system (mimicking user style)
inputs: [incoming_message, user_stance_from_memory, style_fingerprint]
output: [2-3 candidate drafts with one-line notes on what each is doing]
tags: [daily, voice-mimic]
---

Draft a reply that sounds like the user, not like an AI. Pull their voice from style memory — their cadence, their qualifiers, what they don't bother to say, where they're brief versus where they explain. Take their position on this topic from memory; don't invent one. Produce 2-3 candidate drafts with one-line notes on what each is doing differently. Flag if their stated position would be unwise to send and let them decide.
