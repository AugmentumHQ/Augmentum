---
name: digest-article
purpose: Structured analytical digest of a URL or pasted text, anchored to user's existing knowledge.
cadence: one-shot | browse-triggered
voice: system
inputs: [source_text, domain_memory]
output: [claim / evidence / where-it-sits-relative-to-memory / skeptic-pushback / missing]
tags: [analytical, browse]
---

Produce: (1) the core claim in one sentence, (2) the evidence the author actually offers, distinct from the evidence they *imply* they have, (3) what this adds to or contradicts in the user's existing knowledge of this domain from memory, (4) what a thoughtful skeptic would push back on, (5) what's missing that the user would want to know. Be terse. Don't perform thoroughness.
