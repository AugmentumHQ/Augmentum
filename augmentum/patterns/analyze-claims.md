---
name: analyze-claims
purpose: Score each load-bearing claim by evidence type + counterargument.
cadence: one-shot
voice: system
inputs: [source_text]
output: [per-claim: claim / evidence / evidence-type / counterargument; trust-rating]
tags: [analytical, fabric-port]
---

For each load-bearing claim in the source: state the claim → state the evidence given → state the evidence type (citation / anecdote / appeal to authority / hand-wave) → state one strong counterargument the author should have addressed. Mark unsupported claims `UNSUPPORTED`. End with a 1-line trust-rating: `report-grade` / `essay-grade` / `vibes-grade`.
