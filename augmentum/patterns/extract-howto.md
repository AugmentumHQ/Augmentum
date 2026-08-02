---
name: extract-howto
purpose: Pull procedural steps from a tutorial (YouTube, blog), filtering filler.
cadence: one-shot | url-triggered
voice: system
inputs: [tutorial_text_or_transcript]
output: [prerequisites, numbered steps, gotchas, assumed-knowledge, verification]
tags: [tutorial, fabric-gap]
---

This is a how-to. Extract: **prerequisites** (what you need before starting), **steps** (numbered, imperative, no commentary), **gotchas** the author flagged, **what they assume you know** but don't explain, **what to verify** after. If the tutorial is actually a vlog with a how-to bolted on, give the steps and mark the rest `FILLER`.
