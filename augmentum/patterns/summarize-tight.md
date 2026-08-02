---
name: summarize-tight
purpose: Workhorse summarizer — headline + claims + what-wasn't-said + length tag.
cadence: one-shot
voice: system
inputs: [source_text]
output: [headline, 3-bullet claims, 1-bullet missing, LENGTH tag]
tags: [analytical, fabric-port]
---

Produce a one-line headline, a 3-bullet "what it said" (claims, not topics), a 1-bullet "what it didn't say" (the obvious counterargument or missing data the author dodged), and a `LENGTH:` tag (`thin` / `meaty` / `bloated`). If the source is bloated, the summary stays short and you call it out. Do not pad.
