---
name: label-and-rate
purpose: Quality-rate and tag content using the user's existing tag vocabulary.
cadence: event-triggered (save) | one-shot
voice: system
inputs: [content, user_tag_vocabulary_from_memory, user_existing_knowledge]
output: [signal-density, novelty, time-to-value, rigor, 3-5 tags, KEEP/SKIM/SKIP verdict]
tags: [analytical, fabric-port, triage]
---

Rate the content on: **signal density** (S/M/L), **novelty for this user** (low/med/high — pull what they already know from memory), **time-to-value** (minutes-to-read vs. payoff), **rigor** (rigorous / mixed / sloppy). Tag with 3-5 topic labels using the user's existing tag vocabulary from memory, not new ones. End with one-line `KEEP` / `SKIM` / `SKIP` verdict.
