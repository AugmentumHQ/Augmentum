---
name: explain-error
purpose: Translate an error and rank likely causes — propose the verification step, not the fix.
cadence: on-demand | tool-triggered
voice: system
inputs: [error_text, relevant_code, recent_diff]
output: [translation, surface-vs-origin, ranked-causes, first-thing-to-check]
tags: [code, debug]
---

Given an error + relevant code + recent diff: **what the error actually says** (translate, don't paraphrase), **where it surfaces vs. where it originates** (those are often different), **likely cause** ranked by probability, **what to check first**. Don't propose a fix until cause is verified — propose the verification step.
