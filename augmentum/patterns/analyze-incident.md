---
name: analyze-incident
purpose: Postmortem skeleton from logs / timeline / commit context. Blameless.
cadence: event-triggered (build/deploy failure) | on-demand
voice: system
inputs: [logs, timeline, recent_commits, affected_systems]
output: [what / timeline / proximate-cause / contributing-factors / masking / action-items]
tags: [devops, fabric-gap]
---

Given logs / timeline / commit context: produce a postmortem skeleton — **what happened** (in plain language), **timeline** (timestamps, terse), **what triggered it** (proximate cause), **why it propagated** (contributing factors), **what masked it** (why we didn't catch it earlier), **action items** (specific, owner-able). Mark anything that's speculation `?UNCONFIRMED`. Do not write blameful language.
