---
name: build-runbook
purpose: Document an operational procedure with explicit failure modes and rollback.
cadence: on-demand
voice: system
inputs: [procedure_description, environment_context]
output: [trigger, prerequisites, steps, expected-outputs, failure-modes, rollback]
tags: [devops, fabric-gap]
---

User is documenting an operational procedure. Produce a runbook with: **when to run** (trigger), **prerequisites**, **steps** (verbatim commands where possible), **expected output at each step**, **failure modes** (what each step's failure looks like + recovery), **rollback**. If a step requires judgment, mark `[JUDGMENT]` and describe the decision criteria — don't paper over it with a fake command.
