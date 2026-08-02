---
name: test-gap-finder
purpose: Identify untested-by-intent behaviors, not uncovered lines.
cadence: on-demand | pre-PR
voice: system
inputs: [diff_or_module, testing_conventions_from_memory]
output: [edge-cases / error-paths / invariants / integration-boundaries — what each missing test should assert]
tags: [code, testing]
---

For the diff/module: identify untested behaviors — not just uncovered lines, **untested-by-intent**. Categorize: **edge cases not exercised**, **error paths not asserted**, **invariants not checked**, **integration boundaries that mock instead of integrate** (per the user's testing convention from memory). Don't write the tests — describe what each missing test should assert.
