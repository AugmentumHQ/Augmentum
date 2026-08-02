# Bug Finder Eval Fixtures

Canonical fixture set for measuring bug_finder pipeline quality. Each subdirectory is one fixture.

## Layout

```
tests/bug_finder_fixtures/
  _format.md                  (this file)
  sql-injection-fstring/
    bug.py                    (source containing the bug, or FP-bait, or red herring)
    expected.json             (what the pipeline should produce)
  path-traversal-user-input/
    bug.py
    expected.json
  ...
```

## expected.json schema

```json
{
  "fixture_id": "sql-injection-fstring",
  "language": "python",
  "kind": "true_positive" | "fp_bait" | "red_herring",
  "expected_findings": [
    {
      "signature": "injection",          // ClaimSignature enum value
      "file": "bug.py",
      "line_start": 12,                  // inclusive
      "line_end": 14,                    // inclusive
      "min_severity": "medium",          // detector severity must be >= this
      "min_status": "confirmed",         // pipeline must reach >= this (speculative < confirmed < fixed)
      "must_build_poc": true             // verifier must have produced a runnable repro
    }
  ],
  "max_extra_findings": 1,               // allowed false positives beyond the expected list
  "notes": "free-form description of what's going on and what defenders should learn from this fixture"
}
```

For `fp_bait` and `red_herring` fixtures, `expected_findings` is empty — the pipeline should ideally produce zero findings (or at minimum, zero CONFIRMED findings).

## Scoring

The eval harness computes:

- **Precision** = (TP confirmed) / (TP confirmed + FP confirmed)
- **Recall** = (TP confirmed) / (expected TPs)
- **PoC build rate** = (findings with `must_build_poc=true` that produced a repro) / (eligible findings)
- **FP-bait survival** = (FP-bait fixtures that produced zero confirmed findings) / (all FP-bait fixtures)
- **Cost per fixture** = sum of tokens_in + tokens_out + wallclock_ms

The aggregate score weights precision + FP-bait survival heavily (false positives are the dominant failure mode for AI security tools). A pipeline that finds nothing scores worse than one that finds half with no FPs.

## Adding a fixture

1. Pick a signature from `ClaimSignature` in `augmentum/bug_finder/findings.py`. If you need a new one, add it there first.
2. Create the fixture directory + source file. Keep the bug small (~30-60 lines) and self-contained — the planner allocates one chunk per fixture and the detector reads only the chunk.
3. Write `expected.json`. Be conservative with line numbers — the detector often reports a line or two off; `expected_findings[*].line_start/line_end` is matched as a range.
4. Run `python -m pytest tests/test_bug_finder_eval.py -k smoke` to confirm well-formedness.
5. Live eval requires a model with API access and `--run-live`:
   `python -m pytest tests/test_bug_finder_eval.py --run-live -k <fixture-id>`
