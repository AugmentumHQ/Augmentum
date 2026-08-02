# Bug Finder Bench Fixtures

Each fixture is a JSON file describing one bench run shape. The file
encodes:

- **Which workspace to audit** (`workspace_id`)
- **Which mode to use** (`mode`: `explore` or `named-bug`)
- **What goal text the lead should optimize for** (`goal_desc`, `goal_repro`)
- **Which findings the run should produce** (`expected_findings` array)
- **Suggested CLI command** (`notes`)

Match policy for `expected_findings`: greedy first-match across the run's
output. Each expected entry pins:

- `file` (required) — substring match against the finding's `file`
- `claim_signature` (optional) — exact match if pinned, ignored if empty
- `function` (optional) — exact match if pinned, ignored if empty
- `severity_at_least` (optional) — minimum severity floor (`info` / `low` /
  `medium` / `high` / `critical`). Empty = any.

An expected entry without a matching actual finding lands in `unmatched`
and forces the bench's exit code to fail.

## Target workspace

The fixtures here target the **`augmentum-self-test`** workspace
(`45f1e36d-1b8f-42a0-bdcb-fde282363b30`). That workspace has real Augmentum
source at `/workspace/augmentum` — verified to contain `bug_finder`,
`auth`, `proxy/*_routes.py`, etc.

## Fixture catalog

| File | Mode | Purpose |
|---|---|---|
| `augmentum_baseline.json` | (any) | Pipeline-completes smoke test |
| `augmentum_explore.json` | `explore` | Static-pipeline blind sweep |
| `augmentum_named_bug.json` | `named-bug` | Lead-loop focused investigation |

## Running

Smoke (validates the run completes):

```bash
python scripts/bug_finder_test_bench.py \
    --workspace-id 45f1e36d-1b8f-42a0-bdcb-fde282363b30 \
    --model deepseek-v4-pro \
    --max-chunks 6 \
    --focus-paths augmentum/auth,augmentum/proxy \
    --fixture tests/bug_finder_bench_fixtures/augmentum_baseline.json
```

Named-bug (exercises the lead loop end-to-end):

```bash
python scripts/bug_finder_test_bench.py \
    --workspace-id 45f1e36d-1b8f-42a0-bdcb-fde282363b30 \
    --model deepseek-v4-pro \
    --goal-mode named-bug \
    --goal-desc "find blanket except-handlers in route handlers that leak internal error context into HTTP responses" \
    --goal-repro "look at augmentum/proxy/*_routes.py" \
    --max-chunks 6 \
    --focus-paths augmentum/proxy \
    --fixture tests/bug_finder_bench_fixtures/augmentum_named_bug.json
```

## Capturing a baseline

The fixtures ship with empty `expected_findings`. To turn one into a real
regression assertion:

1. Run the bench once (any seeded mode).
2. Pick 1-3 high-confidence findings from the scorecard's "Top findings"
   section — ones with `severity_at_least=medium` and a stable
   `claim_signature`.
3. Add each as an entry under `expected_findings`. Use the substring of
   the file path that's stable across runs (e.g. `auth_routes.py` not
   `/workspace/augmentum/proxy/auth_routes.py`).
4. Re-run. The bench now fails when the model regresses.

Don't over-pin — flaky expectations are worse than no expectations.
