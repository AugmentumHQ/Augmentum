# Coder Eval Harness

Regression-detection corpus for the coder mode pipeline. Patterned after
`tests/agentic_evals/`. The fast pytest suite validates case loadability,
assertion specs, tier classification, and one scripted Hybrid smoke test.
The standalone bench drives `CoderHandler` end-to-end against fixture
workspaces and records file state, tool usage, step traces, and outcomes.

The harness is not about beating SWE-bench. It exists so Coder loop changes
can land without silent regressions on small tasks that already work.

## Layout

```text
tests/coder_evals/
  cases/
    reflex/        Tier 0: single-shot edits
    surgical/      Tier 1: one to three files
    composed/      Tier 2: multi-file tasks
    project/       Tier 3: from-scratch / larger tasks
  bench.py         Hybrid behavior bench and local workspace adapter
  conftest.py      fixture builders and case loader
  properties.py    assertion helpers
  test_runner.py   parametrized pytest checks
```

## Case Format

```yaml
name: add_missing_import
tier: reflex
workspace:
  files:
    main.py: |
      def main():
          print(json.dumps({"hello": "world"}))

user_message: "Add the missing json import to main.py"

backend: scripted
responses:
  - tool_calls:
      - name: code_edit
        input:
          file_path: main.py
          search: "def main():"
          replace: "import json\n\n\ndef main():"
  - content: "Added the missing json import."

assertions:
  - property: file_contains
    path: main.py
    text: "import json"
  - property: max_iterations
    n: 3
  - property: tier_classified_as
    expected: reflex
```

The bench normalizes older seed scripts, including `file_path` to `path` and
`code_read` to `file_read`, and inserts a `file_read` before scripted edits so
the scripts satisfy Hybrid's read-before-edit guard.

## Running

```powershell
pytest tests/coder_evals/ -q
pytest tests/coder_evals/ -k reflex
```

## Hybrid Behavior Bench

Scripted regression run:

```powershell
python scripts/run_coder_hybrid_bench.py --backend scripted
python scripts/run_coder_hybrid_bench.py --backend scripted --case reflex
```

Local model behavior captures:

```powershell
python scripts/run_coder_hybrid_bench.py --backend ollama --list-models
python scripts/run_coder_hybrid_bench.py --backend ollama --model small=llama3.1:8b
python scripts/run_coder_hybrid_bench.py --backend openai --base-url http://localhost:1234/v1 --model medium=qwen3-coder
```

Reports land under `.augmentum/coder-bench/<timestamp>/` by default:

- `summary.md`: outcome totals, step trace, final prose, failures.
- `results.json`: full machine-readable run bundle.
- `<case>-<model>.json`: one result per case/model pair.

Outcome labels:

- `perfect`: assertions and post-run verification passed.
- `ended_early`: the model stopped without satisfying the case.
- `partial`: the model changed something but missed one or more criteria.
- `loop_stopped`: Hybrid safeguards broke a loop or streak.
- `backend_error`: the model/backend failed before the run could finish.

## Covered Today

- `cases/reflex/case_add_missing_import.yaml`
- `cases/surgical/case_fix_null_check.yaml`
- `cases/composed/case_extract_helper.yaml`
- `cases/project/case_cli_calculator.yaml`

## Not Covered Yet

- Live local-model runs are non-deterministic and should be treated as
  behavior captures, not merge-blocking unit tests.
- Browser/service-heavy tasks need additional fixture support before they
  should be added to the seed corpus.
- Real user-session captures are still future work; synthetic seeds are enough
  to test the harness and catch obvious loop regressions.
