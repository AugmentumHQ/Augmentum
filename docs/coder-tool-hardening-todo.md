# Coder Tool Hardening — Remaining Work

Tracking doc for the silent-drop / false-success / data-loss hardening thread
(2026-07-27 → 07-28). Everything below is **uncommitted** on a shared
multi-agent tree. See memory `project_coder_silent_param_drop_classfix`.

## Done (built + verified, awaiting restart to go live)
- **shell_exec** — explicit `timeout` (pure wall-clock, ≤600), `run_in_background`
  (detach via WorkspaceServiceManager), `cwd`. `_MAX_SHELL_TIMEOUT`/`_clamp_timeout`.
- **Layer 1 (systemic)** — `coerce_params(stripped_out=…)` + `Tool._note_ignored_params`
  surfaces every stripped model-supplied arg in the ToolResult (output/error +
  metadata) instead of a debug log. Covers every tool + every future param.
- **Layer 2 (honor high-traffic)** — code_grep `case_insensitive`/`glob`,
  shell_exec `cwd`, test_run `timeout` (+ fixed latent 45s outer-cap: TestRunTool
  now overrides `Tool.timeout`).
- **file_write** — added the missing `_maybe_snapshot_before_write` (overwrites
  were invisible in review + unrestorable on Reject).
- **apply_patch** — in-band markers gate `--check` AND apply (was reporting
  "Patch applied"/success on a failed apply); `patch_path` on all failure returns;
  best-effort `.diff` cleanup on success.

## Left to do
1. **Recovery read-back (reframed B1/B3)** — model-authored tool inputs are already
   persisted untruncated in `coder_turn_events.payload` before execute; nothing
   reads them back. Build a read-back path (NOT a new staging layer) so a
   truncated/lost `content` / `edits[]` / `patch` can be recovered instead of
   asking the model to "resend the COMPLETE array". Must degrade gracefully when
   the ledger row is absent (recording is best-effort, `log.debug`).
2. **Subagent `complete_run` into the `finally`** (`augmentum/agents/dispatch.py`
   ~446-469) — a non-CancelledError exception currently loses the whole run (row
   stuck `running`, `output_text` NULL); only `run_broker.py` boot-sweep recovers.
3. **Reconcile `test_invalid_arguments_string`** (`tests/test_tool_calling_tiers.py`)
   to the `POSITIONAL_GUESS_KEY` sentinel change — it asserts the parsed dict
   `== {}` but now carries the guess marker (harmless at runtime; `coerce_params`
   pops it before execute).
4. **Restart the augmentum container** to load the `coder/tools.py` edits
   (file_write snapshot, apply_patch gating, param honoring). The `containers.py`
   file_write auto-mkdir + write-OK marker is already live.
5. **Live-verify against a real model** — tool fan-out + chain limit, and the new
   shell_exec/test_run timeout + background paths end-to-end.

## Deferred by design (Layer 1 already de-silences them)
- **file_write `append`** — the tool's contract is "write COMPLETE body"; append →
  `shell_exec` `>>`. Layer 1 now tells the model the arg was ignored.
- **publish_ports custom `port`** — recreates the container against a fixed common
  set; honoring an arbitrary port is a deeper container-layer change.
- **git scoped `paths`** — `git add -A` is fine in the isolated per-workspace repo.

## Known pre-existing failures (NOT from this thread — do not attribute)
- `test_has_expected_read_only_tools` (28 vs 23 count drift — other session's browser tools).
- 4× `_to_openai_payload() missing 'request'` (llama backend signature, other session).

## Security / hygiene before commit
- **Revoke the bench API key** used for probing (`sk-aug-…`, id `akey_…`) once
  verification concludes — must NOT be committed anywhere.
- Shared tree: commit with `git commit --only <paths>`, never bare `git commit`.
