---
name: delegate
description: >
  Delegate token-heavy generation to an Augmentum-served model
  (default deepseek-v4-flash) and review the result instead of
  generating it yourself. Use when the user says "/delegate",
  "use deepseek for this", "save tokens", "have augmentum do it",
  "panel review this", "get a second opinion on the diff",
  or when a task is generation-heavy but judgment-light: writing
  boilerplate/tests/docstrings, drafting long docs, mechanical
  refactors, converting formats, summarizing long files, or
  generating fixtures/sample data. Do NOT use for tasks where the
  hard part is judgment (architecture, subtle bug fixes, security),
  where your own edit tools are already cheaper, or when the
  Augmentum stack is down.
---

# Delegate to Augmentum

Route generation-heavy work to a local/cheap model through Augmentum's
`/v1` API. You stay the reviewer: craft a tight prompt, ship minimal
context, then verify and apply the output yourself. The economics only
work if your tokens go to REVIEW, not to re-generating.

## The tool

```bash
python scripts/claude_delegate.py "PROMPT" \
    [--file path/to/context.py]... \
    [--model d/deepseek-v4-flash] [--max-tokens 8192]
```

- Stdlib-only; auth auto-resolves from `~/.augmentum/cli.env`.
- `--file` embeds file contents as `<file path=...>` blocks (repeatable).
- Long prompts: pipe via `--stdin` instead of shell-quoting.
- Answer prints to stdout; a `[delegate: … tokens=…]` line goes to stderr.
- Exit 3 = no API key, exit 4 = Augmentum unreachable/HTTP error —
  on either, fall back to doing the task yourself and tell the user.

## Workflow (mandatory shape)

1. **Scope the ask.** One focused deliverable per call — "write pytest
   cases for `WriteChurnTracker` covering X/Y/Z", not "improve tests".
   Name the exact output format you want (unified diff, full file body,
   markdown doc) so the reply is mechanically applicable.
2. **Ship minimal context.** Pass only the files the delegate genuinely
   needs via `--file`. Never paste secrets, `.env` contents, or
   credentials into the prompt.
3. **Review, don't trust.** Read the output critically — you own the
   result. Check it against the actual codebase (imports exist, APIs
   are real, style matches). Local models hallucinate APIs more than
   you do.
4. **Apply it yourself** with your own Edit/Write tools — never ask the
   delegate to "apply" anything; it has no filesystem access.
5. **Verify** with the usual cheapest-falsifiable check (run the tests
   it wrote, ruff, import check) before reporting done.
6. **Attribute honestly.** Tell the user the draft came from the
   delegate model and what you changed in review.
7. **Record the lesson.** If review caught a defect, save it so the
   next call doesn't repeat it:

   ```bash
   python scripts/claude_delegate.py --lesson "GENERAL RULE" \
       [--for-model deepseek-v4-flash] [--kind chat|mission]
   ```

   Write the CLASS, not the instance ("every import must be exercised",
   not "don't import pytest here"). Lessons auto-inject into future
   calls as HARD REQUIREMENTS next to the task (budgeted: newest 12 /
   ~2400 chars). Ledger lives at `.claude/delegate-lessons.jsonl` —
   `--lessons-show` to audit, `--no-lessons` to opt out of injection,
   `--dry-run` to inspect the composed payload. Also record positive
   patterns when a prompt shape works unusually well. If the SAME
   lesson gets violated twice despite injection, stop delegating that
   task family and tell the user the lesson isn't landing.

## Iteration

If the output is close but flawed, ONE follow-up call quoting the
specific defect is fine. If it misses twice, stop delegating — do it
yourself; a third round-trip burns more tokens than it saves.

## Panel review (default after shipping substantial work)

After you finish a substantial change — and whenever the user asks for a
review, second opinion, or panel — fan out 2–3 chat delegates over the
artifact (the diff, the new file, the doc) with DELIBERATELY DIFFERENT
lenses. Same-lens peers agree on their hallucinations; the value is in
decorrelated verdicts.

```bash
git diff [--staged] -- <files> > /tmp/panel_diff.patch   # or copy the file
python scripts/claude_delegate.py --file /tmp/panel_diff.patch --max-tokens 2048 "LENS PROMPT"
```

Run the panelists in PARALLEL (one Bash block, multiple calls). Standard
lenses — pick 2–3 that fit the artifact:

- **Refuter**: "Your ONLY job is to REFUTE this change: find concrete
  inputs/paths where it is wrong or violates its stated contract. List
  each as input -> actual -> why wrong. If none, say NO DEFECTS FOUND.
  Do not suggest rewrites."
- **Boundary**: empty/degenerate inputs, idempotence, concurrency,
  first-use and restart behavior. PASS/FAIL per case, one-line reason.
- **Domain lens** matched to the artifact: security (injection, authz,
  user_id scoping), API-contract (does the wire shape match callers),
  regression (what existing behavior could this break).

Then YOU arbitrate: panelists will disagree — that disagreement is the
signal. Verify any claimed defect against the real code before acting on
it (panelists hallucinate defects too); concede real ones, dismiss false
ones explicitly, and report the panel's verdicts + your rulings to the
user with attribution. A finding that survives arbitration is handled
like any review finding: fix the class, and record a `--lesson` if the
panel itself misbehaved (e.g. rewrote instead of refuting).

Keep panels stateless — never feed one panelist another's output; that
recorrelates them. Synthesis is your job, not theirs.

## Mission mode (delegate a ticket, not a snippet)

For jobs that need tools and a filesystem (multi-file changes, run the
tests, iterate), queue a full headless coder turn on an Augmentum
workspace instead of a chat completion:

```bash
python scripts/claude_delegate.py --list-workspaces
python scripts/claude_delegate.py --task "PROMPT" --workspace <id> \
    --model deepseek-v4-flash --wait        # blocks, prints result + git diff
python scripts/claude_delegate.py --job <job_id>   # poll a queued mission
python scripts/claude_delegate.py --diff <workspace_id>
```

- Missions run in the WORKSPACE's container, not this repo's checkout —
  only delegate work targeting code that lives (or is checked out) in
  that workspace. Never point a mission at a workspace the user is
  actively using without asking.
- Review contract is the same: read the `--diff` output critically
  before treating the mission as done; the closeout JSON lists
  changed files and the finish reason.
- Ask the user which workspace when it isn't obvious — never pick one
  for them.

## Model choice

Default `d/deepseek-v4-flash` (the `d/` prefix is direct mode — skips
Augmentum's classifier, raw pipe to the model). For harder generation,
`d/deepseek-v4-pro` exists but is slower/costlier — surface the choice
to the user rather than auto-upgrading. `python scripts/claude_delegate.py
--help` for all knobs.
