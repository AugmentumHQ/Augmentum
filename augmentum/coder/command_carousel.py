"""Command-carousel detector — windowed dedup on normalized shell re-runs.

Live failure this encodes (2026-07-07, three Qwen3.6-35B native runs on
one session building a TUI IDE: ctr_f19332da… hit the 150 hard cap,
ctr_d5daf4a8… and ctr_1506fdfc… ran 147 iterations each). Every one was
the SAME shape: the model re-ran essentially one pytest invocation dozens
of times, varying only the OUTPUT-SHAPING suffix —

    pytest ide/tests/ -v --tb=short 2>&1 | tail -40
    pytest ide/tests/ -v --tb=short 2>&1 | tail -60
    pytest ide/tests/ -v --tb=short 2>&1 | tail -100
    pytest ide/tests/ ... | grep -E '(PASSED|FAILED)'
    pytest ide/tests/ ... | grep -E '(PASSED|FAILED)' | wc -l

— interleaved with inline ``python3 -c "..."`` probes repeated 10×. It
fell through EVERY existing guard, by design:

- :class:`~augmentum.coder.duplicate_calls.DuplicateCallTracker` scopes
  itself to READ_ONLY_TOOLS and its docstring explicitly excludes shell
  / test re-runs as "legitimate repeats" — and it keys on the EXACT
  ``(tool, input)``, so ``| tail -40`` and ``| tail -60`` are different
  keys and never collide;
- :class:`~augmentum.coder.probe_signal.ProbeSignalTracker` only fires
  on BYTE-IDENTICAL output across edits — test output varies every run;
- ``action_stagnation_break`` keys on the same tool NAME for 20 iters —
  the model interleaves shell_exec / code_edit / file_read, so the name
  never stays constant;
- ``test_failure_streak`` needs a PURE failure streak — tests
  intermittently pass, resetting it.

The tell is structural and identical to the grep-carousel that motivated
``duplicate_calls``: a rotating set of near-identical commands whose
MEANINGFUL result (how many tests pass / what error) is not moving, but
whose surface form varies just enough to dodge exact-match dedup. The
normalization here strips the output-shaping pipeline (``| tail/head/
grep/wc/sed/...``) and redirections so the carousel collapses to one key,
and the "did it make progress" test is a SIGNAL hash (pytest pass/fail
counts, error signatures) rather than raw bytes — so a legitimate
red→green test loop (each run improves) is never counted, while a
re-run that establishes nothing is.

Three rungs, mirroring the read-carousel ladder (Matt, 2026-07-06:
"re-orient the model without the damage but maintaining the lesson"):

1. **nudge** — one-shot: this result is established, change the code or
   change the check.
2. **reorient** — surgical context repair: stub the duplicate shell
   RESULTS in place (first kept as ground truth, pairing intact — reuses
   ``duplicate_calls.prune_duplicate_results``) + a reorientation note.
3. **escalate** — hand the turn to the heavyweight buddy.

Plus an orthogonal **flaky-test** flag (#5): when the same normalized
command yields a DIFFERENT signal across runs with NO edit in between,
the test is non-deterministic — the model cannot stabilize it and should
note it and move on (the 150-iter run spent ~70 iterations fighting a
pytest-inside-pytest subprocess test that flapped pass↔fail).

Pure bookkeeping — no I/O, no logging. The loop owns message appends,
result pruning, meta chunks, and the model swap.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

# ── Command normalization ─────────────────────────────────────────────

# Trailing pipe stages that only RESHAPE output — stripping them exposes
# the underlying command. Anchored so a stage anywhere after the first
# such pipe (and everything downstream) is removed: `cmd | grep x | wc -l`
# → `cmd`. A LEADING grep/sort/awk (the primary command) is untouched
# because it has no preceding pipe.
_FILTER_PIPE = re.compile(
    r"\s*\|\s*(?:tail|head|grep|egrep|fgrep|wc|sed|cut|sort|uniq|awk|"
    r"tr|column|less|more|cat|xargs)\b.*$",
)
# Shell redirections — `2>&1`, `>foo`, `2>/dev/null`, `&>x`, `1>&2`.
_REDIRECT = re.compile(r"\s*(?:\d*>&?\d*|\d*>>?)\s*\S*")
_WS = re.compile(r"\s+")


def normalize_command(command: str) -> str:
    """Collapse a shell command to its carousel-identity.

    Strips the output-shaping pipe tail and redirections, then
    whitespace-normalizes. `pytest x -v 2>&1 | tail -80` and
    `pytest x -v | grep FAIL | wc -l` both reduce to `pytest x -v`.
    Deliberately conservative — it does NOT touch flags or paths, so
    genuinely different invocations stay distinct.
    """
    c = (command or "").strip()
    c = _FILTER_PIPE.sub("", c)
    c = _REDIRECT.sub(" ", c)
    return _WS.sub(" ", c).strip()


# ── Signal extraction (progress, not bytes) ───────────────────────────

_PYTEST_COUNT = re.compile(r"(\d+)\s+(passed|failed|error|errors|xfailed|xpassed)")
# Error/traceback fingerprints — the LAST exception line is the stable
# identity of a failure across runs with reflowed tracebacks.
_ERR_LINE = re.compile(r"^(?:E\s{2,}|[A-Za-z_][\w.]*(?:Error|Exception|Failure):)", re.M)


def extract_signal(output: str) -> str:
    """A hash of the MEANINGFUL outcome of a command's output.

    Prefers a structured pytest tally (passed/failed/error counts) so a
    test run that moves the needle produces a different signal from one
    that doesn't. Falls back to exception fingerprints, then to a coarse
    whitespace-normalized hash of the whole output — so a plain re-run
    with identical output still collapses to one signal.
    """
    text = output or ""
    counts = _PYTEST_COUNT.findall(text)
    if counts:
        tally = {}
        for n, kind in counts:
            tally[kind] = tally.get(kind, 0) + int(n)
        return "pytest:" + ",".join(f"{k}={tally[k]}" for k in sorted(tally))
    errs = _ERR_LINE.findall(text)
    if errs:
        joined = "|".join(errs[:5])
        return "err:" + hashlib.sha256(joined.encode("utf-8", "replace")).hexdigest()[:16]
    return "raw:" + hashlib.sha256(
        _WS.sub(" ", text.strip()).encode("utf-8", "replace")
    ).hexdigest()[:16]


def _passed_count(signal: str) -> int:
    """Tests passing in a pytest signal, or -1 for non-pytest signals."""
    if not signal.startswith("pytest:"):
        return -1
    m = re.search(r"passed=(\d+)", signal)
    return int(m.group(1)) if m else 0


def _failed_count(signal: str) -> int:
    if not signal.startswith("pytest:"):
        return -1
    m = re.search(r"(?:failed|error)s?=(\d+)", signal)
    return int(m.group(1)) if m else 0


def _is_improvement(new_sig: str, best_sig: str | None) -> bool:
    """Did ``new_sig`` make real progress over the best signal seen?

    For pytest signals: strictly more passing OR strictly fewer failing.
    For non-pytest signals: a brand-new signal (best is None or differs)
    counts as progress — the command surfaced something not seen before.
    """
    if best_sig is None:
        return True
    if new_sig.startswith("pytest:") and best_sig.startswith("pytest:"):
        return (
            _passed_count(new_sig) > _passed_count(best_sig)
            or (
                _failed_count(new_sig) >= 0
                and _failed_count(new_sig) < _failed_count(best_sig)
            )
        )
    # Non-pytest: novelty is progress; a repeat is not.
    return new_sig != best_sig


@dataclass
class _CmdRecord:
    """Per-normalized-command bookkeeping for one turn."""

    command: str
    count: int = 0
    stale_runs: int = 0                       # re-runs with no signal improvement
    tool_ids: list[str] = field(default_factory=list)
    first_output: str = ""
    best_signal: str | None = None
    last_signal: str | None = None
    last_epoch: int = -1
    nudged: bool = False
    reoriented: bool = False
    flaky_flagged: bool = False
    just_flagged_flaky: bool = False          # transient: set only on the flagging run


@dataclass
class CommandCarouselTracker:
    """Per-turn windowed detector for shell/test command carousels.

    ``observe`` records one successful, non-empty shell command and its
    output and returns one of ``"" / "nudge" / "reorient" / "escalate"``.
    The returned record's ``just_flagged_flaky`` is True on the single
    call where a non-deterministic test was first detected (orthogonal
    to the spin ladder — emit a separate one-shot nudge).

    ``note_mutations`` advances the edit epoch, exactly like
    :class:`ProbeSignalTracker`, so "did edits land between two runs of
    this command" is answerable — the axis that separates a stuck
    carousel (no edits, no signal change) from a flaky test (no edits,
    signal DID change) from legitimate iteration (edits between runs).
    """

    nudge_at: int
    reorient_margin: int
    escalate_margin: int = 3
    _epoch: int = 0
    records: dict[str, _CmdRecord] = field(default_factory=dict)
    reoriented_this_turn: bool = False

    @property
    def reorient_at(self) -> int:
        return self.nudge_at + self.reorient_margin

    def note_mutations(self, count: int) -> None:
        if count > 0:
            self._epoch += 1

    def observe(
        self, *, tool_id: str, command: str, output: str,
    ) -> tuple[str, _CmdRecord | None]:
        if self.nudge_at <= 0 or not (command or "").strip() or not (output or "").strip():
            return "", None
        key = normalize_command(command)
        if not key:
            return "", None
        rec = self.records.get(key)
        if rec is None:
            rec = _CmdRecord(command=key)
            self.records[key] = rec
        rec.count += 1
        rec.tool_ids.append(tool_id)
        rec.just_flagged_flaky = False
        if not rec.first_output:
            rec.first_output = (output or "")[:2400]

        sig = extract_signal(output)
        edits_between = self._epoch > rec.last_epoch and rec.last_epoch >= 0

        # ── Flaky-test detection (#5) ──────────────────────────────────
        # Same command, signal CHANGED, and NO edit landed in between →
        # the command is non-deterministic. One-shot flag per command.
        if (
            rec.last_signal is not None
            and sig != rec.last_signal
            and not edits_between
            and not rec.flaky_flagged
            and (sig.startswith("pytest:") or rec.last_signal.startswith("pytest:"))
        ):
            rec.flaky_flagged = True
            rec.just_flagged_flaky = True

        # ── Spin accounting ────────────────────────────────────────────
        if _is_improvement(sig, rec.best_signal):
            rec.best_signal = sig
            rec.stale_runs = 0
        else:
            rec.stale_runs += 1
        rec.last_signal = sig
        rec.last_epoch = self._epoch

        # ── Ladder (driven by stale_runs, not raw count) ───────────────
        if rec.reoriented and rec.stale_runs >= self.reorient_at + self.escalate_margin:
            return "escalate", rec
        if rec.stale_runs >= self.reorient_at and not rec.reoriented:
            rec.reoriented = True
            if self.reoriented_this_turn:
                return "escalate", rec
            self.reoriented_this_turn = True
            return "reorient", rec
        if rec.stale_runs >= self.nudge_at and not rec.nudged:
            rec.nudged = True
            return "nudge", rec
        return "", rec

    def reset(self) -> None:
        """Fresh ladder after a model handoff — the buddy gets its own
        budget rather than inheriting the looping model's stale counts."""
        self.records.clear()
        self.reoriented_this_turn = False


# ── User-facing bodies ────────────────────────────────────────────────


def carousel_nudge_body(rec: _CmdRecord) -> str:
    same = ""
    if rec.best_signal and rec.best_signal.startswith("pytest:"):
        same = f" The result has been stuck at `{rec.best_signal[7:]}` and is not moving."
    return (
        f"You have re-run `{rec.command[:160]}` {rec.count} times this "
        f"turn (varying only how the output is sliced).{same} Re-running "
        "it cannot change the outcome — the code has to change first. "
        "State in one line what the failure actually is, then either "
        "make a surgical edit that addresses THAT failure, or run a "
        "genuinely different diagnostic (read the failing source, add an "
        "assert that pinpoints the bad value). Do not run this command "
        "again to 'check'."
    )


def carousel_reorientation_body(rec: _CmdRecord) -> str:
    preview = rec.first_output[:600]
    result = (
        f"stuck at `{rec.best_signal[7:]}`"
        if rec.best_signal and rec.best_signal.startswith("pytest:")
        else "returning no new information"
    )
    return (
        "<reorientation>Your working history was just cleaned: you re-ran "
        f"`{rec.command[:160]}` {rec.count} times this turn, {result} the "
        "whole time. The redundant results were pruned from your context "
        "(the first is kept above). What that command establishes:\n"
        f"{preview}\n"
        "Re-running it is EXHAUSTED — the answer is not in running it "
        "again. Re-state the goal in one line, name the specific failure "
        "you have not yet fixed, and take a genuinely different action: "
        "read the failing code, form a hypothesis for WHY it fails, make "
        "one surgical edit, then verify. If you cannot form a hypothesis, "
        "dispatch task_dispatch(role=plan) or ask the user.</reorientation>"
    )


def flaky_test_body(rec: _CmdRecord) -> str:
    return (
        f"The command `{rec.command[:160]}` just produced a DIFFERENT "
        "pass/fail result than its previous run even though you did not "
        "edit any code in between. That test is non-deterministic "
        "(environment- or timing-dependent — e.g. a subprocess/pytest-"
        "inside-pytest test, a clock/network/ordering dependency). You "
        "cannot make a flaky test pass by re-running or rewriting it. "
        "Note it as flaky and MOVE ON to the rest of the work — do not "
        "spend further iterations trying to stabilize it. If it is "
        "central to the task, say so in your final answer and leave it "
        "for the user."
    )


__all__ = [
    "CommandCarouselTracker",
    "carousel_nudge_body",
    "carousel_reorientation_body",
    "extract_signal",
    "flaky_test_body",
    "normalize_command",
]
