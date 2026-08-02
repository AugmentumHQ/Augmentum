"""Observation ledger — standalone soft-failure / repeat-call tracker.

Phase 2 / PR-2.2 of the Integrated Coding Nervous System spec.

Lifted out of :class:`augmentum.coder.state.CoderState` so the shared
:class:`augmentum.loops.LoopRunner` can own one directly without
needing a CoderState instance. The same ledger is what makes the
coder good at NOT looping — pulling it out is a precondition for
giving the agentic / App Builder surface the same hygiene.

Four buckets:

* ``recent_validation_errors`` — schema-shaped failures (caps at 3).
  Surfaced in the sticky reminder so the model sees its own pattern
  of malformed tool calls.
* ``recent_tool_failures`` — soft failures with success=False but not
  schema-related (mtime stale, missing file, missing binary). Caps at
  4, TTL 30 min, dedup key ``(tool, target)``.
* ``recent_tool_calls`` — intent-keyed fingerprint of productive
  successful calls so the model doesn't re-read the same file 5×.
  Caps at 8.
* ``background_processes`` — backgrounded shell commands so the agent
  doesn't keep spawning conflicting servers. Caps at 8.

The methods do not touch any wall-clock ``updated_at`` field — that's
the host (:class:`CoderState` or the runner) responsibility. The
ledger is single-purpose: it tracks observations and exposes the
read/write API the soft breakers consume.

Wiring from CoderState
----------------------
:class:`CoderState` retains the four lists as its own dataclass fields
and constructs the ledger via :meth:`ObservationLedger.from_lists` —
the ledger shares the list references rather than copying. Mutations
through either ``state.recent_validation_errors`` or
``state.ledger.recent_validation_errors`` see the same data, so every
existing caller and every JSON-round-trip path keeps working.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# ── Constants ──────────────────────────────────────────────────────────


FAILURE_LEDGER_TTL_SECONDS: float = 30 * 60
"""How long a soft-failure entry sticks around without a fresh
``last_at`` bump. Recurring failures keep refreshing themselves;
one-off failures fall off after 30 minutes."""


TRACKED_TOOLS_BY_PATH: frozenset[str] = frozenset({
    "file_read", "file_list", "dir_tree",
})
"""Tools whose intent key is the ``path`` argument. A second
``file_read('a.py')`` increments the existing entry's counter; a
``file_read('b.py')`` creates a new one."""


TRACKED_TOOLS_BY_COMMAND: frozenset[str] = frozenset({
    "shell_read", "shell_exec",
})
"""Shell tools — intent key is the exact command string. Truncated
to 120 chars for display, full string used as the dedup key."""


TRACKED_TOOLS_BY_QUERY: frozenset[str] = frozenset({
    "code_grep", "find_files", "code_search",
})
"""Search tools — intent key is whichever of ``pattern`` / ``query``
/ ``text`` is supplied."""


TRACKED_TOOLS_BY_URL: frozenset[str] = frozenset({
    "browser_open", "browser_snapshot", "browser_screenshot",
})
"""Browser navigation/inspection tools — intent key is the ``url``
argument when supplied, else the empty string (which still dedups
within a fixed URL — exactly the trace pattern from 2026-05-30 where
the model called ``browser_screenshot`` repeatedly on the same page
without realising it was looking at the same state)."""


TRACKED_TOOLS_BY_SELECTOR: frozenset[str] = frozenset({
    "browser_click", "browser_type", "browser_verify",
    "browser_wait", "browser_extract", "browser_fill_form",
})
"""Browser interaction tools — intent key is ``selector`` + optional
``text`` (so typing 'hello' vs 'world' into the same field counts as
distinct intents). This catches the failure mode observed in a real
trace: ~10 consecutive ``browser_click .connect-btn`` calls with no
intermediate progress. Pre-2026-05-30 these were untracked entirely
because the original frozen sets only covered file/shell/grep tools.
Wave-2 primitives (2026-07-02) join here: browser_wait keys on
selector|text like click/type; browser_extract additionally folds in
``kind`` (same selector, different extraction = distinct intent);
browser_fill_form keys on its sorted field selectors."""


TRACKED_TOOLS_BY_EXPRESSION: frozenset[str] = frozenset({
    "browser_evaluate",
})
"""Page JS evaluation — intent key is the expression itself. A model
re-evaluating the same expression is checking the same condition;
the value either changed (worth one more look) or didn't (loop)."""


TRACKED_TOOLS_BY_REQUEST: frozenset[str] = frozenset({
    "http_request",
})
"""HTTP probe — intent key is ``METHOD URL``. POST and GET to the
same URL are legitimately distinct intents; same METHOD+URL repeated
is the failure pattern."""


TRACKED_TOOLS_BY_QUERY_DB: frozenset[str] = frozenset({
    "db_inspect",
})
"""Read-only SQLite probe — intent key is the query string."""


# ── Ledger ────────────────────────────────────────────────────────────


@dataclass
class ObservationLedger:
    """Four-bucket observation ledger for the coder/agentic loop.

    Construct directly with ``ObservationLedger()`` for a fresh empty
    ledger, or via :meth:`from_lists` to share references with a host
    container (currently :class:`CoderState`).
    """

    recent_validation_errors: list[dict] = field(default_factory=list)
    recent_tool_failures: list[dict] = field(default_factory=list)
    recent_tool_calls: list[dict] = field(default_factory=list)
    background_processes: list[dict] = field(default_factory=list)

    @classmethod
    def from_lists(
        cls,
        *,
        recent_validation_errors: list[dict],
        recent_tool_failures: list[dict],
        recent_tool_calls: list[dict],
        background_processes: list[dict],
    ) -> ObservationLedger:
        """Build a ledger that shares its bucket lists with the caller.

        Used by :class:`CoderState` so ``state.recent_validation_errors``
        and ``state.ledger.recent_validation_errors`` are the same list
        object — every existing direct-mutation pattern keeps working
        while methods migrate to the ledger one at a time.
        """
        return cls(
            recent_validation_errors=recent_validation_errors,
            recent_tool_failures=recent_tool_failures,
            recent_tool_calls=recent_tool_calls,
            background_processes=background_processes,
        )

    # ── Validation errors ──────────────────────────────────────────────

    def record_validation_error(
        self, *, tool_name: str, error: str, max_kept: int = 3,
    ) -> None:
        """Remember a malformed tool call for the sticky reminder.

        Per-tool dedup: if ``tool_name`` already has an entry, bump
        its counter and refresh the message rather than push a
        duplicate row. Two counters per entry:

        * ``count`` — total per-tool failures, ignoring error content
          (sticky reminder header).
        * ``repeat_count`` — per-tool failures of the SAME error
          signature in a row; resets to 1 on signature change. The
          ``same_validation_error_repeat`` breaker watches this so
          it fires only when the model is genuinely stuck on the
          identical bad call.
        """
        short = (error or "")[:200]
        signature = short.split(".")[0][:80]
        for entry in self.recent_validation_errors:
            if entry.get("tool") == tool_name:
                entry["count"] = int(entry.get("count") or 1) + 1
                entry["error"] = short
                if entry.get("signature") == signature:
                    entry["repeat_count"] = int(entry.get("repeat_count") or 1) + 1
                else:
                    entry["signature"] = signature
                    entry["repeat_count"] = 1
                entry["last_at"] = time.time()
                return
        self.recent_validation_errors.append({
            "tool":         tool_name,
            "error":        short,
            "count":        1,
            "signature":    signature,
            "repeat_count": 1,
            "last_at":      time.time(),
        })
        if len(self.recent_validation_errors) > max_kept:
            del self.recent_validation_errors[: -max_kept]

    def clear_validation_errors(self) -> bool:
        """Drop every entry. Returns True if anything was cleared so
        the host can decide whether to bump its ``updated_at``."""
        if not self.recent_validation_errors:
            return False
        self.recent_validation_errors.clear()
        return True

    # ── Soft tool failures (cross-turn TTL) ────────────────────────────

    def record_tool_failure(
        self,
        *,
        tool_name: str,
        target: str,
        error: str,
        max_kept: int = 4,
    ) -> None:
        """Remember a soft failure (non-schema, success=False).

        Dedupe key is ``(tool_name, target)`` so different files /
        commands track independently. Prunes stale entries before
        recording, so the ledger stays fresh without a scheduler.
        """
        self.prune_stale_tool_failures()
        short = (error or "").split("\n")[0][:200]
        key_target = (target or "")[:160]
        for entry in self.recent_tool_failures:
            if entry.get("tool") == tool_name and entry.get("target") == key_target:
                entry["count"] = int(entry.get("count") or 1) + 1
                entry["error"] = short
                entry["last_at"] = time.time()
                return
        self.recent_tool_failures.append({
            "tool":    tool_name,
            "target":  key_target,
            "error":   short,
            "count":   1,
            "last_at": time.time(),
        })
        if len(self.recent_tool_failures) > max_kept:
            del self.recent_tool_failures[: -max_kept]

    def prune_stale_tool_failures(
        self, *, ttl_seconds: float | None = None,
    ) -> int:
        """Drop failure entries older than ``ttl_seconds`` since
        ``last_at``. Default TTL is
        :data:`FAILURE_LEDGER_TTL_SECONDS`. Returns the count dropped
        (for telemetry / tests)."""
        if ttl_seconds is None:
            ttl_seconds = FAILURE_LEDGER_TTL_SECONDS
        cutoff = time.time() - float(ttl_seconds)
        before = len(self.recent_tool_failures)
        # Mutate in place so any shared reference (CoderState) sees it.
        self.recent_tool_failures[:] = [
            e for e in self.recent_tool_failures
            if float(e.get("last_at") or 0) >= cutoff
        ]
        return before - len(self.recent_tool_failures)

    def clear_tool_failures(self) -> bool:
        """Reset the soft-failure tracker. Returns True if anything
        was cleared."""
        if not self.recent_tool_failures:
            return False
        self.recent_tool_failures.clear()
        return True

    # ── Successful productive calls (repeat detection) ─────────────────

    def record_tool_call(
        self,
        *,
        tool_name: str,
        tool_input: dict,
        iteration: int,
        max_kept: int = 8,
    ) -> None:
        """Remember a productive tool call so the reminder can show it.

        Intent-keyed: ``file_read(path='a.py')`` and
        ``file_read(path='b.py')`` are distinct; the former called
        three times increments a single entry's count. Tools without
        an obvious intent key (e.g. ``task_list``, ``ask_user``) are
        skipped — we only care about the "gathering information" shape.
        """
        if not isinstance(tool_input, dict):
            return
        key = _intent_key(tool_name, tool_input)
        if not key:
            return
        for entry in self.recent_tool_calls:
            if entry.get("tool") == tool_name and entry.get("key") == key:
                entry["count"] = int(entry.get("count") or 1) + 1
                entry["last_iter"] = iteration
                return
        self.recent_tool_calls.append({
            "tool":      tool_name,
            "key":       key,
            "count":     1,
            "last_iter": iteration,
        })
        if len(self.recent_tool_calls) > max_kept:
            del self.recent_tool_calls[: -max_kept]

    def hit_repeat_cap(
        self, *, tool_name: str, tool_input: dict, cap: int = 5,
    ) -> bool:
        """Has this exact ``(tool, key)`` been called ``cap`` or more
        times? Non-destructive — same counter
        :meth:`record_tool_call` populates."""
        if not isinstance(tool_input, dict):
            return False
        return self.repeat_count(
            tool_name=tool_name, tool_input=tool_input,
        ) >= cap

    def repeat_count(
        self, *, tool_name: str, tool_input: dict,
    ) -> int:
        """Return how many times this ``(tool, intent_key)`` has been
        called. Returns 0 for untracked tools or when the intent key
        can't be derived."""
        if not isinstance(tool_input, dict):
            return 0
        key = _intent_key(tool_name, tool_input)
        if not key:
            return 0
        for entry in self.recent_tool_calls:
            if entry.get("tool") == tool_name and entry.get("key") == key:
                return int(entry.get("count") or 0)
        return 0

    def clear_tool_calls_for_path(self, path: str) -> bool:
        """Drop every ``recent_tool_calls`` entry keyed on this path.

        Called after a successful mutation (file_write, code_edit,
        code_multi_edit) so subsequent re-reads of the same path
        start from count=0 — otherwise the preemptive-refusal path
        would block legitimate "verify my edit landed" calls.

        Returns True if anything was dropped so the host can decide
        whether to bump ``updated_at``.
        """
        if not path:
            return False
        before = len(self.recent_tool_calls)
        self.recent_tool_calls[:] = [
            e for e in self.recent_tool_calls
            if not (
                e.get("tool") in TRACKED_TOOLS_BY_PATH
                and e.get("key") == path
            )
        ]
        return len(self.recent_tool_calls) != before

    # ── Background processes ───────────────────────────────────────────

    def record_background_process(
        self, *, command: str, iteration: int, max_kept: int = 8,
    ) -> None:
        """Note that the agent started a backgrounded shell command.

        Exact dedup on the trimmed display string; a re-run of the
        same command increments ``count``. Truncates display to 120
        chars; the full string survives in the shell_exec
        tool_result message.
        """
        trimmed = (command or "").strip()
        if not trimmed:
            return
        display = trimmed if len(trimmed) <= 120 else trimmed[:117] + "…"
        for entry in self.background_processes:
            if entry.get("command") == display:
                entry["count"] = int(entry.get("count") or 1) + 1
                entry["last_iter"] = iteration
                return
        self.background_processes.append({
            "command":   display,
            "iteration": iteration,
            "last_iter": iteration,
            "count":     1,
        })
        if len(self.background_processes) > max_kept:
            del self.background_processes[: -max_kept]


# ── Intent-key resolver (shared by record_tool_call + repeat_count) ───


def _intent_key(tool_name: str, tool_input: dict) -> str:
    """Derive the dedup key for ``record_tool_call`` /
    ``repeat_count``. Returns ``""`` for untracked tools or empty
    arguments — caller treats that as "skip".

    New tool families (2026-05-30): browser/http/db probes were
    previously untracked, which let the model loop on
    ``browser_click('.connect-btn')`` indefinitely. The cap=5
    hard-block now fires for those too."""
    if tool_name in TRACKED_TOOLS_BY_PATH:
        return (tool_input.get("path") or "").strip()
    if tool_name in TRACKED_TOOLS_BY_COMMAND:
        return (tool_input.get("command") or "").strip()
    if tool_name in TRACKED_TOOLS_BY_QUERY:
        return str(
            tool_input.get("pattern")
            or tool_input.get("query")
            or tool_input.get("text")
            or ""
        ).strip()
    if tool_name in TRACKED_TOOLS_BY_URL:
        return (tool_input.get("url") or "").strip()
    if tool_name in TRACKED_TOOLS_BY_SELECTOR:
        # Combine selector + optional text so typing different content
        # into the same field counts as distinct intents (otherwise
        # filling a form one field at a time would all collapse to
        # one key per field). selector alone for click/verify.
        # browser_fill_form has no top-level selector — its intent is
        # the set of fields it touches; browser_extract folds in kind
        # so links-vs-table on the same selector stay distinct.
        fields = tool_input.get("fields")
        if isinstance(fields, dict) and fields:
            return "|".join(sorted(str(k).strip() for k in fields))
        sel = (tool_input.get("selector") or "").strip()
        text = (tool_input.get("text") or "").strip()
        kind = (tool_input.get("kind") or "").strip()
        parts = [p for p in (sel, kind, text) if p]
        return "|".join(parts) if len(parts) > 1 else (parts[0] if parts else "")
    if tool_name in TRACKED_TOOLS_BY_EXPRESSION:
        return (tool_input.get("expression") or "").strip()
    if tool_name in TRACKED_TOOLS_BY_REQUEST:
        method = (tool_input.get("method") or "GET").strip().upper()
        url = (tool_input.get("url") or "").strip()
        return f"{method} {url}".strip() if url else ""
    if tool_name in TRACKED_TOOLS_BY_QUERY_DB:
        return (tool_input.get("query") or "").strip()
    return ""


__all__ = [
    "FAILURE_LEDGER_TTL_SECONDS",
    "ObservationLedger",
    "TRACKED_TOOLS_BY_COMMAND",
    "TRACKED_TOOLS_BY_EXPRESSION",
    "TRACKED_TOOLS_BY_PATH",
    "TRACKED_TOOLS_BY_QUERY",
    "TRACKED_TOOLS_BY_QUERY_DB",
    "TRACKED_TOOLS_BY_REQUEST",
    "TRACKED_TOOLS_BY_SELECTOR",
    "TRACKED_TOOLS_BY_URL",
]
