"""Duplicate read-call tracker + in-place reorientation — coder loop rung.

Live failure this encodes (2026-07-04, deepseek-v4-flash native run
ctr_21c9e2a7…, cancelled by the user at 146 tool calls): 101 of the
146 calls were ``code_grep``, cycling the SAME (pattern, path) pairs —
``uOutlineThickness`` against the same three shader files, over and
over with period ~4. Every existing guard was blind to the shape:

- the identical-call detector needs the same call in CONSECUTIVE
  iterations — a rotating cycle of 3-4 distinct calls never repeats
  back-to-back;
- ``inspection_loop_nudge`` counts read-ONLY iterations — the loop
  interleaved occasional edits, resetting the streak every time;
- the churn ladder watches writes; the probe tracker watches shell
  probes. Nobody watched reads for windowed duplication.

The deeper problem is that a context full of repeated identical calls
is SELF-SUSTAINING: 20 greps in history are few-shot pressure to emit
the 21st, and the duplicated results bloat the window, compacting away
the context that would have broken the loop. So the middle rung here
is not a break — it is REORIENTATION (Matt, 2026-07-06: "instead of
canceling, re-orient the model without the damage but maintaining the
lesson"):

1. **nudge** — one-shot prescriptive message, same as the other
   ladders: this result is established, don't re-run it.
2. **reorient** — surgical context repair: stub out every duplicate
   tool RESULT beyond the first (pairing stays intact — the assistant
   ``tool_calls`` entries and tool_call_ids are untouched, only the
   result content is replaced), then append a reorientation note that
   preserves the lesson: what was run, how many times, what it
   returned, and that the line of inquiry is exhausted. The model
   keeps its goal and its ground truth, loses the self-imitation
   pressure and the token bloat.
3. **escalate** — the same key advancing AFTER a reorient means the
   model demonstrably can't steer out even with a clean window; hand
   the turn to the heavyweight buddy (loop-side, reusing the
   write-churn escalation handoff). No buddy configured → the loop
   falls through to its existing backstops; this ladder never kills
   the turn itself.

Pure bookkeeping + pure list surgery — no I/O, no logging. The loop
owns message appends, meta chunks, and the model swap.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

# Result-content stub written over pruned duplicate results. Kept short
# and self-explaining — the model reads these in place of the dead
# weight, and the reorientation note carries the surviving information.
PRUNED_STUB = (
    "[pruned: duplicate call — this exact tool call was run earlier "
    "with an identical result; see the first occurrence above and the "
    "reorientation note below]"
)

_PREVIEW_CHARS = 600


def _key(tool: str, tool_input: dict | None) -> str:
    """Canonical (tool, input) identity — order-insensitive JSON."""
    try:
        canon = json.dumps(tool_input or {}, sort_keys=True, default=str)
    except Exception:
        canon = str(tool_input)
    return hashlib.sha256(f"{tool}\x00{canon}".encode("utf-8", "replace")).hexdigest()


def _out_hash(output: str) -> str:
    return hashlib.sha256(" ".join((output or "").split()).encode("utf-8", "replace")).hexdigest()


@dataclass
class _KeyRecord:
    tool: str
    input_summary: str
    count: int = 0
    tool_ids: list[str] = field(default_factory=list)
    first_output: str = ""
    output_hashes: set[str] = field(default_factory=set)
    nudged: bool = False
    reoriented: bool = False


@dataclass
class DuplicateCallTracker:
    """Per-turn windowed duplicate detector for read-shaped tools.

    ``observe`` returns one of ``""`` / ``"nudge"`` / ``"reorient"`` /
    ``"escalate"`` — at most one action per key per rung, at most one
    reorient per turn (a second key reaching the reorient rung after a
    reorient already happened escalates instead: the window was already
    cleaned once and the model is still looping).

    ``tracked_tools`` scopes the detector to read/introspection tools.
    Mutating tools have the churn ladder; shell probes have the
    probe-signal tracker; test re-runs (red→green) are legitimate
    repeats and must never be counted here.
    """

    nudge_at: int
    reorient_margin: int
    tracked_tools: frozenset[str]
    escalate_margin: int = 3
    records: dict[str, _KeyRecord] = field(default_factory=dict)
    reoriented_this_turn: bool = False

    def observe(
        self,
        *,
        tool_id: str,
        tool: str,
        tool_input: dict | None,
        output: str,
    ) -> tuple[str, _KeyRecord | None]:
        """Record one successful call; return (action, record)."""
        if tool not in self.tracked_tools:
            return "", None
        key = _key(tool, tool_input)
        rec = self.records.get(key)
        if rec is None:
            summary = json.dumps(tool_input or {}, sort_keys=True, default=str)
            rec = _KeyRecord(tool=tool, input_summary=summary[:300])
            self.records[key] = rec
        rec.count += 1
        rec.tool_ids.append(tool_id)
        if not rec.first_output:
            rec.first_output = (output or "")[:_PREVIEW_CHARS * 4]
        rec.output_hashes.add(_out_hash(output))

        if rec.reoriented and rec.count >= self.reorient_at + self.escalate_margin:
            return "escalate", rec
        if rec.count >= self.reorient_at and not rec.reoriented:
            rec.reoriented = True
            if self.reoriented_this_turn:
                # Window was already repaired once this turn and a
                # DIFFERENT call is now looping too — self-correction
                # isn't happening; confirm the loop upward.
                return "escalate", rec
            self.reoriented_this_turn = True
            return "reorient", rec
        if rec.count >= self.nudge_at and not rec.nudged:
            rec.nudged = True
            return "nudge", rec
        return "", rec

    @property
    def reorient_at(self) -> int:
        return self.nudge_at + self.reorient_margin


def duplicate_nudge_body(rec: _KeyRecord) -> str:
    same = " — with the identical result every time" if len(rec.output_hashes) == 1 else ""
    return (
        f"You have now run `{rec.tool}` with the exact same input "
        f"{rec.count} times this turn{same}. That information is "
        "established; re-running the call cannot produce anything new. "
        "State in one line what you are actually trying to find out, "
        "then get it a DIFFERENT way: a different pattern or path, a "
        "broader read, or a different tool entirely."
    )


def reorientation_body(rec: _KeyRecord) -> str:
    """The lesson that replaces the pruned repetition.

    Written to keep the GROUND TRUTH (first result preview) and the
    meta-lesson (this line of inquiry is exhausted) while the pruned
    stubs remove the few-shot pressure to repeat.
    """
    sameness = (
        "the result was IDENTICAL every single time"
        if len(rec.output_hashes) == 1
        else "the results were near-identical"
    )
    preview = rec.first_output[:_PREVIEW_CHARS]
    return (
        "<reorientation>Your working history was just cleaned: you ran "
        f"`{rec.tool}` with input {rec.input_summary} {rec.count} times "
        f"this turn, and {sameness}. The duplicate results were pruned "
        "from your context (the first occurrence is kept above). What "
        "that call establishes, once and for all:\n"
        f"{preview}\n"
        "This line of inquiry is EXHAUSTED — the answer you are looking "
        "for is not in that call's output, or you already have it. Do "
        "not run it again. Re-state your goal in one line, say what "
        "information is actually missing, and take a genuinely "
        "different next action (different file, different pattern, a "
        "broader search, reading the caller instead of the definition, "
        "or asking the user).</reorientation>"
    )


def prune_duplicate_results(messages: list, rec: _KeyRecord) -> int:
    """Stub out duplicate tool RESULTS in-place; keep the first.

    Only ``role="tool"`` result content is replaced — assistant
    ``tool_calls`` and ``tool_call_id`` pairing are untouched, so the
    history stays schema-valid for strict providers (the DeepSeek
    pairing class). Returns the number of results stubbed.
    """
    keep = set(rec.tool_ids[:1])
    prune = set(rec.tool_ids[1:])
    stubbed = 0
    for i, msg in enumerate(messages):
        if getattr(msg, "role", "") != "tool":
            continue
        tcid = getattr(msg, "tool_call_id", "") or ""
        if tcid in prune and tcid not in keep:
            content = getattr(msg, "content", "") or ""
            if content == PRUNED_STUB:
                continue
            try:
                messages[i] = type(msg)(
                    role="tool", content=PRUNED_STUB, tool_call_id=tcid,
                )
            except Exception:
                # Message type with a different constructor — mutate as
                # a fallback rather than fail the repair.
                try:
                    msg.content = PRUNED_STUB
                except Exception:
                    continue
            stubbed += 1
    return stubbed
