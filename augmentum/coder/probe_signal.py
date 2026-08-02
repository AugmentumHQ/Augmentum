"""Always-green-probe detector — shared by the coder strategy loops.

Live failure this encodes (2026-07-06, 9B native runs ctr_1b36af6c… and
ctr_412d2740…): the model "verified" its work with a hand-rolled shell
script that only PRINTS and exits 0 — a probe that cannot fail. Every
edit→probe→edit cycle looked green, so the model kept churning on
``core/cli.py`` with no failure signal to steer by, and none of the
other detectors could see it:

- silent-success only fires on EMPTY stdout — this probe prints plenty;
- the identical-call detector needs the same call in CONSECUTIVE
  iterations — edits interleave the probe runs;
- the churn ladder catches the rewriting, not the broken feedback loop
  that sustains it.

The tell is structural: the same probe, re-run after file mutations
landed, returning byte-identical output. If edits can't move the
probe's output, the probe measures nothing — it is not verification.
This tracker hashes normalized shell output per normalized command and
counts unchanged-across-edits re-runs; the loop nudges once at
threshold, prescribing a check that can actually FAIL.

Pure bookkeeping — no I/O, no logging. The loop owns message appends
and meta chunks.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_WS = re.compile(r"\s+")


def _norm_hash(text: str) -> str:
    """Whitespace-insensitive content hash — reflowed output (progress
    padding, trailing newlines) shouldn't defeat the comparison."""
    return hashlib.sha256(_WS.sub(" ", text.strip()).encode("utf-8", "replace")).hexdigest()


@dataclass
class ProbeSignalTracker:
    """Per-turn detector for verification probes that can't fail.

    ``note_mutations`` advances a mutation epoch; ``observe_probe``
    records one successful, non-empty shell_exec. A probe whose output
    hash matches its previous run AND whose previous run happened in an
    earlier mutation epoch (i.e. edits landed in between) increments
    that command's no-signal count. When any command's count reaches
    ``nudge_at`` the observe returns ``"nudge"`` — one-shot per turn,
    because the prescription (write a check that can fail) applies to
    the model's whole verification habit, not one command.

    A changed output hash resets that command's count: the probe
    demonstrably carries signal after all.
    """

    nudge_at: int
    _epoch: int = 0
    _last: dict[str, tuple[str, int]] = field(default_factory=dict)
    """cmd_hash -> (output_hash, epoch of last run)."""
    _counts: dict[str, int] = field(default_factory=dict)
    nudge_fired: bool = False

    def note_mutations(self, count: int) -> None:
        """Record that ``count`` file mutations landed (advance epoch)."""
        if count > 0:
            self._epoch += 1

    def observe_probe(self, command: str, output: str) -> str:
        """Record one successful shell probe. Returns ``"nudge"`` when
        the always-green pattern crosses the threshold, else ``""``."""
        if self.nudge_at <= 0 or not command.strip() or not output.strip():
            return ""
        cmd_key = _norm_hash(command)
        out_hash = _norm_hash(output)
        prev = self._last.get(cmd_key)
        self._last[cmd_key] = (out_hash, self._epoch)
        if prev is None:
            return ""
        prev_hash, prev_epoch = prev
        if prev_hash != out_hash:
            self._counts[cmd_key] = 0
            return ""
        if self._epoch <= prev_epoch:
            # No edits between the two runs — a plain re-run proves
            # nothing about the probe; the identical-call detector
            # owns that shape.
            return ""
        self._counts[cmd_key] = self._counts.get(cmd_key, 0) + 1
        if self._counts[cmd_key] >= self.nudge_at and not self.nudge_fired:
            self.nudge_fired = True
            return "nudge"
        return ""

    def reset(self) -> None:
        """Fresh detector after a model handoff — the buddy gets its
        own chance to establish a real verification loop."""
        self._last.clear()
        self._counts.clear()
        self.nudge_fired = False


def probe_no_signal_nudge_body(command: str, repeats: int) -> str:
    """Prescriptive recovery text for the always-green-probe nudge."""
    cmd_preview = command.strip().replace("\n", " ")[:160]
    return (
        f"You have re-run the same probe {repeats + 1} times and its "
        "output was byte-identical every time, even though you edited "
        f"files in between (probe: `{cmd_preview}`). A probe whose "
        "output cannot change cannot verify your changes — a script "
        "that only prints and exits 0 is not verification. Replace it "
        "with a check that can FAIL:\n"
        "1. Run the project's real tests (test_run / pytest) on the "
        "code you touched.\n"
        "2. No tests? Write ONE assert-based check for the specific "
        "behavior you're fixing (a python3 -c one-liner with `assert`, "
        "or a tiny pytest file) — confirm it FAILS before your fix "
        "logic is right, then passes.\n"
        "3. At minimum, make the probe print the actual value you care "
        "about and compare it to the expected value — not a static "
        "success banner."
    )


__all__ = ["ProbeSignalTracker", "probe_no_signal_nudge_body"]
