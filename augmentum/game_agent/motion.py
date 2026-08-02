"""Vision-only motion heuristics — input-context evidence without RAM.

On a game with RAM probes, :class:`~augmentum.game_agent.context.
InputContextTracker` gets its evidence from decoded position facts and
``*_text`` probe activity. On a game with NO probes (the seven generic
platform profiles, any unprofiled title), that evidence never arrives
and MODE= inference degrades to fx-only guessing.

This module closes the gap from pixels alone. It compares consecutive
frame fingerprints region-by-region and classifies the change pattern:

* **screen transition** — most of the frame changed at once (fade,
  scene cut, menu opening full-screen).
* **world motion** — a broad, mid-band-inclusive change: the camera
  scrolled or the player sprite moved through the world.
* **text printing** — change concentrated in the bottom band while the
  rest of the frame holds still: a dialog/text box is typing. (The
  bottom band is where 90%+ of games put their text boxes; a game with
  top-anchored text simply misses this heuristic and still has the
  scene narrator's DIALOG label.)

These are *heuristics* — deliberately conservative, and only consulted
when RAM truth is absent (the orchestrator gates the feed). A wrong
free_move hint costs one suboptimal MODE= line; RAM probes, when they
exist, are strictly better and this module stays silent.

Pure and cheap: works on the same 32x32 RGB fingerprints the dedup pass
already computes (~1ms per compare, no extra image decode).
"""

from __future__ import annotations

from dataclasses import dataclass

from augmentum.game_agent.perception import _FP_EDGE, _fingerprint_bytes

# A fingerprint cell "changed" when any channel moved more than this
# (same scale as perception's dedup threshold — far above shimmer,
# far below real change).
_CELL_THRESHOLD = 24

# Band split (rows of the fingerprint grid): top | mid | bottom.
# The bottom band is deliberately the classic dialog-box strip
# (bottom ~30% of the screen).
_TOP_ROWS = int(_FP_EDGE * 0.35)
_BOTTOM_START = int(_FP_EDGE * 0.70)

# Classification thresholds (fractions of cells changed).
_SCREEN_CHANGE_FRAC = 0.75   # nearly everything moved → transition
_MOTION_FRAC = 0.12          # broad change → world motion / scroll
_TEXT_BOTTOM_FRAC = 0.10     # bottom band active …
_TEXT_QUIET_FRAC = 0.05      # … while top+mid stay this quiet


@dataclass
class MotionReading:
    """One classified frame-to-frame change."""

    changed_frac: float
    top_frac: float
    mid_frac: float
    bottom_frac: float
    screen_changed: bool = False
    world_motion: bool = False
    text_printing: bool = False


class MotionSense:
    """Rolling frame-pair classifier. Feed frames; read the pattern.

    Use when:
    - A frame is in hand at a periodic tick and the orchestrator wants
      input-context evidence for probe-less games.

    Returns:
    - From :meth:`feed`, a :class:`MotionReading` for this frame vs the
      previous one, or ``None`` on the first frame / decode failure /
      no change at all.
    """

    def __init__(self) -> None:
        self._last_fp: bytes | None = None

    def feed(self, png: bytes) -> MotionReading | None:
        fp = _fingerprint_bytes(png)
        if fp is None:
            return None
        prev, self._last_fp = self._last_fp, fp
        if prev is None or len(prev) != len(fp):
            return None

        # Per-cell change mask over the FP_EDGE x FP_EDGE grid.
        # Fingerprints are flat RGB: 3 bytes per cell, row-major.
        edge = _FP_EDGE
        row_changed = [0] * edge
        total = 0
        for i in range(edge * edge):
            base = i * 3
            for c in range(3):
                d = fp[base + c] - prev[base + c]
                if d > _CELL_THRESHOLD or d < -_CELL_THRESHOLD:
                    row_changed[i // edge] += 1
                    total += 1
                    break
        if total == 0:
            return None

        cells = float(edge * edge)
        top = sum(row_changed[:_TOP_ROWS])
        mid = sum(row_changed[_TOP_ROWS:_BOTTOM_START])
        bottom = sum(row_changed[_BOTTOM_START:])
        top_frac = top / (_TOP_ROWS * edge)
        mid_frac = mid / ((_BOTTOM_START - _TOP_ROWS) * edge)
        bottom_frac = bottom / ((edge - _BOTTOM_START) * edge)
        changed_frac = total / cells

        reading = MotionReading(
            changed_frac=changed_frac,
            top_frac=top_frac,
            mid_frac=mid_frac,
            bottom_frac=bottom_frac,
        )
        # Order matters: a full transition is NOT world motion (both
        # would fire on the raw fractions); text printing requires the
        # rest of the frame to hold still, so it excludes the others.
        if changed_frac >= _SCREEN_CHANGE_FRAC:
            reading.screen_changed = True
        elif (
            bottom_frac >= _TEXT_BOTTOM_FRAC
            and top_frac <= _TEXT_QUIET_FRAC
            and mid_frac <= _TEXT_QUIET_FRAC
        ):
            reading.text_printing = True
        elif changed_frac >= _MOTION_FRAC and mid_frac > 0:
            reading.world_motion = True
        return reading


__all__ = ["MotionReading", "MotionSense"]
