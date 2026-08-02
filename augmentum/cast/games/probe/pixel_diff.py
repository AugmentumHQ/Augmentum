"""Canvas response confirmation — did synthetic input change the frame?

The probe captures the game canvas (or full viewport) before and after
firing synthetic input, then asks: did enough pixels change to conclude
the game *responded*? This is the PURE half — given two same-size byte
buffers (raw RGBA or any stable encoding the driver hands us), compute a
changed-pixel ratio and a boolean verdict. No browser, no PIL dependency:
the driver decodes the screenshots to raw bytes before calling in.

A response is intentionally a *low* bar — even a 1-frame flicker or a
score tick is enough to prove input landed. The default threshold is
small; the driver can raise it for noisy (animated-background) games.
"""

from __future__ import annotations

import base64
import binascii

# Default: >0.2% of sampled bytes changed counts as "the game reacted".
# Games with idle animation will trip this on their own, so the probe
# pairs it with a control window (see playwright_probe) — but as a pure
# threshold this is the floor for "something happened".
DEFAULT_DIFF_THRESHOLD = 0.002


def _decode(data: bytes | str) -> bytes:
    """Accept raw bytes or a base64 string; return raw bytes ('' on junk)."""
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=False)
        except (binascii.Error, ValueError):
            return b""
    return b""


def frame_diff_ratio(before: bytes | str, after: bytes | str) -> float:
    """Fraction of bytes that differ between two equal-length frames.

    Mismatched lengths (a resize between captures) count as a full change
    of the overlapping region plus the size delta — i.e. a strong signal
    that *something* changed, returned as a high ratio rather than an
    error. Empty inputs → 0.0 (no evidence either way).
    """
    b = _decode(before)
    a = _decode(after)
    if not b and not a:
        return 0.0
    if not b or not a:
        return 1.0

    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    changed = 0
    for i in range(n):
        if a[i] != b[i]:
            changed += 1
    # Length delta is unambiguous change too.
    size_delta = abs(len(a) - len(b))
    total = max(len(a), len(b))
    return (changed + size_delta) / total


def responded(
    before: bytes | str,
    after: bytes | str,
    *,
    threshold: float = DEFAULT_DIFF_THRESHOLD,
) -> bool:
    """True iff the after-frame differs from the before-frame by more than
    ``threshold`` (fraction of bytes). The driver's "did input land?" gate.
    """
    return frame_diff_ratio(before, after) > float(threshold)
