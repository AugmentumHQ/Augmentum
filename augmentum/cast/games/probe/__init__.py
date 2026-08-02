"""Phase-4 proactive probe — fingerprint a game before the first cast.

The probe loads an unknown game in a headless browser, traps which input
APIs it wires up, confirms it reacts to synthetic input, and persists a
``CastProfile`` (``classified_by='probe'``) so the FIRST cast gets the
right adapter chain instead of waiting for the telemetry demotion
round-trip.

Public surface:

  - :class:`PlaywrightProbe`        — the headless driver (playwright_probe)
  - :func:`build_probe_profile`     — ProbeResult → CastProfile
  - :func:`classify_input_style`    — observed events → input_chain (pure)
  - :class:`CastProbeCoordinator`   — fire-and-forget on first cast (job)

All strictly additive: every entry point degrades to ``None`` / no-op
when Playwright is unavailable, so a probe can make a cast better, never
break one.

See spec: ``docs/superpowers/specs/2026-06-04-universal-cast-pipeline-design.md``
"""

from __future__ import annotations

from augmentum.cast.games.probe.fingerprint import (
    InputFingerprint,
    classify_input_style,
)
from augmentum.cast.games.probe.job import CastProbeCoordinator
from augmentum.cast.games.probe.playwright_probe import (
    PlaywrightProbe,
    ProbeResult,
    build_probe_profile,
)

__all__ = [
    "InputFingerprint",
    "classify_input_style",
    "PlaywrightProbe",
    "ProbeResult",
    "build_probe_profile",
    "CastProbeCoordinator",
]
