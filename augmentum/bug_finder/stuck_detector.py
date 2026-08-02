"""Re-export shim: five-pattern semantic stuck detector.

Substrate moved to ``augmentum.agents.stuck`` in 2026-05-31. Re-exported
here so existing bug_finder import sites continue to work unchanged.
"""

from __future__ import annotations

from augmentum.agents.stuck import StuckDetector, StuckPattern, StuckResult, Turn

__all__ = ["StuckDetector", "StuckPattern", "StuckResult", "Turn"]
