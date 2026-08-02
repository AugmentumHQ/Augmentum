"""Health & strain monitoring.

General-purpose server-strain sampler that complements the existing
``event_loop_stall`` watchdog: instead of one log line per stall, it writes a
durable, queryable time series (``strain_samples``) that correlates strain
against how many clients/browsers/devices were concurrently active — so
multi-browser contention can be hunted after the fact.
"""

from __future__ import annotations

from augmentum.health.strain import StrainMonitor

__all__ = ["StrainMonitor"]
