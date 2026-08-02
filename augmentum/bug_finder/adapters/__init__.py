"""Framework adapters for the bug_finder.

Each adapter teaches the bug_finder how to read a specific framework's
structural conventions — where routes are declared, what marks a
settings file, what test runner is in use. The lead and the
investigator consume the adapter's structured outputs (routes, jobs,
entry points) as deterministic facts instead of asking an LLM to
re-derive them by reading raw files.

Adapters are auto-selected by the comprehension skeleton's framework
detection (``CodebaseSkeleton.framework``). When detection fails or
the framework isn't supported, ``generic`` falls back to language-
level heuristics.

The first concrete adapter is ``fastapi`` — Augmentum's own
framework. Adding Flask, Django, Express, etc. is the path to
running the bug_finder against any codebase.
"""

from augmentum.bug_finder.adapters.base import (
    AdapterRouteHint,
    AdapterSettingHint,
    FrameworkAdapter,
    NullAdapter,
)
from augmentum.bug_finder.adapters.fastapi import FastAPIAdapter

__all__ = [
    "AdapterRouteHint",
    "AdapterSettingHint",
    "FastAPIAdapter",
    "FrameworkAdapter",
    "NullAdapter",
    "adapter_for_framework",
]


_REGISTRY: dict[str, type[FrameworkAdapter]] = {
    "fastapi": FastAPIAdapter,
}


def adapter_for_framework(framework: str) -> FrameworkAdapter:
    """Return a concrete adapter for ``framework`` (lower-cased).

    Falls back to ``NullAdapter`` for unknown frameworks so callers
    can always invoke methods without a None check. The null adapter
    returns empty lists for every query — callers see "no routes,
    no settings" and gracefully degrade.
    """
    framework = (framework or "").strip().lower()
    cls = _REGISTRY.get(framework, NullAdapter)
    return cls()
