"""Fuzzer integration for bug_finder.

Phase 1 (Python only — Atheris): chunk classifier. Future phases add the
harness writer, runner, and triage subagents. See
``docs/superpowers/specs/2026-06-02-bug-finder-fuzzer-integration-design.md``
for the full design.
"""

from augmentum.bug_finder.fuzz.classifier import (
    FuzzVerdict,
    classify_chunk,
    classify_function,
)
from augmentum.bug_finder.fuzz.harness import (
    Harness,
    generate_harness,
)
from augmentum.bug_finder.fuzz.runner import (
    FuzzCrash,
    FuzzRunResult,
    run_fuzz_harness,
)
from augmentum.bug_finder.fuzz.triage import (
    FUZZ_FAMILY,
    triage_fuzz_run,
)

__all__ = [
    "FUZZ_FAMILY",
    "FuzzCrash",
    "FuzzRunResult",
    "FuzzVerdict",
    "Harness",
    "classify_chunk",
    "classify_function",
    "generate_harness",
    "run_fuzz_harness",
    "triage_fuzz_run",
]
