"""Regression coverage for bug_finder prompts.

Tools in the registry aren't enough — the LLM has to be told WHEN to
reach for them. These tests pin that contract so a prompt refactor
can't silently drop the wiring-aware guidance and reopen the FP class
the new tools were built to close.
"""

from __future__ import annotations

from augmentum.bug_finder.investigator import INVESTIGATOR_SYSTEM_PROMPT
from augmentum.bug_finder.prompts import DETECTOR_SYSTEM_PROMPT


_DETECTOR_WIRING_TOOLS = (
    "middleware_chain",
    "decorators_on",
    "get_constant",
    "trace_origin",
    "who_calls",
    "is_reachable_from",
)


_INVESTIGATOR_WIRING_TOOLS = (
    "middleware_chain",
    "decorators_on",
    "trace_origin",
    "who_calls",
    "is_reachable_from",
)


def test_detector_prompt_names_each_wiring_tool() -> None:
    for tool_name in _DETECTOR_WIRING_TOOLS:
        assert tool_name in DETECTOR_SYSTEM_PROMPT, (
            f"detector prompt lost wiring guidance for {tool_name}"
        )


def test_detector_prompt_keeps_wiring_section_anchor() -> None:
    """The 'Wiring-aware checks' header is the anchor that ties the
    guidance together. A refactor that loses it likely loses the
    decision-point framing too."""
    assert "Wiring-aware checks" in DETECTOR_SYSTEM_PROMPT


def test_investigator_prompt_names_each_wiring_tool() -> None:
    for tool_name in _INVESTIGATOR_WIRING_TOOLS:
        assert tool_name in INVESTIGATOR_SYSTEM_PROMPT, (
            f"investigator prompt lost wiring guidance for {tool_name}"
        )
