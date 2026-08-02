"""Investigator parser tests.

Pure logic. The subagent run requires a backend + container and is
exercised when the orchestrator triggers a dispatch_investigate call.
Here we cover: parse_investigator_output's tolerance to permissive
LLM output shapes, the dataclass invariants, and the candidate-filter
heuristics.
"""

from __future__ import annotations

import json

from augmentum.bug_finder.investigator import (
    InvestigatorCandidate,
    InvestigatorOutput,
    parse_investigator_output,
)


def _block(payload: dict) -> str:
    return "thinking...\n\n```json\n" + json.dumps(payload) + "\n```\n"


# ---------------------------------------------------------------------------
# parse_investigator_output
# ---------------------------------------------------------------------------


def test_parse_empty_output_returns_none() -> None:
    assert parse_investigator_output("") is None
    assert parse_investigator_output("no fenced json here") is None


def test_parse_malformed_json_returns_none() -> None:
    assert parse_investigator_output("```json\n{not valid}\n```") is None


def test_parse_missing_candidates_array_returns_none() -> None:
    """The schema requires ``candidates`` as a list. Missing → None."""
    assert parse_investigator_output(
        _block({"pattern": "x"}),
    ) is None


def test_parse_empty_candidates_returns_valid_output() -> None:
    """Explicit empty candidates is a meaningful signal ('I checked,
    no recurrence'). Lead acts on the empty list."""
    out = parse_investigator_output(_block({
        "pattern": "blanket except Exception",
        "candidates": [],
    }))
    assert out is not None
    assert out.pattern == "blanket except Exception"
    assert out.candidates == ()


def test_parse_populated_candidates() -> None:
    out = parse_investigator_output(_block({
        "pattern": "f-string SQL query",
        "candidates": [
            {
                "file": "augmentum/auth/store.py",
                "function": "find_user",
                "line_start": 42,
                "line_end": 50,
                "similar_to": "augmentum/billing/store.py:lookup_invoice",
                "confidence": "high",
                "rationale": "f-string interpolating user input into SQL",
            },
            {
                "file": "augmentum/payments/store.py",
                "function": "list_by_user",
                "line_start": 88,
                "line_end": 96,
                "similar_to": "augmentum/billing/store.py:lookup_invoice",
                "confidence": "medium",
                "rationale": "same shape, sanitization unclear",
            },
        ],
    }))
    assert out is not None
    assert len(out.candidates) == 2
    assert out.candidates[0].confidence == "high"
    assert out.candidates[0].line_start == 42
    assert out.candidates[1].confidence == "medium"


def test_parse_skips_candidates_without_file() -> None:
    """A candidate without ``file`` is unactionable — drop it."""
    out = parse_investigator_output(_block({
        "pattern": "x",
        "candidates": [
            {"file": "good.py", "function": "foo"},
            {"function": "missing-file"},        # dropped
            {"file": "", "function": "empty"},   # dropped
            {"file": "also_good.py", "function": "bar"},
        ],
    }))
    assert out is not None
    files = {c.file for c in out.candidates}
    assert files == {"good.py", "also_good.py"}


def test_parse_defaults_missing_function_to_module_marker() -> None:
    out = parse_investigator_output(_block({
        "pattern": "x",
        "candidates": [{"file": "x.py"}],
    }))
    assert out is not None
    assert out.candidates[0].function == "<module>"


def test_parse_normalizes_invalid_confidence() -> None:
    """Confidence values not in {high, medium, low} fall back to medium."""
    out = parse_investigator_output(_block({
        "pattern": "x",
        "candidates": [
            {"file": "a.py", "function": "f", "confidence": "extreme"},
            {"file": "b.py", "function": "g", "confidence": "HIGH"},
        ],
    }))
    assert out is not None
    confidences = [c.confidence for c in out.candidates]
    assert "medium" in confidences
    assert "high" in confidences


def test_parse_truncates_at_15_candidates() -> None:
    """Hard cap from the prompt; even if the model emits more we
    drop everything past 15 to keep the lead's queue bounded."""
    out = parse_investigator_output(_block({
        "pattern": "x",
        "candidates": [
            {"file": f"f{i}.py", "function": "foo"} for i in range(40)
        ],
    }))
    assert out is not None
    assert len(out.candidates) == 15


def test_parse_tolerates_bare_json_object() -> None:
    out = parse_investigator_output(json.dumps({
        "pattern": "bare", "candidates": [],
    }))
    assert out is not None
    assert out.pattern == "bare"


def test_parse_last_json_block_wins() -> None:
    raw = (
        "Draft:\n```json\n" + json.dumps({
            "pattern": "draft", "candidates": [],
        }) + "\n```\n"
        "Final:\n```json\n" + json.dumps({
            "pattern": "final", "candidates": [],
        }) + "\n```\n"
    )
    out = parse_investigator_output(raw)
    assert out is not None
    assert out.pattern == "final"


# ---------------------------------------------------------------------------
# Dataclass invariants
# ---------------------------------------------------------------------------


def test_candidate_is_frozen() -> None:
    c = InvestigatorCandidate(file="x.py", function="foo")
    try:
        c.confidence = "high"  # type: ignore[misc]
    except (AttributeError, Exception):
        return
    raise AssertionError("InvestigatorCandidate should be frozen")


def test_output_is_frozen() -> None:
    o = InvestigatorOutput(pattern="x", candidates=())
    try:
        o.pattern = "y"  # type: ignore[misc]
    except (AttributeError, Exception):
        return
    raise AssertionError("InvestigatorOutput should be frozen")


def test_candidate_defaults() -> None:
    """Default field values keep the dataclass forgiving for the LLM
    output paths that don't fill every field."""
    c = InvestigatorCandidate(file="x.py", function="foo")
    assert c.line_start == 0
    assert c.line_end == 0
    assert c.confidence == "medium"
    assert c.similar_to == ""
    assert c.rationale == ""
