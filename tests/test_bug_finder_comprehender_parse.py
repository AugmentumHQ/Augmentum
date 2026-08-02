"""bug_finder comprehender JSON-salvage parser (audit 2026-06-17).

The comprehender's old parser only recovered JSON from a *closed* fenced
block (or output starting with '{'). A budget-truncated final emit has no
closing fence and starts with prose, so the whole knowledge map was lost
→ "comprehender couldn't produce parseable JSON" → zero findings. These
pin the hardened salvage path: brace-matching anywhere + truncation repair,
without corrupting JSON that contains URLs.
"""
from __future__ import annotations

import json

from augmentum.bug_finder.comprehender import (
    _last_json_payload,
    _repair_truncated,
    parse_comprehender_output,
)

_FULL = {
    "brief": "Repo summary. Docs at http://example.com/a//b — keep the slashes.",
    "subsystems": [
        {"name": "auth", "purpose": "login", "paths": ["augmentum/auth"],
         "size_files": 10, "pillars": ["user_id_scoping"]},
        {"name": "coder", "purpose": "agentic coding", "paths": ["augmentum/coder"],
         "size_files": 50, "pillars": []},
        {"name": "narrative", "purpose": "story mode", "paths": ["augmentum/narrative"],
         "size_files": 30, "pillars": []},
    ],
    "pillars": [
        {"name": "user_id_scoping",
         "statement": "every user-scoped table accepts user_id",
         "evidence": ["x.py:1"]},
    ],
    "risk_surfaces": [],
    "entry_points": [],
}


def test_complete_fenced_json():
    out = "prose\n```json\n" + json.dumps(_FULL) + "\n```\nmore prose"
    p = _last_json_payload(out)
    assert p is not None
    assert len(p["subsystems"]) == 3


def test_url_in_brief_not_corrupted():
    """`//` in a URL must survive — the repair must NOT strip comments."""
    out = "```json\n" + json.dumps(_FULL) + "\n```"
    p = _last_json_payload(out)
    assert "http://example.com/a//b" in p["brief"]


def test_unfenced_trailing_json():
    out = "Here is the map:\n" + json.dumps(_FULL)
    p = _last_json_payload(out)
    assert p is not None and len(p["pillars"]) == 1


def test_truncated_final_block_salvaged():
    """Opening fence, NO closing fence, cut mid-subsystems — the old
    parser returned None; the new one recovers brief + the subsystems
    that landed before the cut."""
    s = json.dumps(_FULL)
    cut = s.index('"narrative"')  # truncate inside the 3rd subsystem
    out = "Final map:\n```json\n" + s[:cut]
    p = parse_comprehender_output(out)
    assert p is not None
    assert p.brief.startswith("Repo summary")
    # auth + coder fully landed before the cut; narrative was truncated.
    names = [x.name for x in p.subsystems]
    assert "auth" in names and "coder" in names


def test_truncated_midstring_salvaged():
    """Cut in the middle of a string value → string gets closed, object
    + array + outer object closed; still yields a usable dict."""
    s = json.dumps(_FULL)
    cut = s.index("agentic cod")  # mid 'purpose' string of subsystem 2
    out = "```json\n" + s[:cut]
    p = _last_json_payload(out)
    assert p is not None
    assert p["brief"].startswith("Repo summary")


def test_repair_returns_none_for_balanced_input():
    # A balanced (non-truncated) string must not be "repaired" — repair
    # only fires for genuinely open structures.
    assert _repair_truncated('{"a": 1}') is None


def test_garbage_returns_none():
    assert _last_json_payload("no json here at all") is None
    assert _last_json_payload("") is None
    assert parse_comprehender_output("just prose, sorry") is None
