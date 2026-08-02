"""Shared JSON-salvage parser used pipeline-wide by bug_finder
(comprehender/investigator/lead/orchestrator/detector). Audit 2026-06-17.

The whole point: a budget-TRUNCATED model emit (no closing fence, starts
with prose, cut mid-structure) used to lose the entire result at every
stage. These pin the recovery, including the detector-findings path."""
from __future__ import annotations

import json

from augmentum.bug_finder.findings import parse_detector_output
from augmentum.bug_finder.json_salvage import (
    salvage_json_array,
    salvage_json_object,
)


def test_object_complete_fenced():
    p = salvage_json_object('x\n```json\n{"a": 1, "b": [1,2]}\n```\ny')
    assert p == {"a": 1, "b": [1, 2]}


def test_object_unfenced_trailing():
    p = salvage_json_object('here it is: {"a": 1}')
    assert p == {"a": 1}


def test_object_truncated_no_closing_fence():
    """Opening fence, no closer, cut mid-array — recover what landed."""
    full = {"brief": "see http://x/a//b", "items": [{"k": 1}, {"k": 2}, {"k": 3}]}
    s = json.dumps(full)
    cut = s.index('{"k": 3')
    out = "Final:\n```json\n" + s[:cut]
    p = salvage_json_object(out)
    assert p is not None
    assert p["brief"] == "see http://x/a//b"   # URL // not corrupted
    assert [it["k"] for it in p["items"]] == [1, 2]


def test_object_truncated_dangling_key():
    s = json.dumps({"a": 1, "subsystems": [{"name": "x"}, {"name": "y"}]})
    cut = s.index('"y"')  # truncates right after `{"name": `
    p = salvage_json_object("```json\n" + s[:cut])
    assert p is not None
    names = [x.get("name") for x in p.get("subsystems", [])]
    assert "x" in names


def test_array_salvage():
    arr = salvage_json_array('```json\n[{"a": 1}, {"a": 2}]\n```')
    assert arr == [{"a": 1}, {"a": 2}]


def test_garbage_returns_none():
    assert salvage_json_object("no json here") is None
    assert salvage_json_object("") is None
    assert salvage_json_array("nope") is None


# ── detector parser uses the salvage path ─────────────────────────────

def test_parse_detector_output_complete():
    out = '```json\n{"findings": [{"file":"a.py","function":"f","claim":"bug","severity":"high"}]}\n```'
    findings = parse_detector_output(out)
    assert len(findings) == 1
    assert findings[0].severity == "high"


def test_parse_detector_output_bare_array():
    out = '```json\n[{"file":"a.py","function":"f","claim":"bug","severity":"low"}]\n```'
    findings = parse_detector_output(out)
    assert len(findings) == 1


def test_parse_detector_output_truncated_recovers_partial():
    """A detector that emits many findings and gets cut off mid-stream
    still yields the ones that landed — instead of zero (the failure the
    audit named on large codebases)."""
    payload = {"findings": [
        {"file": "a.py", "function": "f1", "claim": "bug one", "severity": "high"},
        {"file": "b.py", "function": "f2", "claim": "bug two", "severity": "medium"},
        {"file": "c.py", "function": "f3", "claim": "bug three", "severity": "low"},
    ]}
    s = json.dumps(payload)
    cut = s.index("bug three")  # truncate inside the 3rd finding's claim
    out = "Findings:\n```json\n" + s[:cut]
    findings = parse_detector_output(out)
    assert len(findings) >= 2  # f1 + f2 recovered; f3 was mid-cut
    claims = {f.claim for f in findings}
    assert "bug one" in claims and "bug two" in claims


def test_parse_detector_output_garbage_is_empty():
    assert parse_detector_output("sorry, no findings block") == []
    assert parse_detector_output("") == []
