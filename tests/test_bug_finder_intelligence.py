"""Tests for the bug_finder intelligence module — call graph queries
+ pattern memory rollups."""

from __future__ import annotations

import time

from augmentum.bug_finder.call_graph import CallGraph, CallSite
from augmentum.bug_finder.intelligence import (
    CallerInfo,
    PatternRecurrence,
    callees_of,
    is_reachable_from,
    pattern_recurrence,
    signature_familiarity_score,
    unresolved_patterns,
    who_calls,
)
from augmentum.bug_finder.patterns import Pattern


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _graph_with(edges: dict[str, list[str]], call_sites: list[CallSite] = None) -> CallGraph:
    nodes = set()
    edges_set: dict[str, set[str]] = {}
    reverse: dict[str, set[str]] = {}
    for caller, callees in edges.items():
        nodes.add(caller)
        edges_set[caller] = set(callees)
        for c in callees:
            reverse.setdefault(c.rsplit(".", 1)[-1], set()).add(caller)
            nodes.add(c)
    return CallGraph(
        nodes=nodes,
        edges=edges_set,
        reverse=reverse,
        call_sites=call_sites or [],
    )


def _pattern(
    *, signature: str, file: str = "x.py",
    hit_count: int = 1, fix_count: int = 0,
    last_seen_at: int | None = None,
    last_severity: str = "medium",
) -> Pattern:
    ts = last_seen_at if last_seen_at is not None else int(time.time())
    return Pattern(
        pattern_id="pat_test",
        user_id="u", workspace_id="w",
        claim_signature=signature, file=file,
        first_seen_at=ts, last_seen_at=ts,
        last_run_id="r",
        hit_count=hit_count, fix_count=fix_count,
        speculative_count=0,
        sample_claim="", last_severity=last_severity, note="",
    )


# ---------------------------------------------------------------------------
# who_calls
# ---------------------------------------------------------------------------


def test_who_calls_returns_deduped_callers() -> None:
    sites = [
        CallSite(caller="m.a", target="eval", file="a.py", line=10),
        CallSite(caller="m.a", target="eval", file="a.py", line=10),
        CallSite(caller="m.b", target="eval", file="b.py", line=22),
    ]
    g = _graph_with({"m.a": ["eval"], "m.b": ["eval"]}, call_sites=sites)
    out = who_calls(g, "eval")
    callers = {c.caller for c in out}
    assert callers == {"m.a", "m.b"}
    assert len(out) == 2


def test_who_calls_respects_limit() -> None:
    sites = [
        CallSite(caller=f"m{i}", target="eval", file=f"f{i}.py", line=i)
        for i in range(50)
    ]
    g = _graph_with({}, call_sites=sites)
    out = who_calls(g, "eval", limit=5)
    assert len(out) == 5


def test_who_calls_empty_when_target_unknown() -> None:
    g = _graph_with({}, call_sites=[])
    assert who_calls(g, "missing") == []


# ---------------------------------------------------------------------------
# callees_of
# ---------------------------------------------------------------------------


def test_callees_of_returns_sorted_bare_names() -> None:
    g = _graph_with({"m.foo": ["bar", "baz"]})
    out = callees_of(g, "m.foo")
    assert out == ["bar", "baz"]


# ---------------------------------------------------------------------------
# is_reachable_from
# ---------------------------------------------------------------------------


def test_reachable_from_direct_callee() -> None:
    g = _graph_with({"a": ["b"]})
    assert is_reachable_from(g, source="a", sink="b")


def test_reachable_from_indirect_within_depth() -> None:
    g = _graph_with({"a": ["b"], "b": ["c"], "c": ["d"]})
    assert is_reachable_from(g, source="a", sink="d", max_depth=4)


def test_reachable_from_not_within_depth() -> None:
    g = _graph_with({"a": ["b"], "b": ["c"], "c": ["d"]})
    assert not is_reachable_from(g, source="a", sink="d", max_depth=2)


def test_reachable_from_unreachable() -> None:
    g = _graph_with({"a": ["b"], "x": ["y"]})
    assert not is_reachable_from(g, source="a", sink="y")


def test_reachable_from_same_node_is_true() -> None:
    g = _graph_with({})
    assert is_reachable_from(g, source="a", sink="a")


def test_reachable_from_breaks_cycles() -> None:
    """A cycle in the graph shouldn't make BFS loop forever."""
    g = _graph_with({"a": ["b"], "b": ["a", "c"]})
    assert is_reachable_from(g, source="a", sink="c", max_depth=10)


# ---------------------------------------------------------------------------
# pattern_recurrence
# ---------------------------------------------------------------------------


def test_pattern_recurrence_aggregates_matched_rows() -> None:
    patterns = [
        _pattern(signature="auth_bypass", file="auth.py",
                 hit_count=3, fix_count=1, last_seen_at=100),
        _pattern(signature="auth_bypass", file="admin.py",
                 hit_count=2, fix_count=0, last_seen_at=200,
                 last_severity="high"),
        _pattern(signature="injection",
                 hit_count=5, last_seen_at=300),
    ]
    rec = pattern_recurrence(patterns, signature="auth_bypass")
    assert rec is not None
    assert rec.hit_count == 5            # 3 + 2
    assert rec.fix_count == 1
    assert rec.unresolved_count == 4     # 2 unresolved in auth.py + 2 in admin.py
    assert rec.familiar is True
    assert rec.last_severity == "high"   # winner: most recent (last_seen_at=200)


def test_pattern_recurrence_returns_none_for_unknown() -> None:
    patterns = [_pattern(signature="x")]
    assert pattern_recurrence(patterns, signature="never") is None


def test_pattern_recurrence_filters_by_file() -> None:
    patterns = [
        _pattern(signature="x", file="auth.py",  hit_count=2),
        _pattern(signature="x", file="other.py", hit_count=5),
    ]
    rec = pattern_recurrence(patterns, signature="x", file="auth")
    assert rec is not None
    assert rec.hit_count == 2   # only auth.py row counts


def test_pattern_recurrence_familiar_threshold() -> None:
    """familiar=True only when hit_count >= 2."""
    patterns = [_pattern(signature="x", hit_count=1)]
    rec = pattern_recurrence(patterns, signature="x")
    assert rec is not None
    assert rec.familiar is False


# ---------------------------------------------------------------------------
# unresolved_patterns
# ---------------------------------------------------------------------------


def test_unresolved_lists_only_unfixed_patterns() -> None:
    patterns = [
        _pattern(signature="a", hit_count=3, fix_count=0, last_seen_at=300),
        _pattern(signature="b", hit_count=2, fix_count=2, last_seen_at=200),  # fixed
        _pattern(signature="c", hit_count=1, fix_count=0, last_seen_at=100),
    ]
    out = unresolved_patterns(patterns)
    sigs = [p.claim_signature for p in out]
    assert "a" in sigs
    assert "c" in sigs
    assert "b" not in sigs


def test_unresolved_sorted_by_recency() -> None:
    patterns = [
        _pattern(signature="old", last_seen_at=10),
        _pattern(signature="new", last_seen_at=999),
        _pattern(signature="mid", last_seen_at=500),
    ]
    out = unresolved_patterns(patterns)
    assert [p.claim_signature for p in out] == ["new", "mid", "old"]


def test_unresolved_respects_limit() -> None:
    patterns = [
        _pattern(signature=f"sig_{i}", last_seen_at=i)
        for i in range(30)
    ]
    assert len(unresolved_patterns(patterns, limit=5)) == 5


# ---------------------------------------------------------------------------
# signature_familiarity_score
# ---------------------------------------------------------------------------


def test_familiarity_score_zero_for_unseen() -> None:
    assert signature_familiarity_score([], signature="x") == 0.0


def test_familiarity_score_increases_with_hits() -> None:
    one = [_pattern(signature="x", hit_count=1)]
    four = [_pattern(signature="x", hit_count=4)]
    many = [_pattern(signature="x", hit_count=20)]
    s1 = signature_familiarity_score(one, signature="x")
    s4 = signature_familiarity_score(four, signature="x")
    sm = signature_familiarity_score(many, signature="x")
    assert 0 < s1 < s4 < sm
    assert sm <= 1.0


def test_familiarity_score_caps_at_one() -> None:
    p = [_pattern(signature="x", hit_count=1000)]
    assert signature_familiarity_score(p, signature="x") == 1.0
