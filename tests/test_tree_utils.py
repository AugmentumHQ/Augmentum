"""Tests for session tree linearization."""
from __future__ import annotations

from augmentum.state.tree_utils import find_node_by_evidence, linearize_to_node


def _make_tree():
    """Simple branching tree: root→A→B→C (main), B→D (branch)."""
    return {
        "root": {"id": "root", "role": "system", "content": "sys", "children": ["A"]},
        "A": {"id": "A", "role": "user", "content": "hello", "children": ["B"]},
        "B": {"id": "B", "role": "assistant", "content": "hi there", "children": ["C", "D"]},
        "C": {"id": "C", "role": "user", "content": "how are you", "children": []},
        "D": {"id": "D", "role": "user", "content": "branch msg", "children": []},
    }


def test_linearize_to_node_main_path():
    tree = _make_tree()
    path = linearize_to_node(tree, "root", "C")
    ids = [m["id"] for m in path]
    assert ids == ["root", "A", "B", "C"]


def test_linearize_to_node_branch_path():
    tree = _make_tree()
    path = linearize_to_node(tree, "root", "D")
    ids = [m["id"] for m in path]
    assert ids == ["root", "A", "B", "D"]


def test_linearize_to_node_not_found():
    tree = _make_tree()
    path = linearize_to_node(tree, "root", "MISSING")
    assert path is None


def test_find_node_by_evidence_exact():
    tree = _make_tree()
    node_id = find_node_by_evidence(tree, "hello")
    assert node_id == "A"


def test_find_node_by_evidence_fuzzy():
    tree = _make_tree()
    node_id = find_node_by_evidence(tree, "hell")
    assert node_id == "A"


def test_find_node_by_evidence_no_match():
    tree = _make_tree()
    node_id = find_node_by_evidence(tree, "completely unrelated text that matches nothing")
    assert node_id is None
