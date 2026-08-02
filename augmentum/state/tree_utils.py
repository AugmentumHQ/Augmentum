"""Session tree linearization utilities for dream context retrieval."""
from __future__ import annotations

from difflib import SequenceMatcher


def linearize_to_node(
    tree: dict[str, dict],
    root_id: str,
    target_id: str,
) -> list[dict] | None:
    """Walk session tree from root to target node, returning the linear ancestor path.
    Uses DFS. Returns None if target_id not found in tree.
    """
    if target_id not in tree:
        return None

    path: list[str] = []

    def _dfs(node_id: str) -> bool:
        path.append(node_id)
        if node_id == target_id:
            return True
        node = tree.get(node_id, {})
        for child_id in node.get("children", []):
            if _dfs(child_id):
                return True
        path.pop()
        return False

    if not _dfs(root_id):
        return None
    return [tree[nid] for nid in path]


def find_node_by_evidence(
    tree: dict[str, dict],
    evidence: str,
    threshold: float = 0.6,
) -> str | None:
    """Fuzzy-match evidence text against message content in tree.
    Returns the node ID of the best match above threshold, or None.
    """
    best_id: str | None = None
    best_score = 0.0
    evidence_lower = evidence.lower()

    for node_id, node in tree.items():
        content = node.get("content", "")
        if not content:
            continue
        score = SequenceMatcher(None, evidence_lower, content.lower()).ratio()
        if score > best_score:
            best_score = score
            best_id = node_id

    if best_score >= threshold:
        return best_id
    return None
