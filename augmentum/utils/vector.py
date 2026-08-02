"""Shared vector + LLM-response helpers used by both memory and dream
compaction systems.

These are pure functions with no app-state coupling, so they can be
freely imported by either subsystem without dragging in dependencies.
Lifted from ``augmentum/memory/compactor.py`` (cosine_similarity) and
``augmentum/memory/consolidator.py`` (parse_merged_response) to avoid
duplicating the implementation between MemoryCompactor and the new
DreamCompactor — both call sites need identical semantics.
"""

from __future__ import annotations

import json


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors.

    Returns 0.0 for zero-norm inputs rather than raising — callers
    typically treat that as "unknown" and skip rather than fail.
    """
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def parse_merged_response(raw: str) -> tuple[str, float] | None:
    """Parse a JSON LLM response containing a merged/summarized statement.

    Accepts either ``{"merged": "...", "importance": 0.8}`` (consolidator
    convention) or ``{"summary": "...", "importance": 0.7}`` (compactor
    convention). Strips markdown code fences if the model wrapped its
    response in ```json ... ```.

    Returns ``(text, importance)`` on success, ``None`` when the response
    isn't valid JSON, isn't a dict, lacks a usable text field, or has a
    text field shorter than 5 chars (rejected as garbage).

    Importance is clamped to ``[0.0, 1.0]`` and defaults to 0.7 when
    missing or non-numeric.
    """
    text = raw.strip()
    # Strip ```json ... ``` style fences
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    body = str(data.get("merged") or data.get("summary") or "").strip()
    if len(body) < 5:
        return None

    try:
        importance = max(0.0, min(1.0, float(data.get("importance", 0.7))))
    except (TypeError, ValueError):
        importance = 0.7

    return body, importance
