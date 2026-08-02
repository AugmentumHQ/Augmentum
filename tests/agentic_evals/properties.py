"""Assertion helpers for agentic eval cases.

Each helper returns (bool, reason). The runner aggregates failures so a
single case can report multiple violations rather than stopping at the
first. Keep helpers pure and string-based — they run against the final
delivered text + captured artifact metadata.
"""

from __future__ import annotations

import re
from typing import Any


def has_artifact_link(text: str, *, _: Any = None) -> tuple[bool, str]:
    """True iff the response contains at least one URL or /api/artifacts path."""
    ok = bool(re.search(r"https?://|/api/artifacts/|download", text, re.IGNORECASE))
    return ok, "no artifact link found" if not ok else ""


def cites_sources(text: str, *, min_citations: int = 1) -> tuple[bool, str]:
    """True iff text has at least ``min_citations`` numeric inline citations."""
    hits = re.findall(r"\[\d+\]", text)
    ok = len(hits) >= min_citations
    return ok, f"expected >={min_citations} citations, got {len(hits)}" if not ok else ""


def length_between(text: str, *, min_chars: int = 0, max_chars: int = 10_000) -> tuple[bool, str]:
    n = len(text)
    ok = min_chars <= n <= max_chars
    return ok, f"length {n} not in [{min_chars},{max_chars}]" if not ok else ""


def mentions_all(text: str, *, keywords: list[str]) -> tuple[bool, str]:
    missing = [k for k in keywords if k.lower() not in text.lower()]
    return (not missing), f"missing keywords: {missing}" if missing else ""


def mentions_any(text: str, *, keywords: list[str]) -> tuple[bool, str]:
    hit = any(k.lower() in text.lower() for k in keywords)
    return hit, f"none of {keywords} mentioned" if not hit else ""


def does_not_mention(text: str, *, forbidden: list[str]) -> tuple[bool, str]:
    """Fails if any forbidden keyword appears — e.g., pipeline-leak phrases."""
    hit = [f for f in forbidden if f.lower() in text.lower()]
    return (not hit), f"forbidden keywords present: {hit}" if hit else ""


def is_conversational(text: str, *, _: Any = None) -> tuple[bool, str]:
    """Heuristic: avoids pipeline-style prefixes like 'Step 1:', '## Review'."""
    leaks = [
        r"^\s*step\s+\d+\s*[:.)]",
        r"^\s*##\s*(?:plan|research|draft|review|deliver)\b",
        r"work[_ ]?notes",
        r"VERDICT\s*[:=]",
    ]
    for pat in leaks:
        if re.search(pat, text, re.IGNORECASE | re.MULTILINE):
            return False, f"pipeline leak: matched /{pat}/"
    return True, ""


PROPERTY_REGISTRY = {
    "has_artifact_link": has_artifact_link,
    "cites_sources": cites_sources,
    "length_between": length_between,
    "mentions_all": mentions_all,
    "mentions_any": mentions_any,
    "does_not_mention": does_not_mention,
    "is_conversational": is_conversational,
}


def apply_assertions(text: str, assertions: list[dict]) -> list[str]:
    """Run each assertion; return list of failure reasons (empty = all pass)."""
    failures: list[str] = []
    for spec in assertions:
        name = spec["property"]
        fn = PROPERTY_REGISTRY.get(name)
        if not fn:
            failures.append(f"unknown property: {name}")
            continue
        kwargs = {k: v for k, v in spec.items() if k != "property"}
        ok, reason = fn(text, **kwargs)
        if not ok:
            failures.append(f"{name}: {reason}")
    return failures
