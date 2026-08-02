"""Topical aggregator — distill observer.recent into named threads.

Sprint 2, Aletheia × Augmentum arc Piece 6.

A *thread* is a coherent cluster of surface events on the same topic.
This module reads the observer's recent deque (cross-modal activity:
``surface.browse.opened``, ``surface.media.played``, ``surface.coder.*``,
etc.) and groups events with shared domain + keyword overlap into
named threads.

Threads are the *fuel for revisit_thread* — the wondering generator
(``companion_runtime/wondering.py``) reads thread output and writes
journal entries when:

1. A thread has ≥ ``min_events`` events within ``window_seconds``
2. The events carry domain + keyword signal (not just stray clicks)
3. User isn't in hush, drives permit, daily cap allows

The aggregator itself is pure-function: no DB writes, no LLM calls.
Caching lives at the *call site* (wondering generator) to keep
this module testable.

Two grouping signals today:
- **Domain** — same host across events
- **Keyword overlap** — shared meaningful words in URLs/payloads

Future polish (deferred to Sprint 6+): embedding-based clustering for
semantic grouping when domain alone is too coarse.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ── Defaults ──────────────────────────────────────────────────────────

# Minimum events for a thread to register. Three is the spec default:
# two events are noise (random hover), three within a tight window are
# signal. Tuning surface for `companion_topical_min_events`.
DEFAULT_MIN_EVENTS: int = 3

# Time window. 4 hours captures a single afternoon's attention without
# spanning multi-day patterns (those are the dream cycle's territory).
DEFAULT_WINDOW_SECONDS: float = 4 * 3600.0

# Stopwords — common words that don't carry topical signal. Kept tight;
# the keyword extractor uses length-floor (>= 4 chars) as the main filter.
_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "on", "in", "with", "to",
    "for", "is", "this", "that", "from", "by", "at", "as", "if", "it",
    "we", "you", "are", "was", "were", "been", "have", "has", "had",
    "will", "would", "could", "should", "may", "might", "can", "did",
    "https", "http", "www", "com", "org", "net", "html", "htm",
})

# Surface-event topic prefix the observer captures. Topical aggregator
# scopes to these — chat.* events are observer's own concern.
_SURFACE_PREFIX: str = "surface."


# ── Public types ──────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class Thread:
    """A detected topical thread.

    ``topic`` is a short human-readable label (the dominant domain, or
    a top keyword when no domain is present). Used by the wondering
    generator as the entry title.

    ``event_count`` is len(event_ids); cached separately for cheap
    UI rendering.

    ``clients`` is the distinct auth-session sources (web / android /
    cast_receiver) whose events fed this thread, dominant first —
    provenance fuel for the wondering's origin record. Events written
    before the emit sites carried ``client`` count as "web".
    """
    topic: str
    event_ids: tuple[str, ...]
    domains: tuple[str, ...]
    keywords: tuple[str, ...]
    first_seen: float
    last_seen: float
    event_count: int
    clients: tuple[str, ...] = ()


# ── Helpers ───────────────────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    """Cheap domain extraction. Empty string on parse failure or
    when no hostname is present (e.g. relative URLs, malformed)."""
    if not url:
        return ""
    try:
        host = urlparse(url).hostname or ""
        # Drop common subdomain prefixes that don't add signal
        if host.startswith("www."):
            host = host[4:]
        if host.startswith("m."):
            host = host[2:]
        return host.lower()
    except Exception:
        return ""


def _extract_keywords(text: str, max_n: int = 5) -> tuple[str, ...]:
    """Extract the top ``max_n`` keywords from a text blob.

    Heuristic-only: ≥4-char alphanumeric tokens, lowercase, minus
    stopwords. Counts the most frequent and returns in descending
    frequency. Future polish: TF-IDF against a corpus, embedding-based
    keyphrase extraction.
    """
    if not text:
        return ()
    # Match word-like tokens of length 4+. Lowercase for matching.
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{3,}", text.lower())
    counter: Counter[str] = Counter(
        w for w in words if w not in _STOPWORDS
    )
    return tuple(w for w, _ in counter.most_common(max_n))


def _event_user_id(entry: dict) -> str:
    """Extract user_id from a recent-deque entry. Empty string when missing."""
    payload = entry.get("payload") or {}
    if isinstance(payload, dict):
        return str(payload.get("user_id") or "")
    return ""


def _event_client(entry: dict) -> str:
    """Auth-session source that produced the event (``payload.client``).

    Events emitted before the provenance enrichment (notes v2 Phase 1)
    carry no client field — those are treated as "web", the only client
    that existed when they were written.
    """
    payload = entry.get("payload") or {}
    if isinstance(payload, dict):
        client = str(payload.get("client") or "").strip().lower()
        if client:
            return client
    return "web"


def _event_text(entry: dict) -> str:
    """Concatenate the searchable text fields from an event payload.

    Most surface events carry a URL (browse), file_id (media/file),
    or query (search). We assemble whatever's present so the keyword
    extractor has something to work with.
    """
    payload = entry.get("payload") or {}
    if not isinstance(payload, dict):
        return ""
    parts: list[str] = []
    for key in ("url", "title", "query", "name", "label", "topic"):
        v = payload.get(key)
        if isinstance(v, str) and v:
            parts.append(v)
    return " ".join(parts)


# ── Aggregator ────────────────────────────────────────────────────────

def aggregate_threads(
    observed_recent: Any,
    *,
    user_id: str,
    min_events: int = DEFAULT_MIN_EVENTS,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    now: float | None = None,
    allowed_clients: frozenset[str] | None = None,
) -> list[Thread]:
    """Group recent surface events into topical threads.

    Iterates ``observed_recent`` (the observer's recent deque or any
    iterable of event dicts), filters to:

    - topic starting with ``surface.``
    - payload.user_id matches ``user_id``
    - timestamp within ``window_seconds`` of ``now``
    - payload.client in ``allowed_clients`` (when provided)

    ``allowed_clients`` is the parsed ``companion_attention_sources``
    setting — a shared TV (cast_receiver) logged in as you must not
    write your attention stream (2026-06-08 incident). ``None`` means
    no filtering; events without a client field count as "web".

    Groups by domain (extracted from URL payloads). Returns threads
    with ≥ ``min_events`` events. Threads are ranked by recency
    (most-recent-last-seen first).

    Pure function: no DB writes, no caching, no side effects. Callers
    cache as appropriate (the wondering generator gates its caller on
    a per-user 60s interval).
    """
    if now is None:
        now = time.time()
    cutoff = now - window_seconds

    # Filter to qualifying events
    qualifying: list[tuple[dict, float, dict]] = []
    for entry in observed_recent:
        if not isinstance(entry, dict):
            continue
        topic = entry.get("topic") or ""
        if not topic.startswith(_SURFACE_PREFIX):
            continue
        try:
            t = float(entry.get("t") or 0.0)
        except (TypeError, ValueError):
            continue
        if t < cutoff:
            continue
        if _event_user_id(entry) != user_id:
            continue
        if allowed_clients is not None and _event_client(entry) not in allowed_clients:
            continue
        payload = entry.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        qualifying.append((entry, t, payload))

    if len(qualifying) < min_events:
        return []

    # Group by domain. Events with no domain go to a "no_domain" bucket
    # but we don't form threads from them today (they'd be too noisy);
    # future polish can keyword-cluster them.
    by_domain: dict[str, list[tuple[dict, float, dict]]] = {}
    for entry, t, payload in qualifying:
        url = str(payload.get("url") or "")
        domain = _extract_domain(url) if url else ""
        if not domain:
            continue
        by_domain.setdefault(domain, []).append((entry, t, payload))

    threads: list[Thread] = []
    for domain, group in by_domain.items():
        if len(group) < min_events:
            continue
        # Build the keyword list from concatenated text across events
        all_text = " ".join(_event_text(e) for e, _, _ in group)
        keywords = _extract_keywords(all_text)
        times = [t for _, t, _ in group]
        # Stable event_ids — use the timestamp string (events don't
        # always carry an explicit id; ts is unique enough at this scale).
        event_ids = tuple(str(e.get("t") or "") for e, _, _ in group)
        client_counter: Counter[str] = Counter(
            _event_client(e) for e, _, _ in group
        )
        threads.append(Thread(
            topic=domain,
            event_ids=event_ids,
            domains=(domain,),
            keywords=keywords,
            first_seen=min(times),
            last_seen=max(times),
            event_count=len(group),
            clients=tuple(c for c, _ in client_counter.most_common()),
        ))

    # Rank by recency (most recently active threads first)
    threads.sort(key=lambda t: t.last_seen, reverse=True)
    return threads


__all__ = [
    "Thread",
    "aggregate_threads",
    "DEFAULT_MIN_EVENTS",
    "DEFAULT_WINDOW_SECONDS",
]
