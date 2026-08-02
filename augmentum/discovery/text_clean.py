"""Shared text cleanup for outbound search-engine queries.

Feed signal titles (YouTube videos, news headlines, RSS items) arrive with
HTML entities (``&#x27;``), raw tags (``<span lang="en" dir="ltr">``), and
trailing decorator chars (``| Channel Name``, `` - Source``). Searxng treats
these as malformed and returns 400 — observed at 3,351/24h before this
landed.

Applied at two boundaries:
  1. Signal ingestion -> cluster name  (prevention; clustering.py)
  2. Cluster name -> searxng query     (bandaid for legacy dirty rows;
     recommender.py)
"""
from __future__ import annotations

import html
import re

_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Truncated/unterminated tags (cluster names are clipped to 55 chars in
# _generate_cluster_name, which sometimes lands mid-tag like `<span lang=`).
# Strip everything from a stray `<` to end of string after the tag pass.
_ORPHAN_LT_RE = re.compile(r"<[^>]*$")
_WS_RE = re.compile(r"\s+")
_TRAILING_JUNK_RE = re.compile(r"[\s|\-—–:•·]+$")
_LEADING_JUNK_RE = re.compile(r"^[\s|\-—–:•·]+")


def clean_text_for_query(text: str, *, max_chars: int = 120) -> str:
    """Make free-form text safe to send as a search engine query.

    Returns empty string when the input is empty OR when cleanup leaves
    nothing meaningful behind (e.g. pure HTML junk). Callers must check
    for empty and skip the query in that case — sending an empty query
    produces a "modifier " stub that's worse than no query at all.
    """
    if not text:
        return ""
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _ORPHAN_LT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    text = _LEADING_JUNK_RE.sub("", text)
    text = _TRAILING_JUNK_RE.sub("", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    return text
