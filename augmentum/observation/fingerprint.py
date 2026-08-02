"""Deterministic fingerprinting for observation lookup keys.

The L0 store keys on a hash of ``(text_prefix, surface, mode)``. Storing
the hash rather than the literal tuple keeps the row narrow (no escaping
surface-name punctuation into the PK), and the deterministic hash means
two ingestion paths writing the same prefix/surface/mode pair land on
the same row and bump the count instead of duplicating.

The prefix is normalized before hashing:

  - lowercased
  - collapsed whitespace
  - stripped leading/trailing whitespace
  - capped at ``_PREFIX_CHAR_CAP`` chars (longer prefixes are rare and
    spread the count too thinly to be useful)

The hash is SHA-1 (16 hex bytes) — chosen over MD5 for it being the
common-denominator hash on every platform Python supports without
extra deps, over SHA-256 for shorter PK rows. Collision risk at our
volumes is negligible; this is not a security boundary.
"""

from __future__ import annotations

import hashlib
import re

# Anything longer than this is a paragraph, not a prefix. Truncation is
# left-aligned — we care about how the next thing follows from the
# prefix's *start*, not its tail.
_PREFIX_CHAR_CAP = 200

# Whitespace collapsing — handles tabs, newlines, multiple spaces.
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_prefix(text: str) -> str:
    """Lowercase + collapse whitespace + truncate. Used by both the
    fingerprint hash AND the corpus exporter so the cache file's text
    keys match what hashed into the store.
    """
    if not text:
        return ""
    collapsed = _WHITESPACE_RE.sub(" ", text).strip().lower()
    if len(collapsed) > _PREFIX_CHAR_CAP:
        collapsed = collapsed[:_PREFIX_CHAR_CAP]
    return collapsed


def fingerprint_prefix(
    text: str, *, surface: str = "chat", mode: str = "",
) -> str:
    """Compute the L0 fingerprint for a (prefix, surface, mode) tuple.

    Returns a 40-char hex string (SHA-1). Deterministic — the same
    inputs always produce the same output, which is the contract the
    upsert path relies on.
    """
    normalized = normalize_prefix(text)
    key = f"{surface}\x1f{mode}\x1f{normalized}".encode()
    return hashlib.sha1(key).hexdigest()
