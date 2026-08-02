"""Deterministic JSON serialization for signed fabric payloads.

Anything we sign or seal (contact cards P1, relay envelopes P3, author
bindings P2) must serialize to the SAME bytes on both sides or the
signature won't verify. Python's ``json.dumps`` is *almost* there but
its defaults (spaces after separators, insertion-order keys, non-ASCII
escaping) are not guaranteed stable across producers.

This is a small, dependency-free JCS-flavoured canonicaliser: sorted
keys, no insignificant whitespace, UTF-8, ``ensure_ascii=False`` so the
byte form is the natural UTF-8 encoding. It is NOT full RFC 8785 (no
number re-normalisation) — fabric payloads only carry strings, ints,
bools, and nested objects/arrays of those, which this handles
deterministically.
"""

from __future__ import annotations

import json
from typing import Any


def canonical_bytes(obj: Any) -> bytes:
    """Serialize ``obj`` to deterministic UTF-8 bytes for signing.

    Sorted keys, comma/colon separators with no spaces, no ASCII
    escaping. The same logical object always produces the same bytes.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
