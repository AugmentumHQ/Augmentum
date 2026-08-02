"""Library id-namespace helpers.

``/api/library/items`` presents a single list over two id-namespaces:

* ``artifacts`` rows — opaque hex ids.
* ``library_publications`` rows — ids prefixed ``pub_`` (see
  :func:`augmentum.library.publications._new_publication_id`).

Per-item operations (pin, tags, activity, collection membership, delete,
download, preview) must route to the right backing store by namespace.
This module is the single source of truth for that discrimination so the
prefix check never gets re-hardcoded (and drifts) across call sites.
"""

from __future__ import annotations

_PUBLICATION_ID_PREFIX = "pub_"


def is_publication_id(item_id: str | None) -> bool:
    """True when ``item_id`` addresses a ``library_publications`` row."""
    return isinstance(item_id, str) and item_id.startswith(_PUBLICATION_ID_PREFIX)
