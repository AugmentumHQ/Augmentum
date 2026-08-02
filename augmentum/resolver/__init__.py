"""Reference Resolver — natural-language → ranked moments.

Resolves descriptive references across every surface that has touched
the user's attention — file_index entries, journal moments, future
session lookups — and returns ranked candidates. Domain-agnostic by
design: the same code path serves a request for a document, an image,
a media file, or a journal entry. The retrieval substrate doesn't
know the difference.

Public surface is intentionally small:

* :func:`resolve_moments` — the one entry point. Takes a natural-
  language query and returns ranked, structured moments.

Everything else is implementation detail.
"""

from __future__ import annotations

from augmentum.resolver.core import Moment, resolve_moments

__all__ = ["Moment", "resolve_moments"]
