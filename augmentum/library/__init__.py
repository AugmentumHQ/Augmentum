"""Save-to-Library subsystem.

One-way snapshots of coder-built artifacts into a per-user catalog that
survives workspace teardown. The preview pane in coder mode anchors the
v1 UX — if the user sees the artifact running, one click saves it.

See ``docs/superpowers/specs/2026-05-26-publish-to-library-design.md``
for the full design.

Module layout
-------------
``publications`` — :class:`PublicationStore` (CRUD on
``library_publications``) + :class:`LibraryStorage` (per-publication
storage dirs, atomic overwrite, screenshot write, meta.json mirror).

``preview_kind`` — Static-vs-dynamic preview classifier used by the
save route to gate the save button at preflight time.
"""

from __future__ import annotations

from augmentum.library.activity import ActivityStore
from augmentum.library.collections import (
    CollectionStore,
    DynamicFilter,
    SlugCollision,
)
from augmentum.library.home import build_home_payload
from augmentum.library.publications import (
    LibraryStorage,
    PublicationStore,
    SizeBudgetExceeded,
    TitleCollision,
)

__all__ = [
    "ActivityStore",
    "CollectionStore",
    "DynamicFilter",
    "LibraryStorage",
    "PublicationStore",
    "SizeBudgetExceeded",
    "SlugCollision",
    "TitleCollision",
    "build_home_payload",
]
