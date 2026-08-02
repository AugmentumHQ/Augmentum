"""ATP exposure of the coder ``pack_search`` tool — zero logic duplication.

The coder's PackSearchTool (augmentum/coder/knowledge_tools.py) never
touches its workspace plumbing — it searches via the module-level pack
manager — so this subclass just satisfies the ``_CoderTool`` constructor
with inert values and pins ATP-only surfaces. Any harness gets Augmentum's
offline DevDocs/Wikipedia/reference retrieval (the self-hosted, private
answer to Context7-style doc lookup).

``coder.tools`` is imported before ``coder.knowledge_tools`` on purpose:
knowledge_tools imports from coder.tools, and coder.tools tail-imports
knowledge_tools — loading knowledge_tools first trips that cycle.
"""

from __future__ import annotations

import augmentum.coder.tools  # noqa: F401 — establishes import order (see docstring)
from augmentum.coder.knowledge_tools import (
    PackSearchTool,
    _installed_reference_packs,
)
from augmentum.tools.base import SurfaceExposure


class AtpPackSearchTool(PackSearchTool):
    def __init__(self) -> None:
        super().__init__(container_manager=None, workspace_id="", state=None)

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(chat=False, coder=False, flow=False)

    def health_check(self) -> bool:
        # Health-gated listing: no packs installed → tool doesn't appear
        # (the coder surface instead rewrites the description, but ATP
        # clients cache the list — absence is the honest signal here).
        return bool(_installed_reference_packs())
