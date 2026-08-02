"""User-uploaded animations for the companion widget atlas.

Phase B of [[project-dance-timeline-authoritative]]. Bundled atlas
entries stay code-defined in ``ui/scripts/anim-atlas.js``; rows here
merge in alongside them at runtime so the conductor's selection
population grows with each upload.
"""
from __future__ import annotations

from augmentum.animations.store import UserAnimationStore

__all__ = ["UserAnimationStore"]
