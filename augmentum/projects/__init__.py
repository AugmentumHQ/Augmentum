"""Project entity — the canonical noun for user-owned coding work.

See docs/superpowers/specs/2026-05-29-integrated-coding-nervous-system.md.
Phase 1 / PR-1.1 introduces the substrate; later PRs route the chat,
App Builder, and Coder surfaces through it.
"""

from __future__ import annotations

from augmentum.projects.store import (
    ProjectRepoStorage,
    ProjectStore,
    SlugCollision,
)

__all__ = [
    "ProjectRepoStorage",
    "ProjectStore",
    "SlugCollision",
]
