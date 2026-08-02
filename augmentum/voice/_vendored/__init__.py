"""Vendored override modules for upstream packages we patch.

Each subdirectory here pins a specific upstream commit + ships a small
override module that adds the behavior Augmentum needs. See per-package
VENDOR.md for the strategy + upgrade procedure.

Pinned vendors:
  - pocket_tts: Mimi codec token tap for presence audio history (Phase 2+)
"""

from __future__ import annotations
