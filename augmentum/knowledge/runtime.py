"""Process-level accessor for the live PackManager.

Knowledge packs are server-level resources (the ``knowledge_packs``
surface is NOT user-scoped — see CLAUDE.md's table list), so a single
module-level holder is the right shape. Set once by server.py at
startup after ``PackManager.scan()``; read by consumers that have no
``app.state`` in reach (the coder tool registry constructs tools far
from any request object).

Deliberately tiny: no lazy construction here — if the server didn't
wire packs (``knowledge_packs_enabled`` off), ``get_pack_manager()``
returns None and callers degrade.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from augmentum.knowledge.packs import PackManager

_pack_manager: PackManager | None = None


def set_pack_manager(manager: PackManager | None) -> None:
    global _pack_manager
    _pack_manager = manager


def get_pack_manager() -> PackManager | None:
    return _pack_manager


__all__ = ["get_pack_manager", "set_pack_manager"]
