"""Offer substrate — chat-LLM-emitted proposals rendered as inline chips.

See ``docs/superpowers/specs/2026-06-02-offer-substrate-design.md``.

The substrate rides on the existing notification primitives:

* Active offers are rows in ``notifications`` on channel
  ``system.offer`` (catalog entry in ``augmentum/notifications/catalog.py``).
* The accept callback runs through the existing notification action
  registry (``augmentum/notifications/actions.py``) — one handler
  pattern (``system.offer``) dispatches to per-kind catalog accept
  handlers.

What this package adds:

* ``store`` — CRUD over the ``offer_suppressions`` table (migration 224).
* ``catalog`` — per-kind catalogs of offer-able targets, each with a
  ``build_preview`` + ``accept`` pair.
* ``dispatcher`` — the entry point the ``propose_offer`` tool calls:
  catalog lookup → suppression check → rate-limit → publish.
* ``handlers`` — the action-callback dispatcher (registered against
  the ``system.offer`` channel pattern).
"""

from __future__ import annotations

from .catalog.base import (
    CATALOG,
    CatalogEntry,
    OfferPreview,
    get_entry,
    list_kinds,
    register_kind,
)
from .dispatcher import (
    PROPOSE_OFFER_RESULT_KEYS,
    ProposeOfferResult,
    propose_offer,
)
from .store import (
    SUPPRESSION_NEVER,
    OfferSuppression,
    delete_suppression,
    is_suppressed,
    list_suppressions,
    set_suppression,
    sweep_expired_suppressions,
)

__all__ = [
    "CATALOG",
    "CatalogEntry",
    "OfferPreview",
    "PROPOSE_OFFER_RESULT_KEYS",
    "ProposeOfferResult",
    "SUPPRESSION_NEVER",
    "OfferSuppression",
    "delete_suppression",
    "get_entry",
    "is_suppressed",
    "list_kinds",
    "list_suppressions",
    "propose_offer",
    "register_kind",
    "set_suppression",
    "sweep_expired_suppressions",
]
