"""Memory-save offer catalog.

The model proposes "I noticed something worth remembering — save it
to your memory?" The user accepts (or doesn't) and the fact lands
in the durable memory store as an ``EXPLICIT`` source — bypassing
the dedup-against-noise heuristics that protect against extracted
facts and going straight into ACTIVE tier with high confidence.

The catalog exposes the *kind* of memory (preference / fact /
instruction); the actual content the model wants to save comes
through ``propose_offer(extra={"content": "..."})`` and lands in
``payload['extra']['content']`` for the accept handler.

This is the right shape because:

* The model has the contextual content — what the user just said,
  what makes it worth remembering. Hard-coding strings in the
  catalog would defeat the purpose.
* The *kind* IS curated. The model can only propose against the
  three ``MemoryType`` slots the user-facing memory layer treats
  as canonical (preference / fact / instruction). Internal types
  like ``ENTITY``, ``ANALYSIS``, ``SKILL`` are not offer-able —
  those are extraction-only paths.
* EXPLICIT source means the memory's content is the user's chosen
  phrasing (here, the model's phrasing approved by the user via
  Accept). PII scrub is skipped — opt-in is the gate.

If ``payload['extra']['content']`` is missing or empty, the accept
handler refuses rather than writing an empty memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from augmentum.memory.models import MemoryType, SourceType
from augmentum.offers.catalog.base import (
    CatalogEntry,
    OfferPreview,
    register_kind,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request


log = get_logger(__name__)


KIND: str = "memory_save"


@dataclass(frozen=True)
class _MemorySpec:
    target_id: str
    memory_type: MemoryType
    label: str
    hint: str


# Three offer-able memory types. Internal types (ENTITY / ANALYSIS /
# SKILL / RELATIONSHIP) are NOT offer-able — they're extraction
# pipeline outputs, not user-facing facts.
_MEMORIES: list[_MemorySpec] = [
    _MemorySpec(
        target_id="preference",
        memory_type=MemoryType.PREFERENCE,
        label="Save as preference",
        hint=(
            "Remember this as a preference (durable opinion or stylistic "
            "choice). Surfaces across all chat sessions."
        ),
    ),
    _MemorySpec(
        target_id="fact",
        memory_type=MemoryType.FACT,
        label="Save as fact",
        hint=(
            "Remember this as a fact (objective statement about you, "
            "your work, your context). Used for grounded recall."
        ),
    ),
    _MemorySpec(
        target_id="instruction",
        memory_type=MemoryType.NARRATIVE,  # closest existing slot
        label="Save as instruction",
        hint=(
            "Remember this as an instruction or working agreement. "
            "Surfaced when later turns might violate it."
        ),
    ),
]


def _truncate(text: str, limit: int = 240) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _make_entry(spec: _MemorySpec) -> CatalogEntry:
    async def _preview(_target_id: str, _user_id: str) -> OfferPreview | None:
        # build_preview doesn't have access to the payload's extra
        # content (preview runs at propose time, content lives in the
        # extra dict alongside). The chip's hint stays generic; the
        # model's ``reason`` field (rendered separately by the UI)
        # carries the specifics like "noticed you prefer X".
        return OfferPreview(
            label=spec.label,
            hint=spec.hint,
            details={
                "scope": "user",
                "memory_type": spec.memory_type.value,
                "source_type": SourceType.EXPLICIT.value,
                "tier_target": "active",
            },
        )

    async def _accept(payload: dict[str, Any], request: "Request") -> dict[str, Any]:
        store = getattr(request.app.state, "memory_store", None)
        user = request.scope.get("user")
        user_id = getattr(user, "id", "") if user is not None else ""

        if store is None:
            return {
                "ok": False,
                "error": "no_memory_store",
                "detail": "Memory subsystem not initialized.",
            }
        if not user_id:
            return {
                "ok": False,
                "error": "no_user",
                "detail": "Memory save is per-user; no authenticated user.",
            }

        extra = payload.get("extra") or {}
        content = (extra.get("content") or "").strip()
        if not content:
            return {
                "ok": False,
                "error": "missing_content",
                "detail": (
                    "The propose_offer call must include "
                    "extra={'content': '<text to remember>'}."
                ),
            }

        # Defensive cap so a runaway extra payload doesn't bloat
        # the memory row. The store itself doesn't enforce a length;
        # 4KB is well above typical preference/fact/instruction text
        # and below anything that would feel like a paste mistake.
        if len(content) > 4096:
            return {
                "ok": False,
                "error": "content_too_long",
                "detail": "Memory content must be <= 4096 characters.",
            }

        try:
            memory_id = await store.store(
                content=content,
                memory_type=spec.memory_type,
                user_id=user_id,
                importance=0.7,            # EXPLICIT user-accepted = high
                confidence=0.95,
                source_type=SourceType.EXPLICIT,
                source_context={
                    "source": "offer",
                    "offer_target_id": spec.target_id,
                },
            )
        except Exception as exc:
            return {
                "ok": False,
                "error": "store_failed",
                "detail": str(exc)[:200],
            }

        log.info(
            "offer_memory_save_accepted",
            memory_type=spec.memory_type.value,
            memory_id=memory_id,
            user_id=user_id,
        )
        return {
            "ok": True,
            "memory_id": memory_id,
            "memory_type": spec.memory_type.value,
            "snippet": _truncate(content),
            "next_step": (
                f"Saved as {spec.memory_type.value}. Visible in Settings → "
                f"Memory; surfaces automatically in future chat turns when "
                f"relevant. Edit or forget from the memory list."
            ),
        }

    return CatalogEntry(
        kind=KIND,
        target_id=spec.target_id,
        title=spec.label + "?",
        scope="user",
        build_preview=_preview,
        accept=_accept,
    )


ENTRIES: list[CatalogEntry] = [_make_entry(s) for s in _MEMORIES]


if ENTRIES:
    register_kind(KIND, ENTRIES)
