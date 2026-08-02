"""Prompt + seed → result cache for image generation."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.image.persistence import ImagePersistence

log = get_logger(__name__)


def _build_cache_key(
    prompt: str,
    negative_prompt: str,
    model: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    cfg_scale: float,
) -> str:
    """Deterministic cache key from generation parameters."""
    raw = f"{prompt}|{negative_prompt}|{model}|{seed}|{width}x{height}|{steps}|{cfg_scale}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


class ImageCache:
    """In-memory + SQLite-backed cache for image generation results.

    Only caches results when a specific seed is provided (seed != -1),
    since random seeds produce different outputs.
    """

    def __init__(self, persistence: ImagePersistence | None = None) -> None:
        self._persistence = persistence
        self._memory: dict[str, str] = {}  # cache_key → image_id

    async def get(
        self,
        prompt: str,
        negative_prompt: str,
        model: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        *,
        user_id: str,
    ) -> str | None:
        """Return cached image_id if a matching generation exists, else None."""
        if seed == -1 or not user_id:
            return None

        key = _build_cache_key(
            prompt, negative_prompt, model, seed, width, height, steps, cfg_scale,
        )

        # Per-user in-memory cache key so identical prompts from different
        # users don't return each other's image_ids.
        mem_key = f"{user_id}:{key}"
        if mem_key in self._memory:
            log.debug("image_cache_hit", source="memory", key=key[:8])
            return self._memory[mem_key]

        # Fall back to DB (user-scoped)
        if self._persistence:
            image_id = await self._persistence.get_cache_entry(key, user_id=user_id)
            if image_id:
                self._memory[mem_key] = image_id
                log.debug("image_cache_hit", source="db", key=key[:8])
                return image_id

        return None

    async def put(
        self,
        prompt: str,
        negative_prompt: str,
        model: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        image_id: str,
        *,
        user_id: str,
    ) -> None:
        """Cache a generation result."""
        if seed == -1 or not user_id:
            return

        key = _build_cache_key(
            prompt, negative_prompt, model, seed, width, height, steps, cfg_scale,
        )
        self._memory[f"{user_id}:{key}"] = image_id

        if self._persistence:
            await self._persistence.save_cache_entry(key, image_id, user_id=user_id)
            log.debug("image_cache_stored", key=key[:8], image_id=image_id)
