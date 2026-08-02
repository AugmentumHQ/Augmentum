"""Overridable prompt registry — the bridge that lets Evolve *change* the app.

An evolved prompt is only useful if the app actually uses it. A prompt the app
reads from an override-or-default is a real **reshape surface**: applying the
evolved text writes a per-user override (verified by read-back, reversible by
clearing it, archived in the lineage) — exactly the Adaptation path, so it reuses
all of it. A prompt hardcoded in Python is a *code* edit instead (the worktree
driver), out of scope here.

A prompt site opts in by registering its key + default and reading
``resolved_prompt(key)`` instead of a constant. The override lives in the main
settings store (it IS the intended change — like a config Adaptation), scoped to
the user; the *record* of the change lands in the isolated growth archive.

Safety: only registered keys are overridable, and the registry holds the
canonical default so clearing the override always restores known-good behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Settings-key prefix for prompt overrides (namespaced so they never collide with
# other settings; the reshape/config surface writes the same key).
KEY_PREFIX = "selfedit_prompt:"


@dataclass
class PromptSpec:
    key: str                 # full settings key (KEY_PREFIX + slug)
    slug: str                # short id used in the UI/URL
    default: str             # the canonical prompt — restored when the override clears
    label: str = ""
    description: str = ""
    user_facing: bool = False  # True = changes user-visible behavior (companion/builder)

    def to_dict(self, *, effective: str = "") -> dict:
        return {
            "slug": self.slug, "key": self.key, "label": self.label,
            "description": self.description, "user_facing": self.user_facing,
            "default": self.default, "effective": effective or self.default,
            "overridden": bool(effective and effective != self.default),
        }


_REGISTRY: dict[str, PromptSpec] = {}


def register_prompt(slug: str, default: str, *, label: str = "", description: str = "",
                    user_facing: bool = False) -> PromptSpec:
    """Register an overridable prompt. Re-registering the same slug updates the
    default (so the canonical text always tracks the code). Returns the spec."""
    spec = PromptSpec(key=KEY_PREFIX + slug, slug=slug, default=default,
                      label=label or slug, description=description, user_facing=user_facing)
    _REGISTRY[slug] = spec
    return spec


def get_spec(slug: str) -> PromptSpec | None:
    return _REGISTRY.get(slug)


def spec_for_key(key: str) -> PromptSpec | None:
    return next((s for s in _REGISTRY.values() if s.key == key), None)


def registered_prompts() -> dict[str, PromptSpec]:
    return dict(_REGISTRY)


def clear_registry() -> None:  # for tests
    _REGISTRY.clear()


async def resolved_prompt(slug: str, *, settings_store: Any, user_id: str,
                          default: str = "") -> str:
    """The effective prompt for ``slug``: the user's override if set, else the
    registered default (or the passed ``default`` if the slug isn't registered).
    Never raises — a store hiccup falls back to the default so a prompt site can't
    break on the override layer."""
    spec = _REGISTRY.get(slug)
    fallback = (spec.default if spec else default) or default
    if settings_store is None or not user_id:
        return fallback
    try:
        val = await settings_store.get_user(user_id, KEY_PREFIX + slug)
    except Exception as exc:  # noqa: BLE001 — the override layer must never break a prompt site
        log.warning("resolved_prompt_lookup_failed", slug=slug, error=repr(exc))
        return fallback
    return val if (val and str(val).strip()) else fallback
