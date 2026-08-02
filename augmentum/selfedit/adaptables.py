"""Adaptable-settings catalog — the discoverable, self-extending list for Adapt.

Law 0 (derive, don't duplicate): instead of a hand-maintained list, the catalog
is **derived from the app's own per-user settings registry** (``_UI_SETTINGS``,
stored under ``ui.<key>`` and read via ``get_user_or_global``). As the app grows
new settings, they appear in Adapt automatically — curated metadata (label / type
/ options) sharpens the known ones, and the rest are inferred and safely shown.

A denylist keeps JSON blobs / internal state out; everything written goes through
the same reshape path (verified by read-back, reversible, archived).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

SETTING_PREFIX = "ui."

# Curated metadata for known keys. Keys NOT here are auto-derived as the app grows.
METADATA: dict[str, dict] = {
    "aiName": {"label": "Assistant name", "type": "text",
               "description": "what the assistant calls itself"},
    "responseStyle": {"label": "Response style", "type": "text",
                      "description": "a short style steer, e.g. concise / balanced / detailed"},
    "voiceAutoRead": {"label": "Auto-read replies aloud", "type": "bool"},
    "voiceSpeed": {"label": "Voice speed", "type": "number",
                   "description": "playback rate, e.g. 0.5–2.0", "default": "1.0"},
    "thinkEnabled": {"label": "Show thinking", "type": "bool"},
    "preserveThinking": {"label": "Keep thinking in history", "type": "bool"},
    "autoTools": {"label": "Automatic tool use", "type": "bool"},
    "softTypography": {"label": "Soft typography (calmer chrome labels)", "type": "bool"},
    "ttsIncludeActionText": {"label": "Read action text aloud", "type": "bool"},
    "dreamEnabled": {"label": "Dreams", "type": "bool"},
    "connectEnabled": {"label": "Connect (peer-to-peer)", "type": "bool"},
    "browseReaderJustify": {"label": "Justify reader text", "type": "bool"},
}

# Never surface: JSON blobs, long free-form, or internal state (not user-tunable).
DENYLIST: frozenset[str] = frozenset({
    "systemPrompt", "aiInstructions", "personalityPresets", "typographyCustomFonts",
    "typographyTextColors", "recentModels", "engineModelLoadProfiles", "workspace",
    "xrSeatLayout", "gameStreamInputPrefs", "mediaRailsHidden", "orbCustomOrder",
    "orbCustomColors", "stopSequences", "onboarding_completed", "typographyPreset",
})

# Name fragments that imply a numeric setting (checked before the bool heuristic).
_NUMBER_HINTS = ("temperature", "topp", "maxtokens", "speed", "size", "height", "width",
                 "threshold", "minutes", "days", "limit", "tokens", "count")


def _humanize(key: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", " ", key).replace("_", " ").strip()
    return s[:1].upper() + s[1:] if s else key


def _infer_type(key: str, maxlen: int) -> str:
    low = key.lower()
    if any(h in low for h in _NUMBER_HINTS):
        return "number"
    if maxlen <= 8:                 # the app stores bools as "true"/"false"
        return "bool"
    return "text"


@dataclass
class Adaptable:
    key: str                          # camelCase UI key (e.g. "voiceAutoRead")
    label: str
    type: str                         # bool | number | text | choice
    description: str = ""
    options: list[str] = field(default_factory=list)
    default: str = ""
    curated: bool = False             # True = has hand-written metadata

    @property
    def settings_key(self) -> str:
        return SETTING_PREFIX + self.key

    def to_dict(self, *, value: Any = None) -> dict:
        cur = "" if value is None else str(value)
        return {
            "key": self.key, "settings_key": self.settings_key, "label": self.label,
            "type": self.type, "description": self.description, "options": self.options,
            "default": self.default, "curated": self.curated,
            "value": cur or self.default,
            "is_set": value is not None and str(value) != "",
        }


def _build(key: str, maxlen: int) -> Adaptable:
    md = METADATA.get(key, {})
    return Adaptable(
        key=key, label=md.get("label") or _humanize(key),
        type=md.get("type") or _infer_type(key, maxlen),
        description=md.get("description", ""), options=md.get("options", []),
        default=md.get("default", ""), curated=key in METADATA)


def derive(ui_settings: dict[str, int] | None = None) -> list[Adaptable]:
    """The adaptables catalog. With ``ui_settings`` (the app's ``_UI_SETTINGS``),
    derive ALL non-denylisted settings — so the list auto-extends as the app grows;
    without it, fall back to the curated set. Curated first, then the rest, by label."""
    keys = list(ui_settings.keys()) if ui_settings else list(METADATA.keys())
    items = [_build(k, (ui_settings or {}).get(k, 256)) for k in keys if k not in DENYLIST]
    items.sort(key=lambda a: (not a.curated, a.label.lower()))
    return items


# Curated-only catalog (stable identity) for standalone use + tests.
CATALOG: list[Adaptable] = derive()
_BY = {a.key: a for a in CATALOG}
_BY.update({a.settings_key: a for a in CATALOG})


def get_adaptable(key: str) -> Adaptable | None:
    return _BY.get(key)


async def catalog_with_values(*, settings_store: Any, user_id: str,
                              ui_settings: dict[str, int] | None = None) -> list[dict]:
    """The catalog (derived from ``ui_settings`` when given) with each setting's
    current per-user value. Never raises — a store hiccup leaves a value blank."""
    getter = getattr(settings_store, "get_user_or_global", None) or \
        getattr(settings_store, "get_user", None)
    out: list[dict] = []
    for a in derive(ui_settings):
        val = None
        if getter and user_id:
            try:
                val = await getter(user_id, a.settings_key)
            except Exception as exc:  # noqa: BLE001 — discovery must never fail on one key
                log.warning("adaptable_value_lookup_failed", key=a.key, error=repr(exc))
        out.append(a.to_dict(value=val))
    return out
