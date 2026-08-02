"""Dataclasses for the universal cast pipeline.

Mirrors :class:`augmentum.game_stream.profiles.GameProfile` in shape —
frozen, declarative, JSON-serialisable. Per-game runtime decisions
(strategy + adapter chain) live here, *not* in code.

See spec: ``docs/superpowers/specs/2026-06-04-universal-cast-pipeline-design.md``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

# Strategy kind constants. Strings (not Enums) for trivial JSON
# serialisation + forward-compat: an unknown strategy string round-trips
# untouched through the registry.
STRATEGY_SHIM = "shim"
STRATEGY_PROXY = "proxy"
STRATEGY_CONTAINERIZED = "containerized"

CastStrategyKind = Literal["shim", "proxy", "containerized"]

# Provenance for classified_by — same forward-compat as strategy.
CLASSIFIED_DEFAULT = "default"
CLASSIFIED_PROBE = "probe"
CLASSIFIED_MANUAL = "manual"
CLASSIFIED_TELEMETRY = "telemetry"

# Built-in adapter ids. Anything else is rejected by the registry on
# upsert so we don't accidentally persist a typo'd chain.
KNOWN_ADAPTERS: frozenset[str] = frozenset({
    "gamepad_api",
    "keyboard",
    "touch",
    "pointer",
})


@dataclass(frozen=True)
class KeymapProfile:
    """Per-adapter keymap overrides. Each field is opaque to the Python
    side — the loader (``ui/scripts/cast-input/universal-input-adapter.js``)
    consumes it. We just shuttle the JSON blobs through.
    """

    keyboard: dict[str, Any] = field(default_factory=dict)
    touch: dict[str, Any] = field(default_factory=dict)
    pointer: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.keyboard or self.touch or self.pointer)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.keyboard:
            out["keyboard"] = dict(self.keyboard)
        if self.touch:
            out["touch"] = dict(self.touch)
        if self.pointer:
            out["pointer"] = dict(self.pointer)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> KeymapProfile:
        if not isinstance(data, dict):
            return cls()
        return cls(
            keyboard=dict(data.get("keyboard") or {}),
            touch=dict(data.get("touch") or {}),
            pointer=dict(data.get("pointer") or {}),
        )


@dataclass(frozen=True)
class CastProfile:
    """Per-game cast configuration. PK is (user_id, title_id)."""

    title_id: str
    user_id: str = ""
    strategy: CastStrategyKind = STRATEGY_SHIM
    embed_url: str = ""                       # for shim/proxy strategies
    container_profile_id: str = ""            # for containerized
    input_chain: tuple[str, ...] = ("gamepad_api",)
    keymap: KeymapProfile | None = None
    quirks: dict[str, Any] = field(default_factory=dict)
    classified_by: str = CLASSIFIED_DEFAULT
    classified_at: float = 0.0
    failed_at: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the on-the-wire JSON shape (route responses).
        Keys mirror the dataclass field names; ``keymap`` is flattened
        to its dict form (or omitted when empty).
        """
        out: dict[str, Any] = {
            "title_id": self.title_id,
            "user_id": self.user_id,
            "strategy": self.strategy,
            "embed_url": self.embed_url,
            "container_profile_id": self.container_profile_id,
            "input_chain": list(self.input_chain),
            "quirks": dict(self.quirks),
            "classified_by": self.classified_by,
            "classified_at": self.classified_at,
            "failed_at": self.failed_at,
            "notes": self.notes,
        }
        if self.keymap and not self.keymap.is_empty():
            out["keymap"] = self.keymap.to_dict()
        return out

    @classmethod
    def from_row(cls, row: Any) -> CastProfile:
        """Hydrate from a SQLite row in the canonical column order
        used by ``CastProfileRegistry._SELECT_COLS``.
        """
        (
            user_id, title_id, strategy, embed_url, container_profile_id,
            input_chain_json, keymap_json, quirks_json,
            classified_by, classified_at, failed_at, notes,
        ) = row
        input_chain = tuple(_safe_json_list(input_chain_json, ["gamepad_api"]))
        keymap_dict = _safe_json_dict(keymap_json, {})
        quirks_dict = _safe_json_dict(quirks_json, {})
        return cls(
            title_id=str(title_id or ""),
            user_id=str(user_id or ""),
            strategy=_coerce_strategy(strategy),
            embed_url=str(embed_url or ""),
            container_profile_id=str(container_profile_id or ""),
            input_chain=input_chain,
            keymap=KeymapProfile.from_dict(keymap_dict) if keymap_dict else None,
            quirks=quirks_dict,
            classified_by=str(classified_by or CLASSIFIED_DEFAULT),
            classified_at=float(classified_at or 0.0),
            failed_at=float(failed_at or 0.0),
            notes=str(notes or ""),
        )

    def merge_fields(self, **fields: Any) -> CastProfile:
        """Return a new profile with select fields overridden. Used for
        manual override paths (registry.override)."""
        data = {
            "title_id": self.title_id,
            "user_id": self.user_id,
            "strategy": self.strategy,
            "embed_url": self.embed_url,
            "container_profile_id": self.container_profile_id,
            "input_chain": self.input_chain,
            "keymap": self.keymap,
            "quirks": dict(self.quirks),
            "classified_by": self.classified_by,
            "classified_at": self.classified_at,
            "failed_at": self.failed_at,
            "notes": self.notes,
        }
        data.update(fields)
        if "input_chain" in fields:
            data["input_chain"] = _coerce_input_chain(fields["input_chain"])
        if "strategy" in fields:
            data["strategy"] = _coerce_strategy(fields["strategy"])
        return CastProfile(**data)


@dataclass(frozen=True)
class HostCapabilities:
    """What the cast host can spend on this cast.

    Strategy.can_handle consults this to bail early on options the host
    can't service (e.g. containerized when no AGSP credits remain).
    """

    has_gpu: bool = False
    has_agsp: bool = False
    agsp_credits_available: int = 0
    has_network_egress: bool = True  # for proxy fetching


@dataclass(frozen=True)
class PreparedCast:
    """Hand-off payload returned by CastStrategy.prepare.

    ``surface_url`` is what library2 sends to the receiver via
    ``POST /api/cast/send`` — the receiver opens it as the main slot's
    surface URL. ``input_chain`` rides separately as a postMessage to
    the loader inside that surface once it mounts.
    """

    title_id: str
    strategy: CastStrategyKind
    surface_url: str
    surface_kind: str = "html.generic"
    input_chain: tuple[str, ...] = ("gamepad_api",)
    keymap: KeymapProfile | None = None
    session_token: str = ""             # for proxy
    container_session_id: str = ""      # for containerized
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "title_id": self.title_id,
            "strategy": self.strategy,
            "surface_url": self.surface_url,
            "surface_kind": self.surface_kind,
            "input_chain": list(self.input_chain),
            "session_token": self.session_token,
            "container_session_id": self.container_session_id,
            "notes": self.notes,
        }
        if self.keymap and not self.keymap.is_empty():
            out["keymap"] = self.keymap.to_dict()
        return out


# ── helpers ──────────────────────────────────────────────────────


def _safe_json_list(raw: Any, default: list[Any]) -> list[Any]:
    if not raw:
        return list(default)
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return list(default)
    return list(parsed) if isinstance(parsed, list) else list(default)


def _safe_json_dict(raw: Any, default: dict[str, Any]) -> dict[str, Any]:
    if not raw:
        return dict(default)
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return dict(default)
    return dict(parsed) if isinstance(parsed, dict) else dict(default)


def _coerce_strategy(value: Any) -> CastStrategyKind:
    s = str(value or "").strip().lower()
    if s in (STRATEGY_SHIM, STRATEGY_PROXY, STRATEGY_CONTAINERIZED):
        return s  # type: ignore[return-value]
    return STRATEGY_SHIM


def _coerce_input_chain(raw: Any) -> tuple[str, ...]:
    """Coerce + validate an input_chain. Drops unknown adapter ids
    and any duplicates while preserving order. Empty chain falls back
    to (gamepad_api,)."""
    if not isinstance(raw, (list, tuple)):
        return ("gamepad_api",)
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        s = str(item or "").strip()
        if s and s in KNOWN_ADAPTERS and s not in seen:
            out.append(s)
            seen.add(s)
    return tuple(out) if out else ("gamepad_api",)
