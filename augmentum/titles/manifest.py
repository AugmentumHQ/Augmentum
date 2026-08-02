"""TitleManifest -- typed projection over an ``artifacts`` row.

A title manifest is the canonical view of a "playable thing." It pulls
fields from the underlying artifact row (which already supports
``metadata`` JSON, ``pinned``, ``last_opened_at``, user-scoping, VFS
integration) and presents them as a typed, source/runtime-aware shape
the rest of AXF consumes.

We do not store a separate ``title_manifests`` table -- the artifact
row IS the manifest, and this dataclass is the lens through which AXF
reads it. New kinds (emulator ROM, web app, streamed game) just add a
new ``metadata.kind`` value; no schema change.

Compatibility note: existing js13k artifacts use ``metadata.kind ==
"game"`` (no further specialisation). AXF treats those as
``KIND_JS13K_GAME`` when ``metadata.source == "js13k"``. Future migration
options:

* leave them at ``kind == "game"`` and continue inferring sub-kind from
  ``metadata.source`` (zero migration cost; what we do today)
* opportunistic rewrite on next pin/edit (future, optional)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


# ── Recognised title kinds ────────────────────────────────────────────
#
# Adding a new kind = one constant + register a Source that produces
# manifests with this kind + register a Runtime that ``supports()`` it.

KIND_JS13K_GAME = "js13k_game"
KIND_STREAMED_GAME = "streamed_game"     # AGSP-bundled native games (Luanti, future)
KIND_EMULATOR_ROM = "emulator_rom"       # User-uploaded ROM, plays via libretro
KIND_WEB_APP = "web_app"                 # URL bookmark
KIND_GIT_PROJECT = "git_project"         # Cloned + built from a public Git repo

TITLE_KINDS: frozenset[str] = frozenset({
    KIND_JS13K_GAME,
    KIND_STREAMED_GAME,
    KIND_EMULATOR_ROM,
    KIND_WEB_APP,
    KIND_GIT_PROJECT,
})


def is_title_kind(kind: str) -> bool:
    return kind in TITLE_KINDS


# Legacy bridge: existing js13k artifacts have ``kind == "game"``
# (set before the AXF refactor). The store maps these to KIND_JS13K_GAME
# at read time without rewriting any rows.
_LEGACY_GAME_KIND = "game"


def _legacy_kind_to_axf(metadata: dict) -> str | None:
    """Map a pre-AXF ``metadata.kind`` value to an AXF kind, or None.

    Returns None when the artifact isn't a title at all (e.g. a doc,
    chart, or app-builder checkpoint).
    """
    raw = metadata.get("kind")
    if raw in TITLE_KINDS:
        return raw
    if raw == _LEGACY_GAME_KIND:
        # Existing js13k pins use the bare ``kind == "game"`` sentinel
        # plus ``source == "js13k"`` for sub-typing. Other future
        # legacy bridges land here too.
        source = metadata.get("source", "")
        if source == "js13k":
            return KIND_JS13K_GAME
        # Unknown legacy game source -- treat as js13k_game tentatively
        # rather than dropping it; the user can re-pin to upgrade.
        return KIND_JS13K_GAME
    return None


@dataclass(frozen=True)
class TitleManifest:
    """Typed projection of an artifacts row that represents a title."""

    # ── Identity ──────────────────────────────────────────────────
    id: str                                 # artifact id
    user_id: str
    title: str                              # display name
    version: str                            # free-form version string

    # ── Categorisation ───────────────────────────────────────────
    kind: str                               # one of TITLE_KINDS
    source_id: str                          # 'js13k', 'internal', 'agsp-profile', ...
    source_remote_id: str                   # source-specific external id

    # ── Runtime ──────────────────────────────────────────────────
    runtime_preferred: str                  # 'browser-iframe', 'agsp-streamed', 'emulator', ...
    runtime_alternates: tuple[str, ...]     # runtime ids the resolver may pick from

    # ── Behavioural ──────────────────────────────────────────────
    capabilities: dict[str, Any]            # input_modes, multiplayer, save_states, ...
    metadata: dict[str, Any]                # genre, tags, screenshots, year, ...

    # ── Library state (mirrored from artifact row) ───────────────
    pinned: bool
    last_played_at: str | None              # ISO datetime or None
    total_play_time_s: int                  # SUM(title_runs.duration_s); computed on read

    # ── Raw metadata blob (for callers that want to read fields the
    # ── manifest doesn't model yet without re-querying the store) ─
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    # ── Construction ─────────────────────────────────────────────

    @classmethod
    def from_artifact_row(
        cls,
        row: dict,
        *,
        total_play_time_s: int = 0,
    ) -> "TitleManifest | None":
        """Project an artifacts row into a TitleManifest.

        Returns None if the row is not a title (wrong ``metadata.kind``).
        """
        meta = _decode_metadata(row.get("metadata"))
        kind = _legacy_kind_to_axf(meta)
        if kind is None:
            return None

        source_id = str(meta.get("source") or _default_source_for_kind(kind))
        source_remote_id = str(meta.get("source_id") or "")

        # Each kind has a sensible default runtime; overridable via
        # ``metadata.runtime_preferred``. The alternates list is the
        # set of runtimes the runtime registry might switch to via the
        # auto-router (e.g. emulator-browser <-> emulator-streamed).
        runtime_preferred = str(
            meta.get("runtime_preferred")
            or _default_runtime_for_kind(kind)
        )
        alternates = meta.get("runtime_alternates")
        if not isinstance(alternates, list):
            alternates = list(_default_alternates_for_kind(kind))

        capabilities = meta.get("capabilities")
        if not isinstance(capabilities, dict):
            capabilities = {}

        return cls(
            id=str(row.get("id", "")),
            user_id=str(row.get("user_id", "")),
            title=str(
                row.get("display_name")
                or meta.get("title")
                or meta.get("name")
                or row.get("filename", "Untitled")
            ),
            version=str(meta.get("version", "")),
            kind=kind,
            source_id=source_id,
            source_remote_id=source_remote_id,
            runtime_preferred=runtime_preferred,
            runtime_alternates=tuple(str(a) for a in alternates),
            capabilities=capabilities,
            metadata={
                k: v for k, v in meta.items()
                # strip housekeeping keys that already have first-class
                # fields above so the .metadata view is the "user-facing"
                # subset (genre, tags, etc.) without duplication
                if k not in {
                    "kind", "source", "source_id", "version",
                    "runtime_preferred", "runtime_alternates", "capabilities",
                    "title", "name",
                }
            },
            pinned=bool(row.get("pinned", 0)),
            last_played_at=row.get("last_opened_at"),
            total_play_time_s=int(total_play_time_s or 0),
            raw_metadata=meta,
        )

    # ── Convenience / serialisation ──────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict for the API surface."""
        return {
            "id": self.id,
            "title": self.title,
            "version": self.version,
            "kind": self.kind,
            "source_id": self.source_id,
            "source_remote_id": self.source_remote_id,
            "runtime_preferred": self.runtime_preferred,
            "runtime_alternates": list(self.runtime_alternates),
            "capabilities": dict(self.capabilities),
            "metadata": dict(self.metadata),
            "library_state": {
                "pinned": self.pinned,
                "last_played_at": self.last_played_at,
                "total_play_time_s": self.total_play_time_s,
            },
        }


# ── Defaults per kind ────────────────────────────────────────────────


def _default_source_for_kind(kind: str) -> str:
    return {
        KIND_JS13K_GAME: "js13k",
        KIND_STREAMED_GAME: "agsp-profile",
        KIND_EMULATOR_ROM: "internal",
        KIND_WEB_APP: "url-bookmark",
        KIND_GIT_PROJECT: "github",
    }.get(kind, "internal")


def _default_runtime_for_kind(kind: str) -> str:
    return {
        KIND_JS13K_GAME: "browser-iframe",
        KIND_STREAMED_GAME: "agsp-streamed",
        KIND_EMULATOR_ROM: "emulator",                # resolves to -browser or -streamed
        KIND_WEB_APP: "browser-iframe",
        KIND_GIT_PROJECT: "browser-iframe",
    }.get(kind, "browser-iframe")


def _default_alternates_for_kind(kind: str) -> tuple[str, ...]:
    # Runtime IDs (matching ids registered in runtime_registry):
    #   "emulator-browser" — EmulatorBrowserRuntime (WASM/EmulatorJS)
    #   "agsp-streamed"    — AgspStreamedRuntime (Dolphin/PCSX2 etc. via AGSP)
    # Heavy systems (gamecube/wii/ps2) are streaming_required and only
    # the streamed runtime claims them; lighter systems get the browser
    # runtime first and skip the streamed alternate entirely.
    return {
        KIND_EMULATOR_ROM: ("emulator-browser", "agsp-streamed"),
    }.get(kind, ())


# ── Helpers ──────────────────────────────────────────────────────────


def _decode_metadata(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            decoded = json.loads(raw)
            return decoded if isinstance(decoded, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}
