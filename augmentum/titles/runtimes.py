"""Runtime registry -- how Titles execute.

A Runtime is the engine that actually plays a title. ``browser-iframe``
mounts the title's HTML in a sandboxed iframe (js13k, web bookmarks,
GitHub-built static sites). ``agsp-streamed`` delegates to the existing
``GameStreamRuntime`` for container-streamed games. Future runtimes
plug in identically: ``emulator-browser`` (EmulatorJS), ``emulator-
streamed`` (RetroArch in a container), ``web-app`` (elevated capability
iframe), ``git-built`` (build pipeline → static iframe).

The runtime layer is *engine-agnostic*. Every runtime takes a
TitleManifest + LaunchContext and returns a ``LaunchHandle`` describing
how the browser should connect (URL, signaling endpoint, embed token).
The substrate services (InputBus, SaveService, TelemetryService) are
runtime-aware but not runtime-coupled -- a controller binding works
identically across runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from augmentum.titles.manifest import (
    KIND_EMULATOR_ROM,
    KIND_GIT_PROJECT,
    KIND_JS13K_GAME,
    KIND_STREAMED_GAME,
    KIND_WEB_APP,
    TitleManifest,
)
from augmentum.titles.rom_systems import get_system
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class BiosMissingError(RuntimeError):
    """Raised by EmulatorBrowserRuntime when launch is blocked by
    missing required BIOS files. The service layer catches this and
    re-raises as TitleNotPlayable so the route maps to a 409 with the
    actionable error message intact."""


class CoreUnavailableError(RuntimeError):
    """Raised by EmulatorBrowserRuntime when the system's libretro core
    isn't bundled in our EmulatorJS build (core_status != 'bundled').
    Same surface as BiosMissingError: caught by the service layer and
    surfaced as an actionable 409, so the user sees a clean message
    instead of EmulatorJS's broken 'start core' fallback UI."""


@dataclass(frozen=True)
class LaunchHandle:
    """Returned by ``Runtime.launch``; carries everything the browser
    needs to mount the title.

    Distinct from a session/run id -- this is the "where do I connect"
    primitive. The TitleService records a corresponding ``title_runs``
    row separately and returns its run_id alongside this handle to the
    caller.
    """

    runtime_id: str
    kind: str                                 # 'iframe' | 'webrtc' | 'native-tab'
    target: str                               # iframe src URL, signaling path, etc.
    session_id: str = ""                      # adapter-specific id (AGSP session, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)


class Runtime(Protocol):
    """Contract for a title runtime."""

    id: str
    label: str
    capabilities: dict[str, Any]              # input_modes, multiplayer, save_states, ...

    async def supports(self, manifest: TitleManifest) -> bool:
        """True iff this runtime can play the given title."""

    async def launch(
        self, manifest: TitleManifest, ctx: dict,
    ) -> LaunchHandle:
        """Start a session and return the connection handle.

        ``ctx`` is the launch-context dict (user_id, requested
        resolution/bitrate for streamed runtimes, etc.). Routes
        translate request bodies into ctx.
        """

    async def stop(self, session_id: str, *, user_id: str = "") -> None:
        """Tear down a session. No-op for stateless runtimes (iframe)."""


# ── Built-in: BrowserIframeRuntime ────────────────────────────────────


class BrowserIframeRuntime:
    """Runtime for titles that play inside a sandboxed browser iframe.

    Covers js13k bundles (existing path), URL bookmarks, GitHub-built
    static sites, and any future "static HTML payload" sources. The
    runtime is stateless: launching just decides which URL to mount,
    and stopping is a no-op (the browser tears the iframe down on its
    own).

    For js13k specifically, the iframe URL is derived from the artifact
    metadata's ``embed_url`` -- the same field the existing game-surface
    UI already reads. This keeps full compatibility with current pins.
    """

    id = "browser-iframe"
    label = "Browser (in-app)"
    capabilities = {
        "input_modes": ["keyboard", "mouse", "touch", "gamepad"],
        "save_states": False,           # title-level; SRAM-style saves still go through SaveService
        "offline_after_load": True,
        "streaming": False,
    }

    async def supports(self, manifest: TitleManifest) -> bool:
        # Currently bound to: js13k pins, web bookmarks, git-built titles.
        # Streamed and emulator titles fall through to their own runtimes.
        return manifest.kind in {
            KIND_JS13K_GAME,
            KIND_WEB_APP,
            KIND_GIT_PROJECT,
        }

    async def launch(
        self, manifest: TitleManifest, ctx: dict,
    ) -> LaunchHandle:
        # The artifacts metadata supplies the embed URL. We don't try
        # to be clever -- if it's missing the route layer surfaces a
        # 400 telling the user to re-import / re-pin.
        embed = (
            manifest.raw_metadata.get("embed_url")
            or manifest.raw_metadata.get("embed_src")
            or manifest.raw_metadata.get("source_url")
            or ""
        )
        if not embed:
            log.warning(
                "browser_iframe_missing_embed",
                title_id=manifest.id,
                kind=manifest.kind,
            )
        return LaunchHandle(
            runtime_id=self.id,
            kind="iframe",
            target=embed,
            session_id="",
            metadata={
                "title": manifest.title,
                "kind": manifest.kind,
            },
        )

    async def stop(self, session_id: str, *, user_id: str = "") -> None:
        # Iframe runtimes are stateless; the browser handles teardown.
        return None


# ── Built-in: EmulatorBrowserRuntime ─────────────────────────────────


class EmulatorBrowserRuntime:
    """Runtime for emulator ROMs played via WASM in the browser.

    The actual emulation is done by EmulatorJS (vendored under
    ``ui/lib/emulator-js/``). This adapter doesn't run any code itself
    -- it produces a LaunchHandle the frontend uses to mount EmulatorJS
    with the right system, core, ROM URL, and save bridge.

    The save bridge is the per-(user, title) save endpoint surface
    (``/api/titles/{id}/saves/*``). EmulatorJS hooks into it via a JS
    shim the frontend stage will install on the iframe.

    Stateless: launching is just resolving URLs + config; stopping is
    a no-op (the browser tears the iframe down on its own).
    """

    id = "emulator-browser"
    label = "Emulator (browser)"
    capabilities = {
        "input_modes": ["keyboard", "gamepad", "touch"],
        "save_states": True,
        "offline_after_load": True,        # ROM cached in browser
        "streaming": False,
    }

    def __init__(self, controller_service=None, bios_store=None) -> None:
        # Late-bound by server.py once the controller substrate is up.
        # None = the canonical defaults from defaults.py travel as
        # ``controls`` via a separate path; iframe falls back to
        # EmulatorJS' own keyboard defaults if even that's missing.
        self._controllers = controller_service
        # bios_store is the per-user BIOS index. None = best-effort
        # launch (no BIOS validation, no bios_url). Set by server.py
        # at startup once the blob substrate is up.
        self._bios = bios_store

    def attach_controller_service(self, controller_service) -> None:
        """Idempotent late-bind. Called from server.py after the
        controller_service is constructed."""
        self._controllers = controller_service

    def attach_bios_store(self, bios_store) -> None:
        """Idempotent late-bind for the BIOS store. Called from
        server.py after the BiosStore is constructed."""
        self._bios = bios_store

    async def supports(self, manifest: TitleManifest) -> bool:
        if manifest.kind != KIND_EMULATOR_ROM:
            return False
        # System must be one we recognise. Imports use rom_systems.py
        # so this should never fail in practice -- defensive check
        # against hand-crafted manifests.
        from augmentum.titles.rom_systems import get_system
        spec = get_system(str(manifest.raw_metadata.get("system_id", "")))
        if spec is None:
            return False
        # Refuse systems that need a non-browser runtime so the
        # registry falls through to AgspStreamedRuntime (which claims
        # them via streaming_profile). Without this, the resolver
        # would land here first and the launch() would raise
        # CoreUnavailableError instead of routing to AGSP.
        if spec.core_status == "streaming_required":
            return False
        return True

    async def launch(
        self, manifest: TitleManifest, ctx: dict,
    ) -> LaunchHandle:
        meta = manifest.raw_metadata
        system_id = str(meta.get("system_id", ""))
        rom_sha = str(meta.get("rom_sha256", ""))
        if not system_id or not rom_sha:
            raise RuntimeError(
                f"emulator title {manifest.id!r} missing system_id or rom_sha256"
            )

        # Refuse non-bundled cores up-front. Without this, EmulatorJS
        # 404s the core .data file at runtime and falls into its
        # "start core" splash UI — looks like a settings screen, isn't.
        # Surface an actionable error tied to the catalog's core_status
        # so the user knows whether it's "coming with streaming runtime"
        # vs. "needs a different EmulatorJS build".
        spec = get_system(system_id)
        if spec is not None and spec.core_status != "bundled":
            if spec.core_status == "streaming_required":
                # In normal flow this branch is unreachable —
                # supports() rejects streaming_required ahead of
                # launch(), so the registry resolves to AgspStreamedRuntime
                # instead. Kept as a defensive guard for callers that
                # bypass the registry (tests, manual invocations).
                raise CoreUnavailableError(
                    f"{spec.label} needs the streaming runtime — "
                    f"call AgspStreamedRuntime instead of "
                    f"EmulatorBrowserRuntime."
                )
            if spec.core_status == "experimental":
                raise CoreUnavailableError(
                    f"{spec.label} support is experimental — its core "
                    f"({spec.libretro_core}) isn't bundled in our "
                    f"EmulatorJS build yet. Vendor the WASM core to "
                    f"enable it."
                )
            # Unknown status string — don't pretend the core exists.
            raise CoreUnavailableError(
                f"{spec.label} marked '{spec.core_status}' — not "
                f"playable through the browser runtime."
            )

        # ROM is served by an internal blob endpoint (the frontend
        # fetches the bytes directly; EmulatorJS feeds them to the
        # core). Same shape as save endpoints -- under the title's
        # ownership scope.
        rom_url = f"/api/titles/{manifest.id}/rom"
        save_bridge_url = f"/api/titles/{manifest.id}/saves"

        # BIOS resolution. For systems whose catalog requires BIOS
        # (psx/ps2/saturn/3do/lynx/amiga/etc.), refuse to launch when
        # the user hasn't installed the required files -- a half-
        # working emulator that falls into the EmulatorJS settings
        # screen is worse UX than an actionable error. For systems
        # where BIOS is optional (gba/nds), we still surface installed
        # BIOS via bios_url to improve accuracy when present.
        bios_required = bool(meta.get("bios_required", False))
        bios_url = ""
        bios_files: list[dict] = []
        user_id = ctx.get("user_id", "") if isinstance(ctx, dict) else ""
        if self._bios is not None and user_id:
            try:
                missing = await self._bios.missing_required(
                    user_id=user_id, system_id=system_id,
                )
                if bios_required and missing:
                    names = ", ".join(b.filename for b in missing)
                    raise BiosMissingError(
                        f"Cannot launch {manifest.title}: missing required "
                        f"BIOS for {system_id} ({names}). Drag the BIOS "
                        f"file(s) onto the Library to install."
                    )
                # Surface every installed BIOS for this system. The
                # iframe template currently consumes a single bios_url
                # (EmulatorJS limitation for multi-BIOS systems like
                # PS2/NDS); we provide the full list as bios_files so
                # the template can grow into multi-BIOS support
                # without a runtime API break.
                installed = await self._bios.list_for_user(
                    user_id=user_id, system_id=system_id,
                )
                for rec in installed:
                    url = (
                        f"/api/titles/bios/{system_id}/"
                        f"{rec.canonical_filename}"
                    )
                    bios_files.append({
                        "canonical_filename": rec.canonical_filename,
                        "url": url,
                        "size_bytes": rec.size_bytes,
                        "sha1": rec.sha1,
                    })
                if bios_files and not bios_url:
                    bios_url = bios_files[0]["url"]
            except BiosMissingError:
                raise
            except Exception as exc:
                # Non-fatal: BIOS resolution failure shouldn't block
                # launch for systems where it's only optional.
                log.warning(
                    "emulator_bios_resolve_failed",
                    title_id=manifest.id, system_id=system_id,
                    error=str(exc),
                )

        # Resolve the user's controller layout (defaults + override).
        # Embed it in the launch handle so the iframe configures
        # EmulatorJS' input mappings before the core boots, no
        # second round-trip needed.
        controls = None
        if self._controllers is not None:
            try:
                user_id = ctx.get("user_id", "") if isinstance(ctx, dict) else ""
                if user_id:
                    layout = await self._controllers.resolve(
                        user_id=user_id, system_id=system_id,
                    )
                    if layout is not None:
                        controls = layout.to_dict()
            except Exception as exc:
                # Don't fail the launch if controller resolution hiccups.
                log.warning(
                    "emulator_controls_resolve_failed",
                    title_id=manifest.id, system_id=system_id,
                    error=str(exc),
                )

        config = {
            "system": system_id,
            "core": str(meta.get("libretro_core", "")),
            "bios_required": bios_required,
            "bios_url": bios_url,
            "bios_files": bios_files,
            "rom_url": rom_url,
            "save_bridge_url": save_bridge_url,
            "title": manifest.title,
            # The frontend mounts EmulatorJS by loading
            # `${emulator_js_path}loader.js` and pointing
            # ``EJS_pathtodata`` at the same path. The bundle's
            # ``loader.js`` + cores live in ``ui/lib/emulator-js/data/``
            # by vendor convention; letting the runtime declare the
            # path here keeps a future config knob ("which fork of
            # EmulatorJS") manageable without UI churn.
            "emulator_js_path": "/ui/lib/emulator-js/data/",
            # Resolved controller layout for this (user, system) pair.
            # ``null`` if the controller substrate is offline; iframe
            # falls back to EmulatorJS' built-in defaults.
            "controls": controls,
        }

        return LaunchHandle(
            runtime_id=self.id,
            kind="emulator",                # frontend stage looks for this
            target=rom_url,                 # the iframe target conceptually
            session_id="",                  # stateless
            metadata=config,
        )

    async def stop(self, session_id: str, *, user_id: str = "") -> None:
        # Browser-side teardown is automatic when the iframe unmounts.
        return None


# ── Built-in: AgspStreamedRuntime ─────────────────────────────────────


class AgspStreamedRuntime:
    """Adapter that delegates to the existing GameStreamRuntime.

    Resolves the AGSP profile from the title's ``source_remote_id`` and
    forwards launch/stop to the running GameStreamRuntime instance
    (which lives at ``app.state.game_stream_runtime``). The substrate
    services don't care that this runtime crosses a process boundary;
    they see the same LaunchHandle shape.
    """

    id = "agsp-streamed"
    label = "Streamed (server)"
    capabilities = {
        "input_modes": ["keyboard", "mouse", "gamepad", "touch"],
        "save_states": True,
        "offline_after_load": False,
        "streaming": True,
    }

    def __init__(
        self,
        game_stream_runtime: Any | None = None,
        bios_store: Any | None = None,
    ) -> None:
        self._gsr = game_stream_runtime
        # BIOS lookup only matters for streamed-emulator profiles whose
        # emulator hard-requires BIOS files (PCSX2 today, future RPCS3).
        # Late-bound from server.py once BiosStore is constructed —
        # mirror of EmulatorBrowserRuntime.attach_bios_store.
        self._bios = bios_store

    def attach(self, game_stream_runtime: Any) -> None:
        """Late-bind the GameStreamRuntime once server.py builds it."""
        self._gsr = game_stream_runtime

    def attach_bios_store(self, bios_store: Any) -> None:
        """Idempotent late-bind for the BIOS store. Called from server.py
        after BiosStore is constructed."""
        self._bios = bios_store

    async def supports(self, manifest: TitleManifest) -> bool:
        if self._gsr is None:
            return False
        # Native streamed games (Luanti and similar — kind=streamed_game,
        # source_remote_id maps 1:1 to a registered profile).
        if manifest.kind == KIND_STREAMED_GAME:
            return True
        # Emulator ROMs whose system is marked streaming_required AND
        # has a streaming_profile pointing at a registered profile
        # (currently: gamecube/wii via the emulator-streamed profile).
        if manifest.kind == KIND_EMULATOR_ROM:
            from augmentum.titles.rom_systems import get_system
            spec = get_system(str(manifest.raw_metadata.get("system_id", "")))
            if spec is None:
                return False
            return bool(spec.streaming_profile)
        return False

    async def launch(
        self, manifest: TitleManifest, ctx: dict,
    ) -> LaunchHandle:
        if self._gsr is None:
            raise RuntimeError("agsp-streamed runtime not initialised")

        # Resolve profile id + per-emulator extras based on manifest kind.
        # Native streamed games carry the profile id in source_remote_id;
        # ROMs resolve via the catalog's streaming_profile field.
        profile_id = ""
        emulator = ""
        system_id = ""
        rom_blob_sha = ""
        if manifest.kind == KIND_EMULATOR_ROM:
            from augmentum.titles.rom_systems import get_system
            system_id = str(manifest.raw_metadata.get("system_id", ""))
            spec = get_system(system_id)
            if spec is None or not spec.streaming_profile:
                raise RuntimeError(
                    f"emulator title {manifest.id!r}: system {system_id!r} "
                    f"has no streaming_profile mapping"
                )
            profile_id = spec.streaming_profile
            emulator = spec.streaming_emulator
            rom_blob_sha = str(manifest.raw_metadata.get("rom_sha256", ""))
            if not rom_blob_sha:
                raise RuntimeError(
                    f"emulator title {manifest.id!r} missing rom_sha256 — "
                    f"cannot bind-mount ROM into streamed container"
                )
        else:
            profile_id = manifest.source_remote_id or manifest.raw_metadata.get(
                "profile_id", ""
            )
            if not profile_id:
                raise RuntimeError(
                    f"title {manifest.id!r} missing AGSP profile_id "
                    f"(in source_remote_id or metadata.profile_id)"
                )

        # Resolve BIOS files for emulators that hard-require them
        # (PCSX2 today, future RPCS3). The bios_store records each
        # installed BIOS by canonical filename + content-addressed
        # blob_sha256; the entrypoint will symlink each from the
        # data volume into the emulator's bios dir at session start.
        # Pre-flight UI prompt (library-game-sources::_emulatorLaunch)
        # already gates the launch on all_required_present, so this
        # list is the *full* set the user has installed; it isn't
        # responsible for blocking incomplete sets.
        bios_files: list[tuple[str, str]] = []
        if (
            emulator == "pcsx2"
            and system_id
            and self._bios is not None
        ):
            try:
                records = await self._bios.list_for_user(
                    user_id=ctx["user_id"], system_id=system_id,
                )
                bios_files = [
                    (r.canonical_filename, r.blob_sha256)
                    for r in records
                    if r.blob_sha256 and r.canonical_filename
                ]
            except Exception as exc:                # noqa: BLE001
                log.warning(
                    "bios_resolve_failed",
                    system_id=system_id,
                    error=str(exc),
                )

        # Phone-as-controller — build the cast-input bridge URL
        # template for the in-container daemon. Falls back to
        # agent_bridge_base_url because both daemons share a network
        # namespace. Empty base disables phone-as-controller; the
        # runtime treats an empty template as a no-op.
        from augmentum.config import settings as _settings
        cast_input_base = (
            _settings.cast_input_bridge_base_url
            or _settings.agent_bridge_base_url
            or ""
        ).rstrip("/")
        cast_input_url_template = (
            f"{cast_input_base}/api/cast/input/container-ws/"
            "{session_id}?token={token}"
            if cast_input_base else ""
        )

        info = await self._gsr.start_session(
            user_id=ctx["user_id"],
            profile_id=profile_id,
            world_id=ctx.get("world_id"),
            bitrate_mbps=ctx.get("bitrate_mbps"),
            resolution=ctx.get("resolution"),
            encoder=ctx.get("encoder"),
            # Streamed-emulator extras (empty strings for native streamed
            # games — the runtime treats them as no-ops).
            emulator=emulator,
            system_id=system_id,
            rom_blob_sha=rom_blob_sha,
            bios_files=bios_files,
            cast_input_bridge_url_template=cast_input_url_template,
        )
        return LaunchHandle(
            runtime_id=self.id,
            kind="webrtc",
            target=info.signaling_path,
            session_id=info.session_id,
            metadata={
                "stream_port": info.stream_port,
                "game_port": info.game_port,
                "bitrate_mbps": info.bitrate_mbps,
                "resolution": info.resolution,
                "profile_id": info.profile_id,
                # Emulator-specific metadata for the frontend's
                # loading-overlay copy ("Loading <system> on <emulator>...").
                "emulator": emulator,
                "system_id": system_id,
            },
        )

    async def stop(self, session_id: str, *, user_id: str = "") -> None:
        if self._gsr is None or not session_id:
            return None
        await self._gsr.stop_session(
            session_id, user_id=user_id, reason="clean",
        )


# ── Registry ──────────────────────────────────────────────────────────


class RuntimeRegistry:
    def __init__(self) -> None:
        self._runtimes: dict[str, Runtime] = {}

    def register(self, runtime: Runtime) -> None:
        self._runtimes[runtime.id] = runtime

    def get(self, runtime_id: str) -> Runtime | None:
        return self._runtimes.get(runtime_id)

    def list(self) -> list[Runtime]:
        return sorted(self._runtimes.values(), key=lambda r: r.label)

    async def resolve_for(self, manifest: TitleManifest) -> Runtime | None:
        """Pick the best runtime for a title.

        Resolution order:
          1. ``manifest.runtime_preferred`` if registered AND supports() returns True
          2. Any runtime in ``manifest.runtime_alternates`` that supports
          3. Any registered runtime that supports
          4. None (route layer surfaces 400)
        """
        preferred = self.get(manifest.runtime_preferred)
        if preferred is not None and await preferred.supports(manifest):
            return preferred
        for alt_id in manifest.runtime_alternates:
            alt = self.get(alt_id)
            if alt is not None and await alt.supports(manifest):
                return alt
        for rt in self._runtimes.values():
            if await rt.supports(manifest):
                return rt
        return None

    def clear(self) -> None:
        """Reset registry (test-only)."""
        self._runtimes.clear()


# Module-level singleton. Server.py wires the AGSP runtime in via
# ``runtime_registry.register(AgspStreamedRuntime(...))`` once the
# GameStreamRuntime is constructed. The browser-iframe runtime is
# registered eagerly at import time since it has no external deps.

runtime_registry = RuntimeRegistry()
runtime_registry.register(BrowserIframeRuntime())
runtime_registry.register(EmulatorBrowserRuntime())
