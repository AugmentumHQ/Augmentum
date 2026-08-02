"""Per-game profile registry.

A profile is the metadata blob the runtime needs to start a particular
game inside the streaming container: which Docker image, what default
resolution, which encoder preferences are sane, what ports to expose.

Adding a new game = registering a ``GameProfile``. The container itself
is built separately (``services/game-stream/Dockerfile.<game>``); the
profile is the *Augmentum-side* knowledge of that image.

Profiles are intentionally lightweight -- they're declarative metadata,
not code. Game-specific runtime hooks (e.g. Luanti bridge mod commands)
live in dedicated adapter modules later.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GameProfile:
    id: str
    display_name: str
    image: str                                          # docker image tag
    default_resolution: str = "1280x720"
    default_bitrate_mbps: int = 4
    supported_encoders: tuple[str, ...] = ("nvenc", "vaapi", "x264")
    recommended_encoder: str = "nvenc"
    # Per-game world settings the UI exposes; runtime forwards them as
    # env vars / mounted config to the container.
    settings_schema: dict[str, dict] = field(default_factory=dict)
    # Optional: a short description used in the Game Portal tile.
    description: str = ""
    # NOTE: ``supported_encoders`` / ``recommended_encoder`` are advisory
    # only today. ``GameStreamRuntime._pick_encoder`` force-returns
    # ``x264enc`` until the cudaconvert NoneType crash on Docker Desktop
    # is fixed (see the long comment there). Don't rely on these fields
    # for runtime decisions; treat them as documentation of *intended*
    # support.
    # Whether this game supports multiple concurrent players in one world.
    multiplayer: bool = False
    # Whether this game has a server-side mod/scripting layer (for
    # future AI-as-player and AI-as-builder integration).
    scriptable: bool = False
    # Whether this profile's container should receive GPU passthrough
    # when the host has it available. Default true — most streamed
    # games (Luanti, future AAA games) benefit from real OpenGL +
    # NVENC. The emulator-streamed profile sets it false because
    # Dolphin/Qt6 SIGBUSes when NVIDIA's libGL is overlay-mounted
    # onto Xvfb (the GLX visual selection hits memory the driver
    # doesn't tolerate on a virtual display); we use Mesa's software
    # libGL + Dolphin's software emulation backend instead.
    requires_gpu: bool = True
    # Whether this profile needs default Linux capabilities + the
    # default seccomp profile (i.e. NOT cap-drop=ALL + no-new-
    # privileges). Default false: most streamed games run fine in
    # the locked-down container. Dolphin (and likely PCSX2/RPCS3)
    # use SIGBUS/SIGSEGV signal handlers for fastmem soft-MMU
    # emulation; under the strict caps, these handlers can't be
    # installed and the emulator SIGBUSes within ~5s of trying to
    # commit its emulated address space. Setting this true gives
    # the container default caps + standard seccomp profile.
    relax_security: bool = False
    # Whether this profile wants browser controllers bridged through
    # /dev/uinput. Session input settings can still disable passthrough.
    wants_gamepad: bool = False
    # Declarative input knobs exposed to the browser. Route handlers
    # round-trip this so the UI can decide which controls to render.
    input_capabilities: dict[str, dict] = field(default_factory=dict)
    # Admission-control weight. 1 unit ≈ "one Luanti's worth of host
    # resources." Read by ``GameStreamRuntime._admit`` to decide whether
    # a host has room for a new session. Heavy emulators (Dolphin,
    # PCSX2) bump to 2 today; future RPCS3 will likely be 4. Tuned
    # against the host's active+resident credit budgets (settings keys
    # ``game_stream_active_credit_budget`` / ``..._resident_...``).
    cost_credits: int = 1
    # Maximum concurrent players a single session of this game supports.
    # Used by couch co-op to decide whether the "+ Players" chip
    # appears on the game tile. 1 = single-player, no invite UI.
    # Emulators default to 4 (XInput's standard); NES profile overrides
    # to 2; Luanti remains 1 today (its own multiplayer story is
    # server-side, not couch).
    max_players: int = 1


class ProfileRegistry:
    """In-memory registry of installed profiles."""

    def __init__(self) -> None:
        self._profiles: dict[str, GameProfile] = {}

    def register(self, profile: GameProfile) -> None:
        self._profiles[profile.id] = profile

    def get(self, profile_id: str) -> GameProfile | None:
        return self._profiles.get(profile_id)

    def list(self) -> list[GameProfile]:
        return sorted(self._profiles.values(), key=lambda p: p.display_name)

    def has(self, profile_id: str) -> bool:
        return profile_id in self._profiles


profile_registry = ProfileRegistry()


# ── Built-in profiles ─────────────────────────────────────────────────
# Each entry is shipped with Augmentum. Image tags reference the
# Dockerfiles under ``services/game-stream/`` (built in Phase 2).

profile_registry.register(GameProfile(
    id="emulator-streamed",
    display_name="Emulator (streamed)",
    image="augmentum-game-stream-emulator-streamed:latest",
    # Heavy emulators (Dolphin, eventually PCSX2/RPCS3) target 60fps
    # at native console resolution. 720p covers GC/Wii native (480p
    # upscaled to 1080p internal via Dolphin's EFB scale, then
    # downsampled by the encoder). 1080p stream resolution would burn
    # NVENC budget without visible improvement on most setups.
    default_resolution="1280x720",
    # Bumped from the Luanti default of 4 -- emulator-rendered output
    # is information-dense (sprites, text, HUD) and more sensitive to
    # bitrate floor than block-y voxel landscapes.
    default_bitrate_mbps=8,
    supported_encoders=("nvenc", "vaapi", "x264"),
    recommended_encoder="nvenc",
    description=(
        "Dolphin (GameCube/Wii), with PCSX2/RPCS3/Citra coming. "
        "Streams from a native emulator running on the server -- "
        "requires GPU passthrough for playable framerates."
    ),
    # Single-player today (Dolphin local-multiplayer over the streamed
    # input layer is possible but not wired; would need WebRTC datachannel
    # plumbing for real-time gamepad sync between participants).
    multiplayer=False,
    scriptable=False,
    # The user-facing settings live in the catalog (system_id auto-
    # selects the emulator). Per-emulator runtime knobs (Dolphin EFB
    # scale, etc.) are deferred until we have a real config UX.
    settings_schema={},
    # NO GPU passthrough — Dolphin SIGBUSes when NVIDIA's libGL is
    # overlay-mounted on Xvfb. Tradeoff: software emulation only,
    # ~5-15 FPS for AAA games. Worth it for stability until we wire
    # a working hardware path (Vulkan, EGL surfaceless, or Sunshine).
    requires_gpu=False,
    # Dolphin's fastmem signal-handler-based soft-MMU SIGBUSes under
    # cap-drop=ALL + no-new-privileges (the seccomp default for our
    # streamed containers). Confirmed via diagnostic: same image,
    # same env, dropped these flags = Dolphin runs fine. PCSX2 and
    # RPCS3 will likely need this same flag when they're wired.
    relax_security=True,
    # Gamepad passthrough: browser Gamepad API events are forwarded
    # by Selkies to a UDS, the in-container bridge translates to
    # /dev/uinput, and SDL2 (PCSX2 + Dolphin both use it) picks up
    # the resulting /dev/input/jsX devices automatically.
    wants_gamepad=True,
    input_capabilities={
        "gamepad": {
            "supported": True,
            "default_enabled": True,
            "bridge": "selkies-uinput",
        },
        "controller_deadzone": {
            "min": 0.0,
            "max": 0.5,
            "default": 0.15,
            "step": 0.01,
        },
        "pointer": {"mode": "absolute-or-emulator-native"},
        "touch": {"supported": False},
    },
    # Dolphin/PCSX2 = heavy CPU on software-OpenGL path. 2 credits.
    cost_credits=2,
    # Most emulator targets (GC/Wii/PSX/PS2/DS) support 4 players via
    # XInput. NES/Genesis cores typically cap at 2 — system_id-aware
    # override happens at session-start time in the runtime if we ever
    # need stricter caps. Defaulting to 4 here is the right ceiling.
    max_players=4,
))

profile_registry.register(GameProfile(
    id="luanti",
    display_name="Voxel World (Luanti)",
    image="augmentum-game-stream-luanti:latest",
    default_resolution="1280x720",
    default_bitrate_mbps=4,
    supported_encoders=("nvenc", "vaapi", "x264"),
    recommended_encoder="nvenc",
    description=(
        "Open-source voxel sandbox -- explore, mine, build. Ships with "
        "Minetest Game (creative sandbox; no mobs or hunger). VoxeLibre / "
        "Mineclonia content packs are a planned follow-up once the Luanti "
        "version in the base image is upgraded past 5.7."
    ),
    # NOTE: structurally single-player today even though the Luanti
    # server inside the container supports multiplayer natively. The
    # streaming runtime spawns one container per session and the
    # server's UDP :30000 isn't port-published, so no second browser
    # can join the same world. Re-enable when we add an explicit
    # "join existing session" path (Phase 5).
    multiplayer=False,
    scriptable=True,
    settings_schema={
        "gamemode": {
            "type": "enum",
            "values": ["survival", "creative"],
            "default": "survival",
            "label": "Game Mode",
        },
        "seed": {
            "type": "string",
            "default": "",
            "label": "World Seed (optional)",
        },
        "pvp": {
            "type": "bool",
            "default": False,
            "label": "Enable PvP",
        },
    },
    wants_gamepad=True,
    input_capabilities={
        "gamepad": {
            "supported": True,
            "default_enabled": True,
            "bridge": "selkies-uinput",
        },
        "controller_deadzone": {
            "min": 0.0,
            "max": 0.5,
            "default": 0.15,
            "step": 0.01,
        },
        "pointer": {
            "mode": "relative",
            "mouse_sensitivity": {
                "min": 0.01,
                "max": 2.0,
                "default": 0.2,
                "step": 0.01,
            },
        },
        "touch": {"supported": True, "default": "auto"},
    },
))


# ── Generic browser-streaming profile ────────────────────────────────
# Not a game per se — it's the "render any URL, stream the result"
# primitive. Anything web-rendered that's too heavy for a weak TV
# (VRM companion, comic reader, notebook, browse panel, web games)
# rides this profile. The caller passes ``target_url`` in
# ``world_settings`` and DockerContainerAdapter forwards it as
# AUGMENTUM_TARGET_URL.
#
# Resource cost: one container per active cast, ~400-600MB RAM with
# Chromium loaded. Idle baseline is zero — containers are spawned on
# demand and reaped when the cast ends. With this single profile the
# entire /ui/ surface area becomes castable to weak hardware.
profile_registry.register(GameProfile(
    id="browser-stream",
    display_name="Browser surface (streamed)",
    image="augmentum-stream-browser:latest",
    # 1080p is the default ceiling for cast surfaces — most TVs are
    # native 1080p/4K and the bitrate is comfortable at 6mbps with
    # NVENC. Bump per-session for higher-fidelity targets (e.g. the
    # notebook surface where typography matters more than motion).
    default_resolution="1920x1080",
    default_bitrate_mbps=6,
    supported_encoders=("nvenc", "vaapi", "x264"),
    recommended_encoder="nvenc",
    description=(
        "Renders any Augmentum web surface server-side and streams it "
        "to the receiver over WebRTC. Used by the cast pipeline for "
        "VRM, comic reader, notebook, browse panel — anything too "
        "heavy for the receiving device's GPU."
    ),
    multiplayer=False,
    scriptable=False,
    # The schema entries here are what the cast pipeline forwards
    # in ``world_settings``. Today the only one used is target_url;
    # leaving the schema declarative documents the contract.
    settings_schema={
        "target_url": {
            "type": "string",
            "default": "",
            "label": "Target URL",
        },
    },
    requires_gpu=True,
    relax_security=False,
    # Gamepad / pointer / touch may layer on later for streamed
    # interactive web surfaces (browse panel, web games). Initial
    # cut is "view-only" — the receiver page renders, no inputs flow
    # back. Wiring inputs is a separate, additive change once we
    # have a real first user for it.
    wants_gamepad=False,
    input_capabilities={
        "gamepad": {"supported": False},
        "pointer": {"mode": "none"},
        "touch": {"supported": False},
    },
))
