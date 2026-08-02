"""Game / title / controller / marketplace settings.

Covers AXF (Augmentum Experience Framework), AGSP (game streaming),
emulator pickup, controller framework, marketplace toggles.
"""

from __future__ import annotations

from augmentum.registry.registry import SettingsRegistry
from augmentum.registry.settings import Setting

_GAME = ("game",)
_GAME_ADV = ("game", "advanced")
_CTRL = ("controller",)
_TITLES = ("titles",)


def register(r: SettingsRegistry) -> None:
    # ============== Titles / marketplace ==============
    r.register(
        Setting(
            key="titles_enabled",
            kind="bool",
            default=True,
            label="Titles substrate",
            description=(
                "Enable the AXF title catalog. Off = no titles tab; "
                "marketplace also hides."
            ),
            section="titles",
            tags=_TITLES,
            voice_aliases=("titles", "games"),
        )
    )
    r.register(
        Setting(
            key="titles_storage_max_mb",
            kind="int",
            default=5000,
            label="Titles storage cap (MB)",
            description=(
                "Soft cap on per-user title-related blob storage (ROMs, "
                "build artifacts). 0 = unlimited."
            ),
            section="titles",
            min_value=0,
            max_value=100000,
            tags=_TITLES,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="marketplace_enabled",
            kind="bool",
            default=False,
            label="Marketplace",
            description=(
                "Enable the curated AXF title marketplace surface. Off by "
                "default; turn on to browse community-published titles."
            ),
            section="marketplace",
            tags=("marketplace",),
            voice_aliases=("marketplace",),
        )
    )

    # ============== Emulator ==============
    r.register(
        Setting(
            key="emulator_browser_enabled",
            kind="bool",
            default=False,
            label="Browser emulator",
            description=(
                "Enable the in-browser emulator (EmulatorJS-based). Off by "
                "default — opt in after pulling ROM packs."
            ),
            section="games.emulator",
            tags=("games", "emulator"),
            voice_aliases=("emulator",),
        )
    )
    r.register(
        Setting(
            key="emulator_rom_max_mb",
            kind="int",
            default=0,
            label="ROM upload cap (MB)",
            description=(
                "Hard ceiling for individual ROM uploads. 0 = use the "
                "default 2 GB cap."
            ),
            section="games.emulator",
            min_value=0,
            max_value=4096,
            tags=("games", "emulator", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="emulator_save_max_per_slot_mb",
            kind="int",
            default=50,
            label="Per-save-slot cap (MB)",
            description=(
                "Per-slot save cap. State saves can reach 5-20 MB on N64."
            ),
            section="games.emulator",
            min_value=1,
            max_value=1024,
            tags=("games", "emulator", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="emulator_save_slots_per_rom",
            kind="int",
            default=8,
            label="Save slots per ROM",
            description=(
                "How many save state slots per ROM. 1 = quicksave/SRAM only; "
                "8 = standard manual states."
            ),
            section="games.emulator",
            min_value=1,
            max_value=32,
            tags=("games", "emulator", "advanced"),
            advanced=True,
        )
    )

    # ============== Game streaming (AGSP) ==============
    r.register(
        Setting(
            key="game_stream_enabled",
            kind="bool",
            default=False,
            label="Game streaming",
            description=(
                "Enable the AGSP game-streaming platform. Off by default — "
                "turn on after bringing up the streaming container."
            ),
            section="games.stream",
            tags=_GAME,
            voice_aliases=("game streaming", "agsp"),
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="game_stream_max_concurrent",
            kind="int",
            default=2,
            label="Concurrent streams per user",
            description=(
                "Per-user concurrent stream cap. Each consumes credits."
            ),
            section="games.stream",
            min_value=1,
            max_value=8,
            tags=_GAME_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="game_stream_default_bitrate_mbps",
            kind="int",
            default=4,
            label="Stream bitrate (Mbps)",
            description=(
                "Default bitrate when the client doesn't request one. "
                "Profiles can override."
            ),
            section="games.stream",
            min_value=1,
            max_value=25,
            tags=_GAME,
        )
    )
    r.register(
        Setting(
            key="game_stream_idle_timeout_seconds",
            kind="int",
            default=3600,
            label="Stream idle timeout (s)",
            description=(
                "How long a session stays warm after the last client "
                "disconnects."
            ),
            section="games.stream",
            min_value=60,
            max_value=7200,
            tags=_GAME_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="game_stream_prefer_hw_encoder",
            kind="bool",
            default=True,
            label="Prefer hardware encoder",
            description=(
                "Prefer NVENC / VAAPI / VideoToolbox when the host supports "
                "it. Falls back to software encoding when no hardware "
                "encoder is present."
            ),
            section="games.stream",
            tags=_GAME_ADV,
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="game_stream_mouse_sensitivity",
            kind="float",
            default=0.2,
            label="Stream mouse sensitivity",
            description=(
                "Mouse sensitivity for streamed sessions. Matches desktop "
                "Luanti default of 0.2."
            ),
            section="games.stream",
            min_value=0.01,
            max_value=2.0,
            tags=_GAME,
        )
    )

    # ============== Game portal ==============
    r.register(
        Setting(
            key="game_portal_enabled",
            kind="bool",
            default=True,
            label="Game portal",
            description=(
                "Enable the web-game browse portal (JS13K, etc.)."
            ),
            section="games.portal",
            tags=_GAME,
        )
    )
    r.register(
        Setting(
            key="game_portal_recommendations",
            kind="enum",
            default="off",
            label="Game recommendations",
            description=(
                "Proactive recommendation level: 'off' = never suggest, "
                "'contextual' = suggest when relevant, 'aggressive' = "
                "spotlight new games regularly."
            ),
            section="games.portal",
            enum_values=("off", "contextual", "aggressive"),
            max_length=16,
            tags=_GAME,
        )
    )
    r.register(
        Setting(
            key="game_portal_default_sources",
            kind="str",
            default="js13k,jams",
            label="Portal default sources",
            description=(
                "Comma-separated list of external catalogs the browse tab "
                "pulls from."
            ),
            section="games.portal",
            max_length=256,
            tags=_GAME,
        )
    )

    # ============== Controllers ==============
    r.register(
        Setting(
            key="controller_remap_enabled",
            kind="bool",
            default=True,
            label="Controller remapping",
            description=(
                "Enable per-user controller remap profiles. Off = use the "
                "raw gamepad axes/buttons."
            ),
            section="controller",
            tags=_CTRL,
        )
    )
    r.register(
        Setting(
            key="controller_haptic_enabled",
            kind="bool",
            default=True,
            label="Haptic rumble",
            description=(
                "Translate game rumble → Gamepad API vibration when the "
                "host browser supports it."
            ),
            section="controller",
            tags=_CTRL,
            voice_aliases=("rumble", "haptics"),
        )
    )
    r.register(
        Setting(
            key="controller_deadzone",
            kind="float",
            default=0.15,
            label="Stick deadzone",
            description=(
                "Analog stick deadzone (0.0-0.5). Below this magnitude the "
                "axis is treated as zero."
            ),
            section="controller",
            min_value=0.0,
            max_value=0.5,
            tags=("controller", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="controller_touch_overlay",
            kind="enum",
            default="auto",
            label="Touch overlay",
            description=(
                "Show the on-screen virtual gamepad. 'auto' = show when no "
                "physical pad is detected; 'always' / 'never' override."
            ),
            section="controller",
            enum_values=("auto", "always", "never"),
            max_length=16,
            tags=_CTRL,
        )
    )
    r.register(
        Setting(
            key="controller_pad_routing",
            kind="enum",
            default="index",
            label="Pad routing",
            description=(
                "How gamepads are routed to player slots. 'index' = pad 0 to "
                "P1, etc. 'identity' = route by remembered pad identity."
            ),
            section="controller",
            enum_values=("index", "identity"),
            max_length=16,
            tags=("controller", "advanced"),
            advanced=True,
        )
    )
