"""User-experience settings — typography, ambient, soundscape,
discovery, offers, notifications, MCP, fabric, library publication.
"""

from __future__ import annotations

from augmentum.registry.registry import SettingsRegistry
from augmentum.registry.settings import Setting

_UI = ("ui",)
_AMBIENT = ("ambient", "grove")
_OFFERS = ("offers",)
_FABRIC = ("fabric",)


def register(r: SettingsRegistry) -> None:
    # ============== Typography ==============
    r.register(
        Setting(
            key="typography_selected",
            kind="str",
            default="system",
            label="Typography preset",
            description=(
                "Active typography preset key. Empty = system font stack."
            ),
            section="ui.typography",
            max_length=64,
            tags=_UI,
            voice_aliases=("typography", "font preset"),
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="typography_text_scale",
            kind="str",
            default="1",
            label="Text scale",
            description=(
                "Global text size multiplier. 0.7–1.4 range; stored as "
                "string for JSON compatibility with the UI."
            ),
            section="ui.typography",
            max_length=8,
            tags=_UI,
            voice_aliases=("text size", "text scale"),
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="typography_custom_fonts",
            kind="str",
            default="[]",
            label="Custom Google Fonts",
            description=(
                "JSON array of custom Google Fonts ({name, key}). Loaded "
                "into the typography picker."
            ),
            section="ui.typography",
            max_length=4096,
            tags=_UI,
            advanced=True,
        )
    )

    # ============== Ambient / Grove ==============
    r.register(
        Setting(
            key="ambient_video",
            kind="str",
            default="",
            label="Ambient video",
            description=(
                "JSON object describing the current ambient background "
                "video ({videoId, title, channel, isLivestream, thumbnail})."
            ),
            section="ambient",
            max_length=2048,
            tags=_AMBIENT,
        )
    )
    r.register(
        Setting(
            key="ambient_volume",
            kind="int",
            default=50,
            label="Ambient volume",
            description=(
                "Ambient background volume, 0-100."
            ),
            section="ambient",
            min_value=0,
            max_value=100,
            tags=_AMBIENT,
            voice_aliases=("ambient volume", "background volume"),
        )
    )
    r.register(
        Setting(
            key="ambient_loop_mode",
            kind="enum",
            default="off",
            label="Ambient loop mode",
            description=(
                "What ambient does at end-of-video: 'off' (stop), 'loop' "
                "(replay current), 'advance' (cycle through favorites)."
            ),
            section="ambient",
            enum_values=("off", "loop", "advance"),
            max_length=16,
            tags=_AMBIENT,
        )
    )
    r.register(
        Setting(
            key="ambient_favorites",
            kind="str",
            default="",
            label="Ambient favorites",
            description=(
                "JSON array of favorite ambient video entries."
            ),
            section="ambient",
            max_length=16384,
            tags=_AMBIENT,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="soundscape_favorites",
            kind="str",
            default="",
            label="Soundscape favorites",
            description=(
                "JSON array of favorite soundscape stations."
            ),
            section="ambient.soundscape",
            max_length=8192,
            tags=_AMBIENT,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="soundscape_last_station",
            kind="str",
            default="",
            label="Last soundscape station",
            description=(
                "JSON object describing the last played station. Used to "
                "resume on app open."
            ),
            section="ambient.soundscape",
            max_length=2048,
            tags=_AMBIENT,
            advanced=True,
        )
    )

    # ============== Discovery ==============
    r.register(
        Setting(
            key="discovery_enabled",
            kind="bool",
            default=True,
            label="Discovery",
            description=(
                "Surface in-app discovery recommendations (knowledge / "
                "media / actions). Off = only explicit search."
            ),
            section="discovery",
            tags=("discovery",),
            voice_aliases=("discovery", "recommendations"),
        )
    )
    r.register(
        Setting(
            key="rsshub_base_url",
            kind="string",
            default="http://rsshub:1200",
            label="RSSHub base URL",
            description=(
                "Where rsshub:// feed shorthands resolve (e.g. "
                "rsshub://github/release/owner/repo in Discovery RSS "
                "subscriptions). Default matches the compose.rsshub "
                "overlay; empty disables expansion."
            ),
            section="discovery",
            tags=("discovery", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="discovery_max_recommendations",
            kind="int",
            default=15,
            label="Max recommendations",
            description=(
                "Cap on number of recommendations shown per surface."
            ),
            section="discovery",
            min_value=5,
            max_value=50,
            tags=("discovery", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="discovery_allow_non_latin",
            kind="bool",
            default=False,
            label="Allow non-Latin content",
            description=(
                "Include results dominated by non-Latin scripts (Chinese, "
                "Japanese, Korean, Arabic, Hebrew, Cyrillic, etc.) in "
                "Discovery For-You. Default off because the existing "
                "domain-reputation and keyword-scoring pipeline was tuned "
                "against an English corpus. Turn on if you read non-Latin "
                "content regularly — the language-script filter that "
                "rejects them by default will be bypassed."
            ),
            section="discovery",
            tags=("discovery",),
            voice_aliases=("non latin discovery", "non-latin content"),
        )
    )

    # ============== Offers ==============
    r.register(
        Setting(
            key="offers_enabled",
            kind="bool",
            default=True,
            label="In-chat offers",
            description=(
                "Let the chat LLM emit Install/Save/Switch proposal chips "
                "when it would help the user. Off = no offer chips."
            ),
            section="offers",
            tags=_OFFERS,
            voice_aliases=("offers", "suggestions"),
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="offers_max_per_day",
            kind="int",
            default=20,
            label="Offers per day cap",
            description=(
                "Daily ceiling on offer chips across all sessions. Prevents "
                "spam after many turns."
            ),
            section="offers",
            min_value=0,
            max_value=200,
            tags=("offers", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="offers_max_per_turn",
            kind="int",
            default=2,
            label="Offers per turn cap",
            description=(
                "Maximum offer chips on a single assistant reply."
            ),
            section="offers",
            min_value=0,
            max_value=10,
            tags=("offers", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="offers_max_pending_per_session",
            kind="int",
            default=5,
            label="Max pending offers",
            description=(
                "How many unresolved offer chips can accumulate in a single "
                "session before older ones are dropped."
            ),
            section="offers",
            min_value=0,
            max_value=50,
            tags=("offers", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="offers_default_expiry_days",
            kind="int",
            default=7,
            label="Offer expiry (days)",
            description=(
                "How long an unactioned offer stays visible before auto-"
                "dismissing."
            ),
            section="offers",
            min_value=1,
            max_value=90,
            tags=("offers", "advanced"),
            advanced=True,
        )
    )

    # ============== Notifications ==============
    r.register(
        Setting(
            key="notifications_enabled",
            kind="bool",
            default=True,
            label="Notifications",
            description=(
                "Master switch for Web Push notifications. Off = no pushes; "
                "in-app inbox still receives."
            ),
            section="notifications",
            tags=("notifications",),
            voice_aliases=("notifications", "push"),
            trust_tier="local_significant",
        )
    )

    # ============== MCP ==============
    r.register(
        Setting(
            key="mcp_enabled",
            kind="bool",
            default=True,
            label="MCP server + clients",
            description=(
                "Install-wide MCP toggle. Off = neither the inbound /mcp "
                "endpoint nor outbound MCPClientManager connect."
            ),
            section="mcp",
            tags=("mcp",),
            voice_aliases=("mcp",),
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="mcp_servers",
            kind="str",
            default="",
            label="MCP server configs",
            description=(
                "JSON array of MCP server configs "
                "([{\"name\": ..., \"command\": ..., \"args\": [...]}])."
            ),
            section="mcp",
            max_length=16384,
            tags=("mcp", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Fabric ==============
    r.register(
        Setting(
            key="fabric_enabled",
            kind="bool",
            default=False,
            label="Fabric (peer mesh)",
            description=(
                "Enable cross-instance peer coordination. Off (default) = "
                "no fabric code path executes; solo installs are bit-for-bit "
                "unaffected."
            ),
            section="fabric",
            tags=_FABRIC,
            trust_tier="admin_only",
            restart_required=True,
        )
    )
    r.register(
        Setting(
            key="local_fabric_icon",
            kind="str",
            default="",
            label="Fabric node icon",
            description=(
                "Operator-chosen visual identifier for THIS node in the "
                "fabric UI. Emoji or short string."
            ),
            section="fabric",
            max_length=64,
            tags=_FABRIC,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Library publication ==============
    r.register(
        Setting(
            key="library_publication_max_bytes",
            kind="int",
            default=52428800,
            label="Per-publication max bytes",
            description=(
                "Size cap on a single Save-to-Library publication. 50 MB default."
            ),
            section="library.publication",
            min_value=1048576,
            max_value=1073741824,
            tags=("library", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="library_publication_user_budget_bytes",
            kind="int",
            default=1073741824,
            label="Per-user library budget",
            description=(
                "Per-user storage budget for Save-to-Library. 1 GB default."
            ),
            section="library.publication",
            min_value=10485760,
            max_value=107374182400,
            tags=("library", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Cast surface ==============
    r.register(
        Setting(
            key="cast_gallery_show_private",
            kind="bool",
            default=False,
            label="Show private in Cast gallery",
            description=(
                "By default private images are hidden from the Cast gallery "
                "to keep shared TVs safe. Flip to show them."
            ),
            section="cast.gallery",
            tags=("cast",),
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="cast_comic_library_ceiling",
            kind="int",
            default=200_000,
            label="Cast comic library ceiling",
            description=(
                "Soft ceiling on cast comic rail chapter count. Raise if you "
                "have a legitimately huge library."
            ),
            section="cast.gallery",
            min_value=1000,
            max_value=10_000_000,
            tags=("cast", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Community install / TV / startup / misc ==============
    r.register(
        Setting(
            key="community_install_enabled",
            kind="bool",
            default=True,
            label="Community install (Open in Augmentum)",
            description=(
                "Master switch for the 'Open in Augmentum' community-install "
                "deep link. Off = link refuses installation."
            ),
            section="community",
            tags=("community",),
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="tv_auto_update",
            kind="bool",
            default=True,
            label="TV auto-update",
            description=(
                "Auto-pull TV catalog updates on startup."
            ),
            section="tv",
            tags=("tv",),
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="startup_warmup",
            kind="bool",
            default=True,
            label="Startup warmup",
            description=(
                "Preload embedding + reranker models in background on "
                "startup so first-query latency is low."
            ),
            section="system",
            tags=("system", "advanced"),
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="metrics_enabled",
            kind="bool",
            default=True,
            label="Metrics collection",
            description=(
                "Collect internal latency / token-throughput metrics for "
                "the diagnostics dashboard."
            ),
            section="system",
            tags=("system",),
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="web_search_topic_hints_enabled",
            kind="bool",
            default=False,
            label="Web-search topic hints",
            description=(
                "When True, web_search appends curated site: hints derived "
                "from the conversation's topic. Improves precision for "
                "specialized topics."
            ),
            section="search.pipeline",
            tags=("search", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="role_min_param_billions",
            kind="float",
            default=1.0,
            label="Role min size (B params)",
            description=(
                "Minimum model size (in billions of parameters) considered "
                "for role-based model selection. Below this, models are "
                "filtered out of role pools."
            ),
            section="system.routing",
            min_value=0.0,
            max_value=200.0,
            tags=("system", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Location / timezone (user identity-adjacent) ==============
    r.register(
        Setting(
            key="timezone",
            kind="str",
            default="",
            label="Timezone",
            description=(
                "IANA timezone (e.g. 'America/New_York'). Empty = auto-"
                "detect from browser."
            ),
            section="general",
            max_length=64,
            tags=("general",),
            voice_aliases=("timezone", "time zone"),
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="location",
            kind="str",
            default="",
            label="Location",
            description=(
                "User location (e.g. 'Portland, OR') for geo-aware search."
            ),
            section="general",
            max_length=128,
            tags=("general",),
            voice_aliases=("location", "where i am"),
            trust_tier="local_reversible",
        )
    )

    # ============== Model role pointers ==============
    r.register(
        Setting(
            key="primary_chat_model",
            kind="str",
            default="",
            label="Primary chat model",
            description=(
                "Model used for chat by default when no per-mode override "
                "is set. Anthropic /v1/messages model aliasing also derives "
                "from this."
            ),
            section="system.routing",
            max_length=256,
            tags=("system", "model"),
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="heavyweight_model",
            kind="str",
            default="",
            label="Heavyweight model",
            description=(
                "Model used for high-stakes one-shot inferences (subagent "
                "verifier, second-opinion, etc.). Empty = use primary."
            ),
            section="system.routing",
            max_length=256,
            tags=("system", "model", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="utility_model",
            kind="str",
            default="",
            label="Utility model",
            description=(
                "Model used for internal utility tasks (memory extraction, "
                "title generation, etc.). Empty = use primary. Pick a small "
                "fast model to cut tokens."
            ),
            section="system.routing",
            max_length=256,
            tags=("system", "model", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="classifier_model",
            kind="str",
            default="",
            label="Classifier model",
            description=(
                "Model used for routing / classification calls. Empty = "
                "fall back to utility."
            ),
            section="system.routing",
            max_length=256,
            tags=("system", "model", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== STT routing ==============
    r.register(
        Setting(
            key="stt_routing_mode",
            kind="enum",
            default="auto",
            label="STT routing mode",
            description=(
                "How voice routes pick STT providers. 'auto' = policy-driven; "
                "'pin' = always use stt_routing_pin_provider."
            ),
            section="voice.routing",
            enum_values=("auto", "pin"),
            max_length=16,
            tags=("voice", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="stt_routing_pin_provider",
            kind="str",
            default="",
            label="Pinned STT provider",
            description=(
                "When stt_routing_mode='pin', the provider ID to use "
                "exclusively. Empty = no pin."
            ),
            section="voice.routing",
            max_length=256,
            tags=("voice", "advanced"),
            advanced=True,
        )
    )

    # ============== TTS Kokoro remainder ==============
    r.register(
        Setting(
            key="tts_kokoro_prosody",
            kind="bool",
            default=True,
            label="Kokoro prosody steering",
            description=(
                "Dynamic prosodic steering — text-aware embedding modulation. "
                "Small per-chunk cost; safer to leave on."
            ),
            section="voice.tts.kokoro",
            tags=("voice", "quality"),
        )
    )
    r.register(
        Setting(
            key="tts_kokoro_quality",
            kind="enum",
            default="int8",
            label="Kokoro quality",
            description=(
                "Quantization for the Kokoro model. 'int8' = CPU-friendly + "
                "fast; 'fp16' = GPU-friendly + slightly higher quality."
            ),
            section="voice.tts.kokoro",
            enum_values=("int8", "fp16"),
            max_length=8,
            tags=("voice", "advanced"),
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
        )
    )

    # ============== Chromium ==============
    r.register(
        Setting(
            key="chromium_binary_path",
            kind="str",
            default="",
            label="Chromium binary path",
            description=(
                "Override the chrome/chromium path used by tools that need "
                "headless browser execution. Empty = auto-discover."
            ),
            section="system",
            max_length=512,
            tags=("system", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== TV update channel ==============
    r.register(
        Setting(
            key="tv_update_channel",
            kind="enum",
            default="stable",
            label="TV update channel",
            description=(
                "Which update channel the cast TV pulls from."
            ),
            section="tv",
            enum_values=("stable", "beta", "dev"),
            max_length=16,
            tags=("tv", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
