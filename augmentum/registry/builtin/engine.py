"""Engine v2 (llama-server subprocess) settings.

All user-tunable engine_* knobs migrated into the declarative substrate.
Every entry is mirrored by the runtime overlay in
``augmentum/proxy/config_routes.py`` into ``_TOOL_SETTINGS`` /
``_STRING_SETTINGS`` so PUT /api/config/* validation flows unchanged.

Range and default values must match ``config.py`` and the literal dicts
in ``config_routes.py`` — drift is caught by ``augmentum.registry.verify``.

Spec: docs/superpowers/specs/2026-06-04-declarative-action-substrate-design.md
"""

from __future__ import annotations

from augmentum.registry.registry import SettingsRegistry
from augmentum.registry.settings import Setting


def _resolve_multislot(persisted: bool | None) -> bool:
    """Resolver for the engine_multislot_enabled tristate. Matches the
    lambda in ``config_routes._TRI_STATE_BOOL_SETTINGS`` — None routes
    to the codebase-default constant."""
    if persisted is None:
        from augmentum.proxy.status_bus import MULTISLOT_DEFAULT_ENABLED

        return MULTISLOT_DEFAULT_ENABLED
    return bool(persisted)


def register(r: SettingsRegistry) -> None:
    # ---- Template / reasoning format ----
    r.register(
        Setting(
            key="engine_use_jinja_template",
            kind="bool",
            default=True,
            label="Use Jinja chat template",
            description=(
                "Force llama-server to use the GGUF's embedded Jinja chat "
                "template. Required for correct thinking-mode behavior on "
                "modern reasoning models. Disable only if a specific GGUF "
                "has a buggy embedded template."
            ),
            section="engine.template",
            tags=("engine", "advanced"),
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )

    r.register(
        Setting(
            key="engine_reasoning_format",
            kind="enum",
            default="deepseek",
            label="Reasoning format",
            description=(
                "How llama-server formats reasoning tokens in the response. "
                "'deepseek' extracts <think>...</think> into the OpenAI-compat "
                "reasoning_content field. 'none' leaves them inline."
            ),
            section="engine.template",
            enum_values=("deepseek", "none"),
            max_length=16,
            tags=("engine", "advanced"),
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )

    # ---- KV cache ----
    r.register(
        Setting(
            key="engine_kv_cache_type",
            kind="enum",
            default="",
            label="KV cache quantization",
            description=(
                "Quantize the KV cache to save VRAM. Empty = no quant (best "
                "quality). 'q8_0' / 'q4_0' / 'f16' trade quality for capacity."
            ),
            section="engine.kv",
            enum_values=("", "q8_0", "q4_0", "f16"),
            max_length=8,
            tags=("engine", "advanced"),
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="engine_kv_ttl_days",
            kind="int",
            default=2,
            label="KV snapshot TTL (days)",
            description=(
                "Sliding TTL for warm KV snapshots. Older snapshots are "
                "garbage-collected. 0 = never expire."
            ),
            section="engine.kv",
            min_value=0,
            max_value=365,
            tags=("engine", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="engine_kv_narrative_ttl_days",
            kind="int",
            default=7,
            label="Narrative KV TTL (days)",
            description=(
                "Narrative sessions get a longer warm-cache window than "
                "regular chat — they benefit more from continuity."
            ),
            section="engine.kv",
            min_value=0,
            max_value=365,
            tags=("engine", "advanced", "narrative"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="engine_kv_max_snapshots_per_model",
            kind="int",
            default=8,
            label="Max KV snapshots per model",
            description=(
                "Hard cap on warm snapshots per model. Secondary safety "
                "rail beyond TTL cleanup — caps disk growth."
            ),
            section="engine.kv",
            min_value=1,
            max_value=100,
            tags=("engine", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="engine_kv_auto_pin_narrative",
            kind="bool",
            default=False,
            label="Pin narrative KV cache",
            description=(
                "Protect narrative session KV snapshots from TTL eviction. "
                "Long-running narrative chats benefit; eats more disk."
            ),
            section="engine.kv",
            tags=("engine", "advanced", "narrative"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="engine_kv_warm_on_start",
            kind="bool",
            default=True,
            label="Warm KV on model load",
            description=(
                "Hydrate slot 0 with the most-recently-used compatible "
                "session right after a model load. Cuts first-prompt latency."
            ),
            section="engine.kv",
            tags=("engine", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ---- Runtime / lifecycle ----
    r.register(
        Setting(
            key="engine_idle_timeout",
            kind="float",
            default=600.0,
            label="Idle unload timeout (s)",
            description=(
                "Seconds of inactivity before auto-unloading a model. "
                "0 = never auto-unload. Frees VRAM during long idles."
            ),
            section="engine.runtime",
            min_value=0.0,
            max_value=86400.0,
            tags=("engine",),
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="engine_health_timeout",
            kind="float",
            default=900.0,
            label="Model load timeout (s)",
            description=(
                "Maximum seconds to wait for llama-server health-check after "
                "a model load. Large GGUFs over Docker bind-mounts (WSL2 9P "
                "/ virtiofs) can take several minutes."
            ),
            section="engine.runtime",
            min_value=60.0,
            max_value=1800.0,
            tags=("engine", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="engine_flash_attn",
            kind="bool",
            default=True,
            label="Flash attention",
            description=(
                "Enable flash-attention kernels in llama-server. Major speedup "
                "on supported hardware; safe to leave on."
            ),
            section="engine.runtime",
            tags=("engine",),
            restart_required=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="engine_parallel_slots",
            kind="int",
            default=0,
            label="Parallel slots",
            description=(
                "Number of concurrent llama-server slots. 0 = single-slot "
                "(simplest, lowest VRAM). >1 enables concurrent requests at "
                "the cost of slot×ctx_size VRAM."
            ),
            section="engine.runtime",
            min_value=0,
            max_value=32,
            tags=("engine", "advanced"),
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="engine_multislot_enabled",
            kind="tristate",
            default=None,
            label="Multislot KV cache",
            description=(
                "Multi-slot KV cache architecture toggle. None = follow "
                "codebase recommendation (env-tuned). True/False overrides."
            ),
            section="engine.runtime",
            tristate_resolver=_resolve_multislot,
            tags=("engine", "advanced"),
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="engine_cache_ram_mib",
            kind="int",
            default=0,
            label="Host-RAM KV cache (MiB)",
            description=(
                "Size of the host-RAM warm-tier cache for evicted slot KV. "
                "0 = disabled. Useful for very long context that exceeds VRAM."
            ),
            section="engine.runtime",
            min_value=0,
            max_value=65536,
            tags=("engine", "advanced"),
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="engine_auto_pair_mmproj",
            kind="bool",
            default=False,
            label="Auto-pair vision projector",
            description=(
                "When loading a vision-capable GGUF, llama_server_manager "
                "auto-attaches the paired mmproj file from the same directory."
            ),
            section="engine.runtime",
            tags=("engine", "vision"),
            restart_required=True,
            trust_tier="admin_only",
        )
    )

    # ---- MTP (multi-token prediction) ----
    r.register(
        Setting(
            key="engine_mtp_enabled",
            kind="bool",
            default=False,
            label="MTP self-speculation",
            description=(
                "Enable multi-token prediction self-speculation. Requires "
                "an MTP-headed GGUF; runtime gate auto-disables if missing."
            ),
            section="engine.mtp",
            tags=("engine", "advanced"),
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="engine_mtp_n_max",
            kind="int",
            default=2,
            label="MTP max draft tokens",
            description=(
                "Maximum tokens drafted per MTP step. Higher = potentially "
                "faster but more wasted work if drafts mis-predict."
            ),
            section="engine.mtp",
            min_value=1,
            max_value=16,
            tags=("engine", "advanced"),
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
        )
    )

    # ---- Reasoning ----
    r.register(
        Setting(
            key="engine_reasoning_budget",
            kind="int",
            default=0,
            label="Reasoning token budget",
            description=(
                "Cap on hidden chain-of-thought tokens per turn. 0 = no cap. "
                "Forces the model to wrap up reasoning before this threshold."
            ),
            section="engine.reasoning",
            min_value=0,
            max_value=131072,
            tags=("engine", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="engine_reasoning_grace_period",
            kind="int",
            default=0,
            label="Reasoning grace period",
            description=(
                "After reasoning_budget is exhausted, allow this many extra "
                "tokens before forcing the closing think-block tag. 0 = no grace."
            ),
            section="engine.reasoning",
            min_value=0,
            max_value=8192,
            tags=("engine", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
