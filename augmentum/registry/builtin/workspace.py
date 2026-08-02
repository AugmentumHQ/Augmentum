"""Workspace-facing settings — files, dream, app builder, body
physics, vision provider, ghost text, tool execution.
"""

from __future__ import annotations

from augmentum.registry.registry import SettingsRegistry
from augmentum.registry.settings import Setting

_FILES = ("files",)
_DREAM = ("dream",)
_BODY = ("body_physics",)


def register(r: SettingsRegistry) -> None:
    # ============== Files / VFS ==============
    r.register(
        Setting(
            key="files_webdav_enabled",
            kind="bool",
            default=True,
            label="WebDAV access",
            description=(
                "Expose the user's file index over WebDAV. Off = file access "
                "only through the HTTP API and UI."
            ),
            section="files.access",
            tags=_FILES,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="files_enrichment_enabled",
            kind="bool",
            default=True,
            label="File enrichment",
            description=(
                "Generate thumbnails + extracted-text descriptions for "
                "uploaded files. Disable to save CPU on import."
            ),
            section="files",
            tags=_FILES,
        )
    )
    r.register(
        Setting(
            key="files_max_thumbnail_px",
            kind="int",
            default=200,
            label="Thumbnail max (px)",
            description=(
                "Maximum thumbnail dimension. Higher = clearer previews, "
                "more disk."
            ),
            section="files",
            min_value=50,
            max_value=500,
            tags=("files", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="files_description_max_chars",
            kind="int",
            default=500,
            label="Description max chars",
            description=(
                "How many characters of extracted text are stored as a "
                "file's searchable description."
            ),
            section="files",
            min_value=100,
            max_value=2000,
            tags=("files", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="files_search_limit",
            kind="int",
            default=20,
            label="File search limit",
            description=(
                "Maximum results returned from file search."
            ),
            section="files",
            min_value=5,
            max_value=100,
            tags=_FILES,
        )
    )
    r.register(
        Setting(
            key="files_upload_max_file_bytes",
            kind="int",
            default=100 * 1024 * 1024,
            label="Per-file upload cap (bytes)",
            description=(
                "Hard cap on a single file upload. 100 MB default."
            ),
            section="files.upload",
            min_value=1024 * 1024,
            max_value=10 * 1024 * 1024 * 1024,
            tags=("files", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="files_upload_max_files_per_request",
            kind="int",
            default=200,
            label="Files per upload request",
            description=(
                "Maximum file count in a single POST /api/files/upload."
            ),
            section="files.upload",
            min_value=1,
            max_value=1000,
            tags=("files", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="files_upload_max_request_bytes",
            kind="int",
            default=500 * 1024 * 1024,
            label="Per-request upload cap (bytes)",
            description=(
                "Aggregate cap across a single upload request. 500 MB default."
            ),
            section="files.upload",
            min_value=1024 * 1024,
            max_value=50 * 1024 * 1024 * 1024,
            tags=("files", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="files_user_storage_quota_bytes",
            kind="int",
            default=10 * 1024 * 1024 * 1024,
            label="Per-user storage quota (bytes)",
            description=(
                "Soft per-user storage ceiling. 0 = unlimited. 10 GB default."
            ),
            section="files.upload",
            min_value=0,
            max_value=1024 * 1024 * 1024 * 1024,
            tags=("files", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Dream ==============
    r.register(
        Setting(
            key="dream_compaction_enabled",
            kind="bool",
            default=True,
            label="Dream compaction",
            description=(
                "Periodically compact dream entries (deduplicate + cluster + "
                "summarize). Required for long-running installs."
            ),
            section="dream.compaction",
            tags=_DREAM,
        )
    )
    r.register(
        Setting(
            key="dream_compaction_interval_hours",
            kind="float",
            default=12.0,
            label="Compaction cadence (h)",
            description=(
                "How often compaction runs. Cheap; safe to leave at default."
            ),
            section="dream.compaction",
            min_value=1.0,
            max_value=168.0,
            tags=("dream", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="dream_dedup_threshold",
            kind="float",
            default=0.85,
            label="Dream dedup threshold",
            description=(
                "Similarity above which two dream entries are merged into one."
            ),
            section="dream.compaction",
            min_value=0.5,
            max_value=0.99,
            tags=("dream", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="dream_cluster_threshold",
            kind="float",
            default=0.65,
            label="Dream cluster threshold",
            description=(
                "Similarity above which entries are grouped into a "
                "thematic cluster for summarization."
            ),
            section="dream.compaction",
            min_value=0.4,
            max_value=0.95,
            tags=("dream", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="dream_cluster_min_size",
            kind="int",
            default=3,
            label="Dream cluster min size",
            description=(
                "Minimum entries per cluster before it qualifies for "
                "cluster-summarization."
            ),
            section="dream.compaction",
            min_value=2,
            max_value=20,
            tags=("dream", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="dream_compaction_max_clusters_per_run",
            kind="int",
            default=5,
            label="Cluster cap per run",
            description=(
                "Per-pass cap so an install with hundreds of clusters doesn't "
                "burn the entire LLM budget on compaction."
            ),
            section="dream.compaction",
            min_value=1,
            max_value=50,
            tags=("dream", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="dream_consolidation_low",
            kind="float",
            default=0.65,
            label="Consolidation low",
            description=(
                "Lower bound of the on-write consolidation similarity band — "
                "new entries within this range get merged into the closest "
                "existing entry."
            ),
            section="dream.compaction",
            min_value=0.4,
            max_value=0.9,
            tags=("dream", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="dream_consolidation_high",
            kind="float",
            default=0.85,
            label="Consolidation high",
            description=(
                "Upper bound of the on-write consolidation similarity band."
            ),
            section="dream.compaction",
            min_value=0.5,
            max_value=0.99,
            tags=("dream", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="dream_time_trim_count_threshold",
            kind="int",
            default=200,
            label="Time-trim threshold",
            description=(
                "Time-trim only fires above this entry count. Below it, "
                "semantic compaction is the only pruning mechanism."
            ),
            section="dream.compaction",
            min_value=50,
            max_value=10000,
            tags=("dream", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="dream_compaction_max_age_days",
            kind="int",
            default=30,
            label="Time-trim max age (days)",
            description=(
                "Age cutoff once the count threshold is exceeded. Older "
                "entries are pruned first."
            ),
            section="dream.compaction",
            min_value=7,
            max_value=3650,
            tags=("dream", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== App Builder ==============
    r.register(
        Setting(
            key="app_builder_improve_pass",
            kind="bool",
            default=True,
            label="App-builder improve pass",
            description=(
                "Run an LLM-driven 'improve' pass after the initial app "
                "generation completes."
            ),
            section="app_builder",
            tags=("app_builder", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="app_builder_max_improve_iterations",
            kind="int",
            default=2,
            label="Improve iteration cap",
            description=(
                "Maximum improve passes per generation run."
            ),
            section="app_builder",
            min_value=0,
            max_value=5,
            tags=("app_builder", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="app_builder_max_fix_iterations",
            kind="int",
            default=4,
            label="Fix iteration cap",
            description=(
                "Maximum fix-error passes when verification fails."
            ),
            section="app_builder",
            min_value=1,
            max_value=8,
            tags=("app_builder", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="app_builder_auto_preview",
            kind="bool",
            default=True,
            label="Auto-preview",
            description=(
                "Open the generated app preview automatically when build "
                "completes."
            ),
            section="app_builder",
            tags=("app_builder",),
        )
    )
    r.register(
        Setting(
            key="app_builder_max_tokens",
            kind="int",
            default=8192,
            label="App-builder max tokens",
            description=(
                "Per-call token cap for the build pipeline."
            ),
            section="app_builder",
            min_value=1024,
            max_value=32768,
            tags=("app_builder", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="app_builder_llm_timeout_seconds",
            kind="int",
            default=600,
            label="Build LLM timeout (s)",
            description=(
                "Per-LLM-call timeout for the build pipeline. Local "
                "30B+ models on cold KV cache routinely take 4-6 minutes."
            ),
            section="app_builder",
            min_value=60,
            max_value=3600,
            tags=("app_builder", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Body physics ==============
    r.register(
        Setting(
            key="body_physics_enabled",
            kind="bool",
            default=False,
            label="Body physics",
            description=(
                "Enable VRM body physics (collision, soft-body, gravity). "
                "Beta — opt in."
            ),
            section="body_physics",
            tags=_BODY,
            voice_aliases=("body physics", "avatar physics"),
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="body_physics_audio_reactions_enabled",
            kind="bool",
            default=True,  # matches _TOOL_SETTING_DEFAULTS
            label="Audio-reactive physics",
            description=(
                "Modulate body physics in time with TTS prosody (chest "
                "expansion on emphasis, etc.)."
            ),
            section="body_physics",
            tags=_BODY,
        )
    )
    r.register(
        Setting(
            key="body_physics_visual_feedback_enabled",
            kind="bool",
            default=True,  # matches _TOOL_SETTING_DEFAULTS
            label="Visual feedback",
            description=(
                "Render collision/soft-body debug overlays."
            ),
            section="body_physics",
            tags=("body_physics", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="body_physics_velocity_aware",
            kind="bool",
            default=True,
            label="Velocity-aware",
            description=(
                "Scale physics response by the avatar's movement velocity."
            ),
            section="body_physics",
            tags=("body_physics", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="body_physics_compliance_gain",
            kind="float",
            default=1.0,
            label="Compliance gain",
            description=(
                "Scale factor on soft-body compliance. Higher = bouncier."
            ),
            section="body_physics",
            min_value=0.0,
            max_value=2.0,
            tags=("body_physics", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="body_physics_rapier_weight",
            kind="float",
            default=0.6,  # matches _TOOL_SETTING_DEFAULTS
            label="Rapier weight",
            description=(
                "Blend factor between Rapier physics and the analytical "
                "fallback. 0 = analytical only; 1 = Rapier only."
            ),
            section="body_physics",
            min_value=0.0,
            max_value=2.0,
            tags=("body_physics", "advanced"),
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="body_physics_recover_hz",
            kind="float",
            default=6.0,  # matches _TOOL_SETTING_DEFAULTS
            label="Recovery rate (Hz)",
            description=(
                "How quickly soft-body recovers to rest pose."
            ),
            section="body_physics",
            min_value=2.0,
            max_value=20.0,
            tags=("body_physics", "advanced"),
            advanced=True,
        )
    )

    # ============== Vision provider ==============
    r.register(
        Setting(
            key="vision_provider_enabled",
            kind="bool",
            default=False,
            label="Vision provider sidecar",
            description=(
                "Bring up the SmolVLM-based vision sibling llama-server. "
                "Off by default; flip on after pulling the vision GGUF."
            ),
            section="vision",
            tags=("vision",),
            restart_required=True,
            trust_tier="admin_only",
        )
    )
    # vision_provider_gpu_layers retired 2026-06-19: the CPU fallback is
    # CPU-by-definition; GPU vision is the classifier slot's job (Slot C).
    r.register(
        Setting(
            key="vision_provider_backend_port",
            kind="int",
            default=8092,
            label="Vision backend port",
            description=(
                "llama-server port for the vision sibling subprocess "
                "(primary engine uses 8091)."
            ),
            section="vision",
            min_value=1024,
            max_value=65535,
            tags=("vision", "advanced"),
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="vision_provider_model_path",
            kind="str",
            default="/models/vision/SmolVLM-256M-Instruct-Q8_0.gguf",
            label="Vision model path",
            description=(
                "Path to the vision GGUF. Default points to the baked-in "
                "Dockerfile.gpu location."
            ),
            section="vision",
            max_length=512,
            tags=("vision", "advanced"),
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="vision_provider_mmproj_path",
            kind="str",
            default="/models/vision/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf",
            label="Vision projector path",
            description=(
                "Path to the vision mmproj file paired with the vision GGUF."
            ),
            section="vision",
            max_length=512,
            tags=("vision", "advanced"),
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
        )
    )

    # ============== Ghost text ==============
    r.register(
        Setting(
            key="ghost_text_enabled",
            kind="bool",
            default=False,
            label="Ghost-text completions",
            description=(
                "LLM-powered inline suggestions in the code editor. "
                "Disabled by default — turns on once you've selected a model."
            ),
            section="ghost_text",
            tags=("ghost_text",),
            voice_aliases=("ghost text", "code completions"),
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="ghost_text_model",
            kind="str",
            default="",
            label="Ghost-text model",
            description=(
                "Model used for ghost-text completions. Empty = use the "
                "current chat model."
            ),
            section="ghost_text",
            max_length=256,
            tags=("ghost_text", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Tool execution ==============
    r.register(
        Setting(
            key="tool_result_max_chars",
            kind="int",
            default=20000,
            label="Tool result max chars",
            description=(
                "Maximum characters per tool result injected into context. "
                "Larger results get truncated."
            ),
            section="tools.execution",
            min_value=1000,
            max_value=128000,
            tags=("tools", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="tool_execution_timeout",
            kind="float",
            default=120.0,
            label="Tool execution timeout (s)",
            description=(
                "Per-tool execution timeout. Tools that overrun this are "
                "cancelled."
            ),
            section="tools.execution",
            min_value=10.0,
            max_value=600.0,
            tags=("tools", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Avatar ==============
    r.register(
        Setting(
            key="avatar_enabled",
            kind="bool",
            default=False,
            label="3D avatar",
            description=(
                "Render Becca's VRM 3D avatar in the companion surface. "
                "Off = flat-card presentation."
            ),
            section="avatar",
            tags=("avatar",),
            voice_aliases=("avatar", "3d avatar"),
            trust_tier="local_reversible",
        )
    )
