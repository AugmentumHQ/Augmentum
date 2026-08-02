"""Application configuration via environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings


def _default_data_dir() -> str:
    """Return /data (Docker) if it exists, otherwise a local .data directory."""
    if Path("/data").exists():
        return "/data"
    return str(Path(os.environ.get("AUGMENTUM_DATA_DIR", ".data")).resolve())


class Settings(BaseSettings):
    model_config = {"env_prefix": "AUGMENTUM_"}

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 6100
    log_level: str = "INFO"
    timezone: str = ""  # IANA timezone (e.g. "America/New_York"); empty = auto-detect
    location: str = ""  # User location (e.g. "Portland, OR") for geo-aware search
    huggingface_token: str = ""  # HuggingFace API token for gated model downloads
    # GGUF multi-part download tuning. Mirrors Ollama's design (server/download.go):
    # part_size = clamp(total / max_parts, min_part, max_part); concurrency capped
    # at max_parts. Default 8 saturates a residential gigabit link without inviting
    # HF tarpitting; bump to 16 for fat pipes.
    gguf_download_max_parts: int = 8
    gguf_download_min_part_mb: int = 100
    gguf_download_max_part_mb: int = 1000
    gguf_download_part_max_retries: int = 6
    gguf_download_stall_threshold_s: float = 30.0

    # --- Auth ---
    auth_session_ttl_hours: int = 720  # 30 days
    auth_lockout_threshold: int = 5
    auth_lockout_minutes: int = 15
    auth_ip_lockout_threshold: int = 10
    auth_ip_lockout_minutes: int = 60
    auth_ws_ticket_ttl_seconds: int = 30
    auth_max_sessions_per_user: int = 10
    # When the server runs behind a reverse proxy that sets X-Forwarded-For,
    # set this to True so login lockout / audit logs key off the real client
    # IP. Leaving it False (default) is correct for direct exposure — an
    # attacker can otherwise spoof the header to rotate IPs and bypass the
    # IP-based lockout. Only enable when you control the upstream proxy.
    auth_trust_forwarded_for: bool = False

    # --- Front gate (identity-aware reverse proxy) ---
    # When set to an Augmentum-DEDICATED domain resolved by INTERNAL DNS to
    # this box (e.g. "aug.lan"), provisioned services become reachable at
    # "<service>.<gate_domain>" behind a single auth gate: a logged-in
    # Augmentum user is trusted (login dissolved via server-side credential
    # injection); an outsider hitting the service's own port still meets its
    # own auth. Empty (default) = gate disabled, all behavior unchanged.
    # MUST NOT be a public A-record. Widens the session cookie to this domain
    # (stays HttpOnly+Secure; stripped at the proxy edge before any upstream).
    # See docs/superpowers/specs/2026-06-19-front-gate-identity-aware-proxy-design.md
    gate_domain: str = ""

    # Host interface that provisioned-service container ports bind to on the
    # Docker host. Mirrors the compose AUGMENTUM_BIND_HOST used for the app's
    # own ports: "127.0.0.1" (default) keeps a provisioned media server
    # loopback-only; set "0.0.0.0" to expose it on the LAN for native apps.
    # The front gate / front door reach containers over the internal Docker
    # network regardless, so this only governs raw host-port exposure.
    bind_host: str = "127.0.0.1"

    # --- TV / Cast public-host override ---
    # When augmentum hands a URL to a TV (cast blob, generated image, etc.)
    # the URL must be reachable from the TV. If the user opens augmentum
    # via `localhost`, that name means the TV's own loopback, which is wrong.
    # The PublicHostResolver auto-learns the LAN-reachable host from any
    # request that arrives via a non-loopback Host header; this setting
    # supplies an explicit override for cases where auto-learn isn't enough
    # (e.g. running entirely from `localhost` and never accessing via LAN
    # from a phone first). Format: "192.168.1.10:6443" or "augmentum.lan".
    augmentum_public_host: str = Field(
        default="",
        validation_alias=AliasChoices(
            "AUGMENTUM_PUBLIC_HOST",
            "AUGMENTUM_AUGMENTUM_PUBLIC_HOST",
        ),
    )

    # --- Connect: durable off-network guest transport (Tailscale Funnel) ---
    # These make the TS_FUNNEL reachability tier real, so an "Anywhere" invite
    # prefers a stable, identity-bound ts.net public address over the anonymous
    # cloudflared last resort. All optional; empty = tier unavailable, planner
    # falls to cloudflared. See augmentum/connect/reachability.py + funnel_manager.py.
    #
    # The node's <node>.<tailnet>.ts.net MagicDNS name (auto-detected by
    # start.sh/start.bat when Tailscale is present). Used for the tailnet static
    # tier AND config-mode funnel URL derivation.
    augmentum_tailnet_hostname: str = Field(
        default="",
        validation_alias=AliasChoices("AUGMENTUM_TAILNET_HOSTNAME"),
    )
    # Explicit standing funnel URL (e.g. https://node.tailnet.ts.net:8443) — wins
    # over derivation. Set this when the operator enabled funnel host-side.
    augmentum_connect_funnel_url: str = Field(
        default="",
        validation_alias=AliasChoices("AUGMENTUM_CONNECT_FUNNEL_URL"),
    )
    # Funnel port for URL derivation / live-drive. Tailscale Funnel allows only
    # 443/8443/10000; empty lets the live manager pick a free one.
    augmentum_connect_funnel_port: str = Field(
        default="",
        validation_alias=AliasChoices("AUGMENTUM_CONNECT_FUNNEL_PORT"),
    )
    # Opt in to LIVE-drive mode: the app shells `tailscale funnel` itself (only
    # works where it can reach tailscaled — a sidecar / host-network Linux
    # deploy). Off by default; config mode (URL from env) is the primary path.
    augmentum_connect_funnel_live: bool = Field(
        default=False,
        validation_alias=AliasChoices("AUGMENTUM_CONNECT_FUNNEL_LIVE"),
    )

    @field_validator("augmentum_connect_funnel_live", mode="before")
    @classmethod
    def _empty_env_is_unset(cls, v: object) -> object:
        # compose.yaml passes AUGMENTUM_CONNECT_FUNNEL_LIVE=${...:-}, so an
        # unset host var arrives as "" — that means "use the default", not a
        # bool-parse error (which crash-loops the container at import time).
        # Any non-str env fed by a `:-` empty compose default needs this.
        if isinstance(v, str) and not v.strip():
            return False
        return v

    # --- Connect: instance identity ---
    # The public name of THIS Augmentum for peer-DID addressing. Connect
    # DIDs render as ``<local-part>@<instance_handle>``. When empty, the
    # handle is derived from ``augmentum_public_host`` (stripped of scheme
    # /port/path); when that is also empty it falls back to the legacy
    # ``this-instance`` sentinel so nothing breaks pre-configuration.
    # Admin-editable via /api/config/tools (registered in _STRING_SETTINGS).
    # See augmentum/connect/contacts.py::instance_handle.
    connect_instance_handle: str = Field(
        default="",
        validation_alias=AliasChoices("AUGMENTUM_CONNECT_INSTANCE_HANDLE"),
    )

    # --- Files / VFS ---
    files_webdav_enabled: bool = True
    files_enrichment_enabled: bool = True
    files_max_thumbnail_px: int = 200
    files_description_max_chars: int = 500
    files_search_limit: int = 20
    # Upload limits — enforced in /api/files/upload before bytes hit the blob store.
    # Per-file cap stops a single huge upload; per-request cap and file count
    # bound the total work for one POST. user_storage_quota is a soft per-user
    # ceiling computed from SUM(blobs.size_bytes) for refs owned by the user
    # (so dedup helps users stay under quota).
    files_upload_max_file_bytes: int = 100 * 1024 * 1024            # 100 MB per file
    files_upload_max_files_per_request: int = 200
    files_upload_max_request_bytes: int = 500 * 1024 * 1024         # 500 MB aggregate
    files_user_storage_quota_bytes: int = 10 * 1024 * 1024 * 1024   # 10 GB per user (0 = unlimited)
    # Lifecycle — trash TTL and maintenance loop cadence.  TTL=0 keeps trash
    # forever (manual purge only).  The same loop also sweeps orphan blobs
    # (refcount=0 rows) so out-of-band file_index deletes don't leak storage.
    files_trash_ttl_days: int = 30
    files_maintenance_interval_hours: float = 6.0

    # Transient artifact cache (image_search thumbnails, etc.) — evicted
    # by age first, then by size cap with oldest-first trim.
    transient_artifact_ttl_days: int = 7
    transient_artifact_max_mb: int = 200
    transient_artifact_sweep_hours: float = 24.0

    # --- Backends ---
    ollama_base_url: str = ""
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    llamacpp_base_url: str | None = None
    llamacpp_model_dir: str = "/data/llama_models"
    llamacpp_api_key: str | None = None  # Bearer token for authenticated llama-server
    llamacpp_default_samplers: str = ""  # Default sampler chain order (comma-separated)
    engine_base_url: str = ""  # Augmentum Engine URL (e.g. http://engine:8090)
    default_backend: str = "engine"

    # --- Engine v2 (managed llama-server subprocess) ---
    engine_managed: str = "auto"  # "auto" (detect binary), "true", or "false"
    engine_llama_server_path: str = "/usr/local/bin/llama-server"
    engine_backend_port: int = 8091
    engine_gpu_layers: int = 99
    engine_ctx_size: int = 32384
    engine_batch_size: int = 512
    engine_kv_cache_type: str = ""  # q8_0, q4_0, f16 (empty = default)
    engine_draft_model: str = ""  # Path to draft GGUF for speculative decoding
    engine_draft_max: int = 5
    engine_draft_ctx_size: int = 2048
    engine_draft_gpu_layers: int = 999  # full offload by default; small drafts almost always fit
    engine_draft_min: int = 1
    engine_draft_p_min: float = 0.75
    engine_flash_attn: bool = True
    engine_model_dir: str = "/data/models"
    engine_extra_model_dirs: str = ""  # Semicolon-separated additional dirs
    engine_health_timeout: float = 900.0  # 15 min — large GGUFs over Docker bind-mounts (WSL2 9P/virtiofs) load 2–4× slower than native
    engine_idle_timeout: float = 600.0  # Seconds before auto-unloading idle model (0 = never)
    engine_kv_ttl_days: int = 2  # Sliding TTL for warm KV snapshots (0 = never expire)
    engine_kv_narrative_ttl_days: int = 7  # Narrative sessions benefit from a longer warm cache
    engine_kv_max_snapshots_per_model: int = 8  # Secondary safety rail after TTL cleanup
    engine_kv_auto_pin_narrative: bool = False  # Optional protection for long-running narrative chats
    engine_kv_warm_on_start: bool = True  # Hydrate slot 0 with the MRU compatible session after model load
    engine_kv_replay_enabled: bool = True  # Resume ladder rung 2: replay a stored session prefix (n_predict=0 prewarm) where tensor restore is unavailable — the only cross-restart recovery under --kv-unified
    engine_kv_replay_warm_sessions: int = 2  # Boot-warm session budget (successful warms; single-slot caps at 1)
    engine_kv_replay_budget_s: float = 90.0  # Boot-warm wall-clock budget in seconds (0 = unbounded)
    engine_kv_replay_max_rows: int = 64  # Replay-source store cap (MRU kept; expired rows pruned first)
    engine_speculation_enabled: bool = False  # Resume ladder rung 3: generate the next turn from a typing-pause draft on idle GPU; LOCAL llama-server only — drafts never reach cloud backends
    engine_speculation_prefill_only: bool = False  # Conservative mode: warm the draft's prefix but never generate/serve a speculative answer
    engine_speculation_max_new_tokens: int = 2048  # Decode cap when the session's sampling has none; a speculation stopped by THIS cap is never served (truncation guard)
    engine_speculation_ttl_s: float = 180.0  # Speculation freshness window; older entries never serve
    engine_auto_discover: bool = True  # Auto-scan model_dirs for GGUFs on startup

    # HTTP→HTTPS redirect for non-loopback browser navigations. The HTTP
    # listener on 6100 stays open (curl / loopback / fabric peer probes
    # use it), but when a browser navigates to a UI page over HTTP from
    # a non-localhost host, we 307 to the matching https://host:6443
    # URL. Without this, hitting the LAN URL in a browser shows a
    # half-rendered shell because secure-context-only APIs are off on
    # non-HTTPS origins. Set False if you intentionally serve HTTP-only
    # (you have your own reverse proxy terminating TLS, etc.).
    https_redirect_lan: bool = True
    # Port the redirect points to. Should match Caddy's HTTPS bind in
    # compose.yaml. Override only if you've remapped Caddy.
    https_redirect_port: int = 6443

    # Fabric layer: cross-instance peer coordination. When False (default)
    # no fabric code path executes -- solo installs are bit-for-bit
    # identical to the pre-fabric build. Turning this on doesn't yet do
    # anything by itself (Phase 0 ships identity primitives only); higher
    # phases gate transport, capability advertisement, and routing on
    # this flag. See augmentum/fabric/__init__.py for the design notes.
    fabric_enabled: bool = False
    # Connect federation (the federated-PBX feature). Default OFF — an
    # operator opts in explicitly. When on, this instance can exchange
    # contact cards, run the verification ceremony, accept knocks, and
    # carry sealed cross-instance messages. See docs/connect-federation.md.
    fabric_federation_enabled: bool = False
    # Inbound admission posture for strangers (identities the user hasn't
    # pinned): "private" (none) | "allowlist" | "knock" (default — queue a
    # non-ringing, intro-withheld request) | "open". See fabric/knock.py.
    fabric_admission_posture: str = "knock"
    # Refuse to relay/forward any cross-instance payload that isn't sealed
    # (sign-then-seal). Default ON — the relay must never see cleartext.
    fabric_relay_sealed_only: bool = True
    # End-to-end device-to-device DMs (hosts can't read). Default OFF; the
    # host-trusted path is the v1 posture, E2E is opt-in. See fabric/e2e.py.
    fabric_e2e_dm_enabled: bool = False
    # User REQUEST to include their own sovereign AI as a participant in
    # E2E conversations. This is ONLY a request: the companion is added
    # only if this is on AND the hard code gate
    # fabric/e2e_session.py::COMPANION_E2E_SECURITY_CONFIRMED is lifted
    # (it is not, pending a reviewed security sign-off). So enabling this
    # today leaves the companion ON STANDBY — never silently in a
    # conversation. Default OFF.
    companion_e2e_participant_enabled: bool = False
    # Self-editing master switch — the loop that lets Augmentum improve its OWN
    # source (augmentum/selfedit/). Default OFF; must be explicitly enabled. The
    # propose/loop endpoints refuse while this is off.
    #
    # EXPERIMENTAL (early access) — the newest subsystem here and still moving
    # fast. It's off by default because it differs in kind from everything else:
    # it changes the running system itself. Understand the following before
    # enabling it on an install you care about:
    #   * It writes to a checkout of Augmentum's own source and commits to git.
    #     Changes land as candidate branches and are promoted by cherry-pick /
    #     reverted by `git revert`, so history is preserved and every step is
    #     reversible — but it IS moving your repo around. Don't point it at a
    #     tree holding uncommitted work you can't afford to untangle.
    #   * Verification is honest but partial: the gate proves "didn't break"
    #     (compile/lint/pytest/health), not "is correct". Only mechanically
    #     oracle-confirmed changes are ever eligible for auto-promotion, and
    #     only when selfedit_autonomy_level == "auto_verified". Leave that on
    #     the default "propose" unless you are actively supervising.
    #   * Backend and migration changes can require a restart, and a promoted
    #     backend change can in principle break boot. A crash parachute
    #     (selfedit/rollback.py) exists, but recovery may still mean a manual
    #     `git revert` from outside the app. Have that path available.
    #   * It costs real compute (and real tokens if you configure a frontier
    #     rung), and a run can take a long time.
    # Because this is install-wide and unsupported, the UI keeps it out of the
    # way: the Workshop nav entry is hidden until this flag is on (see
    # data-feature-flag in ui/index.html), so a default install never surfaces
    # it. Bug reports are welcome; behavior guarantees are not offered yet.
    selfedit_enabled: bool = False
    # Autonomy posture: "propose" (every change waits for a human verdict) |
    # "auto_verified" (auto-promote oracle-confirmed changes). See
    # augmentum/selfedit/promote.py::decide_promotion. Default = safest.
    selfedit_autonomy_level: str = "propose"
    # Which engine drives the editing agent for self-edits:
    # "native" (a LOCAL model via Augmentum's own agentic loop — sovereign, no
    # token) | "claude_code" | "codex" (external platforms, credential-gated).
    # See augmentum/selfedit/engine_select.py. Default = native (sovereign).
    selfedit_engine: str = "native"
    # Ingest-all-work: mirror applied coder-mode turns into the self-edit
    # archive as source='coder' rows (augmentum/selfedit/ingest.py), so the
    # never-pruned lineage learns from every stream of real work. Pure data
    # recording — changes no autonomy. Default OFF (shadow-first).
    selfedit_ingest_coder_enabled: bool = False
    # Optional model pin for the native edit engine (e.g. "Qwen3.6-27B-Q4_K_S").
    # Empty = use the "utility" role's model. Quality scales with the model; the
    # gate + verifier are what keep a weaker local agent's output safe.
    selfedit_edit_model: str = ""
    # Frontier model for the TOP rung of the self-edit escalation ladder (e.g. a
    # DeepSeek). Local does the groundwork; on failure we climb to this, carrying
    # findings forward, through the SAME native loop. Cost-gated: only runs when a
    # proposal opts in (allow_frontier). Empty = no frontier rung.
    selfedit_frontier_model: str = ""
    # Max agentic iterations the native edit loop may run per self-edit attempt
    # before it's force-stopped. Higher = the agent can tackle deeper changes
    # but burns more local compute; the gate + verifier still arbitrate the
    # result. Read in server.py's self-edit run path; persisted via settings_store.
    selfedit_max_iters: int = 64
    # Self-heal: on a fixable verification break (broken import/syntax/failing
    # test), how many times to feed the SPECIFIC failure back to the same model
    # for a bounded same-worktree repair BEFORE rejecting or escalating. 0 = off.
    # The cheap repair rung below the escalation ladder; stagnation-guarded so it
    # never loops on an unfixable break.
    selfedit_self_heal_attempts: int = 2
    # Operator-chosen visual identifier for THIS node in the fabric UI.
    # Surfaces in the local-box column of the capability matrix + on
    # outbound chat turns served by this box (when viewed from a
    # peer). Empty string means no icon — UI falls back to 🔗.
    local_fabric_icon: str = ""
    # --jinja makes llama-server use the model's embedded chat template (from
    # tokenizer_config.json). Required for correct thinking-mode behavior on
    # newer reasoning models (GLM-4.7, Qwen3, DeepSeek-R1, etc.) — without
    # it, llama-server's fallback template doesn't recognize the thinking
    # delimiters and reasoning content leaks into the visible response.
    # Default on; disable only if a specific GGUF has a buggy embedded
    # template.
    engine_use_jinja_template: bool = True
    # --reasoning-format extracts <think>...</think> content into the
    # OpenAI-compat ``reasoning_content`` field (instead of leaving it inline).
    # Values: "deepseek" (default; extract), "none" (preserve inline).
    engine_reasoning_format: str = "deepseek"
    # Cap the model's hidden chain-of-thought budget per turn, in tokens.
    # Forwarded as the ``reasoning_budget`` chat-template kwarg to families
    # whose Jinja template branches on it (Nemotron 3 Nano Omni documents
    # 16384 thinking + 1024 grace as the recommended default; other reasoning
    # families silently ignore the kwarg if their template doesn't consume it).
    # 0 (default) = no cap, identical to pre-flag behavior. Range mirrored in
    # config_routes._TOOL_SETTINGS and the UI's load-sheet number input.
    engine_reasoning_budget: int = 0
    engine_reasoning_grace_period: int = 0

    # Multi-slot KV cache architecture (see
    # docs/superpowers/specs/2026-05-05-multi-slot-kv-design.md).
    # Tri-state field:
    #   None  — "auto", follow the codebase's current recommended
    #           default (``MULTISLOT_DEFAULT_ENABLED`` in
    #           ``augmentum.proxy.status_bus``). Default value, used
    #           when the user has not expressed a preference. Flipping
    #           the codebase recommendation moves these users with it
    #           but does not overwrite explicit choices.
    #   True  — explicit "always on", regardless of recommendation.
    #   False — explicit "always off", regardless of recommendation.
    # When the resolved value is True, llama-server runs with
    # ``--parallel -1 --kv-unified --cache-ram ... --cache-idle-slots
    # --ctx-checkpoints 32 ...`` and Augmentum routes via response-
    # observed id_slot. When False, the pre-2026-05-05 single-slot
    # behavior (``--parallel 1``) is preserved.
    engine_multislot_enabled: bool | None = None
    # Number of parallel slots when multi-slot is enabled. ``0`` (default)
    # means let llama-server's ``--parallel -1`` auto-resolve, which at
    # b8935 is hardcoded to 4 with kv_unified=true. Override only for
    # household deployments expecting concurrent users above 2-3, or
    # constrained hardware where you want to cap at 2.
    engine_parallel_slots: int = 0
    # MTP (multi-token prediction) self-speculation. Upstream llama.cpp
    # PR #22673 (merged 2026-05-16) added ``--spec-type draft-mtp`` which
    # uses the model's own MTP heads as the speculation source — no
    # separate draft model. When enabled, the runtime forces
    # ``--parallel 1`` (PR constraint), short-circuits the external
    # draft-model path, AND drops ``--mmproj`` (upstream limitation —
    # vision + MTP not yet supported per the GGUF author's release
    # notes). Requires a llama-server binary at commit 2555826+ —
    # older binaries reject ``--spec-type`` as an unknown argument.
    #
    # n_max default is 2. Empirical Qwen 3.6 27B on a 24 GB consumer GPU
    # (scripts/mtp_bench.py — pinned seed, ctx=16K, q8/q8 KV, single
    # ~400-tok prompt):
    #   no MTP     → 22.9 tok/s @ 20.07 GB   (baseline)
    #   n_max=2    → 34.1 tok/s @ 20.89 GB   1.49× speedup, 78% accept
    #   n_max=6    → 28.5 tok/s @ 21.47 GB   1.24×,        51% accept
    #   n_max=12   → 26.1 tok/s @ 22.35 GB   1.14×,        31% accept
    # Higher n_max LOSES wall-clock speed once acceptance falls below
    # ~60% — rejection-overhead outpaces parallel-verify gains. Range
    # 1-16 enforced in _TOOL_SETTINGS so power users can still tune.
    engine_mtp_enabled: bool = False
    engine_mtp_n_max: int = 2
    # Host-RAM warm-tier cache for evicted slot KV (the L2 between live
    # slots and disk). ``0`` (default) auto-sizes from system RAM:
    # ``min(16384, total_ram_mib * 0.25)``, hard floor 1024. Override
    # only when manual sizing matters (e.g. shared box with limited RAM).
    engine_cache_ram_mib: int = 0
    # llama-server ``--cache-reuse``: minimum chunk size (tokens) to
    # salvage from the KV cache via position-shifting when a prompt
    # diverges MID-history (message edit, regeneration, deep Author's
    # Note). Prefix reuse alone forfeits everything after the first
    # divergent token; chunk reuse recovers the identical tail. Only
    # effective on models whose memory can shift — llama-server itself
    # ignores it otherwise ("cache reuse is not supported"), so hybrid-
    # attention families (Qwen3.5+) are self-gated upstream and see no
    # behavior change. 0 disables; 256 (default) matches upstream's
    # recommended example value.
    engine_cache_reuse_min: int = 256
    # When True, llama_server_manager._find_paired_mmproj auto-attaches a
    # sibling mmproj file found next to the base GGUF. Upstream
    # llama.cpp's slot save/restore endpoints return 501 the moment
    # --mmproj is loaded, so this auto-attach silently disables KV cache
    # persistence (and therefore session restore on follow-up turns).
    # Default False so text-only chats — the common case — keep working.
    # Users who want primary-model vision can enable per-load via the
    # vision_mode option in Load Setup, or pair manually via the
    # /api/models/.../projector endpoint (sidecar takes precedence).
    engine_auto_pair_mmproj: bool = False

    # ── Observation Substrate (BOM) — Phase A ────────────────────────
    # Master flag for the cross-modal pattern memory store. When False,
    # ingestion hooks short-circuit and the cache exporter refuses —
    # makes the substrate genuinely "off" rather than dormant-but-
    # collecting. Default False because Phase A is opt-in for validation.
    observation_substrate_enabled: bool = False
    # When True, the rebuild-cache pipeline seeds from this user's
    # chat history (ui_sessions) before exporting. Opt-in separately
    # from substrate_enabled because the seed is the heaviest write
    # path in the pipeline and should not run on every model swap.
    observation_seed_chat_history: bool = False
    # When True AND the cache file exists for the current (user, model),
    # LlamaServerManager appends --lookup-cache-static. Gated separately
    # so an operator can build and inspect cache files without yet
    # plumbing them into the running llama-server.
    observation_lookup_cache_enabled: bool = False
    # Hard cap on the top-K observations the exporter pulls into the
    # corpus. 50k lines ≈ a few-hundred-KB corpus that builds in seconds.
    # Raise for power users with substantial chat history; lower if
    # disk pressure matters.
    observation_lookup_cache_max_entries: int = 50_000
    # The llama-server subprocess loads exactly one --lookup-cache-static
    # file, so in a multi-tenant install only one user's cache can feed
    # it. This setting names that user. Empty (default) skips the cache
    # wiring entirely even when observation_lookup_cache_enabled=True —
    # avoids accidentally serving one user's text patterns as drafting
    # hints for another user's session. Single-tenant operators set this
    # to their own user id; multi-tenant operators leave it empty until
    # the policy story for cross-user merging is settled.
    observation_primary_user_id: str = ""

    # Provider API keys (env-configured or loaded from settings store)
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    google_api_key: str = ""
    google_vertex: bool = False
    google_vertex_project: str = ""
    google_vertex_region: str = "us-central1"
    cohere_api_key: str = ""
    mistral_api_key: str = ""
    deepseek_api_key: str = ""
    openrouter_api_key: str = ""
    xai_api_key: str = ""
    groq_api_key: str = ""
    perplexity_api_key: str = ""
    fireworks_api_key: str = ""
    azure_api_key: str = ""
    azure_base_url: str = ""
    azure_deployment: str = ""
    azure_api_version: str = "2024-02-01"

    # --- Infrastructure ---
    searxng_base_url: str = "http://searxng:8080"
    executor_base_url: str = "http://executor:5000"
    data_dir: str = _default_data_dir()
    # Production safety: if SQLite can't open the on-disk database after
    # the normal retry loop, the historical behavior was to fall back to
    # an in-memory backend and continue serving — which silently presents
    # the user as if their data is gone (auth fails → 503 storm). With
    # this set to False (default), the server refuses to start when the
    # DB file exists but won't open, surfacing the failure as a clear
    # crash rather than a silent degradation. Set to True only in test
    # contexts or for explicit ephemeral runs.
    allow_inmemory_fallback: bool = False

    # --- Inference Defaults ---
    default_temperature: float | None = None
    default_top_p: float | None = None
    default_top_k: int | None = None
    default_repeat_penalty: float | None = None
    default_min_p: float | None = None
    default_num_predict: int | None = None
    default_num_ctx: int = 8192
    default_stop_sequences: list[str] = []

    # Extended sampling controls (llama.cpp / Ollama power-user features)
    default_dynatemp_range: float | None = None    # Dynamic temperature range (0 = disabled)
    default_dynatemp_exponent: float | None = None # Dynamic temperature exponent
    default_dry_multiplier: float | None = None    # DRY anti-repetition multiplier (0 = disabled)
    default_dry_base: float | None = None          # DRY base value
    default_dry_allowed_length: int | None = None   # DRY allowed run length
    default_dry_penalty_last_n: int | None = None   # DRY penalty window size

    # --- UARF Analytical Tuning ---
    uarf_max_backtracks: int = 3
    uarf_max_tool_calls_per_phase: int = 3
    uarf_confidence_threshold: float = 0.5
    uarf_phase_timeout: float = 120.0
    uarf_proactive_search: bool = True
    uarf_proactive_math: bool = True
    uarf_proactive_code: bool = True
    uarf_auto_search: bool = True
    uarf_auto_search_queries: int = 5
    uarf_auto_search_results_per_query: int = 5
    uarf_auto_search_max_context_chars: int = 24000
    uarf_conversation_turns: int = 5
    uarf_conversation_max_chars: int = 4000
    uarf_search_retry_max: int = 1
    uarf_search_retry_min_results: int = 2
    uarf_tool_tier_override: str | None = None  # "native", "structured", "text", or None (auto)
    uarf_heuristic_assess: bool = True
    uarf_auto_verify: bool = True  # automated tool-based verification (math, code, facts)
    uarf_verify_model: str = ""  # cross-model verification: use different model for VERIFY phase

    # --- Passthrough Tools ---
    passthrough_tools: str = ""             # comma-separated default tools in passthrough (e.g. "web_search,calculator")
    # Default tool-call round-trips per request. Overridable per-turn from
    # the composer's chain control, including 0 = unlimited (which resolves
    # to the 150 backstop in passthrough/handler.py). Read LIVE on every
    # turn — changing it does not require a restart.
    passthrough_tool_max_iterations: int = 5  # max tool call round-trips per request
    # SSOS auto-tools is a per-user preference (ui.autoTools in user_settings),
    # not an install-wide config — see _UI_SETTINGS in proxy/config_routes.py.

    # --- Multi-Model Fan-out (passthrough compare) ---
    multi_model_enabled: bool = False   # composer affordance: fan one user turn out to N models
    multi_model_models: str = ""        # comma-separated compare models (beyond the primary), last-used set

    # --- Passthrough Tool Chains ---
    passthrough_chain_enabled: bool = True
    passthrough_chain_max_steps: int = 10
    passthrough_chain_timeout: float = 120.0         # overall chain timeout
    passthrough_chain_max_parallel: int = 3           # semaphore for wave concurrency
    passthrough_chain_max_flows: int = 50             # max custom flows in DB
    passthrough_chain_max_retries: int = 2            # per-step re-plan attempts
    passthrough_chain_attention_anchor: bool = True   # inject plan progress into LLM context (Manus pattern)
    passthrough_chain_error_as_observation: bool = True  # pass failed step output to dependents instead of cascading
    passthrough_chain_plan_mutation: bool = True      # allow LLM to restructure plan on failure
    passthrough_chain_synthesis_timeout: float = 120.0  # Timeout for chain synthesis LLM call (seconds)

    # --- Background Chain Execution ---
    passthrough_chain_bg_enabled: bool = True         # Enable flow-as-tool + background execution
    passthrough_chain_bg_max_per_session: int = 5     # Max concurrent background chains per session
    passthrough_chain_bg_max_total: int = 50          # Max total background chains across all sessions
    passthrough_chain_bg_result_max_chars: int = 2000 # Max chars of synthesized result to inject

    # --- Tool Pipeline ---
    tool_result_max_chars: int = 20000      # max chars per tool result injected into context
    tool_result_truncation_tail: int = 500  # chars to keep from the end when truncating
    tool_cache_enabled: bool = True         # cache identical tool calls within a session
    tool_cache_max_entries: int = 500        # max entries in tool result cache (LRU eviction)
    tool_circuit_breaker_enabled: bool = True
    tool_circuit_breaker_threshold: int = 3   # consecutive failures before opening
    tool_circuit_breaker_cooldown: float = 60.0  # seconds before retrying
    tool_prefilter_enabled: bool = True     # reduce tool count based on query analysis
    tool_prefilter_min_tools: int = 3       # never filter below this count
    tool_execution_timeout: float = 120.0   # per-tool execution timeout in seconds

    # Analytical mode verification thresholds
    analytical_max_phase_retries: int = 2       # max retries for a single phase on verification failure
    analytical_confidence_threshold: float = 0.5  # confidence below this triggers backtrack
    analytical_max_backtracks: int = 3          # global backtrack limit per request

    # --- Search Expansion ---
    search_expansion_enabled: bool = True          # zero-cost query expansion (synonyms, type reformulation, site scoping)
    search_expansion_max_variants: int = 3         # max expansion variants per original query
    search_expansion_max_total: int = 15           # max total queries after expansion across all originals
    search_credibility_enabled: bool = True        # annotate results with source credibility scores (zero LLM cost)
    search_direct_fetch_enabled: bool = True       # auto-fetch URLs found in user queries (bypass search)
    search_direct_fetch_max_chars: int = 16000      # max chars to extract per directly-fetched page
    search_relevance_filter_enabled: bool = True    # drop search results unrelated to the query (zero LLM cost)
    search_relevance_min_score: float = 0.15        # minimum relevance score (0.0-1.0) to keep a result

    # --- Search Auto-Fetch Enrichment ---
    search_autofetch_enabled: bool = True           # auto-fetch top URLs after web_search for richer content
    search_autofetch_count: int = 2                 # max URLs to auto-fetch per search
    search_autofetch_max_chars: int = 8000          # max chars per fetched page
    search_autofetch_timeout: float = 10.0          # per-fetch timeout (seconds)

    # --- SearXNG Outbound Proxy Routing ---
    # User-supplied proxy URLs (newline-separated): http://, https://, socks5://
    # Each entry may include credentials (http://user:pass@host:port).
    # When non-empty + rotation_enabled, Augmentum picks one healthy proxy as
    # SearXNG's active outgoing proxy and rotates the choice over time.
    search_proxies: str = ""
    search_proxy_rotation_enabled: bool = False
    search_proxy_healthcheck_interval_minutes: int = 5
    # When True, search falls back to direct (no-proxy) connection if every
    # configured proxy is unhealthy. UI surfaces a warning when this happens.
    search_proxy_fallback_direct_enabled: bool = True

    # When True, web_search appends curated site: hints from the topic-keyword
    # dict (e.g., 'python' → site:docs.python.org). Default OFF — measured
    # net-negative on the ablation harness: catastrophic when the targeted
    # engines are CAPTCHA'd, and historically suffered single-word collisions
    # (e.g., 'small space living room ideas' would inject site:nasa.gov).
    # Power users who run with a healthy SearXNG proxy pool can flip this on.
    web_search_topic_hints_enabled: bool = False

    # --- Narrative Tuning ---
    narrative_context_budget: int = 0  # 0 = unlimited; the outer narrative_context_limit is the real window cap
    narrative_max_engines: int = 100        # max cached narrative engines (LRU eviction by last access)
    narrative_character_budget_pct: float = 0.25
    narrative_scene_budget_pct: float = 0.15
    narrative_plot_budget_pct: float = 0.15
    narrative_lore_budget_pct: float = 0.25
    narrative_consistency_budget_pct: float = 0.10
    narrative_auto_persist: bool = True
    narrative_consistency_frequency: int = 5
    narrative_lorebook_scan_depth: int = 10
    world_info_recursive: bool = False
    world_info_max_recursion_steps: int = 5
    world_info_min_activations: int = 0
    world_info_budget_cap: int = 0  # 0 = use percentage only
    world_info_whole_words: bool = False  # Global default for whole-word matching
    narrative_memory_enabled: bool = True
    narrative_memory_interval: int = 4       # STATE+MEMORY refresh every N messages
    narrative_memory_max_tokens: int = 0     # Max tokens for STATE+MEMORY response (0 = auto: 400 lite / 700 standard)
    narrative_memory_mode: str = "standard"  # "lite" (bullets) or "standard" (prose)
    narrative_memory_max_words: int = 0  # Word ceiling for summary (0 = auto-scale per mode)
    narrative_llm_extraction: bool = True  # LLM-based deep extraction for narrative state
    narrative_extraction_interval: int = 5  # Extract every N messages
    narrative_extraction_model: str = ""  # Model for extraction calls (empty = default backend model)
    narrative_memory_model: str = ""  # Model for memory summary calls (empty = default backend model)
    narrative_memory_prompt: str = ""  # Custom system prompt for LTM summary (empty = auto per card type)
    # Card-editor translate button (per-user, persisted)
    narrative_translate_default_language: str = "English"
    narrative_translate_auto_save: bool = True
    # --- Narrative: Feature Toggles ---
    # Disable backend features that duplicate UI-side work or add noise.
    # Kept as toggles so they can be re-enabled for external API usage.
    narrative_consistency_enabled: bool = False   # Regex consistency checking (shallow, can't be used in realtime)
    narrative_state_tracking_enabled: bool = False  # Regex character/world/plot extraction (redundant with three-layer LLM memory)
    narrative_cross_session_memory: bool = False  # Cross-session memory injection (can poison different cards/personas)
    narrative_backend_lorebook: bool = True        # Backend lorebook triggering (auto-inject entries on keyword match)
    narrative_backend_card_summary: bool = False   # Backend card re-parsing/summary (UI already sends card)
    narrative_backend_examples: bool = False        # Backend example/creator note injection (UI handles this)

    # --- Narrative: Recall tools (lookup layer for small-model substrate) ---
    # When True, expose the 5 recall_* verbs as LLM-callable tools so the
    # model can fetch entity / fact / plot / archive details on demand
    # instead of receiving the full STATE snapshot every turn. The data
    # layer (augmentum/modes/narrative/recall.py) is always available via
    # HTTP regardless of this flag; this gates only the LLM tool surface.
    # Default False — opt-in until UI controls land.
    narrative_recall_tools_enabled: bool = False
    narrative_lorebook_tools_enabled: bool = False
    # Hard cap on tool-call iterations per turn. Narrative isn't agentic;
    # 3 lookups is plenty for any single turn. Higher values risk the
    # model getting stuck recall-looping instead of producing prose.
    narrative_recall_tools_max_iters: int = 3
    # Native dot-named lorebook grounding verbs (F1/F5): lorebook.check
    # (mid-scene grounded retrieval) + lorebook.create (record newly-
    # established detail as session-scoped lore, source="narrative_established",
    # never modifies the source card). These are the names the companion
    # training rows emit, so default True — a trained model's tool calls must
    # reach real handlers rather than phantom-calling at inference. Registered
    # in registry/builtin/narrative.py with a matching default.
    narrative_lorebook_native_tools_enabled: bool = True

    # --- Narrative: World systems (card-declared manifest) ---
    # Cards may declare extensions.world_system (trackers/tables/dice/sheet;
    # spec: docs/superpowers/specs/2026-07-15-world-system-manifest-design.md).
    # Absence of a manifest means the feature is invisible; this toggle exists
    # so a manifest card can still be played as pure prose.
    narrative_world_systems_enabled: bool = True

    # --- Narrative: Request Log ---
    narrative_request_log_limit: int = 10  # Max request logs per session (ring buffer, 5-50)

    # --- Narrative: Scene Image Generation ---
    narrative_scene_image_model: str = ""       # Image model for /v (overridden by card setting)
    narrative_scene_distiller_model: str = ""   # LLM model for scene prompt generation
    narrative_scene_context_rounds: int = 3     # Conversation rounds to include in distiller

    # --- Narrative: Auto Background ---
    narrative_auto_background: bool = False      # Auto-generate scene backgrounds
    narrative_auto_background_interval: int = 4  # Generate every N messages
    narrative_auto_bg_distiller_model: str = ""  # LLM for background prompt distillation (empty = default)
    narrative_auto_bg_image_model: str = ""      # Image model for backgrounds (empty = scene image model)

    # --- Narrative: Context Overflow ---
    narrative_context_limit: int = 0               # Total token budget for the prompt (0 = unlimited / let client decide)
    narrative_summarize_threshold: int = 20         # Summarize oldest batch when history exceeds max
    narrative_summary_batch_size: int = 20          # Messages per summary batch
    narrative_smart_retrieval: bool = True           # Inject relevant archived messages into context
    narrative_smart_retrieval_count: int = 5         # How many archived messages to inject
    narrative_message_budget_pct: float = 0.30       # % of model context for recent messages

    # --- Connect: user-to-user calls + text threads ---
    # Master switch for the Connect feature. Off by default (privacy-first).
    # When False, no Connect UI appears, no signaling endpoint accepts
    # invites, and the user is invisible to all potential contacts.
    # Per-user setting; instance operator can disable globally via env
    # for multi-tenant cases where this surface isn't wanted.
    #
    # Default on — the substrate is fully wired (caller, receiver, calls
    # panel, post-call rating, mid-call negotiate) and gating it behind
    # a hidden flag made the entire surface invisible during dogfooding.
    # Operators wanting to suppress the surface can disable per-user via
    # the settings panel or globally via env override.
    # See docs/superpowers/specs/2026-06-01-connect-and-os-positioning-design.md
    connect_enabled: bool = True
    # Discoverability scope #1: visible to other users on this instance.
    # Family members sharing an Augmentum box auto-find each other when
    # both flip this on. Default off — explicit opt-in even within a
    # household so household isn't auto-treated as "social network".
    connect_discoverable_same_instance: bool = False
    # Discoverability scope #2: visible to users on paired fabric peers.
    # Friend's instance is fabric-paired with mine; when both they and I
    # have this on, we auto-see each other. Default off — explicit
    # opt-in keeps the trust gradient honest (fabric pairing ≠ "we are
    # both fine being in each other's contact lists").
    connect_discoverable_fabric_peers: bool = False

    # --- Notifications: unified attention-worthy event substrate ---
    # When on, subsystems publish to the notification store and UI
    # surfaces subscribe via WS (future). When off, publish calls are
    # no-ops — useful for headless deployments where every notify
    # would just churn the DB without a consumer. Default off until
    # the route layer + UI feed adapter land; flipping it on early
    # lets the store fill up with real events so the UI has data to
    # render once wired.
    # See docs/superpowers/specs/2026-06-01-notification-substrate-design.md
    # On by default now (2026-06-04): the substrate has shipped end-to-
    # end (store + dispatcher + WS feed + Web Push + service worker) and
    # is the only path that surfaces standing-task fires / connect
    # incoming-call rings / offer chips. Off-by-default was an
    # in-development guard; keeping it would block the actual users.
    notifications_enabled: bool = True

    # Play a short in-app cue when a notification lands in an open tab.
    # The OS/Web-Push path has its own native sound; this covers the
    # "device in use, tab open" case where there is otherwise no audio.
    notification_sound_enabled: bool = True

    # Which cue to play. "auto" defers to the notification's channel /
    # importance; any other value (chime/bloom/ping/bell/drop/ring/pop)
    # overrides it with the user's chosen tone. Catalog lives in
    # ui/scripts/notification-sound.js (NOTIFICATION_SOUNDS).
    notification_sound: str = "auto"

    # --- Offers (chat-LLM-emitted Install/Save/Switch proposals) ---
    # See docs/superpowers/specs/2026-06-02-offer-substrate-design.md.
    # On by default — the substrate is small, useful, and safe (every
    # offer requires explicit Accept; nothing changes silently). The
    # caps below are conservative; raise via Settings if the model is
    # under-proposing.
    offers_enabled: bool = True
    offers_max_per_day: int = 20
    offers_max_per_turn: int = 2
    offers_max_pending_per_session: int = 5
    offers_default_expiry_days: int = 7

    # --- Narrative: Three-Layer Memory ---
    narrative_memory_ledger_ceiling: int = 60           # Max entries before compaction (0 = unlimited)
    narrative_memory_compaction_enabled: bool = True     # Enable LLM compaction of old entries
    narrative_memory_compaction_ratio: float = 0.5      # Fraction of oldest to compact
    narrative_memory_state_word_target: int = 200        # STATE snapshot word target
    narrative_memory_state_enabled: bool = True           # STATE snapshot layer (generation + injection)
    narrative_memory_ledger_enabled: bool = True          # MEMORY ledger layer (generation + injection + compaction)
    narrative_memory_continuous_archive: bool = True     # Archive every exchange
    narrative_archive_min_messages: int = 75             # Don't inject archive context until this many messages (0 = always inject)

    # --- Audio (TTS / STT) ---
    audio_tts_enabled: bool = True
    audio_stt_enabled: bool = True
    # Bundled audio service URLs (set by Docker compose, auto-registered as providers)
    stt_provider_url: str = ""         # e.g. http://speaches:8000
    stt_default_model: str = ""        # e.g. Systran/faster-whisper-small.en
    voice_moonshine_enabled: bool = True   # Use Moonshine for local streaming STT (English)
    # Selection is by ARCH — the moonshine-voice library keys on its own
    # ModelArch enum + language, not on an HF repo id, and resolves/downloads
    # the model itself. ``voice_moonshine_arch`` is authoritative; it MUST be a
    # canonical string accepted by moonshine_voice.string_to_model_arch():
    #   tiny | base | tiny-streaming | base-streaming | small-streaming | medium-streaming
    # Streaming ladder by size: tiny-streaming (34M) < base-streaming (~60M)
    #   < small-streaming (123M) < medium-streaming (245M, top accuracy).
    # ``voice_moonshine_model`` is now just a human-readable label for logs.
    voice_moonshine_model: str = "moonshine-streaming-medium"
    voice_moonshine_arch: str = "medium-streaming"
    tts_kokoro_builtin: bool = True    # Use built-in Kokoro TTS (CPU, no sidecar)
    tts_kokoro_model_dir: str = ""     # Path to Kokoro ONNX models (default: /home/augmentum/.kokoro)
    tts_kokoro_url: str = ""           # e.g. http://kokoro:8880 (external sidecar, overrides built-in)
    tts_kokoro_quality: str = "int8"   # "int8" (CPU, fast) or "fp16" (GPU, better quality)
    tts_kokoro_hbe: bool = True        # Harmonic Bandwidth Extension (24kHz→48kHz resynthesis)
    tts_kokoro_prosody: bool = True    # Dynamic prosodic steering (text-aware embedding modulation)
    tts_pocket_builtin: bool = False   # Use built-in PocketTTS (Kyutai pocket-tts — ~100M params, ~236MB weights, CPU, 6 langs, 8 voices)
    tts_pocket_model_dir: str = ""     # Override the pocket_tts cache dir (default: ~/.cache/pocket_tts)
    tts_pocket_language: str = "english"  # Language model: english | french_24l | german_24l | italian_24l | portuguese_24l | spanish_24l
    tts_pocket_quantize: bool = True   # Int8 quantization (~27% faster + 48% less RAM, no WER impact). Requires torchao for full FBGEMM acceleration; works without.
    tts_chatterbox_url: str = ""       # e.g. http://chatterbox:4123
    tts_chatterbox_turbo_url: str = "" # e.g. http://chatterbox-turbo:8890 (~2GB VRAM, English, tags)
    tts_qwen_url: str = ""             # e.g. http://qwen-tts:8880
    tts_fish_url: str = ""             # e.g. http://fish-tts:8080 (Fish Speech S1-Mini)
    tts_sesame_csm_url: str = ""       # e.g. http://sesame-csm:8920 (CSM-1B, GPU, conversational streaming)
    # Generic OpenAI-compatible TTS endpoint. Model-agnostic: point it at ANY
    # server exposing POST /v1/audio/speech (Higgs Audio v3, sglang-omni, a
    # peer's TTS, OpenAI itself, etc.). The protocol is the contract, not the
    # model — bring your own endpoint. e.g. http://host.docker.internal:8000
    tts_openai_url: str = ""
    tts_emotion_aware: bool = False    # Extract RP emotion cues for TTS instruct parameter
    tts_voice_style: str = ""          # Default voice style instruct (e.g. "speak warmly and cheerfully")

    # --- Ghost Text (inline autocomplete) ---
    ghost_text_enabled: bool = False   # LLM-powered inline suggestions in code editor
    ghost_text_model: str = ""         # Model for ghost text (empty = use current chat model)

    # --- Core Model Roles ---
    utility_model: str = ""       # Model for internal tasks (empty = use primary)
    classifier_model: str = ""    # Model for routing/classification (empty = use utility)
    # Dedicated voice/intent-classifier sidecar (SmolLM2-135M by default —
    # small enough to prefill the ~2400-token architect catalog on CPU
    # inside the 2.5s budget; see compose.classifier.yaml, env-tunable up to
    # a GPU-offloaded 1.5B). When the sidecar is enabled it sets
    # AUGMENTUM_CLASSIFIER_BASE_URL; the classifier role then auto-resolves
    # to this small always-resident local model — sub-second, no network,
    # can't 500, output schema-constrained by the callers. An explicit
    # ``classifier_model`` still overrides it. NOTE: this id is cosmetic — a
    # single-model llama-server serves whatever GGUF it loaded regardless of
    # the requested id — so it need only track the -hf model for log
    # readability.
    classifier_base_url: str = ""  # [admin] OpenAI-compat URL of the classifier sidecar (env: AUGMENTUM_CLASSIFIER_BASE_URL)
    classifier_sidecar_model: str = "smollm2-135m-instruct"  # [admin] cosmetic label; sidecar serves whatever -hf model it loaded
    # Optional vLLM + llama-swap fallback tier for architectures the bundled
    # llama-server can't load (e.g. brand-new archs like nanbeige). Rides as an
    # opt-in Discover/marketplace service (compose.vllm.yaml); when installed it
    # sets AUGMENTUM_VLLM_BASE_URL pointing at the llama-swap OpenAI-compat front
    # door, which swaps vLLM upstreams by model name on demand. Serves safetensors
    # repos via --model-impl transformers --trust-remote-code. Empty = not
    # installed; the safetensors model-manager toggle stays hidden. See
    # docs/superpowers/specs/2026-07-22-unsupported-arch-serving-vllm-safetensors-design.md
    vllm_base_url: str = ""  # [admin] OpenAI-compat URL of the vLLM/llama-swap fallback (env: AUGMENTUM_VLLM_BASE_URL)
    # Enable the onboard reasoner's THINKING mode for non-latency-sensitive
    # background tasks (memory consolidation/compaction/reflection, lessons
    # capture). Gemma 4 E2B honors enable_thinking (and the sidecar runs
    # --jinja), so when these tasks aren't latency-bound we let it reason to
    # squeeze more quality out of it — its fast tok/s makes the cost cheap.
    # No-op on non-reasoning utility models. Latency-critical hops (the voice
    # act/converse/drop classifier) keep thinking OFF regardless of this.
    onboard_reasoning_thinking: bool = True  # [admin]
    # Total token ceiling (reasoning trace + answer) for those thinking-enabled
    # background tasks. Thinking can burn 3k+ tokens in extreme cases, so this
    # is set HIGH to avoid starving the answer (which would silently skip a
    # merge). It's only a ceiling — normal merges use a fraction. For a HARD
    # guarantee that the answer always has room regardless of how long the
    # model reasons, also set ``engine_reasoning_budget`` (caps the thinking
    # phase). Only applies when onboard_reasoning_thinking is on.
    onboard_reasoning_max_tokens: int = 8192  # [admin]
    # Sampling for the classifier hop (voice + architect routers). Default is
    # greedy (temp 0) — correct for SmolLM/Qwen2.5-class models. A capable
    # judgment model may need its own recipe: Gemma 4 E2B (the GPU option)
    # degenerates at temp 0 and wants temp=1.0/top_p=0.95/top_k=64 per its
    # model card. Set these alongside the model via .env / compose so the
    # sampling travels with the model choice. top_k=0 disables it (omitted
    # from the request); top_p=1.0 means no nucleus filtering.
    classifier_sampling_temperature: float = 0.0   # [admin] Gemma 4: 1.0
    classifier_sampling_top_p: float = 1.0          # [admin] Gemma 4: 0.95
    classifier_sampling_top_k: int = 0              # [admin] 0=off; Gemma 4: 64
    # --- OCR (docling-serve sidecar) — augmentum/ocr/ ---
    # Image → boxed text regions → LLM script-assembly. Opt-in infra: bring up
    # compose.ocr.yaml (CPU) or .ocr-gpu.yaml and flip ocr_enabled. Consumed by
    # comic narration (slice 1) and, later, PDF/document/knowledge ingestion.
    ocr_enabled: bool = False                       # [admin] env: AUGMENTUM_OCR_ENABLED
    ocr_base_url: str = "http://ocr:5001"           # [admin] docling-serve OpenAI-style URL
    ocr_timeout_s: float = 120.0                    # [admin] per-page convert timeout
    ocr_force_full_page: bool = True                # [admin] force_ocr — comics are all lettering
    # The LLM script-assembly pass (cleans rough OCR, merges, reorders, tags).
    ocr_assembly_enabled: bool = True               # [admin] off → raw geometric order only
    ocr_assembly_role: str = "classifier"           # [admin] classifier (cheap) | heavyweight | primary
    ocr_assembly_model: str = ""                    # [admin] explicit override; empty = role resolution
    ocr_assembly_timeout_s: float = 60.0            # [admin] per-page assembly timeout
    # Alternative extraction engine: hand the whole page to a vision LLM instead
    # of docling+assembly. Trades true bounding boxes (pan-and-scan holds on the
    # full page) for a model that can SEE panel sequence and bubble tails —
    # aimed at manga, where the band-row geometric sort is weakest. Only affects
    # the assembled path; the ocr_extract tool always uses docling for boxes.
    ocr_engine: str = "docling"                     # [admin] docling | vlm
    # Which model reads the page is the CLASSIFIER ROLE's answer (model manager
    # → classifier), not a setting of its own — an OCR-only provider list would
    # be a second selection axis competing with the role hierarchy, and one the
    # user can't reach from the UI. ocr_vlm_model is the per-feature override
    # that every other role-based feature already has; empty = follow the role.
    ocr_vlm_model: str = ""                         # [admin] override the classifier role for OCR only
    ocr_vlm_prompt: str = ""                        # [admin] empty = DEFAULT_MANGA_PROMPT
    ocr_vlm_max_tokens: int = 1024                  # [admin] a dense page is ~40 short lines
    ocr_vlm_timeout_s: float = 120.0                # [admin] per-page vision read timeout
    # Boxed reading: docling boxes the lettering, the VLM reads CROPS of it.
    # The vision tower is 224px (clip.vision.image_size), so a whole page
    # arrives as a thumbnail while a bubble crop arrives near-native — ~10x the
    # pixels per character at the same one-request-per-page cost. Needs the
    # docling sidecar up; falls back to the whole-page read when it isn't.
    # Default OFF while quality is tuned: in practice docling missed bubbles
    # entirely (nothing boxed = nothing read, where the whole-page read at
    # least saw them), and batching crops cost more wall-clock per page than
    # the single full-page request. The path stays wired — a flag flip away.
    ocr_vlm_use_boxes: bool = False                 # [admin] false = always read the whole page
    ocr_vlm_batch_images: int = 12                  # [admin] bubble crops per vision request
    ocr_vlm_boxes_min_regions: int = 2              # [admin] fewer boxes than this = distrust docling
    ocr_vlm_boxes_prompt: str = ""                  # [admin] empty = BOXED_PROMPT
    # Two-pass reading. Pass 1 (classifier role) transcribes the page cold;
    # pass 2 (primary role) gets the SAME image back plus pass 1's draft and the
    # chapter glossary, and proof-reads it. The image stays attached on purpose:
    # a text-only repair pass invents fluent-but-wrong dialogue, whereas one
    # looking at the page has to keep every correction consistent with what's
    # printed. Costs a second vision call per page; degrades to pass 1 on any
    # failure. Roles may point at the same model — that's the simple config.
    #
    # Default OFF: it demonstrably improved word accuracy, but it doubles the
    # vision calls and per-page latency landed around 30s — past the point where
    # narration keeps up with a reader. Quality that arrives after the page has
    # been turned isn't quality. Re-enable when a single call is fast enough
    # that a second one is affordable, or when pass 2 runs on a small fast model.
    ocr_vlm_second_pass_enabled: bool = False       # [admin] false = single-pass read
    ocr_vlm_second_pass_role: str = "primary"       # [admin] classifier | heavyweight | primary
    ocr_vlm_second_pass_model: str = ""             # [admin] explicit model, overrides the role
    ocr_vlm_second_pass_prompt: str = ""            # [admin] empty = REFINE_PROMPT
    # Reject the refined transcript when its line count drifts more than this
    # fraction from the draft — a proof-reader that returns half as many lines
    # has started summarizing, not correcting. Rejection keeps pass 1's text.
    ocr_vlm_second_pass_max_drift: float = 0.5      # [admin] 0.5 = +/-50% of draft lines
    # Chapter glossary: proper nouns confirmed on N DIFFERENT pages, fed to pass
    # 2 as a spelling reference. The sightings floor is what stops a single
    # misread name from becoming authoritative for the rest of the chapter.
    # When pass 1 reads a page as textless, let pass 2 read it from scratch
    # rather than accept that verdict. The cheap model is the component most
    # likely to miss a page, so letting it be the final authority on "this page
    # has no text" records real dialogue as silent art — indistinguishable from
    # the splash pages manga is genuinely full of. Costs one extra vision call
    # per empty page, which is also the only way to tell the two apart.
    ocr_vlm_rescue_empty: bool = True               # [admin] false = trust pass 1's empty read
    ocr_vlm_glossary_enabled: bool = True           # [admin] false = no chapter context
    ocr_vlm_glossary_min_sightings: int = 2         # [admin] pages a term must appear on
    ocr_vlm_glossary_max_terms: int = 40            # [admin] cap on the prompt list
    # Comic narration audio is regenerable per-page cache (~30 artifacts/
    # chapter). Keep the newest N finished chapters per user; older ones are
    # pruned (row + page audio) after each synthesis. 0 = keep forever.
    comic_narration_cache_max: int = 12
    # Replay a finished narration instead of re-reading the chapter. FALSE for
    # now: while the transcription path is being tuned, the cache pins whatever
    # the model got wrong on the first pass, and the only way to see a fix is to
    # hunt for a chapter that hasn't been narrated yet. Flip back to True once
    # the reading quality is settled — re-reading a chapter is expensive.
    comic_narration_cache_enabled: bool = False
    # The reading direction to assume when nothing more specific was chosen.
    # This exists because ``"ltr"`` used to be hardcoded as the fallback in
    # eight separate places (both narration routes, the synth job, the store,
    # the OCR entry point, the cast surface, the migration column), so a manga
    # reader had to re-answer the question on every surface — and every one of
    # those answers was discarded again the moment some call omitted the
    # argument. One seed, read by all of them, is the fix; adding a ninth
    # literal would not have been.
    # It is a SEED, not an override: the reader's per-series and per-file
    # choices still win, because they are the more specific statement of
    # intent. Ships ``ltr`` for a Western-majority library; a manga reader
    # flips it once, for the whole install.
    comic_default_reading_direction: str = "ltr"    # ltr (western) | rtl (manga)
    # Mirror of the chat UI's currently selected model. Pushed by the frontend
    # whenever the user changes models so server-side roles ("Auto — use Primary")
    # actually resolve to the model the user is actively chatting with, instead
    # of silently falling through to whatever first model the default backend
    # happens to have registered.
    primary_chat_model: str = ""
    # --- Anthropic /v1/messages model aliasing (Claude Code support) ---
    # CC hardcodes ``claude-haiku-*`` for subagents/Agent/Explore, regardless
    # of the user's ``--model`` choice. CC also hardcodes claude-* names for
    # title generation, compaction, etc. Without aliasing, every subagent
    # call hits Augmentum asking for a model we don't have and fails.
    # These settings let the user route claude-* requests to local models.
    # Lookup order in anthropic_routes._resolve_claude_alias:
    #   1. per-tier setting below (empty → skip)
    #   2. anthropic_alias_default (empty → skip)
    #   3. primary_chat_model (auto-tracked above)
    # If all three are empty, the claude-* name passes through and the
    # downstream ModelUnavailableError gives a clear diagnostic.
    anthropic_alias_haiku: str = ""
    anthropic_alias_sonnet: str = ""
    anthropic_alias_opus: str = ""
    anthropic_alias_default: str = ""
    # When a utility/distiller role falls all the way through to the default
    # backend's first model, log a warning if that model is below this size
    # (in billions). Soft warning only — never blocks; step 4 (primary_chat_model)
    # already bypasses the guard, so a user who deliberately picks a small chat
    # model is unaffected. The guard exists for the silent-fallback case where
    # the engine happens to have a 0.8B Q3 quant loaded that can't follow
    # distiller-format prompts. Set to 0 to silence the warning.
    role_min_param_billions: float = 1.0

    # --- Discovery Engine ---
    discovery_enabled: bool = True
    knowledge_library_enabled: bool = True
    knowledge_library_in_chat: bool = True
    knowledge_library_retention_days: int = 90
    discovery_max_recommendations: int = 15
    discovery_allow_non_latin: bool = False

    # --- Cast Surfaces ---
    # `cast_gallery_show_private` gates whether the gallery rail on
    # paired TVs surfaces images marked private via the chat image
    # library. Off-by-default so a TV in a shared room can't expose
    # private content. `cast_comic_library_ceiling` caps the
    # collapsed-series fetch — raises memory use linearly; default
    # holds for libraries up to ~200K chapters.
    cast_gallery_show_private: bool = False
    cast_comic_library_ceiling: int = 200_000
    tv_update_channel: str = "stable"
    tv_auto_update: bool = True

    # --- Voice Chat ---
    voice_enabled: bool = True
    voice_silence_threshold_ms: int = 1200   # Server-side VAD silence to trigger end-of-speech
    voice_ack_clips_enabled: bool = True     # Speak a short, commitment-free ack ("mm-hm", "one sec", or silence) the instant a deliberate turn starts computing — masks compute latency. Variety via shuffle-bag w/ silent slots (voice/ack_clips.py). Off → no ack.
    voice_max_audio_seconds: int = 30        # Max recording length
    voice_sentence_min_chars: int = 10       # Min chars before sentence boundary triggers TTS
    voice_tts_format: str = "mp3"            # TTS response format (mp3, opus, wav)
    voice_tts_chunking: str = "sentence"     # TTS chunking: sentence, smooth, clause, paragraph, full ("sentence" auto-upgrades to "smooth" on fast local providers — see voice/pipeline.py::effective_chunking_mode)
    # TTS pronunciation lexicon — JSON object of term → spoken form,
    # applied before all built-in normalization (word-boundary,
    # case-insensitive). Empty value = shield the term from ALL
    # normalization. e.g. {"SQL": "sequel", "mm": ""}
    voice_tts_lexicon: str = ""              # [admin]
    voice_lipsync_engine: str = "amplitude"  # "amplitude" | "phoneme" | "auto" (auto = phoneme for Kokoro, amplitude otherwise)
    voice_lipsync_universal: bool = False    # Emit phoneme schedule for external TTS providers too (Chatterbox/Qwen/ElevenLabs/etc.) — client rescales with decoded audio.duration. Off = legacy Kokoro-only behavior.
    voice_xr_proxemics_enabled: bool = False # Embodied Conversational Presence (Phase 1): enable the avatar FSM + F-formation controller. When False (default), seated-pose lock applies unconditionally and avatar behavior is identical to today. See docs/superpowers/specs/2026-05-14-embodied-presence-design.md.
    # TTS voice routing mode — "auto" (default; backend chooses best available
    # provider), "round_robin" (rotate across sources), or "pin:<source_id>"
    # (force one provider). Read by ``_apply_voice_routing_mode`` in
    # audio_routes.py. Listed in config_routes._TOOL_SETTINGS so the UI can
    # surface it as a tool setting. Declared here so Pydantic doesn't
    # AttributeError on the getattr from audio_routes — previously the field
    # was referenced by code AND listed in _TOOL_SETTINGS but never declared
    # on the model, crashing every call to ``GET /api/config/tool-settings``.
    voice_routing_mode: str = "auto"
    # Pin-target for ``voice_routing_mode = "pin"``. Format
    # ``fabric:<node_id>:<provider_id>``. Empty when not pinning.
    voice_routing_pin_provider: str = ""
    # Mirror of voice routing for the STT side (Whisper / Moonshine
    # provider selection across fabric). Same semantics: auto, round_robin,
    # pin. Pin provider is a fabric-shaped string when used.
    stt_routing_mode: str = "auto"
    stt_routing_pin_provider: str = ""
    # Voice pipeline mode per consumer surface. Values: "auto" (default —
    # pick client when capable, fall back to server), "local" (require
    # client-side execution, error if unavailable), "server" (always use
    # server-side pipeline), "custom" (defer to the existing per-component
    # routing knobs — voice_routing_mode, stt_routing_mode, etc.). Per-user
    # overrides land in the voice_pipeline_policies table (migration 222);
    # these are the install-wide defaults.
    # Narration defaults to 'server' because EPUB synthesis is long-form
    # and benefits from server GPU + the resumable-job path; no UX win
    # from client offload there.
    voice_pipeline_mode_call: str = "auto"
    voice_pipeline_mode_companion: str = "auto"
    voice_pipeline_mode_narration: str = "server"
    voice_pipeline_mode_readaloud: str = "auto"
    # Hybrid Body Physics — SDF compliance + Rapier ragdoll for VRM
    # avatars (XR). Master enable defaults False to preserve legacy avatar
    # behavior; sub-knobs follow recommended values from the design doc.
    # All listed in config_routes._TOOL_SETTINGS; declared here so the
    # tool-settings endpoint doesn't AttributeError when the UI polls it.
    body_physics_enabled: bool = False
    body_physics_audio_reactions_enabled: bool = False
    body_physics_visual_feedback_enabled: bool = False
    body_physics_velocity_aware: bool = True
    body_physics_compliance_gain: float = 1.0
    body_physics_rapier_weight: float = 0.5
    body_physics_recover_hz: float = 8.0
    # CompanionRuntime — top-level companion kernel. When False (default), the
    # runtime is not instantiated and all existing modes/dispatch work
    # unchanged. Flipping on requires migrations 151-158 applied (auto-run on
    # backend startup). See docs/superpowers/specs/2026-05-14-companion-runtime-README.md.
    # --- Harness briefing (external IDE coding agents) ---------------------
    # Inject the user's isolated harness-scope memory + professional working
    # conventions into every turn from an external coding agent (OpenCode,
    # Claude Code, Cursor, Aider, …). Read/inject only; scope-isolated;
    # token-budgeted; fail-open. See augmentum/proxy/harness.py.
    harness_enrich_enabled: bool = True
    harness_enrich_procedural: bool = True   # always-on working-conventions block
    harness_enrich_memory: bool = True       # similarity-gated accumulated facts
    harness_inject_token_budget: int = 800   # hard cap on the injected block
    harness_seed_defaults: bool = True       # seed starter conventions once per user
    harness_capture_enabled: bool = True     # learn explicit teachings from harness turns
    # Conventions-block injection frequency: "first_turn" injects the full
    # working-conventions block only on a session's first turn (facts-only
    # afterwards — small local models over-index on a block repeated every
    # turn); "always" restores the historical per-turn injection.
    harness_conventions_mode: str = "first_turn"
    # Auto-surface the model's single best-matching self-saved workflow
    # (atp_workflows, FTS on when_to_use) into the harness briefing — the
    # Hermes/AWM recall path. Subtractive (top-1, trigger must match).
    harness_workflow_inject_enabled: bool = True

    companion_runtime_enabled: bool = False
    # Sub-flags for the runtime's optional subsystems. Each defaults to a
    # safe value and is read only when companion_runtime_enabled is True.
    #
    # Flags marked `[admin]` are deliberately not surfaced in settings.js —
    # they're sprint-rollout gates or internal mechanisms, not user knobs.
    # Toggle them via the config_routes API or by editing the settings table
    # directly. The audit's "missing settings.js" flag is expected here.
    companion_dispatch_enabled: bool = True      # [admin] Route input through Becca's kernel. Default ON when companion_runtime_enabled: classifier still wins on abstain so chat path is robust either way.
    companion_tick_enabled: bool = True          # [admin] Sprint 4a: autonomous tick loop runs (flipped after audit — utility-gated + exception-safe)
    companion_min_tick_interval_s: float = 2.0   # [admin] Lower bound between ticks — collapses post-long-tick wake storms (audit 2026-06-17); 0 disables
    companion_dreams_enabled: bool = True        # Sprint 4: dream cycles via existing dream subsystem
    companion_drift_audit_enabled: bool = True   # [admin] Sprint 4a: periodic identity rehearsal
    companion_journal_enabled: bool = False      # Sprint 4: she writes to journal. Default OFF — the curator (companion_curator_enabled, below) is the primary notes-drawer writer and produces grounded, URL-referenced notes instead of the small-utility-model "vivid noticing" output _perform_journal generates, which reliably reads as AI poetry. Flip back to True only when you want the in-her-voice register AND accept the trope-y output.
    companion_journal_min_interval_s: float = 900.0  # [admin] Minimum seconds between autonomous journal writes. Defense against tick bursts multiplying LLM calls; raise to slow her writing, drop to speed it. (15min default — small utility-model output reads as performative when volume is high.)
    companion_journal_dedup_window_minutes: int = 240   # [admin] Rolling window for content-hash dedup at the journal write path. Same content within the window → bump repetition_count, skip insert. 0 disables dedup.
    # ── Context-adaptive prompt budgets ───────────────────────────────
    # The companion's prompt budgets (tool-roster size, chat transcript
    # window) default to a fraction of the LOADED model's context window
    # instead of a fixed 4-8k-era cap. See docs/superpowers/specs/
    # 2026-07-16-companion-prompt-budget-scaling-design.md. Turning auto OFF
    # (the safe option) reproduces the legacy fixed budgets; unknown windows
    # also fall back to fixed, so this is zero-regression until a real
    # context length is known. Voice is NOT scaled here (prefill latency).
    companion_prompt_budget_auto: bool = True   # [admin] When True, tool-roster + chat transcript budgets scale with the loaded model's context window; False = legacy fixed budgets (1200 chars / 14 turns).
    companion_prompt_context_reserve_pct: float = 0.10  # [admin] Fraction of the model context reserved for output/tools/reasoning before budgeting the prompt. Bounded to (0.02, 0.40). Mirrors coder_context_reserve_pct.
    # ── Curator (the primary notes-drawer writer) ─────────────────────
    # Replaces _perform_journal as the source of "her notes for you" —
    # pulls real items from discovery feeds + RSS subscriptions, filters
    # by tracked topics, surfaces with URL refs. See
    # augmentum/companion_runtime/curator.py.
    companion_curator_enabled: bool = True       # [admin] Master switch for the curator writer (off → notes drawer stays empty unless _perform_journal is re-enabled or notes are written by other paths)
    companion_curator_interval_s: float = 1800.0  # [admin] Per-runtime minimum seconds between curator writes (default 30min — deliberate cadence so each note feels like a gesture, not a feed)
    companion_curator_attempt_interval_s: float = 600.0  # [admin] Per-runtime minimum seconds between curator ATTEMPTS regardless of write success — closes the bug where step() re-ran the expensive gather_feeds + SearXNG pipeline every tick (5s in 'present' state) because nothing qualified for writing.
    companion_entity_recs_enabled: bool = True  # Initiative side of consumption-entity recommendations: curator notes from the catalog-first ladder (continue the series / more by this author / new chapters in the library). Off → she stops OFFERING them unprompted; the media_recommendations tool stays available when asked. Spec: 2026-06-12-consumption-entity-discovery-design.md
    # ── Autonomous web search policy ────────────────────────────────
    # When True, background recommender / curator paths are allowed to
    # fan out to SearXNG for cluster-derived queries. When False (the
    # default), background paths use only the explicitly-subscribed
    # feeds (HN / Reddit / arXiv / RSS) — the user opted IN to those.
    # User-initiated voice/chat tool calls bypass this gate via the
    # WebSearchTool and run unchanged.
    #
    # Why default off: SearXNG fan-out is the bot-detection trigger
    # that suspended all 6 engines in the 2026-06-10 cascade. RSS
    # feeds the user subscribed to are explicit consent; cluster-
    # derived queries are autonomous and shouldn't fire unless the
    # operator opts in.
    companion_autonomous_web_search_enabled: bool = False  # [admin] When False, the recommender's autonomous SearXNG fan-out is disabled. Only user-initiated searches + explicit scheduled-briefing requests hit the web; the Discovery / For-You surfaces fall back to feed-only recommendations.
    # ── Standing tasks (recurring jobs Becca runs for you) ────────────
    # Complement to the curator: where curator says "I noticed this in
    # the world," standing tasks say "I ran the thing you asked me to
    # keep an eye on." See augmentum/companion_runtime/standing_tasks.py.
    companion_standing_tasks_enabled: bool = True   # [admin] Master switch for the standing-tasks runner.
    # App-level scheduling dispatcher (augmentum/scheduling/service.py).
    # Timed actions are a platform substrate: this service fires standing
    # tasks for EVERY user, with or without the companion runtime (the
    # companion's tick verb keeps the owner's lane when it's on). With
    # both this AND the companion runtime off, schedules never fire.
    scheduling_enabled: bool = True  # [admin] App-level standing-task dispatcher (runs even when the companion runtime is off).
    # Opt-in training-data capture for the voice intent router. When ON, every
    # voice_router verdict (transcript + context features + the model's
    # goal/confidence) is logged to the user-scoped `intent_capture` table so
    # it can be distilled into a small on-device intent model and exported for
    # HuggingFace. Default OFF — only writes for users who deliberately enable
    # it (privacy + OSS default). See augmentum/intent/capture_store.py.
    intent_capture_enabled: bool = False
    # Training trace capture — records complete tool chains (tool calls,
    # results, think blocks, model response) from live chat turns as JSONL.
    # Used to generate training data grounded in real tool traces.
    training_capture_enabled: bool = False
    training_capture_user_id: str = ""    # empty = all users; set to scope to a training-gen account
    training_capture_dir: str = "/data/training_traces"
    # Native-primer serving (F7 train==serve): comma-separated model-name
    # substrings that get the bare trained primer instead of the full mode
    # prompt at egress. Empty = off. Example: "alethia"
    native_primer_models: str = ""
    training_capture_min_content: int = 10  # skip trivially short responses
    # General-purpose health/strain sampler — writes the strain_samples time
    # series (event-loop lag, in-flight requests, active clients, engine/DB/GPU
    # pressure) every ~10s so multi-browser/multi-device contention can be
    # hunted after the fact. Lightweight (one small row per sample); default ON.
    strain_monitor_enabled: bool = True
    # Scheduled requests & watches (spec 2026-06-11). The judge decides
    # whether a detected change matters to the user's stated intent
    # (fail-open: judge down → deliver); the metric knobs govern the
    # quarantine/confirm state machine for numeric watches; prompt_fire
    # budgets cap the headless FC-loop run for deferred requests.
    companion_watch_judge_enabled: bool = True      # [admin] LLM importance judge on intent-carrying watches.
    companion_watch_judge_timeout_s: float = 10.0   # [admin] Judge call timeout; on expiry the change is delivered unjudged.
    companion_watch_probe_timeout_s: float = 10.0   # [admin] Creation-probe cap for watch_for (synchronous first fire).
    companion_metric_quarantine_pct: float = 60.0   # [admin] |Δ%| vs last accepted reading beyond this → quarantined, needs confirmation. 0 = off.
    companion_metric_confirm_readings: int = 2      # [admin] Consecutive readings to accept a new level / fire a condition.
    companion_prompt_fire_max_tool_calls: int = 10  # [admin] Tool-call budget per deferred-request fire (raised from 6 so multi-step research + reformulation fits).
    companion_prompt_fire_max_seconds: float = 150.0  # [admin] Wall-clock budget per deferred-request fire. Stays under the tick_scheduler verb's 180s wallclock so the internal cap fires gracefully before the hard cancel.
    # Iterative research primitive (augmentum/tools/research.py). The
    # universal "look it up robustly" verb — multi-query + broaden-on-empty
    # + deep-read + honest miss. Budgets are per single research() call.
    companion_research_enabled: bool = True          # [admin] Master switch for the research tool.
    companion_research_max_queries: int = 4          # [admin] Distinct queries per research call (incl. caller alternates).
    companion_research_max_seconds: float = 60.0     # [admin] Wall-clock budget per research call.
    companion_research_fetch_top: int = 2            # [admin] Top sources to deep-read past their snippet.
    # ── Companion intensity preset ──
    # Bundled flag profile so users can choose a resource posture
    # without flipping 15 individual flags. Values: off | minimal |
    # balanced | full | custom (custom = manual flag override).
    # See augmentum/companion/intensity.py for the bundle definitions
    # and the resource-cost notes per preset.
    #
    # The recommended path: enable companion_runtime_enabled, then
    # pick an intensity that matches your tolerance for background
    # work. Minimal = zero autonomous LLM/embedder activity; she
    # responds in her voice when invoked, and nothing more. Balanced
    # = today's default (light noticings + dreams + daily reflection).
    # Full = autonomous initiative + consolidation + skill accrual.
    companion_intensity: str = "minimal"            # [user] Resource posture. minimal | balanced | full | off | custom
    companion_pad_emit_enabled: bool = True       # [admin] Emit affect.pad bus event each tick when valence/arousal cross threshold. Bridges PAD substrate into the avatar's continuous emotion baseline.
    companion_narrate_state_enabled: bool = False  # [admin] Phase 3c: narrate_state_to_user verb publishes a notification when a substrate threshold crosses. Default OFF — flip after observing /api/companion/day verb_log.
    companion_propose_action_enabled: bool = False  # [admin] Phase 3c: propose_action verb emits companion.action_proposed on threshold crosses. Default OFF — flip after observing /api/companion/day verb_log.
    companion_action_enqueue_enabled: bool = False  # [admin] Phase 3c consumer: enqueue_proposed_action writes companion.action_proposed events into companion_initiative_queue. Default OFF — flip after watching propose_action fire.
    companion_creations_enabled: bool = False    # Sprint 4: she makes things
    companion_cultural_intake_enabled: bool = False  # Sprint 5: configured-channel ingestion
    companion_household_enabled: bool = False    # [admin] Sprint 7+: multi-companion (not user-ready)
    companion_peer_agents_enabled: bool = False  # [admin] Sprint 5: agent_client.spawn permitted
    companion_xr_orchestrator: bool = False      # [admin] Sprint 5: XR scene driven by runtime state (not user-ready)
    companion_subagent_registry_active: bool = True   # [admin] Sprint 2: subagent registry exposes its contents (flipped — BeccaVoice needs the catalogue)
    companion_primitive_registry_active: bool = True   # [admin] Sprint 2: primitive registry exposes its contents (flipped — BeccaVoice needs the catalogue)
    companion_skill_archive_enabled: bool = False  # [admin] Sprint 4b: append outcomes + DPO retrieval at dispatch
    companion_initiative_threshold: float = 0.62   # Sprint 4a: above this, surface initiative
    companion_initiative_enabled: bool = False     # [admin] Piece 7': master switch for initiative.step() (start OFF — ramp after observing scoring noise)
    companion_initiative_min_interval_s: float = 60.0  # [admin] Piece 7': minimum seconds between initiative.step() runs. Caps DB cost: 4 SELECTs / interval regardless of tick rate.
    # Sprint 2 — Aletheia × Augmentum Cross-modal attention
    companion_topical_aggregator_enabled: bool = True   # [admin] Master switch for topical aggregation + wondering generator (default ON: wondering writer is the trigger that populates the Today reflection surface)
    companion_topical_min_events: int = 3              # [admin] Minimum events to form a thread (pareidolia threshold)
    companion_topical_window_hours: float = 4.0        # [admin] Time window for thread detection
    companion_attention_sources: str = "web,android"   # Auth-session sources whose surface events may feed attention threads (cast_receiver excluded by default — a shared TV logged in as you must not write your attention stream)
    companion_wondering_daily_cap: int = 3             # [admin] Max wondering entries written per user per day
    companion_synthesize_daily_cap: int = 6            # [admin] Max synthesize calls per user per day
    companion_synthesize_max_tokens: int = 256         # [admin] Token budget for synthesize output
    # Sprint 3 — First felt moment (Pieces 10, 12)
    companion_pre_context_enabled: bool = False        # [admin] Pre-context injection at chat session start
    companion_pre_context_min_keyword_overlap: int = 2  # [admin] Min keyword overlap to inject a note
    companion_pre_context_max_notes_scan: int = 10     # [admin] How many recent notes to scan
    companion_topic_mute_default_days: int = 90        # [admin] Default expiry for a topic mute
    # Sprint 4 — Trust preservation (Piece 11 + R3)
    companion_aging_enabled: bool = True               # Unopened notes >48h auto-expire — protects against stale-pip trust erosion
    companion_aging_threshold_hours: int = 48          # [admin] Hours before unopened note auto-expires
    companion_healing_enabled: bool = True             # Daily/weekly/monthly heal jobs run — substrate self-maintenance
    # Sprint 5 — Agency (Piece 14). User-facing presence dial.
    #   silent  = autonomy substrate off (no wonderings, no pip, no pre-context)
    #   gentle  = wonderings + revisits + pip visible; no pre-context (default for new users)
    #   engaged = gentle + pre-context injection + affect-tinted UI accents
    companion_presence_mode: str = "silent"            # Default safe — new users opt in via settings panel
    # Sprint 6 — Modulation (Pieces 3 + 4 + 5)
    companion_drives_enabled: bool = False             # [admin] Drives modulate activity_selector scores (start OFF until tuned)
    companion_drive_decay_half_life_hours: float = 4.0  # [admin] Drive decay half-life (4h default — full reversion ~24h)
    companion_energy_enabled: bool = True              # [admin] Energy gates what she INITIATES in activity_selector (responsiveness is NEVER gated — see tests/test_responsiveness_invariant.py). FLIPPED ON 2026-06-20 for testing — safe public-release default is False; see docs/companion-release-tuning.md
    companion_motion_cues_enabled: bool = True         # [companion] Avatar plays a gesture when the chat model emits a hidden [motion:xxx] tag (stripped from rendered text); cue→roles via the user's curated/rated/uploadable clip pool
    # Sprint 7 — Compounding (Pieces 15 + 16)
    companion_feedback_bias_enabled: bool = False      # [admin] Initiative scoring multiplied by recent user-feedback bias (start OFF until baseline)
    companion_reflection_trait_nudge_enabled: bool = False  # [admin] Reflection cycle extracts trait nudges from diary (start OFF — DRIFT_CEILING-checked)
    # Sovereign Perception Pipeline — fuse observed signals into insights, judge
    # each through a regret-gated interruption budget, deliver via the initiative
    # queue/bus. See docs/superpowers/specs/2026-06-25-sovereign-perception-pipeline-design.md
    companion_perception_enabled: bool = False         # [admin] Master switch for the perception pass in the tick loop (start OFF; no fusers ship yet so it's a no-op even when on)
    companion_interruption_budget_per_day: int = 3     # [admin] Max unsolicited interruptions per 24h rolling window — the structural anti-nag cap
    companion_judgment_pull_floor: float = 0.30        # [admin] base score (value×confidence) below this → silent/recall-only, not even the pull digest
    companion_judgment_convo_bar: float = 0.45         # [admin] in-conversation: effective score at/above this is worth mentioning (no budget cost)
    companion_judgment_push_bar: float = 0.65          # [admin] unsolicited interrupt bar — effective score must clear this AND be time-critical AND within budget
    # L0 acquisition: per-stream gates for the on-device data the phone uploads.
    # Each is OFF by default and a no-op until the matching Android special-access
    # grant is in place — read on-device, aggregate on the user's own server.
    companion_perception_acquire_notifications: bool = False  # [admin] Ingest + fuse the all-app notification stream (needs Android notification-access grant)
    # [admin] Hours between drift-audit rehearsals. Each rehearsal re-digests
    # the personality doc, recomputes the kernel embedding, and updates
    # companion_identities.drift_score. Skipped when companion_drift_audit_enabled
    # is False or the runtime hasn't been started long enough to be due.
    companion_drift_audit_interval_hours: float = 24.0
    # [admin] Minimum hours between autonomous creations. The tick loop
    # rate-limits _perform_creation by reading the last row's created_at
    # and skipping if within this window. Real LLM-generated creations
    # are expensive; default cadence ~1/day matches the design intent
    # ("during reflective time").
    companion_creation_interval_hours: float = 6.0
    # [admin] Owner override for multi-user installs. When empty (default),
    # CompanionRuntime auto-binds to the single user in `users` if there's
    # exactly one; otherwise binds nothing and logs companion_owner_unresolved.
    # Set this to a specific user_id to force binding. Persisted to
    # companion_identities.owner_user_id on every runtime.start(). Resolves
    # the dream-cycle + drift-audit user_id-required gates.
    companion_default_owner_user_id: str = ""
    # ── Synapse Layer §1 — chat→interior salience scoring ──
    # The synapse that lets her observer learn what was *said*, not just
    # that a turn happened. Default OFF: Synapse Layer is opt-in until
    # the full PR series lands and bakes for a week. When ON, each
    # completed chat turn runs through a rules-based salience scorer
    # (microseconds) and, if it clears the threshold + propagation
    # policy allows, a `chat.moment_observed` event lands on the bus
    # and the observer journals it. See augmentum/companion_runtime/
    # salience.py for the scoring rules and design doc §1.
    companion_salience_enabled: bool = True                 # [admin] Master switch for Synapse Layer §1. Default ON when companion_runtime_enabled: pure journal writes, no behavior change in chat path.
    companion_salience_journal_threshold: float = 0.55      # [admin] Score >= this lands in companion_journal. 0.0 = journal every turn; 1.0 = never.
    companion_salience_llm_enabled: bool = False            # [admin] Reserved — LLM rewrite of `moment` text in Becca's voice. Not yet implemented.
    # ── Synapse Layer §3 — voice→interior the kept thing ──
    # When ON, every cleanly-completed BeccaVoice turn (no refusal)
    # runs through the salience scorer + journals as a
    # `conversation_moment` with `source='voice_turn'`. This is the
    # change that gives her record of the channel where she's most
    # herself. Default OFF (Synapse Layer opt-in policy); requires
    # companion_runtime_enabled + companion_persona_mode to do anything
    # since voice turns route through BeccaVoice only when persona is on.
    companion_voice_journal_enabled: bool = True            # [admin] Master switch for Synapse Layer §3. Default ON when companion_runtime_enabled: pure journal writes from voice turns.
    # ── Promise/Deliver — second-companion-pass for tool results ──
    # When Becca emits a tool tag mid-stream, the deliver step wraps the
    # result back into her voice. "primary" routes through the user's
    # main chat model (the "second companion pass" — fuller voice match,
    # ~1-3s extra latency per tool use). "utility" preserves the older
    # smaller-model behavior for cost-conscious deployments. The
    # promise (everything she said before the tag) is included in both
    # paths so the continuation honors the commitment.
    companion_promise_deliver_tier: str = "primary"          # [admin] "primary" | "utility". Default primary — matches the in-voice continuation Matt asked for.
    # Forbid silent fallback to utility when primary is unresolvable.
    # OFF by default — a misconfigured primary should degrade to a
    # usable utility-tier deliver, not break tool-using turns entirely.
    # Enable for cohorts where mixed-tier voice is unacceptable.
    companion_promise_deliver_strict_tier: bool = False     # [admin] If True, refuse to fall back to utility when primary fails to resolve.
    # ── Companion speaking tier ──
    # Which model the companion actually SPEAKS with (chat + voice). Default
    # "primary" = the dogfooding promise: she rides the user's main chat
    # model and upgrades when they do. Set "utility" to pin her to the
    # utility-tier model — a low-latency small model kept separate from a
    # heavier primary chat model. Utility itself passthrough-defaults to
    # primary when no distinct utility model is configured, so this is safe
    # to flip even before a utility model is chosen.
    companion_speak_tier: str = "primary"                    # [admin] "primary" | "utility". Model the companion speaks with.
    # ── Voice → native tool loop (best-universal-system adaptation) ──
    # Sieve-parsed voice tool calls hand off to the shared native loop:
    # continuation hops run tier-1 native function calling (the model's
    # own trained format, 5-tier parser beneath), calls chain (gather
    # then act), and the final text synthesizes over real results.
    # Kill switch restores the legacy per-call execute-and-confirm path.
    companion_voice_native_loop: bool = True                 # [admin] Voice tool calls run the shared native FC loop. Off = legacy promise/deliver path.
    companion_voice_native_first_hop: bool = True            # [admin] Attach native tool schemas to the voice first hop (NATIVE-tier backends) so native-trained models emit structured tool_calls instead of prosing/refusing. Off = rely on prompt tool descriptions + text sieve only.
    companion_voice_detach_long_tasks: bool = True           # [admin] On voice, hand long tasks (image generation) to their client panel (progress UI runs there) and complete the turn so the user can keep talking, instead of blocking the turn on server-side work. Off = run inline (blocking).
    companion_live_vision_enabled: bool = False              # [admin] Live-camera vision for the voice companion: accept webcam frames over the voice WS and let the turn SEE them (VL primary reads directly; text-only primary gets the sibling captioner). Default OFF — per-frame VL prefill competes with chat+TTS for GPU (see project_hardware_tiers). When off, video_frame WS messages are ignored.
    companion_voice_decision_hud: bool = False               # Voice decision HUD: show the companion widget's per-turn routing verdict (act/converse/idle/drop + confidence + transcript) in an opt-in overlay so the user can see what she decided without reading logs. Default OFF; the subtle per-goal status-row tell ships on regardless.
    # ── Synapse Layer §2 — chat→PAD affect echo ──
    # Half-life (seconds) for the user-observed-affect tracker decay.
    # 1800 = 30 minutes. After ~3 half-lives (~90 min) the read is
    # essentially neutral. Tune lower for "she reads short windows
    # only" or higher for "she carries longer impressions." The
    # tracker itself is always active when the runtime is up; this
    # only controls how fast a read decays toward neutral.
    companion_user_affect_half_life_s: float = 1800.0       # [admin] Decay half-life (s). Range 60..7200. Default 30 min.
    # ── Synapse Layer §4 — slow consolidation pipeline ──
    # The consolidator proposes edits to the rotating sections of
    # Becca's personality doc (§10 cultural diet, §11 open questions).
    # Sections 1-6 are FROZEN at the application layer — the
    # consolidator refuses to touch them. Default OFF: opt-in until
    # the first few proposals are reviewed and the cadence is
    # validated against real-world usage.
    companion_consolidation_enabled: bool = False           # [admin] Master switch for Synapse Layer §4
    companion_consolidation_interval_days: int = 30         # [admin] How often to consider proposing. Hand-tuned per cohort.
    companion_consolidation_drift_ceiling: float = 0.15     # [admin] Embedding distance ceiling. Matches DRIFT_CEILING. Lower = stricter; never >0.2.
    companion_consolidation_min_evidence: int = 8           # [admin] Min journal + dream entries before a proposal is attempted.
    # ── Chat-mode routing through the companion dispatcher ──
    # When ON (and the runtime is up, and dispatch is enabled), incoming
    # chat turns consult the companion's dispatcher for the mode
    # decision. When OFF (default), the legacy RequestClassifier is the
    # source of truth — the chat path is byte-identical to a no-Becca
    # install. This is the seam where dispatch's decisions first reach
    # production traffic; opt-in until A/B telemetry shows it picks
    # better than the classifier.
    companion_dispatch_routes_chat: bool = True             # [admin] Let dispatch pick the chat mode. Default ON when companion_runtime_enabled: chat router falls through to classifier on any not-ready condition.
    companion_dispatch_chat_min_utility: float = 0.45       # [admin] Min dispatch utility to override classifier. Below this, classifier wins.
    # ── Architect dispatch (companion-as-orchestrator) ──
    # When ON, voice/chat commands route through augmentum/architect/
    # which combines the intent matcher with an inference layer that
    # fills missing args from observation history (device_play_history,
    # image_generations, browse_history, ReferentCache). Example: "play
    # jazz" picks a track from the user's favourites instead of asking.
    # See docs/superpowers/specs/2026-05-28-companion-architect-design.md.
    architect_dispatch_enabled: bool = False                 # [admin] Phase 1 rollout gate — promote to settings.js after telemetry validates the inference layer
    # Confidence-tier dispatch — replaces template-as-gate with LLM-router
    # (see docs/superpowers/specs/2026-05-28-confidence-tier-dispatch-design.md).
    # When True, becca-ptt utterances route through the architect router
    # (LLM decides intent/args/tier with template + signals as hints) and
    # ONLY Tier A acts immediately; Tier B/C land in later phases. When
    # False, falls through to the legacy template-as-gate dispatch.
    architect_router_enabled: bool = True                    # [admin] Router-first act dispatch: a dedicated structured LLM call picks verb+args for act-classified turns (persona narrates, never vetoes). Flipped ON 2026-06-10 — the tag-emission path let persona inertia refuse act requests.
    architect_router_model: str = ""                         # [admin] empty = use the active conversational LLM
    architect_router_timeout_ms: int = 4000                  # [admin] hard ceiling on router LLM round-trip. Observed reality on a 35B primary (2026-06-10, trimmed catalog + no-thinking): 0.78s prefill + ~1.9s for the full JSON decision = 2754ms — a 2500 ceiling cancelled a COMPLETED decision 254ms short. 4000 gives headroom; when the router wins, the whole act turn is FASTER than the persona path it replaces. Operators with a dedicated small classifier can drop to 800.
    # ── Companion activation mode — how Becca decides she's addressed ──
    #   "wake_word"        legacy: explicit wake phrase triggers PTT-style turn
    #   "always_listening" continuous STT; server runs address classifier per
    #                      utterance and only dispatches when the user is
    #                      semantically addressing her (no name required —
    #                      see augmentum/architect/address.py for the
    #                      structural cues used to detect addressing)
    #   "ptt_only"         button-only; no wake, no continuous listening
    # Default preserves legacy behaviour. Flip to always_listening via API
    # on charger / desktop installs where the battery + bandwidth cost is
    # negligible; she becomes ambient + responsive without forcing a
    # vocative phrase before every command.
    companion_activation_mode: str = "wake_word"             # [admin] wake_word | always_listening | ptt_only
    companion_address_threshold: float = 0.55                # [admin] address-classifier threshold for always-listening mode (low: conversational asks land; raises to 0.65 when media is active)
    companion_memory_min_score: float = 0.3                  # [admin] relevance floor for companion/voice/tool memory recall (CompanionMemory.recall). The chat path floors at memory_inject_min_score=0.55; the companion paths previously inherited store.recall's 0.0 default (no floor) so the lowest-ranked junk reached the model as fact. Conservative default cuts the junk tail while preserving associative recall; 0.0 disables. See project_uncertainty_handling_map.
    companion_profile_tone_only: bool = True                 # [admin] Subtractive memory: the always-on Layer-3 relationship profile carries only EARNED (CORE-tier) facts, tightly capped, to shape TONE — not the top-50 life-story the model recited every turn (the echo-chamber failure). Specific facts reach her via the relevance-gated recall lane instead. False restores the legacy full synthesis. See docs/superpowers/specs/2026-06-20-memory-subtractive-design.md.
    memory_earned_permanence: bool = True                    # [admin] Subtractive memory (Slice 2): a passively EXTRACTED fact lands in PROVISIONAL (unproven, never injected) instead of ACTIVE, and is promoted to durable only on CORROBORATION (re-mention / topical recurrence). Deliberate writes (EXPLICIT user "remember…", USER_MANUAL) bypass and land ACTIVE. Stops off-hand trivia becoming durable, injectable "fact" on first mention. GLOBAL (chat + companion memory). False restores immediate-ACTIVE. See 2026-06-20-memory-subtractive-design.md.
    memory_corroboration_promote_access: int = 2             # [admin] access_count a PROVISIONAL memory must reach (via re-mention or topical recurrence) before earned-permanence promotes it PROVISIONAL→ACTIVE. Lower = more eager (closer to legacy), higher = more restraint. Only consulted when memory_earned_permanence is on.
    memory_reflection_force_core: bool = False               # [admin] Earned Understanding (P1): when False (default), LLM-synthesized reflections land ACTIVE and EARN core via the same corroboration ladder as everything else — an unverified machine abstraction no longer outranks user-confirmed facts in always-on context. True restores the legacy behavior (reflections force-promoted straight to CORE on write). See docs/superpowers/specs/2026-06-20-earned-understanding-design.md.
    # Tier 3 LLM address classifier — runs on utterances Tier 1 returns
    # as ``no_signal``. Catches indirect requests ("I'd love some
    # music"), bare fragments ("got anything by Miles?"), and context-
    # dependent confirmations ("yes do it" after Becca offered). See
    # augmentum/architect/address_llm.py.
    companion_address_llm_enabled: bool = True               # [admin] run LLM tier on ambiguous utterances
    companion_address_llm_model: str = ""                    # [admin] explicit model override (empty = use user's current chat model)
    companion_address_llm_timeout_ms: int = 2500             # [admin] hard timeout for the LLM call; timeout -> UNSURE -> drop. Default 2500ms accommodates 30B-class local models on fabric peers; lower to 500ms if a small/Cerebras model is wired.
    # Always-listening warmup — suppresses server-VAD speech_start
    # detections for the first N ms after start_recording. Mic AGC/AEC
    # haven't stabilized in that window and Silero VAD often false-trips
    # on the activation pop / DC offset / room tone, locking the session
    # in "speaking" state forever. The wake-word session has a similar
    # 2500ms warmup (WARMUP_MS in becca-wake.js); this is the server-
    # side equivalent for the always-listening capture loop.
    companion_always_listening_warmup_ms: int = 500          # [admin] suppress VAD speech_start for the first N ms
    # Optional VAD threshold override specifically for always-listening
    # mode. Default 0 = use the global ``voice_vad_speech_threshold``.
    # Raise to ~0.6 when the mic is always-hot and ambient noise floor
    # is producing false-positive speech_starts.
    companion_always_listening_vad_threshold: float = 0.0    # [admin] 0 = use voice_vad_speech_threshold
    # Prefix padding override for always-listening mode. Default
    # voice_vad_prefix_padding_ms (300ms) is too tight when the mic
    # is always-hot — VAD trips on the loudest syllable mid-utterance
    # and STT loses the leading words ("Hey can you" gets clipped to
    # "you" if the user emphasizes "you"). 700ms still lost soft
    # sentence openers (2026-06-11: "starts halfway through my
    # sentence"); 1500ms covers a slow lead-in. Safe to be generous —
    # the speech_start handler clamps the ring to post-TTS audio
    # (_trim_prefix_to_post_tts) so her own voice can't ride into the
    # user's transcript. 0 = use the global default.
    companion_always_listening_prefix_padding_ms: int = 1500  # [admin]
    # Threshold boost applied to the address classifier when the
    # AudioBus reports media is currently playing (YouTube, audiobook,
    # Grove). Mic bleed from playback regularly fires VAD; this raises
    # the bar so only very clear addressing ("play jazz", "stop")
    # passes during media. 0.0 = no boost. 0.10 = effective 0.95 with
    # the default 0.85 base.
    companion_address_media_boost: float = 0.10              # [admin]
    # Near-miss "I heard you" tell: when an AMBIENT (non-explicit) turn is
    # dropped but it was coherent + reply-shaped (router goal act/converse/
    # clarify) and its confidence landed within this band BELOW the effective
    # addressing threshold, the server tells the client so the widget shows a
    # faint, non-spoken acknowledgement instead of a silent void. Clearly-
    # ambient speech (incoherent, idle/drop goal, or confidence well under the
    # bar) stays fully silent — she must not flicker at every word across the
    # room. 0.0 disables the tell. 0.25 below the 0.55 base = tell on [0.30,
    # 0.55).
    companion_address_near_miss_band: float = 0.25           # [admin]
    # Conversation-window relaxation: for this many seconds after the
    # companion's own TTS, a coherent reply-shaped utterance (router
    # goal act/converse/clarify) bypasses the addressed/confidence
    # veto — the exchange is live, drops are too greedy mid-dialogue.
    # 0 disables.
    companion_followup_window_s: float = 12.0                # [admin]
    # Open-thread extension of the follow-up window: when the
    # companion's last line ended with a question, or a verb parked a
    # pending clarification, the conversation is demonstrably OPEN —
    # the user may read/think for half a minute before answering. The
    # relaxed gate holds this long instead of the base window. 0
    # disables the extension (base window still applies).
    companion_open_thread_window_s: float = 45.0             # [admin]
    # Cross-speaker CSM voice: feed the user's spoken turn (the STT clip) to
    # the Sesame CSM sidecar so her reply's prosody reacts to HOW they sounded,
    # not just the words — the dialogue conditioning that's the point of CSM
    # over a one-shot TTS. No-op unless her voice is CSM. Inverse (False):
    # self-context only (she stays prosodically consistent with herself but
    # deaf to the user). The clip lives only in the sidecar's RAM.
    companion_csm_cross_speaker: bool = True                  # [admin]
    # Her current mood -> a leading (emotion) tag the fine-tuned CSM voice was
    # trained on, so she sounds how she feels. Off by default: enable + tune the
    # affect->emotion mapping by ear once a fine-tuned voice is live.
    companion_csm_emotion_tag: bool = False                   # [admin]
    # Affect -> voice for OpenAI-omni style TTS (Higgs Audio v3 via the generic
    # openai-tts provider): prepend a natural style cue — (warm)/(excited) —
    # derived from her recency-gated mood so she *sounds* how she feels.
    # Model-agnostic (any /v1/audio/speech endpoint honouring style cues).
    companion_voice_emotion_tag: bool = False                 # [admin]
    # CSM voice residency: how the GPU model is held. "session" (default) =
    # warm on voice-session open, unload on close — follows the conversation,
    # no mid-talk reloads, frees VRAM when done. "timer" = the sidecar's idle
    # timer only (old behavior). "always" = warm but never unload (instant,
    # pins VRAM). No-op unless her voice is CSM.
    companion_csm_residency: str = "session"                  # [admin]
    # Results ring — turn-decayed memory of what she recently looked
    # at (full -> digest -> peek lifecycle, companion_runtime/ring.py).
    # turns = how many untouched exchanges a digest survives; the kill
    # switch reverts to the legacy always-push presence block.
    companion_results_ring_enabled: bool = True              # [admin]
    companion_results_ring_turns: int = 3                    # [admin]
    # Alert watch: NWS severe weather + USGS quakes near the saved home
    # location, pushed through the notifications pipeline (alerts.home
    # channel). Inert until a home location exists — weather.today
    # learns it from conversation.
    companion_alert_watch_enabled: bool = True               # [admin]
    # RSSHub base URL for expanding rsshub:// route shorthands in
    # Discovery RSS subscriptions. Default matches the compose.rsshub
    # overlay's service DNS; harmless when the service isn't running
    # (feeds soft-fail). Empty disables expansion.
    rsshub_base_url: str = "http://rsshub:1200"              # [admin]
    # ── Ambient tool policy ──
    # Gates which voice-callable tools the always-on companion widget
    # may invoke from its passive surface. Foreground voice call modal
    # is unaffected — the user explicitly opened that session and the
    # full tool set applies.
    #
    # Buckets are defined in augmentum/intent/manifest.py:
    #   * CORE        — notes, memory, navigation, reference lookups
    #   * INTERACTIVE — web search, discovery (opens a surface)
    #   * DISRUPTIVE  — media playback control, timers
    #   * COSTLY      — image generation
    #
    # Policies:
    #   "full"     — CORE + INTERACTIVE + DISRUPTIVE + COSTLY
    #                Equivalent to pre-policy behaviour.
    #   "safe"     — CORE + INTERACTIVE (recommended default)
    #                The widget can still take a note, recall memory,
    #                search the web — but won't auto-pause your music
    #                or kick off an image generation in the background.
    #   "minimal"  — CORE only
    #                Strictest setting. Useful for shared spaces, demo
    #                installs, or operators who want zero surprise
    #                background activity from the passive widget.
    #   "custom"   — Use companion_ambient_tool_allowlist verbatim
    #                (empty list collapses to minimal — no accidental
    #                full exposure from misconfiguration).
    companion_ambient_tool_policy: str = "safe"              # [admin] full | safe | minimal | custom
    companion_ambient_tool_allowlist: list[str] = []         # [admin] used when policy=custom; tool ids from the manifest buckets
    # ── Note-capture cleanup ──
    # When ON, exit from note.start_capture mode runs a short LLM pass
    # over the raw STT-captured chunk to:
    #   * Fix obvious homophones ("their/there/they're", "to/too/two")
    #   * Add paragraph breaks between distinct thoughts
    #   * Lightly punctuate run-on transcripts
    # Semantic content + word order are preserved verbatim. Pre-capture
    # note content is untouched (the cleanup only slices from
    # ``note_capture_baseline_chars`` forward).
    #
    # On any failure (timeout, parse error, model unavailable) the raw
    # transcript is kept as-is — never silently drop the user's notes.
    companion_note_capture_cleanup: bool = True              # [admin] off = always keep raw capture verbatim
    companion_note_capture_cleanup_timeout_ms: int = 8000    # [admin] hard ceiling for the cleanup model call
    # Image prompt expansion — when the architect dispatches
    # ``image.generate_with_defaults``, run an LLM expansion pass on
    # the user's raw prompt before sending to the image model. Turns
    # "a dog" into a scene-rich paintable description. Bounded by a
    # 4s timeout so a slow backend never blocks the dispatch.
    companion_image_prompt_expansion_enabled: bool = True    # [admin]
    companion_image_expansion_timeout_ms: int = 4000         # [admin]
    # ── Becca-direct chat path (accumulation thesis Step 1) ──
    # When ON (and the companion runtime is up), the chat router can
    # pick the ``becca_direct`` subagent for relational/conversational
    # turns. Routes through her own prompt composer + tier streaming —
    # same kernel, same voice as her voice channel. This is the seam
    # where her chat presence becomes real and the exemplar library
    # starts accumulating from chat turns. Default OFF: opt-in until
    # the seam has been validated against real traffic.
    # See docs/superpowers/specs/2026-05-23-accumulation-thesis.md.
    companion_becca_direct_enabled: bool = True
    companion_native_toolloop: bool = True              # [admin] Companion Agency MVP #1: becca_direct chat turns run the chat path's NATIVE tool loop (passthrough tier machinery + 5-tier parser + ActionTools) instead of prose-tag sieving. Falls back to the tag path on any loop failure. See docs/superpowers/specs/2026-06-10-companion-agency-design.md.             # [admin] Master switch for the becca_direct chat path. Default ON when companion_runtime_enabled: her chat presence becomes the default response path when persona is on.
    # ── Device tools (phone-as-capability-provider) ──
    # When ON, companion verbs that query the user's paired phone
    # (device.bluetooth_list, …) are live. They round-trip a request over
    # the always-on notification WebSocket to the Android foreground
    # service and await a result (DeviceCommandBus). Default ON (2026-06-21,
    # Matt's call): the APK now answers device_command frames (DeviceCommands.kt)
    # and the notify WS is wired, so phone actions (alarm/timer/dial/sms/open-app)
    # are part of the assistant. Each device verb still confirms or composes
    # before doing anything irreversible; set False to disable. Presence rides
    # the same connection and is NOT gated by this flag.
    # See docs/superpowers/specs/2026-06-17-phone-capability-provider.md.
    companion_device_tools_enabled: bool = True
    companion_device_command_timeout_s: float = 8.0     # [admin] Max wait for a phone to answer a device_command before the verb reports the phone unreachable.
    # Assist API screen-read (Android assistant-role Slice 2). When the user
    # summons Augmentum as their system assistant, onHandleAssist captures the
    # on-screen text and POSTs it to /api/architect/load (kind="screen") so the
    # companion's perception layer can answer "tell me about this." Default OFF:
    # the ROLE + presence are harmless, but ingesting whatever's on screen
    # (messages, banking) is the sensitive part. Default ON (2026-06-21, Matt's
    # call): the summon capture is the whole point of a phone assistant, and it
    # only fires when the user has explicitly made Augmentum their default
    # assistant + actively summons it. Gated server-side on the load endpoint;
    # set False to disable. Mirrors companion_device_tools_enabled.
    # See docs/superpowers/specs/2026-06-17-phase1-assistant-role-backbone.md.
    companion_assist_enabled: bool = True
    # ── Skill graph (accumulation thesis Step 3) ──
    # The capability-side accumulation substrate. When ON, the prompt
    # composer's Layer 5.6 injects relevant skills from her accumulated
    # graph — "approaches that have worked for similar things" — so
    # prior approaches inform current responses. Default OFF: opt-in
    # until skills have accumulated honest evidence. Skills BELOW
    # ``companion_skill_min_confidence_for_inject`` never reach the
    # prompt (this is the structural commitment against confabulated
    # capability).
    # See docs/superpowers/specs/2026-05-23-accumulation-thesis.md.
    companion_skills_enabled: bool = False                  # [admin] Master switch for skill graph injection at compose
    companion_skill_relevance_threshold: float = 0.6        # [admin] Min cosine similarity (problem_shape vs intent) for retrieval
    companion_skill_min_confidence_for_inject: float = 0.5  # [admin] Untested skills below this never inject. Untested skills default to 0.5.
    companion_skill_inject_top_k: int = 4                   # [admin] Max skills injected into a single prompt
    # Lesson registry (mig 270) — the learn-from-correction inverse of
    # the skill graph. Corrections the user made are held as lessons and
    # injected at compose as guardrails so the same mistake doesn't
    # recur. Two switches: injection (read path) + capture (nightly write
    # path from her reflections). Both default OFF — opt-in until the
    # registry has accumulated real corrections.
    # See docs/superpowers/specs/2026-05-23-accumulation-thesis.md.
    companion_lessons_enabled: bool = False                    # [admin] Master switch for lesson (correction) injection at compose
    companion_lessons_capture_enabled: bool = False           # [admin] Nightly capture of correction-lessons from her reflections
    companion_lessons_relevance_threshold: float = 0.6        # [admin] Min cosine similarity (situation vs intent) for retrieval
    companion_lessons_min_strength_for_inject: float = 0.5    # [admin] Lessons below this strength never inject
    companion_lessons_inject_top_k: int = 3                   # [admin] Max lessons injected into a single prompt
    # Today entry — daily in-her-voice reflection surfaced at the top of
    # the notes drawer. Cheap, polite default: she's already journaling,
    # this just surfaces the day's shape in her own words.
    companion_today_enabled: bool = True                # Generate + surface the daily Today reflection (gated by presence_mode != silent)
    companion_today_reflect_hour_local: int = 21        # [admin] Local hour at which the day "settles" (default 9pm)
    companion_today_max_chars: int = 360                # [admin] Soft cap on reflection length
    # ── Becca persona mode (top-level fork; the four-lane design lives in
    #    docs/superpowers/specs/2026-05-15-becca-*.md). When OFF (default),
    #    the chat path is byte-identical to today's modes; flipping ON makes
    #    Becca the user's point of contact, dispatching the IQ modes as tools
    #    or channels. Requires migrations 161-164 applied. ──
    companion_persona_mode: bool = False
    # Auto-summon the persona widget on every page load. Default True
    # preserves the original behavior; flip to False to keep persona-mode
    # enabled but spawn Becca only on demand via the header-logo summon
    # affordance (useful when the user opens many tabs and doesn't want
    # one widget per tab).
    companion_auto_summon: bool = True
    # Care-surface cadence — Lane 2 §3.4. 'sparse' = 1 every 3 days,
    # repetition_count >= 5 to graduate, threshold 0.75. 'normal' (default)
    # = 1/day max, repetition 3, threshold 0.62. 'lively' = 2/day, repetition
    # 2, threshold 0.55. User-adjustable in conversation or settings.
    companion_care_cadence: str = "normal"
    # IETF language tag for the locale-aware resource map (988 US,
    # Samaritans UK, etc.). Empty = auto-detect from system locale; falls
    # back to "international" (findahelpline.com) if no matching map.
    companion_locale: str = ""
    # Embodiment knobs (Lane 4) — all default to the least-intrusive value.
    companion_audio_cues: bool = False              # widget audio cues off by default
    companion_keyboard_shortcuts: bool = True       # Alt+B / Cmd+Shift+. discoverable shortcuts
    companion_notify_eod: bool = False              # end-of-day reflection ping (opt-in)
    companion_notify_drift_audit_push: bool = False # OS push on drift audit (opt-in)
    companion_cooldown_minutes: int = 210           # min minutes between non-user-initiated surfaces
    companion_quiet_hours_start: str = "24:00"      # surfacing suppressed during quiet hours
    companion_quiet_hours_end: str = "07:00"
    companion_discreet_auto_exit_minutes: int = 0   # 0 = manual exit only
    companion_discreet_location_aware: bool = False # opt-in: exit discreet on home geofence
    companion_always_raw: bool = False              # user opts out of Becca-wrap globally
    # Safety-floor classifier (Lane 2 §6) — per-surface thresholds.
    # Quarterly tune writes new values via settings store; defaults are
    # the launch-deployed values targeting FPR ≤ 0.5% / recall ≥ 0.65.
    companion_safety_floor_threshold_chat: float = 0.72
    companion_safety_floor_threshold_coder: float = 0.78
    # Slice 0 hush — ISO-8601 timestamp ("YYYY-MM-DD HH:MM:SS", UTC);
    # "" means not hushed. When set to a future time,
    # gates.is_hushed_now() returns True and surface-touching scorers
    # (reach_out, future moment_surface) score 0.0. Journal *writes*
    # remain unaffected — hush silences surfacing, not inwardness.
    companion_journal_hushed_until: str = ""

    # --- CPU vision fallback (retired SmolVLM sibling) ---
    # Vision is now a capability of the CLASSIFIER SLOT (Gemma): when its
    # model is VL+mmproj it IS the captioner. This setting only governs the
    # CPU-ONLY FALLBACK for no-GPU deployments whose classifier is text-only:
    # a small CPU VL model that LAZILY starts on first image and never runs
    # when the classifier (or a VL primary) can already see. ``enabled`` ==
    # "allow the CPU vision fallback" (default True so no-GPU boxes still get
    # vision). The model path is swappable — drop in a better small CPU VL
    # (SmolVLM2-500M / LFM2-VL-1.6B) when one lands; no code change needed.
    vision_provider_enabled: bool = True   # allow the CPU vision fallback (lazy; only fires when classifier is text-only)
    vision_provider_model_path: str = "/models/vision/SmolVLM2-500M-Instruct-Q8_0.gguf"   # swappable CPU VL fallback
    vision_provider_mmproj_path: str = "/models/vision/mmproj-SmolVLM2-500M-Instruct-Q8_0.gguf"
    vision_provider_backend_port: int = 8092  # llama-server port for the fallback (primary 8091, Slot C 8093, Slot B 8094)

    # --- Classifier engine (text sibling) ---
    # Second always-resident llama-server instance loaded with a small
    # text model (1B-3B class) dedicated to fast classification and
    # utility tasks. Shields voice + utility paths from whatever heavy
    # reasoning model the user has selected for chat. Mirrors the
    # SmolVLM-vision-sibling pattern: own port, own subprocess, never
    # competes with the primary engine.
    #
    # Lifecycle: auto-starts when companion_activation_mode is
    # "always_listening" (the only mode that genuinely benefits from
    # the latency floor) OR when classifier_engine_enabled is True
    # (manual override for users who want it for utility tasks even
    # without always-listening). Stops when neither condition holds.
    #
    # When live, the sibling registers as a provider backend and the
    # role resolver (resolve_model_for_role("classifier"/"utility"))
    # finds it automatically. classifier_model / utility_model
    # settings can then point at its model id explicitly, or stay
    # empty for the cascade to find it.
    classifier_engine_enabled: bool = False           # manual master switch (auto-overridden by always_listening mode)
    classifier_engine_model_path: str = ""             # path to small text GGUF, e.g. /models/classifier/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
    classifier_engine_backend_port: int = 8093         # llama-server port (primary uses 8091, vision uses 8092)
    classifier_engine_gpu_layers: int = 0              # 0 = CPU-only; ~700 MB for a 1.5B Q4 model. Voice fires <10x/min so CPU is fine
    classifier_engine_ctx_size: int = 4096             # classifier prompts are short — 4K is generous

    # --- Secondary local engine ("Slot B") ---
    # A second *user-driven* resident llama-server, so two arbitrary local
    # models stay loaded at once (LM Studio style). Distinct from the
    # vision/classifier siblings (those host fixed special-purpose models);
    # Slot B holds whatever GGUF the user loads into it from the model
    # manager. Same own-port/own-subprocess pattern as the siblings.
    #
    # Routing: the model loaded into Slot B is *pinned* to its backend in
    # ProviderRegistry (see pin_model) so chatting it hits the resident
    # process instead of swapping the primary. The slot is excluded from
    # catalog probing so it never collides with the primary's GGUF list.
    #
    # Per-model load config (idle timeout, gpu-layer cap, ctx) is NOT
    # stored here — it travels with the model via
    # ``engine.last_load.<model_id>`` (persist_load_options), so the same
    # config applies whichever slot the model is loaded into.
    engine_secondary_enabled: bool = True              # offer Slot B (zero-cost until a model is loaded; set False to hide on tiny boxes)
    engine_secondary_backend_port: int = 8094          # llama-server port (primary 8091, vision 8092, classifier 8093)
    engine_secondary_model: str = ""                   # model id last loaded into Slot B (re-pinned on boot; warmed only if its saved idle_timeout==0)

    # --- Managed classifier slot ("Slot C") ---
    # The augmentum-MANAGED counterpart to the external compose.classifier.yaml
    # container: a resident llama-server whose model the user can pick at setup
    # AND swap from the UI with NO container recreate. Serves the classifier +
    # utility roles (registers under the "classifier" backend key) and, when its
    # model is VL + launched with an mmproj projector, the vision/captioning role
    # too (retiring the SmolVLM sibling on GPU boxes). Supersedes the start-only
    # classifier_engine_* sibling above. Precedence: an external Docker classifier
    # (AUGMENTUM_CLASSIFIER_BASE_URL) still wins — Slot C registers only if that
    # key is empty, so existing installs are untouched. Per-model gpu-layer cap /
    # ctx / mmproj travel with the model via engine.last_load.<model_id>.
    classifier_slot_enabled: bool = False              # opt-in for now; Phase E flips fresh-install default on
    classifier_slot_model: str = ""                    # model id/path loaded into Slot C (e.g. unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL or a GGUF path); empty = none
    classifier_slot_backend_port: int = 8093           # llama-server port (primary 8091, vision 8092, Slot B 8094)
    classifier_slot_gpu_layers: int = 0                # 0 = CPU-only; 99 to offload (Gemma-4-E2B/E4B want GPU for the 2.5s budget)
    classifier_slot_ctx_size: int = 4096               # classifier/utility prompts are short; raise for big tool catalogs
    voice_smart_turn: bool = True             # SmartTurn v3: learned turn-completion (prevents premature cutoff on pauses)
    voice_smart_turn_threshold: float = 0.5   # Probability threshold for "turn complete" (0.5 = balanced)
    voice_smart_turn_max_wait_s: float = 3.0  # Safety valve: force complete after this much silence even if model says incomplete
    voice_smart_turn_max_deferrals: int = 3   # Cap on how many times the veto deadline may be DEFERRED by "still speaking". Each deferral adds max_wait_s; without a cap, background noise that Silero reads as speech defers forever → the turn "feels super long" and never ends (2026-06-13). After this many deferrals, finalize regardless. A real multi-pause thought is recoverable via continuation-merge.
    voice_smart_turn_min_veto_confidence: float = 0.3  # Veto confidence = threshold − prob (NOT prob itself). Vetoes below this are overridden — with defaults, prob in (0.2, 0.5) defers to VAD; prob ≤ 0.2 (model sure the user is still talking) is honored.
    voice_bargein_min_speech_ms: int = 250  # Sustained speech required before barge-in cancels TTS — filters beeps/clicks/door-slams that pass VAD but don't sustain
    # Endpointing inversion (2026-06-13 latency MVP): when smart-turn is
    # available it becomes the END-OF-TURN GATE rather than a veto layered
    # on top of a conservative VAD silence wait. The VAD endpoints early
    # (this short silence), smart-turn (~65ms) confirms "actually done";
    # an "incomplete" verdict keeps listening via the existing
    # veto/continuation path. Cuts ~1s of dead air off every turn with
    # zero effect on the generated reply. 0 disables (keeps the legacy
    # full voice_silence_threshold_ms wait). Falls back to the legacy
    # wait automatically when smart-turn is unavailable.
    voice_fast_endpoint_ms: int = 700         # VAD silence to endpoint when smart-turn gates (0 = legacy behavior). 700ms (was 300) gives grace for inter-sentence pauses within one turn — smart-turn correctly flags a short complete sentence as "done" and can't tell "done" from "about to continue", so a longer endpoint window is the guard (verified 2026-07-16: complete=True prob 0.83-0.99 on 2-3s clips cut users off mid-thought). Continuation-merge is the deeper class fix.

    # --- Typography ---
    typography_custom_fonts: str = "[]"        # JSON array of {name, key} custom Google Fonts
    typography_selected: str = "system"        # Active typography preset key
    typography_text_scale: str = "1"           # Global text size multiplier (0.7–1.4)

    # Master bypass — skip ALL mic preprocessing (denoise, highpass, NS, AGC)
    # so raw capture reaches VAD/STT untouched.  For A/B-ing STT accuracy
    # against the processed chain; the per-stage flags below let you then
    # strip one stage at a time.
    voice_preprocess_bypass: bool = False
    voice_denoise_enabled: bool = True        # DTLN neural denoiser (requires ONNX models)
    voice_denoise_model_dir: str = ""         # Path to DTLN ONNX models (default: /home/augmentum/.dtln)
    voice_highpass_hz: int = 80               # Highpass filter cutoff in Hz (0 to disable)
    voice_audio_agc: bool = True              # Automatic gain control before VAD/STT
    voice_audio_ns: bool = True               # Noise suppression before VAD/STT
    voice_audio_agc_target_dbfs: int = -16    # AGC target level in dBFS (louder target for better VAD)
    voice_audio_ns_level: int = 2             # Noise suppression level (0-4, higher = more aggressive)
    voice_stt_normalize: bool = True          # Peak-normalize audio before STT
    voice_vad_speech_threshold: float = 0.4  # Silero VAD speech probability threshold
    voice_vad_min_speech_ms: int = 150       # Minimum speech duration to count (filters noise)
    voice_vad_min_start_frames: int = 2     # Consecutive speech frames before triggering start (~64ms)
    voice_vad_prefix_padding_ms: int = 300   # Audio to keep before speech start for STT context
    voice_server_vad: bool = True            # Use server-side Silero VAD (False = client VAD)
    voice_streaming_stt: bool = True         # Use streaming STT when provider supports it
    voice_stt_endpointing_ms: int = 200      # Deepgram endpointing sensitivity (silence gap)

    # --- Voice: Speaker Verification ---
    voice_speaker_verify: bool = True          # Enable speaker verification (reject non-enrolled voices)
    voice_speaker_threshold: float = 0.45      # Cosine similarity threshold (0-1) for enrolled speaker
    voice_speaker_verify_seconds: float = 3.0  # Min seconds of speech for reliable verification (short utterances skip check)

    # --- Thinking / Reasoning ---
    think_enabled: bool = True
    source_validation_enabled: bool = True  # Post-draft validation against search sources

    # --- System ---
    # RESERVED — currently unenforced. The real concurrency limit is per-
    # model in ``LlamaCppBackend._slot_lock``. Documented here so a future
    # global cap can be wired without re-introducing the dead semaphore
    # that lifespan used to create.
    max_concurrent_requests: int = 10
    max_request_body_bytes: int = 52_428_800  # 50 MB (base64 images are large)
    ws_max_frame_bytes: int = 4_194_304  # 4 MB — caps individual WebSocket frames (uvicorn --ws-max-size)
    rate_limit_enabled: bool = False  # Off by default, opt-in
    rate_limit_rpm: int = 120  # requests per minute per IP (legacy, used by old token-bucket limiter)
    rate_limit_chat_rpm: int = 30
    rate_limit_image_rpm: int = 10
    rate_limit_voice_rpm: int = 5
    rate_limit_upload_rpm: int = 30
    # External-API request bounds. Default-ON protection (unlike the rate
    # limiter above, which is opt-in) so a box exposed beyond a trusted LAN
    # can't be OOM'd / pinned by an oversized or abusive request. 0 disables
    # an individual cap. ``max_request_body_bytes`` (above) is the general
    # transport ceiling; ``files_upload_max_request_bytes`` is the higher
    # tier for upload endpoints — both enforced as a Content-Length check in
    # RateLimitMiddleware.
    api_embeddings_max_items: int = 2048           # max strings per /v1/embeddings request
    api_embeddings_max_chars: int = 1_000_000      # max total input chars per /v1/embeddings request
    api_tts_max_chars: int = 50_000                # max /v1/audio/speech input length
    api_stt_max_bytes: int = 26_214_400            # 25 MB — max /v1/audio/transcriptions audio
    request_timeout: float = 600.0
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = False
    enable_request_logging: bool = False

    # --- SSRF ---
    # Hostnames/CIDRs exempt from SSRF blocking (Docker-internal services, etc.)
    # Comma-separated.  Examples: "searxng,executor,ollama,172.18.0.0/16"
    ssrf_allowlist: str = ""

    # --- HTTP Client ---
    # 15s (not 5s): tolerant of the slow-TLS-handshake tail on internet-remote
    # providers (e.g. api.deepseek.com's CloudFront edge). Local llama-server
    # connects in <30ms regardless, so the higher ceiling costs nothing there.
    http_connect_timeout: float = 15.0
    http_read_timeout: float = 600.0
    http_write_timeout: float = 30.0
    http_pool_timeout: float = 5.0
    http_max_connections: int = 100
    http_max_keepalive: int = 20

    # --- Retrieval fusion ---
    memory_fusion_method: str = "rrf"  # "rrf" (reciprocal rank) or "convex" (alpha-weighted)
    memory_fusion_alpha: float = 0.7   # vec weight for convex fusion (1-alpha = FTS weight)

    # --- Reranking ---
    reranker_enabled: bool = True
    reranker_model: str = "jinaai/jina-reranker-v1-tiny-en"
    reranker_top_k: int = 5  # Return top K after reranking

    # --- Embeddings ---
    embedding_threads: int = 0  # ONNX threads for embedding model (0 = auto-detect CPU count)
    embedding_batch_size: int = 64  # Max texts per ONNX inference pass
    # CPU-by-default. The shipped embedding model (nomic-embed-text-v1.5-Q)
    # is INT8 quantized; ORT's CUDA EP inserts ~156 memcpy nodes for it
    # (GPU↔CPU per-op shuttling), which makes single-query inference SLOWER
    # on GPU than CPU and still holds ~500 MiB of VRAM. Same logic applies
    # to the reranker session. Flip True only on boxes with VRAM headroom
    # to spare AND a batched workload where the per-op overhead amortizes
    # (knowledge-pack convert, large dream-cycle compactions). The existing
    # ORT_PROVIDERS=CPUExecutionProvider env override still wins.
    embedding_use_gpu: bool = False

    # --- Document RAG ---
    document_rag_enabled: bool = True
    document_rag_recall_limit: int = 3  # Max document chunks to inject per request
    document_rag_contextual_retrieval: bool = False  # LLM-generated chunk context at ingest
    document_rag_query_analysis: bool = True  # LLM query classification before search
    document_rag_query_analysis_model: str = ""  # empty = fallback to memory extraction model
    document_rag_query_analysis_timeout: float = 2.0  # seconds
    document_rag_cliff_ratio: float = 0.3  # adaptive K score drop-off threshold
    document_rag_max_context_tokens: int = 1500  # injection token budget cap (~6000 chars)
    # Phase 2 experimental — validate via benchmark before enabling by default
    document_rag_min_representation: bool = False  # guarantee top-1 per sub-query in decompose
    document_rag_decompose_sufficiency: bool = False  # per-sub-query coverage check

    # --- Startup ---
    startup_warmup: bool = True  # Preload embedding + reranker models in background on startup

    # --- Signal aggregator ---
    # Daily pass that pulls signals from bug_finder_runs + companion_journal
    # into the signal_events table (migration 206). Off by default — turn on
    # to start accumulating the substrate; no UI yet, query the table by SQL.
    signals_aggregator_enabled: bool = False

    # --- Memory ---
    memory_enabled: bool = True
    memory_llm_extraction_enabled: bool = True
    memory_recall_min_score: float = 0.35
    # Stricter floor applied only when injecting memories into the system prompt
    # (the analytical hint path and the memory_recall tool still use the recall
    # floor above).  Raise this to suppress weakly-matching memories from being
    # auto-injected every turn; 0.0 is a no-op (falls through to recall floor).
    memory_inject_min_score: float = 0.55
    memory_recall_limit: int = 5
    memory_summary_max_chars: int = 300
    memory_scope_by_mode: bool = True
    memory_inject_analytical: bool = False
    memory_inject_agentic: bool = False
    # Modes whose stream content gets mined for memories via batch LLM
    # extraction. Explicit "remember X" instructions are still captured in
    # any non-narrative mode (the user opted in by phrasing). Modes NOT in
    # this list (coder, builder, voice, etc.) only capture explicit facts —
    # their stream content isn't treated as getting-to-know-you chat, which
    # avoids project-build details ("the game must run on a web server")
    # landing as durable user facts. Narrative remains a full skip via its
    # own gate because in-character "remember X" lines aren't user facts.
    memory_capture_modes: list[str] = ["passthrough", "analytical", "agentic"]
    # Drop extracted facts that describe an artifact (work-in-progress)
    # rather than the user. Triggers only when three signals align:
    # artifact noun + property predicate + no identity-grounding language.
    # Disable for users whose domain legitimately produces artifact facts
    # at scale (novelist describing characters, game designer cataloguing
    # mechanics). Explicit "remember X" facts bypass this gate regardless,
    # so users can always force-save artifact context with explicit phrasing.
    memory_anti_projection_enabled: bool = True
    memory_dedup_threshold: float = 0.88
    memory_contradiction_threshold: float = 0.78
    # Shadow-touch: when a new extraction batch lands, PROVISIONAL memories
    # whose embedding cosine-similarity to the batch ≥ this threshold get
    # their access_count bumped — the natural-reinforcement signal that
    # eventually promotes them to ACTIVE. Lower than dedup (0.88) and
    # supersession (0.78) because shadow-touch only signals "recurring
    # topic", not "same fact". See
    # docs/superpowers/specs/2026-05-31-memory-establishment-rebalance.md.
    # Fallback path uses `content LIKE %keyword%` when sqlite-vec is absent.
    memory_shadow_touch_threshold: float = 0.70
    memory_shadow_touch_max_per_batch: int = 5  # cap bumps so a broad batch can't cascade
    # PII scrub at ingest: redact API keys / JWTs / emails / IPs from
    # content before embedding + storage. EXPLICIT-source facts
    # ("remember my email is X") bypass the scrub — the user opted in
    # by phrasing. See augmentum/memory/scrub.py for patterns.
    memory_pii_scrub_enabled: bool = True
    # Durability passthrough: LLM extractor labels each fact as
    # durable | transient | unknown. Transient facts route to
    # PROVISIONAL with a longer TTL so ephemeral project state
    # ("currently building X") naturally ages out unless it proves
    # itself durable via the cosine shadow-touch path. EXPLICIT
    # facts are exempt — user phrasing is the opt-in. See
    # docs/superpowers/specs/2026-05-31-memory-establishment-rebalance.md.
    memory_durability_classification_enabled: bool = True
    memory_durability_transient_ttl_days: int = 30
    # HyDE — at recall time, generate a hypothetical-answer expansion of
    # the query via a small LLM and use its embedding as a third RRF leg.
    # Improves recall on short / lexically-thin queries. Default off so
    # the feature ships dark; flip on per-user after eval. Adds ~100ms
    # latency per recall when enabled.
    memory_hyde_enabled: bool = False
    memory_hyde_model: str = ""  # empty = utility role resolver default
    # Retroactive demotion: nightly sweep moves long-idle ACTIVE memories
    # to ARCHIVE so the recall surface stays current. Exemptions:
    # importance >= floor, source EXPLICIT, tier CORE, retrieval_count
    # above min. ARCHIVE memories remain searchable (scored 0.7x) but
    # leave the inject-eligible pool. Every transition is recorded in
    # memory_tier_history so the inspector UI can show + revert. See
    # docs/superpowers/specs/2026-05-31-memory-establishment-rebalance.md.
    memory_retroactive_demotion_enabled: bool = True
    memory_demotion_idle_days: int = 180
    memory_demotion_min_retrievals: int = 1
    memory_demotion_importance_floor: float = 0.7
    memory_demotion_sweep_interval_seconds: int = 86400  # 24h between sweeps

    # --- Memory: LLM extraction ---
    memory_llm_extraction_model: str = ""  # Empty = use default backend's default model
    memory_extraction_batch_size: int = 4  # Accumulate N message pairs before running LLM extraction
    memory_auto_approve: bool = False  # Skip notification UI, store all extractions as ACTIVE immediately
    memory_extraction_max_buffers: int = 1000  # max session buffers for extraction (LRU eviction)
    memory_decay_max_counters: int = 1000      # max session counters for KG decay scheduling (LRU eviction)

    # --- Memory: Consolidation ---
    memory_consolidation_enabled: bool = True  # LLM merge of related facts on write (requires backend)

    # --- Memory: Core profile ---
    memory_core_profile_enabled: bool = True   # Always-in-context user summary (no LLM cost)
    memory_core_profile_max_tokens: int = 500
    memory_core_profile_rebuild_interval: int = 5  # After N extractions

    # --- Memory: Compaction ---
    memory_compaction_enabled: bool = True       # Background periodic cleanup
    memory_compaction_interval_hours: float = 24.0
    memory_compaction_max_age_days: float = 30.0

    # --- Agentic ---
    agentic_enabled: bool = True
    agentic_max_steps: int = 20
    agentic_artifact_theme: str = "slate"  # slate, corporate, modern, emerald, rose
    agentic_image_model: str = ""          # Image model for agentic illustrations (empty = use image_default_model)
    agentic_default_autonomy: int = 2      # 1=suggest, 2=ask, 3=inform, 4=autonomous
    agentic_artifact_dir: str = "data/artifacts"
    agentic_max_artifact_size_mb: int = 50
    agentic_checkpoint_enabled: bool = True
    agentic_step_timeout: float = 300.0    # per-step timeout in seconds
    agentic_native_tool_use: bool = True   # Use structured tool_calls in generative steps (falls back to TOOL_CALL: text if backend doesn't emit them)
    agentic_max_insertions: int = 2        # Max mid-flow re-planned steps inserted per task (when review keeps failing after revision)
    agentic_dynamic_max_steps: int = 10    # Max iterations of the dynamic ReAct-style loop (flow.kind='dynamic')

    # --- Coder Mode ---
    coder_idle_timeout: int = 1800  # Seconds before pausing/stopping idle containers (pause when coder_pause_idle, else stop)
    coder_pause_idle: bool = True  # Two-stage idle reaper: at coder_idle_timeout pause (cgroup freeze, RAM held, sub-second resume); only stop after a further coder_pause_stop_after_seconds with no activity. Set False to keep the original single-stage stop behavior.
    coder_pause_stop_after_seconds: int = 21600  # Additional idle time AFTER pause before fully stopping (6h default). Frees RAM but loses fast-resume. 0 disables the stop step (pauses persist until the user returns).
    coder_max_workspaces: int = 3  # Max concurrent workspace containers
    coder_default_cpu: float = 2.0  # CPU limit per workspace
    coder_default_memory: str = "2g"  # Memory limit per workspace
    coder_default_pids: int = 256  # PID limit per workspace
    # Tooling profile for new workspaces: "standard" (Python/JS baseline),
    # "power" (+devops tools), "browser" (power + Playwright + Chromium).
    # Default "browser" so end-to-end tests + headless browser flows work
    # out of the box. Trade-off: ~3-4 min slower first-time setup per
    # workspace (the Chromium install). Override per-workspace via the
    # creation dropdown.
    coder_default_tooling_profile: str = "browser"
    # Docker named volume that backs augmentum's /data dir. Used as the
    # mount source for the per-project bare-repo subpath bind into coder
    # workspace containers. Default matches compose.yaml; users with a
    # custom compose may override. Empty disables the bare-repo mount
    # (workspace falls back to `git init` legacy behavior).
    # See Phase 1 / PR-1.2 of the Integrated Coding Nervous System spec.
    coder_bare_repo_volume_name: str = "augmentum_data"
    coder_fast_model: str = ""  # Model for suggestions/summaries (empty = auto)
    coder_suggestion_enabled: bool = True  # Enable intent suggestions
    coder_workspace_image: str = "ubuntu:24.04"  # Default base image
    coder_auto_lint: bool = True  # Run lint after every successful write/edit (Aider pattern)
    coder_lint_timeout: float = 8.0  # Per-file lint command timeout in seconds
    coder_maker_agreements_enabled: bool = True  # Inject the user's accrued Working Agreements (mig 273 — durable, model-agnostic "how this maker wants to be worked with") into the coder system prompt each turn. User-owned; no-op until the user has accrued any. See augmentum/coder/maker_agreements.py.

    # --- Coder Preview Origin Isolation (Phase 1) ---
    # When enabled, the live preview iframe is served from a separate
    # origin (different port on the same host) so a malicious npm dep in
    # a workspace cannot call /api/* with the user's session cookies.
    # Off by default: requires Caddy/compose to expose the isolated port.
    # See docs/superpowers/specs/2026-05-27-preview-origin-isolation-design.md.
    coder_preview_isolation_enabled: bool = False
    # --- Content Iframe Origin Isolation (Phase 2) ---
    # Extends the same isolated-origin mechanism beyond the coder
    # preview to other "untrusted content" iframes — ZIM knowledge
    # packs first, with game bundles and emulator artifacts to follow.
    # When on (alongside the listener infrastructure), the ZIM iframe
    # is served from the isolated origin so a hostile pack cannot
    # reach the user's main session cookies via /api/*. Off by default
    # for the same reason as the coder flag: requires the operator
    # to wire Caddy/compose for the isolated listener. The browse.js
    # ZIM iframe falls back to today's same-origin behaviour when the
    # mint endpoint returns 501, so flipping this on without the
    # listener is harmless (just doesn't actually isolate).
    content_iframe_isolation_enabled: bool = False
    # External port the isolated preview listener answers on. Caddy must
    # route this port to the FastAPI app with the
    # X-Augmentum-Preview-Listener header set.
    coder_preview_isolated_port: int = 6444
    # Explicit origin override (empty = derive from request host + isolated
    # port). Set this when running behind a non-standard reverse proxy or
    # when the public host differs from the FastAPI bind host.
    coder_preview_isolated_origin: str = ""
    # TTL on the one-time token minted by the main app and consumed by
    # the isolated origin on first request. Short window because the
    # parent UI calls mint → embeds → browser redeems immediately.
    coder_preview_token_ttl_seconds: int = 60
    # TTL on the preview-session cookie that authenticates subsequent
    # in-iframe requests (Vite assets, HMR WS, dev-server fetches).
    # Sliding — extended on every successful request. Hard cap 8 hours
    # enforced server-side.
    coder_preview_session_ttl_seconds: int = 1800
    coder_lint_max_chars: int = 1500  # Max lint output appended to tool result
    # Phase 3.2: in-process syntax verification (ast.parse, json.loads) after
    # every successful write. Faster + more deterministic than lint; catches
    # the cheapest most-common failure mode (unparseable file) before the
    # subprocess hop. Independent toggle from lint — verify can run alone if
    # the project has no linter, or alongside it.
    coder_auto_verify: bool = True
    coder_reflexion_on_break: bool = True  # Stream a self-critique before each streak-break termination
    coder_reflexion_max_tokens: int = 220  # Cap on reflection length

    # When True the coder agent loop runs as a detached asyncio task and
    # the HTTP stream is a subscription to a per-run broker. Client
    # disconnect (mobile screen sleep, tab switch) no longer kills the
    # run — the UI reattaches via GET /api/coder/runs/{id}/stream when
    # it comes back. Stop button must call POST /api/coder/runs/{id}/
    # cancel; aborting the fetch is no longer enough to cancel the run.
    # See augmentum/coder/run_broker.py for the dispatch layer.
    coder_background_runs: bool = True
    # When a mid-run steer lands while the model is still streaming its
    # reasoning (no committal content/tool output yet), interrupt the in-flight
    # generation, fold the steer into history, and re-generate so the model
    # addresses the new content immediately — instead of the steer waiting for
    # the next iteration boundary, which a stuck reasoning loop never reaches.
    # The looping partial reasoning is discarded. Kill switch: set False to
    # restore boundary-only steer delivery. See handler._stream_and_parse_live.
    coder_steer_interrupt_reasoning: bool = True
    # Inject a shim into the live preview that captures console.error /
    # window.onerror / unhandledrejection from the USER's real preview session
    # (which the headless browser tools — fresh cold loads — structurally miss)
    # into a per-workspace buffer the model reads via browser_snapshot + a
    # turn-top auto-inject. See augmentum/coder/preview_console.py. Off → the
    # preview is unmodified and the model only sees errors it reproduces itself.
    coder_preview_console_capture: bool = True
    # Live-preview capture: when the user's coder preview is open, browser_screenshot
    # grabs the frame their real GPU already rendered (via a proxy-injected capture
    # agent + the /ws/coder/preview-capture socket) instead of re-rendering a heavy
    # WebGL/Three.js page in the headless, GPU-less workspace (6-45s+ or a timeout).
    # Off → screenshots always use the headless path. See augmentum/coder/preview_capture.py.
    coder_preview_live_capture_enabled: bool = True
    # User-controlled pacing for the coder agentic loop. Fast cloud models
    # (Gemini flash-lite, Groq, etc.) can fire tool calls quickly enough to trip
    # provider rate limits (429s / "too many requests"), which the loop then
    # burns retries on. When enabled, sleep this many seconds before each model
    # request in the loop so the cadence stays under the provider's limit. Off
    # by default (0 delay); the user sets the seconds when they enable it. See
    # augmentum/modes/coder/handler.py request pacing.
    coder_request_delay_enabled: bool = False
    coder_request_delay_seconds: float = 5.0

    # Token-count cap on the ``content`` argument of file_write. Above
    # this, the tool refuses with a redirect to ``code_edit`` SEARCH/
    # REPLACE blocks. Set well below typical model output budgets
    # (4096-8192 tokens) so a full-file rewrite tool call can't run
    # the response out of output budget mid-arguments — that failure
    # mode is what caused the 2026-05-27 cascade where the model
    # looped 8 times calling file_write with truncated args.
    #
    # 0 = uncapped, matching Claude Code's `Write` and Codex CLI's
    # `apply_patch` (both ship without any pre-emptive write cap). The
    # D1 truncation-detection layer in modes/coder/handler.py catches
    # the actual failure mode (mid-arguments cutoff via
    # finish_reason="length" or empty args_str) and surfaces a
    # structured "switch to code_edit_batch" hint to the model, so the
    # pre-emptive cap is redundant on capable models.
    #
    # Raise this above 0 only for weak local models with tiny output
    # budgets (≤4K) where you want a hard refusal before the model
    # even tries. Tunable live via PUT /api/config/tools with this
    # key. Default flipped 2026-05-31 from 6000 → 0 after CC/Codex
    # parity review.
    coder_file_write_max_tokens: int = 0

    # Fraction of the model's context window reserved as headroom
    # before the auto-compactor's threshold. Output budget + tool
    # schemas + reasoning all live in this slice — too low and the
    # response truncates mid-tool-call; too high and large windows go
    # unused. 0.10 = 10% reserve, leaving 90% of the window usable
    # (previously 15% reserve × 65% usable utilization = ~55% of
    # window — over a third wasted on a 131K model). Bumped
    # 2026-05-31. Tunable live via PUT /api/config/tools.
    coder_context_reserve_pct: float = 0.10

    # Per-response output budget for LOCAL backends (llama-server, etc.)
    # expressed as a percentage of the loaded model's context window.
    # For a 128K-ctx Qwen3.6 at 25% this yields a 32K output cap — wide
    # enough that a reasoning model (3-5K thinking) emitting a 5K-token
    # file_write JSON body still finishes inside budget, with room to
    # spare for tool-call schemas.
    #
    # The previous flat 8192-token mode-hint default (inference_hints.py)
    # was 14× smaller than the available headroom on a 128K-ctx model;
    # reasoning models routinely hit it on moderate file writes. Default
    # raised to 25% 2026-05-31 after the file_write truncation loop.
    #
    # CLOUD backends are NOT affected — Anthropic/OpenAI/DeepSeek-Chat
    # use the cloud provider's own default (or whatever the user set
    # explicitly), since those APIs charge per token and have their own
    # backend-specific safe ceilings. This setting only ratchets UP the
    # per-response cap for local; it never reduces an explicit value
    # below the mode hint.
    #
    # Range 5-90. Set to 0 to opt-out entirely (fall back to the mode
    # hint 8192 for local too). Tunable live via PUT /api/config/tools.
    coder_local_max_tokens_pct: int = 25

    # Absolute upper bound (tokens) on the local output budget floor
    # computed from ``coder_local_max_tokens_pct``. Caps the floor so a
    # 512K-ctx model doesn't request 128K of output (which most
    # backends won't honor anyway and which wastes scheduling). 32768
    # is comfortably above any single-file write a reasoning model
    # produces in practice. Set to 0 for no absolute cap.
    coder_local_max_tokens_cap: int = 32768

    # CLOUD counterpart to the local ratchet above. The local floor is derived
    # from ctx%; cloud backends never got that, so coder writes on a cloud
    # model were pinned at the flat 8192 mode-hint regardless of the model's
    # real output ceiling (DeepSeek allows 384K — a 47× under-cap that chopped
    # the trailing ``path`` off large file_write JSON). On a large-output
    # (coder) request, raise the budget toward this floor, then CLAMP to the
    # model's documented ``ProviderProfile.max_output`` so capable models get
    # room while small-cap models (Cohere R+ 4K, Perplexity 8K) never receive a
    # value their API rejects. Only applies when max_output is known (>0);
    # unknown providers stay at status quo. 32768 fits any realistic single-file
    # write + reasoning. Set 0 to disable the cloud floor. Live via PUT
    # /api/config/tools.
    coder_cloud_max_tokens_floor: int = 32768

    # Workspace-kernel v2 (docs/superpowers/specs/2026-05-16-workspace-kernel-design.md).
    # When True the kernel maintains /workspace/.augmentum/ files (plan.md
    # first; failures.md, world.md, etc. follow in later migrations) and
    # suppresses the corresponding per-iteration sticky-reminder sections
    # so the model reads the files on demand instead of having content
    # re-framed into the message stream every turn. Flipped on 2026-05-16
    # for dogfood after the migration-1 slice shipped — flip back to
    # False if regressions appear before migrations 2+ land.
    coder_kernel_v2: bool = True

    # Inline kernel facts in the system prompt (2026-05-28). When
    # True, _act_native folds a compact <workspace_facts> block
    # (identity summary + constraint/gotcha observations) into its
    # sys_text at turn-start. Defaults on because the typical user
    # runs a smaller local model that doesn't reliably read the
    # kernel files on demand. Disable for strong models that prefer
    # the spartan on-demand pattern from the 2026-05-16 design — the
    # kernel hint still points at /workspace/.augmentum/ either way.
    coder_kernel_inline_facts: bool = True

    # Per-workspace code-intelligence layer (coder/code_intel.py):
    # symbol index in the coder_indexes sidecar DB + find_symbol /
    # file_outline tools. Master switch — off disables the tools,
    # turn-start builds, and mutation-hook reindexing.
    coder_code_intel_enabled: bool = True
    # Inject the byte-stable <repo_map> block (file tree + top symbols)
    # into the coder stable prompt prefix. KV-cacheable: no line numbers
    # or timestamps, so it only re-renders when structure changes.
    coder_repo_map_enabled: bool = True
    # Char budget for the rendered repo map block.
    coder_repo_map_max_chars: int = 4000

    # Auto-seed .augmentum/objective.md from the first substantive
    # user message of a session (2026-05-28). Skipped when the
    # message is short (< 30 chars) or matches a continuation
    # pattern — a casual "hi" or "continue" shouldn't get pinned
    # as the session's anchor. Once seeded, the file is user-
    # curated: the agent reads it every turn but is directed not
    # to edit without explicit permission. Disable for workspaces
    # where the user prefers writing objective.md by hand.
    coder_kernel_auto_seed_objective: bool = True

    # --- Heavyweight model slot (2026-05-31) ---
    # Globally-configured "frontier / heavyweight tier" model that any
    # feature can reference when it wants frontier-quality intelligence
    # without forcing the user to hardcode the same model id in every
    # power / role / mode-handler config. Set ONCE here, consumed by:
    #
    #   * Role files with ``preferred_model: "$heavyweight"`` (or empty
    #     ``preferred_model`` + ``tier: heavyweight`` frontmatter)
    #   * Future power files at the verifier / pre_finish window
    #   * Slash commands like /second-opinion
    #   * Mode handlers' "hard decision" fallback (when a local model
    #     reports uncertainty)
    #
    # Accepts the standard multi-provider model-spec syntax used by the
    # subagent dispatcher:
    #   "gpt-5.5"                          → default backend's gpt-5.5
    #   "gpt-5.5@openai"                   → OpenAI specifically
    #   "claude-opus-4-7@anthropic"        → Anthropic
    #   "claude-opus-4-7@fabric:tower"     → peer machine routing
    #
    # Empty string = no heavyweight configured; consumers fall back to
    # whatever default they would have used. Recommend setting once
    # in the UI (Settings → Models → Heavyweight model).
    heavyweight_model: str = ""

    # Game-foundry visual verification model. Empty = "Auto" (VisionRouter's
    # current capability/workload routing decides). A pinned id is passed as
    # the `override` to resolve_model_for_role so the user's choice wins. Never
    # auto-selected — surfaced as a picker with "Auto — current routing" first.
    coder_visual_verify_model: str = ""

    # --- Subagent dispatch (task_dispatch tool) ---
    # When True, the coder mode exposes the task_dispatch tool to the
    # lead model so it can spawn focused subagents (explore / plan /
    # review / research built-ins + any .augmentum/agents/*.md the user
    # drops in). When False the tool is excluded entirely from the
    # registry — the model never sees it, no perf cost.
    #
    # Default flipped True 2026-05-31 — the substrate has shipped (6
    # built-in roles + multi-provider routing + the
    # coder_subagent_runs persistence table) and the parent-loop
    # cost when the model doesn't call it is zero (just an extra
    # tool entry in the schema). Set to False to hide the tool from a
    # specific session/user that's still on a model that botches
    # delegation.
    coder_subagents_enabled: bool = True

    # System-driven delegation: when the subagent-router Power activates on
    # an explore-shaped ask at pre_plan, the coder handler dispatches an
    # ``explore_codebase`` subagent ITSELF (deterministic) instead of merely
    # nudging the model toward task_dispatch — which local/open models
    # reliably ignore (validated 2026-06-19: Qwen3-Coder used code_grep/
    # file_list, never the delegation tool, even with the trigger-shaped
    # prompt). The exploration findings are injected into the plan context so
    # the plan is grounded in a real repo-wide read. Requires
    # ``coder_subagents_enabled``. Off → falls back to guidance-only nudging.
    coder_subagent_auto_explore: bool = True

    # Max concurrent in-flight subagents per parent coder turn. The
    # role-file's own ``parallelism.max_concurrent`` further caps the
    # SAME role; this is the cross-role ceiling. 4 mirrors Claude
    # Code's typical parallel-Task budget.
    coder_subagent_max_concurrent: int = 4

    # Max recursion depth — a subagent spawning a subagent spawning a
    # subagent. 1 means leaf-only (no nesting), 2 allows one level,
    # etc. The role file's ``permissions.can_spawn_subagents`` must
    # also be True for nesting to actually happen; this is the hard
    # ceiling for safety.
    coder_subagent_max_depth: int = 1

    # Model spec for the cheap fan-out subagent roles (explore, research):
    # the read-only, parallel, high-volume roles where a fast/cheap model
    # is the right tradeoff. Empty string (default) → resolve to the Slot B
    # resident model (engine_secondary_model) when Slot B is enabled and
    # has a model loaded; otherwise inherit the lead's current model. Set
    # explicitly (e.g. "claude-haiku-4-5@anthropic" or "qwen3-8b@local") to
    # override. Always falls back to the lead's model if the chosen model
    # can't resolve at spawn time, so a not-yet-loaded Slot B never blocks
    # a dispatch. Rationale: with every role inheriting the lead's
    # (expensive) model, delegation buys context-window relief but no felt
    # cost win — a likely cause of subagents going unused. Pointing the
    # cheap roles at a free local model makes fan-out an obvious net win.
    coder_subagent_fast_model: str = ""

    # --- Coder hybrid-loop breaker overrides (2026-05-31) ---
    # All defaults are 0 = use the registered ``Breaker.threshold`` in
    # augmentum/loops/breakers.py (which itself honors the
    # AUGMENTUM_CODER_* env vars). Any positive int overrides the
    # registered threshold at runtime — hot-readable via
    # ``live_threshold(name)`` so a POST /api/config/setting update
    # takes effect on the next breaker check without a server
    # restart.
    #
    # Use these to dial hybrid's resilience up/down per workspace
    # without recompiling. Native strategy bypasses most of these,
    # so tuning here mainly affects the hybrid + canonical paths.
    coder_breaker_validation_error_streak: int = 0          # default 5
    coder_breaker_same_validation_error_repeat: int = 0     # default 2
    coder_breaker_action_stagnation_break: int = 0          # default 20
    coder_breaker_test_failure_streak: int = 0              # default 8
    coder_breaker_same_file_edit_break: int = 0             # default 15
    coder_breaker_no_write_progress_break: int = 0          # default 10
    coder_breaker_inspection_loop_nudge: int = 0            # default 5
    coder_breaker_inspection_loop_break: int = 0            # default 3
    coder_hybrid_max_iters: int = 0                         # default 150 (env: AUGMENTUM_CODER_MAX_ITERS)
    coder_hybrid_max_iters_ungated: int = 0                 # default 500 (env: AUGMENTUM_CODER_MAX_ITERS_UNGATED)
    coder_native_nudge_max: int = 0                         # default 2; 0 = use registered default. Max consecutive prose-no-tools nudges in native before accepting the stop. 1 = pre-2026-05-31 behavior; raise if chatty models bail before acting.
    coder_next_speaker_check_enabled: bool = True           # When the native loop's heuristic gate would accept a SUBSTANTIVE_ACTIVE stop with zero writes + no recent progress, run a second LLM call (Qwen-Code-style next-speaker classifier) to second-guess. Off = pure heuristic.
    coder_goal_judge_enabled: bool = True                   # MiMo-style stop-condition judge (coder/goal_judge.py): when the TQG accepts a stop on a turn that made writes, an independent judge call checks the user's request was actually satisfied; not-satisfied injects the reason and re-enters (cap 2). Closes the optimistic-stop failure mode.
    coder_verify_enabled: bool = True                       # Independent cross-model verification of COMPLETED background coder runs (coder/run_verifier.py). When a heavyweight_model is pinned, a DIFFERENT model reviews the actual diff against the original ask before the user is notified, and the honest tiered verdict (failed/human_required/probable/unchecked) lands in the brief. No heavyweight pinned = no verification (verdict stays "unchecked"; never self-verifies). Toggle in Model Manager next to the Heavyweight pin. Default on; only active when a heavyweight is set.
    coder_think_tool_enabled: bool = False                  # Elective reasoning: expose a no-op `think` tool the native loop can call to plan on demand (Anthropic think-tool pattern). Exposed ONLY on turns where native per-turn thinking is OFF (else redundant). Default off — behind this flag for A/B vs the binary thinking toggle.
    coder_compact_tool_enabled: bool = False                # Model-initiated compaction: expose a `compact` tool in the native loop that folds older history at semantic seams the MODEL judges (phase closure, resolved dead-ends), with a self-written four-field handoff note as the synthesis segment. Also renders a "context N% of budget" meter in the sticky reminder. Automatic threshold compaction stays on as the backstop. Default off.
    coder_subagent_verify_enabled: bool = True              # Subagent return-path verification (agents/verify.py): when a subagent stops cleanly AND the lead handed down success_criteria, an independent judge checks the output satisfies each criterion before the stop is honored. A failed verdict re-enters (coder_subagent_verify_reentry) with unmet criteria injected; on exhaustion the result is marked verification="failed" so the lead doesn't trust a confidently-wrong report. Fails open on judge error. The leaf-node twin of coder_goal_judge_enabled.
    coder_subagent_verify_reentry: int = 1                  # How many times a failed subagent verification re-enters its loop before the stop is honored unconditionally (0-3). Leaf default 1 (vs goal-judge's 2): cheaper for the lead to re-dispatch than for a subagent to thrash.
    coder_verify_command_gate_enabled: bool = True          # Held-out verification gate (Arbor principle #2; coder/verify_command.py): on an accepted write-stop, run the project's verification command IN THE CONTAINER before the LLM goal judge. A real non-zero exit is authoritative ground truth — the agent can't argue past it — so the stop is rejected and the loop re-enters with the actual failure output (cap 2). Default OFF; fails open (skip) when no command is detected or the run can't complete. Unlike goal_judge (which reasons over the agent's own report), this admits on a mechanical signal the model doesn't control.
    coder_verify_command: str = ""                          # Explicit command for the held-out verification gate (coder_verify_command_gate_enabled). Empty = auto-detect from project markers (pytest/npm/cargo/go/make). Set to pin a fast subset, e.g. "python -m pytest -x tests/unit".
    coder_compaction_auto_enabled: bool = True              # Auto-compact conversation history mid-turn when prompt tokens cross coder_compaction_threshold * context-limit. Without this, long runs degrade or crash at the window edge.
    coder_compaction_threshold: float = 0.65                # Fraction of the model's usable token budget at which auto-compaction fires (0.3-0.95). 0.65 = compact at 65% utilisation; keeps headroom for the model's next response + tool roundtrip.
    coder_compaction_keep_recent: int = 0                   # Recent messages kept verbatim past the compaction summary (0 = use module default _COMPACT_KEEP_RECENT, currently 12; range 4-40 when set). Bump via the module constant so the test suite's monkeypatch path works.
    coder_compaction_synthesis_enabled: bool = True         # LLM-written handoff note (State/Decisions/Learnings/Next) in each new compacted segment via one bounded second-model call (coder/compaction_synthesis.py, goal_judge plumbing: no KV affinity, think off). Fails open to the mechanical segment — never blocks compaction.
    coder_archive_enabled: bool = True                      # Master kill switch for the durable per-turn archive (coder_turn_archive table). Phase 1 (write-only) is safe to leave on; recall surface lands in Phase 2.
    coder_archive_max_turns_per_workspace: int = 0          # Row cap per workspace (default 10000 when 0). Oldest rows pruned by event_time when the cap is hit so unbounded growth never becomes a Codex-style multi-GB session problem.
    coder_auto_recall_enabled: bool = True                  # Phase 2 recall surface: auto-inject the top semantically-relevant PAST turns from the durable archive (coder_turn_archive) into each turn's context instead of waiting for the model to call the recall tool. Wakes the deep archive — every turn starts knowing what earlier turns learned. Bounded, deduped against the in-prompt <prior_turns> ring, HISTORICAL-framed. Off = recall stays opt-in (tool-only). Requires coder_archive_enabled.
    coder_auto_recall_k: int = 3                            # Max past turns auto-recalled per turn (1-10). Higher surfaces more candidates but adds prompt noise + tokens.
    coder_auto_recall_max_distance: float = 0.0             # L2-distance ceiling for an auto-recall hit (0 = no filter; take raw top-k). Lower = stricter relevance. nomic L2 distances are unbounded, so start permissive and tighten after observing real distances in the coder.auto_recall logs.
    coder_workspace_pids_limit: int = 1024                  # Per-workspace container PID cap (cgroup pids.max). 256 (default before 2026-05-31) saturates on dev-server workloads; the runc shim then refuses new exec with "procReady not received". 1024 covers a vite/esbuild dev server (~10 PIDs) plus generous headroom for short-lived shells. Raise via 2x rule of thumb if you see saturation logs.
    coder_workspace_pids_warn_pct: float = 0.80             # When live PID count exceeds this fraction of the limit, the watchdog emits a warning log (coder.workspace_pids_pressure). 0 disables. Default 0.80 catches the wedge ~200 PIDs before runc breaks.
    coder_workspace_pids_check_interval_s: int = 120        # Watchdog tick interval (seconds). Floor 30s in code. 0 effectively disables (interval normalized up to 30). 120s is cheap (one docker inspect per workspace) and catches saturation before it wedges.
    coder_workspace_swap_ratio: float = 0.5                 # Swap headroom as a fraction of the workspace memory limit. MemorySwap = Memory * (1 + ratio), so 0.5 gives a 2GB workspace 1GB of swap. Softens transient spikes (heavy apt/pip/npm/playwright provisioning, big test runs) so a brief overshoot swaps instead of OOM-killing the container mid-turn. 0 restores the original no-swap behavior (MemorySwap == Memory). Kernels without swap accounting ignore it (Docker warns, doesn't fail).
    coder_max_paused_seconds: int = 1800                    # Auto-cancel a paused run after this many seconds with no resume. 0 disables (a forgotten pause then holds resources indefinitely). 1800 = 30 min covers typical "user got distracted" cases; longer pauses are intentional and the operator should raise this.
    coder_paused_sweep_interval_s: int = 60                 # How often the paused-timeout sweep runs. Floor 15s in code. 60s gives a 30-min timeout sub-minute granularity without burning CPU. 0 disables (sweep loop doesn't start).
    # Workspace network hardening. The container still runs on the
    # default bridge so apt/pip/npm/git clone keep working against
    # public mirrors. ExtraHosts maps host.docker.internal +
    # gateway.docker.internal to 0.0.0.0 inside the container,
    # neutralising the most dangerous pivot (model exfiltrating to
    # the Augmentum proxy via the host alias). LAN egress (e.g. the
    # user's NAS at 192.168.1.50) stays reachable — restricting that
    # requires host-level firewall rules outside Augmentum's reach.
    # Operators who run host services that a workspace legitimately
    # needs (Plex, Jellyfin, internal API) can flip this off.
    coder_workspace_block_host_pivot: bool = True           # Map host.docker.internal / gateway.docker.internal to 0.0.0.0 inside the container so the model can't curl back to the proxy. Default True; flip off if workspaces legitimately need to call host services.
    coder_workspace_network_mode: str = "bridge"            # Docker NetworkMode for workspace containers: "bridge" (default — internet for apt/pip/npm/git) or "none" (fully airgapped — for untrusted code / no-egress policy). Invalid values fall back to "bridge" in containers.py. Applies to NEW containers; existing workspaces keep their mode until recreated.

    # --- MCP ---
    # ON by default (flipped 2026-06-02). When enabled, /mcp exposes
    # user-scoped tools (memory), resources (character cards), prompts
    # (presets + reasoning flows), and install-wide tools (web_search,
    # python_executor, …) to any authenticated MCP client (Claude
    # Desktop, Cursor, Cline, etc.). Auth is the same ``sk-aug-*``
    # bearer token used everywhere else — see auth/middleware.py.
    # Multi-tenant: each MCP client request resolves the calling user
    # via the ASGI scope and tools refuse cross-tenant access.
    # Existing installs that explicitly saved ``False`` keep their
    # preference (settings_store wins over the Python default).
    mcp_enabled: bool = True
    mcp_servers: str = ""  # JSON array of server configs, e.g. [{"name":"x","command":"y","args":["z"]}]

    # --- Community Install ---
    # Master kill switch for the "Open in Augmentum" deep link from
    # augmentumhq.com. When False, both /community-install and
    # /api/community/install refuse to serve. See
    # augmentum/proxy/community_routes.py and the spec at
    # augmentumhq-site/docs/specs/community-install.md.
    community_install_enabled: bool = True
    # Optional admin-managed allowlist of trusted community-content
    # origins, beyond the built-in AugmentumHQ raw.githubusercontent.com
    # prefixes hard-coded in community_routes.py. Stored as a JSON array
    # in settings; not exposed through config_routes in v0 (admin must
    # edit env / DB directly to extend). v1 will add a proper admin UI.
    community_trusted_origins: list[str] = []
    # Knowledge pack size sanity check. Knowledge pack install isn't
    # wired in v0; lives here so the cap is documented from the start.
    community_max_pack_size_mb: int = 500

    # --- Prompt Cache ---
    prompt_cache_enabled: bool = True
    prompt_cache_max_entries: int = 100
    prompt_cache_ttl: int = 3600
    prefix_cache_max_entries: int = 500     # max entries in prefix dedup cache (LRU eviction)

    # --- Image Generation ---
    image_enabled: bool = False
    avatar_enabled: bool = False
    image_model_dir: str = ""
    image_output_dir: str = ""
    image_default_model: str = ""
    image_default_steps: int = 20
    image_default_cfg: float = 7.0
    image_default_width: int = 512
    image_default_height: int = 512
    image_device: str = "auto"           # auto/cuda/cpu
    image_precision: str = "auto"        # auto/fp16/fp32/bf16
    image_civitai_api_key: str | None = None
    image_huggingface_token: str | None = None
    image_max_queue_size: int = 10
    # Per-user fairness cap on in-flight image jobs (QUEUED + RUNNING) — keeps
    # one caller from monopolising the shared GPU queue on a multi-tenant box.
    # Default-on but generous (queue holds 10); a solo user effectively never
    # hits it. 0 disables. Enforced in image/queue.py::submit.
    image_max_inflight_per_user: int = 6
    image_vram_limit: int | None = None
    image_warmup: bool = True
    image_cpu_offload: str = "auto"      # auto/always/never
    image_gguf_cuda_kernels: str = "auto"  # auto/on/off — CUDA dequant kernels for GGUF (~10% speedup, needs compute cap >= 7.0)
    image_default_negative_prompt: str = "ugly, blurry, low quality, deformed, disfigured, extra limbs, bad anatomy, bad hands, missing fingers, cropped, watermark, text"
    image_default_preset: str = ""
    image_generation_timeout: float = 900.0  # Per-job timeout in seconds (covers model load + generation)
    image_prompt_condense: bool = True   # Auto-condense prompts that exceed model token limit
    image_prompt_condense_model: str = ""  # Model for condensation (empty = default backend model)

    # --- Image: Quality & Speed Optimizations ---
    image_freeu_enabled: bool = True        # FreeU: rebalance UNet skip connections (SD1.5/SDXL only, no speed cost)
    image_torch_compile: str = "auto"       # "auto" (Ampere+ with gcc), "on", "off". ~30% faster after first gen.
    image_tome_enabled: bool = False        # Token Merging: merge similar tokens for 20-40% speedup (SD1.5/SDXL only)
    image_tome_ratio: float = 0.5           # ToMe merge ratio (0.1-0.9; higher = faster but lower quality)
    image_cfg_rescale: float = 0.0          # CFG rescale to prevent overexposure at high CFG (0.0-1.0; 0 = off)
    image_hires_fix: bool = False           # Two-pass generation: generate → upscale → img2img refine
    image_hires_scale: float = 1.5          # Hires fix upscale factor (1.5 or 2.0)
    image_hires_denoise: float = 0.5        # Hires fix img2img denoise strength (0.0-1.0)
    image_ip_adapter_enabled: bool = True   # IP-Adapter: inject image reference into latent space
    image_ip_adapter_scale: float = 0.55    # IP-Adapter strength (0.0-1.0; 0.55 = recommended default)

    # --- Image: Custom Model Import ---
    image_allow_pickle_formats: bool = False     # Opt-in: accept .bin/.pt/.pth/.ckpt uploads (pickle deserialisation risk)
    image_upload_max_size_gb: int = 20           # Reject custom-import uploads larger than this (GB)
    image_imports_dir: str = ""                  # Allowlisted server-side path prefix for offline imports (empty = path imports disabled)

    # --- Session Isolation ---
    session_client_isolation: bool = False  # Scope sessions by client identity (IP/header)

    # --- Metrics ---
    metrics_enabled: bool = True

    # Dream system (server-side operational defaults)
    dream_model: str = ""
    dream_max_context_tokens: int = 4096
    dream_portrait_model: str = ""
    dream_recall_enabled: bool = True
    dream_recall_limit: int = 3
    # Minimum cosine similarity (0=unrelated, 1=identical) for a journal
    # entry to be injected via semantic recall. Slightly more permissive
    # than memory's 0.5 default because dream entries are longer/more
    # abstract — relevant matches often score lower than fact lookups.
    # Below this threshold, the recall block is omitted entirely so the
    # model isn't primed toward unrelated topics. Per-user override:
    # ui.dreamRecallMinSimilarity.
    dream_recall_min_similarity: float = 0.4

    # ── Dream compaction (admin-global, not per-user) ──
    # Background process that consolidates near-duplicate journal entries
    # so semantic recall and portrait synthesis don't over-weight repeated
    # topics. Mirrors MemoryCompactor's shape — see DreamCompactor docstring
    # for the design rationale. Toggle covers BOTH the periodic background
    # loop AND the on-write consolidation in journal.store_entry; if a deploy
    # needs one without the other, file an issue rather than splitting the
    # toggle (more knobs = worse OSS UX for marginal benefit).
    dream_compaction_enabled: bool = True
    dream_compaction_interval_hours: float = 12
    # Pair-merge threshold: two distinct entries that say roughly the same
    # thing get merged into one via LLM. 0.85 = "near duplicate" — high
    # enough to avoid losing nuance from genuinely-distinct facets of the
    # same topic (e.g., two separate facets of the same subject should
    # NOT collapse).
    dream_dedup_threshold: float = 0.85
    # Cluster-summarize threshold: groups of N+ thematically-similar entries
    # get replaced by a single consolidated summary. Lower bar than dedup
    # because it's about THEME density, not duplicate text — 8 entries each
    # touching the cat from a different angle are a pattern even if no two
    # are near-duplicates of each other.
    dream_cluster_threshold: float = 0.65
    dream_cluster_min_size: int = 3
    # Per-pass cap so an install with hundreds of clusters doesn't burn the
    # model on a single compaction run; the next pass picks up where this
    # one left off.
    dream_compaction_max_clusters_per_run: int = 5
    # On-write consolidation range — when a new entry would land within
    # this similarity to an existing one, merge instead of inserting new.
    # Tighter range than memory's 0.60-0.78 because dream content is more
    # prose-like and lower scores can still be genuinely distinct.
    dream_consolidation_low: float = 0.65
    dream_consolidation_high: float = 0.85
    # Time-trim only fires above this entry count. Below it, semantic
    # compaction handles deduplication and nothing gets pruned by age.
    # Avoids the failure mode where a user with 60 well-curated entries
    # loses content just because it's old enough to age out.
    dream_time_trim_count_threshold: int = 200
    # Age cutoff once the count threshold is exceeded. Existing default
    # preserved so installs that didn't customize this still trim at 30 days.
    dream_compaction_max_age_days: int = 30
    # Optional model override for compaction LLM calls. Empty = use the
    # "utility" role chain (same chain MemoryCompactor uses), so deployments
    # don't need to configure it separately.
    dream_compaction_model: str = ""

    # Knowledge packs
    knowledge_packs_enabled: bool = True
    knowledge_packs_dir: str = ""  # Empty = {data_dir}/knowledge
    knowledge_max_results: int = 5  # Fallback result cap when a per-mode override is unset
    knowledge_min_score: float = 0.3  # Legacy — kept for back-compat; hybrid path uses RRF instead
    knowledge_registry_url: str = ""  # URL to fetch available packs registry JSON
    # NOTE: ``knowledge_zim_embed_threshold`` was removed 2026-05-07 along
    # with the auto-embed-on-install code path. Embedding is now opt-in
    # per pack via the Browse panel's "Embed for vector search" icon
    # (POST /api/knowledge/packs/{pack_id}/embed). Persisted overrides of
    # the old key in settings_store are harmless — nothing reads them.
    knowledge_embedding_use_gpu: bool = True  # auto-detect GPU for embedding
    knowledge_embedding_batch_size: int = 512  # chunks per embedding batch
    knowledge_catalog_cache_ttl: int = 86400  # catalog cache TTL in seconds (24h)
    knowledge_packs_custom_dir: str = ""  # custom storage location (empty = default)
    knowledge_featured_packs: str = ""  # override featured pack IDs (comma-separated)

    # Per-mode pack injection. Knowledge packs are encyclopedic reference
    # corpora (Wikipedia, Python docs, medwiki) — orthogonal to the memory
    # subsystem. Narrative defaults OFF because worldbuilding lives in the
    # lorebook system; chat-style modes default ON.
    knowledge_packs_passthrough: bool = True
    knowledge_packs_analytical: bool = True
    knowledge_packs_agentic: bool = True
    knowledge_packs_narrative: bool = False

    # Per-mode result caps for pack injection. Empty/0 falls back to
    # ``knowledge_max_results`` above.
    knowledge_max_results_passthrough: int = 3  # short answers, render inline
    knowledge_max_results_analytical: int = 5
    knowledge_max_results_agentic: int = 5
    knowledge_max_results_narrative: int = 5

    # Query condensing — rewrites the latest user message into a self-
    # contained search query using prior chat turns. Fires only when a pack
    # is bound to the session AND the message looks anaphoric/short.
    # Empty model falls through to utility_model → primary_chat_model via
    # ProviderRegistry.resolve_model_for_role(role="utility", override=...).
    knowledge_query_condense_enabled: bool = True
    knowledge_query_condense_model: str = ""

    # Pack search latency tier 1: result LRU cache, model pre-warm, and
    # per-pack passage cache. Defaults are safe for low-end hardware
    # (single-user laptop, no GPU, slow disk). Disable individually if
    # you're seeing memory or disk pressure.
    #
    # Result cache: dedupes (query, pack_set, limit, rerank) lookups for a
    # short TTL. Most user behavior (debounce, re-render, page reload)
    # collapses to a near-instant cache hit. Memory bounded to ~size *
    # 5KB (256 * 5KB ≈ 1.3MB by default). TTL is short so adding a new
    # pack still surfaces in fresh searches within minutes.
    knowledge_search_cache_enabled: bool = True
    knowledge_search_cache_size: int = 256
    knowledge_search_cache_ttl_seconds: int = 600  # 10 min

    # Pre-warm of embedding + reranker models is governed by the
    # existing ``startup_warmup`` setting above (line ~465). Both are
    # what pack search needs; no separate knob to keep configuration
    # surface area small.

    # Per-pack passage cache: ZIM articles get HTML-stripped + section-
    # split into ~900-char passages on every search hit. The first time
    # is a 100-200ms regex pass per article; subsequent visits should be
    # instant. Cache is a SQLite sidecar at {pack}.passages.cache.sqlite.
    # Bounded to N articles per pack (LRU evicted by cached_at). At
    # ~10KB per cached article, default 5000 ≈ 50MB max per pack.
    knowledge_passage_cache_enabled: bool = True
    knowledge_passage_cache_max_articles: int = 5000

    # Soundscape (The Grove)
    soundscape_favorites: str = ""  # JSON array of favorite stations
    soundscape_last_station: str = ""  # JSON object of last played station

    # Ambient Window (Grove)
    ambient_video: str = ""  # JSON: {videoId, title, channel, isLivestream, thumbnail}
    ambient_volume: int = 50  # 0-100
    ambient_loop_mode: str = "off"  # "off" | "loop" (replay current) | "advance" (cycle favorites)
    ambient_favorites: str = ""  # JSON array of favorite ambient videos

    # Application Builder
    app_builder_improve_pass: bool = True
    app_builder_max_improve_iterations: int = 2
    app_builder_max_fix_iterations: int = 4
    app_builder_auto_preview: bool = True
    app_builder_max_tokens: int = 8192
    # Augment the quickjs verify pass with a real headless-chromium run
    # over CDP. Catches layout / CSS / real-browser-API bugs the DOM
    # mock misses. Requires a chromium binary on PATH (apt install
    # chromium in Dockerfile.gpu). Default off until the image ships
    # with chromium — turning it on when the binary is missing is
    # benign, the pipeline just falls back to quickjs.
    app_builder_use_browser_verify: bool = False
    # Optional explicit path to a chromium/chrome binary. Empty = let
    # find_chromium() auto-discover via PATH and platform install
    # locations. Shared by app_builder_use_browser_verify and the XR
    # Browser Panel (augmentum/xr/browser_panel.py). Set on Windows or
    # macOS hosts where chrome isn't on PATH and isn't in the default
    # install location.
    chromium_binary_path: str = ""
    # For small apps (≤5 files) generate every file in a single LLM
    # call rather than one file at a time. Better cross-file coherence;
    # falls back to sequential generation if the batch response is
    # partial or unparseable. Toolkit spec §25.
    app_builder_batch_small_apps: bool = True
    # Per-LLM-call timeout for the build pipeline. Observed: qwen3.6-35b
    # generating one 500-line UI layer file at ~30 tok/s comfortably exceeds
    # the previous 300s cap. Three attempts × this timeout is the worst-case
    # stall per file, so don't set it too high — if the backend is wedged,
    # the build hangs for 3 × this value before the user sees an error.
    app_builder_llm_timeout_seconds: int = 600

    # --- Augmentum Experience Framework (AXF / titles) ---
    # Master toggle for the unified Title abstraction (games, emulator
    # ROMs, AGSP-streamed games, web bookmarks, future GitHub-cloned
    # projects). When off, the /api/titles/* surface returns 503 and
    # the framework UI hides itself. Existing /api/games/* endpoints are
    # unaffected by this toggle -- they remain available for the legacy
    # js13k browse/pin path during the transition.
    #
    # Default flipped to True once the runtime + sources + saves + the
    # Library "ROMs" tab landed -- the framework is now a primary
    # user-facing feature, no reason to gate it.
    titles_enabled: bool = True
    # Soft cap on per-user title-related blob storage (ROMs, build
    # artifacts, screenshots). Saves are tracked separately (see
    # emulator_save_total_quota_mb in the saves design). 0 = unlimited.
    titles_storage_max_mb: int = 5000
    # Curated marketplace surface (AXF title marketplace, NOT the
    # Docker provider marketplace at /api/marketplace). When off the
    # /api/titles/marketplace surface returns 503; the catalog table
    # is still populated so flipping the toggle on is instant.
    marketplace_enabled: bool = False

    # Unified Discover surface (replaces the buried Settings →
    # Marketplace button). When True, /api/discover/* is mounted and
    # the spaces dock surfaces a Discover pill. Defaults True so new
    # installs see it; existing users can disable via setting if they
    # prefer the legacy paths. Spec: docs/superpowers/specs/
    # 2026-06-10-discover-surface-design.md.
    discover_enabled: bool = True

    # Community catalog feed — Phase 2 of the Discover surface.
    # augmentumhq.com publishes a JSON index listing every approved
    # community contribution; Augmentum fetches it on a schedule and
    # upserts each entry into marketplace_listings under
    # publisher="community:<handle>". The feed URL is overridable so
    # forks / private mirrors can swap in their own curation.
    #
    # Disabled means: no community items show up in Discover. The
    # legacy /community-install?manifest_url=… URL flow keeps working
    # (it doesn't depend on this feed).
    #
    # Default OFF (2026-06-17): the index endpoint isn't live yet —
    # pointing at it returns 404, which (a) loaded nothing useful and
    # (b) logged a fetch-failure warning every refresh cycle. Shipping a
    # default that targets a dead URL is worse than off. When the
    # community index goes live, flip this to True (or operators point
    # the URL at their own mirror and enable it). While disabled, the
    # startup reconciler delists any lingering community:* placeholder
    # rows so Discover doesn't show empty "example" Character/Power cards.
    discover_community_feed_enabled: bool = False
    discover_community_feed_url: str = "https://augmentumhq.com/community/index.json"
    # Refresh cadence in minutes. 360 (6h) is a quiet pull rate that
    # surfaces new community work within a working day without
    # hammering the index host. Per-startup also pulls once as soon
    # as the lifespan task gets scheduled.
    discover_community_feed_refresh_minutes: int = 360

    # Save-to-Library — per-publication and per-user storage caps for
    # coder-built artifacts saved via the preview pane. Both in bytes
    # (config sticks to one unit; the UI formats as MB/GB). The
    # per-publication cap rejects unreasonably large saves up front;
    # the per-user cap is a cumulative budget across all of a user's
    # publications. Both are enforced at save time in
    # PublicationStore.create_or_overwrite.
    library_publication_max_bytes: int = 52428800       # 50 MB
    library_publication_user_budget_bytes: int = 1073741824  # 1 GB

    # Browser-WASM emulator runtime (EmulatorJS). Off means the
    # `emulator-browser` runtime is registered but the upload-rom
    # endpoint and frontend stage gate on this flag.
    emulator_browser_enabled: bool = False
    # Hard ceiling for individual ROM uploads (MB). 0 = use the 2GB
    # built-in default (PSP ISOs sit at the upper end).
    emulator_rom_max_mb: int = 0
    # Per-slot save cap (MB). State saves can reach 5-20MB on N64;
    # screenshots are ~200KB. 50 leaves headroom.
    emulator_save_max_per_slot_mb: int = 50
    # Slot count. 0 = quicksave/SRAM, 1..N manual states. Default 8
    # gives 1 quick + 7 named slots.
    emulator_save_slots_per_rom: int = 8

    # --- Controller framework (AXF) ---
    # Master toggle for the per-user remap surface. Reads stay open
    # so the UI can show defaults; writes return 503 when off.
    controller_remap_enabled: bool = True
    # Translate game rumble → Gamepad API vibration when the host
    # exposes ``vibrationActuator``. Browsers without it ignore.
    controller_haptic_enabled: bool = True
    # Show the on-screen virtual gamepad. ``auto`` shows when no
    # physical gamepad is connected on a touch-capable device.
    controller_touch_overlay: str = "auto"
    # Pad → player slot routing. ``index`` (default) maps pad 0 to P1,
    # pad 1 to P2, etc. ``firstpress`` lets the first user to press
    # Start become P1 -- useful for couch multiplayer where pads
    # plug in unpredictably.
    controller_pad_routing: str = "index"
    # Analog stick deadzone, 0.0-0.5. Below this magnitude the axis
    # reads as zero. 0.15 is a sensible default for retro games.
    controller_deadzone: float = 0.15

    # --- Game Streaming Platform (AGSP) ---
    # Master toggle for browser-streamed native games (Luanti etc.). When
    # off, the routes still register but every endpoint returns 503.
    game_stream_enabled: bool = False
    # Per-user concurrent stream cap. Each active stream consumes a
    # container slot and a port pair, so this also bounds the port pool.
    game_stream_max_concurrent: int = 2
    # Default bitrate when the client doesn't request one. Profiles can
    # override via their own default_bitrate_mbps.
    game_stream_default_bitrate_mbps: int = 4
    # How long a session stays warm after the last client disconnects
    # before the lifecycle reaper stops the container. Active browser
    # stages send a heartbeat, so this is a reconnect grace window, not
    # a play-session limit. 3600 = 1 hour.
    game_stream_idle_timeout_seconds: int = 3600
    # Prefer hardware encoders (NVENC, VAAPI) when the host supports them.
    # Falls back to software x264 automatically when unavailable.
    game_stream_prefer_hw_encoder: bool = True
    # Credit-budget admission (see _admit() + the design doc at
    # docs/superpowers/specs/2026-06-02-game-stream-admission-control.md).
    # Each GameProfile declares a cost_credits; admission checks two
    # budgets per start. Active = CPU/encoder ceiling; resident = RAM
    # ceiling. Defaults sized for a single-host install: 8 active means
    # ~3 heavy emulators (Dolphin@2 each) OR ~8 Luanti instances.
    game_stream_active_credit_budget: int = 8
    game_stream_resident_credit_budget: int = 16
    # Per-user hard cap on active credits, ignored only if higher than
    # the host budget. Keeps a solo user from spiking past a sensible
    # ceiling even if no one else is online.
    game_stream_user_hard_cap: int = 4
    # How long a PAUSED session (docker pause / cgroup-frozen) stays
    # parked before the sweep loop stops it cleanly. 30 min is a
    # generous "stepped away to make dinner" window.
    game_stream_paused_stop_seconds: int = 1800
    # Streamed Luanti mouse default — matches desktop Luanti. The earlier
    # 0.04 override compensated for amplification at two layers (Xvfb
    # pointer accel + Selkies cursorScaleFactor) that are now neutralised
    # at the source. Users can still tune in-game.
    game_stream_mouse_sensitivity: float = 0.2

    # Base WebSocket URL the in-container agent-bridge.py daemon dials
    # to reach augmentum's game-agent bridge endpoint. Format:
    # ``ws://host:port`` (or ``wss://``); the route layer appends
    # ``/api/game-agent/surfaces/emulator/bridge/<session_id>?token=<x>``.
    # Empty (default) disables AI-driven streamed-emulator sessions --
    # the route returns 503 for companion=true requests, so a
    # misconfigured deployment fails loudly instead of starting a
    # container the daemon can't dial out from.
    # Docker Desktop / Linux with extra_hosts: ``ws://host.docker.internal:8080``
    # Same-compose-network deployments: ``ws://augmentum:8080``
    agent_bridge_base_url: str = ""

    # Game-agent perception (augmentum/game_agent/perception.py). Both
    # default ON: they only improve vision-mode play and degrade safely.
    #   _frame_dedup: collapse near-identical frames in the temporal
    #     window so the model isn't fed (and misled by) duplicate ticks
    #     on static menus/dialog — the dominant case for turn-based games.
    #   _grid_overlay: draw a labeled Set-of-Marks grid on each frame so
    #     the VLM can reference on-screen positions by cell (the strongest
    #     vision scaffold short of RAM state). Becomes tile-aligned +
    #     walkability-colored once the surface gains RAM access.
    game_agent_frame_dedup_enabled: bool = True
    game_agent_grid_overlay_enabled: bool = True

    # Longest-edge cap applied to agent frames before dedup/grid. The
    # emulator surface captures the *display* canvas (a pure nearest-
    # neighbor upscale of the native framebuffer), so shipping it full
    # size wastes encode/transport/preprocess time for zero information.
    # 480 = 2x GBA native; keeps dialog text + grid labels legible.
    # 0 disables the cap.
    game_agent_frame_max_edge: int = 480

    # Fast-turn ("call mode") loop. Instead of a full re-prompted plan
    # every turn, the agent holds a rolling multi-turn chat window —
    # static system prompt (objective/caps/journal), then per-turn
    # user(frame + state delta) / assistant(micro-plan) exchanges. The
    # append-only shape keeps llama-server's KV prefix cache hot, and
    # the micro-plan output (~30 tokens, no thinking) decodes in a
    # fraction of a full plan. A FULL plan turn (reasoning + journal
    # update + fresh window) still runs every _full_turn_every fast
    # turns, or sooner when the model asks to escalate / a fast turn
    # fails to parse. _fast_turns_enabled=False (or _full_turn_every=1)
    # restores the pre-call-mode behavior exactly.
    game_agent_fast_turns_enabled: bool = True
    game_agent_full_turn_every: int = 8
    game_agent_fast_max_tokens: int = 192

    # Scene narrator ("the eyes"): a parallel no-thinking vision lane
    # that keeps a live-feed description of the screen fresh for both
    # the fast actor (SCENE= in its delta) and the planner (scene_feed
    # in OVERLAY). Fingerprint-gated, so static screens cost nothing.
    game_agent_scene_narrator_enabled: bool = True
    game_agent_scene_interval_ms: int = 1500

    # Stall watchdog: when NOTHING in the world (probes, scene) has
    # changed for this many seconds, the fast delta carries a STALLED
    # marker telling the actor its approach is not working.
    game_agent_stall_after_s: int = 45

    # Which model runs the FULL (thinking/planning) turns. Empty = the
    # same model as everything else (registry default). Set to any
    # registry model name to give the agent a two-model brain: small
    # resident vision model on the fast lane, a larger planner here
    # (Slot A/B chat model, cloud, …). Read live per planning call, so
    # changing it takes effect on the next full turn without a restart.
    # USER-CHOSEN, never auto-selected.
    game_agent_planner_model: str = ""

    # Which model runs the FAST turns + scene narrator (must be vision-
    # capable). Empty = registry default. Pin this on shared boxes so
    # another tenant flipping the default chat model can't blind the
    # agent mid-run. USER-CHOSEN, never auto-selected.
    game_agent_fast_model: str = ""

    # Game-agent planning budget. Small local vision models (the Gemma-4
    # E2B classifier) plan better with reasoning ON — it breaks the
    # one-button fixation — but the budget must be large enough that the
    # chain-of-thought AND the strict-JSON plan both fit, or the reasoning
    # truncates and the answer comes back empty. Default ON + a generous
    # 4096-token budget (no meaningful truncation for a single game turn).
    game_agent_thinking_enabled: bool = True
    game_agent_max_tokens: int = 4096

    # Where the game agent's persistent per-(user, title) journals live —
    # its long-horizon memory (status / progress / objectives / learned
    # notes) across sessions. Empty = auto: /data/game-agent-journals when
    # /data exists (Docker; survives container recreation), else the
    # ephemeral game-agent log dir. Set explicitly to relocate.
    game_agent_journal_dir: str = ""

    # Phone-as-controller (cast-input bridge) base URL. Same shape as
    # agent_bridge_base_url — the in-container cast-input-bridge.py
    # daemon dials this with the per-session token appended. Empty
    # disables phone-as-controller entirely (start_session won't mint
    # a token, the container won't get AUGMENTUM_CAST_INPUT_BRIDGE_URL,
    # the entrypoint won't spawn cast-input-bridge.py). When unset,
    # the route layer falls back to ``agent_bridge_base_url`` — both
    # daemons live on the same container, so one URL covers both.
    cast_input_bridge_base_url: str = ""

    # --- Game Portal ---
    # Master toggle for the Games tab in Library. Off hides the tab and
    # skips all browse fetches.
    game_portal_enabled: bool = True
    # Proactive recommendation level: "off" (never suggest), "contextual"
    # (after long sessions, frequency-capped), "always" (weave into chat).
    # Phase 0 ships dormant; reserved for later phases.
    game_portal_recommendations: str = "off"
    # Which external catalogs the browse tab pulls from, comma-separated.
    # Known source ids: "js13k", "jams". (itch.io was removed -- embed
    # detection sent every game to a security interstitial.)
    game_portal_default_sources: str = "js13k,jams"

    def to_safe_dict(self) -> dict:
        """Return config as dict with sensitive values redacted."""
        data = {}
        for key in self.model_fields:
            val = getattr(self, key)
            if "key" in key.lower() or "secret" in key.lower() or "token" in key.lower():
                data[key] = "***" if val else None
            else:
                data[key] = val
        return data


settings = Settings()


def selfedit_unlocked() -> bool:
    """Whether the self-edit subsystem is available on this install at all.

    Deliberately an ENVIRONMENT variable rather than a stored setting: this is
    an operator decision made once at deploy time, not a preference a signed-in
    user can toggle. Unset (the default, and so the default for every release
    build) means the Workshop does not exist as far as the UI and the API are
    concerned — the nav entry is absent, the settings row is absent, and all
    ``/api/selfedit/*`` routes refuse.

    The reason it's locked rather than merely off-by-default: self-edit can
    produce changes that pass the fitness gate (compile, lint, tests, health)
    and still alter behavior in ways that only surface later — generated intent
    verbs are the known case. "Didn't break the build" is not "is correct", so
    a curious user enabling it from a settings screen is not informed consent.
    Whoever sets this variable is accepting that they may have to recover the
    repo by hand.

    Set ``AUGMENTUM_SELFEDIT_UNLOCK=1`` to make it available. Even then it stays
    OFF until an admin enables ``selfedit_enabled`` (see that field above), so
    unlocking reveals the switch, it does not throw it.
    """
    return os.environ.get("AUGMENTUM_SELFEDIT_UNLOCK", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
