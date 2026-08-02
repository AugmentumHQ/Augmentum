"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Receive, Scope, Send

from augmentum.classifier.router import RequestClassifier
from augmentum.config import settings
from augmentum.models.provider_registry import ProviderRegistry
from augmentum.state.backends.sqlite import SQLiteBackend
from augmentum.state.manager import StateManager
from augmentum.utils.logging import get_logger
from augmentum.utils.model_load import load_model_off_loop

if TYPE_CHECKING:
    from augmentum.tools.registry import ToolRegistry

log = get_logger(__name__)


# --- UI shell digest (native-app asset bundling, see C/ /api/ui-version) -----
#
# The Android shell can bundle the SPA's static shell inside the APK and serve
# it off local disk (zero network RTT) ONLY when its bundle byte-matches what
# this server would serve. This digest is the handshake: the app compares its
# baked SHELL_DIGEST against this and enables local serving on an exact match,
# else falls back to the network transparently. Any divergence → no match →
# network fallback, so the worst case is "no speedup", never "stale assets".
#
# The include set + excludes MUST mirror the Android Gradle ``syncWebAssets``
# task. If they ever drift, the digests simply stop matching and the app
# network-loads — fail-safe, not fail-dangerous.
_UI_SHELL_EXTS = frozenset(
    {".html", ".htm", ".js", ".mjs", ".css", ".json", ".woff2", ".woff", ".svg", ".ico", ".png", ".webp", ".gif"}
)
# Top-level dirs to skip entirely. ``mockups`` alone is 4k+ files; ``lib`` is
# 500 MB of on-demand binaries (three.js/emulator cores) — its handful of .js
# helpers network-load fine and don't belong in the launch-critical shell.
# PRUNING these during the walk (vs filtering after) is what keeps the digest
# cheap: we never stat the thousands of files we'd only discard.
_UI_SHELL_EXCLUDE_TOP = frozenset({"mockups", "lib"})
# Re-check the on-disk signature at most this often. Within the window the
# endpoint answers from cache with zero filesystem work — important because the
# app polls it at launch and a Docker bind-mount stat-walk is not free.
_UI_SHELL_TTL_S = 60.0
# key -> (checked_at_monotonic, signature, digest)
_ui_shell_digest_cache: dict[str, tuple[float, tuple[int, int, int], str]] = {}


def _ui_shell_files(ui_dir):
    """Yield (posix_relpath, abs_path) for every file in the bundled shell set.

    Uses os.walk with in-place dir pruning so excluded top-level trees
    (mockups, lib) are never descended into.
    """
    import os
    from pathlib import Path as _Path

    root = _Path(ui_dir)
    root_str = str(root)
    for dirpath, dirnames, filenames in os.walk(root_str):
        rel_dir = os.path.relpath(dirpath, root_str)
        if rel_dir == ".":
            # Prune excluded trees at the root so we don't walk them at all.
            dirnames[:] = [d for d in dirnames if d not in _UI_SHELL_EXCLUDE_TOP]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _UI_SHELL_EXTS or fname.endswith(".map"):
                continue
            abs_path = _Path(dirpath) / fname
            rel = abs_path.relative_to(root).as_posix()
            yield (rel, abs_path)


def _ui_shell_digest(ui_dir) -> str:
    """Content digest over the bundled shell set; TTL- and signature-cached.

    Within ``_UI_SHELL_TTL_S`` of the last check we return the cached digest
    with no filesystem access. After the TTL we do a (pruned) stat-walk for the
    signature (count, max mtime_ns, total size) and only re-read the ~16 MB of
    bytes when that signature actually changed. So the steady state is cheap and
    the expensive hash happens ~once per real UI change.
    """
    import hashlib
    import time
    from pathlib import Path as _Path

    root = _Path(ui_dir)
    if not root.is_dir():
        return ""
    key = str(root)
    now = time.monotonic()
    cached = _ui_shell_digest_cache.get(key)
    if cached and (now - cached[0]) < _UI_SHELL_TTL_S:
        return cached[2]

    files = sorted(_ui_shell_files(root), key=lambda t: t[0])
    count = len(files)
    max_mtime = 0
    total = 0
    for _rel, p in files:
        try:
            st = p.stat()
            max_mtime = max(max_mtime, st.st_mtime_ns)
            total += st.st_size
        except OSError:
            continue
    sig = (count, max_mtime, total)
    if cached and cached[1] == sig:
        # Unchanged on disk — refresh the TTL, skip the content re-read.
        _ui_shell_digest_cache[key] = (now, sig, cached[2])
        return cached[2]

    outer = hashlib.sha256()
    for rel, p in files:
        try:
            data = p.read_bytes()
        except OSError:
            continue
        outer.update(rel.encode("utf-8"))
        outer.update(b"\x00")
        outer.update(hashlib.sha256(data).hexdigest().encode("ascii"))
        outer.update(b"\n")
    digest = outer.hexdigest()
    _ui_shell_digest_cache[key] = (now, sig, digest)
    return digest


def _parse_bool(value: str) -> bool:
    """Parse a string value to bool (for settings restoration)."""
    return value.lower() in ("true", "1", "yes")


def _parse_optional_bool(value: object) -> bool | None:
    """Parse a tri-state setting: None / True / False from various wire shapes.

    Used by ``_SETTINGS_RESTORE_MAP`` for fields like
    ``engine_multislot_enabled`` where the persisted value is genuinely
    optional — absence (no DB row) means "follow the codebase's
    recommended default" (resolved at runtime via a constant) rather
    than the falsy default.

    Accepted shapes (post-DB stringification, also direct calls):
      * ``None`` / ``""`` / ``"none"`` / ``"null"`` / ``"auto"`` → ``None``
      * Truthy strings (``"true"``, ``"1"``, ``"yes"``) → ``True``
      * Anything else → ``False``
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("", "none", "null", "auto"):
            return None
        return normalized in ("true", "1", "yes")
    return bool(value)


def _is_ip_adapter_generation_error(exc: BaseException) -> bool:
    """Return True for diffusers failures caused by IP-Adapter state/inputs."""
    message = str(exc).lower()
    markers = (
        "ip-adapter",
        "ip_adapter",
        "ip_image_proj",
        "image_embeds",
        "encoder_hid_dim_type",
        "shapes cannot be multiplied",
    )
    return any(marker in message for marker in markers)


# SQLite errors classified as transient — worth retrying with backoff
# rather than immediately surfacing as integrity failures. WSL2 / Docker
# Desktop / Windows-hosted volumes drop reads under concurrent I/O load
# (model mmap fault-in, antivirus opening WAL files, container backups);
# all of these clear within a few seconds.
_TRANSIENT_SQLITE_FRAGMENTS = (
    "disk i/o error",
    "database is locked",
    "database is busy",
    "cannot read from database",
)


def _run_quick_check_with_retry(
    db_path: str,
    *,
    backoff_schedule: tuple[float, ...] = (2.0, 8.0),
    connect_timeout_s: float = 30.0,
    sleep: Callable[[float], None] | None = None,
    logger=None,
) -> list[str]:
    """Run ``PRAGMA quick_check`` against ``db_path`` with retry on
    transient SQLite errors.

    Returns the rows of the quick_check result (e.g. ``["ok"]`` for a
    clean database, or human-readable error strings on corruption).

    On transient ``OperationalError`` (disk I/O blip, lock contention),
    sleeps for the next entry in ``backoff_schedule`` and retries. Each
    retry is logged at INFO so the cadence is visible without firing
    the warning rate that integrity-failure dashboards key on. The
    final attempt's exception (if any) is re-raised — the caller
    (``_integrity_check_loop``) downgrades that to a warning.

    Non-transient errors propagate immediately without consuming any
    retry budget — those represent a real database problem.

    ``sleep`` and ``logger`` parameters are dependency-injected so this
    function can be unit-tested without real wall-clock waits or a
    structured-logging side effect.
    """
    import sqlite3 as _sqlite3
    import time as _time

    if sleep is None:
        sleep = _time.sleep
    last_err: Exception | None = None
    total_attempts = len(backoff_schedule) + 1
    for attempt in range(total_attempts):
        try:
            c = _sqlite3.connect(db_path, timeout=connect_timeout_s)
            try:
                rows = c.execute("PRAGMA quick_check").fetchall()
                if attempt > 0 and logger is not None:
                    logger.info(
                        "db_integrity_check_recovered",
                        attempt=attempt,
                        prior_error=str(last_err) if last_err else "",
                    )
                return [r[0] for r in rows] if rows else []
            finally:
                c.close()
        except _sqlite3.OperationalError as e:
            last_err = e
            msg = str(e).lower()
            transient = any(frag in msg for frag in _TRANSIENT_SQLITE_FRAGMENTS)
            if not transient:
                raise
            if attempt >= len(backoff_schedule):
                # All retries exhausted on transient errors — caller
                # surfaces the final warning.
                raise
            pause = backoff_schedule[attempt]
            if logger is not None:
                logger.info(
                    "db_integrity_check_transient_retry",
                    attempt=attempt + 1,
                    backoff_s=pause,
                    error=str(e),
                )
            sleep(pause)
            continue
    # Defensive — loop above always either returns or raises.
    if last_err is not None:
        raise last_err
    return []


# Consolidated map of all persisted config keys and their type cast functions.
# Used by ``_restore_settings`` during startup to reload saved overrides.
_SETTINGS_RESTORE_MAP: dict[str, type | Callable] = {
    # Health / strain monitor (durable strain_samples time series)
    "strain_monitor_enabled": _parse_bool,
    "coder_verify_enabled": _parse_bool,
    # Self-editing master switch + autonomy posture (default OFF / propose)
    "selfedit_enabled": _parse_bool,
    "selfedit_autonomy_level": str,
    "selfedit_engine": str,
    "selfedit_edit_model": str,
    "selfedit_frontier_model": str,
    # selfedit_max_iters is registered in config_routes._TOOL_SETTINGS as an
    # int, so _auto_derive_restore_parsers() restores it with the right type —
    # no manual entry needed (a manual str entry would override the int parser).
    # Intent-router training-data capture (opt-in, default OFF)
    "intent_capture_enabled": _parse_bool,
    # Training trace capture (opt-in, default OFF)
    "training_capture_enabled": _parse_bool,
    # Android assistant-role Slice 2 screen-read ingest (opt-in, default OFF)
    "companion_assist_enabled": _parse_bool,
    # Notifications
    "notifications_enabled": _parse_bool,
    "notification_sound_enabled": _parse_bool,
    "notification_sound": str,
    # Image defaults
    "image_freeu_enabled": _parse_bool,
    "image_torch_compile": str,
    "image_tome_enabled": _parse_bool,
    "image_tome_ratio": float,
    "image_cfg_rescale": float,
    "image_hires_fix": _parse_bool,
    "image_hires_scale": float,
    "image_hires_denoise": float,
    "image_default_model": str,
    "image_default_steps": int,
    "image_default_cfg": float,
    "image_default_width": int,
    "image_default_height": int,
    "image_default_preset": str,
    # Tool / search settings
    "uarf_auto_search": _parse_bool,
    "uarf_auto_search_queries": int,
    "uarf_auto_search_results_per_query": int,
    "uarf_auto_search_max_context_chars": int,
    "uarf_auto_verify": _parse_bool,
    "uarf_proactive_search": _parse_bool,
    "uarf_proactive_math": _parse_bool,
    "uarf_proactive_code": _parse_bool,
    "uarf_max_tool_calls_per_phase": int,
    "uarf_search_retry_max": int,
    "uarf_search_retry_min_results": int,
    "uarf_heuristic_assess": _parse_bool,
    # Narrative scene image settings
    "narrative_scene_image_model": str,
    "narrative_scene_distiller_model": str,
    "narrative_scene_context_rounds": int,
    # Memory / KG settings
    "memory_enabled": _parse_bool,
    "memory_recall_limit": int,
    "memory_recall_min_score": float,
    "memory_summary_max_chars": int,
    "memory_llm_extraction_enabled": _parse_bool,
    "memory_llm_extraction_model": str,
    "memory_core_profile_enabled": _parse_bool,
    "memory_core_profile_max_tokens": int,
    "memory_core_profile_rebuild_interval": int,
    "memory_consolidation_enabled": _parse_bool,
    "memory_compaction_enabled": _parse_bool,
    "memory_compaction_interval_hours": float,
    "memory_compaction_max_age_days": float,
    "memory_extraction_batch_size": int,
    "memory_auto_approve": _parse_bool,
    "memory_scope_by_mode": _parse_bool,
    "memory_inject_analytical": _parse_bool,
    "memory_inject_agentic": _parse_bool,
    "memory_dedup_threshold": float,
    "memory_contradiction_threshold": float,
    # Reranker
    "reranker_enabled": _parse_bool,
    "reranker_model": str,
    "reranker_top_k": int,
    # Document RAG
    "document_rag_enabled": _parse_bool,
    "document_rag_recall_limit": int,
    "document_rag_contextual_retrieval": _parse_bool,
    "document_rag_query_analysis": _parse_bool,
    "document_rag_query_analysis_model": str,
    "document_rag_query_analysis_timeout": float,
    "document_rag_cliff_ratio": float,
    "document_rag_max_context_tokens": int,
    # Application Builder
    "app_builder_improve_pass": _parse_bool,
    "app_builder_max_improve_iterations": int,
    "app_builder_max_fix_iterations": int,
    "app_builder_auto_preview": _parse_bool,
    "app_builder_max_tokens": int,
    "app_builder_llm_timeout_seconds": int,
    # Narrative settings
    # narrative_memory_enabled intentionally excluded — per-session only
    # (stored in SessionMemorySettings, not global config persistence)
    "narrative_llm_extraction": _parse_bool,
    "narrative_extraction_interval": int,
    "narrative_memory_interval": int,
    "narrative_memory_max_words": int,
    "narrative_memory_ledger_ceiling": int,
    "narrative_smart_retrieval": _parse_bool,
    "narrative_smart_retrieval_count": int,
    "narrative_memory_mode": str,
    "narrative_memory_prompt": str,
    "narrative_extraction_model": str,
    "narrative_memory_model": str,
    "narrative_translate_default_language": str,
    "narrative_translate_auto_save": _parse_bool,
    "narrative_auto_background": _parse_bool,
    "narrative_auto_background_interval": int,
    "narrative_auto_bg_distiller_model": str,
    "narrative_auto_bg_image_model": str,
    "game_portal_enabled": _parse_bool,
    "game_portal_recommendations": str,
    "game_portal_default_sources": str,
    # AXF / titles
    "titles_enabled": _parse_bool,
    "titles_storage_max_mb": int,
    "marketplace_enabled": _parse_bool,
    # Save-to-Library caps
    "library_publication_max_bytes": int,
    "library_publication_user_budget_bytes": int,
    "emulator_browser_enabled": _parse_bool,
    "emulator_rom_max_mb": int,
    "emulator_save_max_per_slot_mb": int,
    "emulator_save_slots_per_rom": int,
    # Controllers
    "controller_remap_enabled": _parse_bool,
    "controller_haptic_enabled": _parse_bool,
    "controller_touch_overlay": str,
    "controller_pad_routing": str,
    "controller_deadzone": float,
    # AGSP -- Game Streaming Platform
    "game_stream_enabled": _parse_bool,
    "game_stream_max_concurrent": int,
    "game_stream_default_bitrate_mbps": int,
    "game_stream_idle_timeout_seconds": int,
    "game_stream_prefer_hw_encoder": _parse_bool,
    "game_stream_mouse_sensitivity": float,
    # Search pipeline settings
    "uarf_conversation_max_chars": int,
    "search_expansion_enabled": _parse_bool,
    "search_expansion_max_variants": int,
    "search_expansion_max_total": int,
    "search_credibility_enabled": _parse_bool,
    "search_direct_fetch_enabled": _parse_bool,
    "search_direct_fetch_max_chars": int,
    "search_relevance_filter_enabled": _parse_bool,
    "search_relevance_min_score": float,
    "search_proxies": str,
    "search_proxy_rotation_enabled": _parse_bool,
    "search_proxy_healthcheck_interval_minutes": int,
    "search_proxy_fallback_direct_enabled": _parse_bool,
    # String settings (models)
    "image_prompt_condense_model": str,
    "uarf_verify_model": str,
    # Multi-model fan-out
    "multi_model_enabled": _parse_bool,
    "multi_model_models": str,
    # Chain settings
    "passthrough_chain_enabled": _parse_bool,
    "passthrough_chain_max_steps": int,
    "passthrough_chain_timeout": float,
    "passthrough_chain_synthesis_timeout": float,
    "passthrough_chain_max_parallel": int,
    "passthrough_chain_max_flows": int,
    "passthrough_chain_max_retries": int,
    "passthrough_chain_attention_anchor": _parse_bool,
    "passthrough_chain_error_as_observation": _parse_bool,
    "passthrough_chain_plan_mutation": _parse_bool,
    # Body physics (hybrid SDF + Rapier avatar embodiment)
    "body_physics_enabled": _parse_bool,
    "body_physics_compliance_gain": float,
    "body_physics_rapier_weight": float,
    "body_physics_recover_hz": float,
    "body_physics_audio_reactions_enabled": _parse_bool,
    "body_physics_visual_feedback_enabled": _parse_bool,
    "body_physics_velocity_aware": _parse_bool,
    # Voice settings
    "voice_tts_chunking": str,
    "voice_silence_threshold_ms": int,
    "voice_max_audio_seconds": int,
    "voice_speaker_verify": _parse_bool,
    "voice_speaker_threshold": float,
    "voice_speaker_verify_seconds": float,
    "voice_smart_turn": _parse_bool,
    "voice_smart_turn_threshold": float,
    "voice_smart_turn_max_wait_s": float,
    "voice_smart_turn_min_veto_confidence": float,
    "voice_bargein_min_speech_ms": int,
    "voice_denoise_enabled": _parse_bool,
    "voice_highpass_hz": int,
    "voice_audio_agc": _parse_bool,
    "voice_audio_ns": _parse_bool,
    "voice_audio_agc_target_dbfs": int,
    "voice_audio_ns_level": int,
    # Fabric routing (which TTS/STT provider to dispatch to, including peers)
    "voice_routing_mode": str,
    "voice_routing_pin_provider": str,
    "stt_routing_mode": str,
    "stt_routing_pin_provider": str,
    "tts_emotion_aware": _parse_bool,
    "tts_voice_style": str,
    "voice_tts_lexicon": str,
    # Ghost text (inline autocomplete)
    "ghost_text_enabled": _parse_bool,
    "ghost_text_model": str,
    # Core model roles
    "utility_model": str,
    "classifier_model": str,
    "primary_chat_model": str,        # UI mirror — what "Auto — use Primary" resolves to
    "heavyweight_model": str,         # Frontier-tier slot (verifier/escalation/second-opinion)
    "role_min_param_billions": float, # Soft warn threshold for role fallback
    # Discovery Engine
    "discovery_enabled": _parse_bool,
    "knowledge_library_enabled": _parse_bool,
    "knowledge_library_in_chat": _parse_bool,
    "knowledge_library_retention_days": int,
    "discovery_max_recommendations": int,
    # Agentic settings
    "agentic_max_steps": int,
    "agentic_artifact_theme": str,
    "agentic_image_model": str,
    "agentic_default_autonomy": int,
    # Narrative (additional)
    "narrative_memory_max_tokens": int,
    "narrative_memory_compaction_enabled": _parse_bool,
    "narrative_memory_state_enabled": _parse_bool,
    "narrative_memory_ledger_enabled": _parse_bool,
    "narrative_memory_continuous_archive": _parse_bool,
    "narrative_archive_min_messages": int,
    "narrative_context_limit": int,
    "narrative_context_budget": int,
    "narrative_request_log_limit": int,
    # Tool pipeline
    "tool_result_max_chars": int,
    "tool_execution_timeout": float,
    # Analytical verification thresholds
    "analytical_max_phase_retries": int,
    "analytical_confidence_threshold": float,
    "analytical_max_backtracks": int,
    # String settings
    "timezone": str,
    "location": str,
    "typography_custom_fonts": str,
    "typography_selected": str,
    "typography_text_scale": str,
    "huggingface_token": str,
    # Startup
    "startup_warmup": _parse_bool,
    # Rate limiting
    "rate_limit_enabled": _parse_bool,
    "rate_limit_chat_rpm": int,
    "rate_limit_image_rpm": int,
    "rate_limit_voice_rpm": int,
    # Session isolation
    "session_client_isolation": _parse_bool,
    # Metrics
    "metrics_enabled": _parse_bool,
    # Dream system
    "dream_model": str,
    "dream_max_context_tokens": int,
    "dream_portrait_model": str,
    "dream_recall_enabled": _parse_bool,
    "dream_recall_limit": int,
    # Kokoro TTS
    "tts_kokoro_quality": str,
    "tts_kokoro_hbe": _parse_bool,
    "tts_kokoro_prosody": _parse_bool,
    # Avatar
    "avatar_enabled": _parse_bool,
    # Knowledge Hub
    "knowledge_packs_enabled": _parse_bool,
    "knowledge_max_results": int,
    "knowledge_min_score": float,
    "knowledge_embedding_use_gpu": _parse_bool,
    "knowledge_embedding_batch_size": int,
    # Per-mode pack injection (encyclopedic packs are independent of memory)
    "knowledge_packs_passthrough": _parse_bool,
    "knowledge_packs_analytical": _parse_bool,
    "knowledge_packs_agentic": _parse_bool,
    "knowledge_packs_narrative": _parse_bool,
    "knowledge_max_results_passthrough": int,
    "knowledge_max_results_analytical": int,
    "knowledge_max_results_agentic": int,
    "knowledge_max_results_narrative": int,
    # Query condensing for chat-style follow-ups
    "knowledge_query_condense_enabled": _parse_bool,
    "knowledge_query_condense_model": str,
    "knowledge_catalog_cache_ttl": int,
    "knowledge_packs_custom_dir": str,
    "knowledge_featured_packs": str,
    # Latency tier 1: cache + passage cache (server-internal, but
    # restore-mapped so admin overrides via /api/config/tools survive
    # restart).
    "knowledge_search_cache_enabled": _parse_bool,
    "knowledge_search_cache_size": int,
    "knowledge_search_cache_ttl_seconds": int,
    "knowledge_passage_cache_enabled": _parse_bool,
    "knowledge_passage_cache_max_articles": int,
    "soundscape_favorites": str,
    "soundscape_last_station": str,
    "ambient_video": str,
    "ambient_volume": int,
    "ambient_favorites": str,
    # Auth
    "auth_session_ttl_hours": int,
    "auth_lockout_threshold": int,
    "auth_lockout_minutes": int,
    "auth_ip_lockout_threshold": int,
    "auth_ip_lockout_minutes": int,
    "auth_ws_ticket_ttl_seconds": int,
    "auth_max_sessions_per_user": int,
    # Files / VFS
    "files_webdav_enabled": _parse_bool,
    "files_enrichment_enabled": _parse_bool,
    "files_max_thumbnail_px": int,
    "files_description_max_chars": int,
    "files_search_limit": int,
    # Image IP-adapter (was writable but lost on restart)
    "image_ip_adapter_enabled": _parse_bool,
    "image_ip_adapter_scale": float,
    # Image custom-import trust boundary
    "image_allow_pickle_formats": _parse_bool,
    "image_upload_max_size_gb": int,
    "image_imports_dir": str,
    # Cloud provider credentials — encrypted at rest with "enc:" prefix,
    # decrypted by _restore_settings on load. Without these in the map a
    # restart silently breaks every external backend (Anthropic, Azure,
    # OpenRouter, etc.) until the user re-enters every key.
    "anthropic_api_key": str,
    "anthropic_base_url": str,
    "azure_api_key": str,
    "azure_api_version": str,
    "azure_base_url": str,
    "azure_deployment": str,
    "cohere_api_key": str,
    "deepseek_api_key": str,
    "fireworks_api_key": str,
    "google_api_key": str,
    "google_vertex": _parse_bool,
    "google_vertex_project": str,
    "google_vertex_region": str,
    "groq_api_key": str,
    "mistral_api_key": str,
    "openrouter_api_key": str,
    "perplexity_api_key": str,
    "xai_api_key": str,
    # Multi-slot KV architecture (see
    # docs/superpowers/specs/2026-05-05-multi-slot-kv-design.md).
    # ``engine_multislot_enabled`` is tri-state: ``None`` means "auto"
    # (follow codebase recommendation), ``True``/``False`` are explicit
    # user overrides. ``_parse_optional_bool`` handles all three.
    # Without these in the restore map, the user's toggle is written
    # to the settings DB but never restored on container restart, and
    # the runtime falls back to the dataclass default — silently
    # losing the user's choice.
    "engine_multislot_enabled": _parse_optional_bool,
    "engine_parallel_slots": int,
    "engine_cache_ram_mib": int,
    # Engine defaults — global llama-server load knobs surfaced in
    # Model Manager > Advanced > "Engine Defaults". idle_timeout
    # applies live (idle monitor reads on each tick); the rest take
    # effect on the next subprocess start.
    "engine_idle_timeout": float,
    "engine_use_jinja_template": _parse_bool,
    "engine_kv_warm_on_start": _parse_bool,
    "engine_reasoning_format": str,
    "engine_reasoning_budget": int,
    "engine_reasoning_grace_period": int,
    "engine_kv_cache_type": str,
    # Tier-1 hardware/storage compat knobs.
    "engine_flash_attn": _parse_bool,
    "engine_health_timeout": float,
    # MTP self-speculation (PR #22673). Persisted so the user's
    # toggle survives container restart. Effective runtime gate is
    # in llama_server_manager._build_cli_args (heads-capability
    # check on the loaded GGUF) — see ``llama_mtp_skipped_no_heads``.
    "engine_mtp_enabled": _parse_bool,
    "engine_mtp_n_max": int,
    "engine_auto_pair_mmproj": _parse_bool,
    # Observation Substrate (BOM) — Phase A. See augmentum/observation/.
    "observation_substrate_enabled": _parse_bool,
    "observation_seed_chat_history": _parse_bool,
    "observation_lookup_cache_enabled": _parse_bool,
    "observation_lookup_cache_max_entries": int,
    "observation_primary_user_id": str,
    # Fabric master enable. See augmentum/fabric/__init__.py.
    "fabric_enabled": _parse_bool,
    "local_fabric_icon": str,
    # Audit-flagged backfill: each setting below is registered in
    # config_routes.py (_TOOL_SETTINGS / _STRING_SETTINGS) and writes to
    # the settings DB through /api/config/tools, but was missing from this
    # map — so user overrides survived in the DB and were silently
    # discarded on container restart. validate_wiring.py catches this
    # class of bug; keep new settings in both layers from day one.
    "ambient_loop_mode": str,
    "dream_compaction_enabled": _parse_bool,
    "dream_compaction_interval_hours": float,
    "dream_compaction_max_age_days": int,
    "dream_compaction_max_clusters_per_run": int,
    "dream_dedup_threshold": float,
    "dream_cluster_threshold": float,
    "dream_cluster_min_size": int,
    "dream_consolidation_low": float,
    "dream_consolidation_high": float,
    "dream_time_trim_count_threshold": int,
    "engine_kv_ttl_days": int,
    "engine_kv_narrative_ttl_days": int,
    "engine_kv_max_snapshots_per_model": int,
    "engine_kv_auto_pin_narrative": _parse_bool,
    "engine_kv_replay_enabled": _parse_bool,
    "engine_kv_replay_warm_sessions": int,
    "engine_kv_replay_budget_s": float,
    "engine_kv_replay_max_rows": int,
    "engine_speculation_enabled": _parse_bool,
    "engine_speculation_prefill_only": _parse_bool,
    "engine_speculation_max_new_tokens": int,
    "engine_speculation_ttl_s": float,
    "files_upload_max_file_bytes": int,
    "files_upload_max_files_per_request": int,
    "files_upload_max_request_bytes": int,
    "files_user_storage_quota_bytes": int,
    "voice_lipsync_engine": str,
    "voice_lipsync_universal": _parse_bool,
    "voice_xr_proxemics_enabled": _parse_bool,
    "chromium_binary_path": str,
    # CompanionRuntime flags
    "companion_runtime_enabled": _parse_bool,
    "companion_live_vision_enabled": _parse_bool,
    "companion_voice_decision_hud": _parse_bool,
    "companion_dispatch_enabled": _parse_bool,
    "companion_tick_enabled": _parse_bool,
    "companion_dreams_enabled": _parse_bool,
    "companion_drift_audit_enabled": _parse_bool,
    "companion_journal_enabled": _parse_bool,
    "companion_creations_enabled": _parse_bool,
    "companion_cultural_intake_enabled": _parse_bool,
    "companion_household_enabled": _parse_bool,
    "companion_peer_agents_enabled": _parse_bool,
    "companion_xr_orchestrator": _parse_bool,
    "companion_subagent_registry_active": _parse_bool,
    "companion_primitive_registry_active": _parse_bool,
    "companion_skill_archive_enabled": _parse_bool,
    # Synapse Layer §1 — chat→interior salience scoring
    "companion_salience_enabled": _parse_bool,
    "companion_salience_journal_threshold": float,
    "companion_salience_llm_enabled": _parse_bool,
    # Synapse Layer §3 — voice→interior journaling
    "companion_voice_journal_enabled": _parse_bool,
    # Promise/Deliver — second-companion-pass for tool results
    "companion_promise_deliver_tier": str,
    "companion_promise_deliver_strict_tier": _parse_bool,
    # Synapse Layer §2 — user-observed affect decay
    "companion_user_affect_half_life_s": float,
    # Synapse Layer §4 — slow consolidation
    "companion_consolidation_enabled": _parse_bool,
    "companion_consolidation_interval_days": int,
    "companion_consolidation_drift_ceiling": float,
    "companion_consolidation_min_evidence": int,
    # Chat-mode routing through the companion dispatcher
    "companion_dispatch_routes_chat": _parse_bool,
    "companion_dispatch_chat_min_utility": float,
    # Architect dispatch — companion-as-orchestrator with inferred defaults
    "architect_dispatch_enabled": _parse_bool,
    # Confidence-tier dispatch — LLM router replaces template-as-gate
    "architect_router_enabled": _parse_bool,
    "architect_router_model": str,
    "architect_router_timeout_ms": int,
    "companion_activation_mode": str,
    "companion_address_threshold": float,
    "companion_memory_min_score": float,
    "companion_profile_tone_only": _parse_bool,
    "memory_earned_permanence": _parse_bool,
    "memory_corroboration_promote_access": int,
    "memory_reflection_force_core": _parse_bool,
    "companion_address_llm_enabled": _parse_bool,
    "companion_address_llm_model": str,
    "companion_address_llm_timeout_ms": int,
    "companion_always_listening_warmup_ms": int,
    "companion_always_listening_vad_threshold": float,
    "companion_always_listening_prefix_padding_ms": int,
    "companion_address_media_boost": float,
    "companion_followup_window_s": float,
    "companion_open_thread_window_s": float,
    "companion_results_ring_enabled": _parse_bool,
    "companion_results_ring_turns": int,
    "companion_alert_watch_enabled": _parse_bool,
    "rsshub_base_url": str,
    "companion_image_prompt_expansion_enabled": _parse_bool,
    "companion_image_expansion_timeout_ms": int,
    # Becca-direct chat path (accumulation thesis Step 1)
    "companion_becca_direct_enabled": _parse_bool,
    # Narrative lorebook tools (model-driven lore management)
    "narrative_lorebook_tools_enabled": _parse_bool,
    # Skill graph (accumulation thesis Step 3)
    "companion_skills_enabled": _parse_bool,
    "companion_skill_relevance_threshold": float,
    "companion_skill_min_confidence_for_inject": float,
    "companion_skill_inject_top_k": int,
    # Lesson registry (mig 270) — learn-from-correction inverse
    "companion_lessons_enabled": _parse_bool,
    "companion_lessons_capture_enabled": _parse_bool,
    "companion_lessons_relevance_threshold": float,
    "companion_lessons_min_strength_for_inject": float,
    "companion_lessons_inject_top_k": int,
    "companion_initiative_threshold": float,
    "companion_initiative_enabled": _parse_bool,
    "companion_initiative_min_interval_s": float,
    # Becca persona mode + Lane 2/4 knobs
    "companion_persona_mode": _parse_bool,
    "companion_auto_summon": _parse_bool,
    "companion_care_cadence": str,
    "companion_presence_mode": str,
    "companion_locale": str,
    "companion_audio_cues": _parse_bool,
    "companion_keyboard_shortcuts": _parse_bool,
    "companion_notify_eod": _parse_bool,
    "companion_notify_drift_audit_push": _parse_bool,
    "companion_cooldown_minutes": int,
    "companion_quiet_hours_start": str,
    "companion_quiet_hours_end": str,
    "companion_default_owner_user_id": str,
    "companion_today_enabled": _parse_bool,
    "companion_today_reflect_hour_local": int,
    "companion_today_max_chars": int,
    "companion_discreet_auto_exit_minutes": int,
    "companion_discreet_location_aware": _parse_bool,
    "companion_always_raw": _parse_bool,
    "companion_drift_audit_interval_hours": float,
    "companion_creation_interval_hours": float,
    "companion_safety_floor_threshold_chat": float,
    "companion_safety_floor_threshold_coder": float,
    "companion_journal_hushed_until": str,
    # Companion aging / healing / drives / feedback / reflection
    "companion_aging_enabled": _parse_bool,
    "companion_aging_threshold_hours": int,
    "companion_healing_enabled": _parse_bool,
    "companion_drives_enabled": _parse_bool,
    "companion_drive_decay_half_life_hours": float,
    "companion_feedback_bias_enabled": _parse_bool,
    "companion_reflection_trait_nudge_enabled": _parse_bool,
    # Companion topical aggregator + wondering generator
    "companion_topical_aggregator_enabled": _parse_bool,
    "companion_topical_min_events": int,
    "companion_topical_window_hours": float,
    "companion_attention_sources": str,
    "companion_wondering_daily_cap": int,
    "companion_synthesize_daily_cap": int,
    "companion_synthesize_max_tokens": int,
    # Companion pre-context injection
    "companion_pre_context_enabled": _parse_bool,
    "companion_pre_context_min_keyword_overlap": int,
    "companion_pre_context_max_notes_scan": int,
    "companion_topic_mute_default_days": int,
    # Cast surface
    "cast_comic_library_ceiling": int,
    "cast_gallery_show_private": _parse_bool,
    "tv_auto_update": _parse_bool,
    "tv_update_channel": str,
    "vision_provider_enabled": _parse_bool,
    "vision_provider_model_path": str,
    "vision_provider_mmproj_path": str,
    "vision_provider_backend_port": int,
    "classifier_engine_enabled": _parse_bool,
    "classifier_engine_model_path": str,
    "classifier_engine_backend_port": int,
    "classifier_engine_gpu_layers": int,
    "classifier_engine_ctx_size": int,
    # Managed classifier slot ("Slot C")
    "classifier_slot_enabled": _parse_bool,
    "classifier_slot_model": str,
    "classifier_slot_backend_port": int,
    "classifier_slot_gpu_layers": int,
    "classifier_slot_ctx_size": int,
    # Comic narration cache retention
    "comic_narration_cache_max": int,
    # Coder mode — file_write cap, kernel toggles, subagent dispatch.
    "coder_file_write_max_tokens": int,
    "coder_context_reserve_pct": float,
    "coder_local_max_tokens_pct": int,
    "coder_local_max_tokens_cap": int,
    "coder_cloud_max_tokens_floor": int,
    "coder_subagents_enabled": _parse_bool,
    "coder_subagent_auto_explore": _parse_bool,
    "coder_subagent_max_concurrent": int,
    "coder_subagent_max_depth": int,
    "coder_subagent_fast_model": str,
    # Coder loop breakers — UI-tunable iteration caps. These were in
    # ``config_routes._TOOL_SETTINGS`` but missing here, which meant
    # every restart wiped the user's chosen values back to class defaults.
    # Same family of bug as the "subagents toggle keeps turning off"
    # symptom from 2026-05-31.
    "coder_breaker_validation_error_streak": int,
    "coder_breaker_same_validation_error_repeat": int,
    "coder_breaker_action_stagnation_break": int,
    "coder_breaker_test_failure_streak": int,
    "coder_breaker_same_file_edit_break": int,
    "coder_breaker_no_write_progress_break": int,
    "coder_breaker_inspection_loop_nudge": int,
    "coder_breaker_inspection_loop_break": int,
    # Coder loop tuning — iter caps + nudge cap + classifier toggle.
    "coder_hybrid_max_iters": int,
    "coder_hybrid_max_iters_ungated": int,
    "coder_native_nudge_max": int,
    "coder_next_speaker_check_enabled": _parse_bool,
    "coder_think_tool_enabled": _parse_bool,
    "coder_compact_tool_enabled": _parse_bool,
    # Coder mid-turn compaction tuning.
    "coder_compaction_auto_enabled": _parse_bool,
    "coder_compaction_threshold": float,
    "coder_compaction_keep_recent": int,
    # Coder turn archive — embedded compacted-turn store + PID-budget guard.
    "coder_archive_enabled": _parse_bool,
    "coder_archive_max_turns_per_workspace": int,
    "coder_workspace_pids_limit": int,
    "coder_workspace_pids_warn_pct": float,
    "coder_workspace_pids_check_interval_s": int,
    "coder_workspace_network_mode": str,
    # Narrative recall-tools — LLM-callable lookup layer (opt-in).
    "narrative_recall_tools_enabled": _parse_bool,
    "narrative_recall_tools_max_iters": int,
    "narrative_lorebook_tools_enabled": _parse_bool,
    "narrative_lorebook_native_tools_enabled": _parse_bool,
    "narrative_world_systems_enabled": _parse_bool,
    # Coder pause sweeping — already in config_routes; restore-map missing.
    "coder_max_paused_seconds": int,
    "coder_paused_sweep_interval_s": int,
    # Web search topic hints — same.
    "web_search_topic_hints_enabled": _parse_bool,
    # MCP — install-wide enablement + persisted HTTP server list (JSON).
    "mcp_enabled": _parse_bool,
    "mcp_servers": str,
    # Community install — kill switch for /community-install + /api/community/install.
    "community_install_enabled": _parse_bool,
    # Offers (chat-LLM-emitted Install/Save/Switch chips). See
    # docs/superpowers/specs/2026-06-02-offer-substrate-design.md.
    "offers_enabled": _parse_bool,
    "offers_max_per_day": int,
    "offers_max_per_turn": int,
    "offers_max_pending_per_session": int,
    "offers_default_expiry_days": int,
    # Voice pipeline modes — per-surface auto/local/server/custom.
    "voice_pipeline_mode_call": str,
    "voice_pipeline_mode_companion": str,
    "voice_pipeline_mode_narration": str,
    "voice_pipeline_mode_readaloud": str,
}


def _auto_derive_restore_parsers() -> dict[str, Callable]:
    """Derive restore parsers from ``config_routes._TOOL_SETTINGS`` +
    ``_STRING_SETTINGS`` — the single source of truth for what's
    user-tunable and what its type is.

    Eliminates the "add setting to _TOOL_SETTINGS, forget to add to
    _SETTINGS_RESTORE_MAP, lose its value on every restart" footgun.
    Every new bool/int/float/str setting now persists across restart
    automatically the moment it's registered in the validator dict.

    Manual ``_SETTINGS_RESTORE_MAP`` entries still take precedence
    (merged on top below) for cases that need custom parsing — e.g.
    encrypted-secret strings, comma-list parsers, or any setting
    whose type tuple doesn't map cleanly to ``int(val)`` / ``str(val)``.
    """
    from augmentum.proxy.config_routes import (
        _STRING_SETTINGS,
        _TOOL_SETTINGS,
    )

    type_to_parser: dict[type, Callable] = {
        bool:  _parse_bool,
        int:   int,
        float: float,
    }
    auto: dict[str, Callable] = {}
    for key, tup in _TOOL_SETTINGS.items():
        if not tup:
            continue
        parser = type_to_parser.get(tup[0])
        if parser is not None:
            auto[key] = parser
    for key in _STRING_SETTINGS:
        auto[key] = str
    return auto


async def _restore_settings(
    settings_store: object,
    restore_map: dict[str, type | Callable] | None = None,
) -> None:
    """Restore persisted config overrides from the settings store.

    When ``restore_map`` is None, uses the auto-derived map merged
    with the manual overrides — the production path. Tests can pass
    an explicit map to exercise specific keys in isolation.
    """
    from augmentum.utils.secrets import decrypt_api_key

    if restore_map is None:
        # Manual entries take precedence so encrypted-secret / custom-
        # parser cases keep their hand-rolled handling.
        auto = _auto_derive_restore_parsers()
        restore_map = {**auto, **_SETTINGS_RESTORE_MAP}

    for cfg_key, cast_fn in restore_map.items():
        val = await settings_store.get(cfg_key)
        if val is not None:
            try:
                # Decrypt values stored with encryption (enc: prefix)
                if isinstance(val, str) and val.startswith("enc:"):
                    val = decrypt_api_key(val)
                object.__setattr__(settings, cfg_key, cast_fn(val))
            except (ValueError, TypeError) as exc:
                log.warning("settings_restore_failed", key=cfg_key, value=val, error=str(exc))


class _MaxBodySizeMiddleware:
    """Reject requests whose body exceeds the configured limit.

    Checks Content-Length header up front *and* tracks actual bytes
    received via a wrapping ``receive`` callable so that chunked or
    header-less uploads are also caught.
    """

    # Path prefixes that legitimately ship large bodies. Each of these
    # endpoints enforces its OWN per-file / per-request cap, so we
    # bypass the global limit instead of duplicating per-route checks.
    #   /api/titles/upload-rom — single ROM stream, 5 GB cap
    #     (titles_routes._DEFAULT_ROM_MAX_BYTES; bumped to fit Wii
    #     single-layer)
    #   /api/titles/bulk-import — folder-drop multi-file, 5 GB per-file
    #     + 1024-files-per-request caps
    #     (titles_routes._BULK_PER_FILE_MAX_BYTES / _BULK_MAX_FILES);
    #     a 3 GB PS2 ISO would otherwise hit the global 50 MB cap and
    #     bounce as 413 before the route handler ran
    #   /api/files/upload — generic VFS upload, files_upload_max_request_bytes
    _LARGE_BODY_PATHS = (
        "/api/titles/upload-rom",
        "/api/titles/bulk-import",
        "/api/files/upload",
    )

    # Path suffixes that legitimately ship large bodies but carry a variable
    # segment (so a startswith prefix can't match). Endpoints matched here
    # MUST enforce their own per-request cap. ``/upload`` covers the coder
    # workspace upload (``/api/coder/files/{workspace_id}/upload`` — capped at
    # 200 MB in coder_routes.upload_files); without it a >50 MB code-folder
    # upload bounces at the global cap before the route's own limit applies.
    _LARGE_BODY_SUFFIXES = (
        "/upload",
    )

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        # Bypass for paths that explicitly handle their own size caps
        # (currently ROM uploads and file uploads). Without this,
        # large legitimate uploads get rejected by the global 50 MB
        # default before the route-specific cap can apply.
        path = scope.get("path", "")
        if (
            any(path.startswith(p) for p in self._LARGE_BODY_PATHS)
            or any(path.endswith(s) for s in self._LARGE_BODY_SUFFIXES)
        ):
            await self._app(scope, receive, send)
            return

        # Fast-reject via Content-Length header when present.
        headers = dict(scope.get("headers", []))
        cl = headers.get(b"content-length")
        if cl is not None:
            try:
                if int(cl) > self._max_bytes:
                    response = JSONResponse(
                        {"error": f"Request body too large (limit {self._max_bytes} bytes)"},
                        status_code=413,
                    )
                    await response(scope, receive, send)
                    return
            except (ValueError, TypeError):
                pass

        # Wrap receive to track actual bytes and enforce the limit
        # regardless of whether Content-Length was sent.
        bytes_received = 0
        max_bytes = self._max_bytes

        async def _counting_receive() -> dict:
            nonlocal bytes_received
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                bytes_received += len(body)
                if bytes_received > max_bytes:
                    raise _BodyTooLargeError
            return message

        try:
            await self._app(scope, _counting_receive, send)
        except _BodyTooLargeError:
            response = JSONResponse(
                {"error": f"Request body too large (limit {self._max_bytes} bytes)"},
                status_code=413,
            )
            await response(scope, receive, send)


class _BodyTooLargeError(Exception):
    """Raised internally when the request body exceeds the size limit."""


def _build_tool_registry(http_client: httpx.AsyncClient) -> ToolRegistry:
    """Create and populate the tool registry with all available tools."""
    from augmentum.tools.calculator import CalculatorTool
    from augmentum.tools.document_parse import DocumentParseTool
    from augmentum.tools.file_ops import FileOpsTool
    from augmentum.tools.hash_tool import HashTool
    from augmentum.tools.json_tool import JsonTool
    from augmentum.tools.math_verify import MathVerifyTool
    from augmentum.tools.python_exec import PythonExecTool
    from augmentum.tools.registry import ToolRegistry
    from augmentum.tools.search_files import SearchFilesTool
    from augmentum.tools.text_analysis import TextAnalysisTool
    from augmentum.tools.unit_converter import UnitConverterTool
    from augmentum.tools.web import WebTool
    from augmentum.tools.web_fetch import WebFetchTool
    from augmentum.tools.web_search import WebSearchTool
    from augmentum.tools.wikipedia import WikipediaTool
    from augmentum.tools.youtube import YouTubeTool

    registry = ToolRegistry()

    # Core tools — both receive the shared httpx client so web_fetch can
    # invoke the browse_fetch dispatch chain (Wikipedia REST / arXiv Atom /
    # Reddit feed / PDF text / etc.) instead of falling back to bare
    # trafilatura extraction on every URL.
    _fetch_tool = WebFetchTool(http_client=http_client)
    _search_tool = WebSearchTool(
        http_client=http_client, base_url=settings.searxng_base_url,
    )
    registry.register(_search_tool)
    registry.register(_fetch_tool)
    registry.register(WebTool(search_tool=_search_tool, fetch_tool=_fetch_tool))
    # Iterative research primitive — multi-query + reformulation-on-empty +
    # deep-read + honest miss, built on the search/fetch tools above. The
    # universal "look it up robustly" verb every surface carries.
    from augmentum.tools.research import ResearchTool
    registry.register(
        ResearchTool(search_tool=_search_tool, fetch_tool=_fetch_tool)
    )
    registry.register(WikipediaTool(http_client=http_client))
    registry.register(YouTubeTool(http_client=http_client, searxng_url=settings.searxng_base_url))
    registry.register(DocumentParseTool(base_dir=f"{settings.data_dir}/workdir"))
    registry.register(
        PythonExecTool(http_client=http_client, base_url=settings.executor_base_url)
    )
    registry.register(
        MathVerifyTool(http_client=http_client, executor_base_url=settings.executor_base_url)
    )
    registry.register(FileOpsTool(base_dir=f"{settings.data_dir}/workdir"))

    # Utility tools
    registry.register(CalculatorTool())
    # DateTimeTool removed — datetime is injected into every system prompt
    # via _inject_datetime(), making the tool redundant.
    registry.register(UnitConverterTool())
    registry.register(TextAnalysisTool())
    registry.register(JsonTool())
    registry.register(HashTool())
    registry.register(SearchFilesTool())

    return registry


async def _init_image_subsystem(app: FastAPI) -> None:
    """Initialize the image generation subsystem."""
    from augmentum.image.cache import ImageCache
    from augmentum.image.hardware import detect_hardware
    from augmentum.image.lora_manager import LoraManager
    from augmentum.image.model_manager import ModelManager as ImageModelManager
    from augmentum.image.pipeline_registry import PipelineRegistry
    from augmentum.image.presets import PresetManager
    from augmentum.image.queue import GenerationQueue

    # Resolve image directories from data_dir if not explicitly set
    if not settings.image_model_dir:
        object.__setattr__(settings, "image_model_dir", f"{settings.data_dir}/image_models")
    if not settings.image_output_dir:
        object.__setattr__(settings, "image_output_dir", f"{settings.data_dir}/image_output")

    # Ensure directories exist
    from pathlib import Path
    Path(settings.image_model_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.image_output_dir).mkdir(parents=True, exist_ok=True)

    log.info("image_subsystem_init", model_dir=settings.image_model_dir)

    # Hardware detection
    vram_limit = settings.image_vram_limit
    hw = detect_hardware(vram_limit=vram_limit)
    app.state.image_hardware = hw

    # Pipeline registry
    pipeline_reg = PipelineRegistry()
    app.state.image_pipeline_registry = pipeline_reg

    # Model manager — also scan /opt/augmentum/image_models for image-baked
    # defaults (e.g. DreamShaper 8 in the GPU variant). The system_dir lives
    # outside /data so it survives the volume mount; ModelManager merges
    # user-installed (writable) and image-baked (read-only) models.
    _SYSTEM_IMAGE_MODELS = "/opt/augmentum/image_models"
    model_mgr = ImageModelManager(
        settings.image_model_dir,
        system_dir=_SYSTEM_IMAGE_MODELS,
    )
    app.state.image_model_manager = model_mgr

    # LoRA manager
    lora_mgr = LoraManager(settings.image_model_dir)
    app.state.image_lora_manager = lora_mgr

    # Preset manager
    preset_mgr = PresetManager()
    app.state.image_preset_manager = preset_mgr

    # Persistence (if SQLite is available)
    persistence = None
    state_mgr = getattr(app.state, "state_manager", None)
    if state_mgr and isinstance(state_mgr.backend, SQLiteBackend):
        from augmentum.image.persistence import ImagePersistence
        persistence = ImagePersistence(state_mgr.backend.conn)
        app.state.image_persistence = persistence

    # Cache
    cache = ImageCache(persistence=persistence)
    app.state.image_cache = cache

    async def _resolve_source_image(source: str, persistence_ref, user_id: str = ""):
        """Load a PIL Image from base64 data, file path, or image_id.

        Returns ``None`` for any input that cannot be resolved to an image
        (record cleaned up, file missing, malformed base64, etc.). Callers
        rely on this to fall back gracefully — for IP-Adapter, that means
        plain generation without a reference rather than crashing the whole
        request. The previous implementation fell through to ``b64decode``
        for anything it didn't recognize, which raised ``binascii.Error``
        on every image_id-shaped string and took the entire ebook
        chapter generation down with it.
        """
        import base64
        import binascii
        import io

        from PIL import Image

        # Handle /api/image/<id> URLs — extract the ID. ``is_image_url``
        # records that the input was unambiguously an image-store
        # reference: if neither the persistence lookup nor the file-path
        # check succeeds, return ``None`` instead of falling through to
        # base64 decoding (which is guaranteed to fail on a hex/UUID
        # string and would mask a real "stale URL" symptom as a generic
        # decode crash).
        is_image_url = source.startswith("/api/image/")
        if is_image_url:
            source = source.rsplit("/", 1)[-1]

        # Try as image_id first (short hex string, no slashes)
        if (
            len(source) < 64 and "/" not in source and "\\" not in source
            and persistence_ref and user_id
        ):
            gen = await persistence_ref.get_generation(source, user_id=user_id)
            if gen and gen.get("file_path") and os.path.exists(gen["file_path"]):
                return await asyncio.to_thread(
                    lambda: Image.open(gen["file_path"]).convert("RGB")
                )
            if is_image_url:
                log.info("source_image_id_not_found", image_id=source[:80])
                return None

        # Try as file path
        if os.path.exists(source):
            resolved = Path(source).resolve()
            allowed_dirs = [Path(settings.image_output_dir).resolve()]
            if settings.image_model_dir:
                allowed_dirs.append(Path(settings.image_model_dir).resolve())
            if settings.data_dir:
                allowed_dirs.append(Path(settings.data_dir).resolve())
            if not any(
                resolved == d or str(resolved).startswith(str(d) + os.sep)
                for d in allowed_dirs
            ):
                log.warning("source_image_path_traversal_blocked", path=str(source))
                return None
            return await asyncio.to_thread(
                lambda: Image.open(source).convert("RGB")
            )

        # Reached here with an /api/image/ input means the file behind
        # the image_id is gone (or the persistence row had a stale path).
        # Don't try base64 — the tail of the URL isn't encoded image data.
        if is_image_url:
            log.info("source_image_file_missing", image_id=source[:80])
            return None

        # Treat as base64. Strip a data URI prefix if present, then decode
        # defensively — malformed input here returns ``None`` rather than
        # propagating a ``binascii.Error`` to the caller.
        if "," in source and source.index(",") < 100:
            source = source.split(",", 1)[1]
        try:
            image_data = base64.b64decode(source)
        except (ValueError, binascii.Error):
            log.info("source_image_base64_decode_failed", length=len(source))
            return None
        try:
            return Image.open(io.BytesIO(image_data)).convert("RGB")
        except (OSError, ValueError):
            log.info("source_image_base64_not_an_image", bytes=len(image_data))
            return None

    # Generation queue with worker
    async def _generate_fn(job):
        """The actual generation function called by the queue worker."""
        from augmentum.image.schemas import PipelineType

        def _stage(msg: str) -> None:
            """Update the job's human-readable progress stage."""
            job.stage = msg

        _stage("Resolving model")

        # Ensure a pipeline is loaded
        model_path = job.model
        if not model_path:
            model_path = settings.image_default_model
        if not model_path:
            # Prefer the currently loaded model, then any downloaded model,
            # then the VRAM-recommended model (which may not be downloaded)
            if pipeline_reg.is_loaded:
                model_path = pipeline_reg.current_model
            else:
                downloaded = model_mgr.list_local_models()
                if downloaded:
                    model_path = downloaded[0].get("name", "")
                else:
                    model_path = hw.recommended_model

        # Resolve local model path
        local_path = model_mgr.get_model_path(model_path)
        if local_path:
            model_path = local_path

        # Auto-download when no model is available.
        # DreamShaper 8 FP16 (~1.1GB) is the default — it's a high-quality
        # SD1.5 fine-tune that's SMALLER than vanilla SD1.5 FP16 (2.1GB),
        # runs on CPU or any GPU, and produces dramatically better output.
        # Falls back to vanilla SD1.5 if DreamShaper download fails.
        if not model_path or not os.path.exists(model_path):
            base_dir = settings.image_model_dir or f"{settings.data_dir}/image_models"

            # Check for existing models (DreamShaper or vanilla SD1.5)
            ds8_dir = os.path.join(base_dir, "dreamshaper-8")
            sd15_dir = os.path.join(base_dir, "sd-v1-5")

            if os.path.exists(ds8_dir):
                model_path = ds8_dir
            elif os.path.exists(sd15_dir):
                model_path = sd15_dir
            elif job.ip_adapter_image or not model_path:
                _stage("Downloading DreamShaper 8 (first time, ~1.1GB)")
                try:
                    from huggingface_hub import snapshot_download
                    await asyncio.to_thread(
                        snapshot_download,
                        "Lykon/dreamshaper-8",
                        local_dir=ds8_dir,
                        local_dir_use_symlinks=False,
                        variant="fp16",
                        token=settings.huggingface_token or None,
                        ignore_patterns=[
                            "*.bin", "*.msgpack", "*.onnx", "*.xml",
                            "flax_model.*", "*.safetensors.index.json",
                            "*fp32*",
                        ],
                    )
                    model_path = ds8_dir
                    log.info("dreamshaper8_auto_downloaded", path=ds8_dir)
                except Exception as exc:
                    log.warning("dreamshaper8_download_failed", error=str(exc),
                                hint="Falling back to vanilla SD1.5")
                    # Fallback: vanilla SD1.5
                    try:
                        _stage("Downloading SD1.5 fallback (~2GB)")
                        await asyncio.to_thread(
                            snapshot_download,
                            "stable-diffusion-v1-5/stable-diffusion-v1-5",
                            local_dir=sd15_dir,
                            local_dir_use_symlinks=False,
                            token=settings.huggingface_token or None,
                            ignore_patterns=[
                                "*.bin", "*.msgpack", "*.onnx", "*.xml",
                                "flax_model.*", "*.safetensors.index.json",
                            ],
                        )
                        model_path = sd15_dir
                        log.info("sd15_fallback_downloaded", path=sd15_dir)
                    except Exception as exc2:
                        log.warning("sd15_fallback_failed", error=str(exc2))

        # Determine pipeline type
        pipeline_type_str = hw.recommended_pipeline
        from augmentum.image.model_manager import _detect_pipeline_type
        if local_path:
            pipeline_type = _detect_pipeline_type(local_path)
        else:
            pipeline_type = PipelineType(pipeline_type_str)

        # Pre-load safety check (live VRAM + system RAM)
        # Skip if same model is already loaded (VRAM already consumed by it)
        # or if swapping models (current model will be unloaded first).
        from augmentum.image.hardware import pre_load_safety_check

        already_loaded = (
            pipeline_reg.is_loaded
            and pipeline_reg.current_model == model_path
        )
        is_swap = (
            pipeline_reg.is_loaded
            and pipeline_reg.current_model != model_path
        )
        if not already_loaded and not is_swap:
            pipeline_type_str_for_check = pipeline_type.value
            safety_err = pre_load_safety_check(model_path, pipeline_type_str_for_check, hw)
            if safety_err:
                raise RuntimeError(safety_err)

        # Load pipeline (swaps if needed)
        _stage("Loading model")
        device = settings.image_device
        if device == "auto":
            device = hw.device
        pipeline = await pipeline_reg.load(
            model_path, pipeline_type, device=device, dtype=settings.image_precision,
        )

        _stage("Preparing prompt")

        # Auto-condense prompt if it exceeds model's token limit
        if settings.image_prompt_condense and job.prompt and job.condense_prompt:
            try:
                from augmentum.image.prompt_condenser import (
                    condense_prompt,
                    detect_token_limit,
                    estimate_tokens,
                    needs_condensing,
                )

                diffusers_pipe = getattr(pipeline, "diffusers_pipe", None)
                if diffusers_pipe is None:
                    log.debug("prompt_condense_skip", reason="no diffusers_pipe property")
                else:
                    token_limit = detect_token_limit(diffusers_pipe)
                    est_tokens = estimate_tokens(job.prompt)
                    log.info(
                        "prompt_condense_check",
                        estimated_tokens=est_tokens,
                        token_limit=token_limit,
                        needs_condensing=needs_condensing(job.prompt, token_limit),
                    )
                    if needs_condensing(job.prompt, token_limit):
                        provider_reg = getattr(app.state, "provider_registry", None)
                        if not provider_reg or not provider_reg.backends:
                            log.warning("prompt_condense_skip", reason="no LLM backend available")
                        else:
                            backend, condense_model = await provider_reg.resolve_model_for_role(
                                "utility",
                                override=job.condense_model or settings.image_prompt_condense_model,
                                settings=settings,
                            )
                            log.info("prompt_condense_using", model=condense_model or "(auto)")
                            original_len = len(job.prompt)
                            job.prompt = await condense_prompt(
                                job.prompt, token_limit, backend,
                                model=condense_model,
                                image_model=job.model,
                            )
                            log.info(
                                "prompt_condensed_result",
                                original_chars=original_len,
                                condensed_chars=len(job.prompt),
                                model=settings.image_prompt_condense_model or "(default)",
                            )
            except Exception:
                log.warning("prompt_condense_failed", exc_info=True)

        # Apply model profile — check feature compatibility
        from augmentum.image.model_profiles import resolve_profile
        _profile = resolve_profile(model_path, pipeline_type.value)

        # Apply default negative prompt (skip for models that don't support it)
        if not job.negative_prompt and _profile.features.get("negative_prompt", True):
            from augmentum.image.prompt_condenser import DEFAULT_NEGATIVE
            job.negative_prompt = settings.image_default_negative_prompt or DEFAULT_NEGATIVE

        # Load LoRAs if specified (with base model compatibility check)
        if job.loras:
            _stage("Loading LoRAs")
            current_pt = pipeline_type.value  # "sd15", "sdxl", "flux"
            for lora in job.loras:
                lora_info = lora_mgr.match_character(lora["name"]) or None
                if not lora_info:
                    # Try direct path lookup
                    lora_path = lora_mgr.get_path(lora["name"])
                else:
                    lora_path = lora_info.path
                    # Check base model compatibility
                    if lora_info.base_model and lora_info.base_model != current_pt:
                        log.warning("lora_base_mismatch",
                                    lora=lora["name"],
                                    lora_base=lora_info.base_model,
                                    pipeline=current_pt)
                        # Skip incompatible LoRA rather than loading garbage
                        continue
                if lora_path:
                    await pipeline.load_lora(lora_path, weight=lora.get("weight", 1.0))

        # Reset bar fields each time the diffusion phase starts so a
        # downstream Saving / postprocess phase doesn't leave a stale
        # "step 20/20" reading on a job that completed and got reused
        # by the next request (job objects are per-submission, but
        # defensive zeroing makes the contract explicit at the boundary).
        job.steps_total = 0
        job.steps_done = 0
        _stage("Generating")

        def _step_cb(done: int, total: int) -> None:
            """Per-diffusion-step progress callback.

            Updates the job's determinate fields so polling UIs render a
            real bar (steps_done / steps_total) AND refines the stage
            text with the running step count so even hover tooltips
            without a bar component show something meaningful. Called
            from a torch thread via diffusers' callback_on_step_end —
            this is the ONLY mutation point during the diffusion loop.
            """
            if total <= 0:
                return
            job.steps_total = total
            job.steps_done = done
            # Keep the text label in sync with the bar — different
            # surfaces consume one or the other.
            job.stage = f"Generating step {done}/{total}"

        # Generate based on job type
        from augmentum.image.schemas import JobType

        if job.job_type == JobType.IMG2IMG:
            source_pil = await _resolve_source_image(job.source_image, persistence, job.user_id)
            if job.width and job.height:
                source_pil = source_pil.resize((job.width, job.height))
            else:
                job.width, job.height = source_pil.size
            result = await pipeline.img2img(
                prompt=job.prompt,
                image=source_pil,
                negative_prompt=job.negative_prompt,
                strength=job.strength,
                steps=job.steps,
                cfg_scale=job.cfg_scale,
                seed=job.seed,
                sampler=job.sampler,
                output_dir=settings.image_output_dir,
                step_callback=_step_cb,
            )
        elif job.job_type == JobType.INPAINT:
            source_pil = await _resolve_source_image(job.source_image, persistence, job.user_id)
            if not job.width or not job.height:
                job.width, job.height = source_pil.size
            mask_pil = await _resolve_source_image(job.mask_image, None, job.user_id)
            if source_pil.size != mask_pil.size:
                mask_pil = mask_pil.resize(source_pil.size)
            if job.mask_blur > 0:
                from PIL import ImageFilter
                mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=job.mask_blur))

            # Apply inpaint mode overrides
            if job.inpaint_mode == "improve":
                # Low denoise preserves content, enhances detail
                if job.strength >= 0.75:
                    job.strength = 0.35
            elif job.inpaint_mode == "modify":
                # Full creative freedom — fill masked region with noise
                job.strength = 1.0
                import numpy as np
                from PIL import Image
                mask_arr = np.array(mask_pil.convert("L"))
                src_arr = np.array(source_pil)
                rng = np.random.default_rng(job.seed if job.seed >= 0 else None)
                noise = rng.integers(0, 256, src_arr.shape, dtype=np.uint8)
                src_arr[mask_arr > 127] = noise[mask_arr > 127]
                source_pil = Image.fromarray(src_arr)

            _do_full_res = False
            if job.inpaint_full_res:
                import numpy as np
                from PIL import Image

                mask_arr = np.array(mask_pil.convert("L"))
                rows = np.any(mask_arr > 127, axis=1)
                cols = np.any(mask_arr > 127, axis=0)

                if rows.any() and cols.any():
                    rmin, rmax = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
                    cmin, cmax = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])

                    pad = job.inpaint_padding
                    rmin = max(0, rmin - pad)
                    rmax = min(source_pil.height - 1, rmax + pad)
                    cmin = max(0, cmin - pad)
                    cmax = min(source_pil.width - 1, cmax + pad)

                    crop_box = (cmin, rmin, cmax + 1, rmax + 1)
                    cropped_src = source_pil.crop(crop_box)
                    cropped_mask = mask_pil.crop(crop_box)
                    crop_w, crop_h = cropped_src.size

                    # Only use full-res if crop is large enough to be meaningful
                    if crop_w >= 16 and crop_h >= 16:
                        _do_full_res = True

            if _do_full_res:
                from PIL import Image

                # Resize crop to model's native resolution (SD1.5=512, SDXL/FLUX=1024)
                _native_res = {"sd15": 512, "sdxl": 1024, "flux": 1024}
                _ptype = getattr(pipeline, "pipeline_type", None)
                model_res = _native_res.get(str(_ptype.value) if _ptype else "", 512)
                target_w = (model_res // 8) * 8
                target_h = (model_res // 8) * 8
                resized_src = cropped_src.resize((target_w, target_h), Image.LANCZOS)
                resized_mask = cropped_mask.resize((target_w, target_h), Image.NEAREST)

                result = await pipeline.inpaint(
                    prompt=job.prompt, image=resized_src, mask=resized_mask,
                    negative_prompt=job.negative_prompt, strength=job.strength,
                    steps=job.steps, cfg_scale=job.cfg_scale, seed=job.seed,
                    sampler=job.sampler, output_dir=settings.image_output_dir,
                    step_callback=_step_cb,
                )

                # Paste result back into original using mask as alpha
                result_img = Image.open(result.file_path)
                result_resized = result_img.resize((crop_w, crop_h), Image.LANCZOS)
                cropped_mask_l = cropped_mask.convert("L")
                full_output = source_pil.copy()
                full_output.paste(result_resized, (cmin, rmin), cropped_mask_l)
                await asyncio.to_thread(full_output.save, result.file_path)
                result.width = full_output.width
                result.height = full_output.height
            else:
                result = await pipeline.inpaint(
                    prompt=job.prompt, image=source_pil, mask=mask_pil,
                    negative_prompt=job.negative_prompt, strength=job.strength,
                    steps=job.steps, cfg_scale=job.cfg_scale, seed=job.seed,
                    sampler=job.sampler, output_dir=settings.image_output_dir,
                    step_callback=_step_cb,
                )
        else:
            # Resolve IP-Adapter reference image(s) — supports single or multiple
            ip_adapter_pil = None
            if job.ip_adapter_image and not settings.image_ip_adapter_enabled:
                log.info("ip_adapter_disabled_by_config", job_id=job.job_id)
            elif job.ip_adapter_image:
                _stage("Loading reference image(s)")
                # Normalize to list for uniform handling
                ref_sources = job.ip_adapter_image if isinstance(job.ip_adapter_image, list) else [job.ip_adapter_image]
                ref_sources = [s for s in ref_sources if s]  # drop empty strings

                if ref_sources:
                    resolved = []
                    for src in ref_sources:
                        # IP-Adapter is a hint, not a hard requirement — a
                        # broken or missing reference must not abort the
                        # whole generation. ``_resolve_source_image``
                        # already returns ``None`` for known miss cases;
                        # this guard catches any unexpected exception
                        # (corrupted file, decoder OOM, etc.) so we
                        # degrade to plain generation instead of crashing.
                        try:
                            pil = await _resolve_source_image(src, persistence, job.user_id)
                        except Exception as exc:
                            log.warning(
                                "ip_adapter_reference_resolution_failed",
                                source=str(src)[:120],
                                error=type(exc).__name__,
                                detail=str(exc)[:200],
                            )
                            pil = None
                        if pil:
                            resolved.append(pil)
                    if resolved:
                        # Single image → pass directly, multiple → pass as list
                        ip_adapter_pil = resolved[0] if len(resolved) == 1 else resolved
                        if not pipeline._ip_adapter_loaded:
                            _stage("Loading IP-Adapter")
                            try:
                                await pipeline.load_ip_adapter(scale=job.ip_adapter_scale)
                            except Exception:
                                log.warning("ip_adapter_load_failed", pipeline=type(pipeline._pipe).__name__)
                                ip_adapter_pil = None
                                try:
                                    await pipeline.unload_ip_adapter()
                                except Exception as exc:
                                    log.debug("ip_adapter_unload_after_failure_failed", error=str(exc))
            _gen_kwargs = dict(
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                width=job.width,
                height=job.height,
                steps=job.steps,
                cfg_scale=job.cfg_scale,
                seed=job.seed,
                sampler=job.sampler,
                output_dir=settings.image_output_dir,
                guidance_rescale=job.guidance_rescale,
                hires_fix=job.hires_fix,
                hires_scale=job.hires_scale,
                hires_denoise=job.hires_denoise,
                clip_skip=job.clip_skip,
                ip_adapter_image=ip_adapter_pil,
                ip_adapter_scale=job.ip_adapter_scale,
                step_callback=_step_cb,
            )
            try:
                result = await pipeline.generate(**_gen_kwargs)
            except (RuntimeError, ValueError, TypeError) as exc:
                if _is_ip_adapter_generation_error(exc):
                    # IP-Adapter incompatible with this model — retry without
                    log.warning(
                        "ip_adapter_generation_retry_without_adapter",
                        job_id=job.job_id,
                        error=str(exc)[:500],
                    )
                    try:
                        await pipeline.unload_ip_adapter()
                    except Exception:
                        log.warning(
                            "ip_adapter_unload_failed_after_generation_error",
                            job_id=job.job_id,
                            exc_info=True,
                        )
                    _gen_kwargs["ip_adapter_image"] = None
                    result = await pipeline.generate(**_gen_kwargs)
                else:
                    raise

        # Diffusion is done — clear the step bar so the Saving / VFS-
        # register / persist phases don't lie about progress. UIs that
        # render the bar gate on steps_total > 0; clearing it switches
        # them back to indeterminate spinner mode for the tail phases.
        job.steps_total = 0
        job.steps_done = 0
        _stage("Saving")

        # Persist generation. ImagePersistence.save_generation refuses to
        # write a row with an empty user_id (it's a multi-tenant guard —
        # without scoping the row would be untouchable from any tenant's
        # UI). Match that contract here so a job that somehow reached the
        # worker without a user can't crash the worker AND can't leave an
        # orphan PNG with no DB row. Loud warning so any new entry point
        # that drops user_id is immediately visible in logs instead of
        # silently failing.
        vfs_registered = True
        if persistence and job.user_id:
            vfs_registered = await persistence.save_generation(
                image_id=result.image_id,
                session_id=job.session_id,
                prompt=job.prompt,
                negative_prompt=job.negative_prompt,
                model=model_path,
                seed=result.seed,
                width=result.width,
                height=result.height,
                steps=job.steps,
                cfg_scale=job.cfg_scale,
                preset=job.preset,
                loras=job.loras,
                file_path=result.file_path,
                job_type=job.job_type.value,
                strength=job.strength,
                source_image_id=job.source_image_id,
                user_id=job.user_id,
                origin=getattr(job, "origin", "") or "",
            )
        elif persistence:
            log.warning(
                "image_persist_skipped_missing_user_id",
                image_id=result.image_id,
                session_id=job.session_id,
                job_type=job.job_type.value,
            )
            vfs_registered = False

        return {
            "image_id": result.image_id,
            "file_path": result.file_path,
            "seed": result.seed,
            "width": result.width,
            "height": result.height,
            "vfs_registered": vfs_registered,
        }

    queue = GenerationQueue(max_size=settings.image_max_queue_size)
    queue.start(_generate_fn)
    app.state.image_queue = queue

    # Register ImageGenerationTool (lazy)
    tool_registry = getattr(app.state, "tool_registry", None)
    if tool_registry:
        from augmentum.tools.image_generation import ImageGenerationTool
        tool_registry.register(ImageGenerationTool(queue=queue, preset_manager=preset_mgr, app_state=app.state))
        log.info("image_generation_tool_registered")

    log.info(
        "image_subsystem_ready",
        device=hw.device,
        tier=hw.tier.value,
        recommended_model=hw.recommended_model,
    )


async def _auto_register_audio_providers(conn) -> None:
    """Register bundled audio services as providers if not already present.

    Called on startup when Docker compose sets the service URL env vars.
    Creates providers as non-default so the user can assign them in Settings.
    """

    _BUNDLED = [
        {
            "id": "moonshine-stt",
            "provider_type": "stt",
            "name": "Moonshine (built-in)",
            "builtin": True,
            "enabled_setting": "voice_moonshine_enabled",
            "default_model": "moonshine-streaming-medium",
        },
        {
            "id": "speaches-stt",
            "provider_type": "stt",
            "name": "Speaches (bundled)",
            "url_setting": "stt_provider_url",
            "model_setting": "stt_default_model",
            "webui_hint": "http://localhost:6200",
        },
        {
            "id": "kokoro-builtin",
            "provider_type": "tts",
            "name": "Kokoro (built-in)",
            "builtin": True,
            "enabled_setting": "tts_kokoro_builtin",
            "default_model": "kokoro",
            "default_voice": "af_heart",
        },
        {
            "id": "pockettts-builtin",
            "provider_type": "tts",
            "name": "Pocket TTS (built-in)",
            "builtin": True,
            "enabled_setting": "tts_pocket_builtin",
            "default_model": "pocket-tts",
            "default_voice": "alba",
        },
        {
            "id": "kokoro-tts",
            "provider_type": "tts",
            "name": "Kokoro TTS (sidecar)",
            "url_setting": "tts_kokoro_url",
            "model_setting": None,
            "default_model": "kokoro",
            "default_voice": "af_heart",
            "webui_hint": "http://localhost:6300/web",
        },
        {
            "id": "chatterbox-tts",
            "provider_type": "tts",
            "name": "Chatterbox TTS (bundled)",
            "url_setting": "tts_chatterbox_url",
            "model_setting": None,
            "default_model": "tts-1",
            "default_voice": "",
            "webui_hint": "http://localhost:6401",
        },
        {
            "id": "chatterbox-turbo",
            "provider_type": "tts",
            "name": "Chatterbox Turbo (bundled)",
            "url_setting": "tts_chatterbox_turbo_url",
            "model_setting": None,
            "default_model": "chatterbox-turbo",
            "default_voice": "",
            "webui_hint": "",
        },
        {
            "id": "qwen-tts",
            "provider_type": "tts",
            "name": "Qwen3 TTS (bundled)",
            "url_setting": "tts_qwen_url",
            "model_setting": None,
            "default_model": "tts-1",
            "default_voice": "Vivian",
            "webui_hint": "",
        },
        {
            "id": "fish-tts",
            "provider_type": "tts",
            "name": "Fish Speech (bundled)",
            "url_setting": "tts_fish_url",
            "model_setting": None,
            "default_model": "openaudio-s1-mini",
            "default_voice": "",
            "webui_hint": "http://localhost:6600",
        },
        {
            "id": "sesame-csm",
            "provider_type": "tts",
            "name": "Sesame CSM (bundled)",
            "url_setting": "tts_sesame_csm_url",
            "model_setting": None,
            "default_model": "sesame-csm-1b",
            # CSM has no built-in voices; the sidecar exposes two bundled
            # example voices so the picker is never empty (see services/
            # sesame-csm/app.py::_BUNDLED_VOICES). This default seeds the
            # fallback + fabric heartbeat with a real selectable voice.
            "default_voice": "conversational_a",
            "webui_hint": "",
        },
        {
            # Model-agnostic OpenAI-compatible TTS. Point tts_openai_url at ANY
            # server exposing POST /v1/audio/speech (Higgs Audio v3 via
            # sglang-omni, a peer's TTS, OpenAI, etc.). Uses the generic
            # /v1/audio/speech dispatch — no per-model branching. Bring your
            # own endpoint; the protocol is the contract, not the model.
            "id": "openai-tts",
            "provider_type": "tts",
            "name": "OpenAI-Compatible TTS (custom endpoint)",
            "url_setting": "tts_openai_url",
            "model_setting": None,
            "default_model": "tts-1",
            "default_voice": "default",
            "webui_hint": "",
        },
    ]

    for spec in _BUNDLED:
        is_builtin = spec.get("builtin", False)

        if is_builtin:
            # Built-in: check the enabled_setting flag
            enabled_setting = spec.get("enabled_setting", "")
            if not getattr(settings, enabled_setting, False):
                continue
            base_url = "builtin"
        else:
            # Sidecar: check if URL is configured via env var
            base_url = getattr(settings, spec.get("url_setting", ""), "")
            if not base_url:
                continue

        # Check if already registered
        cursor = await conn.execute(
            "SELECT id FROM audio_providers WHERE id = ?", (spec["id"],),
        )
        if await cursor.fetchone():
            # Update URL in case it changed
            await conn.execute(
                "UPDATE audio_providers SET base_url = ? WHERE id = ?",
                (base_url, spec["id"]),
            )
            await conn.commit()
            continue

        # Determine default model
        default_model = ""
        if spec.get("model_setting"):
            default_model = getattr(settings, spec["model_setting"], "")
        if not default_model:
            default_model = spec.get("default_model", "")

        # Check if this should be default (first of its type)
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM audio_providers WHERE provider_type = ?",
            (spec["provider_type"],),
        )
        count = (await cursor.fetchone())[0]
        is_default = 1 if count == 0 else 0

        try:
            await conn.execute(
                "INSERT INTO audio_providers "
                "(id, provider_type, name, base_url, default_model, default_voice, is_default) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (spec["id"], spec["provider_type"], spec["name"], base_url,
                 default_model, spec.get("default_voice", ""), is_default),
            )
            await conn.commit()
            log.info(
                "bundled_audio_provider_registered",
                id=spec["id"],
                type=spec["provider_type"],
                is_default=bool(is_default),
            )
        except Exception:
            log.warning("audio_provider_registration_failed", exc_info=True)


async def _migrate_plaintext_keys(conn) -> None:
    """One-time migration: encrypt any plaintext secrets at rest across the
    credential tables. Idempotent — values already ``enc:``-prefixed are
    skipped, so this is safe to run on every startup. Each table names its
    own secret column (provider keys vs media-server access tokens)."""
    from augmentum.utils.secrets import encrypt_api_key

    # (table, secret column). user_media_servers already encrypts on
    # write/read (media/store.py); this sweep catches rows created before
    # that landed so nothing stays plaintext at rest.
    targets = [
        ("providers", "api_key"),
        ("audio_providers", "api_key"),
        ("image_providers", "api_key"),
        ("user_media_servers", "access_token"),
    ]
    for table, col in targets:
        try:
            cursor = await conn.execute(
                f"SELECT id, {col} FROM {table} WHERE {col} IS NOT NULL AND {col} != ''"  # noqa: S608
            )
            rows = await cursor.fetchall()
            migrated = 0
            for row in rows:
                row_id, raw = row[0], row[1]
                if raw and not raw.startswith("enc:"):
                    encrypted = encrypt_api_key(raw)
                    if encrypted != raw:
                        await conn.execute(
                            f"UPDATE {table} SET {col} = ? WHERE id = ?",  # noqa: S608
                            (encrypted, row_id),
                        )
                        migrated += 1
            if migrated:
                await conn.commit()
                log.info("secrets_encrypted_at_rest", table=table, column=col, count=migrated)
        except Exception:
            log.debug("key_migration_skipped", table=table, exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle — create and tear down shared resources."""
    log.info("starting", port=settings.port)

    # Background-task registry. Every long-lived task created during
    # lifespan should be appended here (or via ``_track_bg``) so the
    # shutdown block can cancel them. Without this, the maintenance/
    # warmup/enrichment loops sit in ``await asyncio.sleep(...)`` past
    # SIGTERM and Docker has to SIGKILL the container after the 10s
    # grace, producing log noise and slowing every restart.
    app.state.background_tasks: list[asyncio.Task] = []

    def _track_bg(coro, *, name: str = "") -> asyncio.Task:
        task = asyncio.create_task(coro, name=name) if name else asyncio.create_task(coro)
        app.state.background_tasks.append(task)
        return task

    # Warm the UI-shell digest off the request path so the native app's
    # launch-time /api/ui-version check answers from cache. Pure best-effort:
    # the endpoint recomputes lazily if this hasn't finished yet.
    async def _warm_ui_shell_digest() -> None:
        try:
            from pathlib import Path as _P

            _ui = _P(__file__).resolve().parent.parent.parent / "ui"
            await asyncio.to_thread(_ui_shell_digest, _ui)
        except Exception:
            log.debug("ui_shell_digest_warm_skipped", exc_info=True)

    _track_bg(_warm_ui_shell_digest(), name="warm_ui_shell_digest")

    # NOTE: ``settings.max_concurrent_requests`` is intentionally not wired
    # to a semaphore here. A previous implementation created
    # ``app.state.inference_semaphore = asyncio.Semaphore(...)`` and never
    # acquired it anywhere, which gave the misleading impression of a
    # concurrency cap while leaving traffic completely unbounded. The real
    # concurrency bottleneck for inference is ``LlamaCppBackend._slot_lock``
    # (one slot per loaded model). If a global cap is later needed, acquire
    # a fresh semaphore around the specific paths it should bound — don't
    # resurrect a process-wide lock that nothing references.
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=settings.http_connect_timeout,
            read=settings.http_read_timeout,
            write=settings.http_write_timeout,
            pool=settings.http_pool_timeout,
        ),
        limits=httpx.Limits(
            max_connections=settings.http_max_connections,
            max_keepalive_connections=settings.http_max_keepalive,
        ),
        headers={"User-Agent": "Augmentum/1.0 (https://github.com/augmentum; tool-proxy)"},
    )
    # Separate client for provider chat/inference dispatch so catalog
    # probes and /api/tags listing don't contend with active LLM chat
    # turns for the same outbound connection pool. Same limits and
    # timeouts, independent pool.
    app.state.chat_http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=settings.http_connect_timeout,
            read=settings.http_read_timeout,
            write=settings.http_write_timeout,
            pool=settings.http_pool_timeout,
        ),
        limits=httpx.Limits(
            max_connections=settings.http_max_connections,
            max_keepalive_connections=settings.http_max_keepalive,
        ),
        headers={"User-Agent": "Augmentum/1.0 (https://github.com/augmentum; tool-proxy)"},
    )
    app.state.provider_registry = ProviderRegistry(
        app.state.http_client,
        chat_http_client=app.state.chat_http_client,
    )
    # Probe configured backends — bounded so a misconfigured remote (e.g.,
    # an Ollama address that points at a stopped server) can't block boot
    # forever. probe_backends fans out serially with httpx defaults; in
    # practice a 15s ceiling covers the slowest legitimately-reachable
    # endpoint while bounding the cost of unreachable ones.
    try:
        await asyncio.wait_for(
            app.state.provider_registry.probe_backends(),
            timeout=15.0,
        )
    except TimeoutError:
        log.warning("provider_probe_timeout", timeout_s=15)

    # ---- Engine v2: managed llama-server subprocess ----
    # Resolve the llama-server binary. The Docker image bakes it to
    # /usr/local/bin/llama-server (the default setting). Native installs
    # may put it elsewhere — Apple Silicon Homebrew lands in
    # /opt/homebrew/bin, Linux native /usr/bin, etc. Fall back to PATH
    # lookup when the configured path doesn't resolve, so a Mac user
    # who installed via `brew install llama.cpp` doesn't have to override
    # the setting just to enable Engine v2.
    import shutil
    _engine_binary = settings.engine_llama_server_path
    if not os.path.isfile(_engine_binary):
        _path_lookup = shutil.which("llama-server")
        if _path_lookup:
            _engine_binary = _path_lookup

    _engine_enabled = settings.engine_managed
    if _engine_enabled == "auto":
        _engine_enabled = os.path.isfile(_engine_binary)
        if _engine_enabled:
            log.info(
                "engine_v2_auto_detected",
                path=_engine_binary,
                source=("setting" if _engine_binary == settings.engine_llama_server_path else "PATH"),
            )
    else:
        _engine_enabled = _engine_enabled.lower() in ("true", "1", "yes")

    # Engine v2 init is wrapped so a failure doesn't take down the whole server.
    # If it fails, the server starts without the managed engine — other backends
    # (Ollama, OpenAI, etc.) still work.
    app.state.llama_manager = None
    app.state.token_count_cache = None
    app.state.secondary_slot = None  # second resident local model ("Slot B")
    app.state.classifier_slot = None  # managed classifier/utility/vision model ("Slot C")

    if _engine_enabled:
        try:
            from augmentum.models.llama_cpp import LlamaCppBackend
            from augmentum.models.llama_server_manager import LlamaServerManager
            from augmentum.models.token_count_cache import TokenCountCache

            # Create token cache FIRST so it's available from the moment
            # the manager + backend exist (no race on first request)
            # Keep the token cache isolated from the primary state DB so a
            # corrupt or transient cache file can be rebuilt safely.
            tc_db = os.path.join(settings.data_dir, "cache", "token_count_cache.db")
            kv_manifest_db = os.path.join(settings.data_dir, "cache", "engine_kv_manifest.db")
            token_cache = TokenCountCache(tc_db)
            await token_cache.init_db()

            extra_dirs = [
                d.strip()
                for d in settings.engine_extra_model_dirs.split(";")
                if d.strip()
            ]
            llama_manager = LlamaServerManager(
                llama_server_path=_engine_binary,
                backend_port=settings.engine_backend_port,
                model_dir=settings.engine_model_dir,
                extra_model_dirs=extra_dirs or None,
                gpu_layers=settings.engine_gpu_layers,
                ctx_size=settings.engine_ctx_size,
                batch_size=settings.engine_batch_size,
                kv_manifest_db=kv_manifest_db,
                kv_ttl_days=settings.engine_kv_ttl_days,
                kv_narrative_ttl_days=settings.engine_kv_narrative_ttl_days,
                kv_max_snapshots_per_model=settings.engine_kv_max_snapshots_per_model,
                kv_auto_pin_narrative=settings.engine_kv_auto_pin_narrative,
                kv_warm_on_start=settings.engine_kv_warm_on_start,
            )
            llama_manager.kv_cache_type = settings.engine_kv_cache_type
            llama_manager.draft_model = settings.engine_draft_model
            llama_manager.draft_max = settings.engine_draft_max
            llama_manager.draft_ctx_size = settings.engine_draft_ctx_size
            llama_manager.draft_gpu_layers = settings.engine_draft_gpu_layers
            llama_manager.draft_min = settings.engine_draft_min
            llama_manager.draft_p_min = settings.engine_draft_p_min
            llama_manager.mtp_enabled = settings.engine_mtp_enabled
            llama_manager.mtp_n_max = settings.engine_mtp_n_max
            llama_manager.flash_attn = settings.engine_flash_attn
            llama_manager.idle_timeout = settings.engine_idle_timeout
            llama_manager.health_timeout = settings.engine_health_timeout

            # Attach token cache immediately — the manager exposes it
            # to the backend via the ``token_cache`` property. Using the
            # public setter keeps the attribute's lifecycle visible.
            llama_manager.set_token_cache(token_cache)
            # NOTE: settings_store is attached later (after its own init at
            # the bottom of this lifespan) so the lazy-load path can read
            # the persisted "engine.last_load.<model_id>" defaults.

            # Register as an engine backend
            engine_backend = LlamaCppBackend(
                http_client=app.state.http_client,
                base_url=llama_manager.base_url,
                server_manager=llama_manager,
            )
            app.state.provider_registry.register_backend("engine", engine_backend)

            # Add common GGUF locations (LM Studio, HuggingFace Hub, etc.)
            llama_manager.add_common_model_dirs()

            # Scan additional known directories if they exist
            for extra in [
                settings.llamacpp_model_dir,  # where GGUF downloads go
                "/models/host",               # setup wizard host mount
                "/models/lab",                # additional host mount
                "/data/host-models",          # engine v2 host mount convention
            ]:
                if extra and os.path.isdir(extra) and extra not in llama_manager.model_dirs:
                    llama_manager.model_dirs.append(extra)

            # Restore user-configured model dirs from settings. We also
            # rewrite the persisted value when it contains paths that no
            # longer exist — e.g. a host-mount layout that changed under
            # us (WSL ext4 disk re-pathed, bind mount removed). Without
            # the self-heal, "ghost" entries silently accumulate every
            # time the user touches the Manage Dirs UI: stored ones get
            # filtered by the isdir check but rewritten next save.
            try:
                ss = getattr(app.state, "settings_store", None)
                if ss:
                    saved_dirs = (await ss.get("engine_v2_extra_model_dirs")) or ""
                    if saved_dirs:
                        raw_entries = [d.strip() for d in saved_dirs.split(";") if d.strip()]
                        kept: list[str] = []
                        dropped: list[str] = []
                        for d in raw_entries:
                            if os.path.isdir(d):
                                kept.append(d)
                                if d not in llama_manager.model_dirs:
                                    llama_manager.model_dirs.append(d)
                                    log.info("engine_v2_restored_dir", path=d)
                            else:
                                dropped.append(d)
                        if dropped:
                            await ss.set("engine_v2_extra_model_dirs", ";".join(kept))
                            log.info(
                                "engine_v2_extra_dirs_pruned",
                                dropped=dropped, kept=kept,
                            )
            except Exception as exc:
                # settings_store may not be ready during the early lifespan
                # window; the restore is opportunistic — settings come back
                # via the normal settings restore later in lifespan.
                log.debug("engine_v2_extra_dirs_restore_skipped", error=str(exc))

            if settings.engine_auto_discover:
                try:
                    # Walks all model_dirs and reads GGUF headers from disk —
                    # bounded so a huge dir or slow/network mount can't keep
                    # us in STARTING state past Docker's start_period.
                    n = await asyncio.wait_for(
                        llama_manager.scan_and_cache_profiles(),
                        timeout=30.0,
                    )
                    log.info("engine_v2_profile_scan_done", new_profiles=n)
                except TimeoutError:
                    log.warning("engine_v2_profile_scan_timeout", timeout_s=30)
                except Exception:
                    log.warning("engine_v2_profile_scan_failed", exc_info=True)

            # Force model map refresh so discovered GGUFs appear immediately
            app.state.provider_registry.invalidate_model_map()

            app.state.llama_manager = llama_manager
            app.state.token_count_cache = token_cache

            # Boot-time reconcile: if a llama-server from a prior worker
            # is still alive on our backend port (uvicorn worker swap,
            # crash without proper teardown, etc.) it's silently hoarding
            # VRAM with no manager tracking it. The user would never
            # notice until they tried to load a different model and
            # watched VRAM "clear and refill" mid-send. Reclaim now so
            # idle VRAM is actually idle. No-op when the port is free.
            try:
                reclaimed = await llama_manager.reconcile_stranded_subprocess()
                if reclaimed:
                    log.warning(
                        "engine_v2_boot_reconcile_reclaimed",
                        port=settings.engine_backend_port,
                        note="freed a stranded llama-server subprocess at startup",
                    )
            except Exception:
                # Reconcile is best-effort and must never block startup.
                log.warning("engine_v2_boot_reconcile_failed", exc_info=True)

            log.info("engine_v2_initialized", port=settings.engine_backend_port)
        except Exception:
            log.error("engine_v2_init_failed", exc_info=True)
            # Server continues without managed engine — other backends still work

    # Vision substrate initialization is intentionally deferred until
    # AFTER `_restore_settings` runs (further down in lifespan), so the
    # toggle is sticky across container restarts. Pydantic defaults
    # would otherwise win at this point in boot and the sibling would
    # never auto-start even when the DB has vision_provider_enabled=True.

    app.state.classifier = RequestClassifier()

    # Session lifecycle — coordinates KV cache persistence across modes
    from augmentum.session.lifecycle import SessionLifecycle
    app.state.session_lifecycle = SessionLifecycle(
        state_manager=getattr(app.state, "state_manager", None),
        provider_registry=app.state.provider_registry,
    )

    # Initialize service health registry for graceful degradation
    from augmentum.utils.service_health import ServiceHealthRegistry

    health = ServiceHealthRegistry()
    app.state.service_health = health

    # Register the self-edit Application Health Signal's runtime probes (they read
    # live telemetry — backends, services, DB integrity, strain — lazily at
    # assess time, so registering here is safe regardless of init order).
    try:
        from augmentum.selfedit.probes import register_runtime_probes
        register_runtime_probes(app.state)
    except Exception as exc:  # noqa: BLE001 — health wiring must never break boot
        log.warning("selfedit_health_probe_registration_failed", error=repr(exc))

    async def _check_searxng() -> bool:
        try:
            resp = await app.state.http_client.get(f"{settings.searxng_base_url}/")
            return resp.status_code == 200
        except Exception:
            return False

    async def _check_executor() -> bool:
        try:
            resp = await app.state.http_client.get(
                f"{settings.executor_base_url}/health",
            )
            return resp.status_code == 200
        except Exception:
            return False

    health.register("searxng", check_fn=_check_searxng)
    health.register("executor", check_fn=_check_executor)
    health.register("llm_backend")   # Marked by provider registry
    health.register("image_gen")     # Marked by image pipeline
    health.register("tts")           # Marked by TTS providers
    health.register("stt")           # Marked by STT providers
    app.state.narrative_engines = OrderedDict()  # OrderedDict[str, NarrativeEngine] — lazy per session, LRU eviction
    app.state.narrative_handlers = OrderedDict()  # OrderedDict[str, NarrativeHandler] — cached for group chat state
    app.state.agentic_handlers = OrderedDict()    # OrderedDict[(user_id, session_id), AgenticHandler] — cached so task working memory persists

    # Health / strain monitor counters (read by the strain sampler loop).
    # inflight_requests + active_clients are maintained by the in-flight
    # middleware; last_event_loop_lag_s by the lag monitor; slow_request_count
    # by the slow-request middleware (read-and-reset each sample).
    app.state.inflight_requests = 0
    app.state.active_clients = {}   # {client_id: (last_seen_monotonic, user_id)}
    app.state.last_event_loop_lag_s = 0.0
    app.state.slow_request_count = 0
    app.state.strain_monitor = None  # set during lifespan startup when SQLite-backed

    # Initialize tool registry
    app.state.tool_registry = _build_tool_registry(app.state.http_client)

    # context_peek — the perception contract's pull door: full detail
    # behind the index/digest tiers (open page text, full note, play
    # position, recent results). Needs app.state for the referent
    # caches + notes store, so it registers here rather than in
    # _build_tool_registry.
    from augmentum.tools.context_peek import ContextPeekTool
    app.state.tool_registry.register(ContextPeekTool(app.state))

    # media_recommendations — Gate 1 of the consumption-entity ladder
    # (catalog-grounded "what next" picks). Needs app.state for the
    # state-manager connection, same as context_peek.
    from augmentum.tools.media_recommendations import MediaRecommendationsTool
    app.state.tool_registry.register(MediaRecommendationsTool(app.state))

    # consistency_check — logical validation of statements using an LLM.
    # Needs a ModelBackend, available from the provider registry.
    from augmentum.tools.consistency_check import ConsistencyCheckTool
    app.state.tool_registry.register(
        ConsistencyCheckTool(backend=app.state.provider_registry.default_backend)
    )

    # Narrative lorebook grounding verbs (F1/F5): lorebook.check /
    # lorebook.create. Registered so they EXIST in the catalog and so a
    # trained model's calls resolve to real handlers. Need app.state to
    # reach the live per-session NarrativeEngine (narrative_engines), same
    # as context_peek. The live narrative-mode path dispatches them through
    # the recall loop; these registry objects share that one implementation.
    from augmentum.tools.lorebook_tools import (
        LorebookCheckTool,
        LorebookCreateTool,
    )
    app.state.tool_registry.register(LorebookCheckTool(app.state))
    app.state.tool_registry.register(LorebookCreateTool(app.state))

    # Expose Action primitives (note.create, memory.save, etc.) as
    # tools so the LLM can invoke them via function-calling during a
    # passthrough turn. Direct user phrasing already reaches them via
    # the Tier 1 matcher; this layer adds composition — the model can
    # decide on its own to call note.create + note.show_sticky.
    from augmentum.intent import register_action_tools
    register_action_tools(app.state.tool_registry, app.state)

    # Unified primitive layer (Phase 1): voice manifest derives its
    # tool list from the registry on each lookup. Bind once; the
    # manifest reads ``Tool.surfaces.voice`` at access time, so tools
    # registered later in lifespan are still picked up.
    from augmentum.intent.manifest import bind_registry as _bind_voice_registry
    _bind_voice_registry(app.state.tool_registry)

    # Coder permission registry — backs the permission-approval modal
    # for AUGMENTUM_CODER_PERMISSIONS=confirm_mutations. The audit sink
    # resolves its DB conn lazily per write: app.state.state_manager is
    # wired LATER in create_app, so capturing it here would bind None.
    from augmentum.coder.permission_audit import (
        resolve_store as _resolve_permission_audit_store,
    )
    from augmentum.coder.permissions import PermissionRegistry

    async def _permission_audit_sink(**kwargs) -> None:
        store = _resolve_permission_audit_store(app.state)
        if store is None:
            log.warning("coder.permission_audit_no_store")
            return
        await store.record(**kwargs)

    app.state.permission_registry = PermissionRegistry(
        audit_sink=_permission_audit_sink,
    )

    # Coder review registry — reviewable-turn flow (see
    # augmentum/coder/reviews.py). Handler publishes bundles at turn
    # end; the frontend review panel polls / fetches via the
    # /api/coder/reviews/ routes.
    from augmentum.coder.reviews import ReviewRegistry
    app.state.review_registry = ReviewRegistry()

    # Coder run broker — detaches the agent loop from the HTTP request
    # so a dropped fetch (mobile screen sleep, tab switch) doesn't
    # kill the run. Subscribe/cancel/active-run routes live in
    # coder_routes.py. Orphan sweep runs once at boot to clean up
    # ``status='running'`` rows left behind by the previous process.
    from augmentum.coder.run_broker import (
        CoderRunBroker,
        sweep_orphan_running_runs,
    )
    app.state.coder_run_broker = CoderRunBroker()
    app.state.coder_run_broker.start_sweeper()
    try:
        sm = getattr(app.state, "state_manager", None)
        conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
        swept = await sweep_orphan_running_runs(conn)
        if swept:
            log.info("coder_orphan_runs_swept", count=swept)
    except Exception:
        log.warning("coder_orphan_sweep_at_boot_failed", exc_info=True)

    # Paused-timeout sweep — auto-cancel runs that have been paused
    # longer than ``coder_max_paused_seconds``. Prevents a forgotten
    # pause from holding broker memory + container resources forever.
    # Tunable interval; 0 disables the loop. The handler's
    # CancelledError path emits the cancel reason ("paused_timeout")
    # into the next turn's <prior_turns> so the model sees why the
    # turn was abandoned. See [[coder-paused-timeout]] for design.
    async def _paused_timeout_sweeper() -> None:
        broker = app.state.coder_run_broker
        if broker is None:
            return
        interval = max(15, int(getattr(settings, "coder_paused_sweep_interval_s", 60) or 60))
        log.info("coder_paused_sweeper_started", interval_s=interval)
        while True:
            try:
                await asyncio.sleep(interval)
                max_s = float(getattr(settings, "coder_max_paused_seconds", 1800) or 0)
                if max_s <= 0:
                    continue  # disabled — keep loop running so live re-enable picks up
                cancelled = broker.sweep_paused_timeouts(max_paused_seconds=max_s)
                if cancelled:
                    log.info("coder_paused_timeout_swept", count=cancelled)
            except asyncio.CancelledError:
                log.info("coder_paused_sweeper_cancelled")
                return
            except Exception:
                log.warning("coder_paused_sweeper_iter_failed", exc_info=True)

    _interval = int(getattr(settings, "coder_paused_sweep_interval_s", 60) or 0)
    if _interval > 0:
        _track_bg(_paused_timeout_sweeper(), name="coder_paused_timeout_sweeper")

    # Subagent dispatch (task_dispatch tool) is wired AFTER the
    # state_manager is attached to app.state (search
    # "coder_subagent_dispatch_initialized" below) — the SubagentRunStore
    # needs the live aiosqlite connection, which doesn't exist yet here.
    # Wiring it at this point silently produced store=None (the audit log
    # never persisted a single row). Default to None so any early reader
    # sees a defined attribute.
    app.state.coder_subagent_store = None
    app.state.coder_agent_registry = None

    # Initialize tool circuit breaker (shared across requests)
    if settings.tool_circuit_breaker_enabled:
        from augmentum.tools.circuit_breaker import ToolCircuitBreaker
        app.state.circuit_breaker = ToolCircuitBreaker(
            threshold=settings.tool_circuit_breaker_threshold,
            cooldown=settings.tool_circuit_breaker_cooldown,
        )
    else:
        app.state.circuit_breaker = None

    # Initialize prompt cache, prefix cache, and request deduplicator
    from augmentum.cache.dedup import RequestDeduplicator
    from augmentum.cache.prefix_cache import PrefixCache
    from augmentum.cache.prompt_cache import PromptCache

    app.state.prompt_cache = PromptCache(
        max_size=settings.prompt_cache_max_entries,
        ttl_seconds=settings.prompt_cache_ttl,
    )
    app.state.prefix_cache = PrefixCache()
    app.state.request_deduplicator = RequestDeduplicator()

    # Initialize model manager (cross-backend lifecycle)
    from augmentum.models.model_manager import ModelManager

    app.state.model_manager = ModelManager(app.state.provider_registry)

    # Initialize SQLite state backend
    db_path = f"{settings.data_dir}/augmentum.db"
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    sqlite_backend = SQLiteBackend(db_path)

    # Retry connection — Docker volumes may have stale WAL locks from
    # unclean shutdown or slow NFS/bind mounts.  A brief retry usually
    # recovers without falling back to lossy in-memory mode.
    _max_retries = 3
    _sqlite_ok = False
    for _attempt in range(1, _max_retries + 1):
        try:
            await sqlite_backend.connect()
            _sqlite_ok = True
            # Deferred corruption probe (2026-07-02 boot-latency work):
            # connect() skips the inline PRAGMA quick_check above the
            # size threshold (measured 40.5s on a 1.5GB DB — all of it
            # gating first paint). Run it here as a background task a
            # few seconds after boot I/O settles. Failure touches the
            # recovery stamp + logs at error (never raises) — same
            # operator surface as the inline path, minus the 40s toll
            # on every healthy boot.
            if getattr(sqlite_backend, "deferred_quick_check", False):
                async def _deferred_quick_check(_be=sqlite_backend) -> None:
                    await asyncio.sleep(10)
                    try:
                        await _be.run_quick_check(deferred=True)
                    except Exception:
                        log.error("deferred_quick_check_crashed", exc_info=True)
                _track_bg(_deferred_quick_check(), name="deferred_db_quick_check")
            break
        except Exception:
            if _attempt < _max_retries:
                log.warning(
                    "sqlite_connect_retry",
                    attempt=_attempt,
                    max=_max_retries,
                    path=db_path,
                    exc_info=True,
                )
                await asyncio.sleep(2)
            else:
                # Diagnostics: log file existence and permissions for troubleshooting
                db_file = Path(db_path)
                diag = {
                    "exists": db_file.exists(),
                    "parent_exists": db_file.parent.exists(),
                    "parent_writable": os.access(str(db_file.parent), os.W_OK),
                }
                if db_file.exists():
                    try:
                        diag["size"] = db_file.stat().st_size
                        diag["mode"] = oct(db_file.stat().st_mode)
                    except OSError:
                        # Diagnostic-only; missing/unreadable here just
                        # means we'll log without size/mode fields.
                        pass
                    # Check for stale WAL/SHM that might be locking
                    for ext in ("-wal", "-shm"):
                        wal = Path(db_path + ext)
                        if wal.exists():
                            diag[f"{ext[1:]}_exists"] = True
                            diag[f"{ext[1:]}_size"] = wal.stat().st_size
                log.error(
                    "sqlite_connect_failed",
                    path=db_path,
                    diagnostics=diag,
                    exc_info=True,
                    hint="Check volume permissions (chown 1000:1000 /data), "
                         "the most recent migration for syntax errors / typo'd "
                         "table names, and stale -wal/-shm from an unclean "
                         "shutdown. If you absolutely need to keep serving "
                         "without persistence (NOT recommended; auth + all "
                         "user data go to memory), set the "
                         "AUGMENTUM_ALLOW_INMEMORY_FALLBACK env var or the "
                         "allow_inmemory_fallback setting to true.",
                )

    if not _sqlite_ok:
        # Decision point: refuse to start, or fall through to the
        # in-memory backend? Refusing is the right default — the file
        # likely exists on disk (a typo'd migration, stale WAL, etc.),
        # and falling back makes it APPEAR to the user that all their
        # data was nuked when really we just couldn't open it. A loud
        # crash forces the operator to fix the underlying issue.
        db_file = Path(db_path)
        db_present = db_file.exists() and db_file.stat().st_size > 4096
        allow_fallback = bool(getattr(settings, "allow_inmemory_fallback", False))
        if db_present and not allow_fallback:
            raise RuntimeError(
                f"SQLite database at {db_path} exists ({db_file.stat().st_size} "
                f"bytes) but failed to open after {_max_retries} attempts. "
                f"Refusing to fall back to the in-memory backend — your data is "
                f"on disk and the fallback would silently present the install "
                f"as empty (auth fails, every endpoint 503s with "
                f"`auth_unavailable_denied`). Fix the underlying cause (see "
                f"the preceding `sqlite_connect_failed` log) and restart. "
                f"To override (NOT recommended; ephemeral / setup-only), set "
                f"AUGMENTUM_ALLOW_INMEMORY_FALLBACK=1."
            )

    if _sqlite_ok:
        app.state.state_manager = StateManager(sqlite_backend)

        # Subagent dispatch (task_dispatch tool) — single SubagentRunStore
        # bound to the now-connected aiosqlite connection so audit rows
        # survive process restart. MUST run after the state_manager is
        # attached (above): wiring it earlier resolved conn=None and the
        # audit log silently never persisted. AgentRegistry seeded with
        # built-in roles; user-defined roles in .augmentum/agents/*.md are
        # discovered lazily on first ``task_dispatch`` per workspace.
        try:
            from augmentum.agents.persistence import SubagentRunStore
            from augmentum.agents.presets import BUILTIN_ROLES
            from augmentum.agents.registry import AgentRegistry

            conn = getattr(sqlite_backend, "conn", None)
            app.state.coder_subagent_store = (
                SubagentRunStore(conn) if conn is not None else None
            )
            app.state.coder_agent_registry = AgentRegistry(builtins=BUILTIN_ROLES)
            log.info(
                "coder_subagent_dispatch_initialized",
                builtin_roles=list(BUILTIN_ROLES.keys()),
                store=app.state.coder_subagent_store is not None,
            )
        except Exception:
            app.state.coder_subagent_store = None
            app.state.coder_agent_registry = None
            log.warning("coder_subagent_dispatch_init_failed", exc_info=True)

        # Backup database on startup and rotate old backups.
        # Gated: skip when a fresh backup already exists within the
        # interval window (default 1 hour). VACUUM INTO holds an
        # exclusive write lock for several seconds and back-to-back
        # restarts were causing companion-state writes + auth's
        # failed-attempt inserts to pile up behind the lock and
        # surface as "database is locked" errors. One backup per
        # hour is plenty; missing one bounce is cheap.
        from augmentum.state.backup import (
            backup_database,
            rotate_backups,
            should_skip_startup_backup,
        )

        if should_skip_startup_backup(db_path):
            log.info("backup_skipped_recent", db_path=db_path,
                     hint="newest backup is within the interval window")
        else:
            backup_result = await backup_database(sqlite_backend.conn, db_path)
            if backup_result:
                rotate_backups(Path(backup_result).parent)

        # Load runtime-configured providers from database
        from augmentum.state.provider_store import ProviderStore

        provider_store = ProviderStore(sqlite_backend.conn)
        app.state.provider_store = provider_store
        await app.state.provider_registry.load_runtime_providers(provider_store)
        # Provider URLs cached for discovery filtering
        await app.state.provider_registry.populate_provider_urls(provider_store)

        # Load load balancers
        from augmentum.models.load_balancer import LoadBalancer, LoadBalancerRegistry
        from augmentum.state.balancer_store import BalancerStore

        balancer_store = BalancerStore(sqlite_backend.conn)
        lb_registry = LoadBalancerRegistry()
        balancers = await balancer_store.list_balancers()
        for b in balancers:
            if b.enabled:
                members = await balancer_store.list_members(b.id)
                lb_registry.register(b.id, LoadBalancer(b, members))
        app.state.balancer_store = balancer_store
        app.state.lb_registry = lb_registry
        app.state.provider_registry.set_lb_registry(lb_registry)
        log.info("load_balancers_initialized", count=len(balancers))

        # Encrypt any plaintext API keys still in the database (one-time migration)
        await _migrate_plaintext_keys(sqlite_backend.conn)

        # Initialize persistent settings store and restore saved tokens
        from augmentum.state.settings_store import SettingsStore

        settings_store = SettingsStore(sqlite_backend.conn)
        app.state.settings_store = settings_store

        # Register the self-edit reshape surfaces that have a real actuator+oracle
        # today (the config/Adaptation surface → per-user SettingsStore). This is
        # the documented one-line wire-up from selfedit/surfaces/live.py; the
        # reshape route + engine stay no-ops until a surface is registered here.
        try:
            from augmentum.selfedit.surfaces.live import register_default_surfaces
            app.state.selfedit_reshape_ledger = {}
            register_default_surfaces(settings_store,
                                      revert_ledger=app.state.selfedit_reshape_ledger)
        except Exception as exc:  # noqa: BLE001 — reshape wiring must never break boot
            log.warning("selfedit_reshape_surface_registration_failed", error=repr(exc))

        # Self-edit EDIT DRIVER — the agent that actually changes code. The native
        # engine drives a LOCAL model via Augmentum's OWN agentic loop (sovereign,
        # no token). Wired only when a repo with a WRITABLE .git is reachable (the
        # dev-bind mounts it at /host-augmentum-src) — the candidate worktree needs
        # rw .git. Without it the debt loop stays a SAFE dry-run. Lazy: the model is
        # resolved per-run, so registering here costs nothing and never blocks boot.
        # Skipped entirely unless the operator unlocked the subsystem
        # (AUGMENTUM_SELFEDIT_UNLOCK) — a locked install must not so much as
        # clone its own repo into /data. See config.selfedit_unlocked().
        try:
            import os as _se_os

            from augmentum.config import selfedit_unlocked as _se_unlocked

            se_src = _se_os.environ.get("AUGMENTUM_SELFEDIT_REPO", "/host-augmentum-src")
            if _se_unlocked() and _se_os.path.isdir(_se_os.path.join(se_src, ".git")):
                from augmentum.selfedit.candidate import prepare_writable_repo
                from augmentum.selfedit.engine_select import (
                    DEFAULT_ENGINE,
                    wire_selfedit_driver,
                )
                from augmentum.selfedit.growth_db import get_growth_conn

                # The source mount is read-only in dev-bind; get a repo with a
                # writable .git (a /data --shared clone) so worktrees can be made.
                se_repo = await prepare_writable_repo(se_src, "/data/selfedit/repo")
                # The source is a full checkout at HEAD — the baseline tree for
                # evidence grounding (the --shared clone is --no-checkout, no files).
                app.state.selfedit_source_dir = se_src
                se_engine = (await settings_store.get("selfedit_engine")) or DEFAULT_ENGINE
                se_model = (await settings_store.get("selfedit_edit_model")) or ""
                # Step budget for the native loop (bigger workspace → more steps to
                # locate + edit). Tunable without a code change; default 64.
                try:
                    se_iters = int((await settings_store.get("selfedit_max_iters")) or 64)
                except (TypeError, ValueError):
                    se_iters = 64
                se_iters = max(8, min(se_iters, 200))
                se_conn = await get_growth_conn(app.state)
                se_wired = await wire_selfedit_driver(
                    app.state, se_conn, engine=se_engine, repo_dir=se_repo,
                    registry=app.state.provider_registry, model=se_model,
                    native_role="utility", max_iters=se_iters)
                # Sync the live baseline to the clone HEAD at boot (the clone was
                # just reset to the live source HEAD), so "pending changes" reflects
                # only this session's staged self-edits — not normal host commits
                # landed since the last boot.
                with contextlib.suppress(Exception):
                    from augmentum.selfedit.apply import sync_baseline_to_head
                    await sync_baseline_to_head(se_repo)
                # L2 boot parachute: we reached healthy startup, so reset the
                # entrypoint's boot-attempt counter (the entrypoint increments it
                # each boot; a stuck counter = a crash loop → the parachute fires).
                with contextlib.suppress(Exception):
                    from augmentum.selfedit import rollback as _se_rb
                    _se_rb.mark_boot_healthy(
                        _se_os.path.dirname(_se_os.path.dirname(se_repo)))
                log.info("selfedit_driver_startup", wired=se_wired, engine=se_engine,
                         model=(se_model or "(utility role)"), repo=se_repo)
            else:
                log.info("selfedit_driver_skipped", reason="no writable .git",
                         repo=se_repo)
        except Exception as exc:  # noqa: BLE001 — self-edit driver wiring must never break boot
            log.warning("selfedit_driver_wiring_failed", error=repr(exc))

        # WebXR / Quest-style app sessions. The browser owns WebXR APIs;
        # this store gives immersive sessions durable room, seat, resume,
        # and telemetry state across reloads or headset reconnects.
        from augmentum.xr.session import XRSessionStore

        app.state.xr_store = XRSessionStore(sqlite_backend.conn)

        # Now that the store exists, hand it to the llama-server manager so
        # the lazy-load path can read the per-model "last load" defaults
        # written by the engine load route. Skipped when the engine wasn't
        # constructed (engine_enabled=False).
        if getattr(app.state, "llama_manager", None) is not None:
            app.state.llama_manager.set_settings_store(settings_store)

        # Restore persisted tokens into runtime config — values are stored
        # encrypted (config_routes._STRING_SETTINGS sensitivity heuristic) so
        # decrypt before assigning, otherwise downstream Bearer headers ship
        # the literal `enc:gAAAA...` payload and HF/CivitAI auth fails.
        from augmentum.utils.secrets import decrypt_api_key as _decrypt_at_rest
        hf_token = await settings_store.get("image_huggingface_token")
        if hf_token:
            hf_token = _decrypt_at_rest(hf_token)
        if hf_token:
            object.__setattr__(settings, "image_huggingface_token", hf_token)
            log.info("hf_token_restored_from_db")
        # Also set HF_TOKEN env var so huggingface_hub / diffusers from_pretrained
        # and from_single_file calls auto-authenticate (e.g. GGUF pipeline step 2).
        _effective_hf = hf_token or settings.image_huggingface_token
        if _effective_hf:
            os.environ.setdefault("HF_TOKEN", _effective_hf)
        # Clear sensitive tokens from local scope so they don't leak in tracebacks
        del hf_token, _effective_hf
        civitai_key = await settings_store.get("image_civitai_api_key")
        if civitai_key:
            civitai_key = _decrypt_at_rest(civitai_key)
        if civitai_key:
            object.__setattr__(settings, "image_civitai_api_key", civitai_key)
            log.info("civitai_key_restored_from_db")
        del civitai_key

        # Restore system settings
        tz_val = await settings_store.get("timezone")
        if tz_val:
            object.__setattr__(settings, "timezone", tz_val)
            log.info("timezone_restored_from_db", timezone=tz_val)

        # Restore all persisted config overrides in one pass
        # Pass None so the production path auto-derives from
        # config_routes._TOOL_SETTINGS + _STRING_SETTINGS. Eliminates
        # the "add setting → forget restore-map entry → value resets
        # on restart" footgun documented in
        # _auto_derive_restore_parsers.
        await _restore_settings(settings_store)

        # Re-sync engine settings onto the manager: the manager was
        # constructed earlier in the lifespan (above) using the
        # codebase defaults from ``settings``, before persisted
        # overrides were loaded. Without this re-sync, DB-stored
        # engine_* values (e.g. a user-bumped health_timeout) take
        # effect only after a live PUT /api/config/tools — they
        # silently revert to the codebase default on every restart.
        _llama_mgr = getattr(app.state, "llama_manager", None)
        if _llama_mgr is not None:
            _llama_mgr.kv_cache_type = settings.engine_kv_cache_type
            _llama_mgr.draft_model = settings.engine_draft_model
            _llama_mgr.draft_max = settings.engine_draft_max
            _llama_mgr.draft_ctx_size = settings.engine_draft_ctx_size
            _llama_mgr.draft_gpu_layers = settings.engine_draft_gpu_layers
            _llama_mgr.draft_min = settings.engine_draft_min
            _llama_mgr.draft_p_min = settings.engine_draft_p_min
            _llama_mgr.mtp_enabled = settings.engine_mtp_enabled
            _llama_mgr.mtp_n_max = settings.engine_mtp_n_max
            _llama_mgr.flash_attn = settings.engine_flash_attn
            _llama_mgr.idle_timeout = settings.engine_idle_timeout
            _llama_mgr.health_timeout = settings.engine_health_timeout

        # ── Vision provider substrate (classifier slot IS the captioner) ──────
        # MUST run after `_restore_settings`. Vision is a CAPABILITY of the
        # classifier slot: when its model is VL+mmproj (managed Slot C with
        # Gemma, or an external multimodal classifier), captioning routes
        # there. PrimaryVisionProvider still serves a VL primary directly.
        # SmolVLM is retired to a CPU-ONLY FALLBACK that LAZILY starts only
        # when no VL classifier/primary is available (the no-GPU tier) — so it
        # costs nothing on GPU boxes. ``vision_provider_enabled`` now means
        # "allow the CPU vision fallback" (default True).
        app.state.vision_router = None
        app.state.vision_sibling = None
        try:
            from augmentum.vision import (
                ClassifierVisionProvider,
                PrimaryVisionProvider,
                SmolVLMConfig,
                SmolVLMProvider,
                SmolVLMSibling,
                VisionRouter,
            )

            primary_vision = PrimaryVisionProvider(app.state)

            # CPU fallback — construct (cheap) but DO NOT start the subprocess.
            # The router selects it only when neither the primary nor the
            # classifier can serve vision; caption() lazily cold-starts it then.
            smolvlm_sibling: SmolVLMSibling | None = None
            smolvlm_provider: SmolVLMProvider | None = None
            if settings.vision_provider_enabled and settings.vision_provider_model_path:
                vision_cfg = SmolVLMConfig(
                    base_model_path=settings.vision_provider_model_path,
                    mmproj_path=settings.vision_provider_mmproj_path,
                    backend_port=settings.vision_provider_backend_port,
                    gpu_layers=0,  # the fallback is CPU-by-definition
                )
                smolvlm_sibling = SmolVLMSibling(vision_cfg)
                smolvlm_provider = SmolVLMProvider(
                    smolvlm_sibling, app.state.http_client,
                )

            # Classifier captioner: external Docker classifier OR the managed
            # Slot C (deterministic loopback URL from its port). Availability is
            # gated LIVE on the slot's vision capability so a text-only
            # classifier doesn't claim the captioner role (router falls back).
            external_classifier_url = (
                getattr(settings, "classifier_base_url", "")
                or os.environ.get("AUGMENTUM_CLASSIFIER_BASE_URL", "")
            )
            classifier_url = external_classifier_url
            if not classifier_url and settings.classifier_slot_enabled:
                classifier_url = (
                    f"http://127.0.0.1:{settings.classifier_slot_backend_port}"
                )

            classifier_vision_provider: ClassifierVisionProvider | None = None
            if classifier_url:
                def _classifier_vision_capable() -> bool:
                    # Gate on the managed slot ONLY when the managed slot is
                    # what we're actually calling. With an external Docker
                    # classifier the slot yields the backend key but its object
                    # still lands on app.state (see
                    # ``classifier_slot_yielding_to_external``), never loads a
                    # model, and so reports is_vision_capable()==False forever.
                    # Gating on it there disabled the captioner on EVERY
                    # external-classifier install even when that container was
                    # serving a VL model with its mmproj.
                    if external_classifier_url:
                        return True
                    slot = getattr(app.state, "classifier_slot", None)
                    if slot is not None:
                        return slot.is_vision_capable()
                    return True

                classifier_vision_provider = ClassifierVisionProvider(
                    classifier_url, app.state.http_client,
                    capability_fn=_classifier_vision_capable,
                )

            app.state.vision_router = VisionRouter(
                primary=primary_vision,
                smolvlm=smolvlm_provider,
                classifier=classifier_vision_provider,
            )
            app.state.vision_sibling = smolvlm_sibling
            log.info(
                "vision_router_initialized",
                cpu_fallback_allowed=bool(smolvlm_provider),
                classifier_captioner=bool(classifier_vision_provider),
                classifier_url=classifier_url or None,
            )
        except Exception:
            log.warning("vision_router_init_failed", exc_info=True)

        # Classifier sibling — a third llama-server subprocess hosting
        # a small text model (1-3B class) dedicated to fast classification
        # and utility tasks. Shields voice + utility paths from whatever
        # heavy reasoning model the user has selected for chat.
        #
        # Lifecycle gate: starts when EITHER classifier_engine_enabled
        # is True (manual override) OR companion_activation_mode is
        # "always_listening" (the only voice mode that genuinely
        # benefits from the latency floor — PTT and wake-word are
        # explicit so a slow classifier hop is acceptable there).
        #
        # Provider-registry hookup happens AFTER the sibling reaches
        # READY state — see app.state.classifier_sibling_register_task
        # below. resolve_model_for_role("classifier"/"utility") picks
        # it up automatically once the model_map refresh sees it.
        try:
            from augmentum.architect.classifier_sibling import (
                ClassifierConfig,
                ClassifierSibling,
            )

            activation_mode = (settings.companion_activation_mode or "").lower()
            classifier_should_start = (
                bool(settings.classifier_engine_enabled)
                or activation_mode == "always_listening"
            ) and not settings.classifier_slot_enabled  # Slot C supersedes the start-only sibling (same port 8093)
            classifier_sibling: ClassifierSibling | None = None
            if classifier_should_start and settings.classifier_engine_model_path:
                classifier_cfg = ClassifierConfig(
                    model_path=settings.classifier_engine_model_path,
                    backend_port=settings.classifier_engine_backend_port,
                    gpu_layers=settings.classifier_engine_gpu_layers,
                    ctx_size=settings.classifier_engine_ctx_size,
                )
                classifier_sibling = ClassifierSibling(classifier_cfg)

                async def _start_classifier_and_register() -> None:
                    """Boot the sibling, then register it as a backend
                    in the provider registry so resolve_model_for_role
                    can find it. Done in a background task so lifespan
                    isn't blocked by the model load (~1-2s on CPU).
                    """
                    ok = await classifier_sibling.start()
                    if not ok:
                        return
                    try:
                        from augmentum.models.llama_cpp import LlamaCppBackend
                        registry = app.state.provider_registry
                        base_url = classifier_sibling.base_url
                        if not base_url:
                            log.warning(
                                "classifier_sibling_no_base_url_after_start",
                            )
                            return
                        # Register under the "classifier" key that
                        # resolve_model_for_role actually checks (it does
                        # ``_backends.get("classifier")``). The historical
                        # "classifier_sibling" key was never consulted, so the
                        # in-process sibling silently never resolved and the
                        # classifier role fell through to primary_chat_model.
                        # Don't clobber an explicit Docker sidecar: compose.
                        # classifier.yaml registers this same key from
                        # AUGMENTUM_CLASSIFIER_BASE_URL at startup. The
                        # dedicated sidecar wins; the sibling is the
                        # no-container fallback.
                        if registry._backends.get("classifier") is not None:
                            log.info("classifier_sibling_yielding_to_sidecar")
                            return
                        registry._backends["classifier"] = LlamaCppBackend(
                            registry._http_client, base_url, "",
                        )
                        # Refresh model_map so the sibling's hosted
                        # model id becomes resolvable. The role
                        # resolver in resolve_model_for_role queries
                        # _backends + _model_map; both need the entry.
                        # ``force=True`` bypasses the TTL cache so the
                        # newly-registered backend gets probed now,
                        # not on whatever the next stale-TTL window is.
                        await registry.refresh_model_map(force=True)
                        log.info(
                            "classifier_sibling_registered",
                            base_url=base_url,
                        )
                    except Exception:  # noqa: BLE001
                        log.warning(
                            "classifier_sibling_registration_failed",
                            exc_info=True,
                        )

                asyncio.create_task(_start_classifier_and_register())
            elif classifier_should_start:
                log.info(
                    "classifier_sibling_skipped_no_model_path",
                    activation_mode=activation_mode,
                    enabled=settings.classifier_engine_enabled,
                )

            app.state.classifier_sibling = classifier_sibling
            log.info(
                "classifier_sibling_lifespan",
                will_start=bool(classifier_sibling),
                activation_mode=activation_mode,
                manual_enabled=settings.classifier_engine_enabled,
            )
        except Exception:
            log.warning("classifier_sibling_init_failed", exc_info=True)

        # Secondary local engine ("Slot B") — a second user-driven resident
        # llama-server so two arbitrary local models stay loaded at once.
        # Construct (cheap — no subprocess until a model is loaded) and
        # register its backend so routing can reach it via its pin. The
        # subprocess only spins up on first load/use, so an enabled-but-
        # empty slot costs nothing. Re-pin the last-loaded model from
        # settings so its picker entry routes to the slot after a restart;
        # the model itself lazy-loads on first request, same as the primary.
        try:
            if settings.engine_secondary_enabled:
                from augmentum.models.secondary_slot import (
                    SECONDARY_BACKEND_KEY,
                    SecondarySlot,
                    SecondarySlotConfig,
                )

                primary_mgr = getattr(app.state, "llama_manager", None)
                if primary_mgr is None:
                    log.warning("engine_secondary_skipped_no_primary_engine")
                else:
                    slot = SecondarySlot(
                        SecondarySlotConfig(
                            backend_port=settings.engine_secondary_backend_port,
                            model_dirs=list(primary_mgr.model_dirs),
                            llama_server_path=_engine_binary,
                        ),
                        http_client=app.state.http_client,
                    )
                    slot.set_token_cache(
                        getattr(app.state, "token_count_cache", None),
                    )
                    slot.set_settings_store(
                        getattr(app.state, "settings_store", None),
                    )
                    registry = app.state.provider_registry
                    # Register the backend but keep it OUT of catalog
                    # probing — it shares the primary's GGUF dirs, so
                    # advertising its catalog would collide on every name.
                    registry.register_backend(SECONDARY_BACKEND_KEY, slot.backend)
                    registry.exclude_backend_from_map(SECONDARY_BACKEND_KEY)
                    # Boot-time reconcile: a llama-server from a prior worker
                    # may still be alive on Slot B's port (crash without
                    # teardown, worker swap), silently hoarding VRAM with no
                    # manager tracking it. Reclaim now so idle VRAM is really
                    # idle — parity with the primary engine's boot reconcile.
                    # ``slot.backend`` above already built the manager.
                    try:
                        if slot.manager is not None:
                            reclaimed = await slot.manager.reconcile_stranded_subprocess()
                            if reclaimed:
                                log.warning(
                                    "engine_secondary_boot_reconcile_reclaimed",
                                    port=settings.engine_secondary_backend_port,
                                    note="freed a stranded Slot B llama-server at startup",
                                )
                    except Exception:
                        log.warning("engine_secondary_boot_reconcile_failed", exc_info=True)
                    # Re-pin the last model the user had in Slot B so chat
                    # routes to it after a restart (lazy-loads on first use).
                    last_model = (settings.engine_secondary_model or "").strip()
                    if last_model:
                        registry.pin_model(last_model, SECONDARY_BACKEND_KEY)
                    app.state.secondary_slot = slot
                    log.info(
                        "engine_secondary_initialized",
                        port=settings.engine_secondary_backend_port,
                        pinned_model=last_model or None,
                    )
        except Exception:
            log.warning("engine_secondary_init_failed", exc_info=True)

        # Managed classifier slot ("Slot C") — augmentum-managed, runtime-
        # switchable resident llama-server for the classifier/utility roles
        # (and vision when its model is VL+mmproj). Construct cheap (no
        # subprocess until load), register under the "classifier" backend key
        # that resolve_model_for_role already consults, then background-load
        # the configured model (resident; don't block lifespan on a CPU load).
        # Precedence: an EXTERNAL Docker classifier (AUGMENTUM_CLASSIFIER_BASE_URL)
        # registers this key at registry construction and WINS — Slot C yields,
        # so existing installs are untouched.
        try:
            _external_classifier_url = (
                getattr(settings, "classifier_base_url", "")
                or os.environ.get("AUGMENTUM_CLASSIFIER_BASE_URL", "")
            )
            if settings.classifier_slot_enabled or _external_classifier_url:
                from augmentum.models.classifier_slot import (
                    CLASSIFIER_BACKEND_KEY,
                    ClassifierSlot,
                    ClassifierSlotConfig,
                )

                registry = app.state.provider_registry
                # ALWAYS construct the slot (cheap — no subprocess until load),
                # even when the external Docker classifier holds the key. This
                # is what makes the classifier configurable from the model
                # manager on external-sidecar installs: the slot sits idle and
                # /api/engine/v2/classifier/load can take over the "classifier"
                # key on an explicit user action (take_over=true) — otherwise
                # the load route 404'd and the sidecar's env-frozen model/ctx/
                # mmproj were the only option.
                _yielding = registry._backends.get(CLASSIFIER_BACKEND_KEY) is not None
                primary_mgr = getattr(app.state, "llama_manager", None)
                model_dirs = list(primary_mgr.model_dirs) if primary_mgr else []
                slot = ClassifierSlot(
                    ClassifierSlotConfig(
                        backend_port=settings.classifier_slot_backend_port,
                        model_dirs=model_dirs,
                        llama_server_path=_engine_binary,
                    ),
                    http_client=app.state.http_client,
                )
                slot.set_token_cache(
                    getattr(app.state, "token_count_cache", None),
                )
                slot.set_settings_store(
                    getattr(app.state, "settings_store", None),
                )
                app.state.classifier_slot = slot
                if _yielding:
                    # External sidecar wins the key at boot — the slot stays
                    # idle (no registration, no boot-load) until the user
                    # explicitly takes over from the model manager.
                    log.info("classifier_slot_yielding_to_external")
                else:
                    # Register the backend now (built, idle). resolve_model_for_role
                    # reaches the classifier/utility roles through this key.
                    registry.register_backend(CLASSIFIER_BACKEND_KEY, slot.backend)
                    # Exclude from EVERY catalog listing — parity with Slot B.
                    # The slot's LlamaCppBackend shares model_dirs with the
                    # primary engine, so its list_models() advertises the FULL
                    # GGUF catalog. Left un-excluded, every model name collides
                    # into name@engine / name@classifier variants, the bare
                    # names vanish from the map, and the picker/library shows 0
                    # models per drive until a restart. Role resolution reaches
                    # this backend by key (resolve_model_for_role reads
                    # _backends["classifier"]) — but ONLY on the branch where
                    # ``classifier_model`` is blank. A NAMED classifier model
                    # resolves through the catalog map instead, so exclusion
                    # without a pin sends the role to the primary engine and
                    # thrashes Slot A. The pin below is the other half; see the
                    # long note on the load route in model_routes.py.
                    registry.exclude_backend_from_map(CLASSIFIER_BACKEND_KEY)
                    # Boot reconcile: reclaim a stranded llama-server on the slot's
                    # port (crash without teardown) — parity with Slot B / primary.
                    try:
                        if slot.manager is not None:
                            reclaimed = await slot.manager.reconcile_stranded_subprocess()
                            if reclaimed:
                                log.warning(
                                    "classifier_slot_boot_reconcile_reclaimed",
                                    port=settings.classifier_slot_backend_port,
                                )
                    except Exception:
                        log.warning("classifier_slot_boot_reconcile_failed", exc_info=True)

                    model = (settings.classifier_slot_model or "").strip()
                    if model:
                        # Pin the configured name up front — parity with Slot
                        # B's boot re-pin — so a named ``classifier_model``
                        # routes here from the first request rather than
                        # racing the async load below and landing on the
                        # primary engine (which would swap Slot A's model).
                        # Re-pinned with the REAL hosted id after the load, in
                        # case path resolution normalizes the name.
                        registry.pin_model(model, CLASSIFIER_BACKEND_KEY)

                        async def _load_classifier_slot() -> None:
                            try:
                                path = slot.resolve_model_path(model) or model
                                # Globals are DEFAULTS only — merge_saved lets
                                # the per-model profile (ctx/mmproj/kv the user
                                # set at last load) override them, so restart
                                # no longer silently reverts slot config.
                                load_opts: dict[str, object] = {"idle_timeout": 0}
                                if settings.classifier_slot_gpu_layers:
                                    load_opts["gpu_layers"] = settings.classifier_slot_gpu_layers
                                if settings.classifier_slot_ctx_size:
                                    load_opts["ctx_size"] = settings.classifier_slot_ctx_size
                                model_id = await slot.load(
                                    path, load_options=load_opts, merge_saved=True,
                                )
                                # Make the role resolver return the slot's real
                                # hosted id (the classifier/utility branches return
                                # ``classifier_sidecar_model``).
                                try:
                                    store = getattr(app.state, "settings_store", None)
                                    if store is not None and model_id:
                                        await store.set("classifier_sidecar_model", model_id)
                                        settings.classifier_sidecar_model = model_id
                                except Exception:  # noqa: BLE001
                                    log.warning("classifier_slot_id_sync_failed", exc_info=True)
                                # Re-pin under the real hosted id. The pre-load
                                # pin above used the configured name; if
                                # resolve_model_path normalized it, that pin is
                                # a dead entry and the real name is unpinned —
                                # which is exactly the case that thrashes the
                                # primary engine.
                                if model_id:
                                    if model_id != model:
                                        registry.unpin_model(model)
                                    registry.pin_model(model_id, CLASSIFIER_BACKEND_KEY)
                                await registry.refresh_model_map(force=True)
                                log.info(
                                    "classifier_slot_loaded",
                                    model=model_id,
                                    vision=slot.is_vision_capable(),
                                )
                            except Exception:  # noqa: BLE001 — fail-open; role falls to primary
                                log.warning("classifier_slot_load_failed", exc_info=True)

                        asyncio.create_task(_load_classifier_slot())
                    log.info(
                        "classifier_slot_initialized",
                        port=settings.classifier_slot_backend_port,
                        model=model or None,
                    )
        except Exception:
            log.warning("classifier_slot_init_failed", exc_info=True)

        # Propagate HuggingFace token to env var + image config for in-process downloads
        hf_token = settings.huggingface_token
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
            if not settings.image_huggingface_token:
                object.__setattr__(settings, "image_huggingface_token", hf_token)
            log.info("huggingface_token_restored")

        # Restore custom image presets
        custom_presets_json = await settings_store.get("custom_presets")
        if custom_presets_json:
            import json as _json

            from augmentum.image.presets import GenrePreset
            preset_mgr = getattr(app.state, "image_preset_manager", None)
            if preset_mgr:
                with contextlib.suppress(ValueError, TypeError):
                    for p in _json.loads(custom_presets_json):
                        preset_mgr.add(GenrePreset(
                            name=p["name"], display_name=p.get("display_name", p["name"]),
                            description=p.get("description", ""),
                            positive_tags=p.get("positive_tags", ""),
                            negative_tags=p.get("negative_tags", ""),
                            recommended_model=p.get("recommended_model", ""),
                            cfg_scale=float(p.get("cfg_scale", 7.0)),
                            steps=int(p.get("steps", 20)),
                            sampler=p.get("sampler", ""),
                            scheduler=p.get("scheduler", ""),
                        ))
                log.info("custom_presets_restored")
        # Restore image panel active settings (used by /v command and tools)
        active_img_json = await settings_store.get("image_active_settings")
        if active_img_json:
            import json as _json2
            with contextlib.suppress(ValueError, TypeError):
                app.state.image_active_settings = _json2.loads(active_img_json)
                log.info("image_active_settings_restored")

        # Initialize notes store (browse notes as individual SQLite rows)
        from augmentum.state.notes_store import NotesStore

        notes_store = NotesStore(sqlite_backend.conn)
        app.state.notes_store = notes_store

        # Migrate old JSON blob notes to SQLite rows — claim them for the
        # oldest active user so the new per-user list endpoint still surfaces
        # them after the migration.
        try:
            owner_uid = ""
            try:
                cur = await sqlite_backend.conn.execute(
                    "SELECT id FROM users ORDER BY created_at ASC LIMIT 1",
                )
                row = await cur.fetchone()
                if row:
                    owner_uid = row[0] or ""
            except Exception:
                owner_uid = ""
            migrated = await notes_store.migrate_from_json(
                settings_store, user_id=owner_uid,
            )
            if migrated:
                log.info("browse_notes_migrated", count=migrated)
        except Exception:
            log.debug("notes_migration_check_failed", exc_info=True)

        # Discovery Engine store
        from augmentum.state.discovery_store import DiscoveryStore
        app.state.discovery_store = DiscoveryStore(sqlite_backend.conn)

        # Cross-device reading-position store (Android client sync surface).
        from augmentum.state.sync_store import ReadingPositionStore
        app.state.sync_store = ReadingPositionStore(sqlite_backend.conn)

        # Background-job queue (generic primitive). The store is shared;
        # the runner is a single worker that polls for pending rows.
        # requeue_crashed runs here (not in the runner) so it happens
        # exactly once on startup, before any new handler can race with
        # restart-recovery.
        from augmentum.jobs import JobRunner
        from augmentum.state.jobs_store import JobsStore

        jobs_store = JobsStore(sqlite_backend.conn)
        app.state.jobs_store = jobs_store
        try:
            requeued = await jobs_store.requeue_crashed()
            if requeued:
                log.info("background_jobs_requeued", count=requeued)
        except Exception:
            log.warning("background_jobs_requeue_failed", exc_info=True)
        # Run-ledger orphan sweep — finalizes rows stranded in
        # status='running' by the previous process (coder_turn_runs,
        # claude_runs, pi_runs, xr_sessions). Unlike jobs these are NOT
        # resumable, so they finalize as interrupted rather than requeue.
        try:
            from augmentum.state.run_ledger_sweep import finalize_orphan_runs
            await finalize_orphan_runs(sqlite_backend.conn)
        except Exception:
            log.warning("run_ledger_sweep_failed", exc_info=True)
        # Job reliability monitor (Phase 1.5 contract: the user always
        # hears back) — also the companion's bridge into job
        # completions (wiring program Phase 6): terminal events feed
        # her initiative queue so "your transcription finished" can
        # surface conversationally.
        from augmentum.jobs.monitor import JobMonitor
        app.state.job_monitor = JobMonitor(jobs_store)
        from augmentum.companion_runtime.bridges import make_job_terminal_listener
        app.state.job_monitor.subscribe(make_job_terminal_listener(app.state))
        app.state.job_runner = JobRunner(jobs_store, monitor=app.state.job_monitor)

        # Language-learning vocab state (per-user spaced-repetition rows).
        from augmentum.state.vocab_store import VocabStore
        app.state.vocab_store = VocabStore(sqlite_backend.conn)

        # Companion growth-loop substrate (Phase 1 — migration 216).
        # See docs/superpowers/specs/2026-05-31-companion-growth-loop-design.md
        from augmentum.companion.growth import GrowthStore
        app.state.growth_store = GrowthStore(sqlite_backend.conn)

        # Apply persisted log level if the operator changed it via the UI
        # toggle. Falls back to the env-var default set in setup_logging().
        try:
            persisted_log_level = await settings_store.get("ui.logLevel")
            if persisted_log_level:
                from augmentum.utils.logging import set_log_level
                applied = set_log_level(persisted_log_level)
                log.info("log_level_restored_from_db", level=applied)
        except Exception:
            log.warning("log_level_restore_failed", exc_info=True)

        log.info("persistent_settings_loaded")

        # Run local backend discovery (after providers + dismissed list loaded)
        await app.state.provider_registry.load_dismissed_discoveries(settings_store)
        await app.state.provider_registry.discover_local_backends()

        # Coder mode: container manager
        try:
            import aiodocker
            docker_client = aiodocker.Docker()
            from augmentum.coder.containers import ContainerManager
            app.state.container_manager = ContainerManager(
                docker=docker_client,
                db=sqlite_backend.conn if sqlite_backend else None,
            )
            log.info("container_manager_initialized")
            # Reconcile DB ↔ Docker once before any watcher or route
            # touches workspace state. Catches the daemon-restart cohort
            # (SIGKILL'd containers whose rows still claim running),
            # out-of-band docker pause/stop, and orphan containers with
            # no DB row. Failures here are best-effort — startup must
            # not block on a flaky Docker daemon.
            try:
                summary = await app.state.container_manager.reconcile_with_docker()
                if summary.get("reconciled") or summary.get("orphans"):
                    log.info("workspace_startup_reconcile", **summary)
            except Exception:
                log.warning(
                    "workspace_startup_reconcile_failed", exc_info=True
                )
        except Exception:
            app.state.container_manager = None
            docker_client = None
            log.info("container_manager_unavailable", reason="aiodocker or Docker socket not available")

        # Workspace PID-pressure watchdog. Periodic background scan of
        # running workspace containers vs. their pids.max cgroup. When
        # the live PID count crosses ``coder_workspace_pids_warn_pct``
        # the watchdog emits ``coder.workspace_pids_pressure`` so the
        # operator can react before runc wedges with "procReady not
        # received". Settings-tunable; 0 interval disables. Sleep is
        # interruptible so shutdown drains within the grace window.
        async def _workspace_pids_watchdog() -> None:
            interval = max(30, int(getattr(settings, "coder_workspace_pids_check_interval_s", 120) or 120))
            cm = app.state.container_manager
            if cm is None:
                return
            log.info("workspace_pids_watchdog_started", interval_s=interval)
            while True:
                try:
                    await asyncio.sleep(interval)
                    await cm.check_pids_pressure()
                except asyncio.CancelledError:
                    log.info("workspace_pids_watchdog_cancelled")
                    return
                except Exception:
                    log.warning("workspace_pids_watchdog_iter_failed", exc_info=True)

        if app.state.container_manager is not None:
            _track_bg(_workspace_pids_watchdog(), name="workspace_pids_watchdog")

        # SearXNG outbound-proxy manager. Drives settings.yml + restarts
        # the SearXNG container when the user toggles rotation or the
        # active healthy proxy changes. Init is best-effort — if the
        # bind-mount for SearXNG's settings.yml isn't present (older
        # installs that haven't picked up the compose change yet), the
        # feature is disabled rather than crashing the server.
        try:
            from augmentum.search import ProxyHealthcheckLoop, SearxngProxyManager

            cfg_path = Path(
                os.environ.get(
                    "AUGMENTUM_SEARXNG_CONFIG_PATH",
                    "/srv/searxng_config/settings.yml",
                )
            )
            if cfg_path.is_file():
                app.state.searxng_proxy_manager = SearxngProxyManager(
                    settings_yml_path=cfg_path,
                    searxng_container_name=os.environ.get(
                        "AUGMENTUM_SEARXNG_CONTAINER_NAME", "searxng"
                    ),
                    docker_client=docker_client,
                )
                # Defer settings import to runtime so the closure sees
                # the live module-level settings object (not a copy).
                def _live_settings():
                    from augmentum.config import settings as _s

                    return _s

                app.state.searxng_proxy_healthcheck = ProxyHealthcheckLoop(
                    manager=app.state.searxng_proxy_manager,
                    settings_provider=_live_settings,
                )
                app.state.searxng_proxy_healthcheck.start()
                log.info("searxng_proxy_manager_initialized", config_path=str(cfg_path))
            else:
                app.state.searxng_proxy_manager = None
                app.state.searxng_proxy_healthcheck = None
                log.info(
                    "searxng_proxy_manager_unavailable",
                    reason="settings.yml not writable from Augmentum container",
                    expected_path=str(cfg_path),
                )
        except Exception as _searxng_init_exc:  # noqa: BLE001 — never block server startup
            app.state.searxng_proxy_manager = None
            app.state.searxng_proxy_healthcheck = None
            log.warning(
                "searxng_proxy_manager_init_failed",
                error=str(_searxng_init_exc),
            )

        # Game streaming runtime (AGSP). Store is always created; the
        # runtime is parameterised from settings and reconciles any
        # orphaned containers from a prior run on startup. Adapter
        # selection: real Docker if the coder container_manager is up
        # (reuse its docker client to avoid a second connection),
        # otherwise the stub so the rest of the app still boots in
        # dev environments without Docker.
        try:
            from augmentum.game_stream import (
                GameStreamRuntime,
                PortPool,
                StubContainerAdapter,
            )
            from augmentum.state.game_stream_store import GameStreamStore

            game_stream_store = GameStreamStore(sqlite_backend.conn)
            app.state.game_stream_store = game_stream_store

            gs_max = int(getattr(settings, "game_stream_max_concurrent", 2) or 2)
            gs_idle = int(
                getattr(settings, "game_stream_idle_timeout_seconds", 600) or 600
            )
            gs_prefer_hw = bool(
                getattr(settings, "game_stream_prefer_hw_encoder", True)
            )
            # Credit-budget admission knobs — see _admit() docstring and
            # docs/superpowers/specs/2026-06-02-game-stream-admission-control.md
            gs_active_budget = int(
                getattr(settings, "game_stream_active_credit_budget", 8) or 8,
            )
            gs_resident_budget = int(
                getattr(settings, "game_stream_resident_credit_budget", 16) or 16,
            )
            gs_user_hard_cap = int(
                getattr(settings, "game_stream_user_hard_cap", 4) or 4,
            )
            gs_paused_stop = int(
                getattr(settings, "game_stream_paused_stop_seconds", 1800) or 1800,
            )

            gs_adapter: object = StubContainerAdapter()
            cm = getattr(app.state, "container_manager", None)
            if cm is not None and getattr(cm, "_docker", None) is not None:
                try:
                    from augmentum.game_stream.docker_adapter import (
                        DockerContainerAdapter,
                    )
                    # GPU passthrough: opt-in via env var. Set to 1
                    # in compose.gpu.yaml so users on the GPU stack
                    # get hardware rendering inside agsp containers
                    # automatically (Minetest under llvmpipe is
                    # unusable -- 250% CPU and chunk gen never
                    # completes). Bare CPU stack stays at 0/false.
                    gs_gpu = (
                        os.environ.get("AUGMENTUM_GAME_STREAM_GPU", "")
                        .lower() in ("1", "true", "yes")
                    )
                    # Host networking: default OFF. Tested -- on
                    # Docker Desktop (Mac/Windows) host net does NOT
                    # forward bound ports to the host (unlike bridge
                    # net's PortBindings), so selkies binding
                    # 0.0.0.0:8080 inside the VM is unreachable from
                    # the host's browser. Bridge + port mapping is
                    # the only reliable path on Docker Desktop. The
                    # WebRTC NAT-traversal problem this was meant to
                    # solve is instead handled by selkies' built-in
                    # TURN relay (openrelay default in
                    # entrypoint-base.sh).
                    # Set AUGMENTUM_GAME_STREAM_HOST_NET=1 only if
                    # running on native Linux Docker.
                    gs_host_net = (
                        os.environ.get("AUGMENTUM_GAME_STREAM_HOST_NET", "0")
                        .lower() in ("1", "true", "yes")
                    )
                    gs_adapter = DockerContainerAdapter(
                        cm._docker,
                        gpu_passthrough=gs_gpu,
                        host_network=gs_host_net,
                    )
                    log.info(
                        "game_stream_adapter",
                        impl="docker", gpu=gs_gpu, host_net=gs_host_net,
                    )
                except Exception:
                    log.warning(
                        "game_stream_docker_adapter_init_failed",
                        exc_info=True,
                    )
            else:
                log.info("game_stream_adapter", impl="stub")

            # Couch co-op cleanup hook: when a session stops, sweep
            # any outstanding invite tokens that point at it so a
            # guest can't auto-reconnect to a session that no longer
            # exists. Built here (rather than in the runtime) because
            # only the proxy layer knows about cast_invite_store.
            async def _on_gs_session_stopped(
                session_id: str, user_id: str,
            ) -> None:
                store = getattr(app.state, "cast_invite_store", None)
                if store is not None:
                    store.revoke_for_session(session_id)
                # Hide any QR overlay tied to this session on the
                # owning user's receivers. Idempotent on the receiver
                # side — no overlay = no-op.
                registry = getattr(app.state, "receiver_registry", None)
                if registry is not None and user_id:
                    try:
                        from augmentum.cast.receiver_protocol import (
                            CMD_HIDE_INVITE_QR,
                            ReceiverCmd,
                        )
                        await registry.broadcast(
                            user_id,
                            ReceiverCmd(
                                cmd=CMD_HIDE_INVITE_QR,
                                args={"reason": "session_ended"},
                            ),
                        )
                    except Exception:
                        pass

            app.state.game_stream_runtime = GameStreamRuntime(
                store=game_stream_store,
                # Port pool must cover the credit budget, not the old
                # flat per-user count. Two ports per credit slot
                # (stream + game) plus headroom.
                port_pool=PortPool(
                    base=30000,
                    count=max(2, gs_resident_budget * 2),
                ),
                adapter=gs_adapter,
                max_concurrent_per_user=gs_max,
                idle_timeout_seconds=gs_idle,
                prefer_hw_encoder=gs_prefer_hw,
                active_credit_budget=gs_active_budget,
                resident_credit_budget=gs_resident_budget,
                user_hard_cap=gs_user_hard_cap,
                paused_stop_seconds=gs_paused_stop,
                on_session_stopped=_on_gs_session_stopped,
            )
            try:
                await app.state.game_stream_runtime.reconcile_on_startup()
            except Exception:
                log.warning("game_stream_reconcile_failed", exc_info=True)
        except Exception:
            log.warning("game_stream_init_failed", exc_info=True)

        # Connect — age out call_sessions rows orphaned by a prior
        # process death. The in-memory invite-timer registry doesn't
        # survive a restart; without this sweep, a row caught mid-
        # ring sits in 'ringing' / 'invited' forever and pollutes the
        # calls history. See call_lifecycle.recover_stale_invites_on_startup.
        try:
            from augmentum.connect.call_lifecycle import (
                recover_stale_invites_on_startup,
            )
            await recover_stale_invites_on_startup(sqlite_backend.conn)
        except Exception:
            log.warning("connect_stale_invite_recovery_failed", exc_info=True)

        # Augmentum Experience Framework (AXF / titles). Builds the
        # TitleService over the existing artifacts table + the new
        # title_runs table, registers Source+Runtime adapters for the
        # surfaces shipped to date. Browser-iframe runtime is registered
        # eagerly at module import; the rest are wired here.
        try:
            from augmentum.marketplace import (
                MarketplaceSource,
                MarketplaceStore,
                load_catalog_into_store,
            )
            from augmentum.saves import SaveStore
            from augmentum.titles import (
                AgspStreamedRuntime,
                BiosStore,
                InternalRomSource,
                InternalSource,
                Js13kSource,
                TitleService,
                TitleStore,
                runtime_registry,
                source_registry,
            )

            title_store = TitleStore(sqlite_backend.conn)
            app.state.title_store = title_store

            # SaveStore depends on the blob store for save bytes.
            # Defensive: blob_store may not be attached if VFS init
            # failed earlier; fall back gracefully so the rest of the
            # title surface still boots.
            blob_store = getattr(app.state, "blob_store", None)
            if blob_store is not None:
                app.state.save_store = SaveStore(
                    sqlite_backend.conn, blob_store,
                )
                # BIOS files share the same blob substrate as saves --
                # one canonical scph5500.bin shared across users via
                # refcounted dedup. The bulk-import classifier routes
                # detected BIOS dumps here; the launch path reads them
                # back to populate EJS_biosUrl.
                app.state.bios_store = BiosStore(
                    sqlite_backend.conn, blob_store,
                )
            else:
                app.state.save_store = None
                app.state.bios_store = None
                log.info(
                    "save_store_skipped",
                    reason="blob_store not initialised",
                )

            # ── Controller framework ────────────────────────────────
            # Per-system canonical layouts ship as code; per-user
            # remap overrides live in controller_remaps. The service
            # merges them at resolve time. Once built, we late-bind
            # the service into any registered Runtime that wants it
            # (currently emulator-browser; future emulator-streamed
            # consumes the same shape via a different adapter).
            try:
                from augmentum.controllers import (
                    ControllerService,
                    ControllerStore,
                )
                controller_store = ControllerStore(sqlite_backend.conn)
                app.state.controller_store = controller_store
                app.state.controller_service = ControllerService(
                    store=controller_store,
                )
                emulator_runtime = runtime_registry.get("emulator-browser")
                if emulator_runtime is not None and hasattr(
                    emulator_runtime, "attach_controller_service",
                ):
                    emulator_runtime.attach_controller_service(
                        app.state.controller_service,
                    )
                # Late-bind the BIOS store so the runtime can validate
                # required-BIOS presence and populate bios_url/files
                # in the LaunchHandle. Same pattern as controllers.
                if emulator_runtime is not None and hasattr(
                    emulator_runtime, "attach_bios_store",
                ) and getattr(app.state, "bios_store", None) is not None:
                    emulator_runtime.attach_bios_store(app.state.bios_store)
            except Exception:
                log.warning("controller_service_init_failed", exc_info=True)
                app.state.controller_service = None
                app.state.controller_store = None

            # ── Sources ─────────────────────────────────────────────
            # InternalSource: manual / hand-built manifest import.
            # Js13kSource: bridges the existing js13k provider so
            #   /api/titles/_/discover?source_id=js13k works end-to-end.
            # InternalRomSource: emulator ROM upload bridge (the
            #   /api/titles/upload-rom multipart route + this Source
            #   are the two halves of the ROM install flow).
            # MarketplaceSource is registered after the marketplace
            # store + catalog load so it can delegate installs to the
            # *other* registered Sources via install_via.
            source_registry.register(InternalSource(sqlite_backend.conn))
            source_registry.register(Js13kSource(sqlite_backend.conn))
            source_registry.register(InternalRomSource(sqlite_backend.conn))

            # ── Marketplace store + catalog load ──────────────────
            try:
                marketplace_store = MarketplaceStore(sqlite_backend.conn)
                app.state.marketplace_store = marketplace_store
                stats = await load_catalog_into_store(marketplace_store)
                log.info("marketplace_catalog_load_summary", **stats)
                # Discover surface: also ingest the provider catalog so
                # provider services are browsable from /api/discover.
                # Independent of service_manager being available — the
                # ProviderCatalog is a pure JSON read; Docker only
                # matters at install click time.
                try:
                    from augmentum.marketplace import load_providers_into_store
                    prov_stats = await load_providers_into_store(marketplace_store)
                    log.info("marketplace_providers_load_summary", **prov_stats)
                except Exception:
                    log.warning("marketplace_providers_load_failed", exc_info=True)
                # Discover surface: media servers (Jellyfin/Suwayomi/…) —
                # the MEDIA-category catalog entries surfaced as one-click
                # provision-and-connect cards, separate from inference
                # providers above.
                try:
                    from augmentum.marketplace import load_media_servers_into_store
                    media_stats = await load_media_servers_into_store(marketplace_store)
                    log.info("marketplace_media_servers_load_summary", **media_stats)
                except Exception:
                    log.warning("marketplace_media_servers_load_failed", exc_info=True)
                # Phase 2: schedule the community feed refresh in the
                # background. The first pull happens after a brief
                # delay so it doesn't block lifespan startup; the
                # task self-throttles on its configured cadence.
                # No-op when discover_community_feed_enabled=False.
                try:
                    from augmentum.marketplace import schedule_community_feed_refresh
                    schedule_community_feed_refresh(app, marketplace_store)
                except Exception:
                    log.warning("community_feed_schedule_failed", exc_info=True)
                # Community app stores (service-OS phase 4): refresh every
                # registered store's listings in the background after
                # boot. Per-store isolation inside sync_all_stores — a
                # dead URL never blocks boot or its sibling stores.
                try:
                    import asyncio as _aio_stores

                    from augmentum.marketplace.loaders.stores import (
                        sync_all_stores,
                    )
                    _settings_store = getattr(app.state, "settings_store", None)
                    _http = getattr(app.state, "http_client", None)
                    if _settings_store is not None and _http is not None:
                        _task = _aio_stores.create_task(sync_all_stores(
                            _settings_store, marketplace_store, _http,
                        ))
                        getattr(app.state, "background_tasks", []).append(_task)
                except Exception:
                    log.warning("community_stores_schedule_failed", exc_info=True)
                # MarketplaceSource depends on the registry having the
                # underlying installer Sources already, so register
                # last.
                source_registry.register(
                    MarketplaceSource(
                        store=marketplace_store,
                        sources=source_registry,
                    ),
                )
            except Exception:
                log.warning("marketplace_init_failed", exc_info=True)
                app.state.marketplace_store = None

            # ── Runtimes ────────────────────────────────────────────
            # Register the AGSP runtime adapter, late-binding the
            # GameStreamRuntime that may or may not have come up cleanly.
            agsp_adapter = AgspStreamedRuntime(
                getattr(app.state, "game_stream_runtime", None),
            )
            runtime_registry.register(agsp_adapter)

            app.state.title_service = TitleService(
                store=title_store,
                sources=source_registry,
                runtimes=runtime_registry,
            )
            log.info(
                "title_service_initialized",
                runtimes=[r.id for r in runtime_registry.list()],
                sources=[s.id for s in source_registry.list()],
            )
        except Exception:
            log.warning("title_service_init_failed", exc_info=True)
            app.state.title_service = None
            app.state.marketplace_store = None

        # Marketplace: service manager (reuses Docker client from coder mode)
        app.state.service_manager = None
        _docker_client = None
        if app.state.container_manager:
            _docker_client = app.state.container_manager._docker
        if _docker_client:
            try:
                from augmentum.providers.manager import ServiceManager
                app.state.service_manager = ServiceManager(
                    docker=_docker_client,
                    db=sqlite_backend.conn if sqlite_backend else None,
                )
            except Exception:
                log.info("service_manager_unavailable", exc_info=True)
        else:
            try:
                import aiodocker as _aio_docker
                _docker_client = _aio_docker.Docker()
                from augmentum.providers.manager import ServiceManager
                app.state.service_manager = ServiceManager(
                    docker=_docker_client,
                    db=sqlite_backend.conn if sqlite_backend else None,
                )
            except Exception:
                log.info("service_manager_unavailable", reason="Docker not available")

        # Restore enabled managed services in the BACKGROUND (2026-07-02
        # boot-latency work). restore_enabled() can docker-pull a multi-GB
        # image when one is missing — measured 89s pulling speaches pre-yield,
        # every second of it gating first paint. Nothing at boot consumes
        # these services synchronously: audio providers register from DB rows
        # and probe lazily, and each service flips available when its
        # container reports healthy. Failures degrade exactly as before
        # (service stays down, manager logs it).
        if app.state.service_manager is not None:
            # Manifest services (Discover service apps) register their
            # definitions at install time only — rebuild them from the
            # marketplace catalog BEFORE restore_enabled runs, or every
            # installed app fails restore with "Unknown service" after
            # a restart. Cheap (DB + in-memory), so it runs inline.
            try:
                from augmentum.marketplace.runtime_rehydrate import (
                    rehydrate_manifest_services,
                )
                _mstore = getattr(app.state, "marketplace_store", None)
                if _mstore is not None:
                    _rehydrated = await rehydrate_manifest_services(
                        _mstore, app.state.service_manager,
                    )
                    if _rehydrated:
                        log.info(
                            "manifest_services_rehydrated", count=_rehydrated,
                        )
            except Exception:
                log.warning("manifest_service_rehydrate_failed", exc_info=True)

            async def _restore_managed_services(_mgr=app.state.service_manager) -> None:
                try:
                    restored = await _mgr.restore_enabled()
                    if restored:
                        log.info("managed_services_restored", count=restored)
                except Exception:
                    log.warning("managed_services_restore_failed", exc_info=True)
                # Re-register OpenAI-compatible engine backends (e.g. vLLM) from
                # each enabled service's persisted config_json. Runtime
                # register_backend is lost on restart (F4) — without this, an
                # installed vLLM engine's models become unroutable after an
                # Augmentum restart until reinstall.
                try:
                    registry = getattr(app.state, "provider_registry", None)
                    client = getattr(app.state, "http_client", None)
                    db = getattr(_mgr, "_db", None)
                    if registry is not None and client is not None and db is not None:
                        from augmentum.models.openai_compat import OpenAIBackend
                        cur = await db.execute(
                            "SELECT id FROM managed_services WHERE enabled = 1",
                        )
                        ids = [r[0] for r in await cur.fetchall()]
                        for sid in ids:
                            cfg = await _mgr.read_config_json(sid)
                            reg = (cfg or {}).get("backend_registration")
                            if isinstance(reg, dict) and reg.get("key") and reg.get("url"):
                                chat_client = getattr(registry, "_chat_http_client", None) or client
                                registry.register_backend(
                                    str(reg["key"]),
                                    OpenAIBackend(client, str(reg["url"]), "not-needed", chat_client=chat_client),
                                )
                                from augmentum.config import settings as _s
                                setattr(_s, f"{reg['key']}_base_url", str(reg["url"]))
                                log.info("engine_backend_reregistered", key=reg["key"], url=reg["url"])
                except Exception:
                    log.warning("engine_backend_reregister_failed", exc_info=True)
            _track_bg(_restore_managed_services(), name="restore_managed_services")

        async def _warmup_models() -> None:
            """Background task: preload embedding + reranker models."""
            import time
            t0 = time.monotonic()
            try:
                from augmentum.memory.embeddings import EmbeddingService
                await load_model_off_loop(EmbeddingService.get_model)
                log.info("embedding_model_warmed")
            except Exception:
                log.warning("embedding_warmup_failed", exc_info=True)
            try:
                from augmentum.memory.reranker import RerankService
                await load_model_off_loop(RerankService.get_model)
                log.info("reranker_model_warmed")
            except Exception:
                log.warning("reranker_warmup_failed", exc_info=True)
            log.info("warmup_complete", elapsed_s=round(time.monotonic() - t0, 2))

        # Initialize memory store (uses the same SQLite backend)
        if settings.memory_enabled:
            from augmentum.memory.store import MemoryStore

            app.state.memory_store = MemoryStore(sqlite_backend)
            log.info("memory_store_initialized")

            # The Evidence Bus (Earned Understanding P2). Shares the SQLite
            # backend; emitters record user activity as evidence and converge
            # independent sources onto beliefs via the existing ladder.
            from augmentum.memory.evidence import EvidenceStore
            app.state.evidence_store = EvidenceStore(sqlite_backend)
            log.info("evidence_store_initialized")

            # Wire consolidation backend (if enabled and an LLM is available)
            if settings.memory_consolidation_enabled:
                try:
                    consolidation_backend = getattr(
                        app.state.provider_registry, "default_backend", None,
                    )
                    if consolidation_backend:
                        app.state.memory_store.set_consolidation_backend(
                            consolidation_backend,
                            settings.memory_llm_extraction_model or "",
                        )
                        log.info("memory_consolidation_enabled")
                    else:
                        log.warning("memory_consolidation_no_backend")
                except Exception:
                    log.warning("memory_consolidation_init_failed", exc_info=True)

            # Initialize document store for RAG
            if settings.document_rag_enabled:
                from augmentum.documents.store import DocumentStore

                app.state.document_store = DocumentStore(sqlite_backend)
                log.info("document_store_initialized")
            else:
                app.state.document_store = None

            # Initialize core profile manager (no LLM cost, pure ranking)
            app.state.core_profile_manager = None
            if settings.memory_core_profile_enabled:
                from augmentum.memory.core_profile import CoreProfileManager

                app.state.core_profile_manager = CoreProfileManager(
                    app.state.memory_store,
                    max_tokens=settings.memory_core_profile_max_tokens,
                    rebuild_interval=settings.memory_core_profile_rebuild_interval,
                    app_state=app.state,
                )
                # Hydrate per-user counter + last-rebuild timestamp from
                # persisted state. Without this the extraction counter
                # resets on every restart, and the age-based stale check
                # has no reference point.
                try:
                    await app.state.core_profile_manager.initialize()
                except Exception:
                    log.warning("core_profile_manager_initialize_failed", exc_info=True)
                log.info("core_profile_manager_initialized")

            # Initialize memory compactor (background periodic cleanup)
            app.state.memory_compactor = None
            if settings.memory_compaction_enabled:
                from augmentum.memory.compactor import MemoryCompactor

                app.state.memory_compactor = MemoryCompactor(
                    store=app.state.memory_store,
                    registry=app.state.provider_registry,
                    interval_hours=settings.memory_compaction_interval_hours,
                )
                app.state.memory_compactor.start()
                log.info("memory_compactor_started")

            # Dream system — boot via shared lifecycle helper so the same
            # path is reused when ui.dreamEnabled flips at runtime. Boot
            # predicate matches the runtime toggle: alive if ANY tenant has
            # opted in (user_settings) or the install-wide default is on
            # (app_settings). Stage D migrated the UI toggle to per-user
            # storage, so reading only the global row would miss every
            # user's choice after a restart.
            app.state.dream_journal = None
            app.state.dream_portrait_manager = None
            app.state.dream_engine = None
            app.state.dream_scheduler = None

            from augmentum.dream.lifecycle import setup_dream_system, should_dream_run
            if await should_dream_run(settings_store):
                await setup_dream_system(app)

            # CompanionRuntime — top-level kernel for Becca. Inert until
            # the per-feature flags flip on (dispatch/tick/journal/…),
            # but the substrate (identity, state, memory facade, bus)
            # comes up here so routes and XR clients can subscribe.
            # Depends on memory_store + core_profile_manager being set.
            app.state.companion_runtime = None
            if getattr(settings, "companion_runtime_enabled", False):
                try:
                    from augmentum.companion_runtime.runtime import create_runtime
                    app.state.companion_runtime = await create_runtime(
                        backend=sqlite_backend,
                        memory_store=app.state.memory_store,
                        core_profile=app.state.core_profile_manager,
                        companion_id=getattr(settings, "companion_default_id", "becca"),
                        app_state=app.state,
                    )
                    log.info("companion_runtime_initialized",
                             companion_id=app.state.companion_runtime.companion_id)
                except Exception:
                    log.exception("companion_runtime_init_failed")
                    app.state.companion_runtime = None

            # SchedulerService — app-level dispatcher for standing tasks
            # (briefings/reminders/watches/deferred requests). Timed
            # action is a PLATFORM substrate: with the companion ON the
            # service covers every non-owner user (the tick verb keeps
            # the owner's lane); with the companion OFF it dispatches
            # everyone via a headless context, so schedules created from
            # chat/voice/the Schedule UI fire regardless.
            app.state.scheduler_service = None
            if getattr(settings, "scheduling_enabled", True):
                try:
                    from augmentum.scheduling import SchedulerService
                    app.state.scheduler_service = SchedulerService(
                        backend=sqlite_backend,
                        app_state=app.state,
                        companion_runtime=app.state.companion_runtime,
                        companion_id=getattr(
                            settings, "companion_default_id", "becca",
                        ),
                    )
                    await app.state.scheduler_service.start()
                except Exception:
                    log.exception("scheduler_service_init_failed")
                    app.state.scheduler_service = None

            # Accumulation thesis Step 2 — the unified Companion façade.
            # Mounts a single addressable object that composes
            # identity + state + memory + bus + user_affect. Every
            # module that touches her should route through this dict
            # rather than reaching into companion_runtime.* directly.
            # ``app.state.companion_runtime`` stays as a compat surface
            # during the migration; new code should prefer
            # ``app.state.companions["becca"]``.
            app.state.companions = {}
            if app.state.companion_runtime is not None:
                try:
                    from augmentum.companion import Companion
                    companion = Companion(app.state.companion_runtime)
                    app.state.companions[companion.name] = companion
                    log.info("companion_facade_mounted",
                             companion_id=companion.name,
                             names=list(app.state.companions.keys()))
                except Exception:
                    log.exception("companion_facade_mount_failed")

            # Knowledge packs
            app.state.pack_manager = None
            app.state.catalog_client = None
            app.state.install_jobs = {}
            if settings.knowledge_packs_enabled:
                from augmentum.knowledge.packs import PackManager
                pack_dir = Path(
                    settings.knowledge_packs_custom_dir
                    or settings.knowledge_packs_dir
                    or f"{settings.data_dir}/knowledge"
                )
                pack_mgr = PackManager(pack_dir)
                loaded = await pack_mgr.scan()
                app.state.pack_manager = pack_mgr
                # Module-level mirror for consumers without app.state in
                # reach (coder pack_search tool). Packs are server-level.
                from augmentum.knowledge.runtime import set_pack_manager
                set_pack_manager(pack_mgr)
                log.info("knowledge_packs_loaded", count=loaded, dir=pack_dir)

                # Catalog client for browsing Kiwix
                from augmentum.knowledge.catalog import CatalogClient
                cache_dir = pack_dir / "cache"
                cache_dir.mkdir(parents=True, exist_ok=True)
                app.state.catalog_client = CatalogClient(
                    cache_dir=cache_dir,
                    cache_ttl=settings.knowledge_catalog_cache_ttl,
                )

            # Warm up embedding + reranker models in background (non-blocking)
            if settings.startup_warmup:
                _track_bg(_warmup_models(), name="warmup_models")

    else:
        # Fall back to in-memory backend — auth, files, etc. won't persist
        from augmentum.state.backends.memory import MemoryBackend

        app.state.state_manager = StateManager(MemoryBackend())
        app.state.persistence_degraded = True

    # Schedule periodic provisional cleanup + co-occurrence decay
    async def _memory_maintenance_loop():
        """Background: clean expired provisional memories hourly, decay co-occurrence weekly, retroactive demotion daily."""
        import asyncio
        _decay_counter = 0  # track hours to fire decay weekly
        _demote_counter = 0  # track hours to fire demotion daily
        while True:
            await asyncio.sleep(3600)  # hourly
            _decay_counter += 1
            _demote_counter += 1
            try:
                from augmentum.memory.integration import cleanup_provisional_memories
                deleted = await cleanup_provisional_memories(app.state)
                if deleted:
                    log.info("provisional_cleanup_ran", deleted=deleted)

                # Retroactive demotion daily (every 24 hours). ACTIVE
                # memories matching the staleness rule drop to ARCHIVE;
                # see MemoryStore.retroactive_demote for the predicates.
                demote_interval = max(1, int(settings.memory_demotion_sweep_interval_seconds // 3600))
                if _demote_counter >= demote_interval:
                    _demote_counter = 0
                    store = getattr(app.state, "memory_store", None)
                    if store and settings.memory_retroactive_demotion_enabled:
                        demoted = await store.retroactive_demote()
                        if demoted:
                            log.info("retroactive_demotion_ran", demoted=demoted)

                # Decay co-occurrence weekly (every 168 hours)
                if _decay_counter >= 168:
                    _decay_counter = 0
                    store = getattr(app.state, "memory_store", None)
                    if store:
                        affected = await store.decay_cooccurrence()
                        if affected:
                            log.info("cooccurrence_decay_ran", affected=affected)
            except Exception:
                log.debug("memory_maintenance_error", exc_info=True)

    if settings.memory_enabled:
        _track_bg(_memory_maintenance_loop(), name="memory_maintenance")
        log.info("memory_maintenance_scheduled")

    # --- Signal aggregator ---
    # Daily pass over bug_finder_runs + companion_journal into signal_events.
    # See augmentum/signals/aggregator.py for the substrate rationale.
    async def _signals_aggregator_loop():
        from augmentum.signals import aggregate_all_users
        # Initial delay: don't slam the DB during startup warmup; wait 5 min
        # for the rest of the lifecycle to settle.
        await asyncio.sleep(300)
        while True:
            try:
                _sm = getattr(app.state, "state_manager", None)
                _conn = getattr(getattr(_sm, "backend", None), "conn", None)
                if _conn is not None:
                    _results = await aggregate_all_users(_conn)
                    # Phase 6 bridge: fresh signals become initiative
                    # proposals (autonomy-floor gated inside).
                    try:
                        from augmentum.companion_runtime.bridges import (
                            bridge_signal_results,
                        )
                        await bridge_signal_results(app.state, _results)
                    except Exception:
                        log.warning("signals_initiative_bridge_failed", exc_info=True)
            except Exception:
                log.warning("signals_aggregator_failed", exc_info=True)
            await asyncio.sleep(24 * 3600)

    if settings.signals_aggregator_enabled:
        _track_bg(_signals_aggregator_loop(), name="signals_aggregator")
        log.info("signals_aggregator_scheduled")

    # --- Coder idle reaper ---
    # Stops coder workspaces with ``always_on=0`` after
    # ``coder_idle_timeout`` seconds of no activity. Workspaces marked
    # always_on are exempt (intended for dev servers / long-running
    # daemons). Activity is bumped by ``ContainerManager.mark_active``
    # from coder route hits + chat completions; the reaper reads the
    # persisted ``last_active`` directly so a server restart picks up
    # the correct cut-off. See migration 211 for the policy column +
    # `ContainerManager.sweep_idle` for the selection rules.
    async def _coder_idle_watcher():
        # Initial delay matches the signals_aggregator pattern — let
        # the rest of lifespan finish settling before scanning.
        await asyncio.sleep(60)
        # Sweep cadence: every 2 minutes. Cheap query (one indexed
        # scan over project_checkouts WHERE status='running') and
        # the timeout itself is in the 10-30 min range, so finer
        # granularity wouldn't change UX.
        sweep_interval = 120
        while True:
            try:
                mgr = getattr(app.state, "container_manager", None)
                if mgr is not None:
                    stopped = await mgr.sweep_idle(
                        timeout_seconds=settings.coder_idle_timeout,
                    )
                    if stopped:
                        log.info(
                            "coder_idle_sweep",
                            stopped=stopped,
                            timeout_s=settings.coder_idle_timeout,
                        )
            except Exception:
                log.warning("coder_idle_sweep_failed", exc_info=True)
            await asyncio.sleep(sweep_interval)

    if settings.coder_idle_timeout > 0:
        _track_bg(_coder_idle_watcher(), name="coder_idle_watcher")
        log.info(
            "coder_idle_watcher_scheduled",
            timeout_s=settings.coder_idle_timeout,
        )

    # --- Auth ---
    # First-user-wins: when no admin exists yet, /api/auth/setup accepts any
    # request and the first caller becomes admin. Once an admin exists the
    # endpoint 403s. Matches the Open WebUI / Mattermost pattern; trades a
    # narrow first-claim window on shared LANs for zero-friction localhost
    # onboarding (the dominant case).
    _sm_backend = getattr(getattr(app.state, "state_manager", None), "backend", None)
    _sm_conn = getattr(_sm_backend, "conn", None)
    if _sm_conn:
        from augmentum.auth.api_keys import ApiKeyManager
        from augmentum.auth.session_manager import SessionManager
        app.state.session_manager = SessionManager(_sm_conn)
        app.state.api_key_manager = ApiKeyManager(_sm_conn)
        # Chain auth-cache invalidation so any user mutation routed
        # through SessionManager (update_user, delete_user, revoke_all)
        # also drops cached API-key validations for that user. Without
        # this hook, deactivating a user leaves their sk-aug-* keys
        # valid for up to ``ApiKeyManager._cache_ttl`` seconds.
        app.state.session_manager.register_user_cache_invalidator(
            app.state.api_key_manager.invalidate_user_cache,
        )
        # Backfill the durable first-run setup latch for installs created
        # before it existed: if any user is present, persist the
        # auth_setup_completed flag so /api/auth/setup can never silently
        # re-open if the users table is later emptied. No-op on a genuinely
        # fresh install. Best-effort — never block startup on it.
        try:
            await app.state.session_manager.ensure_setup_latch()
        except Exception:
            log.debug("ensure_setup_latch_failed", exc_info=True)
        # Self-heal: an install with users but no admin can't reach
        # admin-only settings (coder subagents, tool config). Promote the
        # owner if no admin exists. No-op once any admin is present.
        try:
            await app.state.session_manager.ensure_admin_exists()
        except Exception:
            log.debug("ensure_admin_exists_failed", exc_info=True)
    else:
        app.state.session_manager = None
        app.state.api_key_manager = None
        log.warning("auth_session_manager_unavailable", reason="No SQLite backend")

    async def _auth_cleanup_loop():
        while True:
            await asyncio.sleep(3600)
            try:
                sm = app.state.session_manager
                if sm:
                    await sm.cleanup_expired()
            except Exception as exc:
                log.warning("auth_cleanup_failed", error=str(exc))

    _track_bg(_auth_cleanup_loop(), name="auth_cleanup")

    # --- Background job runner ---
    # Started after auth cleanup so handlers that resolve user context
    # find the SessionManager ready. Handlers are registered here with
    # a factory that closes over `app` — lookups against app.state
    # happen at job-run time, so services that finish initialising
    # after this block (http_client, file_index) are still reachable
    # when the runner dispatches work.
    _job_runner = getattr(app.state, "job_runner", None)
    if _job_runner:
        from augmentum.jobs import register_handler
        from augmentum.jobs.handlers.bug_finder_run import (
            make_bug_finder_run_handler,
        )
        from augmentum.jobs.handlers.coder_background_run import (
            make_coder_background_run_handler,
        )
        from augmentum.jobs.handlers.coder_research_run import (
            make_coder_research_run_handler,
        )
        from augmentum.jobs.handlers.file_caption import (
            make_file_caption_handler,
        )
        from augmentum.jobs.handlers.gguf_download import (
            make_gguf_download_handler,
        )
        from augmentum.jobs.handlers.gutenberg_fetch import (
            make_gutenberg_fetch_handler,
        )
        from augmentum.jobs.handlers.image_pull import (
            make_image_pull_handler,
        )
        from augmentum.jobs.handlers.journal_vec_backfill import (
            make_journal_vec_backfill_handler,
        )
        from augmentum.jobs.handlers.media_server_detach import (
            make_media_server_detach_handler,
        )
        from augmentum.jobs.handlers.service_install import (
            make_service_install_handler,
        )
        from augmentum.jobs.handlers.addon_install import (
            make_addon_install_handler,
        )
        from augmentum.jobs.handlers.lang_pack_install import (
            make_lang_pack_install_handler,
        )
        from augmentum.jobs.handlers.media_sync import (
            make_media_sync_handler,
        )
        from augmentum.jobs.handlers.narration_synth import (
            make_narration_synth_handler,
        )
        from augmentum.jobs.handlers.comic_narration_synth import (
            make_comic_narration_synth_handler,
        )
        from augmentum.jobs.handlers.wake_word_corpus_download import (
            make_wake_word_corpus_download_handler,
        )
        from augmentum.jobs.handlers.wake_word_training import (
            make_wake_word_training_handler,
        )
        register_handler("gutenberg_fetch", make_gutenberg_fetch_handler(app))
        register_handler("media_sync", make_media_sync_handler(app))
        register_handler("gguf_download", make_gguf_download_handler(app))
        register_handler("bug_finder_run", make_bug_finder_run_handler(app))
        register_handler(
            "coder_background_run", make_coder_background_run_handler(app),
        )
        register_handler(
            "coder_research_run", make_coder_research_run_handler(app),
        )
        register_handler("lang_pack_install", make_lang_pack_install_handler(app))
        register_handler("narration_synth", make_narration_synth_handler(app))
        register_handler("comic_narration_synth", make_comic_narration_synth_handler(app))
        register_handler("image_pull", make_image_pull_handler(app))
        register_handler("service_install", make_service_install_handler(app))
        register_handler("addon_install", make_addon_install_handler(app))
        register_handler("wake_word_training", make_wake_word_training_handler(app))
        register_handler(
            "wake_word_corpus_download",
            make_wake_word_corpus_download_handler(app),
        )
        register_handler("file_caption", make_file_caption_handler(app))
        register_handler(
            "journal_vec_backfill",
            make_journal_vec_backfill_handler(app),
        )
        register_handler(
            "media_server_detach",
            make_media_server_detach_handler(app),
        )
        _job_runner.start()
        _job_monitor = getattr(app.state, "job_monitor", None)
        if _job_monitor:
            _job_monitor.start()

        # One-shot enqueue of journal_vec backfill if there are
        # journal rows from before migration 177 that need their vec
        # mirror populated. Paged + sleeped so it never stalls. Guarded
        # by a count check so we don't enqueue a no-op on every boot.
        try:
            backend = getattr(app.state, "backend", None)
            if backend is not None and backend.conn is not None:
                cur = await backend.conn.execute(
                    """
                    SELECT COUNT(*) FROM companion_journal j
                    WHERE j.embedding IS NOT NULL
                      AND NOT EXISTS (
                          SELECT 1 FROM companion_journal_vec v
                          WHERE v.journal_id = j.id
                      )
                    """
                )
                row = await cur.fetchone()
                await cur.close()
                missing = int(row[0]) if row else 0
                if missing > 0:
                    await app.state.jobs_store.create(
                        job_type="journal_vec_backfill",
                        payload={},
                    )
                    log.info(
                        "journal_vec_backfill_enqueued",
                        missing_rows=missing,
                    )
        except Exception as exc:
            # Vec table missing / extension not loaded → nothing to do.
            # Any other error is logged but doesn't block startup.
            log.info(
                "journal_vec_backfill_check_skipped",
                error=str(exc)[:200],
            )

    async def _files_maintenance_loop():
        # Adapter dispatch + orphan blob sweep, configurable cadence + TTL.
        # Replaces the old trash-only loop that bypassed adapter.delete()
        # and leaked blobs as a result.
        from augmentum.vfs.adapters import get_adapter
        from augmentum.vfs.maintenance import run_maintenance

        interval = max(60.0, float(settings.files_maintenance_interval_hours) * 3600.0)
        while True:
            await asyncio.sleep(interval)
            try:
                await run_maintenance(
                    file_index=getattr(app.state, "file_index", None),
                    blob_store=getattr(app.state, "blob_store", None),
                    adapter_lookup=get_adapter,
                    trash_ttl_days=settings.files_trash_ttl_days,
                )
            except Exception:
                log.warning("files_maintenance_error", exc_info=True)

    _track_bg(_files_maintenance_loop(), name="files_maintenance")

    async def _transient_artifact_sweep_loop():
        # Evict transient artifacts (image_search thumbnails etc.) by
        # age + size cap. Runs globally — these rows are cache, not user data.
        interval = max(60.0, float(settings.transient_artifact_sweep_hours) * 3600.0)
        while True:
            await asyncio.sleep(interval)
            store = getattr(app.state, "artifact_store", None)
            if store is None:
                continue
            try:
                await store.prune_transient(
                    max_age_days=settings.transient_artifact_ttl_days,
                    max_total_mb=settings.transient_artifact_max_mb,
                )
            except Exception:
                log.warning("transient_artifact_sweep_error", exc_info=True)

    _track_bg(_transient_artifact_sweep_loop(), name="transient_artifact_sweep")

    # --- File Index & VFS ---
    if _sm_conn:
        from augmentum.vfs import set_file_index
        from augmentum.vfs.bridges import (
            VFS,
            ArtifactBridge,
            ChatImageBridge,
            DocumentBridge,
            ImageBridge,
            KnowledgeBridge,
            VoiceBridge,
        )
        from augmentum.vfs.index import FileIndexService

        file_index = FileIndexService(_sm_conn)
        # Wire the jobs_store so register() can enqueue caption jobs for
        # image uploads. The Piece-2 captioner runs background via the
        # SmolVLM sibling, writing descriptions back to file_index.
        if getattr(app.state, "jobs_store", None) is not None:
            file_index.set_jobs_store(app.state.jobs_store)
        app.state.file_index = file_index
        set_file_index(file_index)

        # Comic series identity store — Phase A of the comic-library plan.
        # One store per process, shared by sync.py (catalog ingest resolves
        # series), the per-page delivery route, and the future library
        # surface. Mirror the file_index wiring pattern exactly.
        from augmentum.media.comic_series_store import (
            ComicSeriesStore,
            set_comic_series_store,
        )
        comic_series_store = ComicSeriesStore(_sm_conn)
        app.state.comic_series_store = comic_series_store
        set_comic_series_store(comic_series_store)

        from augmentum.media.library_store import (
            MediaLibraryStore,
            set_media_library_store,
        )
        media_library_store = MediaLibraryStore(_sm_conn)
        app.state.media_library_store = media_library_store
        set_media_library_store(media_library_store)

        # Content-addressed blob store + adapter registry. Uploads is the
        # first adapter wired; future sources (Dropbox, S3, etc.) register
        # themselves here and the rest of the pipeline picks them up via
        # the registry rather than any new switch statements.
        from augmentum.vfs import register_adapter
        from augmentum.vfs.adapters.uploads import UploadsAdapter
        from augmentum.vfs.blobs import BlobStore

        blob_store = BlobStore(_sm_conn)
        app.state.blob_store = blob_store

        # Back-fill stores that depend on blob_store but were initialised
        # earlier in the lifespan (the AXF/titles block runs before this
        # one; it reads getattr(app.state, "blob_store", None), gets None,
        # and latches save_store/bios_store to None). Without this, BIOS
        # uploads + ROM saves both 503 with "Storage subsystems
        # unavailable" forever after restart. Re-creating the stores here
        # is cheap (constructors don't touch the network) and the title
        # surface picks them up via app.state on the next request.
        if getattr(app.state, "save_store", None) is None:
            from augmentum.saves import SaveStore
            app.state.save_store = SaveStore(_sm_conn, blob_store)
        if getattr(app.state, "bios_store", None) is None:
            from augmentum.titles import BiosStore
            app.state.bios_store = BiosStore(_sm_conn, blob_store)
            # Late-bind into the emulator-browser runtime if it exists,
            # mirroring the wiring at lines ~2140-2143 so the runtime
            # can populate bios_url at launch time.
            from augmentum.titles import runtime_registry
            emu = runtime_registry.get("emulator-browser")
            if emu is not None and hasattr(emu, "attach_bios_store"):
                emu.attach_bios_store(app.state.bios_store)
            # Same wiring for the streamed runtime — PCSX2-streamed
            # resolves BIOS through self._bios; without this the
            # branch AttributeErrors on first launch.
            agsp = runtime_registry.get("agsp-streamed")
            if agsp is not None and hasattr(agsp, "attach_bios_store"):
                agsp.attach_bios_store(app.state.bios_store)

        uploads_adapter = UploadsAdapter(_sm_conn, blob_store)
        app.state.uploads_adapter = uploads_adapter
        register_adapter(uploads_adapter)

        # Bookmarks: saved external URLs (videos, articles) — no blob,
        # all state lives in file_index.source_metadata. Same adapter
        # protocol so trash/purge/audit work uniformly.
        from augmentum.vfs.adapters.bookmarks import BookmarksAdapter
        bookmarks_adapter = BookmarksAdapter(_sm_conn)
        app.state.bookmarks_adapter = bookmarks_adapter
        register_adapter(bookmarks_adapter)

        # Media-server rows (Audiobookshelf / Emby / ...). One adapter
        # object per provider slug so trash and source-chip filtering
        # work the same way every other source does. No blobs — streams
        # are proxied through /api/media/stream/{file_id} at play time.
        from augmentum.vfs.adapters.media_server import MediaServerAdapter
        for _slug in ("audiobookshelf", "emby", "jellyfin"):
            register_adapter(MediaServerAdapter(_slug, _sm_conn))

        # Gallery images + chat-attachment images. Without these adapters,
        # purging a trashed image from the Files panel drops the file_index
        # row but leaves image_generations / chat_images orphaned — the
        # Gallery→Files cascade is symmetric now. Images adapter lazily
        # grabs ImagePersistence at delete time so it doesn't depend on
        # the image subsystem already being up.
        from augmentum.vfs.adapters.chat_images import ChatImagesAdapter
        from augmentum.vfs.adapters.images import ImagesAdapter
        _img_out_dir = (
            getattr(settings, "image_output_dir", "")
            or f"{settings.data_dir}/image_output"
        )
        register_adapter(ImagesAdapter(_sm_conn, _img_out_dir))
        register_adapter(ChatImagesAdapter(_sm_conn))

        # Agentic artifacts + RAG documents. Both have native delete paths
        # that already cascade into file_index; these adapters close the
        # reverse direction so Files-panel trash-purge doesn't strand the
        # backing row + disk artifact. Stores are lazily constructed inside
        # each adapter so wiring order stays flexible.
        from augmentum.vfs.adapters.artifacts import ArtifactsAdapter
        from augmentum.vfs.adapters.documents import DocumentsAdapter
        register_adapter(ArtifactsAdapter(_sm_conn))
        if _sm_backend is not None:
            register_adapter(DocumentsAdapter(_sm_backend))

        # Save-to-Library publications. Storage root mirrors the
        # image_output / artifacts convention under data_dir. The
        # PublicationStore is stashed on app.state for the save routes
        # and is also wrapped in a VFS adapter so publications surface
        # in the unified Files panel.
        from augmentum.library.publications import LibraryStorage, PublicationStore
        from augmentum.vfs.adapters.library_published import LibraryPublishedAdapter
        _library_storage = LibraryStorage(
            Path(settings.data_dir) / "library_published"
        )
        _publication_store = PublicationStore(_sm_conn, _library_storage)
        app.state.publication_store = _publication_store
        register_adapter(LibraryPublishedAdapter(_sm_conn, _publication_store))

        vfs = VFS()
        artifact_dir = str(Path(settings.data_dir) / "artifacts")
        knowledge_dir = str(Path(settings.data_dir) / "knowledge")
        voices_dir = str(Path(settings.data_dir) / "voices")
        vfs.register_bridge(ArtifactBridge(_sm_conn, artifact_dir))
        vfs.register_bridge(ImageBridge(_sm_conn))
        vfs.register_bridge(DocumentBridge(_sm_conn))
        vfs.register_bridge(KnowledgeBridge(_sm_conn, knowledge_dir))
        vfs.register_bridge(VoiceBridge(_sm_conn, voices_dir))
        vfs.register_bridge(ChatImageBridge(_sm_conn))
        app.state.vfs = vfs

        # Unified thumbnail service — dispatch preview generation through
        # the adapter registry so gallery / Files / chat / comic surfaces
        # all share one cache and one URL shape. See vfs/thumbnails.py for
        # the producer contract.
        from augmentum.vfs import set_thumbnail_service
        from augmentum.vfs.thumbnails import ThumbnailService
        thumb_dir = str(Path(settings.data_dir) / "thumbs")
        thumbnail_service = ThumbnailService(thumb_dir)
        app.state.thumbnail_service = thumbnail_service
        set_thumbnail_service(thumbnail_service)

        # Populate file index for existing users (one-time), then reconcile
        # any rows that were stranded before source delete paths learned to
        # cascade into file_index, and finally backfill kind for any rows the
        # SQL classifier couldn't resolve from mime alone.
        from augmentum.vfs.populate import (
            backfill_kind,
            populate_from_existing,
            reconcile_orphaned_images,
            reconcile_private_images,
            reconcile_stranded,
            repair_real_paths,
        )
        async def _populate_index():
            try:
                # Populate only users who've never been indexed — we track
                # this in app_settings so purging trash until file_index is
                # empty doesn't cause the one-time backfill to re-run and
                # resurrect everything from the source tables.
                cursor = await _sm_conn.execute("SELECT id FROM users")
                users = [row[0] for row in await cursor.fetchall()]
                for uid in users:
                    marker_key = f"file_index.populated:{uid}"
                    cur = await _sm_conn.execute(
                        "SELECT value FROM app_settings WHERE key = ?", (marker_key,),
                    )
                    if (await cur.fetchone()):
                        continue
                    # Migration back-compat: if the user already has indexed
                    # rows, populate has run before — set the marker and skip.
                    cur = await _sm_conn.execute(
                        "SELECT 1 FROM file_index WHERE user_id = ? LIMIT 1", (uid,),
                    )
                    already_indexed = (await cur.fetchone()) is not None
                    if not already_indexed:
                        await populate_from_existing(file_index, _sm_conn, uid)
                    await _sm_conn.execute(
                        "INSERT OR REPLACE INTO app_settings (key, value) VALUES (?, ?)",
                        (marker_key, "1"),
                    )
                    await _sm_conn.commit()

                await reconcile_stranded(file_index, _sm_conn)
                await reconcile_private_images(_sm_conn)
                # Inverse sweep: pull non-private image_generations rows
                # into file_index when they're missing (covers the case
                # where the realtime register_file path was skipped or
                # bugged, leaving the gallery silently underpopulated).
                await reconcile_orphaned_images(file_index, _sm_conn)
                await repair_real_paths(_sm_conn)
                await backfill_kind(_sm_conn)
            except Exception:
                log.warning("file_index_init_failed", exc_info=True)
        _track_bg(_populate_index(), name="populate_file_index")

        # Start enrichment background loop
        if settings.files_enrichment_enabled:
            from augmentum.vfs.enrichment import enrichment_loop
            _track_bg(enrichment_loop(file_index, _sm_conn), name="files_enrichment")

        log.info("vfs_initialized")

        # --- WebDAV ---
        if settings.files_webdav_enabled:
            try:
                from a2wsgi import WSGIMiddleware

                from augmentum.vfs.webdav import create_webdav_app
                dav_wsgi = create_webdav_app(vfs)
                app.mount("/dav", WSGIMiddleware(dav_wsgi))
                log.info("webdav_mounted", path="/dav")
            except ImportError:
                log.warning("webdav_deps_missing", message="Install wsgidav and a2wsgi for WebDAV support")

    # Auto-register bundled audio providers (from Docker compose env vars)
    if isinstance(app.state.state_manager.backend, SQLiteBackend):
        await _auto_register_audio_providers(app.state.state_manager.backend.conn)

    # Seed recommended Kokoro voice blends (idempotent — skips existing)
    if isinstance(app.state.state_manager.backend, SQLiteBackend) and settings.tts_kokoro_builtin:
        try:
            from augmentum.voice.kokoro_tts import RECOMMENDED_BLENDS
            _conn = app.state.state_manager.backend.conn
            for blend in RECOMMENDED_BLENDS:
                await _conn.execute(
                    "INSERT OR IGNORE INTO voice_mixes (name, blend_spec, provider_id) "
                    "VALUES (?, ?, 'kokoro-builtin')",
                    (blend["name"], blend["spec"]),
                )
            await _conn.commit()
        except aiosqlite.OperationalError as exc:
            # voice_mixes may not exist yet on a brand-new DB before the
            # migration runner has caught up — seed is opportunistic and
            # will succeed on the next startup once migrations have run.
            log.debug("voice_mixes_seed_skipped", error=str(exc))

    # Seed bundled avatars (idempotent — skips existing, skips missing VRM files)
    if isinstance(app.state.state_manager.backend, SQLiteBackend):
        try:
            from augmentum.avatar.bundled import seed_bundled_avatars
            from augmentum.avatar.store import AvatarStore
            avatar_store = AvatarStore(app.state.state_manager.backend.conn)
            bundled_dir = os.path.join(getattr(settings, "data_dir", "/data"), "bundled-avatars")
            # Fallback to ui/lib/bundled-avatars for local dev (non-Docker)
            if not os.path.isdir(bundled_dir):
                bundled_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "ui", "lib", "bundled-avatars")
            seeded = await seed_bundled_avatars(avatar_store, bundled_dir)
            if seeded:
                log.info("bundled_avatars_seeded", count=seeded)
        except Exception:
            log.warning("bundled_avatar_seeding_failed", exc_info=True)

    # Initialize MCP client (if enabled)
    app.state.mcp_client = None
    if settings.mcp_enabled:
        try:
            import json

            from augmentum.mcp.bridge import register_mcp_tools
            from augmentum.mcp.client import MCPClientManager

            mcp_client = MCPClientManager()
            app.state.mcp_client = mcp_client

            # Auto-connect servers from config
            if settings.mcp_servers:
                try:
                    servers = json.loads(settings.mcp_servers)
                except json.JSONDecodeError:
                    log.warning("mcp_servers_invalid_json", raw=settings.mcp_servers)
                    servers = []
                for srv in servers:
                    try:
                        name = srv["name"]
                        if "url" in srv:
                            await mcp_client.connect_http(
                                name, srv["url"], headers=srv.get("headers"),
                            )
                        else:
                            await mcp_client.connect_stdio(
                                name, srv["command"], args=srv.get("args"),
                                env=srv.get("env"),
                            )
                        register_mcp_tools(
                            mcp_client, name, app.state.tool_registry,
                        )
                    except Exception:
                        log.warning(
                            "mcp_server_connect_failed",
                            server=srv.get("name", "?"),
                            exc_info=True,
                        )

            log.info("mcp_client_initialized")
        except Exception:
            log.warning("mcp_init_failed", exc_info=True)

    # Initialize agentic task store
    app.state.task_store = None
    app.state.tool_call_cache = None
    app.state.build_run_store = None
    if isinstance(app.state.state_manager.backend, SQLiteBackend):
        try:
            from augmentum.modes.agentic.task_state import TaskStore, ToolCallCache

            app.state.task_store = TaskStore(sqlite_backend.conn)
            # Cache shares the SQLite connection — agentic resume replays
            # successful tool calls from this table instead of rerunning
            # (web search, image gen, artifact creation are all expensive).
            app.state.tool_call_cache = ToolCallCache(sqlite_backend.conn)
            log.info("task_store_initialized")
        except Exception:
            log.warning("task_store_init_failed", exc_info=True)

        try:
            from augmentum.builds import BuildRunStore

            app.state.build_run_store = BuildRunStore(sqlite_backend.conn)
            interrupted = await app.state.build_run_store.mark_running_interrupted(
                reason="Build interrupted by server restart before completion.",
            )
            log.info("build_run_store_initialized")
            if interrupted:
                log.warning("build_run_store_interrupted_running", count=interrupted)
        except Exception:
            log.warning("build_run_store_init_failed", exc_info=True)

    # Initialize device substrate (TVs, lights, sensors, internal surfaces)
    app.state.device_registry = None
    if isinstance(app.state.state_manager.backend, SQLiteBackend):
        try:
            from augmentum.devices.cast_tokens import CastTokenStore
            from augmentum.devices.events import EventBus
            from augmentum.devices.registry import DeviceRegistry
            from augmentum.devices.sessions import SessionRuntime
            from augmentum.devices.store import (
                DevicePairingStore,
                DevicePlayHistoryStore,
                DeviceStore,
            )

            _conn = sqlite_backend.conn
            _cast_token_store = CastTokenStore()
            app.state.cast_token_store = _cast_token_store
            _device_registry = DeviceRegistry(
                device_store=DeviceStore(_conn),
                pairing_store=DevicePairingStore(_conn),
                history_store=DevicePlayHistoryStore(_conn),
                sessions=SessionRuntime(),
                bus=EventBus(),
                http_client=app.state.http_client,
                cast_token_store=_cast_token_store,
            )

            # Register drivers. Each driver implements one wire protocol
            # and declares which capabilities it supports — adding a new
            # device kind is one driver, zero changes elsewhere.
            from augmentum.devices.drivers.cast_custom import CastCustomDriver
            from augmentum.devices.drivers.dlna import DlnaDriver
            from augmentum.devices.drivers.emby_remote import EmbyRemoteDriver
            from augmentum.media.store import MediaServerStore

            _device_registry.register_driver(
                DlnaDriver(http_client=app.state.http_client),
            )

            # Google Cast — server-side sender via pychromecast. The
            # augmentum container is the orchestrator: it runs mDNS
            # discovery, holds TLS connections to Cast devices, and
            # dispatches LOAD / transport / volume commands. Phone and
            # laptop senders just hit the augmentum API; off-network
            # casts work because nothing requires the user's browser
            # to be on the same LAN as the TV.
            _device_registry.register_driver(CastCustomDriver())

            # Emby/Jellyfin session bridge — pulls TVs that those
            # servers have already discovered on the LAN, surfaces them
            # as devices, routes playback through their session
            # controller. Sidesteps Docker networking entirely; the only
            # connection augmentum needs to make is to Emby itself
            # (which is already configured).
            def _provider_client_factory(provider: str, http_client):
                # Late import to avoid circular dependency with
                # augmentum.proxy.media_routes which itself imports
                # augmentum.devices on app start.
                from augmentum.proxy.media_routes import _provider_client
                return _provider_client(provider, http_client)

            _device_registry.register_driver(EmbyRemoteDriver(
                http_client=app.state.http_client,
                media_server_store_factory=lambda: MediaServerStore(_conn),
                file_index_factory=lambda: app.state.file_index,
                provider_client_factory=_provider_client_factory,
            ))

            await _device_registry.start()
            app.state.device_registry = _device_registry
            log.info(
                "device_registry_initialized",
                drivers=[d.id for d in _device_registry.list_drivers()],
            )
        except Exception:
            log.warning("device_registry_init_failed", exc_info=True)

    # ── Cast render output + HTML renderer ────────────────────────
    # Store is always present (single-machine grace — no chrome needed
    # to serve previously stored outputs). HTMLRenderer is created lazily
    # only when find_chromium() locates a binary; first render call
    # starts the Chrome subprocess on demand.
    app.state.render_output_store = None
    app.state.html_renderer = None
    try:
        from augmentum.cast.output_store import RenderOutputStore
        app.state.render_output_store = RenderOutputStore()
        log.info("render_output_store_initialized")
    except Exception:
        log.warning("render_output_store_init_failed", exc_info=True)

    try:
        from augmentum.cast.html_renderer import HTMLRenderer
        from augmentum.tools.application_cdp import find_chromium
        chromium = find_chromium()
        if chromium:
            # Constructed but not started — start() fires on first
            # render call. Keeps idle resource footprint at zero.
            app.state.html_renderer = HTMLRenderer(chromium_path=chromium)
            log.info("html_renderer_configured", chromium=chromium)
        else:
            log.info("html_renderer_skipped_no_chromium")
    except Exception:
        log.warning("html_renderer_init_failed", exc_info=True)

    # ── Cast receiver registry (WebSocket sessions to TVs/receivers) ─
    # Backed by trusted_receivers + receiver_cast_events for durable
    # per-device identity + audit log. Both stores share the main
    # SQLite connection; the registry uses them on event paths.
    app.state.receiver_registry = None
    app.state.trusted_receiver_store = None
    app.state.cast_event_store = None
    try:
        from augmentum.cast.cast_events import CastEventStore
        from augmentum.cast.receiver_registry import ReceiverRegistry
        from augmentum.cast.trusted_receivers import TrustedReceiverStore
        # NOTE: SQLiteBackend is imported at module scope (line 27). A local
        # `from ... import SQLiteBackend` here would make Python treat the
        # name as a function-local across the entire lifespan body, breaking
        # the SQLiteBackend(db_path) construction much earlier at line 1878
        # with UnboundLocalError.
        sm = app.state.state_manager
        # Stopper for render-stream containers spawned by /api/cast/
        # render-stream/start. Wired into the receiver registry so
        # detach() + surface_closed reap orphan agsp-* containers
        # instead of leaving them running until reboot. None when the
        # game_stream runtime didn't come up — registry tolerates it.
        gs_runtime = getattr(app.state, "game_stream_runtime", None)
        stream_session_stopper = None
        if gs_runtime is not None:
            async def _stop_render_session(sid: str, uid: str) -> None:
                await gs_runtime.stop_session(
                    sid, user_id=uid, reason="cast_disconnected",
                )
            stream_session_stopper = _stop_render_session
        if isinstance(sm.backend, SQLiteBackend):
            trusted_store = TrustedReceiverStore(sm.backend.conn)
            event_store = CastEventStore(sm.backend.conn)
            app.state.trusted_receiver_store = trusted_store
            app.state.cast_event_store = event_store
            app.state.receiver_registry = ReceiverRegistry(
                trusted_store=trusted_store,
                event_store=event_store,
                stream_session_stopper=stream_session_stopper,
            )
            log.info("cast_receiver_registry_initialized", durable=True)
        else:
            app.state.receiver_registry = ReceiverRegistry(
                stream_session_stopper=stream_session_stopper,
            )
            log.warning(
                "cast_receiver_registry_initialized",
                durable=False,
                reason="non-sqlite backend; trusted/events not persisted",
            )
    except Exception:
        log.warning("cast_receiver_registry_init_failed", exc_info=True)

    # ── Cast input registry (phone↔container gamepad routing) ───────
    # Wired regardless of backend — it's RAM-only. Failure to import
    # falls back to None and /api/cast/input/* endpoints return 503.
    app.state.cast_input_registry = None
    try:
        from augmentum.cast.input_bridge import CastInputRegistry
        app.state.cast_input_registry = CastInputRegistry()
        log.info("cast_input_registry_initialized")
    except Exception:
        log.warning("cast_input_registry_init_failed", exc_info=True)

    # ── Cast game profile registry + classifier (per-(user, title)) ──
    # SQLite-backed. Routes return 503 when the backend isn't sqlite or
    # when init failed — callers fall back to the historical default
    # (same-origin strategy + gamepad_api adapter).
    app.state.cast_profile_registry = None
    app.state.cast_classifier = None
    app.state.cast_telemetry_demoter = None
    try:
        from augmentum.cast.games.classifier import CastClassifier
        from augmentum.cast.games.registry import CastProfileRegistry
        from augmentum.cast.games.telemetry import TelemetryDemoter
        if isinstance(sm.backend, SQLiteBackend):
            cast_profile_registry = CastProfileRegistry(sm.backend.conn)
            app.state.cast_profile_registry = cast_profile_registry
            app.state.cast_classifier = CastClassifier(
                profile_registry=cast_profile_registry,
            )
            # Phase 4 (telemetry half): consumes the loader's
            # input_telemetry surface_events + demotes under-reaching
            # strategies. The classifier's read-side already promotes on
            # a recent failed_at; this is what writes it.
            app.state.cast_telemetry_demoter = TelemetryDemoter(
                profile_registry=cast_profile_registry,
            )
            log.info("cast_profile_registry_initialized", durable=True)
        else:
            log.warning(
                "cast_profile_registry_skipped",
                reason="non-sqlite backend",
            )
    except Exception:
        log.warning("cast_profile_registry_init_failed", exc_info=True)

    # ── Cast origin-proxy substrate (Strategy 2) ────────────────────
    # RAM-only session store + on-disk asset cache + httpx fetcher.
    # Wired regardless of backend. The OriginProxyStrategy was already
    # registered with the strategy_registry at module-import time; we
    # attach the session store to it here so can_handle starts
    # returning True (until then it returns False so the classifier
    # falls through to the cheap shim).
    app.state.cast_proxy_session_store = None
    app.state.cast_proxy_fetcher = None
    try:
        from pathlib import Path as _Path

        from augmentum.cast.games.proxy.fetcher import (
            AssetCache,
            ProxyFetcher,
        )
        from augmentum.cast.games.proxy.session_store import ProxySessionStore
        from augmentum.cast.games.strategies import strategy_registry as _strat_reg

        proxy_session_store = ProxySessionStore()
        cache_root = _Path("/data/cast-game-cache")
        asset_cache = AssetCache(cache_root)
        proxy_fetcher = ProxyFetcher(cache=asset_cache)

        app.state.cast_proxy_session_store = proxy_session_store
        app.state.cast_proxy_fetcher = proxy_fetcher
        # Promote the proxy strategy from "registered but inert" to
        # "actively serving" by attaching the live session store.
        proxy_strategy = _strat_reg.get("proxy")
        if proxy_strategy is not None and hasattr(proxy_strategy, "attach_session_store"):
            proxy_strategy.attach_session_store(proxy_session_store)
        log.info("cast_proxy_substrate_initialized", cache_root=str(cache_root))
    except Exception:
        log.warning("cast_proxy_substrate_init_failed", exc_info=True)

    # ── Cast probe coordinator (Phase 4, proactive half) ─────────────
    # Fire-and-forget headless probe on the first cast of an unknown
    # title → pre-classifies its input_chain so the first cast is right.
    # Strictly additive: degrades to no-op when Playwright/Chromium isn't
    # in the image. server_origin left empty → probe contributes the
    # input_chain only; strategy escalation stays with the telemetry
    # demotion loop (safe default).
    app.state.cast_probe_coordinator = None
    try:
        cast_profile_registry = getattr(app.state, "cast_profile_registry", None)
        if cast_profile_registry is not None:
            from augmentum.cast.games.probe import (
                CastProbeCoordinator,
                PlaywrightProbe,
            )
            from augmentum.cast.games.strategies import (
                strategy_registry as _probe_strat_reg,
            )
            app.state.cast_probe_coordinator = CastProbeCoordinator(
                probe=PlaywrightProbe(),
                profile_registry=cast_profile_registry,
                strategy_registry=_probe_strat_reg,
                server_origin="",
            )
            log.info("cast_probe_coordinator_initialized")
    except Exception:
        log.warning("cast_probe_coordinator_init_failed", exc_info=True)

    # ── Cast pair-token store (QR-scan auth bootstrap) ───────────────
    app.state.pair_store = None
    try:
        from augmentum.cast.pair_store import PairStore
        app.state.pair_store = PairStore()
        log.info("cast_pair_store_initialized")
    except Exception:
        log.warning("cast_pair_store_init_failed", exc_info=True)

    # --- Mobile pair-token store (Android QR auth bootstrap) ---------------
    app.state.mobile_pair_store = None
    try:
        from augmentum.auth.mobile_pairing import MobilePairStore
        app.state.mobile_pair_store = MobilePairStore()
        log.info("mobile_pair_store_initialized")
    except Exception:
        log.warning("mobile_pair_store_init_failed", exc_info=True)

    # ── Cast invite-token store (couch co-op join tokens) ────────────
    # RAM-only mirror of pair_store. Tokens grant short-lived
    # "join this session" auth to guest phones holding them. See
    # augmentum/cast/invite_store.py.
    app.state.cast_invite_store = None
    try:
        from augmentum.cast.invite_store import InviteStore
        app.state.cast_invite_store = InviteStore()
        log.info("cast_invite_store_initialized")
    except Exception:
        log.warning("cast_invite_store_init_failed", exc_info=True)

    # ── Guest profile store (couch co-op Phase 2) ────────────────────
    # SQLite-backed — guests are persistent identities under the host's
    # account. Only wired when the SQLite backend is in use; the memory
    # backend test path skips it (route handlers return 503).
    app.state.guest_store = None
    app.state.guest_device_store = None
    if isinstance(getattr(sqlite_backend, "conn", None), object) and \
            getattr(sqlite_backend, "conn", None) is not None:
        try:
            from augmentum.state.guest_store import GuestStore
            app.state.guest_store = GuestStore(sqlite_backend.conn)
            log.info("guest_store_initialized")
        except Exception:
            log.warning("guest_store_init_failed", exc_info=True)
        # Phase 3 device fingerprint store. Same conn — device row
        # lookups happen alongside profile lookups so co-locating
        # them on a single aiosqlite avoids a second connection.
        try:
            from augmentum.state.guest_device_store import GuestDeviceStore
            app.state.guest_device_store = GuestDeviceStore(
                sqlite_backend.conn,
            )
            log.info("guest_device_store_initialized")
        except Exception:
            log.warning("guest_device_store_init_failed", exc_info=True)

    # ── Stream-auth redeem store (server-side render Chrome cookie) ──
    # One-shot tokens minted by /api/cast/render-stream/start that
    # the rendering container's Chrome consumes once to pick up the
    # user's session cookie. See augmentum/cast/stream_auth_redeem.py.
    app.state.stream_auth_redeem_store = None
    try:
        from augmentum.cast.stream_auth_redeem import StreamAuthRedeemStore
        app.state.stream_auth_redeem_store = StreamAuthRedeemStore()
        log.info("cast_stream_auth_redeem_store_initialized")
    except Exception:
        log.warning("cast_stream_auth_redeem_store_init_failed", exc_info=True)

    # ── Coder-preview origin isolation stores ────────────────────────
    # Token store: one-time pvt_* tokens minted by the main app and
    # consumed by the isolated preview origin on iframe-mount.
    # Session store: sliding-TTL pvs_* cookie values bound to the
    # isolated origin (scoped to its host:port so they never reach
    # Augmentum's main /api/*). See preview_auth.py + the design spec
    # at docs/superpowers/specs/2026-05-27-preview-origin-isolation-design.md.
    app.state.preview_token_store = None
    app.state.preview_session_store = None
    try:
        from augmentum.coder.preview_auth import (
            PreviewSessionStore,
            PreviewTokenStore,
        )
        app.state.preview_token_store = PreviewTokenStore(
            default_ttl_s=float(settings.coder_preview_token_ttl_seconds),
        )
        app.state.preview_session_store = PreviewSessionStore(
            sliding_ttl_s=float(settings.coder_preview_session_ttl_seconds),
        )
        log.info(
            "coder_preview_auth_stores_initialized",
            isolation_enabled=settings.coder_preview_isolation_enabled,
            isolated_port=settings.coder_preview_isolated_port,
        )
    except Exception:
        log.warning("coder_preview_auth_stores_init_failed", exc_info=True)

    # ── Voice fanout (multi-target voice render targeting) ───────────
    # Lets the existing /ws/voice WS mirror its emits to additional
    # consumers — TV-cast receivers, future renderers — without
    # touching pipeline.py. See augmentum/voice/fanout.py.
    app.state.voice_fanout = None
    try:
        from augmentum.voice.fanout import VoiceFanout
        app.state.voice_fanout = VoiceFanout()
        log.info("voice_fanout_initialized")
    except Exception:
        log.warning("voice_fanout_init_failed", exc_info=True)

    # Initialize surface sessions (paired browsers/TVs/phones/control planes).
    app.state.surface_store = None
    app.state.surface_runtime = None
    app.state.surface_access_token_store = None
    if isinstance(app.state.state_manager.backend, SQLiteBackend):
        try:
            from augmentum.surfaces import (
                SurfaceAccessTokenStore,
                SurfaceRuntime,
                SurfaceStore,
            )

            app.state.surface_store = SurfaceStore(sqlite_backend.conn)
            app.state.surface_runtime = SurfaceRuntime()
            app.state.surface_access_token_store = SurfaceAccessTokenStore()
            log.info("surface_sessions_initialized")
        except Exception:
            log.warning("surface_sessions_init_failed", exc_info=True)

    # Initialize artifact storage and tools
    app.state.artifact_store = None
    if isinstance(app.state.state_manager.backend, SQLiteBackend):
        try:
            from augmentum.tools.artifact_storage import ArtifactStore

            artifact_store = ArtifactStore(sqlite_backend.conn)
            app.state.artifact_store = artifact_store

            # Register artifact tools
            from augmentum.tools.artifact_chart import ChartTool
            from augmentum.tools.artifact_document import DocumentTool
            from augmentum.tools.artifact_presentation import PresentationTool
            from augmentum.tools.artifact_spreadsheet import SpreadsheetTool

            app.state.tool_registry.register(DocumentTool(artifact_store))
            app.state.tool_registry.register(PresentationTool(artifact_store))
            app.state.tool_registry.register(SpreadsheetTool(artifact_store))
            app.state.tool_registry.register(ChartTool(artifact_store))

            # Unified primitive layer (Phase 1) — artifact conversion +
            # postprocess tools. Same logic the Artifact Studio HTTP
            # routes use; surfacing as Tools makes them LLM/voice-callable.
            from augmentum.tools.background_remove import BackgroundRemoveTool
            from augmentum.tools.document_convert import DocumentConvertTool
            from augmentum.tools.image_convert import ImageConvertTool

            app.state.tool_registry.register(ImageConvertTool(artifact_store))
            app.state.tool_registry.register(BackgroundRemoveTool(artifact_store))
            app.state.tool_registry.register(DocumentConvertTool(artifact_store))

            from augmentum.tools.artifact_ebook import EbookTool

            app.state.tool_registry.register(EbookTool(artifact_store, app_state=app.state))

            # ATP browser tools — agent-browser sidecar verbs for external
            # harnesses (/v1/tools). ATP-only surfaces; health-gated on
            # the sidecar container, so registering unconditionally is safe.
            try:
                from augmentum.tools.browser_tools import ATP_BROWSER_TOOL_CLASSES
                for _browser_cls in ATP_BROWSER_TOOL_CLASSES:
                    app.state.tool_registry.register(_browser_cls(app.state))
            except Exception:
                log.warning("atp_browser_tools_registration_failed", exc_info=True)

            # ATP vision/OCR tools — text-only harness models describe or
            # OCR the screenshots the browser tools produce. Same ATP-only
            # surface + health-gating story as the browser tools.
            try:
                from augmentum.tools.vision_tools import ATP_VISION_TOOL_CLASSES
                for _vision_cls in ATP_VISION_TOOL_CLASSES:
                    app.state.tool_registry.register(_vision_cls(app.state))
            except Exception:
                log.warning("atp_vision_tools_registration_failed", exc_info=True)

            # ATP memory_store — explicit "remember this" from any harness,
            # staged through the human-gated harvest pipeline (never a live
            # memory write).
            try:
                from augmentum.tools.harness_memory import HarnessMemoryStoreTool
                app.state.tool_registry.register(HarnessMemoryStoreTool(app.state))
            except Exception:
                log.warning("atp_memory_store_registration_failed", exc_info=True)

            # ATP pack_search / sandbox_shell / agent-bridge tools — reuse
            # of existing substrates (knowledge packs, coder workspaces,
            # notifications). ATP-only surfaces, health-gated.
            try:
                from augmentum.proxy.agent_bridge import register_bridge_action_handler
                from augmentum.tools.agent_bridge_tools import ATP_BRIDGE_TOOL_CLASSES
                from augmentum.tools.pack_search_atp import AtpPackSearchTool
                from augmentum.tools.sandbox_tools import SandboxShellTool
                app.state.tool_registry.register(AtpPackSearchTool())
                app.state.tool_registry.register(SandboxShellTool(app.state))
                for _bridge_cls in ATP_BRIDGE_TOOL_CLASSES:
                    app.state.tool_registry.register(_bridge_cls(app.state))
                register_bridge_action_handler()
            except Exception:
                log.warning("atp_bridge_tools_registration_failed", exc_info=True)

            # ATP flow_status — poll tool for background flow tasks
            # (flow_deep_research etc.) so external harnesses can retrieve
            # results that chat surfaces get via context injection.
            try:
                from augmentum.tools.flow_status import FlowStatusTool
                app.state.tool_registry.register(FlowStatusTool(app.state))
            except Exception:
                log.warning("atp_flow_status_registration_failed", exc_info=True)

            # ATP recipes — named per-user macros that replay a sequence of
            # ATP tool calls in one shot (kills repeated tool choreography
            # like ensure_auth -> navigate -> screenshot). Steps run through
            # the same registry + context injection, so isolation holds.
            try:
                from augmentum.tools.recipe_tool import AtpRecipeTool
                app.state.tool_registry.register(AtpRecipeTool(app.state))
            except Exception:
                log.warning("atp_recipe_registration_failed", exc_info=True)

            # ATP workflow — self-minted soft procedural memory (Hermes/AWM
            # style). The model saves a playbook that worked and refines it;
            # matches auto-surface into the harness briefing by FTS trigger.
            try:
                from augmentum.tools.workflow_tool import WorkflowTool
                app.state.tool_registry.register(WorkflowTool(app.state))
            except Exception:
                log.warning("atp_workflow_registration_failed", exc_info=True)

            # Image search tool — stores downloaded images as artifacts,
            # NOT in the image gallery (image_generations table).
            from augmentum.tools.image_search import ImageSearchTool

            app.state.tool_registry.register(ImageSearchTool(
                http_client=app.state.http_client,
                artifact_store=artifact_store,
            ))

            # Lightweight export tools (markdown, CSV, code files)
            from augmentum.tools.export_tools import (
                CodeExportTool,
                CsvExportTool,
                MarkdownExportTool,
            )

            app.state.tool_registry.register(MarkdownExportTool(artifact_store))
            app.state.tool_registry.register(CsvExportTool(artifact_store))
            app.state.tool_registry.register(CodeExportTool(artifact_store))

            # Offer substrate — `propose_offer` is the chat LLM's
            # entrypoint for surfacing install/save/switch chips.
            # Catalog kinds register themselves on import (see
            # augmentum/offers/catalog/__init__.py); the system.offer
            # action handler binds at offers_routes.py import time.
            from augmentum.tools.propose_offer import ProposeOfferTool

            app.state.tool_registry.register(ProposeOfferTool(app.state))

            # Standing-tasks: schedule_briefing wraps standing_tasks.add_task
            # for natural-language briefing setup ("wake me at 9 with X").
            # list_briefings + cancel_briefing close the lifecycle loop so
            # the user doesn't have to open the topics modal once a
            # briefing is in place. See standing_tasks.py for the engine.
            from augmentum.tools.manage_briefings import (
                CancelBriefingTool,
                ListBriefingsTool,
            )
            from augmentum.tools.notify import NotifyTool
            from augmentum.tools.recommend_now import RecommendNowTool
            from augmentum.tools.schedule_action import ScheduleActionTool
            from augmentum.tools.schedule import ScheduleTool
            from augmentum.tools.schedule_briefing import ScheduleBriefingTool
            from augmentum.tools.schedule_deadline import ScheduleDeadlineTool
            from augmentum.tools.schedule_reminder import ScheduleReminderTool
            from augmentum.tools.schedule_request import ScheduleRequestTool
            from augmentum.tools.watch_for import WatchForTool

            # Unified single entrypoint (one tools-panel button) that routes to
            # the specialized tools below. The individuals stay registered for
            # voice/companion; the chat panel shows only `schedule`.
            app.state.tool_registry.register(ScheduleTool(app.state))
            app.state.tool_registry.register(ScheduleActionTool(app.state))
            app.state.tool_registry.register(ScheduleBriefingTool(app.state))
            app.state.tool_registry.register(ScheduleDeadlineTool(app.state))
            app.state.tool_registry.register(ScheduleRequestTool(app.state))
            app.state.tool_registry.register(ListBriefingsTool(app.state))
            app.state.tool_registry.register(CancelBriefingTool(app.state))
            # Companion verbs architecture — Phase 4 core verbs.
            app.state.tool_registry.register(NotifyTool(app.state))
            app.state.tool_registry.register(ScheduleReminderTool(app.state))
            app.state.tool_registry.register(WatchForTool(app.state))
            app.state.tool_registry.register(RecommendNowTool(app.state))

            # Language-partner tools — surfaced to character cards seeded
            # by augmentum/learning/partners.py. Registered globally so any
            # character (not just partners) can call them; per-card tool
            # filtering uses the card's `toolAllowlist`. Empty if neither
            # vocab_store nor pack_manager is wired (e.g. learning system
            # disabled at build time).
            from augmentum.tools.language_partner import all_language_partner_tools
            for _lp_tool in all_language_partner_tools(app.state):
                app.state.tool_registry.register(_lp_tool)

            # Reference Resolver — surfaces hybrid retrieval across
            # file_index + companion_journal so the model can resolve
            # natural-language references like "that manga with the
            # quintessential quintuplets". Available to every mode that
            # composes against the tool_registry. CompanionMemory is
            # optional — falls through to file-only retrieval when the
            # runtime isn't initialized (e.g. feature flag off).
            from augmentum.tools.resolve_moments import ResolveMomentsTool
            _resolver_memory = None
            _cr = getattr(app.state, "companion_runtime", None)
            if _cr is not None:
                _resolver_memory = getattr(_cr, "memory", None)
            app.state.tool_registry.register(
                ResolveMomentsTool(
                    file_index=getattr(app.state, "file_index", None),
                    memory=_resolver_memory,
                )
            )

            from augmentum.tools.artifact_application import ApplicationBuilderTool

            # Pinned resolution: resolve the user's model once per build, reuse for
            # all pipeline calls.  Cleared when a new build starts with a different model.
            _pinned_model: str = ""
            _pinned_backend = None
            _pinned_clean: str = ""

            async def _app_builder_call_llm(messages: list, max_tokens: int = 4096, model: str = "", grammar: str = "") -> str:
                """Internal LLM caller for application builder pipeline.

                Uses the specified model (from the original user request)
                so internal calls go to the same backend/model the user chose.
                Load balancers are resolved once and pinned for the pipeline.
                """
                nonlocal _pinned_model, _pinned_backend, _pinned_clean
                import asyncio as _aio

                from augmentum.models.base import InternalChatRequest, Message
                # Resolve model — never send empty model to a backend
                effective_model = model if model and model != "default" else ""
                if effective_model:
                    # If model changed (new build with different selection), re-resolve
                    if effective_model != _pinned_model:
                        try:
                            _pinned_backend, _pinned_clean = await app.state.provider_registry.resolve_backend_with_fabric(effective_model)
                            _pinned_model = effective_model
                            log.info("app_builder.model_pinned", requested=effective_model, resolved=_pinned_clean)
                        except Exception:
                            log.warning("app_builder.model_resolve_failed", model=effective_model, exc_info=True)
                            try:
                                _pinned_backend, _pinned_clean = await app.state.provider_registry.resolve_backend_with_fabric("")
                            except Exception:
                                _pinned_backend, _pinned_clean = None, ""
                            _pinned_model = ""
                    backend_ = _pinned_backend
                    clean_ = _pinned_clean
                else:
                    # No model specified — prefer the user's primary chat
                    # model, then the default backend, then any remaining backend.
                    clean_ = ""
                    backend_ = None
                    try:
                        backend_, clean_ = await app.state.provider_registry.resolve_backend_with_fabric("")
                    except Exception:
                        log.debug("app_builder_auto_model_resolve_failed", exc_info=True)
                    # If primary/default resolution returned empty, try all backends.
                    if not clean_:
                        if not backend_:
                            backend_ = app.state.provider_registry.default_backend
                        for bk_name, bk in app.state.provider_registry.backends.items():
                            if bk is backend_:
                                continue
                            # Don't auto-pick a routing-only backend (the
                            # secondary engine slot) as a general default.
                            if app.state.provider_registry.is_listing_excluded(bk_name) is True:
                                continue
                            try:
                                models = await bk.list_models()
                                if models and models[0].name:
                                    backend_ = bk
                                    clean_ = models[0].name
                                    break
                            except Exception as exc:
                                log.debug("backend_list_models_failed", backend=bk_name, error=str(exc))
                                continue
                    if not clean_:
                        log.warning("app_builder.no_model_resolved", hint="Set a model in the chat before building")
                # Enable cache_prompt for llama.cpp — reuses KV cache for shared
                # prompt prefixes across pipeline calls, saving 15-25s per build.
                # Grammar constraints force valid output format on structured passes.
                raw_opts = {}
                keep_alive = None
                from augmentum.models.llama_cpp import LlamaCppBackend
                from augmentum.models.ollama import OllamaBackend
                if isinstance(backend_, LlamaCppBackend):
                    raw_opts["cache_prompt"] = True
                    if grammar:
                        raw_opts["grammar"] = grammar
                elif isinstance(backend_, OllamaBackend):
                    # Override Ollama's 2048 default context for pipeline calls
                    raw_opts["num_ctx"] = max(getattr(settings, 'app_builder_max_tokens', 8192) * 2, 8192)
                    # Keep model loaded for entire pipeline duration (default 5min is too short)
                    keep_alive = "30m"
                req_ = InternalChatRequest(
                    model=clean_,
                    messages=[Message(role=m["role"], content=m["content"]) for m in messages],
                    stream=False,
                    max_tokens=max_tokens,
                    raw_options=raw_opts,
                    keep_alive=keep_alive,
                )
                # Retry with backoff — handles transient failures like model
                # being unloaded by LM Studio when another request loads a
                # different model.  Wait + retry lets the backend reload.
                # Timeout is configurable; bare asyncio.TimeoutError has empty
                # str() so we explicitly log the exception type and wrap the
                # terminal raise with a meaningful message for the UI.
                llm_timeout = getattr(settings, "app_builder_llm_timeout_seconds", 600)
                last_err: Exception | None = None
                for attempt in range(3):
                    try:
                        resp_ = await _aio.wait_for(backend_.chat(req_), timeout=llm_timeout)
                        break
                    except Exception as _retry_err:
                        last_err = _retry_err
                        err_type = type(_retry_err).__name__
                        err_msg = str(_retry_err)[:200] or "(no message)"
                        if attempt < 2:
                            wait = (attempt + 1) * 5  # 5s, 10s
                            log.warning("app_builder.llm_retry",
                                        attempt=attempt + 1, wait=wait,
                                        error_type=err_type, error=err_msg,
                                        timeout=llm_timeout, model=clean_)
                            await _aio.sleep(wait)
                        else:
                            log.warning("app_builder.llm_exhausted",
                                        error_type=err_type, error=err_msg,
                                        timeout=llm_timeout, model=clean_)
                            final_msg = str(last_err) or f"{err_type} after {llm_timeout}s × 3 attempts"
                            raise RuntimeError(
                                f"LLM call failed for {clean_ or '(no model)'}: {err_type}: {final_msg}"
                            ) from last_err
                content = resp_.message.content or ""
                # Reasoning model fallback: if content is empty but reasoning has the code
                if not content.strip() and hasattr(resp_.message, 'reasoning_content') and resp_.message.reasoning_content:
                    content = resp_.message.reasoning_content
                # Store last usage on the function for the pipeline to read
                usage = resp_.usage if hasattr(resp_, 'usage') else None
                if not content.strip():
                    log.warning(
                        "app_builder.llm_empty_response",
                        model=clean_,
                        requested_model=effective_model,
                        grammar=bool(grammar),
                        finish_reason=getattr(resp_, "finish_reason", ""),
                        prompt_tokens=usage.prompt_tokens if usage else 0,
                        completion_tokens=usage.completion_tokens if usage else 0,
                        max_tokens=max_tokens,
                        messages=len(messages),
                    )
                _app_builder_call_llm._last_usage = {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                }
                return content

            def _app_builder_settings() -> dict:
                return {
                    "app_builder_improve_pass": settings.app_builder_improve_pass,
                    "app_builder_max_improve_iterations": settings.app_builder_max_improve_iterations,
                    "app_builder_max_fix_iterations": settings.app_builder_max_fix_iterations,
                    "app_builder_auto_preview": settings.app_builder_auto_preview,
                    "app_builder_max_tokens": settings.app_builder_max_tokens,
                    "app_builder_use_browser_verify": getattr(settings, "app_builder_use_browser_verify", False),
                    "app_builder_batch_small_apps": getattr(settings, "app_builder_batch_small_apps", True),
                    "app_builder_pipeline_v2": getattr(settings, "app_builder_pipeline_v2", False),
                    "app_builder_llm_timeout_seconds": getattr(settings, "app_builder_llm_timeout_seconds", 600),
                }

            app.state.tool_registry.register(
                ApplicationBuilderTool(artifact_store, _app_builder_call_llm, _app_builder_settings, app_state=app.state)
            )
            log.info("app_builder.registered")

            # Draft section tool — parallel document section generation
            from augmentum.tools.draft_section import DraftSectionTool

            app.state.tool_registry.register(DraftSectionTool(
                backend=app.state.provider_registry.default_backend,
                provider_registry=app.state.provider_registry,
            ))
            log.info("artifact_tools_registered")

            # Unified primitive layer (Phase 1): bind HTTP routes for
            # every Tool that declared ``surfaces.http_route``. Idempotent;
            # tools registered later (or with already-existing routes)
            # are skipped.
            try:
                from augmentum.tools.auto_routes import register_tool_routes
                _bound = register_tool_routes(app, app.state.tool_registry)
                if _bound:
                    log.info("tool_auto_routes_bound", count=len(_bound), routes=_bound)
            except Exception:
                log.warning("tool_auto_routes_failed", exc_info=True)
        except Exception:
            log.warning("artifact_store_init_failed", exc_info=True)

    # Initialize reasoning flow store
    app.state.flow_store = None
    if isinstance(app.state.state_manager.backend, SQLiteBackend):
        try:
            from augmentum.reasoning.store import FlowStore

            app.state.flow_store = FlowStore(sqlite_backend.conn)
            await app.state.flow_store.seed_builtins()
        except Exception:
            log.warning("reasoning_flow_store_init_failed", exc_info=True)

    # Initialize custom flow store (tool chain flows)
    app.state.custom_flow_store = None
    if isinstance(app.state.state_manager.backend, SQLiteBackend):
        try:
            from augmentum.tools.custom_flows import CustomFlowStore

            app.state.custom_flow_store = CustomFlowStore(sqlite_backend.conn)
            # seed_defaults() is per-user now (Tier 0 multi-tenant rollout); the
            # /api/flows/list route triggers it lazily for the calling user.
        except Exception:
            log.warning("custom_flow_store_init_failed", exc_info=True)

    # Initialize repo-local Power registry (native packs + SKILL.md compat)
    app.state.power_registry = None
    try:
        from augmentum.powers import PowerRegistry

        app.state.power_registry = PowerRegistry()
    except Exception:
        log.warning("power_registry_init_failed", exc_info=True)

    # Initialize background chain manager
    app.state.background_chain_manager = None
    if settings.passthrough_chain_bg_enabled:
        try:
            from augmentum.tools.background_chain import BackgroundChainManager

            app.state.background_chain_manager = BackgroundChainManager(
                max_per_session=settings.passthrough_chain_bg_max_per_session,
                max_total=settings.passthrough_chain_bg_max_total,
                provider_registry=app.state.provider_registry,
            )
        except Exception:
            log.warning("background_chain_manager_init_failed", exc_info=True)

    # Register flow tools (flows as callable tools for the LLM)
    if (
        app.state.custom_flow_store
        and app.state.background_chain_manager
        and getattr(app.state, "tool_registry", None)
        and getattr(app.state, "provider_registry", None)
    ):
        try:
            from augmentum.proxy.handler_factory import register_flow_tools_async

            default_backend = app.state.provider_registry.default_backend
            if default_backend:
                flow_tool_count = await register_flow_tools_async(
                    app.state.tool_registry,
                    app.state.custom_flow_store,
                    app.state.background_chain_manager,
                    default_backend,
                    provider_registry=app.state.provider_registry,
                )
                if flow_tool_count:
                    log.info("flow_tools_registered", count=flow_tool_count)
        except Exception:
            log.warning("flow_tools_registration_failed", exc_info=True)

    # Seed built-in prompt presets (narrative mode) for the oldest active
    # user. New users who want the built-ins can install them from the UI —
    # lazy per-user seeding is a future enhancement.
    if isinstance(app.state.state_manager.backend, SQLiteBackend):
        try:
            from augmentum.modes.narrative.prompt_presets import PromptPresetStore

            owner_uid = ""
            try:
                cur = await sqlite_backend.conn.execute(
                    "SELECT id FROM users ORDER BY created_at ASC LIMIT 1",
                )
                row = await cur.fetchone()
                if row:
                    owner_uid = row[0] or ""
            except Exception:
                owner_uid = ""
            preset_store = PromptPresetStore(sqlite_backend.conn)
            await preset_store.seed_builtins(user_id=owner_uid)
        except Exception:
            log.warning("prompt_preset_seed_failed", exc_info=True)

    # Initialize MCP server (expose Augmentum tools to external clients)
    app.state.mcp_server = None
    if settings.mcp_enabled:
        try:
            from augmentum.mcp.server import create_mcp_server, mount_mcp_server

            memory_store = getattr(app.state, "memory_store", None)
            mcp_srv = create_mcp_server(
                app.state.tool_registry, memory_store, app=app,
            )
            app.state.mcp_server = mcp_srv
            mount_mcp_server(app, mcp_srv)
            # FastAPI's ``app.mount`` does NOT run the sub-app's lifespan,
            # and FastMCP's Streamable-HTTP transport initializes its task
            # group only inside ``session_manager.run()`` — without this,
            # every request to /mcp/ 500s with "Task group is not
            # initialized" (found live 2026-07-18; the mount had never
            # actually served a request). Enter the context here and exit
            # it in the shutdown half of this lifespan.
            _mcp_session_ctx = mcp_srv.session_manager.run()
            await _mcp_session_ctx.__aenter__()
            app.state._mcp_session_ctx = _mcp_session_ctx
        except Exception:
            log.warning("mcp_server_init_failed", exc_info=True)

    # Initialize image generation subsystem (if enabled)
    app.state.image_queue = None
    app.state.image_pipeline_registry = None
    if settings.image_enabled:
        try:
            await _init_image_subsystem(app)
        except Exception as exc:
            log.error(
                "image_subsystem_init_failed",
                error=str(exc),
                hint="Install image dependencies: pip install .[image]",
                exc_info=True,
            )

    # Always init image persistence for read-only history access
    if not getattr(app.state, "image_persistence", None):
        state_mgr = getattr(app.state, "state_manager", None)
        if state_mgr and isinstance(state_mgr.backend, SQLiteBackend):
            try:
                from augmentum.image.persistence import ImagePersistence
                app.state.image_persistence = ImagePersistence(state_mgr.backend.conn)
            except Exception:
                log.debug("image_persistence_fallback_failed", exc_info=True)

    # Clean up orphaned image history entries (files deleted but DB rows remain)
    _img_persist = getattr(app.state, "image_persistence", None)
    if _img_persist:
        try:
            _img_out = settings.image_output_dir or f"{settings.data_dir}/image_output"
            purged = await _img_persist.cleanup_orphaned(_img_out)
            if purged:
                log.info("image_orphan_cleanup", purged=purged)
        except Exception:
            log.debug("image_orphan_cleanup_failed", exc_info=True)

    # Initialize resource ledger (VRAM/RAM monitoring across subsystems).
    # Uses a DEDICATED aiosqlite connection (separate worker thread) so its
    # writes don't queue behind auth/state/chat reads on the main shared
    # connection. The main connection had been logging 1+ second waits on
    # simple SELECTs because resource snapshot writes (every 2-30s) and
    # their fsyncs were sitting at the head of the worker-thread queue.
    # Tables are independent (resource_snapshots, resource_profiles —
    # no FKs to user data), so a separate connection has no consistency
    # cost.
    #
    # Pragmas: identical to the main connection (NORMAL synchronous).
    # An earlier revision used ``synchronous=OFF`` here on the theory
    # that "lost observability rows on power failure are not a
    # correctness issue." That theory was wrong about the failure
    # mode. ``synchronous=OFF`` doesn't just risk this connection's own
    # rows — it hands write ordering for the entire shared DB file to
    # the OS. Under WSL2 / Docker Desktop (where fsync semantics are
    # already weaker than native Linux), an unclean shutdown during a
    # checkpoint window can corrupt unrelated tables on the same file.
    # We saw exactly that pattern (rowid out-of-order + double-page
    # references in file_index) on 2026-05-09. After our snapshot-
    # gating fix, this connection writes ~once per model load event,
    # so the per-fsync cost is rounding error.
    from augmentum.resource.ledger import ResourceLedger
    from augmentum.state.backends.sqlite import (
        AUGMENTUM_DB_PRAGMAS,
        _install_query_timing,
        install_safe_rollback,
    )

    _db_conn = None
    _ledger_conn = None
    if isinstance(getattr(app.state, "state_manager", None), StateManager):
        backend = app.state.state_manager.backend
        if isinstance(backend, SQLiteBackend):
            try:
                _ledger_conn = await aiosqlite.connect(db_path)
                _ledger_conn.row_factory = aiosqlite.Row
                for pragma in AUGMENTUM_DB_PRAGMAS:
                    await _ledger_conn.execute(pragma)
                # The resource ledger is BEST-EFFORT telemetry: missed
                # snapshots are recoverable from the next collect() call.
                # 30s busy_timeout (the standard) blocks the entire ledger
                # event loop while a single startup write storm clears,
                # for no gain — the snapshot would be stale by then anyway.
                # 3s lets contended writes drop fast so the loop can keep
                # collecting fresh state.
                await _ledger_conn.execute("PRAGMA busy_timeout=3000")
                # Inherit slow-query diagnostic on this connection too.
                _install_query_timing(_ledger_conn)
                # Persistent connection — safe-rollback prevents a failed
                # DML from leaving Python sqlite3 in stuck-transaction
                # state that pins a WAL snapshot. See
                # ``install_safe_rollback`` docstring for full background.
                install_safe_rollback(_ledger_conn)
                app.state.resource_ledger_conn = _ledger_conn
                _db_conn = _ledger_conn
                log.info(
                    "resource_ledger_dedicated_conn",
                    db_path=db_path,
                    synchronous="NORMAL",
                )
            except Exception:
                # Fall back to the shared backend connection if the
                # dedicated one fails to open — reverts to prior behaviour.
                log.warning(
                    "resource_ledger_dedicated_conn_failed",
                    db_path=db_path,
                    exc_info=True,
                )
                _db_conn = backend.conn

    # Hot-read connection. Same pattern + rationale as the resource
    # ledger conn above: aiosqlite serialises every query on a given
    # connection through a single worker thread, so high-frequency
    # background readers (fabric capability extractors, job-queue
    # claim, audio provider lookups, file_index stats) end up behind
    # whatever write/long-query the main connection is processing.
    # Splitting them onto a dedicated reader connection eliminates
    # that head-of-line block — same DB file, same WAL, just a
    # different worker thread. Writes still go through the main
    # connection so we don't multiply SQLite-side writer contention.
    #
    # Consumers should accept ``None`` and fall through to the main
    # connection so paths that need read-after-write within a request
    # keep their existing semantics (and so unit tests without
    # lifespan setup keep working).
    app.state.read_conn = None
    if isinstance(getattr(app.state, "state_manager", None), StateManager):
        backend = app.state.state_manager.backend
        if isinstance(backend, SQLiteBackend):
            try:
                _read_conn = await aiosqlite.connect(db_path)
                _read_conn.row_factory = aiosqlite.Row
                for pragma in AUGMENTUM_DB_PRAGMAS:
                    await _read_conn.execute(pragma)
                # Shorter busy_timeout than the main 30s — every consumer
                # on this connection is a non-critical background reader
                # that can degrade gracefully (return cached / empty /
                # log warning). Blocking 30s on a SELECT here defeats
                # the whole point of splitting it off.
                await _read_conn.execute("PRAGMA busy_timeout=3000")
                # Per-query timing diagnostics + safe-rollback parity
                # with the main connection. The latter is technically
                # belt-and-suspenders here (we don't issue DML on this
                # conn), but cheap and keeps the lifecycle uniform.
                _install_query_timing(_read_conn)
                install_safe_rollback(_read_conn)
                app.state.read_conn = _read_conn
                log.info(
                    "read_conn_initialised",
                    db_path=db_path,
                    busy_timeout_ms=3000,
                )

                # Wire the read connection into stores that benefit
                # most. JobsStore's claim_next_pending SELECT runs
                # every runner tick and was showing up in slow_db_op
                # logs (3.3s contended) — moving the SELECT to the
                # read conn lifts that off the main writer thread.
                # The UPDATE-to-claim stays on the main connection.
                jobs_store_state = getattr(app.state, "jobs_store", None)
                if jobs_store_state is not None and hasattr(
                    jobs_store_state, "attach_read_conn"
                ):
                    jobs_store_state.attach_read_conn(_read_conn)
            except Exception:
                log.warning(
                    "read_conn_init_failed",
                    db_path=db_path,
                    exc_info=True,
                )

    resource_ledger = ResourceLedger(db=_db_conn)
    resource_ledger.set_model_manager(app.state.model_manager)
    resource_ledger.set_provider_registry(app.state.provider_registry)
    if getattr(app.state, "llama_manager", None):
        resource_ledger.set_llama_manager(app.state.llama_manager)
    if getattr(app.state, "secondary_slot", None):
        resource_ledger.set_secondary_slot(app.state.secondary_slot)
    if getattr(app.state, "classifier_slot", None):
        resource_ledger.set_classifier_slot(app.state.classifier_slot)
    if getattr(app.state, "image_pipeline_registry", None):
        resource_ledger.set_image_subsystem(app.state.image_pipeline_registry)
    if getattr(app.state, "jobs_store", None):
        resource_ledger.set_jobs_store(app.state.jobs_store)
    app.state.resource_ledger = resource_ledger

    try:
        await resource_ledger.collect()
    except Exception:
        log.warning("initial_resource_collect_failed", exc_info=True)

    # Background resource sampler — keeps the probe caches warm OFF the request
    # path so /api/resources/status never awaits a live docker/HTTP/nvidia-smi
    # probe (spec §4.5/§4.6). Auto-restarts on crash; cancelled at shutdown.
    from augmentum.resource.sampler import resource_sampler_loop
    _sampler_holder: dict[str, asyncio.Task | None] = {"task": None}

    def _on_sampler_done(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            log.error("resource_sampler_loop_crashed", error=str(exc))
            new_task = asyncio.create_task(resource_sampler_loop(app.state))
            new_task.add_done_callback(_on_sampler_done)
            _sampler_holder["task"] = new_task
            app.state.resource_sampler_task = new_task

    _sampler_holder["task"] = asyncio.create_task(resource_sampler_loop(app.state))
    _sampler_holder["task"].add_done_callback(_on_sampler_done)
    app.state.resource_sampler_task = _sampler_holder["task"]

    # Schedule periodic session cleanup (every 6 hours, deletes sessions >30 days old)
    _cleanup_holder: dict[str, asyncio.Task | None] = {"task": None}
    state_mgr = getattr(app.state, "state_manager", None)
    if state_mgr and isinstance(state_mgr.backend, SQLiteBackend):
        async def _session_cleanup_loop():
            while True:
                await asyncio.sleep(6 * 3600)  # every 6 hours
                try:
                    conn = state_mgr.backend.conn
                    cursor = await conn.execute(
                        "DELETE FROM sessions WHERE updated_at < datetime('now', '-30 days')",
                    )
                    deleted = cursor.rowcount
                    if deleted:
                        await conn.commit()
                        log.info("session_cleanup", deleted=deleted)
                except Exception:
                    log.debug("session_cleanup_failed", exc_info=True)

        def _on_cleanup_done(task: asyncio.Task) -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                log.error("session_cleanup_loop_crashed", error=str(exc))
                # Auto-restart the cleanup loop
                new_task = asyncio.create_task(_session_cleanup_loop())
                new_task.add_done_callback(_on_cleanup_done)
                _cleanup_holder["task"] = new_task

        _cleanup_holder["task"] = asyncio.create_task(_session_cleanup_loop())
        _cleanup_holder["task"].add_done_callback(_on_cleanup_done)

    # Media library upkeep: (1) on boot, clear any media server stuck in
    # 'syncing' from a restart that killed a sync mid-run (the catalog stays
    # indexed; only the flag is stale); (2) re-sync every server daily so new
    # upstream content (movies/books added after provision) shows up without a
    # manual Sync. Provision already auto-syncs the first time; this keeps it
    # fresh thereafter. Mirrors the cleanup loop's crash-restart shape.
    _media_resync_holder: dict[str, asyncio.Task | None] = {"task": None}
    if state_mgr and isinstance(state_mgr.backend, SQLiteBackend):
        async def _media_resync_loop():
            from augmentum.media.store import MediaServerStore
            from augmentum.media.sync import enqueue_media_sync
            store = MediaServerStore(state_mgr.backend.conn)
            # Let boot DB contention (VACUUM backup, health checks) clear before
            # writing — running immediately races the startup write lock
            # (OperationalError: database is locked).
            await asyncio.sleep(60)
            try:
                n = await store.reset_stale_syncing()
                log.info("media_stale_syncing_reset", count=n)
            except Exception:
                log.warning("media_stale_syncing_reset_failed", exc_info=True)
            while True:
                await asyncio.sleep(24 * 3600)  # daily; first re-sync a day in
                try:
                    servers = await store.list_all()
                    for s in servers:
                        await enqueue_media_sync(
                            app.state, user_id=s.user_id, server_id=s.id,
                        )
                    if servers:
                        log.info("media_periodic_resync_enqueued", count=len(servers))
                except Exception:
                    log.debug("media_periodic_resync_failed", exc_info=True)

        def _on_media_resync_done(task: asyncio.Task) -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                log.error("media_resync_loop_crashed", error=str(exc))
                new_task = asyncio.create_task(_media_resync_loop())
                new_task.add_done_callback(_on_media_resync_done)
                _media_resync_holder["task"] = new_task

        _media_resync_holder["task"] = asyncio.create_task(_media_resync_loop())
        _media_resync_holder["task"].add_done_callback(_on_media_resync_done)

    # Periodic WAL checkpoint. Default ``wal_autocheckpoint=1000`` fires
    # PASSIVE inline on commits past ~4MB, but PASSIVE can't reclaim
    # frames behind a long-lived reader's snapshot — and we have
    # several persistent connections (main backend, dream journal,
    # resource ledger) where a cursor or transaction can hold the
    # snapshot for tens of minutes. When that happens the WAL grows
    # unbounded and every writer's 30s ``busy_timeout`` eventually
    # blows, producing the ``database is locked`` storm we saw on
    # 2026-05-20 (45MB WAL, 11230 frames, only 339 reclaimable).
    #
    # The loop: every 5min run PASSIVE and log the (busy, log, ckpt)
    # tuple — gives a continuous signal. When the WAL grows past
    # ~20MB AND ckpt stays under 25% of the log for two consecutive
    # ticks, escalate to TRUNCATE and emit ``wal_pinned_warning`` so
    # the next investigator has a grep target. One TRUNCATE at startup
    # covers the case where the previous shutdown's 5s-bounded TRUNCATE
    # timed out (see ``wal_checkpoint_complete`` in lifespan_shutdown).
    _checkpoint_holder: dict[str, asyncio.Task | None] = {"task": None}
    if state_mgr and isinstance(state_mgr.backend, SQLiteBackend):
        async def _wal_checkpoint_loop():
            conn = state_mgr.backend.conn
            # Count consecutive ticks where the WAL is large AND mostly
            # un-checkpointable. Two in a row = a real pin; escalate.
            pinned_ticks = 0

            # Startup TRUNCATE: covers the case where the previous
            # shutdown's TRUNCATE hit the 5s ceiling and left a fat
            # WAL behind. Bounded so a startup-time reader can't stall
            # the loop indefinitely.
            try:
                cursor = await asyncio.wait_for(
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)"),
                    timeout=10.0,
                )
                row = await cursor.fetchone()
                if row:
                    log.info(
                        "wal_checkpoint_startup",
                        busy=row[0], log_frames=row[1], checkpointed=row[2],
                    )
            except TimeoutError:
                log.warning("wal_checkpoint_startup_timeout")
            except Exception:
                log.debug("wal_checkpoint_startup_failed", exc_info=True)

            while True:
                await asyncio.sleep(300)  # every 5 minutes
                try:
                    cursor = await conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    row = await cursor.fetchone()
                    if not row:
                        continue
                    busy, log_frames, checkpointed = row[0], row[1], row[2]
                    log.info(
                        "wal_checkpoint_passive",
                        busy=busy,
                        log_frames=log_frames,
                        checkpointed=checkpointed,
                    )
                    # Pin detection: WAL is large (>=5000 frames ≈ 20MB
                    # at default page_size=4KB) AND <25% reclaimable.
                    # Below 5000 frames is normal bursty traffic — a
                    # full reclaim there has no urgency.
                    pinned = (
                        log_frames >= 5000
                        and 4 * checkpointed < log_frames
                    )
                    pinned_ticks = pinned_ticks + 1 if pinned else 0
                    if pinned_ticks >= 2:
                        log.warning(
                            "wal_pinned_warning",
                            log_frames=log_frames,
                            checkpointed=checkpointed,
                            consecutive_ticks=pinned_ticks,
                            note=(
                                "WAL not checkpointing; long-lived reader "
                                "holds an old snapshot. Attempting TRUNCATE."
                            ),
                        )
                        try:
                            cursor = await conn.execute(
                                "PRAGMA wal_checkpoint(TRUNCATE)",
                            )
                            trow = await cursor.fetchone()
                            if trow:
                                log.info(
                                    "wal_checkpoint_truncate_attempt",
                                    busy=trow[0],
                                    log_frames=trow[1],
                                    checkpointed=trow[2],
                                )
                                # busy=0 means the reader released and
                                # TRUNCATE succeeded — drop the counter
                                # so we don't keep escalating.
                                if trow[0] == 0:
                                    pinned_ticks = 0
                        except Exception:
                            log.debug(
                                "wal_checkpoint_truncate_failed", exc_info=True,
                            )
                except Exception:
                    log.debug("wal_checkpoint_passive_failed", exc_info=True)

        def _on_checkpoint_done(task: asyncio.Task) -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                log.error("wal_checkpoint_loop_crashed", error=str(exc))
                new_task = asyncio.create_task(_wal_checkpoint_loop())
                new_task.add_done_callback(_on_checkpoint_done)
                _checkpoint_holder["task"] = new_task

        _checkpoint_holder["task"] = asyncio.create_task(_wal_checkpoint_loop())
        _checkpoint_holder["task"].add_done_callback(_on_checkpoint_done)

    # Studio Tool Palette: sweep un-acted Generate-tab stagings every 5min.
    # Promise: nothing silently rots in the user's image library — if they
    # generate and never click Use / Save / Regenerate, the staged image
    # gets deleted after 30 minutes. See augmentum/proxy/studio_routes.py
    # for the staging registry shape + sweep logic.
    _studio_sweep_holder: dict[str, asyncio.Task | None] = {"task": None}
    try:
        from augmentum.proxy.studio_routes import studio_staging_sweep_loop

        def _on_studio_sweep_done(task: asyncio.Task) -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                log.warning("studio_staging_sweep_loop_crashed", error=str(exc))
        _studio_sweep_holder["task"] = asyncio.create_task(
            studio_staging_sweep_loop(app),
        )
        _studio_sweep_holder["task"].add_done_callback(_on_studio_sweep_done)
    except Exception:
        log.warning("studio_staging_sweep_loop_spawn_failed", exc_info=True)

    # Daily SQLite integrity probe. ``PRAGMA integrity_check`` returns
    # ``[("ok",)]`` on a clean DB and a list of human-readable error
    # strings otherwise (out-of-order rowids, double-page references,
    # orphaned pages). Without this, corruption can grow undetected for
    # weeks because most queries don't hit the bad pages.
    #
    # First run is delayed 30 minutes after startup so the initial boot
    # storm (model loads, FastEmbed init, knowledge pack scan) doesn't
    # share the worker thread with a multi-second integrity scan.
    _integrity_holder: dict[str, asyncio.Task | None] = {"task": None}
    if state_mgr and isinstance(state_mgr.backend, SQLiteBackend):
        # The check runs against a SHORT-LIVED, dedicated sqlite3
        # connection on a worker thread. Originally it shared the main
        # aiosqlite connection — which meant the 1-2s full-table scan
        # blocked every other DB op for the duration. With WAL mode the
        # check is a read-only snapshot and a fresh connection sees a
        # consistent view; no locking impact on the live workload.
        #
        # Uses ``PRAGMA quick_check`` rather than ``integrity_check``:
        # quick_check covers the high-value subset (page-level
        # corruption + B-tree consistency) for roughly 10x less I/O
        # against the same DB. integrity_check additionally validates
        # index ordering against table data, which is rare to corrupt
        # independently and not worth the extra read pressure for a
        # background sentinel. If quick_check ever surfaces something
        # we can run a manual integrity_check from a maintenance
        # window for the deeper diagnostic.
        #
        # Retry contract + transient classification live in
        # ``_run_quick_check_with_retry`` (module level, unit-tested).
        # Total budget: 3 attempts spaced 2s, 8s — enough to outwait
        # WSL2 / Docker Desktop I/O storms without delaying loop
        # progress past ~10s. Final exhaustion bubbles up to the
        # ``except`` below as a warning.
        def _run_integrity_check_sync() -> list[str]:
            return _run_quick_check_with_retry(db_path, logger=log)

        async def _integrity_check_loop():
            await asyncio.sleep(30 * 60)  # initial delay
            while True:
                try:
                    findings = await asyncio.to_thread(_run_integrity_check_sync)
                    if findings == ["ok"]:
                        log.info("db_integrity_ok")
                    else:
                        log.warning(
                            "db_integrity_compromised",
                            findings_count=len(findings),
                            sample=findings[:5],
                        )
                except Exception:
                    log.warning("db_integrity_check_failed", exc_info=True)
                await asyncio.sleep(24 * 3600)  # daily

        def _on_integrity_done(task: asyncio.Task) -> None:
            if task.cancelled():
                return
            exc = task.exception()
            if exc:
                log.error("db_integrity_loop_crashed", error=str(exc))
                new_task = asyncio.create_task(_integrity_check_loop())
                new_task.add_done_callback(_on_integrity_done)
                _integrity_holder["task"] = new_task

        _integrity_holder["task"] = asyncio.create_task(_integrity_check_loop())
        _integrity_holder["task"].add_done_callback(_on_integrity_done)

    # Start background service health checks (non-blocking)
    await health.start(interval=30)

    # Event-loop lag monitor. Sleeps on a fixed interval and measures the
    # actual elapsed wall-clock vs the expected. A healthy loop measures
    # ~period; a stalled loop reports a positive lag that says "something
    # synchronous blocked the loop for this long." Surfaces the kind of
    # transient unresponsiveness that causes the container's healthcheck
    # endpoint to time out under heavy generation load — without that
    # signal the only evidence is autoheal restarts.
    # Background sampler thread for deep stalls. The async lag monitor
    # below catches the lag AFTER the blocking work finished — too late to
    # see what was holding the GIL. This thread runs OUTSIDE the loop and
    # samples sys._current_frames() on the main thread when the loop has
    # gone quiet for too long, capturing the actual stack of the blocker
    # while it's still running. Daemon so it doesn't block shutdown.
    import sys as _sys
    import threading as _threading
    import traceback as _traceback

    _loop_heartbeat = {"t": time.monotonic()}
    _deep_stall_threshold_s = 5.0

    def _stall_sampler() -> None:
        main_thread_id = _threading.main_thread().ident
        last_dump = 0.0
        while True:
            time.sleep(1.0)
            now = time.monotonic()
            silence = now - _loop_heartbeat["t"]
            if silence < _deep_stall_threshold_s:
                continue
            # Throttle: one dump per stall episode, not per second of it.
            if now - last_dump < _deep_stall_threshold_s:
                continue
            last_dump = now
            frames = _sys._current_frames()
            main_frame = frames.get(main_thread_id) if main_thread_id else None
            stack_lines = (
                "".join(_traceback.format_stack(main_frame))
                if main_frame is not None else "<no main frame>"
            )
            log.warning(
                "event_loop_stall_sample",
                silent_for_s=round(silence, 2),
                main_stack=stack_lines,
            )

    _threading.Thread(target=_stall_sampler, daemon=True,
                      name="loop-stall-sampler").start()

    async def _event_loop_lag_monitor(period_s: float = 1.0, threshold_s: float = 1.0) -> None:
        last = time.monotonic()
        while True:
            try:
                await asyncio.sleep(period_s)
            except asyncio.CancelledError:
                return
            now = time.monotonic()
            lag = (now - last) - period_s
            last = now
            _loop_heartbeat["t"] = now
            # Expose the latest lag so the strain sampler can record it as part
            # of the general-purpose health time series (not just on stall).
            app.state.last_event_loop_lag_s = round(max(lag, 0.0), 3)
            if lag >= threshold_s:
                log.warning(
                    "event_loop_stall",
                    lag_s=round(lag, 2),
                    period_s=period_s,
                    threshold_s=threshold_s,
                )

    app.state.loop_lag_task = asyncio.create_task(_event_loop_lag_monitor())

    # General-purpose strain sampler — a durable, queryable health time series
    # (strain_samples) that correlates server strain against how many clients
    # were concurrently active. The static counterpart of the event_loop_stall
    # watchdog; trips a grep-able ``strain_sample`` WARNING on bad moments.
    # Always started; sampling is gated on the setting inside so a runtime
    # toggle takes effect without a restart.
    async def _strain_sampler_loop(period_s: float = 10.0) -> None:
        backend = getattr(getattr(app.state, "state_manager", None), "backend", None)
        conn = getattr(backend, "conn", None) if backend else None
        if conn is None:
            log.info("strain_monitor_inactive_no_sqlite")
            return
        from augmentum.health import StrainMonitor
        # Pass the DB path so the monitor opens its OWN connection for the
        # write/contention probe — issuing BEGIN IMMEDIATE on the shared
        # backend connection crashes ("cannot start a transaction within a
        # transaction") when another coroutine holds an open transaction.
        db_path = getattr(backend, "_db_path", None)
        monitor = StrainMonitor(conn, app, db_path=db_path)
        app.state.strain_monitor = monitor
        try:
            while True:
                try:
                    await asyncio.sleep(period_s)
                except asyncio.CancelledError:
                    return
                if not getattr(settings, "strain_monitor_enabled", True):
                    continue
                try:
                    await monitor.sample_and_store()
                except Exception as exc:  # never let the sampler kill itself
                    log.debug("strain_sample_loop_error", error=str(exc))
        finally:
            await monitor.aclose()

    _track_bg(_strain_sampler_loop(), name="strain_sampler")

    # Fabric layer: no-op when settings.fabric_enabled is False (default).
    # Owns its own lifecycle in augmentum.fabric.lifespan -- the call here
    # is intentionally a one-liner so server.py churn stays minimal.
    from augmentum.fabric.lifespan import start_fabric_if_enabled
    await start_fabric_if_enabled(app)

    # mDNS LAN advertisement so the Android TV receiver (and future
    # native clients) can resolve this node without a subnet sweep.
    # No-op when zeroconf isn't importable or no usable LAN IP exists.
    from augmentum.cast.mdns import start_mdns
    await start_mdns(app)

    yield

    # Shutdown order: background tasks → engines → DB → HTTP client

    # MCP Streamable-HTTP session manager — exit the context entered at
    # mount time so its task group unwinds cleanly before the loop closes.
    _mcp_ctx = getattr(app.state, "_mcp_session_ctx", None)
    if _mcp_ctx is not None:
        with contextlib.suppress(Exception):
            await _mcp_ctx.__aexit__(None, None, None)

    # mDNS deregistration first — quick, idempotent, frees the socket
    # so a fast restart doesn't trip "address in use".
    from augmentum.cast.mdns import stop_mdns
    await stop_mdns(app)

    # Fabric teardown runs before the DB closes so the coordinator
    # can issue clean disconnect frames + cancel the client task
    # while its (read-only) DB handle is still live. No-op when
    # fabric was never started.
    from augmentum.fabric.lifespan import stop_fabric
    await stop_fabric(app)

    # 1. Cancel session cleanup
    _cleanup_task = _cleanup_holder.get("task")
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _cleanup_task

    # 1b. Cancel event-loop lag monitor
    _lag_task = getattr(app.state, "loop_lag_task", None)
    if _lag_task and not _lag_task.done():
        _lag_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _lag_task

    # 1c. Cancel all tracked background loops (memory maintenance, auth
    # cleanup, files maintenance, transient artifact sweep, file index
    # populate, enrichment, warmup). Without this they sit in
    # ``await asyncio.sleep(...)`` past SIGTERM and Docker SIGKILLs the
    # container after the 10s grace, slowing every restart.
    bg_tasks = list(getattr(app.state, "background_tasks", []) or [])
    for task in bg_tasks:
        if not task.done():
            task.cancel()
    if bg_tasks:
        await asyncio.gather(*bg_tasks, return_exceptions=True)
        log.info("background_tasks_cancelled", count=len(bg_tasks))

    # 1d. Stop the SearXNG proxy healthcheck loop (own state, not tracked
    # in background_tasks above).
    _searxng_hc = getattr(app.state, "searxng_proxy_healthcheck", None)
    if _searxng_hc is not None:
        _searxng_hc.stop()

    # 2. Stop service health checks
    await health.stop()

    # 2b. Stop managed llama-server if running
    _llama_mgr = getattr(app.state, "llama_manager", None)
    if _llama_mgr is not None:
        try:
            await _llama_mgr.stop()
            log.info("engine_v2_stopped")
        except Exception:
            log.warning("engine_v2_stop_failed", exc_info=True)

    # 2c. Stop the secondary slot ("Slot B") subprocess if running
    _secondary = getattr(app.state, "secondary_slot", None)
    if _secondary is not None:
        try:
            await _secondary.unload()
            log.info("engine_secondary_stopped")
        except Exception:
            log.warning("engine_secondary_stop_failed", exc_info=True)

    # 2d. Stop the managed classifier slot ("Slot C") subprocess if running
    _classifier_slot = getattr(app.state, "classifier_slot", None)
    if _classifier_slot is not None:
        try:
            await _classifier_slot.unload()
            log.info("classifier_slot_stopped")
        except Exception:
            log.warning("classifier_slot_stop_failed", exc_info=True)

    # 3. Stop background workers
    bg = getattr(app.state, "background_chain_manager", None)
    if bg:
        bg.shutdown()

    if getattr(app.state, "memory_compactor", None):
        await app.state.memory_compactor.stop()

    _job_runner = getattr(app.state, "job_runner", None)
    if _job_runner:
        try:
            await _job_runner.stop()
            log.info("job_runner_stopped")
        except Exception:
            log.warning("job_runner_stop_failed", exc_info=True)

    _job_monitor = getattr(app.state, "job_monitor", None)
    if _job_monitor:
        try:
            await _job_monitor.stop()
        except Exception:
            log.warning("job_monitor_stop_failed", exc_info=True)

    if getattr(app.state, "dream_scheduler", None) or getattr(app.state, "dream_journal", None):
        from augmentum.dream.lifecycle import teardown_dream_system
        await teardown_dream_system(app)

    # SchedulerService teardown — before the companion runtime so a
    # mid-flight step() isn't left dispatching against a stopped context.
    _sched_svc = getattr(app.state, "scheduler_service", None)
    if _sched_svc is not None:
        try:
            await _sched_svc.stop()
        except Exception:
            log.warning("scheduler_service_stop_failed", exc_info=True)

    # CompanionRuntime teardown. Stops the tick task (Sprint 4a) and
    # closes the presence bus so WebSocket subscribers see EOF.
    _companion_rt = getattr(app.state, "companion_runtime", None)
    if _companion_rt is not None:
        try:
            await _companion_rt.stop()
            log.info("companion_runtime_stopped")
        except Exception:
            log.warning("companion_runtime_stop_failed", exc_info=True)

    if getattr(app, "state", None) and getattr(app.state, "pack_manager", None):
        await app.state.pack_manager.close()

    if getattr(app.state, "mcp_client", None):
        await app.state.mcp_client.disconnect_all()

    if getattr(app.state, "image_queue", None):
        await app.state.image_queue.stop()
    if getattr(app.state, "image_pipeline_registry", None):
        await app.state.image_pipeline_registry.unload()

    # 3. Clear engine state
    app.state.narrative_engines.clear()
    if getattr(app.state, "agentic_handlers", None) is not None:
        app.state.agentic_handlers.clear()

    # Cancel any in-flight coder agent tasks. The startup sweep on
    # next boot also marks their rows ``cancelled`` so the UI doesn't
    # see a "running" turn after a restart.
    if getattr(app.state, "coder_run_broker", None) is not None:
        try:
            await app.state.coder_run_broker.shutdown()
        except Exception:
            log.warning("coder_run_broker_shutdown_failed", exc_info=True)

    # Close Docker client for coder mode
    if getattr(app.state, "container_manager", None):
        try:
            await app.state.container_manager._docker.close()
        except Exception as exc:
            log.debug("docker_client_close_failed", error=str(exc))

    # Stop device registry (drivers tear down listeners, event bus closes)
    device_registry = getattr(app.state, "device_registry", None)
    if device_registry is not None:
        try:
            await device_registry.stop()
        except Exception as exc:
            log.warning("shutdown_device_registry_stop_failed", error=str(exc))

    # Stop the HTML renderer if its Chrome subprocess was started. The
    # store is in-memory only — drops with the process, no shutdown
    # needed.
    html_renderer = getattr(app.state, "html_renderer", None)
    if html_renderer is not None:
        try:
            await html_renderer.stop()
        except Exception as exc:
            log.warning("shutdown_html_renderer_stop_failed", error=str(exc))

    # Close any open receiver WebSockets cleanly so the receivers see
    # a graceful close frame rather than a torn TCP.
    receiver_registry = getattr(app.state, "receiver_registry", None)
    if receiver_registry is not None:
        try:
            await receiver_registry.close_all()
        except Exception as exc:
            log.warning("shutdown_receiver_registry_close_failed", error=str(exc))

    # Same for cast-input WSes (phone-as-controller + container daemons).
    cast_input_registry = getattr(app.state, "cast_input_registry", None)
    if cast_input_registry is not None:
        try:
            await cast_input_registry.close_all()
        except Exception as exc:
            log.warning("shutdown_cast_input_registry_close_failed", error=str(exc))

    # Origin-proxy fetcher owns an httpx.AsyncClient; close it cleanly.
    cast_proxy_fetcher = getattr(app.state, "cast_proxy_fetcher", None)
    if cast_proxy_fetcher is not None:
        try:
            await cast_proxy_fetcher.aclose()
        except Exception as exc:
            log.warning("shutdown_cast_proxy_fetcher_close_failed", error=str(exc))

    # 4. WAL checkpoint then close database (after all background tasks are stopped)
    if isinstance(app.state.state_manager.backend, SQLiteBackend):
        try:
            # Bounded so a stuck writer lock can't push the checkpoint past
            # docker's 10s SIGTERM grace — a SIGKILL mid-truncate produces
            # torn WAL pages that corrupt the DB on next startup. On
            # timeout, skip and let the next startup replay the WAL.
            await asyncio.wait_for(
                app.state.state_manager.backend.conn.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)",
                ),
                timeout=5.0,
            )
            log.info("wal_checkpoint_complete")
        except TimeoutError:
            log.warning(
                "wal_checkpoint_timeout",
                note="skipped; next startup will replay WAL",
            )
        except Exception:
            log.warning("wal_checkpoint_failed", exc_info=True)
        await app.state.state_manager.backend.close()

    # 4a-bis. Stop the background resource sampler before closing its DB conn.
    _sampler_task = getattr(app.state, "resource_sampler_task", None)
    if _sampler_task is not None and not _sampler_task.done():
        _sampler_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await _sampler_task
        app.state.resource_sampler_task = None

    # 4b. Close the dedicated resource-ledger connection if we opened one.
    _ledger_conn = getattr(app.state, "resource_ledger_conn", None)
    if _ledger_conn is not None:
        try:
            await _ledger_conn.close()
        except Exception as exc:
            log.warning("shutdown_resource_ledger_close_failed", error=str(exc))
        app.state.resource_ledger_conn = None

    # 4c. Close the hot-read connection if we opened one.
    _read_conn_close = getattr(app.state, "read_conn", None)
    if _read_conn_close is not None:
        try:
            await _read_conn_close.close()
        except Exception as exc:
            log.warning("shutdown_read_conn_close_failed", error=str(exc))
        app.state.read_conn = None

    # 5. Close module-level httpx clients from audio/cloud-image routes
    from augmentum.proxy.audio_routes import close_audio_clients
    from augmentum.proxy.cloud_image_routes import close_cloud_image_clients

    try:
        await close_audio_clients()
    except Exception as exc:
        log.warning("shutdown_audio_clients_close_failed", error=str(exc))
    try:
        await close_cloud_image_clients()
    except Exception as exc:
        log.warning("shutdown_cloud_image_clients_close_failed", error=str(exc))

    # 6. Save all session KV caches before closing
    lifecycle = getattr(app.state, "session_lifecycle", None)
    if lifecycle:
        try:
            await lifecycle.on_shutdown()
        except Exception as exc:
            log.warning("shutdown_session_lifecycle_failed", error=str(exc))

    # 7. Close HTTP clients last
    await app.state.http_client.aclose()
    await app.state.chat_http_client.aclose()
    log.info("shutdown_complete")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(
        title="Augmentum",
        version="0.1.0",
        description=(
            "Privacy-first personal AI box with an OpenAI-compatible API for "
            "chat, embeddings, text-to-speech, speech-to-text, and image "
            "generation — point any OpenAI client at this base URL. Runtime "
            "feature discovery: `GET /api/capabilities`."
        ),
        license_info={"name": "AGPL-3.0-or-later"},
        lifespan=lifespan,
    )

    from augmentum.devices.host_resolver import PublicHostResolver
    app.state.public_host_resolver = PublicHostResolver(
        configured=settings.augmentum_public_host,
    )

    # Request body size limit
    app.add_middleware(_MaxBodySizeMiddleware, max_bytes=settings.max_request_body_bytes)

    # Per-IP sliding window rate limiting (per-endpoint-group limits)
    from augmentum.proxy.middleware.rate_limit import (
        RateLimitMiddleware as SlidingRateLimitMiddleware,
    )

    app.add_middleware(
        SlidingRateLimitMiddleware,
        limits={
            "chat": settings.rate_limit_chat_rpm,
            "image": settings.rate_limit_image_rpm,
            "voice": settings.rate_limit_voice_rpm,
            "upload": settings.rate_limit_upload_rpm,
        },
        enabled=settings.rate_limit_enabled,
    )

    # CORS safety: credentials + wildcard origin is a security vulnerability.
    cors_creds = settings.cors_allow_credentials
    if cors_creds and settings.cors_origins == ["*"]:
        log.warning("cors_insecure", msg="allow_credentials disabled because allow_origins is ['*']")
        cors_creds = False

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=cors_creds,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class _PublicHostObserverMiddleware:
        def __init__(self, app: ASGIApp) -> None:
            self._app = app

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] == "http":
                app_obj = scope.get("app")
                resolver = getattr(
                    getattr(app_obj, "state", None),
                    "public_host_resolver",
                    None,
                )
                if resolver is not None:
                    try:
                        resolver.observe_scope(scope)
                    except Exception as exc:
                        log.debug("public_host_observe_failed", error=str(exc))
            await self._app(scope, receive, send)

    app.add_middleware(_PublicHostObserverMiddleware)

    # Diagnostic: log any HTTP request that takes >500ms in handler time.
    # Streaming endpoints (chat) routinely hold a request for tens of seconds —
    # those are skipped by checking the path prefix. The point is to catch
    # *non-streaming* endpoints that should be quick (auth/status, ui/state,
    # config GET/PUT, etc.) when they're not. Pairs with ``slow_db_op`` from
    # the sqlite backend wrapper — a slow request without slow_db_op points
    # at non-DB blocking; matching slow_db_op lines tell us which query.
    import time as _slowreq_time

    _SLOW_REQUEST_MS = float(os.environ.get("AUGMENTUM_SLOW_REQUEST_MS", "500"))
    # Endpoints that legitimately take seconds-to-minutes by design (LLM
    # streaming, audio synthesis, image generation, dream cycles, knowledge
    # pack imports). Skipping them keeps the slow_request log focused on
    # endpoints that *should* be fast — auth checks, polling endpoints,
    # state reads.
    _STREAMING_PREFIXES = (
        # LLM completion paths
        "/api/chat", "/v1/chat", "/api/generate", "/v1/completions",
        "/v1/embeddings",
        # Long-running mode handlers
        "/api/agentic/", "/api/coder/",
        # Media + voice (large response bodies, transcoding)
        "/api/media/stream/", "/stream/", "/api/voice/",
        "/api/audio/", "/v1/audio/",
        # Image generation (10-30s typical)
        "/api/image/", "/v1/images/",
        # Dream cycles (LLM call + persistence)
        "/api/dream/",
        # Narrative scene image (image-gen pipeline)
        "/api/narrative/scene-image",
        # Knowledge pack import / convert (multi-minute on large packs)
        "/api/knowledge/packs/install",
        "/api/knowledge/packs/convert",
        # Server-Sent Event channels — long-lived by design.
        # ``/api/system/events`` (cross-feature live updates),
        # ``/api/models/downloads/.../stream`` (download progress),
        # ``/api/build/.../stream`` (architect build status),
        # plus any path containing ``/stream`` we missed above.
        "/api/system/events",
    )

    class _SlowRequestLogMiddleware:
        """Slow-request logging + general-purpose in-flight / active-client
        tracking for the strain monitor. Counts EVERY http request as in-flight
        (including streams — they hold resources), stamps the per-tab client id
        so concurrent multi-browser use is observable, and only applies the
        slow-request *timing* to non-streaming requests (a long stream isn't
        "slow")."""

        def __init__(self, app: ASGIApp) -> None:
            self._app = app

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope.get("type") != "http":
                await self._app(scope, receive, send)
                return
            path = scope.get("path", "")

            # Per-tab client id (X-Augmentum-Client) → active-client registry.
            # Best-effort; auth may not have run yet, so user_id is whatever's
            # on the scope at this point (filled in for authed routes).
            try:
                cid = ""
                for k, v in scope.get("headers", ()):
                    if k == b"x-augmentum-client":
                        cid = v.decode("latin-1")[:64]
                        break
                if cid:
                    user = scope.get("user")
                    uid = getattr(user, "id", "") if user else ""
                    app.state.active_clients[cid] = (_slowreq_time.monotonic(), uid)
            except Exception:
                pass

            try:
                app.state.inflight_requests += 1
            except Exception:
                pass

            is_stream = any(path.startswith(p) for p in _STREAMING_PREFIXES)
            started = _slowreq_time.monotonic()
            try:
                await self._app(scope, receive, send)
            finally:
                try:
                    app.state.inflight_requests -= 1
                except Exception:
                    pass
                if not is_stream:
                    elapsed_ms = (_slowreq_time.monotonic() - started) * 1000.0
                    if elapsed_ms >= _SLOW_REQUEST_MS:
                        try:
                            app.state.slow_request_count += 1
                        except Exception:
                            pass
                        log.warning(
                            "slow_request",
                            path=path,
                            method=scope.get("method", ""),
                            elapsed_ms=round(elapsed_ms, 1),
                        )

    app.add_middleware(_SlowRequestLogMiddleware)

    # Security headers middleware — raw ASGI to avoid breaking WebSocket upgrades
    # (BaseHTTPMiddleware intercepts WebSocket handshakes and causes them to hang)
    # Frameable prefixes: HTTP responses that the SPA legitimately embeds in
    # same-origin iframes (artifact previews, files-mode previews and the
    # download stream that PDFs/videos/HTML pull from).
    _FRAMEABLE_PREFIXES = (
        "/api/artifacts/",
        "/api/files/render/",
        "/api/files/download/",
        # Knowledge-pack ZIM articles render in a sandboxed iframe inside
        # the Browse panel. SAMEORIGIN here just unblocks the parent app
        # from embedding the route at all; the route response sets its
        # own CSP tuned for SPA-style ZIM content (see below).
        "/api/knowledge/zim/",
        # Coder workspace live preview — proxied dev server (Vite/Next/CRA/
        # etc.) rendered in the coder UI's preview pane. Without
        # SAMEORIGIN the global middleware stamps DENY here too and the
        # iframe can't load. The preview proxy already strips the
        # upstream's X-Frame-Options / CSP (see coder_routes._DROP_
        # RESPONSE_HEADERS_BASE) so the dev server can't override us
        # back to DENY.
        "/api/coder/preview/",
        # Cast origin-proxy — receiver TV iframes the rewritten game
        # from our origin via /ui/play-web/. Without SAMEORIGIN the
        # middleware stamps DENY and the play-web iframe can't load
        # the proxied page. The proxy fetcher already strips the
        # upstream X-Frame-Options so the source can't override.
        "/api/cast/game-proxy/",
    )

    # Paths whose handlers set their own CSP — the global middleware skips
    # CSP for these so the route's policy is authoritative. Without this,
    # browsers intersect the two CSP headers and the more restrictive of
    # each directive wins, which can silently break route-specific needs
    # (e.g., font-src data: in the ZIM iframe) the global policy doesn't
    # cover. Other security headers (X-Frame-Options, referrer-policy,
    # permissions-policy) still come from the middleware uniformly.
    _CSP_OVERRIDE_PREFIXES = (
        "/api/knowledge/zim/",
    )

    class _SecurityHeadersMiddleware:
        def __init__(self, app: ASGIApp) -> None:
            self._app = app

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http":
                await self._app(scope, receive, send)
                return

            async def send_with_headers(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    path = scope.get("path", "")
                    is_frameable = any(path.startswith(p) for p in _FRAMEABLE_PREFIXES)
                    csp_override = any(path.startswith(p) for p in _CSP_OVERRIDE_PREFIXES)

                    # Same-host port wildcard for the AGSP stream viewer.
                    # We allow http(s)://<this-host>:* in frame-src so the
                    # in-app stream stage can iframe Selkies running on
                    # whichever port the port pool allocated. Host comes
                    # from the request, so localhost users get
                    # 'http://localhost:*', LAN users get their LAN IP.
                    # Echoing the user's own host back into their CSP adds
                    # no privilege beyond what they already accessed --
                    # they hit us at this host, we let them iframe a
                    # different port on the same host.
                    same_host_frame_src = ""
                    for name, value in scope.get("headers", []):
                        if name == b"host" and value:
                            host_str = value.decode("latin-1", "replace")
                            host_only = host_str.split(":", 1)[0].strip()
                            # Reject anything that doesn't look like a
                            # plain hostname/IP -- defensive against a
                            # weird Host header trying to inject CSP
                            # directives.
                            import re as _csp_host_re
                            if host_only and _csp_host_re.match(
                                r"^[A-Za-z0-9._\-]+$", host_only,
                            ):
                                same_host_frame_src = (
                                    f"http://{host_only}:* "
                                    f"https://{host_only}:* "
                                )
                            break

                    extra = [
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-xss-protection", b"1; mode=block"),
                        (b"referrer-policy", b"strict-origin-when-cross-origin"),
                        (b"permissions-policy", b"camera=(self), microphone=(self), geolocation=()"),
                        (b"x-frame-options", b"SAMEORIGIN" if is_frameable else b"DENY"),
                    ]

                    csp = (
                        "default-src 'self'; upgrade-insecure-requests; "
                        "script-src 'self' 'unsafe-inline' 'unsafe-eval' blob: "
                        "https://cdnjs.cloudflare.com https://cdn.jsdelivr.net https://unpkg.com https://esm.sh "
                        "https://www.gstatic.com "
                        "https://www.youtube.com; "
                        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com "
                        "https://cdnjs.cloudflare.com https://cdn.jsdelivr.net https://esm.sh; "
                        "font-src 'self' https://fonts.gstatic.com "
                        "https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
                        "connect-src 'self' ws: wss: data: blob: https://cdn.jsdelivr.net https://esm.sh "
                        "https://storage.googleapis.com "
                        "https://www.youtube.com https://*.youtube.com https://*.googlevideo.com; "
                        "img-src 'self' data: blob: https://avatars.charhub.io https://sv.risuai.xyz "
                        "https://i.ytimg.com https://*.ytimg.com "
                        # img.youtube.com is YouTube's other official thumbnail
                        # host (serves /vi/<id>/hqdefault.jpg directly — it is
                        # NOT a *.ytimg.com subdomain). Emitted as a fallback by
                        # youtube_routes / browse_routes / discovery_routes and
                        # by yt-dlp thumbnail_url metadata, so it must be allowed.
                        "https://img.youtube.com "
                        "https://*.imgur.com https://i.imgur.com "
                        "https://files.catbox.moe "
                        "https://thumbnails.libretro.com "
                        "https://cdn.discordapp.com https://media.discordapp.net "
                        "https://i.ibb.co https://i.postimg.cc "
                        "https://raw.githubusercontent.com https://cloud.githubusercontent.com "
                        # Curated games tab thumbnails hosted on the project's own
                        # canonical domains (not on raw.githubusercontent.com).
                        "https://hexgl.bkcore.com https://*.decisionproblem.com "
                        "https://archive.org https://*.archive.org "
                        # Fandom / Wikia CDN — community-wiki cover art surfaced
                        # by cardsmith Phase 2 wiki lane, browse-mode embeds,
                        # and lorebook references. All subdomains for legacy
                        # static.wikia.nocookie.net + the newer img.wikia paths.
                        "https://*.nocookie.net "
                        "http://*.imgur.com http://i.imgur.com; "
                        "media-src 'self' blob: https://*.somafm.com https://*.freemusicarchive.org http://* https://*; "
                        "child-src 'self' blob: https://www.youtube.com https://player.vimeo.com "
                        "https://www.dailymotion.com https://*.dailymotion.com https://www.tiktok.com https://clips.twitch.tv "
                        "https://platform.twitter.com; "
                        "frame-src 'self' blob: "
                        + same_host_frame_src
                        + "https://www.youtube.com https://player.vimeo.com "
                        "https://www.dailymotion.com https://*.dailymotion.com https://www.tiktok.com https://clips.twitch.tv "
                        "https://platform.twitter.com "
                        # Curated games tab — every entry in
                        # ui/scripts/library-game-sources.js::_CURATED_WEB_GAMES
                        # plus *.github.io for project-page-hosted entries
                        # (2048, Space Huggers) and any future GitHub-Pages
                        # game we add. The iframe is sandboxed in
                        # ui/index.html (allow-scripts allow-forms allow-popups
                        # allow-pointer-lock allow-same-origin) so each game
                        # only sees its own origin's storage. Adding a curated
                        # game on a new domain requires adding the origin here.
                        "https://*.github.io "
                        "https://hextris.io https://www.puzzlescript.net "
                        "https://david-peter.de https://hellowordl.net "
                        "https://wikitrivia.tomjwatson.com "
                        "https://adarkroom.doublespeakgames.com "
                        "https://www.decisionproblem.com "
                        "https://hexgl.bkcore.com https://sandspiel.club "
                        "https://patatap.com https://bemuse.ninja; "
                        "object-src 'none'; "
                        "base-uri 'self'; "
                        # frame-ancestors 'self' is now the default --
                        # 'none' was over-restrictive and produced
                        # confusing console errors when Chrome's
                        # internal iframe-error page tried to fall
                        # back to the parent origin (e.g. a stream
                        # iframe failing to load triggered "Framing
                        # ... violates frame-ancestors 'none'"
                        # noise unrelated to the actual failure).
                        # 'self' still blocks third-party embeds
                        # while letting same-origin iframes work.
                        # The is_frameable distinction is preserved
                        # purely for the legacy x-frame-options header.
                        + "frame-ancestors 'self'"
                    )
                    if not csp_override:
                        extra.append((b"content-security-policy", csp.encode()))

                    message = {
                        **message,
                        "headers": list(message.get("headers", [])) + extra,
                    }
                await send(message)

            await self._app(scope, receive, send_with_headers)

    app.add_middleware(_SecurityHeadersMiddleware)

    # Auth middleware — validates tokens, attaches user to scope.
    # session_manager is None at creation time; middleware reads from
    # app.state.session_manager at request time (set during lifespan).
    from augmentum.auth.middleware import AuthMiddleware
    app.add_middleware(AuthMiddleware)

    # Isolated-origin listener gate. Requests arriving on the isolated
    # preview port carry the X-Augmentum-Preview-Listener: true header
    # (set by Caddy on the :6444 upstream route). On those requests,
    # ONLY paths matching one of the allowlist prefixes are allowed
    # — every other path returns 404 without further processing. This
    # is defense in depth: a misconfigured Caddy that accidentally
    # routes a non-preview port to the main app must NEVER expose
    # /api/me or any other endpoint on the isolated origin. Auth
    # handoff (cookies don't cross origins) is handled inside each
    # content kind's route — see preview_auth.py.
    #
    # Added AFTER AuthMiddleware so it wraps it (Starlette middleware
    # order is reverse-of-add): the listener gate runs FIRST on
    # request, short-circuiting before Auth ever sees a non-preview
    # path on the isolated port.
    #
    # Each prefix maps to a "kind" tracked on the preview-token /
    # preview-session record so a token of one kind can't unlock a
    # path belonging to another (defense in depth on top of the per-
    # route validation):
    #   /api/coder/preview/  → kind=workspace (coder dev-server)
    #   /api/knowledge/zim/  → kind=knowledge_pack (ZIM articles)
    #   /api/artifacts/      → kind=artifact_app (saved app/zip preview)
    #   /api/library/publications/ → kind=publication (played bundle)
    # The coder dev-server proxy must forward every method (dev servers
    # accept POST/PUT for HMR, form posts, etc.), so it's listed as a
    # full-method prefix. The remaining prefixes are pure content
    # servers — GET/HEAD only. This matters for /api/artifacts/, whose
    # namespace also contains mutation routes (POST /import, /fix,
    # DELETE …): restricting to GET/HEAD keeps those unreachable on the
    # isolated origin even though the prefix is broad.
    _PREVIEW_LISTENER_HEADER = b"x-augmentum-preview-listener"
    _ISOLATED_PROXY_PREFIXES: tuple[str, ...] = (
        "/api/coder/preview/",
    )
    _ISOLATED_READONLY_PREFIXES: tuple[str, ...] = (
        "/api/knowledge/zim/",
        "/api/artifacts/",
        "/api/library/publications/",
    )

    class _PreviewListenerGate:
        def __init__(self, app: ASGIApp) -> None:
            self._app = app

        async def __call__(
            self, scope: Scope, receive: Receive, send: Send,
        ) -> None:
            if scope["type"] != "http":
                await self._app(scope, receive, send)
                return
            # Cheap O(1) header check — short-circuit for the common
            # case (request on the main port carries no listener
            # header).
            is_isolated = False
            for name, value in scope.get("headers", []):
                if name == _PREVIEW_LISTENER_HEADER:
                    is_isolated = value == b"true"
                    break
            if not is_isolated:
                await self._app(scope, receive, send)
                return
            path = scope.get("path", "")
            method = scope.get("method", "GET").upper()
            is_proxy = any(path.startswith(p) for p in _ISOLATED_PROXY_PREFIXES)
            is_readonly = (
                method in ("GET", "HEAD")
                and any(path.startswith(p) for p in _ISOLATED_READONLY_PREFIXES)
            )
            if not (is_proxy or is_readonly):
                # Non-allowlisted path (or a non-GET/HEAD on a read-only
                # content prefix) on the isolated port — reject. Plain
                # 404 (not 403) so we don't reveal the path exists on the
                # main port. Logged at INFO so an operator with a
                # misconfigured Caddy sees the rejection pattern.
                log.warning(
                    "preview_isolated_port_rejected_path",
                    path=path[:120], method=method,
                )
                await send({
                    "type": "http.response.start",
                    "status": 404,
                    "headers": [(b"content-type", b"application/json")],
                })
                await send({
                    "type": "http.response.body",
                    "body": b'{"detail":"not found"}',
                })
                return
            # Mark the scope so downstream handlers (notably the
            # preview proxy) know to use the preview-session cookie
            # path instead of the main session cookie.
            scope["augmentum_preview_isolated"] = True
            await self._app(scope, receive, send)

    app.add_middleware(_PreviewListenerGate)

    # Fabric peer-auth middleware. Runs OUTSIDE AuthMiddleware so it
    # gets first crack at requests carrying X-Fabric-* headers (cross-
    # peer dispatch from another Augmentum). When a peer signature
    # verifies, the middleware pre-populates scope["user"] and the
    # AuthMiddleware tolerance check (3 lines added at top of __call__)
    # honours it. When no fabric headers are present, this middleware
    # is a pure pass-through.
    #
    # PREVIOUSLY gated on ``settings.fabric_enabled`` — but the gate
    # was evaluated at app-construction time when settings comes from
    # env + defaults only. Operators who enable fabric via the UI (most
    # of them; AUGMENTUM_FABRIC_ENABLED env override is the rare path)
    # land their toggle in ``settings_store``, which lifespan reads
    # AFTER the middleware chain is built. Result: ``fabric_started``
    # fires at lifespan time but the middleware was never installed,
    # and every cross-peer dispatch hit ``AuthMiddleware._send_401``.
    #
    # Always-install is correct: the middleware exits in O(1) on every
    # request that doesn't carry ``X-Fabric-Sender`` (the first header
    # check), so "fabric off" deployments pay only a dict lookup per
    # request. Worth less than the wall of broken cross-peer dispatch.
    from augmentum.fabric.peer_middleware import FabricPeerMiddleware
    app.add_middleware(FabricPeerMiddleware)

    # CSRF Origin/Referer check. Runs OUTERMOST (added last → wraps every
    # other middleware) so a forged cross-site POST is rejected before we
    # spend a DB round-trip validating its session token.
    #
    # Threat model: an attacker page on evil.tld tricks an authenticated
    # user's browser into POSTing to Augmentum. SameSite=Strict cookies
    # already block this in modern browsers, but we belt-and-braces the
    # defense at the server: every state-changing method that arrives
    # with the session cookie must carry an Origin/Referer pointing to
    # the same host as the request itself. Bearer-token requests are
    # exempt — the attacker can't forge an `Authorization: Bearer …`
    # header from a victim's browser without already having the key.
    _CSRF_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
    # Public endpoints that must work pre-cookie (or with no cookie) and
    # therefore can't enforce the same-origin invariant. Login/setup are
    # the entry points; status is read-only health.
    _CSRF_EXEMPT_PATHS = frozenset({
        "/api/auth/login",
        "/api/auth/setup",
        "/api/auth/status",
    })

    def _csrf_same_host(host: str, origin_or_referer: str) -> bool:
        """True if origin_or_referer's host part matches host."""
        if not origin_or_referer:
            return False
        # Strip scheme://, then strip path/query, leaving just host[:port].
        s = origin_or_referer
        if "://" in s:
            s = s.split("://", 1)[1]
        s = s.split("/", 1)[0]
        return s.casefold() == host.casefold()

    class _CSRFOriginMiddleware:
        def __init__(self, app: ASGIApp) -> None:
            self._app = app

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http":
                await self._app(scope, receive, send)
                return

            method = scope.get("method", "GET").upper()
            if method in _CSRF_SAFE_METHODS:
                await self._app(scope, receive, send)
                return

            path = scope.get("path", "/")
            if path in _CSRF_EXEMPT_PATHS:
                await self._app(scope, receive, send)
                return

            headers = dict(scope.get("headers", []))

            # Bearer-token exemption: an attacker page can't auto-attach an
            # Authorization header (browsers don't do that), so requests
            # presenting one are by construction not CSRF — allow through.
            auth_header = headers.get(b"authorization", b"").decode("latin-1", errors="ignore")
            if auth_header.lower().startswith("bearer "):
                await self._app(scope, receive, send)
                return

            cookie_header = headers.get(b"cookie", b"").decode("latin-1", errors="ignore")
            has_session_cookie = any(
                part.strip().startswith("augmentum_session=")
                for part in cookie_header.split(";")
            )
            if not has_session_cookie:
                # No session cookie + no bearer → request can't act as a
                # logged-in user, so CSRF is moot. Auth middleware will
                # 401 it anyway.
                await self._app(scope, receive, send)
                return

            host = headers.get(b"host", b"").decode("latin-1", errors="ignore")
            origin = headers.get(b"origin", b"").decode("latin-1", errors="ignore")
            referer = headers.get(b"referer", b"").decode("latin-1", errors="ignore")

            # Origin is preferred (always sent on POST in modern browsers
            # and stripped of path/query); Referer is the fallback for
            # clients/proxies that suppress Origin.
            same_origin = (
                _csrf_same_host(host, origin)
                if origin
                else _csrf_same_host(host, referer)
            )

            if not same_origin:
                log.warning(
                    "csrf_origin_rejected",
                    method=method, path=path, host=host,
                    origin=origin or "(missing)",
                    referer=referer[:200] or "(missing)",
                )
                body = b'{"error":"CSRF: Origin/Referer does not match Host"}'
                await send({
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                })
                await send({"type": "http.response.body", "body": body})
                return

            await self._app(scope, receive, send)

    app.add_middleware(_CSRFOriginMiddleware)

    # Iframe-origin guard for sensitive endpoints.
    #
    # SameSite=Strict cookies + the CSRF middleware above already block
    # CROSS-origin attacks against state-changing endpoints. This guard
    # closes the harder case: a same-origin iframe (game-surface local
    # bundle, ZIM knowledge pack, EmulatorJS) running scripts loaded
    # from `/api/knowledge/zim/...` or similar paths. With
    # `allow-same-origin allow-scripts` on those iframes the iframe's
    # JS can `fetch('/api/auth/keys', {credentials: 'include'})` and
    # mint a persistent API key under the user's session — the cookie
    # rides along because same-origin.
    #
    # Defence: for the SHORT LIST of credential-minting / admin-only
    # endpoints, require the Referer header to point at a known
    # "top-level page" path (the SPA shell or `/`). An iframe's
    # auto-attached Referer will carry the iframe's URL (e.g.
    # `/api/knowledge/zim/...`) — pattern match rejects.
    #
    # Bypass surface: a determined attacker inside the iframe could
    # use `fetch(url, {referrerPolicy: 'no-referrer'})` to strip
    # Referer. We close that hole by REQUIRING Referer on the
    # sensitive endpoints — missing Referer fails the gate. Cost: a
    # handful of users running strict browser-level referrer stripping
    # extensions will see 403 on API-key creation. They can either
    # whitelist Augmentum in the extension OR use the parent UI's
    # built-in key management page (which sends Referer).
    #
    # NOT applied to broad write endpoints — only the credential and
    # account-management surface. The risk-reward elsewhere doesn't
    # justify the edge-case Referer-stripping breakage.
    _IFRAME_GUARD_SENSITIVE_PATHS = frozenset({
        # API-key minting — the highest-leverage abuse target. A
        # successful POST returns a token that survives unpairing,
        # session logout, and origin changes.
        "/api/auth/keys",
    })
    # Sensitive PATH PREFIXES (admin surfaces + per-key revoke). All
    # POST/PUT/DELETE under these prefixes go through the guard. Note
    # the trailing slash on `/api/auth/keys/`: it matches per-key
    # revoke (DELETE /api/auth/keys/<id>) without false-matching a
    # hypothetical sibling like `/api/auth/keys-list`.
    _IFRAME_GUARD_SENSITIVE_PREFIXES = (
        "/api/auth/keys/",
        "/api/auth/users",
        "/api/auth/audit",
    )
    # Referer paths the parent app uses. Anything else looks like an
    # iframe (or a script extension stripping Referer to the origin
    # only, which we also reject — the cost of a strict gate).
    #
    # The SPA is mounted at ``/ui/`` (server.py:~6403 — ``app.mount("/ui",
    # StaticFiles(html=True))``), so the legitimate parent app's Referer
    # path is ``/ui/`` or ``/ui/index.html``. Earlier versions assumed the
    # SPA lived at ``/`` and only listed those — this rejected legitimate
    # API-key minting from the main app. The bare ``/`` paths are kept
    # for any future setup that serves the SPA at the root.
    _IFRAME_GUARD_TOP_LEVEL_PATHS = frozenset({
        "", "/", "/index.html",
        "/ui/", "/ui/index.html",
    })

    def _iframe_guard_path_matches(path: str) -> bool:
        if path in _IFRAME_GUARD_SENSITIVE_PATHS:
            return True
        return any(path.startswith(pfx) for pfx in _IFRAME_GUARD_SENSITIVE_PREFIXES)

    def _iframe_guard_referer_ok(referer: str, host: str) -> bool:
        """True if Referer is same-host AND its PATH is a known top-level page.

        Returns False on empty Referer (forces explicit Referer presence
        — closes the `referrerPolicy: 'no-referrer'` bypass). The host
        check is redundant with the CSRF middleware that runs before us,
        but kept here so the gate is self-contained.
        """
        if not referer:
            return False
        # Strip scheme://: extract host + path.
        s = referer
        if "://" in s:
            s = s.split("://", 1)[1]
        # Split host from path on the first slash.
        idx = s.find("/")
        if idx < 0:
            referer_host = s
            path = "/"
        else:
            referer_host = s[:idx]
            path = s[idx:]
        # Truncate query/fragment from path.
        for sep in ("?", "#"):
            cut = path.find(sep)
            if cut >= 0:
                path = path[:cut]
        # Host must match (case-insensitive). Empty Host means we can't
        # validate, so refuse.
        if not host or referer_host.casefold() != host.casefold():
            return False
        return path in _IFRAME_GUARD_TOP_LEVEL_PATHS

    class _IframeOriginGuard:
        def __init__(self, app: ASGIApp) -> None:
            self._app = app

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http":
                await self._app(scope, receive, send)
                return
            method = scope.get("method", "GET").upper()
            # Read-only methods don't need the gate; only state-changing.
            if method in _CSRF_SAFE_METHODS:
                await self._app(scope, receive, send)
                return
            path = scope.get("path", "/")
            if not _iframe_guard_path_matches(path):
                await self._app(scope, receive, send)
                return

            headers = dict(scope.get("headers", []))
            # Bearer-token exemption — same rationale as the CSRF
            # middleware above. An API key holder is already
            # authenticated; this gate targets cookie-borne sessions.
            auth_header = headers.get(b"authorization", b"").decode("latin-1", errors="ignore")
            if auth_header.lower().startswith("bearer "):
                await self._app(scope, receive, send)
                return

            referer = headers.get(b"referer", b"").decode("latin-1", errors="ignore")
            host = headers.get(b"host", b"").decode("latin-1", errors="ignore")
            if _iframe_guard_referer_ok(referer, host):
                await self._app(scope, receive, send)
                return

            log.warning(
                "iframe_origin_guard_rejected",
                method=method, path=path,
                referer=referer[:200] or "(missing)",
            )
            body = b'{"error":"This endpoint must be called from the main app, not an embedded preview"}'
            await send({
                "type": "http.response.start",
                "status": 403,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": body})

    app.add_middleware(_IframeOriginGuard)

    # HTTP→HTTPS redirect for browser navigations from a non-loopback host.
    # Runs OUTERMOST (added last) so it fires before auth / CSRF / etc.
    # for requests we're about to redirect anyway — no point CSRF-checking
    # a 307.
    #
    # Why this exists: the augmentum container exposes HTTP on 6100 for
    # localhost development, container-to-container traffic from Caddy,
    # and fabric peer-probe requests over plain HTTP (no TLS needed
    # there — peer auth uses signed envelopes, not TLS identity). But
    # when a browser hits http://<lan-ip>:6100 the page LOADS (we serve
    # HTML) but the UI's JS bootstrap silently fails because the browser
    # gates ``crypto.subtle`` / ``serviceWorker.register`` / etc. behind
    # secure-context (HTTPS or localhost only). The operator sees a
    # half-rendered shell with no error.
    #
    # Redirect rules (any "no" exits to pass-through):
    #   - Method is GET                                     (state-mutating verbs aren't browser nav)
    #   - X-Forwarded-Proto != "https"                       (Caddy already terminated TLS)
    #   - Host is not loopback / localhost                   (loopback HTTP is fine)
    #   - Accept includes "text/html"                        (filters API + asset requests)
    #   - Path is not a known HTTP-required endpoint         (fabric /hello, /pair, internal health)
    #
    # The match is tight enough that fabric peer-to-peer flows, ollama
    # /api/* clients that hit us over HTTP, and direct curl probes all
    # pass through untouched. Only a human typing the URL into a browser
    # bar gets redirected.
    _HTTPS_REDIRECT_BYPASS_PATHS = (
        "/api/fabric/hello",
        "/api/fabric/pair",
        "/api/auth/status",  # cheap pre-login pings should not bounce
        "/health",
    )
    _HTTPS_REDIRECT_LOOPBACK_HOSTS = frozenset({
        "localhost", "127.0.0.1", "::1", "0.0.0.0",
    })

    class _HttpsRedirectMiddleware:
        """Bounce browser HTTP nav from LAN IPs to the HTTPS edge.

        Pure ASGI middleware (no Starlette base) so we can short-circuit
        before any downstream cost. Disabled when
        ``settings.https_redirect_lan`` is False.
        """

        def __init__(self, app: ASGIApp) -> None:
            self._app = app
            self._port = int(settings.https_redirect_port or 6443)

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http" or not settings.https_redirect_lan:
                await self._app(scope, receive, send)
                return

            if scope.get("method", "").upper() != "GET":
                await self._app(scope, receive, send)
                return

            # Headers in scope are list[(bytes, bytes)] with lowercased
            # names. Fish out the three we care about without allocating
            # a dict.
            host_b = b""
            xfp_b = b""
            accept_b = b""
            for name, value in scope.get("headers", ()):
                if name == b"host":
                    host_b = value
                elif name == b"x-forwarded-proto":
                    xfp_b = value
                elif name == b"accept":
                    accept_b = value

            if xfp_b.lower() == b"https":
                # Caddy proxied us — already TLS-terminated.
                await self._app(scope, receive, send)
                return

            host_str = host_b.decode("latin-1", errors="ignore")
            host_only = host_str.split(":", 1)[0].strip().lower()
            if host_only in _HTTPS_REDIRECT_LOOPBACK_HOSTS or host_only.endswith(".localhost"):
                await self._app(scope, receive, send)
                return

            # Path bypass list — fabric peer probes, health endpoints
            path = scope.get("path", "")
            if any(path.startswith(p) for p in _HTTPS_REDIRECT_BYPASS_PATHS):
                await self._app(scope, receive, send)
                return

            # Only redirect human-driven navigations. API clients and
            # asset fetches use Accept: application/json or */*, not
            # text/html, so this filter is cheap and reliable.
            if b"text/html" not in accept_b.lower():
                await self._app(scope, receive, send)
                return

            query = scope.get("query_string", b"").decode("latin-1", errors="ignore")
            target = f"https://{host_only}:{self._port}{path}"
            if query:
                target += f"?{query}"

            # 307 preserves the method, which matters in the unlikely
            # case someone redirects a POST. (Filtered to GETs above, so
            # in practice this is always GET-to-GET.)
            response_headers = [
                (b"location", target.encode("latin-1")),
                (b"content-length", b"0"),
                (b"cache-control", b"no-store"),
            ]
            await send({
                "type": "http.response.start",
                "status": 307,
                "headers": response_headers,
            })
            await send({
                "type": "http.response.body",
                "body": b"",
                "more_body": False,
            })

    app.add_middleware(_HttpsRedirectMiddleware)

    # Health check — mimics Ollama's root endpoint
    @app.get("/", response_class=PlainTextResponse)
    async def health_check() -> str:
        return "Ollama is running"

    @app.get("/api/capabilities")
    async def capabilities(request: Request) -> JSONResponse:
        """Report server capabilities so the UI can adapt."""
        from augmentum.proxy.capabilities_routes import build_capability_payload

        return JSONResponse(await build_capability_payload(request))

    @app.post("/api/backends/discover")
    async def discover_backends(request: Request) -> JSONResponse:
        """Rescan local network for LLM servers.

        Pass {"clear_dismissed": true} to also clear previously dismissed services.
        """
        registry = getattr(app.state, "provider_registry", None)
        if not registry:
            return JSONResponse({"discovered": []})

        # Optionally clear dismissed list for a fresh scan
        try:
            body = await request.json()
        except Exception:
            body = {}
        if body.get("clear_dismissed"):
            registry._dismissed_urls.clear()
            ss = getattr(app.state, "settings_store", None)
            if ss:
                await registry.save_dismissed_discoveries(ss)

        await registry.discover_local_backends()
        await registry.probe_backends()
        return JSONResponse({
            "discovered": registry._discovered,
            "backends": list(registry.backends.keys()),
        })

    @app.post("/api/backends/remove")
    async def remove_backend(request: Request) -> JSONResponse:
        """Remove a discovered backend and persist the dismissal."""
        registry = getattr(app.state, "provider_registry", None)
        if not registry:
            return JSONResponse({"error": "no registry"}, status_code=500)
        body = await request.json()
        key = body.get("key", "")

        # Find the URL before removing (for persistent dismissal)
        dismissed_url = ""
        for d in registry._discovered:
            if d["key"] == key:
                dismissed_url = d.get("url", "")
                break

        try:
            registry.unregister_backend(key)
            registry._discovered = [d for d in registry._discovered if d["key"] != key]

            # Persist the dismissal so it survives restarts
            if dismissed_url:
                registry.dismiss_discovery_url(dismissed_url)
                ss = getattr(app.state, "settings_store", None)
                if ss:
                    await registry.save_dismissed_discoveries(ss)

            return JSONResponse({"status": "removed", "key": key})
        except ValueError as e:
            from augmentum.utils.secrets import sanitize_error_detail
            return JSONResponse(
                {"error": sanitize_error_detail(str(e))}, status_code=400,
            )

    @app.get("/api/tools")
    async def list_tools(surface: str = "") -> JSONResponse:
        """List registered tools with their schemas for UI consumption.

        ``?surface=flow`` filters to tools exposed to reasoning-flow steps
        (SurfaceExposure.flow) so the flow editor doesn't offer
        conversational action verbs as step tools.
        """
        from augmentum.capabilities.frontdesk import CapabilityContext
        from augmentum.capabilities.tool_host import ToolHost

        return JSONResponse(
            ToolHost().list_ui_tools(CapabilityContext(app.state), surface=surface)
        )

    @app.get("/api/tools/metrics")
    async def tool_metrics() -> JSONResponse:
        """Return per-tool call counts, success rates, and latencies."""
        from augmentum.capabilities.frontdesk import CapabilityContext
        from augmentum.capabilities.tool_host import ToolHost

        return JSONResponse(ToolHost().metrics(CapabilityContext(app.state)))

    @app.get("/api/health")
    async def deep_health() -> JSONResponse:
        """Deep health check — probes all registered backends."""
        registry = getattr(app.state, "provider_registry", None)
        backends_status: dict[str, str] = {}
        if registry:
            for key, backend in registry.backends.items():
                try:
                    models = await backend.list_models()
                    backends_status[key] = f"ok ({len(models)} models)"
                except Exception as exc:
                    # repr() so ConnectTimeout(TimeoutError()) and similar
                    # message-less exceptions still tell us the type.
                    log.warning("health_check_backend_error", backend=key, error=repr(exc))
                    backends_status[key] = "error: unreachable"
        healthy = all("ok" in v for v in backends_status.values()) if backends_status else False
        return JSONResponse(
            {"status": "healthy" if healthy else "degraded", "backends": backends_status},
            status_code=200 if healthy else 503,
        )

    @app.get("/api/ui-version")
    async def ui_version() -> JSONResponse:
        """Content digest of the served UI shell — the native-app bundle handshake.

        The Android shell compares its baked SHELL_DIGEST to this; an exact
        match lets it serve the shell from the APK (zero network RTT). Any
        mismatch (server updated, app not rebuilt, filter drift) → the app
        network-loads as before. Cheap: stat-walk unless the shell changed.
        """
        import asyncio
        from pathlib import Path as _Path

        _ui = _Path(__file__).resolve().parent.parent.parent / "ui"
        digest = await asyncio.to_thread(_ui_shell_digest, _ui)
        return JSONResponse(
            {"digest": digest},
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/api/health/services")
    async def service_health(request: Request) -> JSONResponse:
        """Service health registry — lightweight status of all tracked dependencies."""
        health = getattr(request.app.state, "service_health", None)
        if not health:
            return JSONResponse({})
        return JSONResponse(health.snapshot())

    @app.get("/api/selfedit/health")
    async def selfedit_health(request: Request) -> JSONResponse:
        """The Application Health Signal — one comprehensive, app-wide health
        report composed from live telemetry (backends, services, DB integrity,
        strain). This is the single source of truth the self-edit promotion gate
        and rollback consume; exposed here so it's observable too."""
        if request.scope.get("user") is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        from augmentum.selfedit import health as _h
        report = await _h.assess(ref="live", at=_h.now())
        return JSONResponse(report.to_dict(), status_code=200 if report.ok else 503)

    @app.get("/api/selfedit/attempts")
    async def selfedit_attempts(request: Request) -> JSONResponse:
        """The self-edit lineage — the permanent, never-pruned archive of every
        attempt (objective, candidate, gate verdict, outcome, lesson). User-scoped.
        Read-only here; the Workshop surface (P3) consumes it."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        from augmentum.selfedit.growth_db import get_growth_conn
        conn = await get_growth_conn(request.app.state)
        if conn is None:
            return JSONResponse({"attempts": [], "note": "growth store unavailable"})
        from augmentum.selfedit import store as _s
        try:
            limit = max(1, min(int(request.query_params.get("limit", 50)), 500))
        except (TypeError, ValueError):
            limit = 50
        attempts = await _s.list_attempts(conn, user_id=user.id, limit=limit)
        return JSONResponse({"attempts": attempts})

    @app.get("/api/selfedit/attempts/{attempt_id}")
    async def selfedit_attempt(attempt_id: str, request: Request) -> JSONResponse:
        """One self-edit attempt by id (user-scoped)."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        from augmentum.selfedit.growth_db import get_growth_conn
        conn = await get_growth_conn(request.app.state)
        if conn is None:
            return JSONResponse({"error": "growth store unavailable"}, status_code=503)
        from augmentum.selfedit import store as _s
        attempt = await _s.get_attempt(conn, attempt_id=attempt_id, user_id=user.id)
        if attempt is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(attempt)

    @app.get("/api/selfedit/preferences")
    async def selfedit_preferences(request: Request) -> JSONResponse:
        """What the system has learned about you — kept/reverted per change-shape,
        confidence, and which shapes are trusted (lifted toward 'probable')."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        from augmentum.selfedit.growth_db import get_growth_conn
        from augmentum.selfedit.preferences import (
            MIN_SAMPLES,
            TRUST_THRESHOLD,
            PreferenceStore,
        )
        gconn = await get_growth_conn(request.app.state)
        if gconn is None:
            return JSONResponse({"preferences": [], "note": "growth store unavailable"})
        rows = [s.to_dict() for s in await PreferenceStore(gconn).summary(user_id=user.id)]
        return JSONResponse({"preferences": rows, "min_samples": MIN_SAMPLES,
                             "trust_threshold": TRUST_THRESHOLD})

    @app.get("/api/selfedit/activation")
    async def selfedit_activation(request: Request) -> JSONResponse:
        """The verified skill graph — what the system has learned WORKS, derived as
        a pure fold of the never-pruned archive. Read-only selection signal: the
        regions of verified success vs repeated failure, plus an optional scored
        query (``?surface=&files=a,b``). Changes no autonomy; it only advises."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        from augmentum.selfedit import activation as _act
        from augmentum.selfedit.growth_db import get_growth_conn
        gconn = await get_growth_conn(request.app.state)
        if gconn is None:
            return JSONResponse({"graph": {}, "note": "growth store unavailable"})
        graph, calibration = await _act.load_graph_and_calibration(gconn, user_id=user.id)
        out: dict = {"graph": graph.to_dict(), "calibration": calibration.to_dict()}
        surface = (request.query_params.get("surface") or "").strip()
        files = [f for f in (request.query_params.get("files") or "").split(",") if f.strip()]
        if surface or files:
            out["query"] = graph.score(_act.query_atoms(surface=surface, files=files)).to_dict()
        return JSONResponse(out)

    @app.get("/api/selfedit/palate")
    async def selfedit_palate(request: Request) -> JSONResponse:
        """The Palate — a legible, per-user model of YOUR taste, distilled from the
        archive's keep/revert labels. Read-only: "what the Palate has learned about
        you" (per-shape keep-rates as plain statements) plus, with ``?surface=&
        intent=&files=a,b``, its verdict on a proposed change (p_keep + confidence
        + rationale). Cold-start honest — it defers to you until calibrated, and
        changes no autonomy (the palatable tier is a later, earned phase)."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        from augmentum.selfedit import palate as _palate
        from augmentum.selfedit import store as _store
        from augmentum.selfedit.growth_db import get_growth_conn
        gconn = await get_growth_conn(request.app.state)
        if gconn is None:
            return JSONResponse({"profile": {}, "note": "growth store unavailable"})
        attempts = await _store.list_attempts(gconn, user_id=user.id, limit=500)
        from augmentum.selfedit.preferences import PreferenceStore
        prefs = [p.to_dict() for p in
                 await PreferenceStore(gconn).summary(user_id=user.id)]
        out: dict = {"profile": _palate.palate_profile(attempts, prefs)}
        surface = (request.query_params.get("surface") or "").strip()
        intent = (request.query_params.get("intent") or "").strip()
        files = [f for f in (request.query_params.get("files") or "").split(",") if f.strip()]
        if surface or intent or files:
            model = _palate.build_palate(attempts, prefs)
            feats = _palate.features_from_target(
                surface=surface, intent_class=intent, files=files)
            out["query"] = model.assess(feats).to_dict()
        return JSONResponse(out)

    @app.get("/api/selfedit/retrodiction")
    async def selfedit_retrodiction(request: Request) -> JSONResponse:
        """The archive as a labeled benchmark — how many settled (human-decided)
        cases exist to replay candidate graders against, by source and stored
        oracle tier. Read-only; the first sliver of the verification-coverage
        gauge. (Running an actual replay is a caller concern — see
        selfedit/retrodiction.py — this reports the benchmark itself.)"""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        from augmentum.selfedit import store as _store
        from augmentum.selfedit.growth_db import get_growth_conn
        from augmentum.selfedit.retrodiction import benchmark_summary
        from augmentum.selfedit.trust import archive_trust
        gconn = await get_growth_conn(request.app.state)
        if gconn is None:
            return JSONResponse({"benchmark": {}, "note": "growth store unavailable"})
        attempts = await _store.list_attempts(gconn, user_id=user.id, limit=500)
        out: dict = {"benchmark": benchmark_summary(attempts),
                     # the fail-closed integrity gauge: how honest the engine's
                     # verification is right now + the coverage-gap classes that
                     # are the Oracle Foundry's demand signal
                     "trust": archive_trust(attempts),
                     "window_rows": len(attempts)}
        if len(attempts) >= 500:  # list_attempts cap — never read as the whole archive
            out["note"] = "summarizes the latest 500 archive rows, not the full archive"
        return JSONResponse(out)

    @app.get("/api/selfedit/coverage")
    async def selfedit_coverage(request: Request) -> JSONResponse:
        """The Oracle Foundry's coverage map: (surface × intent-class) →
        best-oracle-tier over the archive — the autonomy frontier — plus the
        ranked worklist of interruption clusters, each carrying a composed
        oracle-authoring objective for the CAPABILITY lane. Read-only; the
        human picks a cell and fires — the foundry never launches an edit."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        from augmentum.selfedit import store as _store
        from augmentum.selfedit.foundry import coverage_summary
        from augmentum.selfedit.growth_db import get_growth_conn
        gconn = await get_growth_conn(request.app.state)
        if gconn is None:
            return JSONResponse({"cells": [], "worklist": [],
                                 "note": "growth store unavailable"})
        attempts = await _store.list_attempts(gconn, user_id=user.id, limit=500)
        out = coverage_summary(attempts)
        out["window_rows"] = len(attempts)
        if len(attempts) >= 500:  # list_attempts cap — same honesty as retrodiction
            out["note"] = "summarizes the latest 500 archive rows, not the full archive"
        return JSONResponse(out)

    @app.post("/api/selfedit/ingest/git")
    async def selfedit_ingest_git(request: Request) -> JSONResponse:
        """Backfill the live repo's commit history into the archive
        (ingest-all-work, source=``git``). Idempotent — re-runs skip existing
        rows. Gated by ``selfedit_enabled``; needs the wired repo dir."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse(
                {"error": "self-edit is disabled", "hint": "enable selfedit_enabled to use this"},
                status_code=403)
        from augmentum.selfedit.growth_db import get_growth_conn
        from augmentum.selfedit.ingest import ingest_git_history
        gconn = await get_growth_conn(request.app.state)
        if gconn is None:
            return JSONResponse({"error": "growth store unavailable"}, status_code=503)
        repo_dir = getattr(request.app.state, "selfedit_repo_dir", "")
        if not repo_dir:
            return JSONResponse(
                {"error": "no repo wired", "hint": "selfedit_repo_dir is not configured"},
                status_code=503)
        body = {}
        with contextlib.suppress(Exception):
            body = await request.json()
        limit = int(body.get("limit") or 2000)
        result = await ingest_git_history(gconn, repo_dir=repo_dir,
                                          user_id=user.id, limit=limit)
        return JSONResponse(result, status_code=200 if result.get("ok") else 502)

    async def _selfedit_enabled(request: Request) -> bool:
        """The gate every ``/api/selfedit/*`` route consults.

        Two conditions, both required:

        1. ``AUGMENTUM_SELFEDIT_UNLOCK`` is set in the environment — the operator
           has deliberately made the subsystem available on this install. Unset
           on a default install, which is why hiding the UI is not the whole
           story: without this check a curious user could still drive the loop
           with curl. Locked installs behave as though the routes don't exist.
        2. The ``selfedit_enabled`` master switch is on (default OFF).
        """
        from augmentum.config import selfedit_unlocked

        if not selfedit_unlocked():
            return False
        store = getattr(request.app.state, "settings_store", None)
        if store is None:
            return False
        val = await store.get("selfedit_enabled")
        return (val or "").strip().lower() in ("1", "true", "yes", "on")

    async def _heal_attempts(request: Request) -> int:
        """The self-heal repair budget (``selfedit_self_heal_attempts``, default 2)
        — how many times a fixable verification break is fed back to the model for
        a same-worktree repair before rejecting/escalating."""
        store = getattr(request.app.state, "settings_store", None)
        if store is None:
            return 2
        try:
            return max(0, min(5, int((await store.get("selfedit_self_heal_attempts")) or 2)))
        except (TypeError, ValueError):
            return 2

    async def _read_demand(request: Request, user_id: str) -> list:
        """Lived user friction (open ``signal_events``, MAIN DB) as structural
        demand DebtTargets for the needs-you lane. Best-effort + strictly
        user-scoped: any failure yields [] and the loop stays audit-only."""
        try:
            backend = request.app.state.state_manager.backend
            from augmentum.selfedit.demand import demand_targets
            return await demand_targets(backend.conn, user_id=user_id)
        except Exception as exc:  # noqa: BLE001 — demand degrades to audit-only
            log.warning("selfedit_read_demand_failed", error=repr(exc))
            return []

    @app.post("/api/selfedit/propose")
    async def selfedit_propose(request: Request) -> JSONResponse:
        """Propose a self-edit. Gated by the ``selfedit_enabled`` master switch
        (default OFF). Returns the debt-paydown PLAN (a dry-run triage of what the
        loop would attempt) — the safe preview. Live editing additionally requires
        a configured edit driver (``app.state.selfedit_driver`` + repo); until that
        is wired this stays preview-only by design."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse(
                {"error": "self-edit is disabled", "hint": "enable selfedit_enabled to use this"},
                status_code=403)
        from augmentum.selfedit.growth_db import get_growth_conn
        conn = await get_growth_conn(request.app.state)
        if conn is None:
            return JSONResponse({"error": "growth store unavailable"}, status_code=503)

        body = {}
        with contextlib.suppress(Exception):
            body = await request.json()
        driver = getattr(request.app.state, "selfedit_driver", None)
        repo_dir = getattr(request.app.state, "selfedit_repo_dir", "")
        live = bool(body.get("live")) and driver is not None and bool(repo_dir)
        fresh = bool(body.get("fresh"))  # re-run the full audit (~minutes) vs cached
        from augmentum.selfedit import loop as _loop
        from augmentum.selfedit import scanners as _sc

        async def _cached_audit(_dir):
            """Read the last full audit from history (instant) so the plan opens
            fast. The full audit (`fresh`) is slow (~minutes) and reserved for an
            explicit re-audit. Falls back to a live run only if no history exists."""
            import os
            rel = os.path.join(".claude", "skills", "augmentum-dev", "references",
                               "audit_history.jsonl")
            for base in (repo_dir or ".", ".", "/host-augmentum-src"):
                try:
                    with open(os.path.join(base, rel), encoding="utf-8") as f:
                        lines = [ln for ln in f.read().splitlines() if ln.strip()]
                    if lines:
                        return lines[-1]
                except (FileNotFoundError, OSError):
                    continue
            return await _sc.default_audit_runner(_dir)

        from augmentum.selfedit.preferences import PreferenceStore

        # Escalation ladder (cheapest first): a local model does the groundwork →
        # the user's primary → a frontier model (cost-gated by allow_frontier). All
        # rungs run through the SAME native loop; only the model changes, with each
        # rung's findings carried up so the stronger model doesn't repeat the work.
        rungs: list = []
        allow_frontier = bool(body.get("allow_frontier"))
        if live:
            with contextlib.suppress(Exception):
                from augmentum.selfedit.escalate import RungSpec, build_ladder
                _store = request.app.state.settings_store
                _edit = (await _store.get("selfedit_edit_model")) or ""
                _primary = (await _store.get("primary_chat_model")) or ""
                _frontier = (await _store.get("selfedit_frontier_model")) or ""
                specs = [RungSpec(model=_edit, label=(_edit or "local utility"))]
                if _primary and _primary != _edit:
                    specs.append(RungSpec(model=_primary, label=_primary))
                if _frontier and _frontier not in (_edit, _primary):
                    specs.append(RungSpec(model=_frontier, label=_frontier, frontier=True))
                rungs = await build_ladder(
                    conn, request.app.state.provider_registry, specs,
                    allow_frontier=allow_frontier,
                )
        # Baseline tree for evidence grounding: the full-checkout source at HEAD.
        evidence_tree = getattr(request.app.state, "selfedit_source_dir", "") if live else ""
        # Demand-side: lived user friction (signal_events, MAIN DB) joins the
        # needs-you lane beside audit findings. Best-effort, strictly user-scoped.
        demand = await _read_demand(request, user.id)
        try:
            report = await _loop.run_debt_loop(
                repo_dir=repo_dir or ".", user_id=user.id, conn=conn,
                driver=driver or _loop.null_edit_driver,
                live_audit_runner=(_sc.default_audit_runner if fresh else _cached_audit),
                max_attempts=int(body.get("max_attempts", 1) or 1),
                dry_run=not live,
                preference_store=PreferenceStore(conn),
                rungs=(rungs or None),
                evidence_tree=(evidence_tree or None),
                target_id=str(body.get("target", "") or ""),
                demand=demand,
                max_heal_attempts=await _heal_attempts(request),
            )
        except Exception as exc:  # noqa: BLE001 — surface, never 500 silently
            log.warning("selfedit_propose_failed", error=repr(exc))
            return JSONResponse({"error": f"propose failed: {exc!r}"}, status_code=500)
        out = report.to_dict()
        out["live"] = live
        out["fresh"] = fresh
        # Annotate the structural (human-decides) items with the verified skill
        # graph's per-debt-class trust — read-only context so the reviewer sees
        # "the system has landed this class before" vs "this class keeps getting
        # reverted." Cold (low confidence) until attempts of the class accrue;
        # never reorders or auto-decides — it only informs the human at the
        # decision point. Best-effort: a graph failure never breaks the preview.
        try:
            from augmentum.selfedit import activation as _act
            graph = await _act.load_graph(conn, user_id=user.id)
            for item in out.get("structural", []):
                sig = graph.score_target(item.get("scanner", ""), item.get("metric", ""))
                if sig.confidence >= 0.2:
                    item["region_trust"] = sig.to_dict()
        except Exception as exc:  # noqa: BLE001 — advisory annotation only
            log.warning("selfedit_propose_annotate_failed", error=repr(exc))
        # The Palate: annotate each needs-you item with a taste prediction — "the
        # system thinks you'll (keep|revert) this" — so the human spends attention
        # where it teaches most (uncertain items) and can fast-track confident
        # keeps. Read-only, cold-start honest (only attaches where it 'speaks'),
        # never reorders or decides. Distilled from settled keep/revert history.
        try:
            from augmentum.selfedit import palate as _palate
            from augmentum.selfedit import store as _pstore
            from augmentum.selfedit.preferences import PreferenceStore as _PrefStore
            _attempts = await _pstore.list_attempts(conn, user_id=user.id, limit=500)
            _prefs = [p.to_dict() for p in await _PrefStore(conn).summary(user_id=user.id)]
            _model = _palate.build_palate(_attempts, _prefs)
            if _model.n_labels:
                for item in out.get("structural", []):
                    feats = _palate.features_from_target(
                        surface=item.get("surface", "") or "",
                        intent_class=item.get("metric", "") or "",
                        origin=item.get("origin", "audit"))
                    pv = _model.assess(feats)
                    if pv.speaks:
                        item["palate"] = pv.to_dict()
        except Exception as exc:  # noqa: BLE001 — advisory annotation only
            log.warning("selfedit_propose_palate_failed", error=repr(exc))
        # Driver availability, independent of this call's live/dry-run mode — a
        # dry-run preview always reports live=False, so the UI needs a separate
        # signal to know the green lane CAN run (else it wrongly disables Fix).
        driver_ready = driver is not None and bool(repo_dir)
        out["driver_ready"] = driver_ready
        out["ladder"] = [getattr(r, "label", "") for r in rungs]
        out["allow_frontier"] = allow_frontier
        if not live:
            out["note"] = (
                "preview only — pass live to run the auto-lane; structural items are for your review"
                if driver_ready else
                "the coder driver isn't connected yet — the auto-lane runs once it "
                "is; structural items are for your review")
        return JSONResponse(out)

    @app.post("/api/selfedit/run")
    async def selfedit_run(request: Request) -> JSONResponse:
        """Launch a self-edit as a SERVER-OWNED background run and return its
        ``run_id`` immediately, so the client can attach to the live stream and be
        guided through the pipeline (target → agent → verify → verdict) as it
        happens — instead of blocking minutes for the final verdict. Gated by
        ``selfedit_enabled``; requires a wired edit driver."""
        nonlocal_app_state = request.app.state
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse({"error": "self-edit is disabled"}, status_code=403)
        from augmentum.selfedit.growth_db import get_growth_conn
        conn = await get_growth_conn(nonlocal_app_state)
        if conn is None:
            return JSONResponse({"error": "growth store unavailable"}, status_code=503)
        driver = getattr(nonlocal_app_state, "selfedit_driver", None)
        repo_dir = getattr(nonlocal_app_state, "selfedit_repo_dir", "")
        if driver is None or not repo_dir:
            return JSONResponse(
                {"error": "the edit driver isn't connected — can't run a live edit yet"},
                status_code=409)
        body = {}
        with contextlib.suppress(Exception):
            body = await request.json()
        target = str(body.get("target", "") or "")
        allow_frontier = bool(body.get("allow_frontier"))
        fresh = bool(body.get("fresh"))
        max_attempts = int(body.get("max_attempts", 1) or 1)

        from augmentum.selfedit import loop as _loop
        from augmentum.selfedit import scanners as _sc
        from augmentum.selfedit.live import launch_live_run
        from augmentum.selfedit.preferences import PreferenceStore

        async def _cached_audit(_dir):
            import os
            rel = os.path.join(".claude", "skills", "augmentum-dev", "references",
                               "audit_history.jsonl")
            for base in (repo_dir or ".", ".", "/host-augmentum-src"):
                try:
                    with open(os.path.join(base, rel), encoding="utf-8") as f:
                        lines = [ln for ln in f.read().splitlines() if ln.strip()]
                    if lines:
                        return lines[-1]
                except (FileNotFoundError, OSError):
                    continue
            return await _sc.default_audit_runner(_dir)

        # Build the ladder (settings → edit/primary/frontier).
        rungs: list = []
        with contextlib.suppress(Exception):
            from augmentum.selfedit.escalate import RungSpec, build_ladder
            store = nonlocal_app_state.settings_store
            _edit = (await store.get("selfedit_edit_model")) or ""
            _primary = (await store.get("primary_chat_model")) or ""
            _frontier = (await store.get("selfedit_frontier_model")) or ""
            specs = [RungSpec(model=_edit, label=(_edit or "local utility"))]
            if _primary and _primary != _edit:
                specs.append(RungSpec(model=_primary, label=_primary))
            if _frontier and _frontier not in (_edit, _primary):
                specs.append(RungSpec(model=_frontier, label=_frontier, frontier=True))
            rungs = await build_ladder(conn, nonlocal_app_state.provider_registry,
                                       specs, allow_frontier=allow_frontier)
        evidence_tree = getattr(nonlocal_app_state, "selfedit_source_dir", "")
        ladder_labels = [getattr(r, "label", "") for r in rungs]

        import uuid as _uuid
        run_id = _uuid.uuid4().hex
        title = f"Fix {target}" if target else "Fix the top auto-lane finding"

        async def _coro() -> dict:
            report = await _loop.run_debt_loop(
                repo_dir=repo_dir or ".", user_id=user.id, conn=conn,
                driver=driver,
                live_audit_runner=(_sc.default_audit_runner if fresh else _cached_audit),
                max_attempts=max_attempts, dry_run=False,
                preference_store=PreferenceStore(conn),
                rungs=(rungs or None),
                evidence_tree=(evidence_tree or None),
                target_id=target,
                max_heal_attempts=await _heal_attempts(request),
            )
            attempted = report.attempted or []
            gated = [o for o in attempted if o.status == "gated"]
            last = attempted[-1] if attempted else None
            return {
                "status": "done", "ok": True,
                "attempts": len(attempted), "gated": len(gated),
                "final_status": (last.status if last else "none"),
                "tier": (last.verdict.tier if (last and last.verdict) else ""),
                "baseline_score": round(report.baseline_score, 1),
            }

        launch_live_run(
            nonlocal_app_state, user_id=user.id, run_id=run_id, title=title,
            target=target, ladder=ladder_labels, coro_factory=_coro,
        )
        return JSONResponse({"run_id": run_id, "title": title, "target": target,
                             "ladder": ladder_labels, "allow_frontier": allow_frontier})

    @app.get("/api/selfedit/run/{run_id}/stream")
    async def selfedit_run_stream(run_id: str, request: Request):
        """SSE: replay the run's buffered events (past ``?since``), then tail the
        live bus until the terminal ``done``. Closing it only detaches — the
        server-owned run keeps going (re-attach with a higher ``since``)."""
        import json as _json

        from fastapi.responses import StreamingResponse

        from augmentum.selfedit.live import get_live_run_manager
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse({"error": "self-edit is disabled"}, status_code=403)
        since = 0
        with contextlib.suppress(Exception):
            since = int(request.query_params.get("since", "0") or 0)
        mgr = get_live_run_manager(request.app.state)
        run = mgr.get(run_id)

        def _sse(obj: dict) -> str:
            return f"data: {_json.dumps(obj)}\n\n"

        async def gen():
            if run is None or run.user_id != user.id:
                yield _sse({"kind": "failed", "text": "run not found"})
                return
            # Subscribe BEFORE snapshotting so events in the gap are caught + deduped.
            q = run.subscribe() if not run.finished.is_set() else None
            last = since
            for ev in run.snapshot(since=since)["events"]:
                last = ev.get("seq") or last
                yield _sse(ev)
            if q is None or run.finished.is_set():
                if q is not None:
                    run.unsubscribe(q)
                # ensure the viewer gets a terminal frame even if it attached late
                if run.result is not None:
                    yield _sse({"kind": "done", **run.result})
                return
            try:
                while True:
                    item = await q.get()
                    s = item.get("seq") or 0
                    if s and s <= last:
                        continue
                    last = s or last
                    yield _sse(item)
                    if item.get("kind") == "done":
                        break
            finally:
                run.unsubscribe(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.get("/api/selfedit/run/{run_id}")
    async def selfedit_run_snapshot(run_id: str, request: Request) -> JSONResponse:
        """A non-streaming snapshot of a live run (status + buffered events past
        ``?since``) — the resume check + a polling fallback for clients that can't
        hold an SSE connection. Returns 404 once the run is evicted (archive holds
        the durable copy)."""
        from augmentum.selfedit.live import get_live_run_manager
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse({"error": "self-edit is disabled"}, status_code=403)
        since = 0
        with contextlib.suppress(Exception):
            since = int(request.query_params.get("since", "0") or 0)
        mgr = get_live_run_manager(request.app.state)
        run = mgr.get(run_id)
        if run is None or run.user_id != user.id:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(run.snapshot(since=since))

    @app.post("/api/selfedit/run/{run_id}/stop")
    async def selfedit_run_stop(run_id: str, request: Request) -> JSONResponse:
        """Cancel a live self-edit run. The candidate worktree is thrown away; the
        attempt is archived as cancelled (the lesson survives)."""
        from augmentum.selfedit.live import get_live_run_manager
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse({"error": "self-edit is disabled"}, status_code=403)
        mgr = get_live_run_manager(request.app.state)
        stopped = await mgr.stop(run_id, user_id=user.id)
        return JSONResponse({"stopped": stopped})

    @app.post("/api/selfedit/capability")
    async def selfedit_capability(request: Request) -> JSONResponse:
        """The self-EVOLVING lane: author a NEW capability from a plain request.
        Synthesizes an acceptance test, then drives the SAME edit engine to
        implement against it — so it streams in the live theater (run_self_edit is
        instrumented) and a verified result lands in the Go-live pending set, just
        like a debt fix. Returns a ``run_id`` to attach to immediately."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse({"error": "self-edit is disabled"}, status_code=403)
        st = request.app.state
        driver = getattr(st, "selfedit_driver", None)
        repo_dir = getattr(st, "selfedit_repo_dir", "")
        if driver is None or not repo_dir:
            return JSONResponse(
                {"error": "the edit driver isn't connected — can't author live yet"},
                status_code=409)
        from augmentum.selfedit.growth_db import get_growth_conn
        conn = await get_growth_conn(st)
        if conn is None:
            return JSONResponse({"error": "growth store unavailable"}, status_code=503)
        body = {}
        with contextlib.suppress(Exception):
            body = await request.json()
        ask = str(body.get("request", "") or "").strip()
        if not ask:
            return JSONResponse({"error": "request is required"}, status_code=400)

        registry = st.provider_registry

        async def _model_invoke(prompt: str) -> str:
            """Plain text-in/text-out for the acceptance-test synthesis. Prefers the
            configured edit model, falls back to the utility role."""
            from augmentum.models.base import InternalChatRequest, Message
            store = st.settings_store
            edit_model = (await store.get("selfedit_edit_model")) or ""
            backend = resolved = None
            if edit_model and hasattr(registry, "resolve_backend_with_fabric"):
                backend, resolved = await registry.resolve_backend_with_fabric(edit_model)
            if backend is None:
                backend, resolved = await registry.resolve_model_for_role("utility")
            if backend is None:
                return ""
            resp = await backend.chat(InternalChatRequest(
                model=resolved or edit_model,
                messages=[Message(role="user", content=prompt)],
                temperature=0.3, max_tokens=2048, stream=False))
            return getattr(getattr(resp, "message", None), "content", "") or ""

        import uuid as _uuid

        from augmentum.selfedit import scanners as _sc
        from augmentum.selfedit.capabilities import author_capability
        from augmentum.selfedit.live import launch_live_run
        from augmentum.selfedit.preferences import PreferenceStore
        run_id = _uuid.uuid4().hex
        title = f"Build: {ask[:48]}"

        async def _coro() -> dict:
            outcome, err = await author_capability(
                ask, repo_dir=repo_dir, conn=conn, driver=driver,
                model_invoke=_model_invoke, user_id=user.id,
                preference_store=PreferenceStore(conn),
                run_audit=_sc.default_audit_runner,  # no-regression check for the verifier
                max_heal_attempts=await _heal_attempts(request),  # self-heal a fixable break
            )
            if outcome is None:
                return {"status": "failed", "ok": False, "error": err or "synthesis failed"}
            return {"status": "done", "ok": True,
                    "final_status": getattr(outcome, "status", ""),
                    "tier": (outcome.verdict.tier if getattr(outcome, "verdict", None) else "")}

        launch_live_run(st, user_id=user.id, run_id=run_id, title=title,
                        target="capability", ladder=[], coro_factory=_coro)
        return JSONResponse({"run_id": run_id, "title": title, "kind": "capability"})

    @app.post("/api/selfedit/debt/advise")
    async def selfedit_debt_advise(request: Request) -> JSONResponse:
        """LLM-agency advisor over the debt list: the model reads what the audit
        actually flagged and recommends the best choices — order, approach,
        grouping, what to skip — versatile, not a fixed map. The deterministic
        triage stays the safety floor (mechanical vs structural/red-tier); the
        agent annotates + prioritizes WITHIN it, never reclassifies. Gated."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse({"error": "self-edit is disabled"}, status_code=403)

        # Triage the latest (cached) audit — the safety floor.
        from augmentum.selfedit import debt as _debt
        from augmentum.selfedit import scanners as _sc2
        repo_dir = getattr(request.app.state, "selfedit_repo_dir", "")
        import os as _os
        rel = _os.path.join(".claude", "skills", "augmentum-dev", "references",
                            "audit_history.jsonl")
        audit_text = ""
        for base in (repo_dir or ".", ".", "/host-augmentum-src"):
            try:
                with open(_os.path.join(base, rel), encoding="utf-8") as f:
                    lines = [ln for ln in f.read().splitlines() if ln.strip()]
                if lines:
                    audit_text = lines[-1]
                    break
            except (FileNotFoundError, OSError):
                continue
        if not audit_text:
            return JSONResponse({"available": False, "note": "no audit available to advise on"})
        triage = _debt.triage(_sc2.parse_audit_json(audit_text))

        # Resolve the user's chat model for the agent's reasoning.
        store = getattr(request.app.state, "settings_store", None)
        pr = getattr(request.app.state, "provider_registry", None)
        model = (await store.get("primary_chat_model")) if store else ""
        if not model or pr is None:
            return JSONResponse({"available": False, "note": "no model available for the advisor"})
        try:
            backend, clean = await pr.resolve_backend_with_fabric(model, user_id=user.id)
        except Exception as exc:  # noqa: BLE001
            log.warning("selfedit_advise_resolve_failed", error=repr(exc))
            backend = None
        if backend is None:
            return JSONResponse({"available": False, "note": "no model backend for the advisor"})

        from augmentum.models.base import InternalChatRequest, Message, response_text
        from augmentum.selfedit.debt_advisor import advise as _advise
        from augmentum.selfedit.prompts import resolved_prompt

        async def _chat(prompt: str) -> str:
            resp = await backend.chat(InternalChatRequest(
                model=clean or model, messages=[Message(role="user", content=prompt)],
                stream=False, max_tokens=1600, temperature=0.2))
            return response_text(resp, thinking_fallback=False)

        # honor an evolved/override advisor prompt if one is set (else the default)
        advisor_prompt = await resolved_prompt("debt_advisor", settings_store=store, user_id=user.id)
        # the learning loop: feed what the user tends to keep/revert as a hint, plus
        # the verified skill-graph (regions that ship vs roll back). Both read-only.
        prefs: list[dict] = []
        activation: dict = {}
        with contextlib.suppress(Exception):
            from augmentum.selfedit import activation as _act
            from augmentum.selfedit.growth_db import get_growth_conn
            from augmentum.selfedit.preferences import PreferenceStore
            gconn = await get_growth_conn(request.app.state)
            if gconn is not None:
                prefs = [s.to_dict() for s in await PreferenceStore(gconn).summary(user_id=user.id)]
                activation = (await _act.load_graph(gconn, user_id=user.id)).to_dict()
        try:
            advice = await _advise(triage, chat=_chat, prompt=advisor_prompt,
                                   preferences=prefs, activation=activation)
        except Exception as exc:  # noqa: BLE001 — never 500; floor still stands
            log.warning("selfedit_advise_failed", error=repr(exc))
            return JSONResponse({"available": False, "note": f"advisor error: {exc!r}"})
        return JSONResponse(advice.to_dict())

    @app.get("/api/selfedit/prompts")
    async def selfedit_prompts(request: Request) -> JSONResponse:
        """The overridable prompts Evolve can target — registered slug, label, the
        canonical default, and the current effective text (override or default).
        Applying an evolved prompt is a config reshape on the spec's key; reverting
        clears it back to the default."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        # importing these modules registers their overridable prompts
        from augmentum.selfedit import debt_advisor as _da  # noqa: F401
        from augmentum.selfedit.prompts import registered_prompts, resolved_prompt
        store = getattr(request.app.state, "settings_store", None)
        out = []
        for slug, spec in registered_prompts().items():
            effective = await resolved_prompt(slug, settings_store=store, user_id=user.id)
            out.append(spec.to_dict(effective=effective))
        return JSONResponse({"prompts": out})

    @app.post("/api/selfedit/evolve/start")
    async def selfedit_evolve_start(request: Request) -> JSONResponse:
        """Start a background evolve session — improve a prompt toward a goal via
        the GEPA loop on the user's model (synthetic eval set, rubric judge,
        reflective mutation, held-out accept). Returns a run_id to poll; the run is
        slow (minutes, many model calls), so it never blocks the request. Gated."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse({"error": "self-edit is disabled"}, status_code=403)
        body = {}
        with contextlib.suppress(Exception):
            body = await request.json()
        prompt = str(body.get("prompt", "")).strip()
        goal = str(body.get("goal", "")).strip()
        if not prompt or not goal:
            return JSONResponse({"error": "prompt and goal are both required"}, status_code=400)

        store = getattr(request.app.state, "settings_store", None)
        pr = getattr(request.app.state, "provider_registry", None)
        model = (await store.get("primary_chat_model")) if store else ""
        if not model or pr is None:
            return JSONResponse({"error": "no model available for evolution"}, status_code=503)
        try:
            backend, clean = await pr.resolve_backend_with_fabric(model, user_id=user.id)
        except Exception as exc:  # noqa: BLE001
            log.warning("selfedit_evolve_resolve_failed", error=repr(exc))
            backend = None
        if backend is None:
            return JSONResponse({"error": "no model backend for evolution"}, status_code=503)

        import uuid as _uuid
        from augmentum.models.base import InternalChatRequest, Message, response_text
        from augmentum.selfedit.evolve import run_evolve_session

        async def _chat(messages: list[dict]) -> str:
            resp = await backend.chat(InternalChatRequest(
                model=clean or model,
                messages=[Message(role=m["role"], content=m["content"]) for m in messages],
                stream=False, max_tokens=1400, temperature=0.4))
            return response_text(resp, thinking_fallback=False)

        runs = getattr(request.app.state, "selfedit_evolve_runs", None)
        if runs is None:
            runs = {}
            request.app.state.selfedit_evolve_runs = runs
        run_id = _uuid.uuid4().hex
        runs[run_id] = {"status": "running", "user_id": user.id, "goal": goal}
        n_cases = max(3, min(int(body.get("n_cases", 6) or 6), 10))
        max_iters = max(1, min(int(body.get("max_iterations", 2) or 2), 3))

        async def _work():
            try:
                result = await run_evolve_session(prompt=prompt, goal=goal, chat=_chat,
                                                  n_cases=n_cases, max_iterations=max_iters)
                d = result.to_dict()
                d["evolved_prompt"] = result.best_variant
                d["baseline_prompt"] = result.baseline
                runs[run_id] = {"status": "done", "user_id": user.id, "result": d}
                log.info("selfedit_evolve_complete", run_id=run_id, accepted=result.accepted)
            except Exception as exc:  # noqa: BLE001 — record the failure, never crash the loop
                log.warning("selfedit_evolve_run_failed", run_id=run_id, error=repr(exc))
                runs[run_id] = {"status": "failed", "user_id": user.id, "error": repr(exc)[:300]}

        asyncio.create_task(_work())
        return JSONResponse({"run_id": run_id, "status": "running"})

    @app.get("/api/selfedit/evolve/{run_id}")
    async def selfedit_evolve_get(run_id: str, request: Request) -> JSONResponse:
        """Poll a background evolve run (running | done | failed), user-scoped."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        runs = getattr(request.app.state, "selfedit_evolve_runs", {}) or {}
        st = runs.get(run_id)
        if not st or st.get("user_id") != user.id:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse({k: v for k, v in st.items() if k != "user_id"})

    @app.post("/api/selfedit/attempts/{attempt_id}/verdict")
    async def selfedit_verdict(attempt_id: str, request: Request) -> JSONResponse:
        """Capture the human verdict on an attempt (keep | revert) — the taste
        oracle and the P7 preference signal. Records it permanently; applies the
        promote/revert when a repo is wired (else records the decision for the
        wired path to honor). Gated by ``selfedit_enabled``."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse({"error": "self-edit is disabled"}, status_code=403)
        from augmentum.selfedit.growth_db import get_growth_conn
        conn = await get_growth_conn(request.app.state)
        if conn is None:
            return JSONResponse({"error": "growth store unavailable"}, status_code=503)

        body = {}
        with contextlib.suppress(Exception):
            body = await request.json()
        decision = str(body.get("decision", "")).strip().lower()
        note = str(body.get("note", ""))[:2000]
        if decision not in ("keep", "revert"):
            return JSONResponse({"error": "decision must be 'keep' or 'revert'"}, status_code=400)

        from augmentum.selfedit import store as _s
        attempt = await _s.get_attempt(conn, attempt_id=attempt_id, user_id=user.id)
        if attempt is None:
            return JSONResponse({"error": "not found"}, status_code=404)

        repo_dir = getattr(request.app.state, "selfedit_repo_dir", "")
        applied = False
        apply_error = ""
        from augmentum.selfedit import promote as _p
        if decision == "keep":
            outcome = f"user kept{': ' + note if note else ''}"
            # apply only when a repo is wired AND there's a candidate to promote.
            # Surface a promote failure (dirty tree, real conflict) instead of
            # swallowing it — a silent apply=False looked like "kept" but the code
            # never landed; the human must know their change didn't take.
            if repo_dir and attempt.get("candidate_ref"):
                try:
                    new_sha = await _p.git_promote(repo_dir, attempt["candidate_ref"])
                    applied = True
                    await _s.finalize(conn, attempt_id=attempt_id, user_id=user.id,
                                      status="promoted", outcome=outcome,
                                      lesson="kept by the user (human-confirmed)",
                                      promoted_commit=new_sha)
                except Exception as exc:  # noqa: BLE001 — surface, never silent
                    apply_error = str(exc)[:400]
                    log.warning("selfedit_verdict_promote_failed",
                                attempt_id=attempt_id, error=apply_error)
            if not applied:
                # The KEEP is still a real taste verdict (it teaches the Palate via
                # the preference tally below) even when the apply couldn't land.
                reason = (f"apply failed: {apply_error}" if apply_error
                          else "apply pending repo wiring")
                await _s.finalize(conn, attempt_id=attempt_id, user_id=user.id,
                                  status=attempt.get("status", "gated"),
                                  outcome=f"{outcome} ({reason})",
                                  lesson="kept by the user (human-confirmed)")
        else:  # revert
            outcome = f"user reverted{': ' + note if note else ''}"
            if repo_dir and attempt.get("promoted_commit"):
                try:
                    await _p.git_revert(repo_dir, attempt["promoted_commit"])
                    applied = True
                except Exception as exc:  # noqa: BLE001 — surface, never silent
                    apply_error = str(exc)[:400]
                    log.warning("selfedit_verdict_revert_failed",
                                attempt_id=attempt_id, error=apply_error)
            await _s.finalize(conn, attempt_id=attempt_id, user_id=user.id,
                              status="rolled_back" if applied else "rejected",
                              outcome=outcome,
                              lesson="reverted by the user — code restored, record kept")
        # The learning loop: teach the system this change-shape from the verdict.
        with contextlib.suppress(Exception):
            from augmentum.selfedit.preferences import PreferenceStore, change_shape
            shape = change_shape(attempt.get("surface", ""),
                                 (attempt.get("gate_verdict") or {}).get("intent_class", ""))
            await PreferenceStore(conn).record(user_id=user.id, shape=shape,
                                               kept=(decision == "keep"))
        return JSONResponse({"ok": True, "decision": decision, "applied": applied,
                             "apply_error": apply_error})

    # ── Staged apply / checkpoints / restart ──────────────────────────────
    # Kept code edits COLLECT in the isolated clone (status='promoted'). These
    # routes let the user take the collected set live in one deliberate step —
    # checkpoint → write to the live tree → restart — and revert to any prior
    # checkpoint. Backend (augmentum/) applies live; frontend (ui/) too once the
    # compose mount is writable.

    def _apply_ctx(request: Request):
        st = request.app.state
        repo_dir = getattr(st, "selfedit_repo_dir", "")
        from augmentum.selfedit import apply as _ap
        return st, repo_dir, _ap

    @app.get("/api/selfedit/pending")
    async def selfedit_pending(request: Request) -> JSONResponse:
        """The staged set: kept edits that are committed in the clone but not yet
        live, with the diff stat + per-file applyability (the ui:ro gate)."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse({"error": "self-edit is disabled"}, status_code=403)
        st, repo_dir, _ap = _apply_ctx(request)
        if not repo_dir:
            return JSONResponse({"error": "no self-edit repo wired"}, status_code=409)
        from augmentum.selfedit.growth_db import get_growth_conn
        conn = await get_growth_conn(st)
        live = _ap.live_tree(st)
        pending = await _ap.compute_pending(repo_dir, live, conn=conn, user_id=user.id)
        out = pending.to_dict()
        out["checkpoints"] = _ap.list_checkpoints(repo_dir)[:10]
        out["ui_writable"] = _ap.subtree_writable(live, "ui")
        return JSONResponse(out)

    @app.post("/api/selfedit/apply")
    async def selfedit_apply(request: Request) -> JSONResponse:
        """Take the collected changes live: checkpoint → write to the live tree →
        (optionally) restart this container. Returns BEFORE the restart so the
        client gets the result and can poll for the app's return."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse({"error": "self-edit is disabled"}, status_code=403)
        st, repo_dir, _ap = _apply_ctx(request)
        if not repo_dir:
            return JSONResponse({"error": "no self-edit repo wired"}, status_code=409)
        from augmentum.selfedit.growth_db import get_growth_conn
        conn = await get_growth_conn(st)
        live = _ap.live_tree(st)
        body = {}
        with contextlib.suppress(Exception):
            body = await request.json()
        do_restart = bool(body.get("restart", True))
        res = await _ap.apply_pending(repo_dir, live, conn=conn, user_id=user.id,
                                      label=str(body.get("label", "") or "apply"))
        out = res.to_dict()
        restarting = do_restart and out["needs_restart"]
        if restarting:
            _ap.schedule_restart(delay=1.5)
        out["restarting"] = restarting
        return JSONResponse(out)

    @app.get("/api/selfedit/checkpoints")
    async def selfedit_checkpoints(request: Request) -> JSONResponse:
        """The restore points — every apply snapshots the prior content first."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse({"error": "self-edit is disabled"}, status_code=403)
        st, repo_dir, _ap = _apply_ctx(request)
        if not repo_dir:
            return JSONResponse({"error": "no self-edit repo wired"}, status_code=409)
        return JSONResponse({"checkpoints": _ap.list_checkpoints(repo_dir)})

    @app.post("/api/selfedit/checkpoints/{checkpoint_id}/restore")
    async def selfedit_restore(checkpoint_id: str, request: Request) -> JSONResponse:
        """Revert to a prior state: restore a checkpoint's files + restart."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse({"error": "self-edit is disabled"}, status_code=403)
        st, repo_dir, _ap = _apply_ctx(request)
        if not repo_dir:
            return JSONResponse({"error": "no self-edit repo wired"}, status_code=409)
        from augmentum.selfedit.growth_db import get_growth_conn
        conn = await get_growth_conn(st)
        live = _ap.live_tree(st)
        body = {}
        with contextlib.suppress(Exception):
            body = await request.json()
        do_restart = bool(body.get("restart", True))
        res = await _ap.restore_checkpoint(repo_dir, live, checkpoint_id=checkpoint_id,
                                           conn=conn, user_id=user.id)
        if res.get("error"):
            return JSONResponse(res, status_code=404)
        restarting = do_restart and res.get("needs_restart")
        if restarting:
            _ap.schedule_restart(delay=1.5)
        res["restarting"] = bool(restarting)
        return JSONResponse(res)

    @app.post("/api/selfedit/restart")
    async def selfedit_restart(request: Request) -> JSONResponse:
        """Restart this container (the changes-already-written case, or a manual
        nudge). The app returns immediately, then restarts ~1.5s later."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse({"error": "self-edit is disabled"}, status_code=403)
        _st, _repo, _ap = _apply_ctx(request)
        _ap.schedule_restart(delay=1.5)
        return JSONResponse({"restarting": True})

    @app.post("/api/selfedit/reshape")
    async def selfedit_reshape(request: Request) -> JSONResponse:
        """Reshape a delivery surface from a request — the surface-agnostic
        self-edit path. Today the live surface is ``config`` (per-user Adaptation:
        layout/density/theme/shortcuts), whose oracle is a mechanical read-back, so
        a valid set earns ``verified`` and auto-applies instantly + reversibly, and
        lands in the same never-pruned archive as a code edit. Gated by
        ``selfedit_enabled``; writes are scoped to the requesting user."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse({"error": "self-edit is disabled"}, status_code=403)
        from augmentum.selfedit.growth_db import get_growth_conn
        conn = await get_growth_conn(request.app.state)
        if conn is None:
            return JSONResponse({"error": "growth store unavailable"}, status_code=503)

        body = {}
        with contextlib.suppress(Exception):
            body = await request.json()
        surface = str(body.get("surface", "config")).strip() or "config"
        change_class = str(body.get("change_class", "adaptation")).strip() or "adaptation"
        key = str(body.get("key", "")).strip()
        if not key:
            return JSONResponse({"error": "key is required"}, status_code=400)
        # config read-back oracle compares stored (str) to intended → coerce to str.
        value = body.get("value")
        payload = {"key": key, "value": "" if value is None else str(value)}

        from augmentum.selfedit.surfaces.base import ReshapeChange
        from augmentum.selfedit.surfaces.engine import ReshapeRequest, run_reshape_request
        from augmentum.selfedit.surfaces.live import build_store_recorder

        change = ReshapeChange(surface=surface, change_class=change_class,
                               payload=payload, intent=str(body.get("ask", "") or key),
                               actor=user.id)

        async def _passthrough(_req, _surfaces):
            return change

        on_start, on_finish = build_store_recorder(conn)
        try:
            result = await run_reshape_request(
                ReshapeRequest(ask=change.intent, actor=user.id, surface_hint=surface),
                classify=_passthrough, on_start=on_start, on_finish=on_finish,
            )
        except Exception as exc:  # noqa: BLE001 — surface, never silent 500
            log.warning("selfedit_reshape_failed", error=repr(exc))
            return JSONResponse({"error": f"reshape failed: {exc!r}"}, status_code=500)
        return JSONResponse(result.to_dict())

    @app.get("/api/selfedit/adaptables")
    async def selfedit_adaptables(request: Request) -> JSONResponse:
        """The discoverable catalog of per-user settings the Adapt lane can change,
        with each setting's current value. Gated."""
        user = request.scope.get("user")
        if user is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        if not await _selfedit_enabled(request):
            return JSONResponse({"error": "self-edit is disabled"}, status_code=403)
        from augmentum.proxy.config_routes import _UI_SETTINGS
        from augmentum.selfedit.adaptables import catalog_with_values
        store = getattr(request.app.state, "settings_store", None)
        # derive from the app's OWN per-user settings registry → auto-extends as
        # the app grows new settings (Law 0: derive, don't duplicate).
        items = await catalog_with_values(settings_store=store, user_id=user.id,
                                          ui_settings=_UI_SETTINGS)
        return JSONResponse({"adaptables": items})

    @app.get("/api/health/strain")
    async def strain_history(request: Request) -> JSONResponse:
        """Recent server-strain samples (strain_samples time series).

        Query params: ``minutes`` (default 60, max 1440), ``limit`` (default
        500, max 5000). Returns newest-first. Server-level health data —
        requires an authenticated user (any), no per-user scoping.
        """
        if request.scope.get("user") is None:
            return JSONResponse({"error": "auth required"}, status_code=401)
        backend = getattr(getattr(request.app.state, "state_manager", None), "backend", None)
        conn = getattr(backend, "conn", None) if backend else None
        if conn is None:
            return JSONResponse({"samples": [], "note": "no sqlite backend"})
        try:
            minutes = max(1, min(int(request.query_params.get("minutes", 60)), 1440))
        except (TypeError, ValueError):
            minutes = 60
        try:
            limit = max(1, min(int(request.query_params.get("limit", 500)), 5000))
        except (TypeError, ValueError):
            limit = 500
        try:
            cur = await conn.execute(
                "SELECT timestamp, event_loop_lag_ms, inflight_requests, slow_requests, "
                "active_clients, active_users, ws_presence, ws_notify, "
                "sessions_narrative, sessions_agentic, sessions_coder, "
                "engine_model, engine_secondary, db_write_ms, "
                "gpu_used_mb, gpu_free_mb, ram_used_mb, ram_free_mb, proc_rss_mb, context_json "
                "FROM strain_samples WHERE timestamp >= datetime('now', ?) "
                "ORDER BY timestamp DESC LIMIT ?",
                (f"-{minutes} minutes", limit),
            )
            cols = [c[0] for c in cur.description]
            rows = await cur.fetchall()
            await cur.close()
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        samples = [dict(zip(cols, r)) for r in rows]
        return JSONResponse({"minutes": minutes, "count": len(samples), "samples": samples})

    # Global exception handlers — structured error responses, no stack traces leaked
    @app.exception_handler(ClientDisconnect)
    async def client_disconnect_handler(request: Request, _exc: ClientDisconnect) -> JSONResponse:
        # Mobile clients drop connections constantly while bodies are still
        # streaming. Without this handler the exception falls through to
        # ``generic_error_handler`` which calls ``log.error(..., exc_info=True)``
        # and rich-renders the entire ``scope`` (headers, cookie, body) plus
        # every frame's locals — a 3-4 second sync workload on the event
        # loop. With dozens of mobile disconnects per session, the loop
        # repeatedly stalls, auth queues, and Caddy times out every other
        # request → "narrative disconnects + auth fails until restart".
        # Treat the disconnect as a normal client event: one structured
        # info line, no traceback, 499 status (response is dropped anyway
        # since the peer is already gone).
        log.info("client_disconnect_during_request", path=request.url.path,
                 method=request.method)
        return JSONResponse({"error": "client disconnected"}, status_code=499)

    @app.exception_handler(httpx.ConnectError)
    async def backend_connect_error(_request: Request, exc: httpx.ConnectError) -> JSONResponse:
        log.error("backend_connection_failed", error=str(exc))
        return JSONResponse(
            {"error": "Cannot connect to model backend. Is it running?"},
            status_code=502,
        )

    @app.exception_handler(httpx.ReadTimeout)
    async def backend_timeout(_request: Request, exc: httpx.ReadTimeout) -> JSONResponse:
        log.error("backend_timeout", error=str(exc))
        return JSONResponse(
            {"error": "Model backend timed out. Try a shorter prompt or increase timeout."},
            status_code=504,
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError) -> JSONResponse:
        log.warning("value_error", error=str(exc))
        return JSONResponse({"error": "Invalid request parameters"}, status_code=400)

    @app.exception_handler(Exception)
    async def generic_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_error", error=str(exc), exc_info=True)
        return JSONResponse(
            {"error": "Internal server error"},
            status_code=500,
        )

    # Normalize every error on a /v1/* path to the OpenAI ``{"error": {...}}``
    # envelope (audio TTS/STT raise plain HTTPException → {"detail": ...}
    # otherwise). Non-/v1 paths keep FastAPI's default error shape.
    from augmentum.proxy.openai_errors import register_openai_compat_error_handlers
    register_openai_compat_error_handlers(app)

    # Import and include routers
    from augmentum.proxy.agentic_routes import router as agentic_router
    from augmentum.proxy.anthropic_routes import router as anthropic_router
    from augmentum.proxy.artifact_routes import router as artifact_router
    from augmentum.proxy.audio_routes import router as audio_router
    from augmentum.proxy.canvas_routes import router as canvas_router
    from augmentum.proxy.auth_routes import router as auth_router
    from augmentum.proxy.gate_routes import router as gate_router
    from augmentum.proxy.lexicon_routes import router as lexicon_router
    from augmentum.proxy.avatar_routes import router as avatar_router
    from augmentum.proxy.balancer_routes import router as balancer_router
    from augmentum.proxy.browse_routes import router as browse_router
    from augmentum.proxy.bug_finder_routes import router as bug_finder_router
    from augmentum.proxy.build_routes import router as build_router
    from augmentum.proxy.animations_routes import router as animations_router
    from augmentum.proxy.cache_routes import router as cache_router
    from augmentum.proxy.dance_routes import router as dance_router
    from augmentum.proxy.playlist_routes import router as playlist_router
    from augmentum.proxy.capabilities_routes import router as capabilities_router
    from augmentum.proxy.cardsmith_routes import router as cardsmith_router
    from augmentum.proxy.cast_routes import (
        pair_router as cast_pair_router,
    )
    from augmentum.proxy.cast_routes import (
        render_output_router as cast_render_output_router,
    )
    from augmentum.proxy.cast_routes import (
        router as cast_router,
    )
    from augmentum.proxy.cast_routes import (
        stream_auth_router as cast_stream_auth_router,
    )
    from augmentum.proxy.cast_games_routes import router as cast_games_router
    from augmentum.proxy.cast_game_proxy_routes import (
        proxy_router as cast_game_proxy_router,
    )
    from augmentum.proxy.cast_game_proxy_routes import (
        start_router as cast_game_proxy_start_router,
    )
    from augmentum.proxy.character_routes import router as character_router
    from augmentum.proxy.chat_image_routes import router as chat_image_router
    from augmentum.proxy.chat_routes import router as chat_router
    from augmentum.proxy.community_routes import router as community_router
    from augmentum.proxy.cloud_image_routes import router as cloud_image_router
    from augmentum.proxy.coder_permission_routes import router as coder_permission_router
    from augmentum.proxy.coder_review_routes import router as coder_review_router
    from augmentum.proxy.coder_routes import router as coder_router
    from augmentum.proxy.coder_subagents_routes import router as coder_subagents_router
    from augmentum.proxy.config_routes import router as config_router
    from augmentum.proxy.external_coder_routes import router as external_coder_router
    from augmentum.proxy.pi_run_routes import router as pi_run_router
    from augmentum.proxy.settings_registry_routes import router as settings_registry_router
    from augmentum.proxy.content_isolation_routes import router as content_isolation_router
    from augmentum.proxy.controllers_routes import router as controllers_router
    from augmentum.proxy.device_routes import (
        cast_blob_router,
    )
    from augmentum.proxy.device_routes import (
        router as device_router,
    )
    from augmentum.proxy.discovery_routes import router as discovery_router
    from augmentum.proxy.sync_routes import router as sync_router
    from augmentum.proxy.document_routes import router as document_router
    from augmentum.proxy.dream_routes import router as dream_router
    from augmentum.proxy.executor_routes import router as executor_router
    from augmentum.proxy.connect_routes import router as connect_router
    from augmentum.proxy.fabric_routes import router as fabric_router
    from augmentum.proxy.portal_routes import router as portal_router
    from augmentum.proxy.notifications_routes import (
        router as notifications_router,
    )
    from augmentum.proxy.offers_routes import router as offers_router
    from augmentum.proxy.files_routes import router as files_router
    from augmentum.proxy.flow_routes import router as flow_router
    from augmentum.proxy.foundry_routes import router as foundry_router
    from augmentum.proxy.game_agent_routes import router as game_agent_router
    from augmentum.proxy.game_stream_routes import router as game_stream_router
    from augmentum.proxy.games_routes import router as games_router
    from augmentum.proxy.grove_routes import router as grove_router
    from augmentum.proxy.image_routes import router as image_router
    from augmentum.proxy.jobs_routes import router as jobs_router
    from augmentum.proxy.knowledge_routes import router as knowledge_router
    from augmentum.proxy.learning_routes import router as learning_router
    from augmentum.proxy.library_routes import router as library_router
    from augmentum.proxy.library_save_routes import router as library_save_router
    from augmentum.proxy.livetv_routes import router as livetv_router
    from augmentum.proxy.marketplace_routes import router as marketplace_router
    from augmentum.proxy.mcp_routes import router as mcp_router
    from augmentum.proxy.media_routes import router as media_router
    from augmentum.proxy.memory_routes import router as memory_router
    from augmentum.proxy.metrics_routes import router as metrics_router
    from augmentum.proxy.mobile_pair_routes import router as mobile_pair_router
    from augmentum.proxy.model_routes import engine_router, llamacpp_router
    from augmentum.proxy.model_routes import router as model_router
    from augmentum.proxy.comic_narration_routes import router as comic_narration_router
    from augmentum.proxy.narrative_routes import router as narrative_router
    from augmentum.proxy.note_intelligence_routes import router as note_intel_router
    from augmentum.proxy.notes_routes import router as notes_router
    from augmentum.proxy.observation_routes import router as observation_router
    from augmentum.proxy.ollama_routes import router as ollama_router
    from augmentum.proxy.openai_routes import router as openai_router
    from augmentum.proxy.persona_routes import router as persona_router
    from augmentum.proxy.power_routes import router as power_router
    from augmentum.proxy.presence_routes import router as presence_router
    from augmentum.proxy.provider_routes import router as provider_router
    from augmentum.proxy.intent_capture_routes import router as intent_capture_router
    from augmentum.proxy.reasoning_routes import router as reasoning_router
    from augmentum.proxy.resource_routes import router as resource_router
    from augmentum.proxy.session_routes import router as session_router
    from augmentum.proxy.studio_routes import router as studio_router
    from augmentum.proxy.system_events import router as system_events_router
    from augmentum.proxy.surface_routes import (
        public_router as surface_public_router,
    )
    from augmentum.proxy.surface_routes import (
        router as surface_router,
    )
    from augmentum.proxy.titles_bios_routes import router as titles_bios_router
    from augmentum.proxy.titles_marketplace_routes import router as titles_marketplace_router
    from augmentum.proxy.titles_routes import router as titles_router
    from augmentum.proxy.titles_saves_routes import router as titles_saves_router
    from augmentum.proxy.ui_routes import router as ui_router
    from augmentum.proxy.voice_enrollment_routes import router as voice_enrollment_router
    from augmentum.proxy.voice_routes import router as voice_router
    from augmentum.proxy.wake_word_routes import router as wake_word_router
    from augmentum.proxy.world_routes import router as world_router
    from augmentum.proxy.xr_routes import router as xr_router
    from augmentum.proxy.youtube_routes import router as youtube_router

    app.include_router(portal_router)
    app.include_router(character_router)
    app.include_router(cardsmith_router)
    app.include_router(comic_narration_router)
    from augmentum.proxy.companion_routes import router as companion_router
    from augmentum.proxy.companion_growth_routes import (
        router as companion_growth_router,
    )
    from augmentum.proxy.vision_routes import router as vision_api_router
    from augmentum.proxy.architect_routes import router as architect_router
    from augmentum.proxy.capability_routes import router as capability_router
    from augmentum.proxy.perception_routes import router as perception_router
    from augmentum.proxy.calendar_routes import router as calendar_router
    app.include_router(companion_router)
    app.include_router(companion_growth_router)
    app.include_router(calendar_router)
    app.include_router(vision_api_router)
    app.include_router(architect_router)
    app.include_router(perception_router)
    app.include_router(capability_router)
    app.include_router(intent_capture_router)
    app.include_router(chat_router)
    app.include_router(community_router)
    app.include_router(narrative_router)
    app.include_router(agentic_router)
    app.include_router(artifact_router)
    app.include_router(build_router)
    app.include_router(canvas_router)
    app.include_router(persona_router)
    app.include_router(ollama_router)
    app.include_router(openai_router)
    app.include_router(anthropic_router)
    from augmentum.proxy.harness_routes import router as harness_router
    app.include_router(harness_router)
    app.include_router(ui_router)
    app.include_router(world_router)
    app.include_router(xr_router)
    app.include_router(cache_router)
    app.include_router(config_router)
    app.include_router(settings_registry_router)
    app.include_router(model_router)
    app.include_router(llamacpp_router)
    app.include_router(engine_router)
    app.include_router(provider_router)
    app.include_router(system_events_router)
    app.include_router(balancer_router)
    app.include_router(session_router)
    app.include_router(memory_router)
    app.include_router(image_router)
    app.include_router(cloud_image_router)
    app.include_router(mcp_router)
    app.include_router(reasoning_router)
    app.include_router(audio_router)
    app.include_router(lexicon_router)
    app.include_router(chat_image_router)
    app.include_router(document_router)
    app.include_router(voice_router)
    from augmentum.proxy.voice_turn_routes import router as voice_turn_router
    app.include_router(voice_turn_router)
    app.include_router(voice_enrollment_router)
    app.include_router(flow_router)
    app.include_router(capabilities_router)
    app.include_router(power_router)
    # notification_router removed — SSE endpoints not consumed by frontend.
    # File kept at proxy/notification_routes.py for future use.
    app.include_router(resource_router)
    app.include_router(executor_router)
    app.include_router(browse_router)
    app.include_router(notes_router)
    app.include_router(observation_router)
    from augmentum.proxy.atp_routes import router as atp_router
    app.include_router(atp_router)
    app.include_router(presence_router)
    app.include_router(studio_router)
    app.include_router(metrics_router)
    app.include_router(dream_router)
    app.include_router(avatar_router)
    app.include_router(dance_router)
    app.include_router(playlist_router)
    app.include_router(animations_router)
    app.include_router(coder_router)
    app.include_router(content_isolation_router)
    app.include_router(coder_permission_router)
    app.include_router(coder_review_router)
    app.include_router(coder_subagents_router)
    app.include_router(external_coder_router)
    from augmentum.proxy.coding_routes import router as coding_router
    app.include_router(coding_router)
    app.include_router(pi_run_router)
    app.include_router(bug_finder_router)
    app.include_router(knowledge_router)
    app.include_router(library_save_router)
    app.include_router(library_router)
    app.include_router(learning_router)
    app.include_router(grove_router)
    app.include_router(youtube_router)
    app.include_router(discovery_router)
    app.include_router(sync_router)
    app.include_router(marketplace_router)
    app.include_router(note_intel_router)
    app.include_router(auth_router)
    app.include_router(gate_router)
    app.include_router(mobile_pair_router)
    app.include_router(fabric_router)
    app.include_router(connect_router)
    app.include_router(notifications_router)
    app.include_router(offers_router)
    app.include_router(wake_word_router)
    app.include_router(files_router)
    app.include_router(media_router)
    app.include_router(livetv_router)
    app.include_router(games_router)
    app.include_router(surface_router)
    app.include_router(surface_public_router)
    app.include_router(device_router)
    app.include_router(cast_blob_router)
    app.include_router(cast_router)
    app.include_router(cast_render_output_router)
    app.include_router(cast_pair_router)
    app.include_router(cast_stream_auth_router)
    app.include_router(cast_games_router)
    app.include_router(cast_game_proxy_start_router)
    app.include_router(cast_game_proxy_router)
    app.include_router(jobs_router)
    app.include_router(game_stream_router)
    app.include_router(game_agent_router)
    app.include_router(foundry_router)
    # Game-agent LLM bridge: lazy SlowPathLLM that delegates to whatever
    # backend ProviderRegistry resolves as the default at call time.
    # Wired as a getter so the registry can be created later in a
    # startup event without breaking app construction; also means
    # provider re-registration at runtime is picked up by the next
    # session without a server restart.
    from augmentum.game_agent.llm_bridge import (
        make_game_agent_chat_llm,
        make_game_agent_llm,
    )
    from augmentum.game_agent.voice_bridge import VoiceBridge
    app.state.game_agent_llm = make_game_agent_llm(
        lambda: app.state.provider_registry,
    )
    # Fast-turn ("call mode") sibling: rolling multi-message window for
    # sub-second micro-plans between FULL planning turns. Same lazy
    # registry, same model resolution.
    app.state.game_agent_chat_llm = make_game_agent_chat_llm(
        lambda: app.state.provider_registry,
    )
    # Companion voice bridge: text -> TTS audio bytes via whichever
    # provider Augmentum has configured. Lazy state-manager lookup so
    # this survives create_app running before startup events. Returns
    # None when state_manager isn't wired yet; VoiceBridge handles
    # None by falling silent rather than crashing.
    def _voice_conn():
        sm = getattr(app.state, "state_manager", None)
        backend = getattr(sm, "backend", None) if sm else None
        return getattr(backend, "conn", None) if backend else None
    app.state.game_agent_voice = VoiceBridge(_voice_conn)
    app.include_router(titles_router)
    app.include_router(titles_marketplace_router)
    app.include_router(titles_saves_router)
    app.include_router(titles_bios_router)
    app.include_router(controllers_router)

    # Discover surface — unified browse + install across provider
    # services, titles, and community content. Spec:
    # docs/superpowers/specs/2026-06-10-discover-surface-design.md
    from augmentum.proxy.discover_routes import router as discover_router
    app.include_router(discover_router)

    from augmentum.proxy.search_routes import router as search_router
    app.include_router(search_router)

    # Serve /.well-known/security.txt at the RFC 9116 canonical path so
    # security researchers can find a contact without going through /ui.
    # Public path (auth-exempt) — see augmentum/auth/middleware.py.
    @app.get("/.well-known/security.txt", include_in_schema=False)
    async def _security_txt():
        from fastapi.responses import PlainTextResponse
        repo_root = Path(__file__).resolve().parent.parent.parent
        security_txt = repo_root / ".well-known" / "security.txt"
        if security_txt.exists():
            return PlainTextResponse(security_txt.read_text(encoding="utf-8"))
        return PlainTextResponse("", status_code=404)

    # Serve the bundled web UI as static files (must be last — catch-all mount)
    from fastapi.staticfiles import StaticFiles

    repo_root = Path(__file__).resolve().parent.parent.parent

    # Scoped vendored-asset mount: the SillyTavern BVH motion pack lives in
    # `poses/external/sillytavern-pack/` (~99MB across 150 .bvh files plus
    # the auto-generated bvh-manifest.json). It's third-party public-domain
    # content used by the scene-test mockup's BVH library dropdown.
    # We mount JUST that subdirectory so we don't expose the rest of
    # `poses/` (which contains user-private body atlases, affordance bones,
    # synthesized poses, etc).
    bvh_dir = repo_root / "poses" / "external" / "sillytavern-pack"
    if bvh_dir.exists():
        app.mount("/bvh-library", StaticFiles(directory=str(bvh_dir)), name="bvh-library")

    # Scoped /poses mount: serves the runtime-shippable subset of poses/
    # via an explicit filename allowlist. Lets the live avatar pipeline
    # fetch atlases, affordance vocab, body-geometry, and VRMA motion
    # clips (all of which the JS defaults at `/poses/<name>`), while
    # keeping user pose captures, intermediate mining outputs, and the
    # extraction Python scripts un-exposed.
    #
    # Allowlist deliberately enumerated rather than glob-only so adding
    # a new runtime asset class is a code change, not an accidental data
    # leak from someone dropping a file in the dir.
    poses_dir = repo_root / "poses"
    if poses_dir.exists():
        import fnmatch

        from starlette.exceptions import HTTPException

        # Flat-level allowlist (top of poses/).
        _POSES_ALLOWED = (
            "body-atlas-*.json",          # voxel atlases per VRM
            "body-geometry-*.json",       # legacy capsule geometry
            "affordances.json",           # affordance vocab variants
            "affordances-bones.json",
            "affordances-v3-vrm.json",
            "affordance-region-tags.json",  # mined affordance-region resolutions
            "landmark-cross-vrm-stats.json",  # mined landmark reliability stats
            "primitives-manifest.json",   # auto-generated index of poses/primitives/
            "*.vrma",                     # VRMA motion clips
        )
        # Nested allowlist: paths of form <subdir>/<file> matching here.
        _POSES_ALLOWED_NESTED = (
            ("primitives", "*.json"),     # pose primitive library
        )

        class _PosesStaticFiles(StaticFiles):
            async def get_response(self, path, scope):
                # Reject path traversal + Windows backslashes outright.
                if "\\" in path or ".." in path.split("/"):
                    raise HTTPException(status_code=404)
                parts = path.split("/")
                if len(parts) == 1:
                    # Top-level file
                    if not any(fnmatch.fnmatch(path, pat) for pat in _POSES_ALLOWED):
                        raise HTTPException(status_code=404)
                elif len(parts) == 2:
                    # One-level-nested file
                    subdir, fname = parts
                    ok = any(
                        subdir == sub and fnmatch.fnmatch(fname, pat)
                        for sub, pat in _POSES_ALLOWED_NESTED
                    )
                    if not ok:
                        raise HTTPException(status_code=404)
                else:
                    raise HTTPException(status_code=404)

                # Precompressed delivery: the body-atlas JSONs are 50-60 MB raw
                # and ship a baked .json.gz sibling (~4x smaller, gzip-9). When
                # the client accepts gzip and the sibling exists, serve it with
                # Content-Encoding: gzip — the browser transparently decompresses
                # so the consumer sees byte-identical JSON (zero visual change)
                # at a quarter of the wire transfer. Delegating to super() on the
                # .gz path keeps StaticFiles' ETag / Last-Modified / 304 /
                # Content-Length handling intact; we only add the encoding hints.
                if path.endswith(".json"):
                    from starlette.datastructures import Headers
                    accept_enc = Headers(scope=scope).get("accept-encoding", "")
                    if "gzip" in accept_enc.lower():
                        gz_rel = path + ".gz"
                        if (Path(self.directory) / gz_rel).is_file():
                            resp = await super().get_response(gz_rel, scope)
                            if resp.status_code == 200:
                                resp.headers["Content-Encoding"] = "gzip"
                                resp.headers["Content-Type"] = "application/json"
                                resp.headers["Vary"] = "Accept-Encoding"
                            return resp
                return await super().get_response(path, scope)

        app.mount("/poses", _PosesStaticFiles(directory=str(poses_dir)), name="poses")

    ui_dir = repo_root / "ui"
    if ui_dir.exists():
        # MIME registrations for files Python's stdlib mimetypes module
        # doesn't know about. EmulatorJS' libretro core ``.data`` files
        # are 7z-compressed binary blobs; without the registration
        # Starlette's StaticFiles serves them as ``text/plain;
        # charset=utf-8``, which is wrong (and confuses some
        # caching/CDN layers even when fetch().arrayBuffer() ignores
        # the type). Same for raw ``.wasm`` bytes.
        import mimetypes
        mimetypes.add_type("application/octet-stream", ".data")
        mimetypes.add_type("application/wasm", ".wasm")
        # Self-hosted reading face (ui/fonts/*.woff2). Some stdlib builds
        # don't map .woff2, leaving StaticFiles to serve it as text/plain,
        # which trips strict caches; pin the correct type.
        mimetypes.add_type("font/woff2", ".woff2")

        # Root-level shim for the push service worker. The SPA is
        # mounted at /ui, so the SW file is served from /ui/notification-
        # sw.js — but service workers are canonically registered from
        # the origin root. Some browsers also tighten scope rules when
        # the SW is served from a sub-path. Serving the same bytes from
        # /notification-sw.js keeps push-subscribe.js's registration path
        # right + decouples push from where the SPA happens to be
        # mounted.
        from fastapi.responses import FileResponse as _FileResponse

        _sw_path = ui_dir / "notification-sw.js"

        @app.get("/notification-sw.js", include_in_schema=False)
        async def _notification_sw():
            return _FileResponse(
                str(_sw_path),
                media_type="application/javascript",
                # Short-cache: SW updates need to propagate quickly so
                # we can fix bugs without telling users to clear caches.
                # Browsers also ignore long-cache on the SW file itself
                # but this is the explicit contract.
                headers={"Cache-Control": "max-age=0, no-cache"},
            )

        # Static-asset caching, scoped to the Android WebView ONLY.
        #
        # The native Android shell loads this exact SPA inside a WebView over
        # the network, so a cold launch re-fetches ~120 <script>/<link> assets.
        # Starlette's StaticFiles sends ETag/Last-Modified but NO freshness
        # lifetime, so the WebView issues a conditional GET (revalidation) per
        # asset on every load — a round-trip storm that's invisible on a
        # desktop localhost but very visible over phone Wi-Fi/Tailscale RTT.
        #
        # We collapse that storm by handing the WebView a real cache lifetime:
        #   - static assets → max-age + stale-while-revalidate (serve instantly
        #     from cache, revalidate in the background → stale by at most one
        #     load, which self-heals; never makes the server look unreachable)
        #   - HTML documents → no-cache (always revalidate the entry point so a
        #     deploy propagates immediately; the ETag keeps it a cheap 304)
        #
        # Crucially this is gated on the AugmentumAndroid User-Agent (set in
        # AugmentumWebViewScreen.kt). Desktop browsers — where active local
        # development happens and localhost RTT is ~0 — keep today's
        # always-fresh behavior, so we don't reintroduce the stale-bundle pain
        # that got the service worker disabled (see ui/index.html).
        _UI_STATIC_CACHE = "public, max-age=600, stale-while-revalidate=86400"

        class _UiStaticFiles(StaticFiles):
            async def get_response(self, path, scope):
                resp = await super().get_response(path, scope)
                try:
                    from starlette.datastructures import Headers

                    if resp.status_code not in (200, 304):
                        return resp
                    # Treat extensionless paths (the html=True index redirect
                    # target) as HTML to be safe.
                    lower = path.lower()
                    last = lower.rsplit("/", 1)[-1]
                    is_html = lower.endswith((".html", ".htm")) or "." not in last
                    is_code = is_html or lower.endswith((".js", ".mjs", ".css"))

                    ua = Headers(scope=scope).get("user-agent", "")
                    if "AugmentumAndroid" in ua:
                        # Native WebView over phone RTT. CODE assets (html/js/
                        # css) must revalidate every load — otherwise a shipped
                        # fix sits behind the SWR lifetime (served stale for up
                        # to max-age, then stale-once-more under SWR), which is
                        # exactly why fixes "never reached the app". `no-cache`
                        # keeps them cached but forces a conditional GET: a cheap
                        # 304 when unchanged, fresh the instant we redeploy.
                        # Non-code static (fonts/wasm/images/data) keeps the
                        # cache lifetime — those are big + rarely change, so the
                        # cold-launch latency win is preserved where it matters.
                        resp.headers["Cache-Control"] = "no-cache" if is_code else _UI_STATIC_CACHE
                    elif is_code:
                        # Desktop + mobile-web browsers. StaticFiles sends
                        # ETag/Last-Modified but NO freshness lifetime — which
                        # does NOT mean "always revalidate". With no
                        # Cache-Control the browser applies HEURISTIC caching
                        # (~10% of the time since Last-Modified) and serves
                        # STALE .js/.css/.html WITHOUT a conditional GET. That is
                        # why shipped frontend fixes appear not to land until a
                        # manual hard-refresh — the browser never asks the
                        # server. `no-cache` forces a conditional GET on every
                        # load: the ETag makes it a cheap 304 when unchanged and
                        # a fresh 200 the instant we redeploy. (On localhost RTT
                        # is ~0; over Tailscale it's one cheap 304 per code
                        # asset — correctness beats a stale bundle.) Non-code
                        # static (fonts/wasm/images) keeps default heuristic
                        # caching; those rarely change and a stale font is
                        # harmless.
                        resp.headers["Cache-Control"] = "no-cache"
                except Exception:  # pragma: no cover - header shaping is best-effort
                    log.warning("ui_cache_header_failed", path=path, exc_info=True)
                return resp

        # Dedupe the bundled VRMA clips. ~20 .vrma files are shared between
        # the avatar animation atlas (served here, /ui/lib/animations/<name>,
        # via avatar-vrma-library.js's dynamic loader) and the enumerated pose
        # library (poses/, served at /poses/<name>). They used to be committed
        # byte-for-byte in BOTH dirs (~25 MB doubled). Now poses/ is the single
        # on-disk home: this mount serves ui/lib/animations/'s own clips
        # directly and transparently falls back to poses/ for any .vrma missing
        # here — so the frontend URL is unchanged, but the shared clips live
        # once. Mounted BEFORE /ui so this more-specific path wins.
        _anim_dir = ui_dir / "lib" / "animations"
        if _anim_dir.exists():
            from starlette.exceptions import HTTPException as _StarletteHTTPException

            class _AnimationsStaticFiles(_UiStaticFiles):
                async def get_response(self, path, scope):
                    try:
                        return await super().get_response(path, scope)
                    except _StarletteHTTPException as exc:
                        # Fall back to the canonical poses/ copy for shared
                        # .vrma clips that no longer live here.
                        if exc.status_code == 404 and path.lower().endswith(".vrma"):
                            cand = poses_dir / Path(path).name
                            if cand.is_file():
                                from starlette.responses import FileResponse

                                return FileResponse(str(cand))
                        raise

            app.mount(
                "/ui/lib/animations",
                _AnimationsStaticFiles(directory=str(_anim_dir), html=False),
                name="ui-animations",
            )

        app.mount("/ui", _UiStaticFiles(directory=str(ui_dir), html=True), name="ui")

    return app
