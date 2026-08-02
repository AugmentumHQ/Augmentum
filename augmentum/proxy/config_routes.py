"""Configuration management API routes."""

from __future__ import annotations

import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.auth.guards import require_admin
from augmentum.config import settings
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/config", tags=["config"])


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""

# Setting keys whose value space is tri-state (None / True / False)
# rather than bool. PUT-side: the handler interprets ``None`` (or
# ``"auto"``) as "delete the persisted override and revert the runtime
# field to None so the codebase recommendation is followed". GET-side:
# returns the persisted value AS-IS (None / True / False) plus a
# companion ``<key>_resolved`` field with the bool the runtime
# actually behaves as. See
# docs/superpowers/specs/2026-05-05-multi-slot-kv-design.md.
_TRI_STATE_BOOL_SETTINGS: dict[str, callable] = {
    # Resolver: takes the persisted Optional[bool] and returns the
    # bool the system actually behaves as. Imported lazily to avoid a
    # circular dep with the engine layer.
    "engine_multislot_enabled": (
        lambda v: __import__(
            "augmentum.proxy.status_bus", fromlist=["MULTISLOT_DEFAULT_ENABLED"]
        ).MULTISLOT_DEFAULT_ENABLED if v is None else bool(v)
    ),
}

# Tool settings that users can adjust at runtime.
# Maps setting key → (type cast, min, max).
_TOOL_SETTINGS: dict[str, tuple[type, int | float, int | float]] = {
    "strain_monitor_enabled": (bool, 0, 1),
    "selfedit_enabled": (bool, 0, 1),  # self-edit master switch (default OFF)
    "selfedit_max_iters": (int, 1, 512),  # native edit-loop iteration cap (default 64)
    "selfedit_self_heal_attempts": (int, 0, 5),  # repair passes on a fixable break (0=off)
    "selfedit_ingest_coder_enabled": (bool, 0, 1),  # ingest-all-work: coder turns → archive (default OFF)
    "intent_capture_enabled": (bool, 0, 1),
    "training_capture_enabled": (bool, 0, 1),
    "training_capture_min_content": (int, 0, 10000),
    "uarf_auto_search": (bool, 0, 1),
    "uarf_auto_search_queries": (int, 1, 10),
    "uarf_auto_search_results_per_query": (int, 1, 10),
    "uarf_auto_search_max_context_chars": (int, 1000, 128000),
    "uarf_auto_verify": (bool, 0, 1),
    "uarf_proactive_search": (bool, 0, 1),
    "uarf_proactive_math": (bool, 0, 1),
    "uarf_proactive_code": (bool, 0, 1),
    "uarf_max_tool_calls_per_phase": (int, 1, 10),
    "uarf_search_retry_max": (int, 0, 5),
    "uarf_search_retry_min_results": (int, 0, 10),
    "uarf_heuristic_assess": (bool, 0, 1),
    # narrative_memory_enabled is per-session only (SessionMemorySettings)
    "narrative_llm_extraction": (bool, 0, 1),
    "narrative_extraction_interval": (int, 1, 20),
    "narrative_memory_interval": (int, 5, 50),
    "narrative_memory_max_tokens": (int, 0, 2000),
    "narrative_memory_max_words": (int, 0, 20000),
    "narrative_memory_ledger_ceiling": (int, 0, 500),
    "narrative_memory_compaction_enabled": (bool, 0, 1),
    "narrative_smart_retrieval": (bool, 0, 1),
    "narrative_smart_retrieval_count": (int, 1, 20),
    "narrative_request_log_limit": (int, 5, 50),
    "narrative_memory_state_enabled": (bool, 0, 1),
    "narrative_memory_ledger_enabled": (bool, 0, 1),
    "narrative_memory_continuous_archive": (bool, 0, 1),
    "narrative_archive_min_messages": (int, 0, 500),
    "narrative_context_limit": (int, 0, 500000),
    "narrative_scene_context_rounds": (int, 1, 10),
    "narrative_auto_background": (bool, 0, 1),
    "narrative_auto_background_interval": (int, 1, 20),
    "narrative_translate_auto_save": (bool, 0, 1),
    "game_portal_enabled": (bool, 0, 1),
    # AXF / titles
    "titles_enabled": (bool, 0, 1),
    "titles_storage_max_mb": (int, 0, 100000),
    "marketplace_enabled": (bool, 0, 1),
    # Save-to-Library caps. Bounds are wide so deployments can dial
    # them as needed; the defaults in config.py are the user-friendly
    # values (50 MB per publication, 1 GB per user).
    "library_publication_max_bytes": (int, 1048576, 1073741824),       # 1MB-1GB
    "library_publication_user_budget_bytes": (int, 10485760, 107374182400),  # 10MB-100GB
    "emulator_browser_enabled": (bool, 0, 1),
    "emulator_rom_max_mb": (int, 0, 4096),
    "emulator_save_max_per_slot_mb": (int, 1, 1024),
    "emulator_save_slots_per_rom": (int, 1, 32),
    # Controller framework
    "controller_remap_enabled": (bool, 0, 1),
    "controller_haptic_enabled": (bool, 0, 1),
    "controller_deadzone": (float, 0.0, 0.5),
    # AGSP -- Game Streaming Platform
    "game_stream_enabled": (bool, 0, 1),
    "game_stream_max_concurrent": (int, 1, 8),
    "game_stream_default_bitrate_mbps": (int, 1, 25),
    "game_stream_idle_timeout_seconds": (int, 60, 7200),
    "game_stream_prefer_hw_encoder": (bool, 0, 1),
    "game_stream_mouse_sensitivity": (float, 0.01, 2.0),
    # Search pipeline settings
    "search_expansion_enabled": (bool, 0, 1),
    "search_expansion_max_variants": (int, 1, 10),
    "search_expansion_max_total": (int, 5, 50),
    "search_credibility_enabled": (bool, 0, 1),
    "search_direct_fetch_enabled": (bool, 0, 1),
    "search_direct_fetch_max_chars": (int, 1000, 128000),
    "search_relevance_filter_enabled": (bool, 0, 1),
    "search_relevance_min_score": (float, 0.0, 1.0),
    "search_proxy_rotation_enabled": (bool, 0, 1),
    "search_proxy_healthcheck_interval_minutes": (int, 1, 1440),
    "search_proxy_fallback_direct_enabled": (bool, 0, 1),
    "web_search_topic_hints_enabled": (bool, 0, 1),
    "uarf_conversation_max_chars": (int, 500, 32000),
    # Multi-model fan-out
    "multi_model_enabled": (bool, 0, 1),
    # Chain settings
    "passthrough_chain_enabled": (bool, 0, 1),
    "passthrough_chain_max_steps": (int, 1, 20),
    "passthrough_chain_timeout": (float, 10.0, 600.0),
    "passthrough_chain_max_parallel": (int, 1, 10),
    "passthrough_chain_max_flows": (int, 1, 200),
    "passthrough_chain_max_retries": (int, 0, 10),
    "passthrough_chain_attention_anchor": (bool, 0, 1),
    "passthrough_chain_error_as_observation": (bool, 0, 1),
    "passthrough_chain_plan_mutation": (bool, 0, 1),
    "passthrough_chain_synthesis_timeout": (float, 10.0, 600.0),
    # Image quality & speed
    "image_max_inflight_per_user": (int, 0, 100),  # per-user GPU-queue fairness cap; 0 = off
    "image_freeu_enabled": (bool, 0, 1),
    "image_tome_enabled": (bool, 0, 1),
    "image_tome_ratio": (float, 0.1, 0.9),
    "image_cfg_rescale": (float, 0.0, 1.0),
    "image_hires_fix": (bool, 0, 1),
    "image_hires_scale": (float, 1.0, 4.0),
    "image_hires_denoise": (float, 0.0, 1.0),
    "image_ip_adapter_enabled": (bool, 0, 1),
    "image_ip_adapter_scale": (float, 0.0, 1.0),
    # Image custom-import trust boundary
    "image_allow_pickle_formats": (bool, 0, 1),
    "image_upload_max_size_gb": (int, 1, 500),
    # Voice detection
    "voice_silence_threshold_ms": (int, 400, 3000),
    "voice_max_audio_seconds": (int, 5, 120),
    "voice_speaker_verify": (bool, 0, 1),
    "voice_smart_turn": (bool, 0, 1),
    "voice_smart_turn_threshold": (float, 0.1, 0.95),
    "voice_smart_turn_max_wait_s": (float, 0.5, 30.0),
    "voice_smart_turn_max_deferrals": (int, 0, 20),
    "voice_smart_turn_min_veto_confidence": (float, 0.0, 0.5),
    "voice_bargein_min_speech_ms": (int, 0, 2000),
    "voice_fast_endpoint_ms": (int, 0, 1500),
    "voice_ack_clips_enabled": (bool, 0, 1),
    "voice_preprocess_bypass": (bool, 0, 1),
    "voice_denoise_enabled": (bool, 0, 1),
    "voice_highpass_hz": (int, 0, 1000),
    "voice_audio_agc": (bool, 0, 1),
    "voice_audio_ns": (bool, 0, 1),
    "voice_audio_agc_target_dbfs": (int, -40, 0),
    "voice_audio_ns_level": (int, 0, 4),
    # Vision provider (SmolVLM sibling + capability router)
    "vision_provider_enabled": (bool, 0, 1),  # allow CPU vision fallback (retired SmolVLM sibling)
    "vision_provider_backend_port": (int, 1024, 65535),
    # Secondary local engine ("Slot B") — second resident user-chosen model
    "engine_secondary_enabled": (bool, 0, 1),
    "engine_secondary_backend_port": (int, 1024, 65535),
    # Managed classifier slot ("Slot C") — swappable classifier/utility/vision model
    "classifier_slot_enabled": (bool, 0, 1),
    "classifier_slot_backend_port": (int, 1024, 65535),
    "classifier_slot_gpu_layers": (int, 0, 999),
    "classifier_slot_ctx_size": (int, 512, 131072),
    # Comic narration cache retention (finished chapters kept per user; 0 = forever)
    "comic_narration_cache_max": (int, 0, 500),
    "voice_speaker_threshold": (float, 0.2, 0.9),
    "voice_speaker_verify_seconds": (float, 1.0, 10.0),
    "voice_lipsync_universal": (bool, 0, 1),
    "voice_xr_proxemics_enabled": (bool, 0, 1),
    # App-level scheduling dispatcher (augmentum/scheduling/service.py) —
    # fires standing tasks for every user, companion on or off.
    "scheduling_enabled": (bool, 0, 1),
    "companion_standing_tasks_enabled": (bool, 0, 1),
    # CompanionRuntime flags (Sprint 1 substrate + per-sprint sub-flags)
    "companion_runtime_enabled": (bool, 0, 1),
    "companion_assist_enabled": (bool, 0, 1),
    "companion_live_vision_enabled": (bool, 0, 1),
    "companion_voice_decision_hud": (bool, 0, 1),
    "companion_dispatch_enabled": (bool, 0, 1),
    "companion_tick_enabled": (bool, 0, 1),
    "companion_dreams_enabled": (bool, 0, 1),
    "companion_perception_acquire_notifications": (bool, 0, 1),  # L0 notification ingest+fuse (default OFF; needs Android grant)
    "companion_drift_audit_enabled": (bool, 0, 1),
    "companion_journal_enabled": (bool, 0, 1),
    "companion_creations_enabled": (bool, 0, 1),
    "companion_cultural_intake_enabled": (bool, 0, 1),
    "companion_entity_recs_enabled": (bool, 0, 1),
    "companion_household_enabled": (bool, 0, 1),
    "companion_peer_agents_enabled": (bool, 0, 1),
    "companion_xr_orchestrator": (bool, 0, 1),
    "companion_subagent_registry_active": (bool, 0, 1),
    "companion_primitive_registry_active": (bool, 0, 1),
    "companion_skill_archive_enabled": (bool, 0, 1),
    # Synapse Layer §1 — chat→interior salience scoring
    "companion_salience_enabled": (bool, 0, 1),
    "companion_salience_journal_threshold": (float, 0.0, 1.0),
    "companion_salience_llm_enabled": (bool, 0, 1),
    # Synapse Layer §3 — voice→interior journaling
    "companion_voice_journal_enabled": (bool, 0, 1),
    # Promise/Deliver — second-companion-pass strict-tier flag.
    # Tier string itself ("primary" / "utility") is in _STRING_SETTINGS.
    "companion_promise_deliver_strict_tier": (bool, 0, 1),
    # Synapse Layer §2 — user-observed affect decay
    "companion_user_affect_half_life_s": (float, 60.0, 7200.0),
    # Synapse Layer §4 — slow consolidation
    "companion_consolidation_enabled": (bool, 0, 1),
    "companion_consolidation_interval_days": (int, 1, 365),
    "companion_consolidation_drift_ceiling": (float, 0.0, 0.20),
    "companion_consolidation_min_evidence": (int, 1, 100),
    # Chat-mode routing through the companion dispatcher
    "companion_dispatch_routes_chat": (bool, 0, 1),
    "companion_dispatch_chat_min_utility": (float, 0.0, 1.0),
    # Architect dispatch — companion-as-orchestrator with inferred defaults
    "architect_dispatch_enabled": (bool, 0, 1),
    # Confidence-tier dispatch — LLM router replaces template-as-gate
    "architect_router_enabled": (bool, 0, 1),
    "architect_router_timeout_ms": (int, 500, 10000),
    "companion_address_threshold": (float, 0.5, 1.0),
    "companion_memory_min_score": (float, 0.0, 1.0),
    "companion_profile_tone_only": (bool, 0, 1),
    "memory_earned_permanence": (bool, 0, 1),
    "memory_corroboration_promote_access": (int, 1, 10),
    "memory_reflection_force_core": (bool, 0, 1),
    "companion_address_llm_enabled": (bool, 0, 1),
    "companion_address_llm_timeout_ms": (int, 100, 5000),
    "companion_always_listening_warmup_ms": (int, 0, 5000),
    "companion_always_listening_vad_threshold": (float, 0.0, 1.0),
    "companion_always_listening_prefix_padding_ms": (int, 0, 3000),
    "companion_address_media_boost": (float, 0.0, 0.5),
    "companion_followup_window_s": (float, 0.0, 60.0),
    "companion_open_thread_window_s": (float, 0.0, 180.0),
    "companion_results_ring_enabled": (bool, 0, 1),
    "companion_csm_cross_speaker": (bool, 0, 1),
    "companion_results_ring_turns": (int, 1, 10),
    "companion_alert_watch_enabled": (bool, 0, 1),
    # Scheduled requests & watches (spec 2026-06-11)
    "companion_watch_judge_enabled": (bool, 0, 1),
    "companion_watch_judge_timeout_s": (float, 1.0, 60.0),
    "companion_watch_probe_timeout_s": (float, 1.0, 60.0),
    "companion_metric_quarantine_pct": (float, 0.0, 500.0),
    "companion_metric_confirm_readings": (int, 1, 10),
    "companion_prompt_fire_max_tool_calls": (int, 1, 20),
    "companion_prompt_fire_max_seconds": (float, 10.0, 600.0),
    "companion_research_enabled": (bool, 0, 1),
    "companion_research_max_queries": (int, 1, 8),
    "companion_research_max_seconds": (float, 10.0, 300.0),
    "companion_research_fetch_top": (int, 0, 4),
    "companion_image_prompt_expansion_enabled": (bool, 0, 1),
    "companion_image_expansion_timeout_ms": (int, 500, 20000),
    # Becca-direct chat path (accumulation thesis Step 1)
    "companion_becca_direct_enabled": (bool, 0, 1),
    # Skill graph (accumulation thesis Step 3)
    "companion_skills_enabled": (bool, 0, 1),
    "companion_skill_relevance_threshold": (float, 0.0, 1.0),
    "companion_skill_min_confidence_for_inject": (float, 0.0, 1.0),
    "companion_skill_inject_top_k": (int, 1, 20),
    # Lesson registry (mig 270) — learn-from-correction inverse
    "companion_lessons_enabled": (bool, 0, 1),
    "companion_lessons_capture_enabled": (bool, 0, 1),
    "companion_lessons_relevance_threshold": (float, 0.0, 1.0),
    "companion_lessons_min_strength_for_inject": (float, 0.0, 1.0),
    "companion_lessons_inject_top_k": (int, 1, 20),
    "companion_initiative_threshold": (float, 0.0, 1.0),
    "companion_initiative_enabled": (bool, 0, 1),
    "companion_initiative_min_interval_s": (float, 5.0, 3600.0),
    "companion_topical_aggregator_enabled": (bool, 0, 1),
    "companion_topical_min_events": (int, 2, 10),
    "companion_topical_window_hours": (float, 0.5, 24.0),
    "companion_wondering_daily_cap": (int, 0, 10),
    "companion_synthesize_daily_cap": (int, 0, 20),
    "companion_synthesize_max_tokens": (int, 64, 1024),
    "companion_pre_context_enabled": (bool, 0, 1),
    "companion_pre_context_min_keyword_overlap": (int, 1, 5),
    "companion_pre_context_max_notes_scan": (int, 1, 50),
    "companion_topic_mute_default_days": (int, 1, 3650),
    "companion_aging_enabled": (bool, 0, 1),
    "companion_aging_threshold_hours": (int, 1, 720),
    "companion_healing_enabled": (bool, 0, 1),
    "companion_min_tick_interval_s": (float, 0.0, 60.0),
    # Registered so it persists + restores across restart — it had a
    # config.py default but no validator entry, so toggling it off
    # reverted to on after restart (audit 2026-06-17).
    "companion_pad_emit_enabled": (bool, 0, 1),
    "companion_drives_enabled": (bool, 0, 1),
    "companion_drive_decay_half_life_hours": (float, 0.5, 168.0),
    "companion_energy_enabled": (bool, 0, 1),
    "companion_motion_cues_enabled": (bool, 0, 1),
    "companion_feedback_bias_enabled": (bool, 0, 1),
    "companion_reflection_trait_nudge_enabled": (bool, 0, 1),
    "companion_drift_audit_interval_hours": (float, 0.5, 720.0),  # [admin] 30min..30d
    "companion_creation_interval_hours": (float, 0.25, 720.0),  # [admin] 15min..30d
    "companion_today_enabled": (bool, 0, 1),
    "companion_today_reflect_hour_local": (int, 0, 23),
    "companion_today_max_chars": (int, 80, 2000),
    # Becca persona mode (top-level fork) + Lane 2/4 numeric knobs.
    # String-valued companion_care_cadence/locale/quiet_hours_* live in
    # _STRING_SETTINGS below.
    "companion_persona_mode": (bool, 0, 1),
    "companion_auto_summon": (bool, 0, 1),
    "companion_audio_cues": (bool, 0, 1),
    "companion_keyboard_shortcuts": (bool, 0, 1),
    "companion_notify_eod": (bool, 0, 1),
    "companion_notify_drift_audit_push": (bool, 0, 1),
    "companion_cooldown_minutes": (int, 0, 10_080),  # cap = 1 week
    "companion_discreet_auto_exit_minutes": (int, 0, 1440),  # cap = 24h
    "companion_discreet_location_aware": (bool, 0, 1),
    "companion_always_raw": (bool, 0, 1),
    "companion_safety_floor_threshold_chat": (float, 0.0, 1.0),
    "companion_safety_floor_threshold_coder": (float, 0.0, 1.0),
    "tts_emotion_aware": (bool, 0, 1),
    "tts_kokoro_hbe": (bool, 0, 1),
    "tts_kokoro_prosody": (bool, 0, 1),
    # Fabric voice routing — string-valued mode lives in _STRING_SETTINGS;
    # nothing numeric to register here. The pin_provider strings live there
    # too. Kept as a comment marker so future numeric routing controls
    # (e.g. round-robin weights) have an obvious place to land.
    # Connect federation (federated-PBX). Posture string is in
    # _STRING_SETTINGS; these are the boolean operator switches.
    "fabric_federation_enabled": (bool, 0, 1),
    "fabric_relay_sealed_only": (bool, 0, 1),
    "fabric_e2e_dm_enabled": (bool, 0, 1),
    "companion_e2e_participant_enabled": (bool, 0, 1),  # REQUEST only; hard-gated on standby
    # Ghost text (inline autocomplete)
    "ghost_text_enabled": (bool, 0, 1),
    # Discovery Engine
    "discovery_enabled": (bool, 0, 1),
    "knowledge_library_enabled": (bool, 0, 1),
    "knowledge_library_in_chat": (bool, 0, 1),
    "knowledge_library_retention_days": (int, 0, 3650),
    "discovery_max_recommendations": (int, 5, 50),
    # Agentic
    "agentic_max_steps": (int, 1, 100),
    "agentic_default_autonomy": (int, 1, 4),  # 1=suggest, 2=ask, 3=inform, 4=autonomous
    # Narrative context injection
    "narrative_context_budget": (int, 0, 128000),  # 0 = unlimited
    # Tool pipeline
    "tool_result_max_chars": (int, 1000, 128000),
    "tool_execution_timeout": (float, 10.0, 600.0),
    # Role resolver: refuse to silently fall through to a model below this size.
    # Prevents "Auto" picking a 0.8B model that can't follow distiller formats.
    "role_min_param_billions": (float, 0.0, 200.0),
    # Analytical verification thresholds
    "analytical_max_phase_retries": (int, 0, 5),
    "analytical_confidence_threshold": (float, 0.0, 1.0),
    "analytical_max_backtracks": (int, 0, 10),
    # Document RAG v2
    "document_rag_query_analysis": (int, 0, 1),
    "document_rag_cliff_ratio": (float, 0.1, 0.8),
    "document_rag_max_context_tokens": (int, 500, 5000),
    "document_rag_query_analysis_timeout": (float, 0.5, 5.0),
    # Application Builder
    "app_builder_improve_pass": (bool, 0, 1),
    "app_builder_max_improve_iterations": (int, 0, 5),
    "app_builder_max_fix_iterations": (int, 1, 8),
    "app_builder_auto_preview": (bool, 0, 1),
    "app_builder_max_tokens": (int, 1024, 32768),
    "app_builder_llm_timeout_seconds": (int, 60, 3600),
    # Memory dedup/contradiction thresholds
    "memory_dedup_threshold": (float, 0.5, 1.0),
    "memory_contradiction_threshold": (float, 0.3, 1.0),
    # Onboard-reasoner thinking for non-latency background tasks (Gemma E2B)
    "onboard_reasoning_thinking": (bool, 0, 1),
    "onboard_reasoning_max_tokens": (int, 512, 32768),
    # Metrics
    "metrics_enabled": (bool, 0, 1),
    # Startup
    "startup_warmup": (bool, 0, 1),
    # Rate limiting
    "rate_limit_enabled": (bool, 0, 1),
    "rate_limit_chat_rpm": (int, 1, 300),
    "rate_limit_image_rpm": (int, 1, 100),
    "rate_limit_voice_rpm": (int, 1, 50),
    # Session isolation
    "session_client_isolation": (bool, 0, 1),
    # Avatar
    "avatar_enabled": (bool, 0, 1),
    # Body Physics — hybrid SDF + Rapier body physics for VR/MR avatar
    # embodiment. Local soft-tissue compliance from the per-VRM SDF body
    # atlas drives indentation/jiggle on close-range touches; Rapier
    # ragdoll handles global chain dynamics (swaying torso, pendulum
    # motion). Defaults assume the avatar is present and tuned for a
    # natural feel without overwhelming low-end GPUs; turning these
    # off restores rigid-skeleton behavior.
    "body_physics_enabled": (bool, 0, 1),
    # Compliance gain: scales the local SDF-driven displacement
    # response. 1.0 = calibrated default; <1 stiffens, >1 makes the
    # surface "give" more on contact. UI slider hints 0..2; backend
    # allows the same range so users can disable compliance entirely
    # with 0.
    "body_physics_compliance_gain": (float, 0.0, 2.0),
    # Rapier weight: blend factor for the global ragdoll chain's
    # contribution to bone deltas. 0 = pure SDF, 2 = exaggerated
    # secondary motion. 0.6 = subtle and physically plausible.
    "body_physics_rapier_weight": (float, 0.0, 2.0),
    # Recovery rate (Hz): critically-damped spring frequency that
    # returns deformed regions to rest. Higher = snappier (less
    # lingering jiggle), lower = floatier. 6.0 Hz matches the
    # body-atlas Phase 1 calibration on Becca.
    "body_physics_recover_hz": (float, 2.0, 20.0),
    # Audio reactions: route impact events to the voice/SFX channel
    # (soft thumps, fabric rustle, etc.). Independent toggle so users
    # can keep physics on but mute the audio coupling.
    "body_physics_audio_reactions_enabled": (bool, 0, 1),
    # Visual feedback: contact-point glow, deformation shading, and
    # other render-side cues that surface the physics state. Off-loads
    # cheaply to the renderer; mainly a stylistic choice.
    "body_physics_visual_feedback_enabled": (bool, 0, 1),
    # Velocity-aware response: scale compliance + audio amplitude by
    # the incoming hand velocity so a tap feels different from a
    # press. Cheap on the math side but a few mid-range mobile GPUs
    # see a hit from the extra per-frame sampling — opt-out kept here
    # for those users.
    "body_physics_velocity_aware": (bool, 0, 1),
    # Provider toggles
    "google_vertex": (bool, 0, 1),
    # Knowledge packs
    "knowledge_packs_enabled": (bool, 0, 1),
    "knowledge_max_results": (int, 1, 20),
    "knowledge_min_score": (float, 0.0, 1.0),
    "knowledge_embedding_use_gpu": (bool, 0, 1),
    "knowledge_embedding_batch_size": (int, 32, 4096),
    "knowledge_catalog_cache_ttl": (int, 0, 604800),
    "ambient_volume": (int, 0, 100),
    # Auth
    "auth_session_ttl_hours": (int, 1, 8760),  # 1h to 1y
    "auth_lockout_threshold": (int, 1, 100),
    "auth_lockout_minutes": (int, 1, 1440),
    "auth_ip_lockout_threshold": (int, 1, 1000),
    "auth_ip_lockout_minutes": (int, 1, 1440),
    "auth_ws_ticket_ttl_seconds": (int, 5, 300),
    "auth_max_sessions_per_user": (int, 1, 100),
    # Files / VFS
    "files_webdav_enabled": (bool, 0, 1),
    "files_enrichment_enabled": (bool, 0, 1),
    "files_max_thumbnail_px": (int, 50, 500),
    "files_description_max_chars": (int, 100, 2000),
    "files_search_limit": (int, 5, 100),
    # Upload limits — exposed as configurable so users can fit their use
    # case (short clips vs full media library) without editing config.py.
    # Bounds chosen wide enough to cover everything from "tiny CSV" to
    # "raw 4K video"; sane defaults live in config.py.
    "files_upload_max_file_bytes": (int, 1024 * 1024, 10 * 1024 * 1024 * 1024),         # 1 MB – 10 GB
    "files_upload_max_files_per_request": (int, 1, 1000),
    "files_upload_max_request_bytes": (int, 1024 * 1024, 50 * 1024 * 1024 * 1024),      # 1 MB – 50 GB
    "files_user_storage_quota_bytes": (int, 0, 1024 * 1024 * 1024 * 1024),              # 0 (off) – 1 TB
    # External-API request bounds (enforced default-on; see config.py).
    "max_request_body_bytes": (int, 0, 2 * 1024 * 1024 * 1024),                         # 0 (off) – 2 GB general transport cap
    "api_embeddings_max_items": (int, 0, 100_000),                                      # 0 = unbounded
    "api_embeddings_max_chars": (int, 0, 100_000_000),                                  # 0 = unbounded
    "api_tts_max_chars": (int, 0, 10_000_000),                                          # 0 = unbounded
    "api_stt_max_bytes": (int, 0, 1024 * 1024 * 1024),                                  # 0 = unbounded (1 GB ceiling)
    # Dream compaction — admin-global; UI exposes these in the dream tab's
    # Advanced section behind a data-admin-only gate. Bounds chosen to span
    # "very conservative" to "very aggressive" without permitting nonsensical
    # values that would crash the compactor (e.g. min cluster size of 1
    # would dedup-spiral any single entry).
    "dream_compaction_enabled": (bool, None, None),
    "dream_compaction_interval_hours": (float, 1.0, 168.0),
    "dream_dedup_threshold": (float, 0.5, 0.99),
    "dream_cluster_threshold": (float, 0.4, 0.95),
    "dream_cluster_min_size": (int, 2, 20),
    "dream_compaction_max_clusters_per_run": (int, 1, 50),
    "dream_consolidation_low": (float, 0.4, 0.9),
    "dream_consolidation_high": (float, 0.5, 0.99),
    "dream_time_trim_count_threshold": (int, 50, 10000),
    "dream_compaction_max_age_days": (int, 7, 3650),
    # Engine KV cache persistence — controls how long llama-server slot
    # save files (the on-disk KV state that lets a chat resume without
    # re-prefilling its prompt) live before eviction. Changes apply on
    # the next save / model reload via update_tool_settings's manager
    # propagation below; see _propagate_kv_settings.
    "engine_kv_ttl_days": (int, 0, 365),  # 0 = never expire
    "engine_kv_narrative_ttl_days": (int, 0, 365),
    "engine_kv_max_snapshots_per_model": (int, 1, 100),
    "engine_kv_auto_pin_narrative": (bool, 0, 1),
    # Engine defaults — global model-loading knobs. Each one's
    # propagation behavior is documented inline in
    # ``_propagate_kv_settings`` below: idle_timeout applies live;
    # the rest take effect on the next subprocess start.
    "engine_idle_timeout": (float, 0.0, 86400.0),  # 0 = never unload, max 24h
    "engine_use_jinja_template": (bool, 0, 1),
    "engine_kv_warm_on_start": (bool, 0, 1),
    # Resume ladder (restore→replay→cold). Replay settings are read
    # live from the config module by kv_resume.py / llama_cpp.py — no
    # manager propagation needed.
    "engine_kv_replay_enabled": (bool, 0, 1),
    "engine_kv_replay_warm_sessions": (int, 0, 16),
    "engine_kv_replay_budget_s": (float, 0.0, 3600.0),
    "engine_kv_replay_max_rows": (int, 4, 1024),
    # Speculative turn generation (ladder rung 3) — same live-read
    # convention as the replay settings above.
    "engine_speculation_enabled": (bool, 0, 1),
    "engine_speculation_prefill_only": (bool, 0, 1),
    "engine_speculation_max_new_tokens": (int, 16, 32768),
    "engine_speculation_ttl_s": (float, 5.0, 3600.0),
    # Reasoning budget cap (chat-template kwarg). 0 = no cap. Ceiling
    # generous so per-model model card recommendations stay in range
    # (Nemotron Omni: 16384 + 1024; future models may go higher).
    "engine_reasoning_budget": (int, 0, 131072),
    "engine_reasoning_grace_period": (int, 0, 8192),
    # Hardware/storage compatibility knobs. flash_attn must be off
    # on Pascal-class cards (GTX 1080 etc.) which lack the required
    # CUDA capability; on by default for Ampere+. health_timeout is
    # how long we wait for /health to return 200 — slow disks or
    # huge models on bind mounts may legitimately exceed the 5-min
    # default. 60 s floor; 30 min ceiling (any longer and the user
    # has bigger problems than a setting can fix).
    "engine_flash_attn": (bool, 0, 1),
    "engine_health_timeout": (float, 60.0, 1800.0),
    # Fabric: master enable for cross-instance peer coordination.
    # When False (default) every fabric code path no-ops. Toggling on
    # by itself doesn't start any fabric work -- higher phases gate
    # transport / advertisement / routing on this flag.
    "fabric_enabled": (bool, 0, 1),

    # Multi-slot KV cache architecture (see
    # docs/superpowers/specs/2026-05-05-multi-slot-kv-design.md). Off
    # by default; flip on to run llama-server with --parallel -1
    # --kv-unified --cache-ram <auto> --cache-idle-slots ... and let
    # Augmentum route requests via response-observed id_slot. Requires
    # a llama-server restart to take effect (the args are baked at
    # subprocess start; toggling at runtime updates the setting but
    # next ``start.bat`` reload picks it up).
    "engine_multislot_enabled": (bool, 0, 1),
    # Number of parallel slots when multi-slot is enabled. 0 = auto
    # (llama-server's --parallel -1 resolves to 4 at b8935). 1-32
    # explicit. Default 0 (auto). Override only for households or
    # constrained hardware.
    "engine_parallel_slots": (int, 0, 32),
    # MTP self-speculation (upstream PR #22673). Enables
    # --spec-type draft-mtp at llama-server start, gated at runtime by
    # the loaded GGUF actually having built-in MTP heads (DeepSeek
    # V3/V4, Qwen 3.6, Gemma 4 MTP builds). Without heads the runtime
    # silently skips MTP — see ``llama_mtp_skipped_no_heads`` log line.
    # n_max bounded at 8 because acceptance rate falls off a cliff
    # past 3-4 in the upstream PR's measurements.
    "engine_mtp_enabled": (bool, 0, 1),
    # Range 1-16. Earlier guidance ("falls off past 3-4") was stale —
    # empirically n_max=12 on Qwen 3.6 27B with cached prefix gives
    # ~70% speedup over n_max=2 (~45→77 tok/s) for ~1.5 GB VRAM cost.
    # Ceiling at 16 to keep the verify-batch buffer sane; past that
    # diminishing returns + real OOM risk on tight VRAM budgets.
    "engine_mtp_n_max": (int, 1, 16),
    # Host-RAM warm-tier KV cache size (--cache-ram). 0 = auto-size
    # from system RAM (min(16384, total_ram_mib*0.25), floor 1024).
    # Manual values 1024-65536 MiB. Default 0 (auto).
    "engine_cache_ram_mib": (int, 0, 65536),
    # Min chunk size (tokens) for mid-prompt KV salvage via shifting
    # (--cache-reuse). 0 = disabled. Self-gated by llama-server on
    # models whose memory can't shift (hybrid attention), so leaving it
    # on is safe for every family.
    "engine_cache_reuse_min": (int, 0, 4096),
    # Auto-pair a sibling mmproj when one exists alongside the base GGUF.
    # Default OFF because llama.cpp returns 501 on /slots/.../save+restore
    # whenever --mmproj is loaded, which silently kills KV session
    # restoration for every chat on a vision-capable model — most of
    # which never actually send an image. Vision is still available via
    # per-load opt-in (vision_mode in Load Setup) or manual sidecar
    # pairing through /api/models/.../projector.
    "engine_auto_pair_mmproj": (bool, 0, 1),
    # Observation Substrate (BOM) — Phase A. Three opt-in flags so the
    # pipeline can be staged: enable substrate → enable seeding →
    # enable lookup-cache wiring. See augmentum/observation/ for the
    # implementation and exporter.py for the cache-build path.
    "observation_substrate_enabled": (bool, 0, 1),
    "observation_seed_chat_history": (bool, 0, 1),
    "observation_lookup_cache_enabled": (bool, 0, 1),
    # 1k - 500k bound: under 1k the cache is too sparse to help; over
    # 500k the corpus builds slowly enough that model swaps stall.
    "observation_lookup_cache_max_entries": (int, 1_000, 500_000),
    # observation_primary_user_id is a string — registered in
    # _STRING_SETTINGS below, not here.
    # Cast Comics rail — hard ceiling on chapter rows fetched when
    # building the series-collapsed rail and drill-in. Auto-grows up
    # to this value based on the user's actual library size; only
    # needs adjustment for libraries past ~200k chapters. Warn-logs
    # when truncating so users know to raise it.
    "cast_comic_library_ceiling": (int, 1000, 10_000_000),
    # Cast Gallery — include private images (off by default). When on,
    # the gallery's "Private" category surfaces images marked private
    # via the chat image library. Off-by-default so a TV in a shared
    # room doesn't paint private content on the wall.
    "cast_gallery_show_private": (bool, 0, 1),
    "tv_auto_update": (bool, 0, 1),
    # Coder hybrid-loop breaker tuning. 0 = use the registered default
    # in augmentum/loops/breakers.py. Any positive int overrides at
    # runtime via live_threshold() — no restart needed. Upper bounds
    # are generous because the user is the one deciding how much
    # patience to give the agent.
    "coder_breaker_validation_error_streak": (int, 0, 100),
    "coder_breaker_same_validation_error_repeat": (int, 0, 100),
    "coder_breaker_action_stagnation_break": (int, 0, 200),
    "coder_breaker_test_failure_streak": (int, 0, 100),
    "coder_breaker_same_file_edit_break": (int, 0, 200),
    "coder_breaker_no_write_progress_break": (int, 0, 200),
    "coder_breaker_inspection_loop_nudge": (int, 0, 100),
    "coder_breaker_inspection_loop_break": (int, 0, 100),
    "coder_verify_enabled": (bool, 0, 1),
    "coder_hybrid_max_iters": (int, 0, 5000),
    "coder_hybrid_max_iters_ungated": (int, 0, 10000),
    "coder_native_nudge_max": (int, 0, 10),
    "game_agent_frame_dedup_enabled": (bool, 0, 1),  # collapse redundant near-identical frames in the perception window
    "game_agent_grid_overlay_enabled": (bool, 0, 1),  # draw a labeled Set-of-Marks grid on agent frames for spatial grounding
    "game_agent_thinking_enabled": (bool, 0, 1),  # let the planner reason before answering (helps small local models)
    "game_agent_max_tokens": (int, 256, 8192),  # planning-reply budget; large enough that reasoning + JSON both fit
    "game_agent_frame_max_edge": (int, 0, 4096),  # longest-edge cap on agent frames; 0 = ship the display canvas raw
    "game_agent_fast_turns_enabled": (bool, 0, 1),  # rolling call-window micro-plan turns between full plans
    "game_agent_full_turn_every": (int, 1, 64),  # run a FULL plan turn every N fast turns (1 = every turn full)
    "game_agent_fast_max_tokens": (int, 32, 1024),  # micro-plan reply budget (strict tiny JSON, no thinking)
    "game_agent_scene_narrator_enabled": (bool, 0, 1),  # parallel vision lane: live-feed scene description
    "game_agent_scene_interval_ms": (int, 500, 10000),  # narrator cadence; fingerprint-gated so static screens skip
    "game_agent_stall_after_s": (int, 10, 600),  # STALLED marker after this many seconds of zero world change
    "coder_next_speaker_check_enabled": (bool, 0, 1),
    "coder_goal_judge_enabled": (bool, 0, 1),
    "coder_think_tool_enabled": (bool, 0, 1),  # elective `think` tool exposure (native-thinking-off turns only)
    "coder_compact_tool_enabled": (bool, 0, 1),  # model-initiated `compact` tool (native loop) + sticky context meter
    "coder_verify_command_gate_enabled": (bool, 0, 1),  # held-out verify gate; command string in _STRING_SETTINGS
    "coder_maker_agreements_enabled": (bool, 0, 1),  # inject accrued Working Agreements (mig 273) into the coder prompt
    "coder_auto_recall_enabled": (bool, 0, 1),  # auto-inject relevant past turns from the durable archive into each turn's context
    "coder_auto_recall_k": (int, 1, 10),  # max past turns auto-recalled per turn
    "coder_auto_recall_max_distance": (float, 0.0, 100.0),  # L2-distance ceiling for an auto-recall hit; 0 = no filter
    "coder_compaction_auto_enabled": (bool, 0, 1),
    "coder_compaction_threshold": (float, 0.3, 0.95),
    "coder_compaction_keep_recent": (int, 4, 40),
    "coder_compaction_synthesis_enabled": (bool, 0, 1),  # LLM handoff note in new compacted segments; fails open to mechanical
    "coder_request_delay_enabled": (bool, 0, 1),  # pace the agentic loop to stay under fast-provider rate limits
    "coder_request_delay_seconds": (float, 0.0, 120.0),  # seconds to sleep before each loop request when pacing is on
    # Narrative recall-tools (LLM-callable lookup layer).
    "narrative_recall_tools_enabled": (bool, 0, 1),
    "narrative_recall_tools_max_iters": (int, 1, 10),
    "narrative_lorebook_tools_enabled": (bool, 0, 1),
    "narrative_lorebook_native_tools_enabled": (bool, 0, 1),
    "narrative_world_systems_enabled": (bool, 0, 1),
    # Connect lives in _UI_SETTINGS (per-user, no admin gate) — these
    # keys used to be here, but each user owning their own
    # discoverability + connect-enabled is structurally a UI setting,
    # not an install-wide tool config. See the camelCase entries
    # connectEnabled / connectDiscoverableSameInstance / etc. in
    # _UI_SETTINGS below.
    # Notification substrate. See
    # docs/superpowers/specs/2026-06-01-notification-substrate-design.md
    "notifications_enabled": (bool, 0, 1),
    "notification_sound_enabled": (bool, 0, 1),
    # Offers (chat-LLM-emitted Install/Save/Switch chips). See
    # docs/superpowers/specs/2026-06-02-offer-substrate-design.md.
    "offers_enabled": (bool, 0, 1),
    "offers_max_per_day": (int, 0, 200),
    "offers_max_per_turn": (int, 0, 10),
    "offers_max_pending_per_session": (int, 0, 50),
    "offers_default_expiry_days": (int, 1, 90),
    "coder_archive_enabled": (bool, 0, 1),
    "coder_archive_max_turns_per_workspace": (int, 0, 1_000_000),
    "coder_workspace_pids_limit": (int, 256, 16_384),
    "coder_workspace_pids_warn_pct": (float, 0.0, 0.99),
    "coder_workspace_pids_check_interval_s": (int, 0, 3600),
    "coder_max_paused_seconds": (int, 0, 86_400),
    "coder_paused_sweep_interval_s": (int, 0, 600),
    "coder_pause_idle": (bool, 0, 1),
    "coder_pause_stop_after_seconds": (int, 0, 7 * 86_400),
    # file_write per-call token cap. 0 = uncapped (default, matches
    # Claude Code / Codex CLI). D1 truncation-detection in the coder
    # handler catches mid-arguments cutoff at runtime; raise above 0
    # only for weak local models with tiny output budgets that benefit
    # from a hard pre-emptive refusal.
    "coder_file_write_max_tokens": (int, 0, 32000),
    # Subagent dispatch — feature flag + concurrency/depth caps.
    "coder_subagents_enabled": (bool, 0, 1),
    "coder_subagent_auto_explore": (bool, 0, 1),
    "coder_subagent_max_concurrent": (int, 1, 16),
    "coder_subagent_max_depth": (int, 1, 4),
    # Subagent return-path verification — judge subagent output against
    # the lead's success_criteria before honoring the stop.
    "coder_subagent_verify_enabled": (bool, 0, 1),
    "coder_subagent_verify_reentry": (int, 0, 3),
    # Fraction of the model context window kept as headroom for
    # output + tool schemas + reasoning. Lower = more usable window
    # for coder context, higher = safer truncation margin. 0.02 floor
    # / 0.40 ceiling clamps a misconfig from starving the model.
    "coder_context_reserve_pct": (float, 0.02, 0.40),
    # Local-backend per-response output budget as % of ctx. 0 disables
    # (falls back to the flat mode-hint default for local too). Range
    # 5-90 keeps the floor sane — too low and we under-use the window;
    # too high and the model rarely needs that much output and we
    # waste KV scheduling.
    "coder_local_max_tokens_pct": (int, 0, 90),
    # Absolute ceiling on the computed local output floor. 0 = no
    # absolute cap (use the % directly). Caps at 256K so a configured
    # value can't induce a server-side rejection.
    "coder_local_max_tokens_cap": (int, 0, 262144),
    "coder_cloud_max_tokens_floor": (int, 0, 384000),
    # MCP (Model Context Protocol). Install-wide toggle — controls both
    # the inbound /mcp surface (Augmentum-as-server) and the outbound
    # MCPClientManager that connects to external MCP servers.
    "mcp_enabled": (bool, 0, 1),
    # Community install (Open in Augmentum) master kill switch.
    # See augmentum/proxy/community_routes.py.
    "community_install_enabled": (bool, 0, 1),
}


# Defaults for tool settings whose corresponding fields are NOT yet
# declared on the Settings dataclass in ``augmentum/config.py``. The
# GET endpoint does ``getattr(settings, key)`` which would raise on a
# fresh boot before any PUT has populated the attr via
# ``object.__setattr__``. Seeding these at module import time ensures
# the first read returns the documented default; once a user issues a
# PUT, persistence and propagation flow through the normal path
# unchanged. New settings registered in ``_TOOL_SETTINGS`` whose key
# isn't also a field on ``Settings`` should be added here too — the
# audit's wiring checker will flag missing defaults otherwise.
_TOOL_SETTING_DEFAULTS: dict[str, object] = {
    # Body physics is beta — defaults OFF. User opts in via Personalize →
    # Body Physics → "Enable body physics". Toggle flows through the
    # coordinator which gates compliance gain / rapier weight / contact
    # reactor enablement live (no restart required).
    "body_physics_enabled": False,
    "body_physics_compliance_gain": 1.0,
    "body_physics_rapier_weight": 0.6,
    "body_physics_recover_hz": 6.0,
    "body_physics_audio_reactions_enabled": True,
    "body_physics_visual_feedback_enabled": True,
    "body_physics_velocity_aware": True,
    # Cast Comics rail ceiling — default 200k covers any realistic
    # personal library; raise this if you legitimately have more
    # chapters and the warning log fires.
    "cast_comic_library_ceiling": 200_000,
    # Cast Gallery — private images excluded by default for shared-TV
    # safety. Flip via admin settings PUT or the controller's
    # "Private" chip gate.
    "cast_gallery_show_private": False,
    "tv_auto_update": True,
}

for _bp_key, _bp_default in _TOOL_SETTING_DEFAULTS.items():
    # ``hasattr`` here covers both the "declared on Settings" path and
    # the "previously seeded by an earlier import" path so we never
    # clobber a value already set (e.g. by a hypothetical future
    # _restore_settings entry).
    if not hasattr(settings, _bp_key):
        object.__setattr__(settings, _bp_key, _bp_default)


def _propagate_kv_settings(request: Request, updated: dict[str, object]) -> None:
    """Mirror engine_kv_* changes onto the live LlamaServerManager.

    The manager caches these as instance attrs at construction (see
    server.py wiring). Without this, edits via /api/config/tools update
    the Settings singleton but the manager keeps using the old values
    until the next process restart, which defeats the point of a
    runtime-editable UI.
    """
    manager = getattr(request.app.state, "llama_manager", None)
    if manager is None:
        return
    if "engine_kv_ttl_days" in updated:
        manager.kv_ttl_days = int(updated["engine_kv_ttl_days"])
    if "engine_kv_narrative_ttl_days" in updated:
        manager.kv_narrative_ttl_days = int(updated["engine_kv_narrative_ttl_days"])
    if "engine_kv_max_snapshots_per_model" in updated:
        manager.kv_max_snapshots_per_model = int(updated["engine_kv_max_snapshots_per_model"])
    if "engine_kv_auto_pin_narrative" in updated:
        manager.kv_auto_pin_narrative = bool(updated["engine_kv_auto_pin_narrative"])
    # Idle timeout applies live — the idle monitor reads
    # ``self.idle_timeout`` on each tick, so a runtime change takes
    # effect on the next 30s monitor cycle without restart.
    if "engine_idle_timeout" in updated:
        manager.idle_timeout = float(updated["engine_idle_timeout"])
    # flash_attn + health_timeout: mirror to the manager's instance
    # attrs so the NEXT model start picks them up. Doesn't affect the
    # currently-running subprocess (its CLI args are baked); does
    # affect any subsequent ``manager.start()`` call (idle reload,
    # model swap, explicit restart).
    if "engine_flash_attn" in updated:
        manager.flash_attn = bool(updated["engine_flash_attn"])
    if "engine_health_timeout" in updated:
        manager.health_timeout = float(updated["engine_health_timeout"])
    # MTP toggles affect only the NEXT subprocess start (same caveat
    # as flash_attn). Runtime gate in _build_cli_args still applies —
    # toggling on without an MTP-headed GGUF loaded is a no-op + log.
    if "engine_mtp_enabled" in updated:
        manager.mtp_enabled = bool(updated["engine_mtp_enabled"])
    if "engine_mtp_n_max" in updated:
        manager.mtp_n_max = int(updated["engine_mtp_n_max"])
    # The remaining engine_* knobs (kv_cache_type, use_jinja_template,
    # reasoning_format, kv_warm_on_start) are baked into the
    # llama-server subprocess command line at startup. Updating them
    # here only changes the persisted setting; the running subprocess
    # keeps the values it was started with. Next subprocess start
    # (model swap, idle reload, manual restart) picks up the new
    # values. The UI surfaces this caveat per-row.

# String settings that accept free-form text (no min/max range).
_STRING_SETTINGS: dict[str, int] = {
    "game_agent_journal_dir": 256,  # game agent long-horizon journal dir; "" = auto (/data when present)
    "game_agent_planner_model": 256,  # model for FULL planning turns; "" = same as default (user-chosen, never auto)
    "game_agent_fast_model": 256,  # model for fast turns + scene narrator (vision-capable); "" = default
    "training_capture_user_id": 64,  # scope training capture to this user ID (empty = all)
    "training_capture_dir": 256,  # output dir for training traces
    "native_primer_models": 256,  # F7 primer-served model name patterns (comma-sep)
    "passthrough_tools": 2048,  # comma-separated default tools for passthrough mode
    "comic_default_reading_direction": 8,  # "ltr" | "rtl" — seed for comic + narration direction
    "selfedit_autonomy_level": 16,  # "propose" | "auto_verified" — see selfedit/promote.py
    "selfedit_engine": 24,  # "native" | "claude_code" | "codex" — see selfedit/engine_select.py
    "selfedit_edit_model": 128,  # native edit-engine model pin (empty = utility role)
    "selfedit_frontier_model": 128,  # top-rung frontier model for the escalation ladder
    "connect_instance_handle": 128,  # Connect: public name of this Augmentum for peer-DID addressing (empty = derive from public host). See augmentum/connect/contacts.py::instance_handle
    "fabric_admission_posture": 16,  # federated-PBX stranger posture: private|allowlist|knock|open. See augmentum/fabric/knock.py
    "tts_openai_url": 256,  # generic OpenAI-compatible /v1/audio/speech endpoint (bring your own model)
    "notification_sound": 16,  # "auto" | chime/bloom/ping/bell/drop/ring/pop
    "multi_model_models": 2048,  # comma-separated compare models for passthrough fan-out
    "timezone": 64,  # IANA timezone (e.g. "America/New_York")
    "companion_activation_mode": 32,  # wake_word | always_listening | ptt_only
    "companion_csm_residency": 16,  # session | timer | always — CSM voice GPU residency
    "coder_workspace_network_mode": 16,  # "bridge" | "none" — invalid values fall back to bridge in containers.py
    "coder_verify_command": 256,  # held-out verification gate command; "" = auto-detect (see config.py)
    "companion_attention_sources": 256,  # comma-separated auth-session sources allowed to feed attention threads
    "companion_promise_deliver_tier": 16,  # "primary" | "utility" — see config.py
    "companion_speak_tier": 16,  # "primary" | "utility" — model the companion speaks with (see config.py)
    "companion_address_llm_model": 128,  # optional override for the Tier 3 classifier
    "location": 128,  # User location for geo-aware search (e.g. "Portland, OR")
    "image_prompt_condense_model": 256,  # max length
    "image_imports_dir": 512,  # Allowlisted server-side path for offline imports (empty = disabled)
    "chromium_binary_path": 512,  # Explicit chrome/chromium path; empty = auto-discover. See augmentum/tools/application_cdp.py::find_chromium.
    "uarf_verify_model": 256,  # cross-model verification model
    "memory_llm_extraction_model": 256,  # LLM model for memory extraction
    "narrative_scene_image_model": 256,  # image model for /v in narrative mode
    "narrative_extraction_model": 256,  # LLM model for narrative extraction
    "narrative_memory_model": 256,  # LLM model for memory summary
    "narrative_memory_mode": 16,  # "lite" or "standard"
    "narrative_memory_prompt": 2000,  # Custom LTM system prompt
    "narrative_translate_default_language": 64,  # Default target language for card translate
    "search_proxies": 4096,  # Newline-separated proxy URLs for SearXNG outbound (http/https/socks5)
    "narrative_scene_distiller_model": 256,  # LLM model for scene prompt generation
    "narrative_auto_bg_distiller_model": 256,  # LLM for background distillation
    "narrative_auto_bg_image_model": 256,  # image model for auto backgrounds
    "image_torch_compile": 10,  # "auto", "on", "off"
    # Engine defaults (string-valued). UI-side dropdowns constrain to
    # the documented value sets; backend stores any string ≤ N chars.
    "engine_reasoning_format": 16,  # "deepseek" | "none" — see --reasoning-format
    "engine_kv_cache_type": 8,      # "" | "q8_0" | "q4_0" | "f16" — KV cache quant
    # Model id last loaded into the secondary slot ("Slot B"). Written by
    # the /api/engine/v2/secondary/load route so the picker entry re-pins
    # to the slot after a restart. Per-model load config travels with the
    # model via engine.last_load.<model_id>, not here.
    "engine_secondary_model": 256,
    # Managed classifier slot ("Slot C") model id/path — set via the
    # /api/engine/v2/classifier/load route; re-loaded on boot.
    "classifier_slot_model": 256,
    # Observation Substrate (BOM) — single-tenant primary user whose
    # cache feeds llama-server's --lookup-cache-static slot. See the
    # config.py comment for the multi-tenant caveat.
    "observation_primary_user_id": 64,
    "voice_tts_chunking": 16,  # TTS chunking mode: sentence, clause, full
    "voice_tts_lexicon": 4000,  # JSON term→spoken map for TTS pronunciation
    "voice_lipsync_engine": 16,  # "amplitude" | "phoneme" | "auto" — see voice/phoneme_lipsync.py
    "tts_voice_style": 256,  # Default TTS voice style instruct (e.g. "speak warmly")
    "coder_subagent_fast_model": 256,  # Fast/cheap model for explore+research fan-out roles (empty = Slot B, then parent)
    "ghost_text_model": 256,  # Model for ghost text autocomplete (empty = current model)
    "utility_model": 256,       # Core role: internal tasks
    "classifier_model": 256,    # Core role: routing/classification
    "primary_chat_model": 256,  # Mirror of UI's selected chat model (for "Auto" role resolution)
    # Core role: frontier-tier model for quality-critical work
    # (Bug Finder verifier, stagnation escalation, future
    # /second-opinion, narrative summarizer escalation, classifier
    # hard-case fallback). Per-workspace overrides in
    # coder_workspaces.bug_finder_verifier_model take precedence.
    # Accepts the model-spec syntax (id | id@provider |
    # id@fabric:peer_id) routed through resolve_backend_with_fabric.
    "heavyweight_model": 256,
    # Game-foundry visual verification: which vision-capable model judges
    # Blender renders + game frames in the coder foundry loop. Empty = "Auto"
    # (fall through the VisionRouter's current capability/workload routing —
    # primary VL, else classifier-with-mmproj, else SmolVLM CPU fallback). A
    # pinned id routes there explicitly. Model-spec syntax (id | id@provider |
    # id@fabric:peer_id) via resolve_backend_with_fabric.
    "coder_visual_verify_model": 256,
    "local_fabric_icon": 8,     # Phase 8: operator-chosen emoji for THIS node in the fabric UI
    "typography_custom_fonts": 4000,  # JSON array of custom Google Font objects
    "typography_selected": 64,  # Active typography preset key
    "typography_text_scale": 8,  # Global text size multiplier
    "agentic_artifact_theme": 32,  # Artifact theme: slate, corporate, modern, emerald, rose
    "agentic_image_model": 256,  # Image model override for agentic illustrations
    "document_rag_query_analysis_model": 256,
    "huggingface_token": 256,  # HuggingFace API token for gated model downloads
    "tts_kokoro_quality": 8,   # "int8" (CPU) or "fp16" (GPU)
    # Fabric voice routing mode + pin target. See augmentum/config.py for
    # semantics (auto/round_robin/pin). Pin provider strings can be long
    # (fabric:<node_id>:<provider_id> = ~80 chars) — 256 leaves headroom.
    "voice_routing_mode": 16,
    "voice_routing_pin_provider": 256,
    "stt_routing_mode": 16,
    "stt_routing_pin_provider": 256,
    "knowledge_packs_custom_dir": 512,
    "knowledge_featured_packs": 1024,
    "soundscape_favorites": 8000,
    "soundscape_last_station": 1000,
    "ambient_video": 512,
    "ambient_loop_mode": 16,
    "ambient_favorites": 8000,
    # Becca persona mode — string-valued knobs. companion_care_cadence is
    # one of {sparse, normal, lively}; the values are enforced at the
    # application layer, not by length.
    "companion_care_cadence": 16,
    "companion_presence_mode": 16,  # silent | gentle | engaged
    "companion_locale": 16,  # IETF language tag, e.g. "en-US"
    "companion_quiet_hours_start": 8,  # "HH:MM" or "24:00"
    "companion_quiet_hours_end": 8,
    "companion_default_owner_user_id": 64,  # [admin] override; resolves dream + drift user_id
    "companion_journal_hushed_until": 32,  # ISO-8601 "YYYY-MM-DD HH:MM:SS", "" = not hushed
    "vision_provider_model_path": 512,  # absolute path to SmolVLM base GGUF
    "vision_provider_mmproj_path": 512,  # absolute path to paired mmproj GGUF
    "tv_update_channel": 16,  # "stable", "beta"
    # Provider API keys
    "anthropic_api_key": 256,
    "anthropic_base_url": 512,
    "google_api_key": 256,
    "google_vertex_project": 128,
    "google_vertex_region": 64,
    "cohere_api_key": 256,
    "mistral_api_key": 256,
    "deepseek_api_key": 256,
    "openrouter_api_key": 256,
    "xai_api_key": 256,
    "groq_api_key": 256,
    "perplexity_api_key": 256,
    "fireworks_api_key": 256,
    "azure_api_key": 256,
    "azure_base_url": 512,
    "azure_deployment": 128,
    "azure_api_version": 32,
    # Game Portal
    "game_portal_recommendations": 32,     # "off" | "contextual" | "always"
    "game_portal_default_sources": 512,    # CSV of source ids
    # Controllers (string-valued; numeric/bool controller settings live in
    # _TOOL_SETTINGS above). Frontend sync was wired in settings.js but
    # the API layer dropped these as "unknown" until registered here.
    "controller_touch_overlay": 8,   # "auto" | "on" | "off"
    "controller_pad_routing": 16,    # "index" | "firstpress"
    # JSON array of MCP server configs, e.g.
    # [{"name":"github","url":"https://..."},{"name":"local","command":"npx","args":["-y","@x/mcp"]}].
    # stdio entries (those with "command") are honored only when seeded
    # from AUGMENTUM_MCP_SERVERS env var on startup — the /v1/mcp/connect
    # API rejects them. HTTP entries can be added live and are persisted
    # here for restart-survivability.
    "mcp_servers": 16384,
    # Voice pipeline mode per consumer surface. One of:
    # "auto" | "local" | "server" | "custom". See config.py for semantics.
    # The resolver in voice/pipeline_resolver.py consumes these.
    "voice_pipeline_mode_call": 8,
    "voice_pipeline_mode_companion": 8,
    "voice_pipeline_mode_narration": 8,
    "voice_pipeline_mode_readaloud": 8,
}


# ---------------------------------------------------------------------------
# Declarative-action-substrate overlay (Phase 1B).
#
# Settings registered in ``augmentum.registry`` are merged into the
# literal dicts above so the live validator/restore paths read those
# declarations through the existing endpoints unchanged. Registry wins
# on conflict — this is intentional, the registry is the canonical
# source going forward.
#
# Phase 1B keeps the literal entries in place (no deletions) so any
# subsystem still importing them by name continues to work. Phase 1C
# will remove the literal entries for fully-migrated settings, once
# the audit confirms zero in-tree readers of the duplicates.
#
# Spec: docs/superpowers/specs/2026-06-04-declarative-action-substrate-design.md
# ---------------------------------------------------------------------------
def _overlay_declarative_registry() -> None:
    try:
        from augmentum.registry.builtin import load_into_default_registry
        from augmentum.registry.registry import get_registry
    except ImportError:
        # Defensive — substrate import shouldn't fail, but a partial
        # checkout / packaging edge case shouldn't break startup.
        return
    load_into_default_registry()
    registry = get_registry()
    _TOOL_SETTINGS.update(registry.to_tool_settings())
    _STRING_SETTINGS.update(registry.to_string_settings())


_overlay_declarative_registry()


@router.get("/")
async def get_config() -> dict:
    """Get current configuration (sanitized — no API keys)."""
    return settings.to_safe_dict()


@router.get("/section/{section}")
async def get_config_section(section: str) -> dict:
    """Get a specific configuration section."""
    safe = settings.to_safe_dict()
    prefix = section.lower() + "_"
    return {k: v for k, v in safe.items() if k.startswith(prefix)}


# Tools hidden from the passthrough dropdown because they are subsumed by the
# unified ``web`` tool, internal-only, or auto-included utilities.
_PASSTHROUGH_HIDDEN: frozenset[str] = frozenset({
    "web_search",      # subsumed by unified "web" tool
    "web_fetch",       # subsumed by unified "web" tool
    "memory_recall",   # internal — auto-injected by memory system
    "math_verify",     # internal — used by UARF verify phase
    "document_parse",  # subsumed by unified file attach + RAG injection
    # Artifact tools whose creation needs a confirmed outline — proposed as an
    # offer chip in passthrough Auto rather than picked here.
    "create_document",
    "create_presentation",
    # NOTE: create_chart / create_spreadsheet are deliberately NOT hidden
    # (2026-07-26). They used to be, on the rationale that artifact tools
    # "require multi-step planning (analytical/agentic only)" — stale since
    # passthrough's _execute_tool gained the artifact-pipeline intercept.
    # Ticking one here is the explicit-consent path; this listing already
    # greys them out via health_check() when matplotlib/openpyxl are absent.
    # Auto-included utilities — hidden because they're always present when any tool is active
    "calculator",
    "datetime",
    "unit_converter",
    # Internal utilities — not useful for direct user selection
    "hash",
    "json_tool",
    "text_analysis",
    "consistency_check",
    "draft_section",
    # Scheduling — collapsed behind the single unified `schedule` tool
    # (augmentum/tools/schedule.py). The individuals stay registered for
    # voice/companion, but the chat panel shows one clean button.
    "schedule_briefing",
    "schedule_reminder",
    "schedule_deadline",
    "watch_for",
    "schedule_request",
    "schedule_action",
    "list_briefings",
    "cancel_briefing",
})

# Utility tools that are auto-included whenever the user enables any tool.
# No external deps, no user-facing cost — always useful for the LLM.
#
# The briefing trio (schedule/list/cancel) is NOT here. It's surfaced
# unconditionally under the companion_runtime_enabled gate inside
# handler_factory._resolve_passthrough_tools, because it represents
# substrate (the companion's own capabilities), not opt-in utility.
PASSTHROUGH_AUTO_TOOLS: frozenset[str] = frozenset({
    "calculator",
    # "datetime" removed — DateTimeTool no longer exists (the date/time is
    # injected into every system prompt instead). Leaving it here made the
    # passthrough loop try to resolve a non-existent tool on every request
    # ("passthrough_tool_not_found name=datetime").
    "unit_converter",
})

# Scheduling substrate — rides through the SAME auto-include mechanism as
# the utilities above (2026-07-02, replacing a bespoke force-injection
# path in handler_factory): present whenever the user has any tool
# enabled AND a scheduling dispatcher exists (companion runtime or
# SchedulerService), absent on toolless configs (which keeps the pure-
# streaming fast path), removable with the "none" header like everything
# else. The model decides whether to schedule — no keyword gating.
PASSTHROUGH_SCHEDULE_AUTO_TOOLS: tuple[str, ...] = (
    # One unified tool that internally routes to briefing/reminder/deadline/
    # watch + list/cancel — so a single schema entry carries the whole
    # scheduling capability instead of seven competing ones.
    "schedule",
)


@router.get("/passthrough-tools")
async def get_passthrough_tools(request: Request) -> JSONResponse:
    """List tools available for passthrough mode enablement."""
    registry = getattr(request.app.state, "tool_registry", None)
    if not registry:
        return JSONResponse({"tools": [], "defaults": []})

    tools = []
    for tool in registry.list_tools():
        if tool.name in _PASSTHROUGH_HIDDEN:
            continue
        healthy = True
        try:
            healthy = tool.health_check()
        except Exception:
            healthy = False
        tools.append({
            "name": tool.name,
            "description": tool.description,
            "category": tool.category.value,
            "healthy": healthy,
            "requires": getattr(tool, "requires_services", []) or [],
        })

    # Current config defaults
    defaults = []
    if settings.passthrough_tools.strip():
        defaults = [t.strip() for t in settings.passthrough_tools.split(",") if t.strip()]

    return JSONResponse({"tools": tools, "defaults": defaults})


# Narrative background-model choices are PER-USER preferences (migration
# 308): stored in user_settings keyed by the requesting user, resolved
# user-override → install default. One user's pick (local vs API vs a
# fabric peer) must never silently change another user's narrative calls.
_USER_SCOPED_TOOL_KEYS: frozenset[str] = frozenset({
    "narrative_memory_model",
    "narrative_extraction_model",
    "narrative_auto_bg_distiller_model",
    "narrative_auto_bg_image_model",
})


@router.get("/tools")
async def get_tool_settings(request: Request) -> dict:
    """Get current tool/search settings (sensitive values redacted).

    Tri-state bool keys (see ``_TRI_STATE_BOOL_SETTINGS``) include a
    companion ``<key>_resolved`` field showing the bool the runtime
    actually behaves as — useful for UI labels like
    "Auto (currently: enabled)" without the frontend having to know
    the resolution logic.

    Keys in ``_USER_SCOPED_TOOL_KEYS`` return the requesting user's own
    value (install default when the user has no override).
    """
    result = {key: getattr(settings, key) for key in _TOOL_SETTINGS}
    for key in _STRING_SETTINGS:
        val = getattr(settings, key)
        # Redact sensitive fields — match the same pattern as to_safe_dict()
        if ("token" in key.lower() or "key" in key.lower() or "secret" in key.lower()) and val:
            result[key] = "***"
        else:
            result[key] = val
    for key, resolver in _TRI_STATE_BOOL_SETTINGS.items():
        result[f"{key}_resolved"] = resolver(getattr(settings, key, None))
    store = getattr(request.app.state, "settings_store", None)
    uid = getattr(request.scope.get("user"), "id", "") or ""
    if store is not None and uid:
        for key in _USER_SCOPED_TOOL_KEYS:
            try:
                val = await store.get_user(uid, key)
            except Exception:
                log.warning("user_scoped_setting_read_failed", key=key, exc_info=True)
                val = None
            if val is not None:
                result[key] = val
    # Read-only capability flag (not a setting — env-driven, see
    # config.selfedit_unlocked). The UI hides the self-edit switch and the
    # Workshop nav entry unless this is true, so a default install never
    # surfaces the subsystem at all.
    from augmentum.config import selfedit_unlocked

    result["selfedit_unlocked"] = selfedit_unlocked()
    return result


@router.put("/tools")
async def update_tool_settings(request: Request) -> JSONResponse:
    """Update tool/search settings at runtime and persist to DB.

    Admin only — the keys in ``_TOOL_SETTINGS`` and ``_STRING_SETTINGS``
    are install-wide (rate limits, auth thresholds, model routing roles,
    server API keys). Per-user preferences live under ``/api/config/ui``
    and ``/api/config/personalization`` instead — EXCEPT the keys in
    ``_USER_SCOPED_TOOL_KEYS``, which write the requesting user's own
    override and therefore don't require admin when the request contains
    only those keys.
    """
    body = await request.json()
    user_scoped_only = (
        isinstance(body, dict)
        and bool(body)
        and all(k in _USER_SCOPED_TOOL_KEYS for k in body)
    )
    if not user_scoped_only and (forbidden := require_admin(request)) is not None:
        return forbidden
    store = getattr(request.app.state, "settings_store", None)
    uid = getattr(request.scope.get("user"), "id", "") or ""

    updated: dict[str, object] = {}
    errors: list[str] = []

    # Self-edit can't be switched on through the API on a locked install. The UI
    # already hides the control, but hiding is not a gate — refuse the write so
    # a hand-rolled PUT can't leave the stored flag on. See
    # config.selfedit_unlocked() for why this one is env-gated.
    if any(k.startswith("selfedit_") for k in body):
        from augmentum.config import selfedit_unlocked

        if not selfedit_unlocked():
            return JSONResponse(
                {
                    "error": (
                        "self-edit is not available on this install "
                        "(set AUGMENTUM_SELFEDIT_UNLOCK to enable it)"
                    )
                },
                status_code=403,
            )

    for key, value in body.items():
        if key in _USER_SCOPED_TOOL_KEYS:
            coerced = str(value)[: _STRING_SETTINGS.get(key, 256)]
            if store is None or not uid:
                errors.append(f"{key}: per-user setting requires a signed-in user")
                continue
            await store.set_user(uid, key, coerced)
            updated[key] = coerced
            continue

        if key in _STRING_SETTINGS:
            coerced = str(value)[:_STRING_SETTINGS[key]]
            is_sensitive = "token" in key.lower() or "key" in key.lower() or "secret" in key.lower()
            # Don't overwrite tokens with the redacted placeholder "***"
            if is_sensitive and coerced.strip("*") == "":
                continue
            # Scheme guard on URL-bearing settings. anthropic_base_url /
            # azure_base_url and friends are admin-configurable; LAN
            # targets stay allowed (Azure private endpoint, on-prem
            # Anthropic mirror) but file://, gopher:// and other SSRF
            # amplifiers don't reach the backend factory.
            if key.endswith("_base_url") and coerced:
                from augmentum.utils.safe_http import SafeHttpError, validate_provider_url
                try:
                    coerced = validate_provider_url(coerced)
                except SafeHttpError as exc:
                    errors.append(f"{key}: {exc}")
                    continue
            object.__setattr__(settings, key, coerced)
            updated[key] = "***" if is_sensitive and coerced else coerced
            if store:
                # Encrypt sensitive values at rest
                if is_sensitive and coerced:
                    from augmentum.utils.secrets import encrypt_api_key
                    await store.set(key, encrypt_api_key(coerced))
                else:
                    await store.set(key, coerced)
            # Propagate HF token to env var for in-process model downloads
            if key == "huggingface_token" and coerced:
                os.environ["HF_TOKEN"] = coerced
                if not settings.image_huggingface_token:
                    object.__setattr__(settings, "image_huggingface_token", coerced)
            continue

        if key not in _TOOL_SETTINGS:
            errors.append(f"Unknown setting: {key}")
            continue

        # Tri-state bool keys: ``None``/``"auto"``/``""`` → delete the
        # override (revert runtime to None so the codebase
        # recommendation applies on this and future restarts).
        # ``True``/``False`` (or string equivalents) → persist as
        # explicit override.
        if key in _TRI_STATE_BOOL_SETTINGS:
            normalized = value
            if isinstance(normalized, str):
                lower = normalized.strip().lower()
                if lower in ("", "none", "null", "auto"):
                    normalized = None
                elif lower in ("true", "1", "yes"):
                    normalized = True
                elif lower in ("false", "0", "no"):
                    normalized = False
                else:
                    errors.append(f"{key}: invalid tri-state value {value!r}")
                    continue

            if normalized is None:
                object.__setattr__(settings, key, None)
                updated[key] = None
                if store:
                    # SettingsStore.set(key, None) deletes the row —
                    # next restart will see no override and the
                    # codebase recommendation applies.
                    await store.set(key, None)
                continue

            if not isinstance(normalized, bool):
                errors.append(f"{key}: tri-state value must be bool|null, got {value!r}")
                continue

            object.__setattr__(settings, key, normalized)
            updated[key] = normalized
            if store:
                await store.set(key, str(normalized))
            continue

        cast_fn, min_val, max_val = _TOOL_SETTINGS[key]

        try:
            if cast_fn is bool:
                coerced = value if isinstance(value, bool) else str(value).lower() in ("true", "1", "yes")
            else:
                coerced = cast_fn(value)
                if coerced < min_val or coerced > max_val:
                    errors.append(f"{key}: value {coerced} out of range [{min_val}, {max_val}]")
                    continue
        except (ValueError, TypeError):
            errors.append(f"{key}: invalid value {value!r}")
            continue

        # Apply to runtime config
        object.__setattr__(settings, key, coerced)
        updated[key] = coerced

        # Persist to DB
        if store:
            await store.set(key, str(coerced))

    _propagate_kv_settings(request, updated)

    log.info("tool_settings_updated", updated=updated, errors=errors)

    current = {key: getattr(settings, key) for key in _TOOL_SETTINGS}
    for key in _STRING_SETTINGS:
        current[key] = getattr(settings, key)
    result = {"updated": updated, "current": current}
    if errors:
        result["errors"] = errors
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# UI-only settings (frontend preferences persisted server-side)
# ---------------------------------------------------------------------------
# These are stored directly in the settings_store key-value table with a
# "ui." prefix.  They have no corresponding Python config field — the backend
# just round-trips them so that the frontend can retrieve them on any device.

# Allowed UI setting keys and their max value length.
_UI_SETTINGS: dict[str, int] = {
    "systemPrompt": 8000,
    "temperature": 32,
    "topP": 32,
    "maxTokens": 32,
    "stopSequences": 1000,
    "personalizationEnabled": 8,
    "aiName": 256,
    "aiInstructions": 32000,           # bumped from 8000 — long hand-written personas; ~8K tokens, injected verbatim into <persona> every personalized turn
    "responseStyle": 64,
    "personalizeAnalytical": 8,
    "personalizeAgentic": 8,
    "personalityPresets": 200_000,     # JSON array of user-saved persona presets ({id, name, instructions}). 20 presets × ~5KB each + headroom.
    "voiceAutoRead": 8,
    "voiceSpeed": 16,
    "voiceDefaultVoice": 256,
    "companionVoice": 256,             # companion (Becca) TTS voice — provider::voice or bare name; empty = fall back to voiceDefaultVoice
    "companionVoiceVolume": 16,        # companion TTS output gain multiplier (e.g. "2.0"). Boosts her soft voice over music/host-mode media; applied as a Web-Audio gain stage in chat/tts.js
    "readerTtsVoice": 256,             # preferred voice for the EPUB/document read-aloud bar
    "readerTtsSpeed": 16,              # playback speed for the reader bar (independent of voiceSpeed)
    # Comic read-aloud Voice Casts — the reusable utility. A cast is a named set
    # of 5 register-bucket voices (m_low/m_high/f_low/f_high/narrator); the
    # library persists server-side so casts follow the user across devices, and
    # the active cast is the cross-session default the reader/TV narrate with.
    # JSON: [{id, name, slots:{...}}]. ~20 casts × 5 × 256 + overhead.
    "comicVoiceCasts": 64_000,
    "comicVoiceCastActive": 64,       # id of the active cast in the library
    "ttsIncludeActionText": 8,
    "browseDefaultSplit": 8,
    "browseNotesHistoryCollapsed": 8,
    "browseLinkOpenMode": 32,
    "browseReaderSize": 16,
    "browseReaderFamily": 32,
    "browseReaderHeight": 16,
    "browseReaderWidth": 16,
    "browseReaderJustify": 8,
    "notesDefaultFormat": 16,
    "thinkEnabled": 8,
    "preserveThinking": 8,
    "autoTools": 8,                    # bool: SSOS heuristic intent path in passthrough
    # Tool-call round-trips allowed per turn. "" = use the install default,
    # "0" = unlimited (bounded by the 150 backstop in passthrough/handler.py).
    # User-set because the right value is model-dependent: a frontier model
    # with a long context can chain far more than a local 12B.
    "toolChainLimit": 8,
    "typographyPreset": 256,
    "typographyCustomFonts": 32_000,   # JSON blob of custom font presets
    "typographyTextSize": 16,
    "typographyTextColors": 512,       # JSON blob of {primary, secondary, muted}
    "softTypography": 8,               # bool: drop uppercase + letter-spacing on chrome labels
    "recentModels": 2000,              # JSON array of recently used model names
    "engineModelLoadProfiles": 64_000, # JSON map of per-model built-in engine load defaults
    # Connect (peer-to-peer calls + text). All three are per-user;
    # discoverability defaults off (privacy) and surface on by default.
    # See docs/superpowers/specs/2026-06-01-connect-and-os-positioning-design.md
    "connectEnabled": 8,
    "connectDiscoverableSameInstance": 8,
    "connectDiscoverableFabricPeers": 8,
    # Dream system
    "dreamEnabled": 8,
    "dreamMessageThreshold": 8,
    "dreamIdleMinutes": 8,
    "dreamCooldownMinutes": 8,
    "dreamMaxJournalEntries": 8,
    "dreamCompactionAgeDays": 8,
    "dreamPortraitMaxTokens": 8,
    "dreamRecallEnabled": 8,
    "dreamRecallLimit": 8,
    "dreamModel": 256,
    # Onboarding
    "onboarding_completed": 8,
    # One-time consent for including NSFW results in online character search
    # (chub.ai / risurealm). "true" once the user has confirmed the gate.
    "nsfw_search_consent": 8,
    # Orb navigation customization
    "orbCustomOrder": 512,     # JSON array of orb IDs
    "orbCustomColors": 1024,   # JSON object: {colorKey: "#hex"}
    # Workspace layout persistence
    "workspace": 8000,         # JSON: surfaces array + focused + layout
    # XR / VR seat calibration — JSON {x, y, z, rotY, envId} for the user's
    # virtual seat position when entering immersive mode from voice call.
    "xrSeatLayout": 256,
    "gameStreamInputPrefs": 8192,
    # Media console rail visibility — JSON array of rail slugs the user
    # hid in-app (mirrors the per-receiver rails_visible prefs the cast
    # surfaces use, but per-user instead of per-TV).
    "mediaRailsHidden": 256,
    # Media console rail ORDER — JSON array of rail slugs in the user's
    # chosen display order (in-app, per-user). Presentation-only: applied
    # client-side over the catalog default; slugs absent from the list fall
    # to the end in catalog order, stale slugs are ignored.
    "mediaRailsOrder": 512,
    # Comic/manga reader preferences (per-user, server-synced so paged/webtoon,
    # fit, direction, background, crop + auto-scroll follow the user across
    # devices; localStorage is the offline cache). Global defaults + a per-series
    # override map + a per-file reading-direction map.
    "comicReaderPrefs": 512,
    # Maps keyed by series_id / file_id — generous caps so a large library's
    # JSON map isn't truncated mid-string (truncation corrupts it; the client
    # then falls back to its localStorage cache for that key).
    "comicReaderSeriesPrefs": 65536,
    "comicReaderDirPrefs": 65536,
    # Comic-cast music bed — the background music source the user last chose to
    # play UNDER a voiced comic on the TV (cast-comic). JSON descriptor
    # {kind,id,name,genre,url,poster,source} from the shared music-source layer,
    # so the same bed is restored on the next cast. Per-user state (not an
    # install default). 2KB holds a station/file descriptor with headroom.
    "comicCastBed": 2048,
}

# Keys that record per-user UI *state* (what this user arranged / used),
# NOT install-wide defaults. These are read per-user ONLY — they must never
# fall back to a global ``app_settings`` row, or one user's saved layout
# (e.g. a pre-Stage-D owner workspace) leaks into every new account. Compare
# the ``DENYLIST`` in ``augmentum/selfedit/adaptables.py`` (same blobs, kept
# out of the Adapt surface for the same "personal-not-tunable-default" reason).
_UI_PERSONAL_STATE_KEYS: frozenset[str] = frozenset({
    "workspace",            # window/surface layout (the coder+chat leak)
    "orbCustomOrder",       # personal orb arrangement
    "orbCustomColors",      # personal orb colors
    "recentModels",         # this user's recently used models
    "typographyCustomFonts",  # this user's uploaded font presets
    "typographyTextColors",   # this user's text-color overrides
    "xrSeatLayout",         # this user's VR seat calibration
    "gameStreamInputPrefs",  # this user's controller mapping
    "mediaRailsHidden",     # this user's hidden media rails
    "mediaRailsOrder",      # this user's media rail display order
    "comicReaderPrefs",         # this user's global comic-reader defaults
    "comicReaderSeriesPrefs",   # this user's per-series comic-reader overrides
    "comicReaderDirPrefs",      # this user's per-file reading direction
    "comicCastBed",             # this user's last comic-cast music bed choice
    "nsfw_search_consent",      # this user's own NSFW-search consent (never inherited)
    # Identity keys (2026-07-18, closing the audited leak): what the
    # assistant is CALLED, how it's told to behave, and which voice it
    # speaks with are personal state, not install defaults — a new
    # account must never inherit the owner's companion identity
    # ("Alethia") or voice. Ground rules: per-user isolation is hard,
    # voices are never auto-selected, and OSS installs stay
    # persona-agnostic. Owners whose persona predates per-user settings
    # (global row only) re-save once; stale globals are ignored, same
    # as the workspace-layout precedent above.
    "aiName",
    "aiInstructions",
    "responseStyle",
    "voiceDefaultVoice",
    "companionVoice",
})


@router.get("/ui")
async def get_ui_settings(request: Request) -> JSONResponse:
    """Get the caller's UI preferences, falling back to the install default.

    Per-user keys live in ``user_settings(user_id, ui.<key>)`` since Stage D;
    if the caller hasn't saved anything yet, the global ``app_settings``
    value acts as a sensible default.

    Exception: keys in ``_UI_PERSONAL_STATE_KEYS`` record what *this* user did
    (window layout, orb arrangement, recent models, custom fonts, seat
    position…). They are NOT install defaults, so they must be read per-user
    ONLY — never falling back to a global ``app_settings`` row. Otherwise a
    pre-Stage-D global value (e.g. an owner's old coder+chat workspace) bleeds
    into every brand-new account on first login. See the multi-tenant
    pref-leak class.
    """
    store = getattr(request.app.state, "settings_store", None)
    if not store:
        return JSONResponse({})
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    result: dict[str, str] = {}
    for key in _UI_SETTINGS:
        if key in _UI_PERSONAL_STATE_KEYS:
            val = await store.get_user(uid, f"ui.{key}")
        else:
            val = await store.get_user_or_global(uid, f"ui.{key}")
        if val is not None:
            result[key] = val
    return JSONResponse(result)


async def _reconcile_dream_lifecycle(request: Request) -> None:
    """Boot or tear down the dream subsystem to match current settings.

    The dream subsystem is a process singleton with user-scoped data.
    It must be alive whenever *any* tenant (or the install-wide default)
    has ``ui.dreamEnabled = true``. Both the ``/ui`` and
    ``/personalization`` endpoints persist the toggle, so both must
    reconcile — otherwise the box can be checked but the scheduler
    stays cold, which is exactly the symptom that triggered this
    refactor.

    Idempotent: ``setup_dream_system`` / ``teardown_dream_system`` each
    no-op when the system is already in the target state.
    """
    store = getattr(request.app.state, "settings_store", None)
    if store is None:
        return
    from augmentum.dream.lifecycle import (
        setup_dream_system,
        should_dream_run,
        teardown_dream_system,
    )
    wanted = await should_dream_run(store)
    running = getattr(request.app.state, "dream_scheduler", None) is not None
    if wanted and not running:
        await setup_dream_system(request.app)
    elif running and not wanted:
        await teardown_dream_system(request.app)


@router.put("/ui")
async def update_ui_settings(request: Request) -> JSONResponse:
    """Persist the caller's UI preferences (per-user, no cross-tenant bleed).

    Side effect: writes to ``ui.dreamEnabled`` trigger a reconciliation
    of the process-level dream subsystem. ``/personalization`` does the
    same — either endpoint can be the entry point depending on the UI
    flow, so both converge on the same helper.
    """
    body = await request.json()
    store = getattr(request.app.state, "settings_store", None)
    if not store:
        return JSONResponse({"error": "Settings store unavailable"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    updated: dict[str, str] = {}
    for key, value in body.items():
        if key not in _UI_SETTINGS:
            continue
        coerced = str(value) if value is not None else ""
        coerced = coerced[:_UI_SETTINGS[key]]
        await store.set_user(uid, f"ui.{key}", coerced)
        updated[key] = coerced

    if "dreamEnabled" in updated:
        await _reconcile_dream_lifecycle(request)

    return JSONResponse({"updated": updated})


# ---------------------------------------------------------------------------
# Personalization (subset of UI settings used by the personalization system)
# ---------------------------------------------------------------------------

_PERSONALIZATION_KEYS: frozenset[str] = frozenset({
    "personalizationEnabled", "aiName", "aiInstructions",
    "responseStyle", "personalizeAnalytical", "personalizeAgentic",
    "personalityPresets",
    "dreamEnabled", "dreamMessageThreshold", "dreamIdleMinutes",
    "dreamCooldownMinutes", "dreamMaxJournalEntries", "dreamCompactionAgeDays",
    "dreamPortraitMaxTokens", "dreamRecallEnabled", "dreamRecallLimit",
    "dreamModel",
})


@router.get("/personalization")
async def get_personalization(request: Request) -> JSONResponse:
    """Get the caller's personalization settings (per-user)."""
    store = getattr(request.app.state, "settings_store", None)
    if not store:
        return JSONResponse({})
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    result: dict[str, str] = {}
    for key in _PERSONALIZATION_KEYS:
        # Same personal-state exception as GET /ui — identity keys must
        # not fall back to another user's (the owner's) global row.
        if key in _UI_PERSONAL_STATE_KEYS:
            val = await store.get_user(uid, f"ui.{key}")
        else:
            val = await store.get_user_or_global(uid, f"ui.{key}")
        if val is not None:
            result[key] = val
    return JSONResponse(result)


@router.put("/personalization")
async def update_personalization(request: Request) -> JSONResponse:
    """Persist the caller's personalization settings (per-user).

    Side effect: toggling ``dreamEnabled`` reconciles the process-level
    dream subsystem after the caller's setting has been written. The
    subsystem is a process singleton with user-scoped data (DreamJournal,
    DreamEngine, DreamScheduler all key by ``user_id`` internally), so
    it should be alive whenever *any* tenant wants it — not only when
    the current caller does. Concretely:

    * If the caller enables dreams, the system boots if it isn't already.
    * If the caller disables dreams, the system tears down only when no
      other tenant (and no install-wide default) still has it on.

    The decision is made from authoritative storage after the write, so
    concurrent toggles from two users converge — no reliance on a diff
    against pre-write state that could race between requests.
    """
    body = await request.json()
    store = getattr(request.app.state, "settings_store", None)
    if not store:
        return JSONResponse({"error": "Settings store unavailable"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    updated: dict[str, str] = {}
    for key, value in body.items():
        if key not in _PERSONALIZATION_KEYS:
            continue
        max_len = _UI_SETTINGS.get(key, 256)
        coerced = str(value) if value is not None else ""
        coerced = coerced[:max_len]
        await store.set_user(uid, f"ui.{key}", coerced)
        updated[key] = coerced

    if "dreamEnabled" in updated:
        await _reconcile_dream_lifecycle(request)

    return JSONResponse({"updated": updated})


@router.get("/voice-prefs/{mode}")
async def get_voice_prefs(mode: str, request: Request):
    """Load the caller's per-mode voice preferences."""
    valid_modes = ("passthrough", "analytical", "narrative", "agentic")
    if mode not in valid_modes:
        return JSONResponse({"error": f"Invalid mode: {mode}"}, status_code=400)
    key = f"voice_prefs_{mode}"
    store = getattr(request.app.state, "settings_store", None)
    if store is None:
        return JSONResponse({"error": "Settings store not initialized"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    raw = await store.get_user_or_global(uid, key)
    if raw:
        import json
        try:
            return JSONResponse(json.loads(raw))
        except Exception:
            log.warning("voice_prefs_parse_failed", key=key, exc_info=True)
    return JSONResponse({
        "avatar_active": False,
        "avatar_expanded": False,
        "stage_active": False,
        "input_mode": "auto",
    })


@router.put("/voice-prefs/{mode}")
async def put_voice_prefs(mode: str, request: Request):
    """Save the caller's per-mode voice preferences."""
    valid_modes = ("passthrough", "analytical", "narrative", "agentic")
    if mode not in valid_modes:
        return JSONResponse({"error": f"Invalid mode: {mode}"}, status_code=400)
    body = await request.json()
    import json
    key = f"voice_prefs_{mode}"
    store = getattr(request.app.state, "settings_store", None)
    if store is None:
        return JSONResponse({"error": "Settings store not initialized"}, status_code=503)
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    await store.set_user(uid, key, json.dumps(body))
    return JSONResponse({"status": "ok"})


@router.get("/kokoro-status")
async def kokoro_status():
    """Return Kokoro TTS runtime status (requested vs actual quality)."""
    try:
        from augmentum.voice.kokoro_tts import KokoroTTS
        return JSONResponse(KokoroTTS.status())
    except Exception:
        return JSONResponse({"loaded": False})


# ---------------------------------------------------------------------------
# Log level — admin-only runtime knob
# ---------------------------------------------------------------------------

@router.get("/log-level")
async def get_log_level_endpoint(request: Request) -> JSONResponse:
    """Return the currently active log level.

    Admin-only because the level reveals install operational posture and
    a non-admin shouldn't be able to confirm whether DEBUG is captured.
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    from augmentum.utils.logging import get_log_level
    return JSONResponse({
        "level": get_log_level(),
        "allowed": ["DEBUG", "INFO", "WARNING", "ERROR"],
    })


@router.put("/log-level")
async def set_log_level_endpoint(request: Request) -> JSONResponse:
    """Change the live log level (admin-only).

    Effect is immediate — applies to all loggers across the process. The
    chosen level is persisted to ``settings_store`` so it survives restart;
    on next boot the persisted value takes precedence over
    ``AUGMENTUM_LOG_LEVEL`` in the env. The change itself is logged at
    WARNING regardless of the new level so it's always visible in the audit
    trail.

    Body: ``{"level": "DEBUG" | "INFO" | "WARNING" | "ERROR"}``
    """
    if (forbidden := require_admin(request)) is not None:
        return forbidden
    body = await request.json()
    requested = (body.get("level") or "").strip()
    if not requested:
        return JSONResponse({"error": "level is required"}, status_code=400)

    from augmentum.utils.logging import get_log_level, set_log_level
    try:
        new_level = set_log_level(requested)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    # Persist so a process restart restores the operator's chosen level.
    store = getattr(request.app.state, "settings_store", None)
    if store is not None:
        try:
            await store.set("ui.logLevel", new_level)
        except Exception:
            log.warning("log_level_persist_failed", level=new_level, exc_info=True)

    user = request.scope.get("user")
    log.warning(
        "log_level_changed",
        new_level=new_level,
        actor_user_id=user.id if user else "",
    )
    return JSONResponse({
        "level": new_level,
        "previous": get_log_level() if get_log_level() != new_level else None,
    })
