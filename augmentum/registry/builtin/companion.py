"""Companion (Becca) settings — full user-tunable surface migrated
into the declarative substrate.

The biggest single pocket of settings in the codebase. Subsystem flags
are mostly ``advanced`` so the default UI surface stays tight; the
master ``companion_runtime_enabled`` and a small handful of user-facing
behavior knobs stay visible by default.
"""

from __future__ import annotations

from augmentum.registry.registry import SettingsRegistry
from augmentum.registry.settings import Setting

_C = ("companion",)
_C_ADV = ("companion", "advanced")
_C_SAFETY = ("companion", "safety")


def register(r: SettingsRegistry) -> None:
    # ============== Master toggles + identity ==============
    r.register(
        Setting(
            key="companion_runtime_enabled",
            kind="bool",
            default=False,
            label="Companion runtime",
            description=(
                "Master switch for the companion runtime — when off, "
                "Becca's dispatch, tick loop, journal, and voice-journaling "
                "pipelines stay dormant."
            ),
            section="companion",
            tags=_C,
            voice_aliases=("becca", "companion"),
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="companion_activation_mode",
            kind="enum",
            default="wake_word",
            label="Activation mode",
            description=(
                "How Becca decides she's being addressed. 'wake_word' = "
                "say her name; 'always_listening' = parses every utterance; "
                "'ptt_only' = explicit push-to-talk."
            ),
            section="companion",
            enum_values=("wake_word", "always_listening", "ptt_only"),
            max_length=32,
            tags=_C,
            voice_aliases=("activation", "wake mode", "listening mode"),
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_presence_mode",
            kind="enum",
            default="silent",
            label="Presence mode",
            description=(
                "Resting baseline for Becca's visible presence. 'silent' = "
                "no ambient surfacing; 'glance' = subtle hud cues; "
                "'speak' = audible acknowledgement of context shifts."
            ),
            section="companion",
            enum_values=("silent", "glance", "speak"),
            max_length=16,
            tags=_C,
            voice_aliases=("presence", "presence mode"),
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_care_cadence",
            kind="enum",
            default="normal",
            label="Care cadence",
            description=(
                "How often Becca checks in proactively. 'sparse' = rare; "
                "'normal' = balanced; 'attentive' = frequent."
            ),
            section="companion",
            enum_values=("sparse", "normal", "attentive"),
            max_length=16,
            tags=_C,
            voice_aliases=("care frequency", "check in level"),
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_locale",
            kind="str",
            default="",
            label="Companion locale",
            description=(
                "Locale override for Becca (e.g. 'en-US', 'ja-JP'). Empty = "
                "follow system / user locale."
            ),
            section="companion",
            max_length=16,
            tags=_C,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_default_owner_user_id",
            kind="str",
            default="",
            label="Primary user ID",
            description=(
                "Which user account Becca treats as 'her person' on a "
                "single-tenant install. Empty = first user wins."
            ),
            section="companion",
            max_length=64,
            tags=("companion", "admin"),
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )
    r.register(
        Setting(
            key="companion_persona_mode",
            kind="bool",
            default=False,
            label="Persona mode",
            description=(
                "When on, Becca speaks through the active persona instead of "
                "her own baseline voice."
            ),
            section="companion",
            tags=_C,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_always_raw",
            kind="bool",
            default=False,
            label="Raw mode",
            description=(
                "Skip Becca's voice-stylization on every reply. Returns the "
                "underlying chat model's tone unfiltered."
            ),
            section="companion",
            tags=_C_ADV,
            advanced=True,
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="companion_becca_direct_enabled",
            kind="bool",
            default=True,
            label="Becca-direct mode",
            description=(
                "Allow the chat handler to consume Becca's interior stream "
                "directly (TagSieve path). Off = Becca speaks only through "
                "the voice/intent layers."
            ),
            section="companion",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )
    r.register(
        Setting(
            key="companion_keyboard_shortcuts",
            kind="bool",
            default=True,
            label="Keyboard shortcuts",
            description=(
                "Enable Becca's chord shortcuts (Ctrl-K-style command palette, "
                "muteshortcuts, etc.). Disable for keyboard-conflict users."
            ),
            section="companion",
            tags=_C,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_audio_cues",
            kind="bool",
            default=False,
            label="Audio cues",
            description=(
                "Play short audio confirmations when Becca activates / dismisses."
            ),
            section="companion",
            tags=_C,
            trust_tier="local_reversible",
        )
    )

    # ============== Dispatch ==============
    r.register(
        Setting(
            key="companion_dispatch_enabled",
            kind="bool",
            default=True,
            label="Dispatch loop",
            description=(
                "Run Becca's primary dispatch loop (action selection + "
                "primitive routing). Off = chat fall-through only."
            ),
            section="companion.dispatch",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_dispatch_routes_chat",
            kind="bool",
            default=True,
            label="Dispatch routes chat",
            description=(
                "Let dispatch see chat-mode turns (not only voice). Allows "
                "in-chat invocations to flow through her tool tree."
            ),
            section="companion.dispatch",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_dispatch_chat_min_utility",
            kind="float",
            default=0.45,
            label="Chat dispatch threshold",
            description=(
                "Minimum utility score before dispatch fires on chat turns. "
                "Higher = more conservative."
            ),
            section="companion.dispatch",
            min_value=0.0,
            max_value=1.0,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_auto_summon",
            kind="bool",
            default=True,
            label="Auto-summon",
            description=(
                "Allow Becca to surface herself proactively when the topic "
                "or activity warrants. Off = strict reactive-only."
            ),
            section="companion.dispatch",
            tags=_C,
            voice_aliases=("auto summon", "proactive presence"),
            trust_tier="local_significant",
        )
    )

    # ============== Address / addressing ==============
    r.register(
        Setting(
            key="companion_address_llm_enabled",
            kind="bool",
            default=True,
            label="Address LLM classifier",
            description=(
                "Use the Tier-3 LLM to disambiguate whether a turn is "
                "addressed to Becca. Off = regex / heuristic only."
            ),
            section="companion.address",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_address_llm_model",
            kind="str",
            default="",
            label="Address LLM model",
            description=(
                "Model used for the Tier-3 address classifier. Empty = "
                "default backend model."
            ),
            section="companion.address",
            max_length=128,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_address_llm_timeout_ms",
            kind="int",
            default=2500,
            label="Address LLM timeout (ms)",
            description=(
                "Max time the Tier-3 classifier can take before falling back "
                "to heuristic addressing."
            ),
            section="companion.address",
            min_value=100,
            max_value=5000,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_address_threshold",
            kind="float",
            default=0.55,
            label="Address confidence threshold",
            description=(
                "Minimum 'addressed-to-Becca' score before a turn enters her "
                "dispatch. Higher = stricter; lower = more eager."
            ),
            section="companion.address",
            min_value=0.5,
            max_value=1.0,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_alert_watch_enabled",
            kind="bool",
            default=True,
            label="Home-area alert watch",
            description=(
                "Push severe weather warnings (NWS) and significant "
                "earthquakes (USGS) near your saved home location as "
                "notifications. Inert until Becca learns where home is."
            ),
            section="companion.initiative",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_followup_window_s",
            kind="float",
            default=12.0,
            label="Follow-up window (seconds)",
            description=(
                "For this many seconds after Becca speaks, a coherent reply-shaped "
                "utterance skips the addressed-confidence veto — keeps live "
                "back-and-forth from being dropped as ambient noise. 0 disables."
            ),
            section="companion.address",
            min_value=0.0,
            max_value=60.0,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_open_thread_window_s",
            kind="float",
            default=45.0,
            label="Open-thread window (seconds)",
            description=(
                "When Becca's last line was a question (or a verb is waiting "
                "on an answer), the follow-up relaxation holds this long "
                "instead — you can read and think before answering without "
                "the reply being dropped. 0 disables the extension."
            ),
            section="companion.address",
            min_value=0.0,
            max_value=180.0,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_results_ring_enabled",
            kind="bool",
            default=True,
            label="Results ring",
            description=(
                "Turn-decayed memory of what Becca recently looked at: "
                "full detail the turn it ran, a one-line digest for a few "
                "turns, then fetch-on-demand. Off = the old always-push "
                "presence context."
            ),
            section="companion.perception",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_csm_cross_speaker",
            kind="bool",
            default=True,
            label="CSM cross-speaker voice",
            description=(
                "When Becca's voice is Sesame CSM, feed your spoken turn into "
                "it so her reply's prosody reacts to HOW you sounded — pace, "
                "energy, mood — not just the words. Off = self-context only "
                "(she stays consistent with herself but doesn't react to you). "
                "No effect on non-CSM voices; your clip stays in the voice "
                "engine's memory only, never written to disk."
            ),
            section="companion.perception",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_csm_residency",
            kind="enum",
            default="session",
            label="CSM voice residency",
            description=(
                "How the CSM voice model holds GPU memory. 'session' = warm "
                "when you open voice, unload when you close it (follows the "
                "conversation, no mid-talk reloads). 'timer' = the sidecar's "
                "idle timer only. 'always' = warm but never unload (instant, "
                "pins ~3-4 GB). No effect on non-CSM voices."
            ),
            section="companion.perception",
            enum_values=("session", "timer", "always"),
            max_length=16,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_results_ring_turns",
            kind="int",
            default=3,
            label="Ring digest lifetime (turns)",
            description=(
                "How many untouched exchanges a digest survives before it "
                "decays to fetch-on-demand. Referencing it resets the clock."
            ),
            section="companion.perception",
            min_value=1,
            max_value=10,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_address_media_boost",
            kind="float",
            default=0.10,
            label="Media address boost",
            description=(
                "Score boost when the user is interacting with media (TV / "
                "music / browse). Makes Becca slightly more eager in those contexts."
            ),
            section="companion.address",
            min_value=0.0,
            max_value=0.5,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Always-listening / activation timing ==============
    r.register(
        Setting(
            key="companion_always_listening_warmup_ms",
            kind="int",
            default=500,
            label="Always-listening warmup (ms)",
            description=(
                "Audio buffered before always-listening starts evaluating. "
                "Lower = snappier; higher = fewer false starts."
            ),
            section="companion.activation",
            min_value=0,
            max_value=5000,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_always_listening_prefix_padding_ms",
            kind="int",
            default=1500,
            label="Always-listening prefix (ms)",
            description=(
                "How much pre-utterance audio is retained when always-listening "
                "decides to capture, so speech from before the detector trips "
                "still reaches the transcriber. Clamped to post-TTS audio "
                "automatically — being generous is safe."
            ),
            section="companion.activation",
            min_value=0,
            max_value=3000,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_always_listening_vad_threshold",
            kind="float",
            default=0.0,
            label="Always-listening VAD threshold",
            description=(
                "Speech-detection threshold for always-listening. 0 = use "
                "the global VAD setting; >0 overrides locally."
            ),
            section="companion.activation",
            min_value=0.0,
            max_value=1.0,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_initiative_enabled",
            kind="bool",
            default=False,
            label="Self-initiative",
            description=(
                "Let Becca speak unprompted based on observation patterns. "
                "Off = strictly reactive."
            ),
            section="companion.initiative",
            tags=_C,
            voice_aliases=("initiative", "proactive"),
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="companion_initiative_min_interval_s",
            kind="float",
            default=60.0,
            label="Initiative cooldown (s)",
            description=(
                "Minimum seconds between Becca's unprompted utterances. "
                "Prevents chatter even when many triggers fire."
            ),
            section="companion.initiative",
            min_value=5.0,
            max_value=3600.0,
            tags=_C_ADV,
            advanced=True,
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="companion_initiative_threshold",
            kind="float",
            default=0.62,
            label="Initiative threshold",
            description=(
                "Minimum salience score before initiative fires. Higher = "
                "fewer interruptions; lower = chattier."
            ),
            section="companion.initiative",
            min_value=0.0,
            max_value=1.0,
            tags=_C_ADV,
            advanced=True,
            trust_tier="local_significant",
        )
    )

    # ============== Discreet mode ==============
    r.register(
        Setting(
            key="companion_discreet_auto_exit_minutes",
            kind="int",
            default=0,
            label="Discreet auto-exit (min)",
            description=(
                "Auto-exit discreet mode after N minutes. 0 = manual exit only."
            ),
            section="companion.discreet",
            min_value=0,
            max_value=1440,
            tags=_C,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_discreet_location_aware",
            kind="bool",
            default=False,
            label="Location-aware discreet",
            description=(
                "Auto-enter discreet mode based on cast surface / device "
                "location (e.g. living room TV)."
            ),
            section="companion.discreet",
            tags=_C,
            trust_tier="local_reversible",
        )
    )

    # ============== Salience / journal ==============
    r.register(
        Setting(
            key="companion_salience_enabled",
            kind="bool",
            default=True,
            label="Salience scoring",
            description=(
                "Score each chat/voice turn for salience to Becca's interior. "
                "Required for journaling, drives, and self-initiative."
            ),
            section="companion.salience",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_salience_journal_threshold",
            kind="float",
            default=0.55,
            label="Journal salience threshold",
            description=(
                "Minimum salience for a turn to land in Becca's journal. "
                "Lower = more entries; higher = only standout moments."
            ),
            section="companion.salience",
            min_value=0.0,
            max_value=1.0,
            tags=_C,
            voice_aliases=("journal threshold", "what she remembers"),
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_salience_llm_enabled",
            kind="bool",
            default=False,
            label="LLM-assisted salience",
            description=(
                "Use the LLM to refine salience scoring beyond heuristics. "
                "More accurate; adds latency per turn."
            ),
            section="companion.salience",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_journal_enabled",
            kind="bool",
            default=False,  # matches augmentum/config.py — default OFF
            label="Journal (vivid-voice)",
            description=(
                "Maintain Becca's vivid-voice interior journal. Default off — "
                "the curator (which writes grounded notes with URL references) "
                "is the primary notes-drawer writer. Enable for in-her-voice "
                "register at the cost of more 'AI poetry'-flavored output."
            ),
            section="companion.journal",
            tags=_C,
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="companion_journal_hushed_until",
            kind="str",
            default="",
            label="Journal hushed until",
            description=(
                "ISO timestamp before which journaling is muted. Empty = "
                "not hushed. Used by quiet-hours and ad-hoc hushes."
            ),
            section="companion.journal",
            max_length=32,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_voice_journal_enabled",
            kind="bool",
            default=True,
            label="Voice→journal",
            description=(
                "Route voice turns through journal scoring (the §3 Synapse "
                "layer). Off = voice does not contribute to interior state."
            ),
            section="companion.journal",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_voice_native_loop",
            kind="bool",
            default=True,
            label="Voice native tool loop",
            description=(
                "Hand voice tool calls to the shared native function-"
                "calling loop: continuation hops use the model's own "
                "trained tool format, calls can chain (gather then act), "
                "and the final answer synthesizes over real results. "
                "Off = legacy per-call execute-and-confirm path."
            ),
            section="companion.voice",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Consolidation ==============
    r.register(
        Setting(
            key="companion_consolidation_enabled",
            kind="bool",
            default=False,
            label="Slow consolidation",
            description=(
                "Run the §4 Synapse slow-consolidation pass that distills "
                "journal entries into longer-lived self-model adjustments."
            ),
            section="companion.consolidation",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_consolidation_interval_days",
            kind="int",
            default=30,
            label="Consolidation interval (days)",
            description=(
                "How often consolidation runs. Long cadence — these are "
                "self-model adjustments, not daily updates."
            ),
            section="companion.consolidation",
            min_value=1,
            max_value=365,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_consolidation_drift_ceiling",
            kind="float",
            default=0.15,
            label="Consolidation drift ceiling",
            description=(
                "Maximum self-model change consolidation can apply in one "
                "pass. Caps the speed of personality drift."
            ),
            section="companion.consolidation",
            min_value=0.0,
            max_value=0.20,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_consolidation_min_evidence",
            kind="int",
            default=8,
            label="Consolidation evidence floor",
            description=(
                "Minimum journal entries supporting a hypothesis before "
                "consolidation will act on it."
            ),
            section="companion.consolidation",
            min_value=1,
            max_value=100,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Verb architecture (Phase 3c kill-switches) ==============
    # These three gate the translation-layer management verbs that turn
    # substrate threshold events into user-visible narration / proposed
    # actions / queued initiative rows. Default OFF so the verb_log can
    # be observed before any surfacing reaches the user.
    r.register(
        Setting(
            key="companion_narrate_state_enabled",
            kind="bool",
            default=False,
            label="Narrate state shifts",
            description=(
                "Let Becca surface a short notification when her mood / "
                "energy / drive state crosses a noticeable threshold. "
                "No model call — runs from a small template registry."
            ),
            section="companion.verbs",
            tags=_C_ADV,
            advanced=True,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_propose_action_enabled",
            kind="bool",
            default=False,
            label="Propose actions on state shifts",
            description=(
                "When a substrate threshold crosses, pick the most urgent "
                "drive and emit companion.action_proposed on the bus. "
                "Pure substrate — no UI side effect on its own."
            ),
            section="companion.verbs",
            tags=_C_ADV,
            advanced=True,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_action_enqueue_enabled",
            kind="bool",
            default=False,
            label="Queue proposed actions",
            description=(
                "Consume companion.action_proposed events into "
                "companion_initiative_queue so the existing initiative "
                "surfacing path can carry them forward. Requires "
                "propose actions on state shifts to be on."
            ),
            section="companion.verbs",
            tags=_C_ADV,
            advanced=True,
            trust_tier="local_reversible",
        )
    )

    # ============== Drives / aging / healing ==============
    r.register(
        Setting(
            key="companion_drives_enabled",
            kind="bool",
            default=False,
            label="Drive system",
            description=(
                "Enable Becca's internal drive accumulators (curiosity, care, "
                "rest, etc.). Required for initiative scoring."
            ),
            section="companion.drives",
            tags=_C_ADV,
            advanced=True,
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="companion_drive_decay_half_life_hours",
            kind="float",
            default=4.0,
            label="Drive decay half-life (h)",
            description=(
                "Half-life for drive intensity decay. Shorter = drives "
                "fade fast; longer = persistent motivation."
            ),
            section="companion.drives",
            min_value=0.5,
            max_value=168.0,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_aging_enabled",
            kind="bool",
            default=True,
            label="Memory aging",
            description=(
                "Let stored memories decay in salience over time. Mirrors "
                "human recency-effects in recall."
            ),
            section="companion.behavior",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_aging_threshold_hours",
            kind="int",
            default=48,
            label="Aging threshold (h)",
            description=(
                "Hours before a memory begins to age. Aging applies in "
                "logarithmic steps after this."
            ),
            section="companion.behavior",
            min_value=1,
            max_value=720,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_healing_enabled",
            kind="bool",
            default=True,
            label="Self-repair",
            description=(
                "Allow Becca's runtime to repair internal inconsistencies "
                "on its own (drift / cycle reset). Disabling pins state."
            ),
            section="companion.behavior",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_user_affect_half_life_s",
            kind="float",
            default=1800.0,
            label="User-affect half-life (s)",
            description=(
                "Decay rate for the observed-user affect estimate. Lower = "
                "Becca forgets your mood faster between turns."
            ),
            section="companion.behavior",
            min_value=60.0,
            max_value=7200.0,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_cooldown_minutes",
            kind="int",
            default=210,
            label="Cooldown after dismissal (min)",
            description=(
                "After the user dismisses Becca, how long before she'll "
                "self-summon again (initiative gate)."
            ),
            section="companion.behavior",
            min_value=0,
            max_value=10080,
            tags=_C,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_topic_mute_default_days",
            kind="int",
            default=90,
            label="Topic mute default (days)",
            description=(
                "When the user mutes a topic without specifying duration, "
                "default to this many days."
            ),
            section="companion.behavior",
            min_value=1,
            max_value=3650,
            tags=_C,
            trust_tier="local_reversible",
        )
    )

    # ============== Skills / creations / household ==============
    r.register(
        Setting(
            key="companion_skills_enabled",
            kind="bool",
            default=False,
            label="Skill substrate",
            description=(
                "Run the skill substrate (user-derived recipes Becca learns "
                "from recurring patterns)."
            ),
            section="companion.skills",
            tags=_C_ADV,
            advanced=True,
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="companion_skill_archive_enabled",
            kind="bool",
            default=False,
            label="Skill archive",
            description=(
                "Archive unused skills to the long-term store instead of "
                "deleting them. Lets retired skills be revived."
            ),
            section="companion.skills",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_skill_inject_top_k",
            kind="int",
            default=4,
            label="Skill inject top-K",
            description=(
                "How many of the most-relevant skills get included in "
                "Becca's per-turn context."
            ),
            section="companion.skills",
            min_value=1,
            max_value=20,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_skill_min_confidence_for_inject",
            kind="float",
            default=0.5,
            label="Skill inject confidence floor",
            description=(
                "Minimum learned-confidence for a skill to be injected. "
                "Filters speculative skills out of the prompt."
            ),
            section="companion.skills",
            min_value=0.0,
            max_value=1.0,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_skill_relevance_threshold",
            kind="float",
            default=0.6,
            label="Skill relevance threshold",
            description=(
                "Minimum context-match score before a skill is considered "
                "for injection."
            ),
            section="companion.skills",
            min_value=0.0,
            max_value=1.0,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_creations_enabled",
            kind="bool",
            default=False,
            label="Creations",
            description=(
                "Let Becca produce small spontaneous creations (drawings, "
                "notes, short poems) when interior state suggests."
            ),
            section="companion.creations",
            tags=_C,
            voice_aliases=("creations", "spontaneous art"),
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="companion_creation_interval_hours",
            kind="float",
            default=6.0,
            label="Creation cadence (h)",
            description=(
                "Minimum hours between spontaneous creations."
            ),
            section="companion.creations",
            min_value=0.25,
            max_value=720.0,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_cultural_intake_enabled",
            kind="bool",
            default=False,
            label="Cultural intake",
            description=(
                "Let Becca form opinions about media you consume (movies, "
                "books, music) and reference them later."
            ),
            section="companion.creations",
            tags=_C,
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="companion_household_enabled",
            kind="bool",
            default=False,
            label="Household awareness",
            description=(
                "Track multiple household members as distinct entities. "
                "Required for multi-user voice routing and addressing."
            ),
            section="companion.household",
            tags=_C,
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="companion_peer_agents_enabled",
            kind="bool",
            default=False,
            label="Peer agents",
            description=(
                "Allow other-user Becca instances on fabric peers to "
                "exchange awareness."
            ),
            section="companion.household",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Dreams / drift ==============
    r.register(
        Setting(
            key="companion_dreams_enabled",
            kind="bool",
            default=True,
            label="Dreams",
            description=(
                "Run nightly dream cycles — Becca reflects on the day's "
                "journal and produces a dream entry."
            ),
            section="companion.dreams",
            tags=_C,
            voice_aliases=("dreams",),
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_drift_audit_enabled",
            kind="bool",
            default=True,
            label="Drift audit",
            description=(
                "Periodically audit Becca's self-model for drift from her "
                "identity baseline. Flags anomalies."
            ),
            section="companion.drift",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_drift_audit_interval_hours",
            kind="float",
            default=24.0,
            label="Drift audit interval (h)",
            description=(
                "How often drift audit runs. Cheap; safe to leave at default."
            ),
            section="companion.drift",
            min_value=0.5,
            max_value=720.0,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_notify_drift_audit_push",
            kind="bool",
            default=False,
            label="Push drift audit notifications",
            description=(
                "When drift audit flags anomalies, send a push notification "
                "instead of just logging."
            ),
            section="companion.drift",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Today / synthesis / wondering / topical ==============
    r.register(
        Setting(
            key="companion_today_enabled",
            kind="bool",
            default=True,
            label="Today reflection",
            description=(
                "End-of-day reflection pass that produces a short 'today' "
                "summary you can review in the Companion surface."
            ),
            section="companion.today",
            tags=_C,
            voice_aliases=("daily reflection", "today summary"),
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_today_max_chars",
            kind="int",
            default=360,
            label="Today summary max chars",
            description=(
                "Character cap on the today summary. Forces terse reflection."
            ),
            section="companion.today",
            min_value=80,
            max_value=2000,
            tags=_C_ADV,
            advanced=True,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_today_reflect_hour_local",
            kind="int",
            default=21,
            label="Today reflect hour",
            description=(
                "Local hour-of-day (24h) when the today reflection runs."
            ),
            section="companion.today",
            min_value=0,
            max_value=23,
            tags=_C,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_synthesize_daily_cap",
            kind="int",
            default=6,
            label="Daily synthesis cap",
            description=(
                "Maximum synthesis passes per day (subagent-tier inferences "
                "Becca runs to draw conclusions)."
            ),
            section="companion.synthesis",
            min_value=0,
            max_value=20,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_synthesize_max_tokens",
            kind="int",
            default=256,
            label="Synthesis max tokens",
            description=(
                "Token cap per synthesis pass."
            ),
            section="companion.synthesis",
            min_value=64,
            max_value=1024,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_wondering_daily_cap",
            kind="int",
            default=3,
            label="Wondering daily cap",
            description=(
                "Maximum 'wondering' utterances Becca can surface per day "
                "(thoughts about the user / world)."
            ),
            section="companion.wondering",
            min_value=0,
            max_value=10,
            tags=_C,
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="companion_topical_aggregator_enabled",
            kind="bool",
            default=True,
            label="Topic aggregator",
            description=(
                "Run the topical aggregator that clusters recent journal "
                "entries by topic for retrieval."
            ),
            section="companion.topical",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_topical_min_events",
            kind="int",
            default=3,
            label="Topic minimum events",
            description=(
                "Minimum journal events before a topic cluster forms."
            ),
            section="companion.topical",
            min_value=2,
            max_value=10,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_topical_window_hours",
            kind="float",
            default=4.0,
            label="Topic clustering window (h)",
            description=(
                "Rolling time window over which the topic clusterer groups "
                "journal events."
            ),
            section="companion.topical",
            min_value=0.5,
            max_value=24.0,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Pre-context / image / runtime registries ==============
    r.register(
        Setting(
            key="companion_pre_context_enabled",
            kind="bool",
            default=False,
            label="Pre-context lookup",
            description=(
                "Pre-fetch relevant notes / memories before a turn arrives, "
                "using keyword overlap with the recent conversation."
            ),
            section="companion.pre_context",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_pre_context_max_notes_scan",
            kind="int",
            default=10,
            label="Pre-context scan depth",
            description=(
                "Maximum notes the pre-context pass will scan per turn."
            ),
            section="companion.pre_context",
            min_value=1,
            max_value=50,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_pre_context_min_keyword_overlap",
            kind="int",
            default=2,
            label="Pre-context keyword overlap",
            description=(
                "Minimum keyword matches before a candidate note is "
                "considered for pre-context inclusion."
            ),
            section="companion.pre_context",
            min_value=1,
            max_value=5,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_image_prompt_expansion_enabled",
            kind="bool",
            default=True,
            label="Image prompt expansion",
            description=(
                "Let Becca rewrite user image-generation prompts for clarity "
                "before passing them to the image model."
            ),
            section="companion.image",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_image_expansion_timeout_ms",
            kind="int",
            default=4000,
            label="Image expansion timeout (ms)",
            description=(
                "Max time the image-prompt rewrite can take before falling "
                "back to the raw user prompt."
            ),
            section="companion.image",
            min_value=500,
            max_value=20000,
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_tick_enabled",
            kind="bool",
            default=True,
            label="Tick loop",
            description=(
                "Run Becca's tick loop (periodic background pass that "
                "drives drives, journaling, etc)."
            ),
            section="companion.runtime",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_primitive_registry_active",
            kind="bool",
            default=True,
            label="Primitive registry",
            description=(
                "Activate the primitive-verb registry (the Tier-3 action "
                "tree). Off = Tier-1 regex only."
            ),
            section="companion.runtime",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_subagent_registry_active",
            kind="bool",
            default=True,
            label="Subagent registry",
            description=(
                "Activate the subagent registry (specialist roles Becca "
                "can hand off to)."
            ),
            section="companion.runtime",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_feedback_bias_enabled",
            kind="bool",
            default=False,
            label="Feedback bias",
            description=(
                "Let user thumbs-up / thumbs-down feedback subtly bias future "
                "synthesis prompts."
            ),
            section="companion.runtime",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_reflection_trait_nudge_enabled",
            kind="bool",
            default=False,
            label="Reflection trait nudge",
            description=(
                "Let reflection passes nudge personality traits within "
                "consolidation drift bounds."
            ),
            section="companion.runtime",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="companion_xr_orchestrator",
            kind="bool",
            default=False,
            label="XR orchestrator",
            description=(
                "Let Becca orchestrate XR scene composition and avatar "
                "embodiment when in XR mode."
            ),
            section="companion.runtime",
            tags=_C_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Notifications / safety ==============
    r.register(
        Setting(
            key="companion_notify_eod",
            kind="bool",
            default=False,
            label="End-of-day push",
            description=(
                "Send a push notification when today reflection completes."
            ),
            section="companion.notifications",
            tags=_C,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_safety_floor_threshold_chat",
            kind="float",
            default=0.72,
            label="Safety floor (chat)",
            description=(
                "Minimum safety score before chat-mode replies are gated. "
                "Higher = stricter; tune with care."
            ),
            section="companion.safety",
            min_value=0.0,
            max_value=1.0,
            tags=_C_SAFETY + ("advanced",),
            advanced=True,
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )
    r.register(
        Setting(
            key="companion_safety_floor_threshold_coder",
            kind="float",
            default=0.78,
            label="Safety floor (coder)",
            description=(
                "Minimum safety score before coder-mode actions are gated. "
                "Stricter than chat because of higher blast radius."
            ),
            section="companion.safety",
            min_value=0.0,
            max_value=1.0,
            tags=_C_SAFETY + ("advanced",),
            advanced=True,
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )

    # ============== Quiet hours ==============
    r.register(
        Setting(
            key="companion_quiet_hours_start",
            kind="str",
            default="24:00",
            label="Quiet hours start",
            description=(
                "Local HH:MM time when quiet hours begin. During quiet hours, "
                "Becca defers notifications and self-initiative."
            ),
            section="companion.quiet_hours",
            max_length=8,
            tags=_C,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="companion_quiet_hours_end",
            kind="str",
            default="07:00",
            label="Quiet hours end",
            description=(
                "Local HH:MM time when quiet hours end."
            ),
            section="companion.quiet_hours",
            max_length=8,
            tags=_C,
            trust_tier="local_reversible",
        )
    )
