"""Voice + TTS settings — full user-tunable surface migrated into
the declarative substrate.
"""

from __future__ import annotations

from augmentum.registry.registry import SettingsRegistry
from augmentum.registry.settings import Setting

_VOICE = ("voice",)
_VOICE_ADV = ("voice", "advanced")


def register(r: SettingsRegistry) -> None:
    # ---- TTS — style / behavior ----
    r.register(
        Setting(
            key="tts_voice_style",
            kind="str",
            default="",
            label="Voice style",
            description=(
                "Default style instruction sent to TTS providers that accept "
                "one (e.g. 'speak warmly and cheerfully'). Blank = neutral."
            ),
            section="voice.tts",
            max_length=256,
            tags=_VOICE,
            voice_aliases=("voice style", "tts style"),
        )
    )
    r.register(
        Setting(
            key="tts_emotion_aware",
            kind="bool",
            default=False,
            label="Emotion-aware TTS",
            description=(
                "Extract emotion cues from chat turns and pass them as the "
                "TTS instruct parameter when the provider supports it."
            ),
            section="voice.tts",
            tags=_VOICE,
            voice_aliases=("emotional voice", "emotion aware"),
        )
    )
    r.register(
        Setting(
            key="tts_kokoro_hbe",
            kind="bool",
            default=True,
            label="Kokoro HBE upsampling",
            description=(
                "Harmonic Bandwidth Extension — resynthesizes Kokoro's 24kHz "
                "output up to 48kHz for crisper highs. Safe to leave on."
            ),
            section="voice.tts.kokoro",
            tags=("voice", "quality"),
        )
    )

    # ---- VAD / turn detection ----
    r.register(
        Setting(
            key="voice_silence_threshold_ms",
            kind="int",
            default=1200,
            label="End-of-speech silence (ms)",
            description=(
                "How long of a silence before the VAD declares end-of-turn. "
                "Lower = snappier but more interrupts; higher = patient."
            ),
            section="voice.detection",
            min_value=400,
            max_value=3000,
            tags=_VOICE,
        )
    )
    r.register(
        Setting(
            key="voice_max_audio_seconds",
            kind="int",
            default=30,
            label="Max single-turn audio (s)",
            description=(
                "Hard cap on recording length per turn. Force-ends the "
                "utterance once exceeded."
            ),
            section="voice.detection",
            min_value=5,
            max_value=120,
            tags=_VOICE,
        )
    )
    r.register(
        Setting(
            key="voice_bargein_min_speech_ms",
            kind="int",
            default=250,
            label="Barge-in minimum (ms)",
            description=(
                "Minimum speech duration before user audio can interrupt the "
                "assistant's TTS playback. Filters cough / breath false starts."
            ),
            section="voice.detection",
            min_value=0,
            max_value=2000,
            tags=_VOICE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="voice_smart_turn",
            kind="bool",
            default=True,
            label="Smart-turn detection",
            description=(
                "Use the smart-turn model (predicts end-of-utterance from "
                "transcript + prosody) instead of pure silence timing."
            ),
            section="voice.detection",
            tags=_VOICE,
        )
    )
    r.register(
        Setting(
            key="voice_smart_turn_threshold",
            kind="float",
            default=0.5,
            label="Smart-turn threshold",
            description=(
                "Probability threshold for 'turn complete' from the smart-turn "
                "model. 0.5 = balanced; higher = more conservative."
            ),
            section="voice.detection",
            min_value=0.1,
            max_value=0.95,
            tags=_VOICE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="voice_smart_turn_max_wait_s",
            kind="float",
            default=3.0,
            label="Smart-turn max wait (s)",
            description=(
                "Hard ceiling on how long smart-turn will wait for the model "
                "to signal turn-complete before falling back to silence timing."
            ),
            section="voice.detection",
            min_value=0.5,
            max_value=30.0,
            tags=_VOICE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="voice_smart_turn_min_veto_confidence",
            kind="float",
            default=0.3,
            label="Smart-turn veto confidence",
            description=(
                "Below this confidence, smart-turn's 'not done yet' signal is "
                "ignored — silence timing wins. Confidence is how far the "
                "completion probability fell below the turn-complete "
                "threshold; near-zero probability is the MOST confident "
                "'still talking'."
            ),
            section="voice.detection",
            min_value=0.0,
            max_value=0.5,
            tags=_VOICE_ADV,
            advanced=True,
        )
    )

    # ---- Audio processing ----
    r.register(
        Setting(
            key="voice_preprocess_bypass",
            kind="bool",
            default=False,
            label="Bypass all mic preprocessing (raw)",
            description=(
                "Send raw microphone audio straight to voice detection and "
                "transcription — no denoise, high-pass, noise suppression or "
                "gain control. Use this to hear what the transcriber actually "
                "gets, then re-enable stages one at a time to find which one "
                "helps on your mic. Overrides every option below."
            ),
            section="voice.audio",
            tags=_VOICE,
        )
    )
    r.register(
        Setting(
            key="voice_denoise_enabled",
            kind="bool",
            default=True,
            label="Denoise input",
            description=(
                "Run noise reduction on mic input before STT. Improves "
                "transcription in noisy rooms; costs a few ms."
            ),
            section="voice.audio",
            tags=_VOICE,
        )
    )
    r.register(
        Setting(
            key="voice_highpass_hz",
            kind="int",
            default=80,
            label="High-pass filter (Hz)",
            description=(
                "Cut frequencies below this from mic input. 80 Hz removes "
                "rumble without losing voice."
            ),
            section="voice.audio",
            min_value=0,
            max_value=1000,
            tags=_VOICE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="voice_audio_agc",
            kind="bool",
            default=True,
            label="Automatic gain control",
            description=(
                "Auto-normalize mic input loudness. Helps if you move toward "
                "or away from the mic during a session."
            ),
            section="voice.audio",
            tags=_VOICE,
        )
    )
    r.register(
        Setting(
            key="voice_audio_agc_target_dbfs",
            kind="int",
            default=-16,
            label="AGC target loudness (dBFS)",
            description=(
                "Target loudness for AGC. -16 dBFS is the broadcast standard "
                "(loud but not clipping)."
            ),
            section="voice.audio",
            min_value=-40,
            max_value=0,
            tags=_VOICE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="voice_audio_ns",
            kind="bool",
            default=True,
            label="Noise suppression",
            description=(
                "Aggressive noise suppression (separate from denoise). Strips "
                "fans, HVAC, keyboard noise. May over-process in quiet rooms."
            ),
            section="voice.audio",
            tags=_VOICE,
        )
    )
    r.register(
        Setting(
            key="voice_audio_ns_level",
            kind="int",
            default=2,
            label="Noise suppression level",
            description=(
                "How aggressively noise suppression cuts. 0 = mild, 4 = "
                "maximum. Higher levels can affect speech intelligibility."
            ),
            section="voice.audio",
            min_value=0,
            max_value=4,
            tags=_VOICE_ADV,
            advanced=True,
        )
    )

    # ---- Speaker verification ----
    r.register(
        Setting(
            key="voice_speaker_verify",
            kind="bool",
            default=True,
            label="Speaker verification",
            description=(
                "Verify the active speaker matches an enrolled voice before "
                "accepting commands. Required for multi-user safety."
            ),
            section="voice.speaker_verify",
            tags=_VOICE,
        )
    )
    r.register(
        Setting(
            key="voice_speaker_threshold",
            kind="float",
            default=0.45,
            label="Speaker match threshold",
            description=(
                "Similarity score required to consider an utterance a match. "
                "Higher = stricter (more false rejections); lower = looser."
            ),
            section="voice.speaker_verify",
            min_value=0.2,
            max_value=0.9,
            tags=_VOICE_ADV,
            advanced=True,
        )
    )
    r.register(
        Setting(
            key="voice_speaker_verify_seconds",
            kind="float",
            default=3.0,
            label="Speaker verify window (s)",
            description=(
                "How many seconds of audio the speaker verifier consumes "
                "before locking the speaker for the rest of the turn."
            ),
            section="voice.speaker_verify",
            min_value=1.0,
            max_value=10.0,
            tags=_VOICE_ADV,
            advanced=True,
        )
    )

    # ---- Lipsync ----
    r.register(
        Setting(
            key="voice_lipsync_universal",
            kind="bool",
            default=False,
            label="Universal lipsync",
            description=(
                "Enable lipsync metadata for ALL TTS outputs (not just avatar "
                "scenes). Costs a small amount of CPU per generation."
            ),
            section="voice.lipsync",
            tags=_VOICE,
        )
    )
    r.register(
        Setting(
            key="voice_lipsync_engine",
            kind="enum",
            default="amplitude",
            label="Lipsync engine",
            description=(
                "Which lipsync engine generates viseme timing. 'amplitude' = "
                "fast envelope-based; alternative engines may give more "
                "accurate phoneme alignment."
            ),
            section="voice.lipsync",
            enum_values=("amplitude",),
            max_length=16,
            tags=_VOICE_ADV,
            advanced=True,
        )
    )

    # ---- XR / proxemics ----
    r.register(
        Setting(
            key="voice_xr_proxemics_enabled",
            kind="bool",
            default=False,
            label="XR proxemics",
            description=(
                "In XR mode, modulate voice volume based on the avatar's "
                "spatial distance from the listener."
            ),
            section="voice.xr",
            tags=_VOICE_ADV,
            advanced=True,
        )
    )

    # ---- Routing / pipeline ----
    r.register(
        Setting(
            key="voice_routing_mode",
            kind="enum",
            default="auto",
            label="Voice routing mode",
            description=(
                "How voice routes pick TTS providers. 'auto' = round-robin / "
                "policy-driven; 'pin' = always use voice_routing_pin_provider."
            ),
            section="voice.routing",
            enum_values=("auto", "pin"),
            max_length=16,
            tags=_VOICE,
        )
    )
    r.register(
        Setting(
            key="voice_routing_pin_provider",
            kind="str",
            default="",
            label="Pinned TTS provider",
            description=(
                "When routing_mode='pin', the provider ID to use exclusively. "
                "Empty = none pinned (falls back to auto)."
            ),
            section="voice.routing",
            max_length=256,
            tags=_VOICE,
        )
    )
    r.register(
        Setting(
            key="voice_tts_chunking",
            kind="enum",
            default="sentence",
            label="TTS chunking",
            description=(
                "How speech is split for synthesis. 'sentence' starts with "
                "quick clause-sized chunks then whole sentences (and "
                "auto-upgrades to 'smooth' on fast local providers like "
                "Pocket); 'smooth' = whole sentences always, best prosody; "
                "'clause' = lowest latency, choppiest; 'paragraph' batches "
                "several sentences; 'full' waits for the complete reply."
            ),
            section="voice.routing",
            enum_values=("sentence", "smooth", "clause", "paragraph", "full"),
            max_length=16,
            tags=_VOICE_ADV,
            advanced=True,
        )
    )

    r.register(
        Setting(
            key="voice_tts_lexicon",
            kind="str",
            default="",
            label="TTS pronunciation lexicon",
            description=(
                "JSON object mapping terms to how they should be spoken, "
                'e.g. {"SQL": "sequel", "kubectl": "kube control"}. An '
                "empty value shields a term from all automatic expansion "
                '({"mm": ""} keeps a thoughtful \'mm\' from becoming '
                "'millimeters'). Matched on word boundaries, any casing."
            ),
            section="voice.routing",
            max_length=4000,
            tags=_VOICE_ADV,
            advanced=True,
        )
    )

    # ---- Per-surface pipeline mode ----
    _pipeline_enum = ("auto", "local", "server", "custom")
    for surface in ("call", "companion", "narration", "readaloud"):
        default = "server" if surface == "narration" else "auto"
        r.register(
            Setting(
                key=f"voice_pipeline_mode_{surface}",
                kind="enum",
                default=default,
                label=f"Pipeline mode — {surface}",
                description=(
                    f"Voice pipeline mode for the '{surface}' surface. "
                    f"'auto' picks based on capability; 'local' forces in-browser; "
                    f"'server' forces server-side; 'custom' lets the surface override."
                ),
                section="voice.pipeline",
                enum_values=_pipeline_enum,
                max_length=8,
                tags=_VOICE_ADV,
                advanced=True,
            )
        )
