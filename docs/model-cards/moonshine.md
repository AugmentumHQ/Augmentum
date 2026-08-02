# Moonshine (STT)

> Moonshine AI's on-device speech-to-text toolkit for real-time voice agents —
> does work "while the user is still talking." Augmentum's local STT (the
> network-free alternative to the Deepgram streaming path).

- **Role in Augmentum:** in-process local STT — same interface as the Deepgram `StreamingSTTSession`, so voice routes use either interchangeably. Native partial transcripts; also the batch-fallback that rescues an empty stream.
- **Wired in:** `augmentum/voice/moonshine_stt.py` (`MoonshineSTTSession`). Model loaded once at class level, shared across sessions; a SEPARATE `_batch_transcriber` for the non-streaming rescue path (never touched by the streaming lifecycle). Selected via `voice_stt_model` / `voice_stt_model_arch`.
- **License:** MIT (official).

## Capabilities (official — moonshine-ai/moonshine)
- **Variants (English, params / WER):** Tiny 26M (12.66%), Tiny Streaming 34M (12.00%), Base 58M (10.07%), Small Streaming 123M (7.84%), Medium Streaming 245M (6.65%).
- **Architecture:** C++ core with a C interface, ONNX Runtime inference; post-training int8 weights + 8-bit heavy ops; flexible input windows + caching for streaming.
- **Languages (STT):** English, Spanish, Mandarin, Japanese, Korean, Vietnamese, Ukrainian, Arabic. (The toolkit also ships TTS for a wider set — not used here.)
- **Sample rate:** 16 kHz mono PCM.
- **Latency:** Medium Streaming ~107 ms on a MacBook Pro (vs Whisper Large v3 ~11,286 ms). Our code observes ~150–270 ms on CPU.
- **Function-calling / modalities:** STT only (audio → text).

## Augmentum wiring notes
- `_SAMPLE_RATE = 16000`; feed 16-bit PCM frames via `send_audio`.
- `warmup()` pre-loads the shared model in the background during voice connection startup.

## Gotchas (the paid-for lessons)
- **Non-Latin languages need `max_tokens_per_second = 13.0`** (official) to prevent hallucinated runaway output — this is the same hallucination class our voice route's 3-tier transcript filter guards against (word-rate cap, repetition drop).
- The streaming `_shared_transcriber` can wedge if an `add_audio` worker is abandoned on its 5 s timeout — that's exactly why the batch rescue path uses a separate instance. Don't merge them.
- Speaker identification is "still experimental" upstream — don't rely on it.

## Sources
- Official repo / docs: https://github.com/moonshine-ai/moonshine
- `augmentum/voice/moonshine_stt.py`
