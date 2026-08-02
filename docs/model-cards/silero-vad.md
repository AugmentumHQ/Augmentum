# Silero VAD

> "Enterprise-grade" pre-trained voice activity detector — ~2 MB, <1 ms/chunk
> on CPU, trained on 6000+ languages. Augmentum's server-side speech
> start/end detector, the front of the whole voice pipeline.

- **Role in Augmentum:** server-side VAD — turns the inbound 16 kHz PCM stream into speech_start / speech_end events that gate STT and drive endpointing + barge-in.
- **Wired in:** `augmentum/voice/vad.py` (`VadProcessor`, a stateful wrapper) · consumed by `augmentum/proxy/voice_routes.py` (server-VAD mode). State machine: IDLE → SPEECH → TRAILING.
- **License:** MIT (official).

## Capabilities (official — snakers4/silero-vad)
- **Version:** v6.2.1 (released 2026-02-24). Our code targets the v6 line.
- **Model size:** JIT model ~2 MB.
- **Sample rates:** 8000 Hz and 16000 Hz.
- **Chunk size:** 30+ ms. **Latency:** <1 ms per chunk on a single CPU thread.
- **Languages:** trained on corpora covering 6000+ languages.
- **Runtime:** PyTorch or ONNX Runtime; "zero strings attached" (no telemetry/keys/expiry).
- **Function-calling / modalities:** VAD only (audio → speech-probability).

## Augmentum wiring notes (`vad.py`)
- Frames: **512 samples @ 16 kHz = 32 ms each** (`FRAME_SAMPLES=512`), satisfying the 30+ ms minimum.
- Tunables (defaults): `speech_threshold=0.6`, `silence_duration_ms=800`, plus `min_speech_ms` (reject sub-noise) and `prefix_padding_ms` (keep pre-speech audio for STT context).
- Emits `speech_start` / `speech_end` / `speech_discard` `VadEvent`s.

## Gotchas (the paid-for lessons)
- **16 kHz mono PCM16 only on the hot path** — feed exactly 512-sample frames; mismatched frame sizes break the v6 model's expectations.
- `silence_duration_ms` is the raw VAD endpoint; it's deliberately generous (800 ms) and then *narrowed* by the SmartTurn v3 model (see [smart-turn-v3](smart-turn-v3.md)) once that model confirms the user is done — VAD alone over-cuts on natural pauses.

## Sources
- Official repo: https://github.com/snakers4/silero-vad
- `augmentum/voice/vad.py`
