# Kokoro-82M

> Open-weight 82M-param TTS — "comparable quality to larger models while
> significantly faster and more cost-efficient." Augmentum's primary
> in-process TTS voice.

- **Role in Augmentum:** in-process TTS (no sidecar, no HTTP) — the default voice for chat/companion/narration. Streaming per-segment, weighted voice-blend mixing.
- **Wired in:** `augmentum/voice/kokoro_tts.py` (via `kokoro-onnx`, ONNX Runtime CPU) · weights at `resolve_model_dir("kokoro")` (baked into the Docker image).
- **Default artifact:** kokoro-onnx INT8 (~88 MB ONNX; the 82M base quantized) — source: `kokoro_tts.py` header.
- **License:** Apache-2.0 (official).

## Capabilities (official — hexgrad/Kokoro-82M)
- **Parameters:** 82 million.
- **Architecture:** StyleTTS 2 (arXiv 2306.07691) + ISTFTNet vocoder (arXiv 2203.02395); decoder-only, no diffusion.
- **Languages / voices:** 8 languages, 54 voices total. v1.0 released 2025-01-27.
- **Sample rate:** 24,000 Hz.
- **Function-calling / modalities:** TTS only (text → audio).

## Augmentum wiring notes
- `_SAMPLE_RATE = 24000`; PCM → MP3/WAV via ffmpeg (already in the image).
- **Voice mixing:** weighted blend of voice embeddings via numpy (`VOICE_META` carries per-voice grade/gender/lang; `af_heart` is the flagship composite).
- Thread-safe (ONNX Runtime handles concurrent inference).
- Streaming: async generator yields a chunk per text segment (see `augmentum/voice/pipeline.py` `SentenceBuffer` for the chunking tiers).

## Gotchas (the paid-for lessons)
- **Needs espeak-ng** for phoneme conversion (official requirement) — bundled in the image.
- Voice quality varies a LOT by voice (official VOICES grades A→F); `VOICE_META` mirrors those grades so the picker can steer toward A/B voices. Low-grade voices ("limited training data") are best used only inside blends.

## Sources
- Official model card: https://huggingface.co/hexgrad/Kokoro-82M
- kokoro-onnx port: https://github.com/thewh1teagle/kokoro-onnx
- `augmentum/voice/kokoro_tts.py`
