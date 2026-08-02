# PocketTTS (Kyutai)

> ~100M-param flow/streaming TTS from the Moshi team — CPU-real-time, voice
> cloning, small enough for WebAssembly. Augmentum's ultra-light TTS tier for
> hosts without GPU headroom (and the on-device path).

- **Role in Augmentum:** CPU-only TTS tier alongside Kokoro — used when there's no GPU headroom or when latency-on-CPU is the right trade-off; also the on-device (phone) cloned-voice path.
- **Wired in:** `augmentum/voice/pocket_tts.py` (reuses `kokoro_tts._encode_audio` for PCM→wav/mp3/opus). Imports of the `pocket_tts` package are deferred so the module loads even when the dep isn't installed (`is_available` reflects install + weights).
- **Default artifact:** `kyutai/pocket-tts`, cached under `~/.cache/pocket_tts` (upstream default). Default voice `alba`.
- **License:** CC-BY-4.0 (official).

## Capabilities (official — kyutai/pocket-tts)
- **Parameters:** 100M; continuous audio-language-modeling / streaming architecture (Mimi codec; arXiv 2509.06926).
- **Languages:** **English only at present** — "more languages are planned." ⚠️ Our code docstring lists 6 (en/fr/de/it/pt/es); treat that as aspirational/unverified and trust the official card until confirmed on the installed package.
- **Voice cloning:** yes, from a short reference clip. Ships 8 named voices (alba, marius, javert, jean, fantine, cosette, eponine, azelma).
- **Footprint / perf:** runs in browsers via WASM; uses ~2 CPU cores; ~6× real-time on a MacBook Air M4 CPU; **~200 ms to first audio chunk**. (Our code notes ~236 MB weights on disk.)
- **Requirements:** Python 3.10–3.14, PyTorch 2.5+.
- **Function-calling / modalities:** TTS only.

## Augmentum wiring notes
- **No model-level streaming** — we still chunk by sentence for perceived latency, but each chunk is generated atomically (use the `smooth` chunk mode).
- **Not thread-safe** — inference serialized through a lock (upstream documents batch=1 only).

## Gotchas (the paid-for lessons)
- **English-only today** per the official card — don't route non-English to it without verifying the installed package's language list.
- Official limitations: does **not** support adding silence in the text input, and (per the card) **not** int8 quantization of the compute. ⚠️ This contradicts our on-device "int8 ONNX" exploration notes — re-verify before relying on an int8 PocketTTS path.

## Sources
- Official model card: https://huggingface.co/kyutai/pocket-tts
- `augmentum/voice/pocket_tts.py`
