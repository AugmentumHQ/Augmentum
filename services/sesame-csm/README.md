# Sesame CSM TTS sidecar

Conversational streaming TTS for Augmentum, built on
[davidbrowne17/csm-streaming](https://github.com/davidbrowne17/csm-streaming)
(incremental-decode fork of Sesame's CSM-1B). Exposes the same
OpenAI-compatible `/v1/audio/speech` contract as every other Augmentum
TTS provider, so the main app registers it with zero special-casing.

**Why CSM over Pocket/Kokoro:** prosody conditioned on the *conversation
so far*, not just the current text. Pocket can't do this — verified dead
end (see project memory `pocket-tts-conversational-ceiling`). CSM was
trained for it.

## Requirements
- NVIDIA GPU, ~3–4 GB VRAM free (bf16). RTF ~0.28× on a 4090 → real-time.
- **HF token** whose account has accepted **both** gated licenses:
  - https://huggingface.co/sesame/csm-1b
  - https://huggingface.co/meta-llama/Llama-3.2-1B
  - Set `AUGMENTUM_HUGGINGFACE_TOKEN` in `.env`.

## Enable
1. Accept the two HF licenses above (one-time, ~2 min).
2. Add `compose.sesame-csm.yaml` to `.augmentum.conf`.
3. `start.bat` / `start.sh` — first boot downloads ~3–4 GB of weights.
4. It auto-registers as the `sesame-csm` TTS provider; pick it in voice settings.

## Endpoints
- `POST /v1/audio/speech` — OpenAI-compat. Streams WAV/PCM live (low TTFB);
  buffers+ffmpeg for mp3/opus/aac/flac.
- `POST /v1/voices` — clone: upload a reference clip + (ideally) its
  transcript. CSM clones from a `(text, audio)` Segment, so the transcript
  helps materially.
- `GET /v1/voices`, `GET /v1/models`, `GET /health`.

## Shared voices (cross-engine)
CSM mounts the same `/data/voices` store as Chatterbox/Pocket, so every
voice a user clones is **automatically available to CSM** — no re-upload.
The main app's clone flow (`POST /api/audio/voices/clone`) now persists a
`<name>.txt` transcript beside the audio (via the default STT, falling
back to built-in Moonshine), which is exactly what CSM needs to clone well.
Every voice CSM exposes is tagged **`<name>-csm`** so the unified picker
distinguishes CSM's clone of a shared source from the source provider's
own voice. (Kokoro is excluded by nature — its voices are preset
embeddings, not user audio, so there's no source clip to clone from.)

## Conversational context (the point)
The OpenAI contract has no slot for conversation history, so the sidecar
carries it statefully, keyed by the `X-Augmentum-Session` header: it keeps
a short rolling buffer of what **it** just said and re-feeds it (plus the
clone anchor, re-injected each turn to hold voice identity) as CSM context.
That delivers cross-turn prosodic continuity without changing the wire
contract. v1 = self-context + cloning; cross-speaker context (priming the
*user's* audio) and the Mimi audio-history substrate feed are a later phase.

## Idle VRAM
An always-resident CSM would pin ~3–4 GB of VRAM 24/7 for a few seconds of
real use per conversation. So the model **unloads from the GPU after
`CSM_IDLE_UNLOAD_S` idle seconds** (default 90; `0` = never) and
lazy-reloads from the local HF cache on the next request — freeing the
VRAM for the LLM / image-gen the rest of the time. To hide the ~3–5 s cold
reload, ping **`POST /warmup`** at voice-session start (it returns
immediately and loads in the background); by the time STT + the LLM
produce the first sentence, CSM is warm. `GET /health` reports `loaded` +
`idle_for_s`. Note: an in-flight generation can't be unloaded mid-stream.

## Known v1 caveats
- **No watermark on the streaming path** (the fork drops it; Sesame's
  license asks you keep it — revisit before any non-personal/public use).
- **`speed` is a no-op** (CSM has no native rate control), like Pocket.
- **First build may need dep tuning** — flash-attn/vllm/triton are omitted;
  if the fork's import graph hard-requires one, add it to requirements.txt.
- **Watermark + non-streaming high-quality path** not yet wired (could add a
  `?stream=0` buffered+watermarked variant later).
- For the main app to stream low-latency, request `response_format=wav`;
  mp3 (Augmentum's default) buffers before first byte.
