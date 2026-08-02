# Speech (STT / TTS) — Combined Reference Card

> **Verbatim reference** for the cloud **speech** providers Augmentum talks to (STT + TTS).
> Audio providers are **DB-driven** (`audio_providers` table) with per-vendor adapters in
> `voice/tts.py` + `proxy/audio_routes.py`; bundled local engines (Kokoro/Moonshine/Pocket)
> are separate. **Sourced:** 2026-06-25 · Sources per §.

> ℹ️ Known wiring fix (shipped): `fabric/extractors.py` previously double-advertised bundled
> Kokoro/Moonshine (builtin + DB `base_url='builtin'`) → 1-voice clobbered 54-voice. Skip
> builtin rows (commit 3e31540). See the `project_fabric_stt_tts_testing` memory.

---

## 1. Deepgram — STT (+ Aura TTS)

| | |
|---|---|
| **Base URL** | `https://api.deepgram.com` |
| **STT** | `POST /v1/listen` (batch) · **WS** `/v1/listen` (streaming) |
| **Auth** | `Authorization: Token <DEEPGRAM_API_KEY>` |

**STT models:** `nova-3` / `nova-3-general` (top general, multilingual); `nova-2` (+ domain variants `nova-2-phonecall`/`-meeting`/`-medical`/`-finance`/…); **`flux-general-en` / `flux-general-multi`** (conversational model built for voice agents — turn-aware).
**Key params:** `model`, `language` (def `en`), `smart_format`, `diarize`, `punctuate`, `interim_results`, `endpointing` (turn/silence detection), streaming `encoding`/`sample_rate`.
**TTS (Aura):** `POST /v1/speak` — Aura voices; low-latency. (Rate limits: Whisper-compat models 15 concurrent paid / 5 PAYG.)
**Sources:** developers.deepgram.com/docs/models-languages-overview · /reference

---

## 2. ElevenLabs — TTS (+ STT Scribe, Voice Design)

| | |
|---|---|
| **Base URL** | `https://api.elevenlabs.io` (+ regional `us`/`eu`/`in`/`sg` residency hosts) |
| **TTS** | `POST /v1/text-to-speech/{voice_id}` · stream `/{voice_id}/stream` · **WS** `/{voice_id}/stream-input` |
| **Auth** | `xi-api-key: <KEY>` header (or `authorization` bearer) |

**Models:** `eleven_v3`, `eleven_multilingual_v2`, `eleven_flash_v2_5` (low-latency), `eleven_turbo`.
**Params:** `text`, `model_id`, `voice_id` (URL), `language_code`, `output_format` (def `mp3_44100`), `voice_settings`:{`stability`, `similarity_boost`, `style`, `use_speaker_boost`, `speed`}, `enable_ssml_parsing` (def false), `apply_text_normalization` (def auto), `enable_logging`, `inactivity_timeout` (def 20).
**Notable:** word-to-audio alignment available on the WS API; STT = **Scribe** model.
**Sources:** elevenlabs.io/docs/api-reference/text-to-speech

---

## 3. Groq — STT (Whisper)

Same key/base as Groq LLM (`https://api.groq.com/openai/v1`, `POST /audio/transcriptions`). Models: `whisper-large-v3` ($0.111/hr transcribed, 217× speed), `whisper-large-v3-turbo` ($0.04/hr, 228× speed). Audio billed **min 10s/request**. (Full detail in [groq.md](groq.md).)

---

## 4. Rime — TTS

| | |
|---|---|
| **Endpoint** | `POST https://users.rime.ai/v1/rime-tts` (streaming) |
| **Auth** | `Authorization: Bearer <KEY>` |

**Params:** `speaker`* (a **Coda** voice), `text`* (≤3000 chars), `modelId` (`coda` realistic conversational; also `mist`/`mistv2` arcana lineage), `language` (en/es/fr/pt/de/ja), `samplingRate` (def 24000), `timeScaleFactor` (def 1.0; >1 slows, <1 speeds).
**Output:** via `Accept` header — `audio/webm;codecs=opus` (recommended), `audio/mpeg`, `audio/wav`, `audio/L16` (headerless PCM).
**Sources:** docs.rime.ai/api-reference/endpoint/tts

---

## 5. Fish Audio — TTS (voice cloning)

| | |
|---|---|
| **Base URL** | `https://api.fish.audio` · `POST /v1/tts` |
| **Auth** | `Authorization: Bearer <KEY>` · **model** via `model` header: `s1` / `s2-pro` (recommended) |

**Params:** `text`*, `reference_id` (speaker; **array** for multi-speaker on S2-Pro), `references` (zero-shot inline clone — needs MessagePack), `format` (wav/pcm/`mp3` def/opus), `sample_rate` (def 44.1k; 48k opus), `mp3_bitrate` (64/128/192), `chunk_length` (100–300, def 300), `normalize` (def true), `speed` (0.5–2.0), `volume` dB, `normalize_loudness` (S2-Pro), `temperature` (def 0.7), `top_p` (def 0.7), `latency` (`low`/`balanced`/`normal`), `max_new_tokens` (def 1024), `repetition_penalty` (def 1.2), `condition_on_previous_chunks` (def true).
**Multi-speaker (S2-Pro):** inline `<|speaker:0|>…<|speaker:1|>…` tags + `reference_id` array.
**Content types:** `application/json` or `application/msgpack` (msgpack required for direct audio upload).
**Sources:** docs.fish.audio/api-reference/endpoint/openapi-v1/text-to-speech

---

## 6. OpenAI — TTS + STT

Same key/base as OpenAI LLM (`https://api.openai.com/v1`).
- **TTS:** `POST /v1/audio/speech` — models `gpt-4o-mini-tts`, `tts-1`, `tts-1-hd`; params `model`, `input`, `voice` (alloy/echo/fable/onyx/nova/shimmer/…), `response_format` (mp3/opus/aac/flac/wav/pcm), `speed`, `instructions` (gpt-4o-mini-tts steerability).
- **STT:** `POST /v1/audio/transcriptions` (+ `/translations`) — models `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`; params `file`, `model`, `language`, `prompt`, `response_format` (json/text/srt/verbose_json/vtt), `temperature`, `timestamp_granularities`.
- **Realtime audio:** `gpt-4o-realtime-preview` via the Realtime API (WS). See [openai.md](openai.md) for the LLM side.

---

## Known drift / gaps (speech)

- 🟢 The audio-provider system is generic DB rows — per-vendor request shapes (Rime `users.rime.ai`, Fish `model` header + msgpack, ElevenLabs `xi-api-key` + voice-in-URL) each need their adapter; a borrowed OpenAI-audio shape will not work against Rime/Fish/Deepgram.
- 🟢 Deepgram **`flux`** (voice-agent turn-aware) and ElevenLabs WS alignment are higher-value for the voice loop than the generic batch path — not yet specially wired.
- Pricing intentionally omitted where not fetched verbatim — see each provider's pricing page.
