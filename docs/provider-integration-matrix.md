# Augmentum Provider × Modality Matrix

_Canonical wiring reference — generated from code-as-source-of-truth + per-provider doc audit._
_Date: 2026-06-15_

## Legend

| Symbol | Meaning |
|---|---|
| ✅ | **full** — wired and matches current provider docs |
| 🟡 | **partial** — wired but with drift, stale catalog, missing sub-capabilities, or runtime-works-but-no-preset/discovery |
| ⬜ | **none** — provider offers this modality but Augmentum wires nothing |
| — | **n/a** — provider does not offer this modality |

> **Embeddings & Rerank are local-only by design** (`augmentum/memory/embeddings.py` hardcodes `nomic-embed-text-v1.5-Q`; `augmentum/memory/reranker.py` picks among local ONNX cross-encoders via `settings.reranker_model`). There is **no cloud-provider abstraction** for either modality, so every cloud embeddings/rerank offering shows ⬜ — a structural coverage gap, not a per-provider defect.

## Matrix

| Provider | LLM | STT | TTS | Image | Embeddings | Rerank | Vision |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| OpenAI | ✅ | ✅ | ✅ | 🟡 | ⬜ | — | ✅ |
| Google Gemini | ✅ | ⬜ | ⬜ | ⬜ | ⬜ | — | ✅ |
| Anthropic Claude | 🟡 | — | — | — | — | — | ✅ |
| DeepSeek | 🟡 | — | — | — | — | — | — |
| Mistral | 🟡 | — | — | — | ⬜ | — | 🟡 |
| xAI (Grok) | 🟡 | — | — | 🟡 | — | — | ✅ |
| Cohere | ✅ | — | — | — | ⬜ | ⬜ | — |
| Moonshot (Kimi) | 🟡 | — | — | — | — | — | 🟡 |
| Z.AI (GLM) | ✅ | — | — | ⬜ | ⬜ | ⬜ | 🟡 |
| OpenRouter | 🟡 | — | — | — | ⬜ | — | 🟡 |
| Groq | 🟡 | 🟡 | — | — | — | — | 🟡 |
| Perplexity | 🟡 | — | — | — | — | — | — |
| Fireworks AI | 🟡 | ⬜ | — | ⬜ | ⬜ | ⬜ | 🟡 |
| Together AI | ✅ | — | — | 🟡 | ⬜ | ⬜ | ✅ |
| NVIDIA NIM | 🟡 | — | — | — | ⬜ | ⬜ | 🟡 |
| Deepgram | — | 🟡 | ✅ | — | — | — | — |
| ElevenLabs | — | ⬜ | ✅ | — | — | — | — |
| Stability AI | — | — | — | 🟡 | — | — | — |
| Black Forest Labs | — | — | — | 🟡 | — | — | — |
| Fal.ai | — | ⬜ | ⬜ | 🟡 | — | — | — |
| Speaches | — | ✅ | 🟡 | — | — | — | — |
| Voyage AI | — | — | — | — | ⬜ | ⬜ | — |
| Jina AI | — | — | — | — | ⬜ | 🟡 | — |

_Notes: "Vision" = image-input on the LLM chat path. 🟡 vision (Mistral/Moonshot/Z.AI/OpenRouter/Groq/Fireworks/NVIDIA) = works via generic OpenAI-compat passthrough with no first-class capability flag or model registration. Fal also offers video-gen (whole modality Augmentum has zero coverage for — not a column here). Local GPU image (diffusers FLUX/Qwen/SD via `image/pipeline_v2.py`) and local in-process TTS (Kokoro/Pocket) / STT (Moonshine) run in parallel to all cloud providers above._

---

## Drift findings

Existing integrations that disagree with current (June 2026) provider docs. **These are correctness bugs.**

### HIGH

- **Fal · image** — `cloud_image_routes.py:945-948` builds `https://fal.run/v1/{model}`; correct URL has **no `/v1/`** segment → every fal image-gen 404s. Fix: `f"{fal_base.rstrip('/')}/{model}"`.
- **OpenAI · image** — `cloud_image_routes.py:693/994` always injects `response_format=b64_json` + `quality='standard'`; the entire `gpt-image-*` lineup **rejects** `response_format` (400) and quality enum is low/medium/high. Fix: stop sending both for gpt-image-* (dall-e legacy only).
- **Claude · llm (sampling)** — `adapters/claude.py:209-225` sends temperature/top_p/top_k unconditionally (stripped only under thinking); Opus 4.7/4.8/Fable-5 **remove** these params → 400. Fix: `is_no_sampling_model()` covering opus-4-7|opus-4-8|fable-5.
- **Claude · llm (thinking/prefill)** — `converters/claude.py:25-33,46-48` regexes top out at 4.6; 4.7/4.8/Fable-5 match nothing → thinking never enabled + assistant-prefill (Continue) sends trailing assistant turn → 400. Fix: extend `_THINKING_MODEL_RE`/`_ADAPTIVE_MODEL_RE`/`_NO_PREFILL_MODEL_RE`.
- **OpenRouter · llm (parse)** — `openai_compat.py:726,1021` reads reasoning only from `reasoning_content`; OpenRouter returns it in `reasoning` → all OpenRouter reasoning silently dropped. Fix: `data.get('reasoning_content') or data.get('reasoning')` in both parse + delta paths.
- **OpenRouter · llm (control)** — `openai_compat.py:609-674` + `provider_profiles.py:209-215`: reasoning control only reaches OpenAI-family ids; the unified `reasoning` object is never sent → anthropic/deepseek/qwen/glm via OpenRouter get no reasoning control. Fix: `supports_openrouter_reasoning` path emitting `reasoning:{enabled,effort}` / `exclude`.
- **Fal · image (catalog)** — `cloud_image_routes.py:460-463,668-674`: edit-only models listed in txt2img catalog but all fal edits 501; selecting one POSTs txt2img to an edit endpoint. Fix: remove edit rows or implement `_edit_fal`.

### MEDIUM

- **DeepSeek · llm (context)** — `provider_profiles.py:248` omits max_context → inherits 128K default; V4 is **1M**. Coder auto-compactor wastes ~87% of the window. Fix: `max_context=1_000_000` + correct the `openai_compat.py:1222-1224` comment.
- **xAI · llm** — `provider_profiles.py:255-266`: max_context pinned 256K (grok-4.3 is 1M); reasoning_effort allows minimal/xhigh which xAI rejects. Fix: 1M context; clamp effort to none/low/medium/high.
- **Mistral · llm** — `provider_profiles.py:216-221` + `openai_compat.py:609`: root-level `reasoning_effort` never sent; think toggle no-ops on small/medium-3.5. Fix: per-model supports_reasoning_effort (suppress on magistral-* which 422).
- **Gemini · llm (Flash-Lite)** — `converters/gemini.py:21,213`: broad `gemini-3.*flash` regex maps 'min'→'minimal' for gemini-3.1-flash-lite (only low/medium/high) → 400. Fix: Flash-Lite branch flooring 'min'→'low'.
- **Groq · llm** — `provider_profiles.py:286-290`: no reasoning flags; reasoning_effort/reasoning_format never sent → thinking toggle no-op for gpt-oss/qwen3-32b. Fix: add supports_reasoning_effort + emit for those ids.
- **NVIDIA NIM · llm** — `provider_profiles.py:362-374`: bare profile; NIM reasoning families get no thinking signal. Fix: per-model capability resolution (reasoning_effort / extra_body enable_thinking).
- **Fireworks · llm** — `provider_profiles.py:298-302`: bare profile; reasoning toggle no-op for hosted DeepSeek-R1/Qwen3/GLM/Kimi. Fix: reasoning flag + emit reasoning_effort.
- **Perplexity · llm** — `provider_profiles.py:291-297`: supports_reasoning_effort=False → reasoning_effort dropped for sonar-reasoning-pro/deep-research; supports_thinking inert. Fix: add supports_reasoning_effort (scope to reasoning ids).
- **Deepgram · stt (Flux)** — `streaming_stt.py:98` + `audio_routes.py:1719,1811`: flux-* ids hardcoded to /v1/listen + channels/alternatives; Flux needs /v2/listen + TurnInfo → no transcript. Fix: route flux-* to v2/listen + TurnInfo parser, or reject loudly.
- **Stability · image (sd3 ids)** — `cloud_image_routes.py:433-442,778-826`: only sd3.5-large/large-turbo reachable; medium/flash unselectable. Fix: add to catalog+endpoint_map, forward model verbatim.
- **Stability · image (edit suite)** — `cloud_image_routes.py:1054-1107`: edit capped at inpaint+search-replace; outpaint/erase/recolor/bg/upscale/control unwired. Fix: add op field + route ops.
- **BFL · image (edit)** — `cloud_image_routes.py:668-674,859-926`: all BFL editing 501 though Kontext + all FLUX.2 edit via input_image on same endpoint. Fix: route input_image, reuse async poll.
- **BFL · image (catalog)** — `cloud_image_routes.py:443-454`: missing flux-2-max/flex/klein-4b/9b + flux-kontext-max; img2img:no wrong.
- **OpenAI · image (catalog)** — `cloud_image_routes.py:412-420`: advertises retired dall-e-3, missing gpt-image-2/1.5.
- **Claude · llm (catalog/context)** — `adapters/claude.py:39-50,422-429`: 4.7/4.8/Fable-5 absent from picker, report 200K not 1M. Fix: add ids @ 1M; broaden get_context_length.
- **Speaches · tts** — `catalog.json:106-122`: combined STT+TTS container registered STT-only; Kokoro/Piper TTS never auto-registered (runtime /v1/audio/speech works). Fix: add speaches-tts catalog entry.
- **Speaches · stt** — `catalog.json:114`: legacy `WHISPER__TTL` silently ignored by current Pydantic config → model unloads after 300s. Fix: rename to `STT_MODEL_TTL`.
- **Together · image** — `cloud_image_routes.py:421-432,489-490`: stale static catalog (wrong Kontext id casing, missing FLUX.2/Qwen-Image/Wan-2.6). Fix: refresh or allow free-typed ids.
- **Gemini · llm (Pro medium)** — `converters/gemini.py:222`: `_G3_PRO_LEVELS` collapses medium→low though 3/3.1 Pro support medium → silent reasoning downgrade.

### LOW

- **Deepgram · tts** — `audio_routes.py:2072`: probe default `aura-2-en` is not a valid voice id. Fix: `aura-2-thalia-en` (label only).
- **Together · image** — `cloud_image_routes.py:716-719`: `quality` sent to Together (no such param; likely ignored). Fix: gate to dall-e3/stability.
- **Stability · image** — `cloud_image_routes.py:799-818`: ultrawide 21:9/9:21 clamped to 16:9; stale 4:3/3:4 comment.
- **Claude · llm** — `adapters/claude.py:90-98`: legacy beta headers dead on current GA models.
- **Moonshot · llm** — `openai_compat.py:523-524`: temperature sent unconditionally; kimi-k2.6/k2.7-code don't accept it.
- **Moonshot/Z.AI/NVIDIA · llm** — `openai_compat.py:677-680`: reasoning_effort folded in for all thinking_type_toggle providers, but Moonshot/Z.AI don't document it (possible strict-400). Fix: gate behind supports_reasoning_effort.
- **Z.AI · llm** — `openai_compat.py:701-702`: clear_thinking/preserve_thinking only on local-engine branch; cloud replay-prior-reasoning unreachable.
- **Jina · rerank** — `config.py:1342`: default jina-reranker-v1-tiny-en is legacy/English-only (local ONNX, still works). Quality note only.
- **Claude · llm (latent)** — `converters/claude.py:73-89`: budget_tokens branch 400s on 4.7/4.8/Fable-5 if added to thinking regex without also adding to adaptive regex; Fable-5 also 400s on `type:disabled`.
- **BFL · image (latent)** — `cloud_image_routes.py:867-871,923`: ultra ignores width/height (wants aspect_ratio); terminal-failure detection misses 'Failed'/'Request Moderated' → 504.
- **Fal · image (latent)** — `cloud_image_routes.py:937,940-941`: negative_prompt/image_size-object sent to FLUX.2-pro (zero-config, may 422).

---

## Coverage gaps

Provider offers a modality Augmentum doesn't wire. **These are feature decisions, not auto-fix bugs.** Effort tags S/M/L. All cloud embeddings/rerank gaps require building the (currently nonexistent) cloud-backend abstraction in `embeddings.py`/`reranker.py` and re-indexing the vector store on any dim change.

### OpenAI
- ⬜ **Embeddings** (text-embedding-3-large/small) — offered, local-only by design. **S** (post cloud-embed abstraction).

### Google Gemini
- ⬜ **Image-gen** (Nano Banana 2 `gemini-3.1-flash-image`, Nano Banana Pro) — high leverage, key already present; needs `responseModalities=[TEXT,IMAGE]` + inlineData handling (`convert_response` drops image parts today). **M**.
- ⬜ **Embeddings** (gemini-embedding-001) — **S** post-abstraction. ⬜ **TTS** (gemini-2.5-flash-tts) / **STT** (audio via generateContent) — out of primary scope. **M** each.

### Mistral
- ⬜ **OCR** (`/v1/ocr`, mistral-ocr-latest) — purpose-built doc extraction; upgrades knowledge-pack PDF ingest. No local equivalent. **M**.
- ⬜ **Embeddings** (mistral-embed, **codestral-embed** — relevant to Coder retrieval). **S** post-abstraction.

### Cohere
- ⬜ **Embeddings** (**embed-v4.0**, multimodal text+image) — highest-leverage gap, aligns with cross-modal moat. First cloud-embed backend. **M**.
- ⬜ **Rerank** (rerank-v3.5 / v4.0-pro/fast) — slots into packs.py cross-encoder leg. **M** (first cloud-rerank backend).

### xAI
- 🟡 **Image-gen** (grok-imagine-image-quality) — addable as generic provider today; needs built-in preset. **S**.

### Z.AI (GLM)
- ⬜ **Embeddings** (embedding-3) / **Rerank** (glm-rerank) — **S** each post-abstraction. ⬜ **Image** (CogView-4) — **M**.

### OpenRouter
- ⬜ **Embeddings** (unified gateway: gemini-embedding-2 / mistral-embed / cohere / openai) — one key, many models. **S** post-abstraction.

### Groq
- 🟡 **STT** (whisper-large-v3/turbo) — works via manual Custom provider; needs a preset. **S**.

### Fireworks AI
- ⬜ **Embeddings** (qwen3-embedding) — OpenAI-compat, same base_url+key. **S** post-abstraction.
- ⬜ **Rerank** (qwen3-reranker via /v1/rerank). **S** post-abstraction.
- ⬜ **Image** (FLUX/SDXL on custom `/image_generation/{model}` + async Kontext workflow) — needs dedicated adapter. **M**.
- ⬜ **STT** (whisper-v3 on separate audio-prod/audio-turbo host + WS streaming) — dedicated entry. **M**.

### Together AI
- ⬜ **Embeddings** (bge / m2-bert) / **Rerank** (Salesforce/Llama-Rank-V1) — key present, OpenAI-/Cohere-compat. **S** post-abstraction.

### NVIDIA NIM
- ⬜ **Embeddings** (nv-embedqa, requires input_type) — **S** post-abstraction.
- ⬜ **Rerank** (nv-rerankqa) — **non-OAI `/v1/ranking` shape** (query.text/passages[].text), can't reuse generic OAI client. **M**.

### Deepgram
- 🟡 **STT Flux** (flux-general-en/multi, conversational with built-in end-of-turn) — could replace local VAD/endpointing; v2/listen + TurnInfo protocol. **M** (see also HIGH drift on wrong-endpoint routing).

### ElevenLabs
- ⬜ **STT** (Scribe scribe_v2, diarized 90+lang) — xi-api-key already pasted for TTS; needs preset + `/v1/speech-to-text` path. **S-M**.
- 🟡 **Other** (Voice Design / Dubbing / Speech-to-Speech / Sound Effects) — only Voices-listing consumed; could feed companion voice mixes / media dubbing / voice enrollment. **L**.

### Stability AI
- 🟡 **Image** (upscale fast/conservative/creative; control structure/sketch/style; extended edit ops; Image-to-3D) — see MEDIUM drift; mostly coverage beyond inpaint+search-replace. **M**.

### Black Forest Labs
- 🟡 **Image** (FLUX.2 max/flex/klein, Kontext editing) — see MEDIUM drift; editing is the feature. **M**.

### Fal.ai
- ⬜ **Video-gen** (Veo3 / Kling / ByteDance) — **whole new modality** (Augmentum has zero cloud video). **L**.
- ⬜ **STT** (Wizper / Whisper) / **TTS** (MiniMax speech-02-hd, voice-clone) — separate provider entries. **M** each.
- 🟡 **Image editing** (flux-2-pro/edit, kontext, v1/fill) — see HIGH drift.

### Speaches
- 🟡 **TTS** (Kokoro/Piper on same container) — runtime path works; catalog/auto-register missing. **S** (see MEDIUM drift).

### Voyage AI
- ⬜ **Embeddings** (voyage-4 family, multimodal) — **S** post-abstraction.
- ⬜ **Rerank** (**rerank-2.5**, instruction-following, ~8% over Cohere v3.5) — high-leverage. **M** (first cloud-rerank backend).

### Jina AI
- ⬜ **Embeddings** (v4 multimodal) — **S** post-abstraction.
- 🟡 **Rerank** (cloud v3 listwise / m0 multimodal / 131K-ctx) — only local v1-tiny wired; cloud generations are best-in-class. **S-M** post-abstraction.