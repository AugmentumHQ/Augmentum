# Provider Reference Cards

Verbatim, information-dense reference for **every external service Augmentum talks to** —
features, full request-param catalog, models, context/output limits, reasoning/vision/tool
support, and pricing — pulled directly from each provider's official docs.

**Discipline:** numbers and field names are copied **verbatim** from the source pages (no
paraphrase). Each card carries a `Sourced:` date + a `Sources:` URL list. Each card also
reconciles "what the provider supports" against "what Augmentum actually sends" (the
`provider_profiles.py` flags / backend behavior), surfacing drift.

Card format: see [`deepseek.md`](deepseek.md) (the reference exemplar). Cards also carry a
**Model-level handling** section — per-model reasoning allow-lists, context/output caps, sampler
quirks, and pricing — since most real corrections are per-model, not per-provider (e.g. xAI effort
varies by model, Groq gpt-oss vs qwen, Perplexity reasoning-models-only, Moonshot temp clamp).

**Consolidated cards.** Three groups share one dense card each (a section per provider) because
they carry little provider-specific handling beyond what the linked vendor cards already cover:
[`aggregators.md`](aggregators.md) (6 thin OpenAI-compat resellers), [`speech.md`](speech.md)
(6 STT/TTS), [`image.md`](image.md) (8 image), [`embeddings.md`](embeddings.md) (4 embed/rerank).
The substantive first-party LLMs each keep a full standalone card.

---

## LLM — OpenAI-compat profiles (`augmentum/models/provider_profiles.py`)

| Provider | id | Card |
|---|---|---|
| DeepSeek | `deepseek` | ✅ [deepseek.md](deepseek.md) |
| OpenAI | `openai` | ✅ [openai.md](openai.md) |
| ChatGPT bridge (codex-proxy) | `chatgpt_bridge` | ✅ [chatgpt_bridge.md](chatgpt_bridge.md) |
| OpenRouter | `openrouter` | ✅ [openrouter.md](openrouter.md) |
| Mistral AI | `mistral` | ✅ [mistral.md](mistral.md) |
| AI21 Labs | `ai21` | ✅ [ai21.md](ai21.md) |
| xAI (Grok) | `xai` | ✅ [xai.md](xai.md) |
| Azure OpenAI | `azure` | ✅ [azure.md](azure.md) |
| Moonshot (Kimi) | `moonshot` | ✅ [moonshot.md](moonshot.md) |
| Groq | `groq` | ✅ [groq.md](groq.md) |
| Perplexity | `perplexity` | ✅ [perplexity.md](perplexity.md) |
| Fireworks AI | `fireworks` | ✅ [fireworks.md](fireworks.md) |
| Pollinations | `pollinations` | ✅ [aggregators.md](aggregators.md) |
| AIML API | `aimlapi` | ✅ [aggregators.md](aggregators.md) |
| ElectronHub | `electronhub` | ✅ [aggregators.md](aggregators.md) |
| Chutes AI | `chutes` | ✅ [aggregators.md](aggregators.md) |
| NanoGPT | `nanogpt` | ✅ [aggregators.md](aggregators.md) |
| Z.AI (GLM) | `zai` | ✅ [zai.md](zai.md) |
| SiliconFlow | `siliconflow` | ✅ [aggregators.md](aggregators.md) |
| Cohere | `cohere` | ✅ [cohere.md](cohere.md) |
| Together AI | `together` | ✅ [together.md](together.md) |
| NVIDIA NIM | `nvidia` | ✅ [nvidia.md](nvidia.md) |

## LLM — native backends (own adapters, bypass profiles)

| Provider | `provider_type` | Card |
|---|---|---|
| Anthropic (Claude) | `anthropic` | ✅ [anthropic.md](anthropic.md) |
| Google (Gemini / Vertex) | `gemini` | ✅ [gemini.md](gemini.md) |

## Speech — STT / TTS (cloud)

| Provider | Modality | Card |
|---|---|---|
| Deepgram | STT + TTS (Aura) | ✅ [speech.md](speech.md) |
| ElevenLabs | TTS (+ STT Scribe, Voice Design) | ✅ [speech.md](speech.md) |
| Groq | STT (Whisper) | ✅ [speech.md](speech.md) · [groq.md](groq.md) |
| Rime | TTS | ✅ [speech.md](speech.md) |
| Fish Audio | TTS | ✅ [speech.md](speech.md) |
| OpenAI | TTS + STT (Whisper / gpt-4o-audio) | ✅ [speech.md](speech.md) |

## Image generation (cloud)

| Provider | Card |
|---|---|
| OpenAI (gpt-image / DALL·E) | ✅ [image.md](image.md) |
| Stability AI | ✅ [image.md](image.md) |
| Black Forest Labs (FLUX) | ✅ [image.md](image.md) |
| Fal.ai | ✅ [image.md](image.md) |
| Together AI (image) | ✅ [image.md](image.md) |
| Ideogram | ✅ [image.md](image.md) |
| NVIDIA (image) | ✅ [image.md](image.md) |
| xAI (grok-imagine) | ✅ [image.md](image.md) |

## Embeddings / Rerank (candidates — currently local-only; cloud not wired)

> Tracked for completeness; these are **coverage gaps** (`reranker`/`embeddings` have no cloud
> abstraction yet). Cards still useful for the planned cloud-backend.

| Provider | Modality | Card |
|---|---|---|
| Voyage AI | Embeddings + Rerank | ✅ [embeddings.md](embeddings.md) |
| Jina AI | Embeddings + Rerank | ✅ [embeddings.md](embeddings.md) |
| Cohere | Embeddings + Rerank | ✅ [embeddings.md](embeddings.md) |
| Mistral | Embeddings + OCR | ✅ [embeddings.md](embeddings.md) |

## Other external infrastructure

| Service | Notes | Card |
|---|---|---|
| SearXNG | self-hosted meta-search (`searxng_base_url`) | ☐ |
| Jellyfin / Plex / Komga / Audiobookshelf | media servers — see [`docs/integrations/media-servers/`](../integrations/media-servers/) | (existing) |

---

_Status legend: ✅ done · ☐ to do. Update this index when a card lands._
