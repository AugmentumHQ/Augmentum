# Mistral AI — Provider Reference Card

> **Verbatim reference** from Mistral's official docs (`docs.mistral.ai`, `mistral.ai/pricing`).
> **Sourced:** 2026-06-25 · **Sources:** see [§8](#8-sources).

| | |
|---|---|
| **Profile id** (`provider_profiles.py`) | `mistral` |
| **`provider_type`** | `openai` (+ `converter_id="mistral"`) |
| **Base URL** | `https://api.mistral.ai/v1` |
| **Endpoint** | `POST /v1/chat/completions` |
| **Auth** | `Authorization: Bearer <MISTRAL_API_KEY>` |
| **Vision** | ✅ on current generalist models (Large 3, Medium 3.5, Small 4, Ministral 3, Pixtral) |
| **Tools** | ✅ (`parallel_tool_calls` default **true**; `tool_choice` adds Mistral-only **`any`**) |

---

## 1. Augmentum wiring (what WE send)

From `provider_profiles.py` (`mistral`):

| Augmentum field | Value | Reconciliation |
|---|---|---|
| `converter_id` | `mistral` | message-shape conversion |
| `max_context` | `256_000` | ✅ Mistral Large 3 = 256K (cited docs.mistral.ai) |
| `max_output` | `0` (unmodeled) | shared context budget; read per-model |
| _(no OpenAI-family flags)_ | — | strict body → would 400 on unknown keys |

⚠️ Mistral reasoning is **model-selection based** (pick `magistral-*` / Small 4), not a per-request
toggle — there is **no** `reasoning_effort`/`thinking` param in Mistral's API (see §3, §7).

---

## 2. Models (verbatim)

| Model | API id (per docs) | Notes |
|---|---|---|
| Mistral Large 3 (v25.12) | `mistral-large-3` | open-weight general-purpose **multimodal**; 256K ctx |
| Mistral Medium 3.5 (v26.04) | `mistral-medium-3505`* | "frontier-class multimodal, agentic + coding" |
| Mistral Small 4 (v26.03) | `mistral-small-2603` | "Hybrid model unifying instruct, **reasoning**, and coding" |
| Ministral 3 14B/8B/3B (v25.12) | `ministral-3-14b` / `-8b` / `-3b` | multimodal lightweight |
| Magistral Medium / Small | `magistral-*` | **reasoning** models |
| Codestral (v25.08) | `codestral-*` | code completion |
| Devstral 2 | `devstral-*` | agentic coding |
| Pixtral | `pixtral-*` | vision |
| Mistral Embed / Codestral Embed | `mistral-embed` / `codestral-embed` | embeddings |
| OCR 4 (v4.0) | `mistral-ocr-*` | doc OCR, paragraph bounding boxes |
| Voxtral TTS / Transcribe | `voxtral-*` | TTS w/ voice clone; realtime transcription |

_* id inferred from version string in docs; verify exact alias._ Context windows beyond Large 3
(256K) were not in the snapshot — **TODO** per-model ctx.

---

## 3. Request parameters (verbatim)

| Parameter | Type | Req | Default | Notes |
|---|---|:--:|---|---|
| `model` | string | ✅ | — | "ID of the model to use" |
| `messages` | array | ✅ | — | "encoded as a list of dict" |
| `temperature` | number\|null | — | — | "recommend 0.0–0.7 range" |
| `top_p` | number\|null | — | — | nucleus sampling |
| `max_tokens` | integer\|null | — | — | max tokens to generate |
| `n` | integer\|null | — | — | completions per request |
| `stream` | boolean | — | false | partial progress |
| `stop` | string\|array\|null | — | — | stop sequences |
| `random_seed` | integer\|null | — | — | **Mistral's name** for the seed (OpenAI uses `seed`) |
| `frequency_penalty` | number\|null | — | — | — |
| `presence_penalty` | number\|null | — | — | encourages diversity |
| `response_format` | object\|null | — | `{type:"text"}` | `text`, `json_object`, **`json_schema`** |
| `tools` | array\|null | — | — | tools the model may call |
| `tool_choice` | object\|string | — | — | `auto`, `none`, **`any`**, `required`, or tool spec |
| `parallel_tool_calls` | boolean | — | **true** | parallel function calling |
| `safe_prompt` | boolean | — | false | **Mistral-only** — inject safety prompt |

---

## 4. Reasoning

No `reasoning_effort` / `thinking` toggle in the chat API. Reasoning = **choosing a reasoning model**
(`magistral-*`, or Small 4's hybrid mode). Recommended `temperature` 0.0–0.7 (lower than OpenAI's
default 1.0).

---

## 7. Pricing (verbatim, per 1M tokens, USD)

| Model | Input | Output |
|---|---|---|
| Mistral Medium 3.5 | $1.5 | $7.5 |
| Mistral Small 4 | $0.15 | $0.6 |
| Mistral Large 3 | $0.5 | $1.5 |
| Ministral 3B | $0.1 | $0.1 |
| Ministral 8B | $0.15 | $0.15 |
| Ministral 14B | $0.2 | $0.2 |
| Codestral | $0.3 | $0.9 |
| Devstral 2 | $0.4 | $2 |
| Magistral Medium | $2 | $5 |
| Magistral Small | $0.5 | $1.5 |
| Mistral Embed | $0.1 | — |
| Codestral Embed | $0.15 | — |
| OCR 4 | $4 / 1000 pages (std) · $2 / 1000 (batch) | — |
| Voxtral TTS | $0.016 / 1K chars | — |

---

## 8. Known drift / gaps

- 🟢 **Seed param name** — Mistral uses `random_seed`, OpenAI uses `seed`. If Augmentum emits `seed` to a Mistral target, determinism is silently lost. Verify the `mistral` converter maps it.
- 🟢 **Reasoning model-based** — UI/role logic should surface `magistral-*` / Small 4 for reasoning rather than expect a `reasoning_effort` toggle (Mistral has none).
- ☐ Per-model context windows (beyond Large 3 256K) + OCR/embed/Voxtral endpoints not yet captured.

---

## 9. Sources

- Models overview: https://docs.mistral.ai/getting-started/models/models_overview/
- API reference: https://docs.mistral.ai/api/
- Pricing: https://mistral.ai/pricing
