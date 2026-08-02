# Cohere (Command) — Provider Reference Card

> **Verbatim reference** from Cohere's official docs (`docs.cohere.com`).
> **Sourced:** 2026-06-25 · **Sources:** see [§6](#6-sources). (Embeddings/Rerank tracked separately — see README embeddings section.)

| | |
|---|---|
| **Profile id** (`provider_profiles.py`) | `cohere` |
| **`provider_type`** | `openai` **+ `converter_id="cohere"`** (native v2 shape, NOT plain OpenAI) |
| **Base URL** | `https://api.cohere.com/v2` |
| **Endpoint** | `POST /v2/chat` (**not** `/chat/completions`) |
| **Auth** | `Authorization: Bearer <COHERE_API_KEY>` |
| **Nature** | First-party Command family. **Distinct v2 API** — content blocks, documents/citations, safety_mode, native thinking. |

---

## 1. Augmentum wiring (what WE send)

`cohere` is the **only OpenAI-compat profile with a dedicated `converter_id`** — Augmentum translates its OpenAI-shaped internal request into Cohere v2 (`/v2/chat`, content-block messages). The converter is the reconciliation surface:

| Cohere v2 capability | Likely surfaced? | Note |
|---|---|---|
| `messages` content **blocks** (text/image/thinking) | partial | converter must map string↔blocks; vision/thinking blocks easy to drop |
| `thinking` (enable + token budget) | ❓ verify converter | reasoning returns as a **distinct content block**, not `reasoning_content` (#28) |
| `documents` (grounded citations) | ❌ | Cohere's headline RAG/citation feature unused |
| `safety_mode` (CONTEXTUAL/STRICT/OFF) | ❌ | safety control not exposed |
| `k` / `p` sampler naming | needs map | Cohere uses `k`/`p`, not `top_k`/`top_p` (#29) |
| `frequency_penalty`/`presence_penalty` | range differs | Cohere **0.0–1.0**, not −2..2 (#29) |
| `tool_choice` `REQUIRED`/`NONE` | needs map | uppercase enum, not OpenAI's `required`/`none` |

---

## 2. Models & model-level handling (verbatim)

| Model | Context | Max output | Reasoning | Vision |
|---|---|---|---|---|
| `command-a-plus-05-2026` | 128k | 64k | ✓ | ✓ |
| `command-a-reasoning-08-2025` | 256k | 32k | ✓ | ✗ |
| `command-a-03-2025` | 256k | 8k | ✓ | ✗ |
| `command-a-vision-07-2025` | 128k | 8k | ✗ | ✓ |
| `command-a-translate-08-2025` | 8k | 8k | ✗ | ✗ |
| `command-r7b-12-2024` | 128k | 4k | ✗ | ✗ |
| `command-r-08-2024` / `command-r-plus-08-2024` | 128k | 4k | ✗ | ✗ |

**Model-level handling we can use:**
- **Reasoning is model-gated** — only `command-a-plus`, `command-a-reasoning`, `command-a` "think before generating." Enable `thinking` (with optional token budget) **only** for those; reasoning returns as a **thinking content block** in the response, not `reasoning_content`.
- **Vision** only on `command-a-plus` / `command-a-vision` → route images there.
- **Output caps vary wildly** (4k → 64k); `command-a-plus` 64k output is unusually high. Per-model `max_output` matters for Coder budgeting.

---

## 3. Request parameters (verbatim, v2)

`model`, `messages` (role + content blocks), `tools` + `tool_choice` (`REQUIRED`/`NONE`), `documents` (citation sources), `response_format` (text / json_object + schema), `safety_mode` (`CONTEXTUAL` default / `STRICT` / `OFF`), `thinking` (enable + budget), `temperature`, **`k`**, **`p`**, `frequency_penalty`/`presence_penalty` (0.0–1.0), `max_tokens`, `seed`, `stop_sequences`, `stream`.

Response: message `content` array, `tool_calls`, **`citations`**, `finish_reason`, `usage` (billed + actual tokens).

---

## 4. Pricing

Not published in the docs section read (pricing page is separate). Track per-model on cohere.com/pricing.

---

## 5. Known drift / gaps

- 🟡 **#28 native features unused** — `documents`/`citations` (grounded RAG), `safety_mode`, and the `thinking`-block return are Cohere v2's differentiators; via the OpenAI-shaped converter they're likely dropped → Command runs as a plain chat model and reasoning blocks may not be extracted. Verify `converters/cohere` maps thinking blocks + tool_choice enum.
- 🟢 **#29 sampler mapping** — Cohere uses `k`/`p` (not `top_k`/`top_p`) and penalty range **0–1** (not −2..2); ensure the converter renames + clamps, else params are ignored or rejected.
- ℹ️ Reasoning is per-model (only `command-a*`/`-reasoning`) — don't send `thinking` to `command-r`/`r7b`/translate.

---

## 6. Sources

- Chat v2 API: https://docs.cohere.com/reference/chat
- Models: https://docs.cohere.com/docs/models
