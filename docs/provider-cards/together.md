# Together AI — Provider Reference Card

> **Verbatim reference** from Together's official docs (`docs.together.ai`).
> **Sourced:** 2026-06-25 · **Sources:** see [§6](#6-sources). (Together also serves image models — see image cards.)

| | |
|---|---|
| **Profile id** (`provider_profiles.py`) | `together` |
| **`provider_type`** | `openai` (bare profile) |
| **Base URL** | `https://api.together.xyz/v1` |
| **Endpoint** | `POST /v1/chat/completions` |
| **Auth** | `Authorization: Bearer <TOGETHER_API_KEY>` |
| **Nature** | Multi-model open-weights host (DeepSeek, Qwen3, GLM, Kimi, Llama). OpenAI-compat passthrough; reasoning behavior is the **upstream model's**. |

---

## 1. Augmentum wiring (what WE send)

Bare profile — no flags. OpenAI-compat passthrough. Together supports the full OpenAI sampler suite (`temperature`, `top_p`, `top_k`, `repetition_penalty`, `min_p`, `presence/frequency_penalty`, `n`, `stop`, `response_format` json/json_schema, `tools`/`tool_choice`, `seed`, `logprobs`). Reasoning is **not** controlled by us → inherits the cross-provider reasoning class (#30).

---

## 2. Models & model-level handling (verbatim, /1M in→out)

| Model | Context | Input | Output | Class |
|---|---|---|---|---|
| DeepSeek-V4-Pro | **512K** | $1.74 | $3.48 | reasoning |
| Qwen3.6 Plus | **1M** | $0.50 | $3.00 | reasoning |
| Qwen3.7 Max | — | $1.25 | $3.75 | reasoning |
| Qwen3.5 9B | 262K | $0.17 | $0.25 | small reasoning |
| GLM-5.2 | 262K | $1.40 | $4.40 | reasoning (asymmetric) |
| GLM-5.1 | 202K | $1.40 | $4.40 | reasoning (asymmetric) |
| Kimi K2.7 Code | 262K | $0.95 | $4.00 | coding |
| Kimi K2.6 | 262K | $1.20 | $4.50 | reasoning |
| Llama 3.3 70B | 131K | $1.04 | $1.04 | dense |

Per-token billing, no minimums; **batch up to 50% off**.

**Model-level handling we can use:**
- Reasoning models (DeepSeek-V4/Qwen3/GLM/Kimi) return reasoning in **`reasoning_content`** (Augmentum reads ✅) — but **no effort/toggle control** is sent.
- GLM-5.x / DeepSeek-V4 are **asymmetric-think** (`_STARTS_THINKING_FAMILIES`) on Together → subject to the #17 content-loss bug when they return plain content.
- Context windows are huge here (DeepSeek-V4 512K, Qwen3.6 1M) — `max_context` is per-model; don't cap globally.

---

## 3. Request parameters (verbatim)

Standard OpenAI: `model`, `messages`, `max_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `min_p`, `presence_penalty`, `frequency_penalty`, `n`, `stop`, `stream`, `logprobs`, `echo`, `seed`, `response_format` (`json_object`/`json_schema`), `tools`/`tool_choice`, `safety_model` (Together-specific moderation model id).

---

## 4. Pricing (verbatim)

Per-model (§2). Size-tiered for the long tail; batch −50%. See together.ai/pricing for the dedicated/serverless split.

---

## 5. Known drift / gaps

- 🟡 **#30 reasoning not controlled** — bare profile sends no reasoning toggle; hosted DeepSeek/Qwen3/GLM/Kimi reason by default with no Augmentum control (can't reduce latency by disabling, can't raise effort).
- 🔴 **(via #17)** GLM-5.x / DeepSeek-V4 on Together are asymmetric-think families → content-loss bug when they return plain content.
- 🟢 `safety_model` moderation hook unused.

---

## 6. Sources

- Chat overview: https://docs.together.ai/docs/chat-overview
- Serverless models + pricing: https://docs.together.ai/docs/serverless-models
