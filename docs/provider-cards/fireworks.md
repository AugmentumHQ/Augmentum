# Fireworks AI — Provider Reference Card

> **Verbatim reference** from Fireworks' official docs (`docs.fireworks.ai`).
> **Sourced:** 2026-06-25 · **Sources:** see [§6](#6-sources).

| | |
|---|---|
| **Profile id** (`provider_profiles.py`) | `fireworks` |
| **`provider_type`** | `openai` |
| **Base URL** | `https://api.fireworks.ai/inference/v1` |
| **Endpoint** | `POST /inference/v1/chat/completions` |
| **Auth** | `Authorization: Bearer <FIREWORKS_API_KEY>` |
| **Nature** | multi-model host (open weights: DeepSeek, Qwen3, GLM, Kimi, Llama) — **feature-rich, OpenAI-compat + extensions** |

---

## 1. Augmentum wiring (what WE send)

`fireworks` is a **bare profile** — no flags set. Fireworks actually supports a lot we don't send:

| Capability Fireworks supports | We send it? | Note |
|---|---|---|
| `reasoning_effort` (rich enum, §3) | ❌ | reasoning no-op on hosted reasoning models (#20) |
| `prompt_cache_key` (cached ~80–90% cheaper) | ❌ | misses cache discount (#21) |
| `service_tier` (`auto/default/flex/priority`) | ❌ | no tier control (#21) |
| `response_format` json_schema / grammar | ✅ (default) | json_schema works |
| full sampler suite (top_k/min_p/typical_p/repetition_penalty) | partial | via raw_options |

---

## 2. Models & model-level handling (verbatim pricing, input / cached / output per 1M)

| Model | Standard | Priority | Class |
|---|---|---|---|
| DeepSeek V4 Pro | $1.74 / $0.145 / $3.48 | $2.61 / $0.218 / $5.22 | reasoning |
| DeepSeek V4 Flash | $0.14 / $0.028 / $0.28 | — | reasoning |
| Qwen 3.7 Plus | $0.40 / $0.08 / $1.60 | — | reasoning |
| Qwen 3.6 Plus | $0.50 / $0.10 / $3.00 | — | reasoning |
| GLM 5.2 | $1.40 / $0.26 / $4.40 | $2.10 / $0.39 / $6.60 | reasoning (asymmetric) |
| Kimi K2.7 Code | $0.95 / $0.19 / $4.00 | $1.425 / $0.285 / $6.00 | coding |
| Kimi K2.6 | $0.95 / $0.16 / $4.00 | $1.50 / $0.22 / $6.00 | reasoning |

**Size-tiered (generic models, input=output /1M):** <4B $0.10 · 4–16B $0.20 · >16B $0.90 · MoE ≤56B $0.50 · MoE 56.1–176B $1.20. Cached input ~80% off; batch 50% off.

**Model-level handling we can use:** reasoning models (DeepSeek/Qwen3/GLM/Kimi) → emit `reasoning_effort`; GLM 5.2 is an **asymmetric/`_STARTS_THINKING`** family (cloud — see correction #17 NVIDIA-class bug); per-model output caps live on the model listing page (read per-model).

---

## 3. Request parameters (verbatim)

| Parameter | Default | Range / Allowed |
|---|---|---|
| `max_tokens` / `max_completion_tokens` | null | alias; can't set both |
| `temperature` | null | 0–2 |
| `top_p` | null | 0–1 |
| `top_k` | null | 0–100 |
| `min_p` / `typical_p` | null | 0–1 |
| `frequency_penalty` / `presence_penalty` | null | −2 to 2 |
| `repetition_penalty` | null | 0–2 (1.0 = none) |
| `n` | 1 | 1–128 |
| `stop` | null | up to 4 sequences |
| `response_format` | null | `text`, `json_object`, `json_schema`, **`grammar`** |
| `tools` / `tool_choice` | `auto` | `auto`, `none`, `any`, `required` |
| **`reasoning_effort`** | null | `low`, `medium`, `high`, `xhigh`, `max`, `none`, `adaptive`, or integer |
| `seed` | — | deterministic |
| `prompt_cache_key` | — | KV-cache session affinity |
| `service_tier` | `default` | `auto`, `default`, `flex`, `priority` |
| `perf_metrics_in_response` | false | Fireworks-specific |

---

## 5. Known drift / gaps

- 🟡 **reasoning_effort unused** — Fireworks accepts a rich effort enum but the bare profile never sends it → thinking no-op on hosted DeepSeek/Qwen3/GLM/Kimi. Fix: enable per-model.
- 🟢 **cache + tier unused** — `prompt_cache_key` (cached ~80–90% cheaper) and `service_tier` supported but not sent.
- 🔴 **(via #17)** GLM 5.2 / DeepSeek-V4 on Fireworks are cloud asymmetric-think families → subject to the `_inside_think` content-loss bug.

---

## 6. Sources

- Chat API: https://docs.fireworks.ai/api-reference/post-chatcompletions
- Pricing: https://docs.fireworks.ai/serverless/pricing · https://fireworks.ai/pricing
