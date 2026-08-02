# Groq — Provider Reference Card

> **Verbatim reference** from Groq's official docs (`console.groq.com/docs`, `groq.com/pricing`).
> **Sourced:** 2026-06-25 · **Sources:** see [§6](#6-sources).

| | |
|---|---|
| **Profile id** (`provider_profiles.py`) | `groq` |
| **`provider_type`** | `openai` |
| **Base URL** | `https://api.groq.com/openai/v1` |
| **Endpoint** | `POST /openai/v1/chat/completions` |
| **Auth** | `Authorization: Bearer <GROQ_API_KEY>` |
| **Also** | STT (Whisper) on the same key — see speech section |
| **Body validation** | strict — 400s on unknown keys |

---

## 1. Augmentum wiring (what WE send)

| Augmentum field | Value | Reconciliation |
|---|---|---|
| _(bare profile — no flags)_ | — | ⚠️ no reasoning flags → `reasoning_effort`/`reasoning_format` **never sent** → thinking toggle is a no-op on gpt-oss/qwen (§5, #15) |

---

## 2. Models (verbatim — production)

| Model id | Context | Max completion |
|---|---|---|
| `llama-3.1-8b-instant` | 131,072 | 131,072 |
| `llama-3.3-70b-versatile` | 131,072 | 32,768 |
| `openai/gpt-oss-120b` | 131,072 | 65,536 |
| `openai/gpt-oss-20b` | 131,072 | 65,536 |
| `qwen/qwen3-32b` | (preview) | — |

STT: `whisper-large-v3`, `whisper-large-v3-turbo`.

---

## 3. Reasoning (verbatim)

**`reasoning_effort`:**
- **Qwen 3.6 27B:** `none` (disable), `default` (enable)
- **GPT-OSS 20B / 120B:** `low`, `medium`, `high`

**`reasoning_format`** (non-GPT-OSS / Qwen):
- `parsed` — reasoning in a dedicated `message.reasoning` field
- `raw` — reasoning inside `<think>` tags in main content
- `hidden` — final answer only
- Defaults to `raw` or `parsed` when JSON mode / tool use is on; **setting `raw` explicitly with those → 400**.

**GPT-OSS models:** `reasoning_format` **unsupported**; reasoning is in a `reasoning` field by default; toggle with `include_reasoning` (boolean).

---

## 4. Pricing (verbatim, USD)

| Model | Input /1M | Output /1M |
|---|---|---|
| `llama-3.1-8b-instant` | $0.05 | $0.08 |
| `llama-3.3-70b-versatile` | $0.59 | $0.79 |
| `openai/gpt-oss-20b` | $0.075 | $0.30 |
| `openai/gpt-oss-120b` | $0.15 | $0.60 |
| `qwen/qwen3-32b` | $0.29 | $0.59 |
| `whisper-large-v3` (STT) | $0.111 / hour transcribed (217× speed) | — |
| `whisper-large-v3-turbo` (STT) | $0.04 / hour transcribed (228× speed) | — |

"Audio is billed at a minimum of 10s per request."

---

## 5. Known drift / gaps

- ✅ **Reasoning effort — FIXED (R4, 2026-06-25)** — `reasoning_via_groq_params` + `_groq_reasoning_params` now emit per-model `reasoning_effort`: gpt-oss → `low/medium/high` (UI `minimal`/`xhigh`/`max` clamped); qwen3 → `none`/`default` driven by the think toggle. `reasoning_format` deliberately NOT sent (`raw`+JSON/tools→400; default works).
- ✅ **Reasoning return field — already handled (audit false positive)** — Groq's `message.reasoning`/`delta.reasoning` (and inline `<think>`) are read by the existing `reasoning_content or reasoning` fallback + `ThinkingStreamBuffer`. No drop. (`gpt-oss` is also NOT accidentally OpenAI-classed.)
- ☐ Groq STT (Whisper) needs its own preset (matrix coverage gap) — see speech cards.

---

## 6. Sources

- Reasoning: https://console.groq.com/docs/reasoning
- Models: https://console.groq.com/docs/models
- Pricing: https://groq.com/pricing
