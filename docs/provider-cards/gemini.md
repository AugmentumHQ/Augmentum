# Google Gemini — Provider Reference Card

> **Verbatim reference** from Google's official docs (`ai.google.dev/gemini-api`).
> **Sourced:** 2026-06-25 · **Sources:** see [§6](#6-sources).
> **Native backend** — own adapter (`models/adapters/gemini.py` + `converters/gemini.py`), bypasses the OpenAI-compat profiles.

| | |
|---|---|
| **`provider_type`** | `gemini` (native generateContent) |
| **Base URL** | `https://generativelanguage.googleapis.com` |
| **Endpoint** | `POST /v1beta/{model=models/*}:generateContent` (+ `:streamGenerateContent`) |
| **Auth** | `?key=<GEMINI_API_KEY>` query param **or** `Authorization: Bearer <KEY>` |
| **API version** | `v1beta` |

---

## 1. Augmentum wiring (what WE send) — **well-wired, current**

The native converter does proper **per-generation thinking** with the right field per family:

| Behavior | Implementation | Status |
|---|---|---|
| **Gemini 3 thinking** | `gemini-3.*flash` / `gemini-3.*pro` → `thinkingConfig:{thinkingLevel, includeThoughts:true}` (level mapped from effort) | ✅ catches 3 / 3.1 / 3.5 / Flash-Lite |
| **Gemini 2.5 thinking** | `gemini-2.5.*flash` → `thinkingBudget` capped **[0, 24576]**; `gemini-2.5.*pro` → capped **[128, 32768]** | ✅ matches per-model ranges |
| **includeThoughts** | always `true` when thinking on → thoughts streamed as `thinking_delta` | ✅ |
| **maxOutputTokens / temp / topP / topK / stop** | mapped into `generationConfig` | ✅ |

⚠️ One subtlety: `gemini-3.*flash` matches **Flash-Lite** too → it takes the Flash `thinkingLevel` path (fine; verify Flash-Lite honors `thinkingLevel`). The Gemini-3 regexes use an **unescaped `.`** (`gemini-3.*flash`) — harmless for real model ids but slightly loose (#31).

---

## 2. Models & model-level handling (verbatim)

| Model | Thinking control | Status |
|---|---|---|
| **Gemini 3.1 Pro** (preview) | `thinkingLevel` (low/high) | newest Pro |
| **Gemini 3.5 Flash** (stable) | `thinkingLevel` | newest Flash |
| **Gemini 3 Flash** (preview) | `thinkingLevel` | — |
| **Gemini 3.1 Flash-Lite** (stable) | `thinkingLevel` (via Flash path) | — |
| **Gemini 2.5 Pro** (stable) | `thinkingBudget` [128, 32768] | — |
| **Gemini 2.5 Flash** (stable) | `thinkingBudget` [0, 24576] | — |
| **Gemini 2.5 Flash-Lite** (stable) | `thinkingBudget` (Flash path) | — |

**Model-level handling we can use:**
- **Gemini 3.x → `thinkingLevel` string** (low/high), **NOT** `thinkingBudget` int. Augmentum maps effort→level ✅. Don't send a budget int to Gemini 3.
- **Gemini 2.5 → `thinkingBudget` int** with **different caps per tier** (Flash 24576 vs Pro 32768; `0` disables on Flash, min 128 on Pro). ✅ enforced.
- `includeThoughts:true` required to receive reasoning (else thoughts hidden).

---

## 3. Request parameters (verbatim, `generationConfig`)

`maxOutputTokens`, `temperature` (0.0–2.0), `topP`, `topK`, `stopSequences`, `responseMimeType` (`application/json`), `responseSchema`, `thinkingConfig:{thinkingBudget | thinkingLevel, includeThoughts}`.
**safetySettings[]:** category ∈ `HARM_CATEGORY_HATE_SPEECH` / `_HARASSMENT` / `_SEXUALLY_EXPLICIT` / `_DANGEROUS_CONTENT` / `_CIVIC_INTEGRITY`; threshold ∈ `BLOCK_ONLY_HIGH` / `BLOCK_MEDIUM_AND_ABOVE` / `BLOCK_LOW_AND_ABOVE` / `BLOCK_NONE`.

---

## 4. Pricing (verbatim, /1M tokens; tiered by context where shown)

| Model | Input | Output | Cache |
|---|---|---|---|
| Gemini 3.5 Flash | $1.50 | $9.00 | $0.15 (+$1.00/1M/hr) |
| Gemini 3.1 Pro | $2.00 (≤200k) / $4.00 (>200k) | $12.00 / $18.00 | $0.20 / $0.40 (+$4.50/1M/hr) |
| Gemini 3.1 Flash-Lite | $0.25 (txt/img/vid) / $0.50 (audio) | $1.50 | $0.025 / $0.05 |
| Gemini 2.5 Pro | $1.25 (≤200k) / $2.50 (>200k) | $10.00 / $15.00 | $0.125 / $0.25 |
| Gemini 2.5 Flash | $0.30 (txt/img/vid) / $1.00 (audio) | $2.50 | $0.03 / $0.10 |
| Gemini 2.5 Flash-Lite | $0.10 / $0.30 (audio) | $0.40 | $0.01 / $0.03 |

**Pro tiers surcharge >200k-token context** — cost tracking should account for the input>200k jump.

---

## 5. Known drift / gaps

- 🟢 **#31 loose Gemini-3 regex** — `gemini-3.*flash`/`gemini-3.*pro` use an unescaped `.`; harmless for real ids but should be `gemini-3[.\d-]*`. Verify Flash-Lite honors `thinkingLevel` (it routes via the Flash path).
- 🟢 **safetySettings unused** — Augmentum sends no `safetySettings`; defaults may block content the user expects (or vice-versa). Consider exposing.
- 🟢 **>200k context price tier** not surfaced for Pro models (input/output ~2× above 200k).

---

## 6. Sources

- generateContent API: https://ai.google.dev/api/generate-content
- Models: https://ai.google.dev/gemini-api/docs/models
- Pricing: https://ai.google.dev/gemini-api/docs/pricing
- Thinking: https://ai.google.dev/gemini-api/docs/thinking
