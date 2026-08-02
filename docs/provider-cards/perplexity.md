# Perplexity (Sonar) — Provider Reference Card

> **Verbatim reference** from Perplexity's official docs (`docs.perplexity.ai`).
> **Sourced:** 2026-06-25 · **Sources:** see [§7](#7-sources).

| | |
|---|---|
| **Profile id** (`provider_profiles.py`) | `perplexity` |
| **`provider_type`** | `openai` (+ `post_process="strict"`) |
| **Base URL** | `https://api.perplexity.ai` |
| **Endpoint** | `POST /chat/completions` |
| **Auth** | `Authorization: Bearer <PPLX_API_KEY>` |
| **Nature** | **search-grounded** — every chat call can run live web/academic/SEC search |

---

## 1. Augmentum wiring (what WE send)

| Augmentum field | Value | Reconciliation |
|---|---|---|
| `post_process` | `strict` | role alternation enforced ✅ |
| `supports_thinking` | `True` | ⚠️ inert — no thinking toggle; `reasoning_effort` not enabled (#18) |
| `supports_reasoning_effort` | _(default `False`)_ | ⚠️ reasoning models can't get effort (#18) |
| `max_context` | `200_000` | sonar-pro 200K / sonar 128K |
| `max_output` | `8_000` | ✅ (param accepts up to 128000, but model cap ~8K) |

---

## 2. Models & model-level handling (verbatim)

| Model | Class | `reasoning_effort` | Search | Token $/1M (in→out) |
|---|---|---|---|---|
| `sonar` | search | — | ✅ | $1 → $1 |
| `sonar-pro` | search | — | ✅ | $3 → $15 |
| `sonar-reasoning-pro` | **reasoning** (CoT) | ✅ `minimal`/`low`/`medium`/`high` | ✅ | $2 → $8 |
| `sonar-deep-research` | **research** | ✅ | ✅ exhaustive | $2 → $8 (+citation $2, +reasoning $3) |

**Model-level handling we can use:**
- `reasoning_effort` → emit **only** for `sonar-reasoning-pro` / `sonar-deep-research` (search models ignore it). Perplexity accepts `minimal` here (unlike xAI).
- Search models (`sonar`, `sonar-pro`) → expose `search_*` controls (below); reasoning/research models → effort + longer outputs.
- `sonar-deep-research` bills **search queries + citation + reasoning tokens** separately — gate behind explicit user intent (expensive).

---

## 3. Request parameters (verbatim)

**Standard:** `model`, `messages`, `max_tokens` (1–128000), `temperature` (0–2), `top_p` (0–1), `stream`, `stream_mode` (`full`/`concise`), `stop`, `response_format` (`text`/`json_schema`), `reasoning_effort` (`minimal`/`low`/`medium`/`high`).
**NOT supported:** `top_k`, `presence_penalty`, `frequency_penalty` (absent from spec).

**Perplexity-specific (search):**

| Parameter | Allowed | Notes |
|---|---|---|
| `search_mode` | `web`, `academic`, `sec` | source of results |
| `disable_search` | bool | turn off all web search |
| `enable_search_classifier` | bool | classifier decides if search needed |
| `search_domain_filter` | string[] | limit to domains |
| `search_language_filter` | ISO 639-1[] | — |
| `search_recency_filter` | `hour`/`day`/`week`/`month`/`year` | — |
| `search_after_date_filter` / `search_before_date_filter` | `MM/DD/YYYY` | — |
| `return_images` | bool | image results |
| `return_related_questions` | bool | follow-up queries |
| `image_format_filter` / `image_domain_filter` | string[] | — |
| `language_preference` | ISO 639-1 | response language |
| `web_search_options.search_context_size` | `low`/`medium`/`high` | **drives per-request price** |
| `web_search_options.search_type` | `fast`/`pro`/`auto` | speed vs quality |
| `web_search_options.user_location` | object | geo personalization |

---

## 4. Pricing (verbatim, USD)

**Token (/1M):** Sonar $1→$1 · Sonar Pro $3→$15 · Sonar Reasoning Pro $2→$8 · Sonar Deep Research $2→$8 (+citation $2/1M, +reasoning $3/1M).

**Per 1,000 requests, by `search_context_size`:**

| Model | Low | Medium | High |
|---|---|---|---|
| Sonar | $5 | $8 | $12 |
| Sonar Pro | $6 | $10 | $14 |
| Sonar Reasoning Pro | $6 | $10 | $14 |

Deep Research adds **$5 / 1,000 search queries**.

---

## 5. Known drift / gaps

- 🟡 **reasoning_effort not sent** — profile `supports_reasoning_effort=False`; `sonar-reasoning-pro`/`sonar-deep-research` accept `minimal/low/medium/high` but never receive it. Fix: per-model enable (reasoning models only).
- 🟢 **Search surface unused** — none of the `search_*` / `web_search_options` controls or returned citations are surfaced → Perplexity runs as a plain chat model, losing its grounding value, and per-request search billing (set by `search_context_size`) is invisible.
- 🟢 **Unsupported sampler params** — Perplexity rejects/ignores `top_k`/`presence_penalty`/`frequency_penalty`; ensure they aren't emitted.

---

## 7. Sources

- Models: https://docs.perplexity.ai/getting-started/models
- Chat API: https://docs.perplexity.ai/api-reference/chat-completions-post
- Pricing: https://docs.perplexity.ai/getting-started/pricing
