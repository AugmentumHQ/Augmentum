# Moonshot AI (Kimi) — Provider Reference Card

> **Verbatim reference** from Kimi's official docs (`platform.kimi.ai` / `platform.moonshot.ai`).
> **Sourced:** 2026-06-25 · **Sources:** see [§7](#7-sources).

| | |
|---|---|
| **Profile id** (`provider_profiles.py`) | `moonshot` |
| **`provider_type`** | `openai` |
| **Base URL** | `https://api.moonshot.ai/v1` (also Anthropic-compatible endpoint) |
| **Endpoint** | `POST /v1/chat/completions` |
| **Auth** | `Authorization: Bearer <MOONSHOT_API_KEY>` |
| **Context** | 262,144 tokens (256K) |
| **Tools** | ✅ strong — **but `tool_choice="required"` NOT supported** |

---

## 1. Augmentum wiring (what WE send)

| Augmentum field | Value | Reconciliation |
|---|---|---|
| `supports_thinking` | `True` | ✅ |
| `supports_thinking_type_toggle` | `True` | sends top-level `thinking: {"type": "enabled"\|"disabled"}` — ✅ Kimi supports this |
| `max_context` | `256_000` | ✅ (262,144) |
| `max_output` | `32_768` | ✅ K2.6 separate 32K output ceiling |
| `supports_response_format_json_schema` | _(default `True`)_ | ✅ Kimi supports Structured Output (`json_schema`) |
| `supports_prompt_cache_key` | _(default `False`)_ | 🟢 **gap** — Kimi supports `prompt_cache_key` + cache-aware billing; enabling it would unlock cache discounts (§6) |

---

## 2. Models (verbatim)

| Model id | Use | Notes |
|---|---|---|
| `kimi-k2.7-code` | coding / SWE | "most capable coding model to date"; reliable in long contexts |
| `kimi-k2.6` | general / multimodal / agent | "latest and most intelligent"; thinking + instant |
| `kimi-k2.5` | cost-sensitive / comparison | vision + text input; thinking + non-thinking |
| `kimi-k2-instruct` (older) | — | recommended temp 0.6 |

Context = **256K** (doubled from 128K at K2.5).

---

## 3. Request parameters (verbatim notes)

OpenAI-compatible Chat Completions surface, with these **Kimi-specific** behaviors:

- **`temperature`** — range **[0, 1]** (NOT OpenAI's [0, 2]). If `temperature < 0.3` **and** `n > 1` → exception. Anthropic-compat endpoint maps `real_temperature = request_temperature * 0.6`.
- **`tool_choice`** — **does NOT support `"required"`**.
- **`max_completion_tokens` / `max_tokens`** — "length of tokens you expect returned, not input+output." If `input + max_completion_tokens` > context → `invalid_request_error`. `finish_reason` = `length` or `stop`.
- **`response_format`** — `{"type":"text"}` (default) · `{"type":"json_object"}` (JSON mode, must prompt for JSON) · `{"type":"json_schema"}` (Structured Output).
- **Reasoning** — thinking mode exposes a **`reasoning_content`** field; disable via top-level `thinking: {"type":"disabled"}` (official API) or `extra_body={'chat_template_kwargs': {"thinking": False}}`.
- Also supported: `prompt_cache_key`, `safety_identifier`, `tools`, `stream`, `response_format`, `thinking`.

---

## 4. Sampling recommendations (verbatim)

| Mode | temperature | top_p |
|---|---|---|
| Thinking | 1.0 | 0.95 |
| Instant (non-thinking) | 0.6 | 0.95 |
| K2-Instruct (older) | 0.6 | — |

---

## 6. Pricing (verbatim, per 1M tokens, USD)

| Model | Input (cache hit) | Input (cache miss) | Output | Context |
|---|---|---|---|---|
| `kimi-k2.6` | $0.16 | $0.95 | $4.00 | 262,144 |
| `kimi-k2.7-code` | $0.19 | $0.95 | $4.00 | 262,144 |

Cache-aware billing for repeated context; tiered rate limits unlock with cumulative spend.

---

## 6b. Known drift / gaps

- 🟡 **`tool_choice="required"` unsupported** — sending it 400s. Verify Augmentum never emits `required` to Moonshot.
- 🟢 **temperature clamp** — Kimi caps temperature at **1.0**; per-model sampling must clamp (OpenAI defaults assume 2.0 ceiling). Also avoid `temp<0.3` with `n>1`.
- 🟢 **prompt-cache gap** — Kimi supports `prompt_cache_key` + cache-aware billing, but `moonshot` profile leaves `supports_prompt_cache_key=False` → no cache key sent → misses the cheaper cache-hit tier.

---

## 7. Sources

- Chat API: https://platform.kimi.ai/docs/api/chat
- Pricing (k2.6): https://platform.kimi.ai/docs/pricing/chat-k26
- Pricing (k2.7-code): https://platform.kimi.ai/docs/pricing/chat-k27-code
