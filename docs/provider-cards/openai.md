# OpenAI — Provider Reference Card

> **Verbatim reference** from OpenAI's official docs (`developers.openai.com`). Numbers/field
> names copied as-published. **Sourced:** 2026-06-25 · **Sources:** see [§9](#9-sources).

| | |
|---|---|
| **Profile id** (`provider_profiles.py`) | `openai` |
| **`provider_type`** | `openai` (native OpenAI-family — drives the `is_openai_family_model` catch-all) |
| **Base URL** | `https://api.openai.com/v1` |
| **Auth** | `Authorization: Bearer <OPENAI_API_KEY>` |
| **Endpoint** | `POST /v1/chat/completions` (Augmentum's path) · OpenAI also offers `/v1/responses` |
| **Vision** | ✅ "All latest OpenAI models support text and image input … and vision" |
| **Tools / function calling** | ✅ |

> ⚠️ **The model-id catch-all:** any provider serving a `gpt-5* / gpt-4.1* / gpt-4o* / o1* /
> o3* / o4* / codex*` id gets the full OpenAI-family field set below via
> `is_openai_family_model`, regardless of its own profile.

---

## 1. Augmentum wiring (what WE send) — reconcile against §3/§5

From `provider_profiles.py` (`openai` profile):

| Augmentum flag | Value | Effect / reconciliation |
|---|---|---|
| `supports_max_completion_tokens` | `True` | sends `max_completion_tokens` (not legacy `max_tokens`) ✅ |
| `supports_reasoning_effort` | `True` | sends `reasoning_effort` ✅ |
| `supports_reasoning_effort_minimal` | `True` | allows `minimal` (no demotion) ✅ |
| `supports_reasoning_summary` | `True` | `reasoning.summary` — **Responses API only**; Chat Completions ignores it |
| `supports_developer_role` | `True` | `system`→`developer` rewrite — now actually gated on `is_openai_family_model`, so fires only for gpt-5/o-series ids ✅ |
| `supports_prompt_cache_key` | `True` | sends `prompt_cache_key` / `prompt_cache_retention` ✅ |
| `supports_service_tier` | `True` | sends `service_tier` ✅ |
| `max_context` | `400_000` | ⚠️ **DRIFT** — docs list gpt-5.5 / gpt-5.4 at **1M** context (§2). Coder compactor under-uses the window. |
| `max_output` | `128_000` | ✅ matches docs |

---

## 2. Models (verbatim)

| Model id | Context | Max output | Reasoning | Vision |
|---|---|---|---|---|
| `gpt-5.5` | 1M | 128K | Yes | Yes |
| `gpt-5.5-pro` | (pro tier) | — | Yes | Yes |
| `gpt-5.4` | 1M | 128K | Yes | Yes |
| `gpt-5.4-mini` | 400K | 128K | Yes | Yes |
| `gpt-5.4-nano` | (not specified in snapshot) | — | — | Yes |
| `gpt-5.4-pro` | (pro tier) | — | Yes | Yes |

> "GPT-5.5 represents a new class of intelligence for coding and professional work"; mini/nano
> are "lower-latency, lower-cost workloads." Legacy families (`gpt-4o`, `gpt-4.1`, o-series) are
> referenced but not spec'd in the current models snapshot — **TODO: pull legacy specs**.

---

## 3. Request body parameters (verbatim — `POST /v1/chat/completions`)

| Parameter | Type | Req | Allowed / notes |
|---|---|:--:|---|
| `messages` | array | ✅ | roles: `system`, **`developer`**, `user`, `assistant`, `tool` |
| `model` | string | ✅ | model id |
| `max_completion_tokens` | integer | — | max tokens in the completion (current) |
| `max_tokens` | integer | — | **legacy** parameter for response length |
| `reasoning_effort` | string | — | **`none`, `minimal`, `low`, `medium`, `high`, `xhigh`** (default varies; `gpt-5.5` = `medium`). "Some models support only a subset — check the model page." |
| `temperature` | number | — | range 0–2, default 1 |
| `top_p` | number | — | range 0–1 |
| `frequency_penalty` | number | — | range −2 to 2 |
| `presence_penalty` | number | — | range −2 to 2 |
| `n` | integer | — | number of choices |
| `stop` | string / array | — | stop sequences |
| `stream` | boolean | — | — |
| `stream_options` | object | — | incl. `include_usage` |
| `response_format` | object | — | structured output via `json_schema` |
| `tools` | array | — | function/tool definitions |
| `tool_choice` | — | — | incl. `allowed_tools` configuration |
| `logprobs` | boolean | — | — |
| `top_logprobs` | integer | — | number of top alternatives |
| `seed` | integer | — | reproducible outputs |
| `store` | boolean | — | whether to store the completion |
| `metadata` | object | — | custom metadata |
| `modalities` | array | — | e.g. `"text"`, `"audio"` |
| `audio` | object | — | output `format` + `voice` |
| `service_tier` | string | — | **`auto`, `default`, `flex`, `scale`, `priority`** |
| `prompt_cache_key` | string | — | identifier for prompt caching |

---

## 4. Reasoning (verbatim)

- `reasoning_effort` ∈ `none`, `minimal`, `low`, `medium`, `high`, `xhigh`. Default per-model (`gpt-5.5` = `medium`, "the best starting point for gpt-5.5's full balance of quality, reliability and performance").
- **Reasoning tokens** "are not visible via the API" but "still occupy space in the model's context window and are billed as output tokens." Count exposed at `usage.output_tokens_details.reasoning_tokens` (Responses API) / `usage.completion_tokens_details.reasoning_tokens` (Chat Completions).
- "Reserve at least 25,000 tokens for reasoning and outputs when you start experimenting."
- **Reasoning summary:** `summary` = `"auto"` / `"concise"` / `"detailed"`; appears in the `reasoning` output item's `summary` array (Responses API).

---

## 5. `developer` role

OpenAI reasoning models (GPT-5.x / o-series) accept `role: "developer"` in place of `system`;
both work, mixing them in one request raises a warning. Augmentum rewrites `system`→`developer`
**only when the model id is OpenAI-family** (the fix that stops the rewrite leaking onto non-OpenAI
backends served through a borrowed `openai` profile). *(The reasoning-guide page did not restate
the system↔developer relationship; this reflects OpenAI's role spec + our `openai_compat.py:391`.)*

---

## 6. Caching

OpenAI caches prefixes ≥1024 tokens automatically; a stable `prompt_cache_key` raises hit rate
(~10× cheaper cached input, ~80% TTFT reduction). Hit/miss tokens surface in `usage` cache fields.

---

## 7. Pricing (verbatim, per 1M tokens, USD — standard tier)

| Model | Input | Cached input | Output |
|---|---|---|---|
| `gpt-5.5` | $5.00 | $0.50 | $30.00 |
| `gpt-5.5-pro` | $30.00 | — | $180.00 |
| `gpt-5.4` | $2.50 | $0.25 | $15.00 |
| `gpt-5.4-mini` | $0.75 | $0.075 | $4.50 |
| `gpt-5.4-nano` | $0.20 | $0.02 | $1.25 |
| `gpt-5.4-pro` | $30.00 | — | $180.00 |

> "OpenAI is winding down the fine-tuning platform." Legacy (o-series / gpt-4.1 / gpt-4o) pricing
> not in current snapshot — **TODO**.

---

## 8. Known drift / gaps

- ⚠️ Profile `max_context=400_000` vs docs **1M** for gpt-5.5/gpt-5.4 — under-utilizes Coder window.
- ☐ Legacy model specs + pricing (o-series, gpt-4.1, gpt-4o) not yet captured.
- ℹ️ `reasoning_effort` now includes `none` and `xhigh`; our pass-through forwards user value, so both reach the wire for `openai` (only the `minimal→low` demotion is gated, and `openai` allows `minimal`).

---

## 9. Sources

- API reference (chat create): https://developers.openai.com/api/docs/api-reference/chat/create
- Models: https://developers.openai.com/api/docs/models
- Reasoning guide: https://developers.openai.com/api/docs/guides/reasoning
- Pricing: https://developers.openai.com/api/docs/pricing
