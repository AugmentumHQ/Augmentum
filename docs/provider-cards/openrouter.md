# OpenRouter — Provider Reference Card

> **Verbatim reference** from OpenRouter's official docs (raw markdown via the `.md` suffix).
> **Sourced:** 2026-06-25 · **Sources:** see [§8](#8-sources).

| | |
|---|---|
| **Profile id** (`provider_profiles.py`) | `openrouter` |
| **`provider_type`** | `openai` |
| **Base URL** | `https://openrouter.ai/api/v1` |
| **Endpoint** | `POST /api/v1/chat/completions` (OpenAI-spec compatible) |
| **Auth** | `Authorization: Bearer <OPENROUTER_API_KEY>` |
| **Required-ish headers** | `HTTP-Referer`, `X-Title` (app attribution) |
| **Nature** | **Aggregator / re-router** — one key, hundreds of upstream models; **pricing is per-model** (see each model's page) |
| **Docs trick** | append `.md` to any docs URL for raw markdown; `/llms.txt` for a page index |

---

## 1. Augmentum wiring (what WE send)

From `provider_profiles.py` (`openrouter`):

| Augmentum field | Value | Reconciliation |
|---|---|---|
| `extra_headers` | `HTTP-Referer: https://augmentum.dev`, `X-Title: Augmentum` | ✅ matches OR's attribution headers |
| `supports_thinking` | `True` | generic reasoning routing |
| _(no OpenAI-family flags)_ | — | relies on the `is_openai_family_model` catch-all so a `gpt-5*` routed via OR still gets OpenAI-family fields |

⚠️ **Two verified drifts** (see [§7](#7-known-drift--gaps)) around reasoning parse + control.

---

## 2. Request parameters (verbatim)

| Parameter | Type | Default | Range / Allowed | Description |
|---|---|---|---|---|
| `temperature` | float | 1.0 | 0.0–2.0 | "influences the variety in the model's responses" |
| `top_p` | float | 1.0 | 0.0–1.0 | "limits the model's choices to a percentage of likely tokens" |
| `top_k` | integer | 0 | ≥ 0 | "limits the model's choice of tokens at each step" |
| `frequency_penalty` | float | 0.0 | −2.0–2.0 | repetition by input frequency |
| `presence_penalty` | float | 0.0 | −2.0–2.0 | "Adjusts how often the model repeats specific tokens" |
| `repetition_penalty` | float | 1.0 | 0.0–2.0 | reduces repetition from input |
| `min_p` | float | 0.0 | 0.0–1.0 | min probability relative to most likely token |
| `top_a` | float | 0.0 | 0.0–1.0 | "Consider only the top tokens with sufficiently high probabilities" |
| `seed` | integer | None | — | deterministic sampling |
| `max_tokens` / `max_completion_tokens` | integer | — | ≥ 1 | upper limit on generated tokens |
| `stop` | array | — | — | halt on specified tokens |
| `response_format` | map | — | `{ "type": "json_object" }` | "Forces the model to produce specific output format" |
| `structured_outputs` | boolean | — | — | enables JSON-schema response formatting |
| `logit_bias` | map | — | −100–100 | token-id → bias |
| `logprobs` | boolean | — | — | "returns the log probabilities of each output token" |
| `top_logprobs` | integer | — | 0–20 | most-likely tokens w/ logprobs |
| `tools` | array | — | — | OpenAI-spec tool calling |
| `tool_choice` | string / object | — | `none`, `auto`, `required`, or tool object | which tool to invoke |
| `parallel_tool_calls` | boolean | true | — | simultaneous function calling |
| `reasoning` | map | — | sub-fields: `enabled`, `exclude`, `effort`, `max_tokens` | unified reasoning control |
| `reasoning_effort` | enum | — | **`max`, `xhigh`, `high`, `medium`, `low`, `minimal`, `none`** | shorthand for `reasoning.effort` (can't conflict with it) |
| `verbosity` | enum | — | `low`, `medium`, `high`, `xhigh`, `max` | "Constrains the verbosity of the model's response" |
| `web_search_options` | map | — | — | "Configures native web search options" |

> OpenRouter "will default to listed values if certain parameters are absent" (e.g. `temperature`→1.0)
> and accepts undocumented params (forwarded to the upstream). Confirm per-model support on the
> model's provider section. `include_reasoning` is a **deprecated** alias for `reasoning.exclude`.

---

## 3. Reasoning

- Unified `reasoning` object: `{ enabled, exclude, effort, max_tokens }`.
- `reasoning.effort` / `reasoning_effort` ∈ `max`, `xhigh`, `high`, `medium`, `low`, `minimal`, `none`.
- `verbosity` for Anthropic models maps to `output_config.effort`; `xhigh` supported by Claude 4.7 Opus and later.
- **Return shape:** OpenRouter returns reasoning in a **`reasoning`** field on the message/delta (not `reasoning_content`). → see drift §7.

---

## 4. Models & pricing

Aggregator — **per-model pricing**, no flat table. Each model page (`openrouter.ai/<author>/<model>`)
lists input/output (and cache) price. Augmentum reads per-model metadata from OR's `/models`.

---

## 7. Known drift / gaps

- ✅ **Reasoning return field — already handled (audit false positive, corrected 2026-06-25)** — both parse paths already read `reasoning_content or reasoning` (streaming `_iter_stream` delta + non-streaming `_parse_openai_response`); the comment even names OpenRouter. OR reasoning is NOT dropped. The original card over-claimed; verified against committed HEAD during the reasoning fix pass.
- 🟡 **Reasoning control never sent (#4, still open)** — reasoning control only reaches OpenAI-family ids; the unified `reasoning` object is never emitted, so Anthropic/DeepSeek/Qwen/GLM routed via OR get no reasoning control. Fix: a `supports_openrouter_reasoning` path emitting `reasoning:{enabled,effort}` / `exclude`. Distinct from the return-field handling above.
- ☐ Provider-routing preferences / `transforms` / `models` (fallback array) / `route` not in the params snapshot — capture separately.

---

## 8. Sources

- Parameters: https://openrouter.ai/docs/api/reference/parameters (`.md` variant)
- API overview: https://openrouter.ai/docs/api/reference/overview
- Create chat completion: https://openrouter.ai/docs/api/api-reference/chat/send-chat-completion-request
- Drift cross-ref: `docs/provider-integration-matrix.md`
