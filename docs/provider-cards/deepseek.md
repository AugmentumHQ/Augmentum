# DeepSeek — Provider Reference Card

> **Verbatim reference** pulled from DeepSeek's official API docs. Numbers and field
> names are copied as-published; do not paraphrase when updating.
> **Sourced:** 2026-06-25 · **Sources:** see [§9](#9-sources).

| | |
|---|---|
| **Profile id** (`provider_profiles.py`) | `deepseek` |
| **`provider_type`** | `openai` (OpenAI-compat dialect) |
| **Base URL** | `https://api.deepseek.com/v1` (standard) · `https://api.deepseek.com/beta` (prefix/reasoning_content) |
| **Auth** | `Authorization: Bearer <DEEPSEEK_API_KEY>` |
| **Endpoint** | `POST /chat/completions` |
| **Vision** | ❌ none — all models text-only; API 400s on `image_url` parts |
| **Tools / function calling** | ✅ (max 128 functions) |

---

## 1. Augmentum wiring (what WE send) — reconcile against §3

From `augmentum/models/provider_profiles.py` (`deepseek` profile):

| Augmentum flag | Value | Effect on the wire |
|---|---|---|
| `post_process` | `semi` | non-leading system→user, merges consecutive same-role |
| `supports_assistant_prefix` | `True` | Continue button → `{"prefix": true}` on trailing assistant msg, rerouted to `/beta` |
| `assistant_prefix_marker` | `{"prefix": True}` | — |
| `prefix_endpoint_override` | `https://api.deepseek.com/beta` | — |
| `accepts_reasoning_content` | `True` | echoes `reasoning_content` back on prior assistant turns (required mid-tool-loop) |
| `supports_thinking_type_toggle` | `True` | sends `thinking: {"type": "enabled"\|"disabled"}` |
| `supports_response_format_json_schema` | `False` | demotes `json_schema` → `json_object` |
| `supports_vision` | `False` | attachments captioned-to-text instead of sent |
| `max_context` | `1_000_000` | ✅ matches docs |
| `max_output` | `384_000` | ✅ matches docs |
| `supports_developer_role` | (default `False`) | ✅ never sends `developer` (would 400) |
| `supports_max_completion_tokens` | (default `False`) | ✅ sends `max_tokens`, not `max_completion_tokens` |
| `supports_reasoning_effort` | (default `False`) | ⚠️ see §5 — DeepSeek DOES take `thinking.reasoning_effort` (`high`/`max`), which we do **not** currently send |

---

## 2. Models (verbatim)

| Model id | Context | Max output | Notes |
|---|---|---|---|
| `deepseek-v4-flash` | 1M tokens | 384K tokens | non-thinking + thinking modes via `thinking.type` |
| `deepseek-v4-pro` | 1M tokens | 384K tokens | — |

> `deepseek-chat` and `deepseek-reasoner` **deprecate 2026/07/24 15:59 UTC** — they were the
> non-thinking / thinking modes of v4-flash respectively.

---

## 3. Request body parameters (verbatim)

| Parameter | Type | Req | Default | Allowed / Range | Description |
|---|---|:--:|---|---|---|
| `messages` | object[] | ✅ | — | ≥ 1 | "A list of messages comprising the conversation so far" |
| `model` | string | ✅ | — | `deepseek-v4-flash`, `deepseek-v4-pro` | Model identifier |
| `thinking` | object | — | enabled | — | Controls thinking/non-thinking mode |
| `thinking.type` | string | — | `enabled` | `enabled`, `disabled` | Switches modes |
| `thinking.reasoning_effort` | string | — | `high` | `high`, `max` | "Controls the reasoning effort of the model" |
| `max_tokens` | integer | — | — | limited by context length | — |
| `temperature` | number | — | 1 | ≤ 2 | "Higher values make output more random" |
| `top_p` | number | — | 1 | ≤ 1 | Nucleus sampling |
| `response_format` | object | — | — | — | JSON output config |
| `response_format.type` | string | — | `text` | `text`, `json_object` | Output format type |
| `stop` | string / string[] | — | — | up to 16 sequences | Stop sequences |
| `stream` | boolean | — | — | — | Enable streaming |
| `stream_options.include_usage` | boolean | — | — | — | Include token usage in stream |
| `tools` | object[] | — | — | max 128 functions | "Functions the model may generate JSON inputs for" |
| `tool_choice` | object / string | — | `auto` (if tools) | `none`, `auto`, `required` | Tool invocation control |
| `logprobs` | boolean | — | — | — | Return token log probabilities |
| `top_logprobs` | integer | — | — | ≤ 20 | Top token probabilities to return |
| `user_id` | string | — | — | `[a-zA-Z0-9\-_]`, max 512 | "User identity for content safety and isolation" |

### Message object fields
- **All:** `role` (`system`/`user`/`assistant`/`tool`), `content` (string; nullable for assistant), optional `name`.
- **Assistant (Beta):** `prefix` (boolean — "Force model to start answer with supplied prefix"), `reasoning_content` (string — "Input for CoT in thinking mode").
- **Tool:** `tool_call_id` (string).

---

## 4. Response object (verbatim)

`object: chat.completion` · streaming chunks = `chat.completion.chunk` with `delta` replacing `message`.

- `id`, `created` (unix s), `model`, `system_fingerprint`
- `choices[].finish_reason`: `stop`, `length`, `content_filter`, `tool_calls`, **`insufficient_system_resource`**
- `choices[].message.role` (always `assistant`), `.content`, **`.reasoning_content`** (nullable, thinking mode)
- `choices[].message.tool_calls[]`: `id`, `type` (`function`), `function.name`, `function.arguments` (JSON string — validate before use)
- `usage`: `prompt_tokens`, `completion_tokens`, `total_tokens`, **`prompt_cache_hit_tokens`**, **`prompt_cache_miss_tokens`**, `completion_tokens_details.reasoning_tokens`

---

## 5. Reasoning / thinking

- Per-request toggle: `thinking: {"type": "enabled"|"disabled"}` (default **enabled**).
- Effort: `thinking.reasoning_effort` ∈ `high` (default), `max`. **Augmentum does not send this yet** (`supports_reasoning_effort=False`).
- On thinking-mode round-trips mid-tool-loop, prior assistant turns **must** carry `reasoning_content` back or the API 400s ("reasoning_content in thinking mode must be passed back").
- Context-cache hit/miss is automatic and surfaced in `usage.prompt_cache_*`.

---

## 6. Sampling recommendations (verbatim, DeepSeek docs)

| Use case | Temperature |
|---|---|
| Coding / Math | 0.0 |
| Data Cleaning / Data Analysis | 1.0 |
| General Conversation | 1.3 |
| Translation | 1.3 |
| Creative Writing / Poetry | 1.5 |

Platform default temperature = **1.0**. (Docs only detail `temperature`; `top_p`/penalties/`logprobs` exist in the schema above but carry no per-use-case guidance.)

---

## 7. Pricing (verbatim, per 1M tokens, USD)

| Metric | `deepseek-v4-flash` | `deepseek-v4-pro` |
|---|---|---|
| Input — cache hit | $0.0028 | $0.003625 |
| Input — cache miss | $0.14 | $0.435 |
| Output | $0.28 | $0.87 |

No off-peak/time-based discount published as of fetch date.

---

## 8. Known drift / gaps (from `docs/provider-integration-matrix.md`)

- ⚠️ `thinking.reasoning_effort` (`high`/`max`) supported by DeepSeek but not emitted by Augmentum.
- ✅ `max_context` / `max_output` / vision-off / json_schema-demote all already corrected in profile.

---

## 9. Sources

- API reference: https://api-docs.deepseek.com/api/create-chat-completion
- Pricing & models: https://api-docs.deepseek.com/quick_start/pricing
- Sampling: https://api-docs.deepseek.com/quick_start/parameter_settings
