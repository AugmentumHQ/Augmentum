# Thin OpenAI-Compat Aggregators — Combined Reference Card

> **Verbatim reference** for six **multi-model reseller/aggregator** profiles that expose an
> OpenAI-compatible surface and re-serve *other vendors'* models. They carry almost no
> provider-specific handling — model behavior (reasoning shape, caps, sampler quirks) is
> **inherited from the upstream model**, so the per-model handling captured in the dedicated
> cards (DeepSeek, Z.AI/GLM, Qwen via Fireworks, etc.) applies transitively.
>
> **Sourced:** 2026-06-25 · Sources per row in [§4](#4-sources).
> Two of the six surfaced real, actionable handling → corrections #22–#24.

---

## 1. The six at a glance (verbatim wiring from `provider_profiles.py`)

| Provider | id | Base URL | Profile flags WE set | Nature |
|---|---|---|---|---|
| **Pollinations** | `pollinations` | `https://gen.pollinations.ai/v1` | `supports_thinking=True`, `model_list_url=…/text` | free/anon tier; re-serves openai/mistral/searchgpt |
| **AIML API** | `aimlapi` | `https://api.aimlapi.com/v1` | `extra_headers={Content-Type: application/json}` | 150+ models (GPT/Claude/Gemini/Llama/DeepSeek/Qwen), pure OpenAI-SDK passthrough |
| **ElectronHub** | `electronhub` | `https://api.electronhub.ai/v1` | _(bare)_ | multi-vendor reseller, OpenAI-compat passthrough |
| **Chutes AI** | `chutes` | `https://llm.chutes.ai/v1` | `supports_thinking=True` | decentralized serverless inference (open-weights), per-model behavior |
| **NanoGPT** | `nanogpt` | `https://nano-gpt.com/api/v1` | _(bare; bearer auth — `x-api-key` caused silent 401s)_ | pay-as-you-go / crypto; **model-suffix feature system** |
| **SiliconFlow** | `siliconflow` | `https://api.siliconflow.com/v1` | `model_list_params={type: text}` | 60+ open-weights (DeepSeek/Qwen/GLM/Kimi/MiniMax) |

All six: `provider_type=openai`, `Authorization: Bearer <key>`, `POST /chat/completions`, OpenAI request/response schema.

---

## 2. Per-provider notes (only what differs from plain OpenAI-compat)

### Pollinations
- **Auth:** free, no signup for the anon tier (1 req / 15 s); `referrer` param or `auth.pollinations.ai` registration lifts limits.
- **Reasoning:** `reasoning_effort` ∈ `minimal` / `low` / `medium` (default) / `high` — supported, but profile sets only `supports_thinking` (no effort emit) → **#22**.
- **Non-standard params:** `seed`, `temperature` (0.0–**3.0**, wider than OpenAI's 2.0), `referrer`, `private` (hide from public feed).
- **Models:** `openai`, `openai-fast`, `openai-reasoning`, `mistral`, `searchgpt` (named aliases, not vendor model ids).
- **⚠ Base-URL drift (#23):** official text endpoint is documented as `https://text.pollinations.ai/openai`; profile points at `https://gen.pollinations.ai/v1`. Verify the configured host still resolves before trusting this profile.

### SiliconFlow
- **Reasoning (real handling — #24):** `enable_thinking` (bool, default `true` for supported models), `thinking_budget` (128–32 768, default 4096 CoT tokens), and reasoning returned in **`reasoning_content`** (DeepSeek-R1 family). Bare profile sends neither toggle nor budget → thinking is a no-op / uncapped.
- **`min_p`** supported **for Qwen3 only**.
- `tools` up to 128 functions; `response_format` supported.

### NanoGPT
- **Model-suffix feature system:** append suffixes to the model id for `:web` search, memory, PII redaction, caching, reasoning-visibility, and thinking variants. None of these are surfaced by Augmentum → users can't reach NanoGPT's headline features (low-value gap, not a bug).
- **Extended Thinking (Reasoning):** documented control across OpenAI-compat endpoints; inherits the same "never-sent" reasoning-control class as DeepSeek/Kimi.
- Bearer auth confirmed (a prior `x-api-key` override → silent 401s; fixed 2026-06-15).

### Chutes AI / AIML API / ElectronHub
- **Pure passthrough.** No provider-specific request handling beyond OpenAI-compat. Reasoning/vision/tool support and caps are entirely a function of *which upstream model* is selected — consult that model's vendor card.
- Chutes sets `supports_thinking=True` (decentralized open-weights, many reasoning models); AIML/ElectronHub bare.
- AIML advertises 4K→2M context across its catalog (xAI Grok at the top) — `max_context` is meaningless at the profile level; it's per-model.

---

## 3. Known drift / gaps (→ CORRECTIONS #22–#24)

- 🟡 **#22 Pollinations** — `reasoning_effort` (`minimal/low/medium/high`) supported but never sent (only `supports_thinking` set).
- 🟢 **#23 Pollinations** — profile base URL (`gen.pollinations.ai/v1`) differs from documented text endpoint (`text.pollinations.ai/openai`); verify host.
- 🟡 **#24 SiliconFlow** — `enable_thinking` + `thinking_budget` + `reasoning_content` return unused → reasoning toggle no-op and CoT uncapped on DeepSeek/Qwen/GLM.
- 🟢 (NanoGPT) model-suffix features + extended-thinking control unsurfaced — tracked, low value.
- ℹ️ All six inherit the **#17 cloud `_inside_think` content-loss bug** whenever they serve a `_STARTS_THINKING_FAMILIES` model (GLM-4.x, DeepSeek-V4, Qwen3, MiniMax, EXAONE) that returns plain content.

---

## 4. Sources

- Pollinations: https://github.com/pollinations/pollinations/blob/master/APIDOCS.md
- SiliconFlow: https://docs.siliconflow.com/en/api-reference/chat-completions/chat-completions
- NanoGPT: https://docs.nano-gpt.com/llms.txt
- AIML API: https://docs.aimlapi.com/api-references/text-models-llm
- Chutes: https://docs.chutes.ai/
- ElectronHub: https://api.electronhub.ai (OpenAI-compat passthrough; minimal public docs)
