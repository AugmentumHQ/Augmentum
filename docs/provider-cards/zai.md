# Z.AI (GLM) — Provider Reference Card

> **Verbatim reference** from Z.AI's official docs (`docs.z.ai`) + verified provider listings.
> **Sourced:** 2026-06-25 · **Sources:** see [§6](#6-sources).

| | |
|---|---|
| **Profile id** (`provider_profiles.py`) | `zai` |
| **`provider_type`** | `openai` |
| **Base URL** | `https://api.z.ai/api/paas/v4` |
| **Endpoint** | `POST /chat/completions` |
| **Auth** | `Authorization: Bearer <ZAI_API_KEY>` (+ `Accept-Language: en-US`) |
| **Nature** | First-party GLM (ZhipuAI) — **asymmetric-think** family (`glm4x`), MoE 357–358B / 32B active, text + (4.7/5V) vision. |

---

## 1. Augmentum wiring (what WE send) — **the best-wired profile in the OpenAI-compat batch**

| Augmentum field | Value | Reconciliation |
|---|---|---|
| `supports_thinking` | `True` | ✅ |
| `supports_thinking_type_toggle` | `True` | ✅ **correctly emits `thinking:{type:enabled/disabled}`** — the one profile that actually controls reasoning (most others never send anything). |
| `max_context` | `200_000` | ✅ (actual 202,752) |
| `max_output` | `131_072` | ✅ (some Coding-plan deployments silently cap at 98,304) |
| `extra_headers` | `Accept-Language: en-US` | ✅ forces English system strings |
| _(no temp clamp)_ | — | 🟡 GLM temperature is **0–1**, top_p **0.01–1** (not 0–2) → out-of-range sampling rejected (#26) |
| _(only type toggle)_ | — | 🟢 `reasoning_effort` granularity (max…none) supported but unused (#27) |

---

## 2. Models & model-level handling (verbatim)

| Model | Context | Max output | Modality | Notes |
|---|---|---|---|---|
| GLM-4.7 | 200K (202,752) | 128K (131,072) | text **+ image** | flagship; thinking preserved across turns, per-request toggle; MIT open-weight |
| GLM-4.6 | 200K (202,752) | 131,072 | text only | MoE 357B/32B; thinking default ON |
| GLM-5.2 / 5.1 / 5 | 128K | 128K | text | newest line |
| GLM-5V-Turbo / GLM-4.5V | 128K / — | 128K / 16K | **vision** | **returns `<think></think>` and `<\|begin_of_box\|><\|end_of_box\|>` tags in content** — different parse path from `reasoning_content` |
| GLM-4.5 | — | 96K | text | thinking type=enabled default |

**Model-level handling we can use:**
- **Reasoning:** `thinking={"type":"enabled"|"disabled"}` (default enabled, GLM-4.5+). When enabled, reasoning streams in **`delta.reasoning_content`** (separate from `delta.content`) — Augmentum reads this ✅.
- **`reasoning_effort`** (`max/high/medium/low/minimal/none`) also accepted on GLM-4.5+ → finer control than the binary toggle we send (#27).
- **GLM-4.5V / 5V** put reasoning **in-band** as `<think>`/`<|begin_of_box|>` tags in `content`, not `reasoning_content` → vision path needs tag extraction, not field read.
- **Tools:** native `function`, **`web_search`**, and **`retrieval`** tool types (grounding built in).

---

## 3. Request parameters (verbatim)

| Parameter | Range / Allowed |
|---|---|
| `model`, `messages` | required |
| `temperature` | **0.0–1.0** (not 0–2) |
| `top_p` | **0.01–1.0** |
| `max_tokens` | 1–131,072 |
| `stream` | default false |
| `thinking` | `{type: enabled\|disabled}` (default enabled) |
| `reasoning_effort` | `max`, `high`, `medium`, `low`, `minimal`, `none` |
| `tools` | `function`, `web_search`, `retrieval` |
| `tool_choice` | `auto` (default) |
| `response_format` | `text`, `json_object` |

---

## 4. Pricing (verbatim, USD /1M, first-party upstream)

| Model | Input | Output |
|---|---|---|
| GLM-4.7 | $0.60 | $2.20 (OpenRouter resells at $0.40 / $1.75) |
| GLM-4.6 | $0.60 | $2.20 (median across providers ≈ $0.55 / $2.20) |

---

## 5. Known drift / gaps

- 🟡 **#26 temperature/top_p range** — GLM is **0–1** temp, **0.01–1** top_p (half OpenAI's range); per-model sampling must clamp for `zai` (same class as Moonshot #13).
- 🟢 **#27 reasoning_effort unused** — only the binary `thinking` type toggle is sent; GLM-4.5+ also accepts `reasoning_effort` (max…none) for finer control.
- 🔴 **(via #17)** GLM-4.x is `_STARTS_THINKING_FAMILIES` (`glm4x`). When `thinking:disabled` (or a deployment that returns plain content with no leading `</think>` and no `reasoning_content`), the cloud `_inside_think=True` init **routes the whole answer into the thinking channel**. Z.AI is a prime trigger of the content-loss bug.
- 🟢 GLM-4.5V/5V in-band `<think>`/`<|begin_of_box|>` tags need content-side extraction (not the `reasoning_content` field).

---

## 6. Sources

- Chat API: https://docs.z.ai/api-reference/llm/chat-completion
- GLM-4.6 overview (`thinking={"type":"enabled"}`, `reasoning_content` streaming): https://docs.z.ai/guides/llm/glm-4.6
- GLM-4.7 / 4.6 specs + pricing: https://openrouter.ai/z-ai/glm-4.7 · https://openrouter.ai/z-ai/glm-4.6 · https://www.together.ai/models/glm-4-6
