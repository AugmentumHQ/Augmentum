# NVIDIA NIM (build.nvidia.com) — Provider Reference Card

> **Verbatim reference** from NVIDIA's official NIM docs (`docs.api.nvidia.com`, `build.nvidia.com`)
> + verified model examples. **Sourced:** 2026-06-25 · **Sources:** see [§6](#6-sources).

| | |
|---|---|
| **Profile id** (`provider_profiles.py`) | `nvidia` |
| **`provider_type`** | `openai` (+ `post_process="semi"`) |
| **Base URL** | `https://integrate.api.nvidia.com/v1` |
| **Endpoint** | `POST /v1/chat/completions` |
| **Auth** | `Authorization: Bearer <NVIDIA_API_KEY>` |
| **Nature** | NVIDIA-hosted **NIM** microservices re-serving open-weights (Nemotron, DeepSeek, Qwen, GLM, Kimi, Llama). OpenAI-compat **+ non-standard reasoning control**. |

---

## 1. Augmentum wiring (what WE send)

| Augmentum field | Value | Reconciliation |
|---|---|---|
| `post_process` | `semi` | ✅ NIM rejects any system message after position 0 ("System message must be at the beginning."). `semi` converts non-leading system→user + merges same-role runs. Matches SillyTavern's NVIDIA handling. |
| _(no reasoning flags)_ | — | 🔴 **NIM reasoning is `chat_template_kwargs`, NOT `reasoning_effort`** → bare profile sends neither → DeepSeek V4 reasoning models **hang indefinitely** (#25); Nemotron/Qwen thinking uncontrolled. |

---

## 2. Reasoning control — **per-model, non-standard** (verbatim from NVIDIA examples)

NIM does **not** use OpenAI's `reasoning_effort`. It uses a non-standard **`chat_template_kwargs`** object (placed at the JSON root for curl, or in `extra_body` for the OpenAI SDK), and the mechanism differs by family:

| Model family | How to toggle thinking | Verbatim |
|---|---|---|
| **DeepSeek V4** (`deepseek-v4-flash`, `deepseek-v4-pro`) | `chat_template_kwargs:{thinking:true, reasoning_effort:"high"}`; disable `{thinking:false}` | NVIDIA example: `temperature=1, top_p=0.95, max_tokens=16384, extra_body={"chat_template_kwargs":{"thinking":True,"reasoning_effort":"high"}}` |
| **Qwen 3.x** (`qwen3-next-80b-a3b-thinking`, `qwq-32b`, Qwen3.5 VLM) | `chat_template_kwargs:{enable_thinking:false}`; omit tokens via `include_reasoning:false` (**not for streaming**) | NIM VLM docs |
| **Nemotron** (Llama-Nemotron Ultra/Super/Nano) | **system prompt** `detailed thinking on` / `detailed thinking off` | model card |
| **Kimi** (`kimi-k2-thinking`) | thinking integrated into model behavior | model card |

**🔴 The hang trap (#25):** for `deepseek-v4-flash`/`-pro`, NIM **strictly requires** `chat_template_kwargs:{enable_thinking:true, thinking:true}` to stream reasoning — **without it the API can hang and never return**. (Compounded by SDKs that strip unknown config keys.)

**Return shape:** reasoning comes back in a **separate `reasoning_content` field** (also `reasoning` on some) — NOT in-band `<think>` tags. Augmentum's `reasoning_content` reader handles this ✅ (but see #17 below).

**Sampling per state (Nemotron, representative):** reasoning ON → temperature ≈ 0.6, top_p ≈ 0.95; reasoning OFF → **greedy** (temperature 0). Don't run reasoning-on greedy (repetition).

---

## 3. Models (catalog — `build.nvidia.com/models`)

100+ NIMs. Reasoning-capable families relevant to per-model handling: **Llama-Nemotron** (Ultra 253B / Super 49B / Nano), **DeepSeek V4** flash/pro, **Qwen3.x** (incl. thinking + QwQ + VLM), **GLM-4.x**, **Kimi K2 thinking**, plus dense Llama/Gemma. Context/output caps are per-NIM (read the model card) — `max_context` is meaningless at profile level.

---

## 4. Request parameters

Standard OpenAI: `model`, `messages`, `temperature`, `top_p`, `max_tokens`, `stream`, `stop`, `frequency_penalty`, `presence_penalty`, `seed`, `tools`/`tool_choice`, `response_format`.
**NIM-specific:** `chat_template_kwargs` (reasoning toggle, §2), `nvext` (NVIDIA extension wrapper — older/family-specific; current DeepSeek/Qwen examples put `chat_template_kwargs` at the root, not in `nvext`).

---

## 5. Known drift / gaps

- ✅ **#25 reasoning + hang risk — FIXED (R3, 2026-06-25)** — profile flag `reasoning_via_chat_template_kwargs=True` + `_nim_chat_template_kwargs` now emit nested `chat_template_kwargs`: **DeepSeek → `{thinking:bool}` always** (kills the hang, both think states); Qwen/Nemotron/EXAONE/Gemma4/MiMo → `{enable_thinking:bool}`. GLM/Kimi intentionally excluded (uncertain NIM key; only DeepSeek hangs). Reasoning still returns in `reasoning_content` (read ✅).
- 🟢 **Nemotron Ultra/Super system-prompt path** — the *older* Llama-Nemotron Ultra/Super 253B/49B use a `detailed thinking on/off` **system-prompt** toggle (not a kwarg). Nemotron 3 Nano uses `enable_thinking` (now covered). The system-prompt variant is not yet wired (no hang; just uncontrolled).
- 🔴 **(via #17)** NIM serves Qwen3/GLM-4.x/DeepSeek-V4 (all `_STARTS_THINKING_FAMILIES`). When a NIM returns plain content with no leading `</think>` and no `reasoning_content`, the cloud `_inside_think=True` init routes the whole answer into the thinking channel. **This is exactly the "responses in the thinking block" symptom reported on NVIDIA NIM.**
- 🟢 Nemotron reasoning-state sampling (ON=0.6/0.95, OFF=greedy) not wired to a per-model profile.

---

## 6. Sources

- NIM LLM API: https://docs.api.nvidia.com/nim/reference/llm-apis
- DeepSeek V4 on NIM (thinking example): https://build.nvidia.com/deepseek-ai/deepseek-v4-flash · https://build.nvidia.com/deepseek-ai/deepseek-v4-pro
- Qwen3.5 VLM NIM (`chat_template_kwargs.enable_thinking`, `include_reasoning`): https://docs.nvidia.com/nim/vision-language-models/1.7.0/examples/qwen/api.html
- Reasoning return field: https://docs.nvidia.com/nemo/evaluator/latest/evaluation/run-evals/reasoning.html
- Hang-without-kwargs report: https://github.com/anomalyco/opencode/issues/24264
- Catalog: https://build.nvidia.com/models
