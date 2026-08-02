# xAI (Grok) — Provider Reference Card

> **Verbatim reference** from xAI's official docs (`docs.x.ai`).
> **Sourced:** 2026-06-25 · **Sources:** see [§6](#6-sources).

| | |
|---|---|
| **Profile id** (`provider_profiles.py`) | `xai` |
| **`provider_type`** | `openai` |
| **Base URL** | `https://api.x.ai/v1` |
| **Endpoint** | `POST /v1/chat/completions` (OpenAI-compatible) |
| **Auth** | `Authorization: Bearer <XAI_API_KEY>` |
| **Vision** | ✅ (profile default; not restated in models snapshot) |
| **Knowledge cutoff** | November 2024 (Grok 3/4 variants), per docs |

---

## 1. Augmentum wiring (what WE send)

| Augmentum field | Value | Reconciliation |
|---|---|---|
| `supports_thinking` | `True` (`thinking_param=reasoning_effort`) | ✅ |
| `supports_reasoning_effort` | `True` | sends `reasoning_effort` ✅ |
| `supports_reasoning_effort_minimal` | _(default `True`)_ | ⚠️ **BUG** — Grok rejects `minimal`; no demotion happens (§4, correction #10) |
| `max_context` | `1_000_000` | ✅ matches grok-4.3 (the old 256K drift is fixed) |
| `max_output` | `0` (unmodeled) | xAI publishes no per-model output cap |
| _(no other OpenAI-family flags)_ | — | xAI doesn't document `max_completion_tokens`/`prompt_cache_key`/`service_tier` |

---

## 2. Models & pricing (verbatim, per 1M tokens, USD)

| Model id | Context | Input | Output | Reasoning |
|---|---|---|---|---|
| `grok-4.3` | 1M | $1.25 | $2.50 | ✅ |
| `grok-4.20-0309-reasoning` | 1M | $1.25 | $2.50 | ✅ |
| `grok-4.20-0309-non-reasoning` | 1M | $1.25 | $2.50 | — |
| `grok-4.20-multi-agent-0309` | 1M | $1.25 | $2.50 | ✅ |
| `grok-build-0.1` | 256k | $1.00 | $2.00 | — |

> Vision / function-calling / cached-input pricing not in the models snapshot — **TODO**.

---

## 3. Request parameters

OpenAI-compatible Chat Completions surface (see [`openai.md` §3](openai.md#3-request-body-parameters-verbatim--post-v1chatcompletions))
plus `reasoning_effort` (per-model allow-lists below). **Full xAI param page not yet fetched — TODO.**

---

## 4. Reasoning (verbatim)

Per-model `reasoning_effort` allow-lists (they differ!):

| Model | Allowed `reasoning_effort` |
|---|---|
| `grok-4.3` | `none`, **`low` (default)**, `medium`, `high` — **no `xhigh`, no `minimal`** |
| `grok-4.20-multi-agent` | `low`, `medium`, `high`, `xhigh` — **no `none`, no `minimal`** |

Return mechanisms:
1. `reasoning_tokens` in usage metrics.
2. Encrypted reasoning via `include: ["reasoning.encrypted_content"]` (**Responses API**).
3. Reasoning **summaries** streamed alongside the final response via delta events.

---

## 5. Known drift / gaps

- 🟡 **`minimal` not demoted** — `xai` profile keeps `supports_reasoning_effort_minimal` at its `True` default, so a `minimal` effort pick is sent verbatim and **Grok 400s** (no model accepts `minimal`). Fix: set `supports_reasoning_effort_minimal=False` on `xai`.
- 🟡 **Per-model effort variance** — `grok-4.3` rejects `xhigh`; multi-agent rejects `none`. A flat profile can't express this; ideally a per-model effort allow-list / clamp.
- ☐ Full param page, vision flag, cached pricing, and image-gen (grok-imagine) not yet captured.

---

## 6. Sources

- Models & pricing: https://docs.x.ai/docs/models
- Reasoning guide: https://docs.x.ai/docs/guides/reasoning
