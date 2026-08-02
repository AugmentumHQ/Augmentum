# Anthropic (Claude) — Provider Reference Card

> **Verbatim reference** from Anthropic's official docs (`platform.claude.com/docs`).
> **Sourced:** 2026-06-25 · **Sources:** see [§6](#6-sources).
> **Native backend** — own adapter (`models/adapters/claude.py` + `converters/claude.py`), bypasses the OpenAI-compat profiles.

| | |
|---|---|
| **`provider_type`** | `anthropic` (native Messages API) |
| **Base URL** | `https://api.anthropic.com` |
| **Endpoint** | `POST /v1/messages` |
| **Headers** | `x-api-key: <KEY>` · `anthropic-version: 2023-06-01` · `Content-Type: application/json` |
| **Beta headers (conditional)** | `interleaved-thinking-2025-05-14` (non-adaptive thinking models only) · `output-300k-2026-03-24` (300k batch output) |

---

## 1. Augmentum wiring (what WE send) — **well-wired, current**

The native converter is one of the most correct paths in the codebase:

| Behavior | Implementation | Status |
|---|---|---|
| **Adaptive vs traditional thinking** | `_ADAPTIVE_MODEL_RE` = `opus-4-6\|opus-4-7\|opus-4-8\|sonnet-4-6\|fable-5` → `{thinking:{type:"adaptive"}, output_config:{effort}}`; others → `{thinking:{type:"enabled", budget_tokens:N}}` | ✅ matches docs (Opus 4.8/Fable 5 use `effort`, not budget) |
| **Interleaved-thinking beta** | added **only** for non-adaptive thinking models (`interleaved-thinking-2025-05-14`) — adaptive models 400 on it | ✅ |
| **Temperature with thinking** | dropped when thinking on (Claude disallows temperature + thinking) | ✅ |
| **Prefill suppression** | `_NO_PREFILL_MODEL_RE` (= adaptive set) skips assistant prefill (`budget_tokens` 400s on Opus 4.7+/Fable-5) | ✅ |
| **max_tokens** | `request.max_tokens or 4096` | ✅ required by API |
| **Model freshness** | thinking-model regex includes `fable-5`, `opus-4-8` | ✅ current |

---

## 2. Models & model-level handling (verbatim, /1M in→out)

| Model | API ID | Context | Max output | Thinking | Pricing |
|---|---|---|---|---|---|
| **Claude Fable 5** | `claude-fable-5` | 1M | 128k | adaptive **always-on** (no extended) | $10 / $50 |
| Claude Mythos 5 | `claude-mythos-5` | 1M | 128k | adaptive always-on | $10 / $50 (limited availability) |
| **Claude Opus 4.8** | `claude-opus-4-8` | 1M¹ | 128k | adaptive (yes) · extended **no** · **`effort` param default `high`** | $5 / $25 |
| **Claude Sonnet 4.6** | `claude-sonnet-4-6` | 1M | 128k | extended **yes** + adaptive yes | $3 / $15 |
| **Claude Haiku 4.5** | `claude-haiku-4-5-20251001` | 200k | 64k | extended yes · adaptive **no** | $1 / $5 |
| Opus 4.7 / 4.6 (legacy) | `claude-opus-4-7` / `-4-6` | 1M | 128k | adaptive yes | $5 / $25 |

¹ 200k on Microsoft Foundry. ~30% more tokens per text since the Opus 4.7 tokenizer.

**Model-level handling we can use:**
- **`effort`** (Opus 4.8 / Fable 5 / adaptive) defaults `high` — Augmentum maps its `thinking_effort` → `output_config.effort` ✅. Set explicitly per request to trade latency/cost.
- **`budget_tokens`** (Sonnet 4.6 / Haiku 4.5 traditional): min **1024**, must be `< max_tokens`, counts toward `max_tokens`.
- **300k output** available on Opus 4.8/4.7/4.6 + Sonnet 4.6 via batch + `output-300k-2026-03-24` beta.
- **temperature range is 0–1** (default 1.0) — not 0–2.

---

## 3. Request parameters (verbatim)

`model`*, `max_tokens`* (>0; 0 = warm cache), `messages`* (alternating user/assistant, content blocks w/ optional `cache_control`), `system` (string or TextBlock[]), `temperature` (0–1, def 1), `top_p`, `top_k`, `stop_sequences`, `stream`, `tools` (custom/bash/code_execution/web_search…), `tool_choice` (`{type:auto\|any\|tool\|none}`), `thinking` (`{type:enabled\|disabled\|adaptive, budget_tokens, display:summarized\|omitted}`), `output_config` (effort / structured format), `service_tier` (`auto`/`standard_only`), `metadata.user_id`, `inference_geo`, `container`.

**stop_reason:** `end_turn` / `max_tokens` / `stop_sequence` / `tool_use` / `pause_turn` / `refusal`.
**usage:** `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `output_tokens_details.thinking_tokens`, `service_tier`.

---

## 4. Pricing (verbatim)

See §2. Prompt caching + Batch API (−50%) discounts on the pricing page. Cache read ≪ fresh input.

---

## 5. Known drift / gaps

- 🟢 **service_tier / metadata.user_id unused** — `service_tier:"auto"` (priority capacity) and abuse-detection `metadata.user_id` not sent.
- 🟢 **`display` field** — `thinking.display: summarized|omitted` not surfaced (controls thinking-block visibility).
- 🟢 **300k output beta** — `output-300k-2026-03-24` not wired (batch-only extended output).
- ℹ️ **Verify `effort` wire location** — Augmentum puts effort in `output_config:{effort}`; confirm against the current `/effort` guide (top-level vs output_config) — code was tuned against real 400s, so likely correct.

---

## 6. Sources

- Messages API: https://platform.claude.com/docs/en/api/messages
- Models overview: https://platform.claude.com/docs/en/docs/about-claude/models/overview
- Extended/adaptive thinking + effort: https://platform.claude.com/docs/en/build-with-claude/extended-thinking · /adaptive-thinking · /effort
