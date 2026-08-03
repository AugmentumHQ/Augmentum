# The Augmentum Engine

*What Augmentum adds on top of a stock `llama.cpp` `llama-server` binary.*

Augmentum serves its own models through the bundled `llama-server`, but the
binary is only the substrate. Around it sits roughly **27,000 lines of Python
orchestration** (`augmentum/models/`) that turn a single-model runner into a
coordinated, agent-facing inference layer.

The distinction in one line: **Ollama and LM Studio manage a single model in a
single process. Augmentum manages a fleet** — multiple resident models, saved
conversation state, draft-aware speculation, per-hardware calibration, and
routing across local, cloud, and peer backends, all under one VRAM budget.

This document walks through what that orchestration actually does, and how each
capability compares to running Ollama or LM Studio directly.

---

## 1. Multi-slot architecture (Slots A / B / C)

Three independent `llama-server` subprocesses, each with its own port, GPU-layer
budget, idle timeout, and lifecycle — all managed from one proxy.

| Slot | Role | Behaviour |
| --- | --- | --- |
| **A — Primary** | Chat / coder / agentic | The model the user is talking to. Swap it from the UI without touching the other slots. |
| **B — Secondary** | Utility: summarization, memory consolidation, reflection, chat titles | Idle by default (zero VRAM); loads on first use. Can also be pinned as a second chat model per conversation (LM Studio–style). |
| **C — Classifier** | Intent classification, voice routing, vision captioning | Resident — never unloads. Runs a small model (e.g. Gemma-4-E2B/E4B) on a ~2.5s latency budget for voice/architect hops. Doubles as the vision substrate when paired with an mmproj. |

**vs. Ollama / LM Studio:** one process, one model, one config. If you want a
fast classifier *and* a large chat model *and* occasional summarization, you're
manually swapping models — or running three separate instances by hand and
managing port conflicts yourself. Augmentum coordinates all three under a single
VRAM budget.

**Practical benefit:** your chat model doesn't get evicted when the companion
summarizes your journal, and your voice interface doesn't wait fifteen seconds
for a cold model load. Three workloads, three slots, no conflict.

---

## 2. KV session persistence and resume

Conversation state is saved to disk as KV-cache slot files, so returning to a
session skips the entire prefill. A 16K-token conversation that costs 10–30
seconds of prefill on a cold start resumes in milliseconds.

The engine runs a three-rung ladder; every rung failure falls through to the
next, and the cold floor is always correct:

| Rung | Mechanism | When it works | Cost |
| --- | --- | --- | --- |
| **RESTORE** | Load saved K/V tensors from a slot file | Single-slot config (non-unified KV) | I/O only — near-instant |
| **REPLAY** | Recompute KV by prefilling the stored prefix with `n_predict=0` | Any slot config, even cross-model swaps | Real prefill, but runs in free windows (post-boot, session open) |
| **COLD** | First real request prefills | Always | Normal prefill latency |

The replay source is captured *below* the mode layer — the exact
post-augmentation message list is recorded per turn, so replay is byte-identical
to the original by construction. No mode has to re-derive trimmed history.

**vs. Ollama / LM Studio:** Ollama's `keep_alive` delays unloading, but once the
model unloads, all KV state evaporates. LM Studio has no session-persistence
concept. Both reprocess the full context on every cold start.

**Practical benefit:** a two-hour roleplay session or a long coding session
resumes instantly. Close your laptop, come back tomorrow — no 30-second
re-prefill. The engine warms in the background while you're not looking.

---

## 3. Speculative turn generation

While the user is mid-draft (a typing pause, an STT partial), the engine runs
the whole turn against the draft on idle GPU cycles. Three outcomes:

1. **Byte-identical draft** → the finished answer streams with zero engine work
   (0 ms TTFT). The response appears the instant the user presses send.
2. **Edited before send** → speculation is discarded, but its prefill already
   sits in a slot; the user pays only for the diff.
3. **GPU got busy** → speculation is preempted instantly; nothing happened. Real
   traffic always wins.

Hard rules: local-only (drafts never leave the box), never serve a truncated
answer, drafts never touch disk. Byte-exactness is *measured*, not asserted —
every serve attempt logs hit/miss.

**vs. Ollama / LM Studio:** neither has a draft-aware prefill engine. At best,
Ollama's prompt caching caches the system prompt — not a user's in-progress
message.

**Practical benefit:** in voice mode, the assistant can answer before you finish
speaking. In chat, the answer appears the moment you press Enter.

---

## 4. Per-model sampling profiles with layered resolution

A five-layer override stack: **per-call → per-chat → per-model → global → family
default**. When you download a model, the engine auto-imports its family's
known-good sampling:

- Qwen3 → `temperature=0.6`, `top_k=20`
- Gemma-4 → `temperature=1.0`
- DeepSeek-R1 → `temperature=0.6`

Each layer fills only what the layer above left unset, via `resolve_sampling()`.
Per-model overrides are editable in the model library (persisted as JSON,
mirroring the profile store).

**vs. Ollama / LM Studio:** Ollama uses a single temperature in its Modelfile
(per-model, but no family defaults, no layering, no auto-import). LM Studio has
global defaults and per-prompt overrides only. Neither knows that Qwen3 should
have `top_k=20` just from looking at the GGUF.

**Practical benefit:** swap models and the sampling follows. You don't have to
remember to set `top_k` for Qwen or raise temperature for Gemma every time.

---

## 5. Self-calibrating VRAM workspace estimates

An exponential moving average of `observed_peak / predicted_reserve` that
self-corrects the load planner's VRAM estimates over time. The calibration:

- Persists as JSON next to the model-profile cache (survives restart)
- Is per-bucket (Flash Attention on/off — different workspace shapes)
- Has safety bounds `[0.7×, 1.5×]` so a single anomalous reading can't OOM or
  starve
- Requires a minimum sample count before it activates

The load planner uses this to autofit GPU layers, cap context length to available
VRAM, and generate `llama-server` CLI flags that won't OOM.

**vs. Ollama / LM Studio:** Ollama uses a fixed `num_gpu` layer count — no
autofit, no calibration, no adaptation. Set it too high and OOM; too low and
waste GPU. LM Studio's GPU-offload slider is manual, static, and uncalibrated.

**Practical benefit:** drop a new GGUF in the models directory and it loads
without you calculating GPU layers or context limits — and it gets more accurate
over time as calibration accumulates samples from your actual hardware.

---

## 6. Multi-provider federation and unified routing

One `ProviderRegistry` discovers and unifies every backend — local
`llama-server`, Ollama, any OpenAI-compatible endpoint, Anthropic, Gemini, and
remote Augmentum peers (Fabric). A single `resolve_backend_for_model()` call
handles all of it.

The resolution chain:

```
per-chat pin → load balancer (lb/name) → explicit model@backend
→ catalog probe → fabric peer fallback → default
```

Model maps carry a 30s TTL with degraded-probe fast-retry (local backends
re-probe quickly; slow cloud catalogs don't).

**vs. Ollama / LM Studio:** Ollama serves only its own models; LM Studio serves
only locally loaded GGUFs. Neither federates, and neither can route to "the model
on my other machine" or "OpenRouter if local is busy."

**Practical benefit:** pin a conversation to a specific model
(`gemma-4@engine_secondary`), set up a load balancer across two providers, or
fall back to a cloud model when your GPU is busy — all transparent to the chat
UI.

---

## 7. Load balancing with automatic fallback

Virtual `lb/<name>` models fan a request across a pool of backends:

- **Five strategies:** round-robin, random, weighted-random, least-recently-used,
  A/B test
- **Pre-first-token fallback:** if the selected member fails with a retryable
  error (429, 5xx, timeout, connection reset), the request silently falls to the
  next member — the user never sees an error
- **Per-member cooldown:** after a failure, the member is skipped for a cooldown
  period (respects `Retry-After` headers or uses exponential backoff)
- **Non-retryable detection:** auth failures, context-length-exceeded, and
  vision-projector mismatches are *not* retried (they'd fail identically on every
  member)
- **Empty completion = failed attempt:** a stream that returns zero content
  counts as a failure
- **Informative exhaustion:** when all members fail, the user gets a classifiable
  error, not a silent empty turn

**vs. Ollama / LM Studio:** neither has a load-balancer concept. One model, one
endpoint.

**Practical benefit:** run three local instances of the same model and get
automatic failover, or pool `local-llama` + `openrouter-free` + `openrouter-paid`
so you never see a rate-limit error.

---

## 8. Cross-provider thinking control

One centralized `resolve_thinking()` answers three questions for every provider:

1. Does this model support thinking control? (family-regex detection)
2. Should thinking be on for this request? (user setting × model capability)
3. If off, must we send an explicit disable? (critical for Gemini 2.5+/3.x, where
   the default is thinking-on)

Family-aware matchers cover Gemini 3, Gemini 2.5, DeepSeek-R1, DeepSeek hybrid,
Claude, GLM-4, and Qwen3. Each adapter translates the decision into its own wire
format (Claude thinking block, Gemini `thinkingConfig`, OpenAI
`reasoning_effort`).

**vs. Ollama / LM Studio:** Ollama doesn't abstract across providers; LM Studio
is local-only. Neither has a reasoning-control layer.

**Practical benefit:** set "thinking: off" once. The engine sends the correct
directive whether the backend is Claude, Gemini, or a local Qwen3 — and knows
Gemini needs an explicit off while Claude is off by default.

---

## 9. Prefix-stability audit (KV-cache reuse measurement)

A request–response contract that measures whether the provider actually reused
the KV cache. On the request side, `track_prefix_stability()` diffs consecutive
messages for a session key; on the response side, `_audit_kv_reuse()` joins that
with actual token counts:

| Verdict | Meaning |
| --- | --- |
| `stable_hit` | Prefix unchanged, cache used — perfect reuse |
| `stable_miss` | Prefix unchanged, cache **not** used — server waste (slot eviction, or a remote provider ignored `prompt_cache_key`) |
| `changed_expected` | Prefix changed, new prefill expected — normal turn |
| `cold_expected` | First turn in session — no baseline to compare |
| `server_void` | Backend returned no cache telemetry — can't judge |

**vs. Ollama / LM Studio:** neither measures cache reuse across turns. You can't
tell whether your `keep_alive` actually prevented a re-prefill.

**Practical benefit:** you can see when your prompt cache is wasting
money/tokens. The audit surfaces "I sent the same 8K system prompt and the
provider re-charged it" — actionable data, not guesswork.

---

## 10. Mode-aware inference hints

Each dispatch mode gets its own default inference parameters, applied as defaults
that never override the user's explicit settings:

| Mode | Temperature | Top-P | Max tokens | Reasoning effort |
| --- | --- | --- | --- | --- |
| Coder / Agentic | Low (deterministic) | — | Generous | High (multi-step problem solving) |
| Narrative | 0.85 | 0.95 | 4096 | Low (creative flow, not CoT) |
| Passthrough (Chat) | — | — | — | Low (snappy, not deliberative) |
| Analytical | 0.1 | 0.1 | — | Low (per-phase, short, fast) |

It also carries `raw_options` for local backends (`repeat_penalty`, `min_p`) and
per-family defaults that follow the model when you switch modes.

**vs. Ollama / LM Studio:** neither has a concept of dispatch modes — they serve
one model with one config.

**Practical benefit:** temperature drops automatically when you switch from
creative writing to coding. You don't need to remember to turn it down.

---

## 11. API translation layer (Anthropic ↔ OpenAI)

Augmentum speaks Anthropic's Messages API natively: it translates incoming
Anthropic-format requests to the internal canonical format, routes them through
the same backend dispatch, and streams SSE responses back in Anthropic format —
including:

- Full message translation (user/assistant blocks, tool use/results, thinking
  blocks)
- Image-block conversion (Anthropic `source` → base64 → OpenAI `image_url`)
- Tool-call round-trip fidelity
- `prompt_cache_key` computation for prefix reuse
- SSE event protocol (`message_start`, `content_block_start/delta/stop`,
  `message_delta`, `ping`)

Provider-specific adapters (`claude.py`, `gemini.py`) and response converters
(`claude.py`, `gemini.py`, `cohere.py`, `mistral.py`) normalize provider quirks.

**vs. Ollama / LM Studio:** Ollama has an experimental OpenAI-compat endpoint; LM
Studio is OpenAI-compat only. Neither translates between API formats or
normalizes across providers.

**Practical benefit:** any Anthropic client can point at Augmentum and reach any
backend — local GGUF, cloud provider, or peer — without knowing what's on the
other side.

---

## 12. Self-discovery and GGUF profiling

A pure-Python GGUF header parser (`model_profile_cache.py`):

- Extracts architecture, layer counts, embedding length, MoE classification, and
  metadata — with no `llama-cpp-python` dependency
- Caches results to disk (JSON) with an LRU-capped in-memory layer
- Tags models by architecture family (Llama, Qwen, Gemma, Mistral, Phi, DeepSeek,
  …)
- Auto-detects MoE models via tensor-name patterns (so `n_gpu_layers` is set
  correctly for MoE vs dense)
- Scans configurable directories recursively for GGUF files
- Supports operator-declared mmproj pairings via a sidecar JSON
  (`.augmentum-projector.json`) — comparable to Jan's `model.json` or Ollama's
  manifest layer, but operator-controlled

On top, `ModelManager` adds Hugging Face GGUF catalog browsing, download with
progress, file-size precheck, and loading into specific slots.

**vs. Ollama / LM Studio:** Ollama uses its own registry/Modelfile system — you
can't just drop a GGUF in a folder. LM Studio scans folders but doesn't
auto-configure GPU layers, context size, or sampling from architecture.

**Practical benefit:** drop any GGUF in `models/` and it appears in the UI with
correct GPU layers, context limits, and sampling defaults — no Modelfile, no
manual config.

---

## 13. Purpose-built for autonomous agents

The engine isn't a generic inference server; it's designed to serve Augmentum's
own workload — the companion runtime and the coder.

**Companion runtime (`companion_runtime/`):**

- The KV resume ladder means scheduled background ticks (affect, energy,
  consolidation, reflection) don't re-prefill the companion's system prompt on
  every wake
- Role-based routing sends summarization/consolidation to Slot B and
  classification to Slot C — never blocking the user's chat on Slot A
- The native function-calling loop drives both chat and voice from the same
  engine

**Coder (`modes/coder/`):**

- The prefix-stability audit shows exactly how much context the coder
  re-processes per turn
- Mode-aware hints set low temperature + high reasoning effort for code
- Context-token tracking integrates with the token-count cache for instant
  window math

**vs. Ollama / LM Studio:** they're general-purpose. They don't know about
companion ticks, coder tool loops, or mode-specific optimization — every request
is a generic `/v1/chat/completions`.

---

## Summary

| Capability | Ollama | LM Studio | Augmentum Engine |
| --- | --- | --- | --- |
| Multi-model concurrency | One process | One process | **3 independent slots** |
| KV session persistence | Volatile | Volatile | **Save/restore + auto-replay** |
| Speculative generation | — | — | **Draft-aware, 0 ms TTFT** |
| Per-model sampling profiles | Manual Modelfile | Manual only | **Auto-import + 5-layer resolution** |
| VRAM self-calibration | Static `num_gpu` | Manual slider | **EMA-based, autofit** |
| Multi-provider federation | Ollama only | Local only | **Local + cloud + peers + fabric** |
| Load balancing with fallback | — | — | **5 strategies + cooldown** |
| Cross-provider thinking control | N/A | N/A | **Family-aware, explicit-disable** |
| Prefix-cache audit | — | — | **Request–response contract, measured** |
| Mode-aware inference hints | — | — | **Per-mode defaults** |
| API translation (Anthropic ↔ OpenAI) | OAI-only (exp.) | OAI-only | **Full Anthropic ↔ OpenAI** |
| GGUF auto-profiling | Needs Modelfile | Partial | **Architecture detection + auto-config** |
| Drop-in GGUF support | Import required | Yes | **Yes** |
| Role-based model routing | — | — | **Classifier / utility / primary** |
| Autonomous-agent integration | — | — | **Companion + coder native** |

**The short version:** Ollama and LM Studio are model *runners*. Augmentum's
engine is an *agent server*. It doesn't just load a model — it coordinates
several, persists conversation state, speculates against user drafts, calibrates
to your hardware, routes across providers, and optimizes inference per workload.
All of it serves one goal: make the assistant feel instant, persistent, and never
in your way.

---

*Related: [Model Manager](model-manager.md) · [Architecture](ARCHITECTURE.md) ·
[Fabric federation](fabric.md) · bundled-model
[capability cards](model-cards/README.md).*
