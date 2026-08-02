# Load Balancers as First-Class Routing — Fallback, Quota-Cycling & Parity

Design spec. **Nothing built yet.** Status: research complete, awaiting go.

## Why (the journey)

Matt selects a load-balancer virtual model (`lb/<name>`) as his chat model. The
balancer's whole value: point one LB at **12 free-tier API keys/accounts** (each
its own model + ~20 RPD), and have Augmentum **auto-cycle** them on rate-limit so
he never tracks quotas by hand — one selection, effective access to all of them.

The `fallback_enabled` toggle exists in the UI and is stored per balancer. But it
**does nothing**. Root cause (verified 2026-07-29):

- `resolve_backend_for_model` (provider_registry.py:~1518-1553) picks a member,
  computes `LoadBalancer.fallback_order(member)`, and stores it in a contextvar via
  `set_balancer_context(BalancerResolution(... fallback_members=...))`.
- **`get_balancer_context()` has ZERO callers.** Nothing ever reads the fallback
  order back. On a 429 the member's `chat_stream` raises → the SSE/ndjson generators
  catch it → `_make_error_chunk` emits the generic "backend issue" toast
  (streaming.py:839). No retry is ever attempted.

Matt hit this in **narrative**, not passthrough. Wiring it for one mode is a parity
bug waiting to be found — users expect an LB to behave identically everywhere it can
be selected (passthrough, direct, analytical, agentic, narrative, becca_direct,
coder, and every companion_runtime task). This spec makes fallback **universal by
construction**, not per-site.

## The core design decision: a `BalancerBackend` facade

Do **not** sprinkle retry logic across the ~20 `chat_stream` sites + dozens of
`chat()` sites. Instead, when the requested model is an `lb/` model,
`resolve_backend_for_model` returns a **`BalancerBackend(ModelBackend)` facade**
instead of a concrete member backend. Every existing call site keeps calling
`backend.chat_stream(req)` / `backend.chat(req)` unchanged and gets fallback for
free — zero call-site edits, automatic cross-mode parity.

The facade implements the full `ModelBackend` ABC:

| Method | Facade behavior |
|---|---|
| `chat_stream(req)` | fallback loop (below) over members |
| `chat(req)` | atomic fallback loop (simpler — no streaming constraint) |
| `pre_stream_validate(req)` | proxy to the *currently selected* member; must be re-run per attempt |
| `list_models` / `show_model` | proxy to primary member (or union) |
| `get_context_length(model)` | **min** across enabled members (safe ceiling) |
| `supports_mid_conversation_system` | per selected member — matters for the datetime-placement fix; the facade must expose the *attempted* member's value each call |
| `is_vision_paired` / `is_local_engine` | proxy to selected member |

## Fallback loop semantics

**Order:** first the strategy-selected member (`LoadBalancer.select()`), then
`fallback_order(selected)` (remaining, sorted by `priority`), skipping members in
cooldown (below).

**Streaming (`chat_stream`) — the hard part:**
- Try member *i*. Buffer until the **first content token** is produced.
- If the generator raises **before the first token** and the error is **retryable**
  → move to member *i+1*, log `balancer_fallback`, retry with a fresh cloned request
  (`request.model = member.model_name`; never mutate the caller's object).
- Once the first token yields → **commit** to that member and passthrough the rest.
  (We cannot un-send streamed tokens — see Constraints.)
- All members exhausted → raise an **informative** error (below), not the generic toast.

**Non-stream (`chat`):** same loop, atomic — retry any member on a retryable error.

**Retryable classification (reuse `streaming.py::_classify_backend_error`):**
retry `rate_limit` (429), `backend_unavailable` (502/503), `timeout`, connection
errors. Do **NOT** retry `auth_failed`, `context_overflow`, `no_vision_projector`
(`_NON_RETRYABLE_KINDS`) — they fail identically on every member and just burn the
pool + delay the real error.

## Parity is 3 implementations, not N — every provider funnels through 3 adapters

`create_backend_from_profile` maps every LLM provider to exactly one of three
adapter kinds. So universal rate-limit parity means instrumenting **3 adapters**,
and the OpenAI-compat one alone covers ~11 named cloud providers + all future ones:

| Adapter kind | Providers it serves | Rate-limit signal to capture |
|---|---|---|
| **`openai` — OpenAICompatibleBackend** | OpenAI, DeepSeek, Groq, Mistral, Together, Fireworks, xAI/Grok, Perplexity, **NVIDIA NIM**, **OpenRouter**, **Cerebras**, vLLM, llama-swap, any future OpenAI-compat | **Generic HTTP:** `Retry-After` + `X-RateLimit-Reset/Remaining/Limit` on the 429. One implementation → parity for all of them. |
| **`claude`/`anthropic` — ClaudeBackend** | Anthropic API + anthropic-compat | `retry-after` + `anthropic-ratelimit-*` headers (confirm exact names at build) |
| **`gemini`/`google` — GeminiBackend** | Google AI Studio / Gemini | non-standard: parse `RetryInfo.retryDelay` + `QuotaFailure.quotaId` from the 429 **body** |
| local: `engine`/`llama_cpp`/`ollama`/`fabric` | self-hosted | no external quota; a 503/"busy" → short generic cooldown |

**Design payoff (premium experience, minimal setup):** because the vast majority of
providers share the OpenAI-compat 429 + `Retry-After`/`X-RateLimit-*` convention, a
**single generic capture** in `OpenAICompatibleBackend` gives fallback + accurate
cooldown parity across nearly the whole catalog with zero per-provider config. Only
Gemini needs a bespoke body-parser; Anthropic needs its header names. A user drops in
keys for whatever mix of providers and it "just works."

## Provider rate-limit survey (2026-07, web-verified) — why cooldown must be SHORT by default

We currently capture **none** of this (adapters ignore all response headers/bodies).
No provider offers a reliable *proactive* "remaining" to poll, so the mechanism is
**reactive**, driven by the 429. The limit TYPE differs, dictating cooldown length:

| Provider | 429 reset signal | Dominant free limit |
|---|---|---|
| OpenRouter | `Retry-After` + `X-RateLimit-Reset`; proactive `GET /api/v1/key`→`limit_remaining/reset` (per-key credit) | free-model **daily** cap |
| Gemini | body `RetryInfo.retryDelay` + `QuotaFailure.quotaId` (PerDay vs PerMinute) | **RPM ~5** and RPD |
| NVIDIA NIM | `Retry-After` sometimes; no confirmed `X-RateLimit-*` | **RPM ~40** + TPM — not daily |
| Cerebras | 429 bucket message; `Retry-After`/`X-RateLimit-*` per best-practice, unconfirmed | **TPM** rolling |
| Groq/Mistral/Together/Fireworks/xAI/OpenAI | standard `Retry-After` + `X-RateLimit-*` (OpenAI-compat convention) | mostly RPM/TPM |

**This overturns my earlier "next-UTC-day default" guess.** A blanket day-long
cooldown would bench a 40-RPM NVIDIA member for a full day over a 1-minute burst.
The right default is SHORT, escalating only on repeat failure.

## Quota-aware cooldown (the piece that makes the 12-key use case actually work)

**Reactive, signal-driven — no manual RPD config** (no provider exposes reliable
proactive remaining across the board; manual caps would drift and reintroduce the
hand-tracking you're trying to eliminate). On a 429, derive the cooldown from the
provider's own signal, in priority order:
1. `Retry-After` header (seconds or HTTP-date) — most precise, honor verbatim.
2. Gemini: parse `RetryInfo.retryDelay` AND `QuotaFailure.quotaId` — if the quotaId
   says *PerDay* cool until the next daily window; if *PerMinute* cool for the delay.
3. `X-RateLimit-Reset` (epoch) when present.
4. **No signal → SHORT default (~60s) with exponential escalation** on repeat 429s
   from the same member (60s → 5m → 30m, capped). Short because the common free-tier
   hard-fail is RPM/TPM (recovers in seconds), not RPD.

Requires new **response-metadata capture** in the OpenAI-compat + Gemini adapters
(headers + Gemini error body) — today thrown away. Optional per-provider nicety:
poll OpenRouter's `GET /api/v1/key` to pre-empt its daily credit cap.

Round-robin alone will keep hitting a 429'd key every cycle. For "20 RPD × 12 keys,
auto-cycle," members need a **per-member circuit breaker**:

- On a `rate_limit`/`backend_unavailable` from member *m*, mark *m* **cooling** until
  a reset time — parse `Retry-After` when present, else a configurable default
  (e.g. rate-limit → until next UTC day for RPD-style limits, or N minutes).
- `LoadBalancer.select()` and `fallback_order()` **skip cooling members**.
- Optional RPD budget: track requests/day per member; pre-emptively cool a member at
  its declared daily cap so the last request doesn't 429.
- State: a `balancer_member_health` table (member_id, cooling_until, rpd_used,
  rpd_window_start, consecutive_failures) **or** in-memory with periodic persist.
  In-memory is fine for a single-process server; persist so cooldowns survive the
  daily restart.
- When ALL members are cooling → the exhaustion error names the earliest reset.

## Informative errors (kill the useless toast)

When the pool is exhausted, surface (via the existing error-event path) a specific
message: *"All N providers in balancer '<name>' are rate-limited or unavailable
(earliest reset ~HH:MM). Add more members or wait."* — carrying `error_kind` so the
frontend can render it distinctly from a one-off backend blip.

### Fallback is never silent (visibility principle)
Even on a SUCCESSFUL fallback, the turn carries metadata that a fallback happened
(`fallback_occurred`, `member_selected`, `member_served`, `reason` e.g. "A
rate-limited"). The frontend shows a small, non-error notice/badge — *"primary
provider rate-limited; served via <member>"*. Rationale (Matt): a working-but-
degraded pool must be visible, or a run where every answer comes from one member
looks like a silent bug instead of "your other provider keeps erroring." This
applies to ALL strategies, and is mandatory for A/B (above).

## Constraints (documented so they're not surprises)

1. **Pre-first-token only for streaming.** Once tokens reach the user, a mid-stream
   429 can't be recovered by switching providers. Mitigation: 429 "high demand"
   almost always fails on the first byte, and `pre_stream_validate` runs before the
   stream — so the common case is covered. Document the residual (mid-stream failure
   → surfaced, not silently retried).
2. **Member capability skew.** Members may differ in context length, vision, tool
   calling, thinking support. For a **vision** request, skip members whose model
   isn't vision-capable rather than 400 them. Context length = min across members
   (already tabled). Tool-calling: if a request carries tools, prefer/require
   tool-capable members.
3. **A/B strategy (decided).** `AB_TEST` splits traffic for comparison, but fallback
   **still fires on hard failure** — otherwise a persistently-erroring arm turns every
   turn into a coin-flip on getting *any* answer. The user gets a response from the
   healthy arm. BUT the turn MUST record + surface that the selected arm errored and
   which arm actually served (`arm_selected` vs `arm_served`, `fallback_occurred`), so
   a run where "everything came from B" reads as *"A kept erroring"* — not a silent
   bug, and the A/B data stays honest (the perturbed turns are flagged, not counted as
   clean B wins).
4. **Datetime placement.** The facade must report the *attempted member's*
   `supports_mid_conversation_system` so `_ensure_datetime` places the block
   correctly per member (ties into the 2026-07-29 tail-placement fix).

## Implementation checklist

- [ ] `BalancerBackend(ModelBackend)` facade in `augmentum/models/load_balancer.py`
      (or a new `balancer_backend.py`), constructed with the `LoadBalancer` + registry.
- [ ] `resolve_backend_for_model` LB branch returns the facade (keep the contextvar
      for A/B metadata; the facade owns fallback so `get_balancer_context` consumers
      are no longer required — but leave it set for telemetry/UI).
- [ ] Shared `is_retryable_backend_error(exc)` helper (extract from streaming.py so
      the facade and the SSE layer agree).
- [ ] Per-member cooldown/RPD circuit breaker + `select()`/`fallback_order()` skip.
- [ ] Informative exhaustion error + `error_kind`.
- [ ] `balancer_fallback` / `balancer_member_cooldown` structured logs.
- [ ] **Apply the facade at BOTH LB branches** (see audit below) via one shared
      `_resolve_balancer(name) -> BalancerBackend` helper — do NOT duplicate the
      logic, or chat-model vs role-model selection will drift.
- [ ] UI (balancer panel): per-member health (up/cooling/RPD used), functional
      `fallback_enabled`, and the exhaustion state.

## Resolution-path audit (done — this is what makes it universal)

Every chat mode handler (passthrough, direct, narrative, agentic, analytical,
becca_direct, coder) receives its backend as an **injected `self._backend`** from
`handler_factory` — it does not resolve its own. So if the *resolved* backend is the
facade, **all seven modes get fallback with zero handler edits.**

There are **THREE resolvers**, and the LB handling is split across two of them —
both must return the facade (via the shared helper):

| Resolver | LB-aware? | Notes |
|---|---|---|
| `resolve_backend_for_model` | ✅ LB branch @ ~1535 | the user's chat-model path (`lb/…`) |
| `resolve_backend_with_fabric` | ✅ (delegates) | calls `resolve_backend_for_model` @ ~450 — inherits the facade |
| `resolve_model_for_role` | ⚠️ **separate** LB branch @ ~1705-1708 | role models (utility/heavyweight/memory, narrative extraction, coder plan phase, browse, cardsmith). **Must return the facade too** or an `lb/` role silently loses fallback. |

**Out-of-scope bypass sites** (grab `default_backend` directly, never a user-selected
`lb/` model — leave as-is, note in code): `companion_runtime/primitives/
memory_consolidate.py:46`, `reasoning/executor.py` (`default_backend` param).
`model_manager` / `kv_resume` / `kv_speculate` are engine-internal, not LB targets.

## Test matrix (parity is the point)

- Fallback fires on 429 **in every mode**: passthrough, direct, analytical, agentic,
  narrative, becca_direct, coder, one companion_runtime task.
- Retryable → next member; non-retryable (auth/context) → immediate surface, no burn.
- Pre-first-token retry succeeds; post-first-token failure surfaces (documented limit).
- Cooldown: a 429'd member is skipped until reset; all-cooling → informative error.
- Capability skew: vision request skips non-vision members.
- `chat()` (non-stream) path falls over too.
- No regression when `fallback_enabled=False` (single-shot, current behavior).

## Open questions for Matt (1-2 now resolved by the survey)
1. ~~RPD tracking~~ → **resolved: reactive, no manual caps.** No provider exposes
   reliable proactive remaining across the board; manual caps drift and reintroduce
   hand-tracking. Parse the 429 signal instead.
2. ~~Cooldown default~~ → **resolved: short (~60s) with exponential escalation**, not
   next-day. Parse `Retry-After`/`X-RateLimit-Reset`/Gemini `retryDelay`+quotaId when
   present; the short default only applies when the provider gives no signal.
3. ~~A/B balancers~~ → **resolved: fallback still fires on hard failure**, but the
   turn surfaces `arm_selected`/`arm_served`/`fallback_occurred` (visibility principle
   above) so it never looks like a silent bug and the A/B data flags perturbed turns.
4. **Optional nicety:** poll OpenRouter's `GET /api/v1/key` to pre-empt its daily
   credit cap before the last request 429s — OpenRouter-only, low priority.

## Status: design locked, ready to build
All decisions resolved. Build sequence: (1) `BalancerBackend` facade + shared
`_resolve_balancer()` in both LB branches; (2) generic response-metadata capture in
`OpenAICompatibleBackend` + Gemini body-parser + Anthropic headers; (3) per-member
cooldown circuit-breaker; (4) fallback-visibility metadata + exhaustion error;
(5) per-mode parity test matrix.
