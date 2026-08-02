# ChatGPT Bridge (codex-proxy) — Provider Reference Card

> **No public vendor docs page** — this profile wraps a *local* [codex-proxy] instance that
> exposes a ChatGPT Plus/Pro account as OpenAI-compatible endpoints. This card is derived from
> [`openai.md`](openai.md) + our **verified** bridge-specific notes, not a vendor page.
> **Sourced:** 2026-06-25 (behavioral, verified in-repo).

| | |
|---|---|
| **Profile id** (`provider_profiles.py`) | `chatgpt_bridge` |
| **`provider_type`** | `openai` |
| **Base URL** | _user-provided_ (local proxy `host:port`; no canonical URL) |
| **Auth** | proxy-managed (wraps the ChatGPT account session) |
| **Pricing** | ChatGPT **subscription** (Plus/Pro), not per-token |
| **Models** | whatever the account exposes (GPT-5.x family via the Responses bridge) |

---

## 1. Augmentum wiring (what WE send)

From `provider_profiles.py` (`chatgpt_bridge`): mirrors `openai` capabilities (`supports_max_completion_tokens`, `supports_reasoning_effort`, `supports_reasoning_summary`, `supports_developer_role`, `supports_prompt_cache_key`, `supports_service_tier`, `supports_thinking`) **with one verified divergence:**

| Flag | Value | Why |
|---|---|---|
| `supports_reasoning_effort_minimal` | **`False`** | ✅ **verified 2026-05-31:** the bridge 400s on `reasoning_effort="minimal"` (`Invalid enum value. Expected 'low' \| 'medium' \| 'high' \| 'xhigh'`) — its enum predates GPT-5.x's `minimal` tier. Adapter demotes `minimal`→`low`. |
| `max_context` | `256_000` | conservative — the Responses bridge publishes no hard ceiling; stops the Coder compactor clamping to 16K. |
| `supports_developer_role` | `True` | gated downstream on `is_openai_family_model` (so only fires for gpt-5/o-series ids served via the bridge). |

---

## 2. Request params

Same surface as OpenAI Chat Completions (see [`openai.md` §3](openai.md#3-request-body-parameters-verbatim--post-v1chatcompletions)), **except** `reasoning_effort="minimal"` is rejected (auto-demoted to `low`).

---

## 3. Known drift / gaps

- ☐ Bridge enum may gain `minimal` support upstream → re-enable `supports_reasoning_effort_minimal` if so.
- ☐ No model/context introspection (`/v1/models` lacks `context_length`) → `max_context` is a guess; verify against the wrapped account's model.

---

## 4. Sources

- Behavioral, verified in-repo: `provider_profiles.py` (`chatgpt_bridge` profile, notes dated 2026-05-31).
- Underlying surface = OpenAI (see [`openai.md`](openai.md)).

[codex-proxy]: local ChatGPT→OpenAI-compatible proxy (user-run).
