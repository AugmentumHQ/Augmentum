# Sampling reference — recommended params per installed model

Recommended sampling parameters for the LLMs in this install, pulled from the
**official model cards** (verbatim where the card gives numbers). This is the
source-of-truth behind `augmentum/models/sampling_profiles.py`:

- `_MODEL_SAMPLING` — the exact/substring table below (checked first).
- `_FAMILY_SAMPLING` — the coarse fallback for anything not listed.

The auto-import on download (`recommended_for`) seeds the **general-task**
profile. Where a model has a distinct **coding** profile (lower temp), it's
noted here — switch to it per-chat (composer "Tuning") or per-model (model
library "Tuning") when you're coding.

> Rule of thumb across every card: tune **either** temperature **or** top_p,
> not both. Reasoning/thinking models repeat at temp 0 (greedy) — don't.

---

## Qwen 3.x — the bulk of the fleet

Qwen split its guidance by **mode** (thinking vs instruct) and **task**
(general vs precise-coding). Important: base **Qwen3 = temp 0.6**, but
**Qwen3.5 / Qwen3.6 general thinking = temp 1.0** (a real change between
versions — only the *coding* profile stayed at 0.6).

| Model (yours) | General (thinking) | Coding (thinking) | Instruct (non-think) |
|---|---|---|---|
| `Qwen3-4B`, `qwen3-14b-hybrid0` (base Qwen3) | 0.6 / 0.95 / k20 | 0.6 / 0.95 / k20 | 0.7 / 0.8 / k20 |
| `Qwen3.5-4B`, `Qwen3.5-9B`, `Qwen3.5-122B-A10B` | **1.0** / 0.95 / k20 / pp1.5 | 0.6 / 0.95 / k20 / pp0 | 0.7 / 0.8 / k20 / pp1.5 |
| `Qwen3.6-27B` (+ `-Fable-5-Distill`) | **1.0** / 0.95 / k20 / pp0 | 0.6 / 0.95 / k20 / pp0 | 0.7 / 0.8 / k20 / pp1.5 |
| `Qwen3.6-35B-A3B` (MoE) | **1.0** / 0.95 / k20 / pp1.5 | 0.6 / 0.95 / k20 / pp0 | 0.7 / 0.8 / k20 / pp1.5 |
| `Qwen3.6-40B-Deck-Opus-NEO-CODE` (coding finetune) | — | **0.6** / 0.95 / k20 | — |

*(format: temperature / top_p / top_k · pp = presence_penalty)*

- MoE vs dense differ on presence_penalty: 35B-A3B (MoE) general = pp **1.5**;
  27B (dense) general = pp **0.0**. pp 1.5 cuts endless repetition but "may
  occasionally result in language mixing" — lower it if you see that.
- Output length: 32k recommended; up to 81920 for hard math/code.
- Sources: [Qwen3.6-27B card](https://huggingface.co/Qwen/Qwen3.6-27B),
  [Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B),
  [27B-vs-35B pp discussion](https://huggingface.co/Qwen/Qwen3.6-27B/discussions/10),
  [[reference_qwen3_thinking_sampling]].

## GLM-4.x — `Dolphin-Mistral-GLM-4.7-Flash-24B-Venice-Edition`

GLM-4.7-Flash base (despite the "Mistral" in the finetune name, the lineage is
GLM-4.7-Flash).

- **General: temperature 1.0, top_p 0.95.** Code bench (SWE/Terminal): 0.7 / 1.0.
- **Do NOT add top_k / min_p / repeat_penalty.** Community testing on the Flash
  GGUF found `--temp 0.7 --min-p 0 --top-p 0.8 --top-k 20 --repeat-penalty 1.05`
  **broke tool-calling and caused looping**; removing them (defaults) fixed it.
- Asymmetric thinking closer — handled in `utils/thinking.py`; turn on
  Preserved Thinking for multi-turn agentic tasks.
- Sources: [GLM-4.7-Flash card](https://huggingface.co/zai-org/GLM-4.7-Flash),
  [recommended-params discussion](https://huggingface.co/zai-org/GLM-4.7-Flash/discussions/6),
  [unsloth GGUF notes](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/discussions/1).

## NVIDIA Nemotron — `NVIDIA-Nemotron-3-Nano-4B`

Reasoning model.

- **Reasoning-on: temperature 0.6, top_p 0.95** (large max_new_tokens budget).
- Instruct/non-thinking: temperature 0.2, top_k 1 (near-greedy).
- Sources: [Llama-Nemotron paper](https://arxiv.org/pdf/2505.00949),
  [NeMo Evaluator reasoning docs](https://docs.nvidia.com/nemo/evaluator/latest/evaluation/run-evals/reasoning.html).

## Meta Llama — `Llama-3.3-70B-Instruct`

- **temperature 0.6, top_p 0.9** (Meta's `generation_config.json` default).
  Providers commonly start at 0.7 / 0.9; lower to 0–0.3 for code/factual.
- Sources: [Llama parameter quick-reference](https://muxup.com/2025q2/recommended-llm-parameter-quick-reference),
  [Llama 3.3 provider docs](https://docs.aimlapi.com/api-references/text-models-llm/meta/llama-3.3-70b-versatile).

## TheDrummer Rocinante — `Rocinante-XL-16B-v1a` (RP / Mistral-Nemo)

- Card gives **temp 0.7 ("chill") → 1.2 ("nitro")**; seeded at **1.0** with
  **min_p 0.02**. **DRY sampling recommended** (set `dry_multiplier` in the
  composer custom params). top_p/top_k are user-tuned — no official values.
- Source: [Rocinante card](https://huggingface.co/TheDrummer/Rocinante-12B-v1.1)
  (same maintainer guidance applies to the XL-16B).

## Gemma — `gemma-3-1b/4b`, `gemma-4-12B/26B-A4B/31B`

- **temperature 1.0, top_p 0.95, top_k 64** (Gemma 3 & 4). See
  `docs/model-cards/gemma-4-e2b.md` for the per-model gotchas (channel tokens,
  mmproj-for-vision). Note: the "greedy for Gemma classifier-tier" note in the
  bundled card is for the small **utility/classifier** slot only, not chat.

---

# Popular local models (not installed here — for self-hosters on bigger rigs)

Pre-seeded so a user who pulls these on heavier hardware than this box gets the
right auto-imported defaults, even though they're not in this install. Each
deviates from its family default, which is why it earns an exact entry.

| Model | Recommended | Notes / source |
|---|---|---|
| **QwQ-32B** (Qwen reasoning) | 0.6 / 0.95 / k40 · min_p 0 | Don't greedy-decode; top_k 20–40. llama.cpp defaults min_p 0.1 → force 0. [card](https://huggingface.co/Qwen/QwQ-32B/discussions/5) |
| **Mistral Small 3.x** (24B) | **temp 0.15** | Unusually low — official. [card](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506) |
| **Mistral Nemo** (12B) | **temp 0.3** | Lower than older Mistral (larger vocab). [card](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) |
| **Magistral** (Mistral reasoning) | 0.7 / 0.95 | `[THINK]` tokens (see thinking.py). [Mistral sampling](https://docs.mistral.ai/guides/sampling) |
| **Cohere Command-A** | 0.9 / 0.95 · repeat 1.04 | Creative default. [Command-A+](https://huggingface.co/CohereLabs/command-a-plus-05-2026-w4a4) |
| **Cohere Command-R** | temp 0.3 | Grounded/RAG default. [predictable outputs](https://docs.cohere.com/docs/predictable-outputs) |
| **Phi-4-reasoning / -mini-reasoning** | 0.8 / 0.95 / k50 | ChatML + system prompt. [card](https://huggingface.co/microsoft/Phi-4-reasoning-plus) |
| **Phi-4** (base) | temp 0.5 | No official; tech report used 0.5. |
| **Llama 3.1 / 3.2 / 3.3 / 4** | 0.6 / 0.9 | Meta `generation_config` default → now the **llama family** value. |
| **DeepSeek R1 + distills** | 0.6 / 0.95 | Family default (DeepSeek recommends 0.5–0.7). |
| **Mixtral / Mistral 7B / Large** | 0.7 / 0.95 | Mistral family default. |

> These came from mining the popular-model space (the litellm cost table lists
> what people actually run) + each model's own card. Cloud-only models
> (GPT/Claude/Gemini APIs) are intentionally **excluded** — this reference is
> local-oriented.

---

## How to update

When a new model lands, add a row above and (if its values differ from the
family default) an entry to `_MODEL_SAMPLING` in
`augmentum/models/sampling_profiles.py`, then add a test case in
`tests/test_sampling_profiles.py::test_exact_model_overrides_family`. The
download auto-seed will apply it to new pulls automatically; for already-
installed models, open the model library → "Tuning" → "Use recommended".
