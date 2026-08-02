# SmolLM2-135M-Instruct

> HuggingFace's 135M tiny instruct model. Augmentum's **OSS-default classifier
> sidecar** — small enough to prefill the architect router's ~2400-token
> action catalog on CPU inside the 2.5s budget.

- **Role in Augmentum:** classifier sidecar — voice-router + architect-router verdicts (`act / converse / clarify / idle / drop`). Schema-constrained output (verdict JSON only). Never leaks into chat or heavier reasoning (the role resolver only hands this backend to `role=="classifier"`).
- **Wired in:** `compose.classifier.yaml` (launch) · `augmentum/models/provider_registry.py` (`classifier` backend + `resolve_model_for_role`) · `augmentum/architect/` (router callers).
- **Default artifact:** `bartowski/SmolLM2-135M-Instruct-GGUF:Q8_0` via `AUGMENTUM_CLASSIFIER_HF` default (source: `compose.classifier.yaml`). Pulled from HF on first start into the `classifier_models` volume.
- **License:** Apache-2.0.

## Capabilities
- **Modalities:** text only.
- **Function-calling:** no — it doesn't invoke tools; it emits a constrained classification verdict that the router acts on.
- **Context window:** launched at `--ctx-size 8192` (`--parallel 2` halves per slot). The functional load is the ~2400-token catalog prefill every call.
- **Reasoning / thinking:** none (and not wanted — the budget is for the verdict).

## Recommended settings
- **Sampling:** greedy — `temperature=0.0`, `top_p=1.0`, `top_k=0` (source: `compose.classifier.yaml` `AUGMENTUM_CLASSIFIER_SAMPLING_*` defaults). Correct for SmolLM/Qwen2.5; do NOT use Gemma's 1.0/0.95/64 here.
- **CPU-only** by default (`--n-gpu-layers 0`); the hop fires <10×/min so CPU is plenty and it never contends for the chat/TTS GPU.
- **Threads:** `--threads 8` default.

## Gotchas (the paid-for lessons)
- **It's the CPU latency floor, not a quality choice.** 135M does ~2400 tokens in ~1s on 8 CPU threads; a 1.5B-class model on CPU overruns the 2.5s budget and the router times out → companion falls back / drops the utterance. To upgrade judgment, go GPU (see [gemma-4-e2b](gemma-4-e2b.md)) — don't put a bigger model on CPU.
- **Don't shrink the catalog to "fix" latency** — the ~2400 tokens are functional routing context, not waste.
- Replaced the brittle regex switchboard AND the flaky remote-LLM classifier (the DeepSeek fallback that 400'd on `json_schema` and dropped utterances) — keep verdicts local so they can't 500 / time out on a network hop.

## Sources
- SmolLM2 model card (Hugging Face / `HuggingFaceTB`).
- `compose.classifier.yaml`
- `augmentum/models/provider_registry.py::resolve_model_for_role`
