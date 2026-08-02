# Gemma 4 E2B

> Google DeepMind's smallest Gemma 4 — "effective 2B" (5.1B w/ embeddings via
> Per-Layer Embeddings). Natively multimodal incl. video; the GPU opt-in for
> Augmentum's classifier sidecar, and (with its mmproj) reusable as the
> vision/video provider.

- **Role in Augmentum:** classifier sidecar (voice + architect router verdicts) when run on GPU; doubles as the vision/video captioner when launched with its mmproj and `classifier_vision_enabled` is on.
- **Wired in:** `compose.classifier-gpu.yaml` (launch) · `augmentum/models/provider_registry.py` (registered as `classifier` backend, `LlamaCppBackend`) · `augmentum/vision/provider.py::ClassifierVisionProvider` (vision) · `augmentum/utils/thinking.py` (Gemma 4 reasoning parser).
- **Default artifact:** `unsloth/gemma-4-E2B-it-qat-GGUF:UD-Q4_K_XL` via `AUGMENTUM_CLASSIFIER_HF` (source: `compose.classifier-gpu.yaml`). NOT the OSS default — the CPU SmolLM2-135M is (see [smollm2-135m](smollm2-135m.md)).
- **License:** Gemma (Apache-2.0-style open weights).

## Capabilities
- **Modalities:** text, image, **audio** (ASR/AST, E2B/E4B only), **video** (frame sequences, ≤60s @ 1fps). Image before text, audio after text in the prompt.
- **Function-calling:** native, structured tool use — this is why it's a strong classifier/agentic small model.
- **Context window:** 128K tokens.
- **Reasoning / thinking:** built-in, configurable. Enable by putting the `<|think|>` token at the **start of the system prompt**; disable by omitting it. Output uses the asymmetric Gemma-4 channel form `<|channel>thought\n…<channel|>` (closer is a different string from the opener — handled in `thinking.py::_FAMILY_PARSERS`).

## Recommended settings
- **Sampling:** `temperature=1.0`, `top_p=0.95`, `top_k=64` (source: model card + `compose.classifier.yaml` `AUGMENTUM_CLASSIFIER_SAMPLING_*`). Greedy (temp 0) is for SmolLM/Qwen — Gemma 4 needs these.
- **GPU:** `AUGMENTUM_CLASSIFIER_NGL=99` (needs the `server-cuda` image; the CPU `:server` tag silently ignores `--n-gpu-layers`).
- **Vision:** launch with the mmproj — `AUGMENTUM_CLASSIFIER_VISION_ARGS=--mmproj-url <gemma mmproj gguf>` (a text-only launch can't read images). Use a **low visual-token budget** for frames/video (`--image-min-tokens 70`; supported budgets 70/140/280/560/1120) — Gemma's guidance: low budget for classification/captioning/video, high for OCR/docs.

## Gotchas (the paid-for lessons)
- **Vision is off unless the mmproj is loaded.** The weights are multimodal; the running server is text-only until `--mmproj`/`--mmproj-url` is passed. This is why `classifier_vision_enabled` defaults OFF.
- **`<|think|>` token must be disabled** for the latency-critical classifier hop or it burns the 2.5s budget on a reasoning trace (`voice_router_parse_failed`, thinking_chars>0). The classifier callers schema-constrain output to the verdict JSON.
- **Reasoning leaks without the right parser** — Gemma 4 uses the asymmetric channel closer; relies on `thinking.py` + `--jinja`. Keep `skip_special_tokens=False` on the decode path or channel tokens get stripped and reasoning extraction silently fails.
- **CPU can't make the budget** at this size with the ~2400-token architect catalog prefill — Gemma-4-E2B as classifier wants GPU (`NGL=99`); on CPU it overruns 2.5s and the hop falls back / drops. SmolLM2-135M is the CPU floor.

## Sources
- Gemma 4 model card (Google DeepMind / Hugging Face).
- `compose.classifier.yaml`, `compose.classifier-gpu.yaml`
- `augmentum/utils/thinking.py` (Gemma 4 family parser)
- `augmentum/vision/provider.py` (ClassifierVisionProvider)
