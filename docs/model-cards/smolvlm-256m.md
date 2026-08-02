# SmolVLM-256M-Instruct

> HuggingFace's 256M vision-language model. Augmentum's **default vision
> captioner sibling** — a dedicated small VL llama-server that captions images
> without touching the primary chat model's KV cache.

- **Role in Augmentum:** the SmolVLM sibling behind `VisionRouter` — captions images for the chat caption-fallback (text-only primary → inlined caption), file-index backfill, and background pipelines. The "keep the primary KV clean" workload slot. (For multimodal-classifier installs, the Gemma classifier can take this role instead — see [gemma-4-e2b](gemma-4-e2b.md).)
- **Wired in:** `augmentum/vision/provider.py::SmolVLMSibling` + `SmolVLMProvider` · `augmentum/vision/router.py::VisionRouter` · `augmentum/proxy/server.py` (startup, gated by `vision_provider_enabled`).
- **Default artifact:** base `/models/vision/SmolVLM-256M-Instruct-Q8_0.gguf` + projector `/models/vision/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf` (source: `config.py` `vision_provider_model_path` / `vision_provider_mmproj_path`; baked into `Dockerfile.gpu`). User-pickable from installed VL GGUFs via the settings captioner picker (`/api/models/vision/captioner-options`).
- **License:** Apache-2.0.

## Capabilities
- **Modalities:** text, image. (No audio/video — for video frames prefer the Gemma classifier provider.)
- **Function-calling:** no — caption/describe/VQA only.
- **Context window:** small; used for single-image caption turns, not long context.
- **Reasoning / thinking:** none.

## Recommended settings
- **Runs on its own port** (default 8092; primary uses 8091, classifier 8091-via-sidecar). Default **CPU-only** (`gpu_layers=0`); opt-in GPU for VRAM headroom.
- **Caption call:** OpenAI vision Chat Completions shape (`image_url` + text), `temperature=0.2` (steady captions). Goes through the shared `_caption_via_openai_endpoint` helper.
- Started in the background at lifespan (~3-5s CPU load); calls arriving before ready return empty captions and the pipeline degrades gracefully.

## Gotchas (the paid-for lessons)
- **Image format:** llama.cpp's `mtmd_helper`/stb_image only decodes PNG/JPEG/BMP/GIF/etc. WebP/AVIF/HEIC 400 with "Failed to load image" — the provider transcodes to PNG via Pillow first (`_ensure_stb_decodable`).
- **Quality cliff:** it's 256M — fine for "what's in this image" gist, weak on fine detail / small text. For OCR/doc parsing, route to a bigger VL (primary or a higher-budget Gemma).
- **Base + mmproj must be a matching pair** — dim-checked by `validate_mmproj_pair` before load; a mismatched projector crashes llama-server at load. The captioner picker only offers dim-compatible pairs.

## Sources
- SmolVLM model card (Hugging Face / `HuggingFaceTB`).
- `augmentum/vision/provider.py`, `augmentum/vision/router.py`
- `augmentum/config.py` (`vision_provider_*`)
