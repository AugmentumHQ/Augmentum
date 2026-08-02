# Image Generation (cloud) — Combined Reference Card

> **Verbatim reference** for the cloud **image** providers Augmentum talks to. Cloud image
> providers are **DB-driven** (`image_providers` table) and proxied by `proxy/cloud_image_routes.py`,
> which special-cases **Stability / BFL / Fal** auth + request shape (`_is_stability`/`_is_bfl`/`_is_fal`);
> OpenAI/Together use the OpenAI `/images/generations` shape. **Sourced:** 2026-06-25 · Sources per §.

---

## 1. OpenAI — `gpt-image` / DALL·E

| | |
|---|---|
| **Base URL** | `https://api.openai.com/v1` · `POST /images/generations` (+ `/edits`, `/variations`) |
| **Auth** | `Authorization: Bearer <KEY>` |

- **`gpt-image-1`** (+ `gpt-image-1-mini`): `prompt`, `size` (`1024x1024`/`1536x1024`/`1024x1536`/`auto`), `quality` (`low`/`medium`/`high`/`auto`), `background` (`transparent`/`opaque`/`auto`), `output_format` (`png`/`jpeg`/`webp`), `output_compression`, `n`, `moderation` (`auto`/`low`). Returns **b64_json** (no URL). Token-billed (text+image tokens).
- **`dall-e-3`**: `size` (1024²/1792×1024/1024×1792), `quality` (`standard`/`hd`), `style` (`vivid`/`natural`), `n=1`. Returns URL or b64.
**Source:** platform.openai.com/docs/api-reference/images

---

## 2. Stability AI — Stable Image

| | |
|---|---|
| **Base URL** | `https://api.stability.ai` · **multipart/form-data** |
| **Auth** | `Authorization: Bearer <KEY>` · `Accept: image/*` |
| **Endpoints** | `/v2beta/stable-image/generate/ultra` · `/core` · `/sd3` |

**Params:** `prompt`, `negative_prompt`, `aspect_ratio`, `output_format` (png/jpeg/webp), `mode` (text-to-image / image-to-image), `model` (sd3.5-large/-turbo/-medium for the sd3 route), `seed`, `strength` (img2img). **Credit-based** per generation (Ultra > Core).
**Source:** platform.stability.ai/docs/api-reference

---

## 3. Black Forest Labs — FLUX

| | |
|---|---|
| **Base URL** | `https://api.bfl.ai` (+ `api.eu.bfl.ai` / `api.us.bfl.ai`) |
| **Auth** | **`x-key: <KEY>`** header |
| **Pattern** | **async** — `POST /v1/{model}` → `{id, polling_url}` → GET poll until `status=="Ready"` → `result.sample` (signed URL, **valid 10 min**) |

**Models:** `flux-2-pro-preview` (latest), `flux-2-pro`, `flux-2-flex`, `flux-2-klein-9b(-preview)`, `flux-2-max`, `flux-kontext-pro`/`-max` (editing, up to 10 ref images), `flux-pro-1.1`, `flux-pro-1.1-ultra` (4MP), `flux-pro`, `flux-dev`.
**Params:** `prompt`, `width`, `height`, `aspect_ratio`, `seed`, `prompt_upsampling`, `safety_tolerance`, `output_format`, input images (editing).
**Limits:** 24 concurrent (kontext-max 6); 402 = no credits, 429 = rate. **Retrieve result within 10 min.**
**Source:** docs.bfl.ai/quick_start/generating_images

---

## 4. Fal.ai

| | |
|---|---|
| **Base URL** | `fal.run` (direct) · `queue.fal.run` (queue) |
| **Auth** | **`Authorization: Key <id:secret>`** |
| **Model id in path** | e.g. `fal-ai/flux/dev`, `fal-ai/flux-pro`, `fal-ai/ideogram/v3` |

**Calling patterns:** `run` (sync), `subscribe` (queue+autopoll), `submit` (async queue), streaming, `realtime()` (WS <100ms).
**Common params:** `prompt`, `image_size`, `num_inference_steps`, `guidance_scale`, `seed`, `num_images`. Returns CDN URLs.
**Pricing:** pay-per-use — **per-megapixel** or **per-second** depending on model (per model page). 1,000+ models.
**Source:** fal.ai/docs/model-apis

---

## 5. Ideogram — V3

| | |
|---|---|
| **Base URL** | `https://api.ideogram.ai` · `POST /v1/ideogram-v3/generate` · **multipart** |
| **Auth** | **`Api-Key: <KEY>`** header |

**Params:** `prompt`*, `aspect_ratio` (1x1 def, 16x9, 9x16, …), `rendering_speed` (`FLASH`/`TURBO`/`DEFAULT`/`QUALITY`), `style_type` (`AUTO`/`GENERAL`/`REALISTIC`/`DESIGN`/`FICTION`), `magic_prompt` (`AUTO`/`ON`/`OFF`), `negative_prompt`, `seed`, `num_images`, `style_preset` (50+, e.g. `OIL_PAINTING`/`POP_ART`), `color_palette` (preset or hex+weights).
**Response:** `data[]` with `url` (**ephemeral — download to keep**), `resolution`, `is_image_safe`, `seed`, modified `prompt`. Best-in-class **text rendering**.
**Source:** developer.ideogram.ai/api-reference/api-reference/generate-v3

---

## 6. Together AI — image

OpenAI-compat `POST /v1/images/generations` at `https://api.together.xyz/v1` (same key as [together.md](together.md)). Hosts **FLUX.1 [schnell]/[dev]/[pro]** + SDXL. Params: `model`, `prompt`, `width`, `height`, `steps`, `n`, `seed`, `negative_prompt`, `response_format` (b64/url). Per-image, size×steps tiered.

---

## 7. NVIDIA — image (build.nvidia.com)

NIM-hosted image models (FLUX.1-dev, SDXL, Sana, Consistory) via `https://ai.api.nvidia.com/v1/genai/{vendor}/{model}` (auth `Authorization: Bearer <KEY>`). Params per model: `prompt`, `cfg_scale`, `steps`, `seed`, `width`/`height`, `negative_prompt`. Same credit model as NIM LLM ([nvidia.md](nvidia.md)).

---

## 8. xAI — grok-imagine

OpenAI-compat `POST /v1/images/generations` at `https://api.x.ai/v1` (same key as [xai.md](xai.md)). Model `grok-2-image` / `grok-imagine`. Params: `prompt`, `n` (1–10), `response_format` (`url`/`b64_json`). No size/quality knobs (model-fixed). Per-image pricing.

---

## Known drift / gaps (image)

- 🟢 **Three non-OpenAI shapes are special-cased** — Stability (multipart + `Accept: image/*`), BFL (`x-key` + **async poll**, 10-min URL expiry), Fal (`Key id:secret` + queue + model-in-path). A provider added with the wrong base-URL host won't match the special-case branch → falls back to OpenAI shape and fails. Verify `_is_stability`/`_is_bfl`/`_is_fal` host checks cover regional hosts (e.g. `api.eu.bfl.ai`, `api.us.bfl.ai`).
- 🟢 **BFL 10-minute signed-URL expiry** — results must be downloaded/persisted within 10 min of `Ready`; a slow path loses the image.
- 🟢 **Ideogram/Fal ephemeral URLs** — must download to keep; don't store the URL as the artifact.
- 🟢 `gpt-image-1` returns **b64 only** (no URL) and is **token-billed** — cost accounting differs from per-image providers.
