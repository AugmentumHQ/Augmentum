# Image generation

Augmentum generates images itself — it doesn't just call someone else's API
(though it can). Text-to-image, image-to-image, inpainting, upscaling, and
background removal all run through one pipeline, and the same generated images can
illustrate a story, land in an artifact, or be pulled by any client through the
open tools.

## Where the pictures come from — three tiers

A request resolves to the first available of:

1. **Local pipeline** *(self-hosted, GPU)* — a diffusers-based, ComfyUI-inspired
   pipeline that loads Stable-Diffusion-family models generically and runs them on
   your own GPU. No external service, no per-image cost, nothing leaves your box.
2. **Cloud providers** *(optional)* — if you add an image provider (an API you
   supply a key for), Augmentum can dispatch to it — useful on a machine with no
   GPU, or when you want a specific hosted model. You add these deliberately; none
   ship configured.
3. **Fabric peers** *(optional)* — if you run [Fabric](fabric.md), a request can be
   offloaded to another of your machines that *does* have a GPU. Your laptop
   generates on your tower.

Check what's live at **Settings → Images** (or `GET /api/images/availability`) —
it reports whether the local pipeline is ready, how many cloud providers are
enabled, and whether peers can serve.

> **You pick the model.** Augmentum never silently chooses a checkpoint for you.
> Download or select one in the image model manager; the picker shows what's
> installed and what you can pull.

## Getting a model

On a GPU box, open the image model manager and **pull** a checkpoint
(`POST /api/images/models/pull`, progress at `/api/images/models/pull/{task_id}`).
Detection (`/api/images/models/detect`) surfaces models already on disk. Once a
model is installed and selected, generation is available.

No GPU? Add a cloud provider instead (Settings → Images → providers), or point a
Fabric peer with a GPU at this instance.

## What you can do

All of these are in the image surface, and each has an API route under
`/api/images/`:

- **Generate** from a prompt (`/generate`) — with prompt enhancement
  (`/enhance-prompt`) and auto-negative-prompt (`/generate-negative`) helpers.
- **Image-to-image** (`/img2img`) — transform an existing image with a prompt.
- **Inpaint** (`/inpaint`) — paint into a masked region.
- **Upscale** (`/{image_id}/upscale`) and **remove background**
  (`/{image_id}/remove-bg`) — post-process any generated image.
- **Scene generation** (`/generate-scene`) — the hook narrative/companion use to
  illustrate a moment, including extracting visual traits from a character.

Long generations run as background jobs (`/job/{job_id}`,
`/generation-status`), so the UI shows real progress rather than a blank wait.

## From other clients (open tool)

Image generation is also exposed as an **open tool** through the ATP surface
(`/v1/tools`) and the OpenAI-compatible `/v1/images/generations` endpoint — so any
client pointed at Augmentum (or the model itself, mid-conversation) can request an
image the same way it would call web search or the calculator. See
[External API](external-api.md).

## Settings worth knowing

- `image_enabled` — master switch for the local pipeline.
- Precision/variant selection is automatic by default (based on your VRAM) but
  can be forced (`fp32`/`fp16`) in the model manager for quality-vs-memory
  trade-offs.
- Generated images are stored per-user and scoped to your account — like
  everything else, they don't cross tenants.
