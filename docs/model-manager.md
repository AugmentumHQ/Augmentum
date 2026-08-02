# Model Manager — power-user features

Most people never open the advanced settings, so these capabilities go
unnoticed. They're all in **Settings → Model Manager**, they all work today, and
several of them let modest hardware punch well above its weight. Here's what each
one actually unlocks.

Every model you install carries its own **engine profile** — set it once and
Augmentum remembers it for that model. The profile is where all of the below
live, and a one-line summary (e.g. `MoE: experts on CPU · Q8 KV · keep warm`)
shows on the model's row so you can see how it's tuned at a glance.

---

## Run big MoE models on modest VRAM (expert offload)

Mixture-of-Experts models (Qwen3-A3B, larger MoE releases) only activate a
fraction of their weights per token — but the *whole* model still has to live
somewhere. Expert offload keeps the always-hot attention layers on the GPU and
pushes the mostly-idle expert layers to system RAM, so a model that "shouldn't
fit" runs anyway, at a modest speed cost instead of not at all.

In the profile's **GPU layers** mode:

- **`MoE: experts on CPU`** — all expert layers on CPU, attention on GPU. The
  simplest way to fit a large MoE on a small card.
- **`MoE: first N on CPU`** — offload only the first *N* expert layers; tune the
  balance between VRAM used and speed.
- **`MoE: VRAM-balanced`** — Augmentum measures your free VRAM and picks the
  offload split automatically.

If you've ever been told a 30B+ MoE won't run on your GPU, try this first.

## Fit longer context in the same VRAM (KV-cache quantization)

The KV cache — the model's working memory of the conversation — grows with
context length and often uses more VRAM than the weights at long context. You can
store it at lower precision:

- **`Q8_0`** — roughly half the KV memory, negligible quality impact.
- **`Q4_0`** — roughly a quarter, for squeezing out maximum context.

Keys and values are set independently. The practical payoff: **much longer
conversations and documents on the same card**. Shows as `Q8 KV` on the model row.

## GPU layer control

Beyond the MoE modes, standard offload is here too — **Auto** (Augmentum
decides), **CPU only** (no GPU at all), or **Custom** (pin an exact number of
layers to the GPU). Useful when you're running several models at once and
budgeting VRAM by hand.

## Download models — with the quantization you want — from inside the app

No `huggingface-cli`, no manual file wrangling. Pick a GGUF repository from the
built-in catalog (or paste any HuggingFace repo), then choose the **exact quant
file** — Q4_K_M, Q6_K, Q8_0, and so on — and Augmentum pulls it and registers it.
Downloads show live progress and resume.

## Keep models warm (resident) instead of cold-loading

Each model can set an **idle timeout**: how long it stays loaded after its last
use before the engine frees its VRAM. Set a model to **stay resident** and it
never pays the cold-load tax again — the first token comes back immediately.
Great for your daily driver; set a short timeout on models you use rarely so they
don't hold VRAM hostage.

## Assign models to roles

Augmentum runs a few jobs behind the scenes — a small **utility** model for quick
internal tasks, a **classifier** that routes your requests and (if it's a
vision model) captions images, and an optional **heavyweight** model that harder
work escalates to. Each is a dropdown in the Model Manager. The classifier/utility
model is kept resident by default, so routing and voice stay fast — and you can
**swap it at runtime with no container restart**.

## Per-model reasoning and thinking

- **Reasoning format** — for models that emit hidden chain-of-thought, choose how
  it's parsed (`deepseek` / `auto` / `none`). `auto` detects the model family;
  `none` turns extraction off if a specific GGUF misbehaves.
- **Thinking toggle** — for models that support it (Qwen 3.x, GLM-4.x, EXAONE
  4.x, Nemotron), the chat composer's thinking button turns extended reasoning on
  or off per turn.

## Context size

Override a model's context window per-profile — cap it lower to save memory, or
raise it toward the model's maximum for long-document work (bounded by what the
model and your VRAM actually support).

---

**A note on defaults:** you don't need any of this to start. Fresh installs pick
sane defaults, and the onboarding flow walks you through choosing a first model.
These settings are here for when you want to push your specific hardware further —
which, on a self-hosted box, is exactly the point.
