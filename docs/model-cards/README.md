# Model Cards — natively-bundled models

Fast-reference capability cards for the models Augmentum ships / runs
**itself** (the always-on substrate: classifier, vision sibling, TTS, STT,
VAD). User-selected chat/coder models are NOT here — those are arbitrary and
discovered at runtime. This area exists so development decisions are grounded
in real capabilities and the per-model gotchas we've already paid for, instead
of re-deriving them from scattered code comments each time a new model lands.

**When to read:** before wiring a bundled model into a new path (does it do
vision? function-calling? what sampling? what control tokens?), or when a
model's behavior surprises you (leaked reasoning, empty captions, dropped
verdicts).

**Update discipline:** when you bump a bundled model's default (compose `-hf`,
`config.py` paths) or add a new always-on model, add/update its card in the
same change. Each card cites its source of truth (the compose file / config
field / model-card URL) so it stays checkable. Cards are point-in-time —
verify against the cited source before relying on a number.

Related grounding: `CLAUDE.md` (reasoning-parser families, engine v2),
`augmentum/vision/` (vision routing), `augmentum/voice/` (STT/TTS/VAD),
`compose.classifier*.yaml` (classifier launch).

## Cards

| Model | Role in Augmentum | Modalities | Fn-calling | Card |
|-------|-------------------|-----------|-----------|------|
| **Gemma 4 E2B** | Classifier sidecar (GPU opt-in); reused as vision/video provider | text, image, audio, **video** | native | [gemma-4-e2b](gemma-4-e2b.md) |
| **SmolLM2-135M** | Classifier sidecar (CPU default) | text | no | [smollm2-135m](smollm2-135m.md) |
| **SmolVLM-256M** | Vision captioner sibling (port 8092) | text, image | no | [smolvlm-256m](smolvlm-256m.md) |
| **Kokoro-82M** | In-process TTS (primary voice) | text→audio | no | [kokoro](kokoro.md) |
| **PocketTTS (Kyutai)** | CPU / on-device TTS, voice cloning | text→audio | no | [pocket-tts](pocket-tts.md) |
| **Moonshine** | Local STT (Deepgram alternative) | audio→text | no | [moonshine](moonshine.md) |
| **Silero VAD v6** | Server-side speech detection | audio→VAD | no | [silero-vad](silero-vad.md) |
| **Smart Turn v3** | Semantic turn-completion gate | audio→prob | no | [smart-turn-v3](smart-turn-v3.md) |

All cards are grounded in the official model card / repo (cited in each card's
**Sources**); Augmentum-specific config is cited from the repo file that's the
source of truth. Two facts are flagged for re-verification in
[pocket-tts](pocket-tts.md): official "English-only" vs our code's 6-language
claim, and official "no int8 compute" vs our on-device int8 notes.
