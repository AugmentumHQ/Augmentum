# Smart Turn v3 (Pipecat)

> Pipecat's open-weights **semantic** VAD — decides whether a speaker actually
> finished their turn by analysing the raw waveform (not the transcript). ~8M
> params, ~12 ms on CPU. Augmentum's turn-completion gate after Silero silence.

- **Role in Augmentum:** runs AFTER Silero VAD detects silence to predict turn-complete vs mid-thought pause ("um", thinking) — so the companion doesn't cut you off on a natural pause, and finalizes fast when you're genuinely done.
- **Wired in:** `augmentum/voice/smart_turn.py` (singleton ONNX session) · gated by `voice_smart_turn` (default True) in `voice_routes.py`. Model baked at `/home/augmentum/.smart-turn/`, with a data-dir cache + HF download fallback.
- **License:** BSD 2-Clause (official).

## Capabilities (official — pipecat-ai/smart-turn-v3)
- **What:** open-source semantic VAD over the raw waveform; tagged multilingual.
- **Architecture:** Whisper Tiny encoder backbone + shallow linear classifier head. **Params: 8M.**
- **Checkpoint:** 8 MB ONNX (int8) / 32 MB ONNX (unquantized).
- **Latency:** ~12 ms CPU inference (no PyTorch/CUDA needed).
- **Output:** probability the turn is complete.
- **Function-calling / modalities:** audio → turn-complete probability only.

## Augmentum wiring notes (`smart_turn.py` — v3.2 checkpoint)
- Input: up to **8 s** of 16 kHz mono float32 audio (`_MAX_SECONDS=8`).
- Threshold `_DEFAULT_THRESHOLD=0.5` (`voice_smart_turn_threshold`).
- Safety valve `voice_smart_turn_max_wait_s=3.0` — force-complete after this much silence even if the model says "incomplete."
- **Deferral cap** `voice_smart_turn_max_deferrals=3` — each "still speaking" defers the veto deadline by `max_wait_s`; without the cap, background noise Silero reads as speech defers forever and the turn "feels super long" (2026-06-13 fix). After the cap, finalize regardless (a real multi-pause thought is recoverable via continuation-merge).
- `voice_smart_turn_min_veto_confidence=0.3` — veto confidence is `threshold − prob` (NOT `prob`); weak vetoes are overridden, so with defaults prob∈(0.2,0.5) defers to VAD and prob≤0.2 (model sure user is still talking) is honored.

## Gotchas (the paid-for lessons)
- It's a **veto layer on top of Silero**, not a replacement — it narrows the 800 ms VAD silence to ~400 ms once it's confident, but the `max_wait_s` + deferral cap exist because an unbounded "still speaking" signal on noise wedges the turn.
- Confidence math is asymmetric (`threshold − prob`) — read the setting comments before tuning, or you'll invert the intent.

## Sources
- Official model card: https://huggingface.co/pipecat-ai/smart-turn-v3
- `augmentum/voice/smart_turn.py`
