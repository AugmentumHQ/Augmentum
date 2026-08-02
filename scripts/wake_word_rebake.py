#!/usr/bin/env python3
"""One-off Phase 0.5 bake of a wake-word model with personal-sample
oversampling enabled (training.PERSONAL_OVERSAMPLE_FACTOR).

Bakes to a sidecar path so the running production model stays
untouched until the eval harness validates the rebake delivers the
predicted FRR drop on real user voice.

Usage (inside the augmentum container):

    docker exec augmentum-augmentum-1 python3 /app/scripts/wake_word_rebake.py \\
        --avatar-id bundled_f_becca \\
        --phrase "hey becca" \\
        --personal-dir /tmp/becca_recordings_all \\
        --output /tmp/bundled_f_becca_rebake.onnx
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/app")

from augmentum.config import settings  # noqa: E402
from augmentum.voice.kokoro_tts import KokoroTTS  # noqa: E402
from augmentum.voice.wake_word.training import (  # noqa: E402
    PERSONAL_OVERSAMPLE_FACTOR,
    train_wake_word_model,
)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--avatar-id", required=True, help="e.g. bundled_f_becca")
    ap.add_argument("--phrase", required=True, help="e.g. 'hey becca'")
    ap.add_argument("--personal-dir", type=Path, required=True,
                    help="dir of personal-sample WAVs")
    ap.add_argument("--output", type=Path, required=True,
                    help="output ONNX path (sidecar — does NOT overwrite prod)")
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    print(f"== Phase 0.5 rebake — {args.avatar_id}")
    print(f"  phrase                : {args.phrase!r}")
    print(f"  personal-dir          : {args.personal_dir}")
    print(f"  output                : {args.output}")
    print(f"  PERSONAL_OVERSAMPLE_FACTOR: {PERSONAL_OVERSAMPLE_FACTOR}")
    n_personal = len(list(args.personal_dir.glob("*.wav")))
    print(f"  detected personal WAVs: {n_personal}")
    print(f"  effective personal positives: {n_personal * PERSONAL_OVERSAMPLE_FACTOR}")
    print()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("loading Kokoro …", flush=True)
    kokoro = KokoroTTS.instance(model_dir=settings.tts_kokoro_model_dir)
    await asyncio.to_thread(kokoro.load_model)
    if not kokoro.is_available:
        raise SystemExit("Kokoro unavailable")

    t0 = time.monotonic()
    result = await train_wake_word_model(
        phrase=args.phrase,
        output_path=args.output,
        kokoro=kokoro,
        epochs=args.epochs,
        personal_samples_dir=args.personal_dir,
    )
    elapsed = time.monotonic() - t0

    print()
    print("=" * 70)
    print(f"REBAKE COMPLETE in {elapsed:.1f}s")
    print("=" * 70)
    print(json.dumps(result.__dict__, indent=2, default=str))
    print()
    print(f"ONNX written: {args.output}")
    print(f"  size: {args.output.stat().st_size:,} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
