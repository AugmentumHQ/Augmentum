"""Turn weights-only onecxi into a loadable base model dir we can fine-tune FROM.

onecxi ships as a bare model.safetensors (no config/processor). We load base
CSM-1B for the architecture + processor, swap in onecxi's expressive weights
(strict=False keeps base's shared audio-token embedding — the 1 key onecxi
omits), and save a complete HF dir. Saved bf16 to respect tight disk; train.py
reloads it as fp32.

  .venv/Scripts/python.exe assemble_onecxi_base.py
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoProcessor, CsmForConditionalGeneration

BASE = "sesame/csm-1b"
ONECXI = Path.home() / (
    ".cache/huggingface/hub/models--onecxi--csm-english-multi-speaker-v2"
    "/snapshots/9bcd64f8d1278e2d926d7ed5fb839329391c85fa/model.safetensors"
)
OUT = "out/onecxi-base"


def main():
    proc = AutoProcessor.from_pretrained(BASE)
    model = CsmForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float32)
    res = model.load_state_dict(load_file(str(ONECXI)), strict=False)
    print(f"swapped onecxi weights: missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}")
    if res.missing_keys:
        print(f"  kept-from-base: {res.missing_keys}")
    assert len(res.unexpected_keys) == 0, "format mismatch — onecxi keys don't fit CSM arch"
    model.to(torch.bfloat16)
    model.save_pretrained(OUT)
    proc.save_pretrained(OUT)
    print(f"[done] loadable onecxi base -> {OUT}")


if __name__ == "__main__":
    main()
