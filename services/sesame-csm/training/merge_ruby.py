"""Merge a voice LoRA into a standalone fine-tuned CSM model (HF format) and
dump its key structure — step 1 of bridging the voice into the streaming
sidecar's engine.

  .venv/Scripts/python merge_ruby.py ruby
"""
from __future__ import annotations

import sys

import torch
from peft import PeftModel
from transformers import CsmForConditionalGeneration

voice = sys.argv[1] if len(sys.argv) > 1 else "ruby"
print(f"[merge] loading base + {voice} adapter (CPU fp32)...", flush=True)
base = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b", torch_dtype=torch.float32)
merged = PeftModel.from_pretrained(base, f"out/{voice}-lora").merge_and_unload()
out = f"out/{voice}-merged"
merged.save_pretrained(out)
print(f"[merge] saved -> {out}", flush=True)

sd = merged.state_dict()
print(f"\n[keys] {len(sd)} tensors. Structure (layer 0 + top-level heads/embeds):")
seen_prefixes = set()
for k in sd:
    top = k.split(".")[0]
    if top not in seen_prefixes:
        seen_prefixes.add(top)
        print(f"  top-level group: {top}")
print("\n[layer-0 / head / embed keys LoRA-relevant]:")
for k in sd:
    if (".layers.0." in k and ("proj" in k)) or any(s in k for s in
            ("embed", "head", "projection", "codebook")):
        if ".layers." not in k or ".layers.0." in k:
            print(f"  {k}  {tuple(sd[k].shape)}")
