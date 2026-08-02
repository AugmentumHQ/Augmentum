"""Bridge a merged HF CSM voice into the fork's original Sesame format, so the
streaming sidecar can load it for real-time generation.

Patches ONLY the LoRA-touched tensors (backbone + decoder attn/mlp projections)
of the base Sesame model.safetensors with our fine-tuned weights — a pure
rename+copy (shapes are identical, no transpose). Everything else (norms,
embeddings, heads, Mimi codec) stays at base.

  .venv/Scripts/python bridge_to_sesame.py ruby
  -> out/ruby-sesame/model.safetensors  (+ config.json copied)  drop-in for the fork
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

voice = sys.argv[1] if len(sys.argv) > 1 else "ruby"
ATTN = {"q_proj": "q_proj", "k_proj": "k_proj", "v_proj": "v_proj", "o_proj": "output_proj"}
MLP = {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"}

# CRITICAL: HF (rotate-half RoPE) and torchtune/Sesame (interleaved RoPE) lay
# out q_proj/k_proj rows differently. A direct copy scrambles positional
# encoding -> degraded, artifact-laden, erratic audio. We undo HF's permute on
# q/k only (v/o/mlp are rotation-independent). (n_heads, n_kv_heads) per stack:
HEADS = {"backbone": (32, 8), "decoder": (8, 2)}


def hf_to_sesame_rope(w, n_heads):
    """Inverse of HF's Llama q/k permute: HF rotate-half layout -> interleaved."""
    d1, d2 = w.shape
    return w.view(n_heads, 2, d1 // n_heads // 2, d2).transpose(1, 2).reshape(d1, d2)

print("[bridge] loading base Sesame model.safetensors ...", flush=True)
base = load_file(hf_hub_download("sesame/csm-1b", "model.safetensors"))

print(f"[bridge] loading merged HF weights from out/{voice}-merged ...", flush=True)
hf = {}
for s in glob.glob(f"out/{voice}-merged/*.safetensors"):
    hf.update(load_file(s))

patched = dict(base)
n = 0
miss = []
for k, v in hf.items():
    sk = None
    p = k.split(".")
    if k.startswith("backbone_model.layers.") and ".self_attn." in k:
        sk = f"backbone.layers.{p[2]}.attn.{ATTN[p[4]]}.weight"
    elif k.startswith("backbone_model.layers.") and ".mlp." in k and p[4] in MLP:
        sk = f"backbone.layers.{p[2]}.mlp.{MLP[p[4]]}.weight"
    elif k.startswith("depth_decoder.model.layers.") and ".self_attn." in k:
        sk = f"decoder.layers.{p[3]}.attn.{ATTN[p[5]]}.weight"
    elif k.startswith("depth_decoder.model.layers.") and ".mlp." in k and p[5] in MLP:
        sk = f"decoder.layers.{p[3]}.mlp.{MLP[p[5]]}.weight"
    if sk is None:
        continue
    if sk not in base:
        miss.append(sk)
        continue
    if base[sk].shape != v.shape:
        raise SystemExit(f"SHAPE MISMATCH {sk}: base {tuple(base[sk].shape)} vs hf {tuple(v.shape)}")
    val = v.to(base[sk].dtype)
    # q/k need the RoPE-layout fix; v/o/mlp are copied as-is.
    if sk.endswith("attn.q_proj.weight") or sk.endswith("attn.k_proj.weight"):
        stack = "backbone" if sk.startswith("backbone.") else "decoder"
        nh, nkv = HEADS[stack]
        val = hf_to_sesame_rope(val, nh if sk.endswith("q_proj.weight") else nkv)
    patched[sk] = val.contiguous()
    n += 1

if miss:
    print(f"[bridge] WARN {len(miss)} mapped keys not in base (e.g. {miss[:3]})")
print(f"[bridge] patched {n} tensors (expect 140: 16 backbone + 4 decoder layers x 7)", flush=True)

out = Path(f"out/{voice}-sesame")
out.mkdir(parents=True, exist_ok=True)
save_file(patched, str(out / "model.safetensors"), metadata={"format": "pt"})
# the fork's Model.from_pretrained also wants config.json
cfg = hf_hub_download("sesame/csm-1b", "config.json")
shutil.copy(cfg, out / "config.json")
print(f"[bridge] wrote {out}/model.safetensors ({len(patched)} tensors) + config.json", flush=True)
print("[bridge] sanity: all base keys present?",
      all(k in patched for k in base), "| total", len(patched), "vs base", len(base))
