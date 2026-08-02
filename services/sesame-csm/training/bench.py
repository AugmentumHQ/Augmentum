"""Benchmark CSM generation speed: fp32 vs bf16 vs bf16+compile on this GPU.
Finds what reaches real-time (RTF < 1) before we bake it into the server.

  .venv/Scripts/python bench.py
"""
from __future__ import annotations

import time

import torch
from peft import PeftModel
from transformers import AutoProcessor, CsmForConditionalGeneration

BASE = "sesame/csm-1b"
SR = 24000
ADAPTER = "out/ruby-lora"
TEXT = "Hey, I'm really glad you're here. Want to pick up where we left off?"

proc = AutoProcessor.from_pretrained(BASE)


def load(dtype, do_compile):
    m = CsmForConditionalGeneration.from_pretrained(BASE, torch_dtype=dtype, device_map="cuda")
    m = m.to(dtype)                       # force-cast stragglers (Mimi) → uniform dtype
    m = PeftModel.from_pretrained(m, ADAPTER)
    m.eval()
    if do_compile:
        try:
            m.generation_config.cache_implementation = "static"
        except Exception:
            pass
        m.forward = torch.compile(m.forward, mode="reduce-overhead", fullgraph=False)
    return m


def gen(m, text):
    conv = [{"role": "0", "content": [{"type": "text", "text": text}]}]
    inp = proc.apply_chat_template(conv, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    t = time.monotonic()
    with torch.no_grad():
        a = m.generate(**inp, output_audio=True, do_sample=True, temperature=0.7)
    torch.cuda.synchronize()
    wav = a[0].float().cpu().reshape(-1)
    return wav.numel() / SR, time.monotonic() - t


for name, dtype, comp in [("fp32", torch.float32, False),
                          ("bf16", torch.bfloat16, False),
                          ("bf16+compile", torch.bfloat16, True)]:
    try:
        m = load(dtype, comp)
        gen(m, "Warming up the engine now.")        # warmup (compile + cuda-graph capture)
        gen(m, "Warming up once more please.")
        a_s, dt = gen(m, TEXT)
        rtf = dt / a_s if a_s else 0
        print(f"{name:14} {a_s:.1f}s audio in {dt:.1f}s  RTF={rtf:.2f}  ({a_s/dt:.2f}x realtime)", flush=True)
        del m
        torch.cuda.empty_cache()
    except Exception as e:  # noqa: BLE001
        print(f"{name:14} FAILED: {type(e).__name__}: {str(e)[:240]}", flush=True)
        torch.cuda.empty_cache()
