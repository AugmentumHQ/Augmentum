"""Compile-only speed test (bf16 + torch.compile)."""
from __future__ import annotations
import time
import torch
from peft import PeftModel
from transformers import AutoProcessor, CsmForConditionalGeneration

BASE, SR = "sesame/csm-1b", 24000
proc = AutoProcessor.from_pretrained(BASE)
m = CsmForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda").to(torch.bfloat16)
m = PeftModel.from_pretrained(m, "out/ruby-lora")
m.eval()
try:
    m.generation_config.cache_implementation = "static"
except Exception:
    pass
m.forward = torch.compile(m.forward, mode="reduce-overhead", fullgraph=False)


def gen(t):
    conv = [{"role": "0", "content": [{"type": "text", "text": t}]}]
    inp = proc.apply_chat_template(conv, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    s = time.monotonic()
    with torch.no_grad():
        a = m.generate(**inp, output_audio=True, do_sample=True, temperature=0.7)
    torch.cuda.synchronize()
    w = a[0].float().cpu().reshape(-1)
    return w.numel() / SR, time.monotonic() - s


print("compiling + warmup (slow first pass)...", flush=True)
gen("Warming up the engine now.")
gen("Warming up once more.")
a, d = gen("Hey, I'm really glad you're here. Want to pick up where we left off?")
print(f"bf16+compile: {a:.1f}s audio in {d:.1f}s  RTF={d/a:.2f}  ({a/d:.2f}x realtime)", flush=True)
