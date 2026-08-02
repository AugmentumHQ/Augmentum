"""Clean re-probe: NO audio anchor (so no context-seam garble), so we judge each
model's INTRINSIC voice consistency + emotion response on speaker-id 0.

Answers the question the anchored probe muddied: across emotions, does the voice
stay one person (identity) and does delivery change (expression)?
"""
from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from safetensors.torch import load_file
from transformers import AutoProcessor, CsmForConditionalGeneration

from probe_decouple import BASE, EMOTIONS, ONECXI, SR, TARGET


def gen(model, processor, text):
    conv = [{"role": "0", "content": [{"type": "text", "text": text}]}]
    inputs = processor.apply_chat_template(
        conv, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        audio = model.generate(**inputs, output_audio=True)
    wav = audio[0].to(torch.float32).cpu()
    return wav.unsqueeze(0) if wav.dim() == 1 else wav


def run(name, weights):
    print(f"\n=== {name} ===", flush=True)
    processor = AutoProcessor.from_pretrained(BASE)
    model = CsmForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.float32, device_map="cuda")
    if weights:
        model.load_state_dict(load_file(str(weights)), strict=False)
    model.eval()
    out = Path("samples/probe_clean")
    out.mkdir(parents=True, exist_ok=True)
    for emo in EMOTIONS:
        text = TARGET if emo == "neutral" else f"({emo}) {TARGET}"
        try:
            wav = gen(model, processor, text)
            torchaudio.save(str(out / f"{name}__{emo}.wav"), wav, SR)
            print(f"[{name}] {emo:8s} {wav.shape[-1]/SR:5.2f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] {emo:8s} FAIL {str(e)[:120]}", flush=True)
    del model
    torch.cuda.empty_cache()


def build_html():
    out = Path("samples/probe_clean")
    by = {}
    for w in sorted(out.glob("*.wav")):
        by.setdefault(w.stem.split("__")[0], []).append(w)
    cards = "".join(
        f'<div class=card><b>{m}</b>' + "".join(
            f'<div class=clip><span>{f.stem.split("__")[1]}</span>'
            f'<audio controls preload=none src="probe_clean/{f.name}"></audio></div>'
            for f in fs) + "</div>"
        for m, fs in by.items())
    html = f"""<!doctype html><meta charset=utf-8><title>Clean probe (no anchor)</title>
<style>body{{font:15px/1.5 system-ui;margin:24px;background:#14151a;color:#e8e8ea}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}}
.card{{background:#1d1f26;border:1px solid #2a2d36;border-radius:12px;padding:14px}}
.card b{{font-size:18px}}.clip{{margin:8px 0}}.clip span{{display:block;font-size:11px;color:#9aa}}
audio{{width:100%;height:34px}}p.sub{{color:#9aa}}</style>
<h1>Clean probe — no audio anchor (no seam garble)</h1>
<p class=sub>Same sentence, only the (emotion) tag changes. Now judge: does each model
stay ONE voice across the row (identity), and does delivery change (expression)?</p>
<div class=grid>{cards}</div>"""
    Path("samples/probe_clean.html").write_text(html, encoding="utf-8")
    print(f"\n[done] open: {Path('samples/probe_clean.html').resolve()}")


def main():
    run("base-csm", None)
    if ONECXI.exists():
        run("onecxi", ONECXI)
    build_html()


if __name__ == "__main__":
    main()
