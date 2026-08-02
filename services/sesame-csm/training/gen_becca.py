"""Generate the clean emotion set with the Becca-on-onecxi adapter, and measure
the two things that decide success:

  1. Did IDENTITY move toward Becca? (timbre proximity to the Ruby anchor, vs
     onecxi-base's proximity — closer = the light LoRA pinned her identity)
  2. Did onecxi's EXPRESSION survive? (pitch spread across emotions, vs onecxi's
     128 Hz baseline — still wide = expression preserved, not flattened)

Outputs samples/becca/*.wav + samples/becca.html (A/B vs onecxi-base clean clips).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF
from peft import PeftModel
from torchaudio.transforms import MFCC
from transformers import AutoProcessor, CsmForConditionalGeneration

from probe_decouple import EMOTIONS, SR, TARGET

BASE = "./out/onecxi-base"
ADAPTER = "out/becca-onecxi-lora"
_mf = MFCC(sample_rate=SR, n_mfcc=20, melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 64})


def gen(model, processor, text):
    conv = [{"role": "0", "content": [{"type": "text", "text": text}]}]
    inputs = processor.apply_chat_template(
        conv, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        audio = model.generate(**inputs, output_audio=True)
    wav = audio[0].to(torch.float32).cpu()
    return wav.unsqueeze(0) if wav.dim() == 1 else wav


def tvec(p):
    x, sr = torchaudio.load(str(p))
    x = x.mean(0)
    if sr != SR:
        x = AF.resample(x, sr, SR)
    return _mf(x)[1:].mean(1)


def pitch_median(p):
    x, sr = torchaudio.load(str(p))
    x = x.mean(0)
    if sr != SR:
        x = AF.resample(x, sr, SR)
    f0 = AF.detect_pitch_frequency(x.unsqueeze(0), SR).squeeze()
    v = f0[(f0 > 70) & (f0 < 450)]
    return v.median().item() if v.numel() > 2 else float("nan")


def main():
    processor = AutoProcessor.from_pretrained(BASE)
    model = CsmForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.float32, device_map="cuda")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    out = Path("samples/becca")
    out.mkdir(parents=True, exist_ok=True)
    for emo in EMOTIONS:
        text = TARGET if emo == "neutral" else f"({emo}) {TARGET}"
        wav = gen(model, processor, text)
        torchaudio.save(str(out / f"becca__{emo}.wav"), wav, SR)
        print(f"[becca] {emo:8s} {wav.shape[-1]/SR:5.2f}s", flush=True)
    del model
    torch.cuda.empty_cache()

    # --- analysis: identity-to-Ruby + expression-preserved, becca vs onecxi-base ---
    rows = [json.loads(l) for l in open("data/ruby/train.jsonl", encoding="utf-8") if l.strip()]
    pref = {"neutral", "regular", "calm", "serenity", "contentment", "interest"}
    cand = sorted([r for r in rows if 2.3 <= float(r.get("duration", 99)) <= 4.6],
                  key=lambda r: (r.get("emotion") not in pref, -len(r["raw_text"])))
    anchor = tvec(Path("data/ruby") / cand[0]["audio"])

    def report(label, folder, stem):
        sims, pitches = [], []
        for emo in EMOTIONS:
            p = Path(folder) / f"{stem}__{emo}.wav"
            if not p.exists():
                continue
            sims.append(torch.nn.functional.cosine_similarity(tvec(p), anchor, dim=0).item())
            pm = pitch_median(p)
            if pm == pm:  # not NaN
                pitches.append(pm)
        id_to_ruby = sum(sims) / len(sims)
        pspread = max(pitches) - min(pitches)
        print(f"  {label:16s} identity->Ruby {id_to_ruby:.3f} | pitch spread {pspread_fmt(pspread)} Hz")
        return id_to_ruby, pspread

    def pspread_fmt(v):
        return f"{v:5.1f}"

    print("\n=== did it work? (higher identity->Ruby = more Becca; pitch spread ~>=100 = expression kept) ===")
    base_id, base_sp = report("onecxi-base", "samples/probe_clean", "onecxi")
    becca_id, becca_sp = report("becca(onecxi)", "samples/becca", "becca")
    print(f"\n  identity gain toward Becca: {becca_id - base_id:+.3f}")
    print(f"  expression retained: {becca_sp:.0f} Hz vs onecxi {base_sp:.0f} Hz "
          f"({'KEPT' if becca_sp > 0.6 * base_sp else 'FLATTENED'})")

    # html
    o = Path("samples/becca")
    cards = []
    for label, folder, stem in [("onecxi-base", "probe_clean", "onecxi"), ("becca (onecxi+LoRA)", "becca", "becca")]:
        clips = "".join(
            f'<div class=clip><span>{e}</span>'
            f'<audio controls preload=none src="{folder}/{stem}__{e}.wav"></audio></div>'
            for e in EMOTIONS if (Path("samples") / folder / f"{stem}__{e}.wav").exists())
        cards.append(f'<div class=card><b>{label}</b>{clips}</div>')
    html = f"""<!doctype html><meta charset=utf-8><title>Becca on onecxi</title>
<style>body{{font:15px/1.5 system-ui;margin:24px;background:#14151a;color:#e8e8ea}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}}
.card{{background:#1d1f26;border:1px solid #2a2d36;border-radius:12px;padding:14px}}
.card b{{font-size:18px}}.clip{{margin:8px 0}}.clip span{{display:block;font-size:11px;color:#9aa}}
audio{{width:100%;height:34px}}p.sub{{color:#9aa}}</style>
<h1>Becca on onecxi — did identity pin while expression survived?</h1>
<p class=sub>Left = expressive base (onecxi). Right = + light Becca LoRA. Judge: does the
right column sound MORE like your intended Becca, while still varying by emotion?</p>
<div class=grid>{''.join(cards)}</div>"""
    Path("samples/becca.html").write_text(html, encoding="utf-8")
    print(f"\n[done] open: {Path('samples/becca.html').resolve()}")


if __name__ == "__main__":
    main()
