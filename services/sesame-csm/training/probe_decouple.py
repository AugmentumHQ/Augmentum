"""Decoupling probe: does an explicit (emotion) tag vary PROSODY while a cloned
IDENTITY (audio anchor) stays constant, in expressive Mimi-family models?

This de-risks the "train emotion-by-tag into a CPU model, clone identity
separately" plan BEFORE building a trainer. We can't test Pocket's
embedding+tag composition directly (no emotion-trained Pocket exists), so we
test the closest available evidence in the same Mimi family:

  * base CSM-1B (sesame)  — retains Sesame's full emotional range, untouched
  * onecxi/csm-english-multi-speaker-v2 — Expresso-trained, claims more range

For each: hold a Ruby reference clip (identity) + target sentence CONSTANT,
vary only the leading (emotion) tag. If prosody changes audibly AND the voice
stays the same person, emotion and identity are separable in this family.
"whisper" is the litmus — unambiguous if the tag actually conditions delivery.

  .venv/Scripts/python.exe probe_decouple.py

Outputs samples/probe/*.wav + samples/probe.html for A/B listening.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torchaudio
from safetensors.torch import load_file
from transformers import AutoProcessor, CsmForConditionalGeneration

BASE = "sesame/csm-1b"
SR = 24000
TARGET = "I just heard the news a moment ago, and honestly, I don't know what to say about it."
# neutral baseline + a spread + whisper litmus (unambiguous if conditioning works)
EMOTIONS = ["neutral", "happy", "sad", "angry", "whisper"]

ONECXI = Path.home() / (
    ".cache/huggingface/hub/models--onecxi--csm-english-multi-speaker-v2"
    "/snapshots/9bcd64f8d1278e2d926d7ed5fb839329391c85fa/model.safetensors"
)


def load_anchor():
    """Pick a clean, neutral-ish 2.3-4.6s Ruby clip as the identity anchor."""
    rows = [json.loads(l) for l in open("data/ruby/train.jsonl", encoding="utf-8") if l.strip()]
    pref = {"neutral", "regular", "calm", "serenity", "contentment", "interest"}
    cand = [r for r in rows if 2.3 <= float(r.get("duration", 99)) <= 4.6]
    cand.sort(key=lambda r: (r.get("emotion") not in pref, -len(r["raw_text"])))
    r = cand[0]
    wav, sr = torchaudio.load(str(Path("data/ruby") / r["audio"]))
    wav = wav.mean(0)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.numpy(), r["raw_text"], r


def generate(model, processor, anchor_wav, anchor_text, text, *, anchored=True):
    if anchored:
        conv = [
            {"role": "0", "content": [
                {"type": "text", "text": anchor_text},
                {"type": "audio", "audio": anchor_wav}]},
            {"role": "0", "content": [{"type": "text", "text": text}]},
        ]
    else:
        conv = [{"role": "0", "content": [{"type": "text", "text": text}]}]
    inputs = processor.apply_chat_template(
        conv, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        audio = model.generate(**inputs, output_audio=True)
    wav = audio[0].to(torch.float32).cpu()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return wav


def run_model(name, weights_path, anchor):
    aw, at, ar = anchor
    print(f"\n=== {name} ===", flush=True)
    processor = AutoProcessor.from_pretrained(BASE)
    model = CsmForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.float32, device_map="cuda")
    if weights_path:
        sd = load_file(str(weights_path))
        res = model.load_state_dict(sd, strict=False)
        miss, unexp = list(res.missing_keys), list(res.unexpected_keys)
        print(f"[{name}] weights: matched={len(sd)-len(unexp)} missing={len(miss)} unexpected={len(unexp)}")
        if unexp:
            print(f"[{name}]   sample unexpected: {unexp[:3]}")
        if miss:
            print(f"[{name}]   sample missing:    {miss[:3]}")
        if len(unexp) > 0.5 * len(sd):
            print(f"[{name}] !! >50% keys unmatched — wrong format, skipping (needs Sesame->HF remap)")
            del model
            torch.cuda.empty_cache()
            return
    model.eval()

    out = Path("samples/probe")
    out.mkdir(parents=True, exist_ok=True)
    # one plain (no-anchor) neutral as an identity reference point
    for emo in EMOTIONS:
        text = TARGET if emo == "neutral" else f"({emo}) {TARGET}"
        try:
            wav = generate(model, processor, aw, at, text, anchored=True)
            fn = out / f"{name}__{emo}.wav"
            torchaudio.save(str(fn), wav, SR)
            print(f"[{name}] {emo:8s} {wav.shape[-1]/SR:5.2f}s -> {fn.name}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] {emo:8s} FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
    del model
    torch.cuda.empty_cache()


def build_html(anchor_meta):
    out = Path("samples/probe")
    wavs = sorted(out.glob("*.wav"))
    by_model: dict[str, list[Path]] = {}
    for w in wavs:
        m = w.stem.split("__")[0]
        by_model.setdefault(m, []).append(w)
    cards = []
    for m, files in by_model.items():
        rows = "".join(
            f'<div class=clip><span>{f.stem.split("__")[1]}</span>'
            f'<audio controls preload=none src="probe/{f.name}"></audio></div>'
            for f in files)
        cards.append(f'<div class=card><b>{m}</b>{rows}</div>')
    html = f"""<!doctype html><meta charset=utf-8><title>Emotion-tag decoupling probe</title>
<style>body{{font:15px/1.5 system-ui;margin:24px;background:#14151a;color:#e8e8ea}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}}
.card{{background:#1d1f26;border:1px solid #2a2d36;border-radius:12px;padding:14px}}
.card b{{font-size:18px}} .clip{{margin:8px 0}} .clip span{{display:block;font-size:11px;color:#9aa}}
audio{{width:100%;height:34px}} p.sub{{color:#9aa}}</style>
<h1>Emotion-tag decoupling probe</h1>
<p class=sub>Same voice anchor (Ruby clip: {anchor_meta}), same sentence — only the
(emotion) tag changes. <b>Listen for two things:</b> (1) does the delivery actually
change per tag? (2) does it stay the SAME voice? <b>whisper</b> is the litmus.</p>
<div class=grid>{''.join(cards)}</div>"""
    p = Path("samples/probe.html")
    p.write_text(html, encoding="utf-8")
    print(f"\n[done] open: {p.resolve()}")


def main():
    anchor = load_anchor()
    _, at, ar = anchor
    print(f"anchor: emo={ar.get('emotion')!r} dur={ar.get('duration')}s text={at[:50]!r}")
    run_model("base-csm", None, anchor)
    if ONECXI.exists():
        run_model("onecxi", ONECXI, anchor)
    else:
        print(f"[onecxi] weights not found at {ONECXI} — skipping")
    build_html(f"{ar.get('emotion')}, {ar.get('duration')}s")


if __name__ == "__main__":
    main()
