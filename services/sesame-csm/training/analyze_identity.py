"""Did IDENTITY hold across the emotion tags? Rough timbre-proximity proxy.

For each generated clip, compute a time-averaged MFCC vector and its cosine
similarity to (a) the Ruby ANCHOR clip and (b) the model's own NEUTRAL clip.
If emotion clips stay timbre-close to the anchor/neutral while pitch swings
(see analyze_probe.py), emotion and identity are separable -> decoupling holds.

Caveat: MFCC cosine is a WEAK identity proxy (it also tracks content/pitch a
bit). Use it as corroboration for the ear, not proof.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF
from torchaudio.transforms import MFCC

SR = 24000
_mfcc = MFCC(sample_rate=SR, n_mfcc=20, melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 64})


def vec(x: torch.Tensor) -> torch.Tensor:
    m = _mfcc(x)              # [n_mfcc, T]
    m = m[1:]                 # drop c0 (overall energy) -> more timbre, less loudness
    return m.mean(1)


def load(p: str) -> torch.Tensor:
    x, sr = torchaudio.load(p)
    x = x.mean(0)
    if sr != SR:
        x = AF.resample(x, sr, SR)
    return x


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    # anchor = same clip probe_decouple.py picked
    rows = [json.loads(l) for l in open("data/ruby/train.jsonl", encoding="utf-8") if l.strip()]
    pref = {"neutral", "regular", "calm", "serenity", "contentment", "interest"}
    cand = [r for r in rows if 2.3 <= float(r.get("duration", 99)) <= 4.6]
    cand.sort(key=lambda r: (r.get("emotion") not in pref, -len(r["raw_text"])))
    anchor_v = vec(load(str(Path("data/ruby") / cand[0]["audio"])))

    out = Path("samples/probe")
    clips = {w.stem: vec(load(str(w))) for w in sorted(out.glob("*.wav"))}
    print(f"{'clip':28s} {'~simAnchor':>10s} {'~simNeutral':>11s}")
    for model in sorted({k.split('__')[0] for k in clips}):
        nv = clips.get(f"{model}__neutral")
        for emo in ["neutral", "happy", "sad", "angry", "whisper"]:
            k = f"{model}__{emo}"
            if k not in clips:
                continue
            sa = cos(clips[k], anchor_v)
            sn = cos(clips[k], nv) if nv is not None else float("nan")
            print(f"{k:28s} {sa:10.3f} {sn:11.3f}")
        print()
    print("Read: high & STABLE simAnchor across emotions => identity held while prosody moved.")
    print("      A big drop on one emotion => that tag likely pulled a different voice.")


if __name__ == "__main__":
    main()
