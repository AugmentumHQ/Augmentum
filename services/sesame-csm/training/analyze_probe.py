"""Objective read on the decoupling probe WAVs, so we judge by numbers not just ear.

Per clip:  dur, loudness(RMS dBFS), median pitch, pitch STD (monotone detector),
spectral centroid (brightness/breathiness).

Litmus checks across the set:
  * whisper should be QUIETER (lower RMS) + brighter/breathier than neutral
  * expressive model => higher pitch-STD spread across emotions (less monotone)
  * cap-hit (~10.0s) flagged as likely runaway
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as AF

SR = 24000


def rms_db(x: torch.Tensor) -> float:
    r = torch.sqrt(torch.mean(x**2) + 1e-12).item()
    return 20 * math.log10(max(r, 1e-9))


def pitch_stats(x: torch.Tensor) -> tuple[float, float, float]:
    """median Hz, std Hz over plausibly-voiced frames, voiced-fraction."""
    try:
        f0 = AF.detect_pitch_frequency(x.unsqueeze(0), SR).squeeze()
    except Exception:
        return float("nan"), float("nan"), 0.0
    v = f0[(f0 > 70) & (f0 < 450)]
    if v.numel() < 3:
        return float("nan"), float("nan"), float(v.numel()) / max(f0.numel(), 1)
    return v.median().item(), v.std().item(), v.numel() / f0.numel()


def centroid(x: torch.Tensor) -> float:
    spec = torch.stft(x, n_fft=1024, hop_length=256, return_complex=True).abs()
    freqs = torch.linspace(0, SR / 2, spec.shape[0]).unsqueeze(1)
    num = (freqs * spec).sum(0)
    den = spec.sum(0) + 1e-9
    return (num / den).mean().item()


def main():
    out = Path("samples/probe")
    wavs = sorted(out.glob("*.wav"))
    print(f"{'clip':28s} {'dur':>6s} {'RMSdB':>7s} {'pitch':>7s} {'pStd':>6s} {'voiced':>6s} {'centr':>7s}  flags")
    rows = {}
    for w in wavs:
        x, sr = torchaudio.load(str(w))
        x = x.mean(0)
        if sr != SR:
            x = AF.resample(x, sr, SR)
        dur = x.numel() / SR
        r = rms_db(x)
        pm, ps, vf = pitch_stats(x)
        c = centroid(x)
        flags = "CAP-HIT?" if dur >= 9.95 else ""
        rows[w.stem] = dict(dur=dur, rms=r, pm=pm, ps=ps, vf=vf, c=c)
        print(f"{w.stem:28s} {dur:6.2f} {r:7.1f} {pm:7.1f} {ps:6.1f} {vf:6.2f} {c:7.0f}  {flags}")

    # Per-model litmus summary
    for model in sorted({k.split('__')[0] for k in rows}):
        sub = {k.split('__')[1]: v for k, v in rows.items() if k.startswith(model + "__")}
        if "neutral" not in sub:
            continue
        n = sub["neutral"]
        print(f"\n--- {model} ---")
        if "whisper" in sub:
            wq = sub["whisper"]["rms"] - n["rms"]
            wc = sub["whisper"]["c"] - n["c"]
            verdict = "WORKS (quieter)" if wq < -3 else ("weak" if wq < -0.5 else "NO EFFECT")
            print(f"  whisper litmus: {wq:+.1f} dB vs neutral, centroid {wc:+.0f} Hz  -> {verdict}")
        pss = [v["ps"] for v in sub.values() if not math.isnan(v["ps"])]
        pms = [v["pm"] for v in sub.values() if not math.isnan(v["pm"])]
        if pms:
            print(f"  pitch spread across emotions: median range {max(pms)-min(pms):5.1f} Hz "
                  f"| mean within-clip pStd {sum(pss)/len(pss):4.1f} Hz  (higher = more expressive)")
        caps = [e for e, v in sub.items() if v["dur"] >= 9.95]
        if caps:
            print(f"  !! cap-hit (likely runaway): {caps}")


if __name__ == "__main__":
    main()
