"""Stage raw voice data into a CSM training set.

Pipeline (backend-agnostic; the JSONL it writes feeds train.py):

    source records  ->  24 kHz mono wav  ->  silence-split if long
        ->  per-clip transcript (faster-whisper unless provided)
        ->  emotion-tagged text  ->  train.jsonl + audio/

Run:  python prepare_data.py --config config.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torchaudio
import yaml

from sources import load_records


def load_cfg(p: str) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_mono_24k(path: str, sr_target: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)            # (C, N)
    wav = wav.mean(0, keepdim=True)            # -> mono
    if sr != sr_target:
        wav = torchaudio.functional.resample(wav, sr, sr_target)
    return wav.squeeze(0)                       # (N,)


def silence_split(wav: torch.Tensor, sr: int, max_s: float, min_s: float) -> list[torch.Tensor]:
    """Split a clip longer than ``max_s`` at low-energy gaps. A simple RMS gate
    — good enough to keep training clips context-friendly without a VAD dep."""
    max_n = int(max_s * sr)
    if wav.numel() <= max_n:
        return [wav]
    fl, hop = int(0.03 * sr), int(0.01 * sr)
    rms = wav.unfold(0, fl, hop).pow(2).mean(1).sqrt()
    thr = max(rms.mean().item() * 0.5, 1e-4)
    silent = rms < thr
    segs: list[torch.Tensor] = []
    start, n = 0, wav.numel()
    while start < n:
        end = min(start + max_n, n)
        if end < n:                              # walk back to nearest silent frame
            floor = start + int(min_s * sr)
            j = (end - fl // 2) // hop
            while j * hop > floor:
                if j < len(silent) and silent[j]:
                    end = j * hop
                    break
                j -= 1
        seg = wav[start:end]
        if seg.numel() >= int(min_s * sr):
            segs.append(seg)
        start = max(end, start + int(min_s * sr))
    return segs


_whisper = None


def transcribe(wav: torch.Tensor, cfg: dict) -> str:
    global _whisper
    from faster_whisper import WhisperModel
    if _whisper is None:
        dev = cfg["transcribe"].get("device", "cuda")
        ct = "float16" if dev == "cuda" else "int8"
        _whisper = WhisperModel(cfg["transcribe"]["model"], device=dev, compute_type=ct)
    audio = wav.detach().cpu().numpy().astype("float32")
    segs, _ = _whisper.transcribe(audio, language=cfg["transcribe"].get("language"))
    return " ".join(s.text.strip() for s in segs).strip()


def canon_emotion(e: str | None, cfg: dict) -> str | None:
    if e is None:
        return None
    e = cfg["emotion"].get("remap", {}).get(e, e)
    keep = cfg["emotion"].get("keep") or []
    if keep and e not in keep:
        return "__drop__"
    return e


def tag_text(text: str, emotion: str | None, cfg: dict) -> str:
    if cfg["emotion"]["enabled"] and emotion:
        return cfg["emotion"]["tag_format"].format(emotion=emotion) + text
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = load_cfg(ap.parse_args().config)
    sr = int(cfg["sample_rate"])
    out = Path(cfg["out_dir"])
    (out / "audio").mkdir(parents=True, exist_ok=True)

    seg_cfg = cfg.get("segment", {})
    manifest: list[dict] = []
    counts: dict[str, int] = {}

    for r in load_records(cfg):
        emo = canon_emotion(r.emotion, cfg)
        if emo == "__drop__":
            continue
        try:
            wav = to_mono_24k(r.audio_path, sr)
        except Exception as exc:  # noqa: BLE001 — skip an unreadable clip, keep going
            print(f"[skip] {r.audio_path}: {exc}")
            continue
        clips = (silence_split(wav, sr, seg_cfg.get("max_seconds", 12), seg_cfg.get("min_seconds", 1.0))
                 if seg_cfg.get("enabled", True) else [wav])
        for k, clip in enumerate(clips):
            # A provided transcript only aligns to an UNSPLIT clip; otherwise transcribe.
            text = r.text if (r.text and len(clips) == 1) else transcribe(clip, cfg)
            if not text:
                continue
            cid = hashlib.md5(f"{r.audio_path}:{k}".encode()).hexdigest()[:12]
            torchaudio.save(str(out / "audio" / f"{cid}.wav"), clip.unsqueeze(0), sr)
            manifest.append({
                "audio": f"audio/{cid}.wav",
                "text": tag_text(text, emo, cfg),
                "raw_text": text,
                "emotion": emo,
                "speaker": r.speaker_id,
                "duration": round(clip.numel() / sr, 2),
            })
            counts[emo or "none"] = counts.get(emo or "none", 0) + 1

    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(m) for m in manifest), encoding="utf-8")
    total_min = sum(m["duration"] for m in manifest) / 60
    print(f"[done] {len(manifest)} clips, {total_min:.1f} min -> {out / 'train.jsonl'}")
    print("[emotions]", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
    if total_min < 10:
        print("[warn] under ~10 min of audio — fine for a proof, thin for robust emotion coverage")


if __name__ == "__main__":
    main()
