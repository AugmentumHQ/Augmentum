"""Synthesize test lines from a trained adapter, across emotions, for listening.

Run:  python generate.py --config config.yaml --adapter out/becca-lora \
          --prompts eval_prompts.yaml --out samples/

Listen to samples/, then adjust data / epochs and retrain. This is your
fast feedback loop — on a 3090 a full prepare→train→listen cycle is short.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio
import yaml
from peft import PeftModel
from transformers import AutoProcessor, CsmForConditionalGeneration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--adapter", default=None, help="LoRA dir; omit to hear the base model")
    ap.add_argument("--prompts", default="eval_prompts.yaml")
    ap.add_argument("--out", default="samples")
    a = ap.parse_args()

    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sr = int(cfg["sample_rate"])

    processor = AutoProcessor.from_pretrained(cfg["base_model"])
    # fp32 — matches training and avoids CSM's audio-embed dtype clash.
    model = CsmForConditionalGeneration.from_pretrained(
        cfg["base_model"], torch_dtype=torch.float32, device_map="cuda")
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    spk = int(cfg.get("speaker_id", 0))
    fmt = cfg["emotion"]["tag_format"]
    emo_on = cfg["emotion"]["enabled"]
    with open(a.prompts, encoding="utf-8") as f:
        prompts = yaml.safe_load(f)["prompts"]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, pr in enumerate(prompts):
        tag = fmt.format(emotion=pr["emotion"]) if (emo_on and pr.get("emotion")) else ""
        text = tag + pr["text"]
        conv = [{"role": str(spk), "content": [{"type": "text", "text": text}]}]
        inputs = processor.apply_chat_template(
            conv, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            audio = model.generate(**inputs, output_audio=True)
        wav = audio[0].to(torch.float32).cpu()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        fn = out / f"{i:02d}_{pr.get('emotion', 'neutral')}.wav"
        torchaudio.save(str(fn), wav, sr)
        print(f"[ok] {fn}  «{text[:64]}»")


if __name__ == "__main__":
    main()
