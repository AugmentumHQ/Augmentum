"""LoRA fine-tune CSM-1B on a prepared voice set. Sized for an RTX 3090 (bf16).

Run:  python train.py --config config.yaml
Output: a PEFT LoRA adapter at train.output_dir.

VERIFY ON FIRST RUN: CSM training lives in recent ``transformers``
(``CsmForConditionalGeneration`` + ``CsmProcessor``). The data collator below
follows the documented CSM training shape — building a one-turn conversation
per clip and letting the processor emit input_ids + audio-code labels. If your
transformers version's processor signature differs, the ONE spot to adjust is
``Collator.__call__`` (see the sesame/csm-1b model card → "Training"). Pin a
known-good version in requirements.txt and bump deliberately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio
import yaml
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    CsmForConditionalGeneration,
    Trainer,
    TrainingArguments,
)


def load_cfg(p: str) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


class VoiceSet(Dataset):
    """Reads train.jsonl. Supports two row shapes:
      - single-turn: {audio, text, speaker}  (audio relative to root)
      - dialogue:    {turns: [{audio, text, speaker}, ...]}  (audio CWD-relative)
    Returns {"turns": [{text, audio(np), speaker}, ...]} either way."""

    def __init__(self, root: str, sr: int):
        self.root = Path(root)
        self.sr = sr
        lines = (self.root / "train.jsonl").read_text(encoding="utf-8").splitlines()
        self.items = [json.loads(ln) for ln in lines if ln.strip()]

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, path: str):
        wav, sr = torchaudio.load(path)
        wav = wav.mean(0)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav.numpy()

    def __getitem__(self, i: int) -> dict:
        m = self.items[i]
        if "turns" in m:                                   # dialogue: CWD-relative paths
            turns = [{"text": t["text"], "speaker": int(t.get("speaker", 0)),
                      "audio": self._load(t["audio"])} for t in m["turns"]]
        else:                                              # single-turn: root-relative
            turns = [{"text": m["text"], "speaker": int(m.get("speaker", 0)),
                      "audio": self._load(str(self.root / m["audio"]))}]
        return {"turns": turns}


class Collator:
    def __init__(self, processor, sr: int):
        self.p = processor
        self.sr = sr

    def __call__(self, batch: list[dict]) -> dict:
        conversations = [[
            {"role": str(t["speaker"]),
             "content": [{"type": "text", "text": t["text"]},
                         {"type": "audio", "audio": t["audio"]}]}
            for t in b["turns"]
        ] for b in batch]
        return self.p.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            output_labels=True,        # train target = the audio codebook tokens
            padding=True,
            return_tensors="pt",
            sampling_rate=self.sr,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    cfg = load_cfg(ap.parse_args().config)
    tc, sr = cfg["train"], int(cfg["sample_rate"])

    processor = AutoProcessor.from_pretrained(cfg["base_model"])
    # CSM's audio-embedding path stays fp32 while the backbone casts, so a
    # bf16 load collides at _merge_input_ids_with_input_values. fp32 keeps the
    # whole model one dtype — uniform and correct. Toggle via train.bf16.
    load_dtype = torch.bfloat16 if tc.get("bf16", False) else torch.float32
    model = CsmForConditionalGeneration.from_pretrained(
        cfg["base_model"], torch_dtype=load_dtype, device_map="cuda")

    model = get_peft_model(model, LoraConfig(
        r=tc["lora_r"], lora_alpha=tc["lora_alpha"], lora_dropout=tc["lora_dropout"],
        bias="none", target_modules=tc["target_modules"], task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    if tc.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    args = TrainingArguments(
        output_dir=tc["output_dir"],
        per_device_train_batch_size=tc["per_device_batch_size"],
        gradient_accumulation_steps=tc["grad_accum"],
        learning_rate=float(tc["learning_rate"]),
        num_train_epochs=tc["num_epochs"],
        bf16=bool(tc.get("bf16", True)),
        warmup_ratio=tc.get("warmup_ratio", 0.03),
        logging_steps=tc.get("logging_steps", 10),
        save_steps=tc.get("save_steps", 200),
        save_total_limit=3,
        seed=tc.get("seed", 3407),
        gradient_checkpointing=bool(tc.get("gradient_checkpointing", True)),
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    Trainer(model=model, args=args,
            train_dataset=VoiceSet(cfg["out_dir"], sr),
            data_collator=Collator(processor, sr)).train()

    model.save_pretrained(tc["output_dir"])
    processor.save_pretrained(tc["output_dir"])
    print(f"[done] LoRA adapter -> {tc['output_dir']}")


if __name__ == "__main__":
    main()
