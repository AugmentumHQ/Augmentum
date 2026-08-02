# CSM Voice Training Harness

Reusable LoRA fine-tuning harness for Sesame **CSM-1B**. It trains one
**consistent, emotion-taggable voice** and produces a small **LoRA adapter**
you load on top of `sesame/csm-1b`.

Why this beats context-cloning: a fine-tuned voice has a *baked-in, stable
identity*, so it sidesteps the self-conditioning degradation entirely — the
model simply *is* the voice. The emotion you trained becomes a text knob you
drive at inference (`(happy) ...`).

Built to be re-pointed: train on EARS now, swap `config.yaml` to a voice
actor's studio recordings later (see `recording_brief.md`) with no code change.

## Honesty / status
- Written against the documented `transformers` CSM training API
  (`CsmForConditionalGeneration` + `CsmProcessor`) + standard PEFT/Trainer.
- **Not run end-to-end in the authoring env** (no GPU/dataset there). The data
  pipeline (`prepare_data.py`, `sources.py`) is the solid, version-independent
  part. The single spot most likely to need a first-run tweak is
  `train.py::Collator` (and the `generate.py` call), if your `transformers`
  version's CSM processor signature differs — both are flagged in-code. Run it,
  paste me the first error, and we iterate.
- **Hardware target: RTX 3090** (Ampere, 24 GB, native bf16 — no fp16 hacks).
  LoRA uses ~6–12 GB; lots of headroom (raise `per_device_batch_size`).

## License boundary — read once
The default source, **EARS**, is **CC BY-NC 4.0** and is real human voices.
Fine for **personal, private** use (your own Becca). Do **not** ship the
EARS-trained adapter in the public repo or build anything commercial on it, and
note that publicly-recognizable voice cloning raises personality rights the CC
license doesn't cover. For a *distributable* voice, use the `csv` source with
self-recorded / permissively-licensed audio.

## Setup
```bash
cd services/sesame-csm/training
python -m venv .venv && . .venv/Scripts/activate   # Windows; or bin/activate on Linux
pip install -r requirements.txt                    # install the CUDA torch build for your driver
huggingface-cli login                              # gated: accept csm-1b + Llama-3.2-1B licenses
```

## The flow
```
1. get EARS        ->  extract; note speaker dirs (pick a female speaker)
2. config.yaml     ->  set ears_root, ears_speaker, emotion.keep
3. prepare_data.py ->  data/becca/train.jsonl + audio/   (prints per-emotion counts)
4. train.py        ->  out/becca-lora  (~tens of min – 2h on a 3090)
5. generate.py     ->  samples/*.wav   (listen across emotions; tweak; retrain)
6. deploy          ->  load the adapter in the sidecar
```

### 3 — prepare
```bash
python prepare_data.py --config config.yaml
```
Sanity-check the printed per-emotion counts. Tune `config.yaml`
`emotion.keep` / `remap` to align labels, and `segment.max_seconds` if clips
are long.

### 4 — train
```bash
python train.py --config config.yaml
```

### 5 — listen
```bash
python generate.py --config config.yaml --adapter out/becca-lora \
    --prompts eval_prompts.yaml --out samples/
```
Align `eval_prompts.yaml` emotions to the labels you kept. Iterate.

### 6 — deploy to the sidecar
The adapter loads on top of `csm-1b`. Two options:
- **Merge** and point the sidecar at the merged weights:
  `PeftModel.from_pretrained(base, "out/becca-lora").merge_and_unload()` → save.
- **Adapter at load time**: add a `PeftModel.from_pretrained(...)` step after
  `load_csm_1b()` in `../app.py`, gated on a `CSM_ADAPTER_DIR` env. Say the word
  and I'll wire that into the sidecar (one small, env-gated change).

## Reuse for a voice actor (the foundation payoff)
1. Get studio audio per `recording_brief.md` (48 kHz/24-bit mono, per-emotion).
2. Make `metadata.csv`: columns `audio[,text][,emotion]`.
3. In `config.yaml`: `source.type: csv`, set `csv_path` + `audio_root`.
4. Re-run from step 3. Nothing else changes.

## Files
| File | Role |
|---|---|
| `config.yaml` | every knob — the one file you edit per voice |
| `sources.py` | pluggable data sources (`ears`, `csv`) — the reuse seam |
| `prepare_data.py` | raw → 24 kHz → segment → transcribe → emotion-tag → `train.jsonl` |
| `train.py` | LoRA fine-tune (bf16, 3090-sized) → adapter |
| `generate.py` | synth test lines per emotion for listening |
| `eval_prompts.yaml` | test sentences |
| `recording_brief.md` | what to have a voice actor record |
