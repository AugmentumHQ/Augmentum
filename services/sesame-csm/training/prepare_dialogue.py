"""Build emotion-paired DIALOGUE training data for a dialogue-aware voice.

Pairs the target voice's clips (speaker 0 = her) with another speaker's
emotion-MATCHED clips (speaker 1 = the "other person") as conversational
context. This relearns cross-speaker conditioning — the fix for the
single-speaker runaway/forgetting — AND seeds a real prosodic correlation
(her reply's tone tracks the prior turn's emotion). Words don't match by
design: CSM conditions on prosody, not meaning. A fraction stay single-turn
(no context) so plain TTS stays sharp.

  python prepare_dialogue.py --config config_ruby_v2.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


def load_cfg(p: str) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_manifest(d: str) -> list[dict]:
    return [json.loads(ln) for ln in (Path(d) / "train.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_ruby_v2.yaml")
    cfg = load_cfg(ap.parse_args().config)
    dlg = cfg["dialogue"]
    target_dir = dlg["target_data"]               # her prepared clips (speaker 0)
    other_dirs = list(dlg["other_data"])          # context voices (speaker 1)
    frac = float(dlg.get("dialogue_fraction", 0.6))
    rng = random.Random(int(cfg["train"].get("seed", 3407)))

    # Bound sequence length so a two-turn example fits VRAM (~v1's single-turn
    # budget): short context turn + moderate reply. EARS clips are mostly ~12s,
    # so pairing two unbounded clips OOMs.
    max_ctx = float(dlg.get("max_context_s", 4.0))
    max_reply = float(dlg.get("max_reply_s", 8.0))

    target = load_manifest(target_dir)
    # context pool = SHORT other-speaker clips only (paths CWD-relative for train.py)
    by_emotion: dict[str, list[tuple[str, dict]]] = {}
    others_short: list[tuple[str, dict]] = []
    for od in other_dirs:
        for r in load_manifest(od):
            if float(r.get("duration", 99)) > max_ctx:
                continue
            by_emotion.setdefault(r.get("emotion") or "neutral", []).append((od, r))
            others_short.append((od, r))

    def turn(data_dir: str, r: dict, speaker: int) -> dict:
        return {"speaker": speaker, "audio": f"{data_dir}/{r['audio']}", "text": r["text"]}

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    lines: list[dict] = []
    for r in target:
        her = turn(target_dir, r, 0)
        emo = r.get("emotion") or "neutral"
        # only short-enough replies get a context turn; long ones stay single.
        if float(r.get("duration", 99)) <= max_reply and rng.random() < frac and others_short:
            pool = by_emotion.get(emo) or by_emotion.get("neutral") or others_short
            od, ctx = rng.choice(pool)
            lines.append({"turns": [turn(od, ctx, 1), her]})
        else:
            lines.append({"turns": [her]})  # plain single-turn (no context)

    (out / "train.jsonl").write_text("\n".join(json.dumps(ln) for ln in lines), encoding="utf-8")
    n_dlg = sum(1 for ln in lines if len(ln["turns"]) > 1)
    print(f"[dialogue] {len(lines)} examples -> {out / 'train.jsonl'}")
    print(f"[dialogue]   {n_dlg} two-turn (emotion-paired) | {len(lines) - n_dlg} single-turn (plain TTS)")


if __name__ == "__main__":
    main()
