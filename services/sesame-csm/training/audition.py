"""Audition EARS female voices and pick one for Becca.

For each speaker id given, grab a few short comparison clips (streaming the
per-speaker zip, extracting only those clips, deleting the zip) and build a
self-contained samples/voices.html you open in a browser to A/B them.

  python audition.py p011 p012 p014 p089 p003 p024 p006 p002

The "neutral" clip is the SAME Rainbow Passage for every speaker, so it's a
true apples-to-apples voice comparison. Re-run with more ids any time.
"""
from __future__ import annotations

import io
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REL = "https://github.com/facebookresearch/ears_dataset/releases/download/dataset"
OUT = Path("samples/voices")
LOCAL = Path("data/ears")          # already-extracted speakers (e.g. p012)

# (filename, human label) — same across all EARS speakers
CLIPS = [
    ("rainbow_01_regular.wav", "Neutral — same passage"),
    ("emo_contentment_sentences.wav", "Warm / content"),
    ("emo_amusement_sentences.wav", "Cheerful"),
]
CLIP_FILES = [c[0] for c in CLIPS]


def have(spk: str) -> bool:
    return all((OUT / spk / c).exists() for c in CLIP_FILES)


def fetch(spk: str) -> None:
    dest = OUT / spk
    dest.mkdir(parents=True, exist_ok=True)
    if have(spk):
        print(f"[skip] {spk} (already have clips)")
        return
    # reuse a fully-extracted local speaker if present
    if all((LOCAL / spk / c).exists() for c in CLIP_FILES):
        for c in CLIP_FILES:
            shutil.copy(LOCAL / spk / c, dest / c)
        print(f"[copy] {spk} (from data/ears)")
        return
    print(f"[get ] {spk} — streaming zip ...", flush=True)
    tmp = dest / "_dl.zip"
    with urllib.request.urlopen(f"{REL}/{spk}.zip") as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    with zipfile.ZipFile(tmp) as z:
        by_base = {n.split("/")[-1]: n for n in z.namelist()}
        for c in CLIP_FILES:
            if c in by_base:
                with z.open(by_base[c]) as src, open(dest / c, "wb") as dst:
                    dst.write(src.read())
            else:
                print(f"  [warn] {spk} missing {c}")
    tmp.unlink(missing_ok=True)
    print(f"[ok  ] {spk}")


def build_html(speakers: list[str]) -> None:
    stats = {}
    sj = Path("data/speaker_statistics.json")
    if sj.exists():
        stats = json.loads(sj.read_text())

    def meta(spk: str) -> str:
        v = stats.get(spk, {})
        age = v.get("age", "?")
        eth = v.get("ethnicity", "")
        return f"age {age} · {eth}" if eth else f"age {age}"

    cards = []
    for spk in speakers:
        players = "".join(
            f'<div class="clip"><span>{label}</span>'
            f'<audio controls preload="none" src="voices/{spk}/{fn}"></audio></div>'
            for fn, label in CLIPS if (OUT / spk / fn).exists()
        )
        cards.append(
            f'<div class="card"><div class="hd"><b>{spk}</b>'
            f'<span class="m">{meta(spk)}</span></div>{players}</div>')

    html = f"""<!doctype html><meta charset=utf-8>
<title>Becca voice audition</title>
<style>
 body{{font:15px/1.5 system-ui,sans-serif;margin:24px;background:#14151a;color:#e8e8ea}}
 h1{{font-weight:600;margin:0 0 4px}} p.sub{{color:#9aa;margin:0 0 20px}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:14px}}
 .card{{background:#1d1f26;border:1px solid #2a2d36;border-radius:12px;padding:14px}}
 .hd{{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px}}
 .hd b{{font-size:20px}} .m{{color:#9aa;font-size:12px}}
 .clip{{margin:8px 0}} .clip span{{display:block;font-size:11px;color:#9aa;margin-bottom:3px}}
 audio{{width:100%;height:34px}}
 .tip{{margin-top:22px;color:#9aa;font-size:13px}}
</style>
<h1>Becca voice audition</h1>
<p class=sub>Start with the top clip in each card — it's the same passage for everyone,
so you're comparing the <i>voice</i>, not the words. Then check warmth/cheer below.</p>
<div class=grid>{''.join(cards)}</div>
<p class=tip>Pick the one you want and tell me its id (e.g. <b>{speakers[0]}</b>) — I'll
re-stage the full speaker and train Becca on it.</p>
"""
    (OUT.parent / "voices.html").write_text(html, encoding="utf-8")
    print(f"\n[done] open: {(OUT.parent / 'voices.html').resolve()}")


def main() -> None:
    speakers = sys.argv[1:] or ["p012"]
    OUT.mkdir(parents=True, exist_ok=True)
    for spk in speakers:
        try:
            fetch(spk)
        except Exception as exc:  # noqa: BLE001 — one bad speaker shouldn't stop the set
            print(f"[fail] {spk}: {exc}")
    build_html(speakers)


if __name__ == "__main__":
    main()
