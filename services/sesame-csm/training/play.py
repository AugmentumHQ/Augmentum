"""Build an HTML player for a generated samples dir.

  python play.py samples/ruby Ruby

Writes samples/<name>.html with one player per emotion + the prompt text.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

d = Path(sys.argv[1])
name = sys.argv[2] if len(sys.argv) > 2 else d.name
prompts = {}
ep = Path("eval_prompts.yaml")
if ep.exists():
    for i, p in enumerate(yaml.safe_load(ep.read_text())["prompts"]):
        prompts[f"{i:02d}_{p.get('emotion', 'neutral')}"] = p.get("text", "")

cards = ""
for w in sorted(d.glob("*.wav")):
    emo = w.stem.split("_", 1)[1] if "_" in w.stem else w.stem
    txt = prompts.get(w.stem, "")
    cards += (f'<div class=card><div class=hd><b>{emo}</b></div>'
              f'<audio controls preload=none src="{d.name}/{w.name}"></audio>'
              f'<p>{txt}</p></div>')

html = f"""<!doctype html><meta charset=utf-8><title>{name} voice</title>
<style>
 body{{font:15px/1.5 system-ui,sans-serif;margin:24px;background:#14151a;color:#e8e8ea}}
 h1{{font-weight:600}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
 .card{{background:#1d1f26;border:1px solid #2a2d36;border-radius:10px;padding:12px}}
 .hd b{{font-size:18px;text-transform:capitalize}} audio{{width:100%;margin:6px 0;height:34px}}
 p{{color:#9aa;font-size:13px;margin:4px 0 0}}
</style>
<h1>{name} — first training ({len(list(d.glob('*.wav')))} emotions)</h1>
<div class=grid>{cards}</div>
"""
out = d.parent / f"{name.lower()}.html"
out.write_text(html, encoding="utf-8")
print("open:", out.resolve())
