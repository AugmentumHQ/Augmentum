#!/usr/bin/env python
"""Build a language-learning ``.augpack`` from open-source corpora.

Offline packager — run on a dev machine, ship the resulting ``.augpack``
as a static download. Nothing here touches the running server; the pack
is content-addressed-immutable and learner state lives separately in the
``vocab_state`` table.

Example (Japanese):

    python scripts/build_lang_pack.py ja \\
        --jmdict       data/JMdict_e \\
        --tatoeba-sentences data/jpn_sentences.tsv \\
        --tatoeba-links     data/jpn_eng_links.tsv \\
        --freq         data/jp_freq.tsv \\
        --jlpt         data/jp_jlpt.tsv \\
        --name "Japanese (JMdict + Tatoeba)" \\
        --out          data/knowledge_packs/ja.augpack

Source data:
  * JMdict (English edition): http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz
    (gunzip it first; pass the XML path)
  * Tatoeba dumps: https://tatoeba.org/downloads
    ``sentences.csv`` -> id<TAB>lang<TAB>text ; ``links.csv`` -> id<TAB>id
  * ``--freq`` / ``--jlpt``: optional ``key<TAB>int`` TSVs (surface or
    reading -> rank / JLPT level).

Use ``--limit-vocab`` / ``--limit-sentences`` to build a small pack for
testing the runtime lookup path without parsing the full corpus.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a plain script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augmentum.knowledge.lang_pack_builder import build_pack  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a language-learning .augpack")
    p.add_argument("lang_code", help="ISO language code, e.g. 'ja'")
    p.add_argument("--jmdict", required=True, type=Path, help="JMdict XML file")
    p.add_argument("--tatoeba-sentences", type=Path, help="Tatoeba sentences.csv (id<TAB>lang<TAB>text)")
    p.add_argument("--tatoeba-links", type=Path, help="Tatoeba links.csv (id<TAB>translation_id)")
    p.add_argument("--freq", type=Path, help="Optional frequency TSV (surface<TAB>rank)")
    p.add_argument("--jlpt", type=Path, help="Optional JLPT TSV (surface<TAB>level)")
    p.add_argument("--name", default="", help="Human-readable pack name")
    p.add_argument("--out", required=True, type=Path, help="Output .augpack path")
    p.add_argument("--limit-vocab", type=int, default=None, help="Cap vocab entries (testing)")
    p.add_argument("--limit-sentences", type=int, default=None, help="Cap sentences (testing)")
    args = p.parse_args(argv)

    if not args.jmdict.exists():
        print(f"error: JMdict file not found: {args.jmdict}", file=sys.stderr)
        return 2

    t0 = time.time()
    summary = build_pack(
        out_path=args.out,
        lang_code=args.lang_code,
        jmdict_xml=args.jmdict,
        tatoeba_sentences=args.tatoeba_sentences,
        tatoeba_links=args.tatoeba_links,
        freq_tsv=args.freq,
        jlpt_tsv=args.jlpt,
        name=args.name,
        limit_vocab=args.limit_vocab,
        limit_sentences=args.limit_sentences,
    )
    size_mb = args.out.stat().st_size / (1024 * 1024)
    print(
        f"built {args.out} — {summary['vocab']} vocab, "
        f"{summary['sentences']} sentences, {size_mb:.1f} MB, "
        f"{time.time() - t0:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
