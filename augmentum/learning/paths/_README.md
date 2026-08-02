# Learning paths

Hand-curated, pedagogically sequenced units per supported language. These
paths give learning mode a level spine that raw frequency lists cannot provide:
frequency still helps discovery and backfill, while path units decide the
professional beginner progression.

## File layout

One JSON per language: `{lang}.json`. Loaded at runtime by
`augmentum.learning.paths.load_path(lang_code)`.

Currently shipped:

| File | Level system | Units | Vocab | Phrases | Aux content |
|------|--------------|-------|-------|---------|-------------|
| `es.json` | CEFR A1/A2 | 22 | 496 | 90 | assessment, grammar |
| `fr.json` | CEFR A1 | 12 | 296 | 51 | assessment, grammar |
| `ja.json` | JLPT N5/N4 | 22 | 435 | 91 | assessment, grammar, kanji |
| `ko.json` | TOPIK 1 | 10 | 210 | 41 | assessment, grammar |
| `zh.json` | HSK 1/2 | 20 | 376 | 84 | assessment, characters, tones |

Planned next work: deeper levels, richer grammar practice, and better
morphology for languages where whitespace is only a coarse segmentation
signal.

## Schema

```json
{
  "lang": "<ISO-639-1>",
  "version": "1.0.0",
  "level_system": "cefr" | "hsk" | "jlpt" | "topik",
  "name": "<display title>",
  "credits": "<curation provenance>",
  "license": "CC BY 4.0",
  "levels": [
    {
      "code": "A1" | "HSK1" | "N5" | "TOPIK1" | "...",
      "name": "<level display>",
      "can_do": "<one-line can-do statement>",
      "units": [
        {
          "id": "<lang>-<level>-<NN>-<slug>",
          "title": "<unit title>",
          "theme": "<short tag>",
          "goal": "<one sentence: what the learner can do after this unit>",
          "vocab": [
            {"surface": "<word>", "pos": "<pos>", "gloss": "<en gloss>"}
          ],
          "phrases": [
            {"target": "<sentence in target lang>", "en": "<english>"}
          ],
          "grammar_note": "<optional one-line grammar hint>",
          "estimated_minutes": 12
        }
      ]
    }
  ]
}
```

## Loading and use

* `augmentum.learning.paths` exposes `available_langs()`, `load_path(lang)`,
  `path_summary(lang)`, `available_aux(lang)`, and
  `unit_vocab_surfaces(lang, unit_id)`.
* `learning_routes.py` exposes `GET /api/learning/paths/{lang}/manifest`,
  `GET /api/learning/paths/{lang}/unit/{unit_id}`, and aux endpoints.
* `POST /api/learning/packs/{lang}/seed` walks the curated path before falling
  back to frequency, so users start with useful unit vocabulary rather than
  raw high-frequency fragments.
* `GET /api/learning/games/readiness` surfaces path and aux metadata alongside
  per-game gates so the hub can keep curriculum, SRS, and games aligned.

## QA audit

Run the learning-content audit before changing paths, pack builders, or game
readiness rules:

```bash
python scripts/audit_learning_content.py --pretty --output out/learning-content-audit.json
```

Useful variants:

* `--no-packs --no-examples` checks catalog/path distribution only.
* `--lang ja --lang es` narrows the report to specific languages.
* `--pack-dir <dir>` audits a specific installed-pack directory.

The report covers unit/level distribution, duplicate and too-short surfaces,
phrase translation gaps, installed-pack lookup coverage, example-sentence
coverage, sentence counts, level metadata distribution, and game material gates.

## Authoring guidance

* A unit is 15-25 vocab + 4-8 phrases + 1 grammar note.
* Estimated minutes is a session size guide; aim for 8-15 minutes.
* Cover the unit's goal with the vocab; do not pad.
* Phrases should use only vocab from this unit or earlier units in the same
  level. Recycling is a feature.
* Grammar notes are hints, not lessons. Use one sentence and let Companion mode
  handle deeper explanations.
