# Voice recording brief (for a future custom voice)

When you commission a voice actor, hand them this. It produces audio that drops
straight into the `csv` source path of the harness — same pipeline, better
voice. The goal is **coverage of expressive range + clean transfer**, not volume.

## Format specs (give these verbatim)
- **48 kHz, 24-bit, mono WAV.** (We downsample to 24 kHz; capturing higher
  protects quality through processing.)
- Dry signal: **no reverb, compression, EQ, or de-noise** baked in. We want the
  raw voice; processing is our job.
- Quiet room / treated booth, consistent mic distance, consistent gain across
  sessions. Same mic + chain start to finish.
- One emotion/style per file (or per take), named so the label is recoverable,
  e.g. `happy_001.wav`, `whisper_012.wav`.

## What to record (the coverage that matters)
Per emotion/style below, ~3–5 minutes of varied sentences (not one carrier
phrase repeated). Mix declaratives, questions, exclamations, and a few longer
multi-clause lines so prosody has range to learn from.

- **neutral** (the base — record the most here, ~10 min)
- happy / warm
- sad / gentle
- calm / reassuring
- excited / energetic
- tired / soft
- whisper
- (optional) playful, serious, concerned

Plus **~5 min of free, improvised conversational speech** — them just talking
naturally, as if mid-conversation. This spontaneity is what read scripts can't
fake and what makes a companion voice feel alive (it's why studio *dialogue*
sets beat audiobook sets).

## Reference tone (optional but valuable)
Give the actor 2–3 short reference clips of the *target delivery* (the register
you want Becca to live in) so they match tone, not just words. Consistency of
character across emotions matters more than any single great take.

## Deliver
1. The WAVs, named by emotion as above.
2. A `metadata.csv` with columns: `audio,text,emotion`
   - `audio`: filename, `text`: the line (optional — we can transcribe),
     `emotion`: the label.
3. Point `config.yaml` at it: `source.type: csv`, `csv_path`, `audio_root`,
   then run `prepare_data.py` → `train.py` exactly as for EARS.

## Why this format
Each emotion becomes a `(tag) text` training pair, so the model learns the tag
as a controllable knob. Clean per-emotion coverage + consistent character is
the whole game — the harness handles the rest.
