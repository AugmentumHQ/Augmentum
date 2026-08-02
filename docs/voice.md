# Voice & wake word

Augmentum has a full voice pipeline, not just speech-to-text and text-to-speech.
You can talk to it hands-free, it speaks back naturally, and you can train it to
wake to a word of your choosing.

## The pipeline

**Listening (speech → text):**

- **Streaming STT** — transcribes as you speak.
- **Voice-activity detection + smart-turn** — knows when you've actually finished
  a thought, so it doesn't cut you off or wait awkwardly.
- **Denoising** — cleans the input before transcription.
- **Pronunciation lexicon** — teach it how to say (and hear) names and terms it
  otherwise mangles.

**Speaking (text → speech):**

- **Multiple TTS engines** (e.g. Kokoro, Pocket-TTS) — pick the voice and engine
  you like.
- **Emotion + prosody** — output isn't flat; it carries expression.
- **Phoneme lip-sync** — drives the 3D avatar's mouth in time with the audio.

Voice output is server-rendered, so on the phone client it can play through the
system TTS engine — meaning other apps can speak in your chosen voice too.

## Setting it up

Voice input and output are chosen during `setup.sh` (STT and TTS steps) and can
be changed later in the app's voice settings. On the **CPU** tier you get a
lightweight STT/TTS; on **GPU** you can run higher-quality engines. Pick a voice
and speed in settings — note these are intentionally **per-surface** (your phone
and the web app can use different voices).

## Training your own wake word

Instead of a fixed "hey assistant," you can train Augmentum to wake to a phrase
of your choice from just a handful of samples:

1. Open the **wake-word** settings.
2. **Record a few personal samples** of your phrase (the more varied — different
   rooms, distances — the better).
3. Augmentum trains a small personal wake-word **model** against a corpus of
   negatives (so it fires on *your* word, not everything).
4. Once trained, the model is selectable and the assistant listens for it
   hands-free.

You can manage multiple corpora and models, and retrain as needed.

## Tuning

- If it cuts you off or lingers, adjust the **turn-detection / VAD** sensitivity.
- If a name is mispronounced or misheard, add it to the **lexicon**.
- If the wake word misfires or misses, add more **personal samples** (especially
  from the conditions where it struggles) and retrain.
- Denoising helps in noisy rooms but adds a little latency — toggle it based on
  your environment.

## Notes

- Everything runs locally by default — audio doesn't leave your machine unless
  you deliberately configure a cloud STT/TTS provider.
- Wake-word models are personal and stay on your instance.
