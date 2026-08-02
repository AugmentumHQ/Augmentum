"""CSM Voice Studio — interactive generation with best-of-N rejection sampling.

Type a custom script, pick a trained voice + emotion, and the server generates
N candidates, transcribes each (faster-whisper) to score text fidelity, applies
silence/duration heuristics, and ranks them — so the AR "lottery" failures get
filtered instead of shipped. Reference-anchor, temperature and seed controls
included.

  .venv/Scripts/python studio.py        # then open http://localhost:7860

Voices = the LoRA adapters under out/<name>-lora (ruby, tess, ...). Train more
and they auto-appear in the picker.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import torch
import torchaudio
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoProcessor, CsmForConditionalGeneration

BASE = "sesame/csm-1b"
OUT = Path("out")
DATA = Path("data")
SR = 24000
SAMPLES = Path("samples/studio")
SAMPLES.mkdir(parents=True, exist_ok=True)

# EARS label set (emotions + reading styles) — the tags the voices were trained on.
EMOTIONS = ["neutral", "contentment", "amusement", "serenity", "adoration", "amazement",
            "anger", "sadness", "fear", "disgust", "distress", "confusion", "desire",
            "pride", "relief", "interest", "realization", "disappointment", "guilt",
            "embarassment", "cuteness", "extasy", "pain", "whisper", "loud", "fast", "slow"]

app = FastAPI(title="CSM Voice Studio")

_proc = None
_model = None
_is_peft = False
_loaded_adapters: set[str] = set()
_anchor_cache: dict[str, tuple] = {}
_whisper = None


def list_voices() -> list[str]:
    return sorted(d.name[:-5] for d in OUT.glob("*-lora") if (d / "adapter_config.json").exists())


def _load_base() -> None:
    global _proc, _model
    if _model is not None:
        return
    print("[studio] loading CSM-1B (fp32) ...", flush=True)
    _proc = AutoProcessor.from_pretrained(BASE)
    _model = CsmForConditionalGeneration.from_pretrained(
        BASE, torch_dtype=torch.float32, device_map="cuda")
    _model.eval()
    print("[studio] base model ready", flush=True)


def _use_voice(voice: str) -> None:
    """Activate a voice's LoRA adapter (loading it once, then switching)."""
    global _model, _is_peft
    _load_base()
    if not voice:
        return
    adir = OUT / f"{voice}-lora"
    if voice not in _loaded_adapters:
        if not _is_peft:
            _model = PeftModel.from_pretrained(_model, str(adir), adapter_name=voice)
            _is_peft = True
        else:
            _model.load_adapter(str(adir), adapter_name=voice)
        _loaded_adapters.add(voice)
    if _is_peft:
        _model.set_adapter(voice)


def _anchor_for(voice: str):
    """A clean neutral clip from the voice's training data → (text, waveform)."""
    if voice in _anchor_cache:
        return _anchor_cache[voice]
    mani = DATA / voice / "train.jsonl"
    if not mani.exists():
        return None
    rows = [json.loads(ln) for ln in mani.read_text(encoding="utf-8").splitlines() if ln.strip()]
    pick = [r for r in rows if r.get("emotion") == "neutral" and 2 < r.get("duration", 0) < 8]
    if not pick:
        return None
    r = pick[0]
    wav, sr = torchaudio.load(str(DATA / voice / r["audio"]))
    wav = wav.mean(0)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    out = (r.get("raw_text", ""), wav.numpy())
    _anchor_cache[voice] = out
    return out


def _generate(voice, text, emotion, temperature, seed, use_anchor, max_new_tokens):
    tagged = (f"({emotion}) " if emotion else "") + text
    convo = []
    if use_anchor:
        a = _anchor_for(voice)
        if a:
            convo.append({"role": "0", "content": [
                {"type": "text", "text": a[0]}, {"type": "audio", "audio": a[1]}]})
    convo.append({"role": "0", "content": [{"type": "text", "text": tagged}]})
    inputs = _proc.apply_chat_template(
        convo, tokenize=True, return_dict=True, return_tensors="pt").to("cuda")
    if seed is not None:
        torch.manual_seed(seed)
    gen_kwargs = dict(output_audio=True, do_sample=True, temperature=float(temperature))
    if max_new_tokens:
        gen_kwargs["max_new_tokens"] = int(max_new_tokens)
    with torch.no_grad():
        audio = _model.generate(**inputs, **gen_kwargs)
    wav = audio[0].to(torch.float32).cpu().reshape(-1)
    return wav


def _norm(s: str) -> list[str]:
    return "".join(c.lower() if (c.isalnum() or c == " ") else " " for c in s).split()


def _wer(ref: list[str], hyp: list[str]) -> float:
    # Levenshtein over words
    n, m = len(ref), len(hyp)
    if n == 0:
        return 1.0 if m else 0.0
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return d[m] / n


def _score(wav, target: str) -> dict:
    global _whisper
    from faster_whisper import WhisperModel
    if _whisper is None:
        _whisper = WhisperModel("base", device="cuda", compute_type="float16")
    dur = wav.numel() / SR
    rms = float(wav.pow(2).mean().sqrt())
    arr = wav.numpy().astype("float32")
    segs, _ = _whisper.transcribe(arr, language="en")
    hyp = " ".join(s.text.strip() for s in segs).strip()
    wer = _wer(_norm(target), _norm(hyp))
    flags = []
    if rms < 0.006:
        flags.append("near-silent")
    if dur < 0.5:
        flags.append("too-short")
    words = max(1, len(_norm(target)))
    if dur > words * 1.2 + 2:
        flags.append("too-long/runaway")
    score = wer + (0.6 if "near-silent" in flags else 0) \
        + (0.4 if "too-short" in flags else 0) + (0.3 if "too-long/runaway" in flags else 0)
    return {"wer": round(wer, 2), "dur": round(dur, 1), "rms": round(rms, 3),
            "transcript": hyp, "flags": flags, "score": round(score, 3)}


class GenReq(BaseModel):
    text: str
    voice: str = "ruby"
    emotion: str = ""
    n: int = 3
    temperature: float = 0.7
    seed: int | None = None
    use_anchor: bool = True
    max_new_tokens: int | None = None


@app.post("/api/generate")
def api_generate(req: GenReq):
    if not req.text.strip():
        return {"error": "empty text"}
    _use_voice(req.voice)
    cands = []
    n = max(1, min(req.n, 8))
    for i in range(n):
        seed = (req.seed + i) if req.seed is not None else None
        t0 = time.monotonic()
        wav = _generate(req.voice, req.text, req.emotion, req.temperature,
                        seed, req.use_anchor, req.max_new_tokens)
        sc = _score(wav, req.text)
        fn = SAMPLES / f"{int(time.time() * 1000)}_{i}.wav"
        torchaudio.save(str(fn), wav.unsqueeze(0), SR)
        sc.update(url=f"/audio/{fn.name}", gen_s=round(time.monotonic() - t0, 1), idx=i)
        cands.append(sc)
    cands.sort(key=lambda c: c["score"])
    if cands:
        cands[0]["best"] = True
    return {"candidates": cands, "voice": req.voice, "emotion": req.emotion, "text": req.text}


@app.get("/audio/{name}")
def api_audio(name: str):
    return FileResponse(SAMPLES / name)


@app.get("/api/voices")
def api_voices():
    return {"voices": list_voices(), "emotions": EMOTIONS}


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "studio.html").read_text(encoding="utf-8")


# ── OpenAI-compatible TTS provider ────────────────────────────────────
# Lets you plug this server into Augmentum as a custom TTS provider by URL.
# It serves the FINE-TUNED voices (ruby, tess, ...) — pick one via the `voice`
# field. Emotion can ride along as "voice:emotion" (e.g. "ruby:contentment").

def _wav_bytes(wav: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torchaudio.save(buf, wav.unsqueeze(0), SR, format="wav")
    return buf.getvalue()


class SpeechReq(BaseModel):
    model: str = ""
    input: str = ""
    voice: str = "ruby"
    response_format: str = "wav"
    speed: float = 1.0
    instructions: str = ""


@app.get("/v1/models")
def v1_models():
    return {"object": "list", "data": [
        {"id": "csm-voice-studio", "object": "model", "owned_by": "local"}]}


@app.get("/v1/voices")
@app.get("/v1/audio/voices")
def v1_voices():
    # one id per trained voice; Augmentum's picker lists these
    return JSONResponse([{"id": v, "name": v} for v in list_voices()])


@app.post("/v1/audio/speech")
def v1_speech(req: SpeechReq):
    text = (req.input or "").strip()
    if not text:
        return JSONResponse({"error": "input is required"}, status_code=400)
    voice, _, emotion = req.voice.partition(":")   # "ruby:contentment"
    voice = voice or (list_voices() or ["ruby"])[0]
    _use_voice(voice)
    # single take with the reference anchor — keep latency to one generation.
    wav = _generate(voice, text, emotion, 0.7, None, True, None)
    return Response(content=_wav_bytes(wav), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    print("[studio] open http://localhost:7860", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=7860)
