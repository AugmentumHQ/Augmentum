"""Speech-to-text helpers — transcription and audio transcoding.

Extracted from pipeline.py for modularity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.proxy.audio_routes import (
    _audio_client,
    _build_headers,
    _get_default_provider,
    _is_deepgram,
)
from augmentum.utils.http_client import normalize_base_url
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Content Types
# ---------------------------------------------------------------------------

_AUDIO_CONTENT_TYPES = {
    ".webm": "audio/webm",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".opus": "audio/opus",
}


def _content_type_for(filename: str) -> str:
    """Infer audio content type from filename extension."""
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    return _AUDIO_CONTENT_TYPES.get(ext, "audio/webm")


# ---------------------------------------------------------------------------
# Transcoding
# ---------------------------------------------------------------------------


def _transcode_to_wav(audio_bytes: bytes) -> tuple[bytes, str, str]:
    """Transcode audio to WAV via ffmpeg for maximum STT compatibility.

    Returns (wav_bytes, filename, content_type).
    Falls back to original bytes if ffmpeg is unavailable.
    """
    import os
    import shutil
    import subprocess
    import tempfile

    if not shutil.which("ffmpeg"):
        log.debug("ffmpeg_not_found_skipping_transcode")
        return audio_bytes, "recording.webm", "audio/webm"

    inp_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as inp:
            inp.write(audio_bytes)
            inp_path = inp.name

        out_path = inp_path.rsplit(".", 1)[0] + ".wav"

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", inp_path, "-ar", "16000", "-ac", "1",
             "-f", "wav", out_path],
            capture_output=True, timeout=10,
        )

        os.unlink(inp_path)
        inp_path = None

        if result.returncode == 0 and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                wav_bytes = f.read()
            os.unlink(out_path)
            log.debug("audio_transcoded", from_size=len(audio_bytes), to_size=len(wav_bytes))
            return wav_bytes, "recording.wav", "audio/wav"

        if os.path.exists(out_path):
            os.unlink(out_path)

        log.debug("ffmpeg_transcode_failed", stderr=result.stderr[:200] if result.stderr else "")
        return audio_bytes, "recording.webm", "audio/webm"
    except Exception:
        log.debug("transcode_error", exc_info=True)
        for p in (inp_path, out_path):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass
        return audio_bytes, "recording.webm", "audio/webm"


# ---------------------------------------------------------------------------
# STT API
# ---------------------------------------------------------------------------


async def transcribe_audio(
    audio_bytes: bytes,
    conn,
    filename: str = "recording.webm",
    language: str = "",
) -> str:
    """Send audio to the configured STT provider and return transcript text."""
    provider = await _get_default_provider(conn, "stt")
    if not provider:
        raise RuntimeError("No STT provider configured")

    # Transcode webm→wav for broader STT compatibility
    # Run in a thread to avoid blocking the event loop (ffmpeg subprocess)
    if filename.endswith(".webm"):
        import asyncio
        audio_bytes, filename, _ = await asyncio.to_thread(
            _transcode_to_wav, audio_bytes,
        )

    base_url = normalize_base_url(provider["base_url"])
    headers = _build_headers(provider["api_key"], base_url=base_url)
    model = provider["default_model"]
    content_type = _content_type_for(filename)

    async with _audio_client(base_url) as client:
        if _is_deepgram(base_url):
            # Deepgram: POST /v1/listen with raw audio body
            params: dict[str, str] = {}
            if model:
                params["model"] = model
            if language:
                params["language"] = language
            dg_headers = {**headers, "Content-Type": content_type}
            resp = await client.post(
                f"{base_url}/v1/listen",
                content=audio_bytes,
                params=params,
                headers=dg_headers,
            )
            resp.raise_for_status()
            dg_data = resp.json()
            channels = dg_data.get("results", {}).get("channels", [])
            if channels:
                alts = channels[0].get("alternatives", [])
                if alts:
                    return alts[0].get("transcript", "").strip()
            return ""

        # OpenAI-compatible: multipart form
        files_data = {
            "file": (filename, audio_bytes, content_type),
        }
        form_data: dict[str, str] = {}
        if model:
            form_data["model"] = model
        if language:
            form_data["language"] = language

        resp = await client.post(
            f"{base_url}/v1/audio/transcriptions",
            files=files_data,
            data=form_data,
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("text", "").strip()
