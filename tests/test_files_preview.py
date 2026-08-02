"""Tests for the preview helpers introduced in the streaming/HEIC pass.

Covers two areas:
  * `_transcode_image_to_jpeg` — TIFF in / JPEG out (HEIC path needs an
    optional dependency we don't gate the suite on).
  * Range-request streaming via FileResponse — confirm the stack returns
    206 Partial Content with the right body slice when a Range header is
    sent. Built around a minimal FastAPI app to keep the harness tiny.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.testclient import TestClient
from PIL import Image

from augmentum.proxy.files_routes import _transcode_image_to_jpeg

# --- Image transcoding -------------------------------------------------

class TestTranscodeImage:
    def _write_temp(self, fmt: str, suffix: str) -> Path:
        img = Image.new("RGB", (40, 40), "red")
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            tf.write(buf.getvalue())
            return Path(tf.name)

    def test_tiff_transcodes_to_jpeg(self):
        # TIFF is supported by stock Pillow on every install, so this
        # path runs in CI without optional deps.
        path = self._write_temp("TIFF", ".tiff")
        try:
            out = _transcode_image_to_jpeg(path, "tiff")
            assert out is not None
            assert out[:3] == b"\xff\xd8\xff"  # JPEG magic
            # Round-trip the bytes through Pillow to confirm validity.
            img = Image.open(io.BytesIO(out))
            assert img.format == "JPEG"
            assert img.size == (40, 40)
        finally:
            path.unlink(missing_ok=True)

    def test_rgba_image_flattens_to_white_bg(self):
        # JPEG can't carry alpha — the helper composites onto white so
        # transparent corners don't render as black.
        img = Image.new("RGBA", (20, 20), (0, 0, 0, 0))  # fully transparent
        buf = io.BytesIO()
        img.save(buf, format="TIFF")
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tf:
            tf.write(buf.getvalue())
            path = Path(tf.name)
        try:
            out = _transcode_image_to_jpeg(path, "tiff")
            assert out is not None
            jpg = Image.open(io.BytesIO(out))
            # Centre pixel should be white-ish, not black.
            r, g, b = jpg.getpixel((10, 10))
            assert r > 200 and g > 200 and b > 200
        finally:
            path.unlink(missing_ok=True)

    def test_heic_returns_none_when_lib_missing(self):
        # pillow-heif is optional; without it the helper returns None
        # so the route surfaces the friendly fallback shell instead.
        try:
            import pillow_heif  # noqa: F401
            # Library is installed — skip the negative-path assertion.
            return
        except ImportError:
            pass
        with tempfile.NamedTemporaryFile(suffix=".heic", delete=False) as tf:
            tf.write(b"not really heic")
            path = Path(tf.name)
        try:
            assert _transcode_image_to_jpeg(path, "heic") is None
        finally:
            path.unlink(missing_ok=True)

    def test_corrupt_file_returns_none(self):
        # Garbage bytes — Pillow.open() raises; helper swallows + logs.
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tf:
            tf.write(b"definitely not a tiff")
            path = Path(tf.name)
        try:
            assert _transcode_image_to_jpeg(path, "tiff") is None
        finally:
            path.unlink(missing_ok=True)


# --- Range-request streaming -------------------------------------------

def _make_range_app(payload: bytes, suffix: str = ".bin") -> tuple[FastAPI, Path]:
    """Spin up a minimal FastAPI app that serves `payload` via FileResponse,
    matching how the real download route hands files off."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(payload)
    tmp.close()
    path = Path(tmp.name)

    app = FastAPI()

    @app.get("/file")
    def serve():
        return FileResponse(str(path), media_type="application/octet-stream")

    return app, path


class TestRangeRequests:
    def test_full_response_no_range(self):
        payload = b"abcdefghij" * 1000  # 10 KB
        app, path = _make_range_app(payload)
        try:
            with TestClient(app) as client:
                resp = client.get("/file")
                assert resp.status_code == 200
                assert resp.content == payload
                # Starlette advertises range support for FileResponse.
                assert resp.headers.get("accept-ranges") == "bytes"
        finally:
            path.unlink(missing_ok=True)

    def test_partial_response_with_range(self):
        payload = bytes(range(256)) * 4  # 1 KB of distinguishable bytes
        app, path = _make_range_app(payload)
        try:
            with TestClient(app) as client:
                # First 100 bytes.
                resp = client.get("/file", headers={"Range": "bytes=0-99"})
                assert resp.status_code == 206
                assert resp.content == payload[:100]
                assert resp.headers.get("content-range", "").startswith("bytes 0-99/")
                assert resp.headers.get("content-length") == "100"
        finally:
            path.unlink(missing_ok=True)

    def test_seek_into_middle(self):
        payload = b"".join(bytes([i % 256]) * 10 for i in range(100))
        app, path = _make_range_app(payload)
        try:
            with TestClient(app) as client:
                # Pull bytes 500-599 — simulates a video player seeking.
                resp = client.get("/file", headers={"Range": "bytes=500-599"})
                assert resp.status_code == 206
                assert resp.content == payload[500:600]
        finally:
            path.unlink(missing_ok=True)

    def test_open_ended_range(self):
        # bytes=N- means "from N to EOF" — the form video players use to
        # request the rest of the file after the moov atom.
        payload = b"x" * 500 + b"y" * 500
        app, path = _make_range_app(payload)
        try:
            with TestClient(app) as client:
                resp = client.get("/file", headers={"Range": "bytes=500-"})
                assert resp.status_code == 206
                assert resp.content == b"y" * 500
        finally:
            path.unlink(missing_ok=True)
