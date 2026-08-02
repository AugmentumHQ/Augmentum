"""Post-processing operations for generated images: upscaling and background removal."""

from __future__ import annotations

import asyncio
import os
import uuid

from augmentum.config import settings
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Lazy singletons — loaded on first use, freed on demand
_upscale_model = None
_upscale_device = None
_rembg_session = None

TILE_SIZE = 512
TILE_PAD = 10
DEFAULT_UPSCALE_MODEL = "4x-UltraSharp.pth"


def _get_output_dir() -> str:
    d = settings.image_output_dir or os.path.join(settings.data_dir, "image_output")
    os.makedirs(d, exist_ok=True)
    return d


def _get_upscale_models_dir() -> str:
    d = os.path.join(settings.data_dir, "upscale_models")
    os.makedirs(d, exist_ok=True)
    return d


def _resolve_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


ULTRASHARP_URL = (
    "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth"
)


def _ensure_default_model() -> str:
    """Download 4x-UltraSharp if not present. Returns the model path."""
    model_path = os.path.join(_get_upscale_models_dir(), DEFAULT_UPSCALE_MODEL)
    if os.path.isfile(model_path):
        return model_path

    log.info("downloading_upscale_model", model=DEFAULT_UPSCALE_MODEL, url=ULTRASHARP_URL)
    import httpx

    with httpx.stream("GET", ULTRASHARP_URL, follow_redirects=True, timeout=120) as resp:
        resp.raise_for_status()
        tmp_path = model_path + ".tmp"
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1024 * 64):
                f.write(chunk)
    os.replace(tmp_path, model_path)
    log.info("upscale_model_downloaded", path=model_path)
    return model_path


def _load_upscale_model(model_path: str | None = None):
    """Load an upscale model via spandrel. Caches globally."""
    global _upscale_model, _upscale_device
    import torch
    import spandrel

    if model_path is None:
        model_path = _ensure_default_model()

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Upscale model not found: {model_path}. "
            f"Place a .pth upscale model in {_get_upscale_models_dir()}/"
        )

    device = _resolve_device()
    log.info("loading_upscale_model", path=model_path, device=device)

    loader = spandrel.ModelLoader(device=device)
    model = loader.load_from_file(model_path)
    if device == "cuda":
        model = model.half()
    model.eval()

    _upscale_model = model
    _upscale_device = device
    return model


def _upscale_tiled(img_tensor, model, scale: int):
    """Process image in tiles to keep VRAM bounded."""
    import torch

    _, _, h, w = img_tensor.shape
    out_h, out_w = h * scale, w * scale
    output = torch.empty(1, 3, out_h, out_w, device=img_tensor.device, dtype=img_tensor.dtype)

    for y in range(0, h, TILE_SIZE):
        for x in range(0, w, TILE_SIZE):
            # Tile boundaries with padding
            y1 = max(0, y - TILE_PAD)
            x1 = max(0, x - TILE_PAD)
            y2 = min(h, y + TILE_SIZE + TILE_PAD)
            x2 = min(w, x + TILE_SIZE + TILE_PAD)

            tile = img_tensor[:, :, y1:y2, x1:x2]
            with torch.no_grad():
                tile_out = model(tile)

            # Calculate output region (without padding)
            oy1 = (y - y1) * scale
            ox1 = (x - x1) * scale
            oy2 = oy1 + min(TILE_SIZE, h - y) * scale
            ox2 = ox1 + min(TILE_SIZE, w - x) * scale

            out_y = y * scale
            out_x = x * scale
            out_ye = out_y + (oy2 - oy1)
            out_xe = out_x + (ox2 - ox1)

            output[:, :, out_y:out_ye, out_x:out_xe] = tile_out[:, :, oy1:oy2, ox1:ox2]

    return output


async def upscale_image(
    image_path: str, scale: int = 4, model_path: str | None = None,
) -> tuple[str, str, int, int]:
    """Upscale an image. Returns (new_image_id, new_file_path, width, height)."""
    import torch
    import numpy as np
    from PIL import Image

    def _run():
        global _upscale_model
        model = _upscale_model or _load_upscale_model(model_path)

        img = Image.open(image_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

        device = _upscale_device or "cpu"
        tensor = tensor.to(device)
        if device == "cuda":
            tensor = tensor.half()

        # Use tiling for images larger than tile size, direct for small
        _, _, h, w = tensor.shape
        if h > TILE_SIZE or w > TILE_SIZE:
            result = _upscale_tiled(tensor, model, scale)
        else:
            with torch.no_grad():
                result = model(tensor)

        result = result.squeeze(0).float().clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        result = (result * 255).astype(np.uint8)
        out_img = Image.fromarray(result)

        image_id = uuid.uuid4().hex[:16]
        out_path = os.path.join(_get_output_dir(), f"{image_id}.png")
        out_img.save(out_path)

        return image_id, out_path, out_img.width, out_img.height

    return await asyncio.to_thread(_run)


async def remove_background(image_path: str) -> tuple[str, str, int, int]:
    """Remove background from an image. Returns (new_image_id, new_file_path, width, height)."""
    from PIL import Image

    def _run():
        global _rembg_session
        from rembg import remove, new_session

        if _rembg_session is None:
            # Prefer the image-baked model (downloaded once at build time, lives
            # outside the /data volume so it survives volume mounts). Falls
            # through to the persistent data volume if missing — that path is
            # still authoritative for user-installed model variants.
            prebaked_dir = "/home/augmentum/.u2net"
            prebaked_model = os.path.join(prebaked_dir, "isnet-general-use.onnx")
            if os.path.isfile(prebaked_model):
                os.environ.setdefault("U2NET_HOME", prebaked_dir)
                log.info("loading_rembg_model", model="isnet-general-use", cache_dir=prebaked_dir)
            else:
                rembg_dir = os.path.join(settings.data_dir, "rembg_models")
                os.makedirs(rembg_dir, exist_ok=True)
                os.environ.setdefault("U2NET_HOME", rembg_dir)
                log.info("loading_rembg_model", model="isnet-general-use", cache_dir=rembg_dir)
            _rembg_session = new_session("isnet-general-use")

        img = Image.open(image_path).convert("RGBA")
        result = remove(img, session=_rembg_session)

        image_id = uuid.uuid4().hex[:16]
        out_path = os.path.join(_get_output_dir(), f"{image_id}.png")
        result.save(out_path, "PNG")

        return image_id, out_path, result.width, result.height

    return await asyncio.to_thread(_run)


def unload_upscaler() -> None:
    """Free VRAM used by the upscale model."""
    global _upscale_model, _upscale_device
    if _upscale_model is not None:
        del _upscale_model
        _upscale_model = None
        _upscale_device = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        log.info("upscaler_unloaded")


def unload_rembg() -> None:
    """Free memory used by rembg session."""
    global _rembg_session
    if _rembg_session is not None:
        del _rembg_session
        _rembg_session = None
        log.info("rembg_unloaded")
