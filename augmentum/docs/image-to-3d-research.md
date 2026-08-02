# Image-to-3D Model Generation Research

**Date:** March 2026
**Status:** Parked — revisit after core features are stable
**Use Case:** Generate 3D avatars from character card images in narrative mode

## Summary

Single-image-to-3D is feasible on consumer GPUs (6-8 GB VRAM) with sub-second generation times. Two strong open-source candidates exist. The integration pattern maps cleanly to our existing image pipeline architecture.

## Model Comparison

| | TripoSR | SF3D (Stable Fast 3D) |
|---|---------|------|
| **By** | Tripo AI + Stability AI | Stability AI |
| **VRAM** | 6 GB (3.5 GB half-precision) | 6 GB |
| **Speed** | <0.5s (A100), seconds on consumer | ~0.5s |
| **Texture** | Vertex colors (default), UV bake optional (2048px) | UV-unwrapped texture (384x384 triplane), PBR |
| **Materials** | Albedo only | Albedo + Roughness + Metallic |
| **Lighting** | Baked in | Delighting step (relightable) |
| **Mesh** | Marching Cubes | Marching Tetrahedron (smoother) |
| **Output** | .obj, .glb | .glb |
| **License** | MIT | Stability AI Community (free <$1M rev) |
| **Deps** | Simple (pip, xatlas wheels) | Complex (C extensions: texture_baker, uv_unwrapper) |
| **Install** | `pip install -r requirements.txt` | Needs C++ build toolchain |

**Recommendation:** Start with TripoSR (simpler deps, MIT, good enough for testing). Upgrade to SF3D if quality matters.

## Integration Plan

### Architecture (follows image pipeline pattern)

```
augmentum/mesh3d/
  pipeline.py        — Model loading, inference, export (~80 lines)
augmentum/proxy/
  mesh3d_routes.py   — POST /api/mesh3d/generate, GET /api/mesh3d/{id} (~60 lines)
augmentum/state/migrations/
  0XX_mesh3d.sql     — mesh3d_generations table
config.py            — mesh3d_enabled, mesh3d_output_dir, mesh3d_texture_resolution
```

### Core Inference (~10 lines)

```python
from tsr.system import TSR

model = TSR.from_pretrained("stabilityai/TripoSR", config_name="config.yaml", weight_name="model.ckpt")
model.to(device)

# image = PIL Image with background removed
scene_codes = model([image], device=device)
meshes = model.extract_mesh(scene_codes, resolution=256)

# With texture baking:
bake_output = bake_texture(meshes[0], model, scene_codes[0], texture_resolution=1024)
xatlas.export("output.glb", vertices, indices, uvs, normals)
```

### UI Integration

1. "Generate 3D" button on character card in narrative inspector
2. `<model-viewer>` component (Google's web component) or Three.js for .glb display
3. Rotatable/zoomable 3D preview in inspector panel
4. Store .glb in mesh3d_output/, metadata in SQLite

### Config Settings

```python
mesh3d_enabled: bool = False
mesh3d_output_dir: str = ""           # auto: {data_dir}/mesh3d_output
mesh3d_texture_resolution: int = 1024
mesh3d_half_precision: bool = True    # saves ~2.5 GB VRAM
mesh3d_timeout: float = 30.0         # seconds (much faster than image gen)
```

### Dependencies (TripoSR path)

```
# Core
tsr (from TripoSR repo)
trimesh>=4.0
xatlas-python       # pre-built wheels, no compilation

# Background removal
rembg>=2.0          # pulls onnxruntime (~500MB)

# Already in project
torch, numpy, pillow
```

### Key Concerns

- **Anime/stylized art quality** — models trained on photorealistic objects. Back-of-character is guesswork from single image. Need to test with actual character card art.
- **GPU sharing** — must coordinate with image pipeline (same GPU, lazy load/unload pattern)
- **Docker** — straightforward (pip install in Dockerfile). Bare-metal Windows needs testing.
- **rembg** — 500MB dependency for background removal. Could make optional if avatar images already have transparent backgrounds.

## Video Generation (also researched, lower priority)

Local video gen is possible but slow and resource-heavy. Best options:
- **FramePack** — 6 GB VRAM, minutes per clip, up to 60s video
- **Wan2.1 1.3B** — 8 GB VRAM, 480p, ~4 min per 5s clip
- **LTX-Video 2B** — 6-8 GB VRAM, short clips, needs 32 GB+ RAM

Cloud video gen costs $0.04-0.40/second (Hailuo cheapest, Veo 3.1 most expensive).

Not practical for integration until local speed improves significantly.
