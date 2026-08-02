"""Blender asset stage — produce a real GLB + verify render for the 3d path.

For the MVP this runs a deterministic bpy script (a materialed crate) headless
under xvfb, exporting a GLB the generated game loads and a PNG the visual-verify
stage inspects. It proves the Blender→GLB→game→play ring with a genuinely
Blender-made asset. A model-authored bpy script (arbitrary assets) is the
natural next step — it swaps the template for a generated script but keeps this
run+read plumbing.

Runs via the container executor directly (write script, ``xvfb-run blender``,
read outputs) — the same path ``BlenderRunTool`` uses, without constructing a
full coder tool. Needs a workspace on the ``creative`` profile (Blender+xvfb).
"""
from __future__ import annotations

import shlex
from typing import Any

from augmentum.coder.executors import ContainerExecutor
from augmentum.coder.foundry.contract import GameBuildSpec
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Deterministic bpy build: a materialed crate, one sun, a 3/4 camera, Eevee
# render to PNG, GLB export. Reads --engine/--out-glb/--out-png after `--`.
# Engine id differs across Blender versions (EEVEE → EEVEE_NEXT in 4.2+); we
# try the requested id then fall back so the same script works on either.
_BPY_TEMPLATE = r'''
import bpy, sys

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def _arg(name, default):
    return argv[argv.index(name) + 1] if name in argv else default

engine = _arg("--engine", "BLENDER_EEVEE")
out_glb = _arg("--out-glb", "/workspace/out.glb")
out_png = _arg("--out-png", "/workspace/out.png")

bpy.ops.wm.read_factory_settings(use_empty=True)

bpy.ops.mesh.primitive_cube_add(size=1.4)
obj = bpy.context.active_object
bpy.ops.object.modifier_add(type="BEVEL")
try:
    obj.modifiers["Bevel"].width = 0.05
except Exception:
    pass

mat = bpy.data.materials.new("Crate")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (0.62, 0.36, 0.16, 1.0)
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.7
obj.data.materials.append(mat)

cam_data = bpy.data.cameras.new("Cam")
cam = bpy.data.objects.new("Cam", cam_data)
bpy.context.scene.collection.objects.link(cam)
cam.location = (3.0, -3.0, 2.2)
cam.rotation_euler = (1.05, 0.0, 0.785)
bpy.context.scene.camera = cam

sun_data = bpy.data.lights.new("Sun", type="SUN")
sun = bpy.data.objects.new("Sun", sun_data)
bpy.context.scene.collection.objects.link(sun)
sun.location = (4.0, -2.0, 5.0)
sun_data.energy = 4.0

scene = bpy.context.scene
for eng in (engine, "BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
    try:
        scene.render.engine = eng
        break
    except Exception:
        continue
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = out_png
scene.render.resolution_x = 256
scene.render.resolution_y = 256
bpy.ops.render.render(write_still=True)

bpy.ops.export_scene.gltf(filepath=out_glb, export_format="GLB")
print("Saved: '%s'" % out_png)
print("Saved: '%s'" % out_glb)
'''


def make_asset_stage(app_state: Any, *, workspace_id: str, timeout_s: float = 300.0):
    """Return an async ``asset(spec) -> {glb_asset, render_png_bytes}`` stage.

    Writes the GLB under the generated game's ``assets/`` dir so the game can
    load it by the relative path returned in ``glb_asset``. The PNG render is
    returned as bytes for the visual-verify stage.
    """
    cm = getattr(app_state, "container_manager", None)

    async def asset(spec: GameBuildSpec) -> dict:
        if cm is None:
            return {"glb_asset": "", "render_png_bytes": None}
        ex = ContainerExecutor(cm, workspace_id)
        gen_dir = f"/workspace/generated/{spec.slug}"
        rel_glb = f"assets/{spec.slug}.glb"
        out_glb = f"{gen_dir}/{rel_glb}"
        out_png = f"{gen_dir}/assets/{spec.slug}.render.png"
        script_path = f"{gen_dir}/__asset_build.py"

        await ex.run_command(["bash", "-c", f"mkdir -p {gen_dir}/assets"], timeout=15.0)
        await ex.write_file(script_path, _BPY_TEMPLATE)

        cmd = (
            f"cd /workspace && xvfb-run -a blender --background --python-exit-code 1 "
            f"--python {shlex.quote(script_path)} -- "
            f"--engine BLENDER_EEVEE --out-glb {shlex.quote(out_glb)} "
            f"--out-png {shlex.quote(out_png)} 2>&1"
        )
        try:
            out = await ex.run_command(["bash", "-c", cmd], timeout=timeout_s)
        except Exception as exc:
            log.warning("foundry_blender_asset_failed", slug=spec.slug, error=str(exc))
            return {"glb_asset": "", "render_png_bytes": None}

        # Confirm the GLB landed (designed ≠ applied) and read the render.
        glb_ok = False
        try:
            sz = await ex.run_command(
                ["bash", "-c", f"stat -c %s {shlex.quote(out_glb)} 2>/dev/null"],
                timeout=10.0,
            )
            glb_ok = sz.strip().isdigit() and int(sz.strip()) > 0
        except Exception:
            glb_ok = False
        if not glb_ok:
            log.warning("foundry_blender_no_glb", slug=spec.slug, tail=out[-400:])
            return {"glb_asset": "", "render_png_bytes": None}

        render_bytes: bytes | None = None
        try:
            render_bytes = await ex.read_file_bytes(out_png)
        except Exception:
            render_bytes = None

        return {"glb_asset": rel_glb, "render_png_bytes": render_bytes}

    return asset
