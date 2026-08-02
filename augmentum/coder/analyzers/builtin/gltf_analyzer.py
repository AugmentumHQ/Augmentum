"""GLB / glTF / VRM analyzer.

VRM is glTF 2.0 + the ``VRM`` (0.x) or ``VRMC_vrm`` (1.0) extension —
parse with pygltflib and report mesh shape, materials, skin/humanoid
metadata, and the VRM extension contents when present.
"""

from __future__ import annotations

import json

from augmentum.coder.analyzers.registry import (
    AnalysisReport,
    register_analyzer,
)


class GLTFAnalyzer:
    name = "gltf"
    extensions = ("glb", "gltf", "vrm")
    magic_bytes = (b"glTF",)  # GLB binary container magic

    async def analyze(self, path: str, raw: bytes) -> AnalysisReport:
        try:
            from pygltflib import GLTF2  # type: ignore
        except ImportError:
            return AnalysisReport(
                format="glTF (lib missing)",
                summary=(
                    "glTF / VRM file detected but the `pygltflib` package "
                    "isn't installed. Install via `pip install pygltflib`."
                ),
            )

        gltf = GLTF2().load(path)
        if gltf is None:
            return AnalysisReport(
                format="glTF",
                summary="File header looked like glTF but the parser returned no model.",
            )

        meshes = gltf.meshes or []
        materials = gltf.materials or []
        nodes = gltf.nodes or []
        skins = gltf.skins or []
        accessors = gltf.accessors or []
        animations = gltf.animations or []
        textures = gltf.textures or []
        extensions_used = list(gltf.extensionsUsed or [])

        # Vertex count = sum of POSITION accessors across all primitives
        vert_total = 0
        primitive_total = 0
        for mesh in meshes:
            for prim in (mesh.primitives or []):
                primitive_total += 1
                pos_idx = getattr(prim.attributes, "POSITION", None) if prim.attributes else None
                if pos_idx is not None and 0 <= pos_idx < len(accessors):
                    vert_total += accessors[pos_idx].count or 0

        # VRM extension contents — both 0.x (top-level "VRM") and 1.0
        # ("VRMC_vrm") shapes. Pick the first present.
        ext_data = gltf.extensions or {}
        vrm_meta = ext_data.get("VRM") or ext_data.get("VRMC_vrm") or {}
        is_vrm = bool(vrm_meta) or path.lower().endswith(".vrm")

        vrm_lines: list[str] = []
        if vrm_meta:
            meta = vrm_meta.get("meta") or vrm_meta.get("vrm", {}).get("meta") or {}
            humanoid = vrm_meta.get("humanoid") or {}
            bones = humanoid.get("humanBones") or {}
            blendshapes = (
                vrm_meta.get("blendShapeMaster", {}).get("blendShapeGroups")
                or vrm_meta.get("expressions", {}).get("preset")
                or []
            )
            if meta:
                title = meta.get("title") or meta.get("name") or "(untitled)"
                author = meta.get("author") or "(unknown)"
                vrm_lines.append(f"- VRM title: {title}")
                vrm_lines.append(f"- VRM author: {author}")
            if bones:
                bone_count = len(bones) if isinstance(bones, dict) else len(list(bones))
                vrm_lines.append(f"- Humanoid bones mapped: {bone_count}")
            if blendshapes:
                count = len(blendshapes) if hasattr(blendshapes, "__len__") else 0
                vrm_lines.append(f"- Blendshape groups: {count}")

        size_mb = len(raw) / (1024 * 1024) if raw else 0
        format_label = "VRM avatar" if is_vrm else ("GLB binary" if path.endswith(".glb") else "glTF JSON")

        bullets = [
            f"- Format: {format_label}",
            f"- File size: {size_mb:,.2f} MB",
            f"- Meshes: {len(meshes)} ({primitive_total} primitives, ~{vert_total:,} vertices total)",
            f"- Materials: {len(materials)}",
            f"- Textures: {len(textures)}",
            f"- Nodes: {len(nodes)}",
            f"- Skins: {len(skins)}",
            f"- Animations: {len(animations)}",
            f"- Accessors: {len(accessors)}",
        ]
        if extensions_used:
            bullets.append(f"- Extensions used: {', '.join(extensions_used)}")
        bullets.extend(vrm_lines)

        summary = (
            f"{format_label} file\n\n" + "\n".join(bullets)
            + "\n\nFor specific nodes/materials/animations, call "
            "`analyze_file(path)` and inspect `details`."
        )

        return AnalysisReport(
            format=format_label,
            summary=summary,
            details={
                "mesh_count": len(meshes),
                "primitive_count": primitive_total,
                "vertex_total": vert_total,
                "material_count": len(materials),
                "texture_count": len(textures),
                "node_count": len(nodes),
                "skin_count": len(skins),
                "animation_count": len(animations),
                "accessor_count": len(accessors),
                "extensions_used": extensions_used,
                "is_vrm": is_vrm,
                "vrm_meta_keys": sorted(vrm_meta.keys()) if isinstance(vrm_meta, dict) else [],
            },
            raw_size_bytes=len(raw),
        )


register_analyzer(GLTFAnalyzer())
