"""safetensors analyzer — reports tensor shapes + embedded metadata.

The metadata block is where LoRA trigger words, base-model fingerprints,
and training hyperparameters live. Surfacing it makes the file legible
without scanning every shard.
"""

from __future__ import annotations

from augmentum.coder.analyzers.registry import (
    AnalysisReport,
    register_analyzer,
)


class SafetensorsAnalyzer:
    name = "safetensors"
    extensions = ("safetensors",)
    # safetensors files start with a u64 LE header length, no fixed magic
    magic_bytes = ()

    async def analyze(self, path: str, raw: bytes) -> AnalysisReport:
        try:
            from safetensors import safe_open  # type: ignore
        except ImportError:
            return AnalysisReport(
                format="safetensors (lib missing)",
                summary=(
                    "safetensors file detected but the `safetensors` package "
                    "isn't installed."
                ),
            )

        with safe_open(path, framework="pt") as f:
            metadata = f.metadata() or {}
            keys = list(f.keys())
            tensor_summary: list[str] = []
            total_params = 0
            for key in keys[:8]:
                t = f.get_slice(key)
                shape = t.get_shape()
                dtype = t.get_dtype()
                size = 1
                for dim in shape:
                    size *= int(dim)
                total_params += size
                tensor_summary.append(f"  - {key}: {tuple(shape)} {dtype}")
            for key in keys[8:]:
                t = f.get_slice(key)
                shape = t.get_shape()
                size = 1
                for dim in shape:
                    size *= int(dim)
                total_params += size

        size_mb = len(raw) / (1024 * 1024) if raw else 0

        meta_bullets: list[str] = []
        for k in sorted(metadata.keys())[:15]:
            v = metadata.get(k, "")
            if len(str(v)) > 120:
                v = str(v)[:117] + "..."
            meta_bullets.append(f"  - {k}: {v}")

        bullets = [
            f"- File size: {size_mb:,.2f} MB",
            f"- Tensor count: {len(keys)}",
            f"- Total parameters: {total_params:,}",
        ]

        sections = [
            "safetensors archive\n",
            "\n".join(bullets),
        ]
        if metadata:
            sections.append("\n**Metadata:**\n" + "\n".join(meta_bullets))
        if tensor_summary:
            sections.append(
                f"\n**First {len(tensor_summary)} tensors:**\n"
                + "\n".join(tensor_summary)
            )
        if len(keys) > 8:
            sections.append(f"\n…and {len(keys) - 8} more tensors.")

        summary = "\n".join(sections)

        return AnalysisReport(
            format="safetensors",
            summary=summary,
            details={
                "tensor_count": len(keys),
                "total_params": total_params,
                "size_mb": round(size_mb, 2),
                "metadata": dict(metadata),
                "tensor_keys_preview": keys[:16],
            },
            raw_size_bytes=len(raw),
        )


register_analyzer(SafetensorsAnalyzer())
