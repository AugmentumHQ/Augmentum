"""GGUF model analyzer — reports arch, ctx, vocab, key metadata.

Replaces the ~250 LOC hand-rolled binary parser in
``augmentum/models/model_profile_cache.py`` for *Coder* reads (the
model_profile_cache module itself stays on its own reader until a
separate migration PR — see project memory: "Powers extension pass").
"""

from __future__ import annotations

from augmentum.coder.analyzers.registry import (
    AnalysisReport,
    register_analyzer,
)


class GGUFAnalyzer:
    name = "gguf"
    extensions = ("gguf",)
    magic_bytes = (b"GGUF",)

    async def analyze(self, path: str, raw: bytes) -> AnalysisReport:
        try:
            from gguf import GGUFReader  # type: ignore
        except ImportError:
            return AnalysisReport(
                format="GGUF (lib missing)",
                summary=(
                    "GGUF model file detected but the `gguf` Python package "
                    "isn't installed. Install via `pip install gguf` to get "
                    "structured metadata."
                ),
            )

        # GGUFReader can read directly from a path; we have bytes in memory
        # for unified dispatch. Write to a BytesIO equivalent if needed,
        # but the lib supports path-based reading which is what most callers
        # have anyway.
        reader = GGUFReader(path)

        fields = reader.fields or {}
        kv = {}
        for key, field in fields.items():
            try:
                parts = field.parts
                value: object
                if not parts:
                    value = ""
                elif len(parts) == 1:
                    value = parts[0].tolist() if hasattr(parts[0], "tolist") else parts[0]
                else:
                    last = parts[-1]
                    value = last.tolist() if hasattr(last, "tolist") else last
                # Trim very long arrays / nested structures for the summary
                if isinstance(value, list) and len(value) > 6:
                    value = f"[{len(value)} items]"
                kv[key] = value
            except Exception:
                kv[key] = "<unreadable>"

        arch = str(kv.get("general.architecture", "unknown"))
        name = str(kv.get("general.name", path.rsplit("/", 1)[-1]))
        ctx = kv.get(f"{arch}.context_length") or kv.get("llama.context_length")
        vocab_size = kv.get(f"{arch}.vocab_size") or kv.get("tokenizer.ggml.token_type")
        embed = kv.get(f"{arch}.embedding_length")
        block_count = kv.get(f"{arch}.block_count")
        rope_freq = kv.get(f"{arch}.rope.freq_base")
        tensor_count = len(reader.tensors) if hasattr(reader, "tensors") else 0

        size_mb = len(raw) / (1024 * 1024) if raw else 0
        if not raw:
            try:
                import os
                size_mb = os.path.getsize(path) / (1024 * 1024)
            except OSError:
                size_mb = 0

        bullets = [
            f"- Architecture: {arch}",
            f"- Name: {name}",
            f"- File size: {size_mb:,.1f} MB",
            f"- Tensors: {tensor_count}",
        ]
        if ctx is not None:
            bullets.append(f"- Context length: {ctx}")
        if embed is not None:
            bullets.append(f"- Embedding dim: {embed}")
        if block_count is not None:
            bullets.append(f"- Transformer blocks: {block_count}")
        if rope_freq is not None:
            bullets.append(f"- RoPE freq base: {rope_freq}")
        if vocab_size is not None:
            bullets.append(f"- Vocab size: {vocab_size}")

        summary = (
            f"GGUF model file: **{name}**\n\n"
            + "\n".join(bullets)
            + "\n\nFor any other metadata field, call "
            "`analyze_file(path)` and inspect the `details` block."
        )

        return AnalysisReport(
            format="GGUF model",
            summary=summary,
            details={
                "architecture": arch,
                "name": name,
                "context_length": ctx,
                "embedding_dim": embed,
                "block_count": block_count,
                "vocab_size": vocab_size,
                "tensor_count": tensor_count,
                "rope_freq_base": rope_freq,
                "size_mb": round(size_mb, 1),
                "metadata_keys": sorted(kv.keys())[:80],
            },
            raw_size_bytes=int(size_mb * 1024 * 1024),
        )


register_analyzer(GGUFAnalyzer())
