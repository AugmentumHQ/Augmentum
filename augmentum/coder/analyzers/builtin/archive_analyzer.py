"""Archive analyzer — zip / tar / tgz / 7z. Lists structure without extracting."""

from __future__ import annotations

import os
import tarfile
import zipfile
from collections import Counter

from augmentum.coder.analyzers.registry import (
    AnalysisReport,
    register_analyzer,
)


def _ext_dist(names: list[str]) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    for name in names:
        _, ext = os.path.splitext(name)
        counts[(ext or "(no-ext)").lower()] += 1
    return counts.most_common(8)


def _build_summary(format_label: str, names: list[str], sizes: list[int], total_size: int) -> str:
    file_count = len(names)
    largest_idx = max(range(len(sizes)), key=lambda i: sizes[i]) if sizes else -1
    bullets = [
        f"- Format: {format_label}",
        f"- Entries: {file_count}",
        f"- Total uncompressed size: {total_size / (1024 * 1024):,.2f} MB",
    ]
    if largest_idx >= 0:
        bullets.append(
            f"- Largest entry: {names[largest_idx]} "
            f"({sizes[largest_idx] / (1024 * 1024):,.2f} MB)"
        )

    dist = _ext_dist(names)
    dist_block = ""
    if dist:
        dist_block = "\n\n**File-type breakdown:**\n" + "\n".join(
            f"  - {ext}: {count}" for ext, count in dist
        )

    preview = names[:20]
    preview_block = "\n\n**First entries:**\n" + "\n".join(
        f"  - {n}" for n in preview
    )
    if file_count > 20:
        preview_block += f"\n  …and {file_count - 20} more"

    return (
        f"{format_label}\n\n" + "\n".join(bullets) + dist_block + preview_block
    )


class ZipAnalyzer:
    name = "zip"
    extensions = ("zip", "cbz", "epub", "apk", "jar", "war", "ipa")
    magic_bytes = (b"PK\x03\x04",)

    async def analyze(self, path: str, raw: bytes) -> AnalysisReport:
        names: list[str] = []
        sizes: list[int] = []
        try:
            with zipfile.ZipFile(path) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    names.append(info.filename)
                    sizes.append(info.file_size)
        except zipfile.BadZipFile as exc:
            return AnalysisReport(
                format="zip (corrupt)",
                summary=f"File has zip magic but couldn't be opened: {exc}",
            )

        total = sum(sizes)
        return AnalysisReport(
            format="ZIP archive",
            summary=_build_summary("ZIP archive", names, sizes, total),
            details={
                "entry_count": len(names),
                "total_uncompressed_bytes": total,
                "extension_distribution": dict(_ext_dist(names)),
                "entries_preview": names[:40],
            },
            raw_size_bytes=len(raw),
        )


class TarAnalyzer:
    name = "tar"
    extensions = ("tar", "tgz", "tar.gz", "tbz2", "tar.bz2", "txz", "tar.xz")
    # Tar files don't have a fixed magic at byte 0; ustar marker is at 257.
    # We rely on extension dispatch + try-open semantics.
    magic_bytes = ()

    async def analyze(self, path: str, raw: bytes) -> AnalysisReport:
        names: list[str] = []
        sizes: list[int] = []
        try:
            with tarfile.open(path) as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    names.append(member.name)
                    sizes.append(member.size)
        except (tarfile.TarError, OSError) as exc:
            return AnalysisReport(
                format="tar (unreadable)",
                summary=f"Tar archive couldn't be opened: {exc}",
            )

        total = sum(sizes)
        return AnalysisReport(
            format="TAR archive",
            summary=_build_summary("TAR archive", names, sizes, total),
            details={
                "entry_count": len(names),
                "total_uncompressed_bytes": total,
                "extension_distribution": dict(_ext_dist(names)),
                "entries_preview": names[:40],
            },
            raw_size_bytes=len(raw),
        )


register_analyzer(ZipAnalyzer())
register_analyzer(TarAnalyzer())
