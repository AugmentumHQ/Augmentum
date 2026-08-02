"""Audio file analyzer — tags + duration + format via mutagen."""

from __future__ import annotations

from augmentum.coder.analyzers.registry import (
    AnalysisReport,
    register_analyzer,
)


class AudioAnalyzer:
    name = "audio"
    extensions = ("mp3", "flac", "wav", "ogg", "m4a", "aac", "opus", "wma")
    magic_bytes = (
        b"ID3",            # MP3 with ID3v2
        b"fLaC",           # FLAC
        b"RIFF",           # WAV (also AVI, but extension filters)
        b"OggS",           # Ogg / Opus
    )

    async def analyze(self, path: str, raw: bytes) -> AnalysisReport:
        try:
            from mutagen import File as MutagenFile  # type: ignore
        except ImportError:
            return AnalysisReport(
                format="audio (lib missing)",
                summary=(
                    "Audio file detected but the `mutagen` package isn't "
                    "installed. Install via `pip install mutagen`."
                ),
            )

        audio = MutagenFile(path, easy=True)
        if audio is None:
            return AnalysisReport(
                format="audio (unrecognized)",
                summary=f"File extension says audio but mutagen couldn't parse it.",
            )

        info = audio.info
        duration_s = float(getattr(info, "length", 0) or 0)
        sample_rate = getattr(info, "sample_rate", None)
        channels = getattr(info, "channels", None)
        bitrate = getattr(info, "bitrate", None)
        codec = type(audio).__name__.replace("MutagenFile", "")

        tags = dict(audio.tags or {}) if audio.tags is not None else {}
        # mutagen easy returns lists for everything; collapse single-item lists
        def _flatten(v):
            if isinstance(v, list) and len(v) == 1:
                return v[0]
            return v
        tags = {k: _flatten(v) for k, v in tags.items()}

        size_mb = len(raw) / (1024 * 1024) if raw else 0
        mins, secs = divmod(int(duration_s), 60)
        hrs, mins = divmod(mins, 60)
        duration_label = (
            f"{hrs}:{mins:02d}:{secs:02d}" if hrs else f"{mins}:{secs:02d}"
        )

        bullets = [
            f"- Codec: {codec}",
            f"- Duration: {duration_label} ({duration_s:.1f}s)",
            f"- File size: {size_mb:,.2f} MB",
        ]
        if sample_rate:
            bullets.append(f"- Sample rate: {sample_rate} Hz")
        if channels:
            bullets.append(f"- Channels: {channels}")
        if bitrate:
            bullets.append(f"- Bitrate: {bitrate // 1000} kbps")

        tag_lines: list[str] = []
        priority_keys = ("title", "artist", "album", "albumartist", "date",
                         "tracknumber", "genre", "composer")
        for key in priority_keys:
            if key in tags:
                tag_lines.append(f"  - {key}: {tags[key]}")
        remaining = [k for k in tags if k not in priority_keys][:10]
        for key in remaining:
            tag_lines.append(f"  - {key}: {tags[key]}")

        sections = ["Audio file\n", "\n".join(bullets)]
        if tag_lines:
            sections.append("\n**Tags:**\n" + "\n".join(tag_lines))

        return AnalysisReport(
            format=f"audio/{codec.lower()}",
            summary="\n".join(sections),
            details={
                "codec": codec,
                "duration_s": duration_s,
                "sample_rate": sample_rate,
                "channels": channels,
                "bitrate_bps": bitrate,
                "tags": tags,
                "size_mb": round(size_mb, 2),
            },
            raw_size_bytes=len(raw),
        )


register_analyzer(AudioAnalyzer())
