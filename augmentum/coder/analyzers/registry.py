"""Registry + dispatch for file analyzers.

Two lookup keys: file extension and magic-byte prefix. Extension is
fast and covers the common case; magic-bytes is the fallback for
files with stripped or wrong extensions. Both keys can map to the
same analyzer — dispatch picks the most specific match.

Analyzer authors register via ``register_analyzer`` (decorator or
imperative). Each analyzer returns an :class:`AnalysisReport` whose
``summary`` text is what the model actually consumes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class AnalysisReport:
    """Result of running an analyzer against a file.

    ``summary`` is the prose the model consumes via ``file_read``. Keep
    it under ~2000 chars — the whole point is to be cheaper than the
    raw content. ``details`` is structured data for follow-up tool
    calls (``analyze_file(path)`` returns the full report so the model
    can drill in if it needs specific fields).
    """

    format: str                       # human label, e.g. "GGUF model"
    summary: str                      # prose for the model
    details: dict[str, Any] = field(default_factory=dict)
    raw_size_bytes: int = 0
    cache_key: str = ""               # if non-empty, future identical files reuse this report
    truncated: bool = False            # True if the analyzer skipped content for safety


class FileAnalyzer(Protocol):
    """Contract for a file-type analyzer.

    Implementations are async because some analyzers may want to call
    out (model-led generic falls through to the active LLM). Most
    builtins are CPU-bound on a parser lib and can ``return`` directly.
    """

    name: str
    extensions: tuple[str, ...]
    magic_bytes: tuple[bytes, ...]

    async def analyze(self, path: str, raw: bytes) -> AnalysisReport: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_BY_EXT: dict[str, FileAnalyzer] = {}
_BY_MAGIC: list[tuple[bytes, FileAnalyzer]] = []


def register_analyzer(analyzer: FileAnalyzer) -> FileAnalyzer:
    """Register an analyzer for its declared extensions + magic prefixes.

    Idempotent: re-registering the same name replaces the prior entry
    (useful for hot-reload during development). Returns the analyzer
    so it can be used as a decorator.
    """
    for ext in analyzer.extensions:
        normalized = ext.lower().lstrip(".")
        _BY_EXT[normalized] = analyzer
    for magic in analyzer.magic_bytes:
        # Keep longer magic prefixes first so dispatch prefers the most
        # specific match (e.g. "RIFF....WAVE" wins over plain "RIFF").
        _BY_MAGIC.append((magic, analyzer))
        _BY_MAGIC.sort(key=lambda pair: -len(pair[0]))
    log.debug(
        "analyzer_registered",
        name=getattr(analyzer, "name", "?"),
        extensions=list(analyzer.extensions),
        magic_count=len(analyzer.magic_bytes),
    )
    return analyzer


def _resolve(path: str, raw: bytes | None) -> FileAnalyzer | None:
    """Find a matching analyzer by extension, then by magic bytes."""
    _, ext = os.path.splitext(path)
    if ext:
        analyzer = _BY_EXT.get(ext.lower().lstrip("."))
        if analyzer is not None:
            return analyzer
    if raw:
        for magic, analyzer in _BY_MAGIC:
            if raw.startswith(magic):
                return analyzer
    return None


def is_analyzable(path: str, raw: bytes | None = None) -> bool:
    """True if any registered analyzer claims this path."""
    return _resolve(path, raw) is not None


async def analyze_file(
    path: str, raw: bytes, *, raise_on_error: bool = False,
) -> AnalysisReport | None:
    """Run the matching analyzer; return None if no handler matches.

    Errors from the analyzer itself are caught and logged at warning
    level — falling back to None lets the caller decide whether to
    serve raw content, defer to the generic model-led path, or surface
    a user-facing error. Set ``raise_on_error=True`` to bubble.
    """
    analyzer = _resolve(path, raw)
    if analyzer is None:
        return None
    try:
        report = await analyzer.analyze(path, raw)
    except Exception as exc:
        if raise_on_error:
            raise
        log.warning(
            "analyzer_failed",
            analyzer=getattr(analyzer, "name", "?"),
            path=path, error=str(exc)[:200],
        )
        return None
    return report
