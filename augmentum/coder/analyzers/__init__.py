"""File analyzer substrate for Coder.

Mirrors the existing image-captioner pattern (``vision/router.py`` +
``jobs/handlers/file_caption.py``) but for non-vision file types that
the chat model can't read raw without poisoning its own context — large
binary formats (GGUF, GLB/VRM, safetensors), structurally dense docs
(PDF, DOCX, EPUB), and high-volume data dumps (SQLite, archives).

The model never sees raw bytes for registered types. ``file_read`` on a
``.vrm`` returns ``"40 accessors covering vertex/index/skin data; mesh
has 33,720 verts; 17 blendshape groups; 1 humanoid skin"`` instead of a
two-thousand-line hex preview that would degrade decoding quality.

Unknown extensions fall through to a model-led generic analyzer that
writes a small Python parser, runs it, captures the structured output,
and caches the parser at ``.augmentum/analyzers/<ext>.py`` for the
workspace. Future reads of the same type reuse the cached parser
without round-tripping to the model.
"""

from __future__ import annotations

# Import builtin handlers so their @register_analyzer side-effects run. These
# names are intentionally unused — the import IS the registration.
from augmentum.coder.analyzers.builtin import (  # noqa: F401
    archive_analyzer as _archive_analyzer,
)
from augmentum.coder.analyzers.builtin import (
    audio_analyzer as _audio_analyzer,
)
from augmentum.coder.analyzers.builtin import (
    gguf_analyzer as _gguf_analyzer,
)
from augmentum.coder.analyzers.builtin import (
    gltf_analyzer as _gltf_analyzer,
)
from augmentum.coder.analyzers.builtin import (
    safetensors_analyzer as _safetensors_analyzer,
)
from augmentum.coder.analyzers.builtin import (
    sqlite_analyzer as _sqlite_analyzer,
)
from augmentum.coder.analyzers.registry import (
    AnalysisReport,
    FileAnalyzer,
    analyze_file,
    is_analyzable,
    register_analyzer,
)

__all__ = [
    "AnalysisReport",
    "FileAnalyzer",
    "analyze_file",
    "is_analyzable",
    "register_analyzer",
]
