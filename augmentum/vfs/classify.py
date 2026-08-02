"""Kind classification — derive a user-facing category (image/document/audio/
video/archive/code/other) from mime_type + filename extension.

`source` tells us *which backing table* owns a row — used for cascading deletes.
`kind` tells us *what the user sees it as* — used to drive the files-panel tabs.
Until this split, the "artifacts" source was a catch-all: every chart, ebook,
spreadsheet, presentation, downloaded image, and zip that flowed through
ArtifactStore showed up under the Artifacts tab. Classifying by mime instead
lets an image be an image regardless of whether image_generations, artifacts,
or chat_images produced it.
"""

from __future__ import annotations


_IMAGE_EXTS = frozenset({
    "png", "jpg", "jpeg", "gif", "webp", "svg", "bmp", "ico", "tif", "tiff", "avif", "heic",
})
_AUDIO_EXTS = frozenset({
    "mp3", "wav", "ogg", "flac", "m4a", "aac", "opus", "wma",
})
_VIDEO_EXTS = frozenset({
    "mp4", "webm", "mkv", "mov", "avi", "wmv", "flv", "m4v",
})
_DOCUMENT_EXTS = frozenset({
    "pdf", "docx", "doc", "odt", "rtf", "txt", "md", "markdown", "rst",
    "html", "htm", "csv", "tsv", "xlsx", "xls", "ods",
    "pptx", "ppt", "odp", "key",
    "epub", "mobi", "azw3",
    "json", "xml", "yaml", "yml", "toml", "log",
})
_ARCHIVE_EXTS = frozenset({
    "zip", "tar", "gz", "tgz", "bz2", "7z", "rar", "xz",
})
# Comic-book archive formats — semantically distinct from generic archives
# so the Files panel / Library surface can route them to the comic reader
# instead of the zip viewer. See augmentum/media/providers/komga.py +
# suwayomi.py for the upstream mime types we expect.
_COMIC_EXTS = frozenset({"cbz", "cbr", "cbt", "cb7"})
_COMIC_MIMES = frozenset({
    "application/vnd.comicbook+zip",
    "application/vnd.comicbook-rar",
    "application/x-cbr",
    "application/x-cbz",
})
_CODE_EXTS = frozenset({
    "py", "js", "ts", "jsx", "tsx", "vue", "svelte",
    "rs", "go", "java", "kt", "swift", "cs", "cpp", "cc", "c", "h", "hpp",
    "rb", "php", "sh", "bash", "zsh", "fish",
    "sql", "css", "scss", "less",
    "dockerfile", "makefile",
})


def _ext_of(name: str) -> str:
    if not name:
        return ""
    dot = name.rfind(".")
    if dot < 1 or dot == len(name) - 1:
        # "Dockerfile" / "Makefile" — no dot but matchable by full lowercase name
        low = name.lower()
        if low in _CODE_EXTS:
            return low
        return ""
    return name[dot + 1:].lower()


def derive_kind(mime_type: str | None, name: str | None) -> str:
    """Return one of: image, document, audio, video, archive, code, other.

    Prefers mime_type when it starts with a well-known top-level, falls back to
    the filename extension. Stays stable: don't rename these tokens without
    updating the SQL backfill and the UI tabs.
    """
    mime = (mime_type or "").lower()

    # Comic archives take precedence over generic archive classification —
    # the mime types below LOOK like zip/rar subtypes but the Library
    # surface + reader specifically handle them. Check before the image/
    # audio/video branches because none of those overlap.
    if mime in _COMIC_MIMES:
        return "comic"

    if mime.startswith("image/"):
        return "image"
    if mime.startswith("audio/"):
        return "audio"
    if mime.startswith("video/"):
        return "video"

    # Specific application/* mimes we treat as documents
    if mime in {
        "application/pdf",
        "application/epub+zip",
        "application/json",
        "application/xml",
        "application/x-yaml",
        "application/yaml",
        "application/msword",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
    }:
        return "document"
    if "officedocument" in mime or "opendocument" in mime:
        return "document"
    if mime.startswith("text/"):
        return "document"

    # Archives
    if mime == "application/zip" or "x-tar" in mime or "gzip" in mime or "x-rar" in mime or "7z" in mime:
        return "archive"

    # Fall back to filename extension (covers octet-stream / empty mimes).
    # Comic extensions beat the generic archive branch — a ``.cbz`` is a
    # zip, but the user is going to read it, not extract files from it.
    ext = _ext_of(name or "")
    if ext in _COMIC_EXTS:    return "comic"
    if ext in _IMAGE_EXTS:    return "image"
    if ext in _AUDIO_EXTS:    return "audio"
    if ext in _VIDEO_EXTS:    return "video"
    if ext in _ARCHIVE_EXTS:  return "archive"
    if ext in _DOCUMENT_EXTS: return "document"
    if ext in _CODE_EXTS:     return "code"

    return "other"
