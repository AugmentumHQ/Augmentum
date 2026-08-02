"""Static-file preview registry.

Single source of truth for which workspace files the UI knows how to preview
without a running dev server, what MIME type to serve them under, and how to
transform the bytes before they reach the iframe.

The registry is consumed by two callers:

- ``GET /api/coder/preview-file/{ws}/{path:path}`` (coder_routes.py) —
  looks up by extension, calls the renderer, returns the bytes with the
  right Content-Type and headers.
- ``GET /api/coder/preview-types`` (coder_routes.py) — returns the list of
  supported extensions so the frontend can decide whether to show the
  Preview context-menu item without maintaining a parallel list.

Adding a new previewable type is a one-line change to ``_TYPES`` below
plus a renderer if it needs transformation. The frontend picks it up
automatically on next page load.
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from dataclasses import dataclass

# Renderer signature: (raw_bytes, base_href_or_empty) -> bytes_to_serve.
# base_href is the URL prefix the iframe needs so relative paths inside
# the file resolve through the same preview-file route. Renderers that
# don't emit HTML ignore it. Returning the input unchanged is the
# expected behavior for binary types.
Renderer = Callable[[bytes, str], bytes]


@dataclass(frozen=True)
class PreviewType:
    """One previewable category.

    ``extensions`` are lowercase, leading-dot form (``.html``, ``.md``).
    ``media_type`` is what we serve in Content-Type after rendering.
    ``kind`` is a short label the UI uses to pick an icon / heading.
    ``renderer`` transforms bytes; for binary types this is just the
    passthrough.
    """

    extensions: tuple[str, ...]
    media_type: str
    kind: str
    renderer: Renderer


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _passthrough(data: bytes, _base_href: str) -> bytes:
    return data


def _html_with_base(data: bytes, base_href: str) -> bytes:
    """Inject ``<base href>`` into HTML so root-relative + relative asset
    paths resolve through the same preview-file route as the parent.

    Strategy: insert the tag right after ``<head>`` if present, otherwise
    after ``<html...>``, otherwise at the very top. Idempotent — if a
    ``<base>`` tag already exists we leave the document alone (the
    author opted out).

    We intentionally do NOT try to rewrite every ``href=/foo`` to a
    proxy-prefixed form (that's the heavy machinery the dev-server
    proxy uses for server-emitted absolute URLs). For static files the
    author wrote, ``<base href>`` covers the common case at zero cost.
    """
    if not base_href:
        return data
    # Decode tolerantly for the injection scan; bytes get sliced back
    # together so non-UTF8 binary blobs (extremely unlikely in an HTML
    # file) survive without re-encoding errors.
    try:
        text = data.decode("utf-8", errors="replace")
    except Exception:
        return data
    lower = text.lower()
    if "<base" in lower:
        return data
    base_tag = f'<base href="{html.escape(base_href, quote=True)}">'
    # Insert after <head ...> if found.
    head_pos = lower.find("<head")
    if head_pos >= 0:
        gt = text.find(">", head_pos)
        if gt >= 0:
            return (text[: gt + 1] + base_tag + text[gt + 1 :]).encode("utf-8")
    # Fallback: after <html ...> tag.
    html_pos = lower.find("<html")
    if html_pos >= 0:
        gt = text.find(">", html_pos)
        if gt >= 0:
            return (text[: gt + 1] + base_tag + text[gt + 1 :]).encode("utf-8")
    # No structural anchors — prepend at the very top.
    return (base_tag + text).encode("utf-8")


_MARKDOWN_CSS = """
:root { color-scheme: light dark; }
body {
  font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  max-width: 880px; margin: 2rem auto; padding: 0 1.5rem;
  color: #1a1a1a; background: #fafafa;
}
@media (prefers-color-scheme: dark) {
  body { color: #e4e4e4; background: #1a1a1a; }
  a { color: #6ea8ff; }
  code, pre { background: #2a2a2a; }
  blockquote { border-left-color: #444; color: #aaa; }
  hr { border-color: #333; }
  table th, table td { border-color: #333; }
}
h1, h2, h3, h4 { line-height: 1.25; margin: 1.5em 0 .6em; }
h1 { font-size: 2em; border-bottom: 1px solid #ddd; padding-bottom: .3em; }
h2 { font-size: 1.5em; border-bottom: 1px solid #eee; padding-bottom: .3em; }
p, ul, ol, blockquote { margin: 0 0 1em; }
ul, ol { padding-left: 1.6em; }
li > p { margin: 0; }
code { font-family: SFMono-Regular, Menlo, Consolas, monospace; font-size: .9em;
  background: #f0f0f0; padding: .15em .35em; border-radius: 3px; }
pre { background: #f0f0f0; padding: 1em; border-radius: 6px; overflow-x: auto; }
pre code { background: transparent; padding: 0; font-size: .92em; }
blockquote { border-left: 4px solid #ddd; color: #555; margin: 0 0 1em;
  padding: 0 1em; }
table { border-collapse: collapse; margin: 0 0 1em; }
table th, table td { border: 1px solid #ddd; padding: .4em .6em; text-align: left; }
img { max-width: 100%; height: auto; }
hr { border: 0; border-top: 1px solid #ddd; margin: 2em 0; }
"""


def _markdown_render(data: bytes, base_href: str) -> bytes:
    """Render markdown → standalone HTML page using markdown-it (already
    a dependency for the document/ebook artifact tools).

    Falls back to a ``<pre>``-wrapped view if the library isn't available
    in the runtime environment — the user still sees the file rather than
    an opaque error.

    Includes a compact built-in stylesheet (no external CSS fetch) so the
    page is readable without the user having to wire one up. Honors
    prefers-color-scheme.
    """
    try:
        from markdown_it import MarkdownIt

        md = MarkdownIt("commonmark", {"html": False, "linkify": True, "typographer": True})
        body_html = md.render(data.decode("utf-8", errors="replace"))
    except Exception:
        body_html = "<pre>" + html.escape(data.decode("utf-8", errors="replace")) + "</pre>"
    base_tag = f'<base href="{html.escape(base_href, quote=True)}">' if base_href else ""
    doc = (
        "<!doctype html><html><head>"
        '<meta charset="utf-8">'
        f"{base_tag}"
        f"<style>{_MARKDOWN_CSS}</style>"
        "</head><body>"
        f"{body_html}"
        "</body></html>"
    )
    return doc.encode("utf-8")


_JSON_CSS = """
:root { color-scheme: light dark; }
body { font: 14px/1.5 SFMono-Regular, Menlo, Consolas, monospace;
  margin: 0; padding: 1.25rem 1.5rem; background: #fafafa; color: #1a1a1a; }
@media (prefers-color-scheme: dark) {
  body { background: #1a1a1a; color: #e4e4e4; }
  .err { color: #ff8484; }
}
pre { margin: 0; white-space: pre-wrap; word-break: break-word; }
.err { color: #c00; }
"""


def _json_pretty(data: bytes, _base_href: str) -> bytes:
    """Pretty-print JSON in a styled HTML wrapper.

    Invalid JSON falls through to the raw text with an error banner — useful
    for diagnosing "why won't this parse" without bouncing out to the editor.
    """
    text = data.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
        pretty = json.dumps(parsed, indent=2, ensure_ascii=False)
        body = f"<pre>{html.escape(pretty)}</pre>"
    except Exception as exc:
        body = (
            f'<pre class="err">JSON parse failed: {html.escape(str(exc))}</pre>'
            f"<pre>{html.escape(text)}</pre>"
        )
    doc = (
        "<!doctype html><html><head>"
        '<meta charset="utf-8">'
        f"<style>{_JSON_CSS}</style>"
        "</head><body>"
        f"{body}"
        "</body></html>"
    )
    return doc.encode("utf-8")


_TEXT_CSS = """
:root { color-scheme: light dark; }
body { font: 13px/1.5 SFMono-Regular, Menlo, Consolas, monospace;
  margin: 0; padding: 1.25rem 1.5rem; background: #fafafa; color: #1a1a1a; }
@media (prefers-color-scheme: dark) {
  body { background: #1a1a1a; color: #e4e4e4; }
}
pre { margin: 0; white-space: pre-wrap; word-break: break-word; }
"""


def _text_wrap(data: bytes, _base_href: str) -> bytes:
    """Wrap arbitrary text in a styled, scrollable HTML page.

    For logs and free-form text where the user wants something readable
    without trapping the rendering in the editor pane.
    """
    text = data.decode("utf-8", errors="replace")
    doc = (
        "<!doctype html><html><head>"
        '<meta charset="utf-8">'
        f"<style>{_TEXT_CSS}</style>"
        "</head><body>"
        f"<pre>{html.escape(text)}</pre>"
        "</body></html>"
    )
    return doc.encode("utf-8")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Adding a type:
# 1. Add a PreviewType entry below.
# 2. If it needs transformation (e.g. JSON pretty-print), add a renderer
#    above and reference it; otherwise use _passthrough.
# 3. That's it — the route + the frontend pick it up via _by_ext() /
#    list_extensions() automatically.
# ---------------------------------------------------------------------------

_TYPES: tuple[PreviewType, ...] = (
    # HTML — the headline use case. Renderer injects <base href> so
    # author-relative paths resolve through the same proxy route.
    PreviewType((".html", ".htm"), "text/html; charset=utf-8", "html", _html_with_base),
    # Markdown — render to HTML with a readable stylesheet.
    PreviewType((".md", ".markdown"), "text/html; charset=utf-8", "markdown", _markdown_render),
    # SVG — image, but also text-shaped. Serve as image/svg+xml so the
    # browser renders it; <base href> handled by passthrough (SVG
    # rarely needs it).
    PreviewType((".svg",), "image/svg+xml", "svg", _passthrough),
    # Raster images — browser-native rendering.
    PreviewType((".png",), "image/png", "image", _passthrough),
    PreviewType((".jpg", ".jpeg"), "image/jpeg", "image", _passthrough),
    PreviewType((".gif",), "image/gif", "image", _passthrough),
    PreviewType((".webp",), "image/webp", "image", _passthrough),
    PreviewType((".bmp",), "image/bmp", "image", _passthrough),
    PreviewType((".ico",), "image/x-icon", "image", _passthrough),
    PreviewType((".avif",), "image/avif", "image", _passthrough),
    # PDF — browser-native PDF viewer.
    PreviewType((".pdf",), "application/pdf", "pdf", _passthrough),
    # JSON — pretty-printed in a styled wrapper. Invalid JSON shows
    # the parse error inline rather than failing the request.
    PreviewType((".json",), "text/html; charset=utf-8", "json", _json_pretty),
    # Plain text / logs — styled, scrollable, mobile-readable.
    PreviewType((".txt", ".log"), "text/html; charset=utf-8", "text", _text_wrap),
    # Audio — browser-native <audio>-compatible MIME types.
    PreviewType((".mp3",), "audio/mpeg", "audio", _passthrough),
    PreviewType((".wav",), "audio/wav", "audio", _passthrough),
    PreviewType((".ogg",), "audio/ogg", "audio", _passthrough),
    PreviewType((".m4a",), "audio/mp4", "audio", _passthrough),
    PreviewType((".flac",), "audio/flac", "audio", _passthrough),
    # Video — browser-native <video>-compatible MIME types.
    PreviewType((".mp4",), "video/mp4", "video", _passthrough),
    PreviewType((".webm",), "video/webm", "video", _passthrough),
    PreviewType((".ogv",), "video/ogg", "video", _passthrough),
    PreviewType((".mov",), "video/quicktime", "video", _passthrough),
    # CSS / JS — sometimes useful to inspect served as-is (a <base href>
    # in the parent HTML will pull these via the same route). Plain text
    # MIME so the browser doesn't try to execute when opened standalone.
    PreviewType((".css",), "text/css; charset=utf-8", "code", _passthrough),
    PreviewType((".js", ".mjs"), "text/javascript; charset=utf-8", "code", _passthrough),
)


# Built once at import — extension lookup is hot, doing it linearly is fine
# given the registry is ~25 entries, but a dict is friendlier to read.
_EXT_INDEX: dict[str, PreviewType] = {
    ext: t for t in _TYPES for ext in t.extensions
}


def by_extension(ext: str) -> PreviewType | None:
    """Return the PreviewType for a leading-dot lowercase extension, or None.

    Callers should normalize: ``ext = ('.' + tail).lower()`` where ``tail``
    is the suffix after the last ``.``. Returns None for unregistered
    extensions; the route layer turns that into 415.
    """
    return _EXT_INDEX.get(ext)


def extension_for_path(path: str) -> str:
    """Lowercase, leading-dot extension for a path. Empty string for none."""
    if "." not in path:
        return ""
    tail = path.rsplit(".", 1)[-1].lower()
    if not tail or "/" in tail:
        return ""
    return "." + tail


def list_extensions() -> list[str]:
    """All registered extensions, sorted — for the discovery endpoint."""
    return sorted(_EXT_INDEX.keys())


def extensions_by_kind() -> dict[str, list[str]]:
    """Group extensions by ``kind`` so the UI can show grouped iconography."""
    out: dict[str, list[str]] = {}
    for t in _TYPES:
        out.setdefault(t.kind, []).extend(t.extensions)
    for k in out:
        out[k] = sorted(out[k])
    return out


def render(path: str, data: bytes, base_href: str) -> tuple[bytes, str] | None:
    """One-shot lookup-and-render. Returns (rendered_bytes, media_type) or
    None when the extension isn't registered."""
    ext = extension_for_path(path)
    if not ext:
        return None
    t = _EXT_INDEX.get(ext)
    if t is None:
        return None
    return t.renderer(data, base_href), t.media_type
