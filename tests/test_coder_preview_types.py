"""Tests for the static-file preview type registry.

The registry (augmentum/coder/preview_types.py) is the single source of
truth for which workspace files the UI can render in the preview iframe
without a running dev server. Tests cover:

- Extension lookup (normalization, unknown types)
- HTML <base href> injection invariants (idempotent, placement)
- Markdown rendering produces real HTML
- JSON pretty-print + invalid-JSON fallback
- Text/log wrapper produces a styled HTML page
- Binary passthrough leaves bytes unchanged
- Path validation rejects traversal + non-/workspace paths (via the
  route helper extracted for testability)

The route-level handler (preview_file in coder_routes.py) needs a full
ASGI test harness — covered separately by the live coder route tests.

Run: python -m pytest tests/test_coder_preview_types.py -v
"""

from __future__ import annotations

import json

import pytest

from augmentum.coder.preview_types import (
    by_extension,
    extension_for_path,
    extensions_by_kind,
    list_extensions,
    render,
)


# ---------------------------------------------------------------------------
# Extension lookup
# ---------------------------------------------------------------------------


class TestExtensionLookup:
    def test_known_extensions_resolve(self):
        assert by_extension(".html") is not None
        assert by_extension(".md") is not None
        assert by_extension(".png") is not None
        assert by_extension(".pdf") is not None

    def test_unknown_extension_returns_none(self):
        assert by_extension(".exe") is None
        assert by_extension(".zip") is None
        assert by_extension(".") is None
        assert by_extension("") is None

    def test_case_normalization_is_callers_responsibility(self):
        # Registry stores lowercase; uppercase doesn't match — callers
        # normalize. extension_for_path() does the normalization for them.
        assert by_extension(".HTML") is None
        assert by_extension(".html") is not None

    def test_extension_for_path_normalizes(self):
        assert extension_for_path("/workspace/index.HTML") == ".html"
        assert extension_for_path("/workspace/foo/BAR.MD") == ".md"
        assert extension_for_path("/workspace/no_extension") == ""
        assert extension_for_path("/workspace/.hidden") == ".hidden"

    def test_extension_for_path_ignores_path_dots(self):
        # A trailing slash + dot shouldn't be treated as an extension.
        assert extension_for_path("/workspace/foo/") == ""
        # Dotted dirs (e.g. /workspace/.augmentum/something) — the rightmost
        # dot is what matters; trust the tail extraction.
        assert extension_for_path("/workspace/.augmentum/x.json") == ".json"

    def test_list_extensions_sorted(self):
        exts = list_extensions()
        assert exts == sorted(exts)
        assert len(exts) >= 20  # we ship more than 20 types

    def test_extensions_by_kind_groups(self):
        kinds = extensions_by_kind()
        assert "html" in kinds
        assert "image" in kinds
        assert ".png" in kinds["image"]
        assert ".html" in kinds["html"]
        # Each kind's list is sorted.
        for k, exts in kinds.items():
            assert exts == sorted(exts), f"kind {k!r} not sorted: {exts}"


# ---------------------------------------------------------------------------
# HTML <base href> injection
# ---------------------------------------------------------------------------


class TestHtmlBaseInjection:
    def test_base_href_inserted_after_head(self):
        src = b"<html><head><title>x</title></head><body>hi</body></html>"
        out, mt = render("/workspace/index.html", src, "/api/coder/preview-file/wsid/workspace/")
        assert mt == "text/html; charset=utf-8"
        assert b"<base href=" in out
        # Order: <head> ... <base> ... <title>
        head_pos = out.lower().find(b"<head")
        base_pos = out.lower().find(b"<base")
        title_pos = out.lower().find(b"<title")
        assert head_pos < base_pos < title_pos

    def test_base_href_inserted_after_html_when_no_head(self):
        src = b"<html><body>bare</body></html>"
        out, _ = render("/workspace/x.html", src, "/api/coder/preview-file/wsid/workspace/")
        assert b"<base href=" in out
        # Order: <html> ... <base> ... <body>
        html_pos = out.lower().find(b"<html")
        base_pos = out.lower().find(b"<base")
        body_pos = out.lower().find(b"<body")
        assert html_pos < base_pos < body_pos

    def test_base_href_prepended_when_no_html(self):
        src = b"<p>just a fragment</p>"
        out, _ = render("/workspace/frag.html", src, "/api/coder/preview-file/wsid/workspace/")
        assert out.startswith(b"<base href=")

    def test_existing_base_tag_preserved(self):
        """If the author already declared a <base>, we leave it alone."""
        src = b'<html><head><base href="/x/"><title>z</title></head><body></body></html>'
        out, _ = render("/workspace/x.html", src, "/api/coder/preview-file/wsid/workspace/")
        # Only the original <base> remains; we didn't inject a second one.
        assert out.count(b"<base") == 1

    def test_empty_base_href_skips_injection(self):
        """Renderer signature accepts empty base_href; should be a no-op."""
        src = b"<html><head></head><body>x</body></html>"
        out, _ = render("/workspace/x.html", src, "")
        assert b"<base href=" not in out
        assert out == src  # passthrough when no base to inject

    def test_base_href_attribute_is_escaped(self):
        """A workspace path with `"` or `<` shouldn't break the attribute."""
        # The route never produces such a base_href today (we construct it
        # ourselves), but the renderer treats it as user-supplied and escapes.
        src = b"<html><head></head><body></body></html>"
        out, _ = render("/workspace/x.html", src, '/api/coder/preview-file/wsid/workspace/"evil/')
        # The injected attribute must be HTML-escaped.
        assert b"&quot;evil/" in out
        # Raw quote outside the attribute boundary would break parsing —
        # ensure the unescaped form is absent.
        assert b'"evil/"' not in out.split(b"<base href=", 1)[1].split(b">", 1)[0]


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


class TestMarkdownRender:
    def test_basic_markdown_produces_html(self):
        out, mt = render("/workspace/README.md", b"# Hello\n\nworld", "/base/")
        assert mt == "text/html; charset=utf-8"
        assert b"<h1>" in out
        assert b"Hello" in out
        assert b"<p>world</p>" in out

    def test_doctype_and_charset(self):
        out, _ = render("/workspace/x.md", b"# x", "")
        assert out.startswith(b"<!doctype html>")
        assert b'charset="utf-8"' in out

    def test_base_href_included_when_provided(self):
        out, _ = render("/workspace/sub/x.md", b"# x", "/api/coder/preview-file/wsid/workspace/sub/")
        assert b"<base href=" in out
        assert b"/workspace/sub/" in out

    def test_no_base_tag_when_empty(self):
        out, _ = render("/workspace/x.md", b"# x", "")
        assert b"<base" not in out

    def test_markdown_extension_alias(self):
        out_md, _ = render("/workspace/x.md", b"# x", "")
        out_markdown, _ = render("/workspace/x.markdown", b"# x", "")
        # Both produce <h1>x</h1> wrapped in the same template.
        assert b"<h1>x</h1>" in out_md
        assert b"<h1>x</h1>" in out_markdown


# ---------------------------------------------------------------------------
# JSON renderer
# ---------------------------------------------------------------------------


class TestJsonRender:
    def test_valid_json_pretty_printed(self):
        out, mt = render("/workspace/data.json", b'{"a":1,"b":[2,3]}', "")
        assert mt == "text/html; charset=utf-8"
        # html.escape escapes quotes — check for the escaped form.
        assert b"&quot;a&quot;: 1" in out
        assert b"&quot;b&quot;" in out

    def test_invalid_json_shows_error_inline(self):
        out, _ = render("/workspace/broken.json", b"{not valid", "")
        assert b"JSON parse failed" in out
        # Raw text included so the user sees what was there.
        assert b"{not valid" in out

    def test_unicode_preserved(self):
        out, _ = render("/workspace/u.json", '{"x":"é"}'.encode("utf-8"), "")
        # ensure_ascii=False keeps the é literal in the pretty output.
        assert "é".encode("utf-8") in out


# ---------------------------------------------------------------------------
# Text wrapper
# ---------------------------------------------------------------------------


class TestTextWrap:
    def test_log_file_wrapped_in_pre(self):
        out, mt = render("/workspace/app.log", b"line1\nline2\n", "")
        assert mt == "text/html; charset=utf-8"
        assert b"<pre>" in out
        assert b"line1" in out
        assert b"line2" in out

    def test_html_in_text_is_escaped(self):
        out, _ = render("/workspace/x.txt", b"<script>alert(1)</script>", "")
        # The <script> must be escaped, not rendered.
        assert b"&lt;script&gt;" in out
        assert b"<script>alert" not in out


# ---------------------------------------------------------------------------
# Binary passthrough
# ---------------------------------------------------------------------------


class TestBinaryPassthrough:
    def test_png_bytes_unchanged(self):
        png_magic = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        out, mt = render("/workspace/logo.png", png_magic, "/ignored/")
        assert mt == "image/png"
        assert out == png_magic

    def test_pdf_bytes_unchanged(self):
        pdf_magic = b"%PDF-1.4\n" + b"\x00" * 50
        out, mt = render("/workspace/spec.pdf", pdf_magic, "")
        assert mt == "application/pdf"
        assert out == pdf_magic

    def test_svg_bytes_unchanged(self):
        svg = b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'
        out, mt = render("/workspace/icon.svg", svg, "")
        assert mt == "image/svg+xml"
        assert out == svg

    def test_unknown_extension_returns_none(self):
        # render() returns None for unregistered types — route turns this into 415.
        assert render("/workspace/x.exe", b"MZ", "") is None
        assert render("/workspace/Makefile", b"all:", "") is None


# ---------------------------------------------------------------------------
# Path validation (the route's _validate_preview_path)
# ---------------------------------------------------------------------------
#
# We import the helper directly from coder_routes for unit testing. The
# route handler itself wraps this with the auth / ownership / fs read
# layers — exercised in the route tests, not here.


class TestPathValidation:
    @pytest.fixture
    def validate(self):
        from augmentum.proxy.coder_routes import _validate_preview_path
        return _validate_preview_path

    def test_workspace_relative_path_accepted(self, validate):
        assert validate("/workspace/foo.html") is None
        assert validate("/workspace/sub/dir/bar.png") is None

    def test_bare_workspace_root_rejected(self, validate):
        # `/workspace` (no trailing slash + no file) has nothing to serve.
        assert validate("/workspace") is not None

    def test_outside_workspace_rejected(self, validate):
        assert validate("/etc/passwd") is not None
        assert validate("/usr/local/bin/sh") is not None
        assert validate("/") is not None

    def test_traversal_rejected(self, validate):
        assert validate("/workspace/../etc/passwd") is not None
        assert validate("/workspace/foo/../../bar") is not None

    def test_backslash_rejected(self, validate):
        assert validate("/workspace/..\\etc\\passwd") is not None

    def test_empty_path_rejected(self, validate):
        assert validate("") is not None


# ---------------------------------------------------------------------------
# Base-href URL builder (the route's _preview_base_href)
# ---------------------------------------------------------------------------


class TestBaseHrefBuilder:
    @pytest.fixture
    def build(self):
        from augmentum.proxy.coder_routes import _preview_base_href
        return _preview_base_href

    def test_workspace_root_file(self, build):
        out = build("ws1", "/workspace/index.html")
        assert out == "/api/coder/preview-file/ws1/workspace/"

    def test_nested_file(self, build):
        out = build("ws1", "/workspace/app/dist/index.html")
        assert out == "/api/coder/preview-file/ws1/workspace/app/dist/"

    def test_trailing_slash_present(self, build):
        """Without the trailing slash, <img src='logo.png'> would resolve
        to /api/coder/preview-file/ws1/workspace<no-slash>logo.png — not
        the same prefix. Test pins the invariant explicitly."""
        out = build("ws1", "/workspace/x.html")
        assert out.endswith("/")
