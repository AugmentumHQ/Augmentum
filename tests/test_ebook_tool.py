"""Tests for the EPUB storybook renderer."""
from __future__ import annotations

import io
import zipfile

import pytest


def _make_chapters(n: int = 3, with_images: bool = False) -> list[dict]:
    """Build N sample chapters for testing."""
    chapters = []
    for i in range(1, n + 1):
        ch = {
            "heading": f"Chapter {i}: Test Scene {i}",
            "body": f"Once upon a time in chapter {i}. " * 20,
        }
        if with_images:
            ch["_image_path"] = ""
        chapters.append(ch)
    return chapters


class TestEpubRenderer:
    def test_basic_epub_structure(self):
        from augmentum.tools.artifact_ebook import _render_epub
        chapters = _make_chapters(3)
        data = _render_epub("Test Story", "Test Author", chapters)
        assert isinstance(data, bytes)
        assert len(data) > 0
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            assert names[0] == "mimetype"
            assert zf.read("mimetype") == b"application/epub+zip"
            assert "META-INF/container.xml" in names
            assert "OEBPS/content.opf" in names
            assert "OEBPS/toc.ncx" in names
            assert "OEBPS/toc.xhtml" in names
            assert "OEBPS/style.css" in names
            assert "OEBPS/cover.xhtml" in names
            assert "OEBPS/title.xhtml" in names
            assert "OEBPS/chapter-01.xhtml" in names
            assert "OEBPS/chapter-02.xhtml" in names
            assert "OEBPS/chapter-03.xhtml" in names

    def test_mimetype_not_compressed(self):
        from augmentum.tools.artifact_ebook import _render_epub
        chapters = _make_chapters(1)
        data = _render_epub("Test", "", chapters)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            info = zf.getinfo("mimetype")
            assert info.compress_type == zipfile.ZIP_STORED

    def test_chapter_content_has_drop_cap(self):
        from augmentum.tools.artifact_ebook import _render_epub
        chapters = [{"heading": "The Beginning", "body": "Once upon a time."}]
        data = _render_epub("Story", "", chapters)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            ch1 = zf.read("OEBPS/chapter-01.xhtml").decode()
            assert 'class="first"' in ch1

    def test_chapter_number_words(self):
        from augmentum.tools.artifact_ebook import _render_epub
        chapters = _make_chapters(3)
        data = _render_epub("Story", "", chapters)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            ch1 = zf.read("OEBPS/chapter-01.xhtml").decode()
            ch2 = zf.read("OEBPS/chapter-02.xhtml").decode()
            assert "Chapter One" in ch1
            assert "Chapter Two" in ch2

    def test_alternating_image_placement(self):
        from augmentum.tools.artifact_ebook import _render_epub
        chapters = _make_chapters(2)
        data = _render_epub("Story", "", chapters)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            ch1 = zf.read("OEBPS/chapter-01.xhtml").decode()
            assert "illustration-" not in ch1

    def test_cover_fallback_without_image(self):
        from augmentum.tools.artifact_ebook import _render_epub
        chapters = _make_chapters(1)
        data = _render_epub("My Story", "Author", chapters, cover_image_path=None)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            cover = zf.read("OEBPS/cover.xhtml").decode()
            assert "My Story" in cover

    def test_opf_manifest_lists_all_files(self):
        from augmentum.tools.artifact_ebook import _render_epub
        chapters = _make_chapters(2)
        data = _render_epub("Story", "", chapters)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            opf = zf.read("OEBPS/content.opf").decode()
            assert "chapter-01.xhtml" in opf
            assert "chapter-02.xhtml" in opf
            assert "style.css" in opf
            assert "cover.xhtml" in opf
            assert "title.xhtml" in opf
            assert "toc.ncx" in opf

    def test_single_chapter(self):
        from augmentum.tools.artifact_ebook import _render_epub
        chapters = [{"heading": "Only Chapter", "body": "Short story."}]
        data = _render_epub("Solo", "", chapters)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            assert "OEBPS/chapter-01.xhtml" in names
            assert "OEBPS/chapter-02.xhtml" not in names

    def test_empty_chapters(self):
        from augmentum.tools.artifact_ebook import _render_epub
        data = _render_epub("Empty", "", [])
        assert isinstance(data, bytes)

    def test_markdown_in_body(self):
        from augmentum.tools.artifact_ebook import _render_epub
        chapters = [{"heading": "Test", "body": "This is **bold** and *italic*."}]
        data = _render_epub("MD Test", "", chapters)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            ch1 = zf.read("OEBPS/chapter-01.xhtml").decode()
            assert "<strong>" in ch1 or "<b>" in ch1 or "**" not in ch1

    def test_default_theme_is_storybook(self):
        from augmentum.tools.artifact_ebook import _STORYBOOK_CSS, _build_epub_css
        # Empty, blank, the 'storybook' sentinel, and unknown names all map
        # to the warm-parchment default.
        for name in ("", "  ", "storybook", "STORYBOOK", "not-a-theme"):
            assert _build_epub_css(name) == _STORYBOOK_CSS

    def test_light_reading_theme_recolors_page(self):
        from augmentum.tools.artifact_ebook import (
            _EPUB_THEMES,
            _STORYBOOK_CSS,
            _build_epub_css,
        )
        css = _build_epub_css("sepia")
        t = _EPUB_THEMES["sepia"]
        assert css != _STORYBOOK_CSS
        assert css.startswith("/* Augmentum EPUB — sepia theme */")
        # storybook's parchment/brown literals replaced by the theme's palette
        assert "#faf9f6" not in css and "#2c1810" not in css
        assert t["bg"] in css and t["fg"] in css and t["accent"] in css
        # light theme keeps a (recoloured) prefers-color-scheme block
        assert "@media (prefers-color-scheme: dark)" in css

    def test_dark_reading_theme_drops_media_block(self):
        from augmentum.tools.artifact_ebook import _EPUB_THEMES, _build_epub_css
        css = _build_epub_css("night")
        assert css.startswith("/* Augmentum EPUB — night theme */")
        assert _EPUB_THEMES["night"]["bg"] in css
        # the base palette IS dark, so the override would only fight it
        assert "@media (prefers-color-scheme: dark)" not in css
        # 'night' is a sans theme — Georgia is gone
        assert 'Georgia, "Times New Roman", serif' not in css

    def test_theme_name_threads_into_epub(self):
        from augmentum.tools.artifact_ebook import _render_epub
        chapters = [{"heading": "Ch1", "body": "Hello world. More text."}]
        data = _render_epub("Themed", "", chapters, theme_name="midnight")
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            css = zf.read("OEBPS/style.css").decode()
            assert "midnight theme" in css
            assert "#0e1320" in css  # midnight page background

    def test_reading_settings_defaults_are_noop(self):
        from augmentum.tools.artifact_ebook import _STORYBOOK_CSS, _build_epub_css
        # Missing / default keys must not change a single byte.
        for reading in (None, {}, {"font": "", "size": "md", "leading": "normal"}):
            assert _build_epub_css("", reading) == _STORYBOOK_CSS
        assert "/* reading settings */" not in _build_epub_css("", {"font": ""})

    def test_reading_settings_append_overrides(self):
        from augmentum.tools.artifact_ebook import _build_epub_css
        css = _build_epub_css("sepia", {"font": "sans", "size": "xl", "leading": "relaxed"})
        block = css.split("/* reading settings */")[-1]
        # font-family goes on body; size/leading go on p/li/blockquote so they
        # beat the preview shell's `.serif p` rule.
        assert "body{font-family:" in block and "sans-serif" in block
        assert "body p,body li,body blockquote{" in block
        assert "font-size:1.45rem" in block
        assert "line-height:2.05" in block
        # bogus values (and the 'md'/'normal' defaults) are ignored
        assert "/* reading settings */" not in _build_epub_css("", {"font": "comic", "size": "huge", "leading": "loose"})
        assert "/* reading settings */" not in _build_epub_css("", {"size": "md", "leading": "normal"})

    def test_reading_settings_thread_into_epub(self):
        from augmentum.tools.artifact_ebook import _render_epub
        data = _render_epub(
            "R", "", [{"heading": "C1", "body": "Words."}],
            theme_name="paper", reading={"size": "lg"},
        )
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            css = zf.read("OEBPS/style.css").decode()
            assert "paper theme" in css and "/* reading settings */" in css
            assert "body p,body li,body blockquote{font-size:1.2rem}" in css

    def test_preview_floods_page_with_book_bg(self):
        from augmentum.proxy.artifact_routes import _epub_to_html
        from augmentum.tools.artifact_ebook import _render_epub
        data = _render_epub("Bk", "", [{"heading": "C1", "body": "Words and words."}], theme_name="night")
        p = tmp = __import__("tempfile").mktemp(suffix=".epub")
        __import__("pathlib").Path(p).write_bytes(data)
        try:
            html = _epub_to_html(p, "Bk", "/dl")
        finally:
            __import__("os").unlink(p)
        assert html and "html,body{background:#16181c}" in html
        # the device-dark-mode block must not survive into the preview
        assert "@media (prefers-color-scheme: dark)" not in html


import os
import tempfile


class TestEpubWithImages:
    """Test EPUB rendering with actual image files."""

    def _create_test_image(self, path: str, width: int = 200, height: int = 150):
        """Create a minimal test image."""
        from PIL import Image
        img = Image.new("RGB", (width, height), color=(100, 150, 200))
        img.save(path)

    def test_chapter_images_embedded(self, tmp_path):
        from augmentum.tools.artifact_ebook import _render_epub

        img_path = str(tmp_path / "scene.png")
        self._create_test_image(img_path)

        chapters = [
            {
                "heading": "The Garden",
                "body": "Flowers bloomed everywhere.",
                "_image_path": img_path,
                "image_caption": "A beautiful garden",
            },
        ]

        data = _render_epub("Test", "Author", chapters)

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            assert "OEBPS/images/ch01.jpg" in names
            img_data = zf.read("OEBPS/images/ch01.jpg")
            assert img_data[:2] == b"\xff\xd8"  # JPEG magic bytes

            ch1 = zf.read("OEBPS/chapter-01.xhtml").decode()
            assert "images/ch01.jpg" in ch1
            assert "A beautiful garden" in ch1
            assert "illustration-right" in ch1

            opf = zf.read("OEBPS/content.opf").decode()
            assert "img-ch01" in opf

    def test_cover_image_embedded(self, tmp_path):
        from augmentum.tools.artifact_ebook import _render_epub

        cover_path = str(tmp_path / "cover.png")
        self._create_test_image(cover_path, 400, 600)

        chapters = [{"heading": "Ch1", "body": "Text."}]
        data = _render_epub("My Book", "Me", chapters, cover_image_path=cover_path)

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            assert "OEBPS/images/cover.jpg" in zf.namelist()
            cover_html = zf.read("OEBPS/cover.xhtml").decode()
            assert "images/cover.jpg" in cover_html
            assert "My Book" in cover_html

            opf = zf.read("OEBPS/content.opf").decode()
            assert "cover-image" in opf

    def test_rgba_image_converted(self, tmp_path):
        from PIL import Image
        from augmentum.tools.artifact_ebook import _render_epub

        img_path = str(tmp_path / "rgba.png")
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        img.save(img_path)

        chapters = [{"heading": "Test", "body": "Text.", "_image_path": img_path}]
        data = _render_epub("Test", "", chapters)

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            img_data = zf.read("OEBPS/images/ch01.jpg")
            assert img_data[:2] == b"\xff\xd8"

    def test_even_chapter_floats_left(self, tmp_path):
        from augmentum.tools.artifact_ebook import _render_epub

        img1 = str(tmp_path / "s1.png")
        img2 = str(tmp_path / "s2.png")
        self._create_test_image(img1)
        self._create_test_image(img2)

        chapters = [
            {"heading": "Ch1", "body": "Text one.", "_image_path": img1},
            {"heading": "Ch2", "body": "Text two.", "_image_path": img2},
        ]
        data = _render_epub("Test", "", chapters)

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            ch1 = zf.read("OEBPS/chapter-01.xhtml").decode()
            ch2 = zf.read("OEBPS/chapter-02.xhtml").decode()
            assert "illustration-right" in ch1
            assert "illustration-left" in ch2
