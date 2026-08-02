"""Tests for augmentum.vfs.epub_extractor — metadata + cover extraction."""

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

from augmentum.vfs.epub_extractor import extract

_CONTAINER_XML = (
    '<?xml version="1.0"?>'
    '<container version="1.0" '
    'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
    '<rootfiles><rootfile full-path="OEBPS/content.opf" '
    'media-type="application/oebps-package+xml"/></rootfiles>'
    '</container>'
)


def _cover_png_bytes() -> bytes:
    img = Image.new("RGB", (8, 8), color=(200, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_epub(opf_body: str, *, include_cover: bool = True) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("mimetype", "application/epub+zip")
        zf.writestr("META-INF/container.xml", _CONTAINER_XML)
        zf.writestr("OEBPS/content.opf", opf_body)
        if include_cover:
            zf.writestr("OEBPS/cover.png", _cover_png_bytes())
        zf.writestr(
            "OEBPS/ch1.xhtml",
            "<?xml version='1.0'?><html><body><p>text</p></body></html>",
        )
    return buf.getvalue()


def _tmp_epub(data: bytes) -> str:
    path = Path(tempfile.mkstemp(suffix=".epub")[1])
    path.write_bytes(data)
    return str(path)


_OPF_EPUB3 = (
    '<?xml version="1.0"?>'
    '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
    'unique-identifier="bookid">'
    '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
    '<dc:title>The Great Novel</dc:title>'
    '<dc:creator>Jane Doe</dc:creator>'
    '<dc:publisher>Example Press</dc:publisher>'
    '<dc:language>en-US</dc:language>'
    '<dc:date>2024-01-15</dc:date>'
    '<dc:description>A thrilling tale.</dc:description>'
    '</metadata>'
    '<manifest>'
    '<item id="cover" href="cover.png" media-type="image/png" '
    'properties="cover-image"/>'
    '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
    '</manifest>'
    '<spine><itemref idref="ch1"/></spine>'
    '</package>'
)

_OPF_EPUB2 = (
    '<?xml version="1.0"?>'
    '<package xmlns="http://www.idpf.org/2007/opf" version="2.0" '
    'unique-identifier="bookid">'
    '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
    '<dc:title>Old Style Book</dc:title>'
    '<dc:creator>John Smith</dc:creator>'
    '<meta name="cover" content="cover-image"/>'
    '</metadata>'
    '<manifest>'
    '<item id="cover-image" href="cover.png" media-type="image/png"/>'
    '<item id="ch1" href="ch1.xhtml" media-type="application/xhtml+xml"/>'
    '</manifest>'
    '<spine><itemref idref="ch1"/></spine>'
    '</package>'
)


class TestEpubExtractor:
    def test_epub3_all_metadata(self):
        path = _tmp_epub(_write_epub(_OPF_EPUB3))
        meta = extract(path)
        assert meta is not None
        assert meta.title == "The Great Novel"
        assert meta.author == "Jane Doe"
        assert meta.publisher == "Example Press"
        assert meta.language == "en-US"
        assert meta.date == "2024-01-15"
        assert meta.description == "A thrilling tale."

    def test_epub3_cover_is_data_uri(self):
        path = _tmp_epub(_write_epub(_OPF_EPUB3))
        meta = extract(path)
        assert meta is not None
        assert meta.cover_thumbnail.startswith("data:image/jpeg;base64,")
        # Should be decodable as a real JPEG
        import base64
        payload = meta.cover_thumbnail.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(payload)))
        assert img.format == "JPEG"

    def test_epub2_cover_via_meta_name(self):
        path = _tmp_epub(_write_epub(_OPF_EPUB2))
        meta = extract(path)
        assert meta is not None
        assert meta.title == "Old Style Book"
        assert meta.cover_thumbnail.startswith("data:image/jpeg;base64,")

    def test_as_source_metadata_omits_empty(self):
        path = _tmp_epub(_write_epub(_OPF_EPUB2))
        meta = extract(path)
        assert meta is not None
        blob = meta.as_source_metadata()
        assert blob == {"title": "Old Style Book", "author": "John Smith"}
        assert "cover_thumbnail" not in blob

    def test_bad_zip_returns_none(self):
        path = _tmp_epub(b"not a real zip")
        assert extract(path) is None

    def test_missing_opf_returns_none(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("mimetype", "application/epub+zip")
            zf.writestr("OEBPS/ch1.xhtml", "<html/>")
        path = _tmp_epub(buf.getvalue())
        assert extract(path) is None

    def test_no_cover_still_returns_metadata(self):
        opf_no_cover = (
            '<?xml version="1.0"?>'
            '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
            'unique-identifier="bookid">'
            '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
            '<dc:title>Coverless</dc:title>'
            '</metadata>'
            '<manifest>'
            '<item id="ch1" href="ch1.xhtml" '
            'media-type="application/xhtml+xml"/>'
            '</manifest>'
            '<spine><itemref idref="ch1"/></spine>'
            '</package>'
        )
        path = _tmp_epub(_write_epub(opf_no_cover, include_cover=False))
        meta = extract(path)
        assert meta is not None
        assert meta.title == "Coverless"
        assert meta.cover_thumbnail == ""
