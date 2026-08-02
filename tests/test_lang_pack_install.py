"""Tests for the language-pack downloader + the lang_pack_install job handler.

Network and the heavy build are mocked at their module boundaries
(``download_to`` / ``build_pack``); we're exercising orchestration, not
JMdict parsing (covered by ``test_lang_pack_builder``).
"""

from __future__ import annotations

import asyncio
import types

import pytest

from augmentum.jobs.context import JobContext, JobRetryable
from augmentum.jobs.handlers import lang_pack_install as lpi
from augmentum.learning import lang_pack_downloader as dl
from augmentum.utils import streamed_download as sd

# ── download_to ──────────────────────────────────────────────────────


class _FakeStreamResp:
    def __init__(self, chunks, *, headers=None, ok=True):
        self._chunks = chunks
        self.headers = headers or {}
        self._ok = ok

    def raise_for_status(self):
        if not self._ok:
            raise RuntimeError("HTTP 404")

    async def aiter_bytes(self, _chunk_size):
        for c in self._chunks:
            yield c


class _FakeStreamCM:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *_a):
        return False


class _FakeClient:
    def __init__(self, by_url):
        self._by_url = by_url
        self.requests: list[tuple[str, str]] = []

    def stream(self, method, url, **_kw):
        self.requests.append((method, url))
        if url not in self._by_url:
            raise RuntimeError(f"unexpected url {url}")
        return _FakeStreamCM(self._by_url[url])


@pytest.fixture(autouse=True)
def _no_ssrf(monkeypatch):
    async def _ok(_url, **_kw):
        return None
    monkeypatch.setattr(sd, "check_ssrf", _ok)


@pytest.mark.asyncio
async def test_download_to_writes_file_and_reports_progress(tmp_path):
    url = "https://example.test/data.tar.bz2"
    client = _FakeClient({url: _FakeStreamResp([b"abc", b"defg", b"hi"],
                                               headers={"content-length": "9"})})
    dest = tmp_path / "data.tar.bz2"
    seen: list[tuple[int, int | None]] = []
    n = await dl.download_to(client, url, dest, on_progress=lambda d, t: seen.append((d, t)))
    assert n == 9
    assert dest.read_bytes() == b"abcdefghi"
    assert not (tmp_path / "data.tar.bz2.part").exists()  # temp cleaned up
    assert seen[-1] == (9, 9)


@pytest.mark.asyncio
async def test_download_to_no_content_length(tmp_path):
    url = "https://example.test/x.gz"
    client = _FakeClient({url: _FakeStreamResp([b"xyz"])})
    dest = tmp_path / "x.gz"
    seen = []
    n = await dl.download_to(client, url, dest, on_progress=lambda d, t: seen.append((d, t)))
    assert n == 3
    assert dest.read_bytes() == b"xyz"
    assert seen and seen[-1][1] is None


@pytest.mark.asyncio
async def test_download_to_http_error_propagates(tmp_path):
    url = "https://example.test/missing"
    client = _FakeClient({url: _FakeStreamResp([], ok=False)})
    with pytest.raises(RuntimeError):
        await dl.download_to(client, url, tmp_path / "out")


def test_filename_for():
    assert dl.filename_for("https://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz") == "JMdict_e.gz"
    assert dl.filename_for("https://downloads.tatoeba.org/exports/sentences.tar.bz2") == "sentences.tar.bz2"
    assert dl.filename_for("https://x/dir/") == "download"
    assert dl.filename_for("https://x/%2e%2e/evil") == ".._evil" or dl.filename_for("https://x/%2e%2e/evil") == "evil"
    # No path traversal in the result.
    assert "/" not in dl.filename_for("https://x/a/b/c")
    assert "\\" not in dl.filename_for("https://x/a\\b")


# ── lang_pack_install handler ────────────────────────────────────────


class _FakeStore:
    def __init__(self):
        self.progress: list[tuple[float, str]] = []
        self.cancelled = False

    async def update_progress(self, _job_id, *, progress, stage=""):
        self.progress.append((progress, stage))

    async def is_cancel_requested(self, _job_id):
        return self.cancelled


class _FakePackManager:
    def __init__(self, has=False):
        self._has = has
        self.scans = 0

    def has_language_pack(self, _lang):
        return self._has

    async def scan(self):
        self.scans += 1
        self._has = True  # after install + scan, it's loaded
        return 1


def _make_app(*, http_client, pack_manager):
    return types.SimpleNamespace(state=types.SimpleNamespace(
        http_client=http_client, pack_manager=pack_manager,
    ))


def _make_ctx(store, payload):
    return JobContext(job_id="job_1", user_id="usr_1", job_type="lang_pack_install",
                      payload=payload, store=store)


@pytest.mark.asyncio
async def test_handler_already_installed_skips(monkeypatch):
    store = _FakeStore()
    app = _make_app(http_client=object(), pack_manager=_FakePackManager(has=True))
    handler = lpi.make_lang_pack_install_handler(app)
    out = await handler(_make_ctx(store, {"lang_code": "ja"}))
    assert out["skipped"] is True
    assert out["lang_code"] == "ja"
    assert store.progress[-1] == (1.0, "already_installed")


@pytest.mark.asyncio
async def test_handler_unknown_lang_raises(monkeypatch):
    app = _make_app(http_client=object(), pack_manager=_FakePackManager())
    handler = lpi.make_lang_pack_install_handler(app)
    with pytest.raises(ValueError):
        await handler(_make_ctx(_FakeStore(), {"lang_code": "klingon"}))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lang", "builder_attr", "expected_kwargs"),
    [
        ("zh", "build_pack_cedict", {"cedict_txt", "hsk_txt"}),
        ("fr", "build_pack_wiktionary", {"wiktionary_jsonl", "frequency_txt"}),
        ("ko", "build_pack_wiktionary", {"wiktionary_jsonl", "frequency_txt"}),
    ],
)
async def test_handler_dispatches_available_non_japanese_builders(
    monkeypatch, tmp_path, lang, builder_attr, expected_kwargs,
):
    monkeypatch.setattr(lpi, "_pack_dir", lambda: tmp_path)

    async def _fake_dl(_client, _url, dest, **_kw):
        dest.write_bytes(b"stub")
        return 4
    monkeypatch.setattr(lpi, "download_to", _fake_dl)

    built: dict = {}

    def _fake_build(*, out_path, lang_code, name, tatoeba_lang, progress, **kwargs):
        built.update(
            out_path=out_path,
            lang_code=lang_code,
            name=name,
            tatoeba_lang=tatoeba_lang,
            kwargs=kwargs,
        )
        progress(0.5, "building")
        out_path.write_bytes(b"AUGPACK")
        return {"vocab": 12, "sentences": 34}
    monkeypatch.setattr(lpi, builder_attr, _fake_build)

    pm = _FakePackManager(has=False)
    app = _make_app(http_client=object(), pack_manager=pm)
    handler = lpi.make_lang_pack_install_handler(app)

    out = await handler(_make_ctx(_FakeStore(), {"lang_code": lang}))

    assert out["lang_code"] == lang
    assert out["vocab"] == 12 and out["sentences"] == 34
    assert built["lang_code"] == lang
    assert built["tatoeba_lang"] == lpi.catalog.TATOEBA_LANG_CODE[lang]
    assert expected_kwargs <= set(built["kwargs"])
    assert (tmp_path / f"{lang}.augpack").exists()
    assert pm.scans == 1


@pytest.mark.asyncio
async def test_handler_no_http_client_retryable():
    app = _make_app(http_client=None, pack_manager=_FakePackManager())
    handler = lpi.make_lang_pack_install_handler(app)
    with pytest.raises(JobRetryable):
        await handler(_make_ctx(_FakeStore(), {"lang_code": "ja"}))


@pytest.mark.asyncio
async def test_handler_happy_path(monkeypatch, tmp_path):
    # Stub the pack dir, the downloader, and the heavy build.
    monkeypatch.setattr(lpi, "_pack_dir", lambda: tmp_path)
    downloaded: list[str] = []

    async def _fake_dl(_client, url, dest, **_kw):
        downloaded.append(url)
        dest.write_bytes(b"stub")
        return 4
    monkeypatch.setattr(lpi, "download_to", _fake_dl)

    built: dict = {}

    def _fake_build(*, out_path, lang_code, name, tatoeba_lang, progress, **kwargs):
        built.update(out_path=out_path, lang_code=lang_code, name=name,
                     tatoeba_lang=tatoeba_lang, kwargs=kwargs)
        progress(0.5, "building")  # exercise the thread→loop bridge
        out_path.write_bytes(b"AUGPACK")
        return {"vocab": 123, "sentences": 456}
    monkeypatch.setattr(lpi, "build_pack", _fake_build)

    pm = _FakePackManager(has=False)
    app = _make_app(http_client=object(), pack_manager=pm)
    store = _FakeStore()
    handler = lpi.make_lang_pack_install_handler(app)

    out = await handler(_make_ctx(store, {"lang_code": "ja"}))

    assert out["lang_code"] == "ja"
    assert out["vocab"] == 123 and out["sentences"] == 456
    assert out["pack_path"].endswith("ja.augpack")
    # All required JA sources were fetched.
    assert any("JMdict_e.gz" in u for u in downloaded)
    assert any("sentences.tar.bz2" in u for u in downloaded)
    assert any("links.tar.bz2" in u for u in downloaded)
    # build_pack received the mapped kwargs.
    assert "jmdict_xml" in built["kwargs"]
    assert "tatoeba_sentences" in built["kwargs"]
    assert "tatoeba_links" in built["kwargs"]
    assert built["tatoeba_lang"] == "jpn"
    # Pack dir got the file; pack manager was rescanned.
    assert (tmp_path / "ja.augpack").exists()
    assert pm.scans == 1
    assert store.progress[-1] == (1.0, "done")
    # Progress is monotone-ish and bounded.
    assert all(0.0 <= p <= 1.0 for p, _ in store.progress)
    # Give the thread→loop bridge a tick to land its update.
    await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_handler_required_source_failure_is_retryable(monkeypatch, tmp_path):
    monkeypatch.setattr(lpi, "_pack_dir", lambda: tmp_path)

    async def _fail_dl(_client, url, _dest, **_kw):
        raise RuntimeError("network down")
    monkeypatch.setattr(lpi, "download_to", _fail_dl)
    monkeypatch.setattr(lpi, "build_pack", lambda **_kw: {"vocab": 0, "sentences": 0})

    app = _make_app(http_client=object(), pack_manager=_FakePackManager())
    handler = lpi.make_lang_pack_install_handler(app)
    with pytest.raises(JobRetryable):
        await handler(_make_ctx(_FakeStore(), {"lang_code": "ja"}))
