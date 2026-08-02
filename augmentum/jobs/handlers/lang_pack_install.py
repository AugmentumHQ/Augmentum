"""``lang_pack_install`` job handler.

Builds a language-learning ``.augpack`` on demand: downloads the curated
source corpora for a language (see
:mod:`augmentum.learning.lang_pack_catalog`), runs
:func:`augmentum.knowledge.lang_pack_builder.build_pack` server-side,
writes ``<lang>.augpack`` into the knowledge-packs directory, and
re-scans :class:`augmentum.knowledge.packs.PackManager` so the new pack
is live.

Registered once on server startup via :func:`make_lang_pack_install_handler`
(the factory closes over ``app`` so ``app.state`` lookups happen at
run-time). Reuses the existing background-job machinery — progress flows
through ``ctx.update_progress`` and surfaces in the same UI that shows
ZIM conversions and GGUF downloads.

Payload shape (set by the install endpoint):

    {"lang_code": "ja"}

Idempotent: a re-run when the pack is already installed and loaded
short-circuits with ``{"skipped": True}`` instead of re-downloading
~100 MB — important because the runner re-queues in-flight jobs on
restart.

Result: ``{lang_code, pack_path, vocab, sentences}`` (or
``{lang_code, skipped: True}``).
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from augmentum.config import settings
from augmentum.jobs.context import JobContext, JobRetryable
from augmentum.knowledge.lang_pack_builder import (
    build_pack,  # noqa: F401 - resolved dynamically by _resolve_build_fn
    build_pack_cedict,  # noqa: F401 - resolved dynamically by _resolve_build_fn
    build_pack_wiktionary,  # noqa: F401 - resolved dynamically by _resolve_build_fn
)
from augmentum.learning import lang_pack_catalog as catalog
from augmentum.learning.lang_pack_downloader import download_to, filename_for
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Each catalog builder id maps to a (function-name, source_key → kwarg)
# pair. The function name is resolved against this module's globals at
# *call time* so that monkeypatch-replacing ``build_pack`` /
# ``build_pack_wiktionary`` in tests reaches the dispatched handler.
# Adding a builder = one entry here + a new build_pack_* function.
_BUILDERS: dict[str, tuple[str, dict[str, str]]] = {
    "jmdict_tatoeba": (
        "build_pack",
        {
            "jmdict": "jmdict_xml",
            "tatoeba_sentences": "tatoeba_sentences",
            "tatoeba_links": "tatoeba_links",
            "jlpt": "jlpt_tsv",
        },
    ),
    "wiktionary_tatoeba": (
        "build_pack_wiktionary",
        {
            "wiktionary": "wiktionary_jsonl",
            "tatoeba_sentences": "tatoeba_sentences",
            "tatoeba_links": "tatoeba_links",
            "frequency": "frequency_txt",
        },
    ),
    "cedict_tatoeba": (
        "build_pack_cedict",
        {
            "cedict": "cedict_txt",
            "tatoeba_sentences": "tatoeba_sentences",
            "tatoeba_links": "tatoeba_links",
            "hsk": "hsk_txt",
        },
    ),
}


def _resolve_build_fn(name: str):
    """Look up the build function in this module's globals at call time so
    pytest's ``monkeypatch.setattr(lpi, "build_pack", …)`` works."""
    return globals()[name]

# Download phase occupies [0, _DL_END] of the progress bar; the build
# phase occupies (_DL_END, 1.0].
_DL_END = 0.40


def _pack_dir() -> Path:
    return Path(
        settings.knowledge_packs_custom_dir
        or settings.knowledge_packs_dir
        or f"{settings.data_dir}/knowledge"
    )


def make_lang_pack_install_handler(app):
    """Build a handler bound to ``app.state`` services."""

    async def handler(ctx: JobContext) -> dict[str, Any]:
        lang_code = str(ctx.payload.get("lang_code") or "").strip()
        if not lang_code:
            raise ValueError("lang_pack_install requires payload.lang_code")
        spec = catalog.get(lang_code)
        if spec is None or spec.status != "available":
            raise ValueError(f"language pack '{lang_code}' is not installable")
        builder_entry = _BUILDERS.get(spec.builder)
        if builder_entry is None:
            raise ValueError(
                f"language pack '{lang_code}' uses unknown builder '{spec.builder}'"
            )
        build_fn_name, source_to_kwarg = builder_entry
        build_fn = _resolve_build_fn(build_fn_name)

        http_client = getattr(app.state, "http_client", None)
        pack_mgr = getattr(app.state, "pack_manager", None)
        if http_client is None:
            raise JobRetryable("http client not ready")

        out_path = _pack_dir() / f"{lang_code}.augpack"

        # Idempotency: already installed and loaded → nothing to do.
        if pack_mgr is not None and pack_mgr.has_language_pack(lang_code):
            await ctx.update_progress(1.0, stage="already_installed")
            return {"lang_code": lang_code, "pack_path": str(out_path), "skipped": True}

        await ctx.update_progress(0.02, stage="starting")
        await ctx.check_cancel()

        with tempfile.TemporaryDirectory(prefix=f"langpack_{lang_code}_") as tmp:
            tmpdir = Path(tmp)
            build_kwargs: dict[str, Any] = {}
            n_sources = max(1, len(spec.sources))

            loop = asyncio.get_running_loop()

            for i, source in enumerate(spec.sources):
                await ctx.check_cancel()
                dest = tmpdir / filename_for(source.url)
                lo = _DL_END * (i / n_sources)
                hi = _DL_END * ((i + 1) / n_sources)
                stage = f"downloading {source.key}"
                await ctx.update_progress(lo, stage=stage)

                # Coarse byte-progress (runs on the loop — it's invoked
                # from aiter_bytes — so a plain ensure_future schedules
                # the throttled DB write).
                last = {"frac": lo}

                def _on_bytes(done: int, total: int | None, *, _lo=lo, _hi=hi,
                              _stage=stage, _last=last) -> None:
                    if not total:
                        return
                    frac = _lo + (_hi - _lo) * min(1.0, done / total)
                    if frac - _last["frac"] >= 0.02:
                        _last["frac"] = frac
                        asyncio.ensure_future(ctx.update_progress(frac, stage=_stage))

                try:
                    await download_to(http_client, source.url, dest, on_progress=_on_bytes)
                except Exception as exc:  # noqa: BLE001
                    if source.required:
                        log.warning("lang_pack_required_source_failed",
                                    lang=lang_code, key=source.key, url=source.url, error=str(exc))
                        raise JobRetryable(f"failed to download {source.key}: {exc}") from exc
                    log.info("lang_pack_optional_source_skipped",
                             lang=lang_code, key=source.key, error=str(exc))
                    continue

                kwarg = source_to_kwarg.get(source.key)
                if kwarg:
                    build_kwargs[kwarg] = dest

            await ctx.check_cancel()
            await ctx.update_progress(_DL_END, stage="building dictionary")

            # build_pack is sync + CPU/IO-heavy → run off the event loop;
            # its progress callback runs *in the worker thread*, so bridge
            # it back to the loop with run_coroutine_threadsafe.
            def _build_progress(frac: float, stage: str) -> None:
                mapped = _DL_END + (1.0 - _DL_END) * max(0.0, min(1.0, frac))
                asyncio.run_coroutine_threadsafe(
                    ctx.update_progress(mapped, stage=stage), loop,
                )

            tatoeba_lang = catalog.TATOEBA_LANG_CODE.get(lang_code, "jpn")
            result = await ctx.run_in_thread(
                build_fn,
                out_path=out_path,
                lang_code=lang_code,
                name=spec.name,
                tatoeba_lang=tatoeba_lang,
                progress=_build_progress,
                **build_kwargs,
            )

        # Pick up the freshly-built pack.
        if pack_mgr is not None:
            try:
                await pack_mgr.scan()
            except Exception:
                log.warning("lang_pack_rescan_failed", lang=lang_code, exc_info=True)

        await ctx.update_progress(1.0, stage="done")
        log.info("lang_pack_installed", lang=lang_code, path=str(out_path), **result)
        return {"lang_code": lang_code, "pack_path": str(out_path), **result}

    return handler
