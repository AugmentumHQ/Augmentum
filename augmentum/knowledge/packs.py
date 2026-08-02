"""PackManager — load, search, and manage .augpack knowledge packs.

Hybrid retrieval pipeline:
  1. Per-pack vector search (sqlite-vec) and FTS5 keyword search run in
     parallel; ZIM packs contribute their own keyword search.
  2. Reciprocal Rank Fusion merges all legs into a single ranked list,
     mirroring DocumentStore.search.
  3. Optional cross-encoder reranking via RerankService for precision.

Score-distance gymnastics from the previous implementation are gone — RRF
is rank-based, so vector L2 and ZIM keyword scores no longer have to be
reconciled to the same scale.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite
import sqlite_vec

from augmentum.config import settings
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class _SearchCacheEntry:
    """One cached search result, with the timestamp used for TTL eviction."""
    results: list  # list[PackResult]
    timestamp: float


@dataclass
class PackMeta:
    """Metadata read from the ``meta`` table of a .augpack file."""

    name: str = ""
    version: str = ""
    description: str = ""
    embedding_model: str = ""
    embedding_dim: int = 0
    chunk_count: int = 0
    source_license: str = ""
    build_date: str = ""
    # ``language`` for language-learning packs (vocab/sentences schema),
    # "" for the usual chunk-based reference packs.
    pack_kind: str = ""
    lang_code: str = ""       # ISO code, language packs only
    vocab_count: int = 0      # entry count, language packs only


@dataclass
class PackResult:
    """A single search result from a knowledge pack."""

    content: str
    title: str
    section: str
    url: str
    pack_id: str
    source: str
    score: float = 0.0  # RRF score (post-rerank if reranker ran)


@dataclass
class PackConnection:
    """An open connection to a .augpack file."""

    conn: aiosqlite.Connection
    meta: PackMeta
    path: Path
    active: bool = True
    has_fts: bool = False  # False until FTS5 mirror confirmed/built; legacy packs lazy-rebuild on scan
    fts_dim_warned: bool = field(default=False, repr=False)  # one-shot: dim-mismatch warning per process


@dataclass
class ZimPack:
    """A ZIM-backed knowledge pack (large packs, keyword search)."""

    reader: Any  # ZimReader
    meta: PackMeta
    path: Path
    active: bool = True
    cache_path: Path | None = None


@dataclass
class LanguagePack:
    """A language-learning pack (``pack_kind=language``).

    Holds a read-only connection to a ``.augpack`` whose schema is
    ``vocab`` / ``vocab_fts`` / ``sentences`` rather than ``chunks``.
    Deliberately kept out of the retrieval (search) path — the language-
    learning routes query it directly via :mod:`augmentum.learning.lang_packs`.
    """

    conn: aiosqlite.Connection
    meta: PackMeta
    path: Path
    active: bool = True


class PackManager:
    """Core data-layer for .augpack knowledge packs.

    Each pack is a standalone SQLite database with sqlite-vec embeddings.
    The manager scans a directory, opens read-only connections, and merges
    vector-similarity search results across all active packs.
    """

    def __init__(self, pack_dir: Path) -> None:
        self._pack_dir = pack_dir
        self._packs: dict[str, PackConnection] = {}
        self._zim_packs: dict[str, ZimPack] = {}
        # Language-learning packs (pack_kind=language). Separate registry —
        # they have a vocab/sentences schema and never enter the retrieval
        # path; the language-learning routes query them directly.
        self._language_packs: dict[str, LanguagePack] = {}
        self._active_state: dict[str, bool] = {}
        self._state_loaded = False
        # LRU search cache. Keys are (query, sorted-pack-id-tuple, limit,
        # rerank-flag); values are _SearchCacheEntry. Bounded LRU behavior
        # via OrderedDict.popitem(last=False) on overflow. TTL enforced at
        # read time. Cleared on pack add/remove/(de)activate so the user
        # never sees stale results across pack-state changes.
        self._search_cache: OrderedDict[tuple, _SearchCacheEntry] = OrderedDict()
        self._search_cache_hits = 0  # Diagnostic counter — surfaced in logs.
        self._search_cache_misses = 0
        # Stale install jobs (failed conversions). Populated by scan() on
        # startup; cleared/repopulated by discard_failed_conversion(). The
        # API surfaces this list so the UI can show "Conversion incomplete"
        # cards with Discard / Retry actions.
        self.failed_conversions: list[dict[str, Any]] = []

    @property
    def pack_dir(self) -> Path:
        """Public accessor for the packs directory."""
        return self._pack_dir

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_count(self) -> int:
        """Number of currently active packs."""
        return sum(1 for p in self._packs.values() if p.active) + sum(
            1 for z in self._zim_packs.values() if z.active
        )

    @property
    def installed(self) -> list[dict[str, Any]]:
        """Return metadata dicts for every loaded pack.

        When a pack_id has BOTH an .augpack and a sidecar .zim (small packs
        that were converted but kept the source for browseability), the
        listing returns one merged row: augpack metadata for chunks/size,
        with the .zim's main_entry_path lifted in so the UI's Browse button
        lights up. .zim-only and .augpack-only packs each get their own row
        with the same schema.
        """
        out: list[dict[str, Any]] = []
        out_by_id: dict[str, dict[str, Any]] = {}
        for pack_id, pc in self._packs.items():
            m = pc.meta
            entry: dict[str, Any] = {
                "pack_id": pack_id,
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "embedding_model": m.embedding_model,
                "embedding_dim": m.embedding_dim,
                "chunk_count": m.chunk_count,
                "source_license": m.source_license,
                "build_date": m.build_date,
                "active": pc.active,
                "path": str(pc.path),
                # Filled below if a .zim sidecar exists for this pack_id.
                "main_entry_path": None,
                # An augpack row always has the vector index (it IS the
                # vector index). UI uses this to render Embed vs Embedded
                # state on the per-pack settings card.
                "has_vector_index": True,
            }
            out.append(entry)
            out_by_id[pack_id] = entry
        for pack_id, zp in self._zim_packs.items():
            # Resolve the ZIM's main page so the settings UI can offer a
            # "Browse" hotlink. Some ZIMs declare main_entry as a redirect;
            # follow one hop. None on any failure so the UI hides the button.
            main_entry_path: str | None = None
            archive = getattr(zp.reader, "_archive", None)
            if archive is not None:
                try:
                    main = archive.main_entry
                    if getattr(main, "is_redirect", False):
                        main = main.get_redirect_entry()
                    main_entry_path = main.path
                except Exception as exc:
                    log.debug("zim_main_entry_resolve_failed", pack_id=pack_id, error=str(exc))

            sidecar_target = out_by_id.get(pack_id)
            if sidecar_target is not None:
                # .zim is a sidecar to an existing .augpack — enrich the
                # augpack row's main_entry_path so Browse lights up. Don't
                # append a duplicate listing row.
                sidecar_target["main_entry_path"] = main_entry_path
                continue

            m = zp.meta
            entry = {
                "pack_id": pack_id,
                "name": m.name,
                "version": m.version,
                "description": m.description,
                "type": "zim",
                "chunk_count": m.chunk_count,
                "source_license": m.source_license,
                "build_date": m.build_date,
                "active": zp.active,
                "path": str(zp.path),
                "main_entry_path": main_entry_path,
                # ZIM-only — no augpack sidecar exists. UI surfaces the
                # "Embed for vector search" button for these packs.
                "has_vector_index": False,
            }
            try:
                entry["file_size"] = zp.path.stat().st_size
            except OSError:
                entry["file_size"] = 0
            out.append(entry)
        return out

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def scan(self) -> int:
        """Scan ``pack_dir`` for .augpack files and open connections.

        For each pack, ensures the chunks_fts mirror exists (lazy-rebuilds
        from chunks for packs imported before FTS support landed). Sets
        query_only=ON only AFTER the rebuild so legacy packs can be
        upgraded in place without writing a new file.

        Returns the number of packs loaded (including previously loaded).
        """
        if not self._pack_dir.exists():
            log.warning("pack directory does not exist", path=str(self._pack_dir))
            return 0

        self._load_state()
        # Surface failed conversions left over from prior install jobs
        # (a .progress.json file alongside an empty .augpack shell). The
        # empty-augpack guard further down catches the data side; this
        # populates ``self.failed_conversions`` so the API + UI can show
        # them with retry/discard affordances.
        self._scan_failed_conversions()

        for entry in self._pack_dir.iterdir():
            if entry.suffix != ".augpack":
                continue
            pack_id = entry.stem
            if pack_id in self._packs or pack_id in self._language_packs:
                continue
            try:
                conn = await aiosqlite.connect(str(entry), uri=False)
                # Load sqlite-vec extension
                await conn.enable_load_extension(True)
                await conn.load_extension(sqlite_vec.loadable_path())
                await conn.enable_load_extension(False)

                # Language packs (pack_kind=language) carry a vocab/sentences
                # schema, not chunks — they'd trip the partial-augpack guard
                # below. Detect them up front and register into a separate
                # registry that never touches the retrieval path.
                try:
                    cur = await conn.execute(
                        "SELECT value FROM meta WHERE key = 'pack_kind'"
                    )
                    row = await cur.fetchone()
                    pack_kind = (row[0] if row and row[0] else "")
                except Exception:
                    pack_kind = ""
                if pack_kind == "language":
                    await conn.execute("PRAGMA query_only = ON")
                    await conn.execute("PRAGMA journal_mode = OFF")
                    meta = await self._read_meta(conn)
                    self._language_packs[pack_id] = LanguagePack(
                        conn=conn,
                        meta=meta,
                        path=entry,
                        active=self._active_state.get(pack_id, True),
                    )
                    log.info(
                        "loaded_language_pack",
                        pack_id=pack_id,
                        lang=meta.lang_code,
                        vocab=meta.vocab_count,
                    )
                    continue

                # Hard guard: empty or partial augpacks (failed conversions
                # left behind a SQLite shell with 0 chunks, OR a partial pack
                # with chunks committed but no meta written) would shadow the
                # original ZIM if it has the same stem — the leg dispatcher
                # checks _packs first and continues, so the ZIM never gets a
                # chance. We also need to skip the partial-meta case to avoid
                # silently serving a fraction of the corpus with blank meta.
                # Both cases are surfaced via failed_conversions for the UI
                # Resume / Discard affordances.
                try:
                    cursor = await conn.execute("SELECT count(*) FROM chunks")
                    row = await cursor.fetchone()
                    chunk_count = int(row[0]) if row else 0
                except Exception:
                    # Missing chunks table = brand new shell, definitely empty.
                    chunk_count = 0
                meta_count = 0
                try:
                    cursor = await conn.execute("SELECT count(*) FROM meta")
                    row = await cursor.fetchone()
                    meta_count = int(row[0]) if row else 0
                except Exception:
                    # Missing meta table = same as empty; treat as broken.
                    meta_count = 0
                if chunk_count == 0 or meta_count == 0:
                    log.warning(
                        "knowledge_pack_skipped_partial_augpack",
                        pack_id=pack_id,
                        path=str(entry),
                        chunks=chunk_count,
                        meta_rows=meta_count,
                        hint=(
                            "failed conversion: "
                            f"{'empty shell' if chunk_count == 0 else 'partial pack (no meta)'}"
                            "; resume or discard via Browse landing"
                        ),
                    )
                    await conn.close()
                    continue

                # Lazy-build the FTS mirror for legacy packs imported before
                # FTS support. Must run BEFORE query_only=ON. Skip silently
                # for packs already on the new format (the trigger keeps the
                # mirror in sync going forward, so this only fires once per
                # legacy pack per machine).
                has_fts = await self._ensure_fts_index(conn, pack_id)

                await conn.execute("PRAGMA query_only = ON")
                await conn.execute("PRAGMA journal_mode = OFF")
                meta = await self._read_meta(conn)
                self._packs[pack_id] = PackConnection(
                    conn=conn,
                    meta=meta,
                    path=entry,
                    active=self._active_state.get(pack_id, True),
                    has_fts=has_fts,
                )
                log.info(
                    "loaded pack",
                    pack_id=pack_id,
                    name=meta.name,
                    chunks=meta.chunk_count,
                    has_fts=has_fts,
                )
            except Exception:
                log.warning("failed to load pack", pack_id=pack_id, exc_info=True)

        # Scan for .zim files
        for entry in sorted(self._pack_dir.iterdir(), key=lambda p: p.name):
            if entry.suffix != ".zim":
                continue
            pack_id = entry.stem
            if pack_id in self._zim_packs:
                continue
            # Skip partial/corrupt downloads (ZIM header is 80+ bytes minimum)
            if entry.stat().st_size < 256:
                log.warning("skipping tiny zim file", pack_id=pack_id, size=entry.stat().st_size)
                continue
            try:
                from augmentum.knowledge.zim_reader import ZimReader

                reader = ZimReader(entry)
                if reader._archive is None:
                    log.warning("zim_archive_unusable", pack_id=pack_id)
                    continue
                article_count = reader.article_count
                meta = PackMeta(
                    name=pack_id.replace("_", " ").replace("-", " ").title(),
                    description=f"ZIM archive ({article_count} articles)",
                    chunk_count=article_count,
                )
                cache_name = f"{pack_id}_cache.augpack"
                cache_file = self._pack_dir / cache_name
                self._zim_packs[pack_id] = ZimPack(
                    reader=reader,
                    meta=meta,
                    path=entry,
                    active=self._active_state.get(pack_id, True),
                    cache_path=cache_file if cache_file.exists() else None,
                )
                log.info(
                    "loaded zim pack",
                    pack_id=pack_id,
                    articles=article_count,
                )
            except Exception:
                log.warning("failed to load zim pack", pack_id=pack_id, exc_info=True)

        return len(self._packs) + len(self._zim_packs)

    async def close(self) -> None:
        """Close all open connections."""
        for pack_id, pc in list(self._packs.items()):
            try:
                await pc.conn.close()
            except Exception:
                log.warning("error closing pack", pack_id=pack_id, exc_info=True)
        self._packs.clear()
        for pack_id, zp in list(self._zim_packs.items()):
            try:
                zp.reader.close()
            except Exception:
                log.warning("error closing zim pack", pack_id=pack_id, exc_info=True)
        self._zim_packs.clear()
        for pack_id, lp in list(self._language_packs.items()):
            try:
                await lp.conn.close()
            except Exception:
                log.warning("error closing language pack", pack_id=pack_id, exc_info=True)
        self._language_packs.clear()

    # ------------------------------------------------------------------
    # Language packs (pack_kind=language)
    # ------------------------------------------------------------------

    def list_language_packs(self) -> list[dict]:
        """Installed language packs, for the Learning toggle/onboarding.

        Returns one dict per pack: ``pack_id``, ``lang_code``, ``name``,
        ``vocab_count``, ``active``. Includes inactive packs so the UI can
        show "installed but disabled".
        """
        self._load_state()
        return [
            {
                "pack_id": pack_id,
                "lang_code": lp.meta.lang_code,
                "name": lp.meta.name,
                "vocab_count": lp.meta.vocab_count,
                "active": lp.active,
            }
            for pack_id, lp in self._language_packs.items()
        ]

    def get_language_pack(self, lang_code: str) -> LanguagePack | None:
        """The first *active* language pack for ``lang_code``, or None."""
        for lp in self._language_packs.values():
            if lp.active and lp.meta.lang_code == lang_code:
                return lp
        return None

    def has_language_pack(self, lang_code: str) -> bool:
        return self.get_language_pack(lang_code) is not None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        *,
        pack_ids: list[str],
        limit: int = 5,
        rerank: bool = True,
    ) -> list[PackResult]:
        """Hybrid pack search: per-pack (vector + FTS5) → RRF merge → optional rerank.

        Mirrors DocumentStore.search shape. ZIM packs participate via their
        own keyword search and join the same RRF stage. Returns at most
        ``limit`` results.

        Empty query returns []; never embeds an empty string.
        """
        if not query.strip() or not pack_ids:
            return []

        # Result cache lookup. Cheap path: same query + same pack set +
        # same caps within TTL window → return immediately. Most common
        # hit pattern is the user's debounce/re-render cycle and sub-
        # second navigation back to a search they just ran. Expired
        # entries are evicted lazily on access.
        cache_key: tuple | None = None
        if settings.knowledge_search_cache_enabled:
            cache_key = (query, tuple(sorted(pack_ids)), limit, rerank)
            cached = self._search_cache.get(cache_key)
            if cached is not None:
                age = time.monotonic() - cached.timestamp
                if age < settings.knowledge_search_cache_ttl_seconds:
                    self._search_cache.move_to_end(cache_key)
                    self._search_cache_hits += 1
                    log.debug(
                        "knowledge_pack_cache_hit",
                        query=query[:60],
                        age_s=round(age, 1),
                        hits=self._search_cache_hits,
                        misses=self._search_cache_misses,
                    )
                    # Return a fresh list of fresh PackResult instances so
                    # downstream mutations (e.g. reranker score writes) don't
                    # poison the cached entry. PackResult is small, copy is
                    # ~µs per entry.
                    return [
                        PackResult(
                            content=r.content, title=r.title, section=r.section,
                            url=r.url, source=r.source, pack_id=r.pack_id, score=r.score,
                        )
                        for r in cached.results
                    ]
                # Expired — evict and fall through.
                del self._search_cache[cache_key]
            self._search_cache_misses += 1

        # Widen the candidate pool when reranking — same heuristic the
        # document store uses. Reranker needs headroom to find the right
        # paragraph; without rerank, just trim a little wider than the
        # final cap so RRF has something to fuse.
        candidate_limit = limit * 10 if rerank else limit * 2

        # Embed once for all augpack vector legs; ZIM doesn't use it.
        from augmentum.memory.embeddings import EmbeddingService
        query_blob: bytes = b""
        try:
            vec = await asyncio.to_thread(EmbeddingService.embed_query, query)
            query_blob = EmbeddingService.to_blob(vec)
        except Exception:
            log.warning("knowledge_pack_query_embed_failed", exc_info=True)
            # Continue — FTS / ZIM legs can still produce results.

        # Collect rank-ordered hit lists from every leg, in parallel.
        leg_tasks: list[asyncio.Task] = []
        leg_kinds: list[str] = []  # parallel array — annotated in commissioning logs
        for pack_id in pack_ids:
            # Augpack and ZIM legs can BOTH exist for the same pack_id when
            # the user has both files (e.g. converted devdocs ZIM into an
            # embedded augpack but kept the ZIM for browseable view). Run
            # whichever exists; RRF merges dedups across legs at fusion
            # time. The earlier ``continue`` here meant clicking the ZIM
            # card silently fell back to augpack legs and broke browsing.
            pc = self._packs.get(pack_id)
            if pc and pc.active:
                if query_blob:
                    leg_tasks.append(asyncio.create_task(
                        self._vector_leg(pack_id, pc, query_blob, candidate_limit)
                    ))
                    leg_kinds.append(f"vec:{pack_id}")
                if pc.has_fts:
                    leg_tasks.append(asyncio.create_task(
                        self._fts_leg(pack_id, pc, query, candidate_limit)
                    ))
                    leg_kinds.append(f"fts:{pack_id}")
            zp = self._zim_packs.get(pack_id)
            if zp and zp.active:
                leg_tasks.append(asyncio.create_task(
                    self._zim_leg(pack_id, zp, query, candidate_limit)
                ))
                leg_kinds.append(f"zim:{pack_id}")

        # TEMP commissioning log — drop once hybrid path is verified stable.
        log.info(
            "knowledge_pack_legs_dispatched",
            legs=leg_kinds,
            query=query[:80],
            candidate_limit=candidate_limit,
        )

        leg_results: list[list[PackResult]] = []
        leg_counts: list[int] = []
        if leg_tasks:
            for task_result in await asyncio.gather(*leg_tasks, return_exceptions=True):
                if isinstance(task_result, BaseException):
                    log.warning("knowledge_pack_leg_failed", error=str(task_result))
                    leg_counts.append(-1)
                    continue
                count = len(task_result) if task_result else 0
                leg_counts.append(count)
                if task_result:
                    leg_results.append(task_result)

        # TEMP commissioning log — pairs with knowledge_pack_legs_dispatched.
        log.info("knowledge_pack_legs_returned", counts=leg_counts, total_legs=len(leg_kinds))

        if not leg_results:
            return []

        # Reciprocal Rank Fusion across all legs. Same constant as
        # DocumentStore._rrf_merge — proven default.
        merged = self._rrf_merge(leg_results, k=60)
        if not merged:
            return []

        # Cross-encoder rerank for precision. Off-loaded so the model's
        # forward pass doesn't stall the event loop.
        if rerank:
            merged = await asyncio.to_thread(
                self._rerank, query, merged, limit,
            )
        else:
            merged = merged[:limit]

        # Cache the final result. Skip empty results — they're cheap to
        # recompute and caching empties would mask new pack additions
        # for users who searched a stale empty corpus then installed a
        # pack. Bounded LRU eviction.
        if cache_key is not None and merged:
            self._search_cache[cache_key] = _SearchCacheEntry(
                results=list(merged),
                timestamp=time.monotonic(),
            )
            while len(self._search_cache) > settings.knowledge_search_cache_size:
                self._search_cache.popitem(last=False)

        return merged

    def _invalidate_search_cache(self) -> None:
        """Clear the result cache. Called whenever pack composition or
        active state changes — adding/removing a pack should make the
        next search reflect that immediately, not wait for TTL."""
        if self._search_cache:
            log.debug(
                "knowledge_pack_cache_invalidated",
                entries=len(self._search_cache),
                hits=self._search_cache_hits,
                misses=self._search_cache_misses,
            )
            self._search_cache.clear()

    async def _vector_leg(
        self,
        pack_id: str,
        pc: PackConnection,
        query_blob: bytes,
        limit: int,
    ) -> list[PackResult]:
        """Vector search against a single augpack's chunks_vec."""
        query_dim = len(query_blob) // 4
        if pc.meta.embedding_dim and pc.meta.embedding_dim != query_dim:
            if not pc.fts_dim_warned:
                log.warning(
                    "knowledge_pack_embedding_dim_mismatch",
                    pack_id=pack_id,
                    pack_dim=pc.meta.embedding_dim,
                    query_dim=query_dim,
                )
                pc.fts_dim_warned = True
            return []
        try:
            cursor = await pc.conn.execute(
                """
                SELECT c.id, c.content, c.title, c.section, c.url, c.source
                FROM chunks_vec v
                JOIN chunks c ON c.id = v.id
                WHERE v.embedding MATCH ? AND k = ?
                """,
                (query_blob, limit),
            )
            rows = await cursor.fetchall()
        except Exception:
            log.warning("knowledge_pack_vector_leg_failed", pack_id=pack_id, exc_info=True)
            return []
        return [
            PackResult(
                content=row[1],
                title=row[2],
                section=row[3] or "",
                url=row[4] or "",
                source=row[5] or "",
                pack_id=pack_id,
            )
            for row in rows
        ]

    async def _fts_leg(
        self,
        pack_id: str,
        pc: PackConnection,
        query: str,
        limit: int,
    ) -> list[PackResult]:
        """FTS5 keyword search against a single augpack's chunks_fts."""
        from augmentum.utils.fts import tokenize_fts_query

        tokenized = tokenize_fts_query(query)
        expressions = list(tokenized) if isinstance(tokenized, tuple) else [tokenized]
        for fts_expr in expressions:
            if not fts_expr.strip():
                continue
            try:
                cursor = await pc.conn.execute(
                    """
                    SELECT c.id, c.content, c.title, c.section, c.url, c.source
                    FROM chunks_fts f
                    JOIN chunks c ON c.id = f.rowid
                    WHERE chunks_fts MATCH ?
                    ORDER BY rank LIMIT ?
                    """,
                    (fts_expr, limit),
                )
                rows = await cursor.fetchall()
            except Exception:
                log.debug(
                    "knowledge_pack_fts_leg_failed",
                    pack_id=pack_id,
                    expr=fts_expr[:60],
                    exc_info=True,
                )
                continue  # try next expression (AND → OR fallback)
            if rows:
                return [
                    PackResult(
                        content=row[1],
                        title=row[2],
                        section=row[3] or "",
                        url=row[4] or "",
                        source=row[5] or "",
                        pack_id=pack_id,
                    )
                    for row in rows
                ]
        return []

    async def _zim_leg(
        self,
        pack_id: str,
        zp: ZimPack,
        query: str,
        limit: int,
    ) -> list[PackResult]:
        """Section-level passage search against a ZIM archive.

        Each article is split into section-aligned passages (~900 chars
        each) with MediaWiki/CSS chrome stripped. The reranker scores
        passages — its trained granularity — and the per-mode budget
        ends up bounding "how many passages fit" rather than "how many
        whole articles fit," which it never could.

        ``limit`` here is the wide candidate pool from ``search()``
        (``limit*10`` with rerank, ``limit*2`` without). Articles fetched
        scales with it so the reranker has enough candidates without
        running away on large queries.
        """
        # Aim for ~4 passages per article. Floor at 3 articles so a small
        # candidate pool doesn't collapse to one article's worth of
        # passages — multi-article retrieval is what makes the encyclopedia
        # use case work.
        max_articles = max(3, (limit + 3) // 4)
        try:
            passages = await asyncio.to_thread(
                zp.reader.search_passages,
                query,
                max_articles=max_articles,
                max_chars=900,
            )
        except Exception:
            log.warning("knowledge_pack_zim_leg_failed", pack_id=pack_id, exc_info=True)
            return []
        # Clip to the requested candidate pool. Reranker downstream picks
        # the actual top-K. Without a clip, an article with 30 short
        # sections would dominate the candidate set and starve other
        # articles' passages from ever reaching the reranker.
        return [
            PackResult(
                content=p.content,
                title=p.title,
                section=p.section,
                url=p.url,
                source="zim",
                pack_id=pack_id,
            )
            for p in passages[:limit]
        ]

    @staticmethod
    def _rrf_merge(
        leg_results: list[list[PackResult]],
        k: int = 60,
    ) -> list[PackResult]:
        """Reciprocal Rank Fusion over multiple ranked result lists.

        A chunk that appears in multiple legs accumulates score from each.
        Dedup key is (pack_id, title, content[:200]) — same chunk across
        vector + FTS legs of the same pack collapses to one entry; ZIM
        articles dedup by (pack_id, url|title).
        """
        scores: dict[tuple, float] = {}
        items: dict[tuple, PackResult] = {}
        for results in leg_results:
            for rank_idx, item in enumerate(results):
                key = (item.pack_id, item.title, item.content[:200])
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank_idx + 1)
                # Keep the first-seen instance — content is identical for
                # vector/FTS hits on the same chunk.
                items.setdefault(key, item)

        ranked: list[PackResult] = []
        for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            item = items[key]
            item.score = score
            ranked.append(item)
        return ranked

    @staticmethod
    def _rerank(query: str, results: list[PackResult], top_k: int) -> list[PackResult]:
        """Cross-encoder rerank — falls back to RRF order on any failure."""
        from augmentum.memory.reranker import RerankService
        try:
            documents = [r.content for r in results]
            scored = RerankService.rerank(query, documents, top_k=top_k)
            reranked: list[PackResult] = []
            for orig_idx, score in scored:
                item = results[orig_idx]
                item.score = score
                reranked.append(item)
            return reranked
        except Exception:
            log.debug("knowledge_pack_rerank_failed_using_rrf_order", exc_info=True)
            return results[:top_k]

    # ------------------------------------------------------------------
    # Schema migration — lazy-build FTS5 mirror for legacy packs
    # ------------------------------------------------------------------

    async def _ensure_fts_index(
        self, conn: aiosqlite.Connection, pack_id: str,
    ) -> bool:
        """Ensure chunks_fts exists; rebuild from chunks for legacy packs.

        Called once per pack at scan time, BEFORE query_only=ON. Modern
        packs (built after the 2026-05 FTS rollout) ship with the table
        and trigger already; this is a one-time migration for packs
        imported under the prior schema.

        Returns True if the FTS leg is usable. False on rebuild failure —
        the pack still loads but search() will skip its FTS leg.
        """
        try:
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='chunks_fts'"
            )
            row = await cursor.fetchone()
            if row is not None:
                return True
        except Exception:
            log.warning("knowledge_pack_fts_check_failed", pack_id=pack_id, exc_info=True)
            return False

        # No FTS table — build it. Pack files are read-only at the
        # filesystem level only by convention; SQLite is happy to write
        # if we don't pin query_only=ON. Wrap in a transaction for
        # atomicity and a fast fail if the file genuinely is read-only.
        started = time.monotonic()
        try:
            await conn.execute("BEGIN")
            await conn.execute(
                "CREATE VIRTUAL TABLE chunks_fts USING fts5("
                "content, content=chunks, content_rowid=id)"
            )
            await conn.execute(
                "CREATE TRIGGER trg_pack_chunks_ai AFTER INSERT ON chunks BEGIN "
                "INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content); "
                "END"
            )
            # Backfill — the trigger covers future inserts; this populates
            # what already exists. SELECT id avoids relying on rowid alias
            # equality if the chunks table ever gets reshuffled.
            await conn.execute(
                "INSERT INTO chunks_fts(rowid, content) "
                "SELECT id, content FROM chunks"
            )
            await conn.execute("COMMIT")
        except Exception:
            try:
                await conn.execute("ROLLBACK")
            except Exception:
                # Rollback-of-rollback: original failure is logged below.
                # No additional action available here.
                log.debug("knowledge_pack_fts_rollback_failed", pack_id=pack_id)
            log.warning(
                "knowledge_pack_fts_rebuild_failed",
                pack_id=pack_id,
                exc_info=True,
            )
            return False

        elapsed = time.monotonic() - started
        try:
            cursor = await conn.execute("SELECT count(*) FROM chunks_fts")
            row = await cursor.fetchone()
            populated = int(row[0]) if row else 0
        except Exception:
            populated = -1
        log.info(
            "pack_fts_rebuilt",
            pack_id=pack_id,
            chunks=populated,
            seconds=round(elapsed, 2),
        )
        return True

    # ------------------------------------------------------------------
    # Activate / Deactivate / Delete
    # ------------------------------------------------------------------

    async def activate(self, pack_id: str) -> bool:
        """Activate a pack so it participates in searches."""
        pc = self._packs.get(pack_id)
        if pc is not None:
            pc.active = True
            self._active_state[pack_id] = True
            self._save_state()
            self._invalidate_search_cache()
            return True
        zp = self._zim_packs.get(pack_id)
        if zp is not None:
            zp.active = True
            self._active_state[pack_id] = True
            self._save_state()
            self._invalidate_search_cache()
            return True
        return False

    async def deactivate(self, pack_id: str) -> bool:
        """Deactivate a pack (keeps connection open, excludes from search)."""
        pc = self._packs.get(pack_id)
        if pc is not None:
            pc.active = False
            self._active_state[pack_id] = False
            self._save_state()
            self._invalidate_search_cache()
            return True
        zp = self._zim_packs.get(pack_id)
        if zp is not None:
            zp.active = False
            self._active_state[pack_id] = False
            self._save_state()
            self._invalidate_search_cache()
            return True
        return False

    async def delete(self, pack_id: str) -> bool:
        """Close connection and delete the pack file(s).

        A pack_id can be backed by an .augpack, a .zim, or BOTH (small
        packs that were converted but kept the source as a browse sidecar).
        Deletes whichever exist and returns True if any file was removed.
        Failure to delete one variant doesn't abort the other — the user
        asked to remove the pack, partial cleanup is better than orphaned
        files.
        """
        # Cache invalidation upfront — even if delete fails partway, the
        # search cache is now suspect.
        self._invalidate_search_cache()
        deleted_any = False
        delete_failed = False

        pc = self._packs.pop(pack_id, None)
        if pc is not None:
            try:
                await pc.conn.close()
            except Exception:
                log.warning("error closing pack before delete", pack_id=pack_id, exc_info=True)
            try:
                os.remove(pc.path)
                log.info("deleted pack", pack_id=pack_id, path=str(pc.path))
                deleted_any = True
            except OSError:
                log.warning("failed to delete pack file", pack_id=pack_id, path=str(pc.path), exc_info=True)
                delete_failed = True

        zp = self._zim_packs.pop(pack_id, None)
        if zp is not None:
            try:
                zp.reader.close()
            except Exception:
                log.warning("error closing zim before delete", pack_id=pack_id, exc_info=True)
            try:
                os.remove(zp.path)
                log.info("deleted zim pack", pack_id=pack_id, path=str(zp.path))
                deleted_any = True
            except OSError:
                log.warning("failed to delete zim file", pack_id=pack_id, path=str(zp.path), exc_info=True)
                delete_failed = True

        if deleted_any:
            self._active_state.pop(pack_id, None)
            self._save_state()
        # Only signal failure if we deleted nothing AND something was
        # supposed to be there. If one variant deleted cleanly and the
        # other failed (rare — would imply a half-removed pack), still
        # report success since the pack is functionally gone.
        if not deleted_any and delete_failed:
            return False
        return deleted_any

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _read_meta(self, conn: aiosqlite.Connection) -> PackMeta:
        """Read the ``meta`` key-value table into a PackMeta dataclass."""
        meta = PackMeta()
        try:
            cursor = await conn.execute("SELECT key, value FROM meta")
            rows = await cursor.fetchall()
            kv = {row[0]: row[1] for row in rows}
            meta.name = kv.get("name", "")
            meta.version = kv.get("version", "")
            meta.description = kv.get("description", "")
            meta.embedding_model = kv.get("embedding_model", "")
            meta.embedding_dim = int(kv.get("embedding_dim", 0))
            meta.chunk_count = int(kv.get("chunk_count", 0))
            meta.source_license = kv.get("source_license", "")
            meta.build_date = kv.get("build_date", "")
            meta.pack_kind = kv.get("pack_kind", "")
            meta.lang_code = kv.get("lang_code", "")
            meta.vocab_count = int(kv.get("vocab_count", 0) or 0)
        except Exception:
            log.warning("failed to read pack meta", exc_info=True)
        return meta

    @property
    def _state_path(self) -> Path:
        return self._pack_dir / ".pack_state.json"

    def _load_state(self) -> None:
        """Load persisted active/inactive flags from the pack directory."""
        if self._state_loaded:
            return
        self._state_loaded = True
        path = self._state_path
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            active = data.get("active", {})
            if isinstance(active, dict):
                self._active_state = {
                    str(pack_id): bool(is_active)
                    for pack_id, is_active in active.items()
                }
        except Exception:
            log.warning("knowledge_pack_state_load_failed", path=str(path), exc_info=True)
            self._active_state = {}

    def _save_state(self) -> None:
        """Persist active/inactive flags so restarts do not reset user choices."""
        try:
            self._pack_dir.mkdir(parents=True, exist_ok=True)
            active: dict[str, bool] = dict(self._active_state)
            active.update({pack_id: pc.active for pack_id, pc in self._packs.items()})
            active.update({pack_id: zp.active for pack_id, zp in self._zim_packs.items()})
            active.update({pack_id: lp.active for pack_id, lp in self._language_packs.items()})
            payload = {"active": active}
            path = self._state_path
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            log.warning("knowledge_pack_state_save_failed", path=str(self._state_path), exc_info=True)

    # ------------------------------------------------------------------
    # Failed conversion detection + cleanup
    # ------------------------------------------------------------------
    #
    # When a ZIM-to-augpack conversion crashes mid-embedding (OOM, server
    # restart, the user closes the tab on a 4-hour run), the install job
    # leaves three artifacts on disk:
    #   - the original .zim file
    #   - a 60KB empty .augpack shell
    #   - a stale .progress.json with the last-reported stage
    #
    # Without intervention the empty augpack would shadow the populated
    # ZIM (same stem) — the existing scan-time guard handles that. But
    # the user has no UI affordance to know the conversion died, no path
    # to retry it, and the .progress.json never gets cleaned. This block
    # surfaces those failures explicitly so users can recover.

    def _scan_failed_conversions(self) -> None:
        """Populate self.failed_conversions from two signals:

        1. ``.progress.json`` files older than 30 minutes (no active job).
           Catches workers killed by signals (OOM, SIGTERM on server restart)
           that didn't get to run their except block.
        2. ``.augpack`` files with chunks committed but no meta rows.
           Catches the common case where the worker DID run its except block
           and the parent install route then deleted the .progress.json (a
           bug in the original cleanup ordering). Without this, those
           failures left an orphaned partial augpack with no UI affordance.

        Each entry is tagged with ``resumable: bool`` indicating whether
        ``POST /api/knowledge/resume-failed`` can pick up where the failed
        run left off (requires paired .zim + at least one committed chunk +
        compatible embedding dim).
        """
        from augmentum.memory.embeddings import EmbeddingService

        self.failed_conversions = []
        if not self._pack_dir.exists():
            return
        now = time.time()

        # --- Source 1: stale .progress.json files ---
        # Track which pack_ids we've already surfaced so source 2 doesn't
        # duplicate them.
        seen_ids: set[str] = set()
        for entry in self._pack_dir.iterdir():
            if not entry.name.endswith(".progress.json"):
                continue
            try:
                age_s = now - entry.stat().st_mtime
                # Skip recent files — they may be live install jobs. The
                # 30-min threshold is generous; a real conversion of a
                # huge pack writes progress every few seconds, so going
                # 30 minutes silent is a strong "stuck or dead" signal.
                if age_s < 30 * 60:
                    continue
                payload = json.loads(entry.read_text(encoding="utf-8"))
                pack_id = entry.name[: -len(".progress.json")]
                augpack_path = entry.with_suffix("").with_suffix(".augpack")
                zim_path = entry.with_suffix("").with_suffix(".zim")
                fc = self._build_failed_conversion(
                    pack_id=pack_id,
                    augpack_path=augpack_path,
                    zim_path=zim_path,
                    progress_path=entry,
                    last_stage=payload.get("stage", ""),
                    last_progress=int(payload.get("current", 0)),
                    last_total=int(payload.get("total", 0)),
                    last_error=payload.get("error", ""),
                    stale_seconds=int(age_s),
                    embedding_dim=EmbeddingService.DIMENSION,
                )
                self.failed_conversions.append(fc)
                seen_ids.add(pack_id)
            except (OSError, json.JSONDecodeError):
                log.debug("failed_conversion_scan_skip", path=str(entry), exc_info=True)

        # --- Source 2: partial augpacks (chunks > 0, meta empty) ---
        # These are the failures whose progress.json got prematurely
        # deleted. We probe each augpack with a lightweight read; the
        # connections opened by scan() are separate (and skip these via
        # the partial-augpack guard) so there's no contention.
        for entry in self._pack_dir.iterdir():
            if entry.suffix != ".augpack":
                continue
            pack_id = entry.stem
            if pack_id in seen_ids:
                continue
            try:
                conn = sqlite3.connect(str(entry), timeout=2.0)
                try:
                    chunk_count = 0
                    meta_count = 0
                    try:
                        cur = conn.execute("SELECT COUNT(*) FROM chunks")
                        chunk_count = int(cur.fetchone()[0])
                    except sqlite3.Error:
                        pass
                    try:
                        cur = conn.execute("SELECT COUNT(*) FROM meta")
                        meta_count = int(cur.fetchone()[0])
                    except sqlite3.Error:
                        pass
                    if meta_count > 0:
                        # Healthy pack — handled by the normal scan path.
                        continue
                    # No meta row(s). Either chunks > 0 (resumable) or
                    # chunks == 0 (empty shell — discard only).
                    zim_path = entry.with_suffix(".zim")
                    fc = self._build_failed_conversion(
                        pack_id=pack_id,
                        augpack_path=entry,
                        zim_path=zim_path,
                        progress_path=None,
                        last_stage="embedding" if chunk_count > 0 else "unknown",
                        last_progress=chunk_count,
                        last_total=0,  # unknown without progress.json
                        last_error="Conversion did not finish (no meta written)",
                        stale_seconds=int(now - entry.stat().st_mtime),
                        embedding_dim=EmbeddingService.DIMENSION,
                    )
                    self.failed_conversions.append(fc)
                    seen_ids.add(pack_id)
                finally:
                    conn.close()
            except (sqlite3.Error, OSError):
                log.debug(
                    "failed_conversion_partial_probe_skip",
                    path=str(entry),
                    exc_info=True,
                )

        if self.failed_conversions:
            log.info(
                "knowledge_pack_failed_conversions_found",
                count=len(self.failed_conversions),
                pack_ids=[fc["pack_id"] for fc in self.failed_conversions],
                resumable=[fc["pack_id"] for fc in self.failed_conversions if fc["resumable"]],
            )

    def _build_failed_conversion(
        self,
        *,
        pack_id: str,
        augpack_path: Path,
        zim_path: Path,
        progress_path: Path | None,
        last_stage: str,
        last_progress: int,
        last_total: int,
        last_error: str,
        stale_seconds: int,
        embedding_dim: int,
    ) -> dict[str, Any]:
        """Build a failed-conversion record with resumability assessment.

        Resumable iff: (a) paired .zim exists, (b) augpack has at least one
        committed chunk so we have a real resume point, and (c) the existing
        chunks_vec dim matches the current embedding model. (c) protects
        against silently appending mismatched-dim vectors after a model
        upgrade.
        """
        augpack_exists = augpack_path.exists()
        zim_exists = zim_path.exists()
        chunks_committed = 0
        existing_dim = 0
        if augpack_exists:
            try:
                conn = sqlite3.connect(str(augpack_path), timeout=2.0)
                try:
                    # chunks_vec is a vec0 virtual table; the SELECT below
                    # returns nothing without sqlite_vec loaded. Without this
                    # load, existing_dim stays 0 (couldn't read) and the
                    # dim-compatibility check falls through to "compatible"
                    # — which lets a stale-model partial pack appear
                    # resumable and only fail at the worker stage. Loading
                    # the extension here gives an accurate verdict to the UI.
                    try:
                        conn.enable_load_extension(True)
                        conn.load_extension(sqlite_vec.loadable_path())
                        conn.enable_load_extension(False)
                    except sqlite3.Error:
                        # Extension load failed — fall through; existing_dim
                        # stays 0 which is a soft "skip the dim check" in
                        # the resumable calc. Worker probe will catch any
                        # real mismatch on actual resume attempt.
                        pass
                    try:
                        cur = conn.execute("SELECT COUNT(*) FROM chunks")
                        chunks_committed = int(cur.fetchone()[0])
                    except sqlite3.Error:
                        pass
                    if chunks_committed > 0:
                        try:
                            cur = conn.execute("SELECT embedding FROM chunks_vec LIMIT 1")
                            row = cur.fetchone()
                            if row is not None:
                                existing_dim = len(row[0]) // 4  # FLOAT32
                        except sqlite3.Error:
                            pass
                finally:
                    conn.close()
            except (sqlite3.Error, OSError):
                pass

        dim_compatible = existing_dim == 0 or existing_dim == embedding_dim
        resumable = (
            zim_exists
            and chunks_committed > 0
            and dim_compatible
        )
        not_resumable_reason = ""
        if not resumable:
            if not zim_exists:
                not_resumable_reason = "Original .zim file missing"
            elif chunks_committed == 0:
                not_resumable_reason = "No embedded chunks yet (start fresh)"
            elif not dim_compatible:
                not_resumable_reason = (
                    f"Embedding dim mismatch (existing {existing_dim}, "
                    f"current model {embedding_dim})"
                )

        return {
            "pack_id": pack_id,
            "progress_path": str(progress_path) if progress_path else "",
            "augpack_path": str(augpack_path) if augpack_exists else "",
            "augpack_size_bytes": augpack_path.stat().st_size if augpack_exists else 0,
            "zim_path": str(zim_path) if zim_exists else "",
            "zim_size_bytes": zim_path.stat().st_size if zim_exists else 0,
            "chunks_committed": chunks_committed,
            "existing_dim": existing_dim,
            "last_stage": last_stage,
            "last_progress": last_progress,
            "last_total": last_total,
            "last_error": last_error,
            "stale_seconds": stale_seconds,
            "resumable": resumable,
            "not_resumable_reason": not_resumable_reason,
        }

    async def discard_failed_conversion(self, pack_id: str) -> bool:
        """Remove the .progress.json + empty .augpack shell for a stuck
        conversion. The original .zim is untouched — the user can re-
        trigger conversion or use the ZIM as-is via the leg dispatcher.

        Returns True if anything was removed.
        """
        match = next(
            (fc for fc in self.failed_conversions if fc["pack_id"] == pack_id),
            None,
        )
        if match is None:
            return False
        removed = False
        for path_key in ("progress_path", "augpack_path"):
            path_str = match.get(path_key, "")
            if not path_str:
                continue
            path = Path(path_str)
            if not path.exists():
                continue
            try:
                path.unlink()
                removed = True
                log.info(
                    "knowledge_pack_failed_conversion_discarded",
                    pack_id=pack_id, path=str(path),
                )
            except OSError:
                log.warning(
                    "knowledge_pack_failed_conversion_discard_failed",
                    pack_id=pack_id, path=str(path), exc_info=True,
                )
        if removed:
            # Re-scan so the empty-augpack shadow goes away and the ZIM
            # surfaces cleanly. Also drops the failed entry from the
            # tracked list.
            await self.scan()
        return removed


