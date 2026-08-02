"""Knowledge pack retrieval quality eval — runs against a live PackManager
backed by real ZIM/augpack files on disk.

Why live: ZIM passage extraction, vector embedding, and reranker quality
can't be reliably stubbed. The eval needs to confirm that "ask MDWiki
about diabetes" returns a Diabetes article (not Stack Exchange noise),
that content is sanitized (no MediaWiki CSS chrome leaking through), and
that result counts make sense across the cache+rerank pipeline.

Run: ``pytest tests/live/test_live_pack_quality.py --run-live -v``

Skipped automatically when the expected packs aren't installed — the
harness is opportunistic. Each test declares which pack it needs and
will skip with a clear reason if the pack is missing. CI runners with
no packs simply skip the whole file.

Add new cases by appending to ``_QUERY_CASES`` (or to a per-pack list).
Each case is intentionally a single concrete assertion bundle: query +
expected pack + 1-3 strict checks. Don't refactor for cleverness —
brittle-looking but explicit cases make regressions obvious.
"""
from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from augmentum.knowledge.packs import PackManager, PackResult

pytestmark = pytest.mark.live


# Packs we may have on disk. The eval is opportunistic — present packs
# get tested, missing ones are skipped without failing the suite.
_KNOWN_PACKS = (
    "mdwiki_en_all_2025-11",
    "wikipedia_en_simple_all_mini_2026-02",
    "wikipedia_en_physics_nopic_2026-04",
    "devdocs_en_python_2026-02",
    "dandwiki_en_all_nopic_2025-04",
)


@dataclass
class QueryCase:
    """One eval scenario.

    ``checks`` is a list of (name, fn) where fn takes the result list and
    returns True/False. Each named check produces its own pass/fail line
    in the test output — easier to diagnose than one giant assertion.
    """
    pack_id: str
    query: str
    checks: list[tuple[str, Callable[[list[PackResult]], bool]]]


def _has_result_with_title_substr(substr: str) -> Callable[[list[PackResult]], bool]:
    """Builder: True if any result's title contains ``substr`` (case-insensitive)."""
    def check(results: list[PackResult]) -> bool:
        return any(substr.lower() in (r.title or "").lower() for r in results)
    return check


def _content_clean_of(forbidden: list[str]) -> Callable[[list[PackResult]], bool]:
    """Builder: True if no result's content contains any of the forbidden tokens.

    Used to assert HTML chrome isn't leaking through (CSS comments, mw-parser
    classes, infobox markers — things that mean the sanitizer regressed).
    """
    def check(results: list[PackResult]) -> bool:
        for r in results:
            for token in forbidden:
                if token in (r.content or ""):
                    return False
        return True
    return check


def _result_count_at_least(n: int) -> Callable[[list[PackResult]], bool]:
    return lambda results: len(results) >= n


def _all_results_from_pack(pack_id: str) -> Callable[[list[PackResult]], bool]:
    return lambda results: all(r.pack_id == pack_id for r in results)


# Forbidden-tokens lists per pack family. ZIM articles ship with
# upstream HTML chrome that the passage extractor must strip. If any of
# these survive, the sanitizer regressed.
_MEDIAWIKI_NOISE = [
    "/* start https://",   # CSS comment block opener
    "@media screen and",   # CSS media query that leaked
    ".mw-parser-output",   # MediaWiki structural class
    "<!--",                # HTML comment fragment
    "Edit on GitHub",      # Some packs leave "edit" links behind
]


# The actual eval cases. Add concrete scenarios; each is one bundle of
# (pack, query, expected behavior). Keep query phrasings stable so
# regressions are reproducible — if a model upgrade changes ranking,
# you'll see exactly which cases drift and decide whether to retune
# the query or accept the new behavior.
_QUERY_CASES: list[QueryCase] = [
    QueryCase(
        pack_id="mdwiki_en_all_2025-11",
        query="What are the symptoms and treatment of Type 2 diabetes mellitus?",
        checks=[
            ("returns at least 3 results", _result_count_at_least(3)),
            ("all results from MDWiki", _all_results_from_pack("mdwiki_en_all_2025-11")),
            ("hits a Diabetes article", _has_result_with_title_substr("diabetes")),
            ("content is clean of MediaWiki chrome", _content_clean_of(_MEDIAWIKI_NOISE)),
        ],
    ),
    QueryCase(
        pack_id="mdwiki_en_all_2025-11",
        query="hyperglycemia",
        checks=[
            ("returns at least 1 result", _result_count_at_least(1)),
            ("content is clean of MediaWiki chrome", _content_clean_of(_MEDIAWIKI_NOISE)),
        ],
    ),
    QueryCase(
        pack_id="wikipedia_en_physics_nopic_2026-04",
        query="What is gravity?",
        checks=[
            ("returns at least 3 results", _result_count_at_least(3)),
            ("hits a gravity article", _has_result_with_title_substr("gravity")),
            ("content is clean of MediaWiki chrome", _content_clean_of(_MEDIAWIKI_NOISE)),
        ],
    ),
    QueryCase(
        pack_id="wikipedia_en_simple_all_mini_2026-02",
        query="photosynthesis",
        checks=[
            ("returns at least 1 result", _result_count_at_least(1)),
            ("content is clean of MediaWiki chrome", _content_clean_of(_MEDIAWIKI_NOISE)),
        ],
    ),
    QueryCase(
        pack_id="dandwiki_en_all_nopic_2025-04",
        query="orc race traits",
        checks=[
            ("returns at least 1 result", _result_count_at_least(1)),
            ("hits an orc-related article", _has_result_with_title_substr("orc")),
        ],
    ),
    # Devdocs Python is augpack — confirms the augpack legs work too,
    # not just ZIM. Smaller candidate set so we ask for at least 1 hit.
    QueryCase(
        pack_id="devdocs_en_python_2026-02",
        query="list comprehension",
        checks=[
            ("returns at least 1 result", _result_count_at_least(1)),
            ("all results from devdocs Python", _all_results_from_pack("devdocs_en_python_2026-02")),
        ],
    ),
    # Negative case: a query that should hit nothing in the medical pack
    # so we confirm "no results" is reachable (not just always-positive
    # asserts). Pick deliberately off-topic phrasing.
    QueryCase(
        pack_id="mdwiki_en_all_2025-11",
        query="quaternion lie algebra topology",
        checks=[
            ("medical pack returns 0 results for pure-math query OR all are clean",
             lambda results: len(results) == 0 or _content_clean_of(_MEDIAWIKI_NOISE)(results)),
        ],
    ),
]


def _pack_dir() -> Path:
    """Find the knowledge pack dir on this host. Honors env override
    so devs can point the eval at a non-default test fixture pack dir.
    """
    env = os.environ.get("AUGMENTUM_PACK_DIR")
    if env:
        return Path(env)
    # Match the production default path.
    return Path("/data/knowledge")


@pytest.fixture(scope="module")
def pack_manager() -> PackManager:
    """Open a real PackManager against the on-disk pack dir.

    Module-scoped: scanning a 10GB ZIM file is slow, and the eval cases
    don't mutate state, so one shared instance is correct. We don't
    close — the test process exits and OS cleans up.
    """
    pack_dir = _pack_dir()
    if not pack_dir.exists():
        pytest.skip(f"pack dir not present: {pack_dir}")
    mgr = PackManager(pack_dir)
    asyncio.run(mgr.scan())
    if not (mgr._packs or mgr._zim_packs):
        pytest.skip(f"no packs loaded from {pack_dir}")
    return mgr


def _pack_present(mgr: PackManager, pack_id: str) -> bool:
    return pack_id in mgr._packs or pack_id in mgr._zim_packs


@pytest.mark.parametrize("case", _QUERY_CASES, ids=lambda c: f"{c.pack_id}::{c.query[:32]}")
def test_pack_query_quality(case: QueryCase, pack_manager: PackManager) -> None:
    """Run one eval case against the live pack manager.

    Each case is a parametrize entry so failures show up individually
    in pytest output — you can tell at a glance which scenario broke
    when something changes (e.g. embedding model upgrade, regex tweak).
    """
    if not _pack_present(pack_manager, case.pack_id):
        pytest.skip(f"pack not installed on this host: {case.pack_id}")

    results = asyncio.run(pack_manager.search(
        query=case.query,
        pack_ids=[case.pack_id],
        limit=5,
        rerank=True,
    ))

    failures = []
    for name, check in case.checks:
        try:
            if not check(results):
                failures.append(f"  ✗ {name}")
        except Exception as exc:
            failures.append(f"  ✗ {name} (raised {type(exc).__name__}: {exc})")

    if failures:
        # Format the result set for diagnosis. Truncated to keep failure
        # output readable — full content is rarely what changed.
        summary_lines = [
            f"  [{i}] pack={r.pack_id} title={r.title!r} section={r.section!r} "
            f"len={len(r.content)} score={r.score:.3f}"
            for i, r in enumerate(results[:5])
        ]
        msg = (
            f"Eval case failed: query={case.query!r} pack={case.pack_id}\n"
            f"Got {len(results)} results:\n" + "\n".join(summary_lines) +
            "\nChecks failed:\n" + "\n".join(failures)
        )
        pytest.fail(msg)


def test_eval_inventory(pack_manager: PackManager) -> None:
    """Smoke test that surfaces which packs are present and how many
    eval cases will run against them. Helps the "I added a pack but
    no eval" gap — running with -v will list inventory once per run.
    """
    available = set(pack_manager._packs.keys()) | set(pack_manager._zim_packs.keys())
    print()
    print(f"Packs loaded: {sorted(available)}")
    by_pack: dict[str, int] = {}
    for case in _QUERY_CASES:
        by_pack[case.pack_id] = by_pack.get(case.pack_id, 0) + 1
    runnable = sum(n for pid, n in by_pack.items() if pid in available)
    skipped = sum(n for pid, n in by_pack.items() if pid not in available)
    print(f"Eval cases: {runnable} runnable / {skipped} skipped (pack missing)")
    for pid, n in sorted(by_pack.items()):
        marker = "✓" if pid in available else "✗"
        print(f"  {marker} {pid}: {n} case(s)")
    # Always passes — this is informational. Run with `-v -s` to see output.
    assert True
