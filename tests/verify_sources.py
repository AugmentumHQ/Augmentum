"""Verify preferred_sources registry — test fetch accessibility for all non-AVOID domains.

Reports which domains are actually scrapeable for auto-fetch enrichment
and which should be downgraded to AVOID.

Usage:
    .venv/Scripts/python tests/verify_sources.py [OPTIONS]

    --quality LEVEL   Minimum quality to test: excellent, good, all (default: all)
    --timeout SECS    Per-fetch timeout (default: 10)
    --max N           Max domains to test (default: all)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from augmentum.tools.preferred_sources import (
    AVOID,
    EXCELLENT,
    GOOD,
    UNKNOWN,
    _SOURCES,
    SourceInfo,
)
from augmentum.tools.web_fetch import WebFetchTool

# Cloudflare / bot-block signatures in fetched content
_BLOCK_SIGNATURES = [
    "just a moment",
    "enable javascript and cookies",
    "checking your browser",
    "cloudflare",
    "ray id",
    "access denied",
    "403 forbidden",
    "captcha",
    "are you a robot",
    "please verify",
    "bot detection",
]

_QUALITY_NAMES = {EXCELLENT: "EXCELLENT", GOOD: "GOOD", UNKNOWN: "UNKNOWN", AVOID: "AVOID"}


def _is_blocked(content: str) -> bool:
    """Check if content looks like a bot-block page."""
    lower = content.lower()
    # Short content with block signature = likely blocked
    if len(content) < 200:
        return any(sig in lower for sig in _BLOCK_SIGNATURES)
    # Longer content but starts with block page
    first_500 = lower[:500]
    return any(sig in first_500 for sig in _BLOCK_SIGNATURES) and len(content) < 500


async def test_domain(
    tool: WebFetchTool,
    domain: str,
    info: SourceInfo,
    timeout: float,
) -> dict:
    """Test a single domain's accessibility."""
    url = f"https://{domain}"
    start = time.monotonic()

    try:
        result = await asyncio.wait_for(
            tool.execute(url=url, max_chars=1000),
            timeout=timeout,
        )
        elapsed = time.monotonic() - start

        if not result.success:
            return {
                "domain": domain,
                "status": "ERROR",
                "chars": 0,
                "elapsed": elapsed,
                "detail": (result.error or "")[:100],
                "quality": info.quality,
            }

        content = result.output or ""
        chars = len(content)

        if _is_blocked(content):
            return {
                "domain": domain,
                "status": "BLOCKED",
                "chars": chars,
                "elapsed": elapsed,
                "detail": content[:80].replace("\n", " "),
                "quality": info.quality,
            }

        if chars < 50:
            return {
                "domain": domain,
                "status": "EMPTY",
                "chars": chars,
                "elapsed": elapsed,
                "detail": content[:80].replace("\n", " "),
                "quality": info.quality,
            }

        return {
            "domain": domain,
            "status": "OK",
            "chars": chars,
            "elapsed": elapsed,
            "detail": content[:80].replace("\n", " "),
            "quality": info.quality,
        }

    except asyncio.TimeoutError:
        return {
            "domain": domain,
            "status": "TIMEOUT",
            "chars": 0,
            "elapsed": timeout,
            "detail": f"Timed out after {timeout}s",
            "quality": info.quality,
        }
    except Exception as e:
        return {
            "domain": domain,
            "status": "ERROR",
            "chars": 0,
            "elapsed": time.monotonic() - start,
            "detail": str(e)[:100],
            "quality": info.quality,
        }


async def run_tests(args: argparse.Namespace) -> None:
    min_quality = {"excellent": EXCELLENT, "good": GOOD, "all": UNKNOWN}.get(
        args.quality, UNKNOWN
    )

    # Collect domains to test (skip AVOID — they're already marked)
    domains: list[tuple[str, SourceInfo]] = []
    for domain, info in sorted(_SOURCES.items()):
        if info.quality == AVOID:
            continue
        if info.quality < min_quality:
            continue
        domains.append((domain, info))

    if args.max:
        domains = domains[:args.max]

    print(f"Testing {len(domains)} domains (min quality: {args.quality})")
    print(f"Timeout: {args.timeout}s per domain")
    print(f"{'=' * 90}\n")

    tool = WebFetchTool()

    # Test in batches of 10 for concurrency without overwhelming
    batch_size = 10
    all_results: list[dict] = []

    for i in range(0, len(domains), batch_size):
        batch = domains[i:i + batch_size]
        tasks = [test_domain(tool, d, info, args.timeout) for d, info in batch]
        results = await asyncio.gather(*tasks)
        all_results.extend(results)

        for r in results:
            q = _QUALITY_NAMES.get(r["quality"], "?")
            print(
                f"  [{r['status']:>7}] {q:>9} {r['domain']:<35} "
                f"{r['chars']:>5}ch {r['elapsed']:5.1f}s  {r['detail'][:40]}"
            )

        # Small delay between batches
        if i + batch_size < len(domains):
            await asyncio.sleep(0.5)

    # Summary
    print(f"\n{'=' * 90}")
    print("SUMMARY\n")

    by_status: dict[str, list[dict]] = {}
    for r in all_results:
        by_status.setdefault(r["status"], []).append(r)

    for status in ["OK", "BLOCKED", "TIMEOUT", "EMPTY", "ERROR"]:
        items = by_status.get(status, [])
        if items:
            print(f"  {status}: {len(items)}")

    print(f"\n  Total: {len(all_results)}")

    # Flag domains that should be downgraded
    downgrades = [
        r for r in all_results
        if r["status"] in ("BLOCKED", "TIMEOUT") and r["quality"] >= GOOD
    ]
    if downgrades:
        print(f"\n{'=' * 90}")
        print("RECOMMENDED DOWNGRADES (currently GOOD+ but inaccessible):\n")
        for r in downgrades:
            q = _QUALITY_NAMES.get(r["quality"], "?")
            print(f"  {q:>9} → AVOID  {r['domain']:<35} ({r['status']})")

    empties = [r for r in all_results if r["status"] == "EMPTY" and r["quality"] >= GOOD]
    if empties:
        print(f"\nLOW CONTENT (currently GOOD+ but returned <50 chars):\n")
        for r in empties:
            q = _QUALITY_NAMES.get(r["quality"], "?")
            print(f"  {q:>9} → check  {r['domain']:<35} ({r['chars']} chars)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify preferred_sources accessibility")
    parser.add_argument("--quality", default="all", choices=["excellent", "good", "all"])
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max", type=int, default=0, help="Max domains to test (0=all)")
    args = parser.parse_args()

    asyncio.run(run_tests(args))


if __name__ == "__main__":
    main()
