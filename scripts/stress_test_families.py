"""Stress-test the Augmentum engine across model families.

Exercises the engine v2 (llama-server) path end-to-end through the proxy's
HTTP API. Picks one representative model from each family bucket, times load
and first-token latency, generates a canned prompt, and verifies that the
reasoning-channel parser correctly extracts hidden reasoning (for reasoning
models) or leaves content alone (for non-reasoning models).

Run before/after a llama-server bump or reasoning parser change to catch
regressions.

Usage:
    python scripts/stress_test_families.py \\
        --base-url http://localhost:6100 \\
        --token $AUGMENTUM_API_TOKEN

Output: JSON report to stdout. Exit code 0 if every bucket passed, else 1.

The bucket list is intentionally small (one model per bucket) so a full run
completes in ~5-10 min. Extend BUCKETS below for wider coverage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Buckets — one representative GGUF per family. Edit this list to grow the
# coverage; the harness scales linearly with its size.
# ---------------------------------------------------------------------------


@dataclass
class Bucket:
    name: str                     # Human label, e.g. "gemma4-small"
    model_match: str              # Substring match against /v1/models ids
    family_expected: str | None   # Expected detect_reasoning_family result
    reasoning_expected: bool      # Should reasoning text be non-empty?
    prompt: str                   # What to ask the model
    timeout_s: float = 120.0      # Per-model cap


BUCKETS: tuple[Bucket, ...] = (
    Bucket(
        name="tiny-dense",
        model_match="Qwen3.5-0.8B",
        family_expected="qwen35",
        reasoning_expected=False,
        prompt="Say hello in exactly three words.",
        timeout_s=60,
    ),
    Bucket(
        name="small-dense-nonreasoning",
        model_match="Nemotron-3-Nano-4B",
        family_expected=None,  # unknown family, default parsers
        reasoning_expected=False,
        prompt="Say hello in exactly three words.",
        timeout_s=60,
    ),
    Bucket(
        name="gemma3-dense",
        model_match="gemma-3-4b",
        family_expected="gemma3",
        reasoning_expected=False,  # Gemma 3 non-thinking variants
        prompt="Say hello in exactly three words.",
        timeout_s=60,
    ),
    Bucket(
        name="gemma4-small",
        model_match="gemma-4-E4B",
        family_expected="gemma4",
        reasoning_expected=True,   # Gemma 4 E4B/B sizes include thinking
        prompt="What is 17 + 25? Think step by step.",
        timeout_s=90,
    ),
    Bucket(
        name="qwen35-reasoning",
        model_match="Qwen3.5-4B",
        family_expected="qwen35",
        reasoning_expected=True,
        prompt="What is 17 + 25? Think step by step.",
        timeout_s=90,
    ),
    Bucket(
        name="qwen35moe",
        model_match="Qwen3.6-35B-A3B",
        family_expected="qwen35",  # Qwen 3.6 brand reuses qwen35 arch
        reasoning_expected=True,
        prompt="What is 17 + 25? Think step by step.",
        timeout_s=180,
    ),
    Bucket(
        # Nemotron 3 Nano Omni 30B-A3B — Mamba2-Transformer hybrid MoE.
        # Covers the nemotron_h_moe arch path (vs Nemotron 3 Nano text
        # which is non-MoE nemotron_h). Catches regressions in:
        #   * thinking parser family lookup (_FAMILY_PARSERS + name needles)
        #   * MoE expert offload autofit for hybrid Mamba/transformer layer mix
        #   * enable_thinking + reasoning_budget chat-template kwarg forwarding
        # Adjust model_match to whichever Nemotron Omni quant you keep on disk;
        # UD-Q4_K_M (~23.9 GB) is the recommended sweet spot per Unsloth.
        name="nemotron3-omni-moe",
        model_match="Nemotron-3-Nano-Omni-30B-A3B",
        family_expected="nemotron_h_moe",
        reasoning_expected=True,
        prompt="What is 17 + 25? Think step by step.",
        timeout_s=240,  # Hybrid MoE + first-token expert fetch over PCIe
    ),
)


# ---------------------------------------------------------------------------
# Result shape
# ---------------------------------------------------------------------------


@dataclass
class BucketResult:
    bucket: str
    model: str | None = None
    status: str = "pending"           # pending / pass / fail / skip
    reason: str = ""
    load_ms: float | None = None
    ttft_ms: float | None = None
    total_ms: float | None = None
    tokens: int = 0
    tok_per_s: float | None = None
    reasoning_chars: int = 0
    content_chars: int = 0
    reasoning_sample: str = ""
    content_sample: str = ""
    detected_family: str | None = None


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class Harness:
    def __init__(
        self, base_url: str, token: str | None, verbose: bool = False
    ) -> None:
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._verbose = verbose
        self._client = httpx.AsyncClient(
            headers=self._headers, timeout=300, http2=False
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _log(self, *args: Any) -> None:
        if self._verbose:
            print("[stress]", *args, file=sys.stderr)

    async def list_models(self) -> list[str]:
        r = await self._client.get(f"{self._base}/v1/models")
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]

    async def load_model(self, model: str) -> float:
        t0 = time.monotonic()
        r = await self._client.post(f"{self._base}/api/models/{model}/load")
        r.raise_for_status()
        body = r.json()
        if not body.get("success", False):
            raise RuntimeError(f"load failed: {body}")
        return (time.monotonic() - t0) * 1000.0

    async def unload_model(self, model: str) -> None:
        try:
            await self._client.post(f"{self._base}/api/models/{model}/unload")
        except Exception as exc:
            self._log(f"unload warning: {exc}")

    async def generate(
        self, model: str, prompt: str, timeout_s: float
    ) -> tuple[float | None, float, int, str, str]:
        """Stream a single-turn generation.

        Returns (ttft_ms, total_ms, tokens, content_text, reasoning_text).
        """
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": 256,
        }
        t_start = time.monotonic()
        ttft: float | None = None
        tokens = 0
        content_buf: list[str] = []
        reasoning_buf: list[str] = []

        async with self._client.stream(
            "POST",
            f"{self._base}/v1/chat/completions",
            json=payload,
            timeout=timeout_s,
        ) as resp:
            resp.raise_for_status()
            async for raw in resp.aiter_lines():
                if not raw.startswith("data: "):
                    continue
                data = raw[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {}) or {}
                if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                    ttft = (time.monotonic() - t_start) * 1000.0
                if delta.get("content"):
                    content_buf.append(delta["content"])
                    tokens += 1
                reasoning = delta.get("reasoning_content") or delta.get("thinking")
                if reasoning:
                    reasoning_buf.append(reasoning)

        total_ms = (time.monotonic() - t_start) * 1000.0
        return ttft, total_ms, tokens, "".join(content_buf), "".join(reasoning_buf)

    async def run_bucket(
        self, bucket: Bucket, available_models: list[str]
    ) -> BucketResult:
        result = BucketResult(bucket=bucket.name)
        pat = re.compile(re.escape(bucket.model_match), re.IGNORECASE)
        candidates = [m for m in available_models if pat.search(m)]
        if not candidates:
            result.status = "skip"
            result.reason = f"no model matching '{bucket.model_match}' in /v1/models"
            return result
        model = candidates[0]
        result.model = model
        self._log(f"bucket {bucket.name} → {model}")

        try:
            # Independent-family detection (same path as the proxy uses).
            from augmentum.utils.thinking import detect_reasoning_family
            result.detected_family = detect_reasoning_family(model=model)
            if (
                bucket.family_expected is not None
                and result.detected_family != bucket.family_expected
            ):
                result.status = "fail"
                result.reason = (
                    f"family mismatch: expected {bucket.family_expected}, "
                    f"got {result.detected_family}"
                )
                return result

            try:
                result.load_ms = await asyncio.wait_for(
                    self.load_model(model), timeout=bucket.timeout_s
                )
            except Exception as exc:
                result.status = "fail"
                result.reason = f"load error: {exc}"
                return result

            try:
                ttft, total_ms, tokens, content, reasoning = await asyncio.wait_for(
                    self.generate(model, bucket.prompt, bucket.timeout_s),
                    timeout=bucket.timeout_s,
                )
            except Exception as exc:
                result.status = "fail"
                result.reason = f"generation error: {exc}"
                return result

            result.ttft_ms = ttft
            result.total_ms = total_ms
            result.tokens = tokens
            if total_ms > 0 and tokens > 0:
                result.tok_per_s = tokens / (total_ms / 1000.0)
            result.content_chars = len(content)
            result.reasoning_chars = len(reasoning)
            result.content_sample = content[:200]
            result.reasoning_sample = reasoning[:200]

            if bucket.reasoning_expected and not reasoning:
                result.status = "fail"
                result.reason = (
                    "reasoning expected but none extracted — family-dispatch "
                    "or parser regression?"
                )
                return result
            if not content:
                result.status = "fail"
                result.reason = "no content produced"
                return result

            # Leak check: control-token markers should never appear in clean content.
            leak_markers = (
                "<think>", "</think>",
                "<|channel|>", "<|message|>", "<|end|>",
                "<|channel>thought", "<channel|>",
            )
            for marker in leak_markers:
                if marker in content:
                    result.status = "fail"
                    result.reason = f"control token leaked into content: '{marker}'"
                    return result

            result.status = "pass"
        finally:
            await self.unload_model(model)

        return result

    async def run_all(self) -> list[BucketResult]:
        available = await self.list_models()
        self._log(f"{len(available)} models available")
        results: list[BucketResult] = []
        for b in BUCKETS:
            results.append(await self.run_bucket(b, available))
        return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default=os.environ.get("AUGMENTUM_URL", "http://localhost:6100"))
    ap.add_argument("--token", default=os.environ.get("AUGMENTUM_API_TOKEN"))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--include", help="Only run buckets whose name contains this substring")
    args = ap.parse_args()

    global BUCKETS
    if args.include:
        BUCKETS = tuple(b for b in BUCKETS if args.include in b.name)

    async def _main() -> int:
        h = Harness(args.base_url, args.token, verbose=args.verbose)
        try:
            results = await h.run_all()
        finally:
            await h.aclose()

        summary = {
            "ran":     len(results),
            "passed":  sum(1 for r in results if r.status == "pass"),
            "failed":  sum(1 for r in results if r.status == "fail"),
            "skipped": sum(1 for r in results if r.status == "skip"),
        }
        print(json.dumps(
            {"summary": summary, "buckets": [asdict(r) for r in results]},
            indent=2,
        ))
        return 0 if summary["failed"] == 0 else 1

    return asyncio.run(_main())


if __name__ == "__main__":
    sys.exit(main())
