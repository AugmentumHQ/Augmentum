#!/usr/bin/env python3
"""MoE Expert Offload Benchmark Suite.

Tests loading and inference for MoE models with expert offloading.
Measures VRAM usage, cold start, warm throughput, and sustained tok/s.

Usage:
    python tests/bench_moe.py [--url URL] [--model MODEL]

Example:
    python tests/bench_moe.py --model MiniMax-M2.5-Q4_K_M-00001-of-00004.gguf
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import httpx


def get_json(url: str, timeout: float = 10) -> dict:
    r = httpx.get(url, timeout=timeout)
    return r.json()


def post_json(url: str, data: dict, timeout: float = 300) -> dict:
    r = httpx.post(url, json=data, timeout=timeout)
    return r.json()


def get_vram_mb() -> int:
    """Get GPU VRAM usage via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return -1


def bench_generation(base: str, prompt: str, max_tokens: int, temperature: float = 0) -> dict:
    """Run a single generation and return timing + usage."""
    t0 = time.time()
    r = post_json(f"{base}/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }, timeout=600)
    elapsed = time.time() - t0

    usage = r.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    content = r.get("choices", [{}])[0].get("message", {}).get("content", "")

    return {
        "elapsed_s": round(elapsed, 2),
        "completion_tokens": completion_tokens,
        "tok_s": round(completion_tokens / max(elapsed, 0.001), 1),
        "content_preview": content[:200],
    }


def run_benchmark(base: str, model: str | None = None):
    """Full MoE benchmark suite."""
    print("=" * 60)
    print("MoE Expert Offload Benchmark")
    print("=" * 60)

    # Health check
    health = get_json(f"{base}/health")
    print(f"\nEngine: {health.get('status', 'unknown')}")

    # VRAM before loading
    vram_before = get_vram_mb()
    print(f"VRAM before load: {vram_before} MB")

    # List models
    models = get_json(f"{base}/v1/models")
    print("\nAvailable models:")
    for m in models.get("data", []):
        size_gb = m.get("size_bytes", 0) / (1024**3)
        shards = m.get("shards", 1)
        loaded = " (loaded)" if m.get("loaded") else ""
        shard_str = f" [{shards} shards]" if shards > 1 else ""
        print(f"  {m['id']}{shard_str} — {size_gb:.1f} GB{loaded}")

    # Load model if specified
    if model:
        print(f"\n--- Loading {model} ---")
        t0 = time.time()
        result = post_json(f"{base}/v1/models/load", {"model": model}, timeout=300)
        load_time = time.time() - t0
        print(f"Load result: {result.get('status', 'unknown')}")
        print(f"Load time: {load_time:.1f}s")

    # VRAM after loading
    vram_after = get_vram_mb()
    print(f"VRAM after load: {vram_after} MB")
    if vram_before > 0 and vram_after > 0:
        print(f"VRAM delta: +{vram_after - vram_before} MB")

    # Expert stats
    expert_stats = get_json(f"{base}/v1/engine/experts/stats")
    print(f"\nExpert offload: {json.dumps(expert_stats, indent=2)}")

    # Benchmark: Cold start (first request)
    print("\n--- Cold Start (first request after load) ---")
    cold = bench_generation(base, "What is 2+2? Answer briefly.", 30)
    print(f"Cold: {cold['completion_tokens']} tokens in {cold['elapsed_s']}s = {cold['tok_s']} tok/s")
    print(f"Preview: {cold['content_preview'][:100]}")

    # Warmup
    print("\n--- Warmup (3 requests) ---")
    for i in range(3):
        w = bench_generation(base, f"Count to {i+3}", 20)
        print(f"  Warmup {i+1}: {w['tok_s']} tok/s")

    # Benchmark: Sustained throughput
    print("\n--- Sustained Throughput (3 runs × 100 tokens) ---")
    prompt = "Explain quantum computing in detail. Cover qubits, superposition, entanglement, and quantum gates."
    results = []
    for i in range(3):
        r = bench_generation(base, prompt, 100)
        results.append(r)
        print(f"  Run {i+1}: {r['completion_tokens']} tokens in {r['elapsed_s']}s = {r['tok_s']} tok/s")

    avg_tps = sum(r['tok_s'] for r in results) / len(results)
    print(f"\n  Average: {avg_tps:.1f} tok/s")

    # Benchmark: Long generation
    print("\n--- Long Generation (500 tokens) ---")
    long_prompt = "Write a detailed essay about the history of artificial intelligence from the 1950s to today."
    long_result = bench_generation(base, long_prompt, 500)
    print(f"  {long_result['completion_tokens']} tokens in {long_result['elapsed_s']}s = {long_result['tok_s']} tok/s")

    # Final VRAM
    vram_final = get_vram_mb()
    print(f"\nVRAM after benchmark: {vram_final} MB")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model:              {model or 'pre-loaded'}")
    print(f"VRAM used:          {vram_after} MB")
    print(f"Expert offload:     {expert_stats.get('enabled', False)}")
    if expert_stats.get('n_experts'):
        print(f"Experts:            {expert_stats['n_experts']} total, {expert_stats.get('n_experts_active', '?')} active")
        print(f"Model size:         {expert_stats.get('model_size_gb', '?')} GB")
        print(f"Expert weights:     {expert_stats.get('expert_weights_gb', '?')} GB (on CPU/SSD)")
        print(f"Shared weights:     {expert_stats.get('shared_weights_gb', '?')} GB (on GPU)")
    print(f"Cold start:         {cold['tok_s']} tok/s")
    print(f"Sustained (avg):    {avg_tps:.1f} tok/s")
    print(f"Long gen (500 tok): {long_result['tok_s']} tok/s")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="MoE Expert Offload Benchmark")
    parser.add_argument("--url", default="http://localhost:8090", help="Engine base URL")
    parser.add_argument("--model", default=None, help="Model to load (GGUF filename)")
    args = parser.parse_args()

    run_benchmark(args.url, args.model)


if __name__ == "__main__":
    main()
