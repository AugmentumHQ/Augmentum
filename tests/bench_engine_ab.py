"""A/B benchmark: raw llama.cpp defaults vs Augmentum Engine optimizations.

Tests each optimization toggle independently to measure exact impact.
Run from host: python tests/bench_engine_ab.py

Requires engine running at localhost:8090.
"""

from __future__ import annotations

import json
import time
import urllib.request

ENGINE_URL = "http://localhost:8090"

# Models to test (adjust to what you have loaded)
TEST_MODELS = [
    {"name": "Nemotron 4B", "id": "NVIDIA-Nemotron-3-Nano-4B-Q4_K_M.gguf", "type": "dense"},
    {"name": "heretic-27b", "id": "heretic-27b-Q4_K_M.gguf", "type": "dense"},
    {"name": "Qwen3-30B MoE", "id": "Qwen3-30B-A3B-Q4_K_M.gguf", "type": "moe"},
]

# Simple prompt that doesn't trigger thinking/reasoning
BENCH_PROMPT = "Count from 1 to 50, one number per line."
BENCH_TOKENS = 150


def api_post(path: str, body: dict, timeout: int = 300) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{ENGINE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def api_get(path: str) -> dict:
    try:
        with urllib.request.urlopen(f"{ENGINE_URL}{path}", timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def load_model(model_id: str) -> bool:
    resp = api_post("/v1/models/load", {"model": model_id})
    return "error" not in resp and resp.get("status") == "loaded"


def bench_generate(n_tokens: int = BENCH_TOKENS, runs: int = 3) -> dict:
    """Generate tokens and return avg tok/s across runs."""
    results = []
    for run in range(runs):
        t0 = time.time()
        resp = api_post("/v1/chat/completions", {
            "model": "test",
            "messages": [{"role": "user", "content": BENCH_PROMPT}],
            "max_tokens": n_tokens,
            "stream": False,
        })
        elapsed = time.time() - t0

        if "error" in resp:
            results.append({"error": resp["error"]})
            continue

        usage = resp.get("usage", {})
        comp = usage.get("completion_tokens", 0)
        prompt = usage.get("prompt_tokens", 0)

        if comp > 0:
            tps = comp / elapsed
            results.append({
                "run": run + 1,
                "completion_tokens": comp,
                "prompt_tokens": prompt,
                "elapsed_ms": round(elapsed * 1000),
                "tok_s": round(tps, 1),
            })
        else:
            results.append({"run": run + 1, "error": "0 tokens", "elapsed_ms": round(elapsed * 1000)})

    # Compute average
    valid = [r for r in results if "tok_s" in r]
    if valid:
        avg_tps = sum(r["tok_s"] for r in valid) / len(valid)
        avg_ms = sum(r["elapsed_ms"] for r in valid) / len(valid)
    else:
        avg_tps = 0
        avg_ms = 0

    return {
        "runs": results,
        "avg_tok_s": round(avg_tps, 1),
        "avg_ms": round(avg_ms),
        "n_runs": len(valid),
    }


def get_engine_config() -> dict:
    status = api_get("/v1/engine/status")
    if "error" in status:
        return status
    return {
        "model": status.get("model", {}).get("name", "?"),
        "kv_cache_type": status.get("config", {}).get("kv_cache_type", "?"),
        "features": {
            k: v for k, v in status.get("features", {}).items()
            if isinstance(v, bool) and v
        },
        "pool": status.get("model_pool", {}),
    }


def run_benchmark():
    print("=" * 60)
    print("AUGMENTUM ENGINE A/B BENCHMARK")
    print("=" * 60)
    print()

    # Engine config
    config = get_engine_config()
    print("Engine config:")
    print(f"  KV cache: {config.get('kv_cache_type', '?')}")
    print(f"  Features: {config.get('features', {})}")
    print()

    # Test each model
    results = {}
    for model in TEST_MODELS:
        print(f"--- {model['name']} ({model['type']}) ---")

        # Load model
        print(f"  Loading {model['id']}...", end=" ", flush=True)
        t0 = time.time()
        ok = load_model(model["id"])
        load_ms = round((time.time() - t0) * 1000)

        if not ok:
            print(f"FAILED ({load_ms}ms)")
            results[model["name"]] = {"error": "load failed", "load_ms": load_ms}
            continue

        # Check if pool hit
        status = api_get("/v1/engine/status")
        pool_hit = "from_pool" in str(status)
        print(f"OK ({load_ms}ms, pool={'hit' if pool_hit else 'miss'})")

        # Warmup
        print("  Warmup...", end=" ", flush=True)
        bench_generate(n_tokens=20, runs=1)
        print("done")

        # Benchmark
        print(f"  Benchmarking ({BENCH_TOKENS} tokens, 3 runs)...", end=" ", flush=True)
        bench = bench_generate(n_tokens=BENCH_TOKENS, runs=3)
        print(f"{bench['avg_tok_s']} tok/s (avg of {bench['n_runs']} runs)")

        for r in bench["runs"]:
            if "tok_s" in r:
                print(f"    Run {r['run']}: {r['tok_s']} tok/s ({r['completion_tokens']} tokens, {r['elapsed_ms']}ms)")
            else:
                print(f"    Run {r['run']}: {r.get('error', 'unknown')}")

        results[model["name"]] = {
            "type": model["type"],
            "load_ms": load_ms,
            "avg_tok_s": bench["avg_tok_s"],
            "runs": bench["runs"],
        }
        print()

    # Summary table
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Type':<6} {'Load':<8} {'tok/s':<8}")
    print("-" * 50)
    for name, data in results.items():
        if "error" in data:
            print(f"{name:<25} {'?':<6} {data.get('load_ms','?'):<8} ERROR")
        else:
            print(f"{name:<25} {data['type']:<6} {data['load_ms']}ms{'':<3} {data['avg_tok_s']}")
    print()
    print(f"Config: KV={config.get('kv_cache_type', '?')}")
    print("To compare: change ENGINE_KV_CACHE_TYPE, ENGINE_EXPERT_CACHE, etc. and re-run")

    # Save results
    out_file = "tests/bench_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": config,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    run_benchmark()
