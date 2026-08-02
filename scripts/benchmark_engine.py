"""Engine v2 benchmark — test load time, tok/s, VRAM across model sizes.

Usage:
    python scripts/benchmark_engine.py [--host HOST] [--cookie COOKIE]

Tests a range of models from tiny to large, including MoE,
measuring load time, TTFT, prompt tok/s, generation tok/s, and VRAM.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import requests

# Models to test, smallest to largest.
# Each entry: (display_name, model_filename_stem)
MODELS = [
    ("Qwen3.5 0.8B Q3_K_M", "Qwen3.5-0.8B-Q3_K_M"),
    ("Qwen3 4B Q4_K_M", "Qwen3-4B-Q4_K_M"),
    ("Gemma 3 4B Q4_K_M", "gemma-3-4b-it-Q4_K_M"),
    ("Qwen3.5 9B Heretic Q4_K_M", "Qwen3.5-9B-heretic-v2.Q4_K_M"),
    ("Gemma 4 26B-A4B MoE Q4_K_M", "gemma-4-26B-A4B-it-heretic-ara.Q4_K_M"),
    ("Gemma 4 E4B Q8_0", "gemma-4-E4B-it-Q8_0"),
]

BENCHMARK_PROMPT = "Explain quantum computing in simple terms. Cover the key concepts of superposition, entanglement, and quantum gates. Then discuss potential real-world applications."
N_PREDICT = 200


def get_session(host: str, cookie: str | None) -> requests.Session:
    s = requests.Session()
    if cookie:
        s.cookies.set("session", cookie)
    # Try to get a valid session by checking status
    try:
        r = s.get(f"{host}/api/engine/v2/status", timeout=5)
        if r.status_code == 401:
            print("ERROR: Authentication required. Pass --cookie with your session cookie.")
            print("  Get it from browser DevTools > Application > Cookies > session")
            sys.exit(1)
    except requests.ConnectionError:
        print(f"ERROR: Cannot connect to {host}")
        sys.exit(1)
    return s


def unload_model(s: requests.Session, host: str) -> None:
    """Unload current model to start fresh."""
    try:
        s.post(f"{host}/api/engine/v2/models/unload", timeout=30)
    except Exception:
        pass
    # Wait for unload
    for _ in range(20):
        try:
            r = s.get(f"{host}/api/engine/v2/status", timeout=5)
            data = r.json()
            if data.get("state") == "idle":
                return
        except Exception:
            pass
        time.sleep(0.5)


def load_model(s: requests.Session, host: str, model_name: str) -> dict:
    """Load a model and measure load time. Returns status dict."""
    t_start = time.monotonic()
    try:
        r = s.post(
            f"{host}/api/engine/v2/models/load",
            json={"model": model_name},
            timeout=300,
        )
        if r.status_code >= 400:
            return {"error": f"Load failed: {r.status_code} {r.text[:200]}"}
    except requests.Timeout:
        return {"error": "Load timed out (300s)"}
    except Exception as exc:
        return {"error": str(exc)}

    # Wait for ready
    for _ in range(600):  # up to 5 minutes
        try:
            r = s.get(f"{host}/api/engine/v2/status", timeout=5)
            data = r.json()
            if data.get("state") == "ready":
                t_end = time.monotonic()
                data["load_time_s"] = round(t_end - t_start, 2)
                return data
            if data.get("state") == "idle":
                return {"error": "Model failed to load (state went to idle)"}
        except Exception:
            pass
        time.sleep(0.5)

    return {"error": "Load timed out waiting for ready"}


def run_benchmark(s: requests.Session, host: str) -> dict:
    """Run the benchmark endpoint."""
    try:
        r = s.post(
            f"{host}/api/engine/v2/benchmark",
            json={"prompt": BENCHMARK_PROMPT, "n_predict": N_PREDICT},
            timeout=120,
        )
        if r.status_code >= 400:
            return {"error": f"Benchmark failed: {r.status_code} {r.text[:200]}"}
        return r.json()
    except requests.Timeout:
        return {"error": "Benchmark timed out"}
    except Exception as exc:
        return {"error": str(exc)}


def run_switch_test(
    s: requests.Session, host: str, model_a: str, model_b: str,
) -> dict:
    """Measure model switch time between two models."""
    # Load model A
    print(f"  Loading {model_a}...")
    unload_model(s, host)
    result_a = load_model(s, host, model_a)
    if "error" in result_a:
        return {"error": f"Failed to load {model_a}: {result_a['error']}"}

    # Switch to model B (measures swap time)
    print(f"  Switching to {model_b}...")
    t_start = time.monotonic()
    try:
        r = s.post(
            f"{host}/api/engine/v2/models/load",
            json={"model": model_b},
            timeout=300,
        )
    except Exception as exc:
        return {"error": str(exc)}

    # Wait for ready
    for _ in range(600):
        try:
            r = s.get(f"{host}/api/engine/v2/status", timeout=5)
            data = r.json()
            if data.get("state") == "ready":
                t_end = time.monotonic()
                return {
                    "from": model_a,
                    "to": model_b,
                    "switch_time_s": round(t_end - t_start, 2),
                }
        except Exception:
            pass
        time.sleep(0.5)

    return {"error": "Switch timed out"}


def print_results(name: str, load_data: dict, bench_data: dict) -> None:
    """Print formatted benchmark results."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    if "error" in load_data:
        print(f"  LOAD ERROR: {load_data['error']}")
        return

    print(f"  Load time:     {load_data.get('load_time_s', '?')}s")

    if profile := load_data.get("profile"):
        print(f"  Architecture:  {profile.get('architecture', '?')}")
        print(f"  Size:          {profile.get('size_gb', '?')} GB")
        print(f"  Layers:        {profile.get('n_layers', '?')}")
        print(f"  MoE:           {profile.get('is_moe', False)}")
        print(f"  Context:       {profile.get('context_length', '?')}")

    if gpu := load_data.get("gpu"):
        print(f"  GPU:           {gpu.get('name', '?')}")
        print(f"  VRAM used:     {gpu.get('vram_used_mib', 0)} / {gpu.get('vram_total_mib', 0)} MiB")

    if "error" in bench_data:
        print(f"  BENCH ERROR:   {bench_data['error']}")
        return

    if b := bench_data.get("benchmark"):
        print(f"  ---")
        print(f"  Prompt tokens: {b.get('prompt_tokens', '?')}")
        print(f"  Gen tokens:    {b.get('generated_tokens', '?')}")
        print(f"  TTFT:          {b.get('ttft_ms', '?')}ms")
        print(f"  Prompt tok/s:  {b.get('prompt_tps_calc', '?')}")
        print(f"  Gen tok/s:     {b.get('generation_tps', '?')}")
        print(f"  Total time:    {b.get('total_s', '?')}s")


def main():
    parser = argparse.ArgumentParser(description="Engine v2 benchmark")
    parser.add_argument("--host", default="http://localhost:6100",
                        help="Augmentum host URL")
    parser.add_argument("--cookie", help="Session cookie for auth")
    parser.add_argument("--models", nargs="*",
                        help="Specific model stems to test (default: all)")
    parser.add_argument("--switch-test", action="store_true",
                        help="Also test model switching speed")
    args = parser.parse_args()

    s = get_session(args.host, args.cookie)

    models = MODELS
    if args.models:
        models = [(m, m) for m in args.models]

    all_results = []

    for name, stem in models:
        print(f"\n>>> Testing: {name}")

        # Unload any current model
        print("  Unloading current model...")
        unload_model(s, args.host)

        # Load
        print(f"  Loading {stem}...")
        load_data = load_model(s, args.host, stem)

        # Benchmark
        bench_data = {}
        if "error" not in load_data:
            print("  Running benchmark...")
            bench_data = run_benchmark(s, args.host)

        print_results(name, load_data, bench_data)
        all_results.append({
            "name": name,
            "model": stem,
            "load": load_data,
            "benchmark": bench_data,
        })

    # Switch test
    if args.switch_test and len(all_results) >= 2:
        # Find two models that loaded successfully
        good = [r for r in all_results if "error" not in r["load"]]
        if len(good) >= 2:
            print(f"\n>>> Switch test: {good[0]['name']} <-> {good[1]['name']}")
            switch = run_switch_test(
                s, args.host, good[0]["model"], good[1]["model"]
            )
            print(f"  Switch time: {switch.get('switch_time_s', switch.get('error', '?'))}s")

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Model':<35} {'Load(s)':>8} {'TTFT(ms)':>9} {'Prompt':>8} {'Gen':>8} {'VRAM':>8}")
    print(f"  {'':<35} {'':>8} {'':>9} {'tok/s':>8} {'tok/s':>8} {'MiB':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*8}")

    for r in all_results:
        name = r["name"][:35]
        if "error" in r["load"]:
            print(f"  {name:<35} {'FAILED':>8}")
            continue

        load_s = r["load"].get("load_time_s", "?")
        vram = r["load"].get("gpu", {}).get("vram_used_mib", "?")
        b = r.get("benchmark", {}).get("benchmark", {})
        ttft = b.get("ttft_ms", "?")
        ptps = b.get("prompt_tps_calc", "?")
        gtps = b.get("generation_tps", "?")

        print(f"  {name:<35} {load_s:>8} {ttft:>9} {ptps:>8} {gtps:>8} {vram:>8}")

    # Save raw results
    with open("benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Raw results saved to benchmark_results.json")


if __name__ == "__main__":
    main()
