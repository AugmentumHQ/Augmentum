#!/usr/bin/env python3
"""MTP / mmproj / KV-cache matrix bench for Qwen 3.6 27B on a single GPU.

Spawns llama-server directly with controlled CLI args, runs a fixed prompt,
captures VRAM via nvidia-smi + timing from /completion's `timings` block
+ MTP acceptance rate from llama-server stderr. Bypasses Augmentum's
manager + auth + KV-restore so results aren't polluted by per-load
state. Intended to be invoked via:

    docker exec augmentum-augmentum-1 python3 /tmp/mtp_bench.py

Outputs a CSV at /tmp/mtp_bench.csv and a markdown summary on stdout.
"""
from __future__ import annotations

import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

BIN = "/usr/local/bin/llama-server"
MODEL = "/models/host/Qwen3.6-27B-UD-Q4_K_XL.gguf"
MMPROJ = "/models/host/mmproj-BF16.gguf"
PORT = 8092
HOST = "127.0.0.1"
BASE = f"http://{HOST}:{PORT}"

# Prompt: ~400-token chat-style with mixed predictable + creative content
PROMPT = """<|im_start|>system
You are a helpful coding assistant. Be concise and direct.
<|im_end|>
<|im_start|>user
I'm working on a Python project that uses asyncio to coordinate multiple background workers, each doing a different task. Right now my code looks like this:

```python
import asyncio

async def worker_a():
    while True:
        data = await fetch_from_queue()
        await process(data)

async def worker_b():
    while True:
        msg = await listen_websocket()
        await handle(msg)

async def main():
    await asyncio.gather(worker_a(), worker_b())
```

The problem is when one worker hits an unhandled exception, the whole gather() crashes and the other worker dies too. I want each worker to keep running independently — if worker_a crashes, worker_b should keep going, and worker_a should restart after a backoff.

What's the cleanest pattern for this? Should I use asyncio.TaskGroup, or shield the tasks, or wrap each in try/except inside the loop? Explain the tradeoffs and show me the recommended approach with code.
<|im_end|>
<|im_start|>assistant
"""

# Common args every config gets
COMMON = [
    "--model", MODEL,
    "--host", HOST,
    "--port", str(PORT),
    "--n-gpu-layers", "65",
    "--parallel", "1",
    "--flash-attn", "on",
    "--jinja",
    "--reasoning-format", "deepseek",
    "--batch-size", "4096",
    "--no-warmup",  # we'll do a warmup request ourselves to control timing
    "--metrics",
]

# Matrix: (name, extra_args, expects_mtp)
MATRIX = [
    # Block A — isolate MTP overhead at ctx=16192 q8_0/q8_0
    ("A1_baseline_no_mtp",
     ["--ctx-size", "16192", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
     False),
    ("A2_mtp_n2",
     ["--ctx-size", "16192", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
      "--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
     True),
    ("A3_mtp_n6",
     ["--ctx-size", "16192", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
      "--spec-type", "draft-mtp", "--spec-draft-n-max", "6"],
     True),
    ("A4_mtp_n12",
     ["--ctx-size", "16192", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
      "--spec-type", "draft-mtp", "--spec-draft-n-max", "12"],
     True),
    # Block B — does mmproj coexist with MTP?
    ("B1_mmproj_no_mtp",
     ["--ctx-size", "16192", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
      "--mmproj", MMPROJ],
     False),
    ("B2_mmproj_mtp_n6",
     ["--ctx-size", "16192", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
      "--mmproj", MMPROJ, "--spec-type", "draft-mtp", "--spec-draft-n-max", "6"],
     True),
    # Block C — VRAM-saving knobs (MTP n=6 no mmproj)
    ("C1_ctx8k_q8q8",
     ["--ctx-size", "8192", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
      "--spec-type", "draft-mtp", "--spec-draft-n-max", "6"],
     True),
    ("C2_ctx16k_q4q8",
     ["--ctx-size", "16192", "--cache-type-k", "q4_0", "--cache-type-v", "q8_0",
      "--spec-type", "draft-mtp", "--spec-draft-n-max", "6"],
     True),
    ("C3_ctx8k_q4q4",
     ["--ctx-size", "8192", "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
      "--spec-type", "draft-mtp", "--spec-draft-n-max", "6"],
     True),
]

# Champions get re-run free-seed for variance (filled at runtime from results)
VARIANCE_RUNS_PER_CHAMPION = 3
PINNED_SEED = 42


def vram_mb() -> int:
    """Return GPU 0 used VRAM in MiB."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    )
    return int(out.stdout.strip().splitlines()[0])


def kill_stale_servers():
    """Aggressive cleanup before each launch — kill ANY llama-server.

    Augmentum's manager will also spin one up on :8091 the moment any
    chat request arrives. For a clean VRAM bench we want a pristine GPU,
    so we kill all llama-server processes (ours on :8092 + augmentum's
    on :8091 if running). Augmentum's will re-spawn lazily on next chat
    after the bench completes — no permanent damage.
    """
    subprocess.run(["pkill", "-f", "llama-server"], capture_output=True)
    # Wait until VRAM actually drops, not just process exit
    deadline = time.time() + 30
    while time.time() < deadline:
        used = vram_mb()
        if used < 3500:  # Empty-GPU floor for this driver, give some slack
            break
        time.sleep(1)
    time.sleep(2)  # extra settle


def wait_ready(timeout: float = 180) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def launch(extra_args, name: str):
    kill_stale_servers()
    cmd = [BIN] + COMMON + extra_args
    # stderr → pipe so we can scrape acceptance lines on shutdown
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    return proc


def shutdown(proc):
    if proc and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def request_completion(seed: int | None, n_predict: int = 200) -> dict:
    body = {
        "prompt": PROMPT,
        "n_predict": n_predict,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,  # Unsloth's recommended Qwen3.6 default
        "stream": False,
        "cache_prompt": False,  # don't share KV between runs
    }
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(
        f"{BASE}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


_ACCEPT_RE = re.compile(
    r"draft acceptance rate\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s+accepted\s*/\s*(\d+)\s+generated",
)


def parse_acceptance(stderr_text: str) -> tuple[float, int, int] | None:
    """Last `draft acceptance rate = X (A accepted / G generated)` line."""
    matches = list(_ACCEPT_RE.finditer(stderr_text))
    if not matches:
        return None
    m = matches[-1]
    return (float(m.group(1)), int(m.group(2)), int(m.group(3)))


def run_config(name: str, extra_args: list[str], expects_mtp: bool,
               seed: int | None = PINNED_SEED) -> dict:
    print(f"\n=== {name} (seed={seed}) ===", flush=True)
    print(f"args: {' '.join(extra_args)}", flush=True)

    vram_idle = vram_mb()
    proc = launch(extra_args, name)
    if not wait_ready():
        shutdown(proc)
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        print(f"  ! FAILED to become ready in 180s", flush=True)
        print(f"  stderr tail: {stderr[-500:]}", flush=True)
        return {
            "name": name, "seed": seed, "ok": False, "error": "launch_timeout",
            "vram_idle_mb": vram_idle,
        }

    # Settle VRAM after warmup
    time.sleep(2)
    vram_loaded = vram_mb()

    try:
        # Warmup request (not measured)
        request_completion(seed=seed, n_predict=32)
        time.sleep(0.5)

        # Measured request
        t0 = time.time()
        resp = request_completion(seed=seed, n_predict=200)
        t_total = time.time() - t0
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace") if hasattr(e, 'read') else ''
        print(f"  ! HTTP error {e.code}: {body[:300]}", flush=True)
        shutdown(proc)
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        return {
            "name": name, "seed": seed, "ok": False, "error": f"http_{e.code}",
            "vram_idle_mb": vram_idle, "vram_loaded_mb": vram_loaded,
            "stderr_tail": stderr[-500:],
        }
    except Exception as e:
        print(f"  ! Request failed: {e}", flush=True)
        shutdown(proc)
        return {
            "name": name, "seed": seed, "ok": False, "error": str(e)[:200],
            "vram_idle_mb": vram_idle, "vram_loaded_mb": vram_loaded,
        }

    vram_peak = vram_mb()

    # Shutdown to release VRAM + read stderr
    shutdown(proc)
    stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""

    timings = resp.get("timings", {})
    accept = parse_acceptance(stderr)

    result = {
        "name": name,
        "seed": seed,
        "ok": True,
        "vram_idle_mb": vram_idle,
        "vram_loaded_mb": vram_loaded,
        "vram_peak_mb": vram_peak,
        "vram_delta_mb": vram_loaded - vram_idle,
        "prompt_tokens": resp.get("tokens_evaluated", 0),
        "gen_tokens": resp.get("tokens_predicted", 0),
        "prompt_tps": round(timings.get("prompt_per_second", 0.0), 1),
        "gen_tps": round(timings.get("predicted_per_second", 0.0), 1),
        "prompt_ms": round(timings.get("prompt_ms", 0.0), 1),
        "predicted_ms": round(timings.get("predicted_ms", 0.0), 1),
        "wall_s": round(t_total, 2),
        "expects_mtp": expects_mtp,
        "mtp_accept_rate": accept[0] if accept else None,
        "mtp_n_accepted": accept[1] if accept else None,
        "mtp_n_generated": accept[2] if accept else None,
    }

    # Wait for VRAM to actually release before next run
    for _ in range(20):
        if vram_mb() < vram_idle + 1000:
            break
        time.sleep(1)

    print(f"  OK  vram {vram_idle}→{vram_loaded} MiB (Δ {vram_loaded-vram_idle})",
          flush=True)
    print(f"      prefill {result['prompt_tps']} tok/s   "
          f"gen {result['gen_tps']} tok/s   "
          f"wall {result['wall_s']}s", flush=True)
    if accept:
        print(f"      MTP accept {accept[0]:.3f} ({accept[1]}/{accept[2]})",
              flush=True)
    elif expects_mtp:
        print(f"      ! MTP expected but no acceptance line in stderr "
              f"(stderr len={len(stderr)})", flush=True)

    return result


def main():
    print(f"# MTP matrix bench — model: {MODEL}", flush=True)
    print(f"# GPU: {subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()}", flush=True)
    print(f"# pinned seed: {PINNED_SEED}", flush=True)
    print(f"# {len(MATRIX)} configs + variance runs", flush=True)

    results = []
    for name, args, expects_mtp in MATRIX:
        r = run_config(name, args, expects_mtp, seed=PINNED_SEED)
        results.append(r)

    # Pick champion (fastest successful gen_tps) for variance runs
    successes = [r for r in results if r.get("ok")]
    if successes:
        champion = max(successes, key=lambda r: r["gen_tps"])
        # Find its config
        for n, a, em in MATRIX:
            if n == champion["name"]:
                champion_args, champion_expects = a, em
                break
        print(f"\n=== Variance runs for champion {champion['name']} "
              f"(seed=None, {VARIANCE_RUNS_PER_CHAMPION}× passes) ===", flush=True)
        for i in range(VARIANCE_RUNS_PER_CHAMPION):
            r = run_config(f"{champion['name']}_var{i+1}", champion_args,
                           champion_expects, seed=None)
            results.append(r)

    # Write CSV
    out_path = "/tmp/mtp_bench.csv"
    fieldnames = sorted(
        {k for r in results for k in r.keys()} - {"stderr_tail"}
    )
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            w.writerow(row)
    print(f"\n[csv → {out_path}]", flush=True)

    # Markdown summary
    print("\n## Summary\n", flush=True)
    print("| Config | VRAM (Δ) MiB | Prefill tok/s | Gen tok/s | "
          "MTP accept | Notes |", flush=True)
    print("|---|---|---|---|---|---|", flush=True)
    for r in results:
        if not r.get("ok"):
            print(f"| {r['name']} | — | — | — | — | FAILED: {r.get('error', '?')} |",
                  flush=True)
            continue
        delta = r["vram_delta_mb"]
        ar = (f"{r['mtp_accept_rate']:.2f} "
              f"({r['mtp_n_accepted']}/{r['mtp_n_generated']})"
              if r.get("mtp_accept_rate") is not None else "—")
        print(f"| {r['name']} | {r['vram_loaded_mb']} ({delta:+d}) | "
              f"{r['prompt_tps']} | {r['gen_tps']} | {ar} | |", flush=True)


if __name__ == "__main__":
    main()
