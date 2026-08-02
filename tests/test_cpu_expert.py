"""Check if CPU expert compute is active."""
from __future__ import annotations
import json
import time
import urllib.request


def main():
    status = json.loads(urllib.request.urlopen("http://localhost:8090/v1/engine/status").read())
    model = status.get("model", {}).get("name", "?")
    features = status.get("features", {})
    print(f"Model: {model}")
    print(f"MoE offload: {features.get('moe_expert_offload', '?')}")
    print(f"CPU expert compute: {features.get('cpu_expert_compute', '?')}")

    # Run 3 benchmarks
    print("\nBenchmark (3 runs):")
    for i in range(3):
        start = time.time()
        req = urllib.request.Request(
            "http://localhost:8090/v1/chat/completions",
            data=json.dumps({
                "model": "test",
                "messages": [{"role": "user", "content": "Count from 1 to 50, one per line."}],
                "max_tokens": 200,
                "temperature": 0.1,
                "stream": False,
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=300)
        d = json.loads(resp.read())
        elapsed = time.time() - start
        ct = d.get("usage", {}).get("completion_tokens", 0)
        print(f"  Run {i+1}: {ct} tokens, {elapsed:.1f}s, {ct/elapsed:.1f} tok/s")

if __name__ == "__main__":
    main()
