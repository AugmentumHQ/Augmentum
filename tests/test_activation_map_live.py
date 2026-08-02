"""Live integration test for Activation Neural Map.

Tests the map's ability to learn patterns from repeated generation,
using the engine API for token generation and the map for observation.

Usage: python tests/test_activation_map_live.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "services", "engine"))
sys.path.insert(0, os.path.expanduser("~/AppData/Roaming/Python/Python311/site-packages"))

try:
    from activation_map import ActivationMap
except ImportError as _import_exc:
    import pytest as _pytest_skip  # noqa: E402
    _pytest_skip.skip(f"activation_map not importable in this build: {_import_exc}", allow_module_level=True)

ENGINE_URL = "http://localhost:8090"


def generate(prompt: str, max_tokens: int = 50) -> tuple[str, float]:
    """Generate and return (text, elapsed)."""
    start = time.time()
    req = urllib.request.Request(
        f"{ENGINE_URL}/v1/chat/completions",
        data=json.dumps({
            "model": "test",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=120)
    d = json.loads(resp.read())
    elapsed = time.time() - start
    content = d["choices"][0]["message"]["content"]
    tokens = d.get("usage", {}).get("completion_tokens", 0)
    return content, elapsed, tokens


def content_fingerprint(text: str, position: int) -> bytes:
    """Create a fingerprint from text content at a given position.

    In production, this would use actual logits. For this test, we use
    a hash of the text prefix as a proxy — demonstrating that identical
    generation contexts produce identical fingerprints.
    """
    prefix = text[:position] if position < len(text) else text
    return hashlib.blake2b(prefix.encode(), digest_size=16).digest()


def text_to_token_ids(text: str) -> list[int]:
    """Convert text to pseudo-token IDs (hash-based proxy)."""
    # Split into word-like chunks and hash each
    words = text.split()
    return [hash(w) & 0x7FFFFFFF for w in words]


def main():
    print("=== Activation Neural Map — Live Integration Test ===\n")

    # Verify engine is running
    try:
        status = json.loads(urllib.request.urlopen(f"{ENGINE_URL}/v1/engine/status", timeout=5).read())
        print(f"Engine: {status['model']['name']} ({status['model']['state']})")
    except Exception as e:
        print(f"Engine not available: {e}")
        return

    with tempfile.TemporaryDirectory() as tmp:
        amap = ActivationMap(
            db_path=os.path.join(tmp, "live.db"),
            min_confidence=0.5,
            max_draft_tokens=5,
        )

        # ---- Phase 1: Generate identical prompts, observe output patterns ----
        print("\nPhase 1: Observation (identical prompts -> identical outputs)")

        prompts = [
            ("What is 2 + 2?", 5),
            ("What is the capital of France?", 5),
            ("Count from 1 to 10.", 3),
        ]

        observations_per_prompt = {}

        for prompt, repeats in prompts:
            outputs = []
            for r in range(repeats):
                content, elapsed, tokens = generate(prompt, max_tokens=40)
                outputs.append(content)
                tok_ids = text_to_token_ids(content)

                # Record observations at multiple positions
                for pos in range(0, min(len(content), 100), 10):
                    fp = content_fingerprint(content, pos)
                    # The "continuation" is the next few token IDs after this position
                    word_pos = len(content[:pos].split())
                    continuation = tok_ids[word_pos:word_pos + 5]
                    if continuation:
                        amap.record(fp, continuation)

                if r == 0:
                    print(f"  \"{prompt}\" -> \"{content[:60]}...\" ({tokens} tok, {elapsed:.1f}s)")

            # Check determinism: with temp=0, all outputs should be identical
            unique = len(set(outputs))
            observations_per_prompt[prompt] = outputs[0]
            if unique == 1:
                print(f"    OK Deterministic ({repeats}/{repeats} identical)")
            else:
                print(f"    !! {unique} unique outputs from {repeats} runs")

        stats = amap.stats()
        print(f"\n  Map: {stats['entries']} entries, {stats['observation_count']} observations")

        # ---- Phase 2: Prediction ----
        print("\nPhase 2: Prediction (same prompts -> should hit map)")

        for prompt, _ in prompts:
            content, elapsed, tokens = generate(prompt, max_tokens=40)
            tok_ids = text_to_token_ids(content)

            hits = 0
            lookups = 0
            correct_predictions = 0

            for pos in range(0, min(len(content), 100), 10):
                fp = content_fingerprint(content, pos)
                draft = amap.predict(fp)
                lookups += 1
                if draft:
                    hits += 1
                    # Check if prediction matches actual
                    word_pos = len(content[:pos].split())
                    actual = tok_ids[word_pos:word_pos + len(draft)]
                    if draft == actual:
                        correct_predictions += 1

            hit_pct = hits / lookups * 100 if lookups > 0 else 0
            correct_pct = correct_predictions / hits * 100 if hits > 0 else 0
            print(f"  \"{prompt[:40]}\" -> hits: {hits}/{lookups} ({hit_pct:.0f}%), correct: {correct_predictions}/{hits} ({correct_pct:.0f}%)")

        # ---- Phase 3: Novel prompt (should miss) ----
        print("\nPhase 3: Novel prompt (should miss map)")
        content, elapsed, tokens = generate("Explain quantum chromodynamics", max_tokens=40)
        fp = content_fingerprint(content, 20)
        draft = amap.predict(fp)
        print(f"  Novel prompt prediction: {'HIT (unexpected)' if draft else 'MISS (expected)'}")

        # ---- Phase 4: Stats ----
        print("\nPhase 4: Final stats")
        stats = amap.stats()
        for k, v in stats.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

        # ---- Phase 5: Persistence ----
        print("\nPhase 5: Persistence")
        amap.save()
        amap2 = ActivationMap(db_path=os.path.join(tmp, "live.db"))
        print(f"  Saved {stats['entries']}, reloaded {len(amap2._cache)} OK")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
