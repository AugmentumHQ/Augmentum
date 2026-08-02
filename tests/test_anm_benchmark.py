"""ANM Benchmark — 100 real-world prompts from training data.

Samples diverse prompts from gemma3_final_train.jsonl across all categories.
Measures ANM observation, hit rate evolution, and projected speculation gains
against a real distribution of user prompts — not hand-picked toy examples.

Pattern: 80 unique prompts + 20 strategic repeats to measure Hebbian learning.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.request

ENGINE_URL = "http://localhost:8090"
DATASET = "F:/Training/completed datasets/gemma3_final_train.jsonl"


def load_prompts(path: str, n_unique: int = 80, seed: int = None) -> list[dict]:
    """Sample diverse prompts from dataset, stratified by category."""
    by_cat: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cat = d.get("category", "unknown")
            inst = d.get("instruction", "").strip()
            if inst and 20 < len(inst) < 500:  # skip too short/long
                by_cat.setdefault(cat, []).append(inst)

    rng = random.Random(seed if seed is not None else int(time.time()))
    for v in by_cat.values():
        rng.shuffle(v)

    # Proportional sampling across categories
    total_available = sum(len(v) for v in by_cat.values())
    sampled = []
    for cat, prompts in sorted(by_cat.items()):
        n = max(2, int(n_unique * len(prompts) / total_available))
        sampled.extend({"prompt": p, "category": cat} for p in prompts[:n])

    rng.shuffle(sampled)
    return sampled[:n_unique]


def build_sequence(unique_prompts: list[dict], n_total: int = 100) -> list[dict]:
    """Build 100-prompt sequence: 80 unique + 20 repeats for learning test.

    Repeats are drawn from the first 10 prompts — giving the ANM 2-3
    observations before the repeat, simulating real user patterns.
    """
    sequence = list(unique_prompts[:80])

    # Pick 10 prompts to repeat (from early in the sequence)
    repeat_pool = unique_prompts[:10]
    repeats = []
    for p in repeat_pool:
        r = dict(p)
        r["is_repeat"] = True
        repeats.append(r)
    for p in repeat_pool:
        r = dict(p)
        r["is_repeat"] = True
        repeats.append(r)

    # Interleave repeats into positions 81-100
    sequence.extend(repeats[:n_total - len(sequence)])
    return sequence[:n_total]


def generate(prompt: str, max_tokens: int = 60) -> tuple[str, int, float]:
    """Generate and return (text, token_count, elapsed)."""
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
    return content, tokens, elapsed


def get_anm_stats() -> dict:
    try:
        return json.loads(urllib.request.urlopen(
            f"{ENGINE_URL}/v1/engine/anm", timeout=5
        ).read())
    except Exception:
        return {}


def main():
    print("=" * 75)
    print("  ANM Benchmark -- 100 Real-World Prompts (gemma3_final_train.jsonl)")
    print("=" * 75)

    # Verify engine
    try:
        status = json.loads(urllib.request.urlopen(
            f"{ENGINE_URL}/v1/engine/status", timeout=5
        ).read())
        model = status["model"]["name"]
        state = status["model"]["state"]
        print(f"\nEngine: {model} ({state})")
    except Exception as e:
        print(f"Engine not available: {e}")
        return

    # Load and sample prompts
    print(f"Loading prompts from {DATASET}...")
    unique = load_prompts(DATASET, n_unique=80)
    cats = {}
    for p in unique:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    print(f"  Sampled {len(unique)} unique prompts:")
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {n}")

    sequence = build_sequence(unique, n_total=100)
    n_repeats = sum(1 for p in sequence if p.get("is_repeat"))
    print(f"  Sequence: {len(sequence)} total ({len(sequence) - n_repeats} unique + {n_repeats} repeats)")

    # Warm up
    print("\nWarming up...")
    generate("Hello", max_tokens=5)

    # Get baseline tok/s
    print("Measuring baseline tok/s...")
    _, base_tok, base_elapsed = generate(
        "Explain how transformers work in machine learning.", max_tokens=150
    )
    baseline_tps = base_tok / base_elapsed
    print(f"Baseline: {baseline_tps:.1f} tok/s\n")

    # Get initial ANM state
    initial_anm = get_anm_stats()
    initial_entries = initial_anm.get("entries", 0)
    print(f"ANM initial state: {initial_entries} entries (from prior sessions)\n")

    # Run benchmark
    print(f"{'#':>3} {'Cat':>10} {'Rep':>3} {'Tok':>4} {'t/s':>6} "
          f"{'Map':>6} {'Hits':>6} {'Rate':>6} {'Pred':>5} {'Prompt':>40}")
    print("-" * 100)

    results = []
    phase_boundaries = {20: "Phase A (1-20): Cold start",
                        50: "Phase B (21-50): Building",
                        80: "Phase C (51-80): Diverse prompts",
                        100: "Phase D (81-100): Repeats"}

    # Track deltas between prompts
    prev_hits = initial_anm.get("hits", 0)
    prev_misses = initial_anm.get("misses", 0)
    prev_preds = initial_anm.get("predictions_made", 0)

    for i, entry in enumerate(sequence):
        prompt = entry["prompt"]
        is_repeat = entry.get("is_repeat", False)
        cat = entry["category"]

        content, tokens, elapsed = generate(prompt, max_tokens=60)
        tps = tokens / elapsed if elapsed > 0 else 0

        anm = get_anm_stats()
        entries = anm.get("entries", 0)
        cum_hits = anm.get("hits", 0)
        cum_misses = anm.get("misses", 0)
        cum_preds = anm.get("predictions_made", 0)
        hr = anm.get("hit_rate", 0) * 100

        # Per-prompt deltas
        prompt_hits = cum_hits - prev_hits
        prompt_misses = cum_misses - prev_misses
        prompt_preds = cum_preds - prev_preds
        prompt_lookups = prompt_hits + prompt_misses
        prompt_hr = prompt_hits / prompt_lookups * 100 if prompt_lookups > 0 else 0

        prev_hits = cum_hits
        prev_misses = cum_misses
        prev_preds = cum_preds

        results.append({
            "i": i + 1,
            "category": cat,
            "is_repeat": is_repeat,
            "tokens": tokens,
            "tps": tps,
            "map_entries": entries,
            "prompt_hits": prompt_hits,
            "prompt_misses": prompt_misses,
            "prompt_lookups": prompt_lookups,
            "prompt_preds": prompt_preds,
            "prompt_hr": prompt_hr,
            "cumulative_hr": hr,
        })

        rep_mark = "R" if is_repeat else " "
        prompt_short = prompt[:38].replace("\n", " ")
        print(f"{i+1:>3} {cat:>10} {rep_mark:>3} {tokens:>4} {tps:>6.1f} "
              f"{entries:>6} {prompt_hits:>3}/{prompt_lookups:<3} "
              f"{prompt_hr:>5.1f}% p={prompt_preds:>3} {prompt_short:>38}")

        # Phase summaries
        if (i + 1) in phase_boundaries:
            print("-" * 100)
            label = phase_boundaries[i + 1]
            phase_start = {20: 0, 50: 20, 80: 50, 100: 80}[i + 1]
            phase_results = results[phase_start:]
            avg_tps = sum(r["tps"] for r in phase_results) / len(phase_results)
            phase_hits = sum(r["prompt_hits"] for r in phase_results)
            phase_lookups = sum(r["prompt_lookups"] for r in phase_results)
            phase_preds = sum(r["prompt_preds"] for r in phase_results)
            phase_tokens = sum(r["tokens"] for r in phase_results)
            phase_hr = phase_hits / phase_lookups * 100 if phase_lookups > 0 else 0
            phase_repeats = sum(1 for r in phase_results if r["is_repeat"])
            print(f"  {label}")
            print(f"  avg={avg_tps:.1f} tok/s  map={entries}  "
                  f"hits={phase_hits}/{phase_lookups} ({phase_hr:.1f}%)  "
                  f"preds={phase_preds}/{phase_tokens}tok  repeats={phase_repeats}")
            print("-" * 100)

    # Final summary
    print("\n" + "=" * 75)
    print("  RESULTS")
    print("=" * 75)

    final_anm = get_anm_stats()
    total_tokens = sum(r["tokens"] for r in results)
    total_time = sum(r["tokens"] / r["tps"] for r in results if r["tps"] > 0)
    avg_tps = total_tokens / total_time if total_time > 0 else 0
    total_hits = sum(r["prompt_hits"] for r in results)
    total_misses = sum(r["prompt_misses"] for r in results)
    total_lookups = total_hits + total_misses
    total_preds = sum(r["prompt_preds"] for r in results)

    print(f"\n  Model:              {model}")
    print(f"  Prompts:            {len(results)} ({len(results) - n_repeats} unique + {n_repeats} repeats)")
    print(f"  Total tokens gen:   {total_tokens}")
    print(f"  Avg tok/s:          {avg_tps:.1f}")
    print(f"  Baseline tok/s:     {baseline_tps:.1f}")
    print(f"")
    print(f"  ANM entries:        {initial_entries} -> {final_anm.get('entries', 0)}")
    print(f"  ANM observations:   {final_anm.get('observation_count', 0)}")
    print(f"  Total lookups:      {total_lookups} (across 100 prompts)")
    print(f"  Total hits:         {total_hits} ({total_hits/total_lookups*100:.1f}%)" if total_lookups else "")
    print(f"  Total predictions:  {total_preds} (confident enough to draft)")
    print(f"  ANM avg confidence: {final_anm.get('avg_confidence', 0):.3f}")

    # Per-prompt averages
    avg_lookups = total_lookups / len(results)
    avg_hits = total_hits / len(results)
    avg_preds = total_preds / len(results)
    print(f"\n  Per-prompt averages:")
    print(f"    Lookups:          {avg_lookups:.1f}")
    print(f"    Hits:             {avg_hits:.1f}")
    print(f"    Predictions:      {avg_preds:.1f}")

    # Hit rate by phase (using per-prompt deltas)
    print(f"\n  Hit rate by phase:")
    for boundary, label in sorted(phase_boundaries.items()):
        start = {20: 0, 50: 20, 80: 50, 100: 80}[boundary]
        phase = results[start:boundary]
        if phase:
            ph = sum(r["prompt_hits"] for r in phase)
            pl = sum(r["prompt_lookups"] for r in phase)
            pp = sum(r["prompt_preds"] for r in phase)
            pt = sum(r["tokens"] for r in phase)
            phr = ph / pl * 100 if pl > 0 else 0
            print(f"    {label:40s}  hits={ph:>4}/{pl:<4} ({phr:>5.1f}%)  preds={pp:>4}/{pt}tok")

    # Repeat vs novel breakdown
    repeat_results = [r for r in results if r["is_repeat"]]
    novel_results = [r for r in results if not r["is_repeat"]]
    print(f"\n  Novel vs Repeat:")
    for label, subset in [("Novel", novel_results), ("Repeat", repeat_results)]:
        if subset:
            sh = sum(r["prompt_hits"] for r in subset)
            sl = sum(r["prompt_lookups"] for r in subset)
            sp = sum(r["prompt_preds"] for r in subset)
            st = sum(r["tokens"] for r in subset)
            shr = sh / sl * 100 if sl > 0 else 0
            print(f"    {label:8s} (n={len(subset):>3}): hits={sh:>4}/{sl:<4} ({shr:>5.1f}%)  "
                  f"preds={sp:>4}/{st}tok  pred_rate={sp/st*100:.1f}%")

    # Projected speedup — CORRECT MATH
    # Each prediction = 1 lookup where map said "I think next 5 tokens are X"
    # In speculation: we'd eval those 5 draft tokens in 1 batch step (cost ~1 step)
    # instead of 5 sequential steps. If K out of 5 are accepted, we save K-1 steps.
    # Conservative estimate: avg 2 tokens accepted per prediction (out of 5 drafted)
    # (100% accuracy on repeats, ~50% on novel shared patterns, blended ~40%)
    if total_preds > 0:
        print(f"\n  Projected speculative speedup:")

        for accept_rate_name, avg_accept in [("pessimistic (1 tok/pred)", 1),
                                              ("conservative (2 tok/pred)", 2),
                                              ("optimistic (3 tok/pred)", 3)]:
            # Each prediction with avg_accept tokens accepted saves (avg_accept - 1) decode steps
            # (We still need 1 step to verify the batch)
            # But we also need 1 step for the predictions that miss entirely
            # Net: total_steps = total_tokens - saved_steps
            # saved_steps = total_preds * (avg_accept - 1)  [each hit saves accept-1 steps]
            # But predictions that are wrong cost 1 extra step (wasted batch)
            # For now assume all predictions are correct (temp=0)
            saved_steps = total_preds * (avg_accept - 1)
            if saved_steps >= total_tokens:
                saved_steps = total_tokens - 1  # can't save more than total
            effective_steps = total_tokens - saved_steps
            speedup = total_tokens / effective_steps
            projected = baseline_tps * speedup
            print(f"    {accept_rate_name:30s}: "
                  f"save {saved_steps:>5} steps ({saved_steps/total_tokens*100:>4.1f}%) "
                  f"-> {speedup:.2f}x -> {projected:.0f} tok/s")

    # Save
    try:
        urllib.request.urlopen(urllib.request.Request(
            f"{ENGINE_URL}/v1/engine/anm/save",
            data=b"{}",
            headers={"Content-Type": "application/json"},
        ), timeout=10)
        print(f"\n  ANM saved to disk.")
    except Exception:
        pass

    print(f"\n{'='*75}")


if __name__ == "__main__":
    main()
