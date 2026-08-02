"""ANM Evolution Test — 100 prompts measuring how prediction evolves.

Runs 100 prompts through the engine, building the Activation Neural Map
from output patterns. Measures:
- Map growth over time
- Hit rate evolution (should increase as map fills)
- Prediction accuracy (Hebbian strengthening of correct patterns)
- Projected tok/s improvement based on draft acceptance

The test uses output-level fingerprinting (hash of prompt + partial output)
as a proxy for logit-level fingerprinting. This works because with temp=0,
identical contexts always produce identical outputs — so the output IS
a deterministic function of the internal state.
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

from activation_map import ActivationMap

ENGINE_URL = "http://localhost:8090"

# --- Prompt categories ---

FACTUAL = [
    "What is the speed of light?",
    "What year did World War 2 end?",
    "What is the chemical formula for water?",
    "Who wrote Romeo and Juliet?",
    "What is the largest planet in our solar system?",
    "What is the boiling point of water in Celsius?",
    "Who painted the Mona Lisa?",
    "What is the square root of 144?",
    "What is the capital of Japan?",
    "How many bones are in the human body?",
]

CODE = [
    "Write a Python function to check if a number is prime.",
    "Write a Python function to reverse a string.",
    "Write a Python function to find the fibonacci sequence.",
    "Write a Python function to sort a list.",
    "Write a Python function to check if a string is a palindrome.",
]

CREATIVE = [
    "Write a haiku about the ocean.",
    "Describe a sunset in three sentences.",
    "Write a limerick about a cat.",
    "Describe rain falling on a city.",
    "Write a short poem about time.",
]

REASONING = [
    "If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly?",
    "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. What does the ball cost?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
    "Three friends split a $30 bill equally. How much does each pay?",
    "If you have 3 apples and take away 2, how many do you have?",
]


def build_prompt_sequence(n: int = 100) -> list[tuple[str, str]]:
    """Build 100 prompts with strategic repetition to test learning.

    Returns list of (prompt, category) tuples.

    Pattern:
    - First 20: diverse (1 each from all categories, establishing baseline)
    - Next 30: repeat first 10 factual prompts 3x each (ANM should learn these)
    - Next 20: mix of new + repeated (test generalization)
    - Next 15: repeat code prompts 3x each (ANM learns code patterns)
    - Final 15: mix everything (measure mature hit rate)
    """
    sequence = []

    # Phase A (1-20): One of each, establishing baseline
    for p in FACTUAL[:5]:
        sequence.append((p, "factual"))
    for p in CODE[:5]:
        sequence.append((p, "code"))
    for p in CREATIVE[:5]:
        sequence.append((p, "creative"))
    for p in REASONING[:5]:
        sequence.append((p, "reasoning"))

    # Phase B (21-50): Repeat factual prompts 3x each
    for p in FACTUAL[:10]:
        sequence.append((p, "factual-repeat"))
    for p in FACTUAL[:10]:
        sequence.append((p, "factual-repeat"))
    for p in FACTUAL[:10]:
        sequence.append((p, "factual-repeat"))

    # Phase C (51-70): Mix of new and repeated
    for p in CREATIVE:
        sequence.append((p, "creative-repeat"))
    for p in REASONING:
        sequence.append((p, "reasoning-repeat"))
    for p in FACTUAL[:5]:
        sequence.append((p, "factual-3rd-repeat"))
    for p in CODE[:5]:
        sequence.append((p, "code-repeat"))

    # Phase D (71-85): Repeat code 3x
    for p in CODE:
        sequence.append((p, "code-repeat"))
    for p in CODE:
        sequence.append((p, "code-repeat"))
    for p in CODE:
        sequence.append((p, "code-repeat"))

    # Phase E (86-100): Mix everything
    for p in (FACTUAL[:5] + CODE[:3] + CREATIVE[:3] + REASONING[:4]):
        sequence.append((p, "final-mix"))

    return sequence[:n]


def generate(prompt: str, max_tokens: int = 80) -> tuple[str, int, float]:
    """Generate and return (text, token_count, elapsed_seconds)."""
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


def context_fingerprint(prompt: str, output_prefix: str) -> bytes:
    """Fingerprint a generation context.

    In production this would hash logits. Here we hash prompt+output_prefix
    as a proxy — with temp=0, identical context = identical logits.
    """
    data = f"{prompt}|||{output_prefix}".encode()
    return hashlib.blake2b(data, digest_size=16).digest()


def tokenize_output(text: str) -> list[int]:
    """Pseudo-tokenize by hashing word chunks."""
    words = text.split()
    return [hash(w) & 0x7FFFFFFF for w in words]


def main():
    print("=" * 70)
    print("  Activation Neural Map -- 100-Prompt Evolution Test")
    print("=" * 70)

    # Verify engine
    try:
        status = json.loads(urllib.request.urlopen(f"{ENGINE_URL}/v1/engine/status", timeout=5).read())
        model = status["model"]["name"]
        print(f"\nEngine: {model} ({status['model']['state']})")
    except Exception as e:
        print(f"Engine not available: {e}")
        return

    # Warm up
    print("Warming up engine...")
    generate("Hello", max_tokens=5)
    generate("Hello", max_tokens=5)

    # Get baseline tok/s
    print("Measuring baseline tok/s...")
    _, base_tokens, base_elapsed = generate(
        "Write a detailed explanation of how neural networks learn.", max_tokens=200
    )
    baseline_tps = base_tokens / base_elapsed
    print(f"Baseline: {baseline_tps:.1f} tok/s ({base_tokens} tokens in {base_elapsed:.2f}s)\n")

    prompts = build_prompt_sequence(100)

    with tempfile.TemporaryDirectory() as tmp:
        amap = ActivationMap(
            db_path=os.path.join(tmp, "evolution.db"),
            min_confidence=0.6,
            max_draft_tokens=5,
        )

        # Tracking
        results = []
        window_hits = 0
        window_lookups = 0
        window_correct = 0
        window_draft_tokens = 0

        print(f"{'#':>3} {'Cat':>15} {'Tok':>4} {'tok/s':>6} {'Map':>6} "
              f"{'Hits':>5} {'Rate':>6} {'Correct':>8} {'Draft':>6} {'Proj':>7}")
        print("-" * 85)

        for i, (prompt, category) in enumerate(prompts):
            # Generate
            content, tokens, elapsed = generate(prompt, max_tokens=80)
            tps = tokens / elapsed if elapsed > 0 else 0

            # Tokenize output
            tok_ids = tokenize_output(content)

            # Simulate ANM interaction at multiple positions in the output
            prompt_hits = 0
            prompt_lookups = 0
            prompt_correct = 0
            prompt_draft_accepted = 0

            # Slide through output, fingerprinting at each word boundary
            words = content.split()
            for pos in range(len(words) - 1):
                prefix = " ".join(words[:pos + 1])
                fp = context_fingerprint(prompt, prefix)

                # PREDICT: what does the map think comes next?
                draft = amap.predict(fp)
                prompt_lookups += 1

                actual_continuation = tok_ids[pos + 1:pos + 1 + 5]

                if draft is not None:
                    prompt_hits += 1
                    # How many draft tokens match actual?
                    matched = 0
                    for d, a in zip(draft, actual_continuation):
                        if d == a:
                            matched += 1
                        else:
                            break
                    if matched > 0:
                        prompt_correct += 1
                        prompt_draft_accepted += matched

                # RECORD: observe what actually happened
                if actual_continuation:
                    amap.record(fp, actual_continuation)

            # Accumulate window stats
            window_hits += prompt_hits
            window_lookups += prompt_lookups
            window_correct += prompt_correct
            window_draft_tokens += prompt_draft_accepted

            # Calculate projected speedup
            # If draft_accepted tokens skip a decode step each,
            # and baseline is N tok/s, then effective = N * (1 + accepted/total)
            total_tokens_so_far = sum(r["tokens"] for r in results) + tokens
            if total_tokens_so_far > 0:
                acceptance_ratio = window_draft_tokens / total_tokens_so_far
                projected_tps = baseline_tps * (1 + acceptance_ratio)
            else:
                projected_tps = baseline_tps

            hit_rate = window_hits / window_lookups * 100 if window_lookups > 0 else 0
            correct_rate = window_correct / max(window_hits, 1) * 100

            stats = amap.stats()
            results.append({
                "i": i + 1,
                "category": category,
                "tokens": tokens,
                "tps": tps,
                "map_size": stats["entries"],
                "hits": prompt_hits,
                "lookups": prompt_lookups,
                "correct": prompt_correct,
                "draft_accepted": prompt_draft_accepted,
                "cumulative_hit_rate": hit_rate,
                "cumulative_correct_rate": correct_rate,
                "projected_tps": projected_tps,
            })

            # Print every prompt
            print(f"{i+1:>3} {category:>15} {tokens:>4} {tps:>6.1f} {stats['entries']:>6} "
                  f"{prompt_hits:>5} {hit_rate:>5.1f}% {correct_rate:>6.1f}% "
                  f"{prompt_draft_accepted:>5}t {projected_tps:>6.1f}")

            # Print phase summaries
            if i + 1 in (20, 50, 70, 85, 100):
                print("-" * 85)
                phase_results = results[max(0, len(results)-15):]
                avg_tps = sum(r["tps"] for r in phase_results) / len(phase_results)
                print(f"    Phase avg: {avg_tps:.1f} tok/s actual, "
                      f"map={stats['entries']} entries, "
                      f"hit_rate={hit_rate:.1f}%, "
                      f"projected={projected_tps:.1f} tok/s")
                print("-" * 85)

        # Final summary
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)

        final_stats = amap.stats()
        total_lookups = window_lookups
        total_hits = window_hits
        total_correct = window_correct

        print(f"\n  Baseline tok/s:        {baseline_tps:.1f}")
        print(f"  Map entries:           {final_stats['entries']}")
        print(f"  Total observations:    {final_stats['observation_count']}")
        print(f"  Total lookups:         {total_lookups}")
        print(f"  Total hits:            {total_hits} ({total_hits/total_lookups*100:.1f}%)")
        print(f"  Correct predictions:   {total_correct} ({total_correct/max(total_hits,1)*100:.1f}% of hits)")
        print(f"  Draft tokens accepted: {window_draft_tokens}")
        print(f"  Avg confidence:        {final_stats['avg_confidence']:.3f}")

        # Evolution summary by phase
        print("\n  Hit rate evolution:")
        phases = [(1, 20, "A: Baseline"), (21, 50, "B: Factual repeat"),
                  (51, 70, "C: Mixed repeat"), (71, 85, "D: Code repeat"),
                  (86, 100, "E: Final mix")]
        for start, end, label in phases:
            phase_r = [r for r in results if start <= r["i"] <= end]
            if phase_r:
                phase_hits = sum(r["hits"] for r in phase_r)
                phase_lookups = sum(r["lookups"] for r in phase_r)
                phase_correct = sum(r["correct"] for r in phase_r)
                phase_draft = sum(r["draft_accepted"] for r in phase_r)
                hr = phase_hits / phase_lookups * 100 if phase_lookups > 0 else 0
                cr = phase_correct / max(phase_hits, 1) * 100
                print(f"    {label:25s}  hits={phase_hits:>4}/{phase_lookups:<4} "
                      f"({hr:>5.1f}%)  correct={cr:>5.1f}%  draft={phase_draft}t")

        # Projected improvement
        if window_draft_tokens > 0:
            total_gen_tokens = sum(r["tokens"] for r in results)
            # Each accepted draft token saves one decode step
            # Effective throughput = baseline * total_tokens / (total_tokens - draft_accepted)
            effective_tokens = total_gen_tokens - window_draft_tokens
            if effective_tokens > 0:
                speedup = total_gen_tokens / effective_tokens
                projected = baseline_tps * speedup
                print("\n  Projected with speculative execution:")
                print(f"    Tokens generated:    {total_gen_tokens}")
                print(f"    Decode steps saved:  {window_draft_tokens} ({window_draft_tokens/total_gen_tokens*100:.1f}%)")
                print(f"    Speedup factor:      {speedup:.2f}x")
                print(f"    Projected tok/s:     {projected:.1f} (from {baseline_tps:.1f} baseline)")

        # Save map
        amap.save()
        print(f"\n  Map saved to disk ({final_stats['entries']} entries)")


if __name__ == "__main__":
    main()
