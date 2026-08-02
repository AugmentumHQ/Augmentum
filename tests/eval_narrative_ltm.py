#!/usr/bin/env python3
"""Narrative LTM model evaluation — grades STATE/LEDGER/ARCHIVE quality across model sizes.

Loads real roleplay conversations, sends them through the three-layer memory
pipeline in chunks (simulating live usage), and collects all LLM outputs for
manual review.

Usage:
    python tests/eval_narrative_ltm.py [--models MODEL1,MODEL2,...] [--chunks N] [--output FILE]

Connects to LM Studio at localhost:1234 (OpenAI-compatible API).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from augmentum.modes.narrative.memory import (
    MEMORY_CATEGORIES,
    STATE_FIELDS,
    CardType,
    MemoryEntry,
    StateSnapshot,
    SummaryMode,
    build_compaction_prompt,
    build_state_memory_prompt,
    parse_state_memory_response,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

LMSTUDIO_URL = "http://localhost:1234/v1"
CONVO_DIR = Path(os.environ.get("AUGMENTUM_ROLEPLAY_CONVO_DIR", "/data/roleplay_train"))
OUTPUT_DIR = Path("tests/eval_results")

# Conversations to test (short, medium, long)
TARGET_CONVOS = [
    # Short (~50 messages)
    "Tavo_The Demon Queen_ Morvana_iIho.jsonl",
    # Medium (~200 messages)
    "Tavo_The Chronicles of Cyraeth_iIii.jsonl",
    # Long (500+ messages)
    "Tavo_The Rising of the Shield Hero_iIdL.jsonl",
]

# Default model tiers to test
DEFAULT_MODELS = [
    "gemma-3-4b-it",                      # ~4B — small
    "crow-9b-opus-4.6-distill-heretic_qwen3.5",  # ~9B — medium
    "qwen3.5-27b",                         # ~27B — large
]

CHUNK_SIZE = 10  # messages per refresh batch
COMPACTION_CEILING = 25  # trigger compaction at this ledger size


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_conversation(filename: str) -> tuple[str, str, list[dict]]:
    """Load a JSONL conversation file → (char_name, user_name, messages)."""
    path = CONVO_DIR / filename
    if not path.exists():
        # Try fuzzy match
        candidates = list(CONVO_DIR.glob(f"*{filename.split('_')[1]}*"))
        if candidates:
            path = candidates[0]
        else:
            raise FileNotFoundError(f"Not found: {path}")

    char_name = "Character"
    user_name = "User"
    messages = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "chat_metadata" in obj:
            char_name = obj.get("character_name", char_name)
            user_name = obj.get("user_name", user_name)
            continue
        if "mes" in obj:
            messages.append(obj)
    return char_name, user_name, messages


def classify_convo(filename: str) -> CardType:
    lower = filename.lower()
    if "rpg" in lower or "isekai" in lower or "chronicles" in lower or "leveling" in lower:
        return CardType.NARRATOR
    return CardType.CHARACTER


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

async def call_lm(
    client: httpx.AsyncClient,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.3,
    max_tokens: int = 700,
) -> tuple[str, float, dict]:
    """Call LM Studio and return (response_text, elapsed_seconds, usage)."""
    t0 = time.monotonic()
    try:
        resp = await client.post(
            f"{LMSTUDIO_URL}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.monotonic() - t0
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return content, elapsed, usage
    except Exception as e:
        elapsed = time.monotonic() - t0
        return f"[ERROR: {e}]", elapsed, {}


# ---------------------------------------------------------------------------
# Quality scoring (automated heuristics)
# ---------------------------------------------------------------------------

@dataclass
class RefreshScore:
    """Automated quality score for a single STATE+MEMORY refresh."""
    fields_populated: int = 0
    fields_expected: int = 0
    entries_count: int = 0
    entries_valid_category: int = 0
    entries_in_range: int = 0
    has_state_header: bool = False
    has_memory_header: bool = False
    parse_succeeded: bool = False
    raw_response: str = ""
    elapsed: float = 0.0
    usage: dict = field(default_factory=dict)

    @property
    def field_coverage(self) -> float:
        return self.fields_populated / max(1, self.fields_expected)

    @property
    def category_accuracy(self) -> float:
        return self.entries_valid_category / max(1, self.entries_count)

    def summary_line(self) -> str:
        return (
            f"fields={self.fields_populated}/{self.fields_expected} "
            f"entries={self.entries_count} "
            f"valid_cat={self.entries_valid_category}/{self.entries_count} "
            f"in_range={self.entries_in_range}/{self.entries_count} "
            f"headers={'Y' if self.has_state_header and self.has_memory_header else 'N'} "
            f"time={self.elapsed:.1f}s"
        )


@dataclass
class CompactionScore:
    """Automated quality score for a compaction pass."""
    input_count: int = 0
    output_count: int = 0
    rounds_preserved: int = 0
    rounds_lost: int = 0
    avg_text_reduction: float = 0.0
    raw_response: str = ""
    elapsed: float = 0.0
    usage: dict = field(default_factory=dict)

    def summary_line(self) -> str:
        return (
            f"in={self.input_count} out={self.output_count} "
            f"preserved={self.rounds_preserved}/{self.input_count} "
            f"text_reduction={self.avg_text_reduction:.0%} "
            f"time={self.elapsed:.1f}s"
        )


def score_refresh(
    raw: str,
    card_type: CardType,
    batch_start: int,
    batch_end: int,
    elapsed: float,
    usage: dict,
) -> tuple[RefreshScore, StateSnapshot, list[MemoryEntry]]:
    """Score a refresh response and return parsed objects."""
    fields = STATE_FIELDS.get(card_type, STATE_FIELDS[CardType.CHARACTER])
    categories = set(MEMORY_CATEGORIES.get(card_type, MEMORY_CATEGORIES[CardType.CHARACTER]))

    score = RefreshScore(
        fields_expected=len(fields),
        raw_response=raw,
        elapsed=elapsed,
        usage=usage,
    )
    score.has_state_header = bool(re.search(r"##\s*STATE", raw, re.IGNORECASE))
    score.has_memory_header = bool(re.search(r"##\s*MEMORY", raw, re.IGNORECASE))

    try:
        snap, entries = parse_state_memory_response(raw, card_type, batch_start, batch_end)
        score.parse_succeeded = True
        score.fields_populated = sum(1 for f in fields if snap.fields.get(f))
        score.entries_count = len(entries)
        score.entries_valid_category = sum(1 for e in entries if e.category in categories)
        score.entries_in_range = sum(1 for e in entries if batch_start <= e.round_num <= batch_end)
        return score, snap, entries
    except Exception:
        score.parse_succeeded = False
        return score, StateSnapshot(fields={}), []


def score_compaction(
    raw: str,
    input_entries: list[MemoryEntry],
    elapsed: float,
    usage: dict,
) -> tuple[CompactionScore, list[MemoryEntry]]:
    """Score a compaction response and return parsed entries."""
    score = CompactionScore(
        input_count=len(input_entries),
        raw_response=raw,
        elapsed=elapsed,
        usage=usage,
    )

    entry_pattern = re.compile(r"\[R(\d+)\|([^\]]+)\]\s*(.+)")
    compacted = []
    for line in raw.strip().split("\n"):
        m = entry_pattern.match(line.strip().lstrip("- "))
        if m:
            compacted.append(MemoryEntry(
                round_num=int(m.group(1)),
                category=m.group(2).strip().lower().replace(" ", "_"),
                content=m.group(3).strip(),
            ))

    score.output_count = len(compacted)
    input_rounds = {e.round_num for e in input_entries}
    output_rounds = {e.round_num for e in compacted}
    score.rounds_preserved = len(input_rounds & output_rounds)
    score.rounds_lost = len(input_rounds - output_rounds)

    # Average text reduction
    reductions = []
    for inp in input_entries:
        match = [c for c in compacted if c.round_num == inp.round_num]
        if match:
            orig_len = len(inp.content)
            new_len = len(match[0].content)
            if orig_len > 0:
                reductions.append(1.0 - new_len / orig_len)
    score.avg_text_reduction = sum(reductions) / len(reductions) if reductions else 0.0

    return score, compacted


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Full evaluation result for one model × one conversation."""
    model: str
    conversation: str
    card_type: str
    total_messages: int
    refreshes: list[RefreshScore] = field(default_factory=list)
    compactions: list[CompactionScore] = field(default_factory=list)
    final_state: dict = field(default_factory=dict)
    final_ledger: list[dict] = field(default_factory=list)
    total_time: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "conversation": self.conversation,
            "card_type": self.card_type,
            "total_messages": self.total_messages,
            "refreshes": [
                {
                    "score": r.summary_line(),
                    "fields_populated": r.fields_populated,
                    "fields_expected": r.fields_expected,
                    "entries_count": r.entries_count,
                    "valid_categories": r.entries_valid_category,
                    "in_range": r.entries_in_range,
                    "headers_ok": r.has_state_header and r.has_memory_header,
                    "parse_ok": r.parse_succeeded,
                    "elapsed": round(r.elapsed, 2),
                    "usage": r.usage,
                    "raw_response": r.raw_response,
                }
                for r in self.refreshes
            ],
            "compactions": [
                {
                    "score": c.summary_line(),
                    "input_count": c.input_count,
                    "output_count": c.output_count,
                    "rounds_preserved": c.rounds_preserved,
                    "rounds_lost": c.rounds_lost,
                    "text_reduction": round(c.avg_text_reduction, 3),
                    "elapsed": round(c.elapsed, 2),
                    "usage": c.usage,
                    "raw_response": c.raw_response,
                }
                for c in self.compactions
            ],
            "final_state": self.final_state,
            "final_ledger": self.final_ledger,
            "total_time": round(self.total_time, 2),
        }


async def evaluate_model_on_conversation(
    client: httpx.AsyncClient,
    model: str,
    convo_file: str,
    chunk_size: int,
    max_messages: int = 200,
    compaction_ceiling: int = COMPACTION_CEILING,
) -> EvalResult:
    """Run one model through one conversation, collecting all outputs."""
    char_name, user_name, raw_messages = load_conversation(convo_file)
    card_type = classify_convo(convo_file)
    fields = STATE_FIELDS.get(card_type, STATE_FIELDS[CardType.CHARACTER])

    result = EvalResult(
        model=model,
        conversation=convo_file,
        card_type=card_type.value,
        total_messages=min(max_messages, len(raw_messages)),
    )

    # Engine state (simulated)
    message_history: list[str] = []
    state_snapshot: StateSnapshot | None = None
    memory_ledger: list[MemoryEntry] = []
    last_summary_at = 0
    message_count = 0

    t_start = time.monotonic()
    cap = min(max_messages, len(raw_messages))

    for i in range(cap):
        msg = raw_messages[i]
        text = msg.get("mes", "")
        if not text:
            continue

        message_history.append(text)
        message_count += 1

        # Check if refresh is due
        if (message_count - last_summary_at) >= chunk_size:
            batch_start = last_summary_at + 1
            batch_end = message_count

            print(f"    [{model[:30]:30s}] Refresh R{batch_start}-R{batch_end} "
                  f"(ledger={len(memory_ledger)})...", end=" ", flush=True)

            # Build prompt
            system, user = build_state_memory_prompt(
                card_type=card_type,
                current_state=state_snapshot,
                memory_ledger=memory_ledger,
                recent_messages=message_history,
                char_name=char_name,
                batch_start=batch_start,
                batch_end=batch_end,
                mode=SummaryMode.STANDARD,
            )

            # Call LLM
            raw_resp, elapsed, usage = await call_lm(
                client, model, system, user,
                temperature=0.3,
                max_tokens=700,
            )

            # Score and parse
            refresh_score, snap, entries = score_refresh(
                raw_resp, card_type, batch_start, batch_end, elapsed, usage,
            )
            result.refreshes.append(refresh_score)

            # Apply
            if snap.fields:
                state_snapshot = snap
            memory_ledger.extend(entries)
            last_summary_at = batch_end

            print(refresh_score.summary_line())

            # Check compaction
            if len(memory_ledger) >= compaction_ceiling:
                compact_count = max(1, int(len(memory_ledger) * 0.33))
                to_compact = memory_ledger[:compact_count]

                print(f"    [{model[:30]:30s}] Compacting {compact_count} entries...", end=" ", flush=True)

                c_system, c_user = build_compaction_prompt(to_compact, card_type)
                c_raw, c_elapsed, c_usage = await call_lm(
                    client, model, c_system, c_user,
                    temperature=0.0,
                    max_tokens=max(600, len(to_compact) * 32),
                )

                c_score, compacted = score_compaction(c_raw, to_compact, c_elapsed, c_usage)
                result.compactions.append(c_score)

                # Rescue missing entries
                if compacted:
                    output_rounds = {e.round_num for e in compacted}
                    rescued = [e for e in to_compact if e.round_num not in output_rounds]
                    if rescued:
                        compacted = sorted(compacted + rescued, key=lambda e: e.round_num)
                    to_keep = memory_ledger[compact_count:]
                    memory_ledger = compacted + to_keep

                print(c_score.summary_line())

    result.total_time = time.monotonic() - t_start
    result.final_state = state_snapshot.fields if state_snapshot else {}
    result.final_ledger = [e.to_dict() for e in memory_ledger]

    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_report(results: list[EvalResult]) -> None:
    """Print a human-readable comparison report."""
    print("\n" + "=" * 90)
    print("NARRATIVE LTM EVALUATION REPORT")
    print("=" * 90)

    # Group by conversation
    by_convo: dict[str, list[EvalResult]] = {}
    for r in results:
        by_convo.setdefault(r.conversation, []).append(r)

    for convo, convo_results in by_convo.items():
        print(f"\n{'-' * 90}")
        print(f"CONVERSATION: {convo}")
        print(f"Card type: {convo_results[0].card_type}, Messages: {convo_results[0].total_messages}")
        print(f"{'-' * 90}")

        for r in convo_results:
            print(f"\n  MODEL: {r.model}")
            print(f"  Total time: {r.total_time:.1f}s | Refreshes: {len(r.refreshes)} | Compactions: {len(r.compactions)}")

            # Refresh summary
            if r.refreshes:
                avg_fields = sum(s.fields_populated for s in r.refreshes) / len(r.refreshes)
                avg_entries = sum(s.entries_count for s in r.refreshes) / len(r.refreshes)
                avg_valid = sum(s.entries_valid_category for s in r.refreshes) / len(r.refreshes)
                avg_time = sum(s.elapsed for s in r.refreshes) / len(r.refreshes)
                headers_ok = sum(1 for s in r.refreshes if s.has_state_header and s.has_memory_header)

                print(f"  Refresh avg: fields={avg_fields:.1f}/{r.refreshes[0].fields_expected} "
                      f"entries={avg_entries:.1f} valid_cat={avg_valid:.1f} "
                      f"headers={headers_ok}/{len(r.refreshes)} time={avg_time:.1f}s")

            # Compaction summary
            if r.compactions:
                avg_preserved = sum(c.rounds_preserved for c in r.compactions) / len(r.compactions)
                avg_lost = sum(c.rounds_lost for c in r.compactions) / len(r.compactions)
                avg_reduction = sum(c.avg_text_reduction for c in r.compactions) / len(r.compactions)

                print(f"  Compaction avg: preserved={avg_preserved:.0f} lost={avg_lost:.0f} "
                      f"text_reduction={avg_reduction:.0%}")

            # Final state
            if r.final_state:
                print("  Final STATE:")
                for k, v in r.final_state.items():
                    print(f"    {k}: {v[:80]}{'...' if len(v) > 80 else ''}")

            # Final ledger (last 5 entries)
            if r.final_ledger:
                print(f"  Final LEDGER ({len(r.final_ledger)} entries, showing last 5):")
                for e in r.final_ledger[-5:]:
                    print(f"    [R{e['round_num']}|{e['category']}] {e['content'][:70]}")

    # Comparison table
    print(f"\n{'=' * 90}")
    print("COMPARISON MATRIX")
    print(f"{'=' * 90}")
    print(f"{'Model':<35s} {'Conv':<20s} {'Fields%':>8s} {'Entries':>8s} {'ValidCat':>9s} {'Headers':>8s} {'Compact':>8s} {'Time':>7s}")
    print("-" * 90)
    for r in results:
        if r.refreshes:
            avg_fields_pct = sum(s.field_coverage for s in r.refreshes) / len(r.refreshes) * 100
            avg_entries = sum(s.entries_count for s in r.refreshes) / len(r.refreshes)
            avg_valid_pct = sum(s.category_accuracy for s in r.refreshes) / len(r.refreshes) * 100
            headers_pct = sum(1 for s in r.refreshes if s.has_state_header and s.has_memory_header) / len(r.refreshes) * 100
            compact_ok = "N/A" if not r.compactions else f"{sum(c.rounds_lost == 0 for c in r.compactions)}/{len(r.compactions)}"
        else:
            avg_fields_pct = avg_entries = avg_valid_pct = headers_pct = 0
            compact_ok = "N/A"

        short_convo = r.conversation.split("_")[1][:18]
        print(f"{r.model[:34]:<35s} {short_convo:<20s} {avg_fields_pct:>7.0f}% {avg_entries:>7.1f} {avg_valid_pct:>8.0f}% {headers_pct:>7.0f}% {compact_ok:>8s} {r.total_time:>6.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Evaluate narrative LTM across model sizes")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model names (default: auto-select 3 tiers)")
    parser.add_argument("--convos", type=str, default=None,
                        help="Comma-separated JSONL filenames (default: short/medium/long)")
    parser.add_argument("--chunks", type=int, default=CHUNK_SIZE,
                        help=f"Messages per refresh batch (default: {CHUNK_SIZE})")
    parser.add_argument("--max-messages", type=int, default=200,
                        help="Max messages per conversation (default: 200)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: tests/eval_results/ltm_eval_TIMESTAMP.json)")
    parser.add_argument("--ceiling", type=int, default=COMPACTION_CEILING,
                        help=f"Compaction ceiling (default: {COMPACTION_CEILING})")
    args = parser.parse_args()

    # Resolve models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = DEFAULT_MODELS

    # Resolve conversations
    if args.convos:
        convos = [c.strip() for c in args.convos.split(",")]
    else:
        convos = TARGET_CONVOS

    # Verify LM Studio is running
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{LMSTUDIO_URL}/models", timeout=5.0)
            available = [m["id"] for m in resp.json().get("data", [])]
            print(f"LM Studio models available: {len(available)}")
        except Exception as e:
            print(f"ERROR: Cannot connect to LM Studio at {LMSTUDIO_URL}: {e}")
            sys.exit(1)

        # Verify requested models exist
        for m in models:
            if m not in available:
                print(f"WARNING: Model '{m}' not found in LM Studio. Available: {', '.join(available[:5])}...")

        # Verify conversations exist
        for c in convos:
            path = CONVO_DIR / c
            if not path.exists():
                print(f"WARNING: Conversation file not found: {path}")

        print("\nEvaluation plan:")
        print(f"  Models: {', '.join(models)}")
        print(f"  Conversations: {len(convos)}")
        print(f"  Chunk size: {args.chunks}")
        print(f"  Compaction ceiling: {args.ceiling}")
        print(f"  Max messages: {args.max_messages}")
        print(f"  Total evals: {len(models) * len(convos)}")
        print()

        compaction_ceiling = args.ceiling

        all_results: list[EvalResult] = []

        for convo in convos:
            print(f"\n{'=' * 70}")
            print(f"CONVERSATION: {convo}")
            print(f"{'=' * 70}")

            for model in models:
                print(f"\n  Running: {model}")
                try:
                    result = await evaluate_model_on_conversation(
                        client, model, convo, args.chunks, args.max_messages,
                        compaction_ceiling=compaction_ceiling,
                    )
                    all_results.append(result)
                except FileNotFoundError as e:
                    print(f"  SKIP: {e}")
                except Exception as e:
                    print(f"  ERROR: {e}")

        # Print report
        if all_results:
            print_report(all_results)

        # Save JSON
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = Path(args.output) if args.output else OUTPUT_DIR / f"ltm_eval_{ts}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            [r.to_dict() for r in all_results],
            indent=2,
            ensure_ascii=False,
        ), encoding="utf-8")
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
