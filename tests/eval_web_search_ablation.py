#!/usr/bin/env python3
"""Web-search ablation harness — measure how each filter layer affects the
payload a model actually receives in a single turn.

Runs N curated queries through 6 configs (A-F), captures the exact text
the model would see (search snippets + fetched bodies + annotations), then
uses an LLM judge to score whether that payload is sufficient to answer.

Configs:
    A: raw                 — no AVOID, no topic hints, no annotation, fetch top-1
    B: + AVOID             — production AVOID filter, otherwise raw
    C: + topic hints       — production site: hints, otherwise raw
    D: + source annotation — production describe_source header, otherwise raw
    E: production default  — all on, fetch top-1
    F: production + top-3  — all on, fetch top-3 bodies

Run from inside the augmentum container so searxng:8080 and the local
llama-server resolve:

    docker exec augmentum-augmentum-1 python /app/tests/eval_web_search_ablation.py \
        --model Qwen3.6-35B-A3B-IQ4_XS

Or override URLs for host-side runs:

    python tests/eval_web_search_ablation.py \
        --searxng-url http://localhost:8888 \
        --llm-url http://localhost:8091/v1 \
        --model Qwen3.6-35B-A3B-IQ4_XS
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from augmentum.tools.preferred_sources import (
    AVOID,
    describe_source,
    domain_quality,
    get_topic_sites,
)
from augmentum.tools.web import (
    _build_search_query,
    _rank_search_results_for_fetch,
    _render_search_results_text,
)
from augmentum.tools.web_fetch import WebFetchTool
from augmentum.tools.web_search import WebSearchTool

OUTPUT_DIR = Path(__file__).parent / "eval_results"


# ── Ablation config matrix ──────────────────────────────────────────────────

@dataclass
class Config:
    name: str
    label: str
    avoid: bool
    topic_hints: bool
    source_annotation: bool
    auto_fetch_top: int


CONFIGS: list[Config] = [
    Config("A", "raw",                avoid=False, topic_hints=False, source_annotation=False, auto_fetch_top=1),
    Config("B", "+ AVOID",            avoid=True,  topic_hints=False, source_annotation=False, auto_fetch_top=1),
    Config("C", "+ topic hints",      avoid=False, topic_hints=True,  source_annotation=False, auto_fetch_top=1),
    Config("D", "+ source annotation",avoid=False, topic_hints=False, source_annotation=True,  auto_fetch_top=1),
    Config("E", "production default", avoid=True,  topic_hints=True,  source_annotation=True,  auto_fetch_top=1),
    Config("F", "production + top-3", avoid=True,  topic_hints=True,  source_annotation=True,  auto_fetch_top=3),
    Config("G", "AVOID + top-3 only", avoid=True,  topic_hints=False, source_annotation=False, auto_fetch_top=3),
]


# ── 10-query MVP corpus ─────────────────────────────────────────────────────
# Curated to span the categories real Augmentum users hit. Each query is
# something a person would type verbatim (no engineered prompts), with a
# brief expected-answer-shape note for judge calibration.

@dataclass
class Query:
    id: str
    category: str
    query: str
    expects: str  # what a usable payload should contain


CORPUS: list[Query] = [
    Query("q01", "current_events",
          "what happened in the latest Apple WWDC keynote",
          "specific announcements from the most recent WWDC"),
    Query("q02", "code_lookup",
          "how to debounce a function in vanilla javascript",
          "working code example using setTimeout/clearTimeout"),
    Query("q03", "weather_live",
          "weather in Tokyo this weekend",
          "actual forecast for Tokyo, ideally with temps and conditions"),
    Query("q04", "recipe_listicle",
          "easy weeknight dinner ideas",
          "5+ specific dinner recipes with actual instructions, not just a list of cuisines"),
    Query("q05", "diy_pinterest_bait",
          "small space living room ideas",
          "concrete design tactics (layout, multi-use furniture, color choices), not just inspo boards"),
    Query("q06", "product_research",
          "best mechanical keyboard for programming under 200 dollars",
          "specific model recommendations with rationale"),
    Query("q07", "history",
          "what caused the fall of the Western Roman Empire",
          "substantive causes (economic, military, political), not just dates"),
    Query("q08", "opinion_synthesis",
          "is learning to code worth it in 2026",
          "balanced take with arguments both directions, not pure hype or pure doom"),
    Query("q09", "regulation",
          "California rules on right turn on red",
          "the actual rule + any exceptions, ideally citing the vehicle code"),
    Query("q10", "long_tail",
          "what is the wespeaker runtime used for",
          "specific to the wespeakerruntime python package — speaker verification"),
]


# ── Pipeline (ablation-aware) ───────────────────────────────────────────────

@dataclass
class RunResult:
    query_id: str
    config: str
    search_query_sent: str = ""
    results_total: int = 0
    results_after_filter: int = 0
    results_fetched: int = 0
    fetched_urls: list[str] = field(default_factory=list)
    avoid_dropped: list[str] = field(default_factory=list)
    payload_chars: int = 0
    payload: str = ""
    latency_ms: int = 0
    error: str = ""
    judge_score: int | None = None
    judge_reason: str = ""


async def run_search(
    q: Query,
    cfg: Config,
    search_tool: WebSearchTool,
    fetch_tool: WebFetchTool,
    fetch_max_chars: int = 12000,
) -> RunResult:
    r = RunResult(query_id=q.id, config=cfg.name)
    t0 = time.monotonic()

    try:
        # 1. Pre-search: optional topic site: hints
        search_query = _build_search_query(q.query) if cfg.topic_hints else q.query
        r.search_query_sent = search_query

        sr = await search_tool.execute(query=search_query, num_results=8, categories="general")
        if not sr.success:
            r.error = f"search_failed: {sr.error}"
            return r

        raw_results = list(sr.metadata.get("results", []))
        r.results_total = len(raw_results)

        # 2. Post-search: optional AVOID filter
        if cfg.avoid:
            ranked = _rank_search_results_for_fetch(q.query, raw_results)
            kept_urls = {x.get("url", "") for x in ranked}
            r.avoid_dropped = [
                x.get("url", "") for x in raw_results
                if x.get("url") and x.get("url") not in kept_urls
            ]
        else:
            # No filter, no sort — match what raw SearXNG returns
            ranked = [x for x in raw_results if x.get("url")]
            r.avoid_dropped = []

        r.results_after_filter = len(ranked)
        if not ranked:
            r.payload = _render_search_results_text(raw_results)
            r.payload_chars = len(r.payload)
            return r

        # 3. Render search-results block
        sections: list[str] = [
            f"Search results for '{q.query}':\n\n{_render_search_results_text(ranked)}"
        ]

        # 4. Fetch top-N
        fetch_urls = [x.get("url", "") for x in ranked if x.get("url")][: cfg.auto_fetch_top]
        for url in fetch_urls:
            fr = await fetch_tool.execute(url=url, max_chars=fetch_max_chars)
            if not fr.success or not (fr.output and fr.output.strip()):
                continue
            header = f"--- Content from {url} ---"
            if cfg.source_annotation:
                desc = describe_source(url)
                if desc:
                    header += f"\n{desc}"
            sections.append(f"{header}\n{fr.output}")
            r.fetched_urls.append(url)

        r.results_fetched = len(r.fetched_urls)
        r.payload = "\n\n".join(sections)
        r.payload_chars = len(r.payload)

    except Exception as exc:
        r.error = f"run_failed: {exc!s}"
    finally:
        r.latency_ms = int((time.monotonic() - t0) * 1000)

    return r


# ── Judge ───────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = (
    "You are a strict JSON-emitting judge. You MUST respond with EXACTLY one JSON "
    "object and NOTHING else — no preamble, no commentary, no markdown fences, no "
    "trailing text. The JSON object must have exactly two keys: \"score\" (integer "
    "1-5) and \"reason\" (string, under 25 words). Do NOT attempt to answer the "
    "user's query; only judge the payload."
)

_JUDGE_USER = """USER QUERY: {query}

EXPECTED ANSWER SHAPE: {expects}

PAYLOAD THE ASSISTANT WOULD SEE:
---
{payload}
---

Score how well the payload would let an assistant answer the query in the SAME turn (no follow-up searches):
  5 = clear, direct, complete answer
  4 = enough for a confident answer with minor synthesis
  3 = partial — useful-but-incomplete answer possible
  2 = thin — mostly guessing or hedging
  1 = useless — payload doesn't address the query

Reply with EXACTLY one JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}"""


def _extract_first_json(text: str) -> dict | None:
    """Pull the first complete JSON object out of `text`, tolerating prose
    before/after and trailing junk that breaks json.loads."""
    start = text.find("{")
    if start == -1:
        return None
    try:
        obj, _end = json.JSONDecoder().raw_decode(text[start:])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


async def judge(
    q: Query,
    payload: str,
    http_client: httpx.AsyncClient,
    llm_url: str,
    model: str,
    api_key: str = "",
) -> tuple[int | None, str]:
    if not payload.strip():
        return 1, "empty payload"

    capped = payload if len(payload) <= 24000 else payload[:24000] + "\n[...truncated]"
    user_msg = _JUDGE_USER.format(query=q.query, expects=q.expects, payload=capped)

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        resp = await http_client.post(
            f"{llm_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                "temperature": 0.0,
                "max_tokens": 250,
                "stream": False,
                # Qwen 3.x / GLM 4.x / EXAONE 4.x respect this — keep the judge
                # out of thinking mode so it doesn't burn 2000 tokens reasoning
                # about a single 1-5 score and then run out of budget for content.
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=180.0,
        )
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        text = (msg.get("content") or "").strip()
        # Fallback: some Qwen variants still emit JSON inside reasoning_content
        # if the chat-template kwarg isn't honored. Try that before giving up.
        if not text:
            text = (msg.get("reasoning_content") or "").strip()
        parsed = _extract_first_json(text)
        if parsed is None:
            return None, f"no_json: {text[:200]}"
        score = int(parsed.get("score", 0))
        if score < 1 or score > 5:
            return None, f"bad_score: {score}"
        return score, str(parsed.get("reason", ""))[:200]
    except Exception as exc:
        return None, f"judge_failed: {exc!s}"


# ── Orchestration ───────────────────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--searxng-url", default="http://searxng:8080")
    parser.add_argument("--llm-url", default="http://127.0.0.1:8091/v1", help="LLM endpoint — defaults to llama-server direct (no auth, no per-user context injection)")
    parser.add_argument("--warmup-url", default="http://127.0.0.1:6100/v1", help="Augmentum URL to ping once at start so it loads the model into llama-server")
    parser.add_argument("--model", required=True, help="answerer model id (only used to label the run — actual answers come from the judge)")
    parser.add_argument("--judge-model", default="", help="model used for judging (defaults to --model)")
    parser.add_argument("--api-key", default="", help="Augmentum API key (sk-aug-...) used only for the warmup call")
    parser.add_argument("--n", type=int, default=10, help="how many queries from CORPUS to run")
    parser.add_argument("--configs", default="A,B,C,D,E,F,G", help="comma-separated config IDs")
    parser.add_argument("--out", default="", help="output JSON path (default auto-timestamped under writable dir)")
    parser.add_argument("--skip-judge", action="store_true", help="capture payloads, skip scoring")
    parser.add_argument("--throttle-seconds", type=float, default=0.0, help="sleep between runs to avoid CAPTCHA-ing SearXNG engines")
    args = parser.parse_args()

    if args.out:
        out_path = Path(args.out)
    else:
        for candidate in (OUTPUT_DIR, Path("/data/eval_results"), Path("/tmp")):
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                test_file = candidate / ".write_test"
                test_file.write_text("ok")
                test_file.unlink()
                out_path = candidate / f"web_search_ablation_{int(time.time())}.json"
                break
            except OSError:
                continue
        else:
            raise RuntimeError("no writable output directory found")

    selected_cfgs = [c for c in CONFIGS if c.name in {x.strip() for x in args.configs.split(",")}]
    queries = CORPUS[: args.n]
    total = len(queries) * len(selected_cfgs)
    judge_model = args.judge_model or args.model
    print(f"running {len(queries)} queries × {len(selected_cfgs)} configs = {total} runs")
    print(f"searxng={args.searxng_url}  llm={args.llm_url}  answerer={args.model}  judge={judge_model}")
    print(f"throttle={args.throttle_seconds}s between runs  →  est. {total * (10 + args.throttle_seconds) / 60:.1f}min")
    print(f"output: {out_path}\n")

    async with httpx.AsyncClient() as client:
        # Warmup: poke Augmentum so it spins llama-server up with the answerer model loaded.
        # After this, judge calls hit llama-server directly (no auth, no per-user context).
        if not args.skip_judge and args.warmup_url and args.api_key:
            print("warming up model via Augmentum...", end=" ", flush=True)
            try:
                t0 = time.monotonic()
                wr = await client.post(
                    f"{args.warmup_url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {args.api_key}"},
                    json={"model": args.model, "messages": [{"role": "user", "content": "reply with the word OK and nothing else"}], "max_tokens": 5, "stream": False},
                    timeout=300.0,
                )
                print(f"status={wr.status_code} ({int(time.monotonic()-t0)}s)")
            except Exception as exc:
                print(f"WARMUP FAILED ({exc}); will try direct anyway")

        search_tool = WebSearchTool(client, base_url=args.searxng_url)
        fetch_tool = WebFetchTool(client)

        all_results: list[RunResult] = []
        for q in queries:
            for cfg in selected_cfgs:
                print(f"  {q.id} [{q.category:20s}] cfg {cfg.name} ({cfg.label})...", end=" ", flush=True)
                r = await run_search(q, cfg, search_tool, fetch_tool)
                if r.error:
                    print(f"ERR {r.error[:80]}")
                else:
                    print(f"{r.results_after_filter} kept, {r.results_fetched} fetched, "
                          f"{r.payload_chars}c, {r.latency_ms}ms")
                if not args.skip_judge and not r.error:
                    # Judge goes to --llm-url (default llama-server direct) → no auth needed
                    score, reason = await judge(q, r.payload, client, args.llm_url, judge_model, "")
                    r.judge_score = score
                    r.judge_reason = reason
                    print(f"      judge: {score} — {reason[:120]}")
                all_results.append(r)
                if args.throttle_seconds > 0:
                    await asyncio.sleep(args.throttle_seconds)

    # Persist
    out_path.write_text(json.dumps({
        "answerer_model": args.model,
        "judge_model": judge_model,
        "searxng_url": args.searxng_url,
        "throttle_seconds": args.throttle_seconds,
        "queries": [asdict(q) for q in queries],
        "configs": [asdict(c) for c in selected_cfgs],
        "results": [asdict(r) for r in all_results],
    }, indent=2))

    # Summary table
    print("\n── summary ──")
    print(f"{'query':6s} " + " ".join(f"{c.name:>4s}" for c in selected_cfgs))
    by_q: dict[str, dict[str, RunResult]] = {}
    for r in all_results:
        by_q.setdefault(r.query_id, {})[r.config] = r
    for q in queries:
        row = [q.id]
        for c in selected_cfgs:
            r = by_q.get(q.id, {}).get(c.name)
            row.append(str(r.judge_score) if (r and r.judge_score is not None) else " . ")
        print(f"{row[0]:6s} " + " ".join(f"{v:>4s}" for v in row[1:]))

    # Aggregate per-config means
    print("\n── per-config mean score ──")
    for c in selected_cfgs:
        scores = [r.judge_score for r in all_results if r.config == c.name and r.judge_score is not None]
        if scores:
            print(f"  {c.name} ({c.label:24s}) n={len(scores)}  mean={sum(scores)/len(scores):.2f}")
        else:
            print(f"  {c.name} ({c.label:24s}) n=0   (no judged runs)")

    print(f"\nfull results: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
