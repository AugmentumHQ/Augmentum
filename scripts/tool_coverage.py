"""Full tool-inventory coverage + chains, schema-driven.

Ensures EVERY one of the 98 tools + 59 verbs appears in training with its REAL
argument schema (so the model knows the whole inventory natively — the primer-as-key
thesis), and generates multi-tool CHAIN scenarios (composition). Reuses the coauthor
voice-law + gate + row-builder. Threaded, resumable.

Usage:
    AUGMENTUM_API_KEY=sk-aug-... python scripts/tool_coverage.py --out F:/Training/elite_select/rows_tools.jsonl --mode single --per 2
    python scripts/tool_coverage.py --out ... --mode chains --passes 4
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coauthor as ca  # VOICE_LAW, WRITERS, call_writer, parse_array, to_rows, API_KEY

SCHEMA_FILE = Path("docs/training/tool-schemas.json")
_lock = threading.Lock()


def load_inventory():
    d = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    cat = d["catalog"]
    tools = []
    for name, t in cat["tools"].items():
        tools.append({
            "name": name,
            "desc": (t.get("description") or "")[:300],
            "args": t.get("input_schema", {}),
            "tag": _tag_for(t.get("surfaces", {}), t.get("category", "")),
            "kind": "tool",
        })
    for vid, v in cat["verbs"].items():
        tools.append({
            "name": vid,
            "desc": (v.get("summary") or "")[:300],
            "args": v.get("arg_schema", {}),
            "tag": ":B",  # verbs are companion/voice actions
            "kind": "verb",
        })
    return tools


def _tag_for(surfaces, category):
    if surfaces.get("coder") and not surfaces.get("chat"):
        return ":-"
    if surfaces.get("artifact_studio"):
        return ":W"
    if surfaces.get("companion"):
        return ":B"
    if surfaces.get("voice"):
        return ":V"
    return ":C"


def _args_brief(schema):
    """Compact the input_schema to name:type(+required) lines for the prompt."""
    props = (schema or {}).get("properties", {})
    req = set((schema or {}).get("required", []))
    lines = []
    for k, v in props.items():
        t = v.get("type", "any")
        d = (v.get("description") or "")[:60]
        star = "*" if k in req else ""
        lines.append(f"{k}{star}: {t}" + (f" — {d}" if d else ""))
    return "\n".join(lines) or "(no arguments)"


def single_prompt(tool, n):
    return (
        f"TOOL: {tool['name']} — {tool['desc']}\n"
        f"ARGUMENTS (the call MUST use these exact names/types; * = required):\n"
        f"{_args_brief(tool['args'])}\n"
        f"SURFACE tag: {tool['tag']}\n\n"
        f"Write {n} short, in-voice scenarios where she uses THIS tool. The tool call's "
        f"\"name\" MUST be exactly \"{tool['name']}\" and its \"arguments\" MUST match the schema "
        f"above (correct keys, sensible values). Structure: human -> assistant{{think, tool}} -> "
        f"tool{{result: plausible+compact}} -> assistant{{think reacting, response}}. Keep her voice "
        f"(no assistant-tells). rejected_final = a hollow/assistant-speak version of the final response."
    )


# Natural chain groups (compose 2-4 of each set).
CHAIN_GROUPS = {
    "research_doc": ["web_search", "web_fetch", "research", "create_document", "export_markdown", "wikipedia"],
    "data_viz": ["web_search", "research", "create_chart", "create_spreadsheet", "export_csv", "calculator"],
    "image_flow": ["image_search", "image_generation", "convert_image", "remove_background"],
    "coder_flow": ["code_search", "code_grep", "find_files", "file_read", "code_edit", "test_run", "git", "shell_exec"],
    "build_app": ["research", "build_application", "create_chart", "image_generation", "create_presentation"],
    "knowledge": ["memory_recall", "wikipedia", "web_search", "create_document", "youtube"],
    "doc_process": ["document_parse", "create_chart", "json_tool", "export_csv", "text_analysis"],
    "companion_action": ["memory_recall", "media_recommendations", "memory.save", "note.create", "schedule_reminder"],
}


def chain_prompt(group_name, tools, n):
    blocks = []
    for t in tools:
        blocks.append(f"- {t['name']} — {t['desc'][:80]} | args: {_args_brief(t['args'])[:160]}")
    return (
        f"TOOLS AVAILABLE (compose 2-4 of these into a natural CHAIN):\n" + "\n".join(blocks) + "\n\n"
        f"Write {n} scenarios where she chains 2-4 of these tools to do something real — interleaved "
        f"think -> tool -> plausible result -> think reacting -> next tool -> ... -> final in-voice "
        f"response. Each tool call's name+arguments MUST be correct per the schemas above. Keep her "
        f"voice. rejected_final = a version that does it without verifying / hollow final response."
    )


def run(jobs, out, workers):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["metadata"].get("cov_key"))
                except Exception:
                    pass
    jobs = [j for j in jobs if j["key"] not in done]
    out_f = open(out, "a", encoding="utf-8")
    stats = {"chosen": 0, "rejected": 0, "drop": 0, "parse_fail": 0}
    import concurrent.futures as cf

    def work(job):
        arr = None
        for attempt in range(4):
            try:
                arr = ca.parse_array(ca.call_writer(job["prompt"], ca.WRITERS[attempt % len(ca.WRITERS)]))
                if arr:
                    break
            except urllib.error.HTTPError as e:
                time.sleep(2 ** attempt if e.code == 429 else 1)
            except Exception:
                time.sleep(1 + attempt)
        if not arr:
            with _lock:
                stats["parse_fail"] += 1
            return
        for sc in arr:
            if not isinstance(sc, dict):
                continue
            sc.setdefault("tag", job.get("tag", ":C"))
            built = ca.to_rows(sc)
            if not built:
                with _lock:
                    stats["drop"] += 1
                continue
            chosen, rejected = built
            chosen["metadata"]["source"] = "tool_coverage"
            chosen["metadata"]["cov_key"] = job["key"]
            with _lock:
                out_f.write(json.dumps(chosen, ensure_ascii=False) + "\n")
                stats["chosen"] += 1
                if rejected:
                    rejected["metadata"]["source"] = "tool_coverage"
                    rejected["metadata"]["cov_key"] = job["key"]
                    out_f.write(json.dumps(rejected, ensure_ascii=False) + "\n")
                    stats["rejected"] += 1
                out_f.flush()

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, _ in enumerate(cf.as_completed(futs), 1):
            if i % 20 == 0:
                print(f"  {i}/{len(jobs)} | chosen={stats['chosen']} rej={stats['rejected']} "
                      f"drop={stats['drop']} pf={stats['parse_fail']}")
    out_f.close()
    print(f"DONE: {stats['chosen']} chosen + {stats['rejected']} rejected ({stats['drop']} gated, "
          f"{stats['parse_fail']} parse-fail), {time.time()-t0:.0f}s -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["single", "chains", "both"], default="both")
    ap.add_argument("--per", type=int, default=2)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if not ca.API_KEY:
        print("Set AUGMENTUM_API_KEY", file=sys.stderr)
        sys.exit(1)

    inv = load_inventory()
    by_name = {t["name"]: t for t in inv}
    jobs = []

    if args.mode in ("single", "both"):
        for p in range(args.passes):
            for t in inv:
                jobs.append({"key": f"single::{t['name']}::p{p}", "tag": t["tag"],
                             "prompt": single_prompt(t, args.per)})

    if args.mode in ("chains", "both"):
        passes = max(args.passes, 4)
        for p in range(passes):
            for gname, names in CHAIN_GROUPS.items():
                tools = [by_name[n] for n in names if n in by_name]
                if len(tools) < 2:
                    continue
                tag = ":-" if gname == "coder_flow" else ":W" if gname in ("build_app", "data_viz") else ":C"
                jobs.append({"key": f"chain::{gname}::p{p}", "tag": tag,
                             "prompt": chain_prompt(gname, tools, args.per)})

    print(f"inventory: {len(inv)} tools/verbs | jobs: {len(jobs)} | mode={args.mode}")
    run(jobs, args.out, args.workers)


if __name__ == "__main__":
    main()
