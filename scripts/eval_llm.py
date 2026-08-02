"""End-to-end LLM bug-detector eval against ground-truth fixtures.

Operates outside the orchestrator's container path — feeds each
fixture's source directly to an LLM detector prompt and scores the
resulting findings. This is NOT the full pipeline (no verifier, no
PoC construction, no fixer); it's specifically measuring "can the
LLM detector find the bugs the deterministic scanners can't?".

For the full-pipeline eval, see eval_runner.py (requires the
orchestrator running in a container).

Usage:
    # Default: claude-sonnet-4-6 via chatgpt-bridge at localhost:8788
    python scripts/eval_llm.py

    # Different model
    python scripts/eval_llm.py --model claude-opus-4-7

    # Different endpoint (e.g. local Augmentum)
    python scripts/eval_llm.py --api-base http://localhost:8080/v1

    # Pair with deterministic findings — give LLM the substrate's
    # candidates as additional context (more realistic to production)
    python scripts/eval_llm.py --include-substrate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[union-attr]
except AttributeError:
    pass

from augmentum.bug_finder.agnostic_stage import run_agnostic_stage


_FIXTURES_DIR = (
    Path(__file__).resolve().parent.parent
    / "tests" / "bug_finder_fixtures"
)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


_DETECTOR_PROMPT = """You are a careful, disproof-oriented security \
auditor. Your job: find real bugs in the given Python file. NOT \
style nits, NOT theoretical risks, NOT pattern-matched false \
positives — actual exploitable or incorrect-behavior bugs.

Rules:
1. For EACH bug you find, ask: can I construct a concrete input that \
exploits or triggers this? If not, do NOT report it.
2. Prefer FEWER, HIGHER-QUALITY findings over many noisy ones.
3. Report your findings as a JSON object inside a ```json fenced \
block. The exact schema:

```json
{
  "findings": [
    {
      "file": "<filename>",
      "line": <integer>,
      "severity": "high|medium|low",
      "claim_signature": "injection|null_deref|bounds_check|race|\
auth_bypass|missing_validation|resource_leak|use_after_free|\
deadlock|logic_error|type_confusion|other",
      "claim": "<one-sentence description of the bug>",
      "exploit_sketch": "<concrete input or trigger that demonstrates it>"
    }
  ]
}
```

If the file has NO real bugs (or all candidates fail step 1's \
disproof test), return `{"findings": []}`. Do not invent bugs to \
seem thorough.

Common patterns to watch for:
- SQL/shell/code injection via string interpolation
- Auth bypass via missing user_id / authz checks
- Null dereference through Optional/None paths
- Off-by-one in bounds checks
- Resource leaks (file/db/socket not closed)
- TOCTOU races between check and use
- Path traversal via unvalidated user input
- Pickle/eval/exec of untrusted data

Common FALSE positives to skip:
- f-strings with ONLY validated/whitelisted variables in SQL
- pickle of LOCAL/trusted cache data
- shell=True on hardcoded commands"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def _call_llm(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    file_path: str,
    source: str,
    timeout: float = 120.0,
) -> tuple[str, dict]:
    """Send a request to the OpenAI-compatible endpoint. Returns
    (response_text, raw_payload). Raises on HTTP errors."""
    user_msg = (
        f"File: `{file_path}`\n\n"
        f"```python\n{source}\n```\n\n"
        "Audit this file. Return your findings as the JSON object "
        "specified by the system instructions."
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0,
        "max_tokens": 4000,
    }
    req = urllib.request.Request(
        url=api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    payload = json.loads(raw)
    choices = payload.get("choices") or []
    if not choices:
        return "", payload
    msg = (choices[0] or {}).get("message") or {}
    return str(msg.get("content") or ""), payload


_JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)


def _parse_findings(response_text: str) -> list[dict]:
    """Extract the LLM's findings list from its response."""
    if not response_text:
        return []
    blocks = _JSON_BLOCK_RE.findall(response_text)
    if not blocks:
        # Maybe it returned a raw JSON object without fencing
        try:
            data = json.loads(response_text.strip())
        except json.JSONDecodeError:
            return []
        return _extract_findings_list(data)
    # Last block wins
    try:
        data = json.loads(blocks[-1].strip())
    except json.JSONDecodeError:
        return []
    return _extract_findings_list(data)


def _extract_findings_list(data) -> list[dict]:  # noqa: ANN001
    if isinstance(data, dict) and isinstance(data.get("findings"), list):
        return [d for d in data["findings"] if isinstance(d, dict)]
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return []


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


# Signatures the taxonomy treats as distinct but that map to the
# same underlying bug class. Path traversal, SSRF, and command/SQL
# injection all fall under "untrusted input flowing to a dangerous
# sink"; different detectors split them differently. Accept any
# member when the expected is the canonical representative.
_SIGNATURE_EQUIVALENTS: dict[str, set[str]] = {
    "injection": {
        "injection", "missing_validation", "type_confusion",
    },
    "missing_validation": {
        "missing_validation", "injection",
    },
    "auth_bypass": {
        "auth_bypass", "missing_validation",
    },
    "logic_error": {
        "logic_error", "bounds_check",
    },
    "bounds_check": {
        "bounds_check", "logic_error",
    },
    "race": {"race", "logic_error"},
    "use_after_free": {"use_after_free", "resource_leak"},
}


def _signatures_compatible(found: str, expected: str) -> bool:
    f = found.strip().lower()
    e = expected.strip().lower()
    if not f or not e:
        return f == e
    if f == e:
        return True
    return f in _SIGNATURE_EQUIVALENTS.get(e, {e})


def _matches(
    finding: dict,
    expected_sig: str,
    expected_file: str,
    expected_line_start: int,
    expected_line_end: int,
    line_tolerance: int = 3,
) -> bool:
    """Tolerant match: same file basename, line within ±tolerance,
    same signature family (or a documented equivalent)."""
    f_file = str(finding.get("file") or "")
    if not f_file.endswith(expected_file):
        return False
    try:
        f_line = int(finding.get("line") or 0)
    except (TypeError, ValueError):
        f_line = 0
    if f_line and not (
        expected_line_start - line_tolerance
        <= f_line
        <= expected_line_end + line_tolerance
    ):
        return False
    f_sig = str(finding.get("claim_signature") or "").strip().lower()
    return _signatures_compatible(f_sig, expected_sig)


@dataclass
class FixtureResult:
    fixture_id: str
    kind: str
    expected_count: int
    tp: int = 0
    fn: int = 0
    fp: int = 0
    raw_response: str = ""
    sample_fp_descs: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    wallclock_s: float = 0.0

    @property
    def detected_all(self) -> bool:
        return self.expected_count > 0 and self.fn == 0


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _eval_one(
    fixture_dir: Path,
    *,
    api_base: str,
    api_key: str,
    model: str,
    include_substrate: bool,
) -> FixtureResult:
    spec = json.loads(
        (fixture_dir / "expected.json").read_text(encoding="utf-8"),
    )
    fid = spec.get("fixture_id") or fixture_dir.name
    kind = spec.get("kind") or "true_positive"
    expected = spec.get("expected_findings") or []
    res = FixtureResult(
        fixture_id=fid, kind=kind, expected_count=len(expected),
    )

    # Pick the source file (every fixture has a single bug.py)
    src_path = fixture_dir / "bug.py"
    if not src_path.is_file():
        src_path = next(
            (p for p in fixture_dir.iterdir()
             if p.is_file() and p.suffix == ".py"
             and p.name not in {"expected.json"}),
            None,
        )
    if src_path is None:
        return res
    source = src_path.read_text(encoding="utf-8")
    file_label = src_path.name

    # Optionally enrich with substrate findings
    extra_context = ""
    if include_substrate:
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory(prefix=f"llmeval_{fid}_") as tmp:
            ws = Path(tmp)
            (ws / "src").mkdir()
            shutil.copy(src_path, ws / "src" / file_label)
            sweep = run_agnostic_stage(ws, record_patterns=False)
            if sweep.seeded_findings:
                lines = ["Deterministic scanner candidates (verify; many are FPs):"]
                for f in sweep.seeded_findings[:10]:
                    lines.append(
                        f"  - {f.severity} {f.function} "
                        f"L{f.evidence_paths[0].split(':')[-1] if f.evidence_paths else '?'}: "
                        f"{f.claim[:120]}"
                    )
                extra_context = "\n".join(lines) + "\n\n"

    prompt = _DETECTOR_PROMPT
    if extra_context:
        prompt = _DETECTOR_PROMPT + "\n\nContext from deterministic scanners:\n" + extra_context

    t0 = time.monotonic()
    try:
        response_text, payload = _call_llm(
            api_base=api_base, api_key=api_key, model=model,
            prompt=prompt, file_path=file_label, source=source,
        )
        res.tokens_in = int(((payload or {}).get("usage") or {}).get("prompt_tokens") or 0)
        res.tokens_out = int(((payload or {}).get("usage") or {}).get("completion_tokens") or 0)
    except urllib.error.HTTPError as e:
        res.raw_response = f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:200]}"
        return res
    except Exception as e:    # noqa: BLE001
        res.raw_response = f"ERROR: {type(e).__name__}: {e}"
        return res
    res.wallclock_s = round(time.monotonic() - t0, 2)
    res.raw_response = response_text

    findings = _parse_findings(response_text)
    unmatched = list(findings)
    for exp in expected:
        sig = exp.get("signature", "")
        f_basename = exp.get("file", "")
        line_start = exp.get("line_start", 0)
        line_end = exp.get("line_end", 0)
        hit = None
        for f in unmatched:
            if _matches(f, sig, f_basename, line_start, line_end):
                hit = f
                break
        if hit is not None:
            res.tp += 1
            unmatched.remove(hit)
        else:
            res.fn += 1
    for f in unmatched:
        res.fp += 1
        if len(res.sample_fp_descs) < 3:
            res.sample_fp_descs.append(
                f"L{f.get('line')} {f.get('claim_signature')}: "
                f"{str(f.get('claim'))[:100]}"
            )
    return res


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM detector eval against ground-truth fixtures",
    )
    parser.add_argument(
        "--api-base", default=os.environ.get("EVAL_API_BASE", "http://localhost:8788/v1"),
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("EVAL_API_KEY", "sk-bridge"),
    )
    parser.add_argument(
        "--model", default=os.environ.get("EVAL_MODEL", "claude-sonnet-4-6"),
    )
    parser.add_argument(
        "--include-substrate", action="store_true",
        help="pair the LLM detector with substrate scanner findings as context",
    )
    parser.add_argument(
        "--only", default="",
        help="comma-separated fixture ids (default: all)",
    )
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    fixtures = sorted(
        p for p in _FIXTURES_DIR.iterdir()
        if p.is_dir() and (p / "expected.json").is_file()
        and (not only or p.name in only)
    )
    if not fixtures:
        print(f"ERROR: no fixtures matching --only={args.only}", file=sys.stderr)
        return 2

    print(f"=== LLM detector ({args.model}) vs. {len(fixtures)} fixtures ===")
    print(f"api: {args.api_base}")
    print(f"substrate context: {'YES' if args.include_substrate else 'no'}")
    print()

    results: list[FixtureResult] = []
    for fx in fixtures:
        r = _eval_one(
            fx,
            api_base=args.api_base, api_key=args.api_key,
            model=args.model, include_substrate=args.include_substrate,
        )
        results.append(r)
        flag = ""
        if r.kind == "true_positive":
            flag = "OK" if r.detected_all else "MISSED"
        elif r.kind == "fp_bait":
            flag = "BAIT-HIT" if r.fp else "clean"
        elif r.kind == "red_herring":
            flag = "PASS" if r.fp == 0 else "TRIPPED"
        print(
            f"{r.fixture_id:<35} {r.kind:<14} "
            f"exp={r.expected_count:>2} tp={r.tp:>2} fn={r.fn:>2} fp={r.fp:>2}  "
            f"{r.wallclock_s:>5.1f}s {r.tokens_in + r.tokens_out:>5d}tok  {flag}"
        )
        for s in r.sample_fp_descs[:1]:
            print(f"    FP: {s}")
        if r.raw_response.startswith(("HTTP ", "ERROR:")):
            print(f"    {r.raw_response[:120]}")

    # Aggregate
    tp_total = sum(r.tp for r in results if r.kind == "true_positive")
    fn_total = sum(r.fn for r in results if r.kind == "true_positive")
    fp_total = sum(r.fp for r in results)
    bait_hits = sum(r.fp for r in results if r.kind == "fp_bait")
    rh_hits = sum(r.fp for r in results if r.kind == "red_herring")

    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    tokens = sum(r.tokens_in + r.tokens_out for r in results)
    wallclock = sum(r.wallclock_s for r in results)

    print()
    print("=== summary ===")
    print(f"true positives:   {tp_total}")
    print(f"false negatives:  {fn_total}   (real bugs missed)")
    print(f"false positives:  {fp_total}   ({bait_hits} on FP bait, {rh_hits} on red herring)")
    print(f"recall:           {recall:.2%}  ({tp_total}/{tp_total + fn_total})")
    print(f"precision:        {precision:.2%}  ({tp_total}/{tp_total + fp_total})")
    print(f"F1:               {f1:.2f}")
    print(f"cost:             {tokens} tokens / {wallclock:.1f}s wallclock")

    tp_fixtures = [r for r in results if r.kind == "true_positive"]
    bait_fixtures = [r for r in results if r.kind == "fp_bait"]
    found_all = sum(1 for r in tp_fixtures if r.detected_all)
    clean_bait = sum(1 for r in bait_fixtures if r.fp == 0)
    print(
        f"detected all expected bugs in: {found_all}/{len(tp_fixtures)} TP fixtures",
    )
    print(
        f"avoided FP bait in:            {clean_bait}/{len(bait_fixtures)} bait fixtures",
    )
    return 0 if recall >= 0.75 and bait_hits == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
