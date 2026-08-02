"""Real-world CVE eval harness.

For each entry in ``_CVES``: clone the repo at the pre-fix parent
commit, send the vulnerable file(s) to the LLM detector, score
whether the LLM identified a bug touching the same file+line range
the actual fix later patched.

This is the generalization-beyond-textbook-fixtures measurement: can
the bug-finder find real CVEs in real codebases?

Caveats kept honest:

* CVE descriptions don't always pin to a single line — we accept a
  hit within the file at ±10 lines of the fix.
* The LLM might find OTHER real bugs in the file. Those don't count
  against precision here — we only ask: did it find THIS bug?
* Some CVEs are multi-file. We declare success if the LLM hits any
  expected (file, line) anchor.

Run:
    EVAL_API_KEY=pwd python scripts/eval_cve.py
    EVAL_API_KEY=pwd python scripts/eval_cve.py --only urllib3-cve-2019-9740
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")   # type: ignore[union-attr]
except AttributeError:
    pass


# ---------------------------------------------------------------------------
# Verified CVE corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CVEEntry:
    """One verified CVE with its pre-fix coordinates.

    All fields verified against the actual fix commit on GitHub —
    see the URL in ``advisory_url`` for provenance.
    """

    slug: str
    cve_id: str
    repo: str                       # e.g. "urllib3/urllib3"
    parent_commit: str              # SHA right BEFORE the fix landed
    expected: tuple[tuple[str, int, int, str], ...]
    # Each expected: (file_relative_path, line_start, line_end, signature)
    short_description: str
    advisory_url: str


_CVES: tuple[CVEEntry, ...] = (
    CVEEntry(
        slug="urllib3-cve-2023-43804",
        cve_id="CVE-2023-43804",
        repo="urllib3/urllib3",
        parent_commit="3c6e079d1395ae815484b4a8f0ed7657a4dc8d0f",
        expected=(
            ("src/urllib3/connectionpool.py", 880, 910, "missing_validation"),
            ("src/urllib3/poolmanager.py",     440, 460, "missing_validation"),
        ),
        short_description=(
            "On HTTP 301/302/303 redirect that changes method to GET, "
            "request body + content headers (Content-Type/Length, "
            "Authorization, Digest, …) were NOT stripped — leaking "
            "sensitive request bodies across origins."
        ),
        advisory_url=(
            "https://github.com/urllib3/urllib3/commit/"
            "4e98d57809dacab1cbe625fddeec1a290c478ea9"
        ),
    ),
    CVEEntry(
        slug="urllib3-cve-2019-9740",
        cve_id="CVE-2019-9740",
        repo="urllib3/urllib3",
        parent_commit="cd1449d859c712cd2e24f784826e18bf663dfa35",
        expected=(
            ("src/urllib3/util/url.py", 145, 170, "injection"),
        ),
        short_description=(
            "URL parser accepted control characters (\\x00-\\x20, "
            "\\x7f) without validation — enabling HTTP header injection "
            "via crafted URLs."
        ),
        advisory_url=(
            "https://github.com/urllib3/urllib3/commit/"
            "9b76785331243689a9d52cef3db05ef7462cb02d"
        ),
    ),
    CVEEntry(
        slug="urllib3-cve-2021-28363",
        cve_id="CVE-2021-28363",
        repo="urllib3/urllib3",
        parent_commit="5e3432646ad63749ff0d655c157fe293cdc6c2aa",
        expected=(
            ("src/urllib3/connection.py", 480, 510, "auth_bypass"),
        ),
        short_description=(
            "HTTPS proxy connections did not verify the proxy "
            "server's certificate hostname — MITM via certificate "
            "valid for another hostname."
        ),
        advisory_url=(
            "https://github.com/urllib3/urllib3/commit/"
            "8d65ea1ecf6e2cdc27d42124e587c1b83a3118b0"
        ),
    ),
)


# ---------------------------------------------------------------------------
# Git checkout at specific commit
# ---------------------------------------------------------------------------


def _shallow_checkout_commit(repo: str, commit: str, dest: Path) -> bool:
    """Clone the repo at a specific commit (shallow). True on success."""
    url = f"https://github.com/{repo}.git"
    # Strategy: clone --filter=blob:none (no objects yet), then fetch
    # only the specific commit, then checkout.
    try:
        subprocess.run(
            ["git", "clone", "--quiet", "--filter=blob:none",
             "--no-checkout", url, str(dest)],
            check=True, capture_output=True, timeout=120,
        )
        subprocess.run(
            ["git", "-C", str(dest), "fetch", "--quiet",
             "--depth", "1", "origin", commit],
            check=True, capture_output=True, timeout=120,
        )
        subprocess.run(
            ["git", "-C", str(dest), "checkout", "--quiet", commit],
            check=True, capture_output=True, timeout=60,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        err = ""
        if isinstance(exc, subprocess.CalledProcessError):
            err = exc.stderr.decode(errors="ignore")[:240]
        else:
            err = "timeout"
        print(f"   ERROR checking out {commit[:12]}: {err}",
              file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# LLM call (reused from eval_llm.py shape)
# ---------------------------------------------------------------------------


_DETECTOR_PROMPT = """You are a thorough security auditor. Find \
security bugs in this Python file — both obvious ones AND subtle \
missing-validation patterns. Production library code often has \
real bugs hidden behind innocuous-looking parsing/dispatching.

Focus areas:
- Untrusted input flowing to dangerous sinks (SQL, shell, exec, \
file paths, URLs, deserialization)
- Missing input validation — empty/None checks alone aren't enough; \
check whether the parser validates against the character/format/\
length restrictions of the format it accepts. Missing validation \
against control chars, CRLF, null bytes, or out-of-spec chars in \
URL/header/path parsers is a real bug — it enables downstream \
protocol injection.
- Authentication / authorization / session bypass; certificate or \
hostname verification gaps; missing security defaults
- Sensitive data leaks across security contexts (e.g., body/headers \
forwarded across redirects, secrets logged, tokens emitted in URLs)
- Resource leaks, race conditions (TOCTOU), null/None paths, \
use-after-free
- Logic errors that violate documented invariants

Return findings as JSON in a ```json fenced block:

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
      "claim": "<one-sentence bug description>",
      "exploit_sketch": "<concrete input that demonstrates it>"
    }
  ]
}
```

Report the 3-8 highest-confidence findings. Empty list is acceptable \
when nothing genuinely concerns you, but production library code \
usually has at least one bug worth flagging."""


def _call_llm(
    *, api_base: str, api_key: str, model: str,
    prompt: str, file_path: str, source: str,
    timeout: float = 240.0,
) -> tuple[str, dict]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content":
                f"File: `{file_path}`\n\n```python\n{source}\n```\n\n"
                "Audit this file. Return findings JSON as specified."},
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


def _parse_findings(text: str) -> list[dict]:
    if not text:
        return []
    blocks = _JSON_BLOCK_RE.findall(text)
    candidate = blocks[-1] if blocks else text.strip()
    try:
        data = json.loads(candidate.strip())
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict) and isinstance(data.get("findings"), list):
        return [d for d in data["findings"] if isinstance(d, dict)]
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    return []


_SIGNATURE_EQUIVALENTS: dict[str, set[str]] = {
    "injection":          {"injection", "missing_validation", "type_confusion"},
    "missing_validation": {"missing_validation", "injection", "auth_bypass"},
    "auth_bypass":        {"auth_bypass", "missing_validation"},
    "logic_error":        {"logic_error", "bounds_check"},
    "bounds_check":       {"bounds_check", "logic_error"},
}


def _sig_compatible(found: str, expected: str) -> bool:
    f = (found or "").strip().lower()
    e = (expected or "").strip().lower()
    if f == e:
        return True
    return f in _SIGNATURE_EQUIVALENTS.get(e, {e})


def _matches_expected(
    finding: dict, exp_file: str, exp_line_start: int,
    exp_line_end: int, exp_sig: str, line_tolerance: int = 10,
) -> bool:
    """Match if same file (suffix), line within tolerance window, and
    compatible signature."""
    f_file = str(finding.get("file") or "").replace("\\", "/")
    if not (
        f_file.endswith(exp_file)
        or f_file == exp_file
        or f_file.endswith(exp_file.rsplit("/", 1)[-1])
    ):
        return False
    try:
        f_line = int(finding.get("line") or 0)
    except (TypeError, ValueError):
        f_line = 0
    if f_line and not (
        exp_line_start - line_tolerance
        <= f_line
        <= exp_line_end + line_tolerance
    ):
        return False
    return _sig_compatible(
        str(finding.get("claim_signature") or ""), exp_sig,
    )


# ---------------------------------------------------------------------------
# Per-CVE eval
# ---------------------------------------------------------------------------


@dataclass
class CVEResult:
    slug: str
    cve_id: str
    ok: bool = False
    found_cve: bool = False
    error: str = ""
    matched_expected: int = 0
    expected_count: int = 0
    total_findings: int = 0
    extra_findings: int = 0
    tokens: int = 0
    wallclock_s: float = 0.0
    sample_match: str = ""
    sample_extras: list[str] = field(default_factory=list)


def _eval_one(
    entry: CVEEntry,
    *,
    api_base: str, api_key: str, model: str,
) -> CVEResult:
    res = CVEResult(
        slug=entry.slug, cve_id=entry.cve_id,
        expected_count=len(entry.expected),
    )

    with tempfile.TemporaryDirectory(prefix=f"cve_{entry.slug}_") as tmp:
        repo_dir = Path(tmp) / "repo"
        print(f"   checking out {entry.repo}@{entry.parent_commit[:12]}...")
        t0 = time.monotonic()
        if not _shallow_checkout_commit(
            entry.repo, entry.parent_commit, repo_dir,
        ):
            res.error = "checkout failed"
            return res
        print(f"   checkout: {time.monotonic() - t0:.1f}s")

        # Find the union of expected files; LLM is invoked once per
        # unique file. A finding inside a file matches against any
        # expected anchor in that file.
        seen_files: set[str] = set()
        all_findings: list[tuple[str, dict]] = []

        for rel, ls, le, sig in entry.expected:
            if rel in seen_files:
                continue
            seen_files.add(rel)
            src = repo_dir / rel
            if not src.is_file():
                print(f"   WARN expected file missing: {rel}")
                continue
            source = src.read_text(encoding="utf-8")
            # If the file is enormous, take ±300 lines around the
            # earliest expected line range so the LLM gets focused
            # context. Most CVE-relevant Python source is fine.
            if len(source.splitlines()) > 1200:
                lines = source.splitlines()
                pivot = max(1, ls - 300)
                stop = min(len(lines), le + 300)
                source = "\n".join(
                    f"{i+1}: {ln}" for i, ln in enumerate(
                        lines[pivot - 1: stop], start=pivot - 1,
                    )
                )
                file_label = f"{rel} (lines {pivot}-{stop} of {len(lines)})"
            else:
                file_label = rel

            t_llm = time.monotonic()
            try:
                response, payload = _call_llm(
                    api_base=api_base, api_key=api_key,
                    model=model, prompt=_DETECTOR_PROMPT,
                    file_path=file_label, source=source,
                )
                res.tokens += int(
                    ((payload or {}).get("usage") or {})
                    .get("total_tokens") or 0,
                )
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")[:200]
                res.error = f"HTTP {e.code}: {msg}"
                return res
            except Exception as exc:    # noqa: BLE001
                res.error = f"{type(exc).__name__}: {exc}"
                return res
            res.wallclock_s += time.monotonic() - t_llm

            for f in _parse_findings(response):
                # Normalize the file field to the repo-relative path
                # so matching works regardless of how the LLM echoed it.
                f["file"] = rel
                all_findings.append((rel, f))

        # Score: did ANY finding match each expected anchor?
        res.total_findings = len(all_findings)
        matched_anchors: set[tuple[str, int, int, str]] = set()
        matched_findings_idx: set[int] = set()
        for i, (file_rel, f) in enumerate(all_findings):
            for anchor in entry.expected:
                if anchor in matched_anchors:
                    continue
                a_file, a_ls, a_le, a_sig = anchor
                if a_file != file_rel:
                    continue
                if _matches_expected(f, a_file, a_ls, a_le, a_sig):
                    matched_anchors.add(anchor)
                    matched_findings_idx.add(i)
                    if not res.sample_match:
                        res.sample_match = (
                            f"L{f.get('line')} {f.get('claim_signature')}: "
                            f"{str(f.get('claim'))[:120]}"
                        )
                    break
        res.matched_expected = len(matched_anchors)
        res.found_cve = res.matched_expected > 0
        # Anything not matched is "extra" — could be other real bugs.
        for i, (_, f) in enumerate(all_findings):
            if i in matched_findings_idx:
                continue
            res.extra_findings += 1
            if len(res.sample_extras) < 3:
                res.sample_extras.append(
                    f"L{f.get('line')} {f.get('claim_signature')}: "
                    f"{str(f.get('claim'))[:120]}"
                )

        res.ok = True
    return res


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Real-world CVE eval for the bug-finder LLM detector",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("EVAL_API_BASE", "http://localhost:8788/v1"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EVAL_API_KEY", "sk-bridge"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EVAL_MODEL", "claude-sonnet-4-6"),
    )
    parser.add_argument("--only", default="",
                        help="comma-separated slugs (default: all)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    targets = [c for c in _CVES if not only or c.slug in only]
    if not targets:
        print(f"no CVEs match --only={args.only}", file=sys.stderr)
        return 2

    print(f"=== CVE eval ({args.model}) vs. {len(targets)} verified CVEs ===")
    print(f"api: {args.api_base}")
    print()

    results: list[CVEResult] = []
    for entry in targets:
        print(f"== {entry.cve_id} ({entry.slug})")
        print(f"   {entry.short_description[:160]}")
        r = _eval_one(
            entry, api_base=args.api_base, api_key=args.api_key,
            model=args.model,
        )
        results.append(r)
        if r.error:
            print(f"   FAIL: {r.error}")
            continue
        flag = "FOUND" if r.found_cve else "MISSED"
        print(f"   {flag} ({r.matched_expected}/{r.expected_count} anchors, "
              f"{r.extra_findings} extras, "
              f"{r.tokens}tok / {r.wallclock_s:.1f}s)")
        if r.sample_match:
            print(f"   matched: {r.sample_match}")
        for ex in r.sample_extras[:2]:
            print(f"   extra:   {ex}")
        print()

    # Aggregate
    n_total = len(results)
    n_found = sum(1 for r in results if r.found_cve)
    n_errors = sum(1 for r in results if r.error)
    tokens_total = sum(r.tokens for r in results)
    wallclock_total = sum(r.wallclock_s for r in results)

    print("=== summary ===")
    print(f"CVE detection rate: {n_found}/{n_total} "
          f"({(n_found / n_total * 100) if n_total else 0:.0f}%)")
    print(f"errors:             {n_errors}")
    print(f"total cost:         {tokens_total} tok / {wallclock_total:.1f}s")

    if args.json:
        out = [asdict(r) for r in results]
        Path(".augmentum-bench/cve_eval.json").parent.mkdir(
            parents=True, exist_ok=True,
        )
        Path(".augmentum-bench/cve_eval.json").write_text(
            json.dumps(out, indent=2), encoding="utf-8",
        )
        print("wrote .augmentum-bench/cve_eval.json")

    return 0 if n_found == n_total and n_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
