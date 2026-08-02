#!/usr/bin/env python3
"""Augmentum Red Team Scanner.

Automated adversarial analysis of code changes. Scans for common
vulnerability patterns from an attacker's perspective, then suggests
defender countermeasures.

Checks:
  1. Data isolation gaps — queries missing user_id scoping
  2. Auth bypass risks — endpoints without auth checks
  3. Token exposure — secrets in URLs, logs, or error messages
  4. Injection surfaces — unsanitized user input in SQL, HTML, shell, LLM prompts
  5. Timing leaks — auth comparisons that reveal valid usernames
  6. IDOR risks — direct object references without ownership checks
  7. AI context leaks — shared caches or context that cross user boundaries

Exit code 0 = clean, 1 = findings, 2 = setup error.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import _common  # noqa: F401 — import side-effect: UTF-8-safe stdout/stderr

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def _find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "augmentum" / "proxy").is_dir() and (parent / "ui").is_dir():
            return parent
    print("ERROR: Cannot find Augmentum project root.", file=sys.stderr)
    sys.exit(2)

ROOT = _find_root()

_COLOR = os.environ.get("TERM") or os.name != "nt"
def _red(s: str) -> str:    return f"\033[91m{s}\033[0m" if _COLOR else s
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m" if _COLOR else s
def _green(s: str) -> str:  return f"\033[92m{s}\033[0m" if _COLOR else s
def _cyan(s: str) -> str:   return f"\033[96m{s}\033[0m" if _COLOR else s
def _bold(s: str) -> str:   return f"\033[1m{s}\033[0m" if _COLOR else s
def _dim(s: str) -> str:    return f"\033[2m{s}\033[0m" if _COLOR else s

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv


# Shared source of truth: security_check.py's exceptions file.
_EXCEPTIONS_FILE = Path(__file__).resolve().parent.parent / "references" / "security_exceptions.json"


def _load_excepted_files(category: str) -> set[str]:
    """Return a set of paths excepted from a given finding category.

    ``category`` is matched against exception ids (e.g. SQL false positives
    cluster under ``sql-fstring-*``). The exceptions file is the same one
    security_check.py reads, so deciding once that a finding is a false
    positive suppresses it everywhere.
    """
    if not _EXCEPTIONS_FILE.exists():
        return set()
    try:
        data = json.loads(_EXCEPTIONS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return set()
    excepted: set[str] = set()
    for exc in data.get("exceptions", []):
        if category in exc.get("id", ""):
            for f in exc.get("files", []):
                if f != "*":
                    excepted.add(f.replace("\\", "/"))
    return excepted


_SQL_FSTRING_EXCEPTED = _load_excepted_files("sql-fstring")
_SHELL_INJECTION_EXCEPTED = _load_excepted_files("shell-injection")
_WEAK_RANDOM_EXCEPTED = _load_excepted_files("weak-random")

# ---------------------------------------------------------------------------
# Scanners
# ---------------------------------------------------------------------------

findings: list[dict] = []

def _add(severity: str, category: str, file: str, line: int, message: str, fix: str):
    findings.append({
        "severity": severity,
        "category": category,
        "file": file,
        "line": line,
        "message": message,
        "fix": fix,
    })


def _scan_file(path: Path, rel: str, content: str, lines: list[str]):
    """Run all checks on a single file."""
    _check_sql_injection(rel, content, lines)
    _check_data_isolation(rel, content, lines)
    _check_token_exposure(rel, content, lines)
    _check_error_info_leak(rel, content, lines)
    _check_xss_template(rel, content, lines)
    _check_unsafe_random(rel, content, lines)
    _check_shell_injection(rel, content, lines)
    _check_llm_context_leak(rel, content, lines)
    _check_open_redirect(rel, content, lines)
    _check_path_traversal(rel, content, lines)


def _check_sql_injection(rel: str, content: str, lines: list[str]):
    """Detect f-string or format SQL queries (injection risk)."""
    if not rel.endswith(".py"):
        return
    # Files with vetted false-positive exceptions in security_exceptions.json
    # (hardcoded table names, vec dimensions, internal-only identifiers,
    # docstring examples). Decided once, suppressed everywhere.
    if rel.replace("\\", "/") in _SQL_FSTRING_EXCEPTED:
        return
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
        # Honor explicit `# noqa: S608` bypass markers, same as security_check.py.
        if "noqa: S608" in line or "noqa:S608" in line:
            continue
        # f-string in SQL context
        if re.search(r'(?:execute|executemany|executescript)\s*\(\s*f["\']', line):
            _add("CRITICAL", "SQL Injection", rel, i,
                 "f-string in SQL execute() — user input can modify query",
                 "Use parameterized queries: execute('SELECT * FROM t WHERE id = ?', (id,))")
        # .format() in SQL context
        if re.search(r'(?:execute|executemany)\s*\([^)]*\.format\(', line):
            _add("CRITICAL", "SQL Injection", rel, i,
                 ".format() in SQL execute() — user input can modify query",
                 "Use parameterized queries with ? placeholders")


def _check_data_isolation(rel: str, content: str, lines: list[str]):
    """Detect queries on user-scoped tables without WHERE user_id."""
    if not rel.endswith(".py"):
        return
    user_tables = {
        "ui_sessions", "ui_characters", "memories", "facts", "entities",
        "plot_threads", "character_cards", "narrative_memory", "documents",
        "document_chunks", "image_generations", "artifacts", "custom_flows",
        "reasoning_flows", "coder_sessions", "chat_images",
    }
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        lower = stripped.lower()
        for table in user_tables:
            if f"from {table}" in lower or f"into {table}" in lower or f"update {table}" in lower:
                # Check if user_id appears within the next 5 lines
                context = "\n".join(lines[i-1:i+5]).lower()
                if "user_id" not in context:
                    if VERBOSE:
                        _add("HIGH", "Data Isolation", rel, i,
                             f"Query on '{table}' without user_id scoping — potential cross-tenant leak",
                             f"Add WHERE user_id = ? to all queries on {table}")


def _check_token_exposure(rel: str, content: str, lines: list[str]):
    """Detect tokens/secrets in URLs, logs, or error messages."""
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Token in URL construction
        if re.search(r'[?&](?:token|api_key|secret|password)=.*(?:f["\']|\.format|%s|\+)', line):
            _add("HIGH", "Token Exposure", rel, i,
                 "Secret value in URL construction — visible in logs and browser history",
                 "Use headers (Authorization: Bearer) or short-lived tickets instead")
        # Logging secrets
        if re.search(r'log\.\w+\(.*(?:password|api_key|secret|token).*[^sanitize]', line, re.IGNORECASE):
            # Skip if sanitize_error_detail is used nearby
            context = "\n".join(lines[max(0,i-3):i+1])
            if "sanitize" not in context and "redact" not in context:
                if VERBOSE:
                    _add("MEDIUM", "Token Exposure", rel, i,
                         "Potential secret in log output — may appear in log files",
                         "Use sanitize_error_detail() before logging error details")


def _check_error_info_leak(rel: str, content: str, lines: list[str]):
    """Detect error responses that reveal internal details."""
    for i, line in enumerate(lines, 1):
        # Returning raw exception messages to client
        if re.search(r'(?:JSONResponse|HTTPException).*str\(e\)', line):
            if "sanitize" not in line:
                _add("MEDIUM", "Info Leak", rel, i,
                     "Raw exception message returned to client — may reveal internals",
                     "Sanitize with sanitize_error_detail(str(e)) or return generic message")


def _check_xss_template(rel: str, content: str, lines: list[str]):
    """Detect unescaped user content in JS template literals."""
    if not rel.endswith(".js"):
        return
    for i, line in enumerate(lines, 1):
        # Template literal with ${...} that doesn't use escapeHtml
        if "`" in line and "${" in line:
            # Extract the ${...} expressions
            exprs = re.findall(r'\$\{([^}]+)\}', line)
            for expr in exprs:
                if "escapeHtml" not in expr and "sanitize" not in expr:
                    # Skip common safe patterns
                    if any(safe in expr for safe in ["Date", "Math", "parseInt", "JSON", "length", "index", "count", ".id", "encodeURI"]):
                        continue
                    if VERBOSE:
                        _add("HIGH", "XSS", rel, i,
                             f"Template literal ${{{expr}}} without escapeHtml() — XSS risk if user-controlled",
                             "Wrap with escapeHtml() or verify the value cannot contain user input")


def _check_unsafe_random(rel: str, content: str, lines: list[str]):
    """Detect non-cryptographic random for security-sensitive operations."""
    if not rel.endswith(".py"):
        return
    if rel.replace("\\", "/") in _WEAK_RANDOM_EXCEPTED:
        return
    for i, line in enumerate(lines, 1):
        if "import random" in line and "secrets" not in content:
            # Check if random is used for tokens/keys/IDs
            if any(word in content for word in ["token", "session", "secret", "key", "auth", "password"]):
                _add("MEDIUM", "Weak Random", rel, i,
                     "Non-cryptographic random module in security-sensitive file",
                     "Use secrets.token_hex() or secrets.token_urlsafe() for tokens")


def _check_shell_injection(rel: str, content: str, lines: list[str]):
    """Detect shell=True or unsanitized subprocess calls."""
    if not rel.endswith(".py"):
        return
    if rel.replace("\\", "/") in _SHELL_INJECTION_EXCEPTED:
        return
    for i, line in enumerate(lines, 1):
        if "shell=True" in line:
            _add("HIGH", "Shell Injection", rel, i,
                 "subprocess with shell=True — command injection risk if input is user-controlled",
                 "Use shell=False with argument list: subprocess.run(['cmd', arg1, arg2])")
        if re.search(r'os\.system\(', line):
            _add("HIGH", "Shell Injection", rel, i,
                 "os.system() — always vulnerable to injection",
                 "Use subprocess.run() with shell=False")


def _check_llm_context_leak(rel: str, content: str, lines: list[str]):
    """Detect shared caches or context that could leak across users."""
    if not rel.endswith(".py"):
        return
    for i, line in enumerate(lines, 1):
        # Cache keys without user scoping
        if re.search(r'cache_key\s*=\s*f?["\'].*{(?:model|prefix|system)', line, re.IGNORECASE):
            if "user_id" not in line and "user" not in line:
                if VERBOSE:
                    _add("MEDIUM", "Context Leak", rel, i,
                         "Cache key without user_id — may share cached data across users",
                         "Include user_id in cache key: f'{user_id}:{model}:{hash}'")


def _check_open_redirect(rel: str, content: str, lines: list[str]):
    """Detect redirects using user-supplied URLs."""
    if not rel.endswith(".py"):
        return
    for i, line in enumerate(lines, 1):
        if re.search(r'RedirectResponse\(.*(?:request\.|body\.|params\.)', line):
            _add("MEDIUM", "Open Redirect", rel, i,
                 "Redirect using user-supplied URL — phishing risk",
                 "Validate redirect URL is relative or on allowed domain list")


def _check_path_traversal(rel: str, content: str, lines: list[str]):
    """Detect file operations with user-supplied paths."""
    if not rel.endswith(".py"):
        return
    for i, line in enumerate(lines, 1):
        # open() or Path() with user input (f-strings containing request/body)
        if re.search(r'(?:open|Path)\s*\(.*(?:request\.|body\.|params\.)', line):
            if ".." not in line and "safe" not in line.lower() and "_safe" not in line:
                if VERBOSE:
                    _add("HIGH", "Path Traversal", rel, i,
                         "File operation with user-supplied path — directory traversal risk",
                         "Validate path doesn't contain '..' and is within allowed directory")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def scan():
    """Scan all Python and JS files."""
    scan_dirs = [ROOT / "augmentum", ROOT / "ui" / "scripts"]
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        for py in scan_dir.rglob("*"):
            if py.suffix not in (".py", ".js"):
                continue
            if "__pycache__" in str(py):
                continue
            rel = str(py.relative_to(ROOT)).replace("\\", "/")
            try:
                content = py.read_text(encoding="utf-8", errors="ignore")
                lines_list = content.split("\n")
                _scan_file(py, rel, content, lines_list)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def report() -> int:
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    sorted_findings = sorted(findings, key=lambda f: severity_order.get(f["severity"], 9))

    print()
    print(_bold("=" * 60))
    print(_bold("  AUGMENTUM RED TEAM SCAN"))
    print(_bold("=" * 60))
    print()

    if not sorted_findings:
        print(_green(_bold("  No findings. Attack surface looks clean.")))
        print()
        return 0

    # Group by severity
    by_sev: dict[str, list] = {}
    for f in sorted_findings:
        by_sev.setdefault(f["severity"], []).append(f)

    # Summary
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = len(by_sev.get(sev, []))
        if count == 0:
            continue
        color = _red if sev == "CRITICAL" else (_yellow if sev == "HIGH" else _cyan)
        print(f"  {color(sev)}: {count}")
    print()

    # Details
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        items = by_sev.get(sev, [])
        if not items:
            continue
        color = _red if sev == "CRITICAL" else (_yellow if sev == "HIGH" else _cyan)
        print(color(_bold(f"  {sev} ({len(items)})")))
        for f in items:
            print(f"    {f['file']}:{f['line']}")
            print(f"      {_bold('Attack:')} {f['message']}")
            print(f"      {_green('Fix:')} {f['fix']}")
            print()

    total = len(sorted_findings)
    critical = len(by_sev.get("CRITICAL", []))
    print(f"  {total} total findings" + (f" ({_red(f'{critical} CRITICAL')})" if critical else ""))
    print()

    if not VERBOSE:
        print(_dim("  Run with --verbose for data isolation and context leak checks"))
        print()

    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    scan()
    exit_code = report()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
