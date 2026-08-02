"""Wrap the augmentum-dev scanner suite as agent-callable Python
functions.

The augmentum-dev skill ships ~11 deterministic scanners. Their
internal conventions differ — some expose ``scan() + findings``, some
emit stdout text. We support two integration styles:

1. **Native** — scanner exposes a module-level ``findings`` list and a
   ``scan()`` function (the pattern in ``red_team_scan.py``). We
   import + call directly, zero subprocess overhead. This is the
   preferred path for new scanners.

2. **Subprocess + stdout parse** — for scanners that print findings
   to stdout (most of the suite today), we run them via
   ``subprocess.run`` and parse the output line-by-line. The parser
   per scanner is small; the framework here factors out the common
   subprocess + per-scanner parsing.

This commit ships the **native** wrapper around ``red_team_scan``
(the highest-leverage scanner for the lead's pattern: adversarial
analysis of auth bypass / data isolation / token exposure / IDOR).
The subprocess wrappers for security_check, runtime_checks,
code_quality, dead_code, db_safety are scaffolded behind a uniform
``run_scanner`` interface — implementations land per scanner as we
need them. The agent tool layer treats every scanner identically.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from augmentum.bug_finder import refs
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScannerFinding:
    """One structured finding from a deterministic scanner."""

    scanner: str           # which scanner emitted this
    severity: str          # critical | high | medium | low | info
    category: str          # scanner-specific category
    file: str              # repo-relative path
    line: int = 0
    message: str = ""
    fix: str = ""
    rule_id: str = ""      # stable id derived from (scanner, category, file, line)


def _make_rule_id(scanner: str, category: str, file: str, line: int) -> str:
    blob = "|".join((scanner, category, file, str(line)))
    return "scan_" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


_VALID_SEVS = frozenset({"critical", "high", "medium", "low", "info"})


def _normalize_severity(value: Any) -> str:
    sev = str(value or "medium").strip().lower()
    return sev if sev in _VALID_SEVS else "medium"


def _normalize(scanner: str, raw: dict) -> ScannerFinding:
    file = str(raw.get("file") or "").strip()
    try:
        line = int(raw.get("line") or 0)
    except (TypeError, ValueError):
        line = 0
    # Scanners vary on field names — security_check uses ``description``,
    # red_team_scan uses ``message``. Accept either.
    category = str(
        raw.get("category") or raw.get("type") or "uncategorized",
    ).strip()
    message = str(
        raw.get("message") or raw.get("description") or "",
    ).strip()
    fix = str(
        raw.get("fix") or raw.get("recommendation") or "",
    ).strip()
    return ScannerFinding(
        scanner=scanner,
        severity=_normalize_severity(raw.get("severity")),
        category=category,
        file=file,
        line=line,
        message=message,
        fix=fix,
        rule_id=_make_rule_id(scanner, category, file, line),
    )


# ---------------------------------------------------------------------------
# Scanner location + capability check
# ---------------------------------------------------------------------------


_REL_SCRIPTS = (".claude", "skills", "augmentum-dev", "scripts")


def scanner_scripts_dir(root: Path) -> Path:
    return root.joinpath(*_REL_SCRIPTS)


def has_augmentum_dev_scanners(root: Path) -> bool:
    return scanner_scripts_dir(root).is_dir()


# ---------------------------------------------------------------------------
# Native (import + call) scanner runner
# ---------------------------------------------------------------------------


_MODULE_CACHE: dict[str, Any] = {}
_MODULE_CACHE_LOCK = threading.Lock()


def _load_scanner_module(filename: str, root: Path) -> Any | None:
    cache_key = f"{root.resolve()}|{filename}"
    with _MODULE_CACHE_LOCK:
        cached = _MODULE_CACHE.get(cache_key)
        if cached is not None:
            return cached

    script_path = scanner_scripts_dir(root) / filename
    if not script_path.is_file():
        return None

    scripts_dir_str = str(scanner_scripts_dir(root).resolve())
    if scripts_dir_str not in sys.path:
        sys.path.insert(0, scripts_dir_str)

    spec = importlib.util.spec_from_file_location(
        f"_aug_dev_scanner_{filename}", str(script_path),
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "bug_finder_scanner_load_failed",
            filename=filename, error=str(exc),
        )
        return None

    with _MODULE_CACHE_LOCK:
        _MODULE_CACHE[cache_key] = module
    return module


def _run_native_scanner(
    scanner_slug: str,
    filename: str,
    root: Path,
) -> list[dict]:
    """Run a scanner that exposes module-level ``findings`` + ``scan()``.

    Returns the raw list[dict] the scanner accumulated. Callers
    normalize via ``_normalize``.
    """
    module = _load_scanner_module(filename, root)
    if module is None:
        return []
    if hasattr(module, "findings"):
        try:
            module.findings.clear()
        except (AttributeError, TypeError):
            module.findings = []
    scan_fn = getattr(module, "scan", None)
    if not callable(scan_fn):
        log.debug(
            "bug_finder_scanner_no_scan_function",
            scanner=scanner_slug,
        )
        return []
    try:
        scan_fn()
    except SystemExit:
        pass
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "bug_finder_scanner_run_failed",
            scanner=scanner_slug, error=str(exc),
        )
        return []
    raw = getattr(module, "findings", None) or []
    return [r for r in raw if isinstance(r, dict)]


def _call_check(module: Any, name: str) -> list[dict]:
    """Best-effort: call ``module.<name>()`` and return a clean
    ``list[dict]``. Tuples (e.g. ``check_css_js_classes`` returning
    two lists) get flattened. Anything else returns an empty list
    rather than raising.
    """
    fn = getattr(module, name, None)
    if not callable(fn):
        return []
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "bug_finder_scanner_check_failed",
            name=name, error=str(exc),
        )
        return []
    if isinstance(result, tuple):
        # check_css_js_classes returns (missing_css, dead_css). We treat
        # both halves as findings; the per-scanner adapter labels them.
        flat: list[dict] = []
        for part in result:
            if isinstance(part, list):
                flat.extend(d for d in part if isinstance(d, dict))
        return flat
    if isinstance(result, list):
        return [d for d in result if isinstance(d, dict)]
    return []


def _run_code_quality(
    scanner_slug: str,
    filename: str,
    root: Path,
) -> list[dict]:
    """Per-scanner adapter for ``code_quality.py``.

    The scanner exposes ``check_*`` functions that return structured
    list[dict]; main() aggregates + prints. We call the check
    functions directly and stamp each finding with its category +
    a sensible severity floor (the scanner doesn't always set one).
    """
    module = _load_scanner_module(filename, root)
    if module is None:
        return []
    out: list[dict] = []

    # (check_function_name, category_slug, severity_floor)
    _CHECKS: tuple[tuple[str, str, str], ...] = (
        ("check_silent_catches",   "silent_catch_block",     "low"),
        ("check_websocket_contract", "websocket_contract_gap", "medium"),
        ("check_error_consistency",  "mixed_error_pattern",    "low"),
        ("check_console_logs",      "console_log_in_prod",    "low"),
        ("check_tech_debt",         "tech_debt_marker",       "info"),
        ("check_model_map_membership", "model_map_misuse",     "low"),
        # CSS/JS audit returns (missing_css, dead_css) — both flat.
        ("check_css_js_classes",    "css_js_mismatch",        "info"),
    )
    for fn_name, category, severity_floor in _CHECKS:
        for raw in _call_check(module, fn_name):
            # Some checks already carry severity; preserve it. Others
            # leave it blank; stamp the per-check floor.
            severity = str(raw.get("severity") or severity_floor)
            out.append({
                **raw,
                "category": raw.get("category") or category,
                "severity": severity,
            })
    return out


def _run_security_check(
    scanner_slug: str,
    filename: str,
    root: Path,
) -> list[dict]:
    """Per-scanner adapter for ``security_check.py``.

    Most check functions take a ``list[dict]`` of suppression entries
    (the security_exceptions.json `exceptions` array) as their only
    argument. We pre-load that, then dispatch each check.
    """
    module = _load_scanner_module(filename, root)
    if module is None:
        return []

    # Load the suppression entries security_check's checks expect.
    exceptions: list[dict] = []
    loader = getattr(module, "_load_security_exceptions", None)
    if callable(loader):
        try:
            data = loader()
            if isinstance(data, list):
                exceptions = data
        except Exception:  # noqa: BLE001
            exceptions = []

    out: list[dict] = []
    # check_X(exceptions) returns tuple[list[dict], list[dict]] for
    # most; list[dict] for ones without a suppression-split.
    _TUPLE_CHECKS: tuple[str, ...] = (
        "check_ssrf_surface",
        "check_template_xss",
        "check_sql_injection",
        "check_key_exposure",
    )
    for fn_name in _TUPLE_CHECKS:
        fn = getattr(module, fn_name, None)
        if not callable(fn):
            continue
        try:
            result = fn(exceptions)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "bug_finder_scanner_check_failed",
                check=fn_name, error=str(exc),
            )
            continue
        if isinstance(result, tuple):
            for half in result:
                if isinstance(half, list):
                    out.extend(d for d in half if isinstance(d, dict))
        elif isinstance(result, list):
            out.extend(d for d in result if isinstance(d, dict))

    # check_stale_exceptions(exceptions) returns list[dict]
    fn = getattr(module, "check_stale_exceptions", None)
    if callable(fn):
        try:
            stale = fn(exceptions)
            if isinstance(stale, list):
                out.extend(d for d in stale if isinstance(d, dict))
        except Exception:  # noqa: BLE001
            pass

    # check_silent_exceptions() takes no args
    fn = getattr(module, "check_silent_exceptions", None)
    if callable(fn):
        try:
            silent = fn()
            if isinstance(silent, list):
                out.extend(d for d in silent if isinstance(d, dict))
        except Exception:  # noqa: BLE001
            pass

    return out


def _run_runtime_checks(
    scanner_slug: str,
    filename: str,
    root: Path,
) -> list[dict]:
    """Per-scanner adapter for ``runtime_checks.py``.

    The check functions return ``tuple[list[str], int]`` — a list of
    formatted lines + a count. We treat each formatted line as a
    finding's ``message`` and parse out file:line when possible.
    """
    module = _load_scanner_module(filename, root)
    if module is None:
        return []
    suppressions_loader = getattr(module, "_load_suppressions", None)
    if not callable(suppressions_loader):
        return []
    try:
        all_suppressions = suppressions_loader()
    except Exception:  # noqa: BLE001
        all_suppressions = {}

    out: list[dict] = []

    _CHECKS: tuple[tuple[str, str, str], ...] = (
        ("check_empty_model",       "empty_model",          "medium"),
        ("check_silent_exceptions", "silent_exception",     "medium"),
        ("check_unhandled_fetch",   "unhandled_fetch",      "low"),
        ("check_state_outside_handlers", "state_outside_handlers", "low"),
    )
    for fn_name, category, severity_floor in _CHECKS:
        fn = getattr(module, fn_name, None)
        if not callable(fn):
            continue
        suppressions = all_suppressions.get(category, []) if isinstance(
            all_suppressions, dict,
        ) else []
        try:
            result = fn(suppressions, False)
        except Exception as exc:  # noqa: BLE001
            log.debug(
                "bug_finder_scanner_check_failed",
                check=fn_name, error=str(exc),
            )
            continue
        # Each check returns (list[str], int).
        lines = result[0] if isinstance(result, tuple) and result else []
        if not isinstance(lines, list):
            continue
        for line_text in lines:
            file, line_no = _parse_file_line(line_text)
            out.append({
                "severity": severity_floor,
                "category": category,
                "file": file,
                "line": line_no,
                "message": str(line_text).strip(),
                "fix": "",
            })
    return out


_FILE_LINE_RE = __import__("re").compile(
    r"([a-zA-Z0-9_./\\-]+\.\w+):(\d+)",
)


def _parse_file_line(text: str) -> tuple[str, int]:
    """Extract the first ``path/foo.py:LN`` reference from a line of
    scanner output. Returns ``("", 0)`` when none found."""
    m = _FILE_LINE_RE.search(str(text))
    if not m:
        return "", 0
    try:
        return m.group(1), int(m.group(2))
    except (TypeError, ValueError):
        return m.group(1), 0


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------


_RESULT_CACHE: dict[str, tuple[float, list[ScannerFinding]]] = {}
_RESULT_CACHE_LOCK = threading.Lock()
_DEFAULT_CACHE_SECONDS = 300.0


def clear_caches() -> None:
    with _MODULE_CACHE_LOCK:
        _MODULE_CACHE.clear()
    with _RESULT_CACHE_LOCK:
        _RESULT_CACHE.clear()


# ---------------------------------------------------------------------------
# Scanner registry — slug → (filename, runner)
# ---------------------------------------------------------------------------


# Native runners pass directly. Stub runners return empty until the
# per-scanner stdout parser is implemented (TODO).


def _stub_runner(scanner_slug: str, filename: str, root: Path) -> list[dict]:
    log.debug(
        "bug_finder_scanner_stub_invoked",
        scanner=scanner_slug,
        note="per-scanner check-function runner not implemented yet",
    )
    return []


# (slug, filename, runner) — runner returns raw dicts the framework
# normalizes. Add new scanners by appending here.
_SCANNER_REGISTRY: tuple[tuple[str, str, Callable], ...] = (
    ("red_team_scan",   "red_team_scan.py",   _run_native_scanner),
    ("code_quality",    "code_quality.py",    _run_code_quality),
    ("security_check",  "security_check.py",  _run_security_check),
    ("runtime_checks",  "runtime_checks.py",  _run_runtime_checks),
    # The remaining two need per-scanner adapters. Leaving as stubs so
    # the agent tool layer can enumerate them honestly.
    ("dead_code",       "dead_code.py",       _stub_runner),
    ("db_safety",       "db_safety.py",       _stub_runner),
)


_REGISTRY_LOOKUP = {slug: (fname, runner) for slug, fname, runner in _SCANNER_REGISTRY}


def available_scanners() -> list[str]:
    """All slugs the registry knows about."""
    return [slug for slug, _, _ in _SCANNER_REGISTRY]


def is_native_scanner(slug: str) -> bool:
    """``True`` when the scanner's runner is the native import-and-call
    path. Stub-only scanners return ``False`` until subprocess parsers
    are implemented."""
    entry = _REGISTRY_LOOKUP.get(slug)
    if entry is None:
        return False
    return entry[1] is _run_native_scanner


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_scanner(
    slug: str,
    *,
    root: Path,
    apply_suppressions: bool = True,
    cache_seconds: float = _DEFAULT_CACHE_SECONDS,
) -> list[ScannerFinding]:
    """Run one scanner, return structured + suppression-filtered findings."""
    entry = _REGISTRY_LOOKUP.get(slug)
    if entry is None:
        log.warning("bug_finder_scanner_unknown", slug=slug)
        return []
    filename, runner = entry

    cache_key = f"{root.resolve()}|{slug}|{apply_suppressions}"
    now = time.monotonic()
    if cache_seconds > 0:
        with _RESULT_CACHE_LOCK:
            entry_cache = _RESULT_CACHE.get(cache_key)
            if entry_cache and (now - entry_cache[0]) < cache_seconds:
                return entry_cache[1]

    raw = runner(slug, filename, root)
    normalized = [_normalize(slug, r) for r in raw]

    if apply_suppressions:
        filtered: list[ScannerFinding] = []
        for f in normalized:
            if refs.is_finding_suppressed(
                root, file=f.file, pattern=f.category,
            ):
                continue
            filtered.append(f)
        normalized = filtered

    if cache_seconds > 0:
        with _RESULT_CACHE_LOCK:
            _RESULT_CACHE[cache_key] = (now, normalized)
    return normalized


def run_all_native_scanners(
    *,
    root: Path,
    apply_suppressions: bool = True,
    cache_seconds: float = _DEFAULT_CACHE_SECONDS,
) -> dict[str, list[ScannerFinding]]:
    """Run every scanner that has a native runner. Stubs are skipped."""
    out: dict[str, list[ScannerFinding]] = {}
    for slug, _, runner in _SCANNER_REGISTRY:
        if runner is _run_native_scanner:
            out[slug] = run_scanner(
                slug, root=root,
                apply_suppressions=apply_suppressions,
                cache_seconds=cache_seconds,
            )
    return out


# ---------------------------------------------------------------------------
# Convenience functions — one per native scanner
# ---------------------------------------------------------------------------


def red_team_scan(root: Path) -> list[ScannerFinding]:
    """Adversarial scanner: data-isolation gaps, auth bypass, token
    exposure, IDOR, AI context leaks."""
    return run_scanner("red_team_scan", root=root)


def code_quality_check(root: Path) -> list[ScannerFinding]:
    """Silent catch blocks, console.log in prod, mixed error response
    formats, websocket contract gaps, tech-debt markers, CSS/JS class
    mismatches."""
    return run_scanner("code_quality", root=root)


def security_check(root: Path) -> list[ScannerFinding]:
    """SSRF / injection / API-key / size-limit scanner with
    suppression awareness."""
    return run_scanner("security_check", root=root)


def runtime_checks(root: Path) -> list[ScannerFinding]:
    """Runtime-only bug patterns: empty model strings, silent
    exception swallowing, unhandled fetch failures, state outside
    handlers."""
    return run_scanner("runtime_checks", root=root)
