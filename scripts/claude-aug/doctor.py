#!/usr/bin/env python3
"""augmentum doctor — deployment health for Augmentum + harness integrations.

The audit (.claude/skills/augmentum-dev/scripts/audit.py) answers "is the
code right"; the doctor answers "is the world right, right now": containers,
live proxy, ATP tool round-trips, and the client-side harness state on
whatever machine it runs on. Stdlib only — runs anywhere the installer
puts it, no repo required.

Usage:
    python doctor.py                 # human-readable table
    python doctor.py --format=json   # machine-readable (audit-compatible)
    python doctor.py --fix           # apply provably-safe remedies
    python doctor.py --skip-live     # config/static checks only (no network)

Exit codes: 0 all pass, 1 warnings only, 2 one or more failures.

Stdout also carries stable `live.<metric>=<count>` lines (0 = clean) so
audit.py can bundle this as a scanner with a trivial parser.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Expected state manifest — intent lives HERE, explicitly. When integration
# work adds capability (new ATP tools, new containers), grow this in the
# same commit, like an audit baseline bump.
# ---------------------------------------------------------------------------

EXPECTED_ATP_TOOLS = {
    "calculator", "consistency_check", "context_peek", "document_parse",
    "hash_tool", "image_search", "json_tool", "math_verify",
    "media_recommendations", "memory_recall", "python_exec", "research",
    "search_files", "text_analysis", "unit_converter", "web_fetch",
    "web_search", "wikipedia", "youtube",
    # browser sidecar exposure (integration item 1) + one-call sign-in
    "browser_action", "browser_evaluate", "browser_navigate",
    "browser_screenshot", "browser_wait", "browser_ensure_auth",
    # named per-user macros over ATP tools (recipe layer)
    "atp_recipe",
    # self-minted soft procedural memory (Hermes/AWM-style workflow layer)
    "workflow",
    # staged, human-gated memory write (integration item 4)
    "memory_store",
    # research delegation (item 5) + artifacts (item 6)
    "flow_deep_research", "task_status",
    "create_chart", "create_document", "create_spreadsheet",
    # agent bridge (notify/approve/review loop) + per-user Docker sandbox
    "agent_checkin", "ask_user", "check_reply", "sandbox_shell",
    # pack_search is live only when knowledge packs are installed —
    # deliberately NOT in the manifest (surfaces via "new unlisted").
}

# Container name regexes (matched against `docker ps -a` names). Only the
# core container is required; the rest degrade to WARN when absent so
# minimal installs don't fail.
CONTAINER_CORE = re.compile(r"^augmentum[-_].*augmentum")
CONTAINERS_OPTIONAL = {
    "searxng (web_search backend)": re.compile(r"searxng"),
    "browser sidecar": re.compile(r"browser"),
}

DEFAULT_BASE_URL = "http://localhost:6100"
AUG_DIR = Path.home() / ".augmentum"

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


# ---------------------------------------------------------------------------
# Config discovery — env first, then claude.env, then defaults. Never
# hardcode a host, key, or username.
# ---------------------------------------------------------------------------

def discover_config() -> dict:
    cfg = {
        "base_url": os.environ.get("ANTHROPIC_BASE_URL", ""),
        "api_key": os.environ.get("AUGMENTUM_API_KEY")
                   or os.environ.get("ANTHROPIC_API_KEY", ""),
    }
    env_file = AUG_DIR / "claude.env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^export ([A-Z_]+)=(.*)$", line.strip())
            if not m:
                continue
            name, value = m.group(1), m.group(2).strip().strip("\"'")
            if name == "ANTHROPIC_BASE_URL" and not cfg["base_url"]:
                cfg["base_url"] = value
            if name == "AUGMENTUM_API_KEY" and not cfg["api_key"]:
                cfg["api_key"] = value
    cfg["base_url"] = (cfg["base_url"] or DEFAULT_BASE_URL).rstrip("/")
    return cfg


def api(cfg: dict, path: str, data: dict | None = None, timeout: int = 8):
    """Return (status_code, parsed_json_or_None). Raises URLError on no-connect."""
    req = urllib.request.Request(
        cfg["base_url"] + path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={
            "x-api-key": cfg["api_key"],
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        return exc.code, None


def run(cmd: list[str], timeout: int = 15) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, "(timed out)"


# ---------------------------------------------------------------------------
# Checks. Each returns (status, evidence, remedy). Registered with a layer
# and a metric name; metric counts 1 on FAIL, 0 otherwise (WARNs get their
# own *_warn metric) so audit.py can weight them independently.
# ---------------------------------------------------------------------------

CHECKS = []  # (layer, name, metric, fn)


def check(layer: str, name: str, metric: str):
    def deco(fn):
        CHECKS.append((layer, name, metric, fn))
        return fn
    return deco


# ---- infra ----------------------------------------------------------------

def _docker_ps() -> list[tuple[str, str]] | None:
    code, out = run(["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Status}}"])
    if code != 0:
        return None
    return [tuple(l.split("\t", 1)) for l in out.splitlines() if "\t" in l]


@check("infra", "core container running", "container_core_down")
def core_container(ctx):
    ps = ctx["docker"]
    if ps is None:
        return SKIP, "docker not available on this machine", ""
    for name, status in ps:
        if CONTAINER_CORE.match(name):
            ctx["core_container"] = name
            if status.startswith("Up"):
                return PASS, f"{name}: {status}", ""
            return FAIL, f"{name}: {status}", f"docker start {name}"
    return FAIL, "no Augmentum core container found", "docker compose up -d"


@check("infra", "no containers restart-looping", "containers_restarting")
def restart_loops(ctx):
    ps = ctx["docker"]
    if ps is None:
        return SKIP, "docker not available", ""
    looping = [n for n, s in ps if "Restarting" in s]
    if looping:
        return WARN, f"restart-looping: {', '.join(looping)}", \
               f"docker logs {looping[0]} --tail 50"
    return PASS, "none restarting", ""


@check("infra", "optional service containers", "containers_optional_down")
def optional_containers(ctx):
    ps = ctx["docker"]
    if ps is None:
        return SKIP, "docker not available", ""
    down = []
    for label, pattern in CONTAINERS_OPTIONAL.items():
        matches = [(n, s) for n, s in ps if pattern.search(n)]
        if matches and not any(s.startswith("Up") for _, s in matches):
            down.append(label)
    if down:
        return WARN, f"down: {', '.join(down)} (dependent tools will be health-gated off)", \
               "docker compose up -d"
    return PASS, "all present optional services up", ""


# ---- proxy ----------------------------------------------------------------

@check("proxy", "proxy reachable + key valid", "proxy_unreachable")
def proxy_reachable(ctx):
    import time
    t0 = time.monotonic()
    try:
        status, _ = api(ctx["cfg"], "/v1/models", timeout=25)
    except (urllib.error.URLError, OSError) as exc:
        return FAIL, f"cannot connect to {ctx['cfg']['base_url']}: {exc}", \
               "start Augmentum, or fix ANTHROPIC_BASE_URL in ~/.augmentum/claude.env"
    elapsed = time.monotonic() - t0
    if status in (401, 403):
        return FAIL, f"HTTP {status} — key rejected", \
               "regenerate the API key in Augmentum UI and update ~/.augmentum/claude.env"
    if status != 200:
        return FAIL, f"HTTP {status} from /v1/models", "check proxy logs"
    ctx["proxy_ok"] = True
    if elapsed > 5:
        return WARN, f"{ctx['cfg']['base_url']} OK but slow ({elapsed:.1f}s) - warming up or overloaded", \
               "re-run in a minute; if persistent, check container resources"
    return PASS, f"{ctx['cfg']['base_url']} OK ({elapsed:.1f}s)", ""


@check("proxy", "claude-* tier aliases resolve", "alias_resolution_broken")
def alias_resolution(ctx):
    # Self-contained: same-layer checks run concurrently, so don't read
    # ctx state written by proxy_reachable.
    body = {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]}
    try:
        status, _ = api(ctx["cfg"], "/v1/messages/count_tokens", body, timeout=25)
    except (urllib.error.URLError, OSError):
        return SKIP, "proxy not reachable (reported separately)", ""
    if status == 200:
        return PASS, "claude-haiku-* resolves via tier aliases", ""
    return FAIL, f"HTTP {status} for model claude-haiku-4-5", \
           "set AUGMENTUM_ANTHROPIC_ALIAS_HAIKU/SONNET/OPUS/DEFAULT in Augmentum's .env (subagents silently fail without them)"


# ---- atp ------------------------------------------------------------------

@check("atp", "ATP tool list vs manifest", "atp_tools_missing")
def atp_list(ctx):
    if not ctx.get("proxy_ok"):
        return SKIP, "proxy not reachable", ""
    try:
        status, data = api(ctx["cfg"], "/v1/tools/list")
    except (urllib.error.URLError, OSError) as exc:
        return FAIL, f"/v1/tools/list unreachable: {exc}", "check ATP routes in proxy"
    if status != 200 or data is None:
        return FAIL, f"HTTP {status} from /v1/tools/list", "check proxy logs"
    live = {t["name"] for t in data.get("tools", [])}
    ctx["atp_live"] = live
    missing = EXPECTED_ATP_TOOLS - live
    extra = live - EXPECTED_ATP_TOOLS
    if missing:
        return FAIL, f"missing (health-gated off?): {', '.join(sorted(missing))}", \
               "backing service down, or update EXPECTED_ATP_TOOLS if intentionally removed"
    note = f"; new unlisted: {', '.join(sorted(extra))} (add to manifest)" if extra else ""
    return PASS, f"{len(live)} tools live{note}", ""


@check("atp", "ATP execute round-trip", "atp_roundtrip_broken")
def atp_roundtrip(ctx):
    # Self-contained (no ctx reads from same-layer atp_list): calculator is
    # in the expected manifest, so just call it.
    if not ctx.get("proxy_ok"):
        return SKIP, "proxy not reachable", ""
    try:
        status, data = api(ctx["cfg"], "/v1/tools/call",
                           {"tool": "calculator", "arguments": {"expression": "6*7"}},
                           timeout=25)
    except (urllib.error.URLError, OSError) as exc:
        return FAIL, f"call failed: {exc}", "check proxy logs"
    if status == 200 and data and data.get("ok") and "42" in str(data.get("output", "")):
        return PASS, "calculator(6*7) -> 42", ""
    return FAIL, f"HTTP {status}, body: {str(data)[:120]}", "tool registry execute path broken"


# ---- harness (client-side; skipped entirely if claude-aug not installed) --

def _harness_installed() -> bool:
    return AUG_DIR.is_dir() and (AUG_DIR / "claude.env").is_file()


@check("harness", "claude.env present + keyed", "claude_env_broken")
def claude_env(ctx):
    if not _harness_installed():
        return SKIP, "claude-aug not installed on this machine", ""
    if not ctx["cfg"]["api_key"]:
        return FAIL, "no AUGMENTUM_API_KEY found", "run scripts/claude-aug/install.{ps1,sh}"
    return PASS, "claude.env parsed, key present", ""


@check("harness", "config JSONs parse", "harness_config_broken")
def config_json(ctx):
    if not _harness_installed():
        return SKIP, "claude-aug not installed", ""
    bad = []
    for f in (AUG_DIR / "claude-config" / ".claude.json",
              AUG_DIR / "claude-config" / "settings.json"):
        if not f.is_file():
            bad.append(f"{f.name} missing")
            continue
        try:
            json.loads(f.read_text(encoding="utf-8-sig"))
        except ValueError as exc:
            bad.append(f"{f.name}: {exc}")
    if bad:
        return FAIL, "; ".join(bad), "re-run the installer"
    return PASS, ".claude.json + settings.json valid", ""


@check("harness", "PowerShell files have BOM", "ps1_bom_missing")
def ps1_bom(ctx):
    if not _harness_installed():
        return SKIP, "claude-aug not installed", ""
    missing = [p for p in AUG_DIR.rglob("*.ps1")
               if not p.read_bytes().startswith(b"\xef\xbb\xbf")]
    if missing:
        names = ", ".join(p.name for p in missing)
        return FAIL, f"BOM-less (PS 5.1 misparses non-ASCII): {names}", \
               "doctor.py --fix re-saves them with BOM"
    return PASS, "all .ps1 files BOM'd", ""


@check("harness", "PowerShell files parse", "ps1_parse_errors")
def ps1_parse(ctx):
    if not _harness_installed() or os.name != "nt":
        return SKIP, "not applicable on this machine", ""
    errs = []
    for p in AUG_DIR.rglob("*.ps1"):
        code, out = run([
            "powershell", "-NoProfile", "-Command",
            "$e=$null;[void][System.Management.Automation.Language.Parser]::ParseFile("
            f"'{p}',[ref]$null,[ref]$e);$e.Count",
        ])
        if code != 0 or (out.strip() and out.strip().splitlines()[-1] != "0"):
            errs.append(p.name)
    if errs:
        return FAIL, f"parse errors in: {', '.join(errs)}", \
               "powershell -Command '. <file>' to see the errors"
    return PASS, "all .ps1 files parse clean (real PS 5.1 parser)", ""


@check("harness", "wrapper env-leak guard intact", "wrapper_leak_guard_missing")
def wrapper_guard(ctx):
    if not _harness_installed():
        return SKIP, "claude-aug not installed", ""
    ps1 = AUG_DIR / "claude-aug.ps1"
    if not ps1.is_file():
        return SKIP, "claude-aug.ps1 not present", ""
    text = ps1.read_text(encoding="utf-8-sig", errors="replace")
    if "$scopedVars" in text and "finally" in text:
        return PASS, "snapshot/restore pattern present", ""
    return FAIL, "env snapshot/restore missing — claude-aug will poison plain `claude` in the same shell", \
           "restore the scopedVars + try/finally pattern (see repo copy)"


@check("harness", "MCP bridge handshake", "bridge_broken")
def bridge_handshake(ctx):
    bridge = AUG_DIR / "atp-mcp-bridge.py"
    if not bridge.is_file():
        return SKIP, "bridge not installed", ""
    if not ctx.get("proxy_ok"):
        return SKIP, "proxy not reachable", ""
    payload = (
        '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n'
        '{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n'
    )
    try:
        proc = subprocess.run(
            [sys.executable, str(bridge)], input=payload, capture_output=True,
            text=True, timeout=30, encoding="utf-8",
        )
    except subprocess.TimeoutExpired:
        return FAIL, "bridge hung >30s", "run it manually with the same two JSON lines"
    count = 0
    for line in proc.stdout.splitlines():
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("id") == 2 and "result" in msg:
            count = len(msg["result"].get("tools", []))
    if count > 0:
        return PASS, f"initialize + tools/list OK ({count} tools)", ""
    return FAIL, f"no tool list returned; stderr: {proc.stderr[:120]}", \
           "check bridge env fallback + proxy reachability"


# ---- hygiene (only when run inside a git repo) ----------------------------

@check("hygiene", "no API keys in tracked files", "keys_in_git")
def keys_tracked(ctx):
    root = Path(__file__).resolve()
    code, _ = run(["git", "-C", str(root.parent), "rev-parse", "--show-toplevel"])
    if code != 0:
        return SKIP, "not in a git repo", ""
    code, out = run(["git", "-C", str(root.parent), "grep", "-lI", "sk-aug-"])
    hits = [l for l in out.splitlines()
            if l.strip() and "doctor.py" not in l and "template" not in l]
    if code == 0 and hits:
        return FAIL, f"sk-aug-* key present in tracked: {', '.join(hits[:5])}", \
               "remove the key, rotate it in the Augmentum UI"
    return PASS, "no live keys in tracked files", ""


@check("hygiene", "uncommitted change volume", "uncommitted_pileup")
def uncommitted(ctx):
    root = Path(__file__).resolve()
    code, out = run(["git", "-C", str(root.parent), "status", "--porcelain"])
    if code != 0:
        return SKIP, "not in a git repo", ""
    n = len([l for l in out.splitlines() if l.strip()])
    if n > 50:
        return WARN, f"{n} uncommitted changes", "commit or stash before proxy-touching work"
    return PASS, f"{n} uncommitted changes", ""


# ---------------------------------------------------------------------------
# Fixers (provably safe only)
# ---------------------------------------------------------------------------

def apply_fixes(results, ctx) -> list[str]:
    applied = []
    by_name = {name: status for _, name, _, status, *_ in results}
    if by_name.get("core container running") == FAIL and ctx.get("core_container"):
        code, _ = run(["docker", "start", ctx["core_container"]], timeout=60)
        applied.append(f"docker start {ctx['core_container']}: {'ok' if code == 0 else 'failed'}")
    if by_name.get("PowerShell files have BOM") == FAIL:
        for p in AUG_DIR.rglob("*.ps1"):
            raw = p.read_bytes()
            if not raw.startswith(b"\xef\xbb\xbf"):
                p.write_bytes(b"\xef\xbb\xbf" + raw)
                applied.append(f"BOM added: {p.name}")
    return applied


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Augmentum deployment doctor")
    ap.add_argument("--format", choices=["text", "json"], default="text")
    ap.add_argument("--fix", action="store_true", help="apply provably-safe remedies")
    ap.add_argument("--skip-live", action="store_true", help="skip network checks")
    args = ap.parse_args()

    ctx: dict = {"cfg": discover_config()}
    ctx["docker"] = None if args.skip_live else _docker_ps()
    if args.skip_live:
        ctx["proxy_ok"] = False

    # Layer order matters for ctx priming (proxy before atp/bridge), but
    # checks within a layer are independent — run each layer's checks
    # concurrently.
    results = []  # (layer, name, metric, status, evidence, remedy)
    layers = []
    for layer, *_ in CHECKS:
        if layer not in layers:
            layers.append(layer)
    for layer in layers:
        layer_checks = [c for c in CHECKS if c[0] == layer]
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futs = {pool.submit(fn, ctx): (layer, name, metric)
                    for layer, name, metric, fn in layer_checks}
            for fut in concurrent.futures.as_completed(futs):
                layer_, name, metric = futs[fut]
                try:
                    status, evidence, remedy = fut.result()
                except Exception as exc:  # a broken check must not kill the doctor
                    status, evidence, remedy = FAIL, f"check crashed: {exc}", "report this"
                results.append((layer_, name, metric, status, evidence, remedy))
    order = {name: i for i, (_, name, _, _) in enumerate(CHECKS)}
    results.sort(key=lambda r: (layers.index(r[0]), order[r[1]]))

    fixes = apply_fixes(results, ctx) if args.fix else []

    metrics = {m: (1 if s == FAIL else 0) for _, _, m, s, _, _ in results}
    n_fail = sum(1 for r in results if r[3] == FAIL)
    n_warn = sum(1 for r in results if r[3] == WARN)

    if args.format == "json":
        print(json.dumps({
            "ok": n_fail == 0,
            "failures": n_fail, "warnings": n_warn,
            "metrics": {"live": metrics},
            "checks": [
                {"layer": l, "name": n, "status": s, "evidence": e, "remedy": r}
                for l, n, _, s, e, r in results
            ],
            "fixes_applied": fixes,
        }, indent=2))
    else:
        icons = {PASS: "  ok ", WARN: " WARN", FAIL: " FAIL", SKIP: " skip"}
        current = None
        for layer, name, _, status, evidence, remedy in results:
            if layer != current:
                print(f"\n[{layer}]")
                current = layer
            print(f"  {icons[status]}  {name} - {evidence}")
            if remedy and status in (WARN, FAIL):
                print(f"          fix: {remedy}")
        print()
        for f in fixes:
            print(f"  fixed: {f}")
        # stable machine-greppable lines for the audit.py adapter
        for metric, v in sorted(metrics.items()):
            print(f"live.{metric}={v}")
        verdict = "HEALTHY" if n_fail == 0 else "UNHEALTHY"
        print(f"\ndoctor: {verdict} ({n_fail} fail, {n_warn} warn)")

    return 2 if n_fail else (1 if n_warn else 0)


if __name__ == "__main__":
    sys.exit(main())
