#!/usr/bin/env python3
"""Generate an interactive HTML health dashboard for Augmentum.

Scans the full project and produces a single-file HTML report showing:
  - Settings wiring completeness (4-layer cross-reference)
  - Route registration status
  - Migration timeline
  - Subsystem dependency map
  - Template literal safety audit

Opens in the default browser.
"""

from __future__ import annotations

import json
import re
import sys
import webbrowser
from pathlib import Path
from datetime import datetime

import _common  # noqa: F401 — import side-effect: UTF-8-safe stdout/stderr


def _find_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "augmentum" / "proxy").is_dir() and (parent / "ui").is_dir():
            return parent
    print("ERROR: Cannot find Augmentum project root.", file=sys.stderr)
    sys.exit(2)


ROOT = _find_root()


# ── Scanners (reuse logic from validate_wiring.py) ──────────────────────

def _extract_dict_keys(text: str, varname: str) -> set[str]:
    keys: set[str] = set()
    pattern = re.compile(rf"{varname}\s*[:\=].*?\{{", re.DOTALL)
    m = pattern.search(text)
    if not m:
        return keys
    start = m.end()
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{": depth += 1
        elif text[i] == "}": depth -= 1
        i += 1
    block = text[start:i - 1]
    for km in re.finditer(r'^\s*"(\w+)"\s*:', block, re.MULTILINE):
        keys.add(km.group(1))
    return keys


def scan_config_fields() -> set[str]:
    text = (ROOT / "augmentum" / "config.py").read_text(encoding="utf-8", errors="replace")
    fields: set[str] = set()
    in_class = False
    for line in text.splitlines():
        if "class Settings" in line:
            in_class = True
            continue
        if in_class:
            if line and not line[0].isspace() and not line.startswith("#"):
                break
            m = re.match(r"\s+(\w+)\s*:\s*\w+", line)
            if m:
                fields.add(m.group(1))
    return fields


def scan_config_routes() -> tuple[set[str], set[str]]:
    text = (ROOT / "augmentum" / "proxy" / "config_routes.py").read_text(encoding="utf-8", errors="replace")
    return _extract_dict_keys(text, "_TOOL_SETTINGS"), _extract_dict_keys(text, "_STRING_SETTINGS")


def scan_restore_map() -> set[str]:
    text = (ROOT / "augmentum" / "proxy" / "server.py").read_text(encoding="utf-8", errors="replace")
    return _extract_dict_keys(text, "_SETTINGS_RESTORE_MAP")


def scan_js_sync() -> set[str]:
    text = (ROOT / "ui" / "scripts" / "settings.js").read_text(encoding="utf-8", errors="replace")
    sync_keys: set[str] = set()
    sync_match = re.search(r"function\s+syncToolSettingsToBackend", text)
    if sync_match:
        start = text.find("{", sync_match.end())
        if start >= 0:
            depth, i = 1, start + 1
            while i < len(text) and depth > 0:
                if text[i] == "{": depth += 1
                elif text[i] == "}": depth -= 1
                i += 1
            block = text[start:i]
            for km in re.finditer(r"(\w+)\s*:", block):
                key = km.group(1)
                if "_" in key and key not in ("Content", "method", "headers"):
                    sync_keys.add(key)
    return sync_keys


def scan_routes() -> list[dict]:
    proxy_dir = ROOT / "augmentum" / "proxy"
    server_text = (proxy_dir / "server.py").read_text(encoding="utf-8", errors="replace")
    results = []
    for rf in sorted(proxy_dir.glob("*_routes.py")):
        name = rf.stem
        if name == "notification_routes":
            continue
        imported = f"from augmentum.proxy.{name}" in server_text
        alias = name.replace("_routes", "_router")
        registered = f"include_router({alias}" in server_text or name in server_text
        # Extract prefix
        rt = rf.read_text(encoding="utf-8", errors="replace")
        pm = re.search(r'prefix\s*=\s*"([^"]+)"', rt)
        prefix = pm.group(1) if pm else "?"
        results.append({
            "name": name,
            "prefix": prefix,
            "imported": imported,
            "registered": registered,
            "ok": imported and registered,
        })
    return results


def scan_migrations() -> list[dict]:
    mig_dir = ROOT / "augmentum" / "state" / "migrations"
    results = []
    for f in sorted(mig_dir.glob("*.sql")):
        m = re.match(r"(\d+)_(.*?)\.sql", f.name)
        if m:
            results.append({
                "number": int(m.group(1)),
                "name": m.group(2).replace("_", " "),
                "file": f.name,
            })
    return results


def scan_subsystems() -> list[dict]:
    """Count files per subsystem for the dependency map."""
    subsystems = [
        ("Proxy Routes", ROOT / "augmentum" / "proxy", "*_routes.py"),
        ("Mode Handlers", ROOT / "augmentum" / "modes", "**/*.py"),
        ("Voice Pipeline", ROOT / "augmentum" / "voice", "*.py"),
        ("Memory System", ROOT / "augmentum" / "memory", "**/*.py"),
        ("Image Pipeline", ROOT / "augmentum" / "image", "**/*.py"),
        ("Tools", ROOT / "augmentum" / "tools", "**/*.py"),
        ("State / Migrations", ROOT / "augmentum" / "state", "**/*.sql"),
        ("Frontend Scripts", ROOT / "ui" / "scripts", "**/*.js"),
        ("Frontend Styles", ROOT / "ui" / "styles", "*.css"),
        ("Tests", ROOT / "tests", "*.py"),
    ]
    results = []
    for label, path, pattern in subsystems:
        if path.is_dir():
            count = len(list(path.glob(pattern)))
            results.append({"name": label, "count": count})
        else:
            results.append({"name": label, "count": 0})
    return results


# ── Build settings cross-reference ──────────────────────────────────────

def build_settings_matrix() -> list[dict]:
    config_fields = scan_config_fields()
    tool_settings, string_settings = scan_config_routes()
    restore_map = scan_restore_map()
    js_sync = scan_js_sync()

    all_keys = sorted(config_fields | tool_settings | string_settings | restore_map | js_sync)
    rows = []
    for key in all_keys:
        in_config = key in config_fields
        in_routes = key in tool_settings or key in string_settings
        in_restore = key in restore_map
        in_js = key in js_sync
        layers = sum([in_config, in_routes, in_restore, in_js])
        # Only show settings that are in at least one API layer (skip server-only fields)
        if in_routes or in_restore or in_js:
            rows.append({
                "key": key,
                "config": in_config,
                "routes": in_routes,
                "restore": in_restore,
                "js": in_js,
                "layers": layers,
                "status": "ok" if layers >= 3 else ("warn" if layers == 2 else "error"),
            })
    return rows


# ── HTML generation ─────────────────────────────────────────────────────

def generate_html() -> str:
    settings = build_settings_matrix()
    routes = scan_routes()
    migrations = scan_migrations()
    subsystems = scan_subsystems()

    ok_count = sum(1 for s in settings if s["status"] == "ok")
    warn_count = sum(1 for s in settings if s["status"] == "warn")
    err_count = sum(1 for s in settings if s["status"] == "error")
    route_ok = sum(1 for r in routes if r["ok"])
    route_total = len(routes)

    data = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "routes": routes,
        "migrations": migrations,
        "subsystems": subsystems,
        "stats": {
            "settings_ok": ok_count,
            "settings_warn": warn_count,
            "settings_err": err_count,
            "settings_total": len(settings),
            "routes_ok": route_ok,
            "routes_total": route_total,
            "migrations": len(migrations),
        },
    }

    return f'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"><title>Augmentum Health Report</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font:14px/1.6 'Segoe UI',system-ui,sans-serif; background:#0f0f1a; color:#e0e0e8; }}
  .header {{ background:linear-gradient(135deg,#161625 0%,#1a1a35 100%); padding:32px 40px; border-bottom:1px solid #2d2d45; }}
  .header h1 {{ font-size:24px; font-weight:700; letter-spacing:-0.02em; }}
  .header h1 span {{ color:#6c8aff; }}
  .header .meta {{ color:#6b6b80; font-size:12px; margin-top:4px; }}
  .stats {{ display:flex; gap:16px; padding:24px 40px; flex-wrap:wrap; }}
  .stat-card {{ background:#1c1c2e; border:1px solid #2d2d45; border-radius:12px; padding:20px 24px; min-width:180px; flex:1; }}
  .stat-card .label {{ font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:#6b6b80; margin-bottom:8px; }}
  .stat-card .value {{ font-size:28px; font-weight:700; }}
  .stat-card .value.green {{ color:#4ade80; }}
  .stat-card .value.yellow {{ color:#fbbf24; }}
  .stat-card .value.red {{ color:#f87171; }}
  .stat-card .value.blue {{ color:#6c8aff; }}
  .tabs {{ display:flex; gap:0; padding:0 40px; border-bottom:1px solid #2d2d45; }}
  .tab {{ padding:12px 24px; cursor:pointer; color:#6b6b80; font-weight:500; border-bottom:2px solid transparent; transition:all 0.15s; }}
  .tab:hover {{ color:#a1a1b5; }}
  .tab.active {{ color:#6c8aff; border-bottom-color:#6c8aff; }}
  .panel {{ display:none; padding:24px 40px; }}
  .panel.active {{ display:block; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ text-align:left; padding:10px 12px; font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:#6b6b80; border-bottom:1px solid #2d2d45; }}
  td {{ padding:10px 12px; border-bottom:1px solid #1c1c2e; font-family:'SF Mono','Cascadia Code',monospace; font-size:13px; }}
  tr:hover td {{ background:#1a1a2c; }}
  .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; }}
  .dot.green {{ background:#4ade80; }}
  .dot.yellow {{ background:#fbbf24; }}
  .dot.red {{ background:#f87171; }}
  .dot.gray {{ background:#3a3a50; }}
  .badge {{ display:inline-block; padding:2px 8px; border-radius:9999px; font-size:11px; font-weight:500; }}
  .badge.ok {{ background:rgba(74,222,128,0.12); color:#4ade80; }}
  .badge.warn {{ background:rgba(251,191,36,0.12); color:#fbbf24; }}
  .badge.error {{ background:rgba(248,113,113,0.12); color:#f87171; }}
  .bar-row {{ display:flex; align-items:center; margin:6px 0; }}
  .bar-label {{ width:140px; font-size:12px; color:#a1a1b5; }}
  .bar {{ height:20px; border-radius:4px; background:#6c8aff; min-width:2px; transition:width 0.3s; }}
  .bar-count {{ margin-left:8px; font-size:12px; color:#6b6b80; }}
  .mig-timeline {{ display:flex; flex-wrap:wrap; gap:4px; padding:8px 0; }}
  .mig-chip {{ padding:3px 8px; border-radius:4px; background:#1c1c2e; border:1px solid #2d2d45; font-size:11px; font-family:monospace; color:#a1a1b5; cursor:default; }}
  .mig-chip:hover {{ background:#242438; color:#e0e0e8; }}
  .filter {{ margin-bottom:16px; }}
  .filter select {{ background:#1c1c2e; color:#e0e0e8; border:1px solid #2d2d45; border-radius:6px; padding:6px 12px; font-size:13px; }}
</style>
</head><body>

<div class="header">
  <h1><span>&#9670;</span> Augmentum Health Report</h1>
  <div class="meta">Generated {data["generated"]} &middot; {data["stats"]["settings_total"]} settings &middot; {data["stats"]["routes_total"]} routes &middot; {data["stats"]["migrations"]} migrations</div>
</div>

<div class="stats">
  <div class="stat-card"><div class="label">Settings Wired</div><div class="value green">{ok_count}/{len(settings)}</div></div>
  <div class="stat-card"><div class="label">Warnings</div><div class="value {"yellow" if warn_count else "green"}">{warn_count}</div></div>
  <div class="stat-card"><div class="label">Errors</div><div class="value {"red" if err_count else "green"}">{err_count}</div></div>
  <div class="stat-card"><div class="label">Routes OK</div><div class="value {"green" if route_ok == route_total else "yellow"}">{route_ok}/{route_total}</div></div>
  <div class="stat-card"><div class="label">Migrations</div><div class="value blue">{len(migrations)}</div></div>
</div>

<div class="tabs">
  <div class="tab active" data-panel="settings">Settings Wiring</div>
  <div class="tab" data-panel="routes">Routes</div>
  <div class="tab" data-panel="migrations">Migrations</div>
  <div class="tab" data-panel="subsystems">Subsystems</div>
</div>

<div class="panel active" id="panel-settings">
  <div class="filter">
    <select id="settings-filter">
      <option value="all">All settings</option>
      <option value="error">Errors only</option>
      <option value="warn">Warnings only</option>
      <option value="ok">OK only</option>
    </select>
  </div>
  <table>
    <thead><tr><th>Setting</th><th>config.py</th><th>config_routes</th><th>server.py</th><th>settings.js</th><th>Status</th></tr></thead>
    <tbody id="settings-body"></tbody>
  </table>
</div>

<div class="panel" id="panel-routes">
  <table>
    <thead><tr><th>Route File</th><th>Prefix</th><th>Imported</th><th>Registered</th><th>Status</th></tr></thead>
    <tbody id="routes-body"></tbody>
  </table>
</div>

<div class="panel" id="panel-migrations">
  <p style="color:#6b6b80;margin-bottom:16px;">{len(migrations)} migrations — latest: {migrations[-1]["file"] if migrations else "none"}</p>
  <div class="mig-timeline" id="mig-timeline"></div>
</div>

<div class="panel" id="panel-subsystems">
  <div id="subsystems-bars"></div>
</div>

<script>
const DATA = {json.dumps(data)};

// Tabs
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {{
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  document.getElementById('panel-' + t.dataset.panel).classList.add('active');
}}));

// Settings table
function renderSettings(filter) {{
  const tbody = document.getElementById('settings-body');
  tbody.innerHTML = '';
  const rows = filter === 'all' ? DATA.settings : DATA.settings.filter(s => s.status === filter);
  rows.forEach(s => {{
    const dot = c => `<span class="dot ${{c ? 'green' : 'red'}}"></span>`;
    const badge = s.status === 'ok' ? '<span class="badge ok">OK</span>'
                : s.status === 'warn' ? '<span class="badge warn">WARN</span>'
                : '<span class="badge error">ERR</span>';
    tbody.innerHTML += `<tr><td>${{s.key}}</td><td>${{dot(s.config)}}</td><td>${{dot(s.routes)}}</td><td>${{dot(s.restore)}}</td><td>${{dot(s.js)}}</td><td>${{badge}}</td></tr>`;
  }});
}}
renderSettings('all');
document.getElementById('settings-filter').addEventListener('change', e => renderSettings(e.target.value));

// Routes table
const rtbody = document.getElementById('routes-body');
DATA.routes.forEach(r => {{
  const dot = c => `<span class="dot ${{c ? 'green' : 'red'}}"></span>`;
  const badge = r.ok ? '<span class="badge ok">OK</span>' : '<span class="badge error">MISSING</span>';
  rtbody.innerHTML += `<tr><td>${{r.name}}</td><td>${{r.prefix}}</td><td>${{dot(r.imported)}}</td><td>${{dot(r.registered)}}</td><td>${{badge}}</td></tr>`;
}});

// Migrations timeline
const mig = document.getElementById('mig-timeline');
DATA.migrations.forEach(m => {{
  mig.innerHTML += `<div class="mig-chip" title="${{m.file}}">${{String(m.number).padStart(3,'0')}} ${{m.name}}</div>`;
}});

// Subsystems
const maxCount = Math.max(...DATA.subsystems.map(s => s.count));
const bars = document.getElementById('subsystems-bars');
DATA.subsystems.forEach(s => {{
  const pct = maxCount > 0 ? (s.count / maxCount * 100) : 0;
  bars.innerHTML += `<div class="bar-row"><span class="bar-label">${{s.name}}</span><div class="bar" style="width:${{pct}}%"></div><span class="bar-count">${{s.count}} files</span></div>`;
}});
</script>
</body></html>'''


def main():
    html = generate_html()
    out = ROOT / "augmentum-health.html"
    out.write_text(html, encoding="utf-8")
    print(f"Generated: {out}")
    try:
        webbrowser.open(f"file:///{out.resolve()}")
    except Exception:
        print("Open the file manually in your browser.")


if __name__ == "__main__":
    main()
