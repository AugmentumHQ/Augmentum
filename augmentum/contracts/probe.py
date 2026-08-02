"""Canonical in-process route-probe harness — runnable against any worktree.

This is the shipped core the augmentum-dev skill AND the self-edit gate share.
It builds the real FastAPI app in-process (lifting tests/conftest.py's mock-app
wiring against a :memory: DB), probes every safe GET route under a per-route
wall-clock, and runs each break through the diagnosis engine.

Run it as a module against a specific worktree:

    python -m augmentum.contracts.probe --out=result.json
    python -m augmentum.contracts.probe --out=cand.json --baseline=base.json
    python -m augmentum.contracts.probe --update-baseline=base.json

The self-edit gate invokes this with ``cwd=<candidate worktree>`` so it imports
and probes the CANDIDATE's code, then diffs against a baseline probe of the
base_ref — only breaks the edit INTRODUCED gate the promotion (the differential
FP-killer). ``--baseline`` marks each finding ``new`` and the exit code is 1
only when a NEW regression/hard_block appears.

Honesty note: in-process probing mocks heavy services, so some handlers 5xx
purely because a service is a mock. That noise is identical across base and
candidate, so the DIFFERENTIAL cancels it — which is the whole point.
"""

from __future__ import annotations

import json
import sys
import threading
import traceback
from pathlib import Path

# augmentum/contracts/probe.py -> augmentum/contracts -> augmentum -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

# Streaming / long-lived endpoints — probing these hangs by design (LLM
# completion, SSE channels, media). Mirrors _STREAMING_PREFIXES in server.py,
# plus the plural /api/builds/ + any path ending in /stream.
_SKIP_PREFIXES = (
    "/api/chat", "/v1/chat", "/api/generate", "/v1/completions", "/v1/embeddings",
    "/api/agentic/", "/api/coder/", "/api/media/stream/", "/stream/", "/api/voice/",
    "/api/audio/", "/v1/audio/", "/api/image/", "/v1/images/", "/api/dream/",
    "/api/narrative/scene-image", "/api/knowledge/packs/install",
    "/api/knowledge/packs/convert", "/api/system/events", "/api/build/", "/api/builds/",
    "/api/artifacts/build-status/",
)


def _skip(path: str) -> bool:
    return any(path.startswith(p) for p in _SKIP_PREFIXES) or path.rstrip("/").endswith("/stream")


# --------------------------------------------------------------------------
# Baseline (a JSON {"known_broken": [route_key, ...]}) — differential state.
# --------------------------------------------------------------------------

def load_baseline(path: str) -> set[str]:
    if not path:
        return set()
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return set(data.get("known_broken", []))
    except Exception:  # noqa: BLE001 — no/unreadable baseline → empty
        return set()


def write_baseline(path: str, keys: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"known_broken": keys}, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# In-process app + probing
# --------------------------------------------------------------------------

def build_probe_app():
    """Construct the real app in-process, wired like tests/conftest.py's `app`
    fixture: a mock session_manager that accepts any bearer token + a real
    :memory: SQLite state backend. Returns (app, backend); keep the backend
    ref alive for the app's lifetime."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from augmentum.auth.models import User
    from augmentum.cache.dedup import RequestDeduplicator
    from augmentum.cache.prefix_cache import PrefixCache
    from augmentum.cache.prompt_cache import PromptCache
    from augmentum.classifier.router import RequestClassifier
    from augmentum.models.provider_registry import ProviderRegistry
    from augmentum.proxy.server import create_app
    from augmentum.state.backends.sqlite import SQLiteBackend
    from augmentum.state.manager import StateManager
    from augmentum.tools.registry import ToolRegistry

    app = create_app()

    test_user = User(
        id="usr_test", username="tester", display_name="Test User",
        role="admin", is_active=True,
    )
    sm = MagicMock()
    sm.validate_token = AsyncMock(return_value=test_user)
    sm.get_user_by_id = AsyncMock(return_value=test_user)
    sm.validate_ws_ticket = MagicMock(return_value=test_user.id)
    app.state.session_manager = sm

    app.state.http_client = MagicMock()
    pr = MagicMock(spec=ProviderRegistry)
    pr.default_backend = MagicMock()
    pr.backends = {}
    pr.probe_deadline_for = MagicMock(return_value=6.0)
    pr.refresh_model_map = AsyncMock(return_value={})
    app.state.provider_registry = pr

    backend = SQLiteBackend(":memory:")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(backend.connect())
    app.state.state_manager = StateManager(backend)

    app.state.classifier = RequestClassifier()
    app.state.narrative_engines = {}
    app.state.tool_registry = ToolRegistry()
    app.state.prompt_cache = PromptCache()
    app.state.prefix_cache = PrefixCache()
    app.state.request_deduplicator = RequestDeduplicator()

    mm = MagicMock()
    mm.list_all_models = AsyncMock(return_value=[])
    mm.get_running_models = AsyncMock(return_value=[])
    app.state.model_manager = mm

    return app, backend


def _fill_path(path: str) -> str:
    import re
    return re.sub(r"\{[^}]+\}", "1", path)


def _new_client(app, *, authed: bool):
    from fastapi.testclient import TestClient

    c = TestClient(app, raise_server_exceptions=authed)
    if authed:
        c.headers.update({"Authorization": "Bearer probe-token"})
    return c


def _probe(client, spec):
    from augmentum.contracts.diagnose import ProbeResult

    url = _fill_path(spec.path)
    query = {q: "1" for q in spec.required_query}
    try:
        resp = client.get(url, params=query)
        body = ""
        try:
            body = resp.text[:200]
        except Exception:  # noqa: BLE001
            pass
        return ProbeResult(route=spec, status=resp.status_code, body_snippet=body)
    except Exception as exc:  # noqa: BLE001 — server exception re-raised by TestClient
        return ProbeResult(
            route=spec, status=None, exception=repr(exc)[:200],
            traceback_text=traceback.format_exc(),
        )


def _probe_one(box, spec, *, app, authed, timeout_s: float = 12.0):
    """Probe one route in a daemon thread under a hard wall-clock. A hang
    (blocked portal) is recorded as a timeout and the poisoned client rebuilt,
    so one bad route can't stall the sweep."""
    from augmentum.contracts.diagnose import ProbeResult

    result: dict = {}

    def work() -> None:
        try:
            result["r"] = _probe(box["c"], spec)
        except Exception as exc:  # noqa: BLE001
            result["r"] = ProbeResult(
                route=spec, status=None, exception=repr(exc)[:200],
                traceback_text=traceback.format_exc(),
            )

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        box["c"] = _new_client(app, authed=authed)
        return ProbeResult(
            route=spec, status=None, timed_out=True, exception=f"exceeded {timeout_s:.0f}s",
        )
    return result.get("r") or ProbeResult(route=spec, status=None, exception="no result")


def run_probe(*, timeout_s: float = 12.0):
    """Build the app, probe every safe GET route (authed + authless security
    pass), diagnose, and return (get_probed, dossiers) where dossiers is the
    list of non-ok findings (crash/hang/authz_flip)."""
    import asyncio

    from augmentum.contracts.diagnose import AUTHZ_FLIP, build_dossier
    from augmentum.contracts.discover import discover_routes

    app, backend = build_probe_app()
    specs = [s for s in discover_routes(app) if s.method == "GET" and not _skip(s.path)]

    # Authed pass.
    dossiers = []
    box = {"c": _new_client(app, authed=True)}
    for spec in specs:
        d = build_dossier(_probe_one(box, spec, app=app, authed=True), REPO_ROOT)
        if d.failure_mode != "ok":
            dossiers.append(d)

    # Security pass — no creds; inject the app's OWN public predicate.
    try:
        from augmentum.auth.middleware import _PUBLIC_PATHS, _PUBLIC_PREFIXES

        def _is_pub(p: str) -> bool:
            return p in _PUBLIC_PATHS or p.startswith(tuple(_PUBLIC_PREFIXES))
    except Exception:  # noqa: BLE001
        _is_pub = None

    abox = {"c": _new_client(app, authed=False)}
    for spec in specs:
        d = build_dossier(
            _probe_one(abox, spec, app=app, authed=False), REPO_ROOT,
            authless=True, is_public_fn=_is_pub,
        )
        if d.failure_mode == AUTHZ_FLIP:
            dossiers.append(d)

    try:
        asyncio.get_event_loop().run_until_complete(backend.close())
    except Exception:  # noqa: BLE001
        pass

    return len(specs), dossiers


def _arg(args, prefix, default=""):
    return next((a.split("=", 1)[1] for a in args if a.startswith(prefix)), default)


def main(argv: list[str] | None = None, *, default_baseline: str = "") -> int:
    from augmentum.contracts.diagnose import ANNOTATE, AUTHZ_FLIP, HANG, REGRESSION

    args = list(sys.argv[1:] if argv is None else argv)
    fmt = _arg(args, "--format=", "text")
    quiet = "--quiet" in args
    timeout_s = float(_arg(args, "--timeout=", "12") or 12)
    baseline_path = _arg(args, "--baseline=", default_baseline)
    update_target = _arg(args, "--update-baseline=", "")
    if "--update-baseline" in args and not update_target:
        update_target = baseline_path or default_baseline

    try:
        get_probed, dossiers = run_probe(timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001
        print(f"contract probe: could not run: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 2

    hangs = [d for d in dossiers if d.failure_mode == HANG]
    regressions = [d for d in dossiers if d.severity == REGRESSION]
    authz_flips = [d for d in dossiers if d.failure_mode == AUTHZ_FLIP]
    annotated = [d for d in dossiers if d.severity == ANNOTATE and d.failure_mode != HANG]
    all_d = regressions + hangs + annotated + authz_flips

    if update_target:
        keys = sorted(
            {d.route.key for d in regressions}
            | {d.route.key for d in authz_flips}
            | {d.route.key for d in hangs}
        )
        write_baseline(update_target, keys)
        print(f"contract baseline updated: {len(keys)} known-broken route(s) -> {update_target}")
        return 0

    baseline = load_baseline(baseline_path)
    new_regress = [d for d in regressions if d.route.key not in baseline]
    new_block = [d for d in authz_flips if d.route.key not in baseline]
    new_hang = [d for d in hangs if d.route.key not in baseline]

    payload = {
        "get_probed": get_probed,
        "regression": len(regressions),
        "new_regression": len(new_regress),
        "hang": len(hangs),
        "new_hang": len(new_hang),
        "annotate": len(annotated),
        "hard_block": len(authz_flips),
        "new_hard_block": len(new_block),
        "findings": [
            {
                "route": d.route.key, "mode": d.failure_mode, "severity": d.severity,
                "locus": d.locus, "source": d.source_line, "exception": d.exception,
                "status": d.status, "handler": d.route.handler, "note": d.note,
                "traceback": (d.traceback or "")[:2500],
                "new": d.route.key not in baseline,
            }
            for d in all_d
        ],
    }

    out = _arg(args, "--out=", "")
    if out:
        try:
            Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"contract probe: could not write --out: {exc}", file=sys.stderr)

    if fmt == "json":
        print(json.dumps(payload, indent=2))
        return 1 if (new_regress or new_block) else 0

    bl = "no baseline" if not baseline else f"{len(baseline)} baselined"
    print(f"contracts: {get_probed} GET probed | {len(regressions)} regression "
          f"({len(new_regress)} NEW) | {len(hangs)} hang ({len(new_hang)} new) | "
          f"{len(annotated)} annotate | {len(authz_flips)} hard_block "
          f"({len(new_block)} NEW) | {bl}")

    if not quiet:
        def _show(title: str, items: list) -> None:
            if not items:
                return
            print(f"\n{title} ({len(items)}):")
            for d in items:
                print(f"  {'[NEW]' if d.route.key not in baseline else '[baselined]'}")
                print(d.render())
                print()

        _show("[SECURITY] authz flips (non-public route answered unauthenticated)", authz_flips)
        _show("[REGRESSION] in-code breaks", regressions)
        _show("[HANG] routes that exceeded the probe timeout", hangs)
        if annotated and "--verbose" in args:
            _show("[ANNOTATE] crashes with no in-repo frame (mock-harness limits)", annotated)
        elif annotated:
            print(f"\n[ANNOTATE] {len(annotated)} crash(es) with no in-repo frame "
                  f"(mock-harness limits; --verbose to list).")

    return 1 if (new_regress or new_block) else 0


if __name__ == "__main__":
    sys.exit(main())
