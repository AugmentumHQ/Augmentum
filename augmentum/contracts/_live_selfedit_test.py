"""LIVE validation (in-container): does deepseek-v4-flash, given the contract
gate's failure report, diagnose and fix a route break?

Runs the REAL loop with the REAL model, isolated from all live state:
  base    = /host-augmentum-src  (pristine working tree, read-only)
  cand    = /data/setest/cand    (a copy of augmentum/ we deliberately break)
  1. inject a broken lazy import into a GET handler (mirrors the _LITE_TEMPLATES
     class: passes boot-smoke, 500s only when the route is hit)
  2. run contract_regression_verifier(base) against cand -> expect FAIL + a
     .augmentum/contract_failures.md report written into cand
  3. drive the real native deepseek-v4-flash self-edit driver with that repair
     context -> the model reads the report, navigates, fixes
  4. re-run the gate -> expect PASS

No git worktrees, no growth DB (conn=None), no writes to the live DB/repo.

    docker exec -u augmentum augmentum-augmentum-1 \
        python -m augmentum.contracts._live_selfedit_test
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import types
from pathlib import Path

SRC = "/host-augmentum-src"
CAND = "/data/setest/cand"
MODEL = "deepseek-v4-flash"
TARGET_REL = "augmentum/proxy/ui_routes.py"
BREAK_MARKER = "from augmentum.modes.narrative.memory import _LIVE_TEST_MISSING_NAME  # LIVE-TEST-BREAK"


def _log(msg: str) -> None:
    print(f"[live-test] {msg}", flush=True)


def make_candidate() -> None:
    cand = Path(CAND)
    if cand.exists():
        shutil.rmtree(cand, ignore_errors=True)
    cand.mkdir(parents=True, exist_ok=True)
    # Only augmentum/ is needed for create_app + the probe (deps are in the
    # container site-packages). Skip caches to keep the copy fast.
    shutil.copytree(
        f"{SRC}/augmentum", f"{CAND}/augmentum",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    _log(f"candidate built at {CAND} (copy of augmentum/)")


def break_a_route() -> str:
    """Inject a broken lazy import at the top of the get_ltm_prompt handler body
    in the candidate. Returns the handler for reference."""
    f = Path(CAND) / TARGET_REL
    txt = f.read_text(encoding="utf-8")
    anchor = '    """Get the current LTM prompt template (custom or default based on card type and mode)."""'
    if anchor not in txt:
        raise SystemExit(f"anchor not found in {TARGET_REL} — handler shape changed")
    broken = txt.replace(anchor, anchor + "\n    " + BREAK_MARKER, 1)
    f.write_text(broken, encoding="utf-8")
    _log(f"injected break into {TARGET_REL}::get_ltm_prompt")
    return "get_ltm_prompt"


async def build_registry():
    import aiosqlite
    import httpx

    from augmentum.models.provider_registry import ProviderRegistry
    from augmentum.state.provider_store import ProviderStore

    conn = await aiosqlite.connect("file:/data/augmentum.db?mode=ro", uri=True)
    conn.row_factory = aiosqlite.Row
    reg = ProviderRegistry(httpx.AsyncClient())
    await reg.load_runtime_providers(ProviderStore(conn))
    _log("provider registry built (read-only)")
    return reg


def repair_prompt(detail: str) -> str:
    return (
        "Your previous edit did NOT pass verification. Fix ONLY the failure below, "
        "editing the file you already changed; do not start over or add unrelated "
        "work:\n\n"
        f"  - [contract_regression] {detail}\n\n"
        "Make the smallest change that resolves it, then stop."
    )


async def main() -> int:
    from augmentum.contracts.selfedit_gate import contract_regression_verifier
    from augmentum.selfedit.engine_select import build_selfedit_driver
    from augmentum.selfedit.orchestrator import EditRequest

    make_candidate()
    break_a_route()

    gate = contract_regression_verifier(base_dir=SRC)

    _log("STEP 1 — probing base_ref + broken candidate (this builds the baseline once)…")
    r1 = await gate.run({"candidate_dir": CAND})
    _log(f"gate #1: status={r1.status}")
    _log(f"   detail: {r1.detail[:240]}")
    report = Path(CAND) / ".augmentum" / "contract_failures.md"
    _log(f"   report written: {report.exists()}")
    if r1.status != "fail":
        _log("EXPECTED FAIL but gate did not fail — aborting")
        return 1

    _log(f"STEP 2 — driving REAL {MODEL} self-edit driver to repair…")
    reg = await build_registry()
    driver = await build_selfedit_driver(
        conn=None, engine="native", registry=reg, model=MODEL, max_iters=40,
    )
    if driver is None:
        _log("driver could not be built — aborting")
        return 2
    cand_obj = types.SimpleNamespace(path=CAND, name="setest", branch="setest", base_sha="")
    edit = await driver(EditRequest(
        candidate=cand_obj,
        objective="A recent edit broke a route in this app. Diagnose and fix it.",
        attempt_id="live-test", user_id="test",
        prior_context=repair_prompt(r1.detail),
    ))
    _log(f"driver finished: ok={getattr(edit, 'ok', '?')} error={getattr(edit, 'error', '') or '-'}")

    _log("STEP 3 — re-running the gate to verify the fix…")
    r2 = await gate.run({"candidate_dir": CAND})
    _log(f"gate #2: status={r2.status}")
    _log(f"   detail: {r2.detail[:200]}")

    ok = r2.status == "pass"
    _log("=" * 60)
    _log(f"RESULT: {'PASS - deepseek fixed the contract-gate break end to end' if ok else 'NOT FIXED - see gate #2'}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
