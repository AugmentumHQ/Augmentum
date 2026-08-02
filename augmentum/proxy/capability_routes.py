"""HTTP surface for capability synthesis — the user-facing face of the
self-evolving lane (augmentum/selfedit/capabilities).

Flow the UI drives:
  1. POST /api/selfedit/capability/triage   {request}        → ready|clarify|refuse
  2. (if clarify) render the questions, collect answers
  3. POST /api/selfedit/capability/synthesize {request, answers?}
        → the acceptance test + the oracle verdict (valid|vacuous|broken)
  4. POST /api/selfedit/capability/build {request, answers?}  (GATED on
        selfedit_enabled) → drives the live engine to implement against the
        valid oracle, returns the attempt outcome for keep/revert.

triage + synthesize are read-only (model calls + a sandboxed pytest collect) and
always available. build mutates via the edit engine, so it's gated OFF by default
(selfedit_enabled) — exactly like the rest of the self-edit surface.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/selfedit/capability", tags=["selfedit"])


def _uid(request: Request) -> str:
    user = request.scope.get("user")
    return getattr(user, "id", "") if user else ""


async def _body(request: Request) -> dict:
    try:
        b = await request.json()
        return b if isinstance(b, dict) else {}
    except Exception:
        return {}


@router.post("/triage")
async def capability_triage(request: Request) -> JSONResponse:
    """Classify a capability request: ready / clarify (questions) / refuse."""
    body = await _body(request)
    req = str(body.get("request", "")).strip()
    if not req:
        return JSONResponse({"error": "request is required"}, status_code=400)
    from augmentum.selfedit.capabilities import triage_capability_request
    from augmentum.selfedit.capabilities.runtime import build_direct_model_invoke
    mi = build_direct_model_invoke(request.app.state, model=str(body.get("model", "")))
    res = await triage_capability_request(req, model_invoke=mi)
    return JSONResponse(res.to_dict())


@router.post("/synthesize")
async def capability_synthesize(request: Request) -> JSONResponse:
    """Author the acceptance test (scaffolded, verb-shaped) and run it through the
    oracle gate. Returns the test + verdict; does NOT write to the repo."""
    body = await _body(request)
    req = str(body.get("request", "")).strip()
    if not req:
        return JSONResponse({"error": "request is required"}, status_code=400)
    answers = body.get("answers") if isinstance(body.get("answers"), dict) else {}
    from augmentum.selfedit.capabilities import (
        apply_clarifications,
        oracle_verdict,
        synthesize_verb_acceptance,
    )
    from augmentum.selfedit.capabilities.runtime import build_direct_model_invoke
    mi = build_direct_model_invoke(request.app.state, model=str(body.get("model", "")))
    clarified = apply_clarifications(req, answers)
    rel, src = await synthesize_verb_acceptance(clarified, model_invoke=mi)
    if not rel:
        return JSONResponse({"status": "rejected", "error": src})
    # The scaffolded test verifies via the boot path (REGISTRY.get after importing
    # the package), so an unbuilt verb fails with a "must be registered" assertion.
    # Derive the impl module deterministically from the test path so the oracle can
    # ALSO recognize a ModuleNotFoundError (partial build) as the right reason.
    # rel = tests/test_authored_<safe>.py  →  module = augmentum.intent.builtin.syn_<safe>
    safe = rel.removeprefix("tests/test_authored_").removesuffix(".py")
    expected = f"augmentum.intent.builtin.syn_{safe}"
    verdict, detail = await oracle_verdict(src, expected_missing_module=expected)
    return JSONResponse({
        "status": "ok", "test_path": rel, "test_source": src,
        "oracle": verdict, "oracle_detail": detail,
        # 'valid' = gate confirmed right-reason failure; 'unverified' = the scaffold
        # is correct-by-construction but pytest couldn't run here (the engine
        # re-runs the real gate in the worktree at build time). Both are buildable;
        # 'vacuous'/'broken' are not.
        "buildable": verdict in ("valid", "unverified"),
    })


@router.post("/build")
async def capability_build(request: Request) -> JSONResponse:
    """Drive the live edit engine to implement against the valid oracle. GATED:
    requires selfedit_enabled + a wired driver. Off by default → returns a clear
    'not enabled' rather than running."""
    body = await _body(request)
    req = str(body.get("request", "")).strip()
    uid = _uid(request)
    if not req:
        return JSONResponse({"error": "request is required"}, status_code=400)
    if not uid:
        return JSONResponse({"error": "authentication required"}, status_code=401)

    store = getattr(request.app.state, "settings_store", None)
    enabled = False
    if store is not None:
        try:
            enabled = (await store.get("selfedit_enabled")) in ("1", "true", "True", 1, True)
        except Exception:
            enabled = False
    driver = getattr(request.app.state, "selfedit_driver", None)
    repo_dir = getattr(request.app.state, "selfedit_repo_dir", "")
    if not (enabled and driver is not None and repo_dir):
        return JSONResponse({
            "status": "not_enabled",
            "detail": ("capability build is gated — needs selfedit_enabled + a wired "
                       "driver. Use /triage and /synthesize to preview the oracle; "
                       "flip selfedit_enabled to build."),
            "selfedit_enabled": bool(enabled),
            "driver_wired": driver is not None,
        })

    answers = body.get("answers") if isinstance(body.get("answers"), dict) else {}
    from augmentum.selfedit.capabilities import (
        apply_clarifications,
        author_capability,
        triage_capability_request,
    )
    from augmentum.selfedit.capabilities.runtime import build_direct_model_invoke
    conn = getattr(getattr(request.app.state, "sqlite_backend", None), "conn", None)
    mi = build_direct_model_invoke(request.app.state, model=str(body.get("model", "")))
    clarified = apply_clarifications(req, answers)
    # Pre-gate: never drive the engine on a vague/unsafe ask — triage must say
    # 'ready' first (this is what stops a junk request burning a full edit cycle).
    triage = await triage_capability_request(clarified, model_invoke=mi)
    if triage.status != "ready":
        return JSONResponse({
            "status": "not_ready", "triage": triage.to_dict(),
            "detail": "resolve triage (clarify/refuse) before building",
        })
    try:
        outcome, err = await author_capability(
            clarified, repo_dir=repo_dir, conn=conn,
            driver=driver, model_invoke=mi, user_id=uid,
        )
    except Exception as exc:  # noqa: BLE001 — surface, never 500 silently
        log.warning("capability_build_failed", error=repr(exc))
        return JSONResponse({"error": f"build failed: {exc!r}"}, status_code=500)
    if outcome is None:
        return JSONResponse({"status": "not_authorable", "detail": err})
    return JSONResponse({
        "status": getattr(outcome, "status", "?"),
        "attempt_id": getattr(outcome, "attempt_id", ""),
        "lesson": getattr(outcome, "lesson", ""),
    })
