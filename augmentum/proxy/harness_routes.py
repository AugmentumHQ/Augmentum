"""Harness harvest review API — the observation-only staging surface.

Harness turns (OpenCode / Claude Code / …) no longer mutate live memory. Each
turn's extracted harvest CANDIDATES are STAGED for review here; a deliberate
pass promotes the good ones into the baseline. This is the "place we can see"
what's worth harvesting and the trends, before anything reaches the baseline.

See augmentum/proxy/harness.py (staging) and augmentum/training/capture.py
(the harvest feed it folds into).
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter(prefix="/api/harness", tags=["harness"])


def _uid(request: Request) -> str:
    user = request.scope.get("user")
    return getattr(user, "id", "") if user else ""


@router.get("/harvest")
async def list_harvest(
    request: Request, limit: int = 100, include_harvested: bool = False,
) -> JSONResponse:
    """Staged harvest candidates for the caller, most recent first."""
    from augmentum.training.capture import read_harness_harvest

    recs = read_harness_harvest(
        user_id=_uid(request), limit=limit, include_harvested=include_harvested,
    )
    return JSONResponse({"records": recs, "count": len(recs)})


@router.get("/harvest/trends")
async def harvest_trends(request: Request) -> JSONResponse:
    """Aggregate trends over the caller's staged candidates (by kind / harness,
    durable share, pending vs promoted vs dismissed)."""
    from augmentum.training.capture import harness_harvest_trends

    return JSONResponse(harness_harvest_trends(user_id=_uid(request)))


async def _decision_args(request: Request) -> tuple[str, int]:
    body = await request.json()
    return str(body.get("obs_id") or ""), int(body.get("candidate_index") or 0)


@router.post("/harvest/promote")
async def promote(request: Request) -> JSONResponse:
    """Promote ONE staged candidate into the baseline (the deliberate harvest).
    Body: {"obs_id": "...", "candidate_index": 0}."""
    from augmentum.proxy.harness import promote_candidate

    uid = _uid(request)
    if not uid:
        return JSONResponse({"status": "error", "error": "auth required"}, status_code=401)
    obs_id, idx = await _decision_args(request)
    if not obs_id:
        return JSONResponse({"status": "error", "error": "obs_id required"}, status_code=400)
    result = await promote_candidate(request.app.state, uid, obs_id, idx)
    code = 200 if result.get("status") == "promoted" else 400
    return JSONResponse(result, status_code=code)


@router.post("/harvest/dismiss")
async def dismiss(request: Request) -> JSONResponse:
    """Dismiss a staged candidate (no baseline write — drops it from the queue).
    Body: {"obs_id": "...", "candidate_index": 0}."""
    from augmentum.proxy.harness import dismiss_candidate

    uid = _uid(request)
    if not uid:
        return JSONResponse({"status": "error", "error": "auth required"}, status_code=401)
    obs_id, idx = await _decision_args(request)
    if not obs_id:
        return JSONResponse({"status": "error", "error": "obs_id required"}, status_code=400)
    return JSONResponse(dismiss_candidate(uid, obs_id, idx))


# ── Agent bridge (external agents ↔ user via notifications) ────────────

@router.get("/agents")
async def list_agents(request: Request) -> JSONResponse:
    """Active external agent sessions (recent heartbeat) + pending asks."""
    from augmentum.proxy import agent_bridge

    uid = _uid(request)
    if not uid:
        return JSONResponse({"agents": []}, status_code=401)
    agents = await agent_bridge.list_agents(request.app.state, user_id=uid)
    return JSONResponse({"agents": agents, "count": len(agents)})


@router.post("/agent/reply")
async def agent_reply(request: Request) -> JSONResponse:
    """User answers an agent request (button action and/or free text).
    Body: {"request_id": "...", "action": "approve|deny|", "text": "..."}."""
    from augmentum.proxy import agent_bridge

    uid = _uid(request)
    if not uid:
        return JSONResponse({"ok": False, "error": "auth required"}, status_code=401)
    body = await request.json()
    result = await agent_bridge.answer_request(
        request.app.state, user_id=uid,
        request_id=str(body.get("request_id") or ""),
        action=str(body.get("action") or ""),
        text=str(body.get("text") or ""),
    )
    return JSONResponse(result, status_code=200 if result.get("ok") else 400)


@router.post("/checkin")
async def agent_checkin(request: Request) -> JSONResponse:
    """Presence heartbeat + reverse-channel pickup for a bare-machine agent.

    The provider-neutral seam every "My machine" client uses (the claude-aug
    SessionStart/Stop hook and the pi extension) to (a) register/refresh its
    session and (b) receive user replies + QUEUED ASSIGNMENTS. Mirrors the
    ``agent_checkin`` MCP tool but over plain REST so a shell hook or a TS
    extension can call it with just the API key.

    Body: {agent_id?, harness, project?, title?, status?, summary?}. Returns
    {agent_id, status, answered_requests, assignments:[{request_id, title,
    task, run_id}]}. Reporting status='done'|'failed' finalizes this session's
    in-flight assignment runs (the Agents history advances to that state)."""
    from augmentum.proxy import agent_bridge

    uid = _uid(request)
    if not uid:
        return JSONResponse({"error": "auth required"}, status_code=401)
    body = await request.json()
    res = await agent_bridge.checkin(
        request.app.state, user_id=uid,
        harness=str(body.get("harness") or "").strip() or "external",
        project=str(body.get("project") or "").strip(),
        agent_id=str(body.get("agent_id") or "").strip(),
        title=str(body.get("title") or "").strip(),
        status=str(body.get("status") or "working").strip(),
        summary=str(body.get("summary") or "").strip(),
    )
    if res is None:
        return JSONResponse({"error": "unavailable"}, status_code=503)
    return JSONResponse(res)


@router.get("/harvest/view", response_class=HTMLResponse)
async def harvest_view(request: Request) -> HTMLResponse:
    """A simple self-contained review page: lists pending harvest candidates with
    Promote / Dismiss buttons. Uses the same-origin session cookie for auth."""
    return HTMLResponse(_REVIEW_HTML)


_REVIEW_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Harness Harvest — Review</title>
<style>
 :root { color-scheme: dark light; }
 body { font: 15px/1.5 system-ui, sans-serif; margin: 0; padding: 24px;
        background: #0f1115; color: #e6e6e6; }
 h1 { font-size: 20px; margin: 0 0 4px; }
 .sub { color: #8a93a2; margin: 0 0 20px; font-size: 13px; }
 .trends { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 22px; }
 .chip { background: #1a1e27; border: 1px solid #2a3140; border-radius: 999px;
         padding: 5px 12px; font-size: 12px; color: #b9c2d0; }
 .chip b { color: #fff; }
 .card { background: #161a22; border: 1px solid #262d3a; border-radius: 12px;
         padding: 14px 16px; margin-bottom: 12px; }
 .meta { font-size: 12px; color: #7c8696; margin-bottom: 8px; }
 .meta .tag { color: #9ecbff; }
 .text { font-size: 15px; margin: 4px 0 6px; }
 .flags { font-size: 12px; color: #e0a458; margin-bottom: 8px; }
 .src { font-size: 12px; color: #6b7382; font-style: italic; margin-bottom: 10px;
        white-space: pre-wrap; }
 button { font: inherit; border: 0; border-radius: 8px; padding: 6px 14px;
          cursor: pointer; margin-right: 8px; }
 .promote { background: #1f6f43; color: #fff; }
 .dismiss { background: #3a2330; color: #f0c0c8; }
 .empty { color: #7c8696; padding: 40px 0; text-align: center; }
 .done { opacity: .45; }
</style></head>
<body>
 <h1>Harness Harvest — Review</h1>
 <p class="sub">Staged from harness sessions. Nothing here is in your baseline
   until you Promote it. Dismiss drops it from the queue.</p>
 <div class="trends" id="trends"></div>
 <div id="list"><p class="empty">Loading…</p></div>
<script>
const esc = s => String(s==null?'':s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function load() {
  const [tr, hv] = await Promise.all([
    fetch('/api/harness/harvest/trends', {credentials:'same-origin'}).then(r=>r.json()),
    fetch('/api/harness/harvest?limit=200', {credentials:'same-origin'}).then(r=>r.json()),
  ]);
  const t = document.getElementById('trends');
  t.innerHTML = [
    ['pending', tr.pending], ['promoted', tr.promoted], ['dismissed', tr.dismissed],
    ['durable', tr.durable_candidates], ['candidates', tr.candidates],
  ].map(([k,v]) => `<span class="chip">${k} <b>${v||0}</b></span>`).join('')
   + Object.entries(tr.by_kind||{}).map(([k,v]) =>
       `<span class="chip">${esc(k)} <b>${v}</b></span>`).join('');
  const list = document.getElementById('list');
  const rows = [];
  (hv.records||[]).forEach(rec => (rec.candidates||[]).forEach((c, i) => {
    if (c.status && c.status !== 'pending') return;
    rows.push(`<div class="card" id="c-${rec.obs_id}-${i}">
      <div class="meta"><span class="tag">${esc(c.kind)}</span>
        ${c.durable?' · durable':''} · ${esc(rec.harness||'?')}
        ${c.target_scope?` · → ${esc(c.target_scope)}`:''}
        · ${esc((rec.timestamp||'').slice(0,16).replace('T',' '))}</div>
      <div class="text">${esc(c.text)}</div>
      ${c.supersedes_baseline_id?`<div class="flags">⚠ would replace baseline: “${esc(c.supersedes_baseline_text)}”</div>`:''}
      <div class="src">from: ${esc((rec.source_message||'').slice(0,200))}</div>
      <button class="promote" onclick="decide('promote','${rec.obs_id}',${i})">Promote</button>
      <button class="dismiss" onclick="decide('dismiss','${rec.obs_id}',${i})">Dismiss</button>
    </div>`);
  }));
  list.innerHTML = rows.length ? rows.join('') :
    '<p class="empty">No pending candidates. ✨</p>';
}
async function decide(action, obs_id, idx) {
  const card = document.getElementById(`c-${obs_id}-${idx}`);
  if (card) card.classList.add('done');
  await fetch(`/api/harness/harvest/${action}`, {
    method:'POST', credentials:'same-origin',
    headers:{'content-type':'application/json'},
    body: JSON.stringify({obs_id, candidate_index: idx}),
  });
  load();
}
load();
</script></body></html>"""
