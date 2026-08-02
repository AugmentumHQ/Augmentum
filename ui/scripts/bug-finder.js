/**
 * bug-finder.js — Bug Finder surface module.
 *
 * Forensic Diagnostic Lab aesthetic — see ui/styles/bug-finder.css.
 * Bug Finder is set-and-forget: a run is enqueued as a background
 * job; the panel polls for completion; the report renders when ready.
 *
 * State held here:
 *   - runs           — list of run summaries (rail)
 *   - currentRunId   — selected run; '' when nothing's open
 *   - currentReport  — full report for currentRunId (lazy-fetched)
 *   - currentTab     — findings | patches | cost | baseline
 *   - filters        — { minConfirm, severity[], signature[] }
 *
 * Polling: while any run on the rail has stop_reason='running' we
 * re-fetch the list every 5s. Simpler than per-job polling and good
 * enough for Phase 1 — finer-grained progress is a Phase 2 concern.
 */

import { escapeHtml, showToast } from './app.js';
import { getModels } from './model-cache.js';
import { rafCoalesce } from './raf-coalesce.js';

// --------------------------------------------------------------------------
// State
// --------------------------------------------------------------------------

const _state = {
  initialized: false,
  shellMounted: false,
  active: false,
  runs: [],
  currentRunId: '',
  currentReport: null,
  currentTab: 'findings',
  filters: {
    minConfirm: 0,
    statuses: new Set(),
    severities: new Set(),
  },
  pollTimer: null,
  fetchInFlight: false,
  launchDefaults: null,
  liveStream: null,
  liveStreamRunId: '',
  runningShellRunId: '',  // which run's live shell is mounted in #bf-detail
  railSig: '',            // signature of last-rendered rail (skip churn)
};

const _STATUS_ORDER = [
  'fixed', 'confirmed', 'fix_failed', 'unconfirmable', 'speculative',
];

const _SEV_ORDER = ['critical', 'high', 'medium', 'low', 'info'];

const _STAGE_LABELS = [
  'PREPARING',
  'PLANNING',
  'DETECTING',
  'VERIFYING',
  'FIXING',
  'COMPLETE',
];


// --------------------------------------------------------------------------
// API
// --------------------------------------------------------------------------

async function _api(path, opts = {}) {
  const resp = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!resp.ok) {
    let detail = '';
    let detailObj = null;
    try {
      const body = await resp.json();
      detail = body?.detail || '';
      // Structured details (e.g., capability-gate refusal) ride as
      // an object; the string fallback is the FastAPI default for
      // simple raises. Expose both so callers can branch.
      if (detail && typeof detail === 'object') {
        detailObj = detail;
        detail = detailObj.message || JSON.stringify(detailObj);
      }
    } catch (_) {}
    const err = new Error(detail || `${resp.status} ${resp.statusText}`);
    err.status = resp.status;
    err.detail = detailObj;
    throw err;
  }
  return resp.json();
}

async function _fetchRuns() {
  const data = await _api('/api/bug-finder/runs?limit=50');
  return Array.isArray(data.runs) ? data.runs : [];
}

async function _fetchRun(runId) {
  return _api(`/api/bug-finder/runs/${encodeURIComponent(runId)}`);
}

async function _createRun(payload) {
  return _api('/api/bug-finder/runs', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}

async function _cancelRun(runId) {
  return _api(`/api/bug-finder/runs/${encodeURIComponent(runId)}`, {
    method: 'DELETE',
  });
}

// ──────────────────────────────────────────────────────────────────────
// SSE live progress stream
// ──────────────────────────────────────────────────────────────────────
//
// The polling loop still backstops everything (5s cadence; survives a
// disconnected stream), but EventSource gives stage transitions a
// sub-second feel. We subscribe when rendering the running detail card
// and tear down on terminal event or surface deactivate. Reconnect is
// implicit: a new card render opens a fresh EventSource, and the hub
// replays buffered events so the rail stays consistent.

const _STAGE_PROGRESS = [
  { stage: 'workspace_ready',   label: 'PREPARING' },
  { stage: 'agnostic_substrate', label: 'SCANNING' },
  { stage: 'comprehending',     label: 'COMPREHENDING' },
  { stage: 'planning',          label: 'PLANNING' },
  { stage: 'detecting',         label: 'DETECTING' },
  { stage: 'fuzzing',           label: 'FUZZING' },
  { stage: 'verifying',         label: 'VERIFYING' },
  { stage: 'fixing',            label: 'FIXING' },
  { stage: 'complete',          label: 'COMPLETE' },
];

// Stages the backend emits that don't get their own segment — they map
// onto an existing display slot. The pipeline branches (named-bug runs
// take `lead_loop` in place of detect→fuzz; pen-test is an opt-in leg
// after verify), and the orchestrator also emits raw snake_case ids.
// Without this aliasing, `findIndex` returns -1 on those stages, which
// blanks every segment (the bar appears to run backwards) and prints
// the raw id as the label. See _paintLiveDashboard's idx guard.
const _STAGE_ALIAS = {
  lead_loop:     'detecting',     // named-bug detection path
  pen_testing:   'verifying',     // active-probe leg runs just after verify
  writing_checks: 'comprehending', // check-writer runs right after comprehension
};

// Live-state buffer per-run; drives the dashboard's counters and feed.
// Reset when a new running detail card mounts. Keep it bounded.
const _liveState = {
  runId: '',
  currentStage: 'workspace_ready',
  lastStageIdx: 0,       // last resolved segment index — never regresses
  chunks: [],            // [{ file, function, line_start, line_end, ... }]
  chunksTotal: 0,
  chunksDone: 0,
  currentChunkFile: '',
  currentChunkFn: '',
  knowledge: null,       // { had_prior_map, brief_chars }
  feed: [],              // list of { ts, kind, label, detail }
  findings: [],          // list of finding payloads in arrival order
  tokensIn: 0,
  tokensOut: 0,
  cost: null,            // { by_stage:{stage:{tokens_in,tokens_out,ms}}, total_in, total_out }
  suspectByChunk: {},    // "file::function" -> suspected_class (planner rationale)
};

function _resetLiveState(runId) {
  _liveState.runId = runId;
  _liveState.currentStage = 'workspace_ready';
  _liveState.lastStageIdx = 0;
  _liveState.chunks = [];
  _liveState.chunksTotal = 0;
  _liveState.chunksDone = 0;
  _liveState.currentChunkFile = '';
  _liveState.currentChunkFn = '';
  _liveState.knowledge = null;
  _liveState.feed = [];
  _liveState.findings = [];
  _liveState.tokensIn = 0;
  _liveState.tokensOut = 0;
  _liveState.cost = null;
  _liveState.suspectByChunk = {};
}

function _appendFeed(label, detail = '') {
  // Terminal-style: append at the end (newest at bottom), keep a deep
  // scrollback so the full run reads like a console log. Trim from the
  // FRONT once we exceed the cap so old lines age out, not recent ones.
  _liveState.feed.push({
    ts: new Date().toISOString().slice(11, 19),
    label,
    detail,
  });
  const CAP = 500;
  if (_liveState.feed.length > CAP) {
    _liveState.feed.splice(0, _liveState.feed.length - CAP);
  }
}

function _subscribeLive(runId) {
  if (!runId) return;
  if (_state.liveStreamRunId === runId && _state.liveStream) return;
  _closeLive();
  let stream;
  try {
    stream = new EventSource(
      `/api/bug-finder/runs/${encodeURIComponent(runId)}/events`,
    );
  } catch (err) {
    console.debug('[BugFinder] SSE open failed', err);
    return;
  }
  _state.liveStream = stream;
  _state.liveStreamRunId = runId;
  stream.addEventListener('stage', (ev) => {
    try {
      const data = JSON.parse(ev.data || '{}');
      _liveState.currentStage = data.stage || _liveState.currentStage;
      _appendFeed(
        `STAGE: ${(data.stage || '').toUpperCase()}`,
        data.note || (data.chunks ? `${data.chunks} chunks` : ''),
      );
      _paintLiveDashboard();
    } catch (err) { /* malformed — ignore */ }
  });
  stream.addEventListener('comprehension_complete', (ev) => {
    try {
      const data = JSON.parse(ev.data || '{}');
      _liveState.knowledge = data;
      _appendFeed(
        data.had_prior_map ? 'KNOWLEDGE LOADED' : 'KNOWLEDGE SKIPPED',
        data.had_prior_map
          ? `${data.brief_chars}-char brief reused`
          : 'no map persisted — running blind',
      );
      _paintLiveDashboard();
    } catch (err) { /* ignore */ }
  });
  stream.addEventListener('check_written', (ev) => {
    try {
      const data = JSON.parse(ev.data || '{}');
      if (data.valid) {
        _appendFeed(
          `✎ WROTE CHECK ${data.check || ''}`,
          `${data.source_lines || 0} lines · pillar "${data.pillar || ''}"`,
        );
      } else {
        _appendFeed(
          `✎ CHECK SKIPPED (${data.pillar || ''})`,
          data.reason || '',
        );
      }
      _paintLiveDashboard();
    } catch (err) { /* ignore */ }
  });
  stream.addEventListener('check_writer_complete', (ev) => {
    try {
      const data = JSON.parse(ev.data || '{}');
      if ((data.written || 0) > 0) {
        _appendFeed(
          `CHECK-WRITER: ${data.written} new check(s)`,
          `${data.seeded || 0} finding(s) seeded this run · all run free next audit`,
        );
      }
      _paintLiveDashboard();
    } catch (err) { /* ignore */ }
  });
  stream.addEventListener('planner_complete', (ev) => {
    try {
      const data = JSON.parse(ev.data || '{}');
      _liveState.chunks = data.chunks || [];
      _liveState.chunksTotal = _liveState.chunks.length;
      // Remember why the planner picked each chunk — surfaced on the
      // "examining" readout so the user sees what it suspects, not just
      // where it's looking.
      _liveState.suspectByChunk = {};
      for (const c of _liveState.chunks) {
        if (c.suspected_class) {
          _liveState.suspectByChunk[`${c.file || ''}::${c.function || ''}`] = c.suspected_class;
        }
      }
      _appendFeed(
        'PLANNER EMITTED CHUNKS',
        `${_liveState.chunksTotal} targets queued for detection`,
      );
      _paintLiveDashboard();
    } catch (err) { /* ignore */ }
  });
  stream.addEventListener('cost', (ev) => {
    try {
      _liveState.cost = JSON.parse(ev.data || '{}');
      _paintLiveDashboard();
    } catch (err) { /* ignore */ }
  });
  stream.addEventListener('chunk_detect_started', (ev) => {
    try {
      const data = JSON.parse(ev.data || '{}');
      _liveState.currentChunkFile = data.file || '';
      _liveState.currentChunkFn = data.function || '';
      _liveState.chunksDone = data.chunks_done || 0;
      _appendFeed(
        `EXAMINING ${data.function || '<module>'}`,
        data.file || '',
      );
      _paintLiveDashboard();
    } catch (err) { /* ignore */ }
  });
  stream.addEventListener('chunk_detect_complete', (ev) => {
    try {
      const data = JSON.parse(ev.data || '{}');
      _liveState.chunksDone = data.chunks_done || _liveState.chunksDone;
      if (data.findings_from_chunk > 0) {
        _appendFeed(
          `✶ ${data.findings_from_chunk} finding(s) in chunk`,
          `${data.function || ''}`,
        );
      }
      _paintLiveDashboard();
    } catch (err) { /* ignore */ }
  });
  stream.addEventListener('subagent_progress', (ev) => {
    try {
      const data = JSON.parse(ev.data || '{}');
      // Phase: responding | tool_call | tool_result | stuck | done
      const phaseLabel = {
        responding:  'THINKING',
        tool_call:   `→ ${data.tool_name || 'tool'}`,
        tool_result: `← ${data.tool_name || 'tool'}`,
        stuck:       'STUCK',
        done:        'DONE',
      }[data.phase] || data.phase || 'progress';
      const roleLabel = (data.role || '').toUpperCase();
      const iter = data.iteration != null ? `i${data.iteration}` : '';
      const head = `${roleLabel}${iter ? ` ${iter}` : ''} · ${phaseLabel}`;
      const preview = (data.text_preview || '').slice(0, 80).replace(/\s+/g, ' ');
      _appendFeed(head, preview);
      // Roll running token counts
      if (typeof data.tokens_in === 'number')  _liveState.tokensIn  = data.tokens_in;
      if (typeof data.tokens_out === 'number') _liveState.tokensOut = data.tokens_out;
      _paintLiveDashboard();
    } catch (err) { /* ignore */ }
  });
  stream.addEventListener('finding_landed', (ev) => {
    try {
      const data = JSON.parse(ev.data || '{}');
      _liveState.findings.push(data);
      _appendFeed(
        `[${(data.severity || '').toUpperCase()}] FINDING`,
        (data.claim || '').slice(0, 100),
      );
      _paintLiveDashboard();
    } catch (err) { /* ignore */ }
  });
  stream.addEventListener('done', (ev) => {
    try {
      const data = JSON.parse(ev.data || '{}');
      _liveState.currentStage = 'complete';
      _appendFeed('RUN COMPLETE', data.stop_reason || 'done');
      _paintLiveDashboard();
    } catch (err) { /* ignore */ }
    _closeLive();
    refresh();
  });
  stream.onerror = () => {
    // EventSource auto-reconnects; we close on terminal explicitly via
    // the `done` handler. A persistent error during a closed stream is
    // harmless — the polling loop has already moved past.
  };
}

function _closeLive() {
  if (_state.liveStream) {
    try { _state.liveStream.close(); } catch (_) { /* noop */ }
  }
  _state.liveStream = null;
  _state.liveStreamRunId = '';
}

function _fmtTok(n) {
  n = Number(n) || 0;
  if (n >= 1000) return `${(n / 1000).toFixed(n >= 10000 ? 0 : 1)}k`;
  return `${n}`;
}

// Burndown buckets in pipeline order, with display labels matching the
// orchestrator's _COST_BUCKET keys.
const _BURN_STAGES = [
  ['comprehend', 'COMPREHEND'],
  ['planner',    'PLANNER'],
  ['lead',       'LEAD'],
  ['detector',   'DETECTOR'],
  ['verifier',   'VERIFIER'],
  ['pentest',    'PEN-TEST'],
  ['fixer',      'FIXER'],
];

// Maps a live stage id (from the `stage` SSE event) onto its burndown
// bucket so the active stage's bar highlights.
const _STAGE_TO_BUCKET = {
  comprehending: 'comprehend',
  planning:      'planner',
  lead_loop:     'lead',
  detecting:     'detector',
  fuzzing:       'detector',
  verifying:     'verifier',
  pen_testing:   'pentest',
  fixing:        'fixer',
};

function _paintBurndown(root) {
  const el = root.querySelector('#bf-burndown');
  if (!el) return;
  const cost = _liveState.cost;
  if (!cost || !cost.by_stage) {
    el.innerHTML = '<div class="bf-burn-empty">Token telemetry appears as each stage reports in…</div>';
    return;
  }
  const stageTotals = {};
  let maxStage = 1;
  for (const [key] of _BURN_STAGES) {
    const s = cost.by_stage[key];
    const t = s ? (s.tokens_in || 0) + (s.tokens_out || 0) : 0;
    stageTotals[key] = t;
    if (t > maxStage) maxStage = t;
  }
  const spent = (cost.total_in || 0) + (cost.total_out || 0);
  // Rough projection: scale current spend by chunk progress during detect.
  let projected = spent;
  if (_liveState.chunksTotal > 0 && _liveState.chunksDone > 0
      && _liveState.chunksDone < _liveState.chunksTotal) {
    projected = Math.round(spent * (_liveState.chunksTotal / _liveState.chunksDone));
  }
  const pct = projected > 0 ? Math.min(100, Math.round((spent / projected) * 100)) : 0;

  const activeBucket = _STAGE_TO_BUCKET[_liveState.currentStage] || '';
  const bars = _BURN_STAGES
    .filter(([key]) => stageTotals[key] > 0)
    .map(([key, label]) => {
      const t = stageTotals[key];
      const w = Math.max(2, Math.round((t / maxStage) * 100));
      const active = key === activeBucket;
      return `
        <div class="bf-burn-row${active ? ' bf-burn-active' : ''}">
          <span class="bf-burn-label">${label}</span>
          <span class="bf-burn-track"><span class="bf-burn-fill" style="width:${w}%"></span></span>
          <span class="bf-burn-val">${_fmtTok(t)}</span>
        </div>`;
    }).join('');

  el.innerHTML = `
    <div class="bf-burn-head">
      <span class="bf-burn-title">TOKEN BURNDOWN</span>
      <span class="bf-burn-total">${_fmtTok(spent)}${projected > spent ? ` <span class="bf-burn-proj">/ ~${_fmtTok(projected)} projected</span>` : ''}</span>
    </div>
    <div class="bf-burn-meter"><span class="bf-burn-meter-fill" style="width:${pct}%"></span></div>
    <div class="bf-burn-bars">${bars || '<div class="bf-burn-empty">No stage has reported spend yet…</div>'}</div>
  `;
}

// Coalesced to one repaint/frame. The EventSource fires many events/sec
// during active detection and each one rebuilt the counters + feed + findings
// via innerHTML; unbatched that pinned the main thread. All ~dozen call sites
// hit the coalesced wrapper below; the heavy work lives in _paintLiveDashboardNow.
const _paintLiveDashboard = rafCoalesce(() => _paintLiveDashboardNow());

function _paintLiveDashboardNow() {
  const root = document.getElementById('bf-detail');
  if (!root) return;
  // Phase progress
  const segs = root.querySelectorAll('.bf-progress-seg');
  if (segs.length) {
    const resolved = _STAGE_ALIAS[_liveState.currentStage] || _liveState.currentStage;
    let idx = _STAGE_PROGRESS.findIndex(s => s.stage === resolved);
    // Unknown / future stage: hold the last known position rather than
    // blanking every segment (which reads as the bar running backwards).
    if (idx < 0) idx = _liveState.lastStageIdx;
    else _liveState.lastStageIdx = idx;
    segs.forEach((seg, i) => {
      if (i < idx)        seg.dataset.state = 'done';
      else if (i === idx) seg.dataset.state = 'active';
      else                seg.dataset.state = 'pending';
    });
    const stageEl = root.querySelector('.bf-progress-stage');
    if (stageEl) {
      const label = _STAGE_PROGRESS[idx]?.label || _liveState.currentStage.toUpperCase();
      stageEl.textContent = label;
    }
  }

  // Counters panel
  const c = root.querySelector('#bf-live-counters');
  if (c) {
    c.innerHTML = `
      <div class="bf-counter">
        <span class="bf-counter-num">${_liveState.chunksDone}<span class="bf-counter-sep">/</span>${_liveState.chunksTotal || '—'}</span>
        <span class="bf-counter-label">CHUNKS</span>
      </div>
      <div class="bf-counter">
        <span class="bf-counter-num">${_liveState.findings.length}</span>
        <span class="bf-counter-label">FINDINGS</span>
      </div>
      <div class="bf-counter">
        <span class="bf-counter-num">${_fmtTok((_liveState.cost?.total_in || 0) + (_liveState.cost?.total_out || 0))}</span>
        <span class="bf-counter-label">TOKENS SPENT</span>
        <span class="bf-counter-sub">${_fmtTok(_liveState.cost?.total_in || 0)} in · ${_fmtTok(_liveState.cost?.total_out || 0)} out</span>
      </div>
      <div class="bf-counter">
        <span class="bf-counter-num">${escapeHtml(_liveState.currentChunkFn || '—')}</span>
        <span class="bf-counter-label">EXAMINING</span>
        ${(() => {
          const sus = _liveState.suspectByChunk[`${_liveState.currentChunkFile}::${_liveState.currentChunkFn}`];
          if (sus) return `<span class="bf-counter-sub">suspects ${escapeHtml(sus)}</span>`;
          return _liveState.currentChunkFile ? `<span class="bf-counter-sub">${escapeHtml(_liveState.currentChunkFile)}</span>` : '';
        })()}
      </div>
    `;
  }

  _paintBurndown(root);

  // Live activity feed — terminal-style, newest at the bottom. Stick to
  // the bottom only when the user is already there, so scrolling up to
  // read history isn't yanked back down by the next event.
  const f = root.querySelector('#bf-live-feed');
  if (f) {
    if (!_liveState.feed.length) {
      f.innerHTML = '<div class="bf-feed-empty">Connecting to the run… first events land here within a few seconds.</div>';
    } else {
      const atBottom = (f.scrollHeight - f.scrollTop - f.clientHeight) < 40;
      f.innerHTML = _liveState.feed.map(item => `
        <div class="bf-feed-row">
          <span class="bf-feed-time">${escapeHtml(item.ts)}</span>
          <span class="bf-feed-label">${escapeHtml(item.label)}</span>
          ${item.detail ? `<span class="bf-feed-detail">${escapeHtml(item.detail)}</span>` : ''}
        </div>
      `).join('');
      if (atBottom) f.scrollTop = f.scrollHeight;
    }
  }

  // Recent-findings ticker
  const tick = root.querySelector('#bf-live-findings');
  if (tick) {
    if (!_liveState.findings.length) {
      tick.innerHTML = `
        <div class="bf-tick-empty">
          No findings yet — the detector is reasoning over chunks. <br/>
          Findings will animate in as they're produced.
        </div>
      `;
    } else {
      const items = _liveState.findings.slice(-10).reverse();
      tick.innerHTML = items.map(d => {
        const mn = _signatureMnemonic(d.claim_signature);
        const sev = (d.severity || 'info').toUpperCase();
        return `
          <div class="bf-tick-row bf-sev-${escapeHtml(d.severity || 'info')}">
            <span class="bf-tick-mn">${escapeHtml(mn)} · ${escapeHtml(sev)}</span>
            <span class="bf-tick-loc">${escapeHtml(d.file || '')}:${escapeHtml(d.function || '')}</span>
            ${_confidenceChip(d)}
            <span class="bf-tick-claim">${escapeHtml((d.claim || '').slice(0, 140))}</span>
          </div>
        `;
      }).join('');
    }
  }
}

// Best-effort: query coder workspaces for the dropdown. The endpoint
// might not be present on every deploy — we degrade to a git-URL-only
// launcher when it 404s.
async function _fetchWorkspaces() {
  try {
    const data = await _api('/api/coder/workspaces');
    return Array.isArray(data.workspaces) ? data.workspaces : [];
  } catch (_) {
    return [];
  }
}

async function _fetchModels() {
  // /api/tags is the canonical aggregate endpoint (ollama-compatible, used
  // by chat/coder/settings). model-cache caches it for 5 min and strips
  // the a/ n/ p/ mode-prefixed variants we don't want in the dropdown.
  try {
    return await getModels();
  } catch (_) {
    return [];
  }
}


// --------------------------------------------------------------------------
// Render helpers
// --------------------------------------------------------------------------

function _relTime(unixSec) {
  if (!unixSec) return '—';
  const now = Date.now() / 1000;
  const delta = Math.max(0, now - Number(unixSec));
  if (delta < 60) return `${Math.round(delta)}s ago`;
  if (delta < 3600) return `${Math.round(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.round(delta / 3600)}h ago`;
  return `${Math.round(delta / 86400)}d ago`;
}

function _absTime(unixSec) {
  if (!unixSec) return '';
  return new Date(Number(unixSec) * 1000).toISOString().slice(0, 16).replace('T', ' ');
}

function _humanizeTarget(run) {
  if (run.git_url) {
    try {
      const u = new URL(run.git_url);
      const tail = u.pathname.replace(/\.git$/, '').replace(/^\//, '');
      return `${u.hostname}/${tail}`;
    } catch (_) {
      return run.git_url;
    }
  }
  if (run.workspace_id) return `ws://${run.workspace_id.slice(0, 12)}`;
  return '—';
}

// Confidence provenance chip — turns the raw confirm-counts into a
// glanceable signal: how many detector runs agreed, and how many distinct
// model families (the FP-killer — cross-family agreement breaks the
// correlated-error pattern). Status drives the colour class.
function _confidenceChip(d) {
  const status = (d.status || '').toLowerCase();
  const runs = d.runs_to_confirm, total = d.total_runs;
  const fams = d.families_to_confirm;
  const bits = [];
  if (typeof runs === 'number' && typeof total === 'number' && total > 0) {
    bits.push(`${runs}/${total} runs`);
  }
  if (typeof fams === 'number' && fams >= 2) bits.push(`${fams} fam`);
  const label = status
    ? status.replace(/_/g, ' ')
    : (bits.length ? 'detected' : '');
  if (!label && !bits.length) return '';
  const cls = ['confirmed', 'fixed'].includes(status) ? 'bf-conf-strong'
    : status === 'speculative' ? 'bf-conf-weak' : 'bf-conf-mid';
  return `<span class="bf-conf ${cls}">${escapeHtml(label)}${bits.length ? ` · ${escapeHtml(bits.join(' · '))}` : ''}</span>`;
}

function _signatureMnemonic(signature) {
  const m = {
    null_deref:           'NULL',
    bounds_check:         'BNDS',
    race:                 'RACE',
    use_after_free:       'UAF',
    injection:            'INJ',
    missing_validation:   'VLD',
    resource_leak:        'LEAK',
    deadlock:             'DLCK',
    auth_bypass:          'AUTH',
    logic_error:          'LOGIC',
    type_confusion:       'TYPE',
    other:                'OTHR',
  };
  return m[signature] || 'OTHR';
}

function _renderFindingId(rawId) {
  // fnd_abcd1234efgh5678 → FND-ABCD·1234 (instrument-style)
  const hex = (rawId || '').replace(/^fnd_/, '').toUpperCase();
  if (!hex) return 'FND-····';
  return `FND-${hex.slice(0, 4)}·${hex.slice(4, 8)}`;
}

function _renderPips(runs_to_confirm, total_runs) {
  // 3 filled, 0 empty: 3◉/3◉ — but separator goes between groups.
  const filled = Math.max(0, Math.min(Number(runs_to_confirm) || 0, total_runs || 0));
  const total = Math.max(filled, Number(total_runs) || 0) || 1;
  const empty = total - filled;
  const filledHtml = '<span class="bf-fpip-filled">'
    + '◉'.repeat(filled) + '</span>';
  const emptyHtml = empty > 0
    ? '<span class="bf-fpip-empty">' + '○'.repeat(empty) + '</span>'
    : '';
  return `<span class="bf-fpips">${filledHtml}${emptyHtml}<span class="bf-fpip-sep"> ${filled}/${total}</span></span>`;
}


// --------------------------------------------------------------------------
// Rail (run history)
// --------------------------------------------------------------------------

function _renderRail() {
  const list = document.getElementById('bf-rail-list');
  if (!list) return;
  if (!_state.runs.length) {
    if (_state.railSig === 'empty') return;
    _state.railSig = 'empty';
    list.innerHTML = `
      <div class="bf-rail-empty">
        <div class="bf-ready-label">READY</div>
        <div>No runs in history. Press <strong>NEW RUN</strong> to begin.</div>
      </div>
    `;
    return;
  }
  // Skip the rebuild when nothing visible changed — otherwise the rail
  // flashes on every 5s poll. The signature captures everything a card
  // renders (id, status, counts, relative time, active selection).
  const sig = _state.currentRunId + '|' + _state.runs.map(r =>
    `${r.run_id}:${r.stop_reason || 'running'}:${r.findings_total || 0}:`
    + `${r.findings_fixed || 0}:${r.findings_confirmed || 0}:${r.findings_fix_failed || 0}:`
    + _relTime(r.started_at)
  ).join(',');
  if (sig === _state.railSig) return;
  _state.railSig = sig;
  list.innerHTML = _state.runs.map(_renderRailCard).join('');
  list.querySelectorAll('.bf-card').forEach(el => {
    el.addEventListener('click', () => {
      const id = el.dataset.runId || '';
      if (id) selectRun(id);
    });
  });
}

function _renderRailCard(run) {
  const target = escapeHtml(_humanizeTarget(run));
  const time = _relTime(run.started_at);
  const status = run.stop_reason || 'running';
  const active = run.run_id === _state.currentRunId ? 'bf-card-active' : '';
  const counts = [];
  if (run.findings_fixed)     counts.push(`<span class="bf-count-fixed">${run.findings_fixed} FIXED</span>`);
  if (run.findings_confirmed) counts.push(`<span class="bf-count-confirmed">${run.findings_confirmed} CONF</span>`);
  if (run.findings_fix_failed) counts.push(`<span class="bf-count-failed">${run.findings_fix_failed} FAIL</span>`);
  if (!counts.length)         counts.push(`<span>${run.findings_total || 0} TOTAL</span>`);
  return `
    <button class="bf-card ${active}" data-run-id="${escapeHtml(run.run_id || '')}">
      <div class="bf-card-line1">
        <span class="bf-card-target">${target}</span>
        <span class="bf-card-time" title="${escapeHtml(_absTime(run.started_at))}">${escapeHtml(time)}</span>
      </div>
      <div class="bf-card-line2">
        <span class="bf-card-counts">${counts.join('')}</span>
        <span class="bf-status-pill bf-status-${escapeHtml(status)}">${escapeHtml(status).toUpperCase()}</span>
      </div>
    </button>
  `;
}


// --------------------------------------------------------------------------
// Detail view (right side)
// --------------------------------------------------------------------------

function _renderDetailEmpty() {
  const root = document.getElementById('bf-detail');
  if (!root) return;
  _state.runningShellRunId = '';
  root.innerHTML = `
    <div class="bf-detail-empty">
      <div class="bf-ready-label">READY</div>
      <div class="bf-empty-hint">Select a run from the history rail, or start a new one.</div>
    </div>
  `;
}

function _renderDetailRunning(run) {
  const root = document.getElementById('bf-detail');
  if (!root) return;
  const runId = run.run_id || '';
  // Already showing this run's live shell? Repaint in place and bail —
  // NEVER rebuild the DOM or reset live state on a poll. Rebuilding every
  // 5s was wiping the activity log back to "Awaiting first event…",
  // flashing the panel, and snapping the stage label back to PREPARING.
  if (_state.runningShellRunId === runId && root.querySelector('#bf-live-dashboard')) {
    _subscribeLive(runId);   // no-op if already subscribed; reconnects if dropped
    _paintLiveDashboard();
    return;
  }
  _state.runningShellRunId = runId;
  _resetLiveState(runId);
  const target = escapeHtml(_humanizeTarget(run));
  const segments = _STAGE_PROGRESS.map(({ label, stage }) => `
    <div class="bf-progress-seg" data-stage="${stage}" title="${label}"></div>
  `).join('');
  root.innerHTML = `
    <div class="bf-detail-header">
      <button class="bf-back-to-rail" type="button" aria-label="Back to runs">‹ RUNS</button>
      <div class="bf-detail-id">RUN ${escapeHtml(run.run_id || '')}</div>
      <h2 class="bf-detail-title">${target}</h2>
      <div class="bf-detail-meta">
        <span>STARTED <strong>${escapeHtml(_absTime(run.started_at))}</strong></span>
        <span class="bf-detail-meta-sep">·</span>
        <span class="bf-status-pill bf-status-running">RUNNING</span>
        <button class="bf-stop-btn" id="bf-stop-btn" data-run-id="${escapeHtml(run.run_id || '')}">STOP RUN</button>
      </div>
      <div class="bf-progress">${segments}</div>
      <div class="bf-progress-stage">CONNECTING…</div>
    </div>

    <div class="bf-live-dashboard" id="bf-live-dashboard">
      <div class="bf-live-counters" id="bf-live-counters"></div>

      <div class="bf-burndown" id="bf-burndown"></div>

      <div class="bf-live-grid">
        <section class="bf-live-feed-panel">
          <div class="bf-live-panel-head">
            <span class="bf-live-panel-title">LIVE ACTIVITY</span>
            <span class="bf-live-panel-hint">live log · newest at bottom</span>
          </div>
          <div class="bf-live-feed" id="bf-live-feed">
            <div class="bf-feed-empty">Awaiting first event…</div>
          </div>
        </section>

        <section class="bf-live-findings-panel">
          <div class="bf-live-panel-head">
            <span class="bf-live-panel-title">FINDINGS AS THEY LAND</span>
            <span class="bf-live-panel-hint">animated in real time</span>
          </div>
          <div class="bf-live-findings" id="bf-live-findings">
            <div class="bf-tick-empty">
              No findings yet — the detector is reasoning over chunks.
            </div>
          </div>
        </section>
      </div>

      <div class="bf-live-explainer">
        <strong>WHAT YOU'RE SEEING:</strong> the agent walks each chunk the
        planner picked, runs the detector against it, and the verifier
        re-attempts a repro to confirm. Comprehension (when populated)
        gives the planner a structural map so it focuses attention.
        Findings appear here the moment they confirm; the final report
        opens in this panel once the run terminates.
      </div>
    </div>
  `;
  _paintLiveDashboard();
  _subscribeLive(run.run_id);
  root.querySelector('#bf-stop-btn')?.addEventListener('click', async (ev) => {
    const btn = ev.currentTarget;
    const runId = btn.dataset.runId || '';
    if (!runId) return;
    btn.disabled = true;
    btn.textContent = 'STOPPING…';
    try {
      const result = await _cancelRun(runId);
      if (result?.cancelled) {
        showToast('Cancel signalled — run will finalize shortly.', 'success');
      } else {
        showToast(`Run already ${result?.status || 'terminal'}; nothing to cancel.`, 'info');
      }
      await refresh();
    } catch (err) {
      showToast(`Cancel failed: ${err.message}`, 'error');
      btn.disabled = false;
      btn.textContent = 'STOP RUN';
    }
  });
}

function _renderDetailReport(run, report) {
  const root = document.getElementById('bf-detail');
  if (!root) return;
  _state.runningShellRunId = '';
  const findings = (report?.findings || []);
  const fixedCount = findings.filter(f => f.status === 'fixed').length;
  const confCount = findings.filter(f => f.status === 'confirmed').length;
  const patchCount = findings.filter(f => (f.patch || '').trim()).length;
  const cost = (report?.cost_ledger || []);
  const containment = run.containment_warning || report?.containment_warning || '';
  const containmentBanner = containment
    ? `<div class="bf-containment-banner">
         <span class="bf-containment-banner-label">CONTAINMENT</span>
         <span>${escapeHtml(containment)}</span>
       </div>`
    : '';
  const stopReason = (run.stop_reason || 'complete');
  root.innerHTML = `
    ${containmentBanner}
    <div class="bf-detail-header">
      <button class="bf-back-to-rail" type="button" aria-label="Back to runs">‹ RUNS</button>
      <div class="bf-detail-id">RUN ${escapeHtml(run.run_id || '')}</div>
      <h2 class="bf-detail-title">${escapeHtml(_humanizeTarget(run))}</h2>
      <div class="bf-detail-meta">
        <span>STARTED <strong>${escapeHtml(_absTime(run.started_at))}</strong></span>
        <span class="bf-detail-meta-sep">·</span>
        <span>DURATION <strong>${_renderDuration(run)}</strong></span>
        <span class="bf-detail-meta-sep">·</span>
        <span class="bf-status-pill bf-status-${escapeHtml(stopReason)}">${escapeHtml(stopReason).toUpperCase()}</span>
      </div>
    </div>
    <div class="bf-tabs">
      ${_renderTab('findings', 'FINDINGS', findings.length)}
      ${_renderTab('patches', 'PATCHES', patchCount)}
      ${_renderTab('cost', 'COST', cost.length)}
      ${_renderTab('baseline', 'BASELINE', '')}
    </div>
    <div class="bf-tab-panel" id="bf-tab-panel"></div>
  `;
  root.querySelectorAll('.bf-tab').forEach(el => {
    el.addEventListener('click', () => {
      _state.currentTab = el.dataset.tab;
      _renderTabPanel(report);
      root.querySelectorAll('.bf-tab').forEach(t => {
        t.classList.toggle('bf-tab-active', t.dataset.tab === _state.currentTab);
      });
    });
  });
  _renderTabPanel(report);
}

function _renderDuration(run) {
  if (!run.completed_at || !run.started_at) return '—';
  const ms = (Number(run.completed_at) - Number(run.started_at)) * 1000;
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60_000)}m ${Math.round((ms % 60_000) / 1000)}s`;
}

function _renderTab(id, label, count) {
  const active = _state.currentTab === id ? 'bf-tab-active' : '';
  const countHtml = count === '' || count == null
    ? ''
    : `<span class="bf-tab-count">${escapeHtml(String(count))}</span>`;
  return `<button class="bf-tab ${active}" data-tab="${id}">${label}${countHtml}</button>`;
}

function _renderTabPanel(report) {
  const panel = document.getElementById('bf-tab-panel');
  if (!panel) return;
  switch (_state.currentTab) {
    case 'patches':  panel.innerHTML = _renderPatchesTab(report); break;
    case 'cost':     panel.innerHTML = _renderCostTab(report); break;
    case 'baseline': panel.innerHTML = _renderBaselineTab(report); break;
    case 'findings':
    default:         _renderFindingsTab(panel, report); return;
  }
  _wireTabHandlers(panel);
}


// --------------------------------------------------------------------------
// Findings tab — the centerpiece
// --------------------------------------------------------------------------

function _renderFindingsTab(panel, report) {
  const findings = (report?.findings || []);
  const hist = report?.confirmation_hist || {};
  panel.innerHTML = `
    <div class="bf-filter-row" id="bf-filters">
      <button class="bf-filter-chip" data-filter="minConfirm" data-value="3">3/3 ONLY</button>
      <button class="bf-filter-chip" data-filter="minConfirm" data-value="2">2+/3</button>
      <button class="bf-filter-chip" data-filter="status" data-value="fixed">FIXED</button>
      <button class="bf-filter-chip" data-filter="status" data-value="confirmed">CONFIRMED</button>
      <button class="bf-filter-chip" data-filter="status" data-value="fix_failed">FIX FAILED</button>
      <button class="bf-filter-chip" data-filter="severity" data-value="critical">CRIT</button>
      <button class="bf-filter-chip" data-filter="severity" data-value="high">HIGH</button>
      <button class="bf-filter-chip" data-filter="severity" data-value="medium">MED</button>
      <button class="bf-filter-chip bf-filter-passive">
        VARIANCE: ${escapeHtml(Object.entries(hist).map(([k, v]) => `${k}=${v}`).join('  ·  ') || 'n/a')}
      </button>
    </div>
    <div id="bf-findings-body"></div>
  `;
  _renderActiveFilterChips(panel);
  panel.querySelectorAll('[data-filter]').forEach(el => {
    if (el.classList.contains('bf-filter-passive')) return;
    el.addEventListener('click', () => _toggleFilter(el.dataset.filter, el.dataset.value));
  });
  _paintFindings(panel, findings);
}

function _renderActiveFilterChips(panel) {
  panel.querySelectorAll('[data-filter]').forEach(el => {
    if (el.classList.contains('bf-filter-passive')) return;
    const filter = el.dataset.filter;
    const value = el.dataset.value;
    let active = false;
    if (filter === 'minConfirm') {
      active = _state.filters.minConfirm === Number(value);
    } else if (filter === 'status') {
      active = _state.filters.statuses.has(value);
    } else if (filter === 'severity') {
      active = _state.filters.severities.has(value);
    }
    el.classList.toggle('bf-filter-active', active);
  });
}

function _toggleFilter(filter, value) {
  if (filter === 'minConfirm') {
    _state.filters.minConfirm = _state.filters.minConfirm === Number(value) ? 0 : Number(value);
  } else if (filter === 'status') {
    if (_state.filters.statuses.has(value)) _state.filters.statuses.delete(value);
    else _state.filters.statuses.add(value);
  } else if (filter === 'severity') {
    if (_state.filters.severities.has(value)) _state.filters.severities.delete(value);
    else _state.filters.severities.add(value);
  }
  const panel = document.getElementById('bf-tab-panel');
  if (!panel) return;
  _renderActiveFilterChips(panel);
  _paintFindings(panel, (_state.currentReport?.findings || []));
}

function _paintFindings(panel, findings) {
  const body = panel.querySelector('#bf-findings-body');
  if (!body) return;
  const filtered = findings.filter(f => {
    if (_state.filters.minConfirm > 0) {
      const rc = Number(f.runs_to_confirm) || 0;
      const tr = Number(f.total_runs) || 0;
      const minRequired = (_state.filters.minConfirm === 3)
        ? tr  // "3/3 only" — must equal total
        : _state.filters.minConfirm;
      if (rc < minRequired) return false;
    }
    if (_state.filters.statuses.size > 0 && !_state.filters.statuses.has(f.status)) return false;
    if (_state.filters.severities.size > 0 && !_state.filters.severities.has(f.severity)) return false;
    return true;
  });
  if (!filtered.length) {
    body.innerHTML = `
      <div class="bf-detail-empty" style="padding-top:60px;">
        <div class="bf-ready-label">FILTERED</div>
        <div class="bf-empty-hint">No findings match the active filters.</div>
      </div>
    `;
    return;
  }
  // Group by status
  const byStatus = {};
  for (const f of filtered) {
    (byStatus[f.status] || (byStatus[f.status] = [])).push(f);
  }
  let html = '';
  for (const status of _STATUS_ORDER) {
    const group = byStatus[status];
    if (!group || !group.length) continue;
    html += `
      <div class="bf-section-head">
        <span class="bf-section-label bf-sl-${escapeHtml(status)}">${escapeHtml(status).toUpperCase()}</span>
        <span class="bf-section-count">${group.length}</span>
        <span class="bf-section-rule"></span>
      </div>
    `;
    for (const f of group) html += _renderFinding(f);
  }
  body.innerHTML = html;
  body.querySelectorAll('[data-finding-toggle]').forEach(head => {
    head.addEventListener('click', () => {
      const card = head.closest('.bf-finding');
      const body = card.querySelector('.bf-finding-body');
      const open = !card.classList.contains('bf-finding-open');
      card.classList.toggle('bf-finding-open', open);
      if (body) body.style.display = open ? 'block' : 'none';
    });
  });
}

function _renderFinding(f) {
  const mnemonic = _signatureMnemonic(f.claim_signature);
  const sevLabel = (f.severity || 'info').toUpperCase();
  const idLabel = _renderFindingId(f.id);
  const pips = _renderPips(f.runs_to_confirm, f.total_runs);
  const status = f.status || 'speculative';
  const evidence = (f.evidence_paths || []);
  const evidenceHtml = evidence.length
    ? `<div class="bf-finding-loc">${evidence.map(p =>
        `<span><strong>${escapeHtml(p)}</strong></span>`).join('')}</div>`
    : '';
  const meta = [];
  if (f.suggested_repro) meta.push(['Suggested repro', f.suggested_repro]);
  if (f.repro_command)   meta.push(['Repro command', f.repro_command]);
  if (f.repro_path)      meta.push(['Repro path', f.repro_path]);
  if (f.invariant)       meta.push(['Invariant', f.invariant]);
  if (f.repro_output)    meta.push(['Verifier evidence', f.repro_output]);
  const metaHtml = meta.length
    ? `<dl class="bf-finding-meta-block">${meta.map(([k, v]) =>
        `<dt>${escapeHtml(k)}</dt><dd>${escapeHtml(v)}</dd>`).join('')}</dl>`
    : '';
  const notes = (f.notes || []);
  const notesHtml = notes.length
    ? `<ul class="bf-finding-notes">${notes.map(n =>
        `<li>${escapeHtml(n)}</li>`).join('')}</ul>`
    : '';
  return `
    <div class="bf-finding">
      <div class="bf-finding-head" data-finding-toggle>
        <span class="bf-fmnemonic bf-sev-${escapeHtml(f.severity || 'info')}">
          ${escapeHtml(mnemonic)}·<span class="bf-fm-sev">${escapeHtml(sevLabel)}</span>
        </span>
        <span class="bf-fid">${escapeHtml(idLabel)}</span>
        ${pips}
        <span class="bf-fstatus bf-fs-${escapeHtml(status)}">${escapeHtml(status).replace('_', ' ').toUpperCase()}</span>
      </div>
      <div class="bf-finding-body" style="display:none;">
        <p class="bf-finding-claim">${escapeHtml(f.claim || '(no claim)')}</p>
        <div class="bf-finding-loc">
          <span>FILE <strong>${escapeHtml(f.file || '')}</strong></span>
          <span>FN <strong>${escapeHtml(f.function || '')}</strong></span>
        </div>
        ${evidenceHtml}
        ${metaHtml}
        ${notesHtml}
      </div>
    </div>
  `;
}


// --------------------------------------------------------------------------
// Patches tab
// --------------------------------------------------------------------------

function _renderPatchesTab(report) {
  const findings = (report?.findings || []).filter(f => (f.patch || '').trim());
  if (!findings.length) {
    return `
      <div class="bf-detail-empty" style="padding-top:60px;">
        <div class="bf-ready-label">NO PATCHES</div>
        <div class="bf-empty-hint">This run produced no accepted patches.</div>
      </div>
    `;
  }
  return findings.map(f => `
    <div class="bf-patch-card" data-patch-id="${escapeHtml(f.id)}">
      <div class="bf-patch-head">
        <span class="bf-patch-title">${escapeHtml(f.file || '')} — ${escapeHtml(f.function || '')}</span>
        <button class="bf-patch-copy" data-patch="${escapeHtml(f.id)}">COPY DIFF</button>
      </div>
      <pre class="bf-patch-diff">${_colorizeDiff(f.patch || '')}</pre>
    </div>
  `).join('');
}

function _colorizeDiff(diff) {
  // Lightweight diff colorization — runs after escapeHtml so user
  // content can't break out. We escape first, then wrap +/- lines
  // with span classes.
  const escaped = escapeHtml(diff);
  return escaped.split('\n').map(line => {
    if (line.startsWith('+++') || line.startsWith('---')) {
      return `<span class="bf-diff-hunk">${line}</span>`;
    }
    if (line.startsWith('@@')) return `<span class="bf-diff-hunk">${line}</span>`;
    if (line.startsWith('+'))  return `<span class="bf-diff-add">${line}</span>`;
    if (line.startsWith('-'))  return `<span class="bf-diff-del">${line}</span>`;
    return line;
  }).join('\n');
}


// --------------------------------------------------------------------------
// Cost tab
// --------------------------------------------------------------------------

function _renderCostTab(report) {
  const ledger = report?.cost_ledger || [];
  if (!ledger.length) {
    return `
      <div class="bf-detail-empty" style="padding-top:60px;">
        <div class="bf-empty-hint">No cost-ledger entries recorded.</div>
      </div>
    `;
  }
  const rows = ledger.map(e => {
    const stopClass = `bf-cost-stop-${escapeHtml(e.stop_reason || '')}`;
    return `
      <tr>
        <td>${escapeHtml(e.stage || '')}</td>
        <td>${escapeHtml(e.role || '')}</td>
        <td>${escapeHtml(e.model || '')}</td>
        <td class="bf-cost-num">${Number(e.iterations || 0).toLocaleString()}</td>
        <td class="bf-cost-num">${Number(e.tokens_in || 0).toLocaleString()}</td>
        <td class="bf-cost-num">${Number(e.tokens_out || 0).toLocaleString()}</td>
        <td class="bf-cost-num">${(Number(e.wallclock_ms || 0) / 1000).toFixed(2)}s</td>
        <td class="${stopClass}">${escapeHtml(e.stop_reason || '')}</td>
      </tr>
    `;
  }).join('');
  const tot_in  = ledger.reduce((a, e) => a + Number(e.tokens_in || 0), 0);
  const tot_out = ledger.reduce((a, e) => a + Number(e.tokens_out || 0), 0);
  const tot_ms  = ledger.reduce((a, e) => a + Number(e.wallclock_ms || 0), 0);
  return `
    <div class="bf-cost-scroll">
      <table class="bf-cost-table">
        <thead>
          <tr>
            <th>STAGE</th>
            <th>ROLE</th>
            <th>MODEL</th>
            <th>ITER</th>
            <th>TOK IN</th>
            <th>TOK OUT</th>
            <th>WALL</th>
            <th>STOP</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
        <tfoot>
          <tr>
            <td colspan="4">TOTAL</td>
            <td>${tot_in.toLocaleString()}</td>
            <td>${tot_out.toLocaleString()}</td>
            <td>${(tot_ms / 1000).toFixed(2)}s</td>
            <td></td>
          </tr>
        </tfoot>
      </table>
    </div>
  `;
}


// --------------------------------------------------------------------------
// Baseline tab
// --------------------------------------------------------------------------

function _renderBaselineTab(report) {
  const b = report?.baseline || {};
  const intake = report?.intake || {};
  const notes = (b.notes || []);
  const notesHtml = notes.length
    ? `<dd><ul class="bf-finding-notes">${notes.map(n => `<li>${escapeHtml(n)}</li>`).join('')}</ul></dd>`
    : '<dd>—</dd>';
  return `
    <dl class="bf-baseline">
      <dt>Intake</dt>
      <dd>${intake.git_url ? `git_url: ${escapeHtml(intake.git_url)}` : `workspace_id: ${escapeHtml(intake.workspace_id || '')}`}</dd>
      ${intake.focus_paths && intake.focus_paths.length
        ? `<dt>Focus paths</dt><dd>${(intake.focus_paths || []).map(p => escapeHtml(p)).join('  ·  ')}</dd>`
        : ''}
      <dt>Detected language</dt>
      <dd>${escapeHtml(b.detected_language || '(unknown)')}</dd>
      <dt>Test command</dt>
      <dd>${escapeHtml(b.test_command || '(none detected)')}</dd>
      <dt>Notes</dt>
      ${notesHtml}
      ${b.baseline_test_stdout
        ? `<dt>Baseline test output</dt><dd><pre class="bf-baseline-stdout">${escapeHtml(b.baseline_test_stdout)}</pre></dd>`
        : ''}
    </dl>
  `;
}


// --------------------------------------------------------------------------
// Tab wiring (copy buttons, etc.)
// --------------------------------------------------------------------------

function _wireTabHandlers(panel) {
  panel.querySelectorAll('.bf-patch-copy').forEach(btn => {
    btn.addEventListener('click', async () => {
      const card = btn.closest('.bf-patch-card');
      const pre = card?.querySelector('.bf-patch-diff');
      const text = pre ? pre.textContent : '';
      try {
        await navigator.clipboard.writeText(text);
        btn.classList.add('bf-copied');
        btn.textContent = 'COPIED';
        setTimeout(() => {
          btn.classList.remove('bf-copied');
          btn.textContent = 'COPY DIFF';
        }, 1400);
      } catch (_) {
        showToast('Copy failed — your browser blocked clipboard access', 'error');
      }
    });
  });
}


// --------------------------------------------------------------------------
// Run launcher modal
// --------------------------------------------------------------------------

async function _openLauncher(prefill = null) {
  _renderShell();
  const modal = document.getElementById('bf-modal');
  if (!modal) return;
  modal.classList.remove('hidden');
  const errorEl = modal.querySelector('#bf-form-error');
  if (errorEl) errorEl.style.display = 'none';

  const defaults = prefill || _state.launchDefaults || {};
  const [workspaces, models] = await Promise.all([
    _fetchWorkspaces(),
    _fetchModels(),
  ]);
  _populateLauncherDropdowns(workspaces, models, defaults);
}

function _closeLauncher() {
  const modal = document.getElementById('bf-modal');
  if (modal) modal.classList.add('hidden');
}

// Depth presets bundle the token-spend levers so the common case is one
// click. Each writes concrete values into the Advanced inputs (which stay
// visible + editable); touching any Advanced field flips the selection to
// "custom" so what-runs always matches what's-shown.
const _DEPTH_PRESETS = {
  quick:    { runs: 1, maxChunks: 15, concurrency: 4, wallclockMin: 10, fixAttempts: 2, fuzz: false, pentest: false },
  standard: { runs: 3, maxChunks: 40, concurrency: 4, wallclockMin: 30, fixAttempts: 3, fuzz: true,  pentest: false },
  deep:     { runs: 5, maxChunks: 80, concurrency: 6, wallclockMin: 60, fixAttempts: 3, fuzz: true,  pentest: false },
};

function _bfNum(id, fallback) {
  const v = parseInt(document.getElementById(id)?.value, 10);
  return Number.isFinite(v) ? v : fallback;
}

// Rough projection so the user can trade depth for spend BEFORE committing.
// The detector dominates: ~6k tokens per pass (in+out), plus a fixed
// overhead band for comprehension (first run) + planner + verify/fix.
// Deliberately labelled "rough" — real spend rides on chunk size + model.
function _bfRecomputeEstimate() {
  const el = document.getElementById('bf-estimate');
  if (!el) return;
  const runs = _bfNum('bf-adv-runs', 3);
  const maxChunks = _bfNum('bf-adv-maxchunks', 40);
  const passes = Math.max(0, runs) * Math.max(0, maxChunks);
  const tok = passes * 6000 + 50000;
  const tokLabel = tok >= 1000 ? `~${Math.round(tok / 1000)}k` : `${tok}`;
  el.innerHTML =
    `<span class="bf-est-num">${passes}</span> detector passes · `
    + `<span class="bf-est-num">${tokLabel}</span> tokens `
    + `<span class="bf-est-soft">(rough — actual rides on chunk size + model)</span>`;
}

function _bfSetMode(mode) {
  document.querySelectorAll('#bf-mode .bf-seg').forEach(b => {
    const on = b.dataset.mode === mode;
    b.dataset.active = on ? 'true' : 'false';
    b.setAttribute('aria-checked', on ? 'true' : 'false');
  });
  const goal = document.getElementById('bf-goal-group');
  if (goal) goal.style.display = mode === 'named-bug' ? '' : 'none';
  const hint = document.getElementById('bf-mode-hint');
  if (hint) {
    hint.textContent = mode === 'named-bug'
      ? 'Hunt one described defect. The lead agent drives a focused investigation loop.'
      : 'Survey the whole workspace for defects. The planner picks where to look.';
  }
}

function _bfCurrentMode() {
  return document.querySelector('#bf-mode .bf-seg[data-active="true"]')?.dataset.mode || 'explore';
}

function _bfSetPreset(preset, { applyValues = true } = {}) {
  document.querySelectorAll('#bf-depth .bf-seg').forEach(b => {
    const on = b.dataset.preset === preset;
    b.dataset.active = on ? 'true' : 'false';
    b.setAttribute('aria-checked', on ? 'true' : 'false');
  });
  const p = _DEPTH_PRESETS[preset];
  if (applyValues && p) {
    const set = (id, v) => { const e = document.getElementById(id); if (e) e.value = v; };
    const chk = (id, v) => { const e = document.getElementById(id); if (e) e.checked = v; };
    set('bf-adv-runs', p.runs);
    set('bf-adv-maxchunks', p.maxChunks);
    set('bf-adv-concurrency', p.concurrency);
    set('bf-adv-wallclock', p.wallclockMin);
    set('bf-adv-fixattempts', p.fixAttempts);
    chk('bf-adv-fuzz', p.fuzz);
    chk('bf-adv-pentest', p.pentest);
  }
  _bfRecomputeEstimate();
}

function _bfCurrentPreset() {
  return document.querySelector('#bf-depth .bf-seg[data-active="true"]')?.dataset.preset || 'standard';
}

// User hand-edited an Advanced field → the active preset no longer
// describes the config. Flip to "custom" without clobbering their values.
function _bfMarkCustom() {
  if (_bfCurrentPreset() !== 'custom') _bfSetPreset('custom', { applyValues: false });
  else _bfRecomputeEstimate();
}

// Persist launcher choices per-workspace (device-local UI preference, like
// the other launcher chrome). The run + report stay server-side.
function _bfPrefsKey(workspaceId) {
  return `bf.launch.${workspaceId || 'default'}`;
}
function _bfSavePrefs(workspaceId, prefs) {
  try { localStorage.setItem(_bfPrefsKey(workspaceId), JSON.stringify(prefs)); } catch (_) { /* quota / private mode */ }
}
function _bfLoadPrefs(workspaceId) {
  try { return JSON.parse(localStorage.getItem(_bfPrefsKey(workspaceId)) || 'null') || {}; } catch (_) { return {}; }
}

function _populateLauncherDropdowns(workspaces, models, defaults = {}) {
  const wsSelect = document.getElementById('bf-workspace-select');
  if (wsSelect) {
    if (!workspaces.length) {
      wsSelect.innerHTML = '<option value="">(no workspaces — create one in Coder first)</option>';
      wsSelect.disabled = true;
    } else {
      wsSelect.disabled = false;
      wsSelect.innerHTML = '<option value="">— pick workspace —</option>'
        + workspaces.map(w => {
          const id = w.id || w.workspace_id || '';
          const name = w.name || id;
          return `<option value="${escapeHtml(id)}">${escapeHtml(name)}</option>`;
        }).join('');
      if (defaults.workspaceId) wsSelect.value = defaults.workspaceId;
    }
  }

  const modelNames = models.map(m =>
    typeof m === 'string' ? m : (m.id || m.name || ''),
  ).filter(Boolean);
  _populateModelField('bf-model-primary', modelNames, defaults.primaryModel || '');
  _populateModelField('bf-model-verifier', modelNames, defaults.verifierModel || '', {
    emptyLabel: '— (none — single-model self-verification) —',
  });

  const focusEl = document.getElementById('bf-focus-paths');
  if (focusEl) {
    focusEl.value = Array.isArray(defaults.focusPaths)
      ? defaults.focusPaths.join('\n')
      : (defaults.focusPaths || '');
  }
  const threatEl = document.getElementById('bf-threat-model');
  if (threatEl) threatEl.value = defaults.threatModel || '';

  // Restore per-workspace launcher prefs (mode + depth + advanced knobs),
  // falling back to the Standard preset on first use for this workspace.
  const prefs = _bfLoadPrefs(defaults.workspaceId || (document.getElementById('bf-workspace-select')?.value || ''));
  _bfSetMode(prefs.mode || 'explore');
  const goalEl = document.getElementById('bf-goal-desc');
  if (goalEl) goalEl.value = prefs.goalDescription || '';
  const preset = prefs.preset || 'standard';
  if (preset === 'custom' && prefs.advanced) {
    _bfSetPreset('custom', { applyValues: false });
    const set = (id, v) => { const e = document.getElementById(id); if (e != null && v != null) e.value = v; };
    const chk = (id, v) => { const e = document.getElementById(id); if (e) e.checked = !!v; };
    const a = prefs.advanced;
    set('bf-adv-runs', a.runs); set('bf-adv-maxchunks', a.maxChunks);
    set('bf-adv-concurrency', a.concurrency); set('bf-adv-wallclock', a.wallclockMin);
    set('bf-adv-fixattempts', a.fixAttempts); set('bf-adv-severity', a.severityFloor);
    chk('bf-adv-fuzz', a.fuzz); chk('bf-adv-pentest', a.pentest); chk('bf-adv-thinking', a.thinking);
    const ens = document.getElementById('bf-adv-ensemble');
    if (ens) ens.value = Array.isArray(a.ensemble) ? a.ensemble.join('\n') : (a.ensemble || '');
    _bfRecomputeEstimate();
  } else {
    _bfSetPreset(preset in _DEPTH_PRESETS ? preset : 'standard');
  }
}

function _populateModelField(id, modelNames, selected, opts = {}) {
  const el = document.getElementById(id);
  if (!el) return;
  if (!modelNames.length) {
    _swapToTextInput(el, id, selected);
    return;
  }
  const emptyLabel = opts.emptyLabel || '';
  const emptyOpt = emptyLabel ? `<option value="">${escapeHtml(emptyLabel)}</option>` : '';
  const options = modelNames.map(n =>
    `<option value="${escapeHtml(n)}">${escapeHtml(n)}</option>`,
  ).join('');
  el.innerHTML = emptyOpt + options;
  if (selected && modelNames.includes(selected)) {
    el.value = selected;
  } else if (emptyLabel) {
    el.value = '';
  }
}

function _swapToTextInput(selectEl, id, value = '') {
  const input = document.createElement('input');
  input.type = 'text';
  input.className = 'bf-form-input';
  input.id = id;
  input.placeholder = 'e.g. claude-opus-4-7';
  input.value = value;
  selectEl.replaceWith(input);
}

function _readLauncherForm() {
  const workspaceId = document.getElementById('bf-workspace-select')?.value?.trim() || '';
  const primaryModel = document.getElementById('bf-model-primary')?.value?.trim() || '';
  const verifierModel = document.getElementById('bf-model-verifier')?.value?.trim() || '';
  const focusPaths = (document.getElementById('bf-focus-paths')?.value || '')
    .split('\n').map(s => s.trim()).filter(Boolean);
  const threatModel = (document.getElementById('bf-threat-model')?.value || '').trim();
  const mode = _bfCurrentMode();
  const goalDescription = (document.getElementById('bf-goal-desc')?.value || '').trim();
  const ensemble = (document.getElementById('bf-adv-ensemble')?.value || '')
    .split('\n').map(s => s.trim()).filter(Boolean);
  const advanced = {
    runs: _bfNum('bf-adv-runs', 3),
    maxChunks: _bfNum('bf-adv-maxchunks', 40),
    concurrency: _bfNum('bf-adv-concurrency', 4),
    wallclockMin: _bfNum('bf-adv-wallclock', 30),
    fixAttempts: _bfNum('bf-adv-fixattempts', 3),
    severityFloor: document.getElementById('bf-adv-severity')?.value || 'info',
    fuzz: !!document.getElementById('bf-adv-fuzz')?.checked,
    pentest: !!document.getElementById('bf-adv-pentest')?.checked,
    thinking: !!document.getElementById('bf-adv-thinking')?.checked,
    ensemble,
  };
  return {
    workspaceId, primaryModel, verifierModel, focusPaths, threatModel,
    mode, goalDescription, preset: _bfCurrentPreset(), advanced,
  };
}

function _showFormError(msg) {
  const el = document.getElementById('bf-form-error');
  if (!el) return;
  el.textContent = msg;
  el.style.display = '';
}

async function _submitLauncher(ev) {
  ev?.preventDefault?.();
  const form = _readLauncherForm();

  if (!form.workspaceId) {
    _showFormError('Pick a workspace to audit.');
    return;
  }
  if (!form.primaryModel) {
    _showFormError('Pick a primary model — planner / detector / fixer all use this.');
    return;
  }

  if (form.mode === 'named-bug' && !form.goalDescription) {
    _showFormError('Describe the bug to hunt for — or switch to Explore mode.');
    return;
  }

  const a = form.advanced;
  const payload = {
    workspace_id: form.workspaceId,
    primary_model: form.primaryModel,
    verifier_model: form.verifierModel,
    focus_paths: form.focusPaths,
    threat_model: form.threatModel,
    // Token-budget levers (route + job handler already decode these).
    detector_runs_per_chunk: a.runs,
    detector_concurrency: a.concurrency,
    max_chunks: a.maxChunks,
    max_fix_attempts_per_finding: a.fixAttempts,
    overall_wallclock_seconds: Math.max(1, a.wallclockMin) * 60,
    enable_fuzz_leg: a.fuzz,
    enable_pen_test_leg: a.pentest,
    detector_enable_thinking: a.thinking || null,
    detector_models: a.ensemble,
    user_goal: {
      mode: form.mode,
      description: form.mode === 'named-bug' ? form.goalDescription : '',
      severity_floor: a.severityFloor || 'info',
    },
  };
  const forceBox = document.getElementById('bf-force-below-min');
  if (forceBox?.checked) payload.force_below_minimum = true;

  // Remember these choices for next time on this workspace.
  _bfSavePrefs(form.workspaceId, {
    mode: form.mode,
    goalDescription: form.goalDescription,
    preset: form.preset,
    advanced: a,
  });

  const submitBtn = document.getElementById('bf-submit-btn');
  if (submitBtn) {
    submitBtn.disabled = true;
    submitBtn.textContent = 'STARTING…';
  }
  try {
    const result = await _createRun(payload);
    showToast(`Run ${result.run_id} queued`, 'success');
    _closeLauncher();
    await refresh();
    if (result.run_id) selectRun(result.run_id);
    _ensurePolling();
  } catch (err) {
    // Capability-gate refusal: surface the override option instead of
    // a generic error so the user can opt in if they know what they're
    // doing. The route returns 422 with a structured detail object.
    if (err?.detail?.error_code === 'primary_model_below_minimum') {
      const group = document.getElementById('bf-force-group');
      if (group) group.style.display = '';
      _showFormError(
        `${err.detail.message}\n\nRecommended floor: ${err.detail.recommended_floor}`,
      );
    } else {
      _showFormError(err.message || 'Failed to start run');
    }
  } finally {
    if (submitBtn) {
      submitBtn.disabled = false;
      submitBtn.textContent = 'START RUN';
    }
  }
}


// --------------------------------------------------------------------------
// Polling
// --------------------------------------------------------------------------

function _anyRunning() {
  return _state.runs.some(r => (r.stop_reason || 'running') === 'running');
}

function _ensurePolling() {
  if (_state.pollTimer) return;
  if (!_state.active) return;
  if (!_anyRunning()) return;
  _state.pollTimer = setInterval(async () => {
    if (!_state.active) {
      _stopPolling();
      return;
    }
    await refresh();
    if (!_anyRunning()) _stopPolling();
  }, 5000);
}

function _stopPolling() {
  if (_state.pollTimer) {
    clearInterval(_state.pollTimer);
    _state.pollTimer = null;
  }
}


// --------------------------------------------------------------------------
// Public API
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Shell HTML (self-mounted on first activate so index.html only needs the
// outer host div + the script tag)
// --------------------------------------------------------------------------

const _SHELL_HTML = `
  <aside class="bf-rail">
    <div class="bf-rail-header">
      <span class="bf-rail-title">BUG FINDER · RUNS</span>
      <div class="bf-rail-header-actions">
        <button class="bf-new-run-btn" id="bf-new-run-btn" type="button">+ NEW RUN</button>
        <button class="bf-rail-close" id="bf-close-btn" type="button" title="Close">✕</button>
      </div>
    </div>
    <div class="bf-rail-list" id="bf-rail-list"></div>
  </aside>
  <section class="bf-detail" id="bf-detail"></section>

  <div class="modal-overlay hidden" id="bf-modal" role="dialog" aria-modal="true" aria-labelledby="bf-modal-title">
    <div class="bf-modal-card">
      <div class="bf-modal-head">
        <span class="bf-modal-title" id="bf-modal-title">NEW BUG-FINDER RUN</span>
        <button class="bf-modal-close" id="bf-modal-close" type="button" aria-label="Close">✕</button>
      </div>
      <form class="bf-form" id="bf-form">
        <div class="bf-form-error" id="bf-form-error" style="display:none"></div>

        <div class="bf-form-group">
          <label class="bf-form-label" for="bf-workspace-select">Workspace</label>
          <select class="bf-form-select" id="bf-workspace-select"></select>
          <div class="bf-form-hint">The coder workspace to audit. Create one in Coder if this list is empty.</div>
        </div>

        <div class="bf-form-group">
          <label class="bf-form-label" for="bf-model-primary">Primary model</label>
          <select class="bf-form-select" id="bf-model-primary"></select>
          <div class="bf-form-hint">Drives planner, detector, and fixer. Capable instruction-followers recommended (Claude 4.x / GPT-5.x / Qwen 3.5+ / DeepSeek V3.x).</div>
        </div>

        <div class="bf-form-group">
          <label class="bf-form-label" for="bf-model-verifier">Verifier model <span class="bf-label-hint">— optional</span></label>
          <select class="bf-form-select" id="bf-model-verifier"></select>
          <div class="bf-form-hint">Different verifier reduces correlated-error false positives. Leave blank for single-model self-verification (the default on local hardware).</div>
        </div>

        <div class="bf-form-group">
          <label class="bf-form-label">Mode</label>
          <div class="bf-segmented" id="bf-mode" role="radiogroup" aria-label="Audit mode">
            <button type="button" class="bf-seg" data-mode="explore" data-active="true" role="radio" aria-checked="true">Explore</button>
            <button type="button" class="bf-seg" data-mode="named-bug" role="radio" aria-checked="false">Find a specific bug</button>
          </div>
          <div class="bf-form-hint" id="bf-mode-hint">Survey the whole workspace for defects. The planner picks where to look.</div>
        </div>

        <div class="bf-form-group" id="bf-goal-group" style="display:none">
          <label class="bf-form-label" for="bf-goal-desc">What should it hunt for?</label>
          <textarea class="bf-form-textarea" id="bf-goal-desc" rows="2" placeholder="e.g. a path-traversal in the file download handler; users can read other users' notes"></textarea>
          <div class="bf-form-hint">Routes through the lead agent's dynamic investigation loop instead of the static sweep.</div>
        </div>

        <div class="bf-form-group">
          <label class="bf-form-label">Depth <span class="bf-label-hint">— value per token</span></label>
          <div class="bf-segmented" id="bf-depth" role="radiogroup" aria-label="Audit depth">
            <button type="button" class="bf-seg" data-preset="quick" role="radio" aria-checked="false">Quick</button>
            <button type="button" class="bf-seg" data-preset="standard" data-active="true" role="radio" aria-checked="true">Standard</button>
            <button type="button" class="bf-seg" data-preset="deep" role="radio" aria-checked="false">Deep</button>
            <button type="button" class="bf-seg" data-preset="custom" role="radio" aria-checked="false">Custom</button>
          </div>
          <div class="bf-estimate" id="bf-estimate"></div>
        </div>

        <div class="bf-form-group">
          <label class="bf-form-label" for="bf-focus-paths">Focus paths <span class="bf-label-hint">— one per line, optional</span></label>
          <textarea class="bf-form-textarea" id="bf-focus-paths" rows="3" placeholder="src/auth/&#10;lib/parsers/"></textarea>
          <div class="bf-form-hint">Restricts planner attention. Empty = survey the whole repo.</div>
        </div>

        <div class="bf-form-group">
          <label class="bf-form-label" for="bf-threat-model">Threat model <span class="bf-label-hint">— markdown, optional</span></label>
          <textarea class="bf-form-textarea" id="bf-threat-model" rows="5" placeholder="### Assets&#10;### Trust boundaries&#10;### Attacker capabilities&#10;### In scope&#10;### Out of scope"></textarea>
          <div class="bf-form-hint">Prepended to detector + verifier prompts. Closes the #1 source of valid-but-rejected findings.</div>
        </div>

        <details class="bf-advanced" id="bf-advanced">
          <summary class="bf-advanced-summary">Advanced — token budget &amp; legs</summary>
          <div class="bf-advanced-body">
            <div class="bf-adv-grid">
              <label class="bf-adv-field">
                <span class="bf-adv-label">Detector runs / chunk</span>
                <input type="number" class="bf-form-input" id="bf-adv-runs" min="1" max="9" step="1" />
                <span class="bf-adv-note">Each run = one full detector pass. Linear token cost.</span>
              </label>
              <label class="bf-adv-field">
                <span class="bf-adv-label">Max chunks</span>
                <input type="number" class="bf-form-input" id="bf-adv-maxchunks" min="1" max="400" step="1" />
                <span class="bf-adv-note">Cap on code sites examined.</span>
              </label>
              <label class="bf-adv-field">
                <span class="bf-adv-label">Concurrency</span>
                <input type="number" class="bf-form-input" id="bf-adv-concurrency" min="1" max="16" step="1" />
                <span class="bf-adv-note">Parallel detectors. Speed, not token cost.</span>
              </label>
              <label class="bf-adv-field">
                <span class="bf-adv-label">Wallclock cap (min)</span>
                <input type="number" class="bf-form-input" id="bf-adv-wallclock" min="1" max="240" step="1" />
                <span class="bf-adv-note">Hard ceiling — returns whatever landed.</span>
              </label>
              <label class="bf-adv-field">
                <span class="bf-adv-label">Max fix attempts</span>
                <input type="number" class="bf-form-input" id="bf-adv-fixattempts" min="0" max="9" step="1" />
                <span class="bf-adv-note">Fix-loop retries per confirmed finding.</span>
              </label>
              <label class="bf-adv-field">
                <span class="bf-adv-label">Severity floor</span>
                <select class="bf-form-select" id="bf-adv-severity">
                  <option value="info">info — report everything</option>
                  <option value="low">low</option>
                  <option value="medium">medium</option>
                  <option value="high">high</option>
                  <option value="critical">critical only</option>
                </select>
                <span class="bf-adv-note">Drops findings below this rank.</span>
              </label>
            </div>

            <div class="bf-adv-toggles">
              <label class="bf-checkbox-label"><input type="checkbox" id="bf-adv-fuzz" /><span>Fuzz leg <span class="bf-label-hint">— atheris crash-finding on fuzzable chunks</span></span></label>
              <label class="bf-checkbox-label"><input type="checkbox" id="bf-adv-pentest" /><span>Pen-test leg <span class="bf-label-hint">— boots the app + active HTTP probes (opt-in, heavier)</span></span></label>
              <label class="bf-checkbox-label"><input type="checkbox" id="bf-adv-thinking" /><span>Detector thinking <span class="bf-label-hint">— chain-of-thought on reasoning models</span></span></label>
            </div>

            <div class="bf-form-group">
              <label class="bf-form-label" for="bf-adv-ensemble">Detector ensemble <span class="bf-label-hint">— one model id per line, optional</span></label>
              <textarea class="bf-form-textarea" id="bf-adv-ensemble" rows="2" placeholder="Qwen3.6-35B-A3B-IQ4_XS&#10;GLM-4.7-Flash-UD-Q4_K_XL"></textarea>
              <div class="bf-form-hint">Round-robins detector runs across these. Findings flagged by 2+ model families earn a stronger confidence signal. Empty = use the primary model every run.</div>
            </div>
          </div>
        </details>

        <div class="bf-form-group bf-form-group-inline" id="bf-force-group" style="display:none">
          <label class="bf-checkbox-label">
            <input type="checkbox" id="bf-force-below-min" />
            <span>Force run with below-floor primary model</span>
          </label>
          <div class="bf-form-hint">The selected model is below the bug-finder capability floor. The prompts expect capable instruction-followers; below-floor models tend to produce malformed JSON that yields zero findings. Override only if you know the model is up to the task.</div>
        </div>

        <div class="bf-form-actions">
          <button type="button" class="bf-btn-secondary" id="bf-cancel-btn">CANCEL</button>
          <button type="submit" class="bf-btn-primary" id="bf-submit-btn">START RUN</button>
        </div>
      </form>
    </div>
  </div>
`;

function _ensureHost() {
  let host = document.getElementById('bug-finder-surface');
  if (host) return host;
  host = document.createElement('div');
  host.id = 'bug-finder-surface';
  host.className = 'bug-finder-surface hidden';
  host.dataset.bfPane = 'rail';
  document.body.appendChild(host);
  return host;
}

// Single-pane mobile shell: the CSS hides whichever pane doesn't match
// `data-bf-pane`. Desktop ignores the attribute and shows both. Helpers
// here just flip the attribute; the actual render stays unchanged.
function _setPane(pane) {
  const host = document.getElementById('bug-finder-surface');
  if (host) host.dataset.bfPane = pane;
}

function _backToRail() {
  _state.currentRunId = '';
  _state.currentReport = null;
  _closeLive();
  document.querySelectorAll('.bf-card').forEach(el => el.classList.remove('bf-card-active'));
  _renderDetailEmpty();
  _setPane('rail');
}

function _renderShell() {
  if (_state.shellMounted) return;
  const host = _ensureHost();
  host.innerHTML = _SHELL_HTML;
  _state.shellMounted = true;
  _wireShellEvents();
  _renderDetailEmpty();
}

function _wireShellEvents() {
  document.getElementById('bf-new-run-btn')?.addEventListener('click', () => _openLauncher());
  document.getElementById('bf-close-btn')?.addEventListener('click', deactivate);
  document.getElementById('bf-modal-close')?.addEventListener('click', _closeLauncher);
  document.getElementById('bf-cancel-btn')?.addEventListener('click', _closeLauncher);
  document.getElementById('bf-modal')?.addEventListener('click', (e) => {
    if (e.target.id === 'bf-modal') _closeLauncher();
  });
  document.getElementById('bf-form')?.addEventListener('submit', _submitLauncher);

  // Launcher controls (mounted once with the shell). Mode + depth are
  // segmented button groups; Advanced edits flip the depth to "custom".
  document.getElementById('bf-mode')?.addEventListener('click', (e) => {
    const seg = e.target.closest('.bf-seg');
    if (seg) _bfSetMode(seg.dataset.mode);
  });
  document.getElementById('bf-depth')?.addEventListener('click', (e) => {
    const seg = e.target.closest('.bf-seg');
    if (!seg) return;
    if (seg.dataset.preset === 'custom') _bfSetPreset('custom', { applyValues: false });
    else _bfSetPreset(seg.dataset.preset);
  });
  ['bf-adv-runs', 'bf-adv-maxchunks', 'bf-adv-concurrency', 'bf-adv-wallclock',
   'bf-adv-fixattempts', 'bf-adv-fuzz', 'bf-adv-pentest'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', _bfMarkCustom);
    document.getElementById(id)?.addEventListener('change', _bfMarkCustom);
  });

  // Back-to-rail is rendered into every detail-header template; one
  // delegated listener handles all of them so we don't rebind on each
  // render cycle.
  document.getElementById('bug-finder-surface')?.addEventListener('click', (e) => {
    if (e.target.closest?.('.bf-back-to-rail')) _backToRail();
  });
}


async function refresh() {
  if (_state.fetchInFlight) return;
  _state.fetchInFlight = true;
  try {
    _state.runs = await _fetchRuns();
  } catch (err) {
    if (err.status === 401) return;
    showToast(`Bug Finder: ${err.message}`, 'error');
  } finally {
    _state.fetchInFlight = false;
  }
  _renderRail();
  if (_state.currentRunId) {
    const row = _state.runs.find(r => r.run_id === _state.currentRunId);
    if (row) {
      if ((row.stop_reason || 'running') === 'running') {
        _state.currentReport = null;
        _renderDetailRunning(row);
      } else if (!_state.currentReport) {
        await selectRun(_state.currentRunId);  // hydrate report once it lands
      }
    }
  }
}

async function selectRun(runId) {
  _state.currentRunId = runId;
  const row = _state.runs.find(r => r.run_id === runId);
  document.querySelectorAll('.bf-card').forEach(el => {
    el.classList.toggle('bf-card-active', el.dataset.runId === runId);
  });
  if (!row) {
    _renderDetailEmpty();
    return;
  }
  // On mobile, swap to the detail pane. Desktop ignores the attribute.
  _setPane('detail');
  if ((row.stop_reason || 'running') === 'running') {
    _state.currentReport = null;
    _renderDetailRunning(row);
    return;
  }
  try {
    const full = await _fetchRun(runId);
    _state.currentReport = full?.report || null;
    _renderDetailReport(full, _state.currentReport);
  } catch (err) {
    showToast(`Failed to load run: ${err.message}`, 'error');
    _renderDetailEmpty();
  }
}

function activate(opts = {}) {
  _renderShell();
  _ensureSurfaceListener();
  _collapseLeftPanelForBugFinder();
  const host = _ensureHost();
  host.classList.remove('hidden');
  // Default to the rail pane on activate. Mobile users see the run
  // list first; selecting a card flips to detail. Desktop ignores
  // this attribute via the CSS media-query gate.
  if (!host.dataset.bfPane) host.dataset.bfPane = 'rail';
  document.body.setAttribute('data-bug-finder-open', '1');
  if (opts.launchDefaults) _state.launchDefaults = opts.launchDefaults;
  _state.active = true;
  refresh().then(() => _ensurePolling());
  if (opts.openLauncher) _openLauncher(opts.launchDefaults || null);
}

// Collapse the coder left panel when Bug Finder opens — its two-column
// grid (rail + detail) already saturates the viewport, and the file tree
// underneath isn't reachable while the overlay is up. We modify the DOM
// directly rather than calling app.js's closePanel() so the user's
// persistent panel preference (localStorage) isn't overwritten.
function _collapseLeftPanelForBugFinder() {
  try {
    const leftPanel = document.querySelector('.left-panel');
    const appEl = document.getElementById('app');
    if (!leftPanel) return;
    if (window.matchMedia('(min-width: 768px)').matches) {
      leftPanel.classList.add('desktop-collapsed');
      if (appEl) appEl.setAttribute('data-panel', 'hidden');
    } else {
      // Mobile: the panel is a slide-over with .open + a backdrop. Close
      // both so the BF surface is the only thing on screen.
      leftPanel.classList.remove('open');
      document.getElementById('panel-backdrop')?.classList.remove('visible');
      document.body.style.overflow = '';
    }
  } catch { /* tolerate missing chrome — BF still works without the dismiss */ }
}

// Bug Finder is a coder-mode overlay (position:absolute, inset:0 on body).
// Without a focus listener it stays painted across every surface the user
// navigates to while a run is active. Tie its visible lifetime to coder.
let _surfaceListenerWired = false;
function _ensureSurfaceListener() {
  if (_surfaceListenerWired) return;
  _surfaceListenerWired = true;
  document.addEventListener('surface:focus-changed', (e) => {
    if (!_state.active) return;
    if (e?.detail?.mode !== 'coder') deactivate();
  });
}

function deactivate() {
  _state.active = false;
  _stopPolling();
  _closeLive();
  const host = document.getElementById('bug-finder-surface');
  if (host) {
    host.classList.add('hidden');
    // Reset to rail so the next activate lands users on the run list.
    host.dataset.bfPane = 'rail';
  }
  document.body.removeAttribute('data-bug-finder-open');
}

function init() {
  if (_state.initialized) return;
  _state.initialized = true;
  // Lazy shell mount on first activate — index.html no longer needs to
  // host the launcher DOM. ``init`` exists only as a stable export so
  // module pre-warmers can fire it without effects.
}

/**
 * Programmatic entry point for surfaces that already know the workspace.
 * Coder calls this when the user opens a kind=bug_finder workspace or
 * clicks the Audit button.
 *
 *   launchForWorkspace({
 *     workspaceId: 'ws_abc',
 *     primaryModel: 'claude-opus-4-7',
 *     verifierModel: '',          // optional — falls back to heavyweight slot
 *     focusPaths: [],              // optional
 *     threatModel: '',             // optional
 *     openLauncher: true,          // open the launcher modal pre-filled
 *   })
 */
function launchForWorkspace(opts = {}) {
  activate({
    launchDefaults: {
      workspaceId: opts.workspaceId || '',
      primaryModel: opts.primaryModel || '',
      verifierModel: opts.verifierModel || '',
      focusPaths: opts.focusPaths || [],
      threatModel: opts.threatModel || '',
    },
    openLauncher: opts.openLauncher !== false,
  });
}

export { activate, deactivate, init, refresh, launchForWorkspace };
