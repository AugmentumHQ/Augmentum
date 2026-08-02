/* ==========================================================================
   Coder Mode — Progress & Transparency
   Renders a thin progress strip directly under the coder status line so the
   user can see what the model is actually doing between actions:
     - model_load / model_swap     → "Loading model · <name>" + indeterminate bar
     - slot_restore                → "Restoring session" + indeterminate bar
     - prefill                     → "Preparing context · NN% · ~tok/s" with
                                     real progress (poll /api/engine/v2/prefill_progress)
     - thinking / responding       → "Reasoning · XXXX tok at NN tok/s · 0:14"
                                     (live elapsed + accumulated tokens)
   Pre-fix the only signal here was text labels — "waiting for model…" with
   no indication whether something was actually happening for 30s+.

   Mounted into #coder-status (the status bar in the input area). All DOM
   is created lazily on first event so coder mode has zero impact on
   non-coder boot.
   ========================================================================== */

const PREFILL_POLL_INTERVAL_MS = 500;
const LOAD_POLL_INTERVAL_MS = 500;
const TPS_UPDATE_INTERVAL_MS = 250;

const _state = {
  /** The #coder-status DOM root we attach to. Resolved on first event. */
  hostEl: null,
  /** The bar wrapper. Null when no stage is active. */
  barEl: null,
  /** Stage event id currently driving the bar (so out-of-order
   *  stage_complete for a SUPERSEDED stage doesn't kill the current bar). */
  activeStageId: '',
  /** Active stage name — also used by the prefill poll loop. */
  activeStage: '',
  /** Prefill poll timer. */
  prefillIntervalId: null,
  /** Model-load poll timer (model_load / model_swap stages). */
  loadIntervalId: null,
  /** Token-rate tracker (set during streaming phases). */
  tokens: 0,
  tokenStartTs: 0,
  tpsIntervalId: null,
  /** Status detail text we last wrote so we don't fight an unrelated writer. */
  detailOwnedByProgress: false,
};

function _ensureHost() {
  if (_state.hostEl && document.body.contains(_state.hostEl)) return _state.hostEl;
  // Home is the cooperative row above the input — next to the auto
  // (plan-mode) badge and the think toggle. A FIXED position: always
  // visible next to where the user's controls are, and it doesn't
  // rotate between tool-call / reasoning blocks the way an in-
  // conversation mount did. The old #coder-status placement clipped
  // the label on narrow screens; it remains as a fallback host for
  // layouts without the composer row.
  _state.hostEl = (
    document.getElementById('coder-coop-row')
    || document.getElementById('coder-status')
    || null
  );
  return _state.hostEl;
}

function _ensureBar() {
  const host = _ensureHost();
  if (!host) return null;
  if (_state.barEl && host.contains(_state.barEl)) return _state.barEl;
  const bar = document.createElement('div');
  bar.className = 'coder-progress-bar';
  if (host.id === 'coder-coop-row') bar.classList.add('in-coop-row');
  bar.innerHTML = (
    '<div class="coder-progress-track">'
    + '<div class="coder-progress-fill"></div>'
    + '</div>'
    + '<span class="coder-progress-label"></span>'
  );
  host.appendChild(bar);
  _state.barEl = bar;
  return bar;
}

function _removeBar() {
  if (_state.barEl && _state.barEl.parentNode) {
    _state.barEl.parentNode.removeChild(_state.barEl);
  }
  _state.barEl = null;
}

function _setBar({ percent, label, indeterminate = false }) {
  const bar = _ensureBar();
  if (!bar) return;
  bar.dataset.indeterminate = indeterminate ? '1' : '0';
  const fill = bar.querySelector('.coder-progress-fill');
  if (fill) {
    if (indeterminate) {
      fill.style.width = '100%';
    } else {
      const pct = Math.max(0, Math.min(100, Math.round(percent || 0)));
      fill.style.width = `${pct}%`;
    }
  }
  const labelEl = bar.querySelector('.coder-progress-label');
  if (labelEl) labelEl.textContent = label || '';
}

// ─── Prefill polling ───────────────────────────────────────────────────────

function _fmtTok(n) {
  if (!n || n <= 0) return '';
  return n >= 10_000 ? `${Math.round(n / 1000)}k` : String(n);
}

async function _pollPrefillTick() {
  try {
    const resp = await fetch('/api/engine/v2/prefill_progress', {
      credentials: 'same-origin',
    });
    if (!resp.ok) return;
    const data = await resp.json();
    if (!data || !data.active) return;
    const progress = data.progress ?? 0;
    const tps = data.tps ?? 0;
    const done = data.tokens_done ?? 0;
    const pct = Math.round(progress * 100);
    // All derived from the engine's own print_timing line — genuine
    // numbers, not animation. total = done/progress; ETA from the
    // measured prefill tok/s. This is the "processing 100k tokens
    // after a 10-file batch read" visibility: the user sees size,
    // rate, and time remaining instead of a frozen label.
    const parts = [`Preparing context · ${pct}%`];
    if (done > 0 && progress > 0.01) {
      const total = Math.round(done / progress);
      parts.push(`${_fmtTok(done)}/${_fmtTok(total)} tok`);
      if (tps > 0) {
        const etaS = Math.round((total - done) / tps);
        if (etaS >= 3) {
          const m = Math.floor(etaS / 60);
          const s = etaS % 60;
          parts.push(m > 0 ? `~${m}:${String(s).padStart(2, '0')} left` : `~${s}s left`);
        }
      }
    }
    if (tps > 0) parts.push(`${Math.round(tps)} tok/s`);
    _setBar({
      percent: pct,
      label: parts.join(' · '),
      indeterminate: false,
    });
  } catch {
    // Network blip — next tick retries.
  }
}

function _startPrefillPolling() {
  if (_state.prefillIntervalId !== null) return;
  _pollPrefillTick();  // first tick immediately
  _state.prefillIntervalId = setInterval(_pollPrefillTick, PREFILL_POLL_INTERVAL_MS);
}

function _stopPrefillPolling() {
  if (_state.prefillIntervalId !== null) {
    clearInterval(_state.prefillIntervalId);
    _state.prefillIntervalId = null;
  }
}

// ─── Model-load polling ──────────────────────────────────────────────────────
// During a cold model load / swap the HTTP response is silent for 30-120s.
// The engine seeds a snapshot (elapsed + an expected_s from recent loads)
// at /api/engine/v2/load_progress — the SAME endpoint chat polls. Reading
// it here turns the featureless indeterminate bar into "Loading <model> ·
// 14s of ~30s" with an honest, 95%-capped fill. Mirror of
// chat/load-progress.js — keep the two in lockstep.

function _shortModel(id) {
  if (!id) return '';
  // Snapshots may carry a path or a bare name; show the basename, trimmed.
  const base = String(id).split(/[\\/]/).pop() || '';
  return base.length > 40 ? `${base.slice(0, 39)}…` : base;
}

function _loadLabel(stageLabel, modelId, elapsedS, expectedS) {
  const name = _shortModel(modelId);
  const head = name ? `Loading ${name}` : (stageLabel || 'Loading model');
  const elapsed = Math.round(elapsedS || 0);
  const expected = Math.round(expectedS || 0);
  if (expected > 0) return `${head} · ${elapsed}s of ~${expected}s`;
  if (elapsed > 0) return `${head} · ${elapsed}s`;
  return head;
}

async function _pollLoadTick() {
  try {
    const resp = await fetch('/api/engine/v2/load_progress', {
      credentials: 'same-origin',
    });
    if (!resp.ok) return;  // 401/500 — silently skip; next tick retries
    const data = await resp.json();
    if (!data || !data.active) return;
    // Backend caps progress at 95% until the model is READY — an honest
    // "still going" signal, so we never show a full bar mid-load.
    const pct = Math.round((data.progress ?? 0) * 100);
    _setBar({
      percent: pct,
      label: _loadLabel(data.stage_label, data.model_id, data.elapsed_s, data.expected_s),
      indeterminate: false,
    });
  } catch {
    // Network blip — next tick retries.
  }
}

function _startLoadPolling() {
  if (_state.loadIntervalId !== null) return;
  _pollLoadTick();  // first tick immediately so the bar appears fast
  _state.loadIntervalId = setInterval(_pollLoadTick, LOAD_POLL_INTERVAL_MS);
}

function _stopLoadPolling() {
  if (_state.loadIntervalId !== null) {
    clearInterval(_state.loadIntervalId);
    _state.loadIntervalId = null;
  }
}

// ─── Token-rate tracker during streaming ───────────────────────────────────

function _fmtElapsed(ms) {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return m > 0 ? `${m}:${String(rem).padStart(2, '0')}` : `${s}s`;
}

function _tpsTick(stageLabel) {
  if (_state.tokenStartTs === 0) return;
  const elapsedMs = Date.now() - _state.tokenStartTs;
  if (elapsedMs <= 0) return;
  const tps = (_state.tokens / elapsedMs) * 1000;
  const tpsStr = tps > 0 ? ` · ${tps.toFixed(1)} tok/s` : '';
  const tokStr = _state.tokens > 0 ? ` · ${_state.tokens} tok` : '';
  _setBar({
    percent: 100,
    label: `${stageLabel}${tokStr}${tpsStr} · ${_fmtElapsed(elapsedMs)}`,
    indeterminate: true,
  });
}

function _startTokenRateTracker(stageLabel) {
  _state.tokens = 0;
  _state.tokenStartTs = Date.now();
  if (_state.tpsIntervalId !== null) clearInterval(_state.tpsIntervalId);
  _state.tpsIntervalId = setInterval(() => _tpsTick(stageLabel), TPS_UPDATE_INTERVAL_MS);
  _tpsTick(stageLabel);  // first tick immediately
}

function _stopTokenRateTracker() {
  if (_state.tpsIntervalId !== null) {
    clearInterval(_state.tpsIntervalId);
    _state.tpsIntervalId = null;
  }
  _state.tokens = 0;
  _state.tokenStartTs = 0;
}

// ─── Public API ────────────────────────────────────────────────────────────

/** Called from coder.js onStage callback when a stage_start / stage_complete
 *  / stage_progress event arrives from the backend. */
export function handleStageEvent(ev) {
  if (!ev) return;
  if (ev.type === 'start') {
    _stopTokenRateTracker();
    _stopLoadPolling();  // tear down any prior load poll before a new stage
    _state.activeStage = ev.stage;
    _state.activeStageId = ev.id || '';
    const label = ev.label || ev.stage || '';
    const text = ev.detail ? `${label} · ${ev.detail}` : label;
    if (ev.stage === 'prefill') {
      // Indeterminate seed bar until the first poll lands real progress.
      _setBar({ percent: 5, label: text, indeterminate: false });
      _startPrefillPolling();
    } else if (ev.stage === 'model_load' || ev.stage === 'model_swap') {
      // Seed an indeterminate bar so something shows within ~16ms, then
      // poll the load snapshot for real elapsed/expected progress
      // (parity with chat — "Loading <model> · 14s of ~30s").
      _setBar({ percent: 5, label: text, indeterminate: true });
      _startLoadPolling();
    } else {
      // slot_restore (and any other slow stage with no progress snapshot)
      // — indeterminate animated bar so the user can tell it isn't frozen.
      _setBar({ percent: 100, label: text, indeterminate: true });
    }
  } else if (ev.type === 'progress') {
    // Optional mid-stage update. Some stages may emit percent; honor it.
    if (typeof ev.percent === 'number') {
      _setBar({
        percent: ev.percent,
        label: ev.message || _state.activeStage,
        indeterminate: false,
      });
    }
  } else if (ev.type === 'complete') {
    // Ignore stale completes (out-of-order arrival from a superseded stage).
    if (ev.id && _state.activeStageId && ev.id !== _state.activeStageId) return;
    if (ev.stage === 'prefill') _stopPrefillPolling();
    else if (ev.stage === 'model_load' || ev.stage === 'model_swap') _stopLoadPolling();
    _state.activeStage = '';
    _state.activeStageId = '';
    // Don't remove the bar here — the next stage_start or the streaming
    // tracker may pick it up. notifyAllClear() handles teardown.
  }
}

/** Called from coder.js onStatus when a 'thinking' / 'responding' arrives,
 *  so the bar shows a live elapsed timer + tok/s during the generation
 *  phase. Idempotent — switching phases just updates the label. */
export function startStreamingTracker(label) {
  _stopPrefillPolling();
  _stopLoadPolling();
  _startTokenRateTracker(label);
}

/** Called from coder.js onContent / onThinking so the tracker can count
 *  the streamed tokens (estimated by whitespace-split — exact count
 *  doesn't matter here, the user-facing value is "is it moving"). */
export function recordStreamedDelta(text) {
  if (!text || _state.tokenStartTs === 0) return;
  // Rough word-ish token estimate. Same scale browsers use for typing-speed
  // indicators — accurate enough for the on-screen tps indicator. Not
  // used for billing or budget logic.
  _state.tokens += Math.max(1, Math.round(text.length / 4));
}

/** Tear everything down. Called when the agent run ends (onComplete /
 *  onError / abort). */
export function notifyAllClear() {
  _stopPrefillPolling();
  _stopLoadPolling();
  _stopTokenRateTracker();
  _state.activeStage = '';
  _state.activeStageId = '';
  _removeBar();
}
