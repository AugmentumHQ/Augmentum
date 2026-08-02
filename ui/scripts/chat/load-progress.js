/* ==========================================================================
   Chat Module — Model Load Progress Poller
   While llama-server is loading a model (cold start or swap), the HTTP
   chat response is silent until the load completes — that wait can run
   30-120s depending on model size + disk + GPU. The manager seeds a
   snapshot with an ``expected_s`` derived from recent successful loads
   (or a coarse file-size heuristic on first load) and exposes it at
   /api/engine/v2/load_progress.

   This module polls that endpoint during the model_load stage and feeds
   each snapshot into the active renderer's setLoadProgress() — the
   user sees "Loading deepseek-v3 · 14s of ~30s" with a soft progress
   bar instead of the panic-inducing "stream stalled" banner that used
   to fire here.

   Started by chat/index.js when stage_start.stage === 'model_load';
   stopped on stage_complete, first content delta, or stream error.
   Same pattern as prefill-progress.js — keep changes in lockstep.
   ========================================================================== */

const POLL_INTERVAL_MS = 500;

let _intervalId = null;
let _renderer = null;
let _model = '';

async function _tick() {
  if (!_renderer) return;
  try {
    // Pass the active model so a load INTO the secondary slot is watched on
    // its own engine, not the primary's (idle) snapshot.
    const q = _model ? `?model=${encodeURIComponent(_model)}` : '';
    const resp = await fetch(`/api/engine/v2/load_progress${q}`, {
      credentials: 'same-origin',
    });
    if (!resp.ok) return;  // 401/500 etc — silently skip; next tick retries
    const data = await resp.json();
    if (!data || !data.active) return;
    _renderer.setLoadProgress({
      model_id: data.model_id || '',
      progress: data.progress ?? 0,
      elapsed_s: data.elapsed_s ?? 0,
      expected_s: data.expected_s ?? 0,
      stage_label: data.stage_label || 'Loading model',
    });
  } catch {
    // Network blip — next tick will try again. No need to break the
    // poll loop on a single transient failure.
  }
}

/** Begin polling. Idempotent — safe to call repeatedly; later calls
 *  replace the renderer target without restarting the timer. */
export function startLoadPolling(renderer, model = '') {
  if (!renderer) return;
  _renderer = renderer;
  _model = model || '';
  if (_intervalId !== null) return;
  // Fire one tick immediately so the bar appears within ~50ms of stage
  // start rather than waiting for the first interval.
  _tick();
  _intervalId = setInterval(_tick, POLL_INTERVAL_MS);
}

/** Stop polling and clear the bar from the renderer's streaming
 *  indicator. Idempotent. */
export function stopLoadPolling() {
  if (_intervalId !== null) {
    clearInterval(_intervalId);
    _intervalId = null;
  }
  if (_renderer && typeof _renderer.clearLoadProgress === 'function') {
    _renderer.clearLoadProgress();
  }
  _renderer = null;
  _model = '';
}
