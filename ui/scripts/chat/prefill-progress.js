/* ==========================================================================
   Chat Module — Prefill Progress Poller
   While llama-server is processing the prompt (prefill stage), its HTTP
   response body is silent — no tokens come back until first token. But
   it does emit ``slot print_timing: ... prompt processing, n_tokens =
   X, progress = Y, ...`` log lines every batch chunk. The manager
   parses those into a snapshot we expose via /api/engine/v2/prefill_progress.

   This module polls that endpoint during the prefill stage and feeds
   each snapshot into the active renderer's setPrefillProgress() — the
   user sees "Preparing context · 47% · 96 tok/s" with a thin progress
   bar instead of waiting on an opaque "Preparing context…" indicator
   for 30-180s on long-context turns.

   Started by chat/index.js when stage_start.stage === 'prefill';
   stopped on stage_complete, first content delta, or stream error.
   ========================================================================== */

const POLL_INTERVAL_MS = 500;

let _intervalId = null;
let _renderer = null;
let _model = '';

async function _tick() {
  if (!_renderer) return;
  try {
    // Pass the active model so prefill on a secondary-slot model is read
    // from its own engine, not the primary's snapshot.
    const q = _model ? `?model=${encodeURIComponent(_model)}` : '';
    const resp = await fetch(`/api/engine/v2/prefill_progress${q}`, {
      credentials: 'same-origin',
    });
    if (!resp.ok) return;  // 401/500 etc — silently skip; next tick retries
    const data = await resp.json();
    if (!data || !data.active) return;
    _renderer.setPrefillProgress({
      progress: data.progress ?? 0,
      tokens_done: data.tokens_done ?? 0,
      tps: data.tps ?? 0,
      elapsed_s: data.elapsed_s ?? 0,
    });
  } catch {
    // Network blip — next tick will try again. No need to break the
    // poll loop on a single transient failure.
  }
}

/** Begin polling. Idempotent — safe to call repeatedly; later calls
 *  replace the renderer target without restarting the timer. */
export function startPrefillPolling(renderer, model = '') {
  if (!renderer) return;
  _renderer = renderer;
  _model = model || '';
  if (_intervalId !== null) return;
  // Fire one tick immediately so the bar appears within ~50ms of stage
  // start rather than waiting for the first interval. The endpoint
  // returns active:false until the manager sees its first progress
  // log line, so the first few ticks may be no-ops — harmless.
  _tick();
  _intervalId = setInterval(_tick, POLL_INTERVAL_MS);
}

/** Stop polling and clear the bar from the renderer's streaming
 *  indicator. Idempotent. */
export function stopPrefillPolling() {
  if (_intervalId !== null) {
    clearInterval(_intervalId);
    _intervalId = null;
  }
  if (_renderer && typeof _renderer.clearPrefillProgress === 'function') {
    _renderer.clearPrefillProgress();
  }
  _renderer = null;
  _model = '';
}
