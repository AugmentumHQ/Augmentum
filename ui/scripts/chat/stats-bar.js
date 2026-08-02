/* ==========================================================================
   Chat Module — Ambient Stats Bar
   Thin always-on strip pinned to the top of the composer (above the
   tool-icon row). Shows the most recent assistant turn's ttft, tok/s,
   and generation tokens. Replaces the per-bubble inline stats line that
   overflowed on narrow viewports.

   The strip's top edge doubles as the context-window meter: a full-width
   hairline track whose fill expands rightward as the conversation fills
   the model's window (green <70%, amber 70-90%, red >90%). This replaced
   the old in-toolbar ctx-bar pill, which got clipped on narrow viewports.
   Both the numbers and the fill are fed by the same `augmentum:turn-stats`
   event that MessageRenderer dispatches from updateStreamMetrics — fired
   on every metrics tick, so the fill grows live during generation.
   ========================================================================== */

const _ZONE_THRESHOLDS = [
  { min: 0.90, zone: 'danger' },
  { min: 0.70, zone: 'warn' },
  { min: 0.0,  zone: 'ok' },
];

function _zone(ratio) {
  for (const { min, zone } of _ZONE_THRESHOLDS) {
    if (ratio >= min) return zone;
  }
  return 'ok';
}

function _fmtMs(ms) {
  if (!(ms > 0)) return '';
  if (ms < 1000) return `${Math.round(ms)}ms`;
  const s = ms / 1000;
  return s >= 10 ? `${Math.round(s)}s` : `${s.toFixed(1)}s`;
}

function _fmtTok(n) {
  if (!(n > 0)) return '';
  if (n < 1000) return String(n);
  const v = n / 1000;
  return v >= 10 ? `${Math.round(v)}K` : `${v.toFixed(1)}K`;
}

function _buildTitle(s) {
  const lines = [];
  if (s.ttftMs > 0) lines.push(`TTFT ${_fmtMs(s.ttftMs)}`);
  if (s.tps > 0) lines.push(`${s.tps.toFixed(1)} tok/s`);
  if (s.evalTokens > 0) lines.push(`${s.evalTokens.toLocaleString()} generated`);
  if (s.contextUsed > 0 && s.contextLen > 0) {
    lines.push(`Context ${s.contextUsed.toLocaleString()} / ${s.contextLen.toLocaleString()}`);
  }
  if (s.promptTokensEvaluated > 0 || s.promptTokensCached > 0) {
    lines.push(
      `Prompt: ${(s.promptTokensEvaluated || 0).toLocaleString()} fresh + `
      + `${(s.promptTokensCached || 0).toLocaleString()} from KV cache`
      + (s.promptTokensCacheWrite > 0
        ? ` + ${s.promptTokensCacheWrite.toLocaleString()} written`
        : ''),
    );
  }
  return lines.join(' · ');
}

let _hideTimer = null;

function _render(stats) {
  const bar = document.getElementById('chat-stats-bar');
  if (!bar) return;

  const parts = [];
  if (stats.ttftMs > 0) parts.push(`${_fmtMs(stats.ttftMs)} ttft`);
  if (stats.tps > 0) parts.push(`${Math.round(stats.tps)} tok/s`);
  if (stats.evalTokens > 0) parts.push(`${_fmtTok(stats.evalTokens)} tok`);

  // Context occupancy renders as the full-width fill meter along the
  // strip's top edge (the hairline "expands" rightward as the window
  // fills), so it no longer needs a textual part — kept in the hover
  // title. Driven by the --ctx-fill width var + zone color.
  let hasCtx = false;
  if (stats.contextUsed > 0 && stats.contextLen > 0) {
    hasCtx = true;
    const ratio = Math.max(0, Math.min(1, stats.contextUsed / stats.contextLen));
    bar.dataset.zone = _zone(ratio);
    bar.style.setProperty('--ctx-fill', `${ratio * 100}%`);
    bar.setAttribute('role', 'meter');
    bar.setAttribute('aria-valuemin', '0');
    bar.setAttribute('aria-valuemax', '100');
    bar.setAttribute('aria-valuenow', String(Math.round(ratio * 100)));
  } else {
    delete bar.dataset.zone;
    bar.style.removeProperty('--ctx-fill');
    bar.removeAttribute('role');
    bar.removeAttribute('aria-valuenow');
  }

  if (parts.length === 0 && !hasCtx) {
    bar.classList.add('hidden');
    return;
  }

  // Build innerHTML with separators we can opacity down via CSS.
  bar.innerHTML = parts
    .map((p) => `<span class="chat-stats-bar-part">${p}</span>`)
    .join('<span class="chat-stats-bar-sep" aria-hidden="true">·</span>');
  bar.setAttribute('title', _buildTitle(stats));
  bar.classList.remove('hidden');

  // Brief pulse on update. Toggle the attribute off after the animation
  // so the next turn re-triggers it cleanly.
  bar.dataset.pulse = '1';
  if (_hideTimer) clearTimeout(_hideTimer);
  _hideTimer = setTimeout(() => {
    delete bar.dataset.pulse;
    _hideTimer = null;
  }, 650);
}

function _clearBar() {
  const bar = document.getElementById('chat-stats-bar');
  if (!bar) return;
  bar.innerHTML = '';
  bar.removeAttribute('title');
  delete bar.dataset.zone;
  delete bar.dataset.pulse;
  bar.style.removeProperty('--ctx-fill');
  bar.removeAttribute('role');
  bar.removeAttribute('aria-valuenow');
  bar.classList.add('hidden');
  if (_hideTimer) {
    clearTimeout(_hideTimer);
    _hideTimer = null;
  }
}

let _initialized = false;
export function initStatsBar() {
  if (_initialized) return;
  _initialized = true;
  document.addEventListener('augmentum:turn-stats', (e) => {
    if (!e.detail) return;
    _render(e.detail);
  });
  // Clear on session switch — stats are per-turn and don't carry
  // meaning across sessions. Before this listener, opening a fresh
  // chat left the bar showing the previous session's last turn's
  // numbers, which read as "this empty chat has 12K tokens" — a
  // state-awareness bug surfaced via dogfood 2026-05-31.
  document.addEventListener('augmentum:session-changed', _clearBar);
}
