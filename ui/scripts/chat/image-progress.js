/**
 * Shared image-generation progress loader.
 *
 * Single component used by:
 *   - chat/illustrate.js     ("Illustrate this moment" button)
 *   - app.js scene-gen flow  ("Generate scene image" composer action)
 *   - narrative/index.js     auto-background turn flow (badge variant)
 *
 * The DOM shape and CSS classes match the legacy `.illustrate-loading`
 * structure so existing styles apply unchanged. The new bit is a
 * polling controller that reads `/api/image/generation-status`,
 * updates the label text + step counter, and toggles the fill into
 * determinate mode when steps_total > 0.
 *
 * Lifecycle:
 *   const ctrl = createImageProgressLoader({ session_id, variant });
 *   parentEl.appendChild(ctrl.element);
 *   ctrl.start();    // begin polling
 *   await fetch(...);
 *   ctrl.stop();     // halts polling AND removes the element
 *
 * Variants:
 *   - "moment"    label: "Illustrating this moment…"          (default in illustrate)
 *   - "scene"     label: "Generating scene image…"             (default in app.js)
 *   - "auto-bg"   label: "Generating background…"              (narrative auto)
 *
 * Why a builder instead of a class: callers want straight-line
 * `start()` / `stop()` semantics with no inheritance; the closure
 * also keeps the poll handle private so the caller can't accidentally
 * leak it.
 */

const _DEFAULT_LABELS = {
  moment: 'Illustrating this moment…',
  scene: 'Generating scene image…',
  'auto-bg': 'Generating background…',
};

/** Format an elapsed-seconds count for the detail line. */
function _fmtElapsed(s) {
  if (s == null || s < 0) return '';
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s % 60)}s`;
}

/** Build the detail-line text from a /generation-status payload.
 *  Order of preference: explicit step counter > stage text only. */
function _buildDetail(status) {
  if (!status) return '';
  const parts = [];
  if (status.steps_total > 0) {
    parts.push(`step ${status.steps_done}/${status.steps_total}`);
  }
  if (status.queue_size > 0 && !status.active) {
    parts.push(`${status.queue_size} ahead`);
  }
  if (status.elapsed_s > 0) {
    parts.push(_fmtElapsed(status.elapsed_s));
  }
  return parts.join(' · ');
}

/** Pick the user-facing label. ``pre_queue`` (distiller running) wins
 *  because it's the earliest informative state; otherwise fall back to
 *  the job stage text; finally the variant's default. */
function _buildLabel(status, fallback) {
  if (status?.pre_queue?.stage) return `${status.pre_queue.stage}…`;
  if (status?.stage) return `${status.stage}…`;
  return fallback;
}

export function createImageProgressLoader({
  session_id = '',
  variant = 'moment',
  pollIntervalMs = 800,
  category = 'user',
} = {}) {
  const label = _DEFAULT_LABELS[variant] || _DEFAULT_LABELS.moment;

  // DOM — matches the legacy illustrate-loading shape so existing CSS applies.
  const root = document.createElement('div');
  root.className = 'illustrate-loading';
  root.dataset.variant = variant;
  root.innerHTML = `
    <div class="illustrate-loading-bar"><div class="illustrate-loading-fill"></div></div>
    <span class="illustrate-loading-text"></span>
    <span class="illustrate-loading-detail"></span>
  `;
  const barEl = root.querySelector('.illustrate-loading-bar');
  const fillEl = root.querySelector('.illustrate-loading-fill');
  const textEl = root.querySelector('.illustrate-loading-text');
  const detailEl = root.querySelector('.illustrate-loading-detail');
  textEl.textContent = label;

  let pollHandle = null;
  let cancelled = false;

  function _applyStatus(status) {
    textEl.textContent = _buildLabel(status, label);
    detailEl.textContent = _buildDetail(status);
    if (status?.steps_total > 0) {
      const pct = Math.max(0, Math.min(100, (status.steps_done / status.steps_total) * 100));
      barEl.dataset.determinate = 'true';
      fillEl.style.width = `${pct}%`;
    } else {
      // Return to indeterminate sweep — model load / save / pre-queue.
      barEl.dataset.determinate = 'false';
      fillEl.style.width = '';
    }
  }

  async function _poll() {
    if (cancelled) return;
    try {
      // Always pass the category filter so the in-message loader
      // doesn't pick up an auto_bg job's progress (the corner badge
      // owns those) and the corner badge doesn't pick up a user-
      // initiated illustrate's progress (the in-message loader owns
      // those). Single endpoint, two clean surfaces.
      const params = new URLSearchParams();
      if (session_id) params.set('session_id', session_id);
      if (category) params.set('category', category);
      const qs = params.toString();
      const url = qs ? `/api/image/generation-status?${qs}` : '/api/image/generation-status';
      const resp = await fetch(url);
      if (resp.ok) {
        const data = await resp.json();
        _applyStatus(data);
      }
    } catch { /* network blip — keep trying */ }
    if (!cancelled) pollHandle = setTimeout(_poll, pollIntervalMs);
  }

  return {
    element: root,
    start() {
      // First poll after a short delay so the server has time to
      // enqueue / start the job. Matches voice.js's existing timing.
      pollHandle = setTimeout(_poll, 300);
    },
    stop() {
      cancelled = true;
      if (pollHandle) {
        clearTimeout(pollHandle);
        pollHandle = null;
      }
      if (root.parentNode) root.parentNode.removeChild(root);
    },
    /** Replace the loader's default label without restarting polling. */
    setBaseLabel(newLabel) {
      textEl.textContent = newLabel;
    },
  };
}
