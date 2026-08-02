/* ==========================================================================
   Sampling editor — one design-system-native dialog, two callers

   Shared by the per-model editor (model library "Tuning" → models.js) and the
   per-chat editor (composer "Tuning" → chat/toolbar/tuning.js) so both look
   and behave identically. Built on the app's .modal-* component (overlay blur,
   scale-in animation, mobile sizing) rather than bespoke inline styles.

   Each field is blank-to-inherit: the placeholder shows the inherited
   (effective) value, so an empty box visibly tells you what it falls back to.
   ========================================================================== */

import { escapeHtml } from './app.js';

/** Canonical field set — shared so the two callers never drift. */
export const SAMPLING_FIELDS = [
  { key: 'temperature',     label: 'Temperature',     step: '0.05', min: '0', max: '2' },
  { key: 'top_p',           label: 'Top-p',           step: '0.01', min: '0', max: '1' },
  { key: 'top_k',           label: 'Top-k',           step: '1',    min: '0', max: '500' },
  { key: 'min_p',           label: 'Min-p',           step: '0.01', min: '0', max: '1' },
  { key: 'repeat_penalty',  label: 'Repeat penalty',  step: '0.01', min: '0', max: '2' },
  { key: 'presence_penalty', label: 'Presence penalty', step: '0.1', min: '0', max: '2' },
];

const _CLOSE_SVG = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
  + 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
  + '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>';

function _num(raw, key) {
  const t = String(raw ?? '').trim();
  if (t === '') return undefined;
  const n = Number(t);
  if (Number.isNaN(n)) return undefined;
  return key === 'top_k' ? Math.round(n) : n;
}

/**
 * Open the sampling editor.
 *
 * @param {object}   opts
 * @param {string}   opts.scopeLabel    Chip text — "This chat" or the model name.
 * @param {string}   opts.helpText      One-line explainer under the header.
 * @param {object}   opts.values        Current override {key: number} (the saved edit).
 * @param {object}   [opts.effective]   Inherited values {key: number} → shown as placeholders.
 * @param {object}   [opts.recommended] Family/card values {key: number} → "Reset to recommended".
 * @param {string[]} [opts.supported]   Field keys the resolved backend actually
 *   honors (from /api/models/{model}/sampling). Fields outside this list are
 *   rendered disabled with a "not honored" note so the editor never offers a
 *   dead control (e.g. min_p/top_k on OpenAI). Omit/null = show every field
 *   (backward compatible when the backend couldn't be resolved).
 * @param {string}   [opts.providerLabel] Provider name for the "not honored by
 *   X" note (e.g. "OpenAI"). Falls back to a generic phrasing.
 * @param {(o:object)=>any} opts.onSave Called with the override object on Save.
 */
export function openSamplingEditor(opts) {
  const {
    scopeLabel = '', helpText = '', values = {},
    effective = {}, recommended = {}, supported = null,
    providerLabel = '', onSave,
  } = opts || {};
  const hasRecommended = recommended && Object.keys(recommended).length > 0;
  const supports = (key) => !Array.isArray(supported) || supported.includes(key);
  const unavailableNote = providerLabel
    ? `not honored by ${providerLabel}`
    : "not honored by this model's provider";

  const rows = SAMPLING_FIELDS.map((f) => {
    const ok = supports(f.key);
    const cur = values[f.key] != null ? values[f.key] : '';
    const eff = effective[f.key] != null ? String(effective[f.key]) : 'inherit';
    const rec = recommended[f.key] != null ? String(recommended[f.key]) : null;
    const meta = !ok
      ? unavailableNote
      : (rec != null
        ? `recommended ${escapeHtml(rec)}`
        : (effective[f.key] != null ? `inherits ${escapeHtml(eff)}` : 'optional'));
    return `
      <div class="sampling-row${ok ? '' : ' sampling-row-unsupported'}">
        <div class="sampling-row-label">
          <span class="sampling-row-name">${escapeHtml(f.label)}</span>
          <span class="sampling-row-meta">${escapeHtml(meta)}</span>
        </div>
        <input class="field-input sampling-input" type="number" inputmode="decimal"
               id="samp-${f.key}" data-key="${f.key}"${ok ? '' : ' disabled'}
               step="${f.step}" min="${f.min}" max="${f.max}"
               value="${escapeHtml(String(cur))}" placeholder="${ok ? escapeHtml(eff) : 'unavailable'}"
               aria-label="${escapeHtml(f.label)}">
      </div>`;
  }).join('');

  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-modal', 'true');
  overlay.setAttribute('aria-label', 'Sampling settings');
  overlay.innerHTML = `
    <div class="modal sampling-modal">
      <div class="modal-header">
        <div class="sampling-head-titles">
          <span class="modal-title">Sampling</span>
          ${scopeLabel ? `<span class="sampling-scope">${escapeHtml(scopeLabel)}</span>` : ''}
        </div>
        <button class="icon-btn small" data-samp="close" type="button" aria-label="Close">${_CLOSE_SVG}</button>
      </div>
      <div class="modal-body">
        ${helpText ? `<p class="sampling-help">${escapeHtml(helpText)}</p>` : ''}
        <div class="sampling-grid">${rows}</div>
      </div>
      <div class="modal-footer sampling-footer-spread">
        ${hasRecommended
          ? '<button class="btn btn-ghost btn-sm" data-samp="recommend" type="button">Use recommended</button>'
          : '<span></span>'}
        <div class="sampling-footer-right">
          <button class="btn btn-ghost btn-sm" data-samp="clear" type="button">Clear</button>
          <button class="btn btn-sm" data-samp="cancel" type="button">Cancel</button>
          <button class="btn btn-primary btn-sm" data-samp="save" type="button">Save</button>
        </div>
      </div>
    </div>`;

  const inputs = () => Array.from(overlay.querySelectorAll('.sampling-input'));
  const read = () => {
    const out = {};
    for (const el of inputs()) {
      // Never persist a knob the backend won't honor — disabled = unsupported.
      // This also cleans up a stale override left over from a model swap.
      if (el.disabled) continue;
      const v = _num(el.value, el.dataset.key);
      if (v !== undefined) out[el.dataset.key] = v;
    }
    return out;
  };

  const close = () => {
    document.removeEventListener('keydown', onKey);
    overlay.remove();
  };
  function onKey(e) {
    if (e.key === 'Escape') { e.preventDefault(); close(); }
  }

  const doSave = async () => {
    const btn = overlay.querySelector('[data-samp="save"]');
    if (btn) { btn.disabled = true; btn.textContent = 'Saving…'; }
    try {
      await onSave?.(read());
      close();
    } catch (err) {
      console.error('[sampling-editor] save failed:', err);
      if (btn) { btn.disabled = false; btn.textContent = 'Save'; }
    }
  };

  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) { close(); return; }
    const action = e.target.closest('[data-samp]')?.dataset.samp;
    if (action === 'close' || action === 'cancel') close();
    else if (action === 'clear') { for (const el of inputs()) { if (!el.disabled) el.value = ''; } }
    else if (action === 'recommend') {
      for (const el of inputs()) {
        if (el.disabled) continue;  // don't fill a knob the provider ignores
        const r = recommended[el.dataset.key];
        el.value = r != null ? String(r) : '';
      }
    } else if (action === 'save') doSave();
  });
  // Enter anywhere in the field grid saves.
  overlay.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.classList?.contains('sampling-input')) {
      e.preventDefault(); doSave();
    }
  });
  document.addEventListener('keydown', onKey);

  document.body.appendChild(overlay);
  // Focus the first field for immediate keyboard editing.
  overlay.querySelector('.sampling-input')?.focus();
  return { close };
}
