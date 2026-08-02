/**
 * Reusable Kokoro voice-mixer component.
 *
 * Blends Kokoro voices (weighted) into a named mix that becomes a
 * selectable voice. Mounted into any host element; manages its own DOM,
 * styles, and server calls. Used by the TTS Recording Studio; the
 * Settings → Voice page still has its own inline copy (TODO: migrate it
 * to this module so there's one implementation).
 *
 * Backend it talks to (all already exist):
 *   GET  /api/audio/providers/bundled        → which bundled TTS containers are up
 *   POST /api/audio/voices/preview?provider_id=&voice=  → audio blob
 *   POST /api/audio/voices/combine?provider_id=         body {voices:[{name,weight}], save_as}
 *   DELETE /api/audio/voices/mixes/{name}
 *
 * Usage:
 *   const mixer = mountVoiceMixer(hostEl, {
 *     voices,                  // optional: array of voice objects (else fetched)
 *     onSaved(voiceId, label), // called after a mix is saved
 *     onUse(voiceId, label),   // called when the user clicks "Use" on a saved mix
 *   });
 *   ...
 *   mixer.destroy();
 *
 * `getKokoroProviderId()` resolves the active Kokoro provider id, or '' if
 * neither built-in nor sidecar Kokoro is available — callers use it to
 * decide whether to surface the mixer at all.
 */

import { escapeHtml, showToast } from './app.js';
import { getVoices, invalidate as invalidateModelCache } from './model-cache.js';

// Styles live in ui/styles/voice-mixer.css and are linked from ui/index.html.

const _RATIO_COLORS = ['#5ec4d4', '#b08ed8', '#e09070', '#8a9cc5', '#6b7a94', '#7fbf7f'];

// --- provider resolution ---------------------------------------------

let _bundledCache = null;
async function _bundled() {
  if (_bundledCache) return _bundledCache;
  try {
    const r = await fetch('/api/audio/providers/bundled');
    _bundledCache = r.ok ? (await r.json()) || {} : {};
  } catch { _bundledCache = {}; }
  return _bundledCache;
}

// Provider ids that support in-process style-vector blending, grouped by
// "family" (you can't blend across families — different latent spaces).
const _BLEND_FAMILIES = {
  kokoro: ['kokoro-builtin', 'kokoro-tts'],
};

/** Resolve a blend-capable provider id (Kokoro preferred), or ''. */
export async function getKokoroProviderId() {
  const b = await _bundled();
  if (b['kokoro-builtin']) return 'kokoro-builtin';
  if (b['kokoro-tts']) return 'kokoro-tts';
  return '';
}
// Back-compat alias for callers that should think in terms of "blend engine".
export const getBlendProviderId = getKokoroProviderId;

// --- helpers ---------------------------------------------------------

function _familyOf(providerId) {
  for (const [fam, ids] of Object.entries(_BLEND_FAMILIES)) {
    if (ids.includes(providerId)) return fam;
  }
  return 'kokoro';
}
function _voiceInFamily(v, family) {
  const p = (typeof v === 'object' && v) ? v.provider_id : '';
  return (_BLEND_FAMILIES[family] || []).includes(p);
}
function _voiceId(v) { return typeof v === 'string' ? v : (v.id || v.voice_id || v.name || ''); }
function _voiceLabel(v) { return typeof v === 'string' ? v : (v.name || v.id || v.voice_id || ''); }
function _shortName(name) { return String(name || '').replace(/^af_|^am_|^bf_|^bm_/, ''); }
function _buildMixString(parts) {
  // Raw weights with `*`; the backend normalises by total, so magnitudes
  // don't matter — preview and save then produce identical audio.
  return parts.map(p => `${p.name}*${p.weight}`).join('+');
}

// --- component -------------------------------------------------------

/**
 * @param {HTMLElement} host
 * @param {{voices?: any[], onSaved?: Function, onUse?: Function}} opts
 * @returns {{el: HTMLElement, destroy: Function}}
 */
export function mountVoiceMixer(host, opts = {}) {
  const onSaved = typeof opts.onSaved === 'function' ? opts.onSaved : null;
  const onUse = typeof opts.onUse === 'function' ? opts.onUse : null;
  let providerId = '';
  let _family = 'kokoro';   // resolved once providerId is known
  const _voicePool = Array.isArray(opts.voices) ? opts.voices.slice() : [];   // unfiltered seed
  let blendVoices = [];

  const root = document.createElement('div');
  root.className = 'vm-root';
  root.innerHTML = `
    <div class="vm-desc">Blend voices together — drag the sliders to set each voice's share.</div>
    <div class="vm-slots"></div>
    <div class="vm-ratio"></div>
    <button type="button" class="vm-add">+ Add voice</button>
    <div class="vm-actions">
      <input type="text" class="vm-name" placeholder="Name your blend" maxlength="64">
      <button type="button" class="vm-btn vm-preview">Preview</button>
      <button type="button" class="vm-btn vm-btn-primary vm-save">Save voice</button>
    </div>
    <div class="vm-saved" hidden></div>
  `;
  host.appendChild(root);

  const slotsEl = root.querySelector('.vm-slots');
  const ratioEl = root.querySelector('.vm-ratio');
  const addBtn = root.querySelector('.vm-add');
  const nameEl = root.querySelector('.vm-name');
  const previewBtn = root.querySelector('.vm-preview');
  const saveBtn = root.querySelector('.vm-save');
  const savedEl = root.querySelector('.vm-saved');

  function _voiceOptions(selectedId) {
    return blendVoices.map(v => {
      const id = _voiceId(v);
      return `<option value="${escapeHtml(id)}"${id === selectedId ? ' selected' : ''}>${escapeHtml(_voiceLabel(v))}</option>`;
    }).join('');
  }

  function _addSlot(selectedId) {
    if (!blendVoices.length) return;
    const row = document.createElement('div');
    row.className = 'vm-slot';
    const fallback = selectedId || _voiceId(blendVoices[slotsEl.children.length % blendVoices.length]);
    row.innerHTML = `
      <select>${_voiceOptions(fallback)}</select>
      <input type="range" min="1" max="100" step="1" value="50">
      <span class="vm-pct">50%</span>
      <button type="button" class="vm-rm" title="Remove">&times;</button>
    `;
    const slider = row.querySelector('input[type=range]');
    const pct = row.querySelector('.vm-pct');
    slider.addEventListener('input', () => { pct.textContent = `${slider.value}%`; _updateRatio(); });
    row.querySelector('select').addEventListener('change', _updateRatio);
    row.querySelector('.vm-rm').addEventListener('click', () => { row.remove(); _updateRatio(); });
    slotsEl.appendChild(row);
    _updateRatio();
  }

  function _getParts() {
    const merged = new Map();   // voiceId → summed weight (so dupes match the ratio bar)
    for (const slot of slotsEl.querySelectorAll('.vm-slot')) {
      const sel = slot.querySelector('select');
      const sld = slot.querySelector('input[type=range]');
      if (!sel || !sel.value) continue;
      merged.set(sel.value, (merged.get(sel.value) || 0) + (parseInt(sld.value, 10) || 1));
    }
    return Array.from(merged, ([name, weight]) => ({ name, weight }));
  }

  function _updateRatio() {
    const parts = _getParts();
    if (parts.length < 2) { ratioEl.innerHTML = ''; return; }
    const total = parts.reduce((s, p) => s + p.weight, 0) || 1;
    ratioEl.innerHTML = `<div class="vm-ratio-track">${
      parts.map((p, i) => {
        const pct = Math.round(p.weight / total * 100);
        const name = _shortName(p.name);
        return `<div class="vm-ratio-bar" style="flex:${p.weight};background:${_RATIO_COLORS[i % _RATIO_COLORS.length]}" title="${escapeHtml(name)}: ${pct}%">${pct >= 14 ? escapeHtml(name) : ''}</div>`;
      }).join('')
    }</div>`;
  }

  async function _preview() {
    const parts = _getParts();
    if (parts.length < 2) { showToast('Pick at least 2 distinct voices to blend.', 'warning'); return; }
    if (!providerId) { showToast('Kokoro is not available.', 'error'); return; }
    previewBtn.disabled = true;
    const orig = previewBtn.textContent;
    previewBtn.textContent = 'Generating…';
    try {
      const params = new URLSearchParams({ provider_id: providerId, voice: _buildMixString(parts) });
      const r = await fetch(`/api/audio/voices/preview?${params}`, { method: 'POST' });
      if (!r.ok) { showToast('Preview failed.', 'error'); return; }
      const url = URL.createObjectURL(await r.blob());
      const audio = new Audio(url);
      audio.play();
      audio.onended = () => URL.revokeObjectURL(url);
    } catch { showToast('Preview failed.', 'error'); }
    finally { previewBtn.disabled = false; previewBtn.textContent = orig; }
  }

  async function _save() {
    const parts = _getParts();
    if (parts.length < 2) { showToast('Pick at least 2 distinct voices to blend.', 'warning'); return; }
    const name = (nameEl.value || '').trim();
    if (!name) { showToast('Give your blend a name.', 'warning'); nameEl.focus(); return; }
    if (!providerId) { showToast('Kokoro is not available.', 'error'); return; }
    saveBtn.disabled = true;
    try {
      const r = await fetch(`/api/audio/voices/combine?provider_id=${encodeURIComponent(providerId)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voices: parts, save_as: name }),
      });
      const data = await r.json().catch(() => ({}));
      if (data && data.status === 'ok') {
        const mixId = data.saved_as || data.combined_voice || name;
        const fullVoice = `${providerId}::${mixId}`;
        showToast(`Saved blended voice “${mixId}”.`, 'success');
        nameEl.value = '';
        invalidateModelCache?.('voices');
        await _refreshVoices();
        _renderSaved();
        onSaved?.(fullVoice, mixId);
      } else {
        showToast((data && data.error) || 'Could not save the blend.', 'error');
      }
    } catch { showToast('Could not save the blend.', 'error'); }
    finally { saveBtn.disabled = false; }
  }

  async function _refreshVoices() {
    try {
      const all = await getVoices(true);
      if (Array.isArray(all)) blendVoices = all.filter(v => _voiceInFamily(v, _family));
      // Re-point any slot whose voice vanished.
      for (const slot of slotsEl.querySelectorAll('.vm-slot')) {
        const sel = slot.querySelector('select');
        const cur = sel.value;
        sel.innerHTML = _voiceOptions(cur);
        if (!sel.value && blendVoices.length) sel.value = _voiceId(blendVoices[0]);
      }
    } catch { /* keep what we have */ }
  }

  function _loadIntoSlots(blendSpec) {
    if (!blendSpec) return;
    const parsed = String(blendSpec).split('+').map(seg => {
      const s = seg.trim();
      let name = s, weight = 50;
      const star = s.lastIndexOf('*');
      if (star > 0) {
        name = s.slice(0, star).trim();
        const w = parseFloat(s.slice(star + 1));
        if (!isNaN(w)) weight = Math.max(1, Math.min(100, Math.round(w)));
      }
      return { name, weight };
    }).filter(p => p.name);
    if (parsed.length < 2) return;
    slotsEl.innerHTML = '';
    for (const p of parsed) {
      _addSlot(p.name);
      const slot = slotsEl.lastElementChild;
      const sld = slot.querySelector('input[type=range]');
      sld.value = String(p.weight);
      slot.querySelector('.vm-pct').textContent = `${p.weight}%`;
    }
    _updateRatio();
  }

  function _renderSaved() {
    const mixes = blendVoices.filter(v => v && (v.is_mix || _voiceId(v).includes('+')));
    if (!mixes.length) { savedEl.hidden = true; savedEl.innerHTML = ''; return; }
    savedEl.hidden = false;
    savedEl.innerHTML = '<div class="vm-saved-label">Saved blends</div>' + mixes.map(v => {
      const id = _voiceId(v);
      const label = _voiceLabel(v);
      const isMix = !!v.is_mix;
      return `<div class="vm-saved-item">
        <span class="vm-saved-name">${escapeHtml(label)}</span>
        <span style="display:flex;gap:4px;flex-shrink:0">
          <button type="button" class="vm-btn vm-mini vm-use" data-voice="${escapeHtml(`${providerId}::${id}`)}" data-label="${escapeHtml(label)}">Use</button>
          ${isMix ? `<button type="button" class="vm-btn vm-mini vm-edit" data-blend="${escapeHtml(id)}">Edit</button>` : ''}
          ${isMix ? `<button type="button" class="vm-btn vm-mini vm-mini-danger vm-del" data-mix="${escapeHtml(v.name || id)}">&times;</button>` : ''}
        </span>
      </div>`;
    }).join('');
    savedEl.querySelectorAll('.vm-use').forEach(b => b.addEventListener('click', () => onUse?.(b.dataset.voice, b.dataset.label)));
    savedEl.querySelectorAll('.vm-edit').forEach(b => b.addEventListener('click', () => _loadIntoSlots(b.dataset.blend)));
    savedEl.querySelectorAll('.vm-del').forEach(b => b.addEventListener('click', async () => {
      const mix = b.dataset.mix;
      if (!confirm(`Delete saved blend "${mix}"?`)) return;
      try {
        const r = await fetch(`/api/audio/voices/mixes/${encodeURIComponent(mix)}`, { method: 'DELETE' });
        if (!r.ok) { showToast('Delete failed.', 'error'); return; }
        showToast('Blend deleted.', 'success');
        invalidateModelCache?.('voices');
        await _refreshVoices();
        _renderSaved();
      } catch { showToast('Delete failed.', 'error'); }
    }));
  }

  addBtn.addEventListener('click', () => _addSlot());
  previewBtn.addEventListener('click', _preview);
  saveBtn.addEventListener('click', _save);

  // Async init: resolve the blend engine, filter the voice pool to that
  // engine's family, then build slots.
  (async () => {
    providerId = await getKokoroProviderId();
    _family = _familyOf(providerId);
    let pool = _voicePool;
    if (!pool.length) {
      try { const all = await getVoices(); if (Array.isArray(all)) pool = all; } catch { /* none */ }
    }
    blendVoices = (Array.isArray(pool) ? pool : []).filter(v => _voiceInFamily(v, _family));
    if (!providerId || !blendVoices.length) {
      root.innerHTML = '<div class="vm-desc">Voice blending needs the built-in Kokoro engine. '
        + 'Enable it under Settings → Voice.</div>';
      return;
    }
    _addSlot();
    _addSlot();
    _renderSaved();
    _updateRatio();
  })();

  return {
    el: root,
    destroy() { root.remove(); },
  };
}
