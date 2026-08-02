/**
 * Parent-side "Read aloud" controls for EPUB viewers — a play/stop button
 * + voice picker + speed picker that drive the shared `read-aloud.js`
 * pipeline. Lives in the parent DOM (not inside the sandboxed preview
 * iframe), so it has the session cookie and the TTS pipeline directly —
 * no postMessage bridge, no iframe sandbox dependencies.
 *
 * The element is returned so the caller can place it wherever fits the
 * surface (e.g. into the Files preview chrome next to Download, or above
 * the Studio book-viewer iframe).
 *
 * `textUrl` is an endpoint returning `{title, chapters:[{heading,text}]}`
 * — `/api/artifacts/{id}/epub-text` for Studio artifacts,
 *   `/api/files/epub-text/{id}` for file-index rows.
 *
 * `narrationUrl` (optional) is the POST endpoint that records a paired TTS
 * narration for this EPUB. When set, the first time the user hits play we
 * fire it once (fire-and-forget) — a passive "audio partner" saver, so a
 * companion audiobook gets built in the background just by listening. It's
 * idempotent server-side, so re-firing is a cheap no-op.
 *
 * Voice + speed persist per-user via the existing `/api/config/ui`
 * voice-prefs bucket (`readerTtsVoice` / `readerTtsSpeed`) so they
 * survive refresh + restart and don't disturb chat-TTS settings.
 *
 * Usage:
 *   const ctl = createReaderControls({ textUrl });
 *   someHost.prepend(ctl.el);
 *   ...
 *   ctl.destroy();
 */

import { escapeHtml, showToast } from './app.js';
import { readAloud, stopReadAloud, isReadAloudActive } from './read-aloud.js';
import { getVoices, getVoicesSync } from './model-cache.js';
import { getSettings, syncVoicePrefsToBackend } from './settings.js';

const SPEEDS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
const _STYLE_ID = 'epub-reader-controls-styles';

function _ensureStyles() {
  if (document.getElementById(_STYLE_ID)) return;
  const st = document.createElement('style');
  st.id = _STYLE_ID;
  st.textContent = `
.epub-reader-bar{display:inline-flex;align-items:center;gap:8px;font-size:var(--text-xs,12px);
  font-family:inherit;flex-wrap:wrap}
.epub-reader-bar .erb-play{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;
  border:1px solid var(--border,#2d2d45);border-radius:var(--radius-sm,6px);
  background:var(--bg,#0f0f1a);color:var(--text,#ececf1);cursor:pointer;font-size:inherit;
  font-family:inherit;white-space:nowrap;transition:background .15s,border-color .15s}
.epub-reader-bar .erb-play:hover{background:var(--accent-soft,rgba(108,138,255,.12))}
.epub-reader-bar.playing .erb-play,.epub-reader-bar.error .erb-play{border-color:#f87171;color:#f87171}
.epub-reader-bar.busy .erb-play{opacity:.6;pointer-events:none}
.epub-reader-bar select{background:var(--bg,#0f0f1a);color:var(--text,#ececf1);
  border:1px solid var(--border,#2d2d45);border-radius:var(--radius-sm,6px);padding:4px 6px;
  font-size:inherit;font-family:inherit;max-width:170px}
@media (max-width:600px){.epub-reader-bar select{max-width:96px}}
`;
  (document.head || document.documentElement).appendChild(st);
}

function _voiceOptionsHtml(voices, selected) {
  let html = '<option value="">Default voice</option>';
  if (!Array.isArray(voices)) return html;
  const groups = {};
  const order = [];
  for (const v of voices) {
    const rawId = typeof v === 'string' ? v : (v.id || v.name || v.voice_id || '');
    if (!rawId) continue;
    const provId = (typeof v === 'object' && v.provider_id) ? v.provider_id : '';
    const provName = (typeof v === 'object' && v.provider_name) ? v.provider_name : '';
    const label = typeof v === 'string' ? v : (v.name || v.id || v.voice_id || rawId);
    const value = provId ? `${provId}::${rawId}` : rawId;
    const key = provName || provId || '';
    if (!groups[key]) { groups[key] = []; order.push(key); }
    groups[key].push({ value, label });
  }
  const single = order.length <= 1;
  for (const key of order) {
    if (!single && key) html += `<optgroup label="${escapeHtml(key)}">`;
    for (const o of groups[key]) {
      html += `<option value="${escapeHtml(o.value)}"${o.value === selected ? ' selected' : ''}>${escapeHtml(o.label)}</option>`;
    }
    if (!single && key) html += '</optgroup>';
  }
  return html;
}

function _chaptersToText(chapters) {
  const parts = [];
  for (const ch of (chapters || [])) {
    if (!ch) continue;
    const h = (ch.heading || '').trim();
    const t = (ch.text || '').trim();
    // The extracted text usually already opens with the heading (an <h1>
    // in the body) — only prepend when it doesn't, to avoid reading the
    // chapter title twice.
    if (h && !t.toLowerCase().startsWith(h.toLowerCase())) parts.push(h);
    if (t) parts.push(t);
  }
  return parts.join('\n\n').trim();
}

/**
 * @param {{textUrl: string}} opts
 * @returns {{el: HTMLElement, destroy: () => void}}
 */
export function createReaderControls(opts = {}) {
  _ensureStyles();
  const textUrl = opts.textUrl || '';
  const narrationUrl = opts.narrationUrl || '';
  let _passiveSaverFired = false;
  function _firePassiveSaver() {
    if (_passiveSaverFired || !narrationUrl) return;
    _passiveSaverFired = true;
    // Build the companion audiobook in the background; idempotent, and a
    // 422 ("needs Kokoro") is fine to swallow — purely passive.
    fetch(narrationUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ voice }),
    }).catch(() => {});
  }
  const s = getSettings?.() || {};
  let voice = s.readerTtsVoice || '';
  let speed = Number(s.readerTtsSpeed) || 1.0;
  if (!SPEEDS.includes(speed)) speed = 1.0;

  const bar = document.createElement('div');
  bar.className = 'epub-reader-bar';
  bar.innerHTML = `
    <button type="button" class="erb-play" title="Read aloud (TTS)">
      <span class="erb-icon" aria-hidden="true">&#128266;</span><span class="erb-label">Read aloud</span>
    </button>
    <select class="erb-voice" title="Voice" aria-label="Read-aloud voice">${_voiceOptionsHtml(getVoicesSync(), voice)}</select>
    <select class="erb-speed" title="Speed" aria-label="Read-aloud speed">
      ${SPEEDS.map(v => `<option value="${v}"${v === speed ? ' selected' : ''}>${v}&#215;</option>`).join('')}
    </select>
  `;
  const playBtn = bar.querySelector('.erb-play');
  const iconEl = bar.querySelector('.erb-icon');
  const labelEl = bar.querySelector('.erb-label');
  const voiceSel = bar.querySelector('.erb-voice');
  const speedSel = bar.querySelector('.erb-speed');

  if (!getVoicesSync()) {
    getVoices().then(vs => { voiceSel.innerHTML = _voiceOptionsHtml(vs, voice); }).catch(() => {});
  }

  function persistPrefs() {
    const st = getSettings?.();
    if (st) { st.readerTtsVoice = voice; st.readerTtsSpeed = speed; }
    syncVoicePrefsToBackend?.().catch(() => {});
  }
  voiceSel.addEventListener('change', () => { voice = voiceSel.value || ''; persistPrefs(); });
  speedSel.addEventListener('change', () => {
    const n = parseFloat(speedSel.value);
    speed = !isNaN(n) ? n : 1.0;
    persistPrefs();
  });

  let textCache = null;
  async function loadText() {
    if (textCache != null) return textCache;
    if (!textUrl) throw new Error('Read-aloud is not configured for this view.');
    const resp = await fetch(textUrl);
    if (!resp.ok) {
      throw new Error(resp.status === 422
        ? 'No readable text found in this EPUB.'
        : `Could not load EPUB text (HTTP ${resp.status}).`);
    }
    const data = await resp.json();
    textCache = _chaptersToText(data.chapters);
    return textCache;
  }

  function setBusy(busy) {
    bar.classList.toggle('busy', !!busy);
    labelEl.textContent = busy ? 'Preparing…' : (bar.classList.contains('playing') ? 'Stop' : 'Read aloud');
  }
  function setPlaying(p) {
    bar.classList.toggle('playing', !!p);
    iconEl.innerHTML = p ? '&#9632;' : '&#128266;';
    labelEl.textContent = p ? 'Stop' : 'Read aloud';
  }
  function flashError(msg) {
    bar.classList.remove('busy', 'playing');
    bar.classList.add('error');
    iconEl.innerHTML = '&#128266;';
    labelEl.textContent = 'Read aloud';
    showToast?.(msg, 'error');
    setTimeout(() => bar.classList.remove('error'), 4000);
  }

  playBtn.addEventListener('click', async () => {
    if (isReadAloudActive() && bar.classList.contains('playing')) {
      stopReadAloud();
      setPlaying(false);
      bar.classList.remove('busy');
      return;
    }
    setBusy(true);
    let text;
    try { text = await loadText(); }
    catch (e) { flashError(e?.message || 'Could not load text for reading.'); return; }
    if (!text) { flashError('Nothing to read here.'); return; }
    bar.classList.remove('busy');
    setPlaying(true);
    _firePassiveSaver();
    try {
      await readAloud(text, null, { voice: voice || undefined, speed: speed || undefined });
    } catch { /* surfaced by read-aloud's own toast */ }
    setPlaying(false);
  });

  return {
    el: bar,
    destroy() {
      if (isReadAloudActive() && bar.classList.contains('playing')) stopReadAloud();
      bar.remove();
    },
  };
}
