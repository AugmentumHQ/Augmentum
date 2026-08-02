/**
 * TTS Recording Studio — a small modal for turning text into an audio
 * file with the configured TTS voice. The result is saved as an audio
 * artifact, so it lands under Files → Audio. Opened from the Files audio
 * section ("New TTS recording").
 *
 * Short text (≤4000 chars) works with any TTS provider; longer text is
 * split and stitched server-side, which requires the built-in Kokoro
 * voice — the backend says so with a clear error.
 */

import { escapeHtml, showToast } from './app.js';
import { getVoices, getVoicesSync } from './model-cache.js';
import { getSettings } from './settings.js';
import { mountVoiceMixer, getKokoroProviderId } from './voice-mixer.js';

// Styles live in ui/styles/tts-studio.css and are linked from ui/index.html.

const SPEEDS = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0];
const FORMATS = ['mp3', 'wav'];
const SINGLE_CALL_LIMIT = 4000;

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

let _open = false;

export function openTtsStudio() {
  if (_open) return;
  _open = true;
  const s = getSettings?.() || {};
  const defVoice = s.readerTtsVoice || s.voiceDefaultVoice || '';
  const defSpeed = SPEEDS.includes(Number(s.readerTtsSpeed)) ? Number(s.readerTtsSpeed) : 1.0;

  const overlay = document.createElement('div');
  overlay.className = 'tts-studio-overlay';
  overlay.innerHTML = `
    <div class="tts-studio-panel" role="dialog" aria-label="TTS Recording Studio">
      <div class="tts-studio-head">
        <h2>TTS Recording Studio</h2>
        <button type="button" class="tts-studio-close" aria-label="Close">&times;</button>
      </div>
      <div class="tts-studio-body">
        <input type="text" class="ts-name" placeholder="Recording name (optional)" maxlength="120">
        <textarea class="ts-text" placeholder="Type or paste the text to speak…"></textarea>
        <div class="tts-studio-count"><span class="ts-count">0</span> characters</div>
        <div class="tts-studio-row">
          <label>Voice<select class="ts-voice">${_voiceOptionsHtml(getVoicesSync(), defVoice)}</select></label>
          <label>Speed<select class="ts-speed">${SPEEDS.map(v => `<option value="${v}"${v === defSpeed ? ' selected' : ''}>${v}&times;</option>`).join('')}</select></label>
          <label>Format<select class="ts-format">${FORMATS.map((f, i) => `<option value="${f}"${i === 0 ? ' selected' : ''}>${f.toUpperCase()}</option>`).join('')}</select></label>
        </div>
        <button type="button" class="tts-studio-secondary ts-blend-toggle" hidden style="align-self:flex-start">+ Create a blended voice</button>
        <div class="ts-mixer-panel" hidden style="border:1px solid var(--border,#2d2d45);border-radius:var(--radius-sm,8px);padding:12px"></div>
      </div>
      <div class="tts-studio-foot">
        <div class="tts-studio-actions">
          <button type="button" class="tts-studio-gen">Generate &amp; save</button>
          <button type="button" class="tts-studio-secondary ts-new" hidden>New recording</button>
        </div>
        <div class="tts-studio-result" hidden></div>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);
  requestAnimationFrame(() => overlay.classList.add('visible'));

  const panel = overlay.querySelector('.tts-studio-panel');
  const nameEl = overlay.querySelector('.ts-name');
  const textEl = overlay.querySelector('.ts-text');
  const countEl = overlay.querySelector('.ts-count');
  const countWrap = overlay.querySelector('.tts-studio-count');
  const voiceEl = overlay.querySelector('.ts-voice');
  const speedEl = overlay.querySelector('.ts-speed');
  const formatEl = overlay.querySelector('.ts-format');
  const genBtn = overlay.querySelector('.tts-studio-gen');
  const newBtn = overlay.querySelector('.ts-new');
  const resultEl = overlay.querySelector('.tts-studio-result');
  const blendToggle = overlay.querySelector('.ts-blend-toggle');
  const mixerPanelEl = overlay.querySelector('.ts-mixer-panel');

  if (!getVoicesSync()) {
    getVoices().then(vs => { voiceEl.innerHTML = _voiceOptionsHtml(vs, defVoice); }).catch(() => {});
  }

  // "+ Create a blended voice" — only when Kokoro is available; mounts the
  // shared voice-mixer inline, and on save/use refreshes the voice picker
  // and selects the new blend.
  let _mixer = null;
  function _selectVoice(fullVoice) {
    if (!fullVoice) return;
    getVoices(true).then(vs => {
      voiceEl.innerHTML = _voiceOptionsHtml(vs, fullVoice);
      voiceEl.value = fullVoice;
    }).catch(() => {});
  }
  getKokoroProviderId().then(pid => {
    if (!pid || !_open) return;
    blendToggle.hidden = false;
    blendToggle.addEventListener('click', () => {
      const showing = !mixerPanelEl.hidden;
      if (showing) { mixerPanelEl.hidden = true; blendToggle.textContent = '+ Create a blended voice'; return; }
      mixerPanelEl.hidden = false;
      blendToggle.textContent = '− Hide voice blender';
      if (!_mixer) {
        _mixer = mountVoiceMixer(mixerPanelEl, {
          onSaved: (fullVoice) => { _selectVoice(fullVoice); showToast('Blended voice ready — selected for this recording.', 'success'); },
          onUse: (fullVoice) => { _selectVoice(fullVoice); },
        });
      }
    });
  }).catch(() => {});

  function close() {
    _open = false;
    try { _mixer?.destroy(); } catch { /* noop */ }
    overlay.classList.remove('visible');
    setTimeout(() => overlay.remove(), 180);
  }
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
  overlay.querySelector('.tts-studio-close').addEventListener('click', close);
  overlay.addEventListener('keydown', (e) => { if (e.key === 'Escape') { e.preventDefault(); close(); } });

  function updateCount() {
    const n = textEl.value.length;
    countEl.textContent = n.toLocaleString();
    countWrap.classList.toggle('warn', n > SINGLE_CALL_LIMIT);
    countWrap.title = n > SINGLE_CALL_LIMIT
      ? 'Longer than 4,000 characters — needs the built-in Kokoro voice (text is split and stitched).'
      : '';
  }
  textEl.addEventListener('input', updateCount);
  updateCount();
  textEl.focus();

  newBtn.addEventListener('click', () => {
    textEl.value = '';
    nameEl.value = '';
    updateCount();
    resultEl.hidden = true;
    resultEl.innerHTML = '';
    newBtn.hidden = true;
    genBtn.disabled = false;
    textEl.focus();
  });

  genBtn.addEventListener('click', async () => {
    const text = textEl.value.trim();
    if (!text) { textEl.focus(); return; }
    genBtn.disabled = true;
    genBtn.textContent = 'Generating…';
    resultEl.hidden = true;
    resultEl.innerHTML = '';
    try {
      const resp = await fetch('/api/files/tts-studio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          name: nameEl.value.trim(),
          voice: voiceEl.value || '',
          speed: parseFloat(speedEl.value) || 1.0,
          format: formatEl.value || 'mp3',
        }),
      });
      if (!resp.ok) {
        let msg = `Synthesis failed (HTTP ${resp.status}).`;
        try { const d = await resp.json(); if (d && d.detail) msg = d.detail; } catch { /* keep default */ }
        resultEl.hidden = false;
        resultEl.innerHTML = `<div class="tts-studio-err">${escapeHtml(msg)}</div>`;
        genBtn.disabled = false;
        return;
      }
      const data = await resp.json();
      resultEl.hidden = false;
      resultEl.innerHTML = `
        <audio controls autoplay src="${escapeHtml(data.download_url)}"></audio>
        <div class="tts-studio-saved">Saved to Files → Audio as “${escapeHtml(data.name || data.filename || 'recording')}”.</div>
      `;
      newBtn.hidden = false;
      showToast('TTS recording saved to Files (Audio).', 'success');
      try { window.dispatchEvent(new CustomEvent('augmentum:tts-recording-saved', { detail: data })); } catch { /* noop */ }
    } catch {
      resultEl.hidden = false;
      resultEl.innerHTML = '<div class="tts-studio-err">Network error — could not reach the server.</div>';
      genBtn.disabled = false;
    } finally {
      genBtn.textContent = 'Generate & save';
      if (!newBtn.hidden) genBtn.disabled = true;   // keep disabled until "New recording"
    }
  });

  // Keep clicks inside the panel from bubbling to the overlay backdrop.
  panel.addEventListener('click', (e) => e.stopPropagation());
}
