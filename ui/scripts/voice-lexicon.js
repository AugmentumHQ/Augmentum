/**
 * Per-voice TTS pronunciation lexicon — the table under Voices.
 *
 * One row = "when speaking with <voice>, say <term> as <phonetics>".
 * Backend: GET/POST/DELETE /api/voice/lexicon (migration 261); applied
 * on the speech endpoints before text cleaning, so entries beat every
 * built-in normalization rule.
 *
 * Preview buttons follow the voice-mixer pattern (POST
 * /api/audio/voices/preview?voice=&text= → audio blob): the add form
 * has one on EACH side — hear the term as the voice says it today,
 * hear your candidate phonetics — so you can iterate until it sounds
 * right before saving. Saved rows get a single test button that speaks
 * the term through the live pipeline (i.e. with the entry applied).
 *
 * Usage: mountVoiceLexicon(hostEl, { voices }) — voices from
 * model-cache getVoices(); falls back to a bare "every voice" select.
 */

import { escapeHtml, showToast } from './app.js';

function _voiceId(v) { return typeof v === 'string' ? v : (v.id || v.voice_id || v.name || ''); }
function _voiceLabel(v) { return typeof v === 'string' ? v : (v.name || v.id || v.voice_id || ''); }

async function _speak(btn, voice, text) {
  const t = (text || '').trim();
  if (!t) { showToast('Type something to preview first.', 'warning'); return; }
  btn.disabled = true;
  const orig = btn.textContent;
  btn.textContent = '…';
  try {
    const params = new URLSearchParams({ voice: voice || '', text: t.slice(0, 200) });
    const r = await fetch(`/api/audio/voices/preview?${params}`, { method: 'POST' });
    if (!r.ok) { showToast('Preview failed.', 'error'); return; }
    const url = URL.createObjectURL(await r.blob());
    const audio = new Audio(url);
    audio.play();
    audio.onended = () => URL.revokeObjectURL(url);
  } catch { showToast('Preview failed.', 'error'); }
  finally { btn.disabled = false; btn.textContent = orig; }
}

export function mountVoiceLexicon(host, opts = {}) {
  const voices = Array.isArray(opts.voices) ? opts.voices : [];
  let entries = [];

  const root = document.createElement('div');
  root.className = 'vlex-root field-group';
  root.innerHTML = `
    <label class="field-label">Pronunciation</label>
    <div style="font-size:var(--text-xs);color:var(--text-muted);line-height:1.5;margin-bottom:var(--space-xs)">
      Teach voices how to say things. Preview both sides — the word as
      written and your phonetic spelling — until it sounds right, then add it.
    </div>
    <div class="vlex-form" style="display:flex;gap:var(--space-xs);flex-wrap:wrap;align-items:center">
      <input type="text" class="field-input vlex-term" placeholder="Word (e.g. kubectl)" maxlength="80" style="flex:1;min-width:110px">
      <button type="button" class="btn btn-sm vlex-play-term" title="Hear it as the voice says it now">&#9658;</button>
      <input type="text" class="field-input vlex-phon" placeholder="Say it as… (e.g. kube control)" maxlength="200" style="flex:1.4;min-width:140px">
      <button type="button" class="btn btn-sm vlex-play-phon" title="Hear your phonetic spelling">&#9658;</button>
      <select class="field-input vlex-voice" style="flex:1;min-width:110px"></select>
      <button type="button" class="btn btn-primary btn-sm vlex-add">Add</button>
    </div>
    <div class="vlex-list" style="margin-top:var(--space-xs)"></div>
  `;
  host.appendChild(root);

  const termEl = root.querySelector('.vlex-term');
  const phonEl = root.querySelector('.vlex-phon');
  const voiceEl = root.querySelector('.vlex-voice');
  const listEl = root.querySelector('.vlex-list');

  voiceEl.innerHTML = `<option value="">Every voice</option>` + voices.map(v => {
    const id = _voiceId(v);
    return `<option value="${escapeHtml(id)}">${escapeHtml(_voiceLabel(v))}</option>`;
  }).join('');

  root.querySelector('.vlex-play-term').onclick = (e) =>
    _speak(e.currentTarget, voiceEl.value, termEl.value);
  root.querySelector('.vlex-play-phon').onclick = (e) =>
    _speak(e.currentTarget, voiceEl.value, phonEl.value);

  function _voiceDisplay(stored) {
    if (!stored) return 'every voice';
    const match = voices.find(v => _voiceId(v).split('::').pop() === stored);
    return match ? _voiceLabel(match) : stored;
  }

  function _render() {
    if (!entries.length) {
      listEl.innerHTML = `<div style="padding:var(--space-xs);color:var(--text-muted);font-size:var(--text-xs)">No pronunciations yet — add one above.</div>`;
      return;
    }
    listEl.innerHTML = entries.map(e => `
      <div class="vlex-row" data-id="${e.id}" style="display:flex;gap:var(--space-xs);align-items:center;padding:4px 6px;border-bottom:1px solid var(--border);font-size:var(--text-sm)">
        <span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis">${escapeHtml(e.term)}</span>
        <span style="flex:1.4;min-width:0;overflow:hidden;text-overflow:ellipsis;color:var(--text-muted)">${escapeHtml(e.phonetics || '(shielded)')}</span>
        <span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;font-size:var(--text-xs);color:var(--text-muted)">${escapeHtml(_voiceDisplay(e.voice))}</span>
        <button type="button" class="btn btn-sm vlex-row-play" title="Hear it (with this entry applied)">&#9658;</button>
        <button type="button" class="btn btn-sm vlex-row-del" title="Remove">&#10005;</button>
      </div>
    `).join('');
    for (const row of listEl.querySelectorAll('.vlex-row')) {
      const id = Number(row.dataset.id);
      const entry = entries.find(x => x.id === id);
      if (!entry) continue;
      row.querySelector('.vlex-row-play').onclick = (e) =>
        _speak(e.currentTarget, entry.voice, entry.term);
      row.querySelector('.vlex-row-del').onclick = async () => {
        try {
          const r = await fetch(`/api/voice/lexicon/${id}`, { method: 'DELETE' });
          const data = await r.json().catch(() => ({}));
          if (!data.ok) { showToast('Couldn’t remove that entry.', 'error'); return; }
          entries = entries.filter(x => x.id !== id);
          _render();
        } catch { showToast('Couldn’t remove that entry.', 'error'); }
      };
    }
  }

  async function _load() {
    try {
      const r = await fetch('/api/voice/lexicon');
      const data = await r.json().catch(() => ({}));
      entries = data.ok ? (data.entries || []) : [];
    } catch { entries = []; }
    _render();
  }

  root.querySelector('.vlex-add').onclick = async () => {
    const term = (termEl.value || '').trim();
    const phonetics = (phonEl.value || '').trim();
    if (!term) { showToast('Type the word to fix first.', 'warning'); termEl.focus(); return; }
    if (!phonetics) { showToast('Type how it should be spoken.', 'warning'); phonEl.focus(); return; }
    try {
      const r = await fetch('/api/voice/lexicon', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ term, phonetics, voice: voiceEl.value }),
      });
      const data = await r.json().catch(() => ({}));
      if (!data.ok) { showToast(data.reason || 'Couldn’t save that.', 'error'); return; }
      termEl.value = ''; phonEl.value = '';
      await _load();
      showToast('Pronunciation saved.', 'success');
    } catch { showToast('Couldn’t save that.', 'error'); }
  };

  _load();
  return { el: root, refresh: _load, destroy: () => root.remove() };
}
