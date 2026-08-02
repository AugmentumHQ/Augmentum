/**
 * Language-learning UI.
 *
 * Surfaces (all browse-panel-local):
 *   1. A "Learning" chip on the browse landing — *always* present once
 *      the browse landing renders; it's the discovery point. Clicking it:
 *        • no language pack installed  → language picker → install → onboarding
 *        • pack installed, toggle off  → onboarding modal
 *        • pack installed, toggle on   → SRS review overlay
 *   2. A "N words due" tile under the chip row (only when something's due).
 *   3. Click-to-define popover on ZIM article iframes (only when the
 *      Learning toggle is on — otherwise no handlers attach at all).
 *   4. The SRS review overlay.
 *   5. Global press-and-hold-to-translate: anywhere a CJK target language
 *      appears (chat bubbles, learning surfaces, games), holding a word
 *      pops the same define + sentence-translate popover. CJK-only because
 *      the script identifies the language unambiguously; Latin-script
 *      targets (es/fr) stay on surfaces 1-3. See `_globalHoldHandler`.
 *
 * Module state is a single cached profile snapshot (`_state`) plus a tiny
 * per-language due-count cache, refreshed on demand. The chip re-renders
 * on browse.js's `augmentum:browse-landing-ready` event so it survives
 * landing view churn; click-to-define wires in on `augmentum:browse-iframe-loaded`.
 */

import { escapeHtml, showToast, app } from './app.js';

const CHIP_ID = 'browse-landing-learning-chip';
const TILE_ID = 'browse-landing-learning-tile';
const POPOVER_ID = 'learning-define-popover';

let _state = null;
let _stateInflight = null;

// ── State ────────────────────────────────────────────────────────

async function _fetchState() {
  if (_stateInflight) return _stateInflight;
  _stateInflight = (async () => {
    try {
      const r = await fetch('/api/learning/state', { credentials: 'same-origin' });
      _state = r.ok ? await r.json() : null;
      return _state;
    } catch {
      _state = null;
      return null;
    } finally {
      _stateInflight = null;
    }
  })();
  return _stateInflight;
}

async function _fetchDue(lang, limit = 1) {
  try {
    const r = await fetch(`/api/learning/srs/due?lang=${encodeURIComponent(lang)}&limit=${limit}`);
    if (!r.ok) return { total: 0, due: [] };
    return await r.json();
  } catch {
    return { total: 0, due: [] };
  }
}

function _activePacks() {
  return (_state && Array.isArray(_state.packs)) ? _state.packs.filter(p => p.active) : [];
}

function _firstActiveTarget() {
  if (!_state) return null;
  const active = _activePacks();
  const targets = Array.isArray(_state.target_langs) ? _state.target_langs : [];
  for (const t of targets) if (active.some(p => p.lang_code === t)) return t;
  return active.length ? active[0].lang_code : null;
}

const _LANG_LABELS = {
  ja: 'Japanese', es: 'Spanish', fr: 'French', de: 'German', zh: 'Chinese', ko: 'Korean',
};
function _langLabel(code) {
  return _LANG_LABELS[code] || (code || '').toUpperCase();
}

function _fmtInterval(days) {
  const d = Math.max(1, Math.round(Number(days) || 1));
  if (d < 30) return `${d}d`;
  if (d < 365) return `${Math.round(d / 30)}mo`;
  return `${Math.round(d / 365)}y`;
}

// ── Kana → Hepburn romaji ────────────────────────────────────────
// Combo keys (2-char) are checked before singles. Katakana is normalised
// to hiragana first; small っ doubles the next consonant ("ch" → "t");
// the long-vowel mark ー / ー extends the previous vowel.
const _ROMA = {
  'きゃ':'kya','きゅ':'kyu','きょ':'kyo','しゃ':'sha','しゅ':'shu','しょ':'sho',
  'ちゃ':'cha','ちゅ':'chu','ちょ':'cho','にゃ':'nya','にゅ':'nyu','にょ':'nyo',
  'ひゃ':'hya','ひゅ':'hyu','ひょ':'hyo','みゃ':'mya','みゅ':'myu','みょ':'myo',
  'りゃ':'rya','りゅ':'ryu','りょ':'ryo','ぎゃ':'gya','ぎゅ':'gyu','ぎょ':'gyo',
  'じゃ':'ja','じゅ':'ju','じょ':'jo','びゃ':'bya','びゅ':'byu','びょ':'byo',
  'ぴゃ':'pya','ぴゅ':'pyu','ぴょ':'pyo','ぢゃ':'ja','ぢゅ':'ju','ぢょ':'jo',
  'ふぁ':'fa','ふぃ':'fi','ふぇ':'fe','ふぉ':'fo','てぃ':'ti','でぃ':'di','とぅ':'tu','どぅ':'du',
  'うぃ':'wi','うぇ':'we','うぉ':'wo','ゔぁ':'va','ゔぃ':'vi','ゔぇ':'ve','ゔぉ':'vo',
  'じぇ':'je','ちぇ':'che','しぇ':'she',
  'あ':'a','い':'i','う':'u','え':'e','お':'o',
  'か':'ka','き':'ki','く':'ku','け':'ke','こ':'ko',
  'が':'ga','ぎ':'gi','ぐ':'gu','げ':'ge','ご':'go',
  'さ':'sa','し':'shi','す':'su','せ':'se','そ':'so',
  'ざ':'za','じ':'ji','ず':'zu','ぜ':'ze','ぞ':'zo',
  'た':'ta','ち':'chi','つ':'tsu','て':'te','と':'to',
  'だ':'da','ぢ':'ji','づ':'zu','で':'de','ど':'do',
  'な':'na','に':'ni','ぬ':'nu','ね':'ne','の':'no',
  'は':'ha','ひ':'hi','ふ':'fu','へ':'he','ほ':'ho',
  'ば':'ba','び':'bi','ぶ':'bu','べ':'be','ぼ':'bo',
  'ぱ':'pa','ぴ':'pi','ぷ':'pu','ぺ':'pe','ぽ':'po',
  'ま':'ma','み':'mi','む':'mu','め':'me','も':'mo',
  'や':'ya','ゆ':'yu','よ':'yo',
  'ら':'ra','り':'ri','る':'ru','れ':'re','ろ':'ro',
  'わ':'wa','ゐ':'wi','ゑ':'we','を':'o','ん':'n','ゔ':'vu',
  'ぁ':'a','ぃ':'i','ぅ':'u','ぇ':'e','ぉ':'o','ゃ':'ya','ゅ':'yu','ょ':'yo',
  '　':' ',
};
const _VOWELS = new Set(['a', 'i', 'u', 'e', 'o']);

function _romaji(kana) {
  if (!kana) return '';
  const s = String(kana).replace(/[ァ-ヶ]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0x60));
  let out = '';
  let i = 0;
  let geminate = false;
  while (i < s.length) {
    const ch = s[i];
    if (ch === 'っ') { geminate = true; i += 1; continue; }
    if (ch === 'ー' || ch === 'ー') {
      const last = out[out.length - 1];
      if (_VOWELS.has(last)) out += last;
      i += 1; continue;
    }
    const two = s.slice(i, i + 2);
    let roma;
    if (_ROMA[two]) { roma = _ROMA[two]; i += 2; }
    else if (_ROMA[ch]) { roma = _ROMA[ch]; i += 1; }
    else { out += ch; i += 1; continue; }
    if (geminate) {
      if (roma.startsWith('ch')) out += 't';
      else if (roma[0] && !_VOWELS.has(roma[0])) out += roma[0];
      geminate = false;
    }
    out += roma;
  }
  return out;
}

// ── Part-of-speech codes → human labels ──────────────────────────
// The label map per language is shipped *inside the pack* and arrives
// via /api/learning/state as `pos_labels_by_lang`. Unrecognised codes
// fall through to the raw string — better than dropping them.
function _posLabelsFor(lang) {
  const map = (_state && _state.pos_labels_by_lang) || {};
  return (lang && map[lang]) || {};
}
function _posLabel(codesStr, lang) {
  if (!codesStr) return '';
  const labels = _posLabelsFor(lang);
  return String(codesStr).split(',').map((c) => {
    const code = c.trim();
    return labels[code] || code;
  }).filter(Boolean).join(' · ');
}

// ── Word audio (on-demand Kokoro TTS) ────────────────────────────
let _audioEl = null;
function _ttsVoice() {
  const v = _state && _state.tts_voice;
  return (v && v !== 'off') ? v : null;
}

// Per-lang voice resolution cache. Without this every TTS call from the
// review overlay / browse popover used `_ttsVoice()` (the user's global
// voice — almost always English), so a Japanese flashcard's reading
// gets spoken by an English voice and sounds like a placeholder. The
// resolver matches the games' resolver: prefer a Kokoro/server voice
// tagged with the target lang, fall back to the user's global voice.
const _LEARNING_VOICE_CACHE = new Map();
async function _resolveVoiceForLang(lang) {
  if (!lang) return _ttsVoice();
  if (_LEARNING_VOICE_CACHE.has(lang)) {
    return _LEARNING_VOICE_CACHE.get(lang) || _ttsVoice();
  }
  let pick = null;
  try {
    const r = await fetch('/api/audio/voices');
    if (r.ok) {
      const voices = await r.json();
      const matches = (Array.isArray(voices) ? voices : []).filter(v => v.lang === lang);
      matches.sort((a, b) => (b.recommended ? 1 : 0) - (a.recommended ? 1 : 0));
      pick = matches.length ? (matches[0].name || matches[0].voice_id || null) : null;
    }
  } catch { /* fall through */ }
  _LEARNING_VOICE_CACHE.set(lang, pick);
  return pick || _ttsVoice();
}

async function _speak(text, btnEl, lang = null) {
  // Resolve voice per-lang first; only fall back to the global voice
  // when no lang-specific voice is installed (or `lang` is null —
  // backward-compat for non-lang-scoped callers).
  const voice = await _resolveVoiceForLang(lang);
  if (!voice || !text) return;
  if (btnEl) btnEl.classList.add('learning-speaking');
  try {
    const r = await fetch('/v1/audio/speech', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'tts-1', voice, input: String(text), response_format: 'mp3' }),
    });
    if (!r.ok) throw new Error(`tts ${r.status}`);
    const blob = await r.blob();
    if (_audioEl) { try { _audioEl.pause(); } catch { /* */ } }
    _audioEl = new Audio(URL.createObjectURL(blob));
    _audioEl.addEventListener('ended', () => { btnEl && btnEl.classList.remove('learning-speaking'); });
    await _audioEl.play();
  } catch {
    if (btnEl) { btnEl.classList.add('learning-speak-error'); setTimeout(() => btnEl.classList.remove('learning-speak-error'), 1200); }
  } finally {
    setTimeout(() => btnEl && btnEl.classList.remove('learning-speaking'), 4000);
  }
}
const _SPEAKER_SVG = '<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>';
const _LENS_SVG = '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>';

// Wire a 🔊 button: tap → speak with the current voice; press-and-hold
// (~450ms) or right-click → open the voice picker. ``getText`` returns the
// text to pronounce at click time. ``lang`` (optional) routes through
// the per-lang voice resolver so Japanese flashcards get spoken by a
// Japanese voice even when the user's global voice is English.
function _attachSpeakButton(btn, getText, lang = null) {
  if (!btn) return;
  let holdTimer = null;
  let longPressed = false;
  const clearHold = () => { if (holdTimer) clearTimeout(holdTimer); holdTimer = null; };
  btn.addEventListener('pointerdown', () => {
    longPressed = false;
    clearHold();
    holdTimer = setTimeout(() => { longPressed = true; holdTimer = null; _openVoicePicker(btn, getText); }, 450);
  });
  btn.addEventListener('pointerup', (e) => {
    e.stopPropagation();
    clearHold();
    if (!longPressed) _speak(getText(), btn, lang);
    longPressed = false;
  });
  btn.addEventListener('pointerleave', clearHold);
  btn.addEventListener('pointercancel', clearHold);
  btn.addEventListener('click', (e) => e.stopPropagation());
  btn.addEventListener('contextmenu', (e) => { e.preventDefault(); _openVoicePicker(btn, getText); });
}

// Stable distinct colour for a saved voice mix / cloned voice, so the
// user can tell "which of their voices" at a glance in the picker.
const _VOICE_PALETTE = ['#5b8def', '#e0729a', '#3fa97a', '#d99836', '#9b7ce0', '#3fb0b0', '#d96c5b', '#7e9c3f'];
function _colorForVoice(name) {
  let h = 0;
  for (let i = 0; i < name.length; i += 1) h = (h * 31 + name.charCodeAt(i)) | 0;
  return _VOICE_PALETTE[Math.abs(h) % _VOICE_PALETTE.length];
}

const _GRADE_RANK = { 'A':0,'A-':1,'B+':2,'B':3,'B-':4,'C+':5,'C':6,'C-':7,'D+':8,'D':9,'D-':10,'F+':11,'F':12 };

async function _openVoicePicker(anchorBtn, getText) {
  document.getElementById('learning-voice-picker')?.remove();
  let voices = [];
  try {
    const r = await fetch('/api/audio/voices');
    if (r.ok) {
      const j = await r.json();
      voices = Array.isArray(j) ? j : (j.voices || []);
    }
  } catch { /* leave empty */ }

  // The "native" group is the voices that match the lang the user is
  // currently studying. Filtering at picker-open-time means the same
  // picker code serves every language the user installs.
  const activeLang = _firstActiveTarget() || 'ja';
  const langLabel = _langLabel(activeLang);
  const native = [];   // Kokoro voices whose lang prefix matches activeLang
  const mine = [];     // the user's saved mixes / cloned voices
  for (const v of voices) {
    const name = v.id || v.name || v.voice || '';
    if (!name) continue;
    const voiceLang = (v.lang || '').toLowerCase();
    const isMix = !!(v.blend_spec || v.kind === 'mix' || v.type === 'mix' || v.is_mix);
    if (isMix) {
      mine.push({ name, label: v.name || name, color: _colorForVoice(name) });
    } else if (voiceLang.startsWith(activeLang)) {
      native.push({ name, label: v.desc || v.name || name, grade: v.grade || '' });
    }
  }
  native.sort((a, b) => (_GRADE_RANK[a.grade] ?? 99) - (_GRADE_RANK[b.grade] ?? 99) || a.name.localeCompare(b.name));
  const current = (_state && _state.tts_voice) || 'off';

  const row = (voice, label, swatch, extra) =>
    `<button type="button" class="learning-vp-row${voice === current ? ' learning-vp-current' : ''}" data-voice="${escapeHtml(voice)}">
       <span class="learning-vp-swatch" style="background:${swatch}"></span>
       <span class="learning-vp-name">${escapeHtml(label)}</span>${extra || ''}
     </button>`;

  let html = `<div class="learning-vp-group">${escapeHtml(langLabel)} voices</div>`;
  if (native.length) {
    for (const v of native) {
      const isBest = v === native[0];
      const sw = isBest ? 'var(--accent)' : 'var(--text-muted)';
      const tag = (isBest ? '<span class="learning-vp-tag">best</span>' : '') +
        (v.grade ? `<span class="learning-vp-grade">${escapeHtml(v.grade)}</span>` : '');
      html += row(v.name, v.label, sw, tag);
    }
  } else {
    html += `<div class="learning-vp-empty">No built-in voices for ${escapeHtml(langLabel)}.</div>`;
  }
  if (mine.length) {
    html += `<div class="learning-vp-group">Your voices <span class="learning-vp-note">— may mispronounce ${escapeHtml(langLabel)} if non-native</span></div>`;
    for (const v of mine) html += row(v.name, v.label, v.color);
  }
  html += `<button type="button" class="learning-vp-row${(!_ttsVoice()) ? ' learning-vp-current' : ''}" data-voice="off">
      <span class="learning-vp-swatch learning-vp-swatch-off"></span>
      <span class="learning-vp-name">Off — no audio</span>
    </button>`;

  const pop = document.createElement('div');
  pop.id = 'learning-voice-picker';
  pop.className = 'learning-voice-picker';
  pop.innerHTML = html;
  document.body.appendChild(pop);

  const rect = anchorBtn.getBoundingClientRect();
  let top = rect.bottom + 6;
  if (top + pop.offsetHeight > window.innerHeight - 8) top = Math.max(8, rect.top - pop.offsetHeight - 6);
  let left = Math.min(rect.left, window.innerWidth - pop.offsetWidth - 8);
  left = Math.max(8, left);
  pop.style.top = `${top}px`;
  pop.style.left = `${left}px`;

  pop.querySelectorAll('.learning-vp-row').forEach((r) => {
    r.addEventListener('click', async (e) => {
      e.stopPropagation();
      const v = r.dataset.voice;
      try {
        const resp = await fetch('/api/learning/state', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ tts_voice: v }),
        });
        if (resp.ok) _state = await resp.json();
      } catch { /* keep going */ }
      pop.remove();
      if (v !== 'off' && typeof getText === 'function') _speak(getText(), anchorBtn);
    });
  });
  setTimeout(() => {
    function onDoc(e) {
      if (!pop.contains(e.target) && e.target !== anchorBtn) { pop.remove(); document.removeEventListener('mousedown', onDoc); }
    }
    document.addEventListener('mousedown', onDoc);
  }, 0);
}

// ── Chip + due tile ──────────────────────────────────────────────

async function _renderChip() {
  const chipsRow = document.querySelector('.browse-landing-chips');
  if (!chipsRow) return;
  await _fetchState();

  let chip = document.getElementById(CHIP_ID);
  if (!chip) {
    chip = document.createElement('button');
    chip.id = CHIP_ID;
    chip.type = 'button';
    chip.className = 'browse-landing-chip browse-landing-chip-learning';
    chipsRow.appendChild(chip);
    chip.addEventListener('click', _onChipClick);
  }

  const installed = _activePacks().length > 0;
  const toggleOn = installed && _state && _state.toggle === 'on';
  let dueCount = 0;
  let progSnapshot = null;
  if (toggleOn) {
    const lang = _firstActiveTarget();
    if (lang) {
      // Resolve due + progress in parallel — the chip teases both, and
      // both endpoints are user-scoped + per-lang already so there's no
      // cross-tenant risk.
      const [dueRes, prog] = await Promise.all([
        _fetchDue(lang).catch(() => ({ total: 0 })),
        _fetchProgress(lang).catch(() => null),
      ]);
      dueCount = (dueRes && dueRes.total) || 0;
      progSnapshot = prog;
    }
  }

  chip.innerHTML = '';
  const icon = document.createElement('span');
  icon.className = 'learning-chip-icon';
  icon.setAttribute('aria-hidden', 'true');
  icon.textContent = '\u{1F4D6}';
  chip.appendChild(icon);
  const label = document.createElement('span');
  label.className = 'learning-chip-label';
  label.textContent = 'Learning';
  chip.appendChild(label);
  // Soft due-indicator dot (pulses on hover) replaces the bare badge —
  // less "notification anxiety" but still readable. Count stays as the
  // numeric overlay so power users see it without hovering.
  if (dueCount > 0) {
    const dot = document.createElement('span');
    dot.className = 'learning-chip-dot';
    dot.setAttribute('aria-hidden', 'true');
    chip.appendChild(dot);
    const badge = document.createElement('span');
    badge.className = 'learning-chip-badge';
    badge.textContent = String(dueCount);
    chip.appendChild(badge);
  } else if (progSnapshot && progSnapshot.day_streak > 0) {
    // Streak-alive, nothing-due: small ember indicator so the user
    // sees the chip "knows" they have momentum.
    const ember = document.createElement('span');
    ember.className = 'learning-chip-ember';
    ember.setAttribute('aria-hidden', 'true');
    chip.appendChild(ember);
  }
  chip.dataset.toggle = toggleOn ? 'on' : 'off';
  chip.dataset.installed = installed ? '1' : '0';

  // Rich tooltip — a daily summary, not just a bare count.
  if (!installed) {
    chip.title = 'Learn a language inside Augmentum — click to get started';
  } else if (!toggleOn) {
    chip.title = 'Learning is set up — click to review or adjust';
  } else {
    const parts = [];
    parts.push(dueCount > 0
      ? `${dueCount} word${dueCount === 1 ? '' : 's'} due`
      : 'Nothing pressing');
    if (progSnapshot) {
      if (progSnapshot.day_streak > 0) parts.push(`${progSnapshot.day_streak}-day streak`);
      if (progSnapshot.cefr_estimate && progSnapshot.cefr_estimate !== '—') {
        parts.push(progSnapshot.cefr_estimate);
      }
    }
    chip.title = parts.join(' · ');
  }

  // Stage-aware tile rendering: foundation learners get gentler
  // language ("ready" not "due") and skip the progress tile entirely
  // (it's all zeros for them). Active learners get the full chrome.
  const stage = _learnerStage(progSnapshot);
  _renderDueTile(dueCount, stage);
  if (installed && toggleOn) _renderProgressTile().catch(() => {});
  else {
    const t = document.getElementById(PROGRESS_TILE_ID);
    if (t) t.remove();
  }
}

function _renderDueTile(dueCount, stage = 'active') {
  const landing = document.getElementById('browse-landing');
  let tile = document.getElementById(TILE_ID);
  if (!landing || dueCount <= 0) {
    if (tile) tile.remove();
    return;
  }
  if (!tile) {
    tile = document.createElement('button');
    tile.id = TILE_ID;
    tile.type = 'button';
    tile.className = 'learning-due-tile';
    // Sit just under the chip row.
    const chipsRow = landing.querySelector('.browse-landing-chips');
    if (chipsRow && chipsRow.parentElement) {
      chipsRow.parentElement.insertBefore(tile, chipsRow.nextSibling);
    } else {
      landing.appendChild(tile);
    }
    tile.addEventListener('click', () => {
      const lang = _firstActiveTarget();
      if (lang) _openReviewOverlay(lang);
    });
  }
  const mins = Math.max(1, Math.ceil(dueCount * 0.15));
  // "ready" frames the cards as a *path* for a new learner; "due" is
  // owed-language reserved for users who've earned a habit ("you have a
  // schedule and X of it is waiting"). Same number, different mood.
  const verb = stage === 'foundation' ? 'ready' : 'due';
  tile.dataset.stage = stage;
  tile.innerHTML =
    `<span class="learning-due-count">${dueCount} word${dueCount === 1 ? '' : 's'} ${verb}</span>` +
    `<span class="learning-due-est">~${mins} min</span>`;
}

// ── Progress card + leech surface ─────────────────────────────────

// "Foundation" learners are within the first few sessions of a pack:
// total > 0 but nothing has graduated to settled yet, and the streak
// hasn't compounded past day 1. Day-1 chrome should celebrate the
// beginning, not measure the absence. "Active" learners are everyone
// past that — they earn the full dashboard / mastery breakdown.
function _learnerStage(prog) {
  if (!prog) return 'foundation';
  const settled = prog.settled || 0;
  const streak = prog.day_streak || 0;
  if (settled === 0 && streak <= 1) return 'foundation';
  return 'active';
}

// Hand-curated learning paths live server-side and are exposed via
// /api/learning/paths/{lang}. We cache the summary per-session — units
// don't change between page loads, so re-fetching is wasteful.
const _PATH_CACHE = new Map();
async function _fetchPath(lang) {
  if (_PATH_CACHE.has(lang)) return _PATH_CACHE.get(lang);
  try {
    const r = await fetch(`/api/learning/paths/${encodeURIComponent(lang)}`);
    if (r.status === 404) { _PATH_CACHE.set(lang, null); return null; }
    if (!r.ok) return null;
    const data = await r.json();
    _PATH_CACHE.set(lang, data);
    return data;
  } catch {
    _PATH_CACHE.set(lang, null);
    return null;
  }
}

// Pick the unit the learner is currently on. Foundation learners get
// unit 1; everyone else gets the unit just past their settled-word
// count. This is a coarse heuristic — units carry roughly equal vocab
// load (~20 words each) so settled / 20 is a fine approximation. Real
// per-unit completion tracking is a follow-up.
function _currentUnit(path, prog) {
  if (!path || !path.levels || !path.levels[0]) return null;
  const units = path.levels[0].units || [];
  if (units.length === 0) return null;
  const settled = (prog && prog.settled) || 0;
  // Rough "20 graduated words per unit" assumption — replace with real
  // tracking once we instrument per-unit completion.
  const approxIdx = Math.min(units.length - 1, Math.floor(settled / 20));
  return { unit: units[approxIdx], index: approxIdx, total: units.length };
}

async function _fetchProgress(lang) {
  try {
    const r = await fetch(`/api/learning/vocab/progress?lang=${encodeURIComponent(lang)}`);
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

async function _fetchLeeches(lang, limit = 30) {
  try {
    const r = await fetch(`/api/learning/vocab/leeches?lang=${encodeURIComponent(lang)}&limit=${limit}`);
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

const PROGRESS_TILE_ID = 'browse-landing-learning-progress';

async function _renderProgressTile() {
  const landing = document.getElementById('browse-landing');
  if (!landing) return;
  const lang = _firstActiveTarget();
  let tile = document.getElementById(PROGRESS_TILE_ID);
  if (!lang) { if (tile) tile.remove(); return; }
  const prog = await _fetchProgress(lang);
  if (!prog || prog.total === 0) {
    if (tile) tile.remove();
    return;
  }
  // Foundation learners (day 1-3 with no settled vocab yet) get a
  // quieter landing — the progress tile would be 0/0/0/— and reads as
  // a deficit. Hide it until the learner has earned at least one
  // settled card or a multi-day streak.
  if (_learnerStage(prog) === 'foundation') {
    if (tile) tile.remove();
    return;
  }
  if (!tile) {
    tile = document.createElement('button');
    tile.id = PROGRESS_TILE_ID;
    tile.type = 'button';
    tile.className = 'learning-progress-tile';
    const dueTile = document.getElementById(TILE_ID);
    const anchor = dueTile || landing.querySelector('.browse-landing-chips');
    if (anchor && anchor.parentElement) {
      anchor.parentElement.insertBefore(tile, anchor.nextSibling);
    } else {
      landing.appendChild(tile);
    }
    tile.addEventListener('click', () => _openProgressOverlay(lang));
  }
  const streakBadge = prog.day_streak > 0
    ? `<span class="learning-progress-streak">🔥 ${prog.day_streak}d</span>`
    : '';
  tile.innerHTML = `
    <div class="learning-progress-cefr">${escapeHtml(prog.cefr_estimate || '—')}</div>
    <div class="learning-progress-body">
      <div class="learning-progress-count">${prog.settled} settled · ${prog.total} total</div>
      <div class="learning-progress-meta">
        <span>${prog.counts.mature || 0} mature</span>
        <span>${prog.counts.leech || 0} leech</span>
        ${streakBadge}
      </div>
    </div>`;
}

async function _openProgressOverlay(lang) {
  const { overlay, close } = _makeOverlay('learning-progress-overlay');
  overlay.innerHTML = `
    <div class="modal learning-progress-modal">
      <div class="modal-body">
        <div class="learning-progress-loading">Loading…</div>
      </div>
    </div>`;
  const [prog, leech] = await Promise.all([
    _fetchProgress(lang),
    _fetchLeeches(lang, 30),
  ]);
  if (!prog) {
    overlay.querySelector('.modal-body').innerHTML =
      `<p>Couldn't load progress for ${escapeHtml(_langLabel(lang))}.</p>
       <button type="button" class="btn btn-primary" id="learning-progress-close">Close</button>`;
    overlay.querySelector('#learning-progress-close').addEventListener('click', close);
    return;
  }
  const c = prog.counts || {};
  const leeches = (leech && leech.leeches) || [];
  const stage = _learnerStage(prog);
  // Proper close button (SVG, styled), same shape the hub uses — replaces
  // the unstyled `×` text character that looked half-finished.
  const closeBtn = `
    <button type="button" class="learning-progress-close-btn" id="learning-progress-close" aria-label="Close">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M6 6l12 12M18 6L6 18"/></svg>
    </button>`;

  if (stage === 'foundation') {
    // Foundation = day 1-3, nothing settled, no streak compounded yet.
    // Showing 5 zeros + "—" CEFR + "0 reviews · 0 day streak" reads as
    // a deficit. Instead: name where they are, list what unlocks, and
    // point them at today's action.
    const totalSeeded = prog.total || 0;
    const dayLabel = prog.day_streak > 0
      ? `Day ${prog.day_streak}`
      : 'Day 1';
    overlay.querySelector('.modal-body').innerHTML = `
      <div class="learning-progress-head">
        <div>
          <div class="learning-progress-day-marker">${escapeHtml(dayLabel)}</div>
          <div class="learning-progress-sub">${escapeHtml(_langLabel(lang))} · your foundation</div>
        </div>
        ${closeBtn}
      </div>
      <div class="learning-progress-foundation-stats">
        <div class="learning-progress-fstat">
          <div class="learning-progress-fstat-n">${totalSeeded}</div>
          <div class="learning-progress-fstat-lbl">in your garden</div>
        </div>
        <div class="learning-progress-fstat">
          <div class="learning-progress-fstat-n">${(c.learning || 0) + (c.new || 0)}</div>
          <div class="learning-progress-fstat-lbl">in rotation</div>
        </div>
        <div class="learning-progress-fstat">
          <div class="learning-progress-fstat-n">${prog.last_7_days_reviews || 0}</div>
          <div class="learning-progress-fstat-lbl">reviews this week</div>
        </div>
      </div>
      <h3 class="learning-progress-section">What unlocks as you practice</h3>
      <ul class="learning-progress-unlock-list">
        <li>
          <div class="learning-progress-unlock-icon">●</div>
          <div class="learning-progress-unlock-body">
            <div class="learning-progress-unlock-title">Settled vocabulary</div>
            <div class="learning-progress-unlock-sub">Cards graduate to "settled" once you've recalled them correctly several times. Settled count drives your CEFR / proficiency estimate.</div>
          </div>
        </li>
        <li>
          <div class="learning-progress-unlock-icon">●</div>
          <div class="learning-progress-unlock-body">
            <div class="learning-progress-unlock-title">Trouble words</div>
            <div class="learning-progress-unlock-sub">Any word you forget twice goes here. Empty for now — show up and that stays true.</div>
          </div>
        </li>
        <li>
          <div class="learning-progress-unlock-icon">●</div>
          <div class="learning-progress-unlock-body">
            <div class="learning-progress-unlock-title">Streak</div>
            <div class="learning-progress-unlock-sub">A day counts the first time you grade a card. ${prog.day_streak > 0 ? 'Today already counts.' : 'Review one card to start.'}</div>
          </div>
        </li>
        <li>
          <div class="learning-progress-unlock-icon">●</div>
          <div class="learning-progress-unlock-body">
            <div class="learning-progress-unlock-title">Proficiency band</div>
            <div class="learning-progress-unlock-sub">Appears once you've settled enough vocabulary to map onto a CEFR / HSK / JLPT band. Honest signal — won't show until you've earned it.</div>
          </div>
        </li>
      </ul>
      <button type="button" class="btn btn-primary learning-progress-action" id="learning-progress-startaction">Start today's review →</button>`;
    overlay.querySelector('#learning-progress-close').addEventListener('click', close);
    overlay.querySelector('#learning-progress-startaction').addEventListener('click', () => {
      close();
      _openReviewOverlay(lang);
    });
    return;
  }

  // Active learner — the full dashboard.
  const leechRows = leeches.length
    ? leeches.map(w => `
        <li class="learning-leech-row" data-id="${escapeHtml(w.word_id)}">
          <div class="learning-leech-head">
            <span class="learning-leech-surface">${escapeHtml(w.surface)}</span>
            <span class="learning-leech-reading">${escapeHtml(w.reading || '')}</span>
            <span class="learning-leech-tag">${escapeHtml(w.mastery_state)} · ${w.lapses} lapse${w.lapses === 1 ? '' : 's'}</span>
          </div>
          <div class="learning-leech-gloss">${escapeHtml(((w.glosses || []).slice(0, 3)).join(' · '))}</div>
          ${w.example && w.example.lang_text ? `
            <div class="learning-leech-example">
              ${escapeHtml(w.example.lang_text)}
              ${w.example.en_text ? `<span>${escapeHtml(w.example.en_text)}</span>` : ''}
            </div>` : ''}
          <div class="learning-leech-grade">
            <button type="button" data-g="1" class="lg-wg-g lg-wg-g-1">Forgot</button>
            <button type="button" data-g="2" class="lg-wg-g lg-wg-g-2">Hard</button>
            <button type="button" data-g="3" class="lg-wg-g lg-wg-g-3">Good</button>
            <button type="button" data-g="4" class="lg-wg-g lg-wg-g-4">Easy</button>
          </div>
        </li>`).join('')
    : `<li class="learning-leech-empty">No trouble words right now — keep it that way.</li>`;
  overlay.querySelector('.modal-body').innerHTML = `
    <div class="learning-progress-head">
      <div>
        <div class="learning-progress-cefr-big">${escapeHtml(prog.cefr_estimate || '—')}</div>
        <div class="learning-progress-sub">${escapeHtml(_langLabel(lang))} · ${prog.settled} settled words</div>
      </div>
      ${closeBtn}
    </div>
    <div class="learning-progress-stats">
      <div class="learning-progress-stat"><div class="learning-progress-stat-n">${c.mature || 0}</div><div>mature</div></div>
      <div class="learning-progress-stat"><div class="learning-progress-stat-n">${c.reviewing || 0}</div><div>reviewing</div></div>
      <div class="learning-progress-stat"><div class="learning-progress-stat-n">${c.learning || 0}</div><div>learning</div></div>
      <div class="learning-progress-stat"><div class="learning-progress-stat-n">${c.new || 0}</div><div>new</div></div>
      <div class="learning-progress-stat learning-progress-stat-leech"><div class="learning-progress-stat-n">${c.leech || 0}</div><div>leech</div></div>
    </div>
    <div class="learning-progress-meta-row">
      <span><b>${prog.last_7_days_reviews}</b> reviews · last 7 days</span>
      <span><b>${prog.day_streak}</b> day streak ${prog.day_streak > 0 ? '🔥' : ''}</span>
    </div>
    <h3 class="learning-progress-section">Trouble words</h3>
    <ul class="learning-leech-list">${leechRows}</ul>`;
  overlay.querySelector('#learning-progress-close').addEventListener('click', close);
  overlay.querySelectorAll('.learning-leech-row').forEach(row => {
    const wordId = row.dataset.id;
    row.querySelectorAll('.lg-wg-g').forEach(btn => {
      btn.addEventListener('click', async () => {
        const g = Number(btn.dataset.g);
        try {
          await fetch('/api/learning/srs/grade', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ lang, word_id: wordId, grade: g }),
          });
        } catch { /* */ }
        row.style.opacity = '0.4';
        row.style.pointerEvents = 'none';
        // Re-fetch progress so the counts on the head update live.
        const fresh = await _fetchProgress(lang);
        if (fresh) {
          overlay.querySelector('.learning-progress-cefr-big').textContent = fresh.cefr_estimate || '—';
          overlay.querySelector('.learning-progress-sub').textContent =
            `${_langLabel(lang)} · ${fresh.settled} settled words`;
        }
      });
    });
  });
}

async function _onChipClick() {
  await _fetchState();
  const installed = _activePacks().length > 0;
  if (!installed) { _openLanguagePicker(); return; }
  if (!_state || _state.toggle !== 'on') { _openOnboardingModal(); return; }
  const lang = _firstActiveTarget();
  if (lang) _openLearningHub(lang);
}

// ── Learning hub ─────────────────────────────────────────────────
//
// The chip used to dump the user straight into SRS flashcards, which
// felt jarring once the surface grew. The hub is the new landing — a
// measured, library-like daily-companion space. Per-lang accent
// cascades through the whole panel, the greeting renders in the
// target language so the immersion starts at glance, and a single
// "today's path" featured action focuses the eye before the secondary
// grid of practice modes. All actions dispatch to existing overlays.

// Per-target-lang accent identity. Each lang gets a quiet warm color
// that swaps the hub's accent variable when you select that lang, so
// the surface feels like *this language* rather than a generic chrome.
const _LANG_ACCENT = {
  ja: { accent: '#5b7ab8', soft: 'rgba(91,122,184,0.14)' },   // indigo — like inkwash
  es: { accent: '#c79a4d', soft: 'rgba(199,154,77,0.16)' },   // ochre — sun on terracotta
  zh: { accent: '#c84d40', soft: 'rgba(200,77,64,0.14)' },    // cinnabar — ink seal
  fr: { accent: '#3d6a8c', soft: 'rgba(61,106,140,0.16)' },   // fauve blue — postcard
  ko: { accent: '#6fa17e', soft: 'rgba(111,161,126,0.16)' },  // jade — celadon
  en: { accent: '#7a8a9c', soft: 'rgba(122,138,156,0.16)' },
};

// Time-of-day greetings in each target lang. Time bands chosen for the
// learner's local clock — `Date.getHours()` is on the device, which is
// the right behaviour (the LLM doesn't enter this loop).
const _LANG_GREETING = {
  ja: { morning: 'おはよう', day: 'こんにちは', evening: 'こんばんは' },
  es: { morning: 'Buenos días', day: 'Buenas tardes', evening: 'Buenas noches' },
  zh: { morning: '早安', day: '你好', evening: '晚上好' },
  fr: { morning: 'Bonjour', day: 'Bon après-midi', evening: 'Bonsoir' },
  ko: { morning: '좋은 아침', day: '안녕하세요', evening: '안녕히 주무세요' },
  en: { morning: 'Good morning', day: 'Hello', evening: 'Good evening' },
};

function _greetingFor(lang) {
  const g = _LANG_GREETING[lang] || _LANG_GREETING.en;
  const h = new Date().getHours();
  if (h < 12) return g.morning;
  if (h < 18) return g.day;
  return g.evening;
}

// Stroke SVG icons (no emoji — emoji rendering varies wildly by OS and
// breaks the typographic voice of the surface). 24×24, currentColor.
const _HUB_ICONS = {
  review: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><rect x="4" y="5" width="14" height="14" rx="2"/><rect x="7" y="8" width="14" height="14" rx="2"/></svg>',
  games:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="12" cy="12" r="9"/><path d="M9 9l6 3-6 3v-6z" fill="currentColor" stroke="none"/></svg>',
  companion: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M4 6a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H9l-3 3v-3H6a2 2 0 0 1-2-2V6z"/><path d="M11 11h7a2 2 0 0 1 2 2v4a2 2 0 0 1-2 2h-1v2l-2-2h-4a2 2 0 0 1-2-2"/></svg>',
  progress: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M4 19V8"/><path d="M10 19V13"/><path d="M16 19V5"/><path d="M3 20h18"/></svg>',
  leech:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><path d="M12 4l1.8 5.5h5.7l-4.6 3.4 1.7 5.4L12 14.9l-4.6 3.4 1.7-5.4-4.6-3.4h5.7L12 4z"/></svg>',
  settings: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/></svg>',
  arrow:  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
};

async function _openLearningHub(initialLang, opts = {}) {
  const welcome = !!opts.welcome;
  const { overlay, close } = _makeOverlay('learning-hub-overlay');
  const targets = _activePacks();
  if (targets.length === 0) { close(); _openLanguagePicker(); return; }

  // Reuse the target_langs order from the user's profile so the hub's
  // primary lang matches what `_firstActiveTarget()` would pick.
  const orderedTargets = (_state && Array.isArray(_state.target_langs))
    ? _state.target_langs.filter(lc => targets.some(p => p.lang_code === lc))
    : targets.map(p => p.lang_code);
  for (const p of targets) {
    if (!orderedTargets.includes(p.lang_code)) orderedTargets.push(p.lang_code);
  }
  let currentLang = orderedTargets.includes(initialLang) ? initialLang : orderedTargets[0];

  // Render the modal shell — the body fills in per-lang on each switch.
  overlay.innerHTML = `
    <div class="modal lh-modal" id="lh-modal" data-lang="${escapeHtml(currentLang)}">
      <button type="button" class="lh-close" id="lh-close" aria-label="Close">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M6 6l12 12M18 6L6 18"/></svg>
      </button>
      <div class="lh-greeting" id="lh-greeting"></div>
      <div class="lh-meta" id="lh-meta"></div>
      <div class="lh-langs" id="lh-langs"></div>
      <div class="lh-stats" id="lh-stats">
        <div class="lh-stat-loading">·  ·  ·</div>
      </div>
      <div class="lh-featured" id="lh-featured"></div>
      <div class="lh-grid" id="lh-grid"></div>
    </div>`;

  const modalEl = overlay.querySelector('#lh-modal');
  overlay.querySelector('#lh-close').addEventListener('click', close);

  const greetingEl = overlay.querySelector('#lh-greeting');
  const metaEl = overlay.querySelector('#lh-meta');
  const langsEl = overlay.querySelector('#lh-langs');
  const statsEl = overlay.querySelector('#lh-stats');
  const featuredEl = overlay.querySelector('#lh-featured');
  const gridEl = overlay.querySelector('#lh-grid');

  function applyLangAccent(lc) {
    const tone = _LANG_ACCENT[lc] || _LANG_ACCENT.en;
    modalEl.style.setProperty('--lh-accent', tone.accent);
    modalEl.style.setProperty('--lh-accent-soft', tone.soft);
    modalEl.dataset.lang = lc;
  }

  function renderHeader(stage, prog) {
    greetingEl.textContent = _greetingFor(currentLang);
    const today = new Date().toLocaleDateString(undefined, {
      weekday: 'long', month: 'long', day: 'numeric',
    });
    // Foundation learners get a "day N" marker in the subtitle —
    // celebrates *days into the journey* rather than a date stamp.
    if (stage === 'foundation' && prog) {
      const day = Math.max(1, prog.day_streak || 1);
      metaEl.textContent = `${_langLabel(currentLang)} · Day ${day}`;
    } else {
      metaEl.textContent = `${_langLabel(currentLang)} · ${today}`;
    }
  }

  function renderLangs() {
    if (orderedTargets.length <= 1) { langsEl.innerHTML = ''; return; }
    langsEl.innerHTML = orderedTargets.map(lc => `
      <button type="button" class="lh-lang${lc === currentLang ? ' active' : ''}"
              data-lang="${escapeHtml(lc)}" title="${escapeHtml(_langLabel(lc))}">
        <span class="lh-lang-code">${escapeHtml(lc)}</span>
        <span class="lh-lang-name">${escapeHtml(_langLabel(lc))}</span>
      </button>
    `).join('');
    langsEl.querySelectorAll('.lh-lang').forEach(btn => {
      btn.addEventListener('click', () => {
        currentLang = btn.dataset.lang;
        applyLangAccent(currentLang);
        renderHeader();
        renderLangs();
        renderBody();
      });
    });
  }

  function renderStats(due, prog, stage) {
    const settled = prog ? prog.settled : 0;
    const total = prog ? prog.total : 0;
    const streak = prog ? prog.day_streak : 0;
    const last7 = prog ? prog.last_7_days_reviews : 0;
    const level = prog && prog.cefr_estimate ? prog.cefr_estimate : '—';
    // Foundation: 3 cells that all CAN show a real number day 1 (ready /
    // vocab / day). Hide CEFR + 7-day since those start at "—" / 0 and
    // read as a punishment. Active: full 5-cell library card.
    const cells = stage === 'foundation'
      ? [
          { n: due, label: 'ready today', accent: due > 0 },
          { n: total, label: 'in your garden' },
          { n: streak, label: streak === 1 ? 'day in' : 'days', accent: streak > 0 },
        ]
      : [
          { n: due, label: 'due today', accent: due > 0 },
          { n: settled, label: 'lexicon' },
          { n: streak, label: 'days', accent: streak > 0 },
          { n: last7, label: '7-day reviews' },
          { n: level, label: 'level', isText: true },
        ];
    statsEl.dataset.cells = String(cells.length);
    statsEl.innerHTML = cells.map(c => `
      <div class="lh-stat${c.accent ? ' accent' : ''}">
        <div class="lh-stat-n${c.isText ? ' text' : ''}">${escapeHtml(String(c.n))}</div>
        <div class="lh-stat-lbl">${escapeHtml(c.label)}</div>
      </div>`).join('');
  }

  async function _confirmReseed() {
    if (!confirm(
      `Reset your ${_langLabel(currentLang)} queue and start fresh from the curated curriculum?\n\n` +
      `Your existing review schedule for ${_langLabel(currentLang)} will be cleared. ` +
      `This is helpful if your queue was seeded with low-value entries (homophones, function words).`
    )) return;
    try {
      const r = await fetch(`/api/learning/packs/${encodeURIComponent(currentLang)}/reseed`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: 30, confirm: true }),
      });
      if (!r.ok) { showToast('Reseed failed', 'warning'); return; }
      const j = await r.json();
      showToast(
        `Reset ${j.cleared} old entries · seeded ${j.seeded} from the path`,
        'success',
      );
      // Refresh hub data so the new stage/featured card show immediately.
      _state = null;
      await _fetchState();
      await _renderChip();
      renderBody();
    } catch {
      showToast('Reseed failed', 'warning');
    }
  }

  function renderFeatured(due, prog, stage, pathInfo) {
    // Today's primary path: due → review; nothing due but streak alive →
    // companion; cold start → games. The framing changes per state so
    // the hub feels responsive to what the learner has earned today.
    let card;
    if (due > 0) {
      const mins = Math.max(1, Math.ceil(due * 0.15));
      // Curated-path framing — much better than generic "Review N cards"
      // because it names *what* they're learning today. Falls through to
      // the old framing when no path ships for this language.
      if (pathInfo && pathInfo.unit) {
        const u = pathInfo.unit;
        const isFirstDay = !prog || (prog.day_streak || 0) < 1;
        card = {
          eyebrow: `${isFirstDay ? 'Begin' : 'Continue'} · Unit ${pathInfo.index + 1} of ${pathInfo.total}`,
          title: u.title,
          sub: `${u.goal} · ${due} word${due === 1 ? '' : 's'} ready`,
          handler: () => { close(); _openReviewOverlay(currentLang); },
        };
      } else if (stage === 'foundation') {
        const isFirstDay = !prog || (prog.day_streak || 0) < 1;
        card = {
          eyebrow: isFirstDay ? 'Begin' : 'Continue',
          title: isFirstDay ? `Start your first session` : `Continue today's session`,
          sub: `${due} word${due === 1 ? '' : 's'} ready · ~${mins} min`,
          handler: () => { close(); _openReviewOverlay(currentLang); },
        };
      } else {
        card = {
          eyebrow: 'Today',
          title: `Review ${due} card${due === 1 ? '' : 's'}`,
          sub: `About ${mins} minute${mins === 1 ? '' : 's'} · keeps your schedule`,
          handler: () => { close(); _openReviewOverlay(currentLang); },
        };
      }
    } else if (prog && prog.day_streak > 0) {
      card = {
        eyebrow: `Day ${prog.day_streak}`,
        title: 'Talk with your partner',
        sub: 'A persistent character chat that remembers the thread',
        handler: async () => {
          close();
          const { openLanguagePartner } = await import('./learning_games/partner_launch.js');
          await openLanguagePartner(currentLang);
        },
      };
    } else {
      card = {
        eyebrow: 'Begin',
        title: 'Play a learning game',
        sub: 'Focused drills, stories, and review games',
        handler: async () => {
          close();
          const mod = await import('./learning_games/hub.js');
          mod.openGamesHub({ lang: currentLang, voice: _ttsVoice() });
        },
      };
    }
    featuredEl.innerHTML = `
      <button type="button" class="lh-feat" aria-label="${escapeHtml(card.title)}">
        <div class="lh-feat-text">
          <div class="lh-feat-eye">${escapeHtml(card.eyebrow)}</div>
          <div class="lh-feat-title">${escapeHtml(card.title)}</div>
          <div class="lh-feat-sub">${escapeHtml(card.sub)}</div>
        </div>
        <div class="lh-feat-arrow">${_HUB_ICONS.arrow}</div>
      </button>`;
    featuredEl.querySelector('.lh-feat').addEventListener('click', card.handler);
  }

  function renderGrid(due, prog, pathInfo) {
    // The grid carries the rest of the verbs — secondary tier. We omit
    // whichever action the featured card surfaced (so we don't show
    // Review twice when 12 are due).
    const featuredKey = due > 0 ? 'review' : (prog && prog.day_streak > 0 ? 'companion' : 'games');
    const all = [
      {
        key: 'review', icon: 'review', title: 'Review',
        sub: due > 0 ? `${due} due` : 'Free practice',
        handler: () => { close(); _openReviewOverlay(currentLang); },
      },
      {
        key: 'games', icon: 'games', title: 'Games',
        sub: 'Drills and stories',
        handler: async () => {
          close();
          const mod = await import('./learning_games/hub.js');
          mod.openGamesHub({ lang: currentLang, voice: _ttsVoice() });
        },
      },
      {
        key: 'companion', icon: 'companion', title: 'Partner',
        sub: 'Narrative character chat',
        handler: async () => {
          close();
          const { openLanguagePartner } = await import('./learning_games/partner_launch.js');
          await openLanguagePartner(currentLang);
        },
      },
      {
        key: 'progress', icon: 'progress', title: 'Progress',
        sub: prog ? `${prog.settled} lexicon · ${prog.day_streak}d streak` : 'No progress yet',
        handler: () => { close(); _openProgressOverlay(currentLang); },
      },
      {
        key: 'leech', icon: 'leech', title: 'Trouble words',
        sub: prog && prog.counts && prog.counts.leech > 0
          ? `${prog.counts.leech} need attention`
          : 'None yet',
        accent: !!(prog && prog.counts && prog.counts.leech > 0),
        handler: () => { close(); _openProgressOverlay(currentLang); },
      },
      {
        key: 'settings', icon: 'settings', title: 'Settings',
        sub: 'Languages, voice, level',
        handler: () => { close(); _openOnboardingModal(); },
      },
    ];
    // When a curated path exists, surface the "Reset & reseed" action.
    // It's destructive so it lives in the grid (not the featured slot)
    // and the click handler always prompts before wiping the queue.
    if (pathInfo) {
      all.push({
        key: 'reseed', icon: 'leech', title: 'Reset to curriculum',
        sub: 'Wipe queue · start from Unit 1',
        handler: () => _confirmReseed(),
      });
    }
    const cards = all.filter(c => c.key !== featuredKey);
    gridEl.innerHTML = cards.map(c => `
      <button type="button" class="lh-card${c.accent ? ' accent' : ''}" data-key="${escapeHtml(c.key)}">
        <div class="lh-card-icon">${_HUB_ICONS[c.icon]}</div>
        <div class="lh-card-text">
          <div class="lh-card-title">${escapeHtml(c.title)}</div>
          <div class="lh-card-sub">${escapeHtml(c.sub)}</div>
        </div>
      </button>`).join('');
    gridEl.querySelectorAll('.lh-card').forEach(btn => {
      const card = cards.find(c => c.key === btn.dataset.key);
      if (card) btn.addEventListener('click', card.handler);
    });
  }

  async function renderBody() {
    statsEl.innerHTML = `<div class="lh-stat-loading">·  ·  ·</div>`;
    featuredEl.innerHTML = '';
    gridEl.innerHTML = '';
    const [dueRes, prog, path] = await Promise.all([
      _fetchDue(currentLang, 1).catch(() => ({ total: 0 })),
      _fetchProgress(currentLang).catch(() => null),
      _fetchPath(currentLang).catch(() => null),
    ]);
    const due = (dueRes && dueRes.total) || 0;
    const stage = _learnerStage(prog);
    const pathInfo = _currentUnit(path, prog);
    modalEl.dataset.stage = stage;
    if (path) modalEl.dataset.hasPath = '1'; else delete modalEl.dataset.hasPath;
    renderHeader(stage, prog);
    renderStats(due, prog, stage);
    renderFeatured(due, prog, stage, pathInfo);
    renderGrid(due, prog, pathInfo);
  }

  applyLangAccent(currentLang);
  renderHeader('active', null);
  renderLangs();

  // Welcome overlay: one-time celebratory layer that fires right after
  // onboarding completes. Tap-anywhere dismisses, the hub stays open
  // with the featured action focused.
  if (welcome) {
    const banner = document.createElement('div');
    banner.className = 'lh-welcome';
    banner.innerHTML = `
      <div class="lh-welcome-card">
        <div class="lh-welcome-eye">Welcome</div>
        <div class="lh-welcome-title">Your ${escapeHtml(_langLabel(currentLang))} practice starts here.</div>
        <div class="lh-welcome-sub">Tap anywhere to begin.</div>
      </div>`;
    modalEl.appendChild(banner);
    const dismiss = () => { banner.classList.add('lh-welcome-out'); setTimeout(() => banner.remove(), 280); };
    banner.addEventListener('click', dismiss);
    setTimeout(() => banner.classList.add('lh-welcome-in'), 20);
  }

  renderBody();
}

// ── Modal helpers ────────────────────────────────────────────────

function _makeOverlay(id) {
  const old = document.getElementById(id);
  if (old) old.remove();
  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  overlay.id = id;
  const close = () => { overlay.remove(); document.removeEventListener('keydown', onKey); };
  function onKey(e) { if (e.key === 'Escape') close(); }
  document.addEventListener('keydown', onKey);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
  document.body.appendChild(overlay);
  return { overlay, close, onKey };
}

// ── Language picker + install ────────────────────────────────────

async function _openLanguagePicker() {
  const { overlay, close } = _makeOverlay('learning-picker-overlay');
  overlay.innerHTML = `
    <div class="modal learning-modal">
      <div class="modal-header"><h3 class="modal-title">Pick a language to learn</h3></div>
      <div class="modal-body">
        <p class="learning-modal-intro">
          Each pack is built locally from open dictionaries — your progress
          never leaves this machine. The first build downloads a corpus and
          takes a few minutes.
        </p>
        <div class="learning-picker-list" id="learning-picker-list">
          <div class="learning-picker-loading">Loading…</div>
        </div>
      </div>
      <div class="modal-footer"><button type="button" class="btn btn-ghost" id="learning-picker-close">Close</button></div>
    </div>`;
  overlay.querySelector('#learning-picker-close').addEventListener('click', close);

  let catalog = [];
  try {
    const r = await fetch('/api/learning/packs/catalog');
    catalog = r.ok ? (await r.json()).packs || [] : [];
  } catch { /* leave empty */ }

  const list = overlay.querySelector('#learning-picker-list');
  if (!list) return;
  if (!catalog.length) {
    list.innerHTML = '<div class="learning-picker-empty">Couldn’t load the catalog.</div>';
    return;
  }
  list.innerHTML = '';
  for (const p of catalog) {
    const row = document.createElement('div');
    row.className = 'learning-picker-row';
    row.dataset.lang = p.lang_code;
    const name = escapeHtml(p.name || _langLabel(p.lang_code));
    let actionHtml;
    if (p.installed) {
      actionHtml = `<button type="button" class="btn btn-secondary learning-picker-action" data-act="setup">Set up</button>`;
    } else if (p.install_job_id) {
      actionHtml = `<div class="learning-install-progress" data-job="${escapeHtml(p.install_job_id)}">
        <div class="learning-install-bar"><div class="learning-install-bar-fill" style="width:0%"></div></div>
        <span class="learning-install-stage">installing…</span></div>`;
    } else if (p.installable) {
      actionHtml = `<button type="button" class="btn btn-primary learning-picker-action" data-act="install">Install · ~${p.approx_download_mb}&nbsp;MB</button>`;
    } else {
      actionHtml = `<span class="learning-picker-soon">Coming soon</span>`;
    }
    row.innerHTML = `
      <div class="learning-picker-meta">
        <span class="learning-picker-name">${name}</span>
        ${p.status === 'planned' ? '<span class="learning-picker-tag">planned</span>' : ''}
      </div>${actionHtml}`;
    list.appendChild(row);

    if (p.install_job_id) _pollInstall(p.lang_code, p.install_job_id, row, close);
    const act = row.querySelector('.learning-picker-action');
    if (act) act.addEventListener('click', () => {
      const which = act.dataset.act;
      if (which === 'setup') { close(); _openOnboardingModal(); }
      else if (which === 'install') _startInstall(p.lang_code, row, close);
    });
  }
}

async function _startInstall(lang, rowEl, closePicker) {
  rowEl.querySelector('.learning-picker-action')?.remove();
  rowEl.querySelector('.learning-picker-soon')?.remove();
  const wrap = document.createElement('div');
  wrap.className = 'learning-install-progress';
  wrap.innerHTML = `<div class="learning-install-bar"><div class="learning-install-bar-fill" style="width:0%"></div></div>
    <span class="learning-install-stage">starting…</span>`;
  rowEl.appendChild(wrap);
  try {
    const r = await fetch(`/api/learning/packs/${encodeURIComponent(lang)}/install`, { method: 'POST' });
    if (!r.ok) throw new Error(`install failed: ${r.status}`);
    const { job_id } = await r.json();
    _pollInstall(lang, job_id, rowEl, closePicker);
  } catch {
    wrap.innerHTML = '<span class="learning-install-error">Couldn’t start install</span>';
  }
}

async function _pollInstall(lang, jobId, rowEl, closePicker) {
  const fill = rowEl.querySelector('.learning-install-bar-fill');
  const stage = rowEl.querySelector('.learning-install-stage');
  let stopped = false;
  async function tick() {
    if (stopped || !document.body.contains(rowEl)) return;
    try {
      const r = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
      if (r.ok) {
        const j = await r.json();
        const job = j.job || j;
        if (fill) fill.style.width = `${Math.round((job.progress || 0) * 100)}%`;
        if (stage) stage.textContent = job.stage || job.status || 'working…';
        if (job.status === 'completed') {
          stopped = true;
          if (stage) stage.textContent = 'done';
          showToast(`${_langLabel(lang)} pack installed`, 'success');
          _state = null;
          await _renderChip();
          if (typeof closePicker === 'function') closePicker();
          _openOnboardingModal();
          return;
        }
        if (job.status === 'failed' || job.status === 'cancelled') {
          stopped = true;
          if (stage) stage.textContent = job.status === 'failed' ? 'install failed' : 'cancelled';
          rowEl.querySelector('.learning-install-bar')?.classList.add('learning-install-bar-error');
          return;
        }
      }
    } catch { /* transient — keep polling */ }
    setTimeout(tick, 2000);
  }
  tick();
}

// ── Onboarding modal ─────────────────────────────────────────────

// Module-level cache so we don't refetch on every modal open / re-render.
let _kokoroVoicesCache = null;
async function _fetchKokoroVoices() {
  if (_kokoroVoicesCache) return _kokoroVoicesCache;
  try {
    const r = await fetch('/api/audio/voices');
    if (r.ok) {
      const j = await r.json();
      _kokoroVoicesCache = Array.isArray(j) ? j : (j.voices || []);
    }
  } catch { /* leave null; we'll degrade to Off-only */ }
  return _kokoroVoicesCache || [];
}

function _ttsOptionsHtml(voices, selectedLangs, currentVoice) {
  // Build language-grouped <option>s filtered to whatever target langs
  // are checked right now. Falls back to "Off only" when no native
  // Kokoro voices exist for any selected target language.
  const groups = [];   // [{lang, voices: [{name,label,grade}]}]
  for (const lang of selectedLangs) {
    const filtered = [];
    for (const v of voices) {
      const name = v.id || v.name || v.voice || '';
      if (!name) continue;
      const isMix = !!(v.blend_spec || v.kind === 'mix' || v.type === 'mix' || v.is_mix);
      if (isMix) continue;
      const voiceLang = (v.lang || '').toLowerCase();
      if (voiceLang.startsWith(lang)) {
        filtered.push({ name, label: v.desc || v.name || name, grade: v.grade || '' });
      }
    }
    filtered.sort((a, b) => (_GRADE_RANK[a.grade] ?? 99) - (_GRADE_RANK[b.grade] ?? 99) || a.name.localeCompare(b.name));
    if (filtered.length) groups.push({ lang, voices: filtered });
  }
  const allNames = new Set(groups.flatMap(g => g.voices.map(v => v.name)));
  // Pick the default selection: keep current if still valid, else first
  // available native voice, else "off".
  let toSelect = (currentVoice && allNames.has(currentVoice)) ? currentVoice : '';
  if (!toSelect && groups.length && groups[0].voices.length) toSelect = groups[0].voices[0].name;
  if (!toSelect) toSelect = 'off';
  let html = '';
  for (const g of groups) {
    html += `<optgroup label="${escapeHtml(_langLabel(g.lang))}">`;
    for (const v of g.voices) {
      const sel = v.name === toSelect ? ' selected' : '';
      const grade = v.grade ? ` (${v.grade})` : '';
      html += `<option value="${escapeHtml(v.name)}"${sel}>${escapeHtml(v.label)}${escapeHtml(grade)}</option>`;
    }
    html += '</optgroup>';
  }
  html += `<option value="off"${toSelect === 'off' ? ' selected' : ''}>Off — no audio</option>`;
  return html;
}

async function _openOnboardingModal() {
  if (!_state) { _fetchState().then(() => _state && _openOnboardingModal()); return; }
  const packs = _activePacks();
  if (!packs.length) { _openLanguagePicker(); return; }
  const { overlay, close } = _makeOverlay('learning-onboarding-overlay');

  // Voices fetched once before rendering — modal can offer real options.
  const voices = await _fetchKokoroVoices();

  const targetChecks = packs.map(p => `
    <label class="learning-modal-check">
      <input type="checkbox" data-lang="${escapeHtml(p.lang_code)}"
             ${(_state.target_langs || []).includes(p.lang_code) ? 'checked' : ''}>
      <span>${escapeHtml(p.name || _langLabel(p.lang_code))}</span>
    </label>`).join('');
  const levels = [['starting', 'Just starting'], ['basics', 'Some basics'],
                  ['intermediate', 'Intermediate'], ['advanced', 'Advanced']];
  const firstLang = packs[0]?.lang_code;
  const cur = (firstLang && _state.levels && _state.levels[firstLang]) || 'starting';
  const levelRadios = levels.map(([v, lbl]) => `
    <label class="learning-modal-radio">
      <input type="radio" name="learning-level" value="${v}" ${cur === v ? 'checked' : ''}>
      <span>${escapeHtml(lbl)}</span>
    </label>`).join('');

  // Initial TTS options: filter by whichever target_langs the user already had
  // (or, if first-time, by all installed packs).
  const initialTargets = (_state.target_langs && _state.target_langs.length)
    ? _state.target_langs
    : packs.map(p => p.lang_code);
  const ttsOptions = _ttsOptionsHtml(voices, initialTargets, _state.tts_voice || '');

  overlay.innerHTML = `
    <div class="modal learning-modal">
      <div class="modal-header"><h3 class="modal-title">Learn languages in Augmentum</h3></div>
      <div class="modal-body">
        <p class="learning-modal-intro">
          Your progress stays on your machine. Click words while you browse;
          review them later — that's the whole loop.
        </p>
        <div class="learning-modal-section">
          <label class="learning-modal-label" for="learning-native">What languages do you speak fluently?</label>
          <input id="learning-native" class="learning-modal-input" type="text" placeholder="e.g. English"
                 value="${escapeHtml(_state.native_lang || 'English')}">
        </div>
        <div class="learning-modal-section">
          <div class="learning-modal-label">What are you learning?</div>
          <div class="learning-modal-checks">${targetChecks}</div>
        </div>
        <div class="learning-modal-section">
          <div class="learning-modal-label">Your level?</div>
          <div class="learning-modal-radios">${levelRadios}</div>
        </div>
        <div class="learning-modal-section">
          <label class="learning-modal-label" for="learning-tts">Word pronunciation audio</label>
          <select id="learning-tts" class="learning-modal-input">${ttsOptions}</select>
        </div>
        <div class="learning-modal-section learning-modal-manage">
          <button type="button" class="btn btn-ghost btn-small" id="learning-manage-packs">+ Add another language</button>
        </div>
        <label class="learning-modal-check learning-modal-seed">
          <input type="checkbox" id="learning-seed" checked>
          <span>Start me with ~30 common words to review</span>
        </label>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-ghost" id="learning-cancel">Cancel</button>
        <button type="button" class="btn btn-primary" id="learning-save">Get started</button>
      </div>
    </div>`;

  // Re-render the TTS dropdown when the user changes which langs are checked.
  const ttsSelect = overlay.querySelector('#learning-tts');
  overlay.querySelectorAll('.learning-modal-checks input').forEach(cb => {
    cb.addEventListener('change', () => {
      const targets = Array.from(overlay.querySelectorAll('.learning-modal-checks input:checked'))
        .map(el => el.dataset.lang);
      const current = ttsSelect.value;
      ttsSelect.innerHTML = _ttsOptionsHtml(voices, targets, current);
    });
  });

  overlay.querySelector('#learning-cancel').addEventListener('click', close);
  overlay.querySelector('#learning-manage-packs')?.addEventListener('click', () => {
    close();
    _openLanguagePicker();
  });
  overlay.querySelector('#learning-save').addEventListener('click', async () => {
    const native = overlay.querySelector('#learning-native').value.trim() || 'English';
    const targets = Array.from(overlay.querySelectorAll('.learning-modal-checks input:checked')).map(el => el.dataset.lang);
    const levelEl = overlay.querySelector('input[name="learning-level"]:checked');
    const level = levelEl ? levelEl.value : 'starting';
    const ttsVoice = overlay.querySelector('#learning-tts')?.value || 'off';
    const wantSeed = !!overlay.querySelector('#learning-seed')?.checked;
    if (!targets.length) { showToast('Pick at least one language to learn', 'warning'); return; }
    const levelsPayload = {};
    targets.forEach(t => { levelsPayload[t] = level; });
    const saveBtn = overlay.querySelector('#learning-save');
    saveBtn.disabled = true;
    try {
      const r = await fetch('/api/learning/state', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ toggle: 'on', native_lang: native, target_langs: targets, levels: levelsPayload, tts_voice: ttsVoice }),
      });
      if (!r.ok) throw new Error(`save failed: ${r.status}`);
      _state = await r.json();
      let seeded = 0;
      if (wantSeed && targets[0]) {
        try {
          const sr = await fetch(`/api/learning/packs/${encodeURIComponent(targets[0])}/seed`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ count: 30 }),
          });
          if (sr.ok) seeded = (await sr.json()).seeded || 0;
        } catch { /* best effort */ }
      }
      showToast(seeded ? `Learning enabled · ${seeded} words ready to review` : 'Learning enabled', 'success');
      close();
      _state = null;
      await _renderChip();
      document.dispatchEvent(new CustomEvent('augmentum:learning-enabled'));
      // Land them on the Hub with a welcome banner, focused on the
      // primary target lang. Replaces the silent return-to-browse-landing
      // that used to follow onboarding — the moment they enabled
      // Learning *is* the moment to show them what they get.
      const firstLang = (targets && targets[0]) || _firstActiveTarget();
      if (firstLang) {
        // Give the chip a tick to repaint before stacking the hub on top.
        setTimeout(() => _openLearningHub(firstLang, { welcome: true }), 80);
      }
    } catch {
      saveBtn.disabled = false;
      showToast('Failed to save', 'error');
    }
  });
}

// ── Press-and-hold to define ─────────────────────────────────────

function _caretAt(doc, x, y) {
  try {
    if (doc.caretRangeFromPoint) {
      const r = doc.caretRangeFromPoint(x, y);
      return r ? { node: r.startContainer, offset: r.startOffset } : null;
    }
    if (doc.caretPositionFromPoint) {
      const p = doc.caretPositionFromPoint(x, y);
      return p ? { node: p.offsetNode, offset: p.offset } : null;
    }
  } catch { /* fall through */ }
  return null;
}

function _dismissPopover() {
  document.getElementById(POPOVER_ID)?.remove();
}

// If there's a non-empty text selection that lies (at least partly) inside
// ``el``, return its text — lets the user highlight part of a sentence and
// have the 🔊 speak just that. Otherwise '' (caller falls back to the full
// sentence). Read at click time; clicking a <button> doesn't clear a text
// selection in practice.
function _selectionWithin(el) {
  if (!el || !window.getSelection) return '';
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed || !sel.rangeCount) return '';
  const txt = sel.toString().trim();
  if (!txt) return '';
  for (let i = 0; i < sel.rangeCount; i += 1) {
    const r = sel.getRangeAt(i);
    if (el.contains(r.commonAncestorContainer) || el.contains(r.startContainer) || el.contains(r.endContainer)) {
      return txt;
    }
  }
  return '';
}

// Sentence-boundary characters used to bracket the click context. Covers
// Latin and CJK punctuation. Newlines also bound a sentence.
const _SENTENCE_BREAKS = /[.!?…。！？\n\r]/;

// Block-level containers we treat as "the surrounding paragraph" when
// flattening text across inline element boundaries. If a word is split
// across <span>/<b>/<i>/per-letter wrappers (common on syntax-styled or
// animated pages), the caret lands in a one-letter text node and the
// old single-node walk only captured a fragment.
const _BLOCK_TAGS = new Set([
  'P', 'DIV', 'LI', 'TD', 'TH', 'BLOCKQUOTE', 'PRE', 'ARTICLE', 'SECTION',
  'MAIN', 'BODY', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'DT', 'DD',
  'FIGCAPTION', 'SUMMARY', 'CAPTION', 'ASIDE', 'HEADER', 'FOOTER', 'NAV',
]);

function _nearestBlock(node) {
  let n = node && (node.nodeType === 1 ? node : node.parentNode);
  while (n && n.nodeType === 1 && !_BLOCK_TAGS.has(n.tagName)) n = n.parentNode;
  return (n && n.nodeType === 1) ? n : (node && node.ownerDocument && node.ownerDocument.body) || null;
}

// Concatenate all text-node contents within ``block`` and map the
// (caretNode, caretOffset) pair into an absolute offset in that
// flattened string. Returns null if the caret node isn't reachable.
function _flattenBlockText(block, caretNode, caretOffset) {
  if (!block) return null;
  const doc = block.ownerDocument || document;
  const walker = doc.createTreeWalker(block, NodeFilter.SHOW_TEXT, null);
  const pieces = [];
  let cumulative = 0;
  let absOffset = null;
  let node;
  while ((node = walker.nextNode())) {
    const t = node.textContent || '';
    if (node === caretNode) {
      absOffset = cumulative + Math.max(0, Math.min(caretOffset, t.length));
    }
    pieces.push(t);
    cumulative += t.length;
  }
  if (absOffset === null) return null;
  return { text: pieces.join(''), offset: absOffset };
}

// Extract (word at caret, surrounding sentence) from a press location.
// Flattens text across inline element boundaries within the nearest
// block, so a word split across <b>/<i>/<span>/per-letter wrappers is
// still captured as one word. The sentence gives the LLM contextual
// breakdown enough material to disambiguate which sense of e.g.
// Spanish "la" applies.
function _extractContextAtCaret(caret) {
  if (!caret || !caret.node) return null;
  let text, off;
  if (caret.node.nodeType === 3) {
    const block = _nearestBlock(caret.node);
    const flat = _flattenBlockText(block, caret.node, caret.offset);
    if (flat) {
      text = flat.text;
      off = flat.offset;
    } else {
      text = caret.node.textContent || '';
      off = Math.max(0, Math.min(caret.offset, text.length));
    }
  } else {
    return null;
  }
  // Walk LEFT until a sentence boundary, or the start of the block.
  let l = off;
  while (l > 0 && !_SENTENCE_BREAKS.test(text[l - 1])) l -= 1;
  // Walk RIGHT until a sentence boundary (inclusive — keep the .! etc).
  let r = off;
  while (r < text.length && !_SENTENCE_BREAKS.test(text[r])) r += 1;
  if (r < text.length) r += 1;
  const sentence = text.slice(l, r).trim();
  // Word at caret: scan left and right within the block-flattened text
  // for letter/digit chars. For CJK we keep a small slice (24 chars
  // from the caret) because the longest-prefix lookup at pos=0 handles
  // segmentation server-side.
  const wordRe = /[\p{L}\p{N}'\-]/u;
  let wl = off;
  while (wl > 0 && wordRe.test(text[wl - 1])) wl -= 1;
  let wr = off;
  while (wr < text.length && wordRe.test(text[wr])) wr += 1;
  let word = text.slice(wl, wr).trim();
  if (word.length > 24) word = word.slice(0, 24);
  if (!word) {
    word = text.slice(off, off + 24).trim();
  }
  return { word, sentence };
}

// Long-press gesture: filter accidental taps + collide less with native
// text selection. Fires `handler(pointerdownEvent)` after _LONG_PRESS_MS
// of no significant movement; cancels on early pointerup, drag past the
// move threshold, scroll, or pointer cancellation. We suppress the OS
// context menu when our long-press actually fired so the browser's
// long-press menu doesn't fight our popover on mobile.
const _LONG_PRESS_MS = 400;
const _LONG_PRESS_MOVE_PX = 10;

function _attachLongPress(target, handler) {
  let timer = null;
  let startX = 0;
  let startY = 0;
  let firedRecently = false;
  let startEvt = null;
  const clear = () => {
    if (timer) { clearTimeout(timer); timer = null; }
    startEvt = null;
  };
  target.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    if (e.target && e.target.closest && e.target.closest('a, button, input, textarea, select')) return;
    clear();
    startX = e.clientX;
    startY = e.clientY;
    startEvt = e;
    timer = setTimeout(() => {
      timer = null;
      firedRecently = true;
      try { handler(startEvt); } finally {
        startEvt = null;
        // Reset the contextmenu suppression flag after a short delay
        // so a *later* legitimate contextmenu still works.
        setTimeout(() => { firedRecently = false; }, 800);
      }
    }, _LONG_PRESS_MS);
  });
  const moveCancel = (e) => {
    if (!timer) return;
    if (Math.abs(e.clientX - startX) > _LONG_PRESS_MOVE_PX
        || Math.abs(e.clientY - startY) > _LONG_PRESS_MOVE_PX) {
      clear();
    }
  };
  target.addEventListener('pointermove', moveCancel, { passive: true });
  target.addEventListener('pointerup', clear);
  target.addEventListener('pointercancel', clear);
  target.addEventListener('pointerleave', clear);
  target.addEventListener('contextmenu', (e) => {
    // Only swallow the OS menu when our hold actually opened a popover —
    // otherwise (e.g. body-level attach over plain English text) native
    // right-click / long-press menus must keep working.
    if (firedRecently && document.getElementById(POPOVER_ID)) {
      e.preventDefault();
      firedRecently = false;
    }
  });
}

// Common click→lookup→popover handler. ``doc`` is the document the click
// happened in (the iframe's contentDocument, or `document` for parent-doc
// surfaces); ``originRect`` is the iframe's bounding rect (or null when the
// click is already in the parent viewport).
function _handleTextClick(doc, e, lang, originRect, sourceLabel) {
  if (e.defaultPrevented) return;
  if (e.target && e.target.closest && e.target.closest('a, button, input, textarea, select')) return;
  const caret = _caretAt(doc, e.clientX, e.clientY);
  const ctx = _extractContextAtCaret(caret);
  if (!ctx || !ctx.word) return;
  const px = (originRect ? originRect.left : 0) + e.clientX;
  const py = (originRect ? originRect.top : 0) + e.clientY;
  _lookupAndShowPopover(lang, ctx.word, px, py, sourceLabel || '', ctx.sentence);
}

function _attachToIframe(iframe) {
  if (!isLearningActive()) return;
  let doc;
  try { doc = iframe.contentDocument; } catch { return; }
  if (!doc || !doc.body || doc.body.dataset.augLearningWired === '1') return;
  doc.body.dataset.augLearningWired = '1';
  const lang = _firstActiveTarget();
  if (!lang) return;
  const sourceUrl = (() => { try { return iframe.contentWindow.location.href; } catch { return ''; } })();
  _attachLongPress(doc.body, (e) => {
    _handleTextClick(doc, e, lang, iframe.getBoundingClientRect(), sourceUrl);
  });
  doc.addEventListener('scroll', _dismissPopover, { passive: true });
}

// Wire press-and-hold-to-define on a parent-document element (e.g. a
// sentence in the reading overlay). Idempotent per element.
function _attachToElement(el, lang, sourceLabel) {
  if (!el || el.dataset.augLearningWired === '1' || !lang) return;
  el.dataset.augLearningWired = '1';
  _attachLongPress(el, (e) => _handleTextClick(document, e, lang, null, sourceLabel));
}

// ── Global hold-to-translate (CJK) ───────────────────────────────
//
// Press-and-hold on *any* target-language text anywhere in the app — chat
// bubbles, learning surfaces, games — pops the same define+translate
// popover the browse iframe uses. Scoped to CJK target languages only:
// their scripts identify the language unambiguously, so we can fire
// against arbitrary on-screen text without misreading English UI chrome
// as the target language. Latin-script targets (es/fr) share the alphabet
// with the UI and need sentence-level disambiguation we don't do yet —
// they keep the existing browse + reading surfaces.

// Hiragana+Katakana → Japanese. Hangul → Korean. Bare Han ideographs are
// ambiguous between Japanese kanji and Chinese — the dictionary probe in
// _resolveCjkAndShow picks whichever language actually resolves the word.
const _RE_KANA = /[぀-ヿ]/;
const _RE_HANGUL = /[가-힣ᄀ-ᇿ㄰-㆏]/;
const _RE_HAN = /[㐀-䶿一-鿿豈-﫿]/;

// Content surfaces where the gesture is allowed. The CJK-script gate makes
// false positives on English chrome near-impossible, but we still scope to
// real content so the hold never fights a button or the composer.
const _HOLD_REGION_SEL = '.message-content, .lg-overlay, [id^="learning-"]';

function _isWantedTarget(code) {
  if (!_state) return false;
  if (!_activePacks().some(p => p.lang_code === code)) return false;
  const targets = Array.isArray(_state.target_langs) ? _state.target_langs : [];
  return targets.length === 0 || targets.includes(code);
}

// Selected+installed CJK target languages consistent with the script in
// `text`, most-specific first. Empty when the text isn't a CJK target.
function _candidateCjkLangs(text) {
  if (!text) return [];
  const out = [];
  if (_RE_HANGUL.test(text) && _isWantedTarget('ko')) out.push('ko');
  if (_RE_KANA.test(text) && _isWantedTarget('ja')) out.push('ja');
  if (_RE_HAN.test(text)) {
    if (_isWantedTarget('zh') && !out.includes('zh')) out.push('zh');
    if (_isWantedTarget('ja') && !out.includes('ja')) out.push('ja');
  }
  return out;
}

// Probe each candidate language's dictionary in order; the first that
// resolves the pressed word wins and shows its popover. Disambiguates
// bare-Han (zh vs ja kanji) and silently no-ops when the caret grabbed
// punctuation/numbers rather than real vocabulary.
async function _resolveCjkAndShow(candidates, ctx, pageX, pageY) {
  for (const lang of candidates) {
    let entries = [];
    try {
      const r = await fetch(`/api/learning/lookup?lang=${encodeURIComponent(lang)}&q=${encodeURIComponent(ctx.word)}&pos=0`);
      if (r.ok) entries = (await r.json()).entries || [];
    } catch { /* try the next candidate */ }
    if (entries.length) {
      _lookupAndShowPopover(lang, ctx.word, pageX, pageY, '', ctx.sentence, entries);
      return;
    }
  }
}

function _globalHoldHandler(e) {
  if (!isLearningActive()) return;
  const tgt = e.target;
  if (!tgt || !tgt.closest) return;
  // Never hijack interactive controls or the composer.
  if (tgt.closest('a, button, input, textarea, select, [contenteditable="true"]')) return;
  // A more specific surface handler (browse iframe / reading overlay) owns
  // wired subtrees — skip so we don't open two popovers.
  if (tgt.closest('[data-aug-learning-wired="1"]')) return;
  if (!tgt.closest(_HOLD_REGION_SEL)) return;
  const caret = _caretAt(document, e.clientX, e.clientY);
  const ctx = _extractContextAtCaret(caret);
  if (!ctx || !ctx.word) return;
  const candidates = _candidateCjkLangs(ctx.word || ctx.sentence);
  if (!candidates.length) return;
  _resolveCjkAndShow(candidates, ctx, e.clientX, e.clientY);
}

let _globalHoldWired = false;
function _attachGlobalHold() {
  if (_globalHoldWired) return;
  _globalHoldWired = true;
  _attachLongPress(document.body, _globalHoldHandler);
  document.addEventListener('scroll', _dismissPopover, { passive: true, capture: true });
}

// Per-sentence breakdown cache shared across all popover invocations —
// clicking three words in the same paragraph fires the contextual LLM
// call once, not three times.
const _POPOVER_BREAKDOWN_CACHE = new Map();

// Natural-language translation of a target-language span to English via
// the user's current model. Single chokepoint shared by the hold popover
// and the highlighted-span breakdown. (Routes through /v1 today; a native
// /api/learning/translate is the follow-up for offline/Android parity.)
async function _translateText(text, lang) {
  const model = (app && app.state && app.state.currentModel) || 'default';
  const r = await fetch('/v1/chat/completions', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      messages: [
        { role: 'system', content: `You are a ${_langLabel(lang)}-to-English translator. Reply with ONLY a natural English translation of the user's message — no preamble, no notes, no quotation marks.` },
        { role: 'user', content: text },
      ],
      stream: false, max_tokens: 256, temperature: 0.2,
    }),
  });
  if (!r.ok) throw new Error(`translate ${r.status}`);
  const j = await r.json();
  return ((((j.choices || [])[0] || {}).message || {}).content || '').trim();
}

async function _lookupAndShowPopover(lang, snippet, pageX, pageY, sourceUrl, sentenceCtx, prefetched) {
  _dismissPopover();
  // The global CJK hold path already probed the dictionary to pick the
  // language; let it hand the entries through so we don't re-fetch.
  let entries = Array.isArray(prefetched) ? prefetched : [];
  if (!entries.length) {
    try {
      const r = await fetch(`/api/learning/lookup?lang=${encodeURIComponent(lang)}&q=${encodeURIComponent(snippet)}&pos=0`);
      if (r.ok) entries = (await r.json()).entries || [];
    } catch { return; }
  }
  if (!entries.length) return;
  const entry = entries[0];

  const pop = document.createElement('div');
  pop.id = POPOVER_ID;
  pop.className = 'learning-define-popover';
  const glosses = Array.isArray(entry.glosses) ? entry.glosses.slice(0, 5) : [];
  const romaji = _romaji(entry.reading || '');
  const pos = _posLabel(entry.pos || '', lang);
  const speakerBtn = _ttsVoice()
    ? `<button type="button" class="learning-pop-speak" aria-label="Pronounce">${_SPEAKER_SVG}</button>` : '';
  // When we have surrounding sentence context AND it's longer than just
  // the clicked word, show a placeholder for the contextual meaning
  // (filled async below). Otherwise show the dictionary gloss directly.
  const hasContext = sentenceCtx && sentenceCtx.length > snippet.length + 2;
  const contextSlot = hasContext
    ? `<div class="learning-pop-context-meaning" id="learning-pop-context-meaning">Reading "${escapeHtml(snippet)}" in context…</div>`
    : '';
  pop.innerHTML = `
    <div class="learning-pop-head">
      <span class="learning-pop-surface">${escapeHtml(entry.surface || '')}</span>
      <span class="learning-pop-reading">${escapeHtml(entry.reading || '')}</span>
      ${romaji ? `<span class="learning-pop-romaji">${escapeHtml(romaji)}</span>` : ''}
      ${speakerBtn}
      <button type="button" class="learning-pop-close" aria-label="Close">&times;</button>
    </div>
    ${contextSlot}
    ${hasContext ? `<details class="learning-pop-dict-details"><summary class="learning-pop-dict-summary">All dictionary senses</summary>
      <div class="learning-pop-glosses">${escapeHtml(glosses.join(' · '))}</div>
      ${pos ? `<div class="learning-pop-pos">${escapeHtml(pos)}</div>` : ''}
    </details>` : `<div class="learning-pop-glosses">${escapeHtml(glosses.join(' · '))}</div>${pos ? `<div class="learning-pop-pos">${escapeHtml(pos)}</div>` : ''}`}
    ${hasContext ? `<div class="learning-pop-sentence">
      <button type="button" class="btn btn-ghost learning-pop-translate">Translate sentence →</button>
      <div class="learning-pop-sentence-out" style="display:none"></div>
    </div>` : ''}
    <button type="button" class="btn btn-primary learning-pop-add">+ Add to my words</button>`;
  document.body.appendChild(pop);
  _attachSpeakButton(pop.querySelector('.learning-pop-speak'), () => entry.reading || entry.surface, lang);

  // Fire the contextual LLM breakdown if we have a sentence — the same
  // path Companion uses. Cached per (lang, sentence) so multi-click on
  // one paragraph doesn't re-fire. Promote a found-in-context meaning
  // into the prominent slot; collapse the dictionary dump behind
  // <details> since context tells the learner what they actually need.
  if (hasContext) {
    (async () => {
      const cacheKey = `${lang}::${sentenceCtx}`;
      let breakdown = _POPOVER_BREAKDOWN_CACHE.get(cacheKey);
      if (!breakdown) {
        try {
          const common = await import('./learning_games/_common.js');
          breakdown = await common.breakdownContextual(sentenceCtx, lang);
        } catch { /* */ }
        _POPOVER_BREAKDOWN_CACHE.set(cacheKey, breakdown || []);
      }
      const slot = pop.querySelector('#learning-pop-context-meaning');
      if (!slot || !document.body.contains(pop)) return;
      if (!Array.isArray(breakdown) || breakdown.length === 0) {
        slot.remove();   // no context available; the dict glosses already show
        return;
      }
      const target = snippet.toLowerCase();
      const hit = breakdown.find(t => (t.token || '').toLowerCase() === target)
        || breakdown.find(t => (t.token || '').toLowerCase().split(/\s+/).includes(target))
        || breakdown.find(t => target.includes((t.token || '').toLowerCase()));
      if (!hit) { slot.remove(); return; }
      slot.classList.add('learning-pop-context-meaning-loaded');
      slot.innerHTML = `
        <div class="learning-pop-context-label">in this sentence</div>
        <div class="learning-pop-context-text">${escapeHtml(hit.meaning || '')}</div>
        <div class="learning-pop-context-role">${escapeHtml(hit.role || '')}</div>`;
    })();
  }

  // Position: anchored just below the click, clamped to the viewport.
  const w = pop.offsetWidth, h = pop.offsetHeight;
  let left = Math.min(pageX, window.innerWidth - w - 12);
  left = Math.max(8, left);
  let top = pageY + 18;
  if (top + h > window.innerHeight - 8) top = Math.max(8, pageY - h - 12);
  pop.style.left = `${left}px`;
  pop.style.top = `${top}px`;

  pop.querySelector('.learning-pop-close').addEventListener('click', _dismissPopover);

  // Whole-sentence translation, on demand — the contextual word meaning
  // above auto-fires; the full sentence is one tap away so we don't spend
  // an LLM call on every hold.
  const trBtn = pop.querySelector('.learning-pop-translate');
  if (trBtn) {
    trBtn.addEventListener('click', async () => {
      const out = pop.querySelector('.learning-pop-sentence-out');
      trBtn.disabled = true;
      trBtn.textContent = 'Translating…';
      try {
        const t = await _translateText(sentenceCtx, lang);
        out.textContent = t || '(no translation returned)';
        out.style.display = '';
        trBtn.style.display = 'none';
      } catch {
        trBtn.disabled = false;
        trBtn.textContent = 'Translation unavailable';
      }
    });
  }

  const addBtn = pop.querySelector('.learning-pop-add');
  addBtn.addEventListener('click', async () => {
    addBtn.disabled = true;
    try {
      const r = await fetch('/api/learning/vocab/add', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lang, word_id: entry.word_id, source_surface: 'browse', source_ref: sourceUrl || '' }),
      });
      const j = r.ok ? await r.json() : { added: false };
      addBtn.textContent = j.added ? '✓ Added — appears in tomorrow’s review' : '✓ Already in your queue';
      addBtn.classList.add('learning-pop-added');
      document.dispatchEvent(new CustomEvent('augmentum:learning-vocab-added', { detail: { lang, word_id: entry.word_id } }));
      setTimeout(_dismissPopover, 1400);
    } catch {
      addBtn.disabled = false;
      addBtn.textContent = 'Try again';
    }
  });

  // Outside-click dismissal (next tick so this click doesn't immediately close it).
  setTimeout(() => {
    function onDoc(e) {
      if (!pop.contains(e.target)) { _dismissPopover(); document.removeEventListener('mousedown', onDoc); }
    }
    document.addEventListener('mousedown', onDoc);
  }, 0);
}

// ── Highlighted-span breakdown ───────────────────────────────────

async function _openBreakdown(lang, anchorEl, text) {
  text = (text || '').trim();
  if (!text) { showToast('Highlight some text first', 'info'); return; }
  document.getElementById('learning-breakdown')?.remove();

  // Try the contextual LLM breakdown first — same path Companion uses.
  // Returns each token with its meaning IN THIS SENTENCE, not a dump
  // of every dictionary sense. Falls back to the dictionary tokenizer
  // (which IS the per-word dump) if the LLM is offline.
  let llmTokens = null;
  try {
    const common = await import('./learning_games/_common.js');
    llmTokens = await common.breakdownContextual(text, lang);
  } catch { /* */ }

  let rows = '';
  let romaji = '';
  if (Array.isArray(llmTokens) && llmTokens.length > 0) {
    rows = llmTokens.map((t) => `
      <div class="learning-bd-row">
        <span class="learning-bd-surface">${escapeHtml(t.token || '')}</span>
        <span class="learning-bd-gloss">${escapeHtml(t.meaning || '')}</span>
        ${t.role ? `<span class="learning-bd-role">${escapeHtml(t.role)}</span>` : ''}
        ${t.lemma && t.lemma !== t.token ? `<span class="learning-bd-lemma">lemma: ${escapeHtml(t.lemma)}</span>` : ''}
      </div>`).join('');
  } else {
    // Dictionary fallback — the legacy per-token dump, kept for
    // offline / LLM-unavailable scenarios.
    let tokens = [];
    try {
      const r = await fetch(`/api/learning/breakdown/${encodeURIComponent(lang)}?q=${encodeURIComponent(text)}`);
      if (r.ok) tokens = (await r.json()).tokens || [];
    } catch { /* leave empty */ }
    romaji = tokens.map((t) => (t.matched ? _romaji(t.reading || '') : (t.text || ''))).join('');
    rows = tokens.map((t, i) => {
      if (!t.matched) return `<div class="learning-bd-raw">${escapeHtml(t.text || '')}</div>`;
      const g = Array.isArray(t.glosses) ? t.glosses.slice(0, 3).join(' · ') : '';
      return `<div class="learning-bd-row" data-i="${i}">
          <span class="learning-bd-surface">${escapeHtml(t.text || t.surface || '')}</span>
          <span class="learning-bd-reading">${escapeHtml(t.reading || '')}</span>
          <span class="learning-bd-gloss">${escapeHtml(g)}</span>
          ${t.word_id ? `<button type="button" class="learning-bd-add" data-word="${escapeHtml(String(t.word_id))}" title="Add to my words">+</button>` : ''}
        </div>`;
    }).join('');
  }

  const audioOn = !!_ttsVoice();
  const pop = document.createElement('div');
  pop.id = 'learning-breakdown';
  pop.className = 'learning-breakdown';
  pop.innerHTML = `
    <div class="learning-bd-head">
      <span class="learning-bd-text">${escapeHtml(text)}</span>
      ${audioOn ? `<button type="button" class="learning-bd-speak" aria-label="Play">${_SPEAKER_SVG}</button>` : ''}
      <button type="button" class="learning-bd-close" aria-label="Close">&times;</button>
    </div>
    ${romaji ? `<div class="learning-bd-romaji">${escapeHtml(romaji)}</div>` : ''}
    <div class="learning-bd-tokens">${rows || '<em class="learning-bd-empty">Couldn\'t analyse this sentence — try again, or click an individual word for the dictionary entry.</em>'}</div>
    <button type="button" class="btn btn-ghost learning-bd-translate">Translate this →</button>
    <div class="learning-bd-translation" style="display:none"></div>`;
  document.body.appendChild(pop);

  // Position near the anchor (or top-centre if none / off-screen).
  const w = pop.offsetWidth, h = pop.offsetHeight;
  if (anchorEl && anchorEl.getBoundingClientRect) {
    const rect = anchorEl.getBoundingClientRect();
    let top = rect.bottom + 8;
    if (top + h > window.innerHeight - 8) top = Math.max(8, rect.top - h - 8);
    let left = Math.min(rect.left, window.innerWidth - w - 12);
    pop.style.top = `${top}px`;
    pop.style.left = `${Math.max(8, left)}px`;
  } else {
    pop.style.top = `${Math.max(8, (window.innerHeight - h) / 3)}px`;
    pop.style.left = `${Math.max(8, (window.innerWidth - w) / 2)}px`;
  }

  _attachSpeakButton(pop.querySelector('.learning-bd-speak'), () => text, lang);
  pop.querySelector('.learning-bd-close').addEventListener('click', () => pop.remove());
  pop.querySelectorAll('.learning-bd-add').forEach((b) => {
    b.addEventListener('click', async () => {
      const wid = b.dataset.word;
      if (!wid) return;
      b.disabled = true;
      try {
        const r = await fetch('/api/learning/vocab/add', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ lang, word_id: wid, source_surface: 'browse' }),
        });
        await r.json().catch(() => ({}));
        b.textContent = '✓';
        b.classList.add('learning-bd-added');
        document.dispatchEvent(new CustomEvent('augmentum:learning-vocab-added', { detail: { lang, word_id: wid } }));
      } catch { b.disabled = false; }
    });
  });
  pop.querySelector('.learning-bd-translate').addEventListener('click', async () => {
    const btn = pop.querySelector('.learning-bd-translate');
    const out = pop.querySelector('.learning-bd-translation');
    btn.disabled = true;
    btn.textContent = 'Translating…';
    try {
      const t = await _translateText(text, lang);
      out.textContent = t || '(no translation returned)';
      out.style.display = '';
      btn.style.display = 'none';
    } catch {
      btn.disabled = false;
      btn.textContent = 'Translation unavailable';
    }
  });
  setTimeout(() => {
    function onDoc(e) {
      if (!pop.contains(e.target) && e.target !== anchorEl) { pop.remove(); document.removeEventListener('mousedown', onDoc); }
    }
    document.addEventListener('mousedown', onDoc);
  }, 0);
}

// ── SRS review overlay ───────────────────────────────────────────

async function _openReviewOverlay(lang) {
  const { overlay, close } = _makeOverlay('learning-review-overlay');
  let cards = [];
  let idx = 0, revealed = false, graded = 0;

  async function load() {
    const data = await _fetchDue(lang, 30);
    cards = Array.isArray(data.due) ? data.due : [];
    idx = 0; revealed = false; graded = 0;
  }

  function renderEmpty() {
    const finished = graded > 0;
    overlay.innerHTML = `
      <div class="modal learning-review-modal">
        <div class="modal-body learning-review-empty">
          <div class="learning-review-empty-icon">${finished ? '\u{2705}' : '\u{1F389}'}</div>
          ${finished
            ? `<p>Done — ${graded} reviewed. Next review tomorrow.</p>`
            : `<p>Nothing due right now — come back tomorrow.</p>
               <p class="learning-review-empty-hint">Or jump-start your queue with high-frequency words:</p>
               <button type="button" class="btn btn-primary learning-review-seed" id="learning-review-seed">Add ~30 common words</button>`}
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-ghost" id="learning-review-readlink">📖 Read sentences</button>
          <button type="button" class="btn btn-ghost" id="learning-review-empty-games">🎮 Play a game</button>
          <button type="button" class="btn btn-ghost" id="learning-review-done">Close</button>
        </div>
      </div>`;
    overlay.querySelector('#learning-review-done').addEventListener('click', () => { close(); _renderChip(); });
    overlay.querySelector('#learning-review-readlink').addEventListener('click', () => { close(); _openReadOverlay(lang); });
    overlay.querySelector('#learning-review-empty-games').addEventListener('click', async () => {
      close();
      const { openGamesHub } = await import('./learning_games/hub.js');
      openGamesHub({ lang, voice: _ttsVoice() });
    });
    const seedBtn = overlay.querySelector('#learning-review-seed');
    if (seedBtn) seedBtn.addEventListener('click', async () => {
      seedBtn.disabled = true; seedBtn.textContent = 'Adding…';
      try {
        const r = await fetch(`/api/learning/packs/${encodeURIComponent(lang)}/seed`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ count: 30 }),
        });
        if (r.ok) {
          await load();
          if (cards.length) { _renderChip(); render(); return; }
        }
        seedBtn.textContent = 'Nothing available to add';
      } catch { seedBtn.disabled = false; seedBtn.textContent = 'Try again'; }
    });
  }

  function render() {
    const card = cards[idx];
    if (!card) { renderEmpty(); return; }
    const iv = card.preview_intervals || {};
    const grades = [
      { g: 1, label: 'Again', cls: 'again' },
      { g: 2, label: 'Hard', cls: 'hard' },
      { g: 3, label: 'Good', cls: 'good' },
      { g: 4, label: 'Easy', cls: 'easy' },
    ];
    const gloss = Array.isArray(card.glosses) ? card.glosses.join(' · ') : '';
    const romaji = _romaji(card.reading || '');
    const pos = _posLabel(card.pos || '', lang);
    const audioOn = !!_ttsVoice();
    const speakerBtn = audioOn
      ? `<button type="button" class="learning-card-speak" aria-label="Pronounce">${_SPEAKER_SVG}</button>` : '';
    const exSpeak = audioOn
      ? `<button type="button" class="learning-example-speak" aria-label="Play sentence">${_SPEAKER_SVG}</button>` : '';
    const exBreak = `<button type="button" class="learning-example-break" aria-label="Break down">${_LENS_SVG}</button>`;
    const example = card.example
      ? `<div class="learning-review-example">
           <div class="learning-review-example-line">
             <span class="learning-review-example-target">${escapeHtml(card.example.lang_text || '')}</span>${exBreak}${exSpeak}
           </div>
           ${card.example.en_text ? `<div class="learning-review-example-en">${escapeHtml(card.example.en_text)}</div>` : ''}
         </div>` : '';
    const pct = cards.length ? Math.round((idx / cards.length) * 100) : 0;
    overlay.innerHTML = `
      <div class="modal learning-review-modal">
        <div class="learning-review-thread"><i style="width:${pct}%"></i></div>
        <div class="learning-review-topbar">
          <button type="button" class="learning-review-readlink" id="learning-review-readlink" title="Read sentences instead">📖 Read</button>
          <button type="button" class="learning-review-readlink" id="learning-review-gameslink" title="Play a game with these words">🎮 Games</button>
          <span class="learning-review-progress">${idx + 1} / ${cards.length}</span>
          <button type="button" class="learning-settings-gear" id="learning-review-gear" title="Manage languages" aria-label="Manage languages">⚙</button>
        </div>
        <div class="modal-body learning-review-body">
          <div class="learning-review-surface">${escapeHtml(card.surface || '')}</div>
          ${revealed ? `
            <div class="learning-review-readingline">
              <span class="learning-review-reading">${escapeHtml(card.reading || '')}</span>
              ${romaji ? `<span class="learning-review-romaji">${escapeHtml(romaji)}</span>` : ''}
              ${speakerBtn}
            </div>
            <div class="learning-review-gloss">${escapeHtml(gloss)}</div>
            ${pos ? `<div class="learning-review-pos">${escapeHtml(pos)}</div>` : ''}
            ${example}
            <div class="learning-review-grades">
              ${grades.map(x => `<button type="button" class="learning-grade learning-grade-${x.cls}" data-g="${x.g}">
                <span class="learning-grade-label">${x.label}</span>
                <span class="learning-grade-iv">${escapeHtml(_fmtInterval(iv[x.g]))}</span></button>`).join('')}
            </div>` : `
            <button type="button" class="btn btn-primary learning-review-show" id="learning-review-show">Show answer</button>`}
        </div>
      </div>`;
    overlay.querySelector('#learning-review-readlink')?.addEventListener('click', () => { close(); _openReadOverlay(lang); });
    overlay.querySelector('#learning-review-gameslink')?.addEventListener('click', async () => {
      close();
      const { openGamesHub } = await import('./learning_games/hub.js');
      openGamesHub({ lang, voice: _ttsVoice() });
    });
    overlay.querySelector('#learning-review-gear')?.addEventListener('click', () => { close(); _openOnboardingModal(); });
    _attachSpeakButton(overlay.querySelector('.learning-card-speak'), () => card.reading || card.surface, lang);
    _attachSpeakButton(overlay.querySelector('.learning-example-speak'),
      () => _selectionWithin(overlay.querySelector('.learning-review-example-target')) || (card.example && card.example.lang_text) || '', lang);
    overlay.querySelector('.learning-example-break')?.addEventListener('click', (e) => {
      const exEl = overlay.querySelector('.learning-review-example-target');
      _openBreakdown(lang, e.currentTarget, _selectionWithin(exEl) || (card.example && card.example.lang_text) || '');
    });
    if (!revealed) {
      overlay.querySelector('#learning-review-show').addEventListener('click', () => { revealed = true; render(); });
    } else {
      overlay.querySelectorAll('.learning-grade').forEach(b => {
        b.addEventListener('click', () => _grade(card, Number(b.dataset.g)));
      });
    }
  }

  async function _grade(card, g) {
    overlay.querySelectorAll('.learning-grade').forEach(b => { b.disabled = true; });
    try {
      await fetch('/api/learning/srs/grade', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lang, word_id: card.word_id, grade: g }),
      });
    } catch { /* keep going — best effort */ }
    graded += 1;
    idx += 1;
    revealed = false;
    render();
  }

  function onKey(e) {
    if (e.key === ' ' || e.key === 'Spacebar') {
      e.preventDefault();
      if (!revealed) { revealed = true; render(); }
      else { const c = cards[idx]; if (c) _grade(c, 3); }
    } else if (['1', '2', '3', '4'].includes(e.key) && revealed) {
      const c = cards[idx]; if (c) _grade(c, Number(e.key));
    }
  }
  document.addEventListener('keydown', onKey);
  // Detach the key handler when the overlay goes away.
  const obs = new MutationObserver(() => {
    if (!document.body.contains(overlay)) { document.removeEventListener('keydown', onKey); obs.disconnect(); }
  });
  obs.observe(document.body, { childList: true });

  await load();
  render();
}

// ── Sentences reading mode (v1) ──────────────────────────────────

async function _openReadOverlay(lang) {
  const { overlay, close } = _makeOverlay('learning-read-overlay');
  let sentences = [];
  let idx = 0, revealed = false;

  async function load() {
    try {
      const r = await fetch(`/api/learning/read/${encodeURIComponent(lang)}?count=20`);
      sentences = r.ok ? (await r.json()).sentences || [] : [];
    } catch { sentences = []; }
    idx = 0; revealed = false;
  }

  function render() {
    if (!sentences.length) {
      overlay.innerHTML = `
        <div class="modal learning-read-modal">
          <div class="modal-body learning-review-empty"><p>No sentences available to read.</p></div>
          <div class="modal-footer"><button type="button" class="btn btn-ghost" id="learning-read-close">Close</button></div>
        </div>`;
      overlay.querySelector('#learning-read-close').addEventListener('click', () => { close(); _renderChip(); });
      return;
    }
    if (idx >= sentences.length) {
      overlay.innerHTML = `<div class="modal learning-read-modal"><div class="modal-body learning-review-empty"><p>Loading more…</p></div></div>`;
      load().then(render);
      return;
    }
    const s = sentences[idx];
    const sentSpeak = _ttsVoice()
      ? `<button type="button" class="learning-read-speak" aria-label="Play sentence">${_SPEAKER_SVG}</button>` : '';
    const sentBreak = `<button type="button" class="learning-read-break" aria-label="Break down">${_LENS_SVG}</button>`;
    const readPct = sentences.length ? Math.round((idx / sentences.length) * 100) : 0;
    overlay.innerHTML = `
      <div class="modal learning-read-modal">
        <div class="learning-review-thread"><i style="width:${readPct}%"></i></div>
        <div class="learning-review-topbar">
          <button type="button" class="learning-review-readlink" id="learning-read-srs" title="Back to review">↩ Review</button>
          <span class="learning-review-progress">${idx + 1} / ${sentences.length}</span>
          <button type="button" class="learning-settings-gear" id="learning-read-gear" title="Manage languages" aria-label="Manage languages">⚙</button>
        </div>
        <div class="modal-body learning-read-body">
          <div class="learning-read-sentenceline">
            <span class="learning-read-sentence" id="learning-read-sentence">${escapeHtml(s.lang_text || '')}</span>${sentBreak}${sentSpeak}
          </div>
          ${revealed
            ? `<div class="learning-read-translation">${escapeHtml(s.en_text || '(no translation)')}</div>`
            : `<button type="button" class="btn btn-ghost learning-read-reveal" id="learning-read-reveal">Show translation</button>`}
          <p class="learning-read-hint">Tap a word to look it up · highlight a phrase and 🔍 breaks it down (🔊 speaks just it).</p>
        </div>
        <div class="modal-footer">
          <button type="button" class="btn btn-ghost" id="learning-read-close">Done</button>
          <button type="button" class="btn btn-primary" id="learning-read-next">Next →</button>
        </div>
      </div>`;
    const sentEl = overlay.querySelector('#learning-read-sentence');
    if (sentEl) _attachToElement(sentEl, lang, 'reading');
    _attachSpeakButton(overlay.querySelector('.learning-read-speak'),
      () => _selectionWithin(sentEl) || s.lang_text || '', lang);
    overlay.querySelector('.learning-read-break')?.addEventListener('click', (e) =>
      _openBreakdown(lang, e.currentTarget, _selectionWithin(sentEl) || s.lang_text || ''));
    overlay.querySelector('#learning-read-close').addEventListener('click', () => { close(); _renderChip(); });
    overlay.querySelector('#learning-read-srs').addEventListener('click', () => { close(); _openReviewOverlay(lang); });
    overlay.querySelector('#learning-read-next').addEventListener('click', () => {
      idx += 1; revealed = false; _dismissPopover(); render();
    });
    overlay.querySelector('#learning-read-reveal')?.addEventListener('click', () => { revealed = true; render(); });
  }

  await load();
  render();
}

// ── Public surface ───────────────────────────────────────────────

export function isLearningActive() {
  return !!(_state && _state.toggle === 'on' && _activePacks().length > 0);
}

export function getTargetLangs() {
  return _state && Array.isArray(_state.target_langs) ? [..._state.target_langs] : [];
}

export async function refresh() {
  _state = null;
  await _renderChip();
}

// ── Init ─────────────────────────────────────────────────────────

document.addEventListener('augmentum:browse-landing-ready', () => { _renderChip(); });
document.addEventListener('augmentum:browse-iframe-loaded', (e) => {
  const iframe = e.detail && e.detail.iframe;
  if (iframe) _attachToIframe(iframe);
});
document.addEventListener('augmentum:learning-vocab-added', () => {
  // A newly-added word becomes due tomorrow, so the count doesn't change
  // today — but refresh the chip anyway in case it's the first one.
  _state = null;
  _renderChip();
});

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => { _renderChip(); _attachGlobalHold(); });
} else {
  _renderChip();
  _attachGlobalHold();
}
