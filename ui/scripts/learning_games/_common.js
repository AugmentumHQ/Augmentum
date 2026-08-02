/**
 * Shared utilities for the learning-games suite.
 *
 * Each game is a self-contained ES module exporting one `launchX({lang,
 * voice, pool, onClose})` async function. Everything below is helpers
 * those launchers can lean on: pool fetch, TTS playback, FSRS grading,
 * overlay scaffolding, common DOM helpers.
 */

import { escapeHtml } from '../app.js';

export { escapeHtml };

// ── Data ────────────────────────────────────────────────────────────

export async function fetchGamePool(lang, count = 30, mode = 'mixed', focusWords = [], options = {}) {
  try {
    const opts = Array.isArray(focusWords) ? (options || {}) : (focusWords || {});
    const focusList = Array.isArray(focusWords)
      ? focusWords
      : (Array.isArray(opts.focusWords) ? opts.focusWords : (Array.isArray(opts.focus) ? opts.focus : []));
    const qs = new URLSearchParams({ lang, count: String(count), mode });
    if (opts.allowDiscovery) qs.set('allow_discovery', 'true');
    // Bias the pool toward specific word_ids when the partner prescribed
    // a focused drill. The backend reorders matched words to the front;
    // non-matches still come along to fill distractor games.
    if (focusList.length) {
      for (const w of focusList.slice(0, 20)) {
        if (w) qs.append('focus', String(w));
      }
    }
    const r = await fetch(`/api/learning/games/pool?${qs}`);
    if (!r.ok) return [];
    const j = await r.json();
    return j.pool || [];
  } catch { return []; }
}


export async function recordResult({ game_id, lang, score, words_played, words_correct, duration_sec, metadata }) {
  try {
    await fetch('/api/learning/games/result', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        game_id, lang,
        score: score | 0,
        words_played: words_played | 0,
        words_correct: words_correct | 0,
        duration_sec: duration_sec | 0,
        metadata: metadata || {},
      }),
    });
  } catch { /* fire-and-forget */ }
}

export async function fetchBestScores(lang) {
  try {
    const r = await fetch(`/api/learning/games/best?lang=${encodeURIComponent(lang)}`);
    if (!r.ok) return {};
    const j = await r.json();
    return j.by_game || {};
  } catch { return {}; }
}

export async function fetchGameReadiness(lang) {
  try {
    const r = await fetch(`/api/learning/games/readiness?lang=${encodeURIComponent(lang)}`);
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}


// Contextual sentence breakdown — asks the same LLM that produced the
// reply to label every token with its role + meaning *in this specific
// context*. Solves the dictionary-dump problem: a learner clicking on
// 'gusta' shouldn't have to mentally pick between '3sg present
// indicative' and '2sg imperative' — the model knows which it is.
//
// Output shape (parsed from the model's JSON):
//   [{token, role, meaning, lemma}, ...]
// Errors fall through to `null` — callers should degrade to the
// existing per-word dictionary popup.
export async function breakdownContextual(text, lang) {
  if (!text || !lang) return null;
  const trimmed = String(text).trim().slice(0, 1000);
  if (!trimmed) return null;
  const sys = `You are a ${lang} reading assistant for a language learner. Given one ${lang} sentence, output a JSON array (and nothing else) labelling every meaningful token with its role and meaning IN THIS SPECIFIC CONTEXT.

For each token, output:
{
  "token": "<exact surface as it appears, no punctuation>",
  "role": "<short grammatical role, e.g. 'greeting', 'subject pronoun', 'indirect object pronoun', 'verb (3sg present)', 'definite article (f sg)', 'noun (m sg)', 'conjunction', 'preposition', 'adjective (f sg)'>",
  "meaning": "<one short clause in English — the meaning the word carries HERE, not every possible meaning>",
  "lemma": "<dictionary headword if the token is inflected, else same as token>"
}

Skip punctuation. Skip the wrapping markers (¿ ¡ ?). For multi-word expressions ('me llamo', 'how are you'), output a single combined token if they function as a unit.

The learner needs the MEANING IN CONTEXT, not a dictionary dump. Example: for "¿Te gusta la playa?", 'te' is "to you (the recipient of the liking)", NOT "second-person singular voseo imperative of abdicar combined with te". Be the helpful tutor explaining what the sentence does.

Output strict JSON array only. No prose before or after.`;
  const user = `Sentence: "${trimmed}"`;
  const raw = (await llmChat(
    [{ role: "system", content: sys }, { role: "user", content: user }],
  )).trim();
  if (!raw) return null;
  // Extract the JSON array — the strip helper has already removed
  // reasoning tokens, but the model may still wrap in code fences or
  // include trailing prose despite the prompt.
  const match = raw.match(/\[[\s\S]*\]/);
  if (!match) return null;
  try {
    const arr = JSON.parse(match[0]);
    if (!Array.isArray(arr)) return null;
    return arr.filter(t => t && typeof t === 'object' && t.token);
  } catch {
    return null;
  }
}

export async function fetchBreakdown(lang, text) {
  try {
    const r = await fetch(`/api/learning/breakdown/${encodeURIComponent(lang)}?q=${encodeURIComponent(text)}`);
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

export async function fetchSentences(lang, count = 8, q = null) {
  const qs = new URLSearchParams({ count: String(count) });
  if (q) qs.set('q', q);
  try {
    const r = await fetch(`/api/learning/read/${encodeURIComponent(lang)}?${qs}`);
    if (!r.ok) return [];
    return (await r.json()).sentences || [];
  } catch { return []; }
}

export async function gradeCard(lang, word_id, grade) {
  try {
    const r = await fetch('/api/learning/srs/grade', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lang, word_id, grade }),
    });
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

export async function addWord(lang, word_id) {
  try {
    const r = await fetch('/api/learning/vocab/add', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lang, word_id, source_surface: 'game' }),
    });
    return r.ok;
  } catch { return false; }
}

// LLM chat helper for games that need light generative output (story,
// translation grading, constellation validation). Keeps a single shape.
//
// Reasoning models occasionally leak chain-of-thought into `content`
// (when the proxy's thinking parser doesn't recognise the family, or
// `--jinja` isn't wired). Strip everything up to the last `</think>` —
// no-op when the response is clean. Symmetric `<think>X</think>Y`,
// asymmetric `X</think>Y`, and plain `Y` all collapse to Y.
function _stripReasoning(text) {
  if (!text) return '';
  if (text.includes('</think>')) {
    return text.split('</think>').pop().trimStart();
  }
  if (text.includes('[/THINK]')) {
    return text.split('[/THINK]').pop().trimStart();
  }
  return text;
}

function _selectedModel(model) {
  // Use the chat UI's selected model (same localStorage key settings.js
  // reads). Sending literal "default" only works if a model is already
  // loaded; on a cold engine the backend has no way to resolve it and
  // returns 500.
  return model
    || (typeof localStorage !== 'undefined' && localStorage.getItem('augmentum-selected-model'))
    || 'default';
}

export async function llmChat(messages, model) {
  try {
    const r = await fetch('/v1/chat/completions', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: _selectedModel(model), messages, stream: false, temperature: 0.6,
      }),
    });
    if (!r.ok) return '';
    const j = await r.json();
    const raw = j?.choices?.[0]?.message?.content || '';
    return _stripReasoning(raw);
  } catch { return ''; }
}

// Streaming variant for prose surfaces (chat / story / quest). Calls
// `onDelta(deltaText, visibleSoFar)` for each visible chunk. Reasoning
// tokens (anything before `</think>` or `[/THINK]`) are suppressed —
// onDelta only fires once the closer is seen, then for every subsequent
// chunk. If the model never emits a closer, the full content is treated
// as visible at end-of-stream and a single onDelta fires with all of it.
// Resolves to the final visible text (already stripped).
export async function llmChatStream(messages, onDelta, model) {
  let raw = '';
  let visible = '';
  let closer = '';   // '</think>' or '[/THINK]' once seen
  function _flush() {
    if (closer) {
      const next = raw.split(closer).pop();
      const delta = next.slice(visible.length);
      visible = next;
      if (delta && onDelta) onDelta(delta, visible.trimStart());
      return;
    }
    if (raw.includes('</think>')) { closer = '</think>'; _flush(); return; }
    if (raw.includes('[/THINK]')) { closer = '[/THINK]'; _flush(); return; }
  }
  try {
    const r = await fetch('/v1/chat/completions', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: _selectedModel(model), messages, stream: true, temperature: 0.6,
      }),
    });
    if (!r.ok || !r.body) return '';
    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split('\n');
      buf = lines.pop() || '';
      for (const line of lines) {
        const s = line.trim();
        if (!s.startsWith('data:')) continue;
        const data = s.slice(5).trim();
        if (data === '[DONE]') continue;
        try {
          const obj = JSON.parse(data);
          const piece = obj?.choices?.[0]?.delta?.content || '';
          if (piece) {
            raw += piece;
            _flush();
          }
        } catch { /* partial chunk, retry next pass */ }
      }
    }
    // No closer ever seen — fall back to showing the whole response.
    if (!closer && raw) {
      visible = raw;
      if (onDelta) onDelta(raw, raw.trimStart());
    }
    return visible.trimStart();
  } catch { return visible.trimStart(); }
}

// ── Voice resolution ────────────────────────────────────────────────

// One TTS voice can't speak every language well — a Kokoro English
// voice mangles Japanese kana and Chinese tones, and vice versa. The
// games take a lang as input, so we can pick the right voice instead
// of asking the user to remember to flip it in settings every time.
//
// Voice tokens returned by `resolveVoiceForLang`:
//   "<name>"                        — server voice (Kokoro/Pocket/external)
//   "browser:<BCP-47>:<voice name>" — Web Speech API system voice
//   null                            — silent (text-only)
//
// The browser tier is a *fallback*: it kicks in only when no server
// voice matches the language (currently the Korean case — Kokoro has
// zero ko voices). Everything below is lazily initialised on the first
// fallback request, so users whose languages have full server-voice
// coverage never call into `speechSynthesis` at all.
const _voiceCache = new Map();   // lang → resolved voice token (or null)

// `null` = uninitialised; `[]` or array = initialised. The flag is the
// gate inside `_initBrowserVoicesOnce` so we run the init exactly once
// per session, only when actually needed.
let _browserVoices = null;
let _browserUnavailable = false;

const _BROWSER_LOCALE = {
  ja: ['ja-JP', 'ja'],
  es: ['es-ES', 'es-MX', 'es-US', 'es-419', 'es'],
  zh: ['zh-CN', 'zh-Hans', 'zh-Hans-CN', 'zh-TW', 'zh'],
  fr: ['fr-FR', 'fr-CA', 'fr'],
  ko: ['ko-KR', 'ko'],
  // Forward-compat for the rest of Kokoro's lineup + common targets.
  en: ['en-US', 'en-GB', 'en'],
  it: ['it-IT', 'it'],
  pt: ['pt-BR', 'pt-PT', 'pt'],
  de: ['de-DE', 'de'],
  hi: ['hi-IN', 'hi'],
};

function _initBrowserVoicesOnce() {
  if (_browserVoices !== null || _browserUnavailable) return;
  if (typeof speechSynthesis === 'undefined') {
    _browserUnavailable = true;
    return;
  }
  _browserVoices = speechSynthesis.getVoices() || [];
  // Chrome/Edge populate voices asynchronously — first getVoices() is
  // often empty, then `voiceschanged` fires once with the real list.
  // We listen once (auto-unregisters) so we don't accumulate handlers.
  try {
    speechSynthesis.addEventListener('voiceschanged', () => {
      _browserVoices = speechSynthesis.getVoices() || [];
    }, { once: true });
  } catch { /* old browsers: no addEventListener on synth */ }
}

function _pickBrowserVoice(lang) {
  _initBrowserVoicesOnce();
  if (_browserUnavailable || !_browserVoices || _browserVoices.length === 0) return null;
  const candidates = _BROWSER_LOCALE[lang] || [lang];
  for (const tag of candidates) {
    const lowered = tag.toLowerCase();
    const hits = _browserVoices.filter(v =>
      v.lang === tag
      || v.lang.toLowerCase() === lowered
      || v.lang.toLowerCase().startsWith(lowered + '-')
    );
    if (!hits.length) continue;
    // Prefer on-device voices: free, fast, no cloud roundtrip = no
    // privacy footgun. Web Speech mostly returns these on modern OSes.
    hits.sort((a, b) => (b.localService ? 1 : 0) - (a.localService ? 1 : 0));
    return hits[0];
  }
  return null;
}

export async function resolveVoiceForLang(lang, fallback = null) {
  if (!lang) return fallback;
  if (_voiceCache.has(lang)) return _voiceCache.get(lang) || fallback;
  let pick = null;
  try {
    const r = await fetch('/api/audio/voices');
    if (r.ok) {
      const voices = await r.json();
      const matches = (Array.isArray(voices) ? voices : []).filter(v => v.lang === lang);
      // Recommended grade > anything else; within tier, first match wins.
      matches.sort((a, b) => (b.recommended ? 1 : 0) - (a.recommended ? 1 : 0));
      pick = matches.length ? (matches[0].name || matches[0].voice_id || null) : null;
    }
  } catch { /* fall through to browser-voice fallback */ }
  if (!pick) {
    const bv = _pickBrowserVoice(lang);
    if (bv) pick = `browser:${bv.lang}:${bv.name}`;
  }
  _voiceCache.set(lang, pick);
  return pick || fallback;
}

// ── Audio ───────────────────────────────────────────────────────────

let _activeAudio = null;
let _activeUtterance = null;

export async function speakWord(text, voice, { rate = 1.0 } = {}) {
  if (!voice || !text) return null;
  stopAudio();
  if (typeof voice === 'string' && voice.startsWith('browser:')) {
    if (typeof speechSynthesis === 'undefined') return null;
    // Token format: browser:<BCP-47>:<voice name>. The voice name can
    // itself contain colons (rare), so re-join everything after index 1.
    const colonAfterScheme = voice.indexOf(':');
    const colonAfterLocale = voice.indexOf(':', colonAfterScheme + 1);
    const locale = voice.slice(colonAfterScheme + 1, colonAfterLocale) || 'en-US';
    const name = voice.slice(colonAfterLocale + 1) || '';
    const u = new SpeechSynthesisUtterance(String(text));
    u.lang = locale;
    u.rate = Math.max(0.1, Math.min(10, rate));
    // Re-bind to the exact SpeechSynthesisVoice object if we can find
    // it — otherwise the browser picks any voice that matches `u.lang`.
    if (name && _browserVoices) {
      const match = _browserVoices.find(v => v.name === name && v.lang === locale);
      if (match) u.voice = match;
    }
    _activeUtterance = u;
    try { speechSynthesis.speak(u); } catch { /* */ }
    return null;
  }
  try {
    const r = await fetch('/v1/audio/speech', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: 'tts-1', voice, input: String(text), response_format: 'mp3' }),
    });
    if (!r.ok) return null;
    const blob = await r.blob();
    const audio = new Audio(URL.createObjectURL(blob));
    if (rate !== 1.0) audio.playbackRate = rate;
    _activeAudio = audio;
    await audio.play();
    return audio;
  } catch { return null; }
}

export function stopAudio() {
  if (_activeAudio) {
    try { _activeAudio.pause(); } catch { /* */ }
    _activeAudio = null;
  }
  if (_activeUtterance) {
    try { speechSynthesis.cancel(); } catch { /* */ }
    _activeUtterance = null;
  }
}

// ── Randomness ──────────────────────────────────────────────────────

export function shuffle(arr) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function pickN(arr, n) { return shuffle(arr).slice(0, n); }

export function pickOne(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

// ── Overlay scaffolding ─────────────────────────────────────────────

export function makeGameOverlay({ id, palette, title, onClose }) {
  document.getElementById(id)?.remove();
  const overlay = document.createElement('div');
  overlay.id = id;
  overlay.className = `lg-overlay lg-overlay-${palette || 'default'}`;
  overlay.setAttribute('data-game', id);

  const cleanup = [];
  const close = () => {
    stopAudio();
    cleanup.forEach(fn => { try { fn(); } catch { /* */ } });
    overlay.remove();
    document.removeEventListener('keydown', onKey);
    if (typeof onClose === 'function') onClose();
  };
  function onKey(e) { if (e.key === 'Escape') close(); }
  document.addEventListener('keydown', onKey);
  document.body.appendChild(overlay);

  return {
    overlay, close,
    addCleanup: (fn) => cleanup.push(fn),
    titleSuffix: title || '',
  };
}

export function fmtScore(n) {
  return String(n).padStart(4, '0');
}

// Themed empty-state overlay. Used when a game can't run on the user's
// current data (pool too small, mic unavailable, pack lacks sentences).
// Routes through makeGameOverlay so each empty state gets the
// Escape-key handler, audio cleanup, fade-in animation, and the
// palette's atmosphere — the per-game _empty() helpers were duplicating
// all that ten times and falling back to a flat beige box.
export function makeEmptyOverlay({ palette, message, emoji, hint }) {
  const { overlay, close, addCleanup } = makeGameOverlay({
    id: `lg-empty-${palette || 'default'}-${Date.now()}`,
    palette: palette || 'default',
  });
  overlay.classList.add('lg-empty-overlay');
  overlay.innerHTML = `
    <div class="lg-empty-stage">
      <div class="lg-empty">
        ${emoji ? `<div class="lg-empty-emoji" aria-hidden="true">${emoji}</div>` : ''}
        <div class="lg-empty-text">${escapeHtml(message)}</div>
        ${hint ? `<div class="lg-empty-hint">${escapeHtml(hint)}</div>` : ''}
        <button type="button" class="btn btn-primary lg-empty-close">OK</button>
      </div>
    </div>`;
  overlay.querySelector('.lg-empty-close').addEventListener('click', () => close());
  return { overlay, close, addCleanup };
}

// Simple word-level diff for STT/typed-answer grading.
export function similarity(a, b) {
  const sa = String(a || '').trim().toLowerCase();
  const sb = String(b || '').trim().toLowerCase();
  if (!sa || !sb) return 0;
  if (sa === sb) return 1;
  const m = sa.length, n = sb.length;
  const dp = Array(n + 1).fill(0).map(() => Array(m + 1).fill(0));
  for (let i = 0; i <= n; i += 1) dp[i][0] = i;
  for (let j = 0; j <= m; j += 1) dp[0][j] = j;
  for (let i = 1; i <= n; i += 1) {
    for (let j = 1; j <= m; j += 1) {
      const cost = sb[i - 1] === sa[j - 1] ? 0 : 1;
      dp[i][j] = Math.min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost);
    }
  }
  return 1 - dp[n][m] / Math.max(m, n);
}

// Confetti-ish particle burst on a DOM element.
export function burstAt(el, color = 'gold') {
  const r = el.getBoundingClientRect();
  const cx = r.left + r.width / 2;
  const cy = r.top + r.height / 2;
  for (let i = 0; i < 14; i += 1) {
    const p = document.createElement('span');
    p.className = 'lg-particle';
    p.style.background = color;
    p.style.left = `${cx}px`;
    p.style.top = `${cy}px`;
    const ang = (i / 14) * Math.PI * 2;
    const dist = 60 + Math.random() * 40;
    p.style.setProperty('--dx', `${Math.cos(ang) * dist}px`);
    p.style.setProperty('--dy', `${Math.sin(ang) * dist}px`);
    document.body.appendChild(p);
    setTimeout(() => p.remove(), 800);
  }
}

// ── Readable L2 text: clickable tokens + contextual-meaning popover ───
//
// The suite's core "translate the language to the user" primitive, lifted
// out of companion_dialogue (the only game that had it) and generalized so
// every game can render target-language text where the learner taps ANY
// word and sees what it means IN THIS SENTENCE — its role, its lemma, the
// whole-sentence breakdown — instead of a flat dictionary gloss[0].
//
// Two-call API:
//   el.innerHTML = await makeClickableHTML(sentence, lang);
//   el.dataset.fullText = sentence;                 // sentence context
//   const reader = attachReaderPopover(stageEl, { lang, voice, onWordAdded });
// Tokens carry class `lg-tok` + data-w. The popover resolves each token's
// containing sentence from the nearest [data-full-text] ancestor, runs one
// cached breakdownContextual() call per sentence, and degrades to a plain
// dictionary lookup when the LLM is unavailable.

const _CJK_TOKENIZE = new Set(['ja', 'zh']);

// Render `text` as HTML with each word wrapped in a clickable `lg-tok` span.
// CJK (no word spaces) uses the pack's dictionary segmenter; everything else
// splits on Unicode word boundaries. Punctuation/whitespace stay plain.
export async function makeClickableHTML(text, lang) {
  if (!text) return '';
  if (_CJK_TOKENIZE.has(lang)) {
    const bd = await fetchBreakdown(lang, text);
    const tokens = bd && Array.isArray(bd.tokens) ? bd.tokens : null;
    if (tokens && tokens.length) {
      return tokens.map((tok) => {
        const t = tok.text || '';
        if (!t) return '';
        return tok.matched
          ? `<span class="lg-tok" data-w="${escapeHtml(t)}">${escapeHtml(t)}</span>`
          : escapeHtml(t);
      }).join('');
    }
    // Segmenter unavailable (offline pack) — fall through to whitespace
    // tokenization rather than dropping click-to-define entirely.
  }
  const wordRe = /[\p{L}\p{N}]+/u;
  return text.split(/([\p{L}\p{N}]+)/u).map((part) => {
    if (!part) return '';
    return wordRe.test(part)
      ? `<span class="lg-tok" data-w="${escapeHtml(part)}">${escapeHtml(part)}</span>`
      : escapeHtml(part);
  }).join('');
}

function _findInBreakdown(breakdown, clickedSurface) {
  if (!Array.isArray(breakdown) || !clickedSurface) return null;
  const target = clickedSurface.toLowerCase();
  return (
    breakdown.find((t) => (t.token || '').toLowerCase() === target)
    || breakdown.find((t) => (t.token || '').toLowerCase().split(/\s+/).includes(target))
    || breakdown.find((t) => target.includes((t.token || '').toLowerCase()))
  );
}

// Install ONE delegated click handler on `rootEl`. Any `.lg-tok` click opens
// the contextual-meaning popover for that word within its sentence. Returns
// `{ destroy() }`. `onWordAdded` (optional) fires after a successful add.
export function attachReaderPopover(rootEl, { lang, voice, onWordAdded } = {}) {
  if (!rootEl) return { destroy() {} };
  _ensureReaderStyles();
  const cache = new Map();      // sentence text → breakdown array
  const inflight = new Map();
  let pop = null;

  async function getBreakdown(text, bubbleEl) {
    if (cache.has(text)) return cache.get(text);
    if (inflight.has(text)) return inflight.get(text);
    bubbleEl?.classList?.add('lg-tok-analysing');
    const p = (async () => {
      const result = await breakdownContextual(text, lang);
      cache.set(text, result || []);
      inflight.delete(text);
      bubbleEl?.classList?.remove('lg-tok-analysing');
      return result || [];
    })();
    inflight.set(text, p);
    return p;
  }

  function dismiss() { pop?.remove(); pop = null; }

  async function dictionaryFallback(word) {
    pop.classList.remove('lg-rdr-pop-context');
    pop.innerHTML = `<div class="lg-rdr-pop-loading">Dictionary lookup for "${escapeHtml(word)}"…</div>`;
    try {
      const r = await fetch(`/api/learning/lookup?lang=${encodeURIComponent(lang)}&q=${encodeURIComponent(word)}`);
      const entries = r.ok ? (await r.json()).entries || [] : [];
      if (!entries.length) {
        pop.innerHTML = `<div class="lg-rdr-pop-empty">No dictionary entry for "${escapeHtml(word)}".</div>`;
        return;
      }
      const e = entries[0];
      pop.innerHTML = `
        <div class="lg-rdr-pop-head">
          <span class="lg-rdr-pop-surface">${escapeHtml(e.surface)}</span>
          <span class="lg-rdr-pop-reading">${escapeHtml(e.reading || '')}</span>
          ${voice ? '<button type="button" class="lg-rdr-pop-speak" aria-label="Speak">🔊</button>' : ''}
          <button type="button" class="lg-rdr-pop-close" aria-label="Close">×</button>
        </div>
        <div class="lg-rdr-pop-gloss">${escapeHtml((e.glosses || []).slice(0, 5).join(' · '))}</div>
        <button type="button" class="btn btn-primary lg-rdr-pop-add">+ Add to my words</button>`;
      pop.querySelector('.lg-rdr-pop-speak')?.addEventListener('click', () => speakWord(e.reading || e.surface, voice));
      pop.querySelector('.lg-rdr-pop-close')?.addEventListener('click', dismiss);
      pop.querySelector('.lg-rdr-pop-add')?.addEventListener('click', async (ev) => {
        const btn = ev.currentTarget;
        btn.disabled = true;
        const ok = await addWord(lang, e.word_id);
        btn.textContent = ok ? '✓ Added' : 'Try again';
        if (ok) { onWordAdded?.(e.word_id); setTimeout(dismiss, 800); } else { btn.disabled = false; }
      });
    } catch {
      pop.innerHTML = `<div class="lg-rdr-pop-empty">Lookup failed.</div>`;
    }
  }

  async function open(anchor, clickedWord, fullText, bubbleEl) {
    dismiss();
    pop = document.createElement('div');
    pop.className = 'lg-rdr-pop lg-rdr-pop-context';
    pop.innerHTML = `<div class="lg-rdr-pop-loading">Reading "${escapeHtml(clickedWord)}" in context…</div>`;
    document.body.appendChild(pop);
    const rect = anchor.getBoundingClientRect();
    pop.style.left = `${Math.max(8, Math.min(rect.left, window.innerWidth - 380))}px`;
    pop.style.top = `${rect.bottom + 6}px`;
    if (rect.bottom + 240 > window.innerHeight) {
      pop.style.top = `${Math.max(8, rect.top - 240)}px`;
    }
    setTimeout(() => {
      function off(e) {
        if (pop && !pop.contains(e.target)) { dismiss(); document.removeEventListener('mousedown', off); }
      }
      document.addEventListener('mousedown', off);
    }, 0);

    const breakdown = await getBreakdown(fullText, bubbleEl);
    if (!pop) return;   // dismissed while awaiting
    if (!Array.isArray(breakdown) || breakdown.length === 0) {
      dictionaryFallback(clickedWord);
      return;
    }
    const hit = _findInBreakdown(breakdown, clickedWord);
    const rows = breakdown.map((t) => {
      const isHit = hit && t === hit;
      return `<li class="lg-rdr-bd-row${isHit ? ' lg-rdr-bd-row-hit' : ''}">
        <span class="lg-rdr-bd-token">${escapeHtml(t.token || '')}</span>
        <span class="lg-rdr-bd-meaning">${escapeHtml(t.meaning || '')}</span>
        <span class="lg-rdr-bd-role">${escapeHtml(t.role || '')}</span>
      </li>`;
    }).join('');
    pop.innerHTML = `
      ${hit ? `
        <div class="lg-rdr-bd-focus">
          <div class="lg-rdr-bd-focus-head">
            <span class="lg-rdr-bd-focus-token">${escapeHtml(hit.token)}</span>
            ${voice ? '<button type="button" class="lg-rdr-pop-speak" aria-label="Speak">🔊</button>' : ''}
            <button type="button" class="lg-rdr-pop-close" aria-label="Close">×</button>
          </div>
          <div class="lg-rdr-bd-focus-meaning">${escapeHtml(hit.meaning || '')}</div>
          <div class="lg-rdr-bd-focus-role">${escapeHtml(hit.role || '')}${hit.lemma && hit.lemma !== hit.token ? ` · lemma: <i>${escapeHtml(hit.lemma)}</i>` : ''}</div>
          <div class="lg-rdr-bd-actions">
            <button type="button" class="btn btn-primary lg-rdr-bd-add" data-surface="${escapeHtml(hit.lemma || hit.token)}">+ Add to my words</button>
          </div>
        </div>` : `
        <div class="lg-rdr-bd-focus">
          <div class="lg-rdr-bd-focus-head">
            <span class="lg-rdr-bd-focus-token">${escapeHtml(clickedWord)}</span>
            <button type="button" class="lg-rdr-pop-close" aria-label="Close">×</button>
          </div>
          <div class="lg-rdr-bd-focus-role">In-sentence meaning below.</div>
        </div>`}
      <div class="lg-rdr-bd-divider"></div>
      <div class="lg-rdr-bd-label">Sentence breakdown</div>
      <ul class="lg-rdr-bd-list">${rows}</ul>`;
    pop.querySelector('.lg-rdr-pop-close')?.addEventListener('click', dismiss);
    pop.querySelector('.lg-rdr-pop-speak')?.addEventListener('click', () => speakWord(hit?.token || clickedWord, voice));
    pop.querySelector('.lg-rdr-bd-add')?.addEventListener('click', async (ev) => {
      const btn = ev.currentTarget;
      const surface = btn.dataset.surface;
      btn.disabled = true;
      btn.textContent = 'Adding…';
      try {
        const r = await fetch(`/api/learning/lookup?lang=${encodeURIComponent(lang)}&q=${encodeURIComponent(surface)}`);
        const entries = r.ok ? (await r.json()).entries || [] : [];
        if (!entries.length) { btn.textContent = 'No dictionary entry'; return; }
        const ok = await addWord(lang, entries[0].word_id);
        if (ok) { onWordAdded?.(entries[0].word_id); btn.textContent = '✓ Added'; btn.classList.add('lg-rdr-bd-added'); }
        else { btn.disabled = false; btn.textContent = 'Try again'; }
      } catch { btn.disabled = false; btn.textContent = 'Try again'; }
    });
  }

  function onClick(e) {
    const tgt = e.target;
    if (!tgt.classList || !tgt.classList.contains('lg-tok')) return;
    const word = tgt.dataset.w;
    if (!word) return;
    const bubbleEl = tgt.closest('[data-full-text]') || rootEl;
    const fullText = (bubbleEl && bubbleEl.dataset && bubbleEl.dataset.fullText)
      ? bubbleEl.dataset.fullText : (bubbleEl?.textContent || word);
    open(tgt, word, fullText, bubbleEl);
  }
  rootEl.addEventListener('click', onClick);
  return { destroy() { rootEl.removeEventListener('click', onClick); dismiss(); } };
}

// ── Reliability + grading helpers ────────────────────────────────────

// Effort-aware FSRS grade. The suite was flat-grading every correct answer 3
// ("Good") regardless of difficulty, and grading noisy signals — which
// inflates the learner's real scheduler. Map a small evidence bundle to a
// conservative grade instead. correct=false → 1 (Again). Friction (extra
// attempts, hints, replays) pulls a correct answer down toward Hard; a fast,
// clean, unaided answer earns Easy.
export function gradeForEffort({ correct, attempts = 1, hintsUsed = 0, replays = 0, ms = 0 } = {}) {
  if (!correct) return 1;
  const friction = Math.max(0, attempts - 1) + hintsUsed + replays;
  if (friction >= 2) return 2;          // Hard — got there but struggled
  if (friction === 1) return 3;         // Good — one stumble
  if (ms > 0 && ms < 2500) return 4;    // Easy — fast and unaided
  return 3;                             // Good — clean but not fast
}

// LLM call that returns parsed JSON or `fallback`, with a timeout so a cold
// model never hangs a game's critical path. For grammar/equivalence judges
// that MUST degrade offline. The underlying request is fire-and-forget after
// the timeout (no abort) — matches the rest of this module's tolerance.
export async function llmJudgeJSON(messages, { fallback = null, timeoutMs = 8000, model } = {}) {
  const raw = await Promise.race([
    llmChat(messages, model),
    new Promise((res) => setTimeout(() => res(''), timeoutMs)),
  ]);
  if (!raw) return fallback;
  const m = raw.match(/\{[\s\S]*\}/);
  if (!m) return fallback;
  try { return JSON.parse(m[0]); } catch { return fallback; }
}

// Model-warming shimmer for generative surfaces, so a cold local model
// (30-60s load) reads as progress, not a freeze. Returns a stop() fn.
export function showWarming(el, label = 'Warming up the model…') {
  if (!el) return () => {};
  _ensureReaderStyles();
  el.innerHTML = `<div class="lg-warming">
    <div class="lg-warming-dots"><span></span><span></span><span></span></div>
    <div class="lg-warming-label">${escapeHtml(label)}</div>
    <div class="lg-warming-elapsed"></div></div>`;
  const t0 = Date.now();
  const tick = setInterval(() => {
    const s = Math.round((Date.now() - t0) / 1000);
    const e = el.querySelector('.lg-warming-elapsed');
    if (e) e.textContent = s > 2 ? `${s}s${s > 25 ? ' · loading a local model the first time can take a minute' : ''}` : '';
  }, 1000);
  return () => clearInterval(tick);
}

// One honest line for degraded states (no voice, no STT, model offline).
// Games were failing silently; this prepends a dismissable notice bar to an
// overlay. kind ∈ {info, warn}.
export function showNotice(overlayOrEl, message, { kind = 'info' } = {}) {
  if (!overlayOrEl) return;
  _ensureReaderStyles();
  let bar = overlayOrEl.querySelector(':scope > .lg-notice');
  if (!bar) {
    bar = document.createElement('div');
    bar.className = 'lg-notice';
    overlayOrEl.prepend(bar);
  }
  bar.dataset.kind = kind;
  bar.textContent = message;
}

let _readerStylesInjected = false;
function _ensureReaderStyles() {
  if (_readerStylesInjected) return;
  _readerStylesInjected = true;
  const css = `
.lg-tok { cursor: pointer; border-radius: 3px; transition: background .12s; }
.lg-tok:hover { background: color-mix(in srgb, var(--accent, #6ea8fe) 24%, transparent); }
.lg-tok-analysing { opacity: .85; }
.lg-rdr-pop { position: fixed; z-index: 100000; max-width: 360px; min-width: 220px;
  background: var(--bg-elevated, #1b1d24); color: var(--text-primary, #e8e8ea);
  border: 1px solid var(--border, rgba(255,255,255,.12)); border-radius: 12px;
  padding: 12px 14px; box-shadow: 0 12px 40px rgba(0,0,0,.45);
  font-size: 14px; line-height: 1.4; animation: lg-rdr-in .14s ease; }
@keyframes lg-rdr-in { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }
.lg-rdr-pop-loading, .lg-rdr-pop-empty { color: var(--text-muted, #9aa0aa); font-style: italic; }
.lg-rdr-pop-head, .lg-rdr-bd-focus-head { display: flex; align-items: center; gap: 8px; }
.lg-rdr-pop-surface, .lg-rdr-bd-focus-token { font-size: 18px; font-weight: 700; }
.lg-rdr-pop-reading { color: var(--text-muted, #9aa0aa); }
.lg-rdr-pop-speak, .lg-rdr-pop-close { margin-left: auto; background: none; border: none;
  color: var(--text-muted, #9aa0aa); cursor: pointer; font-size: 16px; padding: 2px 4px; }
.lg-rdr-pop-close { margin-left: 4px; }
.lg-rdr-pop-gloss { margin: 8px 0; }
.lg-rdr-bd-focus-meaning { margin: 6px 0 2px; font-weight: 600; }
.lg-rdr-bd-focus-role { color: var(--text-muted, #9aa0aa); font-size: 12.5px; }
.lg-rdr-bd-actions { margin-top: 8px; }
.lg-rdr-bd-add { font-size: 13px; }
.lg-rdr-bd-added { background: var(--success, #30a46c) !important; }
.lg-rdr-bd-divider { height: 1px; background: var(--border, rgba(255,255,255,.1)); margin: 10px 0 8px; }
.lg-rdr-bd-label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--text-muted, #9aa0aa); margin-bottom: 4px; }
.lg-rdr-bd-list { list-style: none; margin: 0; padding: 0; max-height: 200px; overflow-y: auto; }
.lg-rdr-bd-row { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; padding: 4px 6px; border-radius: 6px; }
.lg-rdr-bd-row-hit { background: color-mix(in srgb, var(--accent, #6ea8fe) 18%, transparent); }
.lg-rdr-bd-token { font-weight: 600; }
.lg-rdr-bd-meaning { color: var(--text-secondary, #c2c5cc); }
.lg-rdr-bd-role { grid-column: 2; color: var(--text-muted, #9aa0aa); font-size: 11.5px; }
.lg-warming { display: flex; flex-direction: column; align-items: center; gap: 10px; padding: 28px; color: var(--text-muted, #9aa0aa); }
.lg-warming-dots { display: flex; gap: 6px; }
.lg-warming-dots span { width: 9px; height: 9px; border-radius: 50%; background: var(--accent, #6ea8fe); animation: lg-warm-b 1s ease-in-out infinite; }
.lg-warming-dots span:nth-child(2) { animation-delay: .15s; }
.lg-warming-dots span:nth-child(3) { animation-delay: .3s; }
@keyframes lg-warm-b { 0%, 100% { opacity: .3; transform: translateY(0); } 50% { opacity: 1; transform: translateY(-5px); } }
.lg-warming-label { font-size: 14px; }
.lg-warming-elapsed { font-size: 12px; opacity: .7; }
.lg-notice { padding: 8px 14px; font-size: 13px; text-align: center;
  background: color-mix(in srgb, var(--warning, #e0a800) 18%, transparent);
  color: var(--text-primary, #e8e8ea); border-bottom: 1px solid var(--border, rgba(255,255,255,.1)); }
.lg-notice[data-kind="info"] { background: color-mix(in srgb, var(--accent, #6ea8fe) 16%, transparent); }
`;
  const style = document.createElement('style');
  style.id = 'lg-shared-reader-styles';
  style.textContent = css;
  document.head.appendChild(style);
}
