/**
 * Echo Chamber — "Hear it. Decode it. Pick its meaning."
 *
 * A sentence (real Tatoeba example or pack sentence) is spoken via TTS.
 * Four English meaning options appear — one is the true translation, the
 * others are distractor glosses pulled elsewhere in the user's pool.
 * Replays the audio at a slower speed on a miss.
 *
 * Why audio-only: Tatoeba pages let users *read* sentences with English
 * underneath. That short-circuits the listening muscle. Hiding the text
 * forces the same comprehension Anki/Yomichan never train.
 *
 * --- Upgrades (2026-06-17) ---
 * #1  "Decode it" reveal: after every answer (always on miss, optional on
 *     correct) the L2 sentence appears as tappable tokens (makeClickableHTML
 *     + attachReaderPopover). Full sentence breakdown via breakdownContextual
 *     chips. This is the comprehension payoff the tagline promises.
 * #1b No-voice reading-mode: when `voice` is null/unavailable showNotice
 *     makes it explicit, and the sentence is revealed upfront so learners
 *     read for meaning instead of hearing silence.
 * #2  Replay-aware FSRS via gradeForEffort instead of flat grade 3 / 1.
 * #3  Distractor quality: llmJudgeJSON (timeout+fallback) + similarity-based
 *     dedup on fallback distractors + length-tier preference.
 * #4  Type-what-you-heard (dictation) mode: every ~5th question swaps the
 *     4-option layout for an input field; graded by similarity >= 0.75.
 *     Skipped when TTS is unavailable (no-voice mode).
 */

import {
  escapeHtml,
  fetchGamePool,
  fetchSentences,
  speakWord,
  pickN,
  pickOne,
  shuffle,
  makeGameOverlay,
  makeEmptyOverlay,
  gradeCard,
  burstAt,
  fmtScore,
  llmChat,
  llmJudgeJSON,
  recordResult,
  fetchBestScores,
  similarity,
  makeClickableHTML,
  attachReaderPopover,
  gradeForEffort,
  showNotice,
  breakdownContextual,
} from './_common.js';

const ROUND_QUESTIONS = 10;
// Every Nth question (1-indexed) uses dictation mode instead of 4-option.
// Skipped automatically when voice is unavailable.
const DICTATION_INTERVAL = 5;
// similarity threshold for accepting a dictation answer as correct
const DICTATION_THRESHOLD = 0.75;

// ── CSS (guarded: injected once) ────────────────────────────────────────────

function _ensureEchoStyles() {
  if (document.getElementById('lg-echo-styles')) return;
  const style = document.createElement('style');
  style.id = 'lg-echo-styles';
  style.textContent = `
/* ── Echo Chamber layout ── */
.lg-ec { display: flex; flex-direction: column; height: 100%; }
.lg-ec-stage { flex: 0 0 auto; display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: 14px; padding: 28px 20px 18px; position: relative; }
.lg-ec-rings { position: absolute; inset: 0; pointer-events: none; display: flex;
  align-items: center; justify-content: center; }
.lg-ec-rings div { position: absolute; border-radius: 50%;
  border: 2px solid color-mix(in srgb, var(--accent, #bb9bff) 22%, transparent);
  animation: lg-ec-ring 2.6s ease-out infinite; }
.lg-ec-rings div:nth-child(1) { width: 120px; height: 120px; }
.lg-ec-rings div:nth-child(2) { width: 170px; height: 170px; animation-delay: .35s; }
.lg-ec-rings div:nth-child(3) { width: 220px; height: 220px; animation-delay: .7s; }
.lg-ec-rings div:nth-child(4) { width: 275px; height: 275px; animation-delay: 1.05s; }
@keyframes lg-ec-ring {
  0%   { opacity: .6; transform: scale(.85); }
  60%  { opacity: .15; }
  100% { opacity: 0;  transform: scale(1.08); }
}
.lg-ec-speak { width: 76px; height: 76px; border-radius: 50%; border: none; cursor: pointer;
  background: linear-gradient(135deg, var(--accent, #bb9bff) 0%, color-mix(in srgb, var(--accent,#bb9bff) 60%, #6ea8fe) 100%);
  color: #fff; font-size: 28px; position: relative; z-index: 1;
  transition: transform .14s, box-shadow .14s;
  box-shadow: 0 4px 18px color-mix(in srgb, var(--accent,#bb9bff) 45%, transparent); }
.lg-ec-speak:hover { transform: scale(1.07); }
.lg-ec-pulse { animation: lg-ec-pulse .6s ease-out forwards; }
@keyframes lg-ec-pulse {
  0%   { box-shadow: 0 0 0   0 color-mix(in srgb, var(--accent,#bb9bff) 55%, transparent); }
  70%  { box-shadow: 0 0 0 18px transparent; }
  100% { box-shadow: 0 0 0   0 transparent; }
}
.lg-ec-hint { font-size: 13.5px; color: var(--text-muted, #9aa0aa); min-height: 20px; }
.lg-ec-skip { background: none; border: none; color: var(--text-muted, #9aa0aa);
  font-size: 13px; cursor: pointer; padding: 2px 6px; }
.lg-ec-skip:hover { color: var(--text-secondary, #c2c5cc); }

/* options grid */
.lg-ec-options { flex: 1 1 auto; display: grid; grid-template-columns: 1fr 1fr;
  gap: 10px; padding: 0 16px 12px; align-content: start; }
.lg-ec-opt { width: 100%; min-height: 58px; padding: 10px 14px; border-radius: 12px; border: none;
  background: color-mix(in srgb, var(--accent, #bb9bff) 10%, var(--bg-elevated, #1e1f2a));
  color: var(--text-primary, #e8e8ea); font-size: 14px; text-align: center;
  cursor: pointer; transition: background .14s, transform .1s;
  line-height: 1.35; word-break: break-word; }
.lg-ec-opt:hover:not(:disabled) { background: color-mix(in srgb, var(--accent,#bb9bff) 22%, var(--bg-elevated,#1e1f2a)); }
.lg-ec-opt.lg-ec-right { background: color-mix(in srgb, var(--success,#30a46c) 35%, var(--bg-elevated,#1e1f2a));
  border: 2px solid var(--success,#30a46c); pointer-events: none; }
.lg-ec-opt.lg-ec-wrong { background: color-mix(in srgb, var(--danger,#e05c5c) 28%, var(--bg-elevated,#1e1f2a));
  border: 2px solid var(--danger,#e05c5c); pointer-events: none; }

/* dictation area */
.lg-ec-dictation { flex: 1 1 auto; display: flex; flex-direction: column;
  align-items: center; justify-content: center; gap: 14px; padding: 0 20px 20px; }
.lg-ec-dict-label { font-size: 13.5px; color: var(--text-muted,#9aa0aa); }
.lg-ec-dict-input { width: 100%; max-width: 460px; padding: 12px 16px; border-radius: 12px;
  border: 1.5px solid var(--border, rgba(255,255,255,.14));
  background: var(--bg-elevated, #1e1f2a); color: var(--text-primary, #e8e8ea);
  font-size: 16px; outline: none; text-align: center; }
.lg-ec-dict-input:focus { border-color: var(--accent,#bb9bff); }
.lg-ec-dict-submit { margin-top: 4px; }
.lg-ec-dict-result { font-size: 14px; min-height: 20px; text-align: center; }
.lg-ec-dict-result.lg-ec-right { color: var(--success,#30a46c); }
.lg-ec-dict-result.lg-ec-wrong { color: var(--danger,#e05c5c); }

/* reveal panel — the "Decode it" comprehension payoff */
.lg-ec-reveal { flex: 0 0 auto; margin: 0 16px 14px; padding: 12px 14px;
  border-radius: 12px; background: color-mix(in srgb, var(--accent,#bb9bff) 8%, var(--bg-elevated,#1e1f2a));
  border: 1px solid color-mix(in srgb, var(--accent,#bb9bff) 22%, transparent);
  animation: lg-ec-reveal-in .22s ease; }
@keyframes lg-ec-reveal-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.lg-ec-reveal-row { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; margin-bottom: 6px; }
.lg-ec-reveal-lang { font-size: 15px; font-weight: 600; line-height: 1.4; }
.lg-ec-reveal-en { font-size: 13px; color: var(--text-muted,#9aa0aa); }
.lg-ec-reveal-actions { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.lg-ec-reveal-breakdown-toggle { background: none; border: 1px solid var(--border,rgba(255,255,255,.14));
  color: var(--text-secondary,#c2c5cc); border-radius: 8px; padding: 4px 10px;
  font-size: 12px; cursor: pointer; }
.lg-ec-reveal-breakdown-toggle:hover { background: color-mix(in srgb, var(--accent,#bb9bff) 14%, transparent); }
.lg-ec-breakdown-chips { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }
.lg-ec-chip { display: inline-flex; flex-direction: column; align-items: center;
  gap: 2px; padding: 5px 8px; border-radius: 8px; min-width: 36px;
  background: color-mix(in srgb, var(--accent,#bb9bff) 12%, var(--bg,#15161e));
  font-size: 12px; cursor: default; }
.lg-ec-chip-tok { font-weight: 700; font-size: 14px; }
.lg-ec-chip-meaning { color: var(--text-muted,#9aa0aa); font-size: 11px; text-align: center; }
.lg-ec-chip-role { color: color-mix(in srgb, var(--accent,#bb9bff) 80%, transparent);
  font-size: 10px; text-align: center; }
.lg-ec-breakdown-loading { color: var(--text-muted,#9aa0aa); font-size: 12px; font-style: italic; margin-top: 6px; }

/* reading-mode (no-voice) upfront sentence */
.lg-ec-reading-mode { flex: 0 0 auto; margin: 0 16px 10px; padding: 10px 14px;
  border-radius: 10px; background: color-mix(in srgb, var(--accent,#bb9bff) 9%, var(--bg-elevated,#1e1f2a));
  border: 1px solid color-mix(in srgb, var(--accent,#bb9bff) 18%, transparent); }
.lg-ec-reading-sentence { font-size: 16px; font-weight: 600; line-height: 1.5; }

/* HUD */
.lg-hud { display: flex; align-items: center; gap: 12px; padding: 10px 14px 8px;
  border-bottom: 1px solid var(--border, rgba(255,255,255,.08)); }
.lg-close { background: none; border: none; color: var(--text-muted, #9aa0aa);
  font-size: 22px; cursor: pointer; line-height: 1; padding: 0 4px; }
.lg-close:hover { color: var(--text-primary,#e8e8ea); }
.lg-ec-progress { font-size: 13px; color: var(--text-muted,#9aa0aa); margin-right: auto; }
.lg-hud-stats { display: flex; align-items: baseline; gap: 6px; font-size: 16px;
  font-weight: 700; font-variant-numeric: tabular-nums; color: var(--accent,#bb9bff); }
.lg-hud-label { font-size: 11px; font-weight: 400; color: var(--text-muted,#9aa0aa); }

/* end panel */
.lg-end { display: flex; flex-direction: column; align-items: center; gap: 20px;
  padding: 32px 20px; height: 100%; justify-content: center; }
.lg-end-title { font-size: 22px; font-weight: 700; }
.lg-end-stats { display: flex; gap: 24px; flex-wrap: wrap; justify-content: center; }
.lg-end-stat { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.lg-end-stat-n { font-size: 28px; font-weight: 800; color: var(--accent,#bb9bff); }
.lg-end-actions { display: flex; gap: 12px; }
`;
  document.head.appendChild(style);
}

// ── Launch ───────────────────────────────────────────────────────────────────

export async function launchEchoChamber({ lang, voice, focusWords = [] }) {
  _ensureEchoStyles();

  const [pool, sentences, bests] = await Promise.all([
    fetchGamePool(lang, 30, 'mixed', focusWords),
    fetchSentences(lang, 30),
    fetchBestScores(lang),
  ]);
  if (pool.length < 4) {
    return makeEmptyOverlay({
      palette: 'purple', emoji: '🎧',
      message: 'Build up at least 4 words and you can play.',
      hint: 'Echo Chamber needs target words to surround with distractors.',
    });
  }
  const best = (bests.echo_chamber && bests.echo_chamber.best) || 0;

  // Detect whether voice is actually usable (null = no voice for this lang)
  const noVoice = !voice;

  const { overlay, close, addCleanup } = makeGameOverlay({
    id: 'lg-echo', palette: 'purple', title: 'Echo Chamber',
  });

  overlay.innerHTML = `
    <div class="lg-game lg-ec">
      <header class="lg-hud">
        <button type="button" class="lg-close" aria-label="Close">×</button>
        <div class="lg-ec-progress" id="lg-ec-progress">1 / ${ROUND_QUESTIONS}</div>
        <div class="lg-hud-stats">
          <span class="lg-hud-label">score</span> <span id="lg-ec-score">0000</span>
          ${best ? `<span class="lg-hud-label" style="margin-left:8px">best</span> <span>${escapeHtml(fmtScore(best))}</span>` : ''}
        </div>
      </header>
      <div class="lg-ec-stage" id="lg-ec-stage">
        <div class="lg-ec-rings"><div></div><div></div><div></div><div></div></div>
        <button type="button" class="lg-ec-speak" id="lg-ec-speak" aria-label="Replay"${noVoice ? ' hidden' : ''}>▶</button>
        <div class="lg-ec-hint" id="lg-ec-hint">${noVoice ? 'Read the sentence, pick its meaning' : 'Tap to hear it'}</div>
        <button type="button" class="lg-ec-skip" id="lg-ec-skip" title="Skip this question"${noVoice ? ' style="visibility:hidden"' : ''}>Skip ›</button>
      </div>
      <div id="lg-ec-reading-area"></div>
      <div class="lg-ec-options" id="lg-ec-options"></div>
      <div class="lg-ec-dictation" id="lg-ec-dictation" hidden></div>
      <div id="lg-ec-reveal-area"></div>
      <div class="lg-end" id="lg-ec-end" hidden>
        <div class="lg-end-title">Round complete</div>
        <div class="lg-end-stats" id="lg-ec-end-stats"></div>
        <div class="lg-end-actions">
          <button type="button" class="btn btn-primary" id="lg-ec-replay">Play again</button>
          <button type="button" class="btn btn-ghost" id="lg-ec-quit">Done</button>
        </div>
      </div>
    </div>`;

  if (noVoice) {
    showNotice(overlay, `No voice available for ${escapeHtml(lang)} — reading mode`, { kind: 'warn' });
  }

  overlay.querySelector('.lg-close').addEventListener('click', () => close());

  const speakBtn = overlay.querySelector('#lg-ec-speak');
  const hintEl = overlay.querySelector('#lg-ec-hint');
  const optsEl = overlay.querySelector('#lg-ec-options');
  const dictEl = overlay.querySelector('#lg-ec-dictation');
  const progressEl = overlay.querySelector('#lg-ec-progress');
  const scoreEl = overlay.querySelector('#lg-ec-score');
  const endPanel = overlay.querySelector('#lg-ec-end');
  const readingArea = overlay.querySelector('#lg-ec-reading-area');
  const revealArea = overlay.querySelector('#lg-ec-reveal-area');

  let score = 0;
  let qIdx = 0;
  let current = null;
  let answered = false;
  let questions = [];
  let streak = 0;
  let maxStreak = 0;
  const correct = [];
  const missed = [];
  let roundStart = performance.now();

  // Per-question replay tracking for gradeForEffort
  let _replaysThisQ = 0;
  let _answerStartMs = 0;

  // Reader popover handle — destroyed and recreated per question reveal
  let _readerHandle = null;

  // ── Seed building ──────────────────────────────────────────────────────

  function buildSeeds() {
    const sentenceCandidates = pool
      .filter(c => c.example && c.example.lang_text && c.example.en_text)
      .map(c => ({ card: c, prompt: c.example.lang_text, answer: c.example.en_text, length: c.example.lang_text.length }));
    const extras = sentences
      .filter(s => s.en_text)
      .map(s => ({ card: null, prompt: s.lang_text, answer: s.en_text, length: s.lang_text.length }));
    const all = [...sentenceCandidates, ...extras].sort((a, b) => a.length - b.length);

    const seeds = [];
    const used = new Set();
    const buckets = [
      shuffle(all.slice(0, Math.max(4, Math.floor(all.length / 3)))),
      shuffle(all.slice(Math.floor(all.length / 3), Math.floor(all.length * 2 / 3))),
      shuffle(all.slice(Math.floor(all.length * 2 / 3))),
    ];
    const fallback = shuffle(all);
    function takeUnique(source) {
      while (source.length) {
        const candidate = source.shift();
        if (!used.has(candidate.prompt)) return candidate;
      }
      return null;
    }
    for (let i = 0; i < ROUND_QUESTIONS; i += 1) {
      const bucket = buckets[Math.min(2, Math.floor(i / Math.ceil(ROUND_QUESTIONS / 3)))];
      let seed = takeUnique(bucket) || takeUnique(fallback);
      if (!seed) {
        const c = pickOne(pool);
        seed = { card: c, prompt: c.surface, answer: (c.glosses || [])[0] || c.surface, length: 0 };
      }
      used.add(seed.prompt);
      seeds.push(seed);
    }
    return seeds;
  }

  // ── Distractor quality ─────────────────────────────────────────────────
  // Similarity-filtered fallback: avoid near-duplicates of the answer and
  // of each other. Prefer distractors in the same length tier as the answer.

  function fallbackDistractors(seed, allSeeds) {
    const answerLower = String(seed.answer).toLowerCase();
    const answerLen = answerLower.length;

    // Gather a broad pool of candidates
    const raw = pool
      .filter(c => !seed.card || c.word_id !== seed.card.word_id)
      .flatMap(c => c.glosses || [])
      .concat(allSeeds.map(s => s.answer))
      .filter(g => g && g.toLowerCase() !== answerLower);

    // Dedupe by text, then filter near-duplicates of the answer (sim >= 0.6)
    const unique = Array.from(new Set(raw))
      .filter(d => similarity(d, seed.answer) < 0.6);

    // Prefer same rough length tier (within 2× of answer length), then fill from rest
    const sameTier = shuffle(unique.filter(d => {
      const r = d.length / Math.max(1, answerLen);
      return r >= 0.4 && r <= 2.5;
    }));
    const rest = shuffle(unique.filter(d => {
      const r = d.length / Math.max(1, answerLen);
      return !(r >= 0.4 && r <= 2.5);
    }));
    const ordered = [...sameTier, ...rest];

    // Final dedup pass: ensure selected distractors aren't near-duplicates of each other
    const selected = [];
    for (const d of ordered) {
      if (selected.length >= 3) break;
      if (selected.every(s => similarity(s, d) < 0.55)) selected.push(d);
    }
    while (selected.length < 3) selected.push('—');
    return selected;
  }

  // ── Smart distractors via llmJudgeJSON ─────────────────────────────────

  async function buildSmartDistractors(seeds) {
    const lines = seeds.map((s, i) => `${i + 1}. ${String(lang).toUpperCase()}: "${s.prompt}" / EN: "${s.answer}"`).join('\n');
    const sys = `You write multiple-choice distractors for a listening comprehension game. For each numbered ${lang}→English pair, produce 3 plausible-but-wrong English translations: similar topic or register, similar phrasing length, same part of speech as the correct answer, NOT a paraphrase of the correct answer, not trivially unrelated. Reply with strict JSON only (no prose): {"items": [{"i": 1, "distractors": ["...", "...", "..."]}, ...]}. Each distractor must be under 10 words.`;
    const result = await llmJudgeJSON(
      [{ role: 'system', content: sys }, { role: 'user', content: `Pairs:\n${lines}\n\nReturn smart distractors for each.` }],
      { fallback: null, timeoutMs: 4000 },
    );
    if (!result || !Array.isArray(result.items)) return null;
    const byIdx = {};
    for (const item of result.items) {
      const i = Number(item.i) - 1;
      if (i >= 0 && Array.isArray(item.distractors) && item.distractors.length >= 3) {
        byIdx[i] = item.distractors.slice(0, 3);
      }
    }
    return byIdx;
  }

  function buildOptions(seed, distractors) {
    return { ...seed, options: pickN([seed.answer, ...distractors.slice(0, 3)], 4) };
  }

  // Whether question at 1-indexed position i should use dictation mode
  function _isDictation(oneIdx) {
    if (noVoice) return false;    // no audio = no dictation
    return oneIdx % DICTATION_INTERVAL === 0;
  }

  // ── Reading-mode: reveal sentence upfront (no-voice path) ──────────────

  async function renderReadingMode(q) {
    readingArea.innerHTML = '';
    const wrap = document.createElement('div');
    wrap.className = 'lg-ec-reading-mode';
    const sentEl = document.createElement('div');
    sentEl.className = 'lg-ec-reading-sentence lg-tok-area';
    sentEl.dataset.fullText = q.prompt;
    const html = await makeClickableHTML(q.prompt, lang).catch(() => escapeHtml(q.prompt));
    sentEl.innerHTML = html;
    wrap.appendChild(sentEl);
    readingArea.appendChild(wrap);
    // Attach popover — destroy on next question
    if (_readerHandle) { _readerHandle.destroy(); _readerHandle = null; }
    _readerHandle = attachReaderPopover(sentEl, { lang, voice });
  }

  // ── Reveal panel: "Decode it" comprehension payoff ─────────────────────

  async function showReveal(q, wasCorrect) {
    // Destroy any prior reader
    if (_readerHandle) { _readerHandle.destroy(); _readerHandle = null; }
    revealArea.innerHTML = '';

    const wrap = document.createElement('div');
    wrap.className = 'lg-ec-reveal';

    // Build tappable L2 sentence
    const langHtml = await makeClickableHTML(q.prompt, lang).catch(() => escapeHtml(q.prompt));

    const row = document.createElement('div');
    row.className = 'lg-ec-reveal-row';
    const langSpan = document.createElement('span');
    langSpan.className = 'lg-ec-reveal-lang';
    langSpan.dataset.fullText = q.prompt;
    langSpan.innerHTML = langHtml;
    const enSpan = document.createElement('span');
    enSpan.className = 'lg-ec-reveal-en';
    enSpan.textContent = `= ${q.answer}`;
    row.appendChild(langSpan);
    row.appendChild(enSpan);
    wrap.appendChild(row);

    // Actions row: "See breakdown" toggle
    const actions = document.createElement('div');
    actions.className = 'lg-ec-reveal-actions';
    const bdBtn = document.createElement('button');
    bdBtn.type = 'button';
    bdBtn.className = 'lg-ec-reveal-breakdown-toggle';
    bdBtn.textContent = 'See breakdown';
    actions.appendChild(bdBtn);
    wrap.appendChild(actions);

    // Breakdown chips container (populated on first toggle)
    const chipsArea = document.createElement('div');
    chipsArea.hidden = true;
    wrap.appendChild(chipsArea);

    revealArea.appendChild(wrap);

    // Wire reader popover on the lang span
    _readerHandle = attachReaderPopover(langSpan, { lang, voice });

    // Breakdown toggle
    let bdLoaded = false;
    bdBtn.addEventListener('click', async () => {
      if (chipsArea.hidden) {
        chipsArea.hidden = false;
        bdBtn.textContent = 'Hide breakdown';
        if (!bdLoaded) {
          bdLoaded = true;
          chipsArea.innerHTML = `<div class="lg-ec-breakdown-loading">Analysing…</div>`;
          const bd = await breakdownContextual(q.prompt, lang).catch(() => null);
          if (!bd || !bd.length) {
            chipsArea.innerHTML = `<div class="lg-ec-breakdown-loading">Breakdown unavailable offline.</div>`;
            return;
          }
          const chips = bd.map(t => `
            <div class="lg-ec-chip">
              <span class="lg-ec-chip-tok">${escapeHtml(t.token || '')}</span>
              <span class="lg-ec-chip-meaning">${escapeHtml(t.meaning || '')}</span>
              <span class="lg-ec-chip-role">${escapeHtml(t.role || '')}</span>
            </div>`).join('');
          chipsArea.innerHTML = `<div class="lg-ec-breakdown-chips">${chips}</div>`;
        }
      } else {
        chipsArea.hidden = true;
        bdBtn.textContent = 'See breakdown';
      }
    });

    // Auto-open breakdown on a miss so learners always see the payoff
    if (!wasCorrect) {
      bdBtn.click();
    }
  }

  // ── Render a question ──────────────────────────────────────────────────

  async function render() {
    answered = false;
    _replaysThisQ = 0;
    current = questions[qIdx];
    progressEl.textContent = `${qIdx + 1} / ${ROUND_QUESTIONS}`;
    revealArea.innerHTML = '';
    if (_readerHandle) { _readerHandle.destroy(); _readerHandle = null; }

    const isDictQ = _isDictation(qIdx + 1);   // 1-indexed

    if (isDictQ) {
      // Dictation mode: hide options, show input
      optsEl.innerHTML = '';
      optsEl.hidden = true;
      dictEl.hidden = false;
      dictEl.innerHTML = `
        <div class="lg-ec-dict-label">Type what you heard</div>
        <input type="text" class="lg-ec-dict-input" id="lg-ec-dict-input"
          placeholder="Type the English translation…" autocomplete="off" spellcheck="true">
        <button type="button" class="btn btn-primary lg-ec-dict-submit" id="lg-ec-dict-submit">Submit</button>
        <div class="lg-ec-dict-result" id="lg-ec-dict-result"></div>`;
      hintEl.textContent = 'Listen, then type the translation';
      _answerStartMs = performance.now();
      await play(false);
      // Wire submit
      const inp = dictEl.querySelector('#lg-ec-dict-input');
      const sub = dictEl.querySelector('#lg-ec-dict-submit');
      const res = dictEl.querySelector('#lg-ec-dict-result');
      const submitDictation = async () => {
        if (answered) return;
        const typed = (inp.value || '').trim();
        if (!typed) return;
        await onDictationAnswer(typed, res, inp, sub);
      };
      sub.addEventListener('click', submitDictation);
      inp.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); submitDictation(); } });
      inp.focus();
    } else {
      // Multiple-choice mode
      optsEl.hidden = false;
      dictEl.hidden = true;
      dictEl.innerHTML = '';
      optsEl.innerHTML = current.options.map((o, i) => `
        <button type="button" class="lg-ec-opt" data-i="${i}">${escapeHtml(o)}</button>
      `).join('');
      optsEl.querySelectorAll('.lg-ec-opt').forEach(btn => {
        btn.addEventListener('click', () => onAnswer(btn));
      });
      hintEl.textContent = qIdx === 0 ? 'Tap to hear it' : 'Listen';
      _answerStartMs = performance.now();
      if (noVoice) {
        await renderReadingMode(current);
      } else {
        await play(false);
      }
    }
  }

  // ── Audio play ─────────────────────────────────────────────────────────

  async function play(slow = false) {
    if (!current || !current.prompt) return;
    speakBtn.classList.add('lg-ec-pulse');
    await speakWord(current.prompt, voice, slow ? { rate: 0.75 } : undefined);
    setTimeout(() => speakBtn.classList.remove('lg-ec-pulse'), 600);
    hintEl.textContent = slow ? 'Once more…' : 'Pick its meaning';
  }

  if (speakBtn) speakBtn.addEventListener('click', () => {
    _replaysThisQ += 1;
    play(false);
  });

  // ── Keyboard shortcuts ─────────────────────────────────────────────────

  const onKey = (e) => {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
    if (!endPanel.hidden) return;
    const isDictQ = _isDictation(qIdx + 1);
    if (!isDictQ && e.key >= '1' && e.key <= '4' && !answered) {
      const i = Number(e.key) - 1;
      const btn = optsEl.querySelectorAll('.lg-ec-opt')[i];
      if (btn) { e.preventDefault(); btn.click(); }
    } else if ((e.key === 'r' || e.key === 'R' || e.key === ' ') && !noVoice) {
      e.preventDefault();
      _replaysThisQ += 1;
      play(false);
    } else if (e.key === 's' || e.key === 'S') {
      e.preventDefault();
      overlay.querySelector('#lg-ec-skip')?.click();
    }
  };
  document.addEventListener('keydown', onKey);
  addCleanup(() => document.removeEventListener('keydown', onKey));
  addCleanup(() => { if (_readerHandle) { _readerHandle.destroy(); _readerHandle = null; } });

  // ── Skip ───────────────────────────────────────────────────────────────

  overlay.querySelector('#lg-ec-skip').addEventListener('click', () => {
    if (answered) return;
    answered = true;
    streak = 0;
    // Reveal the correct answer
    Array.from(optsEl.querySelectorAll('.lg-ec-opt'))
      .find(b => b.textContent.trim() === String(current.answer).trim())
      ?.classList.add('lg-ec-right');
    showReveal(current, false);
    setTimeout(() => {
      qIdx += 1;
      if (qIdx >= ROUND_QUESTIONS) endRound();
      else render();
    }, 2000);
  });

  // ── Multiple-choice answer handler ─────────────────────────────────────

  async function onAnswer(btn) {
    if (answered) return;
    answered = true;
    const isRight = btn.textContent.trim() === String(current.answer).trim();
    const elapsedMs = performance.now() - _answerStartMs;
    const effortGrade = gradeForEffort({
      correct: isRight,
      attempts: 1,
      hintsUsed: 0,
      replays: _replaysThisQ,
      ms: elapsedMs,
    });

    if (isRight) {
      btn.classList.add('lg-ec-right');
      burstAt(btn, '#bb9bff');
      streak += 1;
      maxStreak = Math.max(maxStreak, streak);
      const bonus = streak >= 5 ? 10 : (streak >= 3 ? 5 : 0);
      score += 10 + bonus;
      if (current.card) correct.push({ card: current.card, grade: effortGrade });
    } else {
      streak = 0;
      btn.classList.add('lg-ec-wrong');
      Array.from(optsEl.querySelectorAll('.lg-ec-opt'))
        .find(b => b.textContent.trim() === String(current.answer).trim())
        ?.classList.add('lg-ec-right');
      if (current.card) missed.push({ card: current.card, grade: effortGrade });
      // Re-play slower while the correct answer is visible
      if (!noVoice) play(true);
    }
    scoreEl.textContent = fmtScore(score);

    // Show decode reveal: always on miss, also on correct for harder questions
    // (long sentences or high replay count) to reinforce the comprehension habit
    const showAlways = !isRight || _replaysThisQ > 0 || (current.prompt && current.prompt.length > 40);
    if (showAlways) {
      await showReveal(current, isRight);
    }

    setTimeout(() => {
      qIdx += 1;
      if (qIdx >= ROUND_QUESTIONS) endRound();
      else render();
    }, isRight ? 1800 : 2800);
  }

  // ── Dictation answer handler ────────────────────────────────────────────

  async function onDictationAnswer(typed, resEl, inp, subBtn) {
    if (answered) return;
    answered = true;
    inp.disabled = true;
    subBtn.disabled = true;
    const elapsedMs = performance.now() - _answerStartMs;
    const sim = similarity(typed, current.answer);
    const isRight = sim >= DICTATION_THRESHOLD;
    const effortGrade = gradeForEffort({
      correct: isRight,
      attempts: 1,
      hintsUsed: 0,
      replays: _replaysThisQ,
      ms: elapsedMs,
    });

    if (isRight) {
      resEl.className = 'lg-ec-dict-result lg-ec-right';
      resEl.textContent = `Correct! "${escapeHtml(current.answer)}"`;
      streak += 1;
      maxStreak = Math.max(maxStreak, streak);
      const bonus = streak >= 5 ? 10 : (streak >= 3 ? 5 : 0);
      score += 10 + bonus;
      if (current.card) correct.push({ card: current.card, grade: effortGrade });
    } else {
      resEl.className = 'lg-ec-dict-result lg-ec-wrong';
      resEl.innerHTML = `Answer: <strong>${escapeHtml(current.answer)}</strong> · you wrote: "${escapeHtml(typed)}"`;
      streak = 0;
      if (current.card) missed.push({ card: current.card, grade: effortGrade });
    }
    scoreEl.textContent = fmtScore(score);

    // Always show reveal on dictation so learners inspect the full sentence
    await showReveal(current, isRight);

    setTimeout(() => {
      qIdx += 1;
      if (qIdx >= ROUND_QUESTIONS) endRound();
      else render();
    }, isRight ? 1800 : 2800);
  }

  // ── End of round ────────────────────────────────────────────────────────

  async function endRound() {
    if (_readerHandle) { _readerHandle.destroy(); _readerHandle = null; }
    revealArea.innerHTML = '';

    // Dedupe by word_id; respect in_queue===false; cap at 5 writes each list
    const seenIds = new Set();
    const toGrade = [];
    for (const { card: c, grade } of correct) {
      if (toGrade.length >= 5) break;
      if (c.in_queue === false) continue;
      if (seenIds.has(c.word_id)) continue;
      seenIds.add(c.word_id);
      toGrade.push({ c, grade });
    }
    const toFail = [];
    for (const { card: c, grade } of missed) {
      if (toFail.length >= 5) break;
      if (c.in_queue === false) continue;
      if (seenIds.has(c.word_id)) continue;
      seenIds.add(c.word_id);
      toFail.push({ c, grade });
    }
    await Promise.all([
      ...toGrade.map(({ c, grade }) => gradeCard(lang, c.word_id, grade)),
      ...toFail.map(({ c, grade }) => gradeCard(lang, c.word_id, grade)),
    ]);

    const beatBest = score > best;
    overlay.querySelector('#lg-ec-end-stats').innerHTML = `
      <div class="lg-end-stat"><div class="lg-end-stat-n">${escapeHtml(fmtScore(score))}</div><div>score${beatBest ? ' · new best!' : ''}</div></div>
      <div class="lg-end-stat"><div class="lg-end-stat-n">${correct.length}/${ROUND_QUESTIONS}</div><div>right</div></div>
      <div class="lg-end-stat"><div class="lg-end-stat-n">${maxStreak}</div><div>best streak</div></div>`;
    endPanel.hidden = false;
    recordResult({
      game_id: 'echo_chamber', lang,
      score, words_played: ROUND_QUESTIONS, words_correct: correct.length,
      duration_sec: Math.round((performance.now() - roundStart) / 1000),
      metadata: { max_streak: maxStreak },
    });
  }

  // ── Replay ─────────────────────────────────────────────────────────────

  overlay.querySelector('#lg-ec-replay').addEventListener('click', async () => {
    score = 0; qIdx = 0; streak = 0; maxStreak = 0;
    correct.length = 0; missed.length = 0;
    scoreEl.textContent = '0000';
    endPanel.hidden = true;
    roundStart = performance.now();
    revealArea.innerHTML = '';
    readingArea.innerHTML = '';
    await prepareRound();
    render();
  });
  overlay.querySelector('#lg-ec-quit').addEventListener('click', () => close());

  // ── Prepare round ──────────────────────────────────────────────────────

  async function prepareRound() {
    hintEl.textContent = 'Preparing…';
    const seeds = buildSeeds();
    const smart = await buildSmartDistractors(seeds);    // already has 4s timeout via llmJudgeJSON
    questions = seeds.map((s, i) => {
      const distractors = (smart && smart[i]) || fallbackDistractors(s, seeds);
      return buildOptions(s, distractors);
    });
  }

  await prepareRound();
  render();
}
