/**
 * Story Weaver — comprehension under narrative.
 *
 * An LLM writes a short branching story in the target language constrained
 * to use the learner's vocab. At each chapter the user picks the L2 word
 * that fits the missing slot. The choice is carried forward into the next
 * chapter's prompt so the narrative actually branches on what happened.
 *
 * Upgrades (2026-06-17):
 *   #1  Gloss removed from buttons — L2-only choices; Hint toggle on demand.
 *   #2  Paragraph is tap-to-understand via makeClickableHTML + attachReaderPopover.
 *   #3  Speaker button narrates the full paragraph.
 *   #4  showWarming on generation; heuristic fallback if blank/missing-▢/non-L2.
 *   #5  POS-filtered distractors so answer isn't guessable by part-of-speech.
 *   #6  Real branching: chosen word + narrative path fed into next chapter's prompt.
 *   #7  gradeForEffort for FSRS (attempts + hintsUsed per chapter).
 */

import {
  escapeHtml,
  fetchGamePool,
  llmChatStream,
  pickN,
  makeGameOverlay,
  makeEmptyOverlay,
  gradeCard,
  gradeForEffort,
  speakWord,
  burstAt,
  recordResult,
  makeClickableHTML,
  attachReaderPopover,
  showWarming,
} from './_common.js';

// ── Style injection (guarded) ─────────────────────────────────────────

function _ensureStyles() {
  if (document.getElementById('lg-story-styles')) return;
  const style = document.createElement('style');
  style.id = 'lg-story-styles';
  style.textContent = `
.lg-sw { display: flex; flex-direction: column; height: 100%; overflow: hidden; }
.lg-sw-paper { flex: 1; display: flex; flex-direction: column; gap: 16px;
  overflow-y: auto; padding: 0 24px 24px; }
.lg-sw-text { font-size: 17px; line-height: 1.75; padding: 18px 20px;
  background: color-mix(in srgb, var(--bg-surface,#16181f) 85%, var(--accent,#6ea8fe) 3%);
  border: 1px solid var(--border,rgba(255,255,255,.09)); border-radius: 12px;
  color: var(--text-primary,#e8e8ea); white-space: pre-wrap; min-height: 80px; }
.lg-sw-text-bar { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.lg-sw-narrate { background: none; border: none; cursor: pointer; font-size: 19px;
  color: var(--text-muted,#9aa0aa); padding: 2px 4px; border-radius: 6px;
  transition: color .15s, background .15s; line-height: 1; }
.lg-sw-narrate:hover { color: var(--accent,#6ea8fe);
  background: color-mix(in srgb, var(--accent,#6ea8fe) 14%, transparent); }
.lg-sw-slot { display: inline-block; min-width: 60px; text-align: center;
  border-bottom: 2px solid var(--accent,#6ea8fe); color: var(--text-muted,#9aa0aa);
  letter-spacing: .06em; padding: 0 4px; cursor: default; }
.lg-sw-slot-filled { color: var(--success,#30a46c); border-color: var(--success,#30a46c);
  font-weight: 700; cursor: pointer; }
.lg-sw-choices { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.lg-sw-choice { padding: 12px 14px; border-radius: 10px;
  background: var(--bg-elevated,#1b1d24);
  border: 1.5px solid var(--border,rgba(255,255,255,.11));
  color: var(--text-primary,#e8e8ea); cursor: pointer;
  font-size: 16px; font-weight: 600; text-align: center;
  transition: border-color .15s, background .15s; }
.lg-sw-choice:hover:not(:disabled) { border-color: var(--accent,#6ea8fe);
  background: color-mix(in srgb, var(--accent,#6ea8fe) 12%, transparent); }
.lg-sw-choice:disabled { opacity: .55; cursor: default; }
.lg-sw-choice-right { border-color: var(--success,#30a46c) !important;
  background: color-mix(in srgb, var(--success,#30a46c) 18%, transparent) !important; }
.lg-sw-choice-wrong { border-color: var(--error,#e5534b) !important;
  background: color-mix(in srgb, var(--error,#e5534b) 14%, transparent) !important; }
.lg-sw-choice-gloss { font-size: 12px; font-weight: 400; margin-top: 4px;
  color: var(--text-muted,#9aa0aa); display: none; }
.lg-sw-choice-gloss.visible { display: block; }
.lg-sw-choice-surface { font-size: 16px; }
.lg-sw-hint-row { display: flex; align-items: center; gap: 10px; }
.lg-sw-hint-btn { background: none; border: 1px solid var(--border,rgba(255,255,255,.14));
  border-radius: 7px; color: var(--text-muted,#9aa0aa); font-size: 12px;
  padding: 4px 10px; cursor: pointer; transition: background .15s; }
.lg-sw-hint-btn:hover { background: color-mix(in srgb, var(--accent,#6ea8fe) 12%, transparent); }
.lg-sw-recast { padding: 8px 12px; border-radius: 8px; font-size: 13px;
  background: color-mix(in srgb, var(--warning,#e0a800) 14%, transparent);
  color: var(--text-secondary,#c2c5cc); margin-top: 8px; }
.lg-sw-title { font-size: 14px; color: var(--text-muted,#9aa0aa); }
.lg-sw-path-chip { display: inline-block; font-size: 11px; padding: 2px 8px;
  border-radius: 99px; background: color-mix(in srgb, var(--accent,#6ea8fe) 16%, transparent);
  color: var(--text-muted,#9aa0aa); margin-left: 6px; }
`;
  document.head.appendChild(style);
}

// ── Constants ─────────────────────────────────────────────────────────

const CHAPTERS = 5;
// Generation timeout before we show the warming spinner (ms).
const WARMING_DELAY_MS = 3000;
// Hard timeout: if no visible text after this long, fall back to static.
const GEN_TIMEOUT_MS = 45000;

// ── Helpers ───────────────────────────────────────────────────────────

/** Build a fallback paragraph from the target's example, inserting ▢. */
function _staticPara(target) {
  const ex = (target.example || '').trim();
  if (ex) {
    const withBlank = ex.includes(target.surface)
      ? ex.replace(target.surface, '▢')
      : `${ex} ▢`;
    return withBlank;
  }
  return `▢`;
}

/**
 * Heuristic: for a non-en lang, if the paragraph is >80 % printable ASCII
 * (excluding the ▢ placeholder) and the lang is not a Latin-script language,
 * the model likely responded in English.  Re-used after the first attempt.
 */
function _looksWrongScript(para, lang) {
  if (['en', 'es', 'fr', 'de', 'it', 'pt'].includes(lang)) return false;
  const cleaned = para.replace('▢', '').replace(/[\s\d.,!?;:()\-'"""'']/g, '');
  if (!cleaned.length) return false;
  const ascii = cleaned.split('').filter(c => c.charCodeAt(0) < 128).length;
  return ascii / cleaned.length > 0.8;
}

/** Try to extract the ▢ from the paragraph (insert if missing). */
function _ensureBlank(para, target) {
  if (para.includes('▢')) return para;
  const replaced = para.replace(target.surface, '▢');
  if (replaced !== para) return replaced;
  // Last resort — append.
  return `${para} ▢`;
}

/**
 * POS-filtered distractor pick: prefer distractors that share the same pos as
 * the target so the answer can't be guessed by part-of-speech mismatch alone.
 * Falls back gracefully when pos metadata is absent or pool is too small.
 */
function _pickDistractors(pool, target, n = 3) {
  const rest = pool.filter(c => c.word_id !== target.word_id);
  const samePOS = rest.filter(c => c.pos && target.pos && c.pos === target.pos);
  if (samePOS.length >= n) return pickN(samePOS, n);
  // Not enough same-POS — fill remainder from the general pool.
  const picked = pickN(samePOS, samePOS.length);
  const ids = new Set(picked.map(c => c.word_id));
  const rest2 = rest.filter(c => !ids.has(c.word_id));
  return [...picked, ...pickN(rest2, n - picked.length)];
}

// ── Main ──────────────────────────────────────────────────────────────

export async function launchStoryWeaver({ lang, voice }) {
  _ensureStyles();

  const pool = await fetchGamePool(lang, 30, 'explore', [], { allowDiscovery: true });
  let roundStart = performance.now();

  if (pool.length < 6) {
    return makeEmptyOverlay({
      palette: 'amber', emoji: '📜',
      message: 'Story Weaver needs at least 6 words to write around.',
      hint: 'Add a handful of words and the loom will start spinning chapters.',
    });
  }

  const { overlay, close, addCleanup } = makeGameOverlay({
    id: 'lg-story', palette: 'amber', title: 'Story Weaver',
  });

  overlay.innerHTML = `
    <div class="lg-game lg-sw">
      <header class="lg-hud">
        <button type="button" class="lg-close" aria-label="Close">×</button>
        <div class="lg-sw-title">A story in <strong>${escapeHtml(lang.toUpperCase())}</strong></div>
        <div class="lg-hud-stats"><span class="lg-hud-label">chapter</span> <span id="lg-sw-chapter">1</span>/${CHAPTERS}</div>
      </header>
      <div class="lg-sw-paper">
        <div class="lg-sw-text-bar">
          <button type="button" class="lg-sw-narrate" id="lg-sw-narrate" aria-label="Read paragraph aloud" title="Listen to paragraph">🔊</button>
        </div>
        <div class="lg-sw-text" id="lg-sw-text"><em>weaving…</em></div>
        <div class="lg-sw-hint-row">
          <button type="button" class="lg-sw-hint-btn" id="lg-sw-hint">Hint</button>
          <span style="font-size:12px;color:var(--text-muted,#9aa0aa)">Shows gloss — costs a hint point</span>
        </div>
        <div class="lg-sw-choices" id="lg-sw-choices"></div>
      </div>
      <div class="lg-end" id="lg-sw-end" hidden>
        <div class="lg-end-title">— the end —</div>
        <div class="lg-end-stats" id="lg-sw-end-stats"></div>
        <div class="lg-end-actions">
          <button type="button" class="btn btn-primary" id="lg-sw-replay">New story</button>
          <button type="button" class="btn btn-ghost" id="lg-sw-quit">Done</button>
        </div>
      </div>
    </div>`;

  overlay.querySelector('.lg-close').addEventListener('click', () => close());

  const textEl      = overlay.querySelector('#lg-sw-text');
  const choicesEl   = overlay.querySelector('#lg-sw-choices');
  const chapterEl   = overlay.querySelector('#lg-sw-chapter');
  const endPanel    = overlay.querySelector('#lg-sw-end');
  const narrateBtn  = overlay.querySelector('#lg-sw-narrate');
  const hintBtn     = overlay.querySelector('#lg-sw-hint');

  // Per-chapter state.
  let chapter = 1;
  let storySoFar = '';
  let narrativePath = [];    // sequence of chosen surfaces for the branching prompt
  let currentParaText = '';  // raw paragraph (blanked) for the narrate button
  let currentTarget = null;
  let hintsUsedThisChapter = 0;
  let attemptsThisChapter = 0;
  let chapStartMs = 0;

  // Round-level state.
  const gradeInputs = [];    // { target, correct, attempts, hintsUsed, ms }
  const missedThisChapter = new Set();
  let used = new Set();

  // Reader popover — attach to the paper area; reinstalled each chapter.
  let _readerHandle = null;
  function _attachReader() {
    _readerHandle?.destroy();
    const paper = overlay.querySelector('.lg-sw-paper');
    _readerHandle = attachReaderPopover(paper, { lang, voice });
  }
  addCleanup(() => _readerHandle?.destroy());
  _attachReader();

  // ── Hint toggle ────────────────────────────────────────────────────

  let _hintShown = false;
  function _resetHint() {
    _hintShown = false;
    hintBtn.textContent = 'Hint';
    hintBtn.style.opacity = '';
    choicesEl.querySelectorAll('.lg-sw-choice-gloss').forEach(el => el.classList.remove('visible'));
  }
  hintBtn.addEventListener('click', () => {
    if (_hintShown) return;
    _hintShown = true;
    hintsUsedThisChapter += 1;
    hintBtn.textContent = 'Hint shown';
    hintBtn.style.opacity = '0.5';
    choicesEl.querySelectorAll('.lg-sw-choice-gloss').forEach(el => el.classList.add('visible'));
  });

  // ── Narrate button ────────────────────────────────────────────────

  narrateBtn.addEventListener('click', () => {
    if (!currentParaText) return;
    const readable = currentParaText.replace('▢', `____`);
    speakWord(readable, voice);
  });

  // ── Keyboard shortcuts ────────────────────────────────────────────

  const onKey = (e) => {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
    if (e.key >= '1' && e.key <= '4') {
      const opts = choicesEl.querySelectorAll('.lg-sw-choice');
      const i = Number(e.key) - 1;
      const btn = opts[i];
      if (btn && !btn.disabled) { e.preventDefault(); btn.click(); }
    }
    if (e.key === 'h' || e.key === 'H') {
      hintBtn.click();
    }
  };
  document.addEventListener('keydown', onKey);
  addCleanup(() => document.removeEventListener('keydown', onKey));

  // ── Generation ────────────────────────────────────────────────────

  async function generateParagraph(target) {
    const pathNote = narrativePath.length
      ? `The story so far has followed this path: the protagonist chose "${narrativePath.join(' → ')}".`
      : '';
    const sys = `You are writing a ${chapter === 1 ? 'opening' : 'continuing'} paragraph of a short story in ${lang}, for a beginner-intermediate learner. Write exactly ONE short paragraph (2-4 sentences) in ${lang}. Use simple grammar. Include the target word "${target.surface}" exactly once, replaced with the placeholder ▢. ${pathNote} Do not translate. Do not explain. Return only the ${lang} paragraph.`;
    const u = `Story so far:\n${storySoFar || '(beginning)'}\n\nTarget word: ${target.surface}\n\nWrite the next paragraph now with ▢ in place of the target word.`;

    let stopWarming = () => {};
    let para = '';
    let resolved = false;

    const warmingTimer = setTimeout(() => {
      if (!resolved) stopWarming = showWarming(textEl, `Writing chapter ${chapter}…`);
    }, WARMING_DELAY_MS);

    const timeoutPromise = new Promise(resolve => setTimeout(() => resolve(''), GEN_TIMEOUT_MS));

    const streamPromise = llmChatStream(
      [{ role: 'system', content: sys }, { role: 'user', content: u }],
      (_d, full) => {
        if (overlay.isConnected && !resolved) {
          stopWarming();                     // clear warming if still showing
          clearTimeout(warmingTimer);
          textEl.textContent = full;         // live preview while streaming
        }
      },
    );

    para = await Promise.race([streamPromise, timeoutPromise]);
    resolved = true;
    clearTimeout(warmingTimer);
    stopWarming();

    if (!overlay.isConnected) return null;

    para = (para || '').trim();

    // Fallback: blank or wrong-script.
    if (!para || _looksWrongScript(para, lang)) {
      // One retry with a tighter prompt.
      if (para) {
        const retry = await Promise.race([
          llmChat_simple(sys, u),
          new Promise(res => setTimeout(() => res(''), 15000)),
        ]);
        if (retry && !_looksWrongScript(retry.trim(), lang)) {
          para = retry.trim();
        }
      }
      if (!para || _looksWrongScript(para, lang)) {
        para = _staticPara(target);
      }
    }

    para = _ensureBlank(para, target);
    return para;
  }

  /** Thin non-streaming helper for the single retry. */
  async function llmChat_simple(sys, u) {
    try {
      const r = await fetch('/v1/chat/completions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: (typeof localStorage !== 'undefined' && localStorage.getItem('augmentum-selected-model')) || 'default',
          messages: [{ role: 'system', content: sys }, { role: 'user', content: u }],
          stream: false, temperature: 0.5,
        }),
      });
      if (!r.ok) return '';
      const j = await r.json();
      return j?.choices?.[0]?.message?.content || '';
    } catch { return ''; }
  }

  // ── Chapter lifecycle ─────────────────────────────────────────────

  async function nextChapter() {
    chapterEl.textContent = String(chapter);
    hintsUsedThisChapter = 0;
    attemptsThisChapter = 0;
    chapStartMs = performance.now();
    _resetHint();
    missedThisChapter.clear();

    const available = pool.filter(c => !used.has(c.word_id));
    if (available.length === 0) { endStory(); return; }

    const target = available[Math.floor(Math.random() * available.length)];
    currentTarget = target;
    used.add(target.word_id);

    const distractors = _pickDistractors(pool, target, 3);

    // Show warming placeholder while generating.
    textEl.innerHTML = '<em>weaving…</em>';
    choicesEl.innerHTML = '';
    hintBtn.style.display = 'none';
    narrateBtn.disabled = true;

    const para = await generateParagraph(target);
    if (!overlay.isConnected) return;
    if (!para) { endStory(); return; }

    currentParaText = para;
    narrateBtn.disabled = false;
    hintBtn.style.display = '';

    await renderParagraph(para);
    renderChoices([target, ...distractors], target, para);
  }

  async function renderParagraph(para) {
    textEl.innerHTML = '';
    const parts = para.split('▢');
    const before = parts[0] || '';
    const after = parts.slice(1).join('▢') || '';

    // Make before/after text clickable via makeClickableHTML + store context.
    const beforeHtml = before ? await makeClickableHTML(before, lang) : '';
    const afterHtml  = after  ? await makeClickableHTML(after,  lang) : '';

    const slot = document.createElement('span');
    slot.className = 'lg-sw-slot';
    slot.textContent = '____';

    // Wrap para text nodes in a container that carries data-full-text.
    const container = document.createElement('span');
    container.dataset.fullText = para.replace('▢', '____');

    const beforeNode = document.createElement('span');
    beforeNode.innerHTML = beforeHtml;
    beforeNode.dataset.fullText = para.replace('▢', '____');

    const afterNode = document.createElement('span');
    afterNode.innerHTML = afterHtml;
    afterNode.dataset.fullText = para.replace('▢', '____');

    container.appendChild(beforeNode);
    container.appendChild(slot);
    container.appendChild(afterNode);
    textEl.appendChild(container);

    // Re-attach reader popover so the newly-rendered .lg-tok spans are covered.
    _attachReader();
  }

  function renderChoices(choices, target, para) {
    const shuffled = pickN(choices, choices.length);
    choicesEl.innerHTML = shuffled.map(c => `
      <button type="button" class="lg-sw-choice" data-id="${escapeHtml(c.word_id)}">
        <div class="lg-sw-choice-surface">${escapeHtml(c.surface)}${c.reading && c.reading !== c.surface ? `<span style="font-size:12px;font-weight:400;margin-left:6px;color:var(--text-muted,#9aa0aa)">${escapeHtml(c.reading)}</span>` : ''}</div>
        <div class="lg-sw-choice-gloss">${escapeHtml((c.glosses || [])[0] || '')}</div>
      </button>`).join('');
    choicesEl.querySelectorAll('.lg-sw-choice').forEach(btn => {
      btn.addEventListener('click', () => onChoice(btn, target, para));
    });
  }

  async function onChoice(btn, target, para) {
    choicesEl.querySelectorAll('.lg-sw-choice').forEach(b => b.disabled = true);
    attemptsThisChapter += 1;
    const chosen = btn.dataset.id;

    if (chosen === target.word_id) {
      btn.classList.add('lg-sw-choice-right');
      burstAt(btn, '#f0c66f');

      const slot = overlay.querySelector('.lg-sw-slot');
      if (slot) {
        slot.textContent = target.surface;
        slot.classList.add('lg-sw-slot-filled');
        slot.addEventListener('click', () => speakWord(target.reading || target.surface, voice));
      }

      const chapterMs = performance.now() - chapStartMs;
      const grade = gradeForEffort({
        correct: true,
        attempts: attemptsThisChapter,
        hintsUsed: hintsUsedThisChapter,
        ms: attemptsThisChapter === 1 ? chapterMs : 0,
      });
      gradeInputs.push({ target, grade });

      // Carry this chapter's chosen word into the narrative path.
      narrativePath.push(target.surface);
      storySoFar += '\n' + para.replace('▢', target.surface);

      setTimeout(() => {
        chapter += 1;
        if (chapter > CHAPTERS) endStory();
        else nextChapter();
      }, 1600);
    } else {
      btn.classList.add('lg-sw-choice-wrong');
      if (!missedThisChapter.has(target.word_id)) {
        missedThisChapter.add(target.word_id);
      }
      const tip = document.createElement('div');
      tip.className = 'lg-sw-recast';
      tip.textContent = `Not quite — try again. (Hint key: H)`;
      choicesEl.appendChild(tip);
      setTimeout(() => {
        choicesEl.querySelectorAll('.lg-sw-choice').forEach(b => b.disabled = false);
        btn.classList.remove('lg-sw-choice-wrong');
        tip.remove();
      }, 1400);
    }
  }

  // ── End-of-story ──────────────────────────────────────────────────

  async function endStory() {
    const seen = new Set();
    const toGrade = gradeInputs
      .filter(g => { if (seen.has(g.target.word_id)) return false; seen.add(g.target.word_id); return true; })
      .filter(g => g.target.in_queue !== false)
      .slice(0, 5);

    await Promise.all(toGrade.map(g => gradeCard(lang, g.target.word_id, g.grade)));

    const chaptersDone = gradeInputs.length;
    overlay.querySelector('#lg-sw-end-stats').innerHTML = `
      <div class="lg-end-stat">
        <div class="lg-end-stat-n">${chaptersDone}/${CHAPTERS}</div>
        <div>chapters woven</div>
      </div>`;
    endPanel.hidden = false;

    recordResult({
      game_id: 'story_weaver', lang,
      score: chaptersDone * 20,
      words_played: CHAPTERS,
      words_correct: chaptersDone,
      duration_sec: Math.round((performance.now() - roundStart) / 1000),
    });
  }

  // ── Replay ───────────────────────────────────────────────────────

  overlay.querySelector('#lg-sw-replay').addEventListener('click', () => {
    chapter = 1;
    storySoFar = '';
    narrativePath = [];
    gradeInputs.length = 0;
    missedThisChapter.clear();
    used = new Set();
    currentParaText = '';
    currentTarget = null;
    endPanel.hidden = true;
    roundStart = performance.now();
    nextChapter();
  });
  overlay.querySelector('#lg-sw-quit').addEventListener('click', () => close());

  nextChapter();
}
