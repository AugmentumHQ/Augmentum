/**
 * ✨ Constellation — sentence-shape assembly.
 *
 * Words from the user's pool appear as stars across a night sky. The
 * player drags lines between stars to form a sentence; the LLM grades
 * whether the connected sequence is grammatically valid in the target
 * language. Correct constellations unlock a one-line "myth" — a fun
 * etymology or trivia tidbit for one of the words.
 *
 * Why this exists: SRS gives you words in isolation. Sentence-building
 * is the bridge from passive recognition to active use, and it's where
 * Duolingo's "tap the tile" UI is genuinely good — but only when the
 * sentence is a fixed template. Free-form constellations let the user
 * *try* sentences and get graded, which is closer to real writing.
 */

import {
  escapeHtml, fetchGamePool, fetchSentences, llmJudgeJSON,
  pickN, makeGameOverlay, makeEmptyOverlay, speakWord, burstAt,
  recordResult, gradeCard, gradeForEffort, showWarming, showNotice,
  makeClickableHTML, attachReaderPopover,
} from './_common.js';

const STAR_COUNT = 8;
const ROUND_GOAL = 3;

// Grammar particles per language — always present as faint inner stars,
// so a beginner can actually build sentences with their content-word
// pool. Without these, JP/ES/etc. learners' nouns-and-verbs constellations
// would always fail LLM validation for missing particles.
const _PARTICLES = {
  ja: [
    { surface: 'は', reading: 'は', glosses: ['(topic)'] },
    { surface: 'が', reading: 'が', glosses: ['(subject)'] },
    { surface: 'を', reading: 'を', glosses: ['(object)'] },
    { surface: 'に', reading: 'に', glosses: ['(to/at)'] },
    { surface: 'の', reading: 'の', glosses: ['(of)'] },
    { surface: 'で', reading: 'で', glosses: ['(by/at)'] },
    { surface: 'と', reading: 'と', glosses: ['(with/and)'] },
  ],
  es: [
    { surface: 'el', glosses: ['the (m)'] },
    { surface: 'la', glosses: ['the (f)'] },
    { surface: 'un', glosses: ['a (m)'] },
    { surface: 'una', glosses: ['a (f)'] },
    { surface: 'de', glosses: ['of/from'] },
    { surface: 'a', glosses: ['to/at'] },
    { surface: 'en', glosses: ['in/on'] },
    { surface: 'y', glosses: ['and'] },
  ],
  fr: [
    { surface: 'le', glosses: ['the (m)'] },
    { surface: 'la', glosses: ['the (f)'] },
    { surface: 'un', glosses: ['a (m)'] },
    { surface: 'une', glosses: ['a (f)'] },
    { surface: 'de', glosses: ['of'] },
    { surface: 'à', glosses: ['to/at'] },
    { surface: 'et', glosses: ['and'] },
  ],
  zh: [
    { surface: '是', glosses: ['(to be)'] },
    { surface: '的', glosses: ['(possessive/adj)'] },
    { surface: '了', glosses: ['(completed)'] },
    { surface: '在', glosses: ['(at/in)'] },
    { surface: '和', glosses: ['and'] },
  ],
  // Korean particles (은/는/이/가/을/를) are bound suffixes — they
  // never stand alone, they attach to the preceding noun. Showing them
  // as standalone stars would teach the wrong mental model. Use common
  // standalone connectors and demonstratives instead, which Korean
  // does use as their own eojeol.
  ko: [
    { surface: '이것', glosses: ['this'] },
    { surface: '그것', glosses: ['that'] },
    { surface: '저것', glosses: ['that (over there)'] },
    { surface: '그리고', glosses: ['and'] },
    { surface: '하지만', glosses: ['but'] },
    { surface: '그래서', glosses: ['so'] },
    { surface: '또', glosses: ['also'] },
    { surface: '있다', glosses: ['(to be / to exist)'] },
  ],
};

function _particlesFor(lang) {
  const base = _PARTICLES[lang] || [];
  return base.map((p, i) => ({
    word_id: `__particle__${lang}__${i}`,
    surface: p.surface,
    reading: p.reading || '',
    pos: 'particle',
    glosses: p.glosses,
    is_particle: true,
    mastery_state: 'particle',
  }));
}

// Guarded style injection — once per page load.
function _ensureStyles() {
  if (document.getElementById('lg-constellation-styles')) return;
  const style = document.createElement('style');
  style.id = 'lg-constellation-styles';
  style.textContent = `
/* ── Constellation game styles ─────────────────────────────────── */
.lg-cons { display: flex; flex-direction: column; height: 100%; }
.lg-cons-sky { flex: 1 1 auto; cursor: default; }
.lg-cons-builder {
  padding: 10px 14px 8px;
  background: rgba(0,0,0,.45);
  display: flex; flex-direction: column; gap: 8px;
}
.lg-cons-title { font-size: 14px; color: rgba(255,255,255,.55); text-align: center; }
.lg-cons-current {
  min-height: 28px; text-align: center;
  font-size: 16px; font-weight: 600; color: #fff8d8;
  letter-spacing: .04em; transition: color .2s;
}
.lg-cons-current.lg-cons-checking { color: rgba(255,248,216,.45); }
.lg-cons-current.lg-cons-bad { color: #ff7070; animation: lg-cons-shake .3s; }
@keyframes lg-cons-shake {
  0%,100% { transform: none; }
  25% { transform: translateX(-6px); }
  75% { transform: translateX(6px); }
}
.lg-cons-actions { display: flex; gap: 10px; justify-content: center; }
.lg-cons-star { cursor: pointer; transition: opacity .18s; }
.lg-cons-star-particle { opacity: .72; }
.lg-cons-star .lg-cons-star-glow { transition: r .18s, opacity .18s; }
.lg-cons-star:hover .lg-cons-star-glow { opacity: .9; }
.lg-cons-star-selected .lg-cons-star-core { fill: #ffd866 !important; }
.lg-cons-star-selected .lg-cons-star-glow { opacity: .85; }
.lg-cons-star-text { fill: rgba(255,255,255,.88); font-size: 13px; pointer-events: none; }
.lg-cons-star-particle .lg-cons-star-text { font-size: 11px; }
.lg-cons-star-locked { opacity: 1; }
.lg-cons-star-locked .lg-cons-star-core { fill: #ffd866 !important; }
.lg-cons-star-locked .lg-cons-star-glow { opacity: .5; }
.lg-cons-line-locked { stroke: rgba(255,220,100,.9) !important; stroke-width: 2.5 !important; }

/* Myth panel */
.lg-cons-myth {
  padding: 10px 14px; background: rgba(20,14,45,.85);
  border-top: 1px solid rgba(255,220,100,.25);
  animation: lg-cons-myth-in .25s ease;
}
@keyframes lg-cons-myth-in { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; } }
.lg-cons-myth-head { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: #ffd866; margin-bottom: 4px; }
.lg-cons-myth-text { font-size: 14px; color: rgba(255,255,255,.88); line-height: 1.5; }
.lg-cons-myth-close {
  float: right; background: none; border: none;
  color: rgba(255,255,255,.4); cursor: pointer;
  font-size: 18px; line-height: 1; padding: 0 2px; margin-left: 8px;
}

/* Grammar-error feedback strip */
.lg-cons-feedback {
  padding: 8px 14px 6px;
  background: rgba(60,10,10,.7);
  border-top: 1px solid rgba(255,100,100,.2);
  font-size: 13px; color: #ff9a9a; line-height: 1.45;
  animation: lg-cons-myth-in .2s ease;
}
.lg-cons-feedback-label { font-size: 10.5px; text-transform: uppercase; letter-spacing: .07em; color: rgba(255,100,100,.6); margin-bottom: 3px; }
.lg-cons-feedback-dismiss {
  float: right; background: none; border: none;
  color: rgba(255,100,100,.5); cursor: pointer;
  font-size: 17px; line-height: 1; padding: 0 2px; margin-left: 6px;
}

/* Hint scaffold */
.lg-cons-scaffold {
  padding: 6px 14px; text-align: center;
  font-size: 12.5px; color: rgba(180,180,220,.55);
  border-top: 1px solid rgba(255,255,255,.06);
  font-style: italic;
}
.lg-cons-scaffold strong { color: rgba(180,180,220,.78); font-style: normal; }

/* Breakdown panel on success */
.lg-cons-breakdown {
  padding: 10px 14px; background: rgba(10,20,50,.8);
  border-top: 1px solid rgba(100,160,255,.18);
  animation: lg-cons-myth-in .25s ease;
}
.lg-cons-breakdown-label {
  font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  color: rgba(120,180,255,.65); margin-bottom: 6px; cursor: pointer;
  user-select: none; display: flex; align-items: center; gap: 6px;
}
.lg-cons-breakdown-label::after { content: '▸'; transition: transform .15s; }
.lg-cons-breakdown-label.open::after { transform: rotate(90deg); }
.lg-cons-breakdown-body { display: none; }
.lg-cons-breakdown-body.open { display: block; }
.lg-cons-breakdown-sentence {
  font-size: 15px; font-weight: 600; color: #fff8d8; margin-bottom: 6px; line-height: 1.6;
}

/* Checking overlay shimmer on the builder */
.lg-cons-builder-checking { position: relative; }
.lg-cons-builder-checking::after {
  content: 'Checking…';
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  background: rgba(0,0,0,.6); color: rgba(255,248,216,.7);
  font-size: 14px; border-radius: 4px; pointer-events: none;
}
`;
  document.head.appendChild(style);
}

export async function launchConstellation({ lang, voice }) {
  _ensureStyles();

  const pool = await fetchGamePool(lang, 40);
  if (pool.length < 6) {
    return makeEmptyOverlay({
      palette: 'night', emoji: '✨',
      message: 'Constellation needs at least 6 word-stars to draw with.',
      hint: 'Add a few more words to your queue and the sky will fill up.',
    });
  }

  // Prefetch example sentences for the scaffold hint (non-blocking).
  const _sentencesPromise = fetchSentences(lang, 5);

  const { overlay, close, addCleanup } = makeGameOverlay({
    id: 'lg-cons', palette: 'night', title: 'Constellation',
  });

  overlay.innerHTML = `
    <div class="lg-game lg-cons">
      <header class="lg-hud">
        <button type="button" class="lg-close" aria-label="Close">×</button>
        <div class="lg-cons-title">Connect the words into a sentence</div>
        <div class="lg-hud-stats">
          <span class="lg-hud-label">found</span> <span id="lg-cons-found">0</span>/${ROUND_GOAL}
        </div>
      </header>
      <svg class="lg-cons-sky" id="lg-cons-sky" viewBox="0 0 1000 600" preserveAspectRatio="xMidYMid slice">
        <defs>
          <radialGradient id="lg-cons-glow">
            <stop offset="0%" stop-color="rgba(255,230,120,1)"/>
            <stop offset="100%" stop-color="rgba(255,230,120,0)"/>
          </radialGradient>
        </defs>
        <g id="lg-cons-bg"></g>
        <g id="lg-cons-lines"></g>
        <g id="lg-cons-stars"></g>
      </svg>
      <div class="lg-cons-scaffold" id="lg-cons-scaffold" hidden></div>
      <div class="lg-cons-builder" id="lg-cons-builder">
        <div class="lg-cons-current" id="lg-cons-current">tap stars to chain them</div>
        <div class="lg-cons-actions">
          <button type="button" class="btn btn-ghost" id="lg-cons-clear">Clear</button>
          <button type="button" class="btn btn-primary" id="lg-cons-check" disabled>Make it a constellation</button>
        </div>
      </div>
      <div class="lg-cons-feedback" id="lg-cons-feedback" hidden></div>
      <div class="lg-cons-myth" id="lg-cons-myth" hidden></div>
      <div class="lg-cons-breakdown" id="lg-cons-breakdown" hidden></div>
      <div class="lg-end" id="lg-cons-end" hidden>
        <div class="lg-end-title">Sky charted</div>
        <div class="lg-end-stats" id="lg-cons-end-stats"></div>
        <div class="lg-end-actions">
          <button type="button" class="btn btn-primary" id="lg-cons-replay">New sky</button>
          <button type="button" class="btn btn-ghost" id="lg-cons-quit">Done</button>
        </div>
      </div>
    </div>`;

  overlay.querySelector('.lg-close').addEventListener('click', () => close());

  const sky = overlay.querySelector('#lg-cons-sky');
  const bgG = overlay.querySelector('#lg-cons-bg');
  const linesG = overlay.querySelector('#lg-cons-lines');
  const starsG = overlay.querySelector('#lg-cons-stars');
  const currentEl = overlay.querySelector('#lg-cons-current');
  const checkBtn = overlay.querySelector('#lg-cons-check');
  const foundEl = overlay.querySelector('#lg-cons-found');
  const mythEl = overlay.querySelector('#lg-cons-myth');
  const feedbackEl = overlay.querySelector('#lg-cons-feedback');
  const breakdownEl = overlay.querySelector('#lg-cons-breakdown');
  const scaffoldEl = overlay.querySelector('#lg-cons-scaffold');
  const builderEl = overlay.querySelector('#lg-cons-builder');
  const endPanel = overlay.querySelector('#lg-cons-end');

  // Per-card attempt counter for gradeForEffort.
  // Keyed by word_id; tracks {attempts, correct}.
  const _attempts = new Map();

  // Track whether the judge is unavailable so we can degrade gracefully.
  let _judgeUnavailable = false;

  let stars = [];       // {id, card, x, y, isParticle}
  let chain = [];       // indices into stars
  let found = 0;
  // Only content-word (non-particle) stars get locked after a valid
  // constellation. Particle stars remain reusable across chains within
  // the same round — they're function words that a learner may need
  // again in the next sentence (は、が、を etc.).
  const locked = new Set();   // star indices that belong to a finalised constellation
  const gradedCards = [];     // {card, attempts, correct} for end-of-round FSRS
  let mythTimeoutId = null;
  // Reader popover for the breakdown panel — destroy on each new sentence.
  let _breakdownReader = null;
  let roundStart = performance.now();

  addCleanup(() => {
    if (mythTimeoutId) { clearTimeout(mythTimeoutId); mythTimeoutId = null; }
    if (_breakdownReader) { _breakdownReader.destroy(); _breakdownReader = null; }
  });

  // ── Scaffold hint ────────────────────────────────────────────────────

  async function _maybeShowScaffold() {
    try {
      const sentences = await _sentencesPromise;
      if (!sentences || !sentences.length) { scaffoldEl.hidden = true; return; }
      // Pick the shortest example so it fits in one line.
      const ex = sentences.slice().sort((a, b) =>
        (a.text || a.sentence || '').length - (b.text || b.sentence || '').length,
      )[0];
      const text = ex.text || ex.sentence || '';
      if (!text) { scaffoldEl.hidden = true; return; }
      scaffoldEl.innerHTML = `Try something like: <strong>${escapeHtml(text)}</strong>`;
      scaffoldEl.hidden = false;
    } catch {
      scaffoldEl.hidden = true;
    }
  }

  // ── Sky construction ─────────────────────────────────────────────────

  function seedSky() {
    bgG.innerHTML = '';
    for (let i = 0; i < 80; i += 1) {
      const x = Math.random() * 1000;
      const y = Math.random() * 600;
      const r = 0.4 + Math.random() * 1.2;
      bgG.insertAdjacentHTML('beforeend',
        `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(2)}" fill="white" opacity="${(0.15 + Math.random() * 0.5).toFixed(2)}"/>`);
    }
  }

  function spawnStars() {
    starsG.innerHTML = '';
    linesG.innerHTML = '';
    stars = [];
    chain = [];

    // Content stars on the outer ring.
    const cards = pickN(pool, STAR_COUNT);
    cards.forEach((card, i) => {
      const ang = (i / STAR_COUNT) * Math.PI * 2 + Math.random() * 0.3;
      const r = 200 + Math.random() * 60;
      const x = 500 + Math.cos(ang) * r;
      const y = 300 + Math.sin(ang) * r * 0.65;
      stars.push({ id: stars.length, card, x, y, isParticle: false });
    });

    // Particle helper stars clustered in the centre — always available,
    // visually subordinate.
    const particles = _particlesFor(lang);
    particles.forEach((p, i) => {
      const ang = (i / Math.max(1, particles.length)) * Math.PI * 2 + 0.4;
      const r = 60 + Math.random() * 30;
      const x = 500 + Math.cos(ang) * r;
      const y = 300 + Math.sin(ang) * r * 0.7;
      stars.push({ id: stars.length, card: p, x, y, isParticle: true });
    });

    stars.forEach((s, i) => {
      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.classList.add('lg-cons-star');
      if (s.isParticle) g.classList.add('lg-cons-star-particle');
      g.dataset.idx = String(i);
      const r = s.isParticle ? 16 : 32;
      const coreR = s.isParticle ? 3 : 6;
      const fontShift = s.isParticle ? 18 : 26;
      g.innerHTML = `
        <circle class="lg-cons-star-glow" cx="${s.x}" cy="${s.y}" r="${r}" fill="url(#lg-cons-glow)"/>
        <circle class="lg-cons-star-core" cx="${s.x}" cy="${s.y}" r="${coreR}" fill="${s.isParticle ? '#a8c8ff' : '#fff8d8'}"/>
        <text x="${s.x}" y="${s.y + fontShift}" text-anchor="middle" class="lg-cons-star-text">${escapeHtml(s.card.surface)}</text>`;
      g.addEventListener('click', () => onStarClick(i));
      starsG.appendChild(g);
    });
    renderCurrent();
    hideFeedback();
    _maybeShowScaffold();
  }

  // ── Chain rendering ───────────────────────────────────────────────────

  function renderCurrent() {
    if (chain.length === 0) {
      currentEl.textContent = 'tap stars to chain them';
      checkBtn.disabled = true;
    } else {
      currentEl.textContent = chain.map(i => stars[i].card.surface).join(' · ');
      checkBtn.disabled = chain.length < 2;
    }
    drawLines();
    starsG.querySelectorAll('.lg-cons-star').forEach((g, idx) => {
      g.classList.toggle('lg-cons-star-selected', chain.includes(idx));
    });
  }

  function drawLines() {
    linesG.innerHTML = '';
    for (let k = 1; k < chain.length; k += 1) {
      const a = stars[chain[k - 1]];
      const b = stars[chain[k]];
      linesG.insertAdjacentHTML('beforeend',
        `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="rgba(255,221,140,.85)" stroke-width="2"/>`);
    }
  }

  function onStarClick(i) {
    if (locked.has(i)) return;   // star already belongs to a finalised constellation
    const pos = chain.indexOf(i);
    if (pos !== -1) {
      // Tap a chained star → truncate the chain at that point (drops it
      // plus everything after). Lets the player rearrange word order
      // without nuking the whole chain — clicking the tail pops one
      // (the original behavior), clicking earlier rewinds further.
      chain = chain.slice(0, pos);
    } else {
      chain.push(i);
    }
    // Dismiss error feedback when the player starts editing the chain —
    // they're already responding to it.
    hideFeedback();
    renderCurrent();
  }

  // ── Feedback strip (grammar errors) ─────────────────────────────────

  function showFeedback(msg) {
    feedbackEl.innerHTML = `
      <button type="button" class="lg-cons-feedback-dismiss" aria-label="Dismiss">×</button>
      <div class="lg-cons-feedback-label">Grammar note</div>
      ${escapeHtml(msg)}`;
    feedbackEl.hidden = false;
    feedbackEl.querySelector('.lg-cons-feedback-dismiss')
      .addEventListener('click', hideFeedback);
  }

  function hideFeedback() {
    feedbackEl.hidden = true;
    feedbackEl.innerHTML = '';
  }

  // ── Clear button ─────────────────────────────────────────────────────

  overlay.querySelector('#lg-cons-clear').addEventListener('click', () => {
    chain = [];
    hideFeedback();
    renderCurrent();
  });

  // ── Check / judge ────────────────────────────────────────────────────

  checkBtn.addEventListener('click', async () => {
    checkBtn.disabled = true;
    builderEl.classList.add('lg-cons-builder-checking');
    currentEl.classList.add('lg-cons-checking');
    hideFeedback();

    const surfaces = chain.map(i => stars[i].card.surface);
    // Record attempt for every content word in the current chain.
    chain.forEach(i => {
      const c = stars[i].card;
      if (!c.is_particle && c.word_id) {
        const prev = _attempts.get(c.word_id) || { attempts: 0 };
        _attempts.set(c.word_id, { ...prev, attempts: prev.attempts + 1 });
      }
    });

    const result = await checkConstellation(lang, surfaces);
    if (!overlay.isConnected) return;

    builderEl.classList.remove('lg-cons-builder-checking');
    currentEl.classList.remove('lg-cons-checking');

    if (result === null) {
      // Judge timed out or returned null → offline degradation.
      _judgeUnavailable = true;
      showNotice(overlay, 'Model unavailable — checking against example sentences instead.', { kind: 'warn' });
      // Degrade: compare chain to the scaffold sentence if present.
      const scaffoldText = scaffoldEl.hidden ? '' : (scaffoldEl.textContent || '');
      const chainStr = surfaces.join(' ');
      // If the chain exactly matches the hint example, accept it. Otherwise
      // inform the learner we can't verify right now and keep the chain intact.
      const hintMatch = scaffoldText && scaffoldText.includes(chainStr);
      if (!hintMatch) {
        showFeedback("Can't reach the model right now. Try matching the hint sentence, or come back when the model is available.");
        checkBtn.disabled = false;
        return;
      }
      // Treat scaffold-match as valid with minimal metadata.
      await _onValid({ valid: true, translation: '(offline match)', myth: null, feedback: null }, surfaces);
      return;
    }

    if (result.valid) {
      await _onValid(result, surfaces);
    } else {
      _onInvalid(result, surfaces);
    }
  });

  async function _onValid(result, surfaces) {
    // Lock only the content-word stars. Particle stars stay available
    // so the learner can reuse は / の / を in the next constellation.
    chain.forEach(i => {
      if (!stars[i].isParticle) locked.add(i);
      const g = starsG.querySelectorAll('.lg-cons-star')[i];
      g.classList.add('lg-cons-star-locked');
    });
    Array.from(linesG.children).forEach(line => line.classList.add('lg-cons-line-locked'));
    burstAt(checkBtn, '#ffd866');
    found += 1;
    foundEl.textContent = String(found);

    // Record success for each content word in the chain.
    chain.forEach(i => {
      const c = stars[i].card;
      if (!c.is_particle && c.word_id) {
        const prev = _attempts.get(c.word_id) || { attempts: 1 };
        _attempts.set(c.word_id, { ...prev, correct: true });
        gradedCards.push(c);
      }
    });

    // Show myth panel.
    const mythText = result.myth || `"${surfaces.join(' ')}" — ${result.translation || ''}`;
    const chainCards = chain.map(i => stars[i].card);
    showMyth(mythText, chainCards);

    // Show grammar breakdown panel for the valid sentence.
    const sentence = surfaces.join(' ');
    await _showBreakdown(sentence);

    chain = [];
    renderCurrent();
    hideFeedback();

    if (found >= ROUND_GOAL) {
      setTimeout(endRound, 1800);
    }
  }

  function _onInvalid(result, surfaces) {
    // Keep the chain intact so the learner can fix the word order.
    // Instead of just flashing red, show an actionable grammar note.
    currentEl.classList.add('lg-cons-bad');
    setTimeout(() => currentEl.classList.remove('lg-cons-bad'), 300);

    const feedbackMsg = result.feedback
      || 'That sequence isn\'t grammatical in this language. Rearrange the words and try again.';
    showFeedback(feedbackMsg);
    checkBtn.disabled = false;
  }

  // ── Breakdown panel ──────────────────────────────────────────────────

  async function _showBreakdown(sentence) {
    // Clean up the previous reader if any.
    if (_breakdownReader) { _breakdownReader.destroy(); _breakdownReader = null; }
    breakdownEl.hidden = false;
    breakdownEl.innerHTML = `
      <div class="lg-cons-breakdown-label" id="lg-cons-bd-toggle">Why this works →</div>
      <div class="lg-cons-breakdown-body" id="lg-cons-bd-body">
        <div class="lg-cons-breakdown-sentence" id="lg-cons-bd-sentence" data-full-text="${escapeHtml(sentence)}"></div>
      </div>`;

    const toggleLabel = breakdownEl.querySelector('#lg-cons-bd-toggle');
    const bdBody = breakdownEl.querySelector('#lg-cons-bd-body');
    const sentenceEl = breakdownEl.querySelector('#lg-cons-bd-sentence');

    // Expand/collapse toggle.
    toggleLabel.addEventListener('click', () => {
      const open = bdBody.classList.toggle('open');
      toggleLabel.classList.toggle('open', open);
    });

    // Auto-expand after short delay so the myth panel can be read first.
    setTimeout(() => {
      bdBody.classList.add('open');
      toggleLabel.classList.add('open');
    }, 900);

    // Render clickable sentence tokens (makeClickableHTML handles CJK segmentation).
    sentenceEl.innerHTML = await makeClickableHTML(sentence, lang);
    sentenceEl.dataset.fullText = sentence;

    // Attach the reader popover so tapping any word shows its contextual role.
    _breakdownReader = attachReaderPopover(breakdownEl, { lang, voice });
  }

  // ── Myth panel ───────────────────────────────────────────────────────

  function showMyth(text, cards) {
    mythEl.innerHTML = `
      <div class="lg-cons-myth-head">★ Myth unlocked</div>
      <div class="lg-cons-myth-text">${escapeHtml(text)}</div>
      <button type="button" class="lg-cons-myth-close">×</button>`;
    mythEl.hidden = false;
    mythEl.querySelector('.lg-cons-myth-close').addEventListener('click', () => {
      mythEl.hidden = true;
      if (mythTimeoutId) { clearTimeout(mythTimeoutId); mythTimeoutId = null; }
    });
    speakWord(cards.map(c => c.reading || c.surface).join(' '), voice);
    if (mythTimeoutId) clearTimeout(mythTimeoutId);
    mythTimeoutId = setTimeout(() => {
      mythEl.hidden = true;
      mythTimeoutId = null;
    }, 8000);
  }

  // ── End round ────────────────────────────────────────────────────────

  async function endRound() {
    // Dedupe and cap to keep the SRS schedule honest.
    const seen = new Set();
    const toGrade = [];
    for (const c of gradedCards) {
      if (toGrade.length >= 5) break;
      if (seen.has(c.word_id)) continue;
      seen.add(c.word_id);
      toGrade.push(c);
    }

    // Grade each content word proportional to the effort it took.
    await Promise.all(toGrade.map(c => {
      const att = _attempts.get(c.word_id) || { attempts: 1, correct: true };
      const grade = gradeForEffort({
        correct: att.correct !== false,
        attempts: att.attempts || 1,
        // No hint/replay concept in constellation, but multiple attempts
        // on the same card within a round already pull the grade down.
        ms: 0,
      });
      return gradeCard(lang, c.word_id, grade);
    }));

    // Words that were attempted but never landed in a valid constellation
    // get grade 1 (Again).
    const failedIds = [];
    for (const [wid, att] of _attempts.entries()) {
      if (!att.correct && !seen.has(wid) && toGrade.length < 5) {
        failedIds.push(wid);
      }
    }
    await Promise.all(failedIds.map(wid => gradeCard(lang, wid, 1)));

    overlay.querySelector('#lg-cons-end-stats').innerHTML = `
      <div class="lg-end-stat"><div class="lg-end-stat-n">${found}</div><div>constellations</div></div>
      <div class="lg-end-stat"><div class="lg-end-stat-n">${toGrade.length}</div><div>graded</div></div>`;
    endPanel.hidden = false;
    recordResult({
      game_id: 'constellation', lang,
      score: found, words_played: STAR_COUNT, words_correct: found,
      duration_sec: Math.round((performance.now() - roundStart) / 1000),
    });
  }

  // ── Replay / quit ────────────────────────────────────────────────────

  overlay.querySelector('#lg-cons-replay').addEventListener('click', () => {
    found = 0; foundEl.textContent = '0';
    locked.clear();
    _attempts.clear();
    gradedCards.length = 0;
    if (mythTimeoutId) { clearTimeout(mythTimeoutId); mythTimeoutId = null; }
    mythEl.hidden = true;
    feedbackEl.hidden = true;
    breakdownEl.hidden = true;
    if (_breakdownReader) { _breakdownReader.destroy(); _breakdownReader = null; }
    endPanel.hidden = true;
    _judgeUnavailable = false;
    roundStart = performance.now();
    seedSky();
    spawnStars();
  });
  overlay.querySelector('#lg-cons-quit').addEventListener('click', () => close());

  seedSky();
  spawnStars();
}

// ── LLM grammar judge ────────────────────────────────────────────────
//
// Returns {valid, translation, myth, feedback} or null (timeout/offline).
// `feedback` is a one-sentence, learner-facing grammar explanation used
// on invalid sentences — actionable, not just "wrong order".

async function checkConstellation(lang, words) {
  const sys = `You are a strict grammar judge for ${lang}. Given an ordered sequence of words, decide if they form a grammatically plausible short sentence in ${lang}. Reply in strict JSON only (no prose outside the JSON object):
{
  "valid": bool,
  "translation": "English translation if valid, else empty string",
  "myth": "one-sentence etymology or trivia about one word if valid, else null",
  "feedback": "if invalid: a single actionable sentence telling the learner exactly what is grammatically wrong and how to fix it (e.g. 'In Japanese the verb goes last — move 食べる to the end'); if valid: null"
}
A sequence of nouns alone is NOT valid. Particles/connectors must be respected. Be honest — invalid is fine. Keep feedback concise and constructive.`;
  const u = `Words in order: ${words.join(' / ')}`;
  return llmJudgeJSON(
    [{ role: 'system', content: sys }, { role: 'user', content: u }],
    { fallback: null, timeoutMs: 8000 },
  );
}
