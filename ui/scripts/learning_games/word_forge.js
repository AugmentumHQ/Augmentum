/**
 * Word Forge — morphology / compounding.
 *
 * Two morpheme/word chips drop onto an anvil. The learner predicts the
 * compound's meaning BEFORE the reveal (recall > passive exposure), then
 * swings the hammer. The LLM validates via llmJudgeJSON (timeout-safe) and
 * simultaneously explains HOW the parts combine. Forged words are tappable
 * via makeClickableHTML + attachReaderPopover. FSRS grade is earned on the
 * prediction step via gradeForEffort.
 *
 * Upgrade summary (2026-06-17):
 *  #1  Morphology lesson — parallel llmJudgeJSON call explains the join rule
 *  #2  Offline/cold safety — llmJudgeJSON timeout, showWarming, showNotice
 *  #3  Hedged copy — "Looks like a real word" when pack lacks the surface
 *  #4  Prediction gate — learner guesses meaning before reveal; gradeForEffort
 *  #5  Morph-role chip labels — pos→role badge; single-char CJK chips enriched
 *      via breakdownContextual; forged word tappable via reader popover
 *  #6  FSRS — gradeCard on prediction step only (not on mere combination)
 */

import {
  escapeHtml, fetchGamePool, llmJudgeJSON, llmChat,
  pickN, makeGameOverlay, makeEmptyOverlay,
  speakWord, addWord, burstAt, shuffle, recordResult,
  gradeForEffort, gradeCard, similarity,
  makeClickableHTML, attachReaderPopover,
  showWarming, showNotice, breakdownContextual,
} from './_common.js';

// ── Shared styles (guarded) ──────────────────────────────────────────────────
function _ensureForgeStyles() {
  if (document.getElementById('lg-forge-styles')) return;
  const style = document.createElement('style');
  style.id = 'lg-forge-styles';
  style.textContent = `
/* ── Word Forge layout ──────────────────────────────────── */
.lg-wf { display: flex; flex-direction: column; height: 100%; }
.lg-wf-title { font-size: 14px; color: var(--text-muted, #9aa0aa); text-align: center; flex: 1; }
.lg-hud-stats { display: flex; gap: 4px; align-items: center; font-size: 14px; white-space: nowrap; }
.lg-hud-label { color: var(--text-muted,#9aa0aa); font-size: 12px; }

/* Stage */
.lg-wf-stage {
  display: flex; flex-direction: column; align-items: center;
  gap: 16px; padding: 24px 20px 12px;
}

/* Anvil row */
.lg-wf-anvil {
  display: flex; align-items: center; gap: 12px;
  background: color-mix(in srgb, var(--bg-elevated,#1b1d24) 80%, transparent);
  border: 1px solid var(--border,rgba(255,255,255,.1));
  border-radius: 16px; padding: 16px 20px;
  min-width: 320px;
}
.lg-wf-plus { font-size: 22px; color: var(--text-muted,#9aa0aa); flex-shrink: 0; }
.lg-wf-slot {
  flex: 1; min-height: 72px; border-radius: 12px;
  border: 2px dashed var(--border,rgba(255,255,255,.15));
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 4px; padding: 8px; transition: border-color .15s, background .15s;
  position: relative;
}
.lg-wf-slot-hover { border-color: var(--accent,#6ea8fe); background: color-mix(in srgb, var(--accent,#6ea8fe) 10%, transparent); }
.lg-wf-slot-full { border-style: solid; border-color: var(--accent,#6ea8fe); }
.lg-wf-empty { color: var(--text-muted,#9aa0aa); font-size: 12px; }
.lg-wf-slot-surface { font-size: 22px; font-weight: 700; line-height: 1.2; }
.lg-wf-slot-gloss { font-size: 11px; color: var(--text-muted,#9aa0aa); }
.lg-wf-slot-role {
  font-size: 10px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
  padding: 1px 6px; border-radius: 99px;
  background: color-mix(in srgb, var(--accent,#6ea8fe) 20%, transparent);
  color: var(--accent,#6ea8fe); margin-top: 2px;
}
.lg-wf-slot-clear {
  position: absolute; top: 4px; right: 4px;
  background: none; border: none; cursor: pointer; font-size: 16px;
  color: var(--text-muted,#9aa0aa); line-height: 1; padding: 2px 4px;
}
.lg-wf-slot-clear:hover { color: var(--text-primary,#e8e8ea); }

/* Hammer */
.lg-wf-hammer {
  display: flex; align-items: center; gap: 8px;
  padding: 10px 28px; border-radius: 12px; font-size: 16px; font-weight: 600;
  cursor: pointer; border: none;
  background: var(--accent,#6ea8fe); color: #fff;
  transition: opacity .15s, transform .1s;
}
.lg-wf-hammer:disabled { opacity: .4; cursor: not-allowed; }
.lg-wf-hammer:not(:disabled):hover { opacity: .88; }
.lg-wf-hammer-icon { font-size: 20px; }
@keyframes lg-wf-swing {
  0%   { transform: rotate(0deg); }
  25%  { transform: rotate(-28deg) scale(1.12); }
  60%  { transform: rotate(14deg); }
  100% { transform: rotate(0deg); }
}
.lg-wf-hammer-swing { animation: lg-wf-swing .42s ease; }

/* Result panel */
.lg-wf-result {
  width: 100%; max-width: 520px; border-radius: 14px; padding: 18px 20px;
  background: var(--bg-elevated,#1b1d24);
  border: 1px solid var(--border,rgba(255,255,255,.1));
}
.lg-wf-result-good { border-color: var(--success,#30a46c); }
.lg-wf-result-bad  { border-color: var(--border,rgba(255,255,255,.1)); }

.lg-wf-result-head { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; flex-wrap: wrap; }
.lg-wf-result-new { font-size: 28px; font-weight: 800; }
.lg-wf-result-reading { font-size: 16px; color: var(--text-muted,#9aa0aa); }
.lg-wf-result-speak {
  background: none; border: none; cursor: pointer; font-size: 18px;
  color: var(--text-muted,#9aa0aa); padding: 2px 4px;
}
.lg-wf-result-gloss { font-size: 15px; color: var(--text-secondary,#c2c5cc); margin-bottom: 8px; }

/* Morphology lesson box */
.lg-wf-lesson {
  background: color-mix(in srgb, var(--accent,#6ea8fe) 10%, transparent);
  border-left: 3px solid var(--accent,#6ea8fe);
  border-radius: 0 8px 8px 0; padding: 10px 14px;
  font-size: 13.5px; line-height: 1.55; margin: 10px 0;
  color: var(--text-primary,#e8e8ea);
}
.lg-wf-lesson-label {
  font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--accent,#6ea8fe); font-weight: 700; margin-bottom: 4px;
}
.lg-wf-lesson-loading { color: var(--text-muted,#9aa0aa); font-style: italic; font-size: 13px; }

/* Prediction box */
.lg-wf-predict {
  width: 100%; max-width: 520px; border-radius: 14px; padding: 18px 20px;
  background: var(--bg-elevated,#1b1d24);
  border: 1px solid color-mix(in srgb, var(--accent,#6ea8fe) 35%, transparent);
  display: flex; flex-direction: column; gap: 12px;
}
.lg-wf-predict-label { font-size: 14px; color: var(--text-secondary,#c2c5cc); }
.lg-wf-predict-preview { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.lg-wf-predict-combo { font-size: 20px; font-weight: 700; }
.lg-wf-predict-plus { color: var(--text-muted,#9aa0aa); }
.lg-wf-predict-input {
  width: 100%; padding: 10px 14px; border-radius: 8px; font-size: 15px;
  background: var(--bg,#13141a); border: 1px solid var(--border,rgba(255,255,255,.12));
  color: var(--text-primary,#e8e8ea); outline: none;
}
.lg-wf-predict-input:focus { border-color: var(--accent,#6ea8fe); }
.lg-wf-predict-options { display: flex; flex-wrap: wrap; gap: 8px; }
.lg-wf-predict-opt {
  padding: 7px 14px; border-radius: 8px; font-size: 14px; cursor: pointer;
  border: 1px solid var(--border,rgba(255,255,255,.15));
  background: var(--bg,#13141a); color: var(--text-primary,#e8e8ea);
  transition: border-color .12s, background .12s;
}
.lg-wf-predict-opt:hover { border-color: var(--accent,#6ea8fe); background: color-mix(in srgb, var(--accent,#6ea8fe) 12%, transparent); }
.lg-wf-predict-opt-correct { border-color: var(--success,#30a46c) !important; background: color-mix(in srgb, var(--success,#30a46c) 18%, transparent) !important; }
.lg-wf-predict-opt-wrong   { border-color: rgba(255,80,80,.4) !important; background: color-mix(in srgb, rgba(255,80,80,1) 10%, transparent) !important; }
.lg-wf-predict-opt-missed  { border-color: var(--success,#30a46c) !important; }
.lg-wf-predict-actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.lg-wf-predict-hint { font-size: 12px; color: var(--text-muted,#9aa0aa); margin-top: 2px; }

.lg-wf-result-actions { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
.lg-wf-result-bad-msg { color: var(--text-muted,#9aa0aa); font-size: 13.5px; margin-top: 4px; }
.lg-wf-offline-msg { display: flex; flex-direction: column; align-items: center; gap: 10px; padding: 20px; color: var(--text-muted,#9aa0aa); text-align: center; }

/* Chip tray */
.lg-wf-tray { flex: 1; overflow-y: auto; padding: 12px 16px 20px; }
.lg-wf-tray-label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--text-muted,#9aa0aa); margin-bottom: 10px; }
.lg-wf-tray-words { display: flex; flex-wrap: wrap; gap: 8px; }

/* Morpheme chips */
.lg-wf-chip {
  display: flex; flex-direction: column; align-items: center; gap: 2px;
  padding: 8px 14px; border-radius: 10px; cursor: grab;
  background: var(--bg-elevated,#1b1d24);
  border: 1px solid var(--border,rgba(255,255,255,.12));
  color: var(--text-primary,#e8e8ea);
  transition: border-color .12s, opacity .12s, transform .08s;
  user-select: none;
}
.lg-wf-chip:hover { border-color: var(--accent,#6ea8fe); transform: translateY(-2px); }
.lg-wf-chip-dragging { opacity: .5; }
.lg-wf-chip-surface { font-size: 18px; font-weight: 700; line-height: 1.2; }
.lg-wf-chip-gloss { font-size: 11px; color: var(--text-muted,#9aa0aa); max-width: 90px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lg-wf-chip-role {
  font-size: 9.5px; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
  padding: 1px 5px; border-radius: 99px;
  background: color-mix(in srgb, var(--text-muted,#9aa0aa) 18%, transparent);
  color: var(--text-muted,#9aa0aa);
}

/* Forged word tappable token override */
.lg-wf-result .lg-tok { font-size: inherit; font-weight: inherit; }
`;
  document.head.appendChild(style);
}

// ── POS → role label (language-agnostic coarse map) ─────────────────────────
const _POS_ROLE = {
  noun: 'root', n: 'root', 名詞: 'root',
  verb: 'verb', v: 'verb', 動詞: 'verb',
  adj: 'adj', adjective: 'adj', 形容詞: 'adj',
  adv: 'adv', adverb: 'adv', 副詞: 'adv',
  prefix: 'prefix', 接頭辞: 'prefix',
  suffix: 'suffix', 接尾辞: 'suffix',
  counter: 'counter', 助数詞: 'counter',
  particle: 'particle', 助詞: 'particle',
  aux: 'aux', auxiliary: 'aux',
  conj: 'conj', conjunction: 'conj',
  pron: 'pron', pronoun: 'pron',
  num: 'num', numeral: 'num',
  prop: 'proper', proper: 'proper',
};

function _posToRole(pos) {
  if (!pos) return null;
  const key = String(pos).toLowerCase().split(/[\s,/_]+/)[0];
  return _POS_ROLE[key] || null;
}

// Enrich a single-character CJK morpheme with its grammatical role via
// breakdownContextual — fires once, cached in the card object. Best-effort.
async function _enrichChipRole(card, lang) {
  if (card._roleLabel !== undefined) return card._roleLabel;
  const surface = card.surface || '';
  // Already have pos — derive from it first (no LLM needed).
  const posRole = _posToRole(card.pos);
  if (posRole) { card._roleLabel = posRole; return posRole; }
  // Only hit LLM for single-char CJK (cheap, high payoff per character).
  if ((lang === 'ja' || lang === 'zh') && [...surface].length === 1) {
    const bd = await breakdownContextual(surface, lang).catch(() => null);
    if (bd && bd.length) {
      const role = bd[0].role || '';
      // Extract a coarse role from the freeform string
      const coarseKey = Object.keys(_POS_ROLE).find(k => role.toLowerCase().includes(k));
      const coarse = coarseKey ? _POS_ROLE[coarseKey] : (role.split(/[\s(,]/)[0].slice(0, 8) || null);
      card._roleLabel = coarse || null;
      return card._roleLabel;
    }
  }
  card._roleLabel = null;
  return null;
}

// ── Combination check (with timeout) ────────────────────────────────────────
async function checkCombination(lang, a, b) {
  const sys = `You are a linguistic oracle. Decide if combining two morphemes in ${lang} produces a real, attested word (compound, derivation, or fused form). Respond with strict JSON only, no prose. Schema: {"valid": bool, "surface": "...", "reading": "...", "gloss": "english meaning in one short phrase"}. If invalid, return {"valid": false}. Be conservative — only accept established words. For Japanese accept both kanji + kana forms (e.g. 朝 + ごはん → 朝ごはん).`;
  const u = `Combine: ${escapeHtml(a.surface)} (${escapeHtml((a.glosses || [])[0] || '')}) + ${escapeHtml(b.surface)} (${escapeHtml((b.glosses || [])[0] || '')}). Is this a real word in ${lang}?`;
  const result = await llmJudgeJSON(
    [{ role: 'system', content: sys }, { role: 'user', content: u }],
    { fallback: null, timeoutMs: 9000 },
  );
  if (!result) return null;
  if (!result.valid) return { valid: false };
  return result;
}

// ── Morphology explanation (parallel call) ───────────────────────────────────
async function fetchMorphologyLesson(lang, a, b, combo) {
  const sys = `You are a morphology tutor for ${lang} learners. Given two morphemes and the compound they form, write ONE or TWO sentences (max 60 words) that explain HOW the parts combine and NAME the joining rule (e.g. "noun-modifier prefix", "left-branching N+N compound", "verb-stem nominalization", "-ing suffix", "stress-shift"). Do NOT repeat the definition — focus on the STRUCTURE. Be concrete and learner-friendly.`;
  const u = `${escapeHtml(a.surface)} (${escapeHtml((a.glosses || [])[0] || '')}) + ${escapeHtml(b.surface)} (${escapeHtml((b.glosses || [])[0] || '')}) → ${escapeHtml(combo.surface)} (${escapeHtml(combo.gloss || '')})`;
  const result = await llmJudgeJSON(
    [{ role: 'system', content: sys }, { role: 'user', content: u }],
    { fallback: null, timeoutMs: 7000 },
  );
  // Accept either {"lesson": "..."} or a plain string (edge-case model variation).
  if (result && typeof result.lesson === 'string') return result.lesson.trim();
  // Fallback: try raw llmChat for plain-text answer.
  const raw = await Promise.race([
    llmChat(
      [{ role: 'system', content: sys + ' Reply with plain text, no JSON.' }, { role: 'user', content: u }],
    ),
    new Promise(res => setTimeout(() => res(''), 7000)),
  ]);
  return raw ? raw.trim().slice(0, 300) : null;
}

// Build 3 multiple-choice distractors from the pool glosses (excluding
// combos of the current two cards). Falls back to null-terminated list.
function _buildDistractors(pool, a, b, correctGloss, n = 3) {
  const glosses = pool
    .filter(c => c.word_id !== a.word_id && c.word_id !== b.word_id)
    .map(c => (c.glosses || [])[0] || '')
    .filter(g => g && g !== correctGloss && similarity(g, correctGloss) < 0.6);
  const picks = pickN(glosses, n * 3).slice(0, n);
  while (picks.length < n) picks.push('(no meaning found)');
  return picks;
}

export async function launchWordForge({ lang, voice }) {
  _ensureForgeStyles();

  // Word Forge needs solid morphemes the user already owns.
  const pool = await fetchGamePool(lang, 40, 'consolidate');
  const sessionStart = performance.now();

  if (pool.length < 8) {
    return makeEmptyOverlay({
      palette: 'iron',
      message: 'The forge needs at least 8 known morphemes to combine.',
      hint: 'Word Forge runs on consolidated vocab — keep adding words and check back.',
    });
  }

  const { overlay, close: rawClose, addCleanup } = makeGameOverlay({
    id: 'lg-forge', palette: 'iron', title: 'Word Forge',
  });

  let recorded = false;
  let forged = 0;
  let totalPredictions = 0;
  let correctPredictions = 0;

  function _record() {
    if (recorded) return;
    recorded = true;
    recordResult({
      game_id: 'word_forge', lang,
      score: forged * 25 + correctPredictions * 10,
      words_played: forged + totalPredictions,
      words_correct: forged,
      duration_sec: Math.round((performance.now() - sessionStart) / 1000),
      metadata: { predictions: totalPredictions, correct_predictions: correctPredictions },
    });
  }

  const close = () => { _record(); rawClose(); };
  addCleanup(() => _record());

  overlay.innerHTML = `
    <div class="lg-game lg-wf">
      <header class="lg-hud">
        <button type="button" class="lg-close" aria-label="Close">×</button>
        <div class="lg-wf-title">Forge new words from old ones</div>
        <div class="lg-hud-stats">
          <span class="lg-hud-label">forged</span>
          <span id="lg-wf-count">0</span>
        </div>
      </header>
      <div class="lg-wf-stage">
        <div class="lg-wf-anvil">
          <div class="lg-wf-slot" id="lg-wf-slot-a"><span class="lg-wf-empty">drop here</span></div>
          <div class="lg-wf-plus">+</div>
          <div class="lg-wf-slot" id="lg-wf-slot-b"><span class="lg-wf-empty">drop here</span></div>
        </div>
        <button type="button" class="lg-wf-hammer" id="lg-wf-hammer" disabled>
          <span class="lg-wf-hammer-icon">&#x1F528;</span>
          <span id="lg-wf-hammer-label">Forge</span>
        </button>
        <div id="lg-wf-predict" hidden></div>
        <div class="lg-wf-result" id="lg-wf-result" hidden></div>
      </div>
      <div class="lg-wf-tray">
        <div class="lg-wf-tray-label">your morphemes — drag or tap to place on the anvil</div>
        <div class="lg-wf-tray-words" id="lg-wf-tray"></div>
      </div>
    </div>`;

  overlay.querySelector('.lg-close').addEventListener('click', () => close());

  const slotA = overlay.querySelector('#lg-wf-slot-a');
  const slotB = overlay.querySelector('#lg-wf-slot-b');
  const hammer = overlay.querySelector('#lg-wf-hammer');
  const hammerLabel = overlay.querySelector('#lg-wf-hammer-label');
  const trayEl = overlay.querySelector('#lg-wf-tray');
  const resultEl = overlay.querySelector('#lg-wf-result');
  const predictEl = overlay.querySelector('#lg-wf-predict');
  const countEl = overlay.querySelector('#lg-wf-count');

  let placed = { a: null, b: null };
  let trayItems = [];
  let trayRenderId = 0;
  // Reader popover handle for the forged result word
  let _readerHandle = null;

  // Enrich chips with roles asynchronously (no await — updates UI progressively).
  async function _enrichChips(items, renderId) {
    let changed = false;
    for (const card of items) {
      if (card._roleLabel !== undefined) continue;
      await _enrichChipRole(card, lang);
      changed = true;
    }
    // Re-render tray with updated roles if still visible.
    if (changed && overlay.isConnected && renderId === trayRenderId) {
      renderTray({ reshuffle: false });
    }
  }

  function renderTray({ reshuffle = true } = {}) {
    if (reshuffle || !trayItems.length) {
      trayItems = shuffle(pool).slice(0, 12);
      trayRenderId += 1;
    }
    const items = trayItems;
    trayEl.innerHTML = '';
    items.forEach(card => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'lg-wf-chip';
      chip.draggable = true;
      chip.dataset.wordId = card.word_id;
      const role = card._roleLabel ? `<span class="lg-wf-chip-role">${escapeHtml(card._roleLabel)}</span>` : '';
      chip.innerHTML = `
        <span class="lg-wf-chip-surface">${escapeHtml(card.surface)}</span>
        ${role}
        <span class="lg-wf-chip-gloss">${escapeHtml((card.glosses || [])[0] || '')}</span>`;
      chip.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('text/plain', card.word_id);
        chip.classList.add('lg-wf-chip-dragging');
      });
      chip.addEventListener('dragend', () => chip.classList.remove('lg-wf-chip-dragging'));
      chip.addEventListener('click', () => placeNext(card));
      trayEl.appendChild(chip);
    });
    // Fire enrichment asynchronously — the next renderTray will have roles.
    _enrichChips(items, trayRenderId);
  }

  function placeNext(card) {
    if (!placed.a) place('a', card);
    else if (!placed.b) place('b', card);
  }

  function _slotHTML(card) {
    const role = card._roleLabel ? `<span class="lg-wf-slot-role">${escapeHtml(card._roleLabel)}</span>` : '';
    return `
      <button type="button" class="lg-wf-slot-clear" aria-label="Clear">×</button>
      <span class="lg-wf-slot-surface">${escapeHtml(card.surface)}</span>
      ${role}
      <span class="lg-wf-slot-gloss">${escapeHtml((card.glosses || [])[0] || '')}</span>`;
  }

  function place(side, card) {
    placed[side] = card;
    const slot = side === 'a' ? slotA : slotB;
    slot.classList.add('lg-wf-slot-full');
    slot.innerHTML = _slotHTML(card);
    slot.querySelector('.lg-wf-slot-clear').addEventListener('click', () => clear(side));
    hammer.disabled = !(placed.a && placed.b);
  }

  function clear(side) {
    placed[side] = null;
    const slot = side === 'a' ? slotA : slotB;
    slot.classList.remove('lg-wf-slot-full');
    slot.innerHTML = `<span class="lg-wf-empty">drop here</span>`;
    hammer.disabled = !(placed.a && placed.b);
  }

  [slotA, slotB].forEach((slot, i) => {
    slot.addEventListener('dragover', (e) => { e.preventDefault(); slot.classList.add('lg-wf-slot-hover'); });
    slot.addEventListener('dragleave', () => slot.classList.remove('lg-wf-slot-hover'));
    slot.addEventListener('drop', (e) => {
      e.preventDefault();
      slot.classList.remove('lg-wf-slot-hover');
      const id = e.dataTransfer.getData('text/plain');
      const card = pool.find(c => c.word_id === id);
      if (card) place(i === 0 ? 'a' : 'b', card);
    });
  });

  async function lookupInPack(surface) {
    try {
      const r = await fetch(`/api/learning/lookup?lang=${encodeURIComponent(lang)}&q=${encodeURIComponent(surface)}`);
      if (!r.ok) return null;
      const j = await r.json();
      return (j.entries || [])[0] || null;
    } catch { return null; }
  }

  // ── Prediction step ──────────────────────────────────────────────────────
  // Returns { correct: bool, attempts: number, ms: number } after the user
  // has committed to a prediction. Runs BEFORE the result panel is shown,
  // while combo + lesson LLM calls are in flight.
  function showPredictionUI(a, b, correctGloss) {
    return new Promise(resolve => {
      predictEl.hidden = false;
      predictEl.className = 'lg-wf-predict';
      const distractors = _buildDistractors(pool, a, b, correctGloss);
      const options = shuffle([correctGloss, ...distractors]);

      predictEl.innerHTML = `
        <div class="lg-wf-predict-label">What do you think <strong>${escapeHtml(a.surface + b.surface)}</strong> means?</div>
        <div class="lg-wf-predict-options" id="lg-wf-pred-opts"></div>
        <div class="lg-wf-predict-hint" id="lg-wf-pred-hint"></div>`;

      const optsEl = predictEl.querySelector('#lg-wf-pred-opts');
      const hintEl = predictEl.querySelector('#lg-wf-pred-hint');
      const t0 = performance.now();
      let attempts = 0;
      let settled = false;

      options.forEach(opt => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'lg-wf-predict-opt';
        btn.textContent = opt;
        btn.addEventListener('click', () => {
          if (settled) return;
          attempts += 1;
          const isCorrect = opt === correctGloss;

          // Highlight all options
          optsEl.querySelectorAll('.lg-wf-predict-opt').forEach(b2 => {
            if (b2.textContent === correctGloss) b2.classList.add('lg-wf-predict-opt-correct');
            else if (b2 === btn && !isCorrect) b2.classList.add('lg-wf-predict-opt-wrong');
            b2.disabled = true;
          });

          hintEl.textContent = isCorrect ? 'Correct!' : `The answer is: ${correctGloss}`;
          settled = true;
          const ms = performance.now() - t0;

          // Short delay so the learner sees the feedback.
          setTimeout(() => {
            resolve({ correct: isCorrect, attempts, ms });
          }, 700);
        });
        optsEl.appendChild(btn);
      });
    });
  }

  // ── Forge result panel ───────────────────────────────────────────────────
  async function showResult(combo, packEntry, lessonText, predResult) {
    if (_readerHandle) { _readerHandle.destroy(); _readerHandle = null; }
    predictEl.hidden = true;
    resultEl.classList.remove('lg-wf-result-bad');
    resultEl.classList.add('lg-wf-result-good');

    // Render the forged word as a tappable token via makeClickableHTML.
    const forgedHTML = await makeClickableHTML(combo.surface, lang).catch(() => escapeHtml(combo.surface));
    const lessonBox = lessonText
      ? `<div class="lg-wf-lesson"><div class="lg-wf-lesson-label">How it works</div>${escapeHtml(lessonText)}</div>`
      : `<div class="lg-wf-lesson"><div class="lg-wf-lesson-label">How it works</div><span class="lg-wf-lesson-loading">Explanation not available offline.</span></div>`;

    if (packEntry && packEntry.word_id) {
      resultEl.innerHTML = `
        <div class="lg-wf-result-head">
          <span class="lg-wf-result-new" data-full-text="${escapeHtml(combo.surface)}">${forgedHTML}</span>
          <button type="button" class="lg-wf-result-speak" aria-label="Speak">&#x1F50A;</button>
          <span class="lg-wf-result-reading">${escapeHtml(combo.reading || '')}</span>
        </div>
        <div class="lg-wf-result-gloss">${escapeHtml(combo.gloss || '')}</div>
        ${lessonBox}
        <div class="lg-wf-result-actions">
          <button type="button" class="btn btn-primary lg-wf-add">+ Add to my words</button>
          <button type="button" class="btn btn-ghost lg-wf-skip">Skip</button>
        </div>`;
    } else {
      // Real word per LLM but pack doesn't index this surface — hedged copy (#3).
      resultEl.innerHTML = `
        <div class="lg-wf-result-head">
          <span class="lg-wf-result-new" data-full-text="${escapeHtml(combo.surface)}">${forgedHTML}</span>
          <button type="button" class="lg-wf-result-speak" aria-label="Speak">&#x1F50A;</button>
          <span class="lg-wf-result-reading">${escapeHtml(combo.reading || '')}</span>
        </div>
        <div class="lg-wf-result-gloss">${escapeHtml(combo.gloss || '')}</div>
        ${lessonBox}
        <div class="lg-wf-result-bad-msg">Looks like a real word, but it's not in your practice pack yet — can't add.</div>
        <button type="button" class="btn btn-ghost lg-wf-skip">Try again</button>`;
    }

    resultEl.hidden = false;

    // Wire tappable forged-word token
    const newWordEl = resultEl.querySelector('.lg-wf-result-new');
    if (newWordEl) {
      _readerHandle = attachReaderPopover(newWordEl, { lang, voice });
    }

    resultEl.querySelector('.lg-wf-result-speak').addEventListener('click', () => {
      speakWord(combo.reading || combo.surface, voice);
    });

    const addBtn = resultEl.querySelector('.lg-wf-add');
    if (addBtn) {
      addBtn.addEventListener('click', async () => {
        addBtn.disabled = true;
        addBtn.textContent = 'Adding…';

        // FSRS grade on prediction step (#6).
        if (predResult) {
          const grade = gradeForEffort({
            correct: predResult.correct,
            attempts: predResult.attempts,
            hintsUsed: 0,
            replays: 0,
            ms: predResult.ms,
          });
          await gradeCard(lang, packEntry.word_id, grade).catch(() => null);
        }

        await addWord(lang, packEntry.word_id);
        forged += 1;
        countEl.textContent = String(forged);

        if (!pool.some(c => c.word_id === packEntry.word_id)) {
          pool.push({
            word_id: packEntry.word_id,
            surface: packEntry.surface,
            reading: packEntry.reading,
            pos: packEntry.pos,
            glosses: packEntry.glosses,
            mastery_state: 'new',
            in_queue: true,
          });
        }
        burstAt(hammer, '#ffa84d');
        if (_readerHandle) { _readerHandle.destroy(); _readerHandle = null; }
        renderTray();
        resultEl.hidden = true;
        clear('a'); clear('b');
      });
    }

    resultEl.querySelector('.lg-wf-skip').addEventListener('click', () => {
      if (_readerHandle) { _readerHandle.destroy(); _readerHandle = null; }
      resultEl.hidden = true;
      clear('a'); clear('b');
    });
  }

  // ── Hammer click ─────────────────────────────────────────────────────────
  hammer.addEventListener('click', async () => {
    if (!placed.a || !placed.b) return;
    hammer.disabled = true;
    hammer.classList.add('lg-wf-hammer-swing');
    setTimeout(() => hammer.classList.remove('lg-wf-hammer-swing'), 500);

    predictEl.hidden = true;
    resultEl.hidden = true;

    // Show warming state while calling LLM (#2).
    hammerLabel.textContent = 'Consulting…';
    let stopWarming = null;
    const warmingTimeout = setTimeout(() => {
      stopWarming = showWarming(resultEl, 'Consulting the forge…');
      resultEl.hidden = false;
    }, 800);

    const a = placed.a;
    const b = placed.b;

    // Fire validation. Lesson call will follow only if valid.
    const combo = await checkCombination(lang, a, b);

    clearTimeout(warmingTimeout);
    if (stopWarming) { stopWarming(); }
    resultEl.hidden = true;
    hammerLabel.textContent = 'Forge';

    if (!overlay.isConnected) return;

    if (combo === null) {
      // Null result = model offline / timeout (#2).
      showNotice(overlay.querySelector('.lg-game'), 'The forge is offline — no model available. Check your provider settings.', { kind: 'warn' });
      hammer.disabled = false;
      return;
    }

    if (!combo.valid) {
      resultEl.classList.remove('lg-wf-result-good');
      resultEl.classList.add('lg-wf-result-bad');
      resultEl.innerHTML = `
        <div class="lg-wf-result-bad-msg">No spark. These don't combine in ${escapeHtml(lang.toUpperCase())}.</div>
        <button type="button" class="btn btn-ghost lg-wf-skip">Try again</button>`;
      resultEl.hidden = false;
      resultEl.querySelector('.lg-wf-skip').addEventListener('click', () => {
        resultEl.hidden = true;
        hammer.disabled = false;
        clear('a'); clear('b');
      });
      return;
    }

    // Valid — fire pack lookup + lesson explanation IN PARALLEL with prediction UI.
    const packLookupPromise = lookupInPack(combo.surface);
    const lessonPromise = fetchMorphologyLesson(lang, a, b, combo);

    // Prediction gate: show before revealing answer (#4).
    const correctGloss = combo.gloss || (combo.surface);
    totalPredictions += 1;
    const predResult = await showPredictionUI(a, b, correctGloss);
    if (predResult.correct) correctPredictions += 1;

    if (!overlay.isConnected) return;

    // Await the parallel calls (should be done or nearly so by now).
    const [packEntry, lessonText] = await Promise.all([packLookupPromise, lessonPromise]);

    if (!overlay.isConnected) return;

    burstAt(hammer, '#ffa84d');
    await showResult(combo, packEntry, lessonText, predResult);
    hammer.disabled = false;
  });

  renderTray();
}
