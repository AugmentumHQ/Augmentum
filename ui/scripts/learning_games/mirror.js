/**
 * Mirror — translation tile assembly.
 *
 * A sentence appears in the source language; assemble its translation by
 * dragging (or clicking) word tiles into numbered slots, then check.
 *
 * Directions:
 *   L2 → L1  (target language → English)   comprehension
 *   L1 → L2  (English → target language)   production
 *
 * Input modes (toggle button):
 *   Tiles  — drag/click pre-tokenised word tiles into slots
 *   Type   — free-text textarea graded by exact → llmJudge → similarity
 */

import {
  escapeHtml, fetchSentences, fetchGamePool, shuffle, pickN,
  makeGameOverlay, makeEmptyOverlay, speakWord, burstAt,
  gradeCard, fetchBreakdown, recordResult,
  llmJudgeJSON, gradeForEffort, makeClickableHTML, attachReaderPopover,
  showNotice, similarity,
} from './_common.js';

const ROUND_QS = 8;
const OVERLAP_THRESHOLD = 0.70;   // fallback content-word overlap ratio

// CJK langs that need dictionary-aware tokenisation (no whitespace gaps).
const _CJK_LANGS = new Set(['ja', 'zh']);

// ── CSS injection (guarded — runs once) ─────────────────────────────────────

function _ensureMirrorStyles() {
  if (document.getElementById('lg-mirror-styles')) return;
  const style = document.createElement('style');
  style.id = 'lg-mirror-styles';
  style.textContent = `
/* ── Mirror layout ───────────────────────────────────────── */
.lg-mr { display: flex; flex-direction: column; height: 100%; }

.lg-mr-stage { flex: 1; display: flex; flex-direction: column;
  gap: 12px; padding: 16px; overflow-y: auto; }

.lg-mr-prompt { display: flex; align-items: flex-start; gap: 8px;
  padding: 12px 14px; border-radius: 10px;
  background: color-mix(in srgb, var(--accent,#6ea8fe) 10%, transparent);
  min-height: 56px; }
.lg-mr-prompt-text { flex: 1; font-size: 20px; line-height: 1.4;
  word-break: break-word; }
.lg-mr-speak { background: none; border: none; cursor: pointer;
  font-size: 18px; color: var(--text-muted,#9aa0aa);
  padding: 2px 4px; line-height: 1; }
.lg-mr-speak:hover { color: var(--text-primary,#e8e8ea); }

/* ── Mode toggle ─────────────────────────────────────────── */
.lg-mr-mode-bar { display: flex; align-items: center; gap: 8px;
  padding: 0 2px; }
.lg-mr-mode-btn { flex: 1; padding: 6px; border-radius: 6px;
  border: 1px solid var(--border,rgba(255,255,255,.12));
  background: none; color: var(--text-muted,#9aa0aa);
  cursor: pointer; font-size: 13px; transition: all .15s; }
.lg-mr-mode-btn.active { background: var(--accent,#6ea8fe);
  color: #fff; border-color: transparent; }

/* ── Slot row ────────────────────────────────────────────── */
.lg-mr-slots { display: flex; flex-wrap: wrap; gap: 6px;
  min-height: 46px; padding: 6px 0; align-items: center; }
.lg-mr-slot { min-width: 56px; height: 40px; padding: 4px 10px;
  border-radius: 7px; border: 2px dashed var(--border,rgba(255,255,255,.2));
  background: var(--bg-elevated,#1b1d24); color: var(--text-primary,#e8e8ea);
  font-size: 14px; cursor: pointer; transition: all .15s;
  display: inline-flex; align-items: center; justify-content: center; }
.lg-mr-slot-empty { color: var(--text-muted,#9aa0aa); opacity: .5; }
.lg-mr-slot-full { border-style: solid;
  border-color: var(--border,rgba(255,255,255,.25)); }
.lg-mr-slot-full:hover { border-color: var(--accent,#6ea8fe); }
.lg-mr-slot-right { border-color: var(--success,#30a46c) !important;
  background: color-mix(in srgb, var(--success,#30a46c) 12%, transparent) !important; }
.lg-mr-slot-wrong { border-color: var(--error,#e5484d) !important;
  background: color-mix(in srgb, var(--error,#e5484d) 12%, transparent) !important; }
.lg-mr-slot-misplace { border-color: var(--warning,#e0a800) !important;
  background: color-mix(in srgb, var(--warning,#e0a800) 10%, transparent) !important; }
.lg-mr-slot-over { border-color: var(--accent,#6ea8fe) !important;
  background: color-mix(in srgb, var(--accent,#6ea8fe) 18%, transparent) !important; }

/* ── Tile tray ───────────────────────────────────────────── */
.lg-mr-tray { display: flex; flex-wrap: wrap; gap: 6px;
  padding: 8px; background: var(--bg-surface,#12141a);
  border-radius: 10px; min-height: 52px; align-items: flex-start; }
.lg-mr-tile { padding: 6px 12px; border-radius: 7px;
  background: var(--bg-elevated,#1b1d24);
  border: 1px solid var(--border,rgba(255,255,255,.14));
  color: var(--text-primary,#e8e8ea); font-size: 14px;
  cursor: grab; transition: all .12s; user-select: none; touch-action: none; }
.lg-mr-tile:hover:not(.lg-mr-tile-used) { border-color: var(--accent,#6ea8fe);
  background: color-mix(in srgb, var(--accent,#6ea8fe) 14%, var(--bg-elevated,#1b1d24)); }
.lg-mr-tile-used { opacity: .3; cursor: not-allowed; pointer-events: none; }
.lg-mr-tile-dragging { opacity: .55; transform: scale(1.06); box-shadow: 0 6px 20px rgba(0,0,0,.45); }

/* ── Type-mode textarea ──────────────────────────────────── */
.lg-mr-textarea { width: 100%; min-height: 72px; resize: vertical;
  padding: 10px 14px; border-radius: 8px;
  border: 1px solid var(--border,rgba(255,255,255,.14));
  background: var(--bg-surface,#12141a); color: var(--text-primary,#e8e8ea);
  font-size: 16px; line-height: 1.45; font-family: inherit; }
.lg-mr-textarea:focus { outline: none;
  border-color: var(--accent,#6ea8fe); }

/* ── Result / explanation panel ─────────────────────────── */
.lg-mr-result { border-radius: 10px;
  background: var(--bg-elevated,#1b1d24);
  border: 1px solid var(--border,rgba(255,255,255,.1));
  padding: 12px 14px; font-size: 14px; line-height: 1.5; }
.lg-mr-result-info { color: var(--text-muted,#9aa0aa); }
.lg-mr-result-good { color: var(--success,#30a46c); font-weight: 600; }
.lg-mr-result-bad  { color: var(--error,#e5484d);   font-weight: 600; }
.lg-mr-result-target { margin-top: 6px; font-size: 15px;
  color: var(--text-primary,#e8e8ea); font-style: italic; }

/* ── Breakdown panel (shown after every answer) ──────────── */
.lg-mr-breakdown { margin-top: 8px; border-top: 1px solid var(--border,rgba(255,255,255,.1));
  padding-top: 8px; }
.lg-mr-breakdown-label { font-size: 11px; text-transform: uppercase;
  letter-spacing: .06em; color: var(--text-muted,#9aa0aa); margin-bottom: 6px; }
.lg-mr-breakdown-ref { font-size: 14px; margin-bottom: 6px;
  color: var(--text-secondary,#c2c5cc); }
.lg-mr-breakdown-ref .lg-tok { cursor: pointer; }
.lg-mr-bd-list { list-style: none; margin: 0; padding: 0;
  display: flex; flex-wrap: wrap; gap: 4px; }
.lg-mr-bd-chip { padding: 3px 8px; border-radius: 5px;
  font-size: 13px; background: var(--bg-surface,#12141a);
  border: 1px solid var(--border,rgba(255,255,255,.09)); }
.lg-mr-bd-chip-miss { border-color: var(--error,#e5484d);
  background: color-mix(in srgb, var(--error,#e5484d) 12%, transparent); }
.lg-mr-bd-chip-token { font-weight: 600; }
.lg-mr-bd-chip-meaning { color: var(--text-muted,#9aa0aa); font-size: 11.5px;
  display: block; }

/* ── Controls ────────────────────────────────────────────── */
.lg-mr-controls { display: flex; gap: 8px; margin-top: 4px; }
.lg-mr-controls .btn { flex: 1; }

/* ── HUD extras ──────────────────────────────────────────── */
.lg-mr-progress { font-size: 13px; color: var(--text-muted,#9aa0aa); }
.lg-mr-flip { background: none; border: 1px solid var(--border,rgba(255,255,255,.14));
  border-radius: 6px; padding: 3px 8px; cursor: pointer;
  color: var(--text-secondary,#c2c5cc); font-size: 14px;
  transition: border-color .12s; }
.lg-mr-flip:hover { border-color: var(--accent,#6ea8fe); }

/* ── Slot-count adjuster ─────────────────────────────────── */
.lg-mr-slot-controls { display: flex; align-items: center; gap: 6px;
  font-size: 12px; color: var(--text-muted,#9aa0aa); }
.lg-mr-slot-adj { background: none; border: 1px solid var(--border,rgba(255,255,255,.12));
  border-radius: 5px; width: 24px; height: 24px; cursor: pointer;
  color: var(--text-secondary,#c2c5cc); display: flex; align-items: center; justify-content: center; }
.lg-mr-slot-adj:hover { border-color: var(--accent,#6ea8fe); }

/* Mirror surface pass */
.lg-mr {
  background:
    radial-gradient(circle at 16% 12%, rgba(93,208,194,0.16), transparent 28%),
    radial-gradient(circle at 82% 20%, rgba(255,216,102,0.08), transparent 24%),
    linear-gradient(180deg, rgba(8,18,22,0.96), rgba(11,15,22,0.98));
}
.lg-mr-stage {
  max-width: 880px;
  gap: 14px;
  padding: 18px;
}
.lg-mr-prompt {
  position: relative;
  overflow: hidden;
  border-radius: 8px;
  border: 1px solid rgba(93,208,194,0.22);
  border-left: 4px solid var(--accent,#5dd0c2);
  background:
    linear-gradient(135deg, rgba(93,208,194,0.16), rgba(255,255,255,0.035)),
    rgba(7,16,21,0.72);
  box-shadow: 0 18px 44px rgba(0,0,0,0.24);
}
.lg-mr-prompt::before {
  content: '';
  position: absolute;
  left: 14px;
  right: 14px;
  bottom: -22px;
  height: 46px;
  border-radius: 50%;
  background: radial-gradient(ellipse, rgba(93,208,194,0.22), transparent 68%);
  pointer-events: none;
}
.lg-mr-prompt-text {
  position: relative;
  letter-spacing: 0;
}
.lg-mr-mode-bar {
  padding: 4px;
  border-radius: 8px;
  background: rgba(255,255,255,0.045);
  border: 1px solid rgba(255,255,255,0.08);
}
.lg-mr-mode-btn {
  min-height: 34px;
  border-radius: 8px;
  font-weight: 800;
  letter-spacing: 0;
}
.lg-mr-mode-btn.active {
  background: linear-gradient(180deg, rgba(93,208,194,0.95), rgba(63,166,157,0.95));
  box-shadow: 0 6px 18px rgba(93,208,194,0.2);
}
.lg-mr-slots {
  border-radius: 8px;
  border: 1px solid rgba(93,208,194,0.18);
  background:
    linear-gradient(180deg, rgba(93,208,194,0.08), rgba(255,255,255,0.03)),
    rgba(0,0,0,0.22);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
}
.lg-mr-slot {
  min-height: 42px;
  border-radius: 8px;
  font-weight: 800;
  letter-spacing: 0;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
}
.lg-mr-slot-empty {
  background:
    repeating-linear-gradient(135deg, rgba(255,255,255,0.04) 0 8px, transparent 8px 16px),
    rgba(255,255,255,0.025);
}
.lg-mr-slot-right,
.lg-mr-slot-wrong,
.lg-mr-slot-misplace {
  animation: lg-mr-slot-pop 240ms ease-out;
}
.lg-mr-tray {
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.08);
  background:
    linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.02)),
    rgba(4,9,14,0.58);
  padding: 12px;
}
.lg-mr-tile {
  position: relative;
  overflow: hidden;
  border-radius: 8px;
  font-weight: 800;
  letter-spacing: 0;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.11), rgba(255,255,255,0.035)),
    rgba(18,28,36,0.94);
  box-shadow: 0 8px 18px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.08);
}
.lg-mr-tile::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(120deg, transparent, rgba(255,255,255,0.16), transparent);
  transform: translateX(-130%);
  transition: transform 320ms ease;
  pointer-events: none;
}
.lg-mr-tile:hover::before,
.lg-mr-tile:focus-visible::before {
  transform: translateX(130%);
}
.lg-mr-tile:focus-visible,
.lg-mr-slot:focus-visible,
.lg-mr-flip:focus-visible,
.lg-mr-speak:focus-visible {
  outline: 3px solid rgba(255,255,255,0.78);
  outline-offset: 3px;
}
.lg-mr-result {
  border-radius: 8px;
  text-align: left;
  border-color: rgba(255,255,255,0.10);
  background:
    linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.025)),
    rgba(8,13,18,0.78);
}
.lg-mr-breakdown {
  margin-top: 10px;
  padding: 10px;
  border-radius: 8px;
  background: rgba(255,255,255,0.035);
}
.lg-mr-breakdown-label {
  letter-spacing: 0;
  font-weight: 850;
}
.lg-mr-bd-chip {
  border-radius: 8px;
}
@keyframes lg-mr-slot-pop {
  0% { transform: translateY(0) scale(1); }
  50% { transform: translateY(-2px) scale(1.04); }
  100% { transform: translateY(0) scale(1); }
}
`;
  document.head.appendChild(style);
}

// ── Normalised content-word overlap (LLM-offline fallback) ───────────────────

function _contentWords(text) {
  const STOPWORDS = new Set([
    'the','a','an','is','are','was','were','be','been','being',
    'to','of','in','on','at','for','with','and','or','but','not',
    'it','its','he','she','they','we','i','you','this','that',
    'have','has','had','do','does','did','will','would','could',
    'can','may','might','shall','should','my','your','his','her','their',
  ]);
  return String(text).toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .split(/\s+/)
    .filter(w => w.length > 1 && !STOPWORDS.has(w));
}

function _overlapCheck(guessed, target) {
  const gWords = new Set(_contentWords(guessed));
  const tWords = _contentWords(target);
  if (!tWords.length) return false;
  const hits = tWords.filter(w => gWords.has(w)).length;
  const ratio = hits / tWords.length;
  // Also require word-count parity is not wildly off (factor-of-2 guard)
  const gLen = _contentWords(guessed).length;
  const tLen = tWords.length;
  const countOk = tLen > 0 && gLen >= Math.ceil(tLen / 2) && gLen <= tLen * 2;
  return ratio >= OVERLAP_THRESHOLD && countOk;
}

// ── LLM equivalence judge (never silently returns wrong) ─────────────────────

async function llmEquivalent(lang, guessed, target) {
  const sys = [
    `You are a strict bilingual judge for ${lang}/English.`,
    `Given two translation attempts, decide if they convey the SAME CORE MEANING.`,
    `Minor word order, pronoun drops, or synonym swaps are OK.`,
    `Respond ONLY with valid JSON: {"same": true} or {"same": false, "reason": "..."}.`,
  ].join(' ');
  const user = `Reference: "${target}"\nAttempt: "${guessed}"\nSame meaning?`;

  const result = await llmJudgeJSON(
    [{ role: 'system', content: sys }, { role: 'user', content: user }],
    { fallback: null, timeoutMs: 8000 },
  );

  if (result !== null && typeof result === 'object' && typeof result.same === 'boolean') {
    return result.same;
  }
  // LLM offline, timed out, or returned garbage → overlap fallback
  return _overlapCheck(guessed, target);
}

// ── Explanation panel: per-token breakdown shown after every answer ───────────

async function _buildBreakdownPanel(container, langText, lang, voice, bdData, guessedTokens) {
  // bdData may be the fetchBreakdown result (has .tokens) or null.
  // We show it only for the L2 side (the side that's in the target lang).
  container.innerHTML = '';
  const label = document.createElement('div');
  label.className = 'lg-mr-breakdown-label';
  label.textContent = `Sentence breakdown (tap to explore)`;
  container.appendChild(label);

  // Render the reference sentence with clickable tokens for the popover.
  const refRow = document.createElement('div');
  refRow.className = 'lg-mr-breakdown-ref';
  refRow.innerHTML = await makeClickableHTML(langText, lang);
  refRow.dataset.fullText = langText;
  container.appendChild(refRow);

  // Per-token chip strip (role + meaning) if we have breakdown data.
  const tokens = bdData && Array.isArray(bdData.tokens)
    ? bdData.tokens.filter(t => t.matched && t.text)
    : [];

  if (tokens.length) {
    // Build a set of tokens the user missed (only relevant for wrong answers).
    const missedSet = new Set(
      (guessedTokens || [])
        .filter(g => !tokens.find(t => (t.text || '').toLowerCase() === g.toLowerCase())),
    );

    const chipList = document.createElement('ul');
    chipList.className = 'lg-mr-bd-list';
    for (const tok of tokens) {
      const chip = document.createElement('li');
      const wasMissed = missedSet.has(tok.text || '');
      chip.className = `lg-mr-bd-chip${wasMissed ? ' lg-mr-bd-chip-miss' : ''}`;
      const meaning = tok.meaning || tok.gloss || '';
      const role = tok.role || tok.pos || '';
      chip.innerHTML = `<span class="lg-mr-bd-chip-token">${escapeHtml(tok.text)}</span>`
        + (meaning ? `<span class="lg-mr-bd-chip-meaning">${escapeHtml(meaning)}${role ? ` · ${escapeHtml(role)}` : ''}</span>` : '');
      chipList.appendChild(chip);
    }
    container.appendChild(chipList);
  }

  // Return the reader popover handle so callers can destroy it on next question.
  return attachReaderPopover(container, { lang, voice });
}

// ── Main export ──────────────────────────────────────────────────────────────

export async function launchMirror({ lang, voice }) {
  const [sents, pool] = await Promise.all([
    fetchSentences(lang, 30),
    fetchGamePool(lang, 30),
  ]);
  const usable = sents.filter(s => s.lang_text && s.en_text);
  if (usable.length < 4) {
    return makeEmptyOverlay({
      palette: 'teal', emoji: '🪞',
      message: 'Mirror needs Tatoeba sentences with English translations.',
      hint: 'Your pack may not include them yet — try a different language pack or seed more sentences.',
    });
  }

  _ensureMirrorStyles();

  const { overlay, close } = makeGameOverlay({
    id: 'lg-mirror', palette: 'teal', title: 'Mirror',
  });

  overlay.innerHTML = `
    <div class="lg-game lg-mr">
      <header class="lg-hud">
        <button type="button" class="lg-close" aria-label="Close">×</button>
        <div class="lg-mr-progress" id="lg-mr-progress">1 / ${ROUND_QS}</div>
        <div class="lg-hud-stats">
          <button type="button" class="lg-mr-flip" id="lg-mr-flip" title="Swap direction (L2↔L1)">⇄</button>
          <span class="lg-hud-label">score</span>
          <span id="lg-mr-score">0000</span>
        </div>
      </header>

      <div class="lg-mr-stage" id="lg-mr-stage">
        <!-- Source prompt -->
        <div class="lg-mr-prompt">
          <div style="flex:1">
            <div class="lg-hud-label" id="lg-mr-prompt-label">target</div>
            <div class="lg-mr-prompt-text" id="lg-mr-prompt-text">—</div>
          </div>
          <button type="button" class="lg-mr-speak" id="lg-mr-speak" aria-label="Hear">🔊</button>
        </div>

        <!-- Input mode toggle (Tiles / Type) -->
        <div class="lg-mr-mode-bar">
          <button type="button" class="lg-mr-mode-btn active" id="lg-mr-mode-tiles">Tiles</button>
          <button type="button" class="lg-mr-mode-btn" id="lg-mr-mode-type">Type</button>
        </div>

        <!-- Tiles input -->
        <div id="lg-mr-tiles-ui">
          <div class="lg-mr-slot-controls">
            <span>Slots:</span>
            <button type="button" class="lg-mr-slot-adj" id="lg-mr-slot-minus" aria-label="Fewer slots">−</button>
            <span id="lg-mr-slot-count">0</span>
            <button type="button" class="lg-mr-slot-adj" id="lg-mr-slot-plus" aria-label="More slots">+</button>
          </div>
          <div class="lg-mr-slots" id="lg-mr-slots"></div>
          <div class="lg-mr-tray" id="lg-mr-tray"></div>
        </div>

        <!-- Type input -->
        <div id="lg-mr-type-ui" hidden>
          <textarea class="lg-mr-textarea" id="lg-mr-textarea"
            rows="3" placeholder="Type your translation here…"></textarea>
        </div>

        <div class="lg-mr-controls">
          <button type="button" class="btn btn-ghost" id="lg-mr-reset">Reset</button>
          <button type="button" class="btn btn-primary" id="lg-mr-check">Check</button>
        </div>

        <div class="lg-mr-result" id="lg-mr-result" hidden></div>
      </div>

      <div class="lg-end" id="lg-mr-end" hidden>
        <div class="lg-end-title">Round complete</div>
        <div class="lg-end-stats" id="lg-mr-end-stats"></div>
        <div class="lg-end-actions">
          <button type="button" class="btn btn-primary" id="lg-mr-replay">Play again</button>
          <button type="button" class="btn btn-ghost" id="lg-mr-quit">Done</button>
        </div>
      </div>
    </div>`;

  overlay.querySelector('.lg-close').addEventListener('click', () => close());

  // ── Element refs ──────────────────────────────────────────────────────────
  const promptText   = overlay.querySelector('#lg-mr-prompt-text');
  const promptLabel  = overlay.querySelector('#lg-mr-prompt-label');
  const speakBtn     = overlay.querySelector('#lg-mr-speak');
  const slotsEl      = overlay.querySelector('#lg-mr-slots');
  const trayEl       = overlay.querySelector('#lg-mr-tray');
  const resultEl     = overlay.querySelector('#lg-mr-result');
  const flipBtn      = overlay.querySelector('#lg-mr-flip');
  const scoreEl      = overlay.querySelector('#lg-mr-score');
  const progressEl   = overlay.querySelector('#lg-mr-progress');
  const endPanel     = overlay.querySelector('#lg-mr-end');
  const tilesUI      = overlay.querySelector('#lg-mr-tiles-ui');
  const typeUI       = overlay.querySelector('#lg-mr-type-ui');
  const textarea     = overlay.querySelector('#lg-mr-textarea');
  const modeTiles    = overlay.querySelector('#lg-mr-mode-tiles');
  const modeType     = overlay.querySelector('#lg-mr-mode-type');
  const slotCountEl  = overlay.querySelector('#lg-mr-slot-count');
  const slotMinusBtn = overlay.querySelector('#lg-mr-slot-minus');
  const slotPlusBtn  = overlay.querySelector('#lg-mr-slot-plus');
  const stageEl      = overlay.querySelector('#lg-mr-stage');

  // ── State ─────────────────────────────────────────────────────────────────
  let score     = 0;
  let qIdx      = 0;
  let direction = 'l2_to_l1';   // l2_to_l1 = target→EN,  l1_to_l2 = EN→target
  let inputMode = 'tiles';       // 'tiles' | 'type'
  let current   = null;
  let slots     = [];            // array of {id,text}|null
  let currentTiles = [];         // all tiles for this question
  let checking  = false;
  let answerMs  = 0;             // timestamp at question start (for gradeForEffort)
  let attempts  = 0;
  let hintsUsed = 0;
  // Slot count range (variable-length)
  let slotMin   = 1;
  let slotMax   = 1;
  let slotTarget = 1;            // current user-chosen slot count

  const correct = [];
  const wrong   = [];
  const usedIdx = new Set();

  // Popover handle for the breakdown panel (destroy on next question).
  let _popoverHandle = null;

  // ── FSRS helpers ─────────────────────────────────────────────────────────
  const _isCJKUnsegmented = _CJK_LANGS.has(lang);

  function _cardsInSentence(sentence) {
    const text = sentence && sentence.lang_text;
    if (!text) return [];
    const seen = new Set();
    const out  = [];
    for (const c of pool) {
      if (!c.surface || c.in_queue === false) continue;
      if (seen.has(c.word_id)) continue;
      let hit = false;
      if (_isCJKUnsegmented) {
        if (c.surface.length >= 2 && text.includes(c.surface)) hit = true;
      } else {
        const escaped = c.surface.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const re = new RegExp(`(^|[^\\p{L}\\p{N}])${escaped}(?=$|[^\\p{L}\\p{N}])`, 'u');
        if (re.test(text)) hit = true;
      }
      if (hit) { seen.add(c.word_id); out.push(c); }
    }
    return out;
  }

  const roundStart = performance.now();

  // ── Sentence selection ───────────────────────────────────────────────────
  function pickQuestion() {
    if (usedIdx.size >= usable.length) usedIdx.clear();
    let i;
    do { i = Math.floor(Math.random() * usable.length); } while (usedIdx.has(i));
    usedIdx.add(i);
    return usable[i];
  }

  // ── Tokenisation ─────────────────────────────────────────────────────────
  function whitespaceTokens(text) {
    return String(text).trim().split(/\s+/).filter(Boolean);
  }

  // Cache breakdown data per question to avoid double-fetching.
  let _currentBdData = null;

  async function tokeniseFor(text, isTargetLang) {
    if (!isTargetLang || !_CJK_LANGS.has(lang)) return whitespaceTokens(text);
    const bd = await fetchBreakdown(lang, text);
    _currentBdData = bd;   // cache for explanation panel
    if (!bd || !Array.isArray(bd.tokens)) return whitespaceTokens(text);
    const out = [];
    for (const tok of bd.tokens) {
      const t = tok.text || '';
      if (!t) continue;
      if (!tok.matched && /^[\s\p{P}]+$/u.test(t)) {
        if (t.trim() && out.length) out[out.length - 1] = out[out.length - 1] + t;
        continue;
      }
      out.push(t);
    }
    return out.length ? out : whitespaceTokens(text);
  }

  // ── Mode toggle ───────────────────────────────────────────────────────────
  function _setInputMode(mode) {
    inputMode = mode;
    modeTiles.classList.toggle('active', mode === 'tiles');
    modeType.classList.toggle('active', mode === 'type');
    tilesUI.hidden = mode !== 'tiles';
    typeUI.hidden  = mode !== 'type';
    resultEl.hidden = true;
    attempts = 0;
    answerMs = performance.now();
  }

  modeTiles.addEventListener('click', () => _setInputMode('tiles'));
  modeType.addEventListener('click',  () => _setInputMode('type'));

  // ── Slot count controls ───────────────────────────────────────────────────
  function _updateSlotCount() {
    slotCountEl.textContent = String(slotTarget);
    slotMinusBtn.disabled = slotTarget <= slotMin;
    slotPlusBtn.disabled  = slotTarget >= slotMax;
    // Resize slots array preserving any filled tiles.
    const prev = slots;
    slots = new Array(slotTarget).fill(null);
    for (let i = 0; i < Math.min(slotTarget, prev.length); i++) slots[i] = prev[i];
    // Return any tiles that no longer have a slot back to the tray.
    for (let i = slotTarget; i < prev.length; i++) {
      if (prev[i]) {
        const btn = trayEl.querySelector(`[data-tile-id="${prev[i].id}"]`);
        if (btn) btn.classList.remove('lg-mr-tile-used');
      }
    }
    renderSlots();
  }

  slotMinusBtn.addEventListener('click', () => {
    if (slotTarget > slotMin) { slotTarget--; _updateSlotCount(); }
  });
  slotPlusBtn.addEventListener('click', () => {
    if (slotTarget < slotMax) { slotTarget++; _updateSlotCount(); }
  });

  // ── Render ───────────────────────────────────────────────────────────────
  async function render() {
    // Destroy previous popover and reset state.
    _popoverHandle?.destroy();
    _popoverHandle = null;
    _currentBdData = null;
    attempts = 0;
    hintsUsed = 0;
    answerMs = performance.now();

    current = pickQuestion();
    progressEl.textContent = `${qIdx + 1} / ${ROUND_QS}`;
    resultEl.hidden = true;
    textarea.value = '';

    const prompt  = direction === 'l2_to_l1' ? current.lang_text : current.en_text;
    const target  = direction === 'l2_to_l1' ? current.en_text   : current.lang_text;
    promptText.textContent = prompt;
    promptLabel.textContent = direction === 'l2_to_l1'
      ? `${lang.toUpperCase()} → EN` : `EN → ${lang.toUpperCase()}`;
    // Don't speak the L2 prompt in production mode — it gives the answer.
    speakBtn.hidden = direction === 'l1_to_l2';

    const targetIsTargetLang = direction === 'l1_to_l2';
    const targetTokens = await tokeniseFor(target, targetIsTargetLang);

    // Variable-length slots: min = ceil(len/2), max = len+2, default = len.
    slotMin    = Math.max(1, Math.ceil(targetTokens.length / 2));
    slotMax    = targetTokens.length + 2;
    slotTarget = targetTokens.length;
    slots      = new Array(slotTarget).fill(null);

    // Also pre-fetch breakdown for the L2 side when in l2→l1 mode, for the
    // explanation panel. For l1→l2 the target IS the L2 side; tokeniseFor
    // already set _currentBdData above.
    if (direction === 'l2_to_l1' && !_currentBdData) {
      // Fetch in parallel; don't block render.
      fetchBreakdown(lang, current.lang_text).then(bd => { _currentBdData = bd; });
    }

    // Build tile pool: correct tokens + up to 4 distractors.
    let distractorSource;
    if (direction === 'l2_to_l1') {
      distractorSource = usable.flatMap(s => whitespaceTokens(s.en_text || ''))
        .filter(t => t.length > 1 && !targetTokens.includes(t));
    } else {
      distractorSource = pool.map(c => c.surface).filter(s => s && !targetTokens.includes(s));
    }
    const distractors = pickN(Array.from(new Set(distractorSource)), Math.min(4, targetTokens.length));
    currentTiles = shuffle([...targetTokens, ...distractors]).map((tok, i) => ({ id: i, text: tok }));

    _updateSlotCount();
    renderTiles(currentTiles);
  }

  // ── Slot rendering + click-to-return ────────────────────────────────────
  function renderSlots() {
    slotsEl.innerHTML = slots.map((s, i) =>
      s == null
        ? `<button type="button" class="lg-mr-slot lg-mr-slot-empty" data-i="${i}" aria-label="Empty slot ${i + 1}">·</button>`
        : `<button type="button" class="lg-mr-slot lg-mr-slot-full" data-i="${i}">${escapeHtml(s.text)}</button>`,
    ).join('');

    slotsEl.querySelectorAll('.lg-mr-slot').forEach(el => {
      // Click-to-return a placed tile.
      el.addEventListener('click', () => {
        const i = Number(el.dataset.i);
        if (slots[i] != null) {
          const tile = slots[i];
          slots[i] = null;
          const trayBtn = trayEl.querySelector(`[data-tile-id="${tile.id}"]`);
          if (trayBtn) trayBtn.classList.remove('lg-mr-tile-used');
          renderSlots();
        }
      });
      // Drag-over target.
      el.addEventListener('dragover', e => { e.preventDefault(); el.classList.add('lg-mr-slot-over'); });
      el.addEventListener('dragleave', () => el.classList.remove('lg-mr-slot-over'));
      el.addEventListener('drop', e => {
        e.preventDefault();
        el.classList.remove('lg-mr-slot-over');
        const tileId = Number(e.dataTransfer.getData('text/plain'));
        _placeOrSwap(tileId, Number(el.dataset.i));
      });
      // Pointer/touch drop target (for touch drag).
      el.addEventListener('pointerup', e => {
        if (_dragging && _dragging.id !== undefined) {
          e.preventDefault();
          _placeOrSwap(_dragging.id, Number(el.dataset.i));
          _endDrag();
        }
      });
    });
  }

  // ── Tile rendering + drag/click ──────────────────────────────────────────
  // Drag state.
  let _dragging = null;
  let _dragGhost = null;

  function _endDrag() {
    _dragging = null;
    if (_dragGhost) { _dragGhost.remove(); _dragGhost = null; }
    document.body.style.userSelect = '';
  }

  // Place tileId into slotIndex, swapping if the slot is already occupied.
  function _placeOrSwap(tileId, slotIndex) {
    const tile = currentTiles.find(t => t.id === tileId);
    if (!tile) return;
    // If the tile was already in a slot, vacate it first.
    const prevSlot = slots.findIndex(s => s && s.id === tileId);
    const displaced = slots[slotIndex];   // tile currently in target slot (may be null)

    if (prevSlot >= 0) {
      slots[prevSlot] = displaced;       // swap
    } else {
      // Tile came from tray — mark it used, put displaced back to tray.
      if (displaced) {
        const btn = trayEl.querySelector(`[data-tile-id="${displaced.id}"]`);
        if (btn) btn.classList.remove('lg-mr-tile-used');
      }
    }
    slots[slotIndex] = tile;
    const trayBtn = trayEl.querySelector(`[data-tile-id="${tileId}"]`);
    if (trayBtn) trayBtn.classList.add('lg-mr-tile-used');
    renderSlots();
  }

  function renderTiles(tiles) {
    trayEl.innerHTML = tiles.map(t =>
      `<button type="button" class="lg-mr-tile" data-tile-id="${t.id}" draggable="true">${escapeHtml(t.text)}</button>`,
    ).join('');

    trayEl.querySelectorAll('.lg-mr-tile').forEach(btn => {
      const id = Number(btn.dataset.tileId);
      const tile = currentTiles.find(t => t.id === id);
      if (!tile) return;

      // ── Click-to-next-empty ──────────────────────────────────────────────
      btn.addEventListener('click', () => {
        if (btn.classList.contains('lg-mr-tile-used')) return;
        const nextEmpty = slots.findIndex(s => s == null);
        if (nextEmpty < 0) return;
        slots[nextEmpty] = tile;
        btn.classList.add('lg-mr-tile-used');
        renderSlots();
      });

      // ── HTML5 drag ───────────────────────────────────────────────────────
      btn.addEventListener('dragstart', e => {
        e.dataTransfer.setData('text/plain', String(id));
        btn.classList.add('lg-mr-tile-dragging');
      });
      btn.addEventListener('dragend', () => btn.classList.remove('lg-mr-tile-dragging'));

      // ── Pointer / touch drag ─────────────────────────────────────────────
      btn.addEventListener('pointerdown', e => {
        if (btn.classList.contains('lg-mr-tile-used')) return;
        e.preventDefault();
        _dragging = tile;
        document.body.style.userSelect = 'none';
        // Create a visual ghost.
        _dragGhost = btn.cloneNode(true);
        _dragGhost.style.cssText = `position:fixed;z-index:99999;pointer-events:none;
          opacity:.75;transform:scale(1.08);`;
        document.body.appendChild(_dragGhost);
        function onMove(ev) {
          const pt = ev.touches ? ev.touches[0] : ev;
          _dragGhost.style.left = `${pt.clientX - 30}px`;
          _dragGhost.style.top  = `${pt.clientY - 20}px`;
        }
        function onUp(ev) {
          // If the pointer didn't land on a slot, find the nearest one.
          const pt = (ev.changedTouches ? ev.changedTouches[0] : ev);
          const hit = document.elementFromPoint(pt.clientX, pt.clientY);
          const slotEl = hit && (hit.classList.contains('lg-mr-slot') ? hit : hit.closest('.lg-mr-slot'));
          if (slotEl && _dragging) {
            _placeOrSwap(_dragging.id, Number(slotEl.dataset.i));
          }
          _endDrag();
          document.removeEventListener('pointermove', onMove);
          document.removeEventListener('pointerup',   onUp);
          document.removeEventListener('touchmove',   onMove);
          document.removeEventListener('touchend',    onUp);
        }
        document.addEventListener('pointermove', onMove, { passive: true });
        document.addEventListener('pointerup',   onUp);
        document.addEventListener('touchmove',   onMove, { passive: true });
        document.addEventListener('touchend',    onUp);
      });
    });
  }

  // ── Reset ─────────────────────────────────────────────────────────────────
  overlay.querySelector('#lg-mr-reset').addEventListener('click', () => {
    slots = slots.map(() => null);
    trayEl.querySelectorAll('.lg-mr-tile').forEach(b => b.classList.remove('lg-mr-tile-used'));
    textarea.value = '';
    resultEl.hidden = true;
    renderSlots();
    attempts = 0;
    answerMs = performance.now();
  });

  // ── Check answer ──────────────────────────────────────────────────────────
  const checkBtn = overlay.querySelector('#lg-mr-check');
  checkBtn.addEventListener('click', async () => {
    if (checking) return;
    const target = direction === 'l2_to_l1' ? current.en_text : current.lang_text;

    let guessed;
    if (inputMode === 'tiles') {
      if (slots.some(s => s == null)) {
        resultEl.innerHTML = `<div class="lg-mr-result-info">Fill every slot first, or remove slots with the − button.</div>`;
        resultEl.hidden = false;
        return;
      }
      guessed = slots.map(s => s.text).join(' ');
    } else {
      guessed = textarea.value.trim();
      if (!guessed) {
        resultEl.innerHTML = `<div class="lg-mr-result-info">Type your translation first.</div>`;
        resultEl.hidden = false;
        return;
      }
    }

    attempts++;
    const norm = t => t.replace(/\s+/g, ' ').trim().toLowerCase();
    const exact = norm(guessed) === norm(target);

    let valid = exact;
    if (!valid) {
      checking = true;
      checkBtn.disabled = true;
      resultEl.innerHTML = `<div class="lg-mr-result-info">Checking…</div>`;
      resultEl.hidden = false;
      valid = await llmEquivalent(lang, guessed, target);
      checking = false;
      checkBtn.disabled = false;
    }

    const elapsed = performance.now() - answerMs;
    const effortGrade = gradeForEffort({
      correct: valid, attempts, hintsUsed, replays: 0, ms: elapsed,
    });

    // Slot highlighting (tiles mode only).
    if (inputMode === 'tiles') {
      const targetTokens = norm(target).split(/\s+/);
      slotsEl.querySelectorAll('.lg-mr-slot').forEach((el, i) => {
        if (valid) {
          el.classList.add('lg-mr-slot-right');
        } else {
          const placed = (slots[i]?.text || '').toLowerCase();
          const expected = targetTokens[i] || '';
          if (!placed) return;
          if (placed === expected) el.classList.add('lg-mr-slot-right');
          else if (norm(target).includes(placed)) el.classList.add('lg-mr-slot-misplace');
          else el.classList.add('lg-mr-slot-wrong');
        }
      });
    }

    // Determine which L2 text and breakdown to show in the explanation.
    const l2Text  = direction === 'l2_to_l1' ? current.lang_text : current.lang_text;
    const guessedTokens = inputMode === 'tiles'
      ? slots.map(s => s?.text || '').filter(Boolean)
      : norm(guessed).split(/\s+/);

    if (valid) {
      const pts = exact ? 15 : (inputMode === 'type' ? 12 : 10);   // type earns slightly more
      score += pts;
      scoreEl.textContent = String(score).padStart(4, '0');
      if (inputMode === 'tiles') burstAt(slotsEl, '#5dd0c2');

      // Speak the L2 sentence after correct L2→L1 answer (hear what you translated).
      if (direction === 'l2_to_l1' && voice) {
        speakWord(current.lang_text, voice).catch(() => {});
      }

      resultEl.innerHTML = `
        <div class="lg-mr-result-good">✓ ${exact ? 'Exactly right.' : 'Equivalent meaning.'}</div>
        <div class="lg-mr-result-target">${escapeHtml(target)}</div>
        <div class="lg-mr-breakdown" id="lg-mr-breakdown-panel"></div>`;
      resultEl.hidden = false;
      correct.push({ sentence: current, grade: effortGrade });

      // Build breakdown panel asynchronously (don't block next-question timer).
      const bdPanel = resultEl.querySelector('#lg-mr-breakdown-panel');
      if (bdPanel) {
        _buildBreakdownPanel(bdPanel, l2Text, lang, voice, _currentBdData, guessedTokens)
          .then(handle => { _popoverHandle = handle; })
          .catch(() => {});
      }

      setTimeout(() => {
        qIdx++;
        if (qIdx >= ROUND_QS) endRound();
        else render();
      }, 2200);
    } else {
      resultEl.innerHTML = `
        <div class="lg-mr-result-bad">Not quite — try again or skip.</div>
        <div class="lg-mr-result-target">${escapeHtml(target)}</div>
        <div class="lg-mr-breakdown" id="lg-mr-breakdown-panel"></div>
        <button type="button" class="btn btn-ghost lg-mr-skip" style="margin-top:8px">Skip</button>`;
      resultEl.hidden = false;

      const bdPanel = resultEl.querySelector('#lg-mr-breakdown-panel');
      if (bdPanel) {
        _buildBreakdownPanel(bdPanel, l2Text, lang, voice, _currentBdData, guessedTokens)
          .then(handle => { _popoverHandle = handle; })
          .catch(() => {});
      }

      wrong.push({ sentence: current, grade: effortGrade });
      resultEl.querySelector('.lg-mr-skip').addEventListener('click', () => {
        qIdx++;
        if (qIdx >= ROUND_QS) endRound();
        else render();
      });
    }
  });

  // Flip direction.
  flipBtn.addEventListener('click', () => {
    direction = direction === 'l2_to_l1' ? 'l1_to_l2' : 'l2_to_l1';
    render();
  });

  // Speak prompt.
  speakBtn.addEventListener('click', () => {
    const text = direction === 'l2_to_l1' ? current.lang_text : current.en_text;
    speakWord(text, voice);
  });

  // Textarea: Enter submits, Shift+Enter = newline.
  textarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); checkBtn.click(); }
  });

  // ── End of round ──────────────────────────────────────────────────────────
  async function endRound() {
    _popoverHandle?.destroy();
    _popoverHandle = null;

    // Grade cards for correctly-translated sentences. Use the per-answer
    // effortGrade accumulated in correct[]/wrong[] rather than always grading 3.
    const goodSeen = new Set();
    const toGrade  = [];   // [{card, grade}]
    for (const { sentence, grade } of correct) {
      for (const c of _cardsInSentence(sentence)) {
        if (toGrade.length >= 5) break;
        if (goodSeen.has(c.word_id)) continue;
        goodSeen.add(c.word_id);
        toGrade.push({ card: c, grade });
      }
      if (toGrade.length >= 5) break;
    }
    const toFail = [];
    for (const { sentence, grade } of wrong) {
      for (const c of _cardsInSentence(sentence)) {
        if (toFail.length >= 5) break;
        if (goodSeen.has(c.word_id)) continue;
        goodSeen.add(c.word_id);
        toFail.push({ card: c, grade });
      }
      if (toFail.length >= 5) break;
    }

    await Promise.all([
      ...toGrade.map(({ card, grade }) => gradeCard(lang, card.word_id, grade)),
      ...toFail.map(({ card })         => gradeCard(lang, card.word_id, 1)),
    ]);

    overlay.querySelector('#lg-mr-end-stats').innerHTML = `
      <div class="lg-end-stat">
        <div class="lg-end-stat-n">${String(score).padStart(4, '0')}</div>
        <div>score</div>
      </div>
      <div class="lg-end-stat">
        <div class="lg-end-stat-n">${correct.length}/${ROUND_QS}</div>
        <div>correct</div>
      </div>
      <div class="lg-end-stat">
        <div class="lg-end-stat-n">${toGrade.length + toFail.length}</div>
        <div>graded</div>
      </div>`;
    endPanel.hidden = false;

    recordResult({
      game_id: 'mirror', lang,
      score, words_played: ROUND_QS, words_correct: correct.length,
      duration_sec: Math.round((performance.now() - roundStart) / 1000),
      metadata: { direction, inputMode },
    });
  }

  overlay.querySelector('#lg-mr-replay').addEventListener('click', () => {
    score = 0; qIdx = 0; correct.length = 0; wrong.length = 0;
    usedIdx.clear();
    scoreEl.textContent = '0000';
    endPanel.hidden = true;
    render();
  });
  overlay.querySelector('#lg-mr-quit').addEventListener('click', () => close());

  // Warn if the LLM is unreachable (so the user knows the fallback is in use).
  (async () => {
    const probe = await llmJudgeJSON(
      [{ role: 'user', content: 'ping' }],
      { fallback: null, timeoutMs: 4000 },
    );
    if (probe === null) {
      showNotice(stageEl,
        'Language model offline — using word-overlap grading as a fallback.',
        { kind: 'warn' });
    }
  })().catch(() => {});

  render();
}
