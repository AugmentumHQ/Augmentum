/**
 * 🫧 Bubble Pop — audio-first recognition.
 *
 * TTS speaks a target word; the player pops the matching bubble.
 * THE AUDIO IS THE PUZZLE — the gloss is hidden behind TTS. Only a hint
 * tap (or audio unavailability) reveals the gloss early.
 *
 * Improvements (2026-06 production pass):
 *  #1 Audio integrity    — target prompt shows only 🔊 "listen…" until
 *                          TTS finishes (or hint is tapped). Replay button.
 *                          If voice unavailable → gloss shown + notice.
 *  #2 Semantic distractors — at higher difficulty ≥2 active bubbles share
 *                            the target's POS tag so they're plausible traps.
 *  #3 Post-round review  — missed words shown with reading + gloss +
 *                          example sentence, L2 text tappable via
 *                          makeClickableHTML + attachReaderPopover.
 *  #4 Effort-aware FSRS  — gradeForEffort replaces flat grade 3/1.
 *                          Audio-only + fast → Easy (4); hint → friction;
 *                          TTS-never-played targets are NOT graded.
 *  #5 Bubble micro-anchor — hover tooltip: reading + first gloss.
 */

import {
  escapeHtml, fetchGamePool, speakWord, stopAudio, pickOne,
  makeGameOverlay, makeEmptyOverlay, gradeCard, burstAt, fmtScore,
  recordResult, fetchBestScores,
  makeClickableHTML, attachReaderPopover,
  gradeForEffort, showNotice, fetchSentences,
} from './_common.js';

// ── Module-level CSS (injected once) ───────────────────────────────────
(function _ensureBubbleStyles() {
  if (document.getElementById('lg-bubble-styles')) return;
  const style = document.createElement('style');
  style.id = 'lg-bubble-styles';
  style.textContent = `
/* ── Prompt / target strip ─────────────────────────────── */
.lg-bp-target {
  display: flex; flex-direction: column; align-items: center;
  gap: 2px; min-width: 0;
}
.lg-bp-prompt-audio {
  display: flex; align-items: center; gap: 8px;
  font-size: 22px; font-weight: 700; letter-spacing: .01em;
  color: var(--text-primary, #e8e8ea);
}
.lg-bp-prompt-audio.lg-bp-fresh { animation: lg-bp-fresh .38s ease; }
@keyframes lg-bp-fresh {
  0%   { opacity: 0; transform: scale(.85); }
  60%  { transform: scale(1.06); }
  100% { opacity: 1; transform: scale(1); }
}
.lg-bp-replay-btn {
  background: none; border: none; cursor: pointer; font-size: 20px;
  color: var(--accent, #6ea8fe); padding: 2px 4px; border-radius: 6px;
  transition: background .12s;
  line-height: 1;
}
.lg-bp-replay-btn:hover { background: color-mix(in srgb, var(--accent,#6ea8fe) 18%, transparent); }
.lg-bp-hint-btn {
  font-size: 11px; padding: 2px 7px; border-radius: 8px;
  border: 1px solid var(--border, rgba(255,255,255,.18));
  background: none; color: var(--text-muted, #9aa0aa);
  cursor: pointer; transition: background .12s;
}
.lg-bp-hint-btn:hover { background: color-mix(in srgb, var(--warning,#e0a800) 18%, transparent); }
.lg-bp-gloss-reveal {
  font-size: 13.5px; color: var(--text-secondary, #c2c5cc);
  margin-top: 2px; text-align: center;
  animation: lg-rdr-in .18s ease;
}

/* ── Bubble micro-anchor tooltip ───────────────────────── */
.lg-bp-bubble { position: absolute; overflow: visible; }
.lg-bp-bubble:hover::after {
  content: attr(data-tip);
  position: absolute; bottom: calc(100% + 6px); left: 50%;
  transform: translateX(-50%);
  background: var(--bg-elevated, #1b1d24);
  color: var(--text-primary, #e8e8ea);
  border: 1px solid var(--border, rgba(255,255,255,.14));
  border-radius: 8px; padding: 4px 9px;
  font-size: 12px; white-space: nowrap;
  pointer-events: none; z-index: 9999;
  box-shadow: 0 4px 16px rgba(0,0,0,.4);
}

/* ── Post-round review strip ───────────────────────────── */
.lg-bp-review {
  margin: 0 0 14px;
  border: 1px solid var(--border, rgba(255,255,255,.10));
  border-radius: 12px; overflow: hidden;
}
.lg-bp-review-title {
  font-size: 12px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--text-muted, #9aa0aa);
  padding: 8px 14px 6px; background: rgba(0,0,0,.15);
}
.lg-bp-review-list { list-style: none; margin: 0; padding: 0; }
.lg-bp-review-item {
  padding: 10px 14px; border-bottom: 1px solid var(--border, rgba(255,255,255,.07));
  display: grid; grid-template-columns: auto 1fr; gap: 2px 12px;
}
.lg-bp-review-item:last-child { border-bottom: none; }
.lg-bp-review-surface {
  grid-row: span 2; font-size: 20px; font-weight: 700;
  color: var(--text-primary, #e8e8ea); align-self: center;
}
.lg-bp-review-reading { font-size: 12px; color: var(--text-muted, #9aa0aa); }
.lg-bp-review-gloss   { font-size: 14px; color: var(--text-secondary, #c2c5cc); }
.lg-bp-review-sentence {
  grid-column: 1 / -1; margin-top: 4px;
  font-size: 13px; color: var(--text-secondary, #c2c5cc);
  line-height: 1.5;
  padding: 4px 8px;
  background: rgba(255,255,255,.04); border-radius: 6px;
}
.lg-bp-review-sentence .lg-tok { color: var(--accent, #6ea8fe); }
`;
  document.head.appendChild(style);
}());

const ROUND_MS = 60_000;
const SPAWN_MS_START = 1100;
const SPAWN_MS_END = 500;
const LIFESPAN_MS_START = 11_000;
const LIFESPAN_MS_END = 6_000;
const TARGET_GAP_MS = 2800;
// After TTS starts, how long we wait before the audio can be called "played"
// (enough that the user has actually heard the word). Used for grading gate.
const AUDIO_PLAYED_MS = 1200;

const _MASTERY_WEIGHT = {
  leech: 3.0, learning: 2.0, reviewing: 1.5, new: 1.5, mature: 1.0,
};

function _pickWeighted(pool) {
  const weights = pool.map(c => _MASTERY_WEIGHT[c.mastery_state] || 1.0);
  const total = weights.reduce((a, b) => a + b, 0);
  let r = Math.random() * total;
  for (let i = 0; i < pool.length; i += 1) {
    r -= weights[i];
    if (r <= 0) return pool[i];
  }
  return pool[pool.length - 1];
}

// ── Tip text for bubble hover ──────────────────────────────────────────
function _bubbleTip(card) {
  const reading = card.reading && card.reading !== card.surface ? card.reading : '';
  const gloss = (card.glosses || [])[0] || '';
  const parts = [];
  if (reading) parts.push(reading);
  if (gloss) parts.push(gloss);
  return parts.join(' · ') || card.surface;
}

export async function launchBubblePop({ lang, voice, focusWords = [] }) {
  const [pool, bests] = await Promise.all([
    fetchGamePool(lang, 30, 'drill', focusWords),
    fetchBestScores(lang),
  ]);
  if (pool.length < 4) {
    return makeEmptyOverlay({
      palette: 'cyan', emoji: '🫧',
      message: 'You need at least 4 words in your queue to play.',
      hint: 'Click words while you browse — or seed common words from the chip menu.',
    });
  }
  const best = (bests.bubble_pop && bests.bubble_pop.best) || 0;

  const { overlay, close, addCleanup } = makeGameOverlay({
    id: 'lg-bubble-pop', palette: 'cyan', title: 'Bubble Pop',
  });

  overlay.innerHTML = `
    <div class="lg-game lg-bp">
      <header class="lg-hud">
        <button type="button" class="lg-close" aria-label="Close">×</button>
        <div class="lg-bp-target">
          <div class="lg-hud-label">listen for</div>
          <div class="lg-bp-prompt-audio" id="lg-bp-prompt" aria-live="polite">…</div>
          <div class="lg-bp-gloss-reveal" id="lg-bp-gloss" hidden></div>
        </div>
        <div class="lg-bp-combo" id="lg-bp-combo" hidden>
          <span class="lg-bp-combo-x">×</span><span id="lg-bp-combo-n">2</span>
        </div>
        <div class="lg-hud-stats">
          <div><span class="lg-hud-label">score</span> <span id="lg-bp-score">0000</span></div>
          ${best ? `<div><span class="lg-hud-label">best</span> <span class="lg-bp-best">${fmtScore(best)}</span></div>` : ''}
          <div><span class="lg-hud-label">time</span> <span id="lg-bp-time">60</span></div>
          <div id="lg-bp-lives">❤❤❤</div>
          <button type="button" class="lg-bp-pause" id="lg-bp-pause" title="Pause (P)">⏸</button>
        </div>
      </header>
      <div class="lg-bp-arena" id="lg-bp-arena">
        <div class="lg-bp-water-layer lg-bp-water-back" aria-hidden="true"></div>
        <div class="lg-bp-water-layer lg-bp-water-mid" aria-hidden="true"></div>
        <div class="lg-bp-sonar" id="lg-bp-sonar" hidden aria-hidden="true">
          <span></span><span></span><span></span>
        </div>
        <div class="lg-bp-fx-layer" id="lg-bp-fx" aria-hidden="true"></div>
      </div>
      <div class="lg-bp-paused" id="lg-bp-paused" hidden>
        <div class="lg-bp-paused-title">Paused</div>
        <button type="button" class="btn btn-primary lg-bp-resume" id="lg-bp-resume">Resume (P)</button>
      </div>
      <div class="lg-end" id="lg-bp-end" hidden>
        <div class="lg-end-title">Round complete</div>
        <div id="lg-bp-review-section"></div>
        <div class="lg-end-stats" id="lg-bp-end-stats"></div>
        <div class="lg-end-actions">
          <button type="button" class="btn btn-primary" id="lg-bp-replay">Play again</button>
          <button type="button" class="btn btn-ghost" id="lg-bp-quit">Done</button>
        </div>
      </div>
    </div>`;

  const arena = overlay.querySelector('#lg-bp-arena');
  const fxLayer = overlay.querySelector('#lg-bp-fx');
  const sonarEl = overlay.querySelector('#lg-bp-sonar');
  const promptEl = overlay.querySelector('#lg-bp-prompt');
  const glossEl = overlay.querySelector('#lg-bp-gloss');
  const scoreEl = overlay.querySelector('#lg-bp-score');
  const timeEl = overlay.querySelector('#lg-bp-time');
  const livesEl = overlay.querySelector('#lg-bp-lives');
  const comboEl = overlay.querySelector('#lg-bp-combo');
  const comboNEl = overlay.querySelector('#lg-bp-combo-n');
  const endPanel = overlay.querySelector('#lg-bp-end');
  const pausedPanel = overlay.querySelector('#lg-bp-paused');
  const pauseBtn = overlay.querySelector('#lg-bp-pause');
  const reviewSection = overlay.querySelector('#lg-bp-review-section');

  overlay.querySelector('.lg-close').addEventListener('click', () => { stop(); close(); });

  let score = 0;
  let lives = 3;
  let bubbleSerial = 0;
  let active = [];   // { id, card, el, spawnedAt, lifespan }
  let targetCard = null;
  let lastSpawn = 0;
  let lastTarget = 0;
  let roundStart = 0;
  let raf = null;
  let running = false;
  let paused = false;
  let pausedAt = 0;
  let combo = 0;
  let maxCombo = 0;
  let popped = 0;
  let missed = 0;

  // Per-target tracking for effort-aware grading:
  // { card, audioPlayedAt, hintsUsed, correct, attempts, ms }
  let _curTarget = null;

  // For grading: cards where TTS actually played (grading gate).
  // Map: word_id → { card, correct, attempts, hintsUsed, ms }
  const _gradeLog = new Map();

  const missedTargets = [];  // cards we track for review

  // Reader popover attached to the review section after round ends
  let _reviewReader = null;

  // ── Voice availability check ──────────────────────────────────────
  // We resolve this once at launch; if voice is null we degrade
  // immediately rather than failing silently on first target.
  const _hasVoice = !!voice;
  if (!_hasVoice) {
    // Defer notice until end-panel is visible; game still runs text-only
    // We'll show the notice once the round ends / in the HUD as a subtle warn.
    showNotice(overlay.querySelector('.lg-game'), 'No voice — gloss shown immediately', { kind: 'warn' });
  }

  function difficulty(t) {
    return Math.max(0, Math.min(1, (t - roundStart) / ROUND_MS));
  }

  function currentSpawnMs(t) {
    const d = difficulty(t);
    return SPAWN_MS_START + (SPAWN_MS_END - SPAWN_MS_START) * d;
  }

  function currentLifespan(t) {
    const d = difficulty(t);
    const base = LIFESPAN_MS_START + (LIFESPAN_MS_END - LIFESPAN_MS_START) * d;
    return base + (Math.random() - 0.5) * 1500;
  }

  const _MASTERY_HUE = { leech: 0, learning: 30, new: 60, reviewing: 180, mature: 260 };

  // ── Semantic distractor seeding (#2) ──────────────────────────────
  // Count how many active non-target bubbles share the target's POS.
  // We need at least 2 at high difficulty. When re-picking the target
  // we use this to prefer targets that already have semantic peers on screen.
  function _posMatchCount(targetPos) {
    if (!targetPos) return 0;
    return active.filter(b =>
      b.card.word_id !== (targetCard && targetCard.word_id)
      && b.card.pos === targetPos
    ).length;
  }

  function _masterySigil(card) {
    switch (card.mastery_state) {
      case 'mature': return 'Bloom';
      case 'reviewing': return 'Hold';
      case 'learning': return 'Grow';
      case 'leech': return 'Care';
      case 'new':
      default: return 'New';
    }
  }

  function _bubbleMarkup(card) {
    const surface = escapeHtml(card.surface || '');
    const reading = card.reading && card.reading !== card.surface
      ? `<span class="lg-bp-bubble-reading">${escapeHtml(card.reading)}</span>`
      : '';
    return `
      <span class="lg-bp-bubble-glass" aria-hidden="true"></span>
      <span class="lg-bp-bubble-lens" aria-hidden="true"></span>
      <span class="lg-bp-bubble-word"><span>${surface}</span>${reading}</span>
      <span class="lg-bp-bubble-sigil">${escapeHtml(_masterySigil(card))}</span>`;
  }

  function spawnCardBubble(card, t, { semantic = false } = {}) {
    const id = ++bubbleSerial;
    const el = document.createElement('div');
    const mastery = card.mastery_state || 'new';
    el.className = `lg-bp-bubble lg-bp-bubble-${mastery}${semantic ? ' lg-bp-bubble-semantic' : ''}`;
    el.dataset.id = String(id);
    el.dataset.wordId = card.word_id;
    el.dataset.mastery = mastery;
    el.dataset.pos = card.pos || '';
    el.dataset.tip = _bubbleTip(card);
    el.tabIndex = 0;
    el.setAttribute('role', 'button');
    el.setAttribute('aria-label', `Pop ${card.surface}`);
    el.innerHTML = _bubbleMarkup(card);

    const xPct = 8 + Math.random() * 84;
    const hue = _MASTERY_HUE[card.mastery_state] ?? Math.floor(Math.random() * 360);
    const size = 78 + Math.random() * 30;
    const drift = (Math.random() - 0.5) * 30;
    const wobble = 2.5 + Math.random() * 2.5;
    const lifespan = currentLifespan(t);

    el.style.setProperty('--bp-hue', String(hue));
    el.style.setProperty('--bp-drift', `${drift}px`);
    el.style.setProperty('--bp-size', `${size}px`);
    el.style.setProperty('--bp-wobble', `${wobble}s`);
    el.style.left = `${xPct}%`;
    el.style.animation = `lg-bp-rise ${lifespan}ms linear forwards, lg-bp-wobble ${wobble}s ease-in-out infinite`;

    el.addEventListener('click', () => onPop(id, card, el));
    el.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        onPop(id, card, el);
      }
    });
    arena.appendChild(el);

    active.push({ id, card, el, spawnedAt: t, lifespan });
    return { id, card, el };
  }

  function spawnBubble(t) {
    spawnCardBubble(_pickWeighted(pool), t);
  }

  // ── Audio-first prompt (#1) ────────────────────────────────────────
  // Show: 🔊 listen… [Hint]   — hide the gloss until audio finishes or hint tapped.
  // After audio completes (or _hasVoice=false): show gloss.
  // Replay button always visible once a target is active.

  function _clearPrompt() {
    promptEl.textContent = '…';
    promptEl.classList.remove('lg-bp-fresh');
    glossEl.hidden = true;
    glossEl.textContent = '';
    _setSonar(null);
  }

  async function _setTarget(card) {
    targetCard = card;
    const gloss = (card.glosses || [])[0] || '';

    _setSonar(card);

    // Track per-target effort data
    _curTarget = {
      card,
      audioPlayedAt: null,    // timestamp when TTS actually started
      hintsUsed: 0,
      correct: false,
      attempts: 0,
      popMs: null,             // ms from audioPlayedAt → correct pop
    };

    if (!_hasVoice || !gloss) {
      // No voice: show gloss immediately
      promptEl.innerHTML = `${escapeHtml(gloss || card.surface)}`;
      promptEl.classList.add('lg-bp-fresh');
      setTimeout(() => promptEl.classList.remove('lg-bp-fresh'), 400);
      glossEl.hidden = true;
      _curTarget.audioPlayedAt = null; // never played — won't grade
      return;
    }

    // Voice available: show audio prompt + hint button
    const hintBtn = `<button type="button" class="lg-bp-hint-btn" id="lg-bp-hint">Hint</button>`;
    const replayBtn = `<button type="button" class="lg-bp-replay-btn" id="lg-bp-replay-audio" title="Replay audio">🔊</button>`;
    promptEl.innerHTML = `🔊 <span style="font-size:15px;color:var(--text-muted,#9aa0aa)">listen…</span> ${hintBtn} ${replayBtn}`;
    promptEl.classList.add('lg-bp-fresh');
    setTimeout(() => promptEl.classList.remove('lg-bp-fresh'), 400);
    glossEl.hidden = true;

    // Wire hint button
    const hintBtnEl = promptEl.querySelector('#lg-bp-hint');
    if (hintBtnEl) {
      hintBtnEl.addEventListener('click', () => {
        if (_curTarget) _curTarget.hintsUsed = (_curTarget.hintsUsed || 0) + 1;
        glossEl.hidden = false;
        glossEl.textContent = gloss;
        hintBtnEl.disabled = true;
        hintBtnEl.textContent = 'hint used';
      });
    }

    // Wire replay button
    const replayAudioEl = promptEl.querySelector('#lg-bp-replay-audio');
    if (replayAudioEl) {
      replayAudioEl.addEventListener('click', () => {
        if (_curTarget) _curTarget.hintsUsed = (_curTarget.hintsUsed || 0) + 0.5; // soft friction
        _playTargetAudio(card, /* isReplay= */ true);
      });
    }

    // Play TTS; reveal gloss after audio ends
    await _playTargetAudio(card, false);
  }

  async function _playTargetAudio(card, isReplay) {
    if (!_hasVoice || paused) return;
    const spoken = card.reading || card.surface;
    const audioObj = await speakWord(spoken, voice);

    // Mark when audio started for grading gate
    const now = performance.now();
    if (_curTarget && _curTarget.card.word_id === card.word_id) {
      if (!_curTarget.audioPlayedAt) _curTarget.audioPlayedAt = now;
      if (isReplay) _curTarget.hintsUsed = (_curTarget.hintsUsed || 0) + 0.5;
    }

    // After audio finishes (or 2.5s timeout), reveal gloss
    const gloss = (card.glosses || [])[0] || '';
    if (gloss) {
      const revealAfter = (audioObj instanceof HTMLAudioElement)
        ? null   // wait for 'ended' event
        : 2000;  // browser TTS or no audio object — fallback delay

      if (audioObj instanceof HTMLAudioElement) {
        audioObj.addEventListener('ended', () => {
          // Only reveal if this is still the active target
          if (targetCard && targetCard.word_id === card.word_id) {
            glossEl.hidden = false;
            glossEl.textContent = gloss;
          }
        }, { once: true });
      } else {
        setTimeout(() => {
          if (targetCard && targetCard.word_id === card.word_id) {
            glossEl.hidden = false;
            glossEl.textContent = gloss;
          }
        }, revealAfter || 2000);
      }
    }
  }

  function multiplier() {
    if (combo >= 5) return 3;
    if (combo >= 3) return 2;
    return 1;
  }

  function showCombo() {
    if (combo >= 3) {
      comboEl.hidden = false;
      comboNEl.textContent = String(multiplier());
      comboEl.classList.add('lg-bp-combo-flash');
      setTimeout(() => comboEl.classList.remove('lg-bp-combo-flash'), 220);
    } else {
      comboEl.hidden = true;
    }
  }

  function _setSonar(card) {
    if (!sonarEl) return;
    if (!card) {
      sonarEl.hidden = true;
      sonarEl.removeAttribute('data-mastery');
      return;
    }
    sonarEl.hidden = false;
    sonarEl.dataset.mastery = card.mastery_state || 'new';
  }

  function _showFloat(text, anchor, kind = 'good') {
    if (!fxLayer) return;
    const fx = document.createElement('div');
    fx.className = `lg-bp-float lg-bp-float-${kind}`;
    fx.textContent = text;
    const a = arena.getBoundingClientRect();
    const r = anchor && anchor.getBoundingClientRect ? anchor.getBoundingClientRect() : null;
    const x = r ? (r.left + r.width / 2 - a.left) : (a.width / 2);
    const y = r ? (r.top + r.height / 2 - a.top) : (a.height / 2);
    fx.style.left = `${Math.max(16, Math.min(a.width - 16, x))}px`;
    fx.style.top = `${Math.max(16, Math.min(a.height - 16, y))}px`;
    fxLayer.appendChild(fx);
    setTimeout(() => fx.remove(), 900);
  }

  function _arenaPulse(kind) {
    arena.classList.remove('lg-bp-arena-good', 'lg-bp-arena-bad', 'lg-bp-arena-miss');
    arena.classList.add(`lg-bp-arena-${kind}`);
    setTimeout(() => arena.classList.remove(`lg-bp-arena-${kind}`), 360);
  }

  function _recordGrade(cur, correct) {
    if (!cur || !cur.card) return;
    const card = cur.card;
    if (card.in_queue === false) return;
    // Only record grading intent if audio actually played for long enough
    const audioReady = cur.audioPlayedAt !== null
      && (performance.now() - cur.audioPlayedAt) >= AUDIO_PLAYED_MS;
    if (!audioReady && !_hasVoice) {
      // Voice unavailable — still grade (gloss was shown)
    } else if (!audioReady) {
      return; // TTS never completed playing — don't grade
    }
    const existing = _gradeLog.get(card.word_id);
    if (!existing) {
      _gradeLog.set(card.word_id, {
        card,
        correct,
        attempts: cur.attempts || (correct ? 1 : 1),
        hintsUsed: cur.hintsUsed || 0,
        ms: cur.popMs || 0,
      });
    } else {
      // Already logged: if we now got it correct, upgrade; merge friction
      if (correct && !existing.correct) {
        existing.correct = true;
        existing.attempts = (existing.attempts || 1) + (cur.attempts || 0);
        existing.hintsUsed += cur.hintsUsed || 0;
      } else if (!correct) {
        existing.attempts = (existing.attempts || 1) + 1;
        existing.hintsUsed += cur.hintsUsed || 0;
      }
    }
  }

  function onPop(id, card, el) {
    if (!running || paused) return;
    if (!targetCard) return;

    if (_curTarget) _curTarget.attempts = (_curTarget.attempts || 0) + 1;

    if (card.word_id === targetCard.word_id) {
      // Correct pop
      if (_curTarget) {
        _curTarget.correct = true;
        const now = performance.now();
        _curTarget.popMs = _curTarget.audioPlayedAt
          ? Math.max(0, now - _curTarget.audioPlayedAt)
          : 0;
        _recordGrade(_curTarget, true);
      }

      combo += 1;
      maxCombo = Math.max(maxCombo, combo);
      const points = 10 * multiplier();
      score += points;
      popped += 1;
      scoreEl.textContent = fmtScore(score);
      _showFloat(`+${points}`, el, 'good');
      _arenaPulse('good');
      el.classList.add('lg-bp-pop');
      burstAt(el, `hsl(${el.style.getPropertyValue('--bp-hue')}, 80%, 70%)`);
      setTimeout(() => el.remove(), 350);
      active = active.filter(b => b.id !== id);
      targetCard = null;
      _curTarget = null;
      _setSonar(null);
      promptEl.innerHTML = combo >= 3 ? `✓ ×${multiplier()}` : '✓';
      glossEl.hidden = true;
      lastTarget = performance.now();
      showCombo();
    } else {
      // Wrong bubble
      if (_curTarget) {
        // Wrong tap counts as attempt but we only record grade on target escape
      }
      combo = 0;
      score = Math.max(0, score - 2);
      scoreEl.textContent = fmtScore(score);
      _showFloat('-2', el, 'bad');
      _arenaPulse('bad');
      el.classList.add('lg-bp-wrong');
      setTimeout(() => el.classList.remove('lg-bp-wrong'), 400);
      showCombo();
    }
  }

  function loseLife() {
    // Log a miss grade for the escaped target (if audio was played)
    if (_curTarget) {
      _recordGrade(_curTarget, false);
      missedTargets.push({ ..._curTarget.card });
    }
    combo = 0;
    showCombo();
    lives -= 1;
    missed += 1;
    _showFloat('miss', null, 'miss');
    _arenaPulse('miss');
    livesEl.textContent = '❤'.repeat(lives) + '🤍'.repeat(3 - lives);
    targetCard = null;
    _curTarget = null;
    _clearPrompt();
    if (lives <= 0) endRound();
  }

  // ── Semantic distractor enforcement (#2) ──────────────────────────
  // If difficulty > 0.4 and active has a target with a pos tag,
  // ensure ≥2 non-target bubbles share that POS; spawn extras if needed.
  function _ensureSemanticDistractors(t, targetPos) {
    if (!targetPos) return;
    const d = difficulty(t);
    if (d < 0.4) return;
    const needed = 2;
    const current = _posMatchCount(targetPos);
    if (current >= needed) return;
    // Find eligible pool cards with same POS, not already on screen
    const onScreen = new Set(active.map(b => b.card.word_id));
    const candidates = pool.filter(c =>
      c.pos === targetPos
      && c.word_id !== (targetCard && targetCard.word_id)
      && !onScreen.has(c.word_id)
    );
    const toSpawn = needed - current;
    for (let i = 0; i < toSpawn && i < candidates.length; i += 1) {
      spawnCardBubble(candidates[i], t, { semantic: true });
    }
  }

  function pickTarget(t) {
    const eligible = active.filter(b => {
      const age = performance.now() - b.spawnedAt;
      const hasGloss = ((b.card.glosses || [])[0] || '').trim().length > 0;
      return hasGloss && age > 700 && age < b.lifespan - 1800;
    });
    if (eligible.length === 0) return;

    // At high difficulty, prefer targets whose POS already has peers on screen
    const d = difficulty(t || performance.now());
    let choice;
    if (d > 0.4) {
      const withPeers = eligible.filter(b => b.card.pos && _posMatchCount(b.card.pos) >= 1);
      choice = withPeers.length ? pickOne(withPeers) : pickOne(eligible);
    } else {
      choice = pickOne(eligible);
    }

    const card = choice.card;
    const tNow = t || performance.now();

    // Ensure semantic distractors after picking (#2)
    _ensureSemanticDistractors(tNow, card.pos);

    // Audio-first target presentation (#1)
    _setTarget(card);
  }

  function loop(t) {
    if (!running || paused) return;
    const elapsed = t - roundStart;
    const remaining = Math.max(0, Math.ceil((ROUND_MS - elapsed) / 1000));
    timeEl.textContent = String(remaining);
    if (elapsed >= ROUND_MS) { endRound(); return; }

    if (t - lastSpawn > currentSpawnMs(t)) {
      lastSpawn = t;
      spawnBubble(t);
    }

    active = active.filter(b => {
      if (t - b.spawnedAt > b.lifespan) {
        if (targetCard && b.card.word_id === targetCard.word_id) {
          loseLife();
        }
        b.el.remove();
        return false;
      }
      return true;
    });

    if (!targetCard && t - lastTarget > TARGET_GAP_MS && active.length > 2) {
      lastTarget = t;
      pickTarget(t);
    }

    raf = requestAnimationFrame(loop);
  }

  function start() {
    running = true;
    paused = false;
    pausedPanel.hidden = true;
    pauseBtn.disabled = false;
    roundStart = performance.now();
    lastSpawn = 0;
    lastTarget = performance.now();
    raf = requestAnimationFrame(loop);
  }

  function pause() {
    if (!running || paused) return;
    paused = true;
    pausedAt = performance.now();
    if (raf) cancelAnimationFrame(raf);
    stopAudio();
    arena.querySelectorAll('.lg-bp-bubble').forEach(b => {
      b.style.animationPlayState = 'paused';
    });
    pausedPanel.hidden = false;
  }

  function resume() {
    if (!paused) return;
    const pausedFor = performance.now() - pausedAt;
    roundStart += pausedFor;
    lastSpawn += pausedFor;
    lastTarget += pausedFor;
    active.forEach(b => { b.spawnedAt += pausedFor; });
    if (_curTarget && _curTarget.audioPlayedAt) {
      _curTarget.audioPlayedAt += pausedFor;
    }
    arena.querySelectorAll('.lg-bp-bubble').forEach(b => {
      b.style.animationPlayState = 'running';
    });
    paused = false;
    pausedPanel.hidden = true;
    raf = requestAnimationFrame(loop);
  }

  const onPauseKey = (e) => {
    if (!running) return;
    if (e.key === 'p' || e.key === 'P') { e.preventDefault(); paused ? resume() : pause(); }
  };
  pauseBtn.addEventListener('click', () => { paused ? resume() : pause(); });
  overlay.querySelector('#lg-bp-resume').addEventListener('click', resume);
  document.addEventListener('keydown', onPauseKey);
  addCleanup(() => document.removeEventListener('keydown', onPauseKey));

  function stop() {
    running = false;
    paused = false;
    if (raf) cancelAnimationFrame(raf);
    active.forEach(b => b.el.remove());
    active = [];
    pausedPanel.hidden = true;
    pauseBtn.disabled = true;
    stopAudio();
    if (_reviewReader) { _reviewReader.destroy(); _reviewReader = null; }
  }

  // ── Post-round missed-words review (#3) ──────────────────────────
  async function _buildReview() {
    if (missedTargets.length === 0) { reviewSection.hidden = true; return; }
    reviewSection.innerHTML = `
      <div class="lg-bp-review">
        <div class="lg-bp-review-title">These slipped away</div>
        <ul class="lg-bp-review-list" id="lg-bp-review-list">
          <li style="padding:10px 14px;color:var(--text-muted,#9aa0aa);font-style:italic">Loading…</li>
        </ul>
      </div>`;
    reviewSection.hidden = false;

    const listEl = reviewSection.querySelector('#lg-bp-review-list');

    // Fetch example sentences for each missed word (in parallel, 1 per word)
    const sentenceMap = new Map();
    await Promise.all(missedTargets.slice(0, 6).map(async (card) => {
      try {
        const sents = await fetchSentences(lang, 1, card.surface);
        const s = (sents || [])[0];
        if (s && s.text) sentenceMap.set(card.word_id, s.text);
      } catch { /* graceful */ }
    }));

    // Build items
    const items = [];
    for (const card of missedTargets.slice(0, 6)) {
      const surface = escapeHtml(card.surface || '');
      const reading = card.reading && card.reading !== card.surface
        ? escapeHtml(card.reading) : '';
      const gloss = escapeHtml((card.glosses || [])[0] || '');
      const exampleText = sentenceMap.get(card.word_id) || '';

      // We'll build example HTML separately with makeClickableHTML
      items.push({ card, surface, reading, gloss, exampleText });
    }

    // Render items — example sentences get makeClickableHTML
    listEl.innerHTML = '';
    for (const { surface, reading, gloss, exampleText, card } of items) {
      const li = document.createElement('li');
      li.className = 'lg-bp-review-item';
      li.innerHTML = `
        <div class="lg-bp-review-surface">${surface}</div>
        ${reading ? `<div class="lg-bp-review-reading">${reading}</div>` : '<div></div>'}
        <div class="lg-bp-review-gloss">${gloss}</div>
        ${exampleText ? `<div class="lg-bp-review-sentence" data-full-text="${escapeHtml(exampleText)}" id="lg-bp-ex-${escapeHtml(card.word_id)}"></div>` : ''}`;
      listEl.appendChild(li);

      // Render clickable example sentence
      if (exampleText) {
        const sentEl = li.querySelector(`#lg-bp-ex-${escapeHtml(card.word_id)}`);
        if (sentEl) {
          try {
            sentEl.innerHTML = await makeClickableHTML(exampleText, lang);
            sentEl.dataset.fullText = exampleText;
          } catch {
            sentEl.textContent = exampleText;
          }
        }
      }
    }

    // Attach one shared reader popover for all L2 tokens in the review strip (#3)
    if (_reviewReader) { _reviewReader.destroy(); _reviewReader = null; }
    _reviewReader = attachReaderPopover(listEl, {
      lang, voice,
      onWordAdded: () => { /* silent success — card is added to queue */ },
    });
    addCleanup(() => { if (_reviewReader) { _reviewReader.destroy(); _reviewReader = null; } });
  }

  // ── Grading (#4) ──────────────────────────────────────────────────
  async function _applyGrades() {
    const seenIds = new Set();
    const toGrade = [];
    const toFail = [];

    for (const [wid, entry] of _gradeLog.entries()) {
      if (seenIds.has(wid)) continue;
      seenIds.add(wid);
      if (entry.card.in_queue === false) continue;
      if (entry.correct) {
        if (toGrade.length < 5) toGrade.push(entry);
      } else {
        if (toFail.length < 5) toFail.push(entry);
      }
    }

    await Promise.all([
      ...toGrade.map(entry => {
        const grade = gradeForEffort({
          correct: true,
          attempts: entry.attempts || 1,
          hintsUsed: entry.hintsUsed || 0,
          ms: entry.ms || 0,
        });
        return gradeCard(lang, entry.card.word_id, grade);
      }),
      ...toFail.map(entry => {
        return gradeCard(lang, entry.card.word_id, 1);
      }),
    ]);

    return { graded: toGrade.length, failed: toFail.length };
  }

  async function endRound() {
    stop();

    // Build review strip before showing end panel
    await _buildReview();

    const { graded } = await _applyGrades();

    const beatBest = score > best;
    overlay.querySelector('#lg-bp-end-stats').innerHTML = `
      <div class="lg-end-stat"><div class="lg-end-stat-n">${fmtScore(score)}</div><div>score${beatBest ? ' · new best!' : ''}</div></div>
      <div class="lg-end-stat"><div class="lg-end-stat-n">${popped}</div><div>popped</div></div>
      <div class="lg-end-stat"><div class="lg-end-stat-n">×${maxCombo || 0}</div><div>best combo</div></div>
      <div class="lg-end-stat"><div class="lg-end-stat-n">${graded}</div><div>graded</div></div>`;
    endPanel.hidden = false;
    recordResult({
      game_id: 'bubble_pop', lang,
      score, words_played: popped + missed, words_correct: popped,
      duration_sec: Math.round((performance.now() - roundStart) / 1000),
      metadata: { max_combo: maxCombo, missed, lives_lost: 3 - lives },
    });
  }

  overlay.querySelector('#lg-bp-replay').addEventListener('click', () => {
    // Destroy review reader before reset
    if (_reviewReader) { _reviewReader.destroy(); _reviewReader = null; }
    reviewSection.innerHTML = '';
    score = 0; lives = 3; combo = 0; maxCombo = 0; popped = 0; missed = 0;
    _gradeLog.clear();
    missedTargets.length = 0;
    targetCard = null;
    _curTarget = null;
    scoreEl.textContent = fmtScore(0);
    livesEl.textContent = '❤❤❤';
    timeEl.textContent = '60';
    comboEl.hidden = true;
    endPanel.hidden = true;
    _clearPrompt();
    start();
  });
  overlay.querySelector('#lg-bp-quit').addEventListener('click', () => close());

  addCleanup(() => stop());
  start();
}
