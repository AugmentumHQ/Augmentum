/**
 * Vocab Quest — adventure with vocab gates.
 *
 * A LLM-driven mini text adventure (5-7 scenes). Each scene renders a
 * natural L2 sentence using the target word plus an English context line.
 * The player must identify what the highlighted L2 word means — real
 * reading comprehension, not a labelling exercise.
 *
 * Closes Krashen's "compelling input" gap: SRS knows what you should
 * see; story knows why you should care.
 *
 * v2 upgrades:
 *  - Scene generates a genuine L2 sentence; question is "What does «word» mean here?"
 *  - makeClickableHTML + attachReaderPopover on the L2 sentence (tap any word)
 *  - Tap-to-hear 🔊 on each answer button (stopPropagation so it doesn't pick)
 *  - Scene continuity: last 1-2 scenes fed into the next prompt
 *  - showWarming + timeout + static fallback for cold/offline models
 *  - gradeForEffort (time-to-answer, attempts) instead of flat 3/1
 *  - Offer addWord for wrong targets that aren't already in the queue
 */

import {
  escapeHtml, fetchGamePool, llmChatStream, pickN, makeGameOverlay,
  makeEmptyOverlay, gradeCard, burstAt, speakWord, addWord, recordResult,
  makeClickableHTML, attachReaderPopover, gradeForEffort, showWarming, showNotice,
} from './_common.js';

const SCENES = 6;
// How long (ms) to wait for the LLM before using the static fallback.
const LLM_TIMEOUT_MS = 30000;

// ── Quest-scoped CSS (guarded, injected once) ───────────────────────
function _ensureQuestStyles() {
  if (document.getElementById('lg-quest-styles')) return;
  const css = `
/* ── vocab-quest option buttons ── */
.lg-vq-opt-row {
  display: grid; grid-template-columns: minmax(0, 1fr) 42px; gap: 8px; min-width: 0;
}
.lg-vq-opt {
  display: flex; flex-direction: column; gap: 2px; align-items: flex-start;
  min-width: 0; width: 100%;
  padding: 10px 14px; border-radius: 10px; border: 1.5px solid var(--border, rgba(255,255,255,.14));
  background: var(--bg-elevated, #1b1d24); color: var(--text-primary, #e8e8ea);
  cursor: pointer; font-size: 15px; transition: border-color .15s, background .15s;
  position: relative;
}
.lg-vq-opt:not([disabled]):hover { border-color: var(--accent, #6ea8fe); background: color-mix(in srgb, var(--accent, #6ea8fe) 10%, var(--bg-elevated, #1b1d24)); }
.lg-vq-opt[disabled] { opacity: .7; cursor: not-allowed; }
.lg-vq-opt-right { border-color: #7fbf63 !important; background: color-mix(in srgb, #7fbf63 18%, var(--bg-elevated, #1b1d24)) !important; }
.lg-vq-opt-wrong  { border-color: #e05c5c !important; background: color-mix(in srgb, #e05c5c 14%, var(--bg-elevated, #1b1d24)) !important; }
.lg-vq-opt-surface  { font-weight: 700; }
.lg-vq-opt-reading  { color: var(--text-muted, #9aa0aa); font-size: 13px; }
.lg-vq-opt-kbd { margin-left: auto; font-size: 11px; color: var(--text-muted, #9aa0aa);
  background: var(--bg-surface, #14151a); border: 1px solid var(--border, rgba(255,255,255,.12));
  border-radius: 4px; padding: 1px 5px; }
.lg-vq-opt-speak {
  display: grid; place-items: center;
  width: 42px; min-height: 100%;
  border-radius: 8px;
  border: 1px solid var(--border, rgba(255,255,255,.14));
  background: rgba(255,255,255,.05);
  cursor: pointer; font-size: 15px; color: var(--text-primary, #e8e8ea);
  transition: border-color .15s, background .15s;
}
.lg-vq-opt-speak:hover { border-color: var(--accent, #6ea8fe); background: color-mix(in srgb, var(--accent, #6ea8fe) 10%, transparent); }
/* ── L2 sentence display ── */
.lg-vq-l2-sentence {
  font-size: 18px; line-height: 1.6; margin: 8px 0 4px;
  padding: 10px 14px; border-radius: 8px;
  background: color-mix(in srgb, var(--accent, #6ea8fe) 8%, transparent);
  border-left: 3px solid var(--accent, #6ea8fe);
}
.lg-vq-l2-target {
  font-weight: 700; color: var(--accent, #6ea8fe);
  text-decoration: underline; text-decoration-style: dotted;
}
.lg-vq-context-line {
  font-size: 13.5px; color: var(--text-muted, #9aa0aa); margin: 4px 0 10px;
  font-style: italic;
}
.lg-vq-prompt { font-size: 15px; font-weight: 600; margin: 8px 0 12px; }
/* ── offer-add bar (wrong target not in queue) ── */
.lg-vq-add-bar {
  display: flex; align-items: center; gap: 10px; margin-top: 10px;
  padding: 8px 12px; border-radius: 8px;
  background: color-mix(in srgb, var(--warning, #e0a800) 14%, transparent);
  font-size: 13px; color: var(--text-primary, #e8e8ea);
}
.lg-vq-add-bar button { font-size: 12px; }
`;
  const style = document.createElement('style');
  style.id = 'lg-quest-styles';
  style.textContent = css;
  document.head.appendChild(style);
}

// ── Static fallback scenes (L2 word present so the mechanic holds offline) ──
// Keys are rough semantic categories; we pick one at random.
const _FALLBACK_TEMPLATES = [
  { emoji: '🌲', context: 'You reach a forest crossroads and consult a sign.',
    l2Template: (surface) => `The sign says: "${escapeHtml(surface)}."`,
    english: 'A traveller must read the sign to choose the right path.' },
  { emoji: '🏚', context: 'The old innkeeper greets you at the door.',
    l2Template: (surface) => `She whispers, "${escapeHtml(surface)}."`,
    english: 'Her words hold the key to finding shelter for the night.' },
  { emoji: '🌊', context: 'A sailor leans over the bow and calls out.',
    l2Template: (surface) => `He shouts: "${escapeHtml(surface)}!"`,
    english: 'The crew waits to hear the word before raising the anchor.' },
];

export async function launchVocabQuest({ lang, voice }) {
  _ensureQuestStyles();

  const pool = await fetchGamePool(lang, 40, 'drill');
  let questStart = performance.now();

  if (pool.length < 6) {
    return makeEmptyOverlay({
      palette: 'forest', emoji: '🗺️',
      message: 'Quest needs at least 6 words to weave a journey.',
      hint: 'The road ahead opens once you have a few words to barter with.',
    });
  }

  const { overlay, close, addCleanup } = makeGameOverlay({
    id: 'lg-quest', palette: 'forest', title: 'Vocab Quest',
  });

  overlay.innerHTML = `
    <div class="lg-game lg-vq">
      <header class="lg-hud">
        <button type="button" class="lg-close" aria-label="Close">×</button>
        <div class="lg-vq-title">Vocab Quest</div>
        <div class="lg-hud-stats">
          <span class="lg-hud-label">scene</span> <span id="lg-vq-scene">1</span>/${SCENES}
          <span class="lg-hud-label">hp</span> <span id="lg-vq-hp">❤❤❤</span>
        </div>
      </header>
      <div class="lg-vq-panel">
        <div class="lg-vq-illustration" id="lg-vq-illo">
          <svg viewBox="0 0 200 200" width="100%" height="100%">
            <circle cx="100" cy="100" r="80" fill="rgba(255,255,255,.04)" stroke="rgba(174,213,129,.4)" stroke-width="1"/>
            <text x="100" y="115" font-size="80" text-anchor="middle">🌲</text>
          </svg>
        </div>
        <div class="lg-vq-scene-text" id="lg-vq-scene-text"><em>setting out…</em></div>
        <div class="lg-vq-prompt" id="lg-vq-prompt">—</div>
        <div class="lg-vq-options" id="lg-vq-options"></div>
      </div>
      <div class="lg-end" id="lg-vq-end" hidden>
        <div class="lg-end-title" id="lg-vq-end-title">— quest complete —</div>
        <div class="lg-end-stats" id="lg-vq-end-stats"></div>
        <div class="lg-end-actions">
          <button type="button" class="btn btn-primary" id="lg-vq-replay">New quest</button>
          <button type="button" class="btn btn-ghost" id="lg-vq-quit">Done</button>
        </div>
      </div>
    </div>`;

  overlay.querySelector('.lg-close').addEventListener('click', () => close());

  const sceneEl    = overlay.querySelector('#lg-vq-scene');
  const sceneTextEl = overlay.querySelector('#lg-vq-scene-text');
  const promptEl   = overlay.querySelector('#lg-vq-prompt');
  const optsEl     = overlay.querySelector('#lg-vq-options');
  const illoEl     = overlay.querySelector('#lg-vq-illo');
  const hpEl       = overlay.querySelector('#lg-vq-hp');
  const endPanel   = overlay.querySelector('#lg-vq-end');

  let scene = 1;
  let hp = 3;
  // journey entries: { surface, outcome:'win'|'lose', contextLine, l2Sentence }
  let journey = [];
  const correct = [];
  const missed  = [];
  let usedTargets = new Set();

  // Active reader popover — torn down at the start of each scene.
  let _activeReader = null;
  addCleanup(() => _activeReader?.destroy());

  const SCENE_EMOJIS = ['🌲', '🏚', '🌊', '🗻', '🌌', '🏛', '🐉', '🔥'];

  // ── LLM scene generation with timeout + fallback ────────────────────
  async function generateScene(target, choices) {
    const gloss = (target.glosses || [])[0] || target.surface;
    const surface = target.reading || target.surface;

    // Feed last 2 journey entries for continuity.
    const recentJourney = journey.slice(-2).map(j =>
      `Scene "${escapeHtml(j.surface)}" (${j.outcome}): ${escapeHtml(j.contextLine || '')}`
    ).join('\n') || '(this is the beginning)';

    const sys = `You are the dungeon master of a short language-learning adventure game. The player is learning ${lang}. Each scene you write must contain EXACTLY:
1. One natural ${lang} sentence that uses the word "${escapeHtml(surface)}" (the target word). It should read naturally in context.
2. One English line of narrative context (1-2 sentences) that places the scene WITHOUT translating the target word.

Format your response as JSON:
{"l2": "<the ${lang} sentence>", "english": "<English narrative context line>"}

Rules:
- The ${lang} sentence MUST naturally contain "${escapeHtml(surface)}" or a grammatically inflected form of it.
- The English context MUST NOT contain the English translation of "${escapeHtml(gloss)}".
- Keep both lines short and vivid — this is an adventure game.
- Output ONLY the JSON object, nothing else.`;

    const u = `Scene ${scene} of ${SCENES}.
Recent story: ${recentJourney}
Target ${lang} word: "${escapeHtml(surface)}" (the player must identify its meaning)
Available distractor surfaces: ${choices.filter(c => c.word_id !== target.word_id).map(c => escapeHtml(c.surface)).join(', ')}

Write the scene JSON now.`;

    // Show warming shimmer while awaiting.
    const stopWarming = showWarming(sceneTextEl, 'Writing the next scene…');

    let raw = '';
    try {
      raw = await Promise.race([
        llmChatStream(
          [{ role: 'system', content: sys }, { role: 'user', content: u }],
          (_d, full) => {
            // Stream progress into the element — not meaningful JSON yet,
            // but lets the user see activity on slow models.
            sceneTextEl.textContent = full.length > 60 ? full.slice(0, 60) + '…' : full;
          },
        ),
        new Promise((res) => setTimeout(() => res(''), LLM_TIMEOUT_MS)),
      ]);
    } catch {
      raw = '';
    }
    stopWarming();

    // Parse the JSON from the response.
    if (raw) {
      const m = raw.match(/\{[\s\S]*\}/);
      if (m) {
        try {
          const parsed = JSON.parse(m[0]);
          if (parsed.l2 && parsed.english) {
            return { l2: parsed.l2, english: parsed.english };
          }
        } catch { /* fall through */ }
      }
    }

    // Static fallback — still contains the target word in an L2 position.
    const fb = _FALLBACK_TEMPLATES[scene % _FALLBACK_TEMPLATES.length];
    showNotice(overlay, 'Model offline or timed out — using a fallback scene.', { kind: 'warn' });
    return {
      l2: fb.l2Template(surface),
      english: fb.english,
    };
  }

  // ── Render a single scene ───────────────────────────────────────────
  async function nextScene() {
    // Tear down any previous reader popover.
    _activeReader?.destroy();
    _activeReader = null;

    sceneEl.textContent = String(scene);
    illoEl.querySelector('text').textContent = SCENE_EMOJIS[(scene - 1) % SCENE_EMOJIS.length];
    sceneTextEl.innerHTML = '<em>setting out…</em>';
    promptEl.textContent = '';
    optsEl.innerHTML = '';

    // Pick four candidate words; avoid reusing targets.
    const choices = pickN(pool, 4);
    const freshChoices = choices.filter(c => !usedTargets.has(c.word_id));
    if (freshChoices.length === 0) usedTargets = new Set();
    const targetPool = freshChoices.length ? freshChoices : choices;
    const target = targetPool[Math.floor(Math.random() * targetPool.length)];
    usedTargets.add(target.word_id);

    const { l2, english } = await generateScene(target, choices);

    // ── Render L2 sentence with clickable tokens ──────────────────────
    const clickableL2 = await makeClickableHTML(l2, lang);
    // Bold/accent the target surface within the clickable HTML.
    // We wrap the target token spans that contain the surface form.
    const targetSurface = escapeHtml(target.reading || target.surface);
    const highlightedL2 = clickableL2.replace(
      new RegExp(`(<span class="lg-tok" data-w="${targetSurface}">)(${targetSurface})(</span>)`, 'g'),
      '$1<strong class="lg-vq-l2-target">$2</strong>$3',
    );

    sceneTextEl.innerHTML = `
      <div class="lg-vq-l2-sentence" data-full-text="${escapeHtml(l2)}">${highlightedL2}</div>
      <div class="lg-vq-context-line">${escapeHtml(english)}</div>`;

    // Attach the reader popover to the whole panel so clicks on any L2 token work.
    _activeReader = attachReaderPopover(overlay.querySelector('.lg-vq-panel'), {
      lang, voice,
      onWordAdded: () => {},
    });

    promptEl.innerHTML = `What does <strong>«${escapeHtml(target.reading || target.surface)}»</strong> mean here?`;

    // Record when the question appeared (for gradeForEffort ms calculation).
    const questionStartMs = performance.now();

    // ── Render answer buttons with tap-to-hear ─────────────────────────
    optsEl.innerHTML = choices.map((c, i) => `
      <div class="lg-vq-opt-row">
        <button type="button" class="lg-vq-opt" data-id="${escapeHtml(c.word_id)}">
          <span class="lg-vq-opt-surface">${escapeHtml(c.surface)}</span>
          <span class="lg-vq-opt-reading">${escapeHtml(c.reading || '')}</span>
          <span class="lg-vq-opt-kbd">${i + 1}</span>
        </button>
        <button type="button" class="lg-vq-opt-speak" aria-label="Hear ${escapeHtml(c.surface)}">&#128266;</button>
      </div>`).join('');

    // Wire speak buttons — stopPropagation so they don't trigger a pick.
    optsEl.querySelectorAll('.lg-vq-opt-speak').forEach((speakBtn, i) => {
      const c = choices[i];
      speakBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        speakWord(c.reading || c.surface, voice);
      });
    });

    optsEl.querySelectorAll('.lg-vq-opt').forEach((btn) => {
      btn.addEventListener('click', () => {
        onChoice(btn, target, choices, l2, questionStartMs);
      });
    });
  }

  // ── Keyboard shortcuts: 1-4 pick an option ─────────────────────────
  const onKey = (e) => {
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
    if (e.key >= '1' && e.key <= '4') {
      const opts = optsEl.querySelectorAll('.lg-vq-opt');
      const btn = opts[Number(e.key) - 1];
      if (btn && !btn.disabled) { e.preventDefault(); btn.click(); }
    }
  };
  document.addEventListener('keydown', onKey);
  addCleanup(() => document.removeEventListener('keydown', onKey));

  // ── Choice handler ──────────────────────────────────────────────────
  function onChoice(btn, target, choices, l2Sentence, questionStartMs) {
    const chosen = choices.find(c => c.word_id === btn.dataset.id);
    optsEl.querySelectorAll('.lg-vq-opt').forEach(b => b.disabled = true);

    const elapsedMs = performance.now() - questionStartMs;

    if (chosen.word_id === target.word_id) {
      btn.classList.add('lg-vq-opt-right');
      burstAt(btn, '#7fbf63');
      correct.push({ card: target, ms: elapsedMs, attempts: 1 });
      journey.push({
        surface: target.surface,
        outcome: 'win',
        contextLine: l2Sentence,
        l2Sentence,
      });
      speakWord(target.reading || target.surface, voice);
      setTimeout(() => {
        scene += 1;
        if (scene > SCENES) endQuest(true);
        else nextScene();
      }, 1300);
    } else {
      btn.classList.add('lg-vq-opt-wrong');
      hp -= 1;
      missed.push({ card: target, ms: elapsedMs });
      hpEl.textContent = '❤'.repeat(Math.max(0, hp)) + '🤍'.repeat(3 - Math.max(0, hp));

      // Reveal correct answer.
      const rightBtn = Array.from(optsEl.querySelectorAll('.lg-vq-opt'))
        .find(b => b.dataset.id === target.word_id);
      rightBtn?.classList.add('lg-vq-opt-right');

      // Offer to add the missed word if it's not already in the queue.
      _maybeOfferAdd(target, optsEl);

      setTimeout(() => {
        if (hp <= 0) {
          journey.push({ surface: target.surface, outcome: 'lose', contextLine: l2Sentence, l2Sentence });
          endQuest(false);
        } else {
          journey.push({ surface: target.surface, outcome: 'lose', contextLine: l2Sentence, l2Sentence });
          scene += 1;
          if (scene > SCENES) endQuest(false);
          else nextScene();
        }
      }, 1800);
    }
  }

  // ── Offer addWord for wrong targets not yet in the queue ────────────
  function _maybeOfferAdd(target, container) {
    if (target.in_queue !== false) return;   // already queued — nothing to offer
    const bar = document.createElement('div');
    bar.className = 'lg-vq-add-bar';
    bar.innerHTML = `
      <span>New word: <strong>${escapeHtml(target.surface)}</strong> — add to your queue?</span>
      <button type="button" class="btn btn-primary" id="lg-vq-addword">+ Add</button>`;
    container.appendChild(bar);
    bar.querySelector('#lg-vq-addword').addEventListener('click', async (e) => {
      const addBtn = e.currentTarget;
      addBtn.disabled = true;
      const ok = await addWord(lang, target.word_id);
      addBtn.textContent = ok ? '✓ Added' : 'Try again';
      if (!ok) addBtn.disabled = false;
    });
  }

  // ── End-of-quest grading + result recording ─────────────────────────
  async function endQuest(victory) {
    _activeReader?.destroy();
    _activeReader = null;

    const seenIds = new Set();
    const toGrade = [];
    for (const entry of correct) {
      if (toGrade.length >= 5) break;
      const c = entry.card;
      if (c.in_queue === false) continue;
      if (seenIds.has(c.word_id)) continue;
      seenIds.add(c.word_id);
      toGrade.push(entry);
    }
    const toFail = [];
    for (const entry of missed) {
      if (toFail.length >= 5) break;
      const c = entry.card;
      if (c.in_queue === false) continue;
      if (seenIds.has(c.word_id)) continue;
      seenIds.add(c.word_id);
      toFail.push(entry);
    }

    await Promise.all([
      ...toGrade.map(entry => {
        const g = gradeForEffort({ correct: true, attempts: 1, ms: entry.ms });
        return gradeCard(lang, entry.card.word_id, g);
      }),
      ...toFail.map(entry => {
        const g = gradeForEffort({ correct: false });
        return gradeCard(lang, entry.card.word_id, g);
      }),
    ]);

    overlay.querySelector('#lg-vq-end-title').textContent =
      victory ? '— quest complete —' : '— the journey ends here —';
    overlay.querySelector('#lg-vq-end-stats').innerHTML = `
      <div class="lg-end-stat"><div class="lg-end-stat-n">${correct.length}/${SCENES}</div><div>scenes cleared</div></div>
      <div class="lg-end-stat"><div class="lg-end-stat-n">${journey.length}</div><div>words spoken</div></div>`;
    endPanel.hidden = false;

    recordResult({
      game_id: 'vocab_quest', lang,
      score: correct.length * 20 + (victory ? 50 : 0),
      words_played: SCENES, words_correct: correct.length,
      duration_sec: Math.round((performance.now() - questStart) / 1000),
      metadata: { victory },
    });
  }

  // ── Replay ──────────────────────────────────────────────────────────
  overlay.querySelector('#lg-vq-replay').addEventListener('click', () => {
    scene = 1; hp = 3; journey = [];
    correct.length = 0; missed.length = 0;
    usedTargets = new Set();
    hpEl.textContent = '❤❤❤';
    endPanel.hidden = true;
    questStart = performance.now();
    nextScene();
  });
  overlay.querySelector('#lg-vq-quit').addEventListener('click', () => close());

  nextScene();
}
