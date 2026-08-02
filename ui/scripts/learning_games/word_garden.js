/**
 * Word Garden — persistent visual collection.
 *
 * Your vocabulary becomes a garden. Every word you've added is a plant.
 * Mastery state drives appearance:
 *   - new        → bare soil mound
 *   - learning   → sprout
 *   - reviewing  → small plant
 *   - mature     → flowering bloom
 *   - leech      → wilting (red tint)
 *   - due        → drooping variant (in_queue && not mature)
 *
 * FIX 2026-06: stopped corrupting the FSRS scheduler by grading from
 * fully-visible cards. Added hidden-recall step: answer/gloss hidden until
 * "Tap to reveal". Grade buttons appear ONLY after reveal.
 * Added due-droop visual, patch-in-place updates, reader popover on example
 * sentences, auto-speak on open (new/learning), and "plant a word" search.
 */

import {
  escapeHtml, fetchGamePool, gradeCard, speakWord, addWord,
  makeGameOverlay, burstAt, gradeForEffort,
  makeClickableHTML, attachReaderPopover,
} from './_common.js';

// ── Garden-scoped styles (injected once) ─────────────────────────────

function _ensureGardenStyles() {
  if (document.getElementById('lg-garden-styles')) return;
  const style = document.createElement('style');
  style.id = 'lg-garden-styles';
  style.textContent = `
/* Due-droop plant variant */
.lg-wg-stage-due .lg-wg-svg {
  transform-origin: 50% 100%;
  transform: rotate(14deg);
  filter: saturate(0.65) brightness(0.85);
  transition: transform 0.4s ease;
}
.lg-wg-plant.lg-wg-stage-due:hover .lg-wg-svg {
  transform: rotate(6deg);
}
/* Due badge on stats */
.lg-wg-stat-due {
  background: rgba(255,160,60,0.18);
  color: hsl(35, 90%, 68%);
  font-weight: 600;
}
/* Recall gate inside detail card */
.lg-wg-recall-gate {
  margin-top: 20px;
}
.lg-wg-reveal-btn {
  display: block; width: 100%;
  padding: 14px;
  background: rgba(127,201,122,0.12);
  border: 1.5px dashed rgba(127,201,122,0.35);
  border-radius: 8px;
  color: rgba(255,255,255,0.8);
  font-size: 14px;
  cursor: pointer;
  transition: background 160ms, border-color 160ms;
}
.lg-wg-reveal-btn:hover {
  background: rgba(127,201,122,0.22);
  border-color: rgba(127,201,122,0.6);
}
.lg-wg-recall-hint {
  font-size: 11px;
  color: rgba(255,255,255,0.4);
  text-align: center;
  margin-top: 8px;
  letter-spacing: 0;
  text-transform: uppercase;
}
/* Example sentence reader area */
.lg-wg-detail-example-reader {
  font-size: 14px;
  color: rgba(255,255,255,0.8);
  margin-top: 12px;
  padding: 10px 12px;
  background: rgba(255,255,255,0.03);
  border-radius: 8px;
  line-height: 1.65;
}
.lg-wg-detail-example-reader .lg-wg-ex-en {
  color: rgba(255,255,255,0.45);
  font-style: italic;
  display: block;
  margin-top: 5px;
  font-size: 12.5px;
}
/* Auto-speak toggle in header */
.lg-wg-autospeak-btn {
  background: none; border: none; cursor: pointer;
  font-size: 14px; padding: 2px 6px;
  color: rgba(255,255,255,0.55);
  border-radius: 6px;
  transition: color 120ms, background 120ms;
}
.lg-wg-autospeak-btn[aria-pressed="true"] {
  color: rgba(127,201,122,0.9);
  background: rgba(127,201,122,0.10);
}
/* Plant-a-word search bar */
.lg-wg-search-bar {
  display: flex; align-items: center; gap: 8px;
  padding: 10px 14px 0;
}
.lg-wg-search-input {
  flex: 1;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 8px;
  color: inherit;
  font-size: 14px;
  padding: 8px 12px;
  outline: none;
  transition: border-color 160ms;
}
.lg-wg-search-input:focus { border-color: rgba(127,201,122,0.5); }
.lg-wg-search-btn {
  padding: 8px 14px;
  background: rgba(127,201,122,0.18);
  border: 1px solid rgba(127,201,122,0.3);
  border-radius: 8px;
  color: hsl(130,50%,72%);
  cursor: pointer;
  font-size: 13px;
  white-space: nowrap;
  transition: background 160ms;
}
.lg-wg-search-btn:hover { background: rgba(127,201,122,0.30); }
.lg-wg-search-results {
  position: absolute; left: 14px; right: 14px; z-index: 200;
  background: rgba(18,26,20,0.97);
  border: 1px solid rgba(127,201,122,0.25);
  border-radius: 8px;
  max-height: 240px; overflow-y: auto;
  box-shadow: 0 10px 32px rgba(0,0,0,0.55);
  animation: lg-rdr-in .12s ease;
}
.lg-wg-search-result-item {
  display: flex; align-items: baseline; gap: 10px;
  padding: 10px 14px;
  cursor: pointer;
  border-bottom: 1px solid rgba(255,255,255,0.06);
  transition: background 120ms;
}
.lg-wg-search-result-item:last-child { border-bottom: none; }
.lg-wg-search-result-item:hover { background: rgba(127,201,122,0.10); }
.lg-wg-search-surface { font-size: 18px; font-weight: 600; }
.lg-wg-search-reading { font-size: 12px; color: rgba(255,255,255,0.5); }
.lg-wg-search-gloss { font-size: 13px; color: rgba(255,255,255,0.65); margin-left: auto; text-align: right; max-width: 160px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.lg-wg-search-empty { padding: 18px; text-align: center; color: rgba(255,255,255,0.4); font-size: 13px; }
.lg-wg-search-loading { padding: 14px; text-align: center; color: rgba(255,255,255,0.4); font-size: 13px; }
/* Planted-success pulse on a newly grown plant */
@keyframes lg-wg-planted {
  0%   { transform: translateX(-50%) scale(0.4); opacity: 0; }
  60%  { transform: translateX(-50%) scale(1.18); opacity: 1; }
  100% { transform: translateX(-50%) scale(1); }
}
.lg-wg-plant-just-planted { animation: lg-wg-planted 480ms cubic-bezier(.22,1,.36,1); }
`;
  document.head.appendChild(style);
}

// ── Main export ───────────────────────────────────────────────────────

export async function launchWordGarden({ lang, voice }) {
  _ensureGardenStyles();

  const { overlay, close, addCleanup } = makeGameOverlay({
    id: 'lg-word-garden', palette: 'green', title: 'Word Garden',
  });

  overlay.innerHTML = `
    <div class="lg-game lg-wg">
      <header class="lg-hud">
        <button type="button" class="lg-close" aria-label="Close">×</button>
        <div class="lg-wg-title">Your garden</div>
        <button type="button" class="lg-wg-autospeak-btn" id="lg-wg-autospeak"
          aria-pressed="true" title="Auto-speak on open">🔊</button>
        <div class="lg-wg-stats" id="lg-wg-stats"></div>
      </header>
      <div class="lg-wg-search-wrapper" style="position:relative">
        <div class="lg-wg-search-bar">
          <input type="search" class="lg-wg-search-input" id="lg-wg-search-input"
            placeholder="Plant a word…" autocomplete="off" spellcheck="false"/>
          <button type="button" class="lg-wg-search-btn" id="lg-wg-search-btn">Search</button>
        </div>
      </div>
      <div class="lg-wg-sky"></div>
      <div class="lg-wg-soil">
        <div class="lg-wg-plot" id="lg-wg-plot"></div>
      </div>
      <div class="lg-wg-detail" id="lg-wg-detail" hidden></div>
    </div>`;

  overlay.querySelector('.lg-close').addEventListener('click', () => close());

  const plot = overlay.querySelector('#lg-wg-plot');
  const statsEl = overlay.querySelector('#lg-wg-stats');
  const detailEl = overlay.querySelector('#lg-wg-detail');
  const autospeakBtn = overlay.querySelector('#lg-wg-autospeak');
  const searchInput = overlay.querySelector('#lg-wg-search-input');
  const searchBtn = overlay.querySelector('#lg-wg-search-btn');
  const searchWrapper = overlay.querySelector('.lg-wg-search-wrapper');

  // ── State ─────────────────────────────────────────────────────────

  // Patch-in-place: hold all cards in a Map so we never rebuild 1000 plants.
  const cardMap = new Map();   // word_id → { card, plantEl }
  const counts = { new: 0, learning: 0, reviewing: 0, mature: 0, leech: 0, due: 0 };
  let autoSpeak = true;
  let readerPopover = null;    // current attachReaderPopover handle

  // Mastery-keyed vertical bands.
  const _BAND = { mature: 50, reviewing: 35, learning: 22, new: 12, leech: 5, due: 18 };

  // ── Auto-speak toggle ──────────────────────────────────────────────

  autospeakBtn.addEventListener('click', () => {
    autoSpeak = !autoSpeak;
    autospeakBtn.setAttribute('aria-pressed', String(autoSpeak));
  });

  // ── Stats render ───────────────────────────────────────────────────

  function renderStats() {
    const dueCount = counts.due || 0;
    statsEl.innerHTML = `
      ${dueCount ? `<span class="lg-wg-stat lg-wg-stat-due">${dueCount} due</span>` : ''}
      <span class="lg-wg-stat lg-wg-stat-mature">${counts.mature || 0} mature</span>
      <span class="lg-wg-stat lg-wg-stat-reviewing">${counts.reviewing || 0} reviewing</span>
      <span class="lg-wg-stat lg-wg-stat-learning">${counts.learning || 0} learning</span>
      <span class="lg-wg-stat lg-wg-stat-new">${counts.new || 0} new</span>`;
  }

  // ── Plant DOM helpers ──────────────────────────────────────────────

  function makePlantEl(card, index) {
    const stage = stageOf(card);
    const seed = hashStr(card.word_id || card.surface || '');
    const x = 4 + (seed % 92);
    const bandBottom = _BAND[stage] ?? _BAND.new;
    const bottomPct = bandBottom + ((seed >>> 8) % 14);
    const scale = 0.88 + ((seed >>> 12) % 18) / 100;

    const plant = document.createElement('button');
    plant.type = 'button';
    plant.className = `lg-wg-plant lg-wg-stage-${stage}`;
    plant.style.left = `${x}%`;
    plant.style.bottom = `${bottomPct}%`;
    plant.style.setProperty('--wg-i', String(index));
    plant.style.setProperty('--wg-hue', String(plantHue(card)));
    plant.style.setProperty('--wg-scale', scale.toFixed(2));
    plant.style.setProperty('--wg-depth', String(Math.round(bottomPct)));
    plant.dataset.wordId = card.word_id;
    plant.dataset.stage = stage;
    plant.innerHTML = plantMarkup(stage, card);
    plant.title = `${card.surface || ''}`;
    plant.setAttribute('aria-label', `${card.surface || card.reading || 'word'}, ${stage}`);
    return plant;
  }

  function updatePlantEl(plantEl, card) {
    const stage = stageOf(card);
    // Strip existing stage class, apply new one
    plantEl.className = plantEl.className.replace(/lg-wg-stage-\S+/, `lg-wg-stage-${stage}`);
    plantEl.style.setProperty('--wg-hue', String(plantHue(card)));
    plantEl.dataset.stage = stage;
    plantEl.innerHTML = plantMarkup(stage, card);
    plantEl.title = `${card.surface || ''}`;
    plantEl.setAttribute('aria-label', `${card.surface || card.reading || 'word'}, ${stage}`);
  }

  function wirePlant(plantEl, fallbackCard) {
    plantEl.addEventListener('click', (e) => {
      e.stopPropagation();
      const entry = cardMap.get(plantEl.dataset.wordId);
      showDetail(entry?.card || fallbackCard, plantEl);
    });
  }

  // ── Initial full render ────────────────────────────────────────────

  async function initialLoad() {
    const pool = await fetchGamePool(lang, 1000, 'garden');
    plot.innerHTML = '';
    Object.keys(counts).forEach(k => { counts[k] = 0; });
    cardMap.clear();

    if (pool.length === 0) {
      plot.innerHTML = `<div class="lg-wg-empty">Your garden is empty.<br>Search above to plant your first seeds.</div>`;
      statsEl.textContent = '';
      return;
    }

    pool.forEach((card, i) => {
      const stage = stageOf(card);
      const bucket = stage === 'due' ? 'due' : (card.mastery_state || 'new');
      counts[bucket] = (counts[bucket] || 0) + 1;

      const plantEl = makePlantEl(card, i);
      cardMap.set(card.word_id, { card, plantEl });
      wirePlant(plantEl, card);
      plot.appendChild(plantEl);
    });

    renderStats();
  }

  // ── Patch-in-place after a grade ──────────────────────────────────

  function patchCard(word_id, updatedCard) {
    const entry = cardMap.get(word_id);
    if (!entry) return;

    // Subtract old bucket
    const oldStage = stageOf(entry.card);
    const oldBucket = oldStage === 'due' ? 'due' : (entry.card.mastery_state || 'new');
    counts[oldBucket] = Math.max(0, (counts[oldBucket] || 1) - 1);

    // Merge updated card (gradeCard returns the updated record)
    const merged = Object.assign({}, entry.card, updatedCard || {});
    entry.card = merged;

    // Add new bucket
    const newStage = stageOf(merged);
    const newBucket = newStage === 'due' ? 'due' : (merged.mastery_state || 'new');
    counts[newBucket] = (counts[newBucket] || 0) + 1;

    // Mutate only this plant's DOM
    updatePlantEl(entry.plantEl, merged);
    renderStats();
  }

  // ── Detail panel ───────────────────────────────────────────────────
  //
  // Two-phase render:
  //   Phase 1 (gate):  shows surface + reading + "Tap to reveal"  — NO gloss/grade
  //   Phase 2 (open):  shows gloss + example + mastery + grade buttons
  //
  // This is the fix for FSRS corruption: grades are only reachable after the
  // user has actually attempted recall (tapped reveal without seeing the answer).

  function showDetail(card, plantEl) {
    // Destroy any previous reader popover before rebuilding the panel
    if (readerPopover) { readerPopover.destroy(); readerPopover = null; }
    detailEl.hidden = false;
    _renderGate(card, plantEl);

    // Auto-speak for cards still being learned
    if (autoSpeak && (card.mastery_state === 'new' || card.mastery_state === 'learning')) {
      speakWord(card.reading || card.surface, voice);
    }
  }

  function _renderGate(card, plantEl) {
    detailEl.innerHTML = `
      <div class="lg-wg-detail-card">
        <button type="button" class="lg-wg-detail-close" aria-label="Close">×</button>
        <div class="lg-wg-detail-surface">${escapeHtml(card.surface)}</div>
        <div class="lg-wg-detail-reading">
          ${escapeHtml(card.reading || '')}
          <button type="button" class="lg-wg-speak" aria-label="Speak">🔊</button>
        </div>
        <div class="lg-wg-detail-mastery">
          ${escapeHtml(card.mastery_state || '')}${card.in_queue ? ' · due today' : ''}
        </div>
        <div class="lg-wg-recall-gate">
          <button type="button" class="lg-wg-reveal-btn" id="lg-wg-reveal">
            Tap to reveal meaning
          </button>
          <div class="lg-wg-recall-hint">Try to recall before revealing</div>
        </div>
      </div>`;

    const closeBtn = detailEl.querySelector('.lg-wg-detail-close');
    closeBtn.addEventListener('click', () => {
      if (readerPopover) { readerPopover.destroy(); readerPopover = null; }
      detailEl.hidden = true;
    });

    detailEl.querySelector('.lg-wg-speak').addEventListener('click', () => {
      speakWord(card.reading || card.surface, voice);
    });

    const revealBtn = detailEl.querySelector('#lg-wg-reveal');
    const revealedAt = Date.now();
    revealBtn.addEventListener('click', () => {
      const elapsed = Date.now() - revealedAt;
      _renderOpen(card, plantEl, elapsed);
    });
  }

  function _renderOpen(card, plantEl, recallMs) {
    if (readerPopover) { readerPopover.destroy(); readerPopover = null; }

    const gloss = (card.glosses || []).slice(0, 4).join(' · ');
    const inQueue = !!card.in_queue;

    detailEl.innerHTML = `
      <div class="lg-wg-detail-card">
        <button type="button" class="lg-wg-detail-close" aria-label="Close">×</button>
        <div class="lg-wg-detail-surface">${escapeHtml(card.surface)}</div>
        <div class="lg-wg-detail-reading">
          ${escapeHtml(card.reading || '')}
          <button type="button" class="lg-wg-speak" aria-label="Speak">🔊</button>
        </div>
        <div class="lg-wg-detail-gloss">${escapeHtml(gloss)}</div>
        ${card.example
          ? `<div class="lg-wg-detail-example-reader" id="lg-wg-ex-reader">
               <span id="lg-wg-ex-lang"></span>
               <span class="lg-wg-ex-en">${escapeHtml(card.example.en_text || '')}</span>
             </div>`
          : ''}
        <div class="lg-wg-detail-mastery">
          Mastery: <b>${escapeHtml(card.mastery_state || '')}</b>
          ${inQueue ? ' · <span style="color:hsl(35,85%,65%)">due today</span>' : ''}
        </div>
        ${inQueue
          ? `<div class="lg-wg-detail-grade" id="lg-wg-grades">
               <button type="button" data-g="1" class="lg-wg-g lg-wg-g-1">Again</button>
               <button type="button" data-g="2" class="lg-wg-g lg-wg-g-2">Hard</button>
               <button type="button" data-g="3" class="lg-wg-g lg-wg-g-3">Good</button>
               <button type="button" data-g="4" class="lg-wg-g lg-wg-g-4">Easy</button>
             </div>`
          : `<div style="margin-top:14px;font-size:12px;color:rgba(255,255,255,0.4);text-align:center">
               Not due — no review needed today
             </div>`}
      </div>`;

    // Wire close
    detailEl.querySelector('.lg-wg-detail-close').addEventListener('click', () => {
      if (readerPopover) { readerPopover.destroy(); readerPopover = null; }
      detailEl.hidden = true;
    });

    detailEl.querySelector('.lg-wg-speak').addEventListener('click', () => {
      speakWord(card.reading || card.surface, voice);
    });

    // Render example sentence with clickable tokens
    if (card.example && card.example.lang_text) {
      const exReader = detailEl.querySelector('#lg-wg-ex-reader');
      const exLangEl = detailEl.querySelector('#lg-wg-ex-lang');
      if (exReader && exLangEl) {
        makeClickableHTML(card.example.lang_text, lang).then((html) => {
          exLangEl.innerHTML = html;
          exReader.dataset.fullText = card.example.lang_text;
          readerPopover = attachReaderPopover(exReader, {
            lang, voice,
            onWordAdded: () => { /* no-op: already in their garden */ },
          });
        });
      }
    }

    // Grade buttons — only rendered when card is due (in_queue)
    if (inQueue) {
      detailEl.querySelectorAll('.lg-wg-g').forEach(btn => {
        btn.addEventListener('click', async () => {
          const raw = Number(btn.dataset.g);
          // Map button choice → gradeForEffort. Recalling quickly = easy;
          // the recall gate is the retrieval event. recallMs = time to reveal.
          const grade = gradeForEffort({
            correct: raw >= 2,
            attempts: raw === 1 ? 2 : 1,   // Again means it didn't come
            hintsUsed: 0,
            replays: 0,
            ms: raw >= 3 ? recallMs : 0,   // only credit speed for positive grades
          });
          // Disable all grade buttons immediately to prevent double-submit
          detailEl.querySelectorAll('.lg-wg-g').forEach(b => { b.disabled = true; });
          const updated = await gradeCard(lang, card.word_id, grade);
          burstAt(plantEl, grade >= 3 ? '#7fc97a' : '#d99a4a');
          if (grade >= 3) {
            plantEl.classList.add('lg-wg-plant-grown');
            setTimeout(() => plantEl.classList.remove('lg-wg-plant-grown'), 1200);
          }
          // Patch-in-place: mutate only the graded plant + stats counters
          patchCard(card.word_id, updated);
          if (readerPopover) { readerPopover.destroy(); readerPopover = null; }
          detailEl.hidden = true;
        });
      });
    }
  }

  // Close detail when clicking outside it
  detailEl.addEventListener('click', (e) => {
    if (e.target === detailEl) {
      if (readerPopover) { readerPopover.destroy(); readerPopover = null; }
      detailEl.hidden = true;
    }
  });

  // ── Search / plant-a-word ──────────────────────────────────────────

  let searchResultsEl = null;

  function dismissSearch() {
    searchResultsEl?.remove();
    searchResultsEl = null;
  }

  async function runSearch(query) {
    const q = (query || '').trim();
    if (!q) { dismissSearch(); return; }

    dismissSearch();
    searchResultsEl = document.createElement('div');
    searchResultsEl.className = 'lg-wg-search-results';
    searchResultsEl.innerHTML = `<div class="lg-wg-search-loading">Searching…</div>`;
    searchWrapper.appendChild(searchResultsEl);

    try {
      const r = await fetch(
        `/api/learning/lookup?lang=${encodeURIComponent(lang)}&q=${encodeURIComponent(q)}&limit=8`
      );
      const entries = r.ok ? ((await r.json()).entries || []) : [];
      if (!entries.length) {
        searchResultsEl.innerHTML = `<div class="lg-wg-search-empty">No results for "${escapeHtml(q)}"</div>`;
        return;
      }
      searchResultsEl.innerHTML = entries.map((e) => `
        <div class="lg-wg-search-result-item" role="button" tabindex="0"
          data-word-id="${escapeHtml(e.word_id)}"
          data-surface="${escapeHtml(e.surface || '')}"
          data-reading="${escapeHtml(e.reading || '')}">
          <span class="lg-wg-search-surface">${escapeHtml(e.surface || '')}</span>
          <span class="lg-wg-search-reading">${escapeHtml(e.reading || '')}</span>
          <span class="lg-wg-search-gloss">${escapeHtml((e.glosses || []).slice(0, 2).join(' · '))}</span>
        </div>`).join('');

      searchResultsEl.querySelectorAll('.lg-wg-search-result-item').forEach(item => {
        const pick = async () => {
          const wordId = item.dataset.wordId;
          if (!wordId) return;
          dismissSearch();
          searchInput.value = '';
          const ok = await addWord(lang, wordId);
          if (ok) {
            // Add a new plant immediately without a full refresh
            const stub = {
              word_id: wordId,
              surface: item.dataset.surface,
              reading: item.dataset.reading,
              mastery_state: 'new',
              in_queue: false,
              glosses: [],
              example: null,
            };
            const index = cardMap.size;
            const plantEl = makePlantEl(stub, index);
            wirePlant(plantEl, stub);
            plantEl.classList.add('lg-wg-plant-just-planted');
            plantEl.addEventListener('animationend', () => {
              plantEl.classList.remove('lg-wg-plant-just-planted');
            }, { once: true });
            plot.appendChild(plantEl);
            cardMap.set(wordId, { card: stub, plantEl });
            counts.new = (counts.new || 0) + 1;
            // Remove empty state if present
            plot.querySelector('.lg-wg-empty')?.remove();
            renderStats();
            burstAt(plantEl, '#7fc97a');
          }
        };
        item.addEventListener('click', pick);
        item.addEventListener('keydown', (e) => { if (e.key === 'Enter') pick(); });
      });
    } catch {
      if (searchResultsEl) {
        searchResultsEl.innerHTML = `<div class="lg-wg-search-empty">Search failed — try again.</div>`;
      }
    }
  }

  searchBtn.addEventListener('click', () => runSearch(searchInput.value));
  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') runSearch(searchInput.value);
    if (e.key === 'Escape') dismissSearch();
  });

  // Dismiss search results on outside click
  document.addEventListener('mousedown', function onOutside(e) {
    if (searchResultsEl && !searchWrapper.contains(e.target)) {
      dismissSearch();
    }
    if (!overlay.isConnected) {
      document.removeEventListener('mousedown', onOutside);
    }
  });

  // ── Cleanup ────────────────────────────────────────────────────────

  addCleanup(() => {
    if (readerPopover) { readerPopover.destroy(); readerPopover = null; }
    dismissSearch();
  });

  // ── Boot ───────────────────────────────────────────────────────────

  await initialLoad();
}

// ── stageOf ──────────────────────────────────────────────────────────
//
// Returns 'due' when the card is in_queue AND not already mature —
// that's the drooping-plant visual indicating "tend me today".
// Mature cards that are also in_queue still show as mature (they've
// passed the bloom threshold; a review won't change that visually).

function stageOf(card) {
  const m = card.mastery_state;
  if (m === 'mature') return 'mature';
  if (m === 'leech')  return 'leech';
  if (card.in_queue)  return 'due';
  if (m === 'reviewing') return 'reviewing';
  if (m === 'learning')  return 'learning';
  return 'new';
}

function hashStr(s) {
  let h = 0;
  for (let i = 0; i < s.length; i += 1) h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  return Math.abs(h);
}

function plantHue(card) {
  return (hashStr(card.surface || card.reading || card.word_id || '') % 60) + 100;
}

function plantChip(card) {
  const surface = String(card.surface || card.reading || 'word');
  return surface.length > 5 ? `${surface.slice(0, 5)}...` : surface;
}

function plantStatus(stage) {
  if (stage === 'due') return 'Review';
  if (stage === 'leech') return 'Care';
  if (stage === 'mature') return 'Bloom';
  if (stage === 'reviewing') return 'Rooted';
  if (stage === 'learning') return 'Sprout';
  return 'Seed';
}

function plantMarkup(stage, card) {
  return `
    <span class="lg-wg-plant-shadow" aria-hidden="true"></span>
    <span class="lg-wg-plant-aura" aria-hidden="true"></span>
    <span class="lg-wg-plant-sprite" aria-hidden="true">${plantSvg(stage, card)}</span>
    <span class="lg-wg-plant-chip" aria-hidden="true">${escapeHtml(plantChip(card))}</span>
    <span class="lg-wg-plant-status" aria-hidden="true">${escapeHtml(plantStatus(stage))}</span>`;
}

// ── SVG plants ────────────────────────────────────────────────────────
//
// 'due' = drooping variant of the card's underlying mastery stage.
// We map to its mastery_state for colour, then the CSS rotate does the droop.

function plantSvg(stage, card) {
  const m = card.mastery_state;
  const hue = plantHue(card);

  if (stage === 'mature') {
    return `<svg viewBox="0 0 60 80" width="56" height="74" class="lg-wg-svg">
      <path d="M30 78 L30 50" stroke="hsl(${hue},40%,35%)" stroke-width="3" fill="none"/>
      <ellipse cx="30" cy="60" rx="8" ry="3" fill="hsl(${hue},40%,30%)" opacity=".5"/>
      <circle cx="30" cy="32" r="14" fill="hsl(${hue + 200},75%,68%)"/>
      <circle cx="20" cy="38" r="10" fill="hsl(${hue + 200},70%,75%)"/>
      <circle cx="40" cy="38" r="10" fill="hsl(${hue + 200},70%,75%)"/>
      <circle cx="22" cy="26" r="9" fill="hsl(${hue + 200},70%,75%)"/>
      <circle cx="38" cy="26" r="9" fill="hsl(${hue + 200},70%,75%)"/>
      <circle cx="30" cy="32" r="5" fill="hsl(${hue + 30},80%,55%)"/>
    </svg>`;
  }

  // For 'due', render the underlying mastery_state's SVG — the CSS droop rotates it.
  const renderStage = stage === 'due' ? (m || 'learning') : stage;

  if (renderStage === 'reviewing') {
    return `<svg viewBox="0 0 60 80" width="48" height="64" class="lg-wg-svg">
      <path d="M30 75 Q28 55 30 35" stroke="hsl(${hue},45%,38%)" stroke-width="3" fill="none"/>
      <ellipse cx="22" cy="42" rx="11" ry="5" fill="hsl(${hue},55%,42%)" transform="rotate(-30 22 42)"/>
      <ellipse cx="38" cy="42" rx="11" ry="5" fill="hsl(${hue},55%,42%)" transform="rotate(30 38 42)"/>
      <ellipse cx="30" cy="34" rx="9" ry="4" fill="hsl(${hue},60%,48%)"/>
    </svg>`;
  }

  if (renderStage === 'learning') {
    return `<svg viewBox="0 0 60 80" width="40" height="52" class="lg-wg-svg">
      <path d="M30 75 L30 55" stroke="hsl(${hue},50%,40%)" stroke-width="2.5" fill="none"/>
      <ellipse cx="22" cy="54" rx="9" ry="4" fill="hsl(${hue},55%,45%)" transform="rotate(-30 22 54)"/>
      <ellipse cx="38" cy="54" rx="9" ry="4" fill="hsl(${hue},55%,45%)" transform="rotate(30 38 54)"/>
    </svg>`;
  }

  if (renderStage === 'leech') {
    return `<svg viewBox="0 0 60 80" width="40" height="52" class="lg-wg-svg lg-wg-wilted">
      <path d="M30 75 Q24 62 32 58" stroke="hsl(20,40%,40%)" stroke-width="2.5" fill="none"/>
      <ellipse cx="22" cy="62" rx="8" ry="3" fill="hsl(20,40%,45%)" transform="rotate(-50 22 62)"/>
    </svg>`;
  }

  // new / fallback = mound
  return `<svg viewBox="0 0 60 30" width="36" height="20" class="lg-wg-svg">
    <ellipse cx="30" cy="22" rx="20" ry="6" fill="hsl(${hue + 10},25%,30%)"/>
    <circle cx="30" cy="18" r="2" fill="hsl(${hue},45%,55%)"/>
  </svg>`;
}
