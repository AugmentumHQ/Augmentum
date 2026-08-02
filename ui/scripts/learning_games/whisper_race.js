/**
 * Whisper Race — speech production.
 *
 * The model speaks a word; you mirror it back into the mic. We capture
 * via MediaRecorder, ship to /v1/audio/transcriptions, score the
 * Levenshtein similarity against the target reading. A live amplitude
 * meter rings the mic icon so the user sees their voice landing.
 *
 * Production cards are the missing half of an SRS — you can recognise
 * thousands of words you can never say aloud. This is the cheapest way
 * to close that gap without forcing typing or full conversation.
 */

import {
  escapeHtml, fetchGamePool, speakWord, pickOne, makeGameOverlay,
  makeEmptyOverlay, gradeCard, similarity, burstAt, fmtScore,
  recordResult, fetchBestScores,
  makeClickableHTML, attachReaderPopover,
  gradeForEffort, llmJudgeJSON, showNotice,
} from './_common.js';

// ─── Whisper-Race-scoped styles ──────────────────────────────────────
// Injected once per page; guards against duplicate injection across replays.
function _ensureWRStyles() {
  if (document.getElementById('lg-whisper-styles')) return;
  const style = document.createElement('style');
  style.id = 'lg-whisper-styles';
  style.textContent = `
/* Char-level diff in the heard-vs-target panel */
.lg-wr-diff { display: inline; font-family: var(--lg-font-display, monospace); font-size: 14px; }
.lg-wr-diff-ok  { color: #7fc97a; }
.lg-wr-diff-bad { color: #ff8a8a; text-decoration: underline wavy rgba(255,138,77,0.7); }

/* Example sentence panel after scoring */
.lg-wr-example {
  margin-top: 12px;
  padding: 10px 14px;
  background: rgba(255,255,255,0.04);
  border-radius: 10px;
  border-left: 3px solid var(--lg-accent, #ff8a4d);
  text-align: left;
  max-width: 420px;
}
.lg-wr-example-l2 {
  font-size: 15px;
  line-height: 1.5;
  color: rgba(255,255,255,0.88);
}
.lg-wr-example-en {
  font-size: 13px;
  color: rgba(255,255,255,0.5);
  margin-top: 4px;
  font-style: italic;
}

/* Slow-replay button row */
.lg-wr-replay-row {
  display: flex;
  gap: 10px;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
}
.lg-wr-slow-btn {
  font-size: 12px;
  padding: 4px 10px;
  opacity: 0.75;
}
.lg-wr-slow-btn:hover { opacity: 1; }

/* Retry notice for timeout */
.lg-wr-timeout-row {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  margin-top: 6px;
}

/* STT-unavailable & timeout state softened */
.lg-wr-status-warn { color: rgba(255,138,77,0.85); }

/* Heard diff panel */
.lg-wr-heard-detail {
  margin-top: 8px;
  font-size: 13px;
  color: rgba(255,255,255,0.65);
}
.lg-wr-heard-label {
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: rgba(255,255,255,0.4);
  margin-bottom: 3px;
}

/* LLM tone badge */
.lg-wr-tone-badge {
  display: inline-block;
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 100px;
  font-weight: 600;
  letter-spacing: 0.05em;
  margin-left: 8px;
  vertical-align: middle;
}
.lg-wr-tone-match   { background: rgba(127,201,122,0.22); color: #7fc97a; border: 1px solid rgba(127,201,122,0.4); }
.lg-wr-tone-close   { background: rgba(255,216,102,0.18); color: #ffd866; border: 1px solid rgba(255,216,102,0.35); }
.lg-wr-tone-wrong   { background: rgba(255,100,100,0.18); color: #ff8a8a; border: 1px solid rgba(255,100,100,0.35); }
`;
  document.head.appendChild(style);
}

// ─── Tonal / CJK language detection ─────────────────────────────────
// Used to decide whether to run llmJudgeJSON for tone-sensitive scoring.
const _TONAL_LANGS = new Set(['zh', 'ja', 'vi', 'th', 'ko']);

// ─── Dynamic pass threshold ──────────────────────────────────────────
// Short words need a tighter match: a 1-char error in a 3-char word is 67% —
// which the old flat 0.6 threshold passed. Require 0.85 for len ≤ 3.
function _passThreshold(target) {
  return target.length <= 3 ? 0.85 : 0.6;
}

// ─── Char-level diff (heard vs target) ──────────────────────────────
// Returns HTML string highlighting matched (green) / mismatched (red) chars.
// Both strings are first lowercased for comparison; the *original* `heard`
// characters are displayed so the user sees exactly what was transcribed.
function _charDiffHTML(heard, target) {
  const h = String(heard || '').toLowerCase();
  const t = String(target || '').toLowerCase();
  if (!h) return '';
  // Build Levenshtein DP table for alignment.
  const m = h.length, n = t.length;
  const dp = Array.from({ length: m + 1 }, (_, i) =>
    Array.from({ length: n + 1 }, (_, j) => (i === 0 ? j : j === 0 ? i : 0))
  );
  for (let i = 1; i <= m; i += 1) {
    for (let j = 1; j <= n; j += 1) {
      if (h[i - 1] === t[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = 1 + Math.min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]);
      }
    }
  }
  // Traceback to classify each character of `heard`.
  const matched = new Array(m).fill(false);
  let i = m, j = n;
  while (i > 0 && j > 0) {
    if (h[i - 1] === t[j - 1]) {
      matched[i - 1] = true;
      i -= 1; j -= 1;
    } else if (dp[i - 1][j - 1] <= dp[i - 1][j] && dp[i - 1][j - 1] <= dp[i][j - 1]) {
      // Substitution — mismatch
      i -= 1; j -= 1;
    } else if (dp[i - 1][j] < dp[i][j - 1]) {
      // Deletion in heard
      i -= 1;
    } else {
      // Insertion in heard
      j -= 1;
    }
  }
  // Render original `heard` characters coloured by match.
  const orig = String(heard || '');
  return Array.from(orig).map((ch, idx) => {
    const cls = matched[idx] ? 'lg-wr-diff-ok' : 'lg-wr-diff-bad';
    return `<span class="lg-wr-diff ${cls}">${escapeHtml(ch)}</span>`;
  }).join('');
}

const ROUND_QS = 8;
// AbortController timeout for the transcription fetch (ms).
const STT_TIMEOUT_MS = 10_000;

export async function launchWhisperRace({ lang, voice, focusWords = [] }) {
  _ensureWRStyles();

  const [pool, bests] = await Promise.all([
    fetchGamePool(lang, 30, 'drill', focusWords),
    fetchBestScores(lang),
  ]);
  const best = (bests.whisper_race && bests.whisper_race.best) || 0;
  if (pool.length < 4) {
    return makeEmptyOverlay({
      palette: 'ember', emoji: '🎙️',
      message: 'Add some words first — the race needs targets.',
      hint: 'Whisper Race pulls 8 words per round from your active queue.',
    });
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    return makeEmptyOverlay({
      palette: 'ember', emoji: '🎙️',
      message: "Your browser doesn't support microphone access.",
      hint: 'Try Chrome, Edge, or Firefox — or load the desktop app for full mic support.',
    });
  }

  const { overlay, close, addCleanup } = makeGameOverlay({
    id: 'lg-whisper', palette: 'ember', title: 'Whisper Race',
  });

  overlay.innerHTML = `
    <div class="lg-game lg-wr">
      <header class="lg-hud">
        <button type="button" class="lg-close" aria-label="Close">×</button>
        <div class="lg-wr-progress" id="lg-wr-progress">1 / ${ROUND_QS}</div>
        <div class="lg-hud-stats"><span class="lg-hud-label">score</span> <span id="lg-wr-score">0000</span></div>
      </header>
      <div class="lg-wr-stage">
        <div class="lg-wr-target">
          <div class="lg-hud-label">say this</div>
          <div class="lg-wr-target-surface" id="lg-wr-surface">—</div>
          <div class="lg-wr-target-reading" id="lg-wr-reading"></div>
          <div class="lg-wr-target-gloss" id="lg-wr-gloss"></div>
        </div>
        <button type="button" class="lg-wr-mic" id="lg-wr-mic" aria-label="Record">
          <div class="lg-wr-mic-ring"></div>
          <div class="lg-wr-mic-ring"></div>
          <div class="lg-wr-mic-ring"></div>
          <span>🎙</span>
        </button>
        <div class="lg-wr-status" id="lg-wr-status">Tap the mic. Speak. Tap again to stop.</div>
        <div class="lg-wr-replay-row">
          <button type="button" class="btn btn-ghost lg-wr-slow-btn" id="lg-wr-slow">🔈 Hear it slowly</button>
          <button type="button" class="btn btn-ghost lg-wr-skip" id="lg-wr-skip">Skip word</button>
        </div>
        <div class="lg-wr-result" id="lg-wr-result" hidden></div>
        <div class="lg-wr-controls" id="lg-wr-controls" hidden>
          <button type="button" class="btn btn-ghost" id="lg-wr-listen">Hear it again</button>
          <button type="button" class="btn btn-primary" id="lg-wr-next">Next →</button>
        </div>
      </div>
      <div class="lg-end" id="lg-wr-end" hidden>
        <div class="lg-end-title">Race complete</div>
        <div class="lg-end-stats" id="lg-wr-end-stats"></div>
        <div class="lg-end-actions">
          <button type="button" class="btn btn-primary" id="lg-wr-replay">Race again</button>
          <button type="button" class="btn btn-ghost" id="lg-wr-quit">Done</button>
        </div>
      </div>
    </div>`;

  overlay.querySelector('.lg-close').addEventListener('click', () => { stopMic(); close(); });

  const surfEl = overlay.querySelector('#lg-wr-surface');
  const readEl = overlay.querySelector('#lg-wr-reading');
  const glossEl = overlay.querySelector('#lg-wr-gloss');
  const micBtn = overlay.querySelector('#lg-wr-mic');
  const statusEl = overlay.querySelector('#lg-wr-status');
  const resultEl = overlay.querySelector('#lg-wr-result');
  const ctlEl = overlay.querySelector('#lg-wr-controls');
  const progressEl = overlay.querySelector('#lg-wr-progress');
  const scoreEl = overlay.querySelector('#lg-wr-score');
  const endPanel = overlay.querySelector('#lg-wr-end');

  // Reader popover — destroyed and re-attached per question via render()
  let _readerPopover = null;

  let score = 0;
  let qIdx = 0;
  let current = null;
  let recorder = null;
  let stream = null;
  let chunks = [];
  let audioCtx = null;
  let analyser = null;
  let rafId = null;
  const correct = [];
  const missed = [];
  const usedIds = new Set();
  let roundStart = performance.now();
  // Per-question effort tracking for gradeForEffort.
  let _replays = 0;      // how many times user tapped "Hear it again" before scoring
  let _attempts = 0;     // how many mic attempts (recordings) this question
  // Gates a second mic tap while we're in the post-stop transcription
  // window — see original comments.
  let transcribing = false;
  // Generation counter so stale in-flight transcriptions can detect they've
  // been superseded by a skip/render.
  let scoreGen = 0;

  function pickQuestion() {
    const remaining = pool.filter(c => !usedIds.has(c.word_id));
    const choice = pickOne(remaining.length ? remaining : pool);
    if (remaining.length === 0) usedIds.clear();
    usedIds.add(choice.word_id);
    return choice;
  }

  function render() {
    // Destroy any existing reader popover before swapping the card.
    if (_readerPopover) { _readerPopover.destroy(); _readerPopover = null; }
    scoreGen += 1;
    current = pickQuestion();
    _replays = 0;
    _attempts = 0;
    progressEl.textContent = `${qIdx + 1} / ${ROUND_QS}`;
    surfEl.textContent = current.surface;
    readEl.textContent = current.reading || '';
    glossEl.textContent = (current.glosses || [])[0] || '';
    resultEl.hidden = true;
    ctlEl.hidden = true;
    statusEl.textContent = 'Tap the mic. Speak. Tap again to stop.';
    statusEl.classList.remove('lg-wr-status-warn');
    micBtn.classList.remove('lg-wr-mic-recording');
    // Auto-speak after a brief settle so the user hears the model before
    // they need to repeat it.
    setTimeout(() => speakWord(current.reading || current.surface, voice), 200);
  }

  async function startMic() {
    chunks = [];
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      statusEl.textContent = 'Mic access denied. Allow microphone in your browser.';
      statusEl.classList.add('lg-wr-status-warn');
      return;
    }
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);
    const buf = new Uint8Array(analyser.frequencyBinCount);
    const tick = () => {
      analyser.getByteFrequencyData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i += 1) sum += buf[i];
      const level = sum / buf.length / 255;
      micBtn.style.setProperty('--wr-level', String(0.4 + level * 1.6));
      rafId = requestAnimationFrame(tick);
    };
    tick();

    recorder = new MediaRecorder(stream, { mimeType: pickMime() });
    recorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) chunks.push(e.data); };
    recorder.onstop = async () => {
      transcribing = true;
      stopVisuals();
      const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
      const attemptGen = scoreGen;
      _attempts += 1;
      try {
        await scoreAttempt(blob, attemptGen);
      } finally {
        transcribing = false;
      }
    };
    recorder.start();
    micBtn.classList.add('lg-wr-mic-recording');
    statusEl.textContent = 'Listening… tap to stop.';
    statusEl.classList.remove('lg-wr-status-warn');
  }

  function pickMime() {
    for (const m of ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg', 'audio/mp4']) {
      if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(m)) return m;
    }
    return '';
  }

  function stopVisuals() {
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
    if (stream) stream.getTracks().forEach(t => t.stop());
    if (audioCtx) audioCtx.close().catch(() => {});
    stream = null; audioCtx = null; analyser = null;
    micBtn.classList.remove('lg-wr-mic-recording');
    micBtn.style.removeProperty('--wr-level');
  }

  function stopMic() {
    try { if (recorder && recorder.state === 'recording') recorder.stop(); }
    catch { /* */ }
    stopVisuals();
  }

  // ─── Tone-aware LLM judge for tonal/CJK languages ──────────────────
  // Returns 'match' | 'close' | 'wrong' | null (null = LLM unavailable)
  async function _judgeTone(heard, target, targetLang) {
    const messages = [
      {
        role: 'system',
        content: `You are a strict ${targetLang} pronunciation judge. A student spoke a word and speech-to-text produced a transcript. Decide whether the transcript matches the target for a NATIVE speaker — considering tones, readings, and homophones.\nOutput ONLY valid JSON: {"verdict":"match"} or {"verdict":"close"} or {"verdict":"wrong"}.\n- match: identical or phonetically equivalent to a native ear\n- close: one tone/reading off but recognisable\n- wrong: clearly wrong`,
      },
      {
        role: 'user',
        content: `Target: "${target}"\nHeard: "${heard}"\nLanguage: ${targetLang}`,
      },
    ];
    const result = await llmJudgeJSON(messages, { fallback: null, timeoutMs: 6000 });
    if (!result || typeof result.verdict !== 'string') return null;
    const v = result.verdict.toLowerCase();
    if (v === 'match' || v === 'close' || v === 'wrong') return v;
    return null;
  }

  async function scoreAttempt(blob, attemptGen) {
    const attemptCard = current;
    statusEl.textContent = 'Scoring…';
    statusEl.classList.remove('lg-wr-status-warn');

    let heard = '';
    let sttFailed = false;
    let sttTimeout = false;

    // ── STT fetch with AbortController timeout ────────────────────────
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), STT_TIMEOUT_MS);
    try {
      const fd = new FormData();
      fd.append('file', blob, 'attempt.webm');
      fd.append('model', 'whisper-1');
      fd.append('language', lang);
      const r = await fetch('/v1/audio/transcriptions', {
        method: 'POST',
        body: fd,
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (r.status === 503 || r.status === 502) {
        sttFailed = true;
      } else if (r.ok) {
        const j = await r.json();
        heard = (j.text || '').trim();
      } else {
        // Other non-ok status — treat as STT down.
        sttFailed = true;
      }
    } catch (err) {
      clearTimeout(timeoutId);
      if (err && err.name === 'AbortError') {
        sttTimeout = true;
      } else {
        sttFailed = true;
      }
    }

    // Guard: stale generation or overlay removed.
    if (attemptGen !== scoreGen || !overlay.isConnected) return;

    // ── Handle STT-down / timeout without grading ─────────────────────
    if (sttFailed) {
      showNotice(overlay, 'Speech-to-text isn\'t configured — add an STT provider in Settings, or use "Hear it slowly" + Skip.', { kind: 'warn' });
      statusEl.textContent = '';
      // Reveal controls so user can skip or hear the word — but don't grade.
      resultEl.innerHTML = `<div class="lg-wr-heard-detail"><div class="lg-wr-heard-label">No STT available</div><div style="color:rgba(255,255,255,0.5);font-size:13px;">Configure a speech-to-text provider in Settings to enable scoring.</div></div>`;
      resultEl.hidden = false;
      ctlEl.hidden = false;
      return;
    }
    if (sttTimeout) {
      statusEl.textContent = '';
      resultEl.innerHTML = `
        <div class="lg-wr-timeout-row">
          <div style="color:rgba(255,138,77,0.85);font-size:14px;">Transcription timed out.</div>
          <button type="button" class="btn btn-ghost lg-wr-slow-btn" id="lg-wr-retry-btn">Try again</button>
        </div>`;
      resultEl.hidden = false;
      resultEl.querySelector('#lg-wr-retry-btn')?.addEventListener('click', () => {
        resultEl.hidden = true;
        statusEl.textContent = 'Tap the mic. Speak. Tap again to stop.';
      });
      ctlEl.hidden = false;
      return;
    }

    // ── No speech detected ────────────────────────────────────────────
    // Don't grade silence — show a prompt to try again or skip.
    if (!heard) {
      statusEl.textContent = '';
      resultEl.innerHTML = `<div style="color:rgba(255,255,255,0.5);font-size:13px;text-align:center;">(no speech detected — tap the mic and try again, or skip)</div>`;
      resultEl.hidden = false;
      ctlEl.hidden = false;
      return;
    }

    // ── Similarity scoring ────────────────────────────────────────────
    const target = attemptCard.reading || attemptCard.surface;
    let sim = Math.max(similarity(heard, target), similarity(heard, attemptCard.surface));

    // ── Tone-aware LLM adjustment (tonal/CJK langs only) ─────────────
    let toneVerdict = null;
    if (_TONAL_LANGS.has(lang)) {
      toneVerdict = await _judgeTone(heard, target, lang);
      // Guard again — LLM judge is async and scoreGen may have advanced.
      if (attemptGen !== scoreGen || !overlay.isConnected) return;
      if (toneVerdict === 'match') {
        sim = Math.max(sim, 0.95);   // treat as essentially correct
      } else if (toneVerdict === 'close') {
        sim = Math.max(sim, 0.72);   // close but not a full pass
        sim = Math.min(sim, 0.84);   // cap below the short-word threshold
      } else if (toneVerdict === 'wrong') {
        sim = Math.min(sim, 0.55);   // force below all thresholds
      }
    }

    const threshold = _passThreshold(target);
    const passed = sim >= threshold;
    const pct = Math.round(sim * 100);
    const points = Math.round(sim * 15);
    score += points;
    scoreEl.textContent = fmtScore(score);

    if (passed) {
      burstAt(micBtn, '#ff8a4d');
      correct.push({ card: attemptCard, replays: _replays, attempts: _attempts, ms: 0 });
    } else if (heard) {
      missed.push(attemptCard);
    }

    // ── Char-level diff display ───────────────────────────────────────
    const diffHTML = _charDiffHTML(heard, target);
    const toneBadge = toneVerdict
      ? `<span class="lg-wr-tone-badge lg-wr-tone-${toneVerdict}">${
          toneVerdict === 'match' ? 'tone match' : toneVerdict === 'close' ? 'tone close' : 'wrong tone'
        }</span>`
      : '';

    // ── Example sentence (meaning grounding) ─────────────────────────
    const exampleL2 = attemptCard.example || '';
    const exampleEn = attemptCard.example_en || '';

    // Build result panel HTML.
    let resultHTML = `
      <div class="lg-wr-bar"><div class="lg-wr-bar-fill" style="width:${pct}%"></div></div>
      <div class="lg-wr-bar-label">${pct}% match · +${points}${toneBadge}</div>
      <div class="lg-wr-heard-detail">
        <div class="lg-wr-heard-label">you said · target: <strong>${escapeHtml(target)}</strong></div>
        <div>${diffHTML || `<span style="color:rgba(255,255,255,0.45);font-style:italic">(no speech)</span>`}</div>
      </div>`;

    if (exampleL2) {
      resultHTML += `
      <div class="lg-wr-example" id="lg-wr-example-box">
        <div class="lg-wr-example-l2" id="lg-wr-example-l2">${escapeHtml(exampleL2)}</div>
        ${exampleEn ? `<div class="lg-wr-example-en">${escapeHtml(exampleEn)}</div>` : ''}
      </div>`;
    }

    resultEl.innerHTML = resultHTML;
    resultEl.hidden = false;
    ctlEl.hidden = false;
    statusEl.textContent = '';

    // ── Attach reader popover to the example sentence ─────────────────
    if (exampleL2) {
      const exampleBox = resultEl.querySelector('#lg-wr-example-box');
      const l2El = resultEl.querySelector('#lg-wr-example-l2');
      if (l2El) {
        // Render clickable tokens for the example sentence.
        makeClickableHTML(exampleL2, lang).then((html) => {
          if (!l2El.isConnected) return;
          l2El.innerHTML = html;
          l2El.dataset.fullText = exampleL2;
          if (_readerPopover) { _readerPopover.destroy(); }
          _readerPopover = attachReaderPopover(exampleBox, { lang, voice });
        }).catch(() => { /* offline pack unavailable — plain text stays */ });
      }
    }
  }

  micBtn.addEventListener('click', () => {
    if (transcribing) return;
    if (recorder && recorder.state === 'recording') {
      stopMic();
    } else {
      startMic();
    }
  });

  // "Hear it again" button — normal-speed replay, counts toward replay tally.
  overlay.querySelector('#lg-wr-listen').addEventListener('click', () => {
    _replays += 1;
    speakWord(current.reading || current.surface, voice);
  });

  // "Hear it slowly" — 0.7x rate, always available in the skip row.
  // Shared handler for both the pre-result row and (below) any retry state.
  function _hearSlowly() {
    speakWord(current.reading || current.surface, voice, { rate: 0.7 });
  }
  overlay.querySelector('#lg-wr-slow').addEventListener('click', _hearSlowly);

  overlay.querySelector('#lg-wr-next').addEventListener('click', () => {
    qIdx += 1;
    if (qIdx >= ROUND_QS) endRound();
    else render();
  });
  overlay.querySelector('#lg-wr-skip').addEventListener('click', () => {
    stopMic();
    qIdx += 1;
    if (qIdx >= ROUND_QS) endRound();
    else render();
  });

  async function endRound() {
    // Destroy any dangling reader popover before the end screen.
    if (_readerPopover) { _readerPopover.destroy(); _readerPopover = null; }

    const seenIds = new Set();
    const toGrade = [];
    for (const { card, replays, attempts } of correct) {
      if (toGrade.length >= 5) break;
      if (card.in_queue === false) continue;
      if (seenIds.has(card.word_id)) continue;
      seenIds.add(card.word_id);
      toGrade.push({ card, replays, attempts });
    }
    const toFail = [];
    for (const c of missed) {
      if (toFail.length >= 5) break;
      if (c.in_queue === false) continue;
      if (seenIds.has(c.word_id)) continue;
      seenIds.add(c.word_id);
      toFail.push(c);
    }
    await Promise.all([
      ...toGrade.map(({ card, replays, attempts }) => {
        const grade = gradeForEffort({ correct: true, attempts, replays });
        return gradeCard(lang, card.word_id, grade);
      }),
      ...toFail.map(c => gradeCard(lang, c.word_id, 1)),
    ]);
    const beatBest = score > best;
    overlay.querySelector('#lg-wr-end-stats').innerHTML = `
      <div class="lg-end-stat"><div class="lg-end-stat-n">${fmtScore(score)}</div><div>score${beatBest ? ' · new best!' : ''}</div></div>
      <div class="lg-end-stat"><div class="lg-end-stat-n">${correct.length}/${ROUND_QS}</div><div>matched</div></div>`;
    endPanel.hidden = false;
    recordResult({
      game_id: 'whisper_race', lang,
      score, words_played: ROUND_QS, words_correct: correct.length,
      duration_sec: Math.round((performance.now() - roundStart) / 1000),
    });
  }

  overlay.querySelector('#lg-wr-replay').addEventListener('click', () => {
    score = 0; qIdx = 0; correct.length = 0; missed.length = 0;
    usedIds.clear();
    scoreEl.textContent = '0000';
    endPanel.hidden = true;
    roundStart = performance.now();
    render();
  });
  overlay.querySelector('#lg-wr-quit').addEventListener('click', () => close());

  addCleanup(() => {
    stopMic();
    if (_readerPopover) { _readerPopover.destroy(); _readerPopover = null; }
  });
  render();
}
