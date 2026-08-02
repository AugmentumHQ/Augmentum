/**
 * 💬 Companion — conversational practice with a partner at your level.
 *
 * A turn-based chat with an AI partner who speaks in the target language,
 * constrained to vocab the user has seen (mastery >= learning) plus 1-2
 * stretch words per turn. Click any word in the partner's reply to define
 * it; the partner uses gentle recasts when you make mistakes ("oh, you
 * meant X?") instead of corrections.
 *
 * This is the only game where vocab is added *from the conversation* —
 * stretch words the partner uses can be tapped to add to the user's
 * queue, so chatting itself becomes a vocab-acquisition surface.
 *
 * --- v2 additions (2026-06-17) ---
 * 1. Visible recasts: after each partner reply a parallel llmJudgeJSON check
 *    inspects the learner's last turn; if the partner silently corrected
 *    something, a small annotation appears under the bubble.
 * 2. "What could I say?" scaffold: a button near the input fires one llmChat
 *    call and pre-fills an editable example reply in the target language.
 * 3. Runtime difficulty adaptation: every 3 learner turns, an invisible system
 *    injection either simplifies (if learner replies in L1 ≥ 2 turns) or
 *    introduces a stretch word (if learner is comfortable).
 * 4. FSRS credit for produced words: after each user turn, exact pool-surface
 *    matches in the typed/spoken text are graded via gradeForEffort.
 * 5. STT honesty: detect 503/empty response and showNotice with actionable
 *    copy; voice toggle hidden when mic is unavailable.
 * 6. CJK ruby: furigana/pinyin annotation on partner bubbles via fetchBreakdown.
 */

import {
  escapeHtml, fetchGamePool, llmChatStream, llmChat, llmJudgeJSON,
  makeGameOverlay, speakWord, addWord, recordResult, fetchBreakdown,
  breakdownContextual, gradeCard, gradeForEffort, showNotice,
} from './_common.js';

// Japanese and Chinese aren't whitespace-segmented, so the regex
// tokenizer below would treat a whole CJK phrase as one giant "word",
// breaking click-to-define. For these langs we use the pack's
// dictionary-aware breakdown API to split the partner's reply into
// word-sized tokens before rendering clickables.
// Also used for ruby annotation (furigana/pinyin) on partner bubbles (feature #6).
const _CJK_TOKENIZE = new Set(['ja', 'zh']);

const TOPICS = [
  { id: 'cafe',    label: '☕ At a café',         seed: 'You meet at a café and order something to drink.' },
  { id: 'travel',  label: '✈️ Travel plans',     seed: 'You\'re planning a weekend trip together.' },
  { id: 'food',    label: '🍳 Cooking',           seed: 'You\'re cooking dinner and deciding what to make.' },
  { id: 'weather', label: '🌧 Weather + day',    seed: 'You\'re catching up about today and the weather.' },
  { id: 'walk',    label: '🚶 A walk outside',   seed: 'You\'re walking through a park, noticing things.' },
];

// Inject styles once. Guarded by id so multiple overlay opens don't pile up.
function _ensureStyles() {
  if (document.getElementById('lg-codlg-styles')) return;
  const style = document.createElement('style');
  style.id = 'lg-codlg-styles';
  style.textContent = `
.lg-co { display:flex; flex-direction:column; height:100%; }
.lg-hud { display:flex; align-items:center; gap:8px; padding:10px 14px;
  border-bottom:1px solid var(--border,rgba(255,255,255,.1)); flex-shrink:0; }
.lg-co-title { flex:1; font-weight:600; font-size:14px; color:var(--text-muted,#9aa0aa); }
.lg-co-mode, .lg-co-newtopic {
  background:var(--bg-raised,#252830); border:1px solid var(--border,rgba(255,255,255,.1));
  border-radius:8px; padding:4px 10px; font-size:13px; cursor:pointer;
  color:var(--text-primary,#e8e8ea); white-space:nowrap; }
.lg-co-mode:hover, .lg-co-newtopic:hover { background:var(--bg-hover,#303340); }
.lg-close { background:none; border:none; color:var(--text-muted,#9aa0aa);
  font-size:22px; cursor:pointer; padding:0 4px; line-height:1; }

/* migration banner */
.lg-co-migrate { display:flex; align-items:center; gap:10px; margin:12px;
  padding:10px 14px; background:color-mix(in srgb,var(--accent,#6ea8fe) 10%,transparent);
  border:1px solid color-mix(in srgb,var(--accent,#6ea8fe) 28%,transparent);
  border-radius:10px; }
.lg-co-migrate-icon { font-size:22px; }
.lg-co-migrate-body { flex:1; }
.lg-co-migrate-title { font-weight:600; font-size:13px; }
.lg-co-migrate-tag { font-size:12px; color:var(--text-muted,#9aa0aa); margin-top:2px; }
.lg-co-migrate-cta { background:var(--accent,#6ea8fe); color:#fff;
  border:none; border-radius:8px; padding:6px 12px; font-size:13px; cursor:pointer; white-space:nowrap; }

/* topic chips */
.lg-co-topics { display:flex; flex-wrap:wrap; gap:8px; padding:16px;
  justify-content:center; overflow-y:auto; }
.lg-co-topic { background:var(--bg-raised,#252830);
  border:1px solid var(--border,rgba(255,255,255,.1));
  border-radius:20px; padding:8px 16px; font-size:14px; cursor:pointer;
  color:var(--text-primary,#e8e8ea); transition:background .15s; }
.lg-co-topic:hover { background:var(--bg-hover,#303340); }

/* conversation thread */
.lg-co-thread { flex:1; overflow-y:auto; padding:12px 14px; display:flex;
  flex-direction:column; gap:10px; }
.lg-co-msg { display:flex; }
.lg-co-msg-you { justify-content:flex-end; }
.lg-co-msg-partner { justify-content:flex-start; }
.lg-co-msg-bubble { max-width:78%; padding:9px 13px; border-radius:14px;
  font-size:14px; line-height:1.5; position:relative; }
.lg-co-msg-you .lg-co-msg-bubble { background:var(--accent,#6ea8fe);
  color:#fff; border-bottom-right-radius:4px; }
.lg-co-msg-partner .lg-co-msg-bubble { background:var(--bg-raised,#252830);
  color:var(--text-primary,#e8e8ea); border-bottom-left-radius:4px; }
.lg-co-msg-streaming .lg-co-dots { display:inline-flex; gap:3px; vertical-align:middle; margin-left:4px; }
.lg-co-msg-streaming .lg-co-dots i { width:5px; height:5px; border-radius:50%;
  background:var(--text-muted,#9aa0aa); animation:lg-co-bounce .9s ease-in-out infinite; }
.lg-co-msg-streaming .lg-co-dots i:nth-child(2) { animation-delay:.15s; }
.lg-co-msg-streaming .lg-co-dots i:nth-child(3) { animation-delay:.3s; }
@keyframes lg-co-bounce { 0%,100%{transform:translateY(0);opacity:.4} 50%{transform:translateY(-4px);opacity:1} }
.lg-co-speak { background:none; border:none; cursor:pointer; font-size:13px;
  padding:0 0 0 6px; vertical-align:middle; opacity:.55; }
.lg-co-speak:hover { opacity:1; }
.lg-co-word { cursor:pointer; border-radius:3px; transition:background .12s; }
.lg-co-word:hover { background:color-mix(in srgb,var(--accent,#6ea8fe) 24%,transparent); }
.lg-co-bubble-analysing { opacity:.8; }

/* recast annotation */
.lg-co-recast { margin-top:5px; padding:5px 9px; font-size:12px; line-height:1.4;
  background:color-mix(in srgb,var(--warning,#e0a800) 16%,transparent);
  border-left:2px solid var(--warning,#e0a800); border-radius:0 6px 6px 0;
  color:var(--text-secondary,#c2c5cc); }
.lg-co-recast-label { font-size:10.5px; text-transform:uppercase; letter-spacing:.05em;
  color:var(--warning,#e0a800); font-weight:600; margin-bottom:2px; }
.lg-co-recast-original { color:var(--text-muted,#9aa0aa); }
.lg-co-recast-arrow { margin:0 4px; }
.lg-co-recast-correction { font-weight:600; color:var(--text-primary,#e8e8ea); }
.lg-co-recast-note { font-size:11.5px; color:var(--text-muted,#9aa0aa); margin-top:2px; }

/* what-could-I-say scaffold */
.lg-co-scaffold { display:flex; justify-content:flex-end; padding:0 14px 4px; }
.lg-co-scaffold-btn { background:none;
  border:1px solid var(--border,rgba(255,255,255,.15));
  border-radius:14px; padding:4px 12px; font-size:12.5px;
  color:var(--text-muted,#9aa0aa); cursor:pointer; transition:all .15s; }
.lg-co-scaffold-btn:hover { background:var(--bg-raised,#252830); color:var(--text-primary,#e8e8ea); }
.lg-co-scaffold-btn:disabled { opacity:.4; cursor:default; }

/* input row */
.lg-co-input { display:flex; gap:8px; padding:10px 14px;
  border-top:1px solid var(--border,rgba(255,255,255,.1)); flex-shrink:0; align-items:flex-end; }
.lg-co-textarea { flex:1; resize:none; background:var(--bg-raised,#252830);
  border:1px solid var(--border,rgba(255,255,255,.12)); border-radius:10px;
  padding:8px 12px; color:var(--text-primary,#e8e8ea); font-size:14px;
  line-height:1.4; min-height:40px; font-family:inherit; }
.lg-co-textarea:focus { outline:none; border-color:var(--accent,#6ea8fe); }
.lg-co-send { white-space:nowrap; }

/* voice mode */
.lg-co-voice { display:flex; flex-direction:column; align-items:center; gap:10px;
  padding:18px 14px; border-top:1px solid var(--border,rgba(255,255,255,.1)); flex-shrink:0; }
.lg-co-mic { background:var(--bg-raised,#252830);
  border:2px solid var(--border,rgba(255,255,255,.15));
  border-radius:50px; padding:12px 24px; cursor:pointer;
  display:flex; align-items:center; gap:8px; font-size:14px;
  color:var(--text-primary,#e8e8ea); transition:all .15s; }
.lg-co-mic:hover { background:var(--bg-hover,#303340); }
.lg-co-mic.lg-co-mic-recording { border-color:var(--error,#f06969);
  background:color-mix(in srgb,var(--error,#f06969) 14%,transparent);
  animation:lg-co-pulse 1.2s ease-in-out infinite; }
@keyframes lg-co-pulse { 0%,100%{box-shadow:0 0 0 0 color-mix(in srgb,var(--error,#f06969) 40%,transparent)}
  50%{box-shadow:0 0 0 10px transparent} }
.lg-co-voice-status { font-size:13px; color:var(--text-muted,#9aa0aa); min-height:18px; text-align:center; }

/* word popover */
.lg-co-pop { position:fixed; z-index:100000; max-width:360px; min-width:220px;
  background:var(--bg-elevated,#1b1d24); color:var(--text-primary,#e8e8ea);
  border:1px solid var(--border,rgba(255,255,255,.12)); border-radius:12px;
  padding:12px 14px; box-shadow:0 12px 40px rgba(0,0,0,.45);
  font-size:14px; line-height:1.4; animation:lg-co-pop-in .14s ease; }
@keyframes lg-co-pop-in { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:none} }
.lg-co-pop-loading, .lg-co-pop-empty { color:var(--text-muted,#9aa0aa); font-style:italic; }
.lg-co-pop-head, .lg-co-bd-focus-head { display:flex; align-items:center; gap:8px; }
.lg-co-pop-surface, .lg-co-bd-focus-token { font-size:18px; font-weight:700; }
.lg-co-pop-reading { color:var(--text-muted,#9aa0aa); }
.lg-co-pop-speak, .lg-co-pop-close { margin-left:auto; background:none; border:none;
  color:var(--text-muted,#9aa0aa); cursor:pointer; font-size:16px; padding:2px 4px; }
.lg-co-pop-close { margin-left:4px; }
.lg-co-pop-gloss { margin:8px 0; }
.lg-co-bd-focus { }
.lg-co-bd-focus-meaning { margin:6px 0 2px; font-weight:600; }
.lg-co-bd-focus-role { color:var(--text-muted,#9aa0aa); font-size:12.5px; }
.lg-co-bd-focus-actions { margin-top:8px; }
.lg-co-bd-add { font-size:13px; }
.lg-co-bd-added { background:var(--success,#30a46c) !important; }
.lg-co-bd-divider { height:1px; background:var(--border,rgba(255,255,255,.1)); margin:10px 0 8px; }
.lg-co-bd-label { font-size:11px; text-transform:uppercase; letter-spacing:.06em;
  color:var(--text-muted,#9aa0aa); margin-bottom:4px; }
.lg-co-bd-list { list-style:none; margin:0; padding:0; max-height:200px; overflow-y:auto; }
.lg-co-bd-row { display:grid; grid-template-columns:auto 1fr; gap:2px 10px;
  padding:4px 6px; border-radius:6px; }
.lg-co-bd-row-hit { background:color-mix(in srgb,var(--accent,#6ea8fe) 18%,transparent); }
.lg-co-bd-token { font-weight:600; }
.lg-co-bd-meaning { color:var(--text-secondary,#c2c5cc); }
.lg-co-bd-role { grid-column:2; color:var(--text-muted,#9aa0aa); font-size:11.5px; }

/* CJK ruby annotations */
.lg-co-ruby ruby { ruby-align:center; }
.lg-co-ruby rt { font-size:.55em; color:var(--text-muted,#9aa0aa); }
`;
  document.head.appendChild(style);
}

export async function launchCompanionDialogue({ lang, voice }) {
  _ensureStyles();

  // Consolidate mode — chat at the user's confident level. Stretch words
  // arrive via the partner's reply, not via the pool.
  const pool = await fetchGamePool(lang, 60, 'consolidate');
  // Build a Set of known surfaces for FSRS grading (feature #4).
  // Lower-cased for case-insensitive matching against typed/spoken text.
  const poolSurfaceSet = new Set(pool.map(c => (c.surface || '').toLowerCase().trim()).filter(Boolean));
  // Pool card lookup by surface for word_id retrieval.
  const poolBySurface = new Map(pool.map(c => [(c.surface || '').toLowerCase().trim(), c]));

  const sessionStart = performance.now();
  let turnsExchanged = 0;
  let wordsAdded = 0;

  const { overlay, close: rawClose } = makeGameOverlay({
    id: 'lg-companion', palette: 'rose', title: 'Companion',
  });
  // Wrap close so we record the session when leaving.
  const close = () => {
    sendGen += 1;   // discard any in-flight reply
    if (turnsExchanged > 0) {
      recordResult({
        game_id: 'companion_dialogue', lang,
        score: turnsExchanged * 5 + wordsAdded * 10,
        words_played: turnsExchanged, words_correct: wordsAdded,
        duration_sec: Math.round((performance.now() - sessionStart) / 1000),
        metadata: { topic: topic && topic.id, words_added: wordsAdded },
      });
    }
    rawClose();
  };

  // Check STT availability proactively so we can show an honest notice
  // rather than a silent "didn't catch that" (feature #5).
  // We don't block launch on this — fire and forget; UI updated when ready.
  const hasMic = !!navigator.mediaDevices?.getUserMedia;
  let sttAvailable = hasMic;   // assume available if mic is present; probe on first use

  overlay.innerHTML = `
    <div class="lg-co lg-game">
      <header class="lg-hud">
        <button type="button" class="lg-close" aria-label="Close">×</button>
        <div class="lg-co-title">Companion chat — ${escapeHtml(lang.toUpperCase())}</div>
        ${hasMic ? `<button type="button" class="lg-co-mode" id="lg-co-mode" title="Toggle voice mode" data-mode="text">⌨ Text</button>` : ''}
        <button type="button" class="lg-co-newtopic" id="lg-co-newtopic" title="New topic">↻</button>
      </header>
      <div class="lg-co-migrate" id="lg-co-migrate">
        <div class="lg-co-migrate-icon" aria-hidden="true">💬</div>
        <div class="lg-co-migrate-body">
          <div class="lg-co-migrate-title">Looking for a real conversation partner?</div>
          <div class="lg-co-migrate-tag">Quick chat is a one-off round. A persistent partner — with memory, drills, and voice — lives on the hub above.</div>
        </div>
        <button type="button" class="lg-co-migrate-cta" id="lg-co-migrate-cta">Open partner →</button>
      </div>
      <div class="lg-co-topics" id="lg-co-topics">
        ${TOPICS.map(t => `<button type="button" class="lg-co-topic" data-id="${escapeHtml(t.id)}">${escapeHtml(t.label)}</button>`).join('')}
      </div>
      <div class="lg-co-thread" id="lg-co-thread"></div>
      <div class="lg-co-scaffold" id="lg-co-scaffold" hidden>
        <button type="button" class="lg-co-scaffold-btn" id="lg-co-scaffold-btn" title="Get an example reply you can edit">💡 What could I say?</button>
      </div>
      <div class="lg-co-input" id="lg-co-input" hidden>
        <textarea class="lg-co-textarea" id="lg-co-textarea" rows="2" placeholder="Reply in ${escapeHtml(lang.toUpperCase())} or English…"></textarea>
        <button type="button" class="btn btn-primary lg-co-send" id="lg-co-send">Send</button>
      </div>
      <div class="lg-co-voice" id="lg-co-voice" hidden>
        <button type="button" class="lg-co-mic" id="lg-co-mic" aria-label="Hold to speak">
          <span class="lg-co-mic-icon">🎙</span>
          <span class="lg-co-mic-label">Tap to speak</span>
        </button>
        <div class="lg-co-voice-status" id="lg-co-voice-status"></div>
      </div>
    </div>`;

  overlay.querySelector('.lg-close').addEventListener('click', () => close());

  // Migration CTA — opens the persistent partner instead of starting a
  // one-off topic. Closes this overlay first so the user lands on the
  // chat surface, not stacked on top of Quick Chat.
  overlay.querySelector('#lg-co-migrate-cta')?.addEventListener('click', async () => {
    try {
      close();
      const hub = await import('./hub.js');
      hub.openGamesHub({ lang, voice }).catch(() => {});
    } catch (err) {
      console.warn('[companion-game] partner migration failed', err);
    }
  });

  const topicsEl = overlay.querySelector('#lg-co-topics');
  const threadEl = overlay.querySelector('#lg-co-thread');
  const inputEl = overlay.querySelector('#lg-co-input');
  const textarea = overlay.querySelector('#lg-co-textarea');
  const sendBtn = overlay.querySelector('#lg-co-send');
  const voiceEl = overlay.querySelector('#lg-co-voice');
  const voiceMicBtn = overlay.querySelector('#lg-co-mic');
  const voiceStatusEl = overlay.querySelector('#lg-co-voice-status');
  const modeBtn = overlay.querySelector('#lg-co-mode');
  const scaffoldRow = overlay.querySelector('#lg-co-scaffold');
  const scaffoldBtn = overlay.querySelector('#lg-co-scaffold-btn');

  let inputMode = 'text';   // 'text' | 'voice'
  let recorder = null;
  let recStream = null;
  let recChunks = [];
  // Gates a second mic tap while we're in the post-stop transcription
  // window — `busy` only flips inside submitUserText, leaving a 1-3s gap
  // where a stale tap would spin up a second MediaRecorder concurrently.
  let transcribing = false;

  function setInputMode(next) {
    inputMode = next;
    if (!modeBtn) return;
    if (next === 'voice') {
      modeBtn.dataset.mode = 'voice';
      modeBtn.textContent = '🎙 Voice';
      if (topic) {
        inputEl.hidden = true;
        voiceEl.hidden = false;
        scaffoldRow.hidden = true;   // scaffold is text-only
      }
    } else {
      modeBtn.dataset.mode = 'text';
      modeBtn.textContent = '⌨ Text';
      if (topic) {
        inputEl.hidden = false;
        voiceEl.hidden = true;
        scaffoldRow.hidden = false;
      }
    }
  }

  if (modeBtn) {
    modeBtn.addEventListener('click', () => {
      if (busy) return;
      setInputMode(inputMode === 'text' ? 'voice' : 'text');
    });
  }

  let topic = null;
  let history = [];   // {role, content}
  let busy = false;
  // Increments on any reset (topic switch / close). An in-flight reply
  // checks this after awaiting; if it changed, the reply is stale and
  // gets dropped instead of appended to the new conversation.
  let sendGen = 0;

  // Difficulty adaptation tracking (feature #3).
  let englishReplyCount = 0;
  let comfortableReplyCount = 0;
  let turnsSinceAdaptation = 0;
  const ADAPT_EVERY = 3;   // inject a system nudge every N learner turns

  function setBusy(b) {
    busy = b;
    sendBtn.disabled = b;
    textarea.disabled = b;
    if (voiceMicBtn) voiceMicBtn.disabled = b;
    if (scaffoldBtn) scaffoldBtn.disabled = b;
  }

  topicsEl.querySelectorAll('.lg-co-topic').forEach(btn => {
    btn.addEventListener('click', () => startTopic(TOPICS.find(t => t.id === btn.dataset.id)));
  });
  overlay.querySelector('#lg-co-newtopic').addEventListener('click', () => {
    sendGen += 1;
    topic = null;
    history = [];
    englishReplyCount = 0;
    comfortableReplyCount = 0;
    turnsSinceAdaptation = 0;
    threadEl.innerHTML = '';
    topicsEl.style.display = '';
    inputEl.hidden = true;
    voiceEl.hidden = true;
    scaffoldRow.hidden = true;
    setBusy(false);
  });

  // ── "What could I say?" scaffold (feature #2) ────────────────────────
  scaffoldBtn && scaffoldBtn.addEventListener('click', async () => {
    if (busy || !topic) return;
    scaffoldBtn.disabled = true;
    scaffoldBtn.textContent = '💡 Thinking…';
    try {
      // Ask the LLM for ONE short example reply at the learner's level.
      // We include the current history so the suggestion is contextual.
      const scaffoldMsgs = [
        ...history,
        {
          role: 'user',
          content: `(OOC instruction — do NOT add this to the conversation)
Suggest ONE short reply (5-10 words) that I could say next in ${lang}, appropriate for a beginner-intermediate learner. Just the ${lang} sentence, nothing else — no translation, no explanation.`,
        },
      ];
      const suggestion = (await llmChat(scaffoldMsgs)).trim();
      if (suggestion) {
        textarea.value = suggestion;
        textarea.focus();
        // Select all so the learner can see it's editable and replace easily.
        textarea.select();
      }
    } catch { /* degrade silently */ }
    scaffoldBtn.disabled = false;
    scaffoldBtn.textContent = '💡 What could I say?';
  });

  async function startTopic(t) {
    if (busy) return;
    topic = t;
    topicsEl.style.display = 'none';
    if (inputMode === 'voice') {
      voiceEl.hidden = false;
      inputEl.hidden = true;
      scaffoldRow.hidden = true;
    } else {
      inputEl.hidden = false;
      voiceEl.hidden = true;
      scaffoldRow.hidden = false;
    }
    setBusy(true);

    const knownSurfaces = pool.map(c => c.surface).slice(0, 30);
    const knownGlosses = pool.flatMap(c => (c.glosses || []).slice(0, 1)).slice(0, 20);
    const sys = `You are a patient, warm conversation partner for a language learner studying ${lang}. You speak primarily in ${lang} with short, natural turns (1-3 sentences). Constraints:

- Vocabulary: prefer words the learner already knows. Their known words include: ${knownSurfaces.join(', ')}. They roughly mean: ${knownGlosses.join(', ')}.
- You may introduce 1-2 new, common words per turn. Don't dump vocab.
- If they make a grammar mistake, do a gentle RECAST in your reply (rephrase correctly without saying "you said X wrong") — do not correct them out loud.
- If they reply in English, gently echo what they said back in ${lang} and continue.
- Never produce a translation unless they explicitly ask.
- Keep every reply under 30 words. No lists. No formatting.
- Scene: ${t.seed}

Begin the conversation now — greet the learner and ask a simple opening question in ${lang}.`;

    history = [{ role: 'system', content: sys }];
    const stream = makeStreamingBubble('partner');
    const gen = ++sendGen;
    const reply = (await llmChatStream(history, (_d, full) => {
      if (gen === sendGen) stream.setText(full);
    })).trim();
    if (gen !== sendGen) { stream.discard(); return; }
    if (!reply) {
      await stream.finalize('(no opening — try another topic)', false);
    } else {
      history.push({ role: 'assistant', content: reply });
      await stream.finalize(reply, false);
      if (inputMode === 'voice' && voice) {
        speakWord(reply, voice).catch(() => {});
      }
    }
    setBusy(false);
  }

  textarea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  sendBtn.addEventListener('click', send);

  async function send() {
    if (busy) return;
    const text = textarea.value.trim();
    if (!text) return;
    textarea.value = '';
    await submitUserText(text);
  }

  // Detect whether the learner's reply is primarily English.
  // Heuristic: count ASCII-only words; if they dominate the token count it's
  // probably English (good enough for the adaptation nudge, not used for grading).
  function _looksEnglish(text) {
    const words = text.trim().split(/\s+/).filter(Boolean);
    if (words.length === 0) return false;
    const ascii = words.filter(w => /^[A-Za-z']+$/.test(w)).length;
    return ascii / words.length > 0.65;
  }

  function _masteryRank(card) {
    const numeric = Number(card?.mastery);
    if (Number.isFinite(numeric) && numeric > 0) return numeric;
    switch (card?.mastery_state) {
      case 'mature': return 4;
      case 'reviewing': return 3;
      case 'learning': return 2;
      case 'leech': return 1;
      case 'new':
      default: return 0;
    }
  }

  // Feature #4: credit FSRS for pool words that appear in the learner's output.
  // Guard: only exact surface matches, only well-known pool words (mastery >= 2),
  // and only if the reply has >= 2 target-language tokens (guard against STT
  // hallucination / trivial one-word responses).
  async function _creditProducedWords(text) {
    try {
      const tokens = text.trim().split(/[\s\p{P}]+/u).filter(Boolean);
      if (tokens.length < 2) return;   // too short to be meaningful
      const graded = new Set();
      for (const tok of tokens) {
        const lower = tok.toLowerCase();
        if (!poolSurfaceSet.has(lower)) continue;
        const card = poolBySurface.get(lower);
        if (!card || !card.word_id) continue;
        if (_masteryRank(card) < 2) continue;   // only credit words they actually know
        if (graded.has(card.word_id)) continue;
        graded.add(card.word_id);
        const g = gradeForEffort({ correct: true, attempts: 1, ms: 1500 });  // treated as "Good"
        gradeCard(lang, card.word_id, g).catch(() => {});
      }
    } catch { /* non-critical */ }
  }

  // Feature #3: runtime difficulty adaptation.
  // Injects a hidden system message every ADAPT_EVERY turns to guide the
  // partner's register — simplify if struggling, stretch if comfortable.
  function _maybeAdaptDifficulty() {
    turnsSinceAdaptation += 1;
    if (turnsSinceAdaptation < ADAPT_EVERY) return;
    turnsSinceAdaptation = 0;

    let nudge = null;
    if (englishReplyCount >= 2) {
      // Learner fell back to English multiple times — simplify.
      nudge = `(Private coaching note — the learner has replied in English ${englishReplyCount} time(s) recently. Simplify your next few turns: shorter sentences, more familiar vocabulary, slower pace. Keep guiding them back to ${lang} gently.)`;
      englishReplyCount = 0;
    } else if (comfortableReplyCount >= 2) {
      // Learner is comfortable — introduce a small stretch.
      const stretchPool = pool.filter(c => _masteryRank(c) >= 3).map(c => c.surface).slice(0, 5);
      const stretch = stretchPool.length ? stretchPool.join(', ') : '';
      nudge = `(Private coaching note — the learner is doing well. Introduce 1-2 slightly more complex structures or a new useful word${stretch ? ` (consider: ${stretch})` : ''}. Keep it natural — don't announce you're teaching.)`;
      comfortableReplyCount = 0;
    }

    if (nudge) {
      history.push({ role: 'system', content: nudge });
    }
  }

  // Feature #1: visible recast check — fires AFTER the partner reply renders
  // so it never blocks the conversation. Uses llmJudgeJSON with a timeout.
  async function _checkRecast(partnerReply, learnerTurn, msgWrapperEl) {
    if (!learnerTurn || !partnerReply) return;
    const gen = sendGen;   // capture current generation; drop if stale
    try {
      const result = await llmJudgeJSON(
        [
          {
            role: 'system',
            content: `You are a grammar-correction detector for a ${lang} language learner. Given a learner's turn and the conversation partner's reply, determine if the partner implicitly corrected a grammar or vocabulary error (a "recast") — rephrasing the learner's meaning correctly without explicitly saying "you made a mistake".

Output JSON: {"recasted": true/false, "original": "<what the learner said that was wrong>", "correction": "<the corrected form in ${lang}>", "note": "<one short English clause explaining the correction — e.g. 'verb should be conjugated as 3rd person singular'>"}

If no recast occurred, output: {"recasted": false}
JSON only. No prose.`,
          },
          {
            role: 'user',
            content: `Learner said: "${learnerTurn}"\nPartner replied: "${partnerReply}"`,
          },
        ],
        { fallback: null, timeoutMs: 7000 },
      );
      if (gen !== sendGen) return;   // conversation reset while we were waiting
      if (!result || !result.recasted) return;
      if (!result.original || !result.correction) return;

      // Find the partner bubble we annotate. It's the last partner message
      // currently in the thread.
      const bubbleMsg = threadEl.querySelector('.lg-co-msg-partner:last-of-type');
      if (!bubbleMsg) return;

      const annotation = document.createElement('div');
      annotation.className = 'lg-co-recast';
      annotation.innerHTML = `<div class="lg-co-recast-label">Gentle correction</div>
<div>
  <span class="lg-co-recast-original">${escapeHtml(result.original)}</span>
  <span class="lg-co-recast-arrow">→</span>
  <span class="lg-co-recast-correction">${escapeHtml(result.correction)}</span>
</div>
${result.note ? `<div class="lg-co-recast-note">${escapeHtml(result.note)}</div>` : ''}`;
      bubbleMsg.appendChild(annotation);
      threadEl.scrollTop = threadEl.scrollHeight;
    } catch { /* degrade silently — non-critical path */ }
  }

  async function submitUserText(text) {
    if (busy) return;
    if (!text) return;
    setBusy(true);
    addMessage('you', text);
    history.push({ role: 'user', content: text });
    turnsExchanged += 1;

    // Track engagement quality for adaptation (feature #3).
    if (_looksEnglish(text)) {
      englishReplyCount += 1;
      comfortableReplyCount = 0;
    } else {
      comfortableReplyCount += 1;
    }
    _maybeAdaptDifficulty();

    // Feature #4: FSRS credit for produced words (non-blocking).
    _creditProducedWords(text).catch(() => {});

    const stream = makeStreamingBubble('partner');
    const gen = ++sendGen;
    const reply = (await llmChatStream(history, (_d, full) => {
      if (gen === sendGen) stream.setText(full);
    })).trim();
    if (gen !== sendGen) { stream.discard(); return; }
    if (!reply) {
      // Pop the user turn so retry doesn't send "user, user, assistant".
      history.pop();
      await stream.finalize('(no reply — try again)', false);
    } else {
      history.push({ role: 'assistant', content: reply });
      await stream.finalize(reply, true);   // make clickable
      // Voice mode: auto-TTS the partner reply so it actually feels
      // like a conversation, not a chat-with-narration. Falls back
      // silently if no voice is wired.
      if (inputMode === 'voice' && voice) {
        speakWord(reply, voice).catch(() => {});
      }
      // Feature #1: fire the recast check after the reply renders.
      // Pass the learner's text (history.at(-2).content == user turn).
      const learnerTurn = history.length >= 2 ? history[history.length - 2].content : text;
      _checkRecast(reply, learnerTurn, null).catch(() => {});
    }
    setBusy(false);
  }

  async function recordOnce() {
    if (busy || transcribing) return;
    if (recorder && recorder.state === 'recording') {
      try { recorder.stop(); } catch { /* */ }
      return;
    }
    recChunks = [];
    try {
      recStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      voiceStatusEl.textContent = 'Mic access denied. Allow microphone in your browser.';
      return;
    }
    const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg', 'audio/mp4']
      .find(m => MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(m)) || '';
    recorder = new MediaRecorder(recStream, mime ? { mimeType: mime } : {});
    recorder.ondataavailable = (e) => { if (e.data && e.data.size > 0) recChunks.push(e.data); };
    recorder.onstop = async () => {
      transcribing = true;
      voiceMicBtn.classList.remove('lg-co-mic-recording');
      try { recStream.getTracks().forEach(t => t.stop()); } catch { /* */ }
      recStream = null;
      const blob = new Blob(recChunks, { type: recorder.mimeType || 'audio/webm' });
      voiceStatusEl.textContent = 'Transcribing…';
      let heard = '';
      let sttFailed = false;
      try {
        const fd = new FormData();
        fd.append('file', blob, 'reply.webm');
        fd.append('model', 'whisper-1');
        fd.append('language', lang);   // ISO-639-1 — whisper supports ja/es/zh/fr/ko
        const r = await fetch('/v1/audio/transcriptions', { method: 'POST', body: fd });
        if (r.ok) {
          heard = ((await r.json()).text || '').trim();
        } else if (r.status === 503) {
          // Feature #5: STT unavailable (service not running / no Whisper model).
          sttAvailable = false;
          sttFailed = true;
        } else {
          sttFailed = true;
        }
      } catch {
        sttFailed = true;
      }
      // Always restore the idle label — both success and failure paths
      // were leaving "Listening… tap to stop" up on the button.
      voiceMicBtn.querySelector('.lg-co-mic-label').textContent = 'Tap to speak';
      if (!heard) {
        if (sttFailed && !sttAvailable) {
          // Feature #5: actionable notice — switch to text or check setup.
          showNotice(overlay, 'Speech-to-text is not available. Switch to ⌨ Text mode, or enable a Whisper model in Settings → Audio.', { kind: 'warn' });
          // Automatically drop back to text mode so the learner isn't stuck.
          setInputMode('text');
        } else if (sttFailed) {
          voiceStatusEl.textContent = 'Transcription failed — tap to try again.';
        } else {
          voiceStatusEl.textContent = 'Didn\'t catch that — speak clearly and try again.';
        }
        transcribing = false;
        return;
      }
      voiceStatusEl.textContent = '';
      transcribing = false;
      await submitUserText(heard);
    };
    recorder.start();
    voiceMicBtn.classList.add('lg-co-mic-recording');
    voiceMicBtn.querySelector('.lg-co-mic-label').textContent = 'Listening… tap to stop';
    voiceStatusEl.textContent = '';
  }

  if (voiceMicBtn) voiceMicBtn.addEventListener('click', recordOnce);

  function makeStreamingBubble(side) {
    const el = document.createElement('div');
    el.className = `lg-co-msg lg-co-msg-${side} lg-co-msg-streaming`;
    el.innerHTML = `<div class="lg-co-msg-bubble"><span class="lg-co-stream-text"></span><span class="lg-co-dots"><i></i><i></i><i></i></span></div>`;
    threadEl.appendChild(el);
    threadEl.scrollTop = threadEl.scrollHeight;
    const textSpan = el.querySelector('.lg-co-stream-text');
    return {
      setText(s) {
        textSpan.textContent = s;
        threadEl.scrollTop = threadEl.scrollHeight;
      },
      async finalize(s, clickable) {
        el.classList.remove('lg-co-msg-streaming');
        // Stash the raw text so the breakdown handler can pass the
        // exact sentence to the LLM analyser (we can't reconstruct it
        // from textContent — that picks up the 🔊 button label too).
        el.dataset.fullText = s;
        const bubbleEl = el.querySelector('.lg-co-msg-bubble');
        const speak = side === 'partner' && voice
          ? `<button type="button" class="lg-co-speak" aria-label="Speak">🔊</button>`
          : '';
        let body;
        if (clickable) {
          body = await renderClickableAsync(s);
        } else {
          body = escapeHtml(s);
        }
        bubbleEl.innerHTML = body + speak;
        bubbleEl.querySelector('.lg-co-speak')?.addEventListener('click', () => speakWord(s, voice));
        threadEl.scrollTop = threadEl.scrollHeight;
      },
      discard() { el.remove(); },
    };
  }

  async function renderClickableAsync(text) {
    if (_CJK_TOKENIZE.has(lang)) {
      const bd = await fetchBreakdown(lang, text);
      const tokens = bd && Array.isArray(bd.tokens) ? bd.tokens : null;
      if (tokens && tokens.length) {
        const base = tokens.map(tok => {
          const t = tok.text || '';
          if (!t) return '';
          if (tok.matched) {
            // Feature #6: include reading annotation in CJK clickable tokens.
            const tEsc = escapeHtml(t);
            const reading = tok.reading || tok.pinyin || '';
            if (reading && reading !== t) {
              return `<ruby class="lg-co-ruby"><span class="lg-co-word" data-w="${tEsc}">${tEsc}</span><rt>${escapeHtml(reading)}</rt></ruby>`;
            }
            return `<span class="lg-co-word" data-w="${tEsc}">${tEsc}</span>`;
          }
          return escapeHtml(t);
        }).join('');
        return base;
      }
      // Breakdown failed (offline pack, etc.) — fall through to the
      // whitespace tokenizer rather than dropping click-to-define.
    }
    return renderClickable(text, true);
  }

  function addMessage(side, text) {
    const el = document.createElement('div');
    el.className = `lg-co-msg lg-co-msg-${side}`;
    el.dataset.fullText = text;
    const bubble = renderClickable(text, side === 'partner');
    const speak = side === 'partner' && voice
      ? `<button type="button" class="lg-co-speak" aria-label="Speak">🔊</button>`
      : '';
    el.innerHTML = `<div class="lg-co-msg-bubble">${bubble}${speak}</div>`;
    el.querySelector('.lg-co-speak')?.addEventListener('click', () => speakWord(text, voice));
    threadEl.appendChild(el);
    threadEl.scrollTop = threadEl.scrollHeight;
  }

  function renderClickable(text, makeClickable) {
    if (!makeClickable) return escapeHtml(text);
    // Tokenize FIRST on raw text, then escape per-token. The old order
    // (escape, then regex over the escaped string) would match the
    // entity bodies — `&amp;` got `amp` wrapped as a clickable word, and
    // `&#39;` even matched digits. Doing it this way keeps quotes,
    // angles, and ampersands rendering as their literal characters.
    const wordRe = /[\p{L}\p{N}]+/u;
    return text.split(/([\p{L}\p{N}]+)/u).map(part => {
      if (!part) return '';
      if (wordRe.test(part)) {
        return `<span class="lg-co-word" data-w="${escapeHtml(part)}">${escapeHtml(part)}</span>`;
      }
      return escapeHtml(part);
    }).join('');
  }

  // Per-bubble breakdown cache. Keyed by the bubble's full text so the
  // contextual analysis fires exactly once per unique reply, no matter
  // how many tokens the learner clicks. Cache survives the session.
  const _breakdownCache = new Map();
  // In-flight promise per bubble — prevents firing the LLM call twice
  // when the user rapid-clicks two tokens of the same bubble.
  const _breakdownInflight = new Map();

  async function _getBubbleBreakdown(bubbleEl, text) {
    if (_breakdownCache.has(text)) return _breakdownCache.get(text);
    if (_breakdownInflight.has(text)) return _breakdownInflight.get(text);
    bubbleEl.classList.add('lg-co-bubble-analysing');
    const p = (async () => {
      const result = await breakdownContextual(text, lang);
      _breakdownCache.set(text, result || []);
      _breakdownInflight.delete(text);
      bubbleEl.classList.remove('lg-co-bubble-analysing');
      return result || [];
    })();
    _breakdownInflight.set(text, p);
    return p;
  }

  function _findInBreakdown(breakdown, clickedSurface) {
    if (!Array.isArray(breakdown) || !clickedSurface) return null;
    const target = clickedSurface.toLowerCase();
    // Exact (case-insensitive), then containment either way — e.g.
    // user clicks "Te" inside the multi-word "Te gusta" token.
    return (
      breakdown.find(t => (t.token || '').toLowerCase() === target)
      || breakdown.find(t => (t.token || '').toLowerCase().split(/\s+/).includes(target))
      || breakdown.find(t => target.includes((t.token || '').toLowerCase()))
    );
  }

  threadEl.addEventListener('click', async (e) => {
    const tgt = e.target;
    if (!tgt.classList || !tgt.classList.contains('lg-co-word')) return;
    const word = tgt.dataset.w;
    if (!word) return;
    const bubbleEl = tgt.closest('.lg-co-msg-bubble');
    const msgEl = bubbleEl ? bubbleEl.closest('.lg-co-msg') : null;
    const fullText = msgEl ? (msgEl.dataset.fullText || msgEl.textContent || '') : word;
    showContextPop(tgt, word, fullText, bubbleEl);
  });

  async function showContextPop(anchor, clickedWord, fullText, bubbleEl) {
    document.getElementById('lg-co-pop')?.remove();
    const pop = document.createElement('div');
    pop.id = 'lg-co-pop';
    pop.className = 'lg-co-pop lg-co-pop-context';
    pop.innerHTML = `<div class="lg-co-pop-loading">Reading "${escapeHtml(clickedWord)}" in context…</div>`;
    document.body.appendChild(pop);
    const rect = anchor.getBoundingClientRect();
    pop.style.left = `${Math.min(rect.left, window.innerWidth - 380)}px`;
    pop.style.top = `${rect.bottom + 6}px`;

    setTimeout(() => {
      function offClick(e) {
        if (!pop.contains(e.target)) { pop.remove(); document.removeEventListener('mousedown', offClick); }
      }
      document.addEventListener('mousedown', offClick);
    }, 0);

    const breakdown = await _getBubbleBreakdown(bubbleEl, fullText);
    if (!Array.isArray(breakdown) || breakdown.length === 0) {
      // LLM didn't give us a usable breakdown — fall back to the
      // dictionary lookup we used to do. Better than nothing.
      _renderDictionaryFallback(pop, clickedWord);
      return;
    }

    const hit = _findInBreakdown(breakdown, clickedWord);
    const sentenceList = breakdown.map(t => {
      const isHit = hit && t === hit;
      return `<li class="lg-co-bd-row${isHit ? ' lg-co-bd-row-hit' : ''}">
        <span class="lg-co-bd-token">${escapeHtml(t.token || '')}</span>
        <span class="lg-co-bd-meaning">${escapeHtml(t.meaning || '')}</span>
        <span class="lg-co-bd-role">${escapeHtml(t.role || '')}</span>
      </li>`;
    }).join('');

    pop.innerHTML = `
      ${hit ? `
        <div class="lg-co-bd-focus">
          <div class="lg-co-bd-focus-head">
            <span class="lg-co-bd-focus-token">${escapeHtml(hit.token)}</span>
            <button type="button" class="lg-co-pop-speak" aria-label="Speak">🔊</button>
            <button type="button" class="lg-co-pop-close" aria-label="Close">×</button>
          </div>
          <div class="lg-co-bd-focus-meaning">${escapeHtml(hit.meaning || '')}</div>
          <div class="lg-co-bd-focus-role">${escapeHtml(hit.role || '')}${hit.lemma && hit.lemma !== hit.token ? ` · lemma: <i>${escapeHtml(hit.lemma)}</i>` : ''}</div>
          <div class="lg-co-bd-focus-actions">
            <button type="button" class="btn btn-primary lg-co-bd-add" data-surface="${escapeHtml(hit.lemma || hit.token)}">+ Add to my words</button>
            <button type="button" class="btn btn-ghost lg-co-bd-deep" data-lemma="${escapeHtml(hit.lemma || hit.token)}">Dictionary entry →</button>
          </div>
        </div>` : `
        <div class="lg-co-bd-focus">
          <div class="lg-co-bd-focus-head">
            <span class="lg-co-bd-focus-token">${escapeHtml(clickedWord)}</span>
            <button type="button" class="lg-co-pop-close" aria-label="Close">×</button>
          </div>
          <div class="lg-co-bd-focus-role">No contextual match found — showing full sentence below.</div>
        </div>`}
      <div class="lg-co-bd-divider"></div>
      <div class="lg-co-bd-label">Sentence breakdown</div>
      <ul class="lg-co-bd-list">${sentenceList}</ul>`;

    pop.querySelector('.lg-co-pop-close')?.addEventListener('click', () => pop.remove());
    pop.querySelector('.lg-co-pop-speak')?.addEventListener('click', () => {
      const speakWordSurface = hit?.token || clickedWord;
      speakWord(speakWordSurface, voice);
    });
    pop.querySelector('.lg-co-bd-deep')?.addEventListener('click', async (e) => {
      const lemma = e.currentTarget.dataset.lemma;
      _renderDictionaryFallback(pop, lemma);
    });
    // One-tap add — the docstring promises chatting is a vocab-acquisition
    // surface; routing every add through "Dictionary entry → + Add" was
    // two clicks too deep. Resolve the lemma to a word_id via lookup,
    // then call addWord.
    pop.querySelector('.lg-co-bd-add')?.addEventListener('click', async (e) => {
      const btn = e.currentTarget;
      const surface = btn.dataset.surface;
      btn.disabled = true;
      btn.textContent = 'Adding…';
      try {
        const r = await fetch(`/api/learning/lookup?lang=${encodeURIComponent(lang)}&q=${encodeURIComponent(surface)}`);
        const entries = r.ok ? (await r.json()).entries || [] : [];
        if (!entries.length) {
          btn.textContent = 'No dictionary entry';
          return;
        }
        const ok = await addWord(lang, entries[0].word_id);
        if (ok) {
          wordsAdded += 1;
          btn.textContent = '✓ Added';
          btn.classList.add('lg-co-bd-added');
        } else {
          btn.disabled = false;
          btn.textContent = 'Try again';
        }
      } catch {
        btn.disabled = false;
        btn.textContent = 'Try again';
      }
    });
  }

  async function _renderDictionaryFallback(pop, word) {
    pop.classList.remove('lg-co-pop-context');
    pop.innerHTML = `<div class="lg-co-pop-loading">Dictionary lookup for "${escapeHtml(word)}"…</div>`;
    try {
      const r = await fetch(`/api/learning/lookup?lang=${encodeURIComponent(lang)}&q=${encodeURIComponent(word)}`);
      const entries = r.ok ? (await r.json()).entries || [] : [];
      if (entries.length === 0) {
        pop.innerHTML = `<div class="lg-co-pop-empty">No dictionary entry for "${escapeHtml(word)}".</div>`;
        return;
      }
      const e = entries[0];
      pop.innerHTML = `
        <div class="lg-co-pop-head">
          <span class="lg-co-pop-surface">${escapeHtml(e.surface)}</span>
          <span class="lg-co-pop-reading">${escapeHtml(e.reading || '')}</span>
          <button type="button" class="lg-co-pop-speak" aria-label="Speak">🔊</button>
          <button type="button" class="lg-co-pop-close" aria-label="Close">×</button>
        </div>
        <div class="lg-co-pop-gloss">${escapeHtml((e.glosses || []).slice(0, 5).join(' · '))}</div>
        <button type="button" class="btn btn-primary lg-co-pop-add">+ Add to my words</button>`;
      pop.querySelector('.lg-co-pop-speak').addEventListener('click', () => speakWord(e.reading || e.surface, voice));
      pop.querySelector('.lg-co-pop-close').addEventListener('click', () => pop.remove());
      pop.querySelector('.lg-co-pop-add').addEventListener('click', async () => {
        await addWord(lang, e.word_id);
        wordsAdded += 1;
        pop.querySelector('.lg-co-pop-add').textContent = '✓ Added';
        setTimeout(() => pop.remove(), 800);
      });
    } catch {
      pop.innerHTML = `<div class="lg-co-pop-empty">Lookup failed.</div>`;
    }
  }
}
