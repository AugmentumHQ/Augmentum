/**
 * Shared read-aloud helper.
 *
 * Centralises the play/stop/abort state machine so every surface that
 * wants a read-aloud button (browse reader, notes, files preview,
 * library artifacts, dream entries, etc.) can share one call site and
 * one active-playback invariant. Only one read-aloud session can be
 * active at a time — starting a new one stops the previous, even
 * across surfaces. The chat TTS module is imported lazily so pages
 * that never trigger read-aloud don't pay for its bundle.
 *
 * Contract:
 *   readAloud(text, button?, opts?) — start (or stop, if already playing
 *     the same text from the same button). Button gets '.playing'
 *     class while active; cleared on stop or error. `opts` may carry
 *     {voice, speed, chunkMode} to override the global voice defaults.
 *   stopReadAloud() — cancel the active session if any.
 *   isReadAloudActive() — truthy while a session is playing.
 *
 * The text argument must already be prose-cleaned. Callers are
 * responsible for stripping their own chrome (browse uses
 * `_extractReadableText`, notes uses a markdown-to-prose pass, etc.)
 * because the right strip strategy is surface-specific.
 */

// Module-scoped state — singleton by design so starting a read-aloud
// from one surface cleanly interrupts one from another.
let _active = false;
let _paused = false;
let _abort = null;
let _activeBtn = null;
// Monotonic session token. Bumped synchronously the instant a new
// read-aloud is requested so concurrent/late-arriving starts (and their
// suspended `await` continuations) can detect they've been superseded
// and bail instead of installing a competing pipeline. This is what
// makes start atomic despite the lazy `import()` awaits inside readAloud.
let _startSeq = 0;
// Whether the ACTIVE voice supports real pause/resume (blob path). The
// streaming-WAV path can't pause, so the mini-player hides its Pause
// button rather than showing one that silently stops. Defaults true.
let _canPause = true;
// True while a read-aloud is being synthesized ON-DEVICE (Android phone-hosted
// voice via the AugmentumAndroid bridge) rather than the server TTS pipeline.
let _onDevice = false;
// Session metadata supplied by the caller (article title, source URL)
// so the docked mini-player can show what's playing and offer a jump-
// back affordance after the user navigates away.
let _activeSession = null;
const _subscribers = new Set();

function _snapshot() {
  return {
    active: _active,
    paused: _paused,
    canPause: _canPause,
    title: _activeSession?.title || '',
    sourceUrl: _activeSession?.sourceUrl || '',
  };
}

// Clear all session state and notify subscribers. Used for the pre-
// playback bailout paths (module load failed, no TTS provider) where
// there's no pipeline to cancel yet.
function _clearSession() {
  _active = false;
  _paused = false;
  _onDevice = false;
  if (_activeBtn) { _activeBtn.classList.remove('playing'); _activeBtn = null; }
  _abort = null;
  _activeSession = null;
  _canPause = true;
  _notify();
}

function _notify() {
  const snap = _snapshot();
  for (const fn of _subscribers) {
    try { fn(snap); } catch (err) { console.warn('[read-aloud] subscriber threw', err); }
  }
}

export function subscribeReadAloud(fn) {
  _subscribers.add(fn);
  try { fn(_snapshot()); } catch (err) { console.warn('[read-aloud] subscriber threw', err); }
  return () => _subscribers.delete(fn);
}

export function isReadAloudActive() {
  return _active;
}

export function isReadAloudPaused() {
  return _paused;
}

export function pauseReadAloud() {
  if (!_active || _paused) return;
  _paused = true;
  import('./chat/tts.js').then(m => m.pauseTtsPlayback?.()).catch(() => {});
  _notify();
}

export function resumeReadAloud() {
  if (!_active || !_paused) return;
  _paused = false;
  import('./chat/tts.js').then(m => m.resumeTtsPlayback?.()).catch(() => {});
  _notify();
}

export function stopReadAloud() {
  if (_abort) {
    try { _abort.abort(); } catch { /* already aborted */ }
  }
  if (_onDevice) {
    try { window.AugmentumAndroid?.stopSpeaking?.(); } catch { /* ignore */ }
    _onDevice = false;
  }
  // Fire-and-forget cancel on the chat TTS pipeline — safe if the
  // module hasn't been loaded yet.
  import('./chat/tts.js').then(m => m.ttsProgressiveCancel?.()).catch(() => {});
  _active = false;
  _paused = false;
  _canPause = true;
  _activeSession = null;
  if (_activeBtn) {
    _activeBtn.classList.remove('playing');
    _activeBtn = null;
  }
  _notify();
}

// Native (Android) invokes this when ON-DEVICE speech finishes naturally, so the
// read-aloud button + state reset without the user tapping stop.
if (typeof window !== 'undefined') {
  window.__augSpeakFinished = () => {
    if (!_onDevice) return;
    _onDevice = false;
    _active = false;
    _paused = false;
    _canPause = true;
    _activeSession = null;
    if (_activeBtn) { _activeBtn.classList.remove('playing'); _activeBtn = null; }
    _notify();
  };
}

/**
 * Start reading `text` aloud. Toggles stop if already reading from the
 * same button. Returns a Promise that resolves when playback ends
 * (naturally or via stop). Callers don't have to await unless they
 * want to chain behavior.
 */
export async function readAloud(text, button, opts = {}) {
  // Toggle off if the SAME button triggered us while already active.
  if (_active && _activeBtn && _activeBtn === button) {
    stopReadAloud();
    return;
  }
  // Different button — stop the previous one first so the visual
  // state stays consistent (newest wins).
  if (_active) stopReadAloud();

  text = (text || '').trim();
  if (!text) return;

  // ── Claim the singleton SYNCHRONOUSLY, before any await. ──
  // readAloud has lazy `import()` awaits below; without a synchronous
  // claim, two rapid clicks (or two surfaces firing in one tick) both
  // slip past the `if (_active)` guard above — which can't fire until
  // _active is set — and each installs its own pipeline, producing
  // overlapping audio that Stop can't fully kill. Setting _active +
  // a monotonic token NOW means a later starter supersedes us and our
  // own suspended continuation bails at the next checkpoint.
  const mySeq = ++_startSeq;
  _active = true;
  _paused = false;
  _onDevice = false;
  _canPause = true;
  _abort = new AbortController();
  _activeBtn = button || null;
  _activeBtn?.classList.add('playing');
  _activeSession = {
    title: (opts && typeof opts === 'object' && opts.title) || '',
    sourceUrl: (opts && typeof opts === 'object' && opts.sourceUrl) || '',
  };
  _notify();

  // True once a newer start has taken over (token bumped) or a stop
  // cleared us. Suspended continuations check this after every await.
  const _superseded = () => mySeq !== _startSeq || !_active;

  // Lazy-load the TTS pipeline.
  let tts;
  try {
    tts = await import('./chat/tts.js');
  } catch (err) {
    if (!_superseded()) {
      _clearSession();
      _showToast('Read-aloud unavailable — TTS module failed to load.', 'error');
    }
    return;
  }
  if (_superseded()) return;
  if (!tts.ttsProgressiveFeed || !tts.ttsProgressiveFinish) {
    _clearSession();
    _showToast('Read-aloud requires a configured TTS provider in Settings.', 'warning');
    return;
  }

  try {
    // Install the progressive TTS pipeline before feeding. Without
    // this, ttsProgressiveFeed is a silent no-op (the feeder guards
    // on `!_activeTtsBuffer || !_activeTtsPipeline`) — which is why
    // a read-aloud session could hit 200 OK for some warm-up request
    // yet produce no audible output.
    //
    // Settings are read lazily so we don't pull in settings.js until
    // a user actually clicks Listen. Falls back to a sentence-mode
    // buffer at 1.0× speed if the voice defaults aren't configured,
    // which lets the TtsAudioPipeline surface a clearer error from
    // its per-chunk fetch rather than silently doing nothing.
    let voice = '';
    let speed = 1.0;
    let chunkMode = 'sentence';
    try {
      const { getSettings } = await import('./settings.js');
      if (_superseded()) return;
      const s = getSettings?.() || {};
      voice = s.voiceDefaultVoice || '';
      speed = s.voiceSpeed || 1.0;
      chunkMode = s.voiceTtsChunking || 'sentence';
    } catch { /* settings not available — use defaults */ }
    if (_superseded()) return;
    // Per-call overrides win over the global voice defaults — lets the
    // EPUB/document reader bar pick its own voice + speed without
    // touching the user's chat-TTS preferences.
    if (opts && typeof opts === 'object') {
      if (opts.voice) voice = opts.voice;
      if (opts.speed && !isNaN(opts.speed)) speed = opts.speed;
      if (opts.chunkMode) chunkMode = opts.chunkMode;
    }

    // Tell subscribers (mini-player) whether Pause will actually work for
    // this voice — streaming-WAV engines can't pause, so the control is
    // hidden rather than shown as a no-op that silently stops playback.
    _canPause = tts.ttsVoiceSupportsPause ? tts.ttsVoiceSupportsPause(voice) : true;
    _notify();

    // On Android, a phone-hosted voice synthesizes ON-DEVICE via the native
    // bridge — offline, no server round-trip. Read-aloud state is already active
    // (set above); route to the bridge and let it drive the audio. Native calls
    // window.__augSpeakFinished when it's done so the button resets.
    const _vid = (voice || '').split('::').pop();
    if (_vid.indexOf('pockettts-local/') === 0 &&
        window.AugmentumAndroid && typeof window.AugmentumAndroid.speak === 'function') {
      // The native bridge has no pause, only stop.
      _onDevice = true;
      _canPause = false;
      _notify();
      try { window.AugmentumAndroid.speak(text, _vid); } catch (_) { /* ignore */ }
      return;
    }

    // Cancel any prior pipeline (chat auto-read can leave one live).
    tts.ttsProgressiveCancel?.();
    const buffer = new tts.TtsSentenceBuffer(chunkMode);
    const pipeline = new tts.TtsAudioPipeline(voice, speed, /*isNarrative=*/false, _activeBtn);
    tts._installActivePipeline(buffer, pipeline);

    // Chunk at sentence boundaries so the TTS engine gets natural
    // pauses and users can stop partway through without the whole
    // thing queuing up in one audio file.
    const sentences = text.match(/[^.!?\n]+[.!?]+|\S+\n|\S[^.!?\n]*$/g) || [text];
    for (const s of sentences) {
      if (_abort.signal.aborted) break;
      tts.ttsProgressiveFeed(s);
    }
    if (!_abort.signal.aborted) {
      tts.ttsProgressiveFinish();
    }

    // Wait for the pipeline to finish playing. `sealed` is set by
    // ttsProgressiveFinish (above) so we resolve only after the
    // pipeline has been told no more chunks are coming AND has
    // drained its queue and playback — otherwise a narrow race
    // could resolve before the first chunk starts fetching.
    await new Promise((resolve) => {
      const tick = () => {
        if (_abort?.signal.aborted) { resolve(); return; }
        if (pipeline.cancelled) { resolve(); return; }
        if (pipeline.sealed && !pipeline.playing && pipeline.queue.length === 0) {
          resolve();
          return;
        }
        setTimeout(tick, 150);
      };
      tick();
    });
  } catch (err) {
    if (!_abort?.signal?.aborted) {
      _showToast('Read-aloud stopped: ' + (err?.message || err), 'error');
    }
  } finally {
    // Only tear down if we're still the current session. A newer start
    // (newest-wins) already owns the singleton and must not be reset by
    // our stale continuation. On-device native playback is excluded too:
    // it returns early above while still speaking and resets via
    // window.__augSpeakFinished when the phone finishes.
    if (mySeq === _startSeq && !_onDevice) {
      _active = false;
      _paused = false;
      _canPause = true;
      _activeBtn?.classList.remove('playing');
      _activeBtn = null;
      _abort = null;
      _activeSession = null;
      _notify();
    }
  }
}

// ── One-shot note narration (shared dedup) ──────────────────────────
// Used by both the live notification path (chime → beat → speak) and the
// drawer-open auto-start, so a single briefing is never narrated twice.
// Keyed by the companion_journal note id, which both paths carry.
const _narratedNotes = new Set();

function _proseForNarration(text) {
  return String(text || '')
    .replace(/https?:\/\/\S+/g, '')   // sources live in chips, not speech
    .replace(/[ \t]{2,}/g, ' ')
    .trim();
}

/**
 * Narrate a task-result note exactly once per session (dedup by note id).
 * Returns true if it started a session, false if deduped or empty. The
 * actual synthesis runs server-side through the user's configured voice
 * provider (via the chat TTS pipeline readAloud drives).
 */
export function narrateNoteOnce(noteId, text, opts = {}) {
  const key = String(noteId == null ? '' : noteId);
  if (key && _narratedNotes.has(key)) return false;
  const prose = _proseForNarration(text);
  if (!prose) return false;
  if (key) _narratedNotes.add(key);
  readAloud(prose, opts.button, opts);
  return true;
}

/**
 * Tiny toast fallback — uses the global app.showToast when available
 * so styling stays consistent, falls back to console on early loads
 * (e.g. service-worker surfaces).
 */
function _showToast(msg, level = 'info') {
  try {
    const win = /** @type {any} */ (window);
    if (win.__augmentum?.showToast) {
      win.__augmentum.showToast(msg, level, 2500);
      return;
    }
  } catch { /* fall through */ }
  console.debug('[read-aloud]', level, msg);
}

// Stop any active playback on page unload / hide so we don't
// continue talking in a backgrounded tab.
if (typeof window !== 'undefined') {
  window.addEventListener('pagehide', () => { if (_active) stopReadAloud(); });
}
