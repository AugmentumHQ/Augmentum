// Android shell media bridge — makes web audio "travel" like the browser.
//
// Inside the Android app the web UI's audio (chat TTS, voice replies, media
// player, narration, learning games, …) is played by the WebView renderer,
// which Android freezes when the app leaves the foreground unless a
// media-playback foreground service is running — the exact mechanism Chrome
// uses for its lock-screen media card. WebView doesn't wire the Media Session
// web API to the system, so the native shell has to be told when audio is
// audible. This file is that single choke point:
//
//   web → native   AugmentumAndroid.webAudioState('playing'|'paused'|'idle', title)
//   native → web   window.__augMediaCommand('play'|'pause'|'toggle'|'stop')
//
// Coverage is deliberately at the platform level, not per-feature, so every
// current and future audio surface is included without individual wiring:
//   * HTMLMediaElement.prototype.play — catches <audio>/<video> INCLUDING
//     detached `new Audio(url)` clips (their events never reach document, so
//     document-level capture listeners would miss them — hence the patch).
//   * AudioBufferSourceNode.prototype.start — catches Web Audio playback
//     (streamed TTS PCM, voice-mode replies). Mic capture uses worklet/
//     media-stream nodes, never buffer sources, so listening doesn't count
//     as "playing". OfflineAudioContext renders are excluded.
//
// Limitation: same-document only — audio inside sandboxed iframes (ZIM
// reader pages) isn't observed.
//
// In a normal browser (no AugmentumAndroid bridge) this file is inert.
(function () {
  'use strict';

  const bridge = window.AugmentumAndroid;
  if (!bridge || typeof bridge.webAudioState !== 'function') return;

  // Grace before reporting idle: TTS plays sentence-by-sentence and the media
  // player crossfades tracks — brief silent gaps must not flap the native
  // foreground service (Android throttles rapid start/stop of services).
  const IDLE_GRACE_MS = 2500;
  // Debounce before reporting playing: the audio-unlock priming path plays a
  // 1-sample buffer on first user interaction; don't flash a media card for it.
  const START_DEBOUNCE_MS = 250;

  const playingEls = new Set();   // media elements currently in 'playing'
  const trackedEls = new WeakSet();
  const activeSrcs = new Set();   // started, not-yet-ended AudioBufferSourceNodes
  const seenCtxs = new WeakSet(); // contexts we watch for suspend/resume
  let pausedByCommand = [];       // media elements the lock-screen pause stopped
  let suspendedByCommand = [];    // AudioContexts the lock-screen pause suspended
  let commandPauseUntil = 0;      // 'pause' events inside this window are ours
  let lastState = 'idle';
  let idleTimer = null;
  let startTimer = null;

  function anythingPlaying() {
    if (playingEls.size > 0) return true;
    for (const src of activeSrcs) {
      try { if (src.context.state === 'running') return true; } catch (_) { /* ctx gone */ }
    }
    return false;
  }

  function report(state) {
    if (state === lastState) return;
    lastState = state;
    if (state === 'idle') { pausedByCommand = []; suspendedByCommand = []; }
    try { bridge.webAudioState(state, document.title || 'Augmentum'); } catch (_) { /* bridge gone */ }
  }

  function recompute() {
    const viaCommand = Date.now() < commandPauseUntil;
    if (anythingPlaying()) {
      if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
      if (lastState === 'playing' || startTimer) return;
      startTimer = setTimeout(() => {
        startTimer = null;
        if (anythingPlaying()) report('playing');
      }, START_DEBOUNCE_MS);
      return;
    }
    if (startTimer) { clearTimeout(startTimer); startTimer = null; }
    if (viaCommand) {
      // Paused from the lock-screen card — keep the session (and the card)
      // alive so the user can resume from there, exactly like the browser.
      if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
      report('paused');
      return;
    }
    // In-page stop/pause/ended — release the native session after a grace
    // period so back-to-back clips don't churn the foreground service.
    if (lastState === 'idle' || idleTimer) return;
    idleTimer = setTimeout(() => { idleTimer = null; report('idle'); }, IDLE_GRACE_MS);
  }

  function trackElement(el) {
    if (trackedEls.has(el)) return;
    trackedEls.add(el);
    el.addEventListener('playing', () => { playingEls.add(el); recompute(); });
    const gone = () => { playingEls.delete(el); recompute(); };
    el.addEventListener('pause', gone);
    el.addEventListener('ended', gone);
    el.addEventListener('emptied', gone);
    el.addEventListener('error', gone);
  }

  function watchContext(ctx) {
    if (!ctx || seenCtxs.has(ctx)) return;
    seenCtxs.add(ctx);
    // The app suspends/resumes contexts itself (autoplay unlock, speech bus);
    // active buffer sources only count while their context runs.
    try { ctx.addEventListener('statechange', recompute); } catch (_) { /* older impl */ }
  }

  const origPlay = HTMLMediaElement.prototype.play;
  HTMLMediaElement.prototype.play = function () {
    try { trackElement(this); } catch (_) { /* never break playback */ }
    return origPlay.apply(this, arguments);
  };

  if (window.AudioBufferSourceNode) {
    const origStart = AudioBufferSourceNode.prototype.start;
    AudioBufferSourceNode.prototype.start = function () {
      try {
        const offline = window.OfflineAudioContext && this.context instanceof OfflineAudioContext;
        if (!offline) {
          const node = this;
          activeSrcs.add(node);
          watchContext(node.context);
          node.addEventListener('ended', () => { activeSrcs.delete(node); recompute(); }, { once: true });
          recompute();
        }
      } catch (_) { /* never break playback */ }
      return origStart.apply(this, arguments);
    };
  }

  // Transport commands from the native media session (lock-screen card,
  // notification buttons, audio-focus loss).
  window.__augMediaCommand = function (cmd) {
    cmd = String(cmd || '').toLowerCase();
    if (cmd === 'toggle') cmd = lastState === 'playing' ? 'pause' : 'play';
    if (cmd === 'pause') {
      commandPauseUntil = Date.now() + 800;
      for (const el of playingEls) {
        try { el.pause(); pausedByCommand.push(el); } catch (_) { /* element gone */ }
      }
      // Suspending the context freezes scheduled PCM in place (a real pause);
      // buffer sources can't be paused individually.
      const ctxs = new Set();
      for (const src of activeSrcs) {
        try { if (src.context.state === 'running') ctxs.add(src.context); } catch (_) { /* ctx gone */ }
      }
      for (const ctx of ctxs) {
        try { ctx.suspend(); suspendedByCommand.push(ctx); } catch (_) { /* already closed */ }
      }
      recompute();
    } else if (cmd === 'play') {
      const els = pausedByCommand; pausedByCommand = [];
      const ctxs = suspendedByCommand; suspendedByCommand = [];
      for (const ctx of ctxs) { try { ctx.resume(); } catch (_) { /* closed */ } }
      for (const el of els) { try { el.play(); } catch (_) { /* element gone */ } }
      recompute();
    } else if (cmd === 'stop') {
      for (const el of playingEls) { try { el.pause(); } catch (_) { /* gone */ } }
      for (const src of Array.from(activeSrcs)) { try { src.stop(); } catch (_) { /* not started */ } }
      activeSrcs.clear();
      playingEls.clear();
      if (idleTimer) { clearTimeout(idleTimer); idleTimer = null; }
      if (startTimer) { clearTimeout(startTimer); startTimer = null; }
      commandPauseUntil = 0;
      report('idle');
    }
  };
})();
