/* ==========================================================================
   Augmentum — Fullscreen Presence (lock-screen / native summon)
   --------------------------------------------------------------------------
   Renders ONLY the companion's idle VRM, fullscreen, with no chat chrome.
   Activated when the page is loaded with `?presence=1` — the Android
   CompanionPresenceActivity opens `/ui/?xrEmbed=1&presence=1` OVER the lock
   screen (resident-companion spec, Layer 3 / Spike 1).

   Reuses the standalone avatar pipeline (`activateAvatarStandalone` — same
   scene / animator / PresenceEngine as the floating companion widget). Idle
   breathing/blinking is self-contained client-side, so she stays alive on the
   lock screen without a live WebSocket. The VRM URL comes from the same
   `/api/avatar/for-session` route the widget uses.

   Also exposes `window.AugmentumAssist` hooks the native shell calls via
   `WebView.evaluateJavascript`: `showIdleAvatar()` and (summon) `startVoice()`.
   ========================================================================== */

import { activateAvatarStandalone, pauseAvatarRender, resumeAvatarRender } from './avatar.js';

// Ambient presence renders at a capped frame rate — idle breathing/blinking is
// slow, so ~30fps is visually identical to 60 but roughly halves GPU/CPU for an
// always-on surface (live wallpaper / lock screen).
const PRESENCE_FPS = 30;

function _presenceRequested() {
  try {
    return new URLSearchParams(window.location.search).get('presence') === '1';
  } catch (_) {
    return false;
  }
}

async function _resolveVrmUrl() {
  try {
    const r = await fetch('/api/avatar/for-session', { credentials: 'same-origin' });
    if (r.ok) {
      const j = await r.json();
      if (j && j.vrm_url) return j.vrm_url;
    }
  } catch (err) {
    console.warn('[presence] /api/avatar/for-session fetch failed', err);
  }
  return null;
}

// Subtle, neutral starfield behind the companion — the same deep palette as
// the voice-call backdrop (--voice-bg-deep #06060e), but STATIC: no canvas or
// animation, so an always-on wallpaper costs ~nothing. Injected once. The stars
// sit on the stage's own background, i.e. BEHIND the avatar (which renders into
// a transparent WebGL canvas appended as a child).
function _ensureStarfieldStyle() {
  if (document.getElementById('augmentum-presence-starry-style')) return;
  const style = document.createElement('style');
  style.id = 'augmentum-presence-starry-style';
  style.textContent = `
    #augmentum-presence-stage.presence-starry {
      background-color: var(--voice-bg-deep, #06060e);
      background-image:
        radial-gradient(1.6px 1.6px at 25px 35px, rgba(255,255,255,0.85), transparent 55%),
        radial-gradient(1.1px 1.1px at 95px 80px, rgba(210,225,255,0.65), transparent 55%),
        radial-gradient(1px 1px at 150px 45px, rgba(255,245,225,0.6), transparent 55%),
        radial-gradient(0.9px 0.9px at 60px 150px, rgba(255,255,255,0.5), transparent 55%),
        radial-gradient(1.3px 1.3px at 175px 165px, rgba(225,235,255,0.55), transparent 55%),
        radial-gradient(ellipse 130% 90% at 50% 32%, #151b35 0%, #0a0a16 48%, var(--voice-bg-deep, #06060e) 100%);
      background-size: 200px 200px, 200px 200px, 200px 200px, 200px 200px, 200px 200px, cover;
      background-repeat: repeat, repeat, repeat, repeat, repeat, no-repeat;
    }`;
  document.head.appendChild(style);
}

function _ensureStage() {
  _ensureStarfieldStyle();
  let el = document.getElementById('augmentum-presence-stage');
  if (!el) {
    el = document.createElement('div');
    el.id = 'augmentum-presence-stage';
    el.className = 'presence-starry';
    Object.assign(el.style, {
      position: 'fixed',
      inset: '0',
      width: '100vw',
      height: '100vh',
      zIndex: '2000',
      // Idle presence is non-interactive; the native shell owns dismiss/tap.
      pointerEvents: 'none',
    });
    document.body.appendChild(el);
  }
  return el;
}

let _started = false;
// Set by _ensureClock; lets the tap-to-talk engage phase the clock out.
let _fadeClock = null;

/** Load + render the default companion's idle VRM fullscreen. Idempotent. */
export async function showIdleAvatar() {
  if (_started) return true;
  const vrmUrl = await _resolveVrmUrl();
  if (!vrmUrl) {
    console.warn('[presence] no vrm_url; cannot show idle avatar');
    return false;
  }
  const ok = await activateAvatarStandalone({ host: _ensureStage(), vrmUrl, targetFps: PRESENCE_FPS });
  _started = !!ok;
  return ok;
}

export function initPresenceFullscreen() {
  if (!_presenceRequested()) return;
  // Native bridge hooks (assist summon + lock-screen presence). pause/resume
  // let the native shell stop ALL render work the instant the wallpaper is
  // hidden (screen off / app in front) and resume on return.
  window.AugmentumAssist = window.AugmentumAssist || {};
  window.AugmentumAssist.showIdleAvatar = showIdleAvatar;
  window.AugmentumAssist.pausePresence = () => {
    try { pauseAvatarRender(); } catch (err) { console.warn('[presence] pause failed', err); }
  };
  window.AugmentumAssist.resumePresence = () => {
    try { resumeAvatarRender(); } catch (err) { console.warn('[presence] resume failed', err); }
  };
  window.AugmentumAssist.startVoice = () => {
    try {
      return typeof window.__beccaTriggerVoiceCall === 'function'
        ? window.__beccaTriggerVoiceCall()
        : undefined;
    } catch (err) {
      console.warn('[presence] startVoice failed', err);
      return undefined;
    }
  };
  showIdleAvatar().then((ok) => {
    if (!ok) return;
    _ensureTapToTalk();
    const onDevice = window.AugmentumAndroid && typeof window.AugmentumAndroid.startDictation === 'function';
    // On Android, route native dictation transcripts into the client-side
    // send-and-speak loop (no WebSocket — works on a self-signed cert).
    if (onDevice) window.__augReceiveTranscript = _presenceSendTranscript;
    try {
      const p = new URLSearchParams(window.location.search);
      // showclock=1 (set by CompanionPresenceActivity, which covers the system
      // keyguard clock) draws our own time/date that fades as she takes focus.
      if (p.get('showclock') === '1') _ensureClock();
      // autovoice=1 (wake-word summon): only the web WS fallback auto-starts a
      // call here; the on-device loop is hold-to-talk (she's up, prompting).
      if (p.get('autovoice') === '1' && !onDevice) {
        setTimeout(() => { try { window.AugmentumAssist.startVoice(); } catch (_) {} }, 400);
      }
    } catch (_) { /* ignore */ }
  });
}

// A lock-screen-style time/date for the presence surface. Prominent at first,
// then phases out — automatically after the screen's been on a few seconds, or
// the moment the user engages (tap-to-talk) — so she takes the focus. Non-
// interactive so taps pass through to the tap-to-talk layer below.
function _ensureClock() {
  if (document.getElementById('augmentum-presence-clock')) return;
  const el = document.createElement('div');
  el.id = 'augmentum-presence-clock';
  Object.assign(el.style, {
    position: 'fixed', top: 'calc(env(safe-area-inset-top, 0px) + 52px)',
    left: '0', right: '0', zIndex: '2001', textAlign: 'center', pointerEvents: 'none',
    color: 'rgba(255,255,255,0.92)', textShadow: '0 1px 14px rgba(0,0,0,0.55)',
    fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif',
    transition: 'opacity 900ms ease', opacity: '1',
  });
  const time = document.createElement('div');
  Object.assign(time.style, { fontSize: '64px', fontWeight: '300', letterSpacing: '-1px', lineHeight: '1' });
  const date = document.createElement('div');
  Object.assign(date.style, { fontSize: '17px', fontWeight: '500', opacity: '0.85', marginTop: '9px' });
  el.appendChild(time);
  el.appendChild(date);
  document.body.appendChild(el);

  const render = () => {
    const now = new Date();
    let h = now.getHours();
    const m = String(now.getMinutes()).padStart(2, '0');
    const ampm = h >= 12 ? 'PM' : 'AM';
    h = h % 12 || 12;
    time.textContent = `${h}:${m} ${ampm}`;
    date.textContent = now.toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' });
  };
  render();
  const tick = setInterval(render, 10000);

  let faded = false;
  _fadeClock = () => {
    if (faded) return;
    faded = true;
    el.style.opacity = '0';
    setTimeout(() => { clearInterval(tick); el.remove(); }, 1000);
  };
  // Auto phase-out once the screen's been on a few seconds.
  setTimeout(() => { try { _fadeClock(); } catch (_) {} }, 7000);
}

let _presenceHint = null;

// Hands-free single listen turn for lock-screen / wake-word summons, where no
// touch reaches the hold-to-talk button. Wait for the surface to settle + mic
// to warm, start listening, then auto-stop after a bounded window so the turn
// completes without a gesture. Follow-up: native VAD endpointing in
// MoonshineSttService for natural end-of-speech instead of this fixed window.
const PRESENCE_AUTOLISTEN_MS = 7000;
function _presenceAutoListen(begin, finish) {
  setTimeout(() => {
    try { begin(); } catch (_) { /* ignore */ }
    setTimeout(() => { try { finish(); } catch (_) { /* ignore */ } }, PRESENCE_AUTOLISTEN_MS);
  }, 700);
}

// Hold-to-talk on the presence surface. On Android this drives the FULLY
// CLIENT-SIDE loop — no WebSocket, so it works on a self-signed cert: hold to
// talk → on-device Moonshine STT → send the text over HTTPS → her reply is read
// aloud (HTTPS / on-device TTS). The tap layer sits just below the avatar stage;
// the native dismiss button (Compose, above the WebView) still wins top-right
// taps. Inert in the live wallpaper (no touch forwarded).
function _ensureTapToTalk() {
  if (document.getElementById('augmentum-presence-tap')) return;
  const tap = document.createElement('div');
  tap.id = 'augmentum-presence-tap';
  Object.assign(tap.style, {
    // Above the avatar stage (z-2000, now an opaque starfield) so the hint is
    // visible. The layer is transparent except the hint, so the avatar shows
    // through, and it captures the hold-to-talk gestures.
    position: 'fixed', inset: '0', zIndex: '2001', touchAction: 'none',
    display: 'flex', alignItems: 'flex-end', justifyContent: 'center',
    paddingBottom: 'calc(env(safe-area-inset-bottom, 0px) + 84px)',
  });
  const hint = document.createElement('div');
  Object.assign(hint.style, {
    color: 'rgba(255,255,255,0.78)', font: '500 15px system-ui, -apple-system, sans-serif',
    background: 'rgba(0,0,0,0.34)', padding: '9px 18px', borderRadius: '999px',
    backdropFilter: 'blur(6px)', webkitBackdropFilter: 'blur(6px)',
    transition: 'opacity 250ms ease', opacity: '1',
  });
  tap.appendChild(hint);
  document.body.appendChild(tap);
  _presenceHint = hint;

  const bridge = window.AugmentumAndroid;
  const onDevice = bridge && typeof bridge.startDictation === 'function';

  if (onDevice) {
    hint.textContent = 'Hold to talk';
    let holding = false;
    const begin = (e) => {
      if (holding) return;
      if (e && e.preventDefault) e.preventDefault();
      holding = true;
      hint.textContent = 'Listening…';
      try { _fadeClock && _fadeClock(); } catch (_) { /* ignore */ }
      try { bridge.startDictation(); } catch (_) { /* ignore */ }
    };
    const finish = () => {
      if (!holding) return;
      holding = false;
      hint.textContent = 'Thinking…';
      // → native transcribes → window.__augReceiveTranscript (overridden below).
      try { bridge.stopDictation(); } catch (_) { /* ignore */ }
    };
    tap.addEventListener('pointerdown', begin);
    tap.addEventListener('pointerup', finish);
    tap.addEventListener('pointercancel', () => {
      holding = false; hint.textContent = 'Hold to talk';
      try { bridge.cancelDictation(); } catch (_) { /* ignore */ }
    });

    // Hands-free summon (lock-screen / "Hey Becca" wake): a secure keyguard
    // won't pass touch to this button, so when we're launched with autovoice=1
    // we auto-listen instead of waiting for a hold. Native mic needs no user
    // gesture (unlike web getUserMedia), so it works over the keyguard.
    try {
      if (new URLSearchParams(location.search).get('autovoice') === '1') {
        _presenceAutoListen(begin, finish);
      }
    } catch (_) { /* ignore */ }
  } else {
    // Web fallback (no native bridge): tap → the existing WS voice call.
    hint.textContent = 'Tap to talk';
    let started = false;
    tap.addEventListener('click', () => {
      if (started) return;
      started = true;
      hint.textContent = 'Listening…';
      setTimeout(() => { hint.style.opacity = '0'; }, 1400);
      try { _fadeClock && _fadeClock(); } catch (_) { /* ignore */ }
      try { window.AugmentumAssist.startVoice(); } catch (_) { /* ignore */ }
    });
  }
}

// The presence transcript handler: on-device STT text → the HTTPS
// companion turn (model-driven tool loop, full agency — she can open /
// play / note / app.act, not just chat). If the companion path is
// unavailable, fall through to a plain chat send with her reply spoken.
// Replaces the default composer-insert handler ONLY on the presence page.
// Cert-free throughout (HTTPS + on-device STT/TTS, no WebSocket).
function _presenceSendTranscript(text) {
  const t = (text || '').trim();
  // Diagnostic + visible-failure: surface what STT heard so an empty
  // transcribe (mic captured silence / decode miss) stops being silent.
  console.info('[presence] transcript received len=' + t.length + ' text=' + JSON.stringify(t));
  if (!t) {
    if (_presenceHint) {
      _presenceHint.textContent = 'Didn’t catch that — hold to talk';
      setTimeout(() => { if (_presenceHint) _presenceHint.textContent = 'Hold to talk'; }, 1800);
    }
    return;
  }
  // Echo the heard text briefly so the user sees she understood them.
  if (_presenceHint) {
    _presenceHint.textContent = '“' + (t.length > 44 ? t.slice(0, 44) + '…' : t) + '”';
  }
  const reset = () => {
    if (_presenceHint) setTimeout(() => { _presenceHint.textContent = 'Hold to talk'; }, 700);
  };
  const showError = (msg) => {
    if (_presenceHint) {
      _presenceHint.textContent = msg;
      setTimeout(() => { if (_presenceHint) _presenceHint.textContent = 'Hold to talk'; }, 2200);
    }
  };
  const sendToChat = () => {
    const input = document.getElementById('chat-input');
    if (input) {
      input.value = t;
      input.dispatchEvent(new Event('input', { bubbles: true }));
    }
    const click = () => {
      const btn = document.getElementById('send-btn');
      if (btn) btn.click();
      reset();
    };
    // Force her reply to be read aloud (client-side TTS over HTTPS — no socket).
    import('./settings.js')
      .then((m) => { try { m.getSettings().voiceAutoRead = true; } catch (_) { /* ignore */ } click(); })
      .catch(click);
  };
  // Companion turn first: her reply is spoken + surface events routed
  // inside runVoiceTurn, so we're done. Unavailable → plain chat send.
  import('./intent-action-router.js')
    .then((m) => m.runVoiceTurn(t, { surface: 'voice' }))
    .then((r) => {
      console.info('[presence] runVoiceTurn result=' + JSON.stringify(r));
      if (r && r.handled) { reset(); return; }
      // Not handled by the companion path — in presence (xrEmbed) there's
      // no composer to fall back to, so surface it instead of dead air.
      const input = document.getElementById('chat-input');
      if (input) { sendToChat(); } else { showError('Companion didn’t respond'); }
    })
    .catch((err) => {
      console.warn('[presence] runVoiceTurn error', err);
      const input = document.getElementById('chat-input');
      if (input) { sendToChat(); } else { showError('Couldn’t reach the server'); }
    });
}

// Self-init on import — no-ops unless `?presence=1`, so normal app loads are
// unaffected.
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPresenceFullscreen, { once: true });
  } else {
    initPresenceFullscreen();
  }
}
