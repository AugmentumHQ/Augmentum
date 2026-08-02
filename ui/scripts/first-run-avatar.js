/**
 * first-run-avatar.js — the "living avatar" first-run wow.
 *
 * Instead of a static welcome card, a first-time user is greeted by a 3D VRM
 * character that is already on screen and *speaks first* (bundled Kokoro TTS,
 * lip-synced). If the box has no chat model yet, the character asks the user to
 * "give it a mind" — a one-click download of a curated small model — and then
 * wakes up. Everything here wires primitives that already exist:
 *
 *   - avatar mount:   activateAvatarStandalone()          (avatar.js)
 *   - speech+lipsync: POST /api/audio/speech → AnalyserNode → avatarState
 *   - greeter choice: GET /api/avatar/bundled             (avatar_routes.py)
 *   - "give a mind":  POST /api/models/pull (engine) + SSE progress stream
 *   - load model:     openEngineLoadSheet()               (models.js)
 *
 * Falls back to the classic onboarding card if the avatar/voice path can't run
 * (no bundled avatars, no WebGL, TTS disabled) so nobody is ever stranded.
 *
 * Entry point is checkFirstRun(), called once from app.js on boot. Dismissal
 * persists server-side via /api/config/ui (onboarding_completed) — same flag
 * the classic onboarding uses, so the two never both fire.
 */
import { escapeHtml } from './app.js';

// Single AudioContext for the greeting clips. Created lazily on the first user
// gesture (the greeter pick) so autoplay policy doesn't block it.
let _audioCtx = null;
let _avatarMod = null;   // cached dynamic import of avatar.js
let _currentSource = null;

export async function checkFirstRun() {
  try {
    const res = await fetch('/api/config/ui');
    if (!res.ok) return _fallbackOnboarding();
    const data = await res.json();
    if (data.onboarding_completed === 'true' || data.onboarding_completed === true) return;
  } catch {
    return; // network hiccup on boot — don't block the app, try again next load
  }

  // Need at least one bundled avatar to greet with. Without it, there's no
  // living-avatar experience to show — use the classic card.
  let avatars = [];
  try {
    const r = await fetch('/api/avatar/bundled');
    if (r.ok) avatars = (await r.json()).avatars || [];
  } catch { /* fall through */ }
  const greeters = avatars.filter(a => a.vrm_url);
  if (!greeters.length) return _fallbackOnboarding();

  try {
    await _showLivingAvatar(greeters);
  } catch (err) {
    console.warn('[first-run] living avatar failed, falling back', err);
    _fallbackOnboarding();
  }
}

function _fallbackOnboarding() {
  import('./onboarding.js').then(m => m.checkOnboarding()).catch(() => {});
}

async function _hasChatBrain() {
  // "Brain" = a usable chat backend already present: an installed engine model
  // OR a connected provider (Ollama/LM Studio/etc.). If either exists this
  // isn't a cold start, so we skip the "give me a mind" download step.
  try {
    const [eng, status] = await Promise.all([
      fetch('/api/engine/v2/models').then(r => r.ok ? r.json() : null).catch(() => null),
      fetch('/api/models/status').then(r => r.ok ? r.json() : null).catch(() => null),
    ]);
    const engModels = (eng && (eng.models || eng.data)) || [];
    if (Array.isArray(engModels) && engModels.length) return true;
    if (Array.isArray(status) && status.length) return true;
    if (status && Array.isArray(status.providers) && status.providers.length) return true;
  } catch { /* treat as no brain */ }
  return false;
}

async function _showLivingAvatar(greeters) {
  const hasBrain = await _hasChatBrain();

  const overlay = document.createElement('div');
  overlay.className = 'firstrun-overlay';
  overlay.innerHTML = `
    <div class="firstrun-stage">
      <div class="firstrun-avatar-host" id="firstrun-host"></div>
      <div class="firstrun-subtitle" id="firstrun-subtitle"></div>
      <div class="firstrun-panel" id="firstrun-panel">
        <div class="firstrun-greeters">
          <div class="firstrun-prompt">Choose who greets you</div>
          <div class="firstrun-greeter-row">
            ${greeters.map((a, i) => {
              const letter = escapeHtml((a.name || '?').slice(0, 1));
              // A fresh instance has no rendered thumbnails — the endpoint
              // serves a 1x1 transparent placeholder (or 404s). Show the img
              // only if it's a real picture; onload/onerror fall back to the
              // letter avatar so cards are never blank.
              const face = a.thumbnail_url
                ? `<img src="${escapeHtml(a.thumbnail_url)}" alt="${escapeHtml(a.name || '')}" loading="lazy"
                       onload="if(this.naturalWidth<=1){this.style.display='none';this.nextElementSibling.style.display='flex'}"
                       onerror="this.style.display='none';this.nextElementSibling.style.display='flex'">
                   <span class="firstrun-greeter-fallback" style="display:none">${letter}</span>`
                : `<span class="firstrun-greeter-fallback">${letter}</span>`;
              return `
              <button class="firstrun-greeter" data-idx="${i}" title="${escapeHtml(a.name || 'Avatar')}">
                ${face}
                <span class="firstrun-greeter-name">${escapeHtml(a.name || 'Avatar')}</span>
              </button>`;
            }).join('')}
          </div>
        </div>
      </div>
      <button class="firstrun-skip" id="firstrun-skip">Skip intro</button>
    </div>
  `;
  document.body.appendChild(overlay);
  _injectStyles();

  overlay.querySelector('#firstrun-skip').onclick = () => _dismiss(overlay);

  overlay.querySelectorAll('.firstrun-greeter').forEach(btn => {
    btn.onclick = async () => {
      const idx = parseInt(btn.dataset.idx, 10);
      const chosen = greeters[idx];
      if (!chosen) return;
      // The click is our user gesture — safe to create/resume audio now.
      _ensureAudioCtx();
      await _beginGreeting(overlay, chosen, hasBrain);
    };
  });
}

async function _beginGreeting(overlay, avatar, hasBrain) {
  // Swap greeter picker → live stage.
  const panel = overlay.querySelector('#firstrun-panel');
  panel.innerHTML = '<div class="firstrun-loading">Waking the room…</div>';

  const host = overlay.querySelector('#firstrun-host');
  try {
    _avatarMod = await import('./avatar.js');
    const ok = await _avatarMod.activateAvatarStandalone({ host, vrmUrl: avatar.vrm_url });
    if (!ok) throw new Error('avatar mount returned false');
  } catch (err) {
    console.warn('[first-run] avatar mount failed', err);
    return _dismissTo(overlay, () => _fallbackOnboarding());
  }

  if (hasBrain) {
    await _speak(overlay,
      `Hi. I'm awake and listening. Say something, or just type — I'm all yours.`);
    _renderReady(overlay, /*justWokeUp=*/false);
  } else {
    await _speak(overlay,
      `Hi. I can see you and hear you — but I don't have a mind yet. ` +
      `Pick one from the list and I'll wake up.`);
    await _renderModelPicker(overlay);
  }
}

// ── Speech (lip-synced through the standalone avatar) ─────────────────────

function _ensureAudioCtx() {
  if (!_audioCtx) {
    try { _audioCtx = new (window.AudioContext || window.webkitAudioContext)(); }
    catch { _audioCtx = null; }
  }
  if (_audioCtx && _audioCtx.state === 'suspended') _audioCtx.resume().catch(() => {});
}

/**
 * Synthesize `text` with the bundled voice and play it through an AnalyserNode
 * bound to the standalone avatar's lip-sync. Resolves when playback ends (or
 * immediately-ish if audio is unavailable — the subtitle still shows).
 */
function _speak(overlay, text) {
  _setSubtitle(overlay, text);
  return new Promise(async (resolve) => {
    let settled = false;
    const done = () => { if (!settled) { settled = true; resolve(); } };
    // Hard cap so a silent/failed clip never blocks the flow.
    const guard = setTimeout(done, Math.min(1600 + text.length * 55, 12000));

    if (!_audioCtx) { return; /* guard resolves */ }
    try {
      const resp = await fetch('/api/audio/speech', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input: text, response_format: 'mp3' }),
      });
      if (!resp.ok) return; // TTS disabled/unavailable — subtitle carries it
      const buf = await resp.arrayBuffer();
      const audioBuf = await _audioCtx.decodeAudioData(buf);

      const source = _audioCtx.createBufferSource();
      source.buffer = audioBuf;
      const analyser = _audioCtx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      analyser.connect(_audioCtx.destination);

      // Bind to the avatar's lip-sync — the same field the call/companion
      // paths set. onTtsPlaybackChange drives the speaking animation state.
      try { _avatarMod.avatarState.analyserNode = analyser; } catch {}
      try { _avatarMod.onTtsPlaybackChange(true); } catch {}

      _currentSource = source;
      source.onended = () => {
        clearTimeout(guard);
        try { _avatarMod.avatarState.analyserNode = null; } catch {}
        try { _avatarMod.onTtsPlaybackChange(false); } catch {}
        if (_currentSource === source) _currentSource = null;
        done();
      };
      source.start();
    } catch (err) {
      console.warn('[first-run] speak failed', err);
      // guard resolves the promise; subtitle already shown
    }
  });
}

function _setSubtitle(overlay, text) {
  const el = overlay.querySelector('#firstrun-subtitle');
  if (el) el.textContent = text;
}

// ── "Give me a mind" — curated model pick + inline download ───────────────

async function _renderModelPicker(overlay) {
  const panel = overlay.querySelector('#firstrun-panel');
  panel.innerHTML = '<div class="firstrun-loading">Finding models that fit your machine…</div>';

  let models = [];
  try {
    const r = await fetch('/api/engine/catalog');
    if (r.ok) {
      const data = await r.json();
      models = (data.models || data.data || []).slice(0, 4);
    }
  } catch { /* handled below */ }

  if (!models.length) {
    // No catalog (offline / engine unavailable). Hand off to the tested
    // Model Manager so the user can still connect a provider or load a GGUF.
    panel.innerHTML = `
      <div class="firstrun-prompt">Give me a mind</div>
      <p class="firstrun-note">I couldn't reach the model catalog. Open the Model
        Manager to connect a provider (Ollama, LM Studio) or load a model you
        already have.</p>
      <button class="firstrun-btn primary" id="firstrun-open-mm">Open Model Manager</button>`;
    panel.querySelector('#firstrun-open-mm').onclick = () => {
      import('./models.js').then(m => m.openModelManager()).catch(() => {});
      _dismiss(overlay);
    };
    return;
  }

  panel.innerHTML = `
    <div class="firstrun-prompt">Give me a mind — pick one and I'll wake up</div>
    <div class="firstrun-model-grid">
      ${models.map((m, i) => {
        const name = m.name || m.id || m.model || 'model';
        const quant = m.default_quant || 'q4_k_m';
        const blurb = m.blurb || m.description || 'A compact model to get started.';
        return `
          <button class="firstrun-model" data-idx="${i}">
            <span class="firstrun-model-name">${escapeHtml(_prettify(name))}</span>
            <span class="firstrun-model-blurb">${escapeHtml(blurb)}</span>
            <span class="firstrun-model-tag">${escapeHtml(quant)}${m.size ? ' · ' + escapeHtml(String(m.size)) : ''}</span>
          </button>`;
      }).join('')}
    </div>
    <p class="firstrun-note">Runs entirely on your machine. You can add more or
      swap later — I'll still be me.</p>`;

  panel.querySelectorAll('.firstrun-model').forEach(btn => {
    btn.onclick = () => {
      const m = models[parseInt(btn.dataset.idx, 10)];
      if (m) _downloadModel(overlay, m);
    };
  });
}

async function _downloadModel(overlay, model) {
  const panel = overlay.querySelector('#firstrun-panel');
  const name = model.name || model.id || model.model || 'model';
  const quant = model.default_quant || 'q4_k_m';
  const combined = `${name}:${quant}`;

  // Mirror models.js pullModel()'s engine branch: split repo:filename only
  // when the tail is an explicit .gguf; otherwise the server resolves the quant.
  const body = { name: combined, backend: 'engine' };
  const colon = combined.lastIndexOf(':');
  if (colon > 0 && combined.slice(colon + 1).toLowerCase().endsWith('.gguf')) {
    body.name = combined.slice(0, colon);
    body.filename = combined.slice(colon + 1);
  }

  panel.innerHTML = `
    <div class="firstrun-prompt">Waking up…</div>
    <div class="firstrun-progress"><div class="firstrun-progress-bar" id="firstrun-bar"></div></div>
    <div class="firstrun-progress-label" id="firstrun-progress-label">Starting download…</div>`;
  _speak(overlay, `Downloading. Give me a moment to come online.`);

  let jobId = '';
  try {
    const resp = await fetch('/api/models/pull', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`pull HTTP ${resp.status}`);
    const data = await resp.json();
    jobId = data.job_id || '';
  } catch (err) {
    console.warn('[first-run] pull failed', err);
    return _downloadFailed(overlay, model);
  }

  if (!jobId) {
    // Some backends complete synchronously / dedupe an in-flight job.
    return _onModelReady(overlay, name);
  }

  const es = new EventSource(`/api/models/downloads/${encodeURIComponent(jobId)}/stream`);
  const bar = () => overlay.querySelector('#firstrun-bar');
  const label = () => overlay.querySelector('#firstrun-progress-label');
  es.onmessage = (ev) => {
    let p; try { p = JSON.parse(ev.data); } catch { return; }
    const pct = Math.round((p.progress || 0) * 100);
    if (bar()) bar().style.width = pct + '%';
    if (label()) label().textContent =
      `${escapeHtml(p.stage || 'downloading')} — ${pct}%`;
    if (p.status === 'completed') { es.close(); _onModelReady(overlay, name); }
    else if (p.status === 'failed' || p.status === 'cancelled') {
      es.close(); _downloadFailed(overlay, model);
    }
  };
  es.onerror = () => { es.close(); /* fall back to a ready check */ _onModelReady(overlay, name); };
}

function _downloadFailed(overlay, model) {
  const panel = overlay.querySelector('#firstrun-panel');
  panel.innerHTML = `
    <div class="firstrun-prompt">That download didn't finish</div>
    <p class="firstrun-note">You can retry, or open the Model Manager to pick
      another or connect a provider.</p>
    <div class="firstrun-btn-row">
      <button class="firstrun-btn" id="firstrun-retry">Try again</button>
      <button class="firstrun-btn primary" id="firstrun-open-mm2">Open Model Manager</button>
    </div>`;
  panel.querySelector('#firstrun-retry').onclick = () => _downloadModel(overlay, model);
  panel.querySelector('#firstrun-open-mm2').onclick = () => {
    import('./models.js').then(m => m.openModelManager()).catch(() => {});
    _dismiss(overlay);
  };
}

async function _onModelReady(overlay, downloadedName) {
  // Resolve the freshly-installed engine model's canonical name, then hand to
  // the tested load sheet so it lands in a serving slot correctly.
  let modelName = downloadedName;
  try {
    const r = await fetch('/api/engine/v2/models');
    if (r.ok) {
      const installed = (await r.json());
      const list = installed.models || installed.data || [];
      const hit = list.find(m => {
        const n = (m.name || m.id || '').toLowerCase();
        return n.includes(String(downloadedName).toLowerCase());
      });
      if (hit) modelName = hit.name || hit.id || modelName;
    }
  } catch { /* use the queued name */ }

  await _speak(overlay, `There we go — I'm awake. Load me in and let's talk.`);

  const panel = overlay.querySelector('#firstrun-panel');
  panel.innerHTML = `
    <div class="firstrun-prompt">I'm awake</div>
    <p class="firstrun-note">One tap loads me into memory so I can answer.</p>
    <button class="firstrun-btn primary" id="firstrun-load">Load ${escapeHtml(_prettify(modelName))}</button>`;
  panel.querySelector('#firstrun-load').onclick = async () => {
    try {
      const mm = await import('./models.js');
      // openEngineLoadSheet is the tested "pick offload + load into slot" UI.
      await mm.openEngineLoadSheet(modelName);
    } catch (err) {
      console.warn('[first-run] load sheet failed, opening model manager', err);
      import('./models.js').then(m => m.openModelManager()).catch(() => {});
    }
    _renderReady(overlay, /*justWokeUp=*/true);
  };
}

function _renderReady(overlay, justWokeUp) {
  const panel = overlay.querySelector('#firstrun-panel');
  panel.innerHTML = `
    <div class="firstrun-prompt">${justWokeUp ? 'Ready when you are' : 'Say hello'}</div>
    <div class="firstrun-btn-row">
      <button class="firstrun-btn primary" id="firstrun-talk"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 0-3 3v6a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v1a7 7 0 0 1-14 0v-1"/><line x1="12" y1="18" x2="12" y2="22"/><line x1="8" y1="22" x2="16" y2="22"/></svg>Start talking</button>
      <button class="firstrun-btn" id="firstrun-type">Type instead</button>
    </div>`;
  panel.querySelector('#firstrun-type').onclick = () => _dismiss(overlay);
  panel.querySelector('#firstrun-talk').onclick = () => {
    _dismiss(overlay);
    // Best-effort: nudge the existing voice UI open if it's present. Non-fatal.
    const mic = document.querySelector('#voice-toggle, #mic-btn, [data-action="voice"]');
    if (mic) { try { mic.click(); } catch {} }
  };
}

// ── Teardown ──────────────────────────────────────────────────────────────

function _dismissTo(overlay, next) {
  _dismiss(overlay);
  if (typeof next === 'function') next();
}

function _dismiss(overlay) {
  if (overlay.classList.contains('firstrun-dismissed')) return;
  overlay.classList.add('firstrun-dismissed');
  try { _currentSource && _currentSource.stop(); } catch {}
  _currentSource = null;
  // Tear down the standalone avatar so it doesn't fight the app's own mount.
  try { _avatarMod && _avatarMod.deactivateAvatar && _avatarMod.deactivateAvatar(); } catch {}
  setTimeout(() => overlay.remove(), 400);
  fetch('/api/config/ui', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ onboarding_completed: 'true' }),
  }).catch(() => {});
}

function _prettify(name) {
  return String(name)
    .replace(/[-_]/g, ' ')
    .replace(/\.gguf$/i, '')
    .replace(/\b\w/g, c => c.toUpperCase());
}

// ── Styles (scoped, injected once) ────────────────────────────────────────

function _injectStyles() {
  if (document.getElementById('firstrun-styles')) return;
  const style = document.createElement('style');
  style.id = 'firstrun-styles';
  style.textContent = `
    /* Themed with Augmentum design tokens so it follows dark/light/midnight/
       nord like the rest of the app — no hardcoded colors, no AI-demo gradient. */
    .firstrun-overlay{position:fixed;inset:0;z-index:var(--z-system-above,10000);
      display:flex;align-items:center;justify-content:center;
      background:var(--bg-primary);opacity:1;transition:opacity .4s ease;
      animation:firstrun-in .5s var(--ease-decelerate,ease-out) both}
    .firstrun-overlay.firstrun-dismissed{opacity:0;pointer-events:none}
    @keyframes firstrun-in{from{opacity:0}to{opacity:1}}
    .firstrun-stage{display:flex;flex-direction:column;align-items:center;
      gap:var(--space-md,16px);width:min(560px,92vw);max-height:94vh}
    .firstrun-avatar-host{width:min(360px,80vw);height:min(420px,52vh);
      border-radius:var(--radius-xl,20px);overflow:hidden;background:transparent}
    .firstrun-avatar-host canvas{width:100%!important;height:100%!important}
    .firstrun-subtitle{min-height:1.4em;color:var(--text-primary);font-size:1.05rem;
      text-align:center;line-height:1.4;max-width:36ch;
      text-shadow:0 1px 12px var(--bg-primary)}
    .firstrun-panel{width:100%;display:flex;flex-direction:column;
      align-items:center;gap:var(--space-2h,12px)}
    .firstrun-prompt{color:var(--text-primary);font-size:1.1rem;font-weight:600;
      text-align:center}
    .firstrun-note{color:var(--text-secondary);font-size:var(--text-xs,.8125rem);
      text-align:center;max-width:42ch;margin:2px 0 0}
    .firstrun-greeter-row,.firstrun-model-grid{display:flex;gap:var(--space-1h,10px);
      flex-wrap:wrap;justify-content:center}
    .firstrun-greeter{display:flex;flex-direction:column;align-items:center;gap:6px;
      background:var(--bg-elevated);border:1px solid var(--border);
      border-radius:var(--radius-lg,16px);padding:var(--space-sm,8px);cursor:pointer;
      color:var(--text-primary);transition:border-color .15s,transform .15s;width:96px}
    .firstrun-greeter:hover{border-color:var(--accent);transform:translateY(-2px)}
    .firstrun-greeter img{width:72px;height:72px;object-fit:cover;
      border-radius:var(--radius-md,10px)}
    .firstrun-greeter-fallback{width:72px;height:72px;display:flex;align-items:center;
      justify-content:center;font-size:2rem;background:var(--bg-secondary);
      border-radius:var(--radius-md,10px);color:var(--text-secondary)}
    .firstrun-greeter-name{font-size:var(--text-xs,.8125rem);color:var(--text-secondary)}
    .firstrun-model{display:flex;flex-direction:column;gap:4px;text-align:left;
      background:var(--bg-elevated);border:1px solid var(--border);
      border-radius:var(--radius-lg,16px);padding:var(--space-2h,12px);cursor:pointer;
      color:var(--text-primary);transition:border-color .15s,transform .15s;
      width:min(240px,90vw)}
    .firstrun-model:hover{border-color:var(--accent);transform:translateY(-2px)}
    .firstrun-model-name{font-weight:600}
    .firstrun-model-blurb{font-size:var(--text-xs,.8125rem);color:var(--text-secondary)}
    .firstrun-model-tag{font-size:.72rem;color:var(--text-muted);margin-top:2px}
    .firstrun-btn{display:inline-flex;align-items:center;gap:8px;padding:10px 24px;
      border-radius:var(--radius-md,8px);font-size:.9rem;font-weight:600;cursor:pointer;
      border:1px solid var(--border);background:var(--bg-elevated);
      color:var(--text-primary);transition:opacity .15s,transform .15s,box-shadow .15s}
    .firstrun-btn:hover{opacity:.9}
    .firstrun-btn svg{width:16px;height:16px}
    .firstrun-btn.primary{background:var(--accent);color:var(--accent-contrast,#fff);
      border-color:var(--accent);
      box-shadow:0 2px 8px color-mix(in srgb,var(--accent) 30%,transparent)}
    .firstrun-btn.primary:hover{transform:translateY(-1px) scale(1.02);opacity:1;
      box-shadow:0 4px 16px color-mix(in srgb,var(--accent) 40%,transparent)}
    .firstrun-btn.primary:active{transform:scale(.98)}
    .firstrun-btn-row{display:flex;gap:var(--space-1h,10px);flex-wrap:wrap;
      justify-content:center}
    .firstrun-progress{width:min(300px,80vw);height:8px;background:var(--bg-elevated);
      border:1px solid var(--border);border-radius:var(--radius-full,6px);overflow:hidden}
    .firstrun-progress-bar{height:100%;width:0;border-radius:var(--radius-full,6px);
      background:var(--accent);transition:width .3s}
    .firstrun-progress-label{color:var(--text-secondary);font-size:var(--text-xs,.82rem)}
    .firstrun-loading{color:var(--text-secondary);font-size:.9rem}
    .firstrun-skip{position:absolute;top:18px;right:22px;background:none;border:none;
      color:var(--text-secondary);font-size:var(--text-xs,.85rem);cursor:pointer;
      transition:color .15s}
    .firstrun-skip:hover{color:var(--text-primary)}
  `;
  document.head.appendChild(style);
}
