/**
 * intent-action-router.js — Dispatch ``intent_action`` WS payloads to
 * the right frontend surface.
 *
 * The server-side action registry (augmentum/intent/) short-circuits
 * conversation-control + navigation utterances. When one fires, the
 * server sends an ``intent_action`` event with a ``surface`` block
 * naming a channel + payload. This module is the single place every
 * voice/text surface delegates to, so the same "open browse" handling
 * works whether the user said it via wake-PTT, the call modal, or the
 * future cast voice path.
 *
 * Channels are flat strings — keep them in sync with the surface_emit
 * values in augmentum/intent/builtin/. Unknown channels log + no-op.
 *
 * Surface openers are lazy-imported so this module doesn't pull every
 * panel into the initial bundle.
 */
import { ttsProgressiveCancel } from './chat/tts.js';
import { stopReadAloud } from './read-aloud.js';

// Lazy openers per surface id. Each returns a Promise so we can
// await sequencing in callers that care. Errors are caught + logged
// so a missing surface module doesn't break the router.
const _NAV_TARGETS = {
  browse:       () => import('./browse.js').then(m => m.openBrowsePanel?.()),
  notes:        async () => {
    const m = await import('./browse.js');
    m.openBrowsePanel?.();
    // Switch to notes tab via the existing event bridge.
    document.dispatchEvent(new CustomEvent('augmentum:switch-browse-tab',
      { detail: { tab: 'notes' } }));
  },
  discovery:    async () => {
    const m = await import('./browse.js');
    m.openBrowsePanel?.();
    document.dispatchEvent(new CustomEvent('augmentum:switch-browse-tab',
      { detail: { tab: 'discovery' } }));
  },
  files:        () => import('./files/index.js').then(m => m.openFiles?.()),
  // Coder is a MODE, not a panel — enter it through the same setMode
  // path the mode pills use. The old target called a nonexistent
  // ``openCoder`` and fell through to re-running module init on an
  // already-initialized coder.js, mounting duplicate transparent
  // chrome over the avatar stage (the "half-translucent VRM",
  // 2026-06-11).
  coder:        () => import('./app.js').then(m => m.setMode?.('coder')),
  library:      () => import('./library.js').then(m => m.openLibrary?.()),
  marketplace:  () => import('./marketplace.js').then(m => m.openMarketplace?.()),
  settings:     () => import('./settings.js').then(m => m.openSettings?.()),
  studio:       () => import('./studio.js').then(m => m.openStudio?.()),
  // Surfaces below don't have a clean exported opener yet; they fall
  // through to event dispatch so any module that listens can handle
  // them. Better to ship the chained event than silently no-op.
  grove:        () => _emit('navigate.open_surface', { surface: 'grove' }),
  today:        () => _emit('navigate.open_surface', { surface: 'today' }),
  observatory:  () => _emit('navigate.open_surface', { surface: 'observatory' }),
  agent:        () => _emit('navigate.open_surface', { surface: 'agent' }),
  voice:        () => {
    if (typeof window.__beccaTriggerVoiceCall === 'function') {
      window.__beccaTriggerVoiceCall();
    }
  },
  // "open youtube" — bare panel on its discover tab. Import first so
  // the module-level media:open-panel listener exists (the panel is
  // lazy-loaded; dispatching before import is a silent no-op).
  youtube:      () => import('./youtube-panel.js').then(() => {
    window.dispatchEvent(new CustomEvent('media:open-panel', {
      detail: { tab: 'discover' },
    }));
  }),
};

/**
 * Dispatch an ``intent_action`` payload. Returns true if a handler
 * was found for the channel.
 */
export function dispatchIntentAction(payload) {
  if (!payload || typeof payload !== 'object') return false;
  const surface = payload.surface;
  const speak = payload.speak;
  const toast = payload.toast;

  if (speak && typeof speak === 'string') {
    _speakAck(speak).catch(err => console.warn('[intent] speak failed', err));
  }
  if (toast && typeof toast === 'string') {
    try { window.__augmentum?.showToast?.(toast, 'info', 1800); } catch (_) {}
  }
  if (!surface || typeof surface !== 'object') {
    return Boolean(speak || toast);
  }

  const channel = surface.channel;
  const channelPayload = surface.payload || {};
  const handled = _routeChannel(channel, channelPayload);
  if (handled) {
    // becca-presence (Slice 1) subscribes to this for the inline
    // verb-tick toast in the status row. Single emission point: any
    // channel that's actually routed produces one user-visible "she
    // did the thing" tick.
    try {
      document.dispatchEvent(new CustomEvent('becca:verb-fired', {
        detail: {
          channel: String(channel || ''),
          label: typeof payload.tick_label === 'string' ? payload.tick_label : '',
        },
      }));
    } catch (_) { /* widget not mounted — silent */ }
  }
  return handled;
}

// Native bridge: the Android assist overlay runs its OWN voice turn (it can't
// reach this in-page router), so when that turn produces on-screen surface
// actions (play media, open a panel, etc.) the overlay hands them off to this
// WebView and calls this global to actually execute them — same dispatcher the
// in-page voice path uses. Each entry is a {channel, payload} surface event.
if (typeof window !== 'undefined') {
  window.__augReplaySurfaces = (surfaces) => {
    if (!Array.isArray(surfaces)) return 0;
    let n = 0;
    for (const sev of surfaces) {
      if (sev && typeof sev === 'object' && sev.channel) {
        try { if (dispatchIntentAction({ surface: sev })) n += 1; } catch (_) { /* ignore */ }
      }
    }
    return n;
  };
}

/**
 * Run a final transcript through the server's HTTPS companion turn
 * (``POST /api/voice/turn``). This is the cert-free agency seam: on-
 * device STT produces the text, the server runs the SAME model-driven
 * companion tool loop the web app uses (becca_direct → native_loop →
 * select_companion_tools), and the MODEL decides which verbs to fire
 * (open / play / note / app.act …) — no regex, no socket.
 *
 * Returns one of:
 *   { handled: true }     — the companion answered; her reply is spoken
 *                           + any surface events routed here. Caller does
 *                           nothing more (no separate chat send).
 *   { handled: false }    — companion path unavailable; caller sends the
 *                           transcript to chat as normal.
 *
 * Never throws — on any transport/parse error it returns
 * ``{ handled: false }`` so the caller always has a clean fall-through.
 *
 * @param {string} text  final transcript
 * @param {{sessionId?: string, surface?: string}} [opts]
 */
export async function runVoiceTurn(text, opts = {}) {
  const transcript = String(text || '').trim();
  if (!transcript) return { handled: false };
  let sid = opts.sessionId;
  if (sid == null) {
    try { sid = window.__augmentum?.state?.currentSessionId || ''; } catch (_) { sid = ''; }
  }
  try {
    const res = await fetch('/api/voice/turn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'same-origin',
      body: JSON.stringify({
        transcript,
        session_id: sid || '',
        surface: opts.surface || 'voice',
      }),
    });
    if (!res.ok) return { handled: false };
    const data = await res.json();
    if (data && data.handled) {
      // Speak her reply (on-device / HTTPS TTS) and route any surface
      // events through the same dispatcher the WS path uses.
      if (data.reply) {
        try { dispatchIntentAction({ speak: String(data.reply) }); } catch (_) { /* ignore */ }
      }
      const surfaces = Array.isArray(data.surfaces) ? data.surfaces : [];
      for (const sev of surfaces) {
        try { dispatchIntentAction({ surface: sev }); } catch (_) { /* ignore */ }
      }
      return { handled: true };
    }
    return { handled: false };
  } catch (err) {
    console.warn('[intent] voice/turn failed', err);
    return { handled: false };
  }
}

function _routeChannel(channel, payload) {
  switch (channel) {
    case 'tts.cancel':
      try { ttsProgressiveCancel?.(); } catch (_) {}
      try { stopReadAloud?.(); } catch (_) {}
      return true;
    case 'tts.repeat_last':
    case 'tts.resynth_last':
    case 'tts.volume_bump':
      _emit(channel, payload);
      return true;
    case 'conversation.close':
      _emit('conversation.close', payload);
      return true;
    case 'turn.abort':
      try { ttsProgressiveCancel?.(); } catch (_) {}
      _emit('turn.abort', payload);
      return true;
    case 'conversation.strike':
      // "Scratch that" — the server already popped the exchange from
      // the model's context; remove the matching bubbles from the voice
      // transcript so what she "forgot" disappears on screen too. Cancel
      // any in-flight TTS first: if she's mid-way through the reply
      // you're scratching, you don't want to keep hearing it.
      try { ttsProgressiveCancel?.(); } catch (_) {}
      import('./voice.js')
        .then(m => m.strikeLastExchangeUI?.())
        .catch(err => console.warn('[intent] strike UI failed', err));
      return true;
    case 'navigate.open_surface': {
      const fn = _NAV_TARGETS[payload?.surface];
      if (typeof fn === 'function') {
        Promise.resolve(fn()).catch(err =>
          console.warn('[intent] nav handler failed', payload?.surface, err),
        );
        return true;
      }
      console.warn('[intent] unknown surface', payload?.surface);
      return false;
    }
    case 'navigate.back':
      _emit('navigate.back', payload);
      return true;
    case 'device': {
      // Phone action (alarm/timer/dial/sms/open-app) — run it natively via the
      // Android bridge instead of a server→phone round-trip. Only works inside
      // the Android app; on desktop there's no phone to act on.
      try {
        const action = payload && payload.action;
        const bridge = window.AugmentumAndroid;
        if (action && bridge && typeof bridge.runDeviceAction === 'function') {
          bridge.runDeviceAction(String(action), JSON.stringify(payload.params || {}));
          return true;
        }
      } catch (err) {
        console.warn('[intent] device action failed', err);
      }
      return false;
    }
    case 'palette.run':
      // App menu (app.act): press a registered palette command by id.
      // runCommandById re-checks the `when` guard — if context moved
      // on since her dispatch, it declines with a toast, not a misfire.
      import('./command-palette.js')
        .then(m => m.runCommandById?.(payload?.command_id))
        .catch(err => console.warn('[intent] palette.run failed', err));
      return true;
    case 'note.open_sticky':
      import('./sticky-note.js').then(m => m.showSticky?.(payload))
        .catch(err => console.warn('[intent] sticky open failed', err));
      return true;
    case 'note.update_sticky':
      import('./sticky-note.js').then(m => m.updateSticky?.(payload))
        .catch(err => console.warn('[intent] sticky update failed', err));
      return true;
    case 'note.capture_started':
      import('./sticky-note.js').then(m => {
        m.showSticky?.(payload);
        m.setCaptureState?.(payload?.note_id, true);
      }).catch(err => console.warn('[intent] capture-start failed', err));
      return true;
    case 'note.capture_ended':
      import('./sticky-note.js')
        .then(m => m.setCaptureState?.(payload?.note_id, false))
        .catch(err => console.warn('[intent] capture-end failed', err));
      return true;
    case 'grove.play':
      // Architect's grove.play_matching primitive resolved a query and
      // wants us to play. Favourites get first shot; when the server
      // marks ``discover_ok`` (voice asks, assistant picks) a miss
      // falls through to station discovery — the catalog holds
      // hundreds of stations, so a genre ask lands on one of them
      // instead of a "no match" toast. Taste over apology.
      import('./grove.js').then(async m => {
        const query = payload?.query || payload?.label || '';
        const allowDiscover = payload?.discover_ok === true
          && typeof m.findAndPlayMatchingOrDiscover === 'function';
        const genreHints = Array.isArray(payload?.genre_hints)
          ? payload.genre_hints : [];
        const result = allowDiscover
          ? await m.findAndPlayMatchingOrDiscover(query, { genreHints })
          : m.findAndPlayMatching?.(query);
        if (!result || result.ok !== true) {
          const reason = result?.reason || 'no-match';
          const msg = reason === 'no-favourites' && !allowDiscover
            ? 'No favourite stations yet — pick one in Grove first.'
            : `Couldn't find a station for "${query}".`;
          try { window.__augmentum?.showToast?.(msg, 'info', 2200); } catch (_) {}
        } else if (result.station?.name) {
          // Tier label makes a wrong landing self-diagnosing: the
          // toast says WHERE the pick came from (favorites / files /
          // youtube / discover), matching the resolution ladder.
          const tier = {
            favorites: 'favorite', files: 'your library',
            youtube: 'YouTube', discover: 'radio',
          }[result.source] || '';
          try {
            window.__augmentum?.showToast?.(
              `♪ ${result.station.name}${tier ? ` · ${tier}` : ''}`,
              'info', 2600,
            );
          } catch (_) {}
        }
      }).catch(err => console.warn('[intent] grove.play failed', err));
      return true;
    case 'image.generate':
      // Architect's image.generate_with_defaults handoff. Image module
      // opens the panel, fills the form with the architect's inferred
      // settings, and clicks generate. The canonical VRAM check + abort
      // + progress UI all run through the existing path.
      import('./image.js').then(m => {
        const ok = m.generateFromArchitect?.(payload);
        if (!ok) {
          console.warn('[intent] image.generate handoff failed');
        }
      }).catch(err => console.warn('[intent] image.generate failed', err));
      return true;
    case 'image.search': {
      // Voice-found web images → native image viewer. image_search
      // downloads each hit to a local artifact (embed_url →
      // /api/artifacts/.../download), so the lightbox shows them like
      // any owned image. Open the top result; the rest are a follow-up
      // (no multi-image viewer exists yet).
      const imgs = Array.isArray(payload?.images) ? payload.images : [];
      const first = imgs[0];
      const url = first && (first.embed_url || first.download_url
        || first.url || first.source_url);
      if (!url) return false;
      import('./image.js').then(m => {
        m.openLightbox?.({ url, prompt: first.title || first.source || '' });
      }).catch(err => console.warn('[intent] image.search failed', err));
      return true;
    }
    case 'youtube.open': {
      // Voice-found video → the watch panel (transcript-synced player).
      // Direct hit plays; a search opens the panel on the top result.
      const p = payload || {};
      import('./youtube-panel.js').then(yt => {
        if (p.youtube_mode === 'direct' && p.video_id) {
          yt.openDirect?.(p);
        } else if (p.youtube_mode === 'search'
            && Array.isArray(p.results) && p.results.length) {
          const v = p.results[0];
          yt.openFromSearch?.(
            v.video_id || v.videoId || '',
            v.title || '', v.channel || v.author || '',
          );
        }
      }).catch(err => console.warn('[intent] youtube.open failed', err));
      return true;
    }
    case 'browse.search':
      // Architect's web.search primitive — open browse panel, run
      // the query through the existing search path.
      import('./browse.js').then(m => {
        const q = payload?.query || '';
        const cat = payload?.category || '';
        m.browseSearchByQuery?.(q, cat);
      }).catch(err => console.warn('[intent] browse.search failed', err));
      return true;
    case 'timer.set':
      // Architect's time.set_timer primitive — display a countdown
      // chip / toast. For Phase 1 we just toast the confirmation;
      // future enhancement is a persistent on-screen chip with
      // remaining-time display.
      try {
        const dur = payload?.duration_s || 0;
        const label = payload?.label || '';
        const mins = Math.round(dur / 60);
        const msg = label
          ? `Timer: ${mins} min — ${label}`
          : `Timer: ${mins} min`;
        window.__augmentum?.showToast?.(msg, 'info', 2200);
      } catch (err) { console.debug('[intent-router] timer toast failed', err); }
      return true;
    case 'media.resume': {
      // Architect's media.resume / media.play emissions. History: pre-
      // 2026-06-12 this called a resumeFromArchitect export that never
      // existed (every play died); the rewrite then branched on
      // content_kind and handed raw entries to activateFile, which
      // still silently no-op'd for media-server rows (missing stream
      // keys) and for anything the Files-grid resolver couldn't see
      // (2026-07-18 class fix). openContent is now the single opener:
      // it fetches the real entry, dispatches on ACTUAL kind (audio →
      // background mini-player, progress-safe; video/comic/book → the
      // canonical viewer cascade), and always lands somewhere honest.
      const fid = payload?.file_id || '';
      if (!fid) return true;
      import('./files/open-content.js')
        .then(m => m.openContent(fid, { label: payload?.content_label || '' }))
        .catch(err => console.warn('[intent] media.resume open failed', err));
      return true;
    }
    case 'browse.open_url':
      // Architect's browse.find hit — open the resolved URL.
      import('./browse.js').then(m => {
        const url = payload?.url || '';
        if (url) m.openInBrowse?.(url);
      }).catch(err => console.warn('[intent] browse.open_url failed', err));
      return true;
    case 'discovery.open':
      // Architect's discovery.show primitive — open the discovery
      // surface with optional kind / topic filters.
      import('./discovery.js').then(m => {
        const kind = payload?.kind || '';
        const topic = payload?.topic || '';
        if (typeof m.openDiscoveryWithFilter === 'function') {
          m.openDiscoveryWithFilter({ kind, topic });
        } else {
          // Fallback: just open discovery (the surface itself can
          // read query-string-ish hints from a global if it cares).
          if (typeof m.openDiscovery === 'function') m.openDiscovery();
          else _emit('navigate.open_surface', { surface: 'discovery' });
        }
      }).catch(err => console.warn('[intent] discovery.open failed', err));
      return true;
    case 'media.transport': {
      // Universal media transport — pause/next/previous dispatched by
      // the architect's media_control primitives. Three concurrent
      // playback systems exist (media-player audio, Grove music, ambient
      // YouTube). We resolve the *active* one by checking which has an
      // unpaused element, then call the matching method. media-player
      // wins ties because audiobook context is the higher-stakes case.
      const action = payload?.action || '';
      if (!action) return false;
      import('./media-player.js').then(async mp => {
        let routed = false;
        const audio = document.querySelector('audio');
        const mpActive = audio && !audio.paused;
        if (mpActive) {
          if (action === 'pause' && typeof mp.pause === 'function') {
            mp.pause(); routed = true;
          } else if (action === 'next' && typeof mp.skipChapterRelative === 'function') {
            mp.skipChapterRelative(1); routed = true;
          } else if (action === 'previous' && typeof mp.skipChapterRelative === 'function') {
            mp.skipChapterRelative(-1); routed = true;
          }
        }
        if (!routed) {
          // Try Grove next — its public surface is on the global since
          // it owns its own AudioContext lifecycle.
          try {
            const grove = await import('./grove.js');
            if (action === 'pause' && typeof grove.pauseGrove === 'function') {
              grove.pauseGrove(); routed = true;
            } else if (action === 'next' && typeof grove.nextGroveTrack === 'function') {
              grove.nextGroveTrack(); routed = true;
            } else if (action === 'previous' && typeof grove.previousGroveTrack === 'function') {
              grove.previousGroveTrack(); routed = true;
            }
          } catch (err) { console.debug('[intent-router] grove transport failed', err); }
        }
        if (!routed) {
          // Last resort — fire a global custom event so the ambient
          // video player (or any other future playback module) can
          // listen. Better than silent no-op.
          window.dispatchEvent(new CustomEvent('augmentum:media-transport', {
            detail: { action },
          }));
        }
      }).catch(err => console.warn('[intent] media.transport failed', err));
      return true;
    }
    case 'media.volume': {
      // media.volume's in-tab leg (wiring program Phase 1) — the
      // server already handled receiver casts; this fires only when
      // nothing is cast. Same foreground resolution as media.transport:
      // media-player's DOM audio element wins, then Grove, then a
      // global event for any future playback module.
      const direction = payload?.direction || '';
      const level = payload?.level;
      const STEP = 10; // percent — matches the server-side ladder step
      import('./media-player.js').then(async mp => {
        let routed = false;
        const audio = document.querySelector('audio');
        if (audio && !audio.paused) {
          if (direction === 'mute' || direction === 'unmute') {
            routed = mp.setMuted?.(direction === 'mute') ?? false;
          } else if (direction === 'set' && typeof level === 'number') {
            routed = mp.setVolume?.(level) ?? false;
          } else if (direction === 'up' || direction === 'down') {
            routed = mp.adjustVolume?.(direction === 'up' ? STEP : -STEP) ?? false;
          }
        }
        if (!routed) {
          try {
            const grove = await import('./grove.js');
            if (grove.isGrovePlaying?.()) {
              if (direction === 'mute' || direction === 'unmute') {
                routed = grove.setGroveMuted?.(direction === 'mute') ?? false;
              } else if (direction === 'set' && typeof level === 'number') {
                routed = grove.setGroveVolume?.(level) ?? false;
              } else if (direction === 'up' || direction === 'down') {
                routed = grove.adjustGroveVolume?.(direction === 'up' ? STEP : -STEP) ?? false;
              }
            }
          } catch (err) { console.debug('[intent-router] grove volume failed', err); }
        }
        if (!routed) {
          window.dispatchEvent(new CustomEvent('augmentum:media-volume', {
            detail: { direction, level },
          }));
        }
      }).catch(err => console.warn('[intent] media.volume failed', err));
      return true;
    }
    case 'media.adjust': {
      // Playback-rate and sleep-timer adjustments — media-player-only
      // concepts (Grove radio has no rate; its "sleep" is closing it).
      const action = payload?.action || '';
      import('./media-player.js').then(mp => {
        if (!mp.isActive?.()) {
          window.__augmentum?.showToast?.(
            'Nothing playing in the player right now', 'info', 2200,
          );
          return;
        }
        if (action === 'speed') {
          if (typeof payload?.rate === 'number') {
            mp.setSpeed?.(payload.rate);
          } else {
            const cur = mp.getState?.().speed || 1;
            const next = payload?.step === 'reset'
              ? 1
              : cur + (payload?.step === 'faster' ? 0.25 : -0.25);
            mp.setSpeed?.(next);
          }
        } else if (action === 'sleep_timer') {
          if (payload?.cancel) mp.setSleepTimer?.(0);
          else if (payload?.end_of_chapter) mp.setSleepTimer?.('end-of-chapter');
          else if (typeof payload?.minutes === 'number') mp.setSleepTimer?.(payload.minutes);
        }
      }).catch(err => console.warn('[intent] media.adjust failed', err));
      return true;
    }
    case 'game.launch': {
      // game.play's launch decision — a library game / ROM / app build.
      // Routed through the shared library dispatcher (launch-picker for
      // ROMs, game-surface for pinned web games, workspace for app
      // builds), the same implementation the Library's Open button uses.
      const aid = payload?.artifact_id || '';
      if (!aid) return true;
      import('./library/open-item.js')
        .then(m => m.openLibraryItemById(aid, { label: payload?.title || '' }))
        .catch(err => console.warn('[intent] game.launch failed', err));
      return true;
    }
    case 'livetv.tune': {
      // liveTV.play's tune decision — a live TV channel from the user's
      // Emby/JF server. Routes through the same POST /api/livetv/play
      // path as clicking a tile in the Files panel rails, opening the
      // HLS player overlay regardless of which panel is active.
      const sid = payload?.server_id || '';
      const cid = payload?.channel_id || '';
      const name = payload?.name || '';
      if (!sid || !cid) return true;
      import('./files/live-tv-rails.js')
        .then(m => m.playLiveTvChannel({ serverId: sid, channelId: cid, name }))
        .catch(err => console.warn('[intent] livetv.tune failed', err));
      return true;
    }
    case 'files.open':
      // Architect's files.find resolved a specific file — open it.
      // openContent fetches the full file_index row first; the old
      // path handed activateFile a synthetic {id, name, mime_type}
      // stub with no source_metadata, so kind detection ran on the
      // wrong field and media-server rows died in the silent guard
      // (same 2026-07-18 class as media.resume).
      import('./files/open-content.js').then(m => {
        const fileId = payload?.file_id || '';
        if (!fileId) return;
        m.openContent(fileId, { label: payload?.title || '' });
      }).catch(err => console.warn('[intent] files.open failed', err));
      return true;
    case 'files.search_open':
      // Architect's files.find fallback — open the files panel with the
      // user's query so they can pick. Files panel listens for the
      // dispatched event to pre-populate its search input.
      import('./files/index.js').then(m => {
        if (typeof m.openFiles === 'function') m.openFiles();
        document.dispatchEvent(new CustomEvent('augmentum:files-search', {
          detail: { query: payload?.query || '' },
        }));
      }).catch(err => console.warn('[intent] files.search_open failed', err));
      return true;
    case 'companion.candidates':
      // media.play's "offer" decision — near-tie matches rendered as
      // clickable cards anchored to the companion widget. Clicking a
      // card starts background playback; typing a follow-up works too
      // (the server parks the same candidates in the ReferentCache).
      import('./companion-candidates.js').then(m => {
        m.showCandidates?.(payload || {});
      }).catch(err => console.warn('[intent] candidates failed', err));
      return true;
    case 'companion.brief_open':
      // A completed coder run the companion delegated — open the brief panel
      // (live preview + diff + actions) so the user can act on it. Payload is
      // the coder_run_completed envelope
      // (jobs/handlers/coder_background_run.py::_emit_run_perception).
      import('./brief-panel.js').then(m => {
        m.openBrief?.(payload || {});
      }).catch(err => console.warn('[intent] brief_open failed', err));
      return true;
    case 'coder.delegate':
      // Acknowledgment for a queued background build (voice/typed pick). No
      // panel — the run's result opens the brief later. Returning true fires
      // the becca:verb-fired tick, which dismisses any lingering pick dock.
      return true;
    case 'coder.open_workspace':
      // "Take me there" after delegating a build — jump to the workspace.
      if (typeof window.openCoderWorkspace === 'function') {
        window.openCoderWorkspace(String(payload?.workspace_id || ''));
      } else {
        console.warn('[intent] coder.open_workspace: opener unavailable');
      }
      return true;
    case 'coder.new_workspace':
      // The companion delegated a build but no existing workspace fit (or the
      // user chose "New workspace"). Open the Coder create UI so the USER picks
      // the template/repo — never auto-select one. The prompt rides along.
      if (typeof window.openCoderNewWorkspace === 'function') {
        window.openCoderNewWorkspace(payload || {});
      } else {
        console.warn('[intent] coder.new_workspace: opener unavailable');
      }
      return true;
    case 'chat.new':
      // Slice 2 chat verb — Becca's "new chat" command. Reuses the
      // existing augmentum:new-session CustomEvent that chat/index.js
      // already listens for (same hook the Ctrl+Shift+S shortcut
      // uses). Zero new wiring on the chat side.
      document.dispatchEvent(new CustomEvent('augmentum:new-session'));
      return true;
    default:
      console.info('[intent] unhandled channel', channel, payload);
      return false;
  }
}

function _emit(name, payload) {
  document.dispatchEvent(new CustomEvent('augmentum:intent', {
    detail: { channel: name, payload },
  }));
}

async function _speakAck(text) {
  const trimmed = String(text || '').trim();
  if (!trimmed) return;
  try {
    const tts = await import('./chat/tts.js');
    // ttsPlayMessage is the one-shot speak path (fetch + play via the
    // audio bus at speech tier). This previously called a function
    // that never existed (ttsSingleShot) behind a typeof guard, so
    // every architect-dispatched action executed correctly and then
    // said NOTHING — "she did it but didn't answer" (2026-06-13).
    if (typeof tts.ttsPlayMessage === 'function') {
      // An intent_action ack is always the companion confirming an
      // action she just took — speak it in HER voice (ui.companionVoice),
      // not the generic read-aloud default. Without this flag a companion
      // action ("Searching the web…") came out in voiceDefaultVoice while
      // her conversational replies used the companion voice, so a single
      // turn switched voices mid-stream (2026-06-18). companion:true lets
      // ttsPlayMessage resolve companionVoice, falling back to the default
      // when it's unset (no behaviour change for users who never set one).
      await tts.ttsPlayMessage(trimmed, null, { companion: true });
    } else {
      console.warn('[intent] no TTS entry point for spoken ack');
    }
  } catch (err) {
    console.warn('[intent] speak ack failed', err);
  }
}
