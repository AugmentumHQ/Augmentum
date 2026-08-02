/**
 * cast-video.js — TV video player surface.
 *
 * Mirrors the audio surface pattern: rich on-TV UI (title, transport,
 * progress, status flags) with the phone as the transport. Adds the
 * pieces specific to video — subtitle track auto-injection, playback
 * speed, fullscreen letterbox.
 *
 * Patches accepted via postMessage from cast-receiver:
 *   {paused: bool}          — pause / play
 *   {position_s: number}    — absolute seek
 *   {seek_delta_s: number}  — relative skip
 *   {speed: number}         — 0.5..3.0 playback rate
 *   {subtitles: bool}       — show / hide the WebVTT track
 *   {volume: 0..1, muted: bool}
 */

import { effectiveDuration } from '../scripts/media-duration.js';

const params = new URLSearchParams(location.search);
const FILE_ID = (params.get('id') || '').trim();
const PROGRESS_POST_INTERVAL_MS = 10 * 1000;
const HUD_AUTO_HIDE_MS = 4000;

const $ = (sel) => document.querySelector(sel);
const elVideo = $('[data-cv-video]');
const elTitle = $('[data-cv-title]');
const elOverline = $('[data-cv-overline]');
const elSub = $('[data-cv-sub]');
const elElapsed = $('[data-cv-elapsed]');
const elDuration = $('[data-cv-duration]');
const elBarFill = $('[data-cv-bar-fill]');
const elState = $('[data-cv-state]');
const elStateMsg = $('[data-cv-state-msg]');
const elHudTop = $('[data-cv-hud-top]');
const elHudBottom = $('[data-cv-hud-bottom]');
const elFlagPaused = $('[data-cv-flag-paused]');
const elFlagSpeed = $('[data-cv-flag-speed]');
const elFlagCC = $('[data-cv-flag-cc]');

let _resumePosition = 0;
// Upstream-reported full runtime in seconds, captured from
// /api/media/details.duration_s on boot. Used as a floor for
// `elVideo.duration` — which, when the server is transcoding, only
// reflects the buffered transcode head and would otherwise produce
// a 0:00 / 0:32 seek bar on a 90-minute movie. See _effectiveDuration.
let _knownDurationS = 0;
// Sticky target speed. Some browsers silently reset playbackRate
// during buffering / readyState changes, so we re-apply on every
// significant lifecycle event until the actual rate matches.
let _targetPlaybackRate = 1.0;
// Sticky target subtitle index. -1 = off. We remember the user's
// choice so re-applying after a track loads / a seek doesn't lose it.
let _targetSubtitleIdx = -1;
// First-frame flag: after the very first frame paints, we never
// show the "Loading…" overlay again. Seeks / brief buffer drains
// would otherwise flash it on top of playable content and the
// user reads that as "the player got stuck."
let _firstFrameShown = false;
// Available subtitle tracks (index, label, language). Populated from
// /api/media/details on boot. Used by both the local attach loop and
// the patch handler when the controller picks an index by number.
let _subtitleTracks = [];


/* ── Util ─────────────────────────────────────────────────────── */

function fmtTime(s) {
  if (!s || !isFinite(s)) return '—:—';
  const total = Math.round(s);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}

// Absolute timestamp the current transcode segment begins at. Zero
// for direct-play streams (where the browser has the whole file's
// byte range). Becomes N when we re-issue the stream with
// ?start_time_s=N to skip past a buffered transcode head — from then
// on, `elVideo.currentTime` is the offset INTO the new segment, and
// the actual playhead position is `_baseOffsetS + elVideo.currentTime`.
let _baseOffsetS = 0;

// Absolute playhead position in the film, in seconds. Use this for
// every UI display, /progress POST, and surface_state echo — never
// `elVideo.currentTime` directly, because that's the position within
// the CURRENT segment (could be anywhere mid-film if we restarted
// the transcode from an offset).
function _effectiveCurrentTime() {
  const raw = Number(elVideo?.currentTime || 0);
  if (!Number.isFinite(raw) || raw < 0) return _baseOffsetS;
  return _baseOffsetS + raw;
}

// All duration consumers (label, seek-bar fill, surface_state echo,
// /progress POSTs) MUST go through this. Reading `elVideo.duration`
// directly returns the buffered transcode window during Emby/Jellyfin
// transcode and lies about the movie's real length.
function _effectiveDuration() {
  const raw = Number(elVideo?.duration);
  const segmentEnd = Number.isFinite(raw) && raw > 0 ? _baseOffsetS + raw : 0;
  return effectiveDuration(segmentEnd, _knownDurationS);
}

// Seek to an absolute timestamp in the film. Prefers an in-element
// seek (cheap, no rebuffer) when the target is inside the current
// stream's seekable range — covers direct-play AND within-segment
// seeks during transcode. Falls back to re-issuing the stream URL
// with ?start_time_s=N when the target is past seekable end: the
// path Emby/Jellyfin take to restart the transcode at a new offset.
//
// Without this, every forward seek past the buffered transcode head
// sets currentTime beyond `seekable.end`, which the browser silently
// clamps (Chrome snaps to 0, Safari snaps to end) — reading to the
// user as "the show restarted itself when I clicked forward."
function _seekTo(targetS) {
  const absoluteTarget = Math.max(0, Number(targetS) || 0);
  // Map the absolute target into the current segment's coordinates.
  // For direct-play (baseOffset=0), segmentTarget === absoluteTarget.
  const segmentTarget = absoluteTarget - _baseOffsetS;
  const seekableEnd = (elVideo.seekable && elVideo.seekable.length > 0)
    ? Number(elVideo.seekable.end(elVideo.seekable.length - 1) || 0)
    : 0;
  // 0.5s headroom: seeking to exactly seekable.end can park the
  // player at "stalled" on some browsers; pull just inside.
  if (segmentTarget >= 0 && segmentTarget <= seekableEnd - 0.5) {
    try { elVideo.currentTime = segmentTarget; } catch (err) {
      console.warn('[cast-video] in-segment seek failed', err);
    }
    return;
  }
  // Out of range — restart the stream from the absolute target. The
  // server forwards ?start_time_s to the upstream provider (Emby /
  // Jellyfin transcode endpoint, or a Range-based skip for direct-
  // play sources). Cache-buster prevents the browser from re-using
  // the prior response body, which would still start from baseOffset.
  console.log(`[cast-video] restart-from-N seek: ${absoluteTarget.toFixed(1)}s`);
  const wasPaused = !!elVideo.paused;
  _baseOffsetS = absoluteTarget;
  _firstFrameShown = false;
  setStateOverlay('Loading…');
  const qs = new URLSearchParams();
  qs.set('start_time_s', String(absoluteTarget));
  qs.set('_seek', String(Date.now()));
  elVideo.src = `/api/media/stream/${encodeURIComponent(FILE_ID)}?${qs}`;
  // The <video autoplay> attribute auto-plays the new src. Honour the
  // user's prior pause state by re-pausing once new media data lands;
  // without this, every seek-while-paused silently resumes playback.
  // Also re-assert element-level mute when the audio graph is live —
  // some browsers reset .muted on src swap, which would cause double
  // playback (graph output + element output) for ~one frame.
  const onPostSeekLoad = () => {
    if (_audioGraphMediaEl === elVideo) {
      try { elVideo.muted = true; } catch {}
    }
    if (wasPaused) {
      try { elVideo.pause(); } catch {}
    }
    elVideo.removeEventListener('loadeddata', onPostSeekLoad);
  };
  elVideo.addEventListener('loadeddata', onPostSeekLoad);
  if (!wasPaused) {
    elVideo.play().catch((err) => {
      console.warn('[cast-video] post-seek play rejected', err);
    });
  }
}

let _hudTimer = null;
function flashHud(ms = HUD_AUTO_HIDE_MS) {
  elHudTop.classList.add('on');
  elHudBottom.classList.add('on');
  clearTimeout(_hudTimer);
  _hudTimer = setTimeout(() => {
    // Keep the bottom HUD visible while paused so the user sees the
    // status. Top fades out either way.
    elHudTop.classList.remove('on');
    if (!elVideo.paused) elHudBottom.classList.remove('on');
  }, ms);
}

// Watchdog: even if no event ever clears the "Loading…" overlay
// (Chromium occasionally swallows loadeddata for some containers),
// auto-tear-down after this many milliseconds so the user is never
// stuck staring at a "Loading…" scrim over playable content.
const LOADING_OVERLAY_MAX_MS = 8000;
let _loadingWatchdog = null;

function setStateOverlay(text) {
  if (!text) {
    elState.hidden = true;
    if (_loadingWatchdog) { clearTimeout(_loadingWatchdog); _loadingWatchdog = null; }
    return;
  }
  // Once the first frame has shown, the user is looking at playable
  // content — surfacing "Loading…" on top of it during a seek or
  // brief stall is more confusing than helpful. Reserve the overlay
  // for actual errors past that point.
  if (_firstFrameShown && text === 'Loading…') return;
  elStateMsg.textContent = text;
  elState.hidden = false;
  // Only the "Loading…" state gets a watchdog — real errors should
  // stay up until something resolves them.
  if (text === 'Loading…') {
    if (_loadingWatchdog) clearTimeout(_loadingWatchdog);
    _loadingWatchdog = setTimeout(() => {
      // If we're still in "Loading…" 8s in, force-hide. If the video
      // truly never started, the user can tell from the black canvas
      // — a stuck "Loading…" overlay is the worse failure mode.
      if (!elState.hidden && elStateMsg.textContent === 'Loading…') {
        console.warn('[cast-video] loading overlay watchdog fired');
        elState.hidden = true;
        _firstFrameShown = true;  // suppress any future "Loading…"
      }
      _loadingWatchdog = null;
    }, LOADING_OVERLAY_MAX_MS);
  }
}


/* ── Metadata ─────────────────────────────────────────────────── */

async function fetchMetadata() {
  if (!FILE_ID) {
    setStateOverlay('No video id provided');
    return;
  }
  try {
    const r = await fetch(`/api/media/details/${encodeURIComponent(FILE_ID)}`, {
      credentials: 'same-origin',
    });
    if (!r.ok) {
      console.warn('[cast-video] details fetch returned', r.status);
      return;
    }
    const body = await r.json();
    applyMetadata(body);
  } catch (err) {
    console.warn('[cast-video] details fetch threw', err);
  }
}

function applyMetadata(body) {
  // /api/media/details returns provider-neutral fields at the top
  // level (current_time_s, duration_s, title, …). Some legacy paths
  // wrap them under .entry.source_metadata — we read both so the
  // shape doesn't have to be stable to land a resume.
  const entry = body.entry || body || {};
  const sm = entry.source_metadata || entry.metadata || {};
  // Upstream-reported full runtime. Captured BEFORE elVideo.duration
  // is reliable so the seek bar + duration label can floor on the
  // real length while the browser is still negotiating the
  // (possibly-transcoded) stream.
  _knownDurationS = Math.max(
    0,
    Number(body.duration_s ?? sm.duration_s ?? entry.duration_s ?? 0),
  );
  if (_knownDurationS > 0) {
    elDuration.textContent = fmtTime(_knownDurationS);
  }
  const title = body.title || entry.name || entry.title || sm.title || 'Untitled';
  const series = body.series_name || body.series || sm.series_name || sm.series || '';
  const season = body.season_number || sm.season_number || '';
  const episode = body.episode_number || sm.episode_number || '';

  document.title = `Augmentum · ${title}`;
  elTitle.textContent = title;

  const provider = body.source_provider || entry.source || sm.provider || 'video';
  elOverline.textContent = String(provider).replace(/_/g, ' ');

  if (series) {
    const bits = [series];
    if (season && episode) bits.push(`S${season}E${episode}`);
    else if (episode) bits.push(`Ep ${episode}`);
    elSub.textContent = bits.join(' · ');
  } else {
    elSub.textContent = (body.year || sm.year) ? String(body.year || sm.year) : '';
  }

  // Resume position — check TOP-LEVEL first (the shape /api/media/details
  // actually returns), then the legacy nested paths. Previous version
  // only checked the nested path so it always read undefined → 0 →
  // never seeked, which is the real reason resume kept appearing broken.
  const savedPos = parseFloat(
    body.current_time_s
    ?? sm.current_time_s
    ?? entry.current_time_s
    ?? 0,
  );
  if (isFinite(savedPos) && savedPos > 2) {
    _resumePosition = savedPos;
    console.log(`[cast-video] resume position resolved: ${savedPos.toFixed(1)}s`);
    // RACE FIX: if metadata is already loaded (cached video, src set
    // synchronously before fetchMetadata resolved), loadedmetadata
    // fired BEFORE _resumePosition was known — its seek branch saw 0
    // and did nothing. Apply the seek inline so the resume lands
    // regardless of which side of the race wins.
    const effDur = _effectiveDuration();
    if (elVideo.readyState >= 1 /* HAVE_METADATA */ && effDur > 0
        && savedPos < effDur - 1) {
      _seekTo(savedPos);
      console.log(`[cast-video] applied resume late: ${savedPos.toFixed(1)}s`);
      _resumePosition = 0;  // already applied; suppress the loadedmetadata branch
    }
  }

  // Build the full subtitle track list from media_sources. The
  // /api/media/details response nests this under ``body.playback``
  // (not the top level — reading the top level was the previous bug
  // that left every <track> element absent). Each selected source
  // has subtitle_tracks[{index, label, language, is_default}] with
  // index === -1 sentinel for "Off". We attach one <track> per
  // real track so the controller picker can switch among them
  // without round-trips.
  const playback = body.playback || body || {};
  const sources = playback.media_sources || body.media_sources || [];
  const selectedSourceId = playback.selected_media_source_id
    || body.selected_media_source_id
    || '';
  const selectedSource =
    sources.find((s) => s.id === selectedSourceId) || sources[0] || {};
  const tracks = (selectedSource.subtitle_tracks || [])
    .filter((t) => t && typeof t.index === 'number' && t.index >= 0);
  _subtitleTracks = tracks.map((t) => ({
    index: t.index,
    label: t.label || t.language || `Subtitle ${t.index}`,
    language: t.language || '',
    language_code: t.language_code || '',
    is_default: !!t.is_default,
    media_source_id: selectedSource.id || '',
  }));
  attachSubtitleTracks();
}

function attachSubtitleTracks() {
  // Remove prior tracks (re-runs are idempotent).
  Array.from(elVideo.querySelectorAll('track')).forEach((t) => t.remove());
  if (_subtitleTracks.length === 0) {
    elFlagCC.hidden = true;
    return;
  }
  for (const t of _subtitleTracks) {
    const track = document.createElement('track');
    track.kind = 'subtitles';
    track.label = t.label;
    if (t.language_code) track.srclang = t.language_code.slice(0, 2);
    const qs = new URLSearchParams({
      media_source_id: t.media_source_id,
      subtitle_stream_index: String(t.index),
    });
    track.src = `/api/media/subtitle/${encodeURIComponent(FILE_ID)}?${qs.toString()}`;
    track.default = false;
    track.dataset.trackIdx = String(t.index);
    elVideo.appendChild(track);
  }
  // Browsers expose textTracks asynchronously; disable all on a
  // microtask so cues don't briefly auto-show before the patch
  // handler picks a target.
  setTimeout(() => {
    for (let i = 0; i < elVideo.textTracks.length; i++) {
      elVideo.textTracks[i].mode = 'disabled';
    }
    // Re-apply the sticky target in case the user toggled CC on
    // before tracks finished attaching.
    reapplySticky();
  }, 0);
}


/* ── Player wiring ────────────────────────────────────────────── */

function startVideo() {
  if (!FILE_ID) return;

  // Attach listeners FIRST. Setting ``elVideo.src`` can fire
  // ``loadedmetadata`` / ``loadeddata`` SYNCHRONOUSLY when the
  // browser already has the bytes cached (range resume after a
  // pagehide post, Service Worker hit, etc.). With the old order
  // (src → setStateOverlay → addEventListener) those events landed
  // before any handler existed, so the overlay never got the hide
  // signal and stayed up until the 8s watchdog forced it down.
  elVideo.addEventListener('loadedmetadata', () => {
    const effDur = _effectiveDuration();
    elDuration.textContent = fmtTime(effDur);
    if (_resumePosition > 0
        && effDur > 0
        && _resumePosition < effDur - 1) {
      _seekTo(_resumePosition);
      console.log(`[cast-video] resumed at ${_resumePosition.toFixed(1)}s`);
    }
    _resumePosition = 0;
  });
  elVideo.addEventListener('loadeddata', () => {
    _firstFrameShown = true;
    setStateOverlay('');
    flashHud(6000);  // longer first-show so user sees title/duration
    // Re-apply sticky targets in case they were set before the video
    // was ready (silent rejection by the browser).
    reapplySticky();
  });
  elVideo.addEventListener('playing', () => {
    setStateOverlay('');
    reapplySticky();
  });
  elVideo.addEventListener('canplay', () => {
    setStateOverlay('');
    reapplySticky();
  });
  elVideo.addEventListener('timeupdate', () => {
    // timeupdate firing past t=0 is the strongest possible "this is
    // playing" signal — overlay must go regardless of which other
    // events the browser ate.
    if (elVideo.currentTime > 0) {
      _firstFrameShown = true;
      setStateOverlay('');
    }
    const pos = _effectiveCurrentTime();
    const dur = _effectiveDuration();
    elElapsed.textContent = fmtTime(pos);
    if (dur > 0) {
      elBarFill.style.width = `${Math.min(100, (pos / dur) * 100)}%`;
    }
    maybePostProgress();
    _echoSurfaceState({
      position_s: pos,
      duration_s: dur,
      paused: elVideo.paused,
    });
  });
  // Defensive belt: any byte arriving (progress / suspend) tears the
  // overlay down. Some streams fire these without ever reaching
  // loadeddata depending on container/codec.
  elVideo.addEventListener('progress', () => {
    if (elVideo.buffered && elVideo.buffered.length > 0
        && elVideo.buffered.end(0) > 0) {
      _firstFrameShown = true;
      setStateOverlay('');
    }
  });
  elVideo.addEventListener('play', () => {
    elFlagPaused.hidden = true;
    _echoSurfaceState({
      paused: false,
      position_s: _effectiveCurrentTime(),
      duration_s: _effectiveDuration(),
    }, { force: true });
  });
  elVideo.addEventListener('pause', () => {
    elFlagPaused.hidden = false;
    flashHud(99999);  // keep HUD up while paused
    postProgress({ force: true });
    _echoSurfaceState({
      paused: true,
      position_s: _effectiveCurrentTime(),
      duration_s: _effectiveDuration(),
    }, { force: true });
  });
  elVideo.addEventListener('ended', () => {
    elFlagPaused.hidden = false;
    flashHud(99999);
    postProgress({ force: true, finished: true });
  });
  elVideo.addEventListener('ratechange', () => {
    if (Math.abs(elVideo.playbackRate - 1.0) < 0.01) {
      elFlagSpeed.hidden = true;
    } else {
      elFlagSpeed.hidden = false;
      elFlagSpeed.textContent = `${elVideo.playbackRate.toFixed(2)}×`;
    }
  });
  elVideo.addEventListener('error', () => {
    setStateOverlay('Video failed to load');
  });

  // Listeners are all attached. NOW set the source + show the
  // loading scrim. The scrim only appears if the element isn't
  // already past HAVE_CURRENT_DATA — a fresh src usually drops
  // readyState to 0, but a re-mount of the same src can keep
  // existing buffers and skip every load event we'd normally hide on.
  elVideo.src = `/api/media/stream/${encodeURIComponent(FILE_ID)}`;
  if (elVideo.readyState < 2 /* HAVE_CURRENT_DATA */) {
    setStateOverlay('Loading…');
  } else {
    // Already playable — short-circuit to the "first frame shown"
    // state so no future "Loading…" appears either.
    _firstFrameShown = true;
  }
  // Defensive heartbeat: a few hundred ms after src is set, check
  // whether the player is past HAVE_CURRENT_DATA without any of our
  // hide-events firing (some Chromium versions on TV WebViews skip
  // both loadedmetadata and loadeddata when the response is a
  // 206 Partial Content with cached headers). One catch here is
  // cheaper than waiting on the 8s watchdog.
  setTimeout(() => {
    if (elVideo.readyState >= 2 && !_firstFrameShown) {
      _firstFrameShown = true;
      setStateOverlay('');
    }
  }, 500);
  setTimeout(() => {
    if (elVideo.readyState >= 2 && !_firstFrameShown) {
      _firstFrameShown = true;
      setStateOverlay('');
    }
  }, 1500);

  elVideo.play().catch((err) => {
    console.warn('[cast-video] autoplay rejected', err);
  });
}


/* ── Progress persistence ─────────────────────────────────────── */

let _lastProgressAt = 0;
let _lastProgressPos = 0;
let _lastStateEchoAt = 0;
const STATE_ECHO_INTERVAL_MS = 5000;

function maybePostProgress() {
  const now = Date.now();
  if (now - _lastProgressAt < PROGRESS_POST_INTERVAL_MS) return;
  postProgress();
}


/* ── Surface-state echo to cast-receiver shell ─────────────────── *
 * The shell forwards these as ``surface_state`` WS events so
 * controller UIs (cast-shelf, cast-control) can render scrubbers
 * and pause buttons that mirror reality. Throttled to match the
 * native-shortcut path in cast-receiver.js (every 5s for position
 * ticks; play/pause/duration changes fire immediately). */

function _echoSurfaceState(patch, { force = false } = {}) {
  if (!window.parent || window.parent === window) return;
  const now = Date.now();
  if (!force && now - _lastStateEchoAt < STATE_ECHO_INTERVAL_MS) return;
  _lastStateEchoAt = now;
  try {
    window.parent.postMessage(
      { type: 'augmentum.surface_state', state: patch },
      '*',
    );
  } catch (err) {
    console.warn('[cast-video] state echo failed', err);
  }
}

function postProgress({ force = false, finished = false } = {}) {
  if (!FILE_ID) return;
  // Both pos AND dur are in absolute-film coordinates. _baseOffsetS
  // gets folded into pos so the server records "watched up to t=N
  // in the original film" rather than "watched 5s of the just-
  // restarted transcode segment that started at offset N."
  const pos = _effectiveCurrentTime();
  const dur = _effectiveDuration();
  if (!isFinite(pos) || !isFinite(dur) || dur <= 0) return;
  // GUARD: never POST current_time_s ≤ 1 unless this is the explicit
  // "finished" event. The boot sequence can briefly land at pos=0
  // between src-set and resume-seek; a force-post fired in that
  // window (e.g. an early ``pause`` event from the src change)
  // would write 0 to file_index AND push 0 upstream, wiping the
  // saved position. Finished=true is the one legitimate zero-ish
  // write — clearing "is_finished" against the original duration.
  if (pos < 1 && !finished) return;
  if (!force && Math.abs(pos - _lastProgressPos) < 1.0) return;
  _lastProgressAt = Date.now();
  _lastProgressPos = pos;
  fetch(`/api/media/progress/${encodeURIComponent(FILE_ID)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify({
      current_time_s: pos,
      duration_s: dur,
      is_finished: !!finished,
    }),
  }).catch((err) => {
    console.warn('[cast-video] progress post failed', err);
  });
}

window.addEventListener('pagehide', () => {
  if (!FILE_ID) return;
  const dur = _effectiveDuration();
  if (!dur) return;
  const pos = _effectiveCurrentTime();
  // Same guard as postProgress — if we're being torn down before
  // playback ever got past the first second, skip the beacon.
  // Forwarding 0 here is what was wiping the saved position when
  // the user switched away from a freshly-mounted but not-yet-seeked
  // surface. The previous post (10s tick or explicit pause) already
  // recorded the last good position; no reason to overwrite it.
  if (pos < 1) return;
  // sendBeacon first; keepalive fetch fallback covers Safari iOS
  // low-power-mode (returns false) and private-browsing (throws).
  const url = `/api/media/progress/${encodeURIComponent(FILE_ID)}`;
  const payload = JSON.stringify({
    current_time_s: pos,
    duration_s: dur,
    is_finished: false,
  });
  let sent = false;
  try {
    sent = navigator.sendBeacon(
      url,
      new Blob([payload], { type: 'application/json' }),
    );
  } catch { /* sendBeacon unavailable or refused */ }
  if (sent) return;
  try {
    fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: payload,
      keepalive: true,
    }).catch(() => {});
  } catch { /* nothing else to try */ }
});


/* ── A/V offset (lip-sync compensation) ───────────────────────────── *
 *
 * Web Audio routing for compensating Emby/Jellyfin partial-transcode
 * lip-sync drift. The graph (when active) is:
 *
 *   elVideo → MediaElementAudioSourceNode → DelayNode → GainNode → destination
 *
 * Direct port of the floating-video.js implementation. The controller
 * (cast-control) owns the slider UI and persistence; the receiver just
 * applies the offset it's told to via patch.audio_offset_ms.
 *
 * Lazy-init: only set up on first non-zero offset. Once initialized,
 * the graph stays alive until the surface tears down — Media-Element-
 * AudioSourceNode can only be created once per element, and detach
 * would silence audio with no clean way to reattach.
 */

let _audioContext = null;
let _audioSourceNode = null;
let _audioDelayNode = null;
let _audioGainNode = null;
let _audioGraphMediaEl = null;
let _audioOffsetMs = 0;
// Sticky targets — when the graph is live the patch handler routes
// volume/mute through the GainNode, but we still need to remember the
// requested values so they apply correctly post-init.
let _targetVolume = 1.0;
let _targetMuted = false;
// Once Web Audio init throws, refuse to retry for the rest of the
// session. One bad source must not be allowed to keep nuking the
// GPU process — the 2026-05-02 RGB-static incident in the desktop
// player traced to repeated createMediaElementSource attempts on a
// not-yet-ready element. Same browser family runs on the TV.
let _audioGraphDisabledForSession = false;

async function _ensureAudioGraph() {
  if (_audioGraphDisabledForSession) return false;
  if (!elVideo) return false;
  if (_audioGraphMediaEl === elVideo && _audioContext && _audioDelayNode) {
    return true;
  }
  // GPU-crash mitigation: never call createMediaElementSource before
  // the decoder has produced a frame. readyState 2 = HAVE_CURRENT_DATA.
  if (elVideo.readyState < 2) return false;
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return false;
    if (!_audioContext) _audioContext = new Ctx();
    // CRITICAL: cast iframes don't have user activation (the controller
    // tap is a gesture in the PHONE's window, not ours), so the context
    // starts ``suspended``. createMediaElementSource on a suspended
    // context would silently siphon the element's audio into a dead
    // graph — the user hears nothing. Await the resume FIRST; if it
    // fails, bail without touching elVideo so native playback survives.
    if (_audioContext.state === 'suspended') {
      try {
        await _audioContext.resume();
      } catch (err) {
        console.warn('[cast-video] audio context resume rejected:', err);
        _audioGraphDisabledForSession = true;
        return false;
      }
    }
    if (_audioContext.state !== 'running') {
      console.warn('[cast-video] audio context not running after resume');
      _audioGraphDisabledForSession = true;
      return false;
    }
    _audioDelayNode = _audioContext.createDelay(2.5);
    _audioDelayNode.delayTime.value = 0;
    _audioGainNode = _audioContext.createGain();
    _audioGainNode.gain.value = _targetMuted ? 0 : _targetVolume;
    _audioSourceNode = _audioContext.createMediaElementSource(elVideo);
    _audioSourceNode.connect(_audioDelayNode);
    _audioDelayNode.connect(_audioGainNode);
    _audioGainNode.connect(_audioContext.destination);
    elVideo.muted = true;
    _audioGraphMediaEl = elVideo;
    return true;
  } catch (err) {
    console.warn('[cast-video] audio graph init failed; disabling for session:', err);
    _audioGraphDisabledForSession = true;
    _detachAudioGraph();
    return false;
  }
}

function _detachAudioGraph() {
  try { _audioSourceNode?.disconnect(); } catch { /* ignore */ }
  try { _audioDelayNode?.disconnect();  } catch { /* ignore */ }
  try { _audioGainNode?.disconnect();   } catch { /* ignore */ }
  _audioSourceNode = null;
  _audioDelayNode = null;
  _audioGainNode = null;
  _audioGraphMediaEl = null;
  // AudioContext stays alive — closing is async and would race with
  // a follow-up reattach attempt. GC handles it on surface teardown.
}

async function _applyAudioOffsetMs(ms) {
  const clamped = Math.max(0, Math.min(2000, Math.round(Number(ms) || 0)));
  _audioOffsetMs = clamped;
  if (clamped === 0 && !_audioGraphMediaEl) return;
  if (clamped > 0 && !(await _ensureAudioGraph())) {
    _audioOffsetMs = 0;
    return;
  }
  if (_audioDelayNode) {
    // 30ms exponential ramp — converges to ~3% of target in ~90ms,
    // well under any plausible slider-drag cadence and far above the
    // audible click threshold of an abrupt delayTime change.
    const targetSec = clamped / 1000;
    try {
      _audioDelayNode.delayTime.setTargetAtTime(
        targetSec, _audioContext.currentTime, 0.03,
      );
    } catch {
      _audioDelayNode.delayTime.value = targetSec;
    }
  }
}

// Route volume + mute through the GainNode when the graph is live,
// else through the element's native attrs. Without this, touching
// elVideo.volume/muted after graph init is silent because the audio
// is no longer flowing through that path.
function _applyVolume(v) {
  _targetVolume = Math.max(0, Math.min(1, Number(v) || 0));
  if (_audioGainNode && !_targetMuted) {
    try { _audioGainNode.gain.value = _targetVolume; } catch {}
  } else if (!_audioGraphMediaEl) {
    elVideo.volume = _targetVolume;
  }
}

function _applyMuted(m) {
  _targetMuted = !!m;
  if (_audioGainNode) {
    try { _audioGainNode.gain.value = _targetMuted ? 0 : _targetVolume; } catch {}
  } else if (!_audioGraphMediaEl) {
    elVideo.muted = _targetMuted;
  }
}


/* ── Patch handler ────────────────────────────────────────────── */

function setSubtitleByIdx(idx) {
  _targetSubtitleIdx = (typeof idx === 'number') ? idx : -1;
  const tracks = elVideo.textTracks;
  if (!tracks || tracks.length === 0) {
    elFlagCC.hidden = true;
    return;
  }
  // Disable all, then enable the one whose data-track-idx matches.
  // The textTracks list mirrors the order of <track> elements; we
  // find the matching child to read its data attribute.
  const trackEls = Array.from(elVideo.querySelectorAll('track'));
  let activeLabel = '';
  for (let i = 0; i < tracks.length; i++) {
    const el = trackEls[i];
    const elIdx = el ? Number(el.dataset.trackIdx) : NaN;
    if (_targetSubtitleIdx === elIdx) {
      tracks[i].mode = 'showing';
      activeLabel = el?.label || '';
    } else {
      tracks[i].mode = 'disabled';
    }
  }
  if (_targetSubtitleIdx < 0 || !activeLabel) {
    elFlagCC.hidden = true;
  } else {
    elFlagCC.hidden = false;
    elFlagCC.textContent = activeLabel.length > 16
      ? activeLabel.slice(0, 14) + '…'
      : activeLabel;
  }
}

function setPlaybackRate(rate) {
  _targetPlaybackRate = Math.max(0.25, Math.min(4.0, Number(rate) || 1.0));
  try {
    elVideo.playbackRate = _targetPlaybackRate;
    // ratechange listener flips the HUD flag; if browser silently
    // refused, the next reapplySticky tick catches it.
  } catch (err) {
    console.warn('[cast-video] playbackRate set failed', err);
  }
}

function reapplySticky() {
  if (Math.abs((elVideo.playbackRate || 1.0) - _targetPlaybackRate) > 0.01) {
    try { elVideo.playbackRate = _targetPlaybackRate; } catch {}
  }
  // Subtitle tracks load asynchronously — re-apply target index in
  // case the target was set before tracks finished attaching.
  if (_targetSubtitleIdx >= 0) {
    setSubtitleByIdx(_targetSubtitleIdx);
  }
}

function handlePatch(patch) {
  if (!patch || typeof patch !== 'object') return;
  flashHud();
  try {
    if (typeof patch.paused === 'boolean') {
      if (patch.paused) elVideo.pause();
      else elVideo.play().catch(() => {});
    }
    if (typeof patch.position_s === 'number') {
      // patch.position_s is the ABSOLUTE timestamp in the film (the
      // controller computes it from the full-runtime seek bar). The
      // seek helper handles in-segment vs restart-from-N decisions.
      _seekTo(patch.position_s);
    }
    if (typeof patch.seek_delta_s === 'number') {
      // Compute the absolute target via _effectiveCurrentTime so a
      // +30s skip lands 30s ahead in FILM TIME, regardless of where
      // the current transcode segment started. Clamp against the
      // effective (full-runtime) duration so you can't skip past
      // end-of-film.
      const dur = _effectiveDuration();
      const next = _effectiveCurrentTime() + patch.seek_delta_s;
      const target = Math.max(0, dur > 0 ? Math.min(dur - 0.5, next) : next);
      _seekTo(target);
    }
    if (typeof patch.speed === 'number') setPlaybackRate(patch.speed);
    // Two ways to control subs: explicit index, or legacy on/off
    // bool (kept for back-compat — picks the first track when on).
    if (typeof patch.subtitle_idx === 'number') {
      setSubtitleByIdx(patch.subtitle_idx);
    } else if (typeof patch.subtitles === 'boolean') {
      const idx = patch.subtitles
        ? (_subtitleTracks.find((t) => t.is_default)?.index
           ?? _subtitleTracks[0]?.index
           ?? -1)
        : -1;
      setSubtitleByIdx(idx);
    }
    if (typeof patch.volume === 'number') {
      _applyVolume(patch.volume);
    }
    if (typeof patch.muted === 'boolean') {
      _applyMuted(patch.muted);
    }
    if (typeof patch.audio_offset_ms === 'number') {
      _applyAudioOffsetMs(patch.audio_offset_ms);
    }
  } catch (err) {
    console.warn('[cast-video] patch apply failed', err);
  }
}

window.addEventListener('message', (ev) => {
  const data = ev.data;
  if (!data || typeof data !== 'object') return;
  if (data.type === 'augmentum.surface_state') handlePatch(data.patch);
});


/* ── Boot ─────────────────────────────────────────────────────── */

fetchMetadata();
startVideo();
