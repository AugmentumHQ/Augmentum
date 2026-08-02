/**
 * cast-audio.js — TV "now playing audio" surface.
 *
 * Reads ?id=<file_id> from the URL, fetches /api/media/details/<id>
 * for metadata (title, author, cover), and points the hidden <audio>
 * element at /api/media/stream/<id>. The visible UI is the cover,
 * metadata, transport bar, and animated VU meter — so the TV shows
 * SOMETHING while the audio plays (vs a blank black screen from the
 * native <audio> element).
 *
 * No input handling — phone remains the controller.
 */

import { effectiveDuration } from '../scripts/media-duration.js';

const params = new URLSearchParams(location.search);
const FILE_ID = (params.get('id') || '').trim();

// Stream-URL mode — when an arbitrary HTTP audio URL is passed instead
// of a file_id (e.g. Grove casting an internet radio station). All
// presentation metadata comes from query params; progress persistence
// is skipped (there's nothing to resume to). Used when FILE_ID is empty.
const STREAM_URL    = (params.get('streamUrl') || '').trim();
const STREAM_TITLE  = (params.get('title')     || '').trim();
const STREAM_AUTHOR = (params.get('author')    || '').trim();
const STREAM_COVER  = (params.get('cover')     || '').trim();
const STREAM_SOURCE = (params.get('source')    || '').trim();

const $ = (sel) => document.querySelector(sel);
const elAudio = $('[data-ca-audio]');
const elTitle = $('[data-ca-title]');
const elAuthor = $('[data-ca-author]');
const elSource = $('[data-ca-source]');
const elCover = $('[data-ca-cover]');
const elArtBlur = $('[data-ca-art-blur]');
const elElapsed = $('[data-ca-elapsed]');
const elDuration = $('[data-ca-duration]');
const elBarFill = $('[data-ca-bar-fill]');
const elVU = $('[data-ca-vu]');


/* ── Helpers ──────────────────────────────────────────────────── */

function fmtTime(s) {
  if (!s || !isFinite(s)) return '—:—';
  const total = Math.round(s);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  const pad = (n) => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(sec)}` : `${m}:${pad(sec)}`;
}

// Upstream-reported full runtime, captured from /api/media/details on
// boot. Floored against elAudio.duration so duration label, bar fill,
// and /progress writes all reflect the real length even while the
// upstream provider is transcoding and the browser-reported duration
// only covers the buffered head. 0 for livestreams (no finite end).
let _knownDurationS = 0;

// Absolute timestamp the current transcode segment begins at. Zero
// for direct-play streams. Becomes N when we re-issue the stream
// with ?start_time_s=N to skip past a buffered transcode head — from
// then on, `elAudio.currentTime` is the offset INTO the new segment
// and the actual playhead position is `_baseOffsetS + elAudio.currentTime`.
let _baseOffsetS = 0;

function _effectiveCurrentTime() {
  const raw = Number(elAudio?.currentTime || 0);
  if (!Number.isFinite(raw) || raw < 0) return _baseOffsetS;
  return _baseOffsetS + raw;
}

function _effectiveDuration() {
  const raw = Number(elAudio?.duration);
  const segmentEnd = Number.isFinite(raw) && raw > 0 ? _baseOffsetS + raw : 0;
  return effectiveDuration(segmentEnd, _knownDurationS);
}

// Same in-segment-vs-restart-from-N logic as cast-video. Most audio
// is direct-play (MP3/AAC) — seekable covers the full file early so
// _seekTo never hits the restart branch. The branch exists for the
// rare audiobookshelf-transcode case and for symmetry with video.
function _seekTo(targetS) {
  if (!FILE_ID) return;  // STREAM_URL livestream — no seek surface
  const absoluteTarget = Math.max(0, Number(targetS) || 0);
  const segmentTarget = absoluteTarget - _baseOffsetS;
  const seekableEnd = (elAudio.seekable && elAudio.seekable.length > 0)
    ? Number(elAudio.seekable.end(elAudio.seekable.length - 1) || 0)
    : 0;
  if (segmentTarget >= 0 && segmentTarget <= seekableEnd - 0.5) {
    try { elAudio.currentTime = segmentTarget; } catch (err) {
      console.warn('[cast-audio] in-segment seek failed', err);
    }
    return;
  }
  console.log(`[cast-audio] restart-from-N seek: ${absoluteTarget.toFixed(1)}s`);
  const wasPaused = !!elAudio.paused;
  _baseOffsetS = absoluteTarget;
  const qs = new URLSearchParams();
  qs.set('start_time_s', String(absoluteTarget));
  qs.set('_seek', String(Date.now()));
  elAudio.src = `/api/media/stream/${encodeURIComponent(FILE_ID)}?${qs}`;
  if (wasPaused) {
    const restorePause = () => {
      try { elAudio.pause(); } catch {}
      elAudio.removeEventListener('loadeddata', restorePause);
    };
    elAudio.addEventListener('loadeddata', restorePause);
  } else {
    elAudio.play().catch((err) => {
      console.warn('[cast-audio] post-seek play rejected', err);
    });
  }
}


/* ── Metadata fetch ───────────────────────────────────────────── */

function applyStreamMetadata() {
  // Stream-URL mode — render purely from URL params, no provider lookup.
  const title = STREAM_TITLE || 'Live Stream';
  document.title = `Augmentum · ${title}`;
  elTitle.textContent = title;
  elAuthor.textContent = STREAM_AUTHOR || ' ';
  elSource.textContent = (STREAM_SOURCE || 'stream').replace(/_/g, ' ');

  if (STREAM_COVER) {
    elCover.src = STREAM_COVER;
    elCover.onload = () => {
      elCover.classList.add('on');
      elArtBlur.style.backgroundImage = `url("${STREAM_COVER}")`;
      elArtBlur.classList.add('on');
    };
    elCover.onerror = () => {
      // Fall back to editorial gradient if the cover URL 404s.
      elCover.style.display = 'none';
    };
  } else {
    // No cover — leave the editorial gradient as the backdrop.
    elCover.style.display = 'none';
  }
}

async function fetchMetadata() {
  if (!FILE_ID) return;
  try {
    const r = await fetch(`/api/media/details/${encodeURIComponent(FILE_ID)}`, {
      credentials: 'same-origin',
    });
    if (!r.ok) {
      console.warn('[cast-audio] details fetch returned', r.status);
      return;
    }
    const body = await r.json();
    applyMetadata(body);
  } catch (err) {
    console.warn('[cast-audio] details fetch threw', err);
  }
}

let _resumePosition = 0;  // seconds, applied once loadedmetadata fires

function applyMetadata(body) {
  const meta = body.entry || body || {};
  const sourceMeta = meta.source_metadata || meta.metadata || {};
  const title = meta.name || meta.title || 'Untitled';
  const author = sourceMeta.author || sourceMeta.artist || sourceMeta.narrator || '';
  const source = meta.source || sourceMeta.provider || 'audio';

  document.title = `Augmentum · ${title}`;
  elTitle.textContent = title;
  elAuthor.textContent = author || ' ';
  elSource.textContent = source.replace(/_/g, ' ');

  // Captured BEFORE the audio element knows its own duration so the
  // label + bar render correctly even on a transcode where the
  // browser will never report the true full runtime.
  _knownDurationS = Math.max(
    0,
    Number(body.duration_s ?? sourceMeta.duration_s ?? meta.duration_s ?? 0),
  );
  if (_knownDurationS > 0) {
    elDuration.textContent = fmtTime(_knownDurationS);
  }

  // Resume point — /api/media/details returns ``current_time_s`` at
  // the TOP LEVEL of the response (provider-neutral shape). Older
  // responses nested it under ``source_metadata`` so we still check
  // both — but reading only the nested path was the real reason
  // resume looked broken (top-level is always present, nested isn't).
  const savedPos = parseFloat(
    body.current_time_s
    ?? sourceMeta.current_time_s
    ?? meta.current_time_s
    ?? 0,
  );
  if (isFinite(savedPos) && savedPos > 2) {
    _resumePosition = savedPos;
    console.log(`[cast-audio] resume position resolved: ${savedPos.toFixed(1)}s`);
    // RACE FIX: same as cast-video — if metadata already loaded
    // (cached audio), loadedmetadata fired before we knew the
    // resume position. Apply seek inline via _seekTo so transcode
    // sources don't get a "snap to 0" if savedPos is past the
    // initial buffered window.
    const effDur = _effectiveDuration();
    if (elAudio.readyState >= 1 && effDur > 0
        && savedPos < effDur - 1) {
      _seekTo(savedPos);
      console.log(`[cast-audio] applied resume late: ${savedPos.toFixed(1)}s`);
      _resumePosition = 0;
    }
  }

  const coverUrl = `/api/media/cover/${encodeURIComponent(FILE_ID)}`;
  elCover.src = coverUrl;
  elCover.onload = () => {
    elCover.classList.add('on');
    elArtBlur.style.backgroundImage = `url("${coverUrl}")`;
    elArtBlur.classList.add('on');
  };
  elCover.onerror = () => {
    // No cover — leave the editorial gradient.
    elCover.style.display = 'none';
  };
}


/* ── Audio wiring ─────────────────────────────────────────────── */

function startAudio() {
  if (!FILE_ID && !STREAM_URL) {
    elTitle.textContent = 'No track ID provided';
    return;
  }
  elAudio.src = FILE_ID
    ? `/api/media/stream/${encodeURIComponent(FILE_ID)}`
    : STREAM_URL;
  elAudio.addEventListener('loadedmetadata', () => {
    const dur = _effectiveDuration();
    if (dur <= 0) {
      // Livestream / icecast — no upstream-known runtime AND the
      // element-reported duration is Infinity/NaN. Show LIVE in place
      // of total time and hide the bar entirely.
      elDuration.textContent = 'LIVE';
      const bar = document.querySelector('[data-ca-bar]');
      if (bar) bar.style.visibility = 'hidden';
    } else {
      elDuration.textContent = fmtTime(dur);
    }
    // Apply the resume position once we have a usable runtime to
    // clamp against (so we can guard against past-the-end values).
    if (_resumePosition > 0
        && dur > 0
        && _resumePosition < dur - 1) {
      _seekTo(_resumePosition);
      console.log(`[cast-audio] resumed at ${_resumePosition.toFixed(1)}s`);
    }
    _resumePosition = 0;  // single-shot
  });
  elAudio.addEventListener('timeupdate', () => {
    const pos = _effectiveCurrentTime();
    const dur = _effectiveDuration();
    elElapsed.textContent = fmtTime(pos);
    if (dur > 0) {
      elBarFill.style.width = `${Math.min(100, (pos / dur) * 100)}%`;
    }
    maybePostProgress();  // throttled inside
    _echoSurfaceState({
      position_s: pos,
      duration_s: dur,
      paused: elAudio.paused,
    });
  });
  elAudio.addEventListener('play', () => {
    elVU.classList.remove('paused');
    _echoSurfaceState({
      paused: false,
      position_s: _effectiveCurrentTime(),
      duration_s: _effectiveDuration(),
    }, { force: true });
  });
  elAudio.addEventListener('pause', () => {
    elVU.classList.add('paused');
    // Write progress immediately on pause so a quick re-pair picks
    // up the latest position without waiting for the next periodic
    // tick to land.
    postProgress({ force: true });
    _echoSurfaceState({
      paused: true,
      position_s: _effectiveCurrentTime(),
      duration_s: _effectiveDuration(),
    }, { force: true });
  });
  elAudio.addEventListener('ended', () => {
    elVU.classList.add('paused');
    postProgress({ force: true, finished: true });
  });
  elAudio.addEventListener('error', () => {
    elTitle.textContent = 'Audio failed to load';
    elVU.classList.add('paused');
  });
  // Some browsers / TV WebViews silently block autoplay until a
  // user gesture; on Augmentum's cast pipeline this UI is loaded
  // inside an iframe that's already in an active media context
  // (the cast WS established a session), so play() should succeed.
  elAudio.play().catch((err) => {
    console.warn('[cast-audio] autoplay rejected', err);
  });
}


/* ── Progress persistence ─────────────────────────────────────── */
//
// Throttled POST to /api/media/progress so resume works across
// re-pairs and across-device handoffs. The endpoint already pushes
// to the upstream provider (ABS / LibriVox state) AND caches in
// file_index, so a fresh cast-audio session can read back via
// /api/media/details on next launch.

const PROGRESS_POST_INTERVAL_MS = 10 * 1000;
let _lastProgressAt = 0;
let _lastProgressPos = 0;

function maybePostProgress() {
  const now = Date.now();
  if (now - _lastProgressAt < PROGRESS_POST_INTERVAL_MS) return;
  postProgress();
}

function postProgress({ force = false, finished = false } = {}) {
  if (!FILE_ID) return;
  // Absolute coordinates — _effectiveCurrentTime folds in _baseOffsetS
  // so a restart-from-N transcode segment doesn't write back the
  // segment-relative position (which would resume from 0 next time).
  const pos = _effectiveCurrentTime();
  const dur = _effectiveDuration();
  if (!isFinite(pos) || !isFinite(dur) || dur <= 0) return;
  // GUARD: never POST current_time_s ≤ 1 unless it's the explicit
  // "finished" event — the boot sequence briefly lands at pos=0
  // between src-set and resume-seek and a force-post in that
  // window (early pause from src-change) would wipe the saved
  // position from disk + upstream.
  if (pos < 1 && !finished) return;
  // Skip if position hasn't actually advanced (rounding / pause).
  // ``force`` overrides this for the pause/end events.
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
    console.warn('[cast-audio] progress post failed', err);
  });
}

// Capture a final progress beat if the user navigates away / the
// surface gets replaced. ``sendBeacon`` is fire-and-forget and not
// blocked by the unload phase — but Safari (especially iOS in
// low-power mode) sometimes refuses the queue and returns false, and
// some private-browsing modes throw outright. Fall through to a
// keepalive ``fetch`` so the saved-position write survives those
// paths too.
window.addEventListener('pagehide', () => {
  if (!FILE_ID) return;
  const dur = _effectiveDuration();
  if (!dur) return;
  const pos = _effectiveCurrentTime();
  // Same guard as postProgress — if torn down before playback
  // started, don't beacon 0 (would wipe the saved position).
  if (pos < 1) return;
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


/* ── Surface-state echo to cast-receiver shell ─────────────────── *
 * The shell forwards these as ``surface_state`` WS events so
 * controller UIs (cast-shelf, cast-control) can render scrubbers
 * and pause buttons that mirror reality. Throttled to match the
 * native-shortcut path in cast-receiver.js (every 5s for position
 * ticks; play/pause/duration changes fire immediately). */

let _lastStateEchoAt = 0;
const STATE_ECHO_INTERVAL_MS = 5000;

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
    console.warn('[cast-audio] state echo failed', err);
  }
}


/* ── Remote-control surface_state ─────────────────────────────── */
//
// The cast-receiver shell forwards every ``surface_state`` patch
// targeted at this iframe via postMessage. We mirror the same patch
// vocabulary the receiver uses natively for media.video / media.audio
// surfaces so the phone-side transport UI doesn't have to special-
// case our iframe — same keys, same semantics.
//
// Accepted patch keys:
//   paused: bool         — true pauses, false plays
//   position_s: number   — absolute seek in seconds
//   seek_delta_s: number — relative seek (skip 30 / skip back 15)
//   volume: 0..1
//   muted: bool

let _targetPlaybackRate = 1.0;

function setPlaybackRate(rate) {
  _targetPlaybackRate = Math.max(0.25, Math.min(4.0, Number(rate) || 1.0));
  try { elAudio.playbackRate = _targetPlaybackRate; }
  catch (err) { console.warn('[cast-audio] playbackRate set failed', err); }
}

// Re-apply sticky speed on lifecycle events that can silently reset
// playbackRate (some browsers do this during seeks / buffer drains).
['loadeddata', 'playing', 'canplay'].forEach((ev) => {
  elAudio.addEventListener(ev, () => {
    if (Math.abs((elAudio.playbackRate || 1.0) - _targetPlaybackRate) > 0.01) {
      try { elAudio.playbackRate = _targetPlaybackRate; } catch {}
    }
  });
});

function handlePatch(patch) {
  if (!patch || typeof patch !== 'object' || !elAudio) return;
  try {
    if (typeof patch.paused === 'boolean') {
      if (patch.paused) elAudio.pause();
      else elAudio.play().catch(() => {});
    }
    if (typeof patch.position_s === 'number') {
      // Absolute target in track-time; _seekTo decides in-segment
      // vs restart-from-N (rare for audio, but real for ABS-
      // transcoded sources).
      _seekTo(patch.position_s);
    }
    if (typeof patch.seek_delta_s === 'number') {
      const dur = _effectiveDuration();
      const next = _effectiveCurrentTime() + patch.seek_delta_s;
      const target = Math.max(0, dur > 0 ? Math.min(dur - 0.5, next) : next);
      _seekTo(target);
    }
    if (typeof patch.speed === 'number') setPlaybackRate(patch.speed);
    if (typeof patch.volume === 'number') {
      elAudio.volume = Math.max(0, Math.min(1, patch.volume));
    }
    if (typeof patch.muted === 'boolean') {
      elAudio.muted = patch.muted;
    }
  } catch (err) {
    console.warn('[cast-audio] patch apply failed', err);
  }
}

window.addEventListener('message', (ev) => {
  const data = ev.data;
  if (!data || typeof data !== 'object') return;
  // cast-receiver wraps patches as ``{type: 'augmentum.surface_state', surface_id, patch}``.
  if (data.type === 'augmentum.surface_state') {
    handlePatch(data.patch);
  }
});


/* ── Boot ─────────────────────────────────────────────────────── */

if (STREAM_URL && !FILE_ID) {
  applyStreamMetadata();
} else {
  fetchMetadata();
}
startAudio();
