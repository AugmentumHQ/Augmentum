/**
 * media-player.js — Global audio playback singleton.
 *
 * One HTMLAudioElement lives on document.body for the whole session.
 * Every surface that wants to play media — detail panel, mini-player,
 * future voice-mode card — reads state from and dispatches commands
 * through this module. Navigation between panels doesn't kill playback
 * because the element isn't tied to any panel's DOM.
 *
 * State shape:
 *   {
 *     fileId, title, author, narrator,
 *     coverUrl, streamUrl,
 *     durationS, currentTimeS,
 *     isPlaying, isLoading, isFinished,
 *     speed, sleepTimerMs, sleepTimerStartedAt,
 *     chapters:        [{title, start, end}, ...],
 *     currentChapterIdx,
 *   }
 *
 * Consumers call ``subscribe(fn)`` to observe; fn receives the current
 * snapshot on every state transition. Return value is an unsubscribe
 * function (pattern borrows from Redux-style stores, keeps coupling low
 * and testing easy).
 */

import { mediaStreamUrl, mediaCoverUrl, pushMediaProgress, addBookmark } from './files/api.js';
import { AudioBus } from './audio-bus.js';
import { MediaSessionBridge } from './media-session.js';
import { recordLastPlayed } from './media-resume.js';
import { isArmed } from './armed-device.js';
import { castOrPlay } from './cast-or-play.js';

// Skip conventions for audiobooks: +30/−15. Longer forward because
// "I spaced out" is the common case; shorter backward because "I missed
// that last sentence" only needs a few seconds. Matches Audible / Libby.
const SKIP_FORWARD_S = 30;
const SKIP_BACK_S = 15;

// Progress push throttle: every 10s during playback is frequent enough
// for cross-device continuity without hammering the upstream server.
const PUSH_INTERVAL_MS = 10_000;

// Default speed bounds — intentionally wider than most apps since some
// power listeners go well past 2×.
const SPEED_MIN = 0.5;
const SPEED_MAX = 3.0;

// State store — one object, mutated in place. Subscribers get a fresh
// shallow snapshot per notification so they can cheaply diff.
//
// Multi-file books (LibriVox ships one MP3 per chapter) are modelled as
// `audioFiles: [{durationS, startS}, ...]` with cumulative book-level
// start offsets. Single-file sources (Audiobookshelf) get an empty array
// and every seek/time read treats `audio.currentTime` as book time.
// All public APIs accept and return book-level times; file boundaries
// are a transport detail managed by _loadFile / _advanceFile.
const _state = {
  fileId: '',
  episodeId: '',
  title: '',
  author: '',
  narrator: '',
  coverUrl: '',
  streamUrl: '',
  durationS: 0,
  currentTimeS: 0,
  isPlaying: false,
  isLoading: false,
  isFinished: false,
  speed: 1.0,
  sleepTimerMs: 0,
  sleepTimerStartedAt: 0,
  sleepTimerEndOfChapter: false,
  chapters: [],
  currentChapterIdx: -1,
  // Multi-file bookkeeping. Empty audioFiles means single-file mode.
  audioFiles: [],
  currentFileIdx: 0,
  // When switching files, the within-file offset to seek to once
  // `loadedmetadata` fires. Cleared by the loadedmetadata handler.
  pendingFileSeek: 0,
};

const _subscribers = new Set();
let _audio = null;
let _lastPush = 0;
let _sleepTimerHandle = null;

function _snapshot() {
  // Shallow copy so observers can freely hold references; nested arrays
  // are treated as read-only by convention (nothing we ship mutates
  // chapters in place).
  return { ..._state };
}

function _notify() {
  const snap = _snapshot();
  for (const fn of _subscribers) {
    try { fn(snap); } catch (err) { console.warn('[media-player] subscriber threw:', err); }
  }
  _maybeRefreshAgentCatalog();
}

// Edge-triggered app-menu liveness: only when isActive() flips (load /
// close), never on the 1Hz position ticks — the catalog sync is
// debounced + deduped downstream, but there's no reason to even poke it.
let _agentLive = false;
function _maybeRefreshAgentCatalog() {
  const live = !!_state.fileId;
  if (live === _agentLive) return;
  _agentLive = live;
  import('./command-palette.js')
    .then((m) => m.refreshAgentCatalog())
    .catch(() => {});
}

let _mediaBusHandle = null;
let _mediaDuckBaseline = null;

function _ensureAudio() {
  if (_audio) return _audio;
  _audio = document.createElement('audio');
  _audio.preload = 'metadata';
  _audio.style.display = 'none';
  document.body.appendChild(_audio);

  _mediaBusHandle = AudioBus.register({
    id: 'media-player',
    tier: 'media',
    // Audiobookshelf is the primary backend for this player. Tagging
    // as 'narration' so the Becca widget plays a listening pose
    // instead of dance VRMAs while an audiobook is rolling. play()
    // re-tags via setKind when the item's entity_kind says 'music'
    // (local music files route through this player too) so music
    // exclusivity and embodiment both see the truth.
    kind: 'narration',
    duck: (level) => {
      if (!_audio || _mediaDuckBaseline !== null) return;
      _mediaDuckBaseline = _audio.volume;
      _audio.volume = _mediaDuckBaseline * level;
    },
    unduck: () => {
      if (!_audio || _mediaDuckBaseline === null) return;
      _audio.volume = _mediaDuckBaseline;
      _mediaDuckBaseline = null;
    },
    // Music is exclusive on the bus: when another music source (Grove
    // radio, YouTube orb) starts while this player is tagged 'music',
    // pause instead of stacking. Narration is NOT exclusive, so an
    // audiobook still coexists with (and ducks) an ambient bed track.
    stop: () => pause(),
  });

  // Lock-screen / headphones / Bluetooth AVRCP / car head-unit AVRCP.
  // The bridge picks the owner via the AudioBus state event — media
  // tier outranks Grove's ambient tier, so an audiobook always wins.
  // previous/nexttrack route to chapter navigation (matching Audible
  // / Libby): a single-press of the steering-wheel skip button is a
  // chapter jump, not a book jump. seekforward/backward use the same
  // +30/-15 defaults as the in-app buttons.
  MediaSessionBridge.register('media-player', {
    getMetadata: () => ({
      title:  _state.title || 'Audiobook',
      artist: _state.author || '',
      album:  _state.narrator ? `Read by ${_state.narrator}` : '',
      artworkUrl: _state.coverUrl || null,
    }),
    getPosition: () => ({
      duration:     _state.durationS || 0,
      position:     _state.currentTimeS || 0,
      playbackRate: _state.speed || 1,
    }),
    handlers: {
      play:          () => resume(),
      pause:         () => pause(),
      stop:          () => pause(),
      seekto:        (d) => { if (typeof d.seekTime === 'number') seek(d.seekTime); },
      seekforward:   (d) => skip(typeof d.seekOffset === 'number' ? d.seekOffset : SKIP_FORWARD_S),
      seekbackward:  (d) => skip(-(typeof d.seekOffset === 'number' ? d.seekOffset : SKIP_BACK_S)),
      previoustrack: () => skipChapterRelative(-1),
      nexttrack:     () => skipChapterRelative(1),
    },
  });

  // --- Lifecycle events ---
  _audio.addEventListener('loadedmetadata', () => {
    // Single-file mode: book duration is the audio element's duration.
    // Multi-file: book duration was set up-front from the sum of
    // per-file durations; don't overwrite it from the current file.
    if (_isMultiFile()) {
      // If upstream didn't give us a duration for this file (happens on
      // the LibriVox feed for a minority of books), backfill from the
      // audio element so our book→file math doesn't silently drift.
      const cur = _state.audioFiles[_state.currentFileIdx];
      if (cur && !cur.durationS && Number.isFinite(_audio.duration)) {
        cur.durationS = _audio.duration;
        _recomputeAudioFileStarts();
      }
    } else if (!_state.durationS && Number.isFinite(_audio.duration)) {
      _state.durationS = _audio.duration;
    }

    // Within-file resume target. For single-file books this is the book-
    // level resume offset (they coincide); for multi-file books this was
    // set by _loadFile so we land mid-chapter when switching files.
    const within = _state.pendingFileSeek;
    _state.pendingFileSeek = 0;
    const fileDuration = _isMultiFile()
      ? (_state.audioFiles[_state.currentFileIdx]?.durationS || _audio.duration || 0)
      : (_state.durationS || _audio.duration || 0);
    if (within > 1 && (!fileDuration || within < fileDuration - 2)) {
      try { _audio.currentTime = within; } catch { /* seek not ready */ }
    }
    _state.isLoading = false;
    _notify();
  });

  _audio.addEventListener('timeupdate', () => {
    const fileS = _audio.currentTime || 0;
    _state.currentTimeS = _isMultiFile()
      ? _fileTimeToBook(_state.currentFileIdx, fileS)
      : fileS;
    _updateCurrentChapter();
    _maybePushProgress();
    _notify();
  });

  _audio.addEventListener('play',  () => {
    _state.isPlaying = true;
    _mediaBusHandle?.claim();
    MediaSessionBridge.setPlaybackState('media-player', 'playing');
    _notify();
  });
  _audio.addEventListener('pause', () => {
    _state.isPlaying = false;
    _mediaBusHandle?.release();
    MediaSessionBridge.setPlaybackState('media-player', 'paused');
    // Force a progress push on pause so a brief "pause + close tab" is
    // still captured upstream.
    _pushNow({ force: true });
    _notify();
  });

  _audio.addEventListener('ended', () => {
    // Multi-file book: one MP3 ending is a chapter boundary, not the
    // end of the book. Advance to the next file and continue playback
    // so LibriVox listening feels like one long stream.
    if (_isMultiFile() && _state.currentFileIdx < _state.audioFiles.length - 1) {
      _loadFile(_state.currentFileIdx + 1, { withinS: 0, autoplay: true });
      return;
    }
    _state.isPlaying = false;
    _state.isFinished = true;
    _mediaBusHandle?.release();
    _pushNow({ force: true, finished: true });
    _notify();
  });

  _audio.addEventListener('waiting', () => {
    _state.isLoading = true;
    _notify();
  });
  _audio.addEventListener('playing', () => {
    _state.isLoading = false;
    _notify();
  });

  return _audio;
}

// --- Multi-file helpers ------------------------------------------------
// A multi-file book is one whose audio arrived as N separate MP3s
// (LibriVox style). We track the sequence + cumulative start offsets so
// a book-level time like "at 2h15m" resolves to the correct file and the
// right offset within it.

function _isMultiFile() {
  return _state.audioFiles.length > 1;
}

function _recomputeAudioFileStarts() {
  // Rebuilds cumulative startS after any per-file duration change.
  // Durations can shift if we discover a missing length on loadedmetadata.
  let cursor = 0;
  for (const af of _state.audioFiles) {
    af.startS = cursor;
    cursor += af.durationS || 0;
  }
}

function _bookTimeToFile(bookS) {
  // Returns { fileIdx, withinS } for a book-level timestamp. Defaults
  // to the first file for negative input; clamps to the last file when
  // past the book's total duration so a stale resume position doesn't
  // explode into out-of-range file access.
  const files = _state.audioFiles;
  if (!files.length) return { fileIdx: 0, withinS: Math.max(0, bookS) };
  if (bookS <= 0) return { fileIdx: 0, withinS: 0 };
  for (let i = 0; i < files.length; i++) {
    const end = files[i].startS + (files[i].durationS || 0);
    if (bookS < end) return { fileIdx: i, withinS: Math.max(0, bookS - files[i].startS) };
  }
  const last = files.length - 1;
  return { fileIdx: last, withinS: files[last].durationS || 0 };
}

function _fileTimeToBook(fileIdx, withinS) {
  const f = _state.audioFiles[fileIdx];
  if (!f) return withinS;
  return (f.startS || 0) + Math.max(0, withinS);
}

function _streamSrcFor(fileIdx) {
  // Multi-file books pin a specific chapter MP3 via ?file=<idx>; single-
  // file sources use the plain stream URL (the backend reads stream_path
  // from source_metadata there).
  const base = mediaStreamUrl(_state.fileId, {
    episodeId: _state.episodeId || '',
  });
  return _isMultiFile() ? `${base}?file=${fileIdx}` : base;
}

function _loadFile(fileIdx, { withinS = 0, autoplay = false } = {}) {
  const audio = _ensureAudio();
  _state.currentFileIdx = fileIdx;
  _state.pendingFileSeek = Math.max(0, withinS);
  _state.streamUrl = _streamSrcFor(fileIdx);
  _state.isLoading = true;
  audio.src = _state.streamUrl;
  audio.playbackRate = _state.speed;
  _notify();
  if (autoplay) {
    audio.play().catch((err) => {
      console.warn('[media-player] play blocked on file switch:', err);
      _state.isPlaying = false;
      _state.isLoading = false;
      _notify();
    });
  }
}

function _updateCurrentChapter() {
  const t = _state.currentTimeS;
  const chapters = _state.chapters;
  if (!chapters.length) {
    if (_state.currentChapterIdx !== -1) {
      _state.currentChapterIdx = -1;
    }
    return;
  }
  let idx = -1;
  // Linear scan is fine; even massive audiobooks top out at ~120 chapters.
  for (let i = 0; i < chapters.length; i++) {
    if (t >= chapters[i].start) idx = i;
    else break;
  }
  if (idx !== _state.currentChapterIdx) {
    _state.currentChapterIdx = idx;
    // Some lock-screen widgets surface the album/chapter line. Push a
    // metadata refresh on chapter advance so the displayed line stays
    // current without waiting for an explicit user action.
    MediaSessionBridge.notifyMetadataChanged('media-player');
  }
}

function _maybePushProgress() {
  // Suppress pushes while a resume seek is still pending. Without this,
  // an early `timeupdate` (audio.currentTime ≈ 0) firing before
  // `loadedmetadata`'s seek lands satisfies the throttle on the very
  // first tick (initial _lastPush = 0) and pushes current_time_s = 0,
  // clobbering the user's saved position both locally and upstream —
  // the "continue listening restarts from 0" bug.
  if (_state.pendingFileSeek > 0) return;
  const now = Date.now();
  if (now - _lastPush < PUSH_INTERVAL_MS) return;
  _pushNow();
}

async function _pushNow({ force = false, finished = false } = {}) {
  if (!_state.fileId) return;
  // Same pre-seek guard as _maybePushProgress, applied here too because
  // the pause/ended handlers call this with force=true and would
  // otherwise bypass the throttle to write a transient 0 over the saved
  // position (e.g. user clicks continue listening, then immediately
  // pauses before the loadedmetadata seek lands).
  if (_state.pendingFileSeek > 0) return;
  if (!force && Date.now() - _lastPush < PUSH_INTERVAL_MS) return;
  _lastPush = Date.now();
  const result = await pushMediaProgress(_state.fileId, {
    current_time_s: _state.currentTimeS,
    duration_s: _state.durationS,
    is_finished: finished || _state.isFinished,
    episode_id: _state.episodeId || '',
  });
  if (result?.progress_pct != null) {
    // Fire a DOM-level event so the Files grid / Library panel can
    // update their progress bars without subscribing to this module.
    // Keeps coupling minimal.
    window.dispatchEvent(new CustomEvent('media-player:progress', {
      detail: {
        fileId: _state.fileId,
        progressPct: result.progress_pct,
        currentTimeS: _state.currentTimeS,
      },
    }));
  }
}

// --- Public API -------------------------------------------------------

export function subscribe(fn) {
  _subscribers.add(fn);
  fn(_snapshot());  // prime the subscriber with current state
  return () => _subscribers.delete(fn);
}

export function getState() {
  return _snapshot();
}

/**
 * Load and start a media row. Fetches rich details from the backend so
 * chapters and current progress are fresh on every open. If the same
 * fileId is already playing, we treat this as a resume-to-top.
 */
export async function play(fileId, { details = null } = {}) {
  // Cast intercept: if a device is armed, route the play to that device
  // instead of starting local audio. Mini-player stays untouched; the
  // cast-shelf row for that receiver takes over as the user-facing
  // playback surface.
  if (isArmed()) {
    let rich = details;
    if (!rich) {
      try {
        const resp = await fetch(`/api/media/details/${encodeURIComponent(fileId)}`);
        if (resp.ok) rich = await resp.json();
      } catch { /* fall through with whatever rich became */ }
    }
    const episodeId = String(rich?.selected_episode_id || '').trim();
    const castResult = await castOrPlay({
      capability: 'media.audio_play@1',
      surface: 'media-player',
      args: {
        content_url: mediaStreamUrl(fileId, { episodeId }),
        file_id: fileId,
        requires_auth: true,
        title: rich?.playback_title || rich?.title || '',
        author: rich?.author || '',
        poster_url: rich?.cover_url || mediaCoverUrl(fileId),
      },
      fallback: null,
    });
    if (castResult.cast) return;
    // Cast failed; fall through to local playback so the user still hears something.
  }

  const audio = _ensureAudio();
  const requestedEpisodeId = String(details?.selected_episode_id || '').trim();
  const isSame = _state.fileId === fileId && _state.episodeId === requestedEpisodeId && audio.src;

  if (!isSame) {
    _state.fileId = fileId;
    _state.episodeId = requestedEpisodeId;
    _state.isLoading = true;
    // Reset multi-file bookkeeping; details may repopulate it below.
    _state.audioFiles = [];
    _state.currentFileIdx = 0;
    _state.pendingFileSeek = 0;
    _notify();
  }

  // Fetch (or accept pre-fetched) rich details so chapters/cover/etc.
  // render immediately. Non-fatal if the call fails — playback still
  // works with whatever the listing gave us.
  let rich = details;
  if (!rich && !isSame) {
    try {
      const resp = await fetch(`/api/media/details/${encodeURIComponent(fileId)}`);
      if (resp.ok) rich = await resp.json();
    } catch { /* network hiccup, fall through with blanks */ }
  }
  if (rich) {
    // Re-tag the bus source by what's actually playing: local music
    // files join music exclusivity (a later radio/YouTube ask stops
    // them instead of stacking, and the widget dances); everything
    // else stays narration so audiobooks coexist with ambient beds.
    const _entityKind = String(
      rich.entity_kind || rich.source_metadata?.entity_kind || '',
    ).toLowerCase();
    _mediaBusHandle?.setKind?.(_entityKind === 'music' ? 'music' : 'narration');
    _state.episodeId     = String(rich.selected_episode_id || '').trim();
    _state.title         = rich.playback_title || rich.title || _state.title;
    _state.author        = rich.author || _state.author;
    _state.narrator      = rich.narrator || '';
    _state.coverUrl      = rich.cover_url || mediaCoverUrl(fileId);
    _state.durationS     = rich.duration_s || _state.durationS;
    _state.currentTimeS  = rich.current_time_s ?? _state.currentTimeS;
    _state.chapters      = Array.isArray(rich.chapters) ? rich.chapters : [];
    _state.isFinished    = !!rich.is_finished;
    // Title/author/cover all just changed — push to lock-screen so
    // the next paint shows the new book instead of the previous one.
    MediaSessionBridge.notifyMetadataChanged('media-player');
    // Audio files only populated for multi-file sources. Single-file
    // sources (ABS) send an empty list, keeping the player in its
    // pre-existing single-file code path.
    const files = Array.isArray(rich.audio_files) ? rich.audio_files : [];
    _state.audioFiles = files.map(f => ({
      durationS: Number(f?.duration_s) || 0,
      startS: 0,  // filled in by _recomputeAudioFileStarts
    }));
    _recomputeAudioFileStarts();
    _updateCurrentChapter();
    _notify();
  } else if (!isSame) {
    _state.coverUrl = mediaCoverUrl(fileId);
    _notify();
  }

  // Load the right file:
  //   - Single-file: plain stream URL, resume handled by loadedmetadata.
  //   - Multi-file: translate resume position to (fileIdx, withinS) and
  //     load that chapter MP3. `pendingFileSeek` carries the within-file
  //     offset to loadedmetadata.
  if (!isSame) {
    if (_isMultiFile()) {
      const { fileIdx, withinS } = _bookTimeToFile(_state.currentTimeS || 0);
      _loadFile(fileIdx, { withinS, autoplay: false });
    } else {
      _state.streamUrl = mediaStreamUrl(fileId, {
        episodeId: _state.episodeId || '',
      });
      _state.pendingFileSeek = _state.currentTimeS || 0;
      audio.src = _state.streamUrl;
      _notify();
    }
    // Log to Discovery history so the user can find this title back
    // later via "For You → History". Audiobooks + podcasts route
    // through this player; differentiate by entity_kind so the rendered
    // history item label reads correctly. Best-effort fetch — never
    // block playback on the signal POST.
    const entityKind = String(rich?.entity_kind || rich?.source_metadata?.entity_kind || '').toLowerCase();
    const contentType = entityKind === 'podcast' ? 'podcast' : 'audiobook';
    fetch('/api/discovery/signal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        signal_type: 'media_play',
        source_url: `augm:media:${fileId}`,
        source_title: _state.title || 'Audiobook',
        content_type: contentType,
        metadata: {
          file_id: fileId,
          kind: 'audio',
          cover_url: _state.coverUrl || '',
          author: _state.author || '',
          duration_s: _state.durationS || 0,
        },
      }),
    }).catch(() => { /* fire-and-forget */ });
  }

  audio.playbackRate = _state.speed;
  // Record this as the most-recently-played audio item so the next
  // page load can offer a one-tap resume toast (parity with Grove
  // radio/ambient YT). Fire-and-forget; localStorage write only.
  recordLastPlayed({
    kind: 'audio',
    fileId,
    title: _state.title || 'Audiobook',
    subtitle: _state.author || '',
    coverUrl: _state.coverUrl || '',
  });
  try { await audio.play(); } catch (err) {
    // Autoplay policies can reject. Leave state as paused; user hits
    // play again and the gesture requirement is satisfied.
    console.warn('[media-player] play blocked:', err);
    _state.isPlaying = false;
    _state.isLoading = false;
    _notify();
  }
}

export function pause() {
  if (_audio && !_audio.paused) _audio.pause();
}

export function resume() {
  if (_audio && _audio.paused) {
    _audio.play().catch((err) => console.warn('[media-player] resume blocked:', err));
  }
}

export function toggle() {
  if (_audio && !_audio.paused) pause();
  else resume();
}

export function seek(seconds) {
  if (!_audio) return;
  const clamped = Math.max(0, Math.min(_state.durationS || Infinity, seconds));
  // Multi-file: if the target lands in a different MP3, switch files.
  // Within the same file we can seek the audio element directly. Single-
  // file mode: currentTime IS book time, so a direct seek is correct.
  if (_isMultiFile()) {
    const { fileIdx, withinS } = _bookTimeToFile(clamped);
    if (fileIdx !== _state.currentFileIdx) {
      _loadFile(fileIdx, { withinS, autoplay: !_audio.paused });
      MediaSessionBridge.notifyPositionChanged('media-player');
      return;
    }
    try { _audio.currentTime = withinS; } catch { /* before metadata */ }
    MediaSessionBridge.notifyPositionChanged('media-player');
    return;
  }
  try { _audio.currentTime = clamped; } catch { /* before metadata */ }
  // Push the new position to the lock-screen scrubber immediately —
  // the 1Hz poll catches drift, but a +30 skip should land on the
  // widget without a perceivable lag.
  MediaSessionBridge.notifyPositionChanged('media-player');
}

export function skip(delta) {
  if (!_audio) return;
  // Use book-level currentTimeS so crossing a file boundary Just Works.
  // Reading _audio.currentTime directly would give us a within-file time,
  // and +30s near the end of a 20min chapter would silently clamp.
  seek((_state.currentTimeS || 0) + delta);
}

export function skipForward()  { skip(SKIP_FORWARD_S); }
export function skipBackward() { skip(-SKIP_BACK_S); }

export function skipToChapter(idx) {
  const ch = _state.chapters[idx];
  if (ch) seek(ch.start);
}

export function skipChapterRelative(delta) {
  if (!_state.chapters.length) return;
  const cur = Math.max(0, _state.currentChapterIdx);
  const next = Math.max(0, Math.min(_state.chapters.length - 1, cur + delta));
  skipToChapter(next);
}

function _fmtClock(seconds) {
  const s = Math.max(0, Math.floor(Number(seconds) || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
    : `${m}:${String(sec).padStart(2, '0')}`;
}

/**
 * Bookmark the current position. The label auto-derives from the current
 * chapter + timestamp ("Chapter 7 · 1:24:05"); an optional note is the
 * user's own text. Fires `media-player:bookmark-added` so any open
 * detail panel refreshes its list. Returns the saved bookmark (or null).
 */
export async function addBookmarkHere(note = '') {
  if (!_state.fileId) return null;
  const pos = _state.currentTimeS || 0;
  const ch = _state.chapters[_state.currentChapterIdx];
  const clock = _fmtClock(pos);
  const label = ch?.title ? `${ch.title} · ${clock}` : clock;
  const bm = await addBookmark(_state.fileId, {
    position_s: pos,
    label,
    note,
    episode_id: _state.episodeId || '',
  });
  if (bm) {
    window.dispatchEvent(new CustomEvent('media-player:bookmark-added', {
      detail: { fileId: _state.fileId, episodeId: _state.episodeId || '', bookmark: bm },
    }));
  }
  return bm;
}

export function setSpeed(rate) {
  const clamped = Math.max(SPEED_MIN, Math.min(SPEED_MAX, Number(rate) || 1));
  _state.speed = clamped;
  if (_audio) _audio.playbackRate = clamped;
  // setPositionState carries playbackRate so the lock-screen scrubber
  // estimates the right "time remaining" between polls.
  MediaSessionBridge.notifyPositionChanged('media-player');
  _notify();
}

/**
 * Sleep timer.
 *
 *   setSleepTimer(minutes)      — stop in N minutes
 *   setSleepTimer('end-of-chapter') — stop when current chapter ends
 *   setSleepTimer(0) or null     — cancel
 */
export function setSleepTimer(value) {
  _clearSleepTimer();
  if (value === 'end-of-chapter') {
    _state.sleepTimerEndOfChapter = true;
    _state.sleepTimerMs = 0;
    _state.sleepTimerStartedAt = Date.now();
    _armEndOfChapterTimer();
    _notify();
    return;
  }
  const minutes = Number(value) || 0;
  if (minutes <= 0) {
    _notify();
    return;
  }
  _state.sleepTimerMs = minutes * 60_000;
  _state.sleepTimerStartedAt = Date.now();
  _state.sleepTimerEndOfChapter = false;
  _sleepTimerHandle = setTimeout(() => {
    pause();
    _clearSleepTimer();
    _notify();
  }, _state.sleepTimerMs);
  _notify();
}

function _armEndOfChapterTimer() {
  // Poll every 2s; cheap and predictable. We could subscribe to
  // timeupdate internally but that couples the timer to notify frequency.
  _sleepTimerHandle = setInterval(() => {
    const ch = _state.chapters[_state.currentChapterIdx];
    if (!ch) return;
    if (_state.currentTimeS >= ch.end - 0.5) {
      pause();
      _clearSleepTimer();
      _notify();
    }
  }, 2000);
}

function _clearSleepTimer() {
  if (_sleepTimerHandle) {
    clearTimeout(_sleepTimerHandle);
    clearInterval(_sleepTimerHandle);
    _sleepTimerHandle = null;
  }
  _state.sleepTimerMs = 0;
  _state.sleepTimerStartedAt = 0;
  _state.sleepTimerEndOfChapter = false;
}

export function sleepTimerRemainingMs() {
  if (_state.sleepTimerEndOfChapter) return null; // indeterminate
  if (!_state.sleepTimerMs) return 0;
  const elapsed = Date.now() - _state.sleepTimerStartedAt;
  return Math.max(0, _state.sleepTimerMs - elapsed);
}

export function close() {
  if (_audio) {
    _audio.pause();
    _audio.removeAttribute('src');
    _audio.load();
  }
  _clearSleepTimer();
  Object.assign(_state, {
    fileId: '',
    episodeId: '',
    title: '',
    author: '',
    narrator: '',
    coverUrl: '',
    streamUrl: '',
    durationS: 0,
    currentTimeS: 0,
    isPlaying: false,
    isLoading: false,
    isFinished: false,
    chapters: [],
    currentChapterIdx: -1,
    audioFiles: [],
    currentFileIdx: 0,
    pendingFileSeek: 0,
  });
  _notify();
}

// Convenience for consumers that want a quick check without subscribing.
export function isActive() {
  return !!_state.fileId;
}

/**
 * Volume control (companion media.volume channel + future UI slider).
 * Percent scale 0-100. When the bus has us ducked, the duck baseline
 * is what unduck restores — so adjust THAT, not the live (ducked)
 * element volume, or the user's change evaporates when TTS finishes.
 */
export function setVolume(pct) {
  if (!_audio) return false;
  const v = Math.max(0, Math.min(100, Number(pct) || 0)) / 100;
  if (_mediaDuckBaseline !== null) _mediaDuckBaseline = v;
  else _audio.volume = v;
  return true;
}

export function adjustVolume(deltaPct) {
  if (!_audio) return false;
  const base = _mediaDuckBaseline !== null ? _mediaDuckBaseline : _audio.volume;
  return setVolume(base * 100 + (Number(deltaPct) || 0));
}

export function setMuted(muted) {
  if (!_audio) return false;
  _audio.muted = !!muted;
  return true;
}

// App menu: the companion can press this via app.act ("wait, I missed
// that", "go back a bit"). Arg-less, context-bound (only live while
// something is loaded), and reversible — the head of the "replay the
// last bit" ask without burning a dedicated verb on it. Liveness
// re-syncs from _notify() below, which fires on every load/close.
import('./command-palette.js').then(({ registerCommand }) => {
  registerCommand({
    id: 'media.skip-back-bit',
    label: 'Skip back 15 seconds',
    group: 'Media',
    keywords: 'skip back rewind replay missed that say again repeat',
    when: () => isActive(),
    agent: {
      description: 'Skip the current playback back 15 seconds to replay the last bit',
      speak: 'Backed it up a touch.',
    },
    run: () => skipBackward(),
  });
  // Skip-forward joins skip-back (wiring program Phase 1) — same
  // shape: arg-less, live only while something is loaded, reversible.
  registerCommand({
    id: 'media.skip-forward-bit',
    label: 'Skip forward 30 seconds',
    group: 'Media',
    keywords: 'skip forward ahead jump past boring intro',
    when: () => isActive(),
    agent: {
      description: 'Skip the current playback forward 30 seconds',
      speak: 'Jumped ahead a bit.',
    },
    run: () => skipForward(),
  });
  // Bookmark the current spot — "remember this", "bookmark this part",
  // "mark my place". Arg-less, live only while something is loaded; the
  // label auto-derives from chapter + timestamp.
  registerCommand({
    id: 'media.bookmark-here',
    label: 'Bookmark this spot',
    group: 'Media',
    keywords: 'bookmark mark my place remember this spot save position note',
    when: () => isActive(),
    agent: {
      description: 'Save a bookmark at the current playback position',
      speak: 'Marked your place.',
    },
    run: () => addBookmarkHere(),
  });
}).catch(() => {});

// Discovery "Resume listening" cards dispatch this; start (or resume)
// playback on the clicked file. Audiobook-kind only for now — other
// kinds would want a different surface (video player / image viewer).
window.addEventListener('discovery:open-file', (e) => {
  const { file_id, kind } = e.detail || {};
  if (!file_id || kind !== 'audiobook') return;
  play(file_id).catch((err) => {
    console.warn('[media-player] discovery open-file failed:', err);
  });
});
