/**
 * Grove Ambient Window — the orb.
 *
 * A circular video viewport in the Grove panel. It primarily hosts the
 * YouTube ambient player, but can temporarily adopt a local/media-server
 * HTML5 video element for music videos while keeping the same detachable
 * orb shell, seek ring, and hover controls.
 *
 * Exports:
 *   init()                — wire DOM, load favorites, restore last video
 *   loadVideo(video)      — load a video into the orb from Discover
 *   getState()            — current state for grove.js coordination
 *   addFavorite(video)    — add to favorites
 *   removeFavorite(id)    — remove from favorites
 *   isFavorite(id)        — check if in favorites
 */

import { loadYouTubeAPI } from './yt-api.js';
import { escapeHtml, showToast } from './app.js';
import { AudioBus } from './audio-bus.js';
import { recordLastPlayed } from './grove-resume.js';

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $ = id => document.getElementById(id);

let _dom = {};

function _cacheDom() {
  _dom = {
    section:    $('grove-ambient-section'),
    container:  $('grove-orb-container'),
    aura:       $('grove-orb-aura'),
    bleed:      $('grove-orb-bleed'),
    arc:        $('grove-orb-arc'),
    arcFill:    $('grove-orb-arc-fill'),
    arcHit:     $('grove-orb-arc-hit'),
    arcDot:     $('grove-orb-arc-dot'),
    seekTooltip:$('grove-orb-seek-tooltip'),
    orb:        $('grove-orb'),
    iframeWrap: $('grove-orb-iframe-wrap'),
    empty:      $('grove-orb-empty'),
    hover:      $('grove-orb-hover'),
    musicOverlay: $('grove-orb-music-overlay'),
    playBtn:    $('grove-orb-play'),
    playIcon:   $('grove-orb-play-icon'),
    prevBtn:    $('grove-orb-prev'),
    nextBtn:    $('grove-orb-next'),
    loopBtn:    $('grove-orb-loop'),
    loopIcon:   $('grove-orb-loop-icon'),
    playlistBtn:$('grove-orb-playlist'),
    slider:     $('grove-ambient-slider'),
    volumeWrap: $('grove-ambient-volume'),
    title:      $('grove-ambient-title'),
    meta:       $('grove-ambient-meta'),
    favBtn:     $('grove-ambient-fav'),
    chips:      $('grove-ambient-chips'),
  };
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let _player = null;
let _currentVideo = null;    // { sourceType, videoId?, title, channel, thumbnail, ... }
let _currentSourceType = '';
let _isPlaying = false;
let _favorites = [];
let _favIndex = -1;
let _hoverTimeout = null;
let _watchTimer = null;      // 30s discovery signal
let _emitted30s = false;
let _reconnectAttempts = 0;
const _MAX_RECONNECTS = 3;
let _errorTitleTimeout = null;
let _errorColorTimeout = null;
let _html5Cleanup = null;
let _sessionCloseHandler = null;
let _resumeYouTubeVideo = null;
let _loopMode = 'off';   // 'off' | 'loop' | 'advance'
const _LOOP_MODES = ['off', 'loop', 'advance'];
const _LOOP_ICONS = {
  // Material-style "repeat" arrows
  off:     '<path d="M7 7h10v3l4-4-4-4v3H5v6h2V7zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2v4z"/>',
  advance: '<path d="M7 7h10v3l4-4-4-4v3H5v6h2V7zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2v4z"/>',
  // "repeat-1" — same arrows + a centered "1"
  loop:    '<path d="M7 7h10v3l4-4-4-4v3H5v6h2V7zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2v4zm-4-1V8h-1l-2 1v1h1.5v6H13z"/>',
};
const _LOOP_TITLES = {
  off: 'Repeat: off',
  loop: 'Repeat: this track',
  advance: 'Repeat: cycle favorites',
};

// Bus registration: YouTube player volume is 0-100 and can't ramp, so we
// step to the ducked level directly. Baseline is whatever the user set.
let _ambientDuckBaseline = null;
const _ambientBusHandle = AudioBus.register({
  id: 'grove-ambient-yt',
  tier: 'ambient',
  // Grove YT plays user-picked music tracks AND lo-fi/ambient bg.
  // Tagging 'music' so the Becca widget dances when Grove is on —
  // matches user intent (they picked a song to listen + dance to).
  kind: 'music',
  duck: (level) => {
    if (!_player || !_player.getVolume || _ambientDuckBaseline !== null) return;
    try {
      _ambientDuckBaseline = _player.getVolume();
      _player.setVolume(Math.round(_ambientDuckBaseline * level));
    } catch { _ambientDuckBaseline = null; }
  },
  unduck: () => {
    if (!_player || _ambientDuckBaseline === null) return;
    try { _player.setVolume(_ambientDuckBaseline); } catch { /* player gone */ }
    _ambientDuckBaseline = null;
  },
  // Music is exclusive on the bus: when another music source (Grove
  // radio, local music file) starts, pause the orb instead of stacking
  // two tracks. The pause flows through the normal state-change handler
  // (release + play-icon update), so no extra bookkeeping here.
  stop: () => { try { _player?.pauseVideo?.(); } catch { /* player gone */ } },
});

// ---------------------------------------------------------------------------
// Keyword → glow color mapping
// ---------------------------------------------------------------------------
const _COLOR_MAP = [
  { keywords: ['rain', 'storm', 'ocean', 'water', 'waves', 'sea'], rgb: '96, 165, 250', hex: '#60a5fa' },
  { keywords: ['fire', 'fireplace', 'cozy', 'warm', 'candle', 'campfire'], rgb: '249, 115, 22', hex: '#f97316' },
  { keywords: ['lo-fi', 'lofi', 'study', 'chill', 'beats', 'chillhop'], rgb: '167, 139, 250', hex: '#a78bfa' },
  { keywords: ['nature', 'forest', 'birds', 'garden', 'creek', 'meadow'], rgb: '52, 211, 153', hex: '#34d399' },
  { keywords: ['night', 'city', 'neon', 'cyberpunk', 'urban', 'downtown'], rgb: '244, 114, 182', hex: '#f472b6' },
  { keywords: ['jazz', 'piano', 'classical', 'orchestra', 'violin'], rgb: '245, 158, 11', hex: '#f59e0b' },
  { keywords: ['space', 'cosmos', 'stars', 'galaxy', 'nebula', 'universe'], rgb: '99, 102, 241', hex: '#6366f1' },
  { keywords: ['snow', 'winter', 'ice', 'arctic', 'frozen', 'blizzard'], rgb: '147, 197, 253', hex: '#93c5fd' },
  { keywords: ['sunrise', 'sunset', 'dawn', 'golden', 'morning'], rgb: '251, 146, 60', hex: '#fb923c' },
];
const _DEFAULT_COLOR = { rgb: '52, 211, 153', hex: '#34d399' };

function _keywordColor(title) {
  const lower = (title || '').toLowerCase();
  for (const entry of _COLOR_MAP) {
    if (entry.keywords.some(kw => lower.includes(kw))) return entry;
  }
  return _DEFAULT_COLOR;
}

function _applyColor(color) {
  if (!_dom.section) return;
  _dom.section.style.setProperty('--orb-color', color.rgb);
  _dom.section.style.setProperty('--orb-hex', color.hex);
}

function _isYouTubeVideo(video = _currentVideo, sourceType = _currentSourceType) {
  return !!(video?.videoId) && String(video?.sourceType || sourceType || 'youtube') === 'youtube';
}

// Playlist controller registers a callback here so we don't create an
// import cycle (playlist.js imports loadVideo/loadMediaVideo from us).
// Returns true if the playlist consumed the ended event.
let _onEndedHook = null;
export function setEndedHook(fn) {
  _onEndedHook = typeof fn === 'function' ? fn : null;
}
function _notifyPlaylistEnded() {
  if (typeof _onEndedHook !== 'function') return false;
  try { return _onEndedHook() === true; }
  catch (err) { console.warn('[Grove Ambient] Playlist ended hook failed:', err); return false; }
}

function _isLocalMediaVideo(video = _currentVideo, sourceType = _currentSourceType) {
  return String(video?.sourceType || sourceType || '').trim() === 'media_server';
}

function _favoriteKey(value) {
  if (!value) return '';
  if (typeof value === 'string') {
    const raw = String(value || '').trim();
    if (!raw) return '';
    return raw.includes(':') ? raw : `youtube:${raw}`;
  }
  if (_isLocalMediaVideo(value, value?.sourceType) && value?.fileId) {
    return `media_server:${String(value.fileId).trim()}`;
  }
  if (value?.videoId) {
    return `youtube:${String(value.videoId).trim()}`;
  }
  return '';
}

function _flushSessionCloseHandler() {
  if (typeof _sessionCloseHandler !== 'function') return;
  try {
    _sessionCloseHandler();
  } catch (err) {
    console.warn('[Grove Ambient] Session close hook failed:', err);
  }
  _sessionCloseHandler = null;
}

async function _openFavoriteEntry(entry) {
  if (!entry) return false;
  if (_isLocalMediaVideo(entry, entry?.sourceType) && entry?.fileId) {
    try {
      const mod = await import('./files/preview.js');
      const opened = await mod.openVideoPreviewById?.(entry.fileId);
      if (!opened) {
        showToast('Could not reopen that saved music video.', 'error');
      }
      return !!opened;
    } catch (err) {
      console.warn('[Grove Ambient] Failed to reopen media favorite:', err);
      showToast('Could not reopen that saved music video.', 'error');
      return false;
    }
  }
  if (entry.videoId) {
    await loadVideo(entry);
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Seek arc — circular playback progress + scrub target
// ---------------------------------------------------------------------------
let _progressTimer = null;
let _scrubbing = false;
let _duration = 0;

function _updateArc(fraction) {
  if (!_dom.arcFill) return;
  fraction = Math.max(0, Math.min(1, fraction || 0));
  // Full circle = 660 (2π × 105). Offset 660 = empty, 0 = full.
  _dom.arcFill.setAttribute('stroke-dashoffset', 660 * (1 - fraction));
  if (_dom.arcDot) _dom.arcDot.style.transform = `rotate(${fraction * 360}deg)`;
  _dom.arc?.setAttribute('aria-valuenow', Math.round(fraction * 100));
}

function _formatTime(s) {
  if (!isFinite(s) || s < 0) s = 0;
  const total = Math.floor(s);
  const m = Math.floor(total / 60);
  const ss = (total % 60).toString().padStart(2, '0');
  if (m >= 60) {
    const h = Math.floor(m / 60);
    const mm = (m % 60).toString().padStart(2, '0');
    return `${h}:${mm}:${ss}`;
  }
  return `${m}:${ss}`;
}

function _pointerToFraction(e) {
  const rect = _dom.arc.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  // 12 o'clock = 0, sweeping clockwise to 1
  let angle = Math.atan2(e.clientX - cx, -(e.clientY - cy));
  if (angle < 0) angle += Math.PI * 2;
  return angle / (Math.PI * 2);
}

function _showTooltip(e, fraction) {
  const tip = _dom.seekTooltip;
  if (!tip || !_dom.section) return;
  const rect = _dom.section.getBoundingClientRect();
  tip.hidden = false;
  tip.style.left = (e.clientX - rect.left) + 'px';
  tip.style.top = (e.clientY - rect.top) + 'px';
  tip.textContent = `${_formatTime(fraction * _duration)} / ${_formatTime(_duration)}`;
}

function _hideTooltip() {
  if (_dom.seekTooltip) _dom.seekTooltip.hidden = true;
}

function _initSeek() {
  const arc = _dom.arc;
  const hit = _dom.arcHit;
  if (!arc || !hit) return;

  hit.addEventListener('pointermove', (e) => {
    if (arc.classList.contains('disabled')) return;
    const f = _pointerToFraction(e);
    _showTooltip(e, f);
    if (_scrubbing) _updateArc(f);
  });

  hit.addEventListener('pointerleave', () => {
    if (!_scrubbing) _hideTooltip();
  });

  hit.addEventListener('pointerdown', (e) => {
    if (arc.classList.contains('disabled') || !_player) return;
    _scrubbing = true;
    arc.classList.add('scrubbing');
    try { hit.setPointerCapture(e.pointerId); } catch { /* unsupported */ }
    const f = _pointerToFraction(e);
    _updateArc(f);
    _showTooltip(e, f);
    e.preventDefault();
  });

  const endScrub = (e, commit) => {
    if (!_scrubbing) return;
    _scrubbing = false;
    arc.classList.remove('scrubbing');
    try { hit.releasePointerCapture(e.pointerId); } catch { /* unsupported */ }
    if (commit && _player?.seekTo && _duration > 0) {
      const f = _pointerToFraction(e);
      _player.seekTo(f * _duration, true);
    }
    _hideTooltip();
  };
  hit.addEventListener('pointerup', (e) => endScrub(e, true));
  hit.addEventListener('pointercancel', (e) => endScrub(e, false));

  // Keyboard: ±10s with arrows, Home/End for jump-to
  arc.addEventListener('keydown', (e) => {
    if (!_player || _duration <= 0) return;
    const cur = _player.getCurrentTime?.() ?? 0;
    let next = null;
    if (e.key === 'ArrowRight') next = cur + 10;
    else if (e.key === 'ArrowLeft') next = cur - 10;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = Math.max(0, _duration - 1);
    if (next !== null) {
      e.preventDefault();
      _player.seekTo(Math.max(0, Math.min(_duration, next)), true);
    }
  });
}

function _startProgressTimer() {
  _stopProgressTimer();
  _progressTimer = setInterval(() => {
    if (_scrubbing || !_player?.getCurrentTime) return;
    const dur = _player.getDuration?.() ?? 0;
    if (!dur || !isFinite(dur)) {
      // Livestream / unknown — disable seek, keep ring empty
      _dom.arc?.classList.add('disabled');
      _dom.arc?.classList.remove('has-duration');
      _duration = 0;
      _updateArc(0);
      return;
    }
    _duration = dur;
    _dom.arc?.classList.remove('disabled');
    _dom.arc?.classList.add('has-duration');
    _updateArc((_player.getCurrentTime() || 0) / dur);
  }, 250);
}

function _stopProgressTimer() {
  if (_progressTimer) { clearInterval(_progressTimer); _progressTimer = null; }
}

function _teardownActivePlayer({ pause = true, flush = true } = {}) {
  _clearWatchTimer();
  _stopProgressTimer();
  _ambientBusHandle?.release();
  _dom.orb?.classList.remove('buffering');

  if (flush) _flushSessionCloseHandler();

  if (_html5Cleanup) {
    try {
      _html5Cleanup({ pause });
    } catch (err) {
      console.warn('[Grove Ambient] HTML5 cleanup failed:', err);
    }
    _html5Cleanup = null;
  } else if (_player && typeof _player.destroy === 'function') {
    try {
      _player.destroy();
    } catch {
      /* destroy can throw during teardown */
    }
  }

  if (_dom.iframeWrap) _dom.iframeWrap.innerHTML = '';
  _player = null;
  _isPlaying = false;
  _updatePlayIcon();
}

// ---------------------------------------------------------------------------
// Player lifecycle
// ---------------------------------------------------------------------------
async function _createPlayer(videoId) {
  const YT = await loadYouTubeAPI();

  // Clear previous
  if (_player && _player.destroy) {
    try { _player.destroy(); } catch { /* destroy can throw during teardown */ }
  }
  _dom.iframeWrap.innerHTML = '<div id="grove-orb-yt-player"></div>';

  return new Promise((resolve) => {
    _player = new YT.Player('grove-orb-yt-player', {
      videoId,
      width: 362,
      height: 204,
      playerVars: {
        controls: 0,
        modestbranding: 1,
        rel: 0,
        showinfo: 0,
        fs: 0,
        playsinline: 1,
        autoplay: 0,
        iv_load_policy: 3,
      },
      events: {
        onReady: () => resolve(_player),
        onStateChange: _onStateChange,
        onError: _onError,
      },
    });
  });
}

function _onStateChange(event) {
  const state = event.data;
  if (state === 1) { // PLAYING
    _isPlaying = true;
    _reconnectAttempts = 0;
    _updatePlayIcon();
    _startWatchTimer();
    _startProgressTimer();
    _ambientBusHandle?.claim();
  } else if (state === 2) { // PAUSED
    _isPlaying = false;
    _updatePlayIcon();
    _clearWatchTimer();
    _ambientBusHandle?.release();
  } else if (state === 3) { // BUFFERING
    _dom.orb?.classList.add('buffering');
  } else if (state === 0) { // ENDED
    _isPlaying = false;
    _updatePlayIcon();
    _clearWatchTimer();
    _ambientBusHandle?.release();
    _stopProgressTimer();
    if (_currentVideo?.isLivestream) {
      _tryReconnect();
    } else if (_loopMode === 'loop') {
      try { _player?.seekTo?.(0, true); _player?.playVideo?.(); }
      catch (err) { console.warn('[Grove Ambient] Loop replay failed:', err); }
    } else if (_notifyPlaylistEnded()) {
      // Playlist is driving — it advanced to the next item.
    } else if (_loopMode === 'advance') {
      _playNextFavorite();
    }
    // 'off' — leave stopped at the end
  }
  if (state !== 3) {
    _dom.orb?.classList.remove('buffering');
  }
}

function _onError(event) {
  console.warn('[Grove Ambient] Player error:', event.data);
  _isPlaying = false;
  _updatePlayIcon();
  _clearWatchTimer();

  if (event.data === 100 || event.data === 101 || event.data === 150) {
    _showError('Video unavailable');
  } else {
    _tryReconnect();
  }
}

function _tryReconnect() {
  if (_reconnectAttempts >= _MAX_RECONNECTS) {
    _showError('Stream ended');
    return;
  }
  _reconnectAttempts++;
  const delay = Math.pow(2, _reconnectAttempts) * 1000;
  setTimeout(() => {
    if (_isYouTubeVideo() && _currentVideo && _player?.loadVideoById) {
      _player.loadVideoById(_currentVideo.videoId);
    }
  }, delay);
}

function _showError(msg) {
  _setVisualState('empty');
  if (_dom.title) _dom.title.textContent = msg;
  clearTimeout(_errorTitleTimeout);
  clearTimeout(_errorColorTimeout);
  _errorTitleTimeout = setTimeout(() => {
    if (!_isPlaying && _dom.title) {
      _dom.title.textContent = '';
    }
  }, 3000);
  _errorColorTimeout = setTimeout(() => {
    if (!_isPlaying) _applyColor(_DEFAULT_COLOR);
  }, 2000);
  _currentVideo = null;
  _persistVideo();
}

function _mountHtml5MediaPlayer(element, api = null) {
  if (!_dom.iframeWrap || !element) return null;
  const mediaEl = element;
  mediaEl.controls = false;
  mediaEl.playsInline = true;
  mediaEl.setAttribute('playsinline', '');
  mediaEl.classList.add('grove-orb-media-el');

  const adapter = {
    _sourceType: 'media_server',
    playVideo: () => {
      if (api?.play) {
        Promise.resolve(api.play()).catch(() => {});
        return;
      }
      mediaEl.play().catch(() => {});
    },
    pauseVideo: () => {
      if (api?.pause) {
        api.pause();
        return;
      }
      mediaEl.pause();
    },
    getCurrentTime: () => {
      if (api?.getCurrentTime) return Math.max(0, Number(api.getCurrentTime() || 0));
      return Math.max(0, Number(mediaEl.currentTime || 0));
    },
    getDuration: () => {
      if (api?.getDuration) return Math.max(0, Number(api.getDuration() || 0));
      return Number.isFinite(mediaEl.duration) ? Math.max(0, Number(mediaEl.duration || 0)) : 0;
    },
    seekTo: (timeS) => {
      if (api?.seekTo) return api.seekTo(timeS);
      try {
        mediaEl.currentTime = Math.max(0, Number(timeS || 0));
        return true;
      } catch {
        return false;
      }
    },
    setVolume: (value) => {
      const next = Math.max(0, Math.min(1, Number(value || 0) / 100));
      if (api?.setVolume) {
        api.setVolume(next);
        return;
      }
      mediaEl.volume = next;
    },
    getVolume: () => {
      const raw = api?.getVolume ? Number(api.getVolume() || 0) : Number(mediaEl.volume || 0);
      return Math.max(0, Math.min(100, Math.round(raw * 100)));
    },
  };

  const onPlaying = () => {
    _isPlaying = true;
    _updatePlayIcon();
    _startProgressTimer();
    _ambientBusHandle?.claim();
    _dom.orb?.classList.remove('buffering');
  };
  const onPause = () => {
    if (mediaEl.ended) return;
    _isPlaying = false;
    _updatePlayIcon();
    _ambientBusHandle?.release();
    _dom.orb?.classList.remove('buffering');
  };
  const onWaiting = () => {
    _dom.orb?.classList.add('buffering');
  };
  const onEnded = () => {
    _isPlaying = false;
    _updatePlayIcon();
    _ambientBusHandle?.release();
    _stopProgressTimer();
    _updateArc(1);
    _dom.orb?.classList.remove('buffering');
    if (_loopMode === 'loop') {
      try {
        adapter.seekTo(0);
        adapter.playVideo();
      } catch (err) { console.warn('[Grove Ambient] Loop replay (html5) failed:', err); }
    } else {
      _notifyPlaylistEnded();
    }
  };
  const onLoadedMetadata = () => {
    const dur = adapter.getDuration();
    if (dur > 0 && isFinite(dur)) {
      _duration = dur;
      _dom.arc?.classList.remove('disabled');
      _dom.arc?.classList.add('has-duration');
    } else {
      _duration = 0;
      _dom.arc?.classList.add('disabled');
      _dom.arc?.classList.remove('has-duration');
      _updateArc(0);
    }
  };

  mediaEl.addEventListener('playing', onPlaying);
  mediaEl.addEventListener('pause', onPause);
  mediaEl.addEventListener('waiting', onWaiting);
  mediaEl.addEventListener('seeking', onWaiting);
  mediaEl.addEventListener('canplay', onLoadedMetadata);
  mediaEl.addEventListener('loadedmetadata', onLoadedMetadata);
  mediaEl.addEventListener('ended', onEnded);

  _dom.iframeWrap.innerHTML = '';
  _dom.iframeWrap.appendChild(mediaEl);

  _html5Cleanup = ({ pause = true } = {}) => {
    mediaEl.removeEventListener('playing', onPlaying);
    mediaEl.removeEventListener('pause', onPause);
    mediaEl.removeEventListener('waiting', onWaiting);
    mediaEl.removeEventListener('seeking', onWaiting);
    mediaEl.removeEventListener('canplay', onLoadedMetadata);
    mediaEl.removeEventListener('loadedmetadata', onLoadedMetadata);
    mediaEl.removeEventListener('ended', onEnded);
    if (pause) {
      try { mediaEl.pause(); } catch { /* ignore */ }
    }
    try { mediaEl.remove(); } catch { /* ignore */ }
  };

  return adapter;
}

// ---------------------------------------------------------------------------
// Visual state management
// ---------------------------------------------------------------------------
function _setVisualState(state) {
  if (!_dom.empty || !_dom.iframeWrap) return;

  if (state === 'playing') {
    _dom.empty.classList.add('hidden');
    _dom.iframeWrap.classList.add('active');
    _dom.musicOverlay?.classList.toggle('active', _isLocalMediaVideo());
  } else {
    _dom.empty.classList.remove('hidden');
    _dom.iframeWrap.classList.remove('active');
    _dom.hover?.classList.remove('visible');
    _dom.musicOverlay?.classList.remove('active');
  }
}

function _updatePlayIcon() {
  if (!_dom.playIcon) return;
  _dom.playIcon.innerHTML = _isPlaying
    ? '<rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/>'
    : '<path d="M8 5v14l11-7z"/>';
}

function _updateTrackInfo() {
  if (!_dom.title || !_dom.meta) return;
  if (!_currentVideo) {
    _dom.title.textContent = '';
    _dom.meta.textContent = '';
    if (_dom.prevBtn) _dom.prevBtn.hidden = true;
    if (_dom.nextBtn) _dom.nextBtn.hidden = true;
    _updateFavButton();
    return;
  }
  _dom.title.textContent = _currentVideo.title || '';
  const parts = [];
  if (_currentVideo.channel) parts.push(escapeHtml(_currentVideo.channel));
  if (_isYouTubeVideo() && _currentVideo.isLivestream) parts.push('<span style="color:var(--orb-hex)">LIVE</span>');
  else if (_currentVideo.duration) parts.push(escapeHtml(_currentVideo.duration));
  _dom.meta.innerHTML = parts.join(' \u00b7 ');
  if (_dom.prevBtn) _dom.prevBtn.hidden = !_isYouTubeVideo();
  if (_dom.nextBtn) _dom.nextBtn.hidden = !_isYouTubeVideo();
  _updateFavButton();
  _updateLoopButton();
}

function _updateFavButton() {
  const btn = _dom.favBtn;
  if (!btn) return;
  if (!_currentVideo) {
    btn.hidden = true;
    return;
  }
  btn.hidden = false;
  const faved = isFavorite(_currentVideo);
  btn.classList.toggle('active', faved);
  btn.setAttribute('aria-pressed', faved ? 'true' : 'false');
  btn.title = faved ? 'Remove from favorites' : 'Add to favorites';
}

function _initFavButton() {
  _dom.favBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!_currentVideo) return;
    if (isFavorite(_currentVideo)) {
      removeFavorite(_currentVideo);
    } else {
      addFavorite(_currentVideo);
    }
    _updateFavButton();
  });
}

function _updateLoopButton() {
  const btn = _dom.loopBtn;
  const icon = _dom.loopIcon;
  if (!btn || !icon) return;
  // Hide for livestreams — there's no end to control
  btn.hidden = !!_currentVideo?.isLivestream;
  icon.innerHTML = _LOOP_ICONS[_loopMode] || _LOOP_ICONS.off;
  btn.title = _LOOP_TITLES[_loopMode] || _LOOP_TITLES.off;
  btn.classList.toggle('active', _loopMode !== 'off');
  btn.setAttribute('aria-pressed', _loopMode !== 'off' ? 'true' : 'false');
  btn.setAttribute('aria-label', _LOOP_TITLES[_loopMode] || _LOOP_TITLES.off);
}

function _setLoopMode(mode, { persist = true } = {}) {
  if (!_LOOP_MODES.includes(mode)) mode = 'off';
  _loopMode = mode;
  _updateLoopButton();
  if (!persist) return;
  if (window.appSettings) window.appSettings.ambientLoopMode = mode;
  fetch('/api/config/tools', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ambient_loop_mode: mode }),
  }).catch((e) => console.warn('[Grove Ambient] Loop mode sync failed:', e));
}

function _initLoopButton() {
  _dom.loopBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    const next = _LOOP_MODES[(_LOOP_MODES.indexOf(_loopMode) + 1) % _LOOP_MODES.length];
    _setLoopMode(next);
  });
}

// ---------------------------------------------------------------------------
// Hover controls
// ---------------------------------------------------------------------------
function _initHover() {
  if (!_dom.orb) return;

  _dom.orb.addEventListener('mouseenter', () => {
    if (!_currentVideo) return;
    _dom.hover?.classList.add('visible');
    _dom.volumeWrap?.classList.add('always-visible');
    _clearHoverTimeout();
    _startHoverTimeout();
  });

  _dom.orb.addEventListener('mousemove', () => {
    if (!_dom.hover?.classList.contains('visible')) return;
    _clearHoverTimeout();
    _startHoverTimeout();
  });

  _dom.orb.addEventListener('mouseleave', () => {
    _dom.hover?.classList.remove('visible');
    _dom.volumeWrap?.classList.remove('always-visible');
    _clearHoverTimeout();
  });

  _dom.playBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!_player) return;
    if (_isPlaying) _player.pauseVideo?.();
    else _player.playVideo?.();
  });

  _dom.prevBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!_isYouTubeVideo()) return;
    _playPrevFavorite();
  });
  _dom.nextBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!_isYouTubeVideo()) return;
    _playNextFavorite();
  });

  _dom.playlistBtn?.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!_currentVideo) return;
    let detail = null;
    if (_isYouTubeVideo()) {
      detail = {
        type: 'youtube',
        videoId: _currentVideo.videoId,
        title: _currentVideo.title || '',
        channel: _currentVideo.channel || '',
        thumbnail: _currentVideo.thumbnail
          || `https://i.ytimg.com/vi/${_currentVideo.videoId}/mqdefault.jpg`,
      };
    } else if (_isLocalMediaVideo() && _currentVideo.fileId) {
      // entityKind is the file source's classification (music_video, episode, ...)
      // The playlist contract only cares about playback kind: audio/video.
      detail = {
        type: 'file',
        fileId: _currentVideo.fileId,
        name: _currentVideo.title || '',
        kind: 'video',
        thumbnail: _currentVideo.thumbnail || '',
      };
    }
    if (detail) {
      window.dispatchEvent(new CustomEvent('playlist:add-item', { detail }));
    }
  });

  _dom.empty?.addEventListener('click', () => {
    document.dispatchEvent(new CustomEvent('grove:open-discover-youtube'));
  });
}

function _startHoverTimeout() {
  _hoverTimeout = setTimeout(() => {
    _dom.hover?.classList.remove('visible');
  }, 2000);
}

function _clearHoverTimeout() {
  if (_hoverTimeout) {
    clearTimeout(_hoverTimeout);
    _hoverTimeout = null;
  }
}

// ---------------------------------------------------------------------------
// Volume
// ---------------------------------------------------------------------------
let _volSaveTimeout = null;

function _initVolume() {
  if (!_dom.slider) return;

  const vol = window.appSettings?.ambientVolume ?? 50;
  _dom.slider.value = vol;

  _dom.slider.addEventListener('input', () => {
    const v = parseInt(_dom.slider.value, 10);
    if (_player && _player.setVolume) _player.setVolume(v);

    clearTimeout(_volSaveTimeout);
    _volSaveTimeout = setTimeout(() => {
      if (window.appSettings) window.appSettings.ambientVolume = v;
      _syncVolume(v);
    }, 300);
  });
}

async function _syncVolume(v) {
  try {
    await fetch('/api/config/tools', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ambient_volume: v }),
    });
  } catch (e) { console.warn('[Grove Ambient] Volume sync failed:', e); }
}

// ---------------------------------------------------------------------------
// Favorites
// ---------------------------------------------------------------------------
async function _loadFavorites() {
  try {
    const resp = await fetch('/api/grove/ambient-favorites', { credentials: 'same-origin' });
    if (resp.ok) {
      const data = await resp.json();
      if (Array.isArray(data)) _favorites = data;
    }
  } catch (e) { console.warn('[Grove Ambient] Failed to load favorites:', e); }
}

async function _saveFavorites() {
  try {
    await fetch('/api/grove/ambient-favorites', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(_favorites),
    });
  } catch (e) { console.warn('[Grove Ambient] Failed to save favorites:', e); }
}

function _renderChips() {
  if (!_dom.chips) return;
  if (_favorites.length === 0) {
    _dom.chips.innerHTML = '';
    return;
  }

  _dom.chips.innerHTML = _favorites.map((v, i) => {
    const active = _favoriteKey(_currentVideo) === _favoriteKey(v) ? ' active' : '';
    return `<button class="grove-ambient-chip${active}" data-idx="${i}" title="${escapeHtml(v.title)}">
      <img src="${escapeHtml(v.thumbnail)}" alt="" loading="lazy">
      <span class="grove-chip-remove" data-idx="${i}" title="Remove favorite" aria-label="Remove favorite">&times;</span>
    </button>`;
  }).join('');

  // Auto-center the active chip in the scroller
  requestAnimationFrame(() => {
    const active = _dom.chips.querySelector('.grove-ambient-chip.active');
    if (active) active.scrollIntoView({ inline: 'center', block: 'nearest', behavior: 'smooth' });
  });
}

// ---------------------------------------------------------------------------
// Favorites Bloom — long-press a chip to spread all favorites into a 5×6
// grid for drag-reorder without scrolling.
// ---------------------------------------------------------------------------
const _BLOOM_COLS = 5;
const _BLOOM_CELL = 38;
const _BLOOM_GAP = 12;
const _BLOOM_PAD = 18;
const _LONG_PRESS_MS = 280;
const _PRESS_CANCEL_PX = 6;

let _bloomActive = false;
let _bloomOrder = [];
let _bloomCells = [];
let _dragVI = -1;
let _pressTimer = null;
let _pressStart = null;

function _cellPos(vi) {
  const col = vi % _BLOOM_COLS;
  const row = Math.floor(vi / _BLOOM_COLS);
  return {
    x: _BLOOM_PAD + col * (_BLOOM_CELL + _BLOOM_GAP),
    y: _BLOOM_PAD + row * (_BLOOM_CELL + _BLOOM_GAP),
  };
}

function _layoutCells(skipVI = -1) {
  _bloomCells.forEach((cell, vi) => {
    if (vi === skipVI) return;
    const p = _cellPos(vi);
    cell.style.left = p.x + 'px';
    cell.style.top  = p.y + 'px';
  });
}

function _renderBloomGrid() {
  const grid = document.getElementById('grove-fav-bloom-grid');
  if (!grid) return;
  grid.innerHTML = '';
  _bloomCells = [];
  const cols = _BLOOM_COLS;
  const rows = Math.max(1, Math.ceil(_bloomOrder.length / cols));
  grid.style.width  = (_BLOOM_PAD * 2 + cols * _BLOOM_CELL + (cols - 1) * _BLOOM_GAP) + 'px';
  grid.style.height = (_BLOOM_PAD * 2 + rows * _BLOOM_CELL + (rows - 1) * _BLOOM_GAP) + 'px';
  grid.style.display = 'block';

  _bloomOrder.forEach((favIdx, vi) => {
    const v = _favorites[favIdx];
    const cell = document.createElement('button');
    cell.className = 'grove-fav-cell';
    cell.dataset.vi = String(vi);
    if (_favoriteKey(_currentVideo) === _favoriteKey(v)) cell.classList.add('active');
    cell.innerHTML = `<img src="${escapeHtml(v.thumbnail)}" alt="" loading="lazy">`;
    cell.style.position = 'absolute';
    const p = _cellPos(vi);
    cell.style.left = p.x + 'px';
    cell.style.top  = p.y + 'px';
    grid.appendChild(cell);
    _bloomCells.push(cell);
  });
}

function _bloomMove(e) {
  if (!_bloomActive || _dragVI < 0) return;
  const grid = document.getElementById('grove-fav-bloom-grid');
  if (!grid) return;
  const r = grid.getBoundingClientRect();
  const lx = e.clientX - r.left;
  const ly = e.clientY - r.top;
  const dragged = _bloomCells[_dragVI];
  dragged.style.transition = 'none';
  dragged.style.left = (lx - _BLOOM_CELL / 2) + 'px';
  dragged.style.top  = (ly - _BLOOM_CELL / 2) + 'px';

  const col = Math.max(0, Math.min(_BLOOM_COLS - 1,
    Math.round((lx - _BLOOM_PAD - _BLOOM_CELL / 2) / (_BLOOM_CELL + _BLOOM_GAP))));
  const totalRows = Math.ceil(_bloomOrder.length / _BLOOM_COLS);
  const row = Math.max(0, Math.min(totalRows - 1,
    Math.round((ly - _BLOOM_PAD - _BLOOM_CELL / 2) / (_BLOOM_CELL + _BLOOM_GAP))));
  let target = Math.min(_bloomOrder.length - 1, row * _BLOOM_COLS + col);

  if (target !== _dragVI) {
    const [favIdx] = _bloomOrder.splice(_dragVI, 1);
    _bloomOrder.splice(target, 0, favIdx);
    const [cellEl] = _bloomCells.splice(_dragVI, 1);
    _bloomCells.splice(target, 0, cellEl);
    cellEl.dataset.vi = String(target);
    _bloomCells.forEach((c, i) => { c.dataset.vi = String(i); });
    _dragVI = target;
    _layoutCells(_dragVI);
  }
}

function _bloomEnd(e) {
  if (!_bloomActive) return;
  document.removeEventListener('pointermove', _bloomMove);
  document.removeEventListener('pointerup', _bloomEnd);
  document.removeEventListener('pointercancel', _bloomEnd);

  const overlay = document.getElementById('grove-fav-bloom');
  const r = overlay.getBoundingClientRect();
  const inside = e && e.clientX >= r.left && e.clientX <= r.right &&
                       e.clientY >= r.top  && e.clientY <= r.bottom;
  const commit = inside && e?.type !== 'pointercancel';

  // Settle dragged cell into its slot before closing
  const dragged = _bloomCells[_dragVI];
  if (dragged) {
    dragged.style.transition = '';
    dragged.classList.remove('dragging');
    const p = _cellPos(_dragVI);
    dragged.style.left = p.x + 'px';
    dragged.style.top  = p.y + 'px';
  }
  setTimeout(() => _closeBloom(commit), 180);
}

function _openBloom(favIdx) {
  if (_favorites.length < 2) return;
  const overlay = document.getElementById('grove-fav-bloom');
  if (!overlay) return;
  _bloomActive = true;
  _bloomOrder = _favorites.map((_, i) => i);
  _dragVI = _bloomOrder.indexOf(favIdx);
  // Anchor overlay to the panel's current viewport so scroll position
  // doesn't push it out of view.
  const panel = document.getElementById('grove-panel');
  if (panel) {
    overlay.style.top    = panel.scrollTop + 'px';
    overlay.style.left   = '0';
    overlay.style.right  = '0';
    overlay.style.height = panel.clientHeight + 'px';
  }
  overlay.hidden = false;
  void overlay.offsetWidth;
  overlay.classList.add('visible');
  _renderBloomGrid();
  if (_dragVI >= 0) _bloomCells[_dragVI].classList.add('dragging');
  if (navigator.vibrate) { try { navigator.vibrate(12); } catch { /* noop */ } }
  document.addEventListener('pointermove', _bloomMove);
  document.addEventListener('pointerup', _bloomEnd);
  document.addEventListener('pointercancel', _bloomEnd);
}

function _closeBloom(commit) {
  const overlay = document.getElementById('grove-fav-bloom');
  if (commit && _bloomOrder.length === _favorites.length) {
    _favorites = _bloomOrder.map(i => _favorites[i]);
    const currentKey = _favoriteKey(_currentVideo);
    _favIndex = currentKey ? _favorites.findIndex(f => _favoriteKey(f) === currentKey) : -1;
    _saveFavorites();
    _renderChips();
  }
  overlay?.classList.remove('visible');
  setTimeout(() => { if (overlay) overlay.hidden = true; }, 200);
  _bloomActive = false;
  _bloomOrder = [];
  _bloomCells = [];
  _dragVI = -1;
}

function _initBloom() {
  document.addEventListener('keydown', (e) => {
    if (_bloomActive && e.key === 'Escape') {
      document.removeEventListener('pointermove', _bloomMove);
      document.removeEventListener('pointerup', _bloomEnd);
      document.removeEventListener('pointercancel', _bloomEnd);
      _closeBloom(false);
    }
  });
}

function _initChips() {
  // Long-press → bloom. Cancels if pointer moves >6px (treat as scroll/click).
  _dom.chips?.addEventListener('pointerdown', (e) => {
    if (e.target.closest('.grove-chip-remove')) return;
    const chip = e.target.closest('.grove-ambient-chip');
    if (!chip) return;
    const idx = parseInt(chip.dataset.idx, 10);
    if (!Number.isFinite(idx)) return;
    _pressStart = { x: e.clientX, y: e.clientY };
    _pressTimer = setTimeout(() => {
      _pressTimer = null;
      _pressStart = null;
      _openBloom(idx);
    }, _LONG_PRESS_MS);
  });

  const cancelPress = () => {
    if (_pressTimer) { clearTimeout(_pressTimer); _pressTimer = null; }
    _pressStart = null;
  };
  _dom.chips?.addEventListener('pointermove', (e) => {
    if (!_pressStart) return;
    if (Math.hypot(e.clientX - _pressStart.x, e.clientY - _pressStart.y) > _PRESS_CANCEL_PX) {
      cancelPress();
    }
  });
  _dom.chips?.addEventListener('pointerup', cancelPress);
  _dom.chips?.addEventListener('pointercancel', cancelPress);
  _dom.chips?.addEventListener('pointerleave', cancelPress);

  // Wheel: translate vertical scroll to horizontal so desktop wheel/trackpad
  // scrolls the strip without leaving the orb area. Touch is native via overflow-x.
  _dom.chips?.addEventListener('wheel', (e) => {
    if (e.deltaY === 0) return;
    const max = _dom.chips.scrollWidth - _dom.chips.clientWidth;
    if (max <= 0) return;
    const next = _dom.chips.scrollLeft + e.deltaY;
    // Only intercept when we can actually scroll in that direction
    if ((e.deltaY > 0 && _dom.chips.scrollLeft < max) ||
        (e.deltaY < 0 && _dom.chips.scrollLeft > 0)) {
      e.preventDefault();
      _dom.chips.scrollLeft = next;
    }
  }, { passive: false });

  _dom.chips?.addEventListener('click', (e) => {
    // Suppress click that follows a long-press / bloom interaction
    if (_bloomActive) { e.preventDefault(); e.stopPropagation(); return; }
    // Remove button on chip
    const removeBtn = e.target.closest('.grove-chip-remove');
    if (removeBtn) {
      e.stopPropagation();
      const idx = parseInt(removeBtn.dataset.idx, 10);
      if (_favorites[idx]) {
        removeFavorite(_favorites[idx]);
        _renderChips();
      }
      return;
    }
    const chip = e.target.closest('.grove-ambient-chip');
    if (!chip) return;
    const idx = parseInt(chip.dataset.idx, 10);
    if (_favorites[idx]) void _openFavoriteEntry(_favorites[idx]);
  });
}

function _playNextFavorite() {
  if (_favorites.length === 0) return;
  _favIndex = (_favIndex + 1) % _favorites.length;
  void _openFavoriteEntry(_favorites[_favIndex]);
}

function _playPrevFavorite() {
  if (_favorites.length === 0) return;
  _favIndex = (_favIndex - 1 + _favorites.length) % _favorites.length;
  void _openFavoriteEntry(_favorites[_favIndex]);
}

export function addFavorite(video) {
  const key = _favoriteKey(video);
  if (!key) return false;
  if (_favorites.some(f => _favoriteKey(f) === key)) return false;
  _favorites.push(video);
  if (_favorites.length > 30) _favorites.shift();
  _saveFavorites();
  _renderChips();
  _updateFavButton();
  return true;
}

export function removeFavorite(videoId) {
  const key = _favoriteKey(videoId);
  const idx = _favorites.findIndex(f => _favoriteKey(f) === key);
  if (idx === -1) return false;
  _favorites.splice(idx, 1);
  _saveFavorites();
  _renderChips();
  _updateFavButton();
  return true;
}

export function isFavorite(videoId) {
  const key = _favoriteKey(videoId);
  return !!key && _favorites.some(f => _favoriteKey(f) === key);
}

// ---------------------------------------------------------------------------
// Discovery signals
// ---------------------------------------------------------------------------
function _emitSignal(signalType, data = {}) {
  if (!window.appSettings?.discoveryEnabled) return;
  const body = { signal_type: signalType, ...data };
  fetch('/api/discovery/signal', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).catch(() => {});
}

function _startWatchTimer() {
  _clearWatchTimer();
  if (!_isYouTubeVideo()) return;
  if (_emitted30s) return;
  _watchTimer = setTimeout(() => {
    if (_isPlaying && _currentVideo && _isYouTubeVideo()) {
      _emitSignal('video_watch', {
        title: _currentVideo.title,
        url: `https://youtube.com/watch?v=${_currentVideo.videoId}`,
        weight: 1.5,
        source: 'grove_ambient',
        raw_content: _currentVideo.title,
        videoId: _currentVideo.videoId,
        channel: _currentVideo.channel,
        progress: 30,
      });
      _emitted30s = true;
    }
  }, 30000);
}

function _clearWatchTimer() {
  if (_watchTimer) {
    clearTimeout(_watchTimer);
    _watchTimer = null;
  }
}

// ---------------------------------------------------------------------------
// Settings persistence
// ---------------------------------------------------------------------------
function _persistVideo() {
  if (!_isYouTubeVideo(_currentVideo, _currentSourceType) && _currentVideo) return;
  const value = _currentVideo ? JSON.stringify(_currentVideo) : '';
  if (window.appSettings) window.appSettings.ambientVideo = value;
  fetch('/api/config/tools', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ambient_video: value }),
  }).catch(() => {});
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export async function loadVideo(video) {
  if (!video || !video.videoId) return;

  clearTimeout(_errorTitleTimeout);
  clearTimeout(_errorColorTimeout);

  const canReusePlayer = _currentSourceType === 'youtube'
    && !!_player
    && typeof _player.loadVideoById === 'function';
  _resumeYouTubeVideo = null;
  if (!canReusePlayer) {
    _teardownActivePlayer({ pause: true, flush: true });
  }
  _currentSourceType = 'youtube';
  _currentVideo = {
    ...video,
    sourceType: 'youtube',
  };
  _emitted30s = false;
  _reconnectAttempts = 0;

  const color = _keywordColor(video.title);
  _applyColor(color);
  _updateTrackInfo();
  _favIndex = _favorites.findIndex(f => f.videoId === video.videoId);
  _renderChips();
  _setVisualState('playing');

  if (!canReusePlayer || !_player) {
    await _createPlayer(video.videoId);
    const vol = parseInt(_dom.slider?.value ?? 50, 10);
    _player.setVolume(vol);
    _player.playVideo();
  } else {
    _player.loadVideoById(video.videoId);
    const vol = parseInt(_dom.slider?.value ?? 50, 10);
    _player.setVolume(vol);
  }

  _emitSignal('video_open', {
    title: video.title,
    url: `https://youtube.com/watch?v=${video.videoId}`,
    weight: 1.0,
    source: 'grove_ambient',
    videoId: video.videoId,
    channel: video.channel,
  });

  _persistVideo();
  recordLastPlayed({ type: 'ambient', name: video.title || 'ambient video' });
}

export async function loadMediaVideo({
  element,
  video = {},
  api = null,
  onClose = null,
} = {}) {
  if (!element) return;

  clearTimeout(_errorTitleTimeout);
  clearTimeout(_errorColorTimeout);

  let resumeVideo = _resumeYouTubeVideo;
  if (_isYouTubeVideo()) {
    resumeVideo = _currentVideo ? { ..._currentVideo } : resumeVideo;
  }

  _teardownActivePlayer({ pause: true, flush: true });
  _resumeYouTubeVideo = resumeVideo?.videoId ? resumeVideo : null;
  _currentSourceType = 'media_server';
  _currentVideo = {
    sourceType: 'media_server',
    title: video.title || 'Music Video',
    channel: video.channel || '',
    thumbnail: video.thumbnail || '',
    duration: video.duration || '',
    fileId: video.fileId || '',
    entityKind: video.entityKind || 'music_video',
  };
  _sessionCloseHandler = typeof onClose === 'function' ? onClose : null;
  _emitted30s = false;
  _reconnectAttempts = 0;

  const color = _keywordColor(_currentVideo.title);
  _applyColor(color);
  _updateTrackInfo();
  _renderChips();
  _setVisualState('playing');

  _player = _mountHtml5MediaPlayer(element, api);
  const vol = parseInt(_dom.slider?.value ?? 50, 10);
  _player?.setVolume?.(vol);
  _player?.playVideo?.();

  recordLastPlayed({ type: 'ambient', name: _currentVideo.title || 'music video' });
}

export async function dismissCurrent() {
  if (_isLocalMediaVideo()) {
    const restore = _resumeYouTubeVideo?.videoId ? { ..._resumeYouTubeVideo } : null;
    _resumeYouTubeVideo = null;
    _teardownActivePlayer({ pause: true, flush: true });
    _currentSourceType = '';
    _currentVideo = null;
    if (restore?.videoId) {
      _currentSourceType = 'youtube';
      _currentVideo = {
        ...restore,
        sourceType: 'youtube',
      };
      const color = _keywordColor(_currentVideo.title);
      _applyColor(color);
      _updateTrackInfo();
      _favIndex = _favorites.findIndex(f => f.videoId === restore.videoId);
      _renderChips();
      _setVisualState('empty');
      return;
    }
    _setVisualState('empty');
    _updateTrackInfo();
    _renderChips();
    return;
  }
  if (_isPlaying) {
    _player?.pauseVideo?.();
  }
}

export function getState() {
  return {
    currentVideo: _currentVideo,
    sourceType: _currentSourceType,
    isPlaying: _isPlaying,
    favorites: _favorites,
  };
}

// ---------------------------------------------------------------------------
// Detach controller integration
//   Exposed for ui/scripts/grove-orb-detach.js. The orb container may be
//   re-parented into a body-level floating shell; these helpers give the
//   detach controller read access to state (to render the grove slot
//   placeholder title) and write access to YT playback quality (which
//   drops on compact and lifts on focus to save bandwidth).
// ---------------------------------------------------------------------------

/** Return the orb container element — the DOM node the detach controller moves. */
export function getOrbContainer() {
  return _dom.container || document.getElementById('grove-orb-container');
}

/** Return the ambient section (the orb's home slot). */
export function getOrbSection() {
  return _dom.section || document.getElementById('grove-ambient-section');
}

/** Map size mode → YT suggested playback quality. Silent if player isn't ready. */
export function setQualityForSize(size) {
  if (!_isYouTubeVideo() || !_player || typeof _player.setPlaybackQuality !== 'function') return;
  const map = { compact: 'small', standard: 'medium', focus: 'large' };
  const q = map[size];
  if (!q) return;
  try { _player.setPlaybackQuality(q); } catch { /* no-op: cross-origin can throw */ }
}

export async function init() {
  _cacheDom();
  if (!_dom.section) return;

  _initHover();
  _initVolume();
  _initSeek();
  _initChips();
  _initBloom();
  _initFavButton();
  _initLoopButton();
  _setLoopMode(window.appSettings?.ambientLoopMode || 'off', { persist: false });

  // Cross-surface: receive "Send to Ambient" from media cards
  window.addEventListener('media:send-to-ambient', (e) => {
    const video = e.detail;
    if (!video) return;
    const videoData = {
      videoId: video.videoId || video.video_id || '',
      title: video.title || '',
      channel: video.channel || '',
      thumbnail: video.thumbnail || '',
      isLivestream: video.isLivestream || false,
    };
    if (videoData.videoId) {
      loadVideo(videoData);
    }
  });

  await _loadFavorites();
  _renderChips();

  const saved = window.appSettings?.ambientVideo;
  if (saved) {
    try {
      const video = JSON.parse(saved);
      if (video && video.videoId) {
        _currentSourceType = 'youtube';
        _currentVideo = {
          ...video,
          sourceType: 'youtube',
        };
        const color = _keywordColor(video.title);
        _applyColor(color);
        _updateTrackInfo();
        _favIndex = _favorites.findIndex(f => f.videoId === video.videoId);
        _renderChips();
      }
    } catch (e) { console.warn('[Grove Ambient] Failed to parse saved video:', e); }
  }
}
