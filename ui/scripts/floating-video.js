/**
 * FloatingVideo is the app-wide watch surface for video content.
 *
 * It adopts an existing iframe/video element into a persistent player that
 * survives navigation. For HTML5 video rows backed by a media-server file,
 * it also owns progress writes so the Files surface stays coherent while
 * the user works elsewhere in the app.
 */

import { AudioBus } from './audio-bus.js';
import {
  fetchMediaCastLoad,
  fetchMediaOutputs,
  fetchMediaRemoteSession,
  fetchMediaTransportSession,
  playMediaOnRemoteSession,
  playMediaOnTransportReceiver,
  pushMediaProgress,
  sendMediaRemoteSessionGeneral,
  sendMediaRemoteSessionPlaystate,
  sendMediaTransportSessionGeneral,
  sendMediaTransportSessionPlaystate,
} from './files/api.js';
import {
  closeSession,
  getState as getVideoCompanionState,
  inferDeviceKind,
  normalizeShellMode,
  openSession,
  setDeviceKind,
  setLayout,
  setShellMode as setCompanionShellMode,
  updateSession,
} from './video-companion-session.js';
import { openCastPicker } from './cast-picker.js';

let _root = null;
let _iframeSlot = null;
let _audioChip = null;
let _titleEl = null;
let _thumbEl = null;
let _channelEl = null;
let _remotePanelEl = null;
let _remoteThumbEl = null;
let _remoteTitleEl = null;
let _remoteSubtitleEl = null;
let _remoteNoteEl = null;
let _remoteCurrentTimeEl = null;
let _remoteTotalTimeEl = null;
let _remoteSeekEl = null;
let _remoteVolumeEl = null;
let _localProgressEl = null;
let _localCurrentTimeEl = null;
let _localTotalTimeEl = null;
let _localSeekEl = null;
// Transport controls — appear in the local progress bar when the
// custom overlay is the only playback chrome (managed-seek active).
let _localPlayBtn = null;
let _localPlayIconEl = null;
let _localPauseIconEl = null;
let _localSkipBackBtn = null;
let _localSkipForwardBtn = null;
let _localMuteBtn = null;
let _localVolumeOnIconEl = null;
let _localVolumeOffIconEl = null;
let _localVolumeEl = null;
let _localVolumeDragging = false;
let _dragHandleEl = null;
let _detailsBtn = null;
let _playlistBtn = null;
let _nextBtn = null;
let _tracksBtn = null;
let _outputsBtn = null;
let _pipBtn = null;
let _fullscreenBtn = null;
let _popoverEl = null;
let _resizeHandleEl = null;

let _adoptedIframe = null;
let _metadata = {};
let _api = null;
let _mode = 'companion';
let _busHandle = null;
let _duckBaseline = null;
let _buttonsWired = false;

let _fileId = '';
let _nextItem = null;
let _onNext = null;
let _progressState = null;
// Poll handle for api-driven sources (YouTube iframe) where there's no
// HTMLVideoElement to bind ``timeupdate`` against — we read currentTime
// off ``_api`` periodically and re-render the seek bar from that.
let _apiPoll = null;
let _playbackMenu = null;
let _playbackBusy = false;

let _dragStart = null;
let _containerPos = { x: null, y: null };
let _resizeStart = null;
let _pipBoundNode = null;
let _pipHandlers = null;
let _pipActive = false;
let _pipReturnMode = 'companion';
let _ignorePiPLeave = false;
let _popoverMode = '';
let _outputState = {
  loading: false,
  loadedForFileId: '',
  serverId: '',
  provider: '',
  supportsProviderRemote: false,
  remoteSessions: [],
  transportReceivers: [],
  error: '',
};
let _outputLoadSeq = 0;
let _castSdkPromise = null;
let _castConfigured = false;
let _remoteSession = null;
let _remotePollTimer = 0;
let _remotePollInFlight = false;
let _remoteSeekDragging = false;
let _remoteVolumeDragging = false;
let _localSeekDragging = false;
// Post-seek hold: after _commitLocalSeek, the api may take ~200-
// 1500ms before getCurrentTime reports the new position. Until then,
// re-renders would snap the slider back to the OLD position. We pin
// the displayed time to ``_localSeekTargetTimeS`` for up to
// ``_localSeekCooldownUntil`` (or until the player catches up to
// within tolerance, whichever comes first).
let _localSeekTargetTimeS = 0;
let _localSeekCooldownUntil = 0;

// --- Audio sync (A/V offset) state -----------------------------------
// Web Audio routing for compensating Emby/Jellyfin partial-transcode
// lip-sync drift. The graph (when active) is:
//
//   videoEl → MediaElementAudioSourceNode → DelayNode → GainNode → destination
//
// Lazy-init: only set up when the user picks a non-zero offset, so
// users on clean playback paths pay zero overhead. Once initialized,
// the graph stays alive until the player closes — MediaElementAudio-
// SourceNode can only be created once per element, and detaching
// would make the audio go silent without a way to reattach.
//
// Volume + mute control routes through the GainNode once the graph
// is live; otherwise it stays on videoEl.volume directly.
let _audioContext = null;
let _audioSourceNode = null;
let _audioDelayNode = null;
let _audioGainNode = null;
let _audioGraphMediaEl = null;   // the element the graph is attached to
let _audioOffsetMs = 0;          // current offset in ms (0..2000)
// Source signature (e.g. "jellyfin_main|mp4|h264|aac") set on open() —
// when present, offsets are stored/loaded under this key so tuning on
// one file applies to the next file with the same transcode pipeline.
// Empty string falls back to legacy per-file_id storage.
let _syncProfileKey = '';
let _localSyncPopoverEl = null;
let _localSyncRangeEl = null;
let _localSyncValueEl = null;
let _localSyncBtn = null;
let _localSyncDotEl = null;
let _remoteProgressEvent = {
  fileId: '',
  currentTimeS: 0,
  durationS: 0,
};

function _resetOutputState(nextFileId = '') {
  _outputLoadSeq += 1;
  _outputState = {
    loading: false,
    loadedForFileId: nextFileId || '',
    serverId: '',
    provider: '',
    supportsProviderRemote: false,
    remoteSessions: [],
    transportReceivers: [],
    error: '',
  };
}

function _ensureDom() {
  if (_root) return true;
  _root = document.getElementById('floating-video-root');
  if (!_root) {
    console.warn('[floating-video] #floating-video-root not found');
    return false;
  }
  _iframeSlot = _root.querySelector('.fv-iframe-slot');
  _audioChip = _root.querySelector('.fv-audio-chip');
  _titleEl = _root.querySelector('.fv-title');
  _thumbEl = _root.querySelector('.fv-thumb');
  _channelEl = _root.querySelector('.fv-channel');
  _remotePanelEl = _root.querySelector('.fv-remote-panel');
  _remoteThumbEl = _root.querySelector('.fv-remote-thumb');
  _remoteTitleEl = _root.querySelector('.fv-remote-title');
  _remoteSubtitleEl = _root.querySelector('.fv-remote-subtitle');
  _remoteNoteEl = _root.querySelector('.fv-remote-note');
  _remoteCurrentTimeEl = _root.querySelector('.fv-remote-time-cur');
  _remoteTotalTimeEl = _root.querySelector('.fv-remote-time-total');
  _remoteSeekEl = _root.querySelector('.fv-remote-range');
  _remoteVolumeEl = _root.querySelector('.fv-remote-volume-range');
  _localProgressEl = _root.querySelector('.fv-local-progress');
  _localCurrentTimeEl = _root.querySelector('.fv-local-time-cur');
  _localTotalTimeEl = _root.querySelector('.fv-local-time-total');
  _localSeekEl = _root.querySelector('.fv-local-range');
  _localPlayBtn = _root.querySelector('.fv-local-play');
  _localPlayIconEl = _root.querySelector('.fv-local-play-icon');
  _localPauseIconEl = _root.querySelector('.fv-local-pause-icon');
  _localSkipBackBtn = _root.querySelector('.fv-local-skip-back');
  _localSkipForwardBtn = _root.querySelector('.fv-local-skip-forward');
  _localMuteBtn = _root.querySelector('.fv-local-mute');
  _localVolumeOnIconEl = _root.querySelector('.fv-local-volume-on-icon');
  _localVolumeOffIconEl = _root.querySelector('.fv-local-volume-off-icon');
  _localVolumeEl = _root.querySelector('.fv-local-volume-range');
  _localSyncBtn = _root.querySelector('.fv-local-sync-btn');
  _localSyncDotEl = _root.querySelector('.fv-local-sync-dot');
  _localSyncPopoverEl = _root.querySelector('.fv-local-sync-popover');
  _localSyncRangeEl = _root.querySelector('.fv-local-sync-range');
  _localSyncValueEl = _root.querySelector('.fv-local-sync-value');
  _dragHandleEl = _root.querySelector('.fv-drag-handle');
  _detailsBtn = _root.querySelector('[data-fv-action="details"]');
  _playlistBtn = _root.querySelector('[data-fv-action="playlist"]');
  _nextBtn = _root.querySelector('[data-fv-action="next"]');
  _tracksBtn = _root.querySelector('[data-fv-action="tracks"]');
  _outputsBtn = _root.querySelector('[data-fv-action="outputs"]');
  _pipBtn = _root.querySelector('[data-fv-action="pip"]');
  _fullscreenBtn = _root.querySelector('[data-fv-mode="fullscreen"]');
  _popoverEl = _root.querySelector('.fv-popover');
  _resizeHandleEl = _root.querySelector('.fv-resize-handle');

  if (!_buttonsWired) {
    _root.querySelectorAll('[data-fv-mode]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        setMode(btn.dataset.fvMode);
      });
    });
    _detailsBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      _openDetails();
    });
    _playlistBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      let detail = null;
      if (_metadata.videoId) {
        detail = {
          type: 'youtube',
          videoId: _metadata.videoId,
          title: _metadata.title || '',
          channel: _metadata.channel || '',
          thumbnail: _metadata.thumbnail
            || `https://i.ytimg.com/vi/${_metadata.videoId}/mqdefault.jpg`,
        };
      } else if (_fileId) {
        detail = {
          type: 'file',
          fileId: _fileId,
          name: _metadata.title || '',
          kind: 'video',
          thumbnail: _metadata.thumbnail || '',
        };
      }
      if (detail) {
        window.dispatchEvent(new CustomEvent('playlist:add-item', { detail }));
      }
    });
    _nextBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      void _playNext();
    });
    _tracksBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      _togglePlaybackPopover();
    });
    _outputsBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      void _toggleOutputPopover();
    });
    _pipBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      void toggleNativePiP();
    });
    _root.querySelector('.fv-close')?.addEventListener('click', (e) => {
      e.stopPropagation();
      close();
    });
    _audioChip?.addEventListener('click', (e) => {
      if (e.target.closest('.fv-btn') || e.target.closest('.fv-close')) return;
      setMode('companion');
    });
    _remotePanelEl?.addEventListener('click', (e) => {
      const remoteBtn = e.target.closest('[data-fv-remote-action]');
      if (!remoteBtn) return;
      e.stopPropagation();
      void _handleRemoteAction(remoteBtn.dataset.fvRemoteAction || '');
    });
    _remoteSeekEl?.addEventListener('pointerdown', () => {
      _remoteSeekDragging = true;
    });
    _remoteSeekEl?.addEventListener('pointerup', () => {
      _remoteSeekDragging = false;
      void _commitRemoteSeek();
    });
    _remoteSeekEl?.addEventListener('change', () => {
      _remoteSeekDragging = false;
      void _commitRemoteSeek();
    });
    _remoteVolumeEl?.addEventListener('pointerdown', () => {
      _remoteVolumeDragging = true;
    });
    _remoteVolumeEl?.addEventListener('pointerup', () => {
      _remoteVolumeDragging = false;
      void _commitRemoteVolume();
    });
    _remoteVolumeEl?.addEventListener('change', () => {
      _remoteVolumeDragging = false;
      void _commitRemoteVolume();
    });
    _localSeekEl?.addEventListener('pointerdown', () => {
      _localSeekDragging = true;
    });
    _localSeekEl?.addEventListener('input', () => {
      if (!_localSeekDragging || !_localSeekEl || !_localCurrentTimeEl) return;
      const durationS = _currentPlaybackDurationS();
      const frac = Math.max(0, Math.min(1, Number(_localSeekEl.value || 0) / 1000));
      _localCurrentTimeEl.textContent = _fmtRemoteTime(frac * durationS);
    });
    _localSeekEl?.addEventListener('pointerup', () => {
      _localSeekDragging = false;
      void _commitLocalSeek();
    });
    _localSeekEl?.addEventListener('change', () => {
      _localSeekDragging = false;
      void _commitLocalSeek();
    });
    // Transport buttons — play/pause and the two skip-by-10s buttons.
    // All three route through _api when present (preserves managed-
    // seek semantics) and fall back to direct mediaEl manipulation
    // for browser-native playback. Each call ends in
    // _renderLocalPlaybackUi so the play/pause icon flips immediately
    // — the timeupdate event eventually does the same, but waiting
    // ~250ms for the next tick makes the button feel laggy.
    _localPlayBtn?.addEventListener('click', () => {
      _toggleLocalPlayPause();
    });
    _localSkipBackBtn?.addEventListener('click', () => {
      _seekLocalBy(-10);
    });
    _localSkipForwardBtn?.addEventListener('click', () => {
      _seekLocalBy(10);
    });
    _localMuteBtn?.addEventListener('click', () => {
      _toggleLocalMute();
    });
    // Volume slider mirrors the seek-bar pattern: mark dragging so the
    // render loop doesn't fight the user's drag, commit on
    // input/change. mediaEl.volume is 0..1; the input is 0..100.
    _localVolumeEl?.addEventListener('pointerdown', () => {
      _localVolumeDragging = true;
    });
    _localVolumeEl?.addEventListener('input', () => {
      _commitLocalVolume();
    });
    _localVolumeEl?.addEventListener('pointerup', () => {
      _localVolumeDragging = false;
    });
    _localVolumeEl?.addEventListener('change', () => {
      _localVolumeDragging = false;
      _commitLocalVolume();
    });
    // Audio sync — button toggles popover, slider applies offset
    // live (input fires per pointer-step). Reset button clears.
    _localSyncBtn?.addEventListener('click', (e) => {
      e.stopPropagation();
      if (_localSyncPopoverEl?.hidden === false) {
        _closeAudioSyncPopover();
      } else {
        _openAudioSyncPopover();
      }
    });
    _localSyncRangeEl?.addEventListener('input', () => {
      _applyAudioOffsetMs(Number(_localSyncRangeEl.value || 0));
    });
    _localSyncPopoverEl?.addEventListener('click', (e) => {
      // Stop bubble so the outside-click handler doesn't dismiss when
      // the user clicks inside the popover (e.g. on the slider).
      e.stopPropagation();
      const reset = e.target.closest('[data-fv-local-action="reset-sync"]');
      if (reset) _applyAudioOffsetMs(0);
    });
    if (_dragHandleEl) {
      _dragHandleEl.addEventListener('pointerdown', _onDragStart);
    }
    if (_resizeHandleEl) {
      _resizeHandleEl.addEventListener('pointerdown', _onResizeStart);
    }
    document.addEventListener('fullscreenchange', () => {
      if (_mode === 'fullscreen' && !document.fullscreenElement) {
        _applyMode('companion');
      }
    });
    window.addEventListener('resize', _refreshViewport);
    // Keyboard shortcuts — only active when the floating video is the
    // current foreground player and the user isn't typing into an input.
    // Pattern matches what YouTube/Plex/Jellyfin own; native HTML5
    // controls used to capture some of these for free, but we now hide
    // them unconditionally so the custom overlay has to provide parity.
    window.addEventListener('keydown', _onWindowKeyDown);
    document.addEventListener('click', (e) => {
      if (!_root || _root.hidden) return;
      if (!_popoverEl || _popoverEl.hidden) return;
      if (_root.contains(e.target)) return;
      _closePlaybackPopover();
    });
    _popoverEl?.addEventListener('click', (e) => {
      const mediaSourceBtn = e.target.closest('[data-fv-playback-source]');
      if (mediaSourceBtn) {
        e.stopPropagation();
        void _selectPlayback('mediaSource', mediaSourceBtn.dataset.fvPlaybackSource || '');
        return;
      }
      const audioBtn = e.target.closest('[data-fv-playback-audio]');
      if (audioBtn) {
        e.stopPropagation();
        void _selectPlayback('audio', Number(audioBtn.dataset.fvPlaybackAudio));
        return;
      }
      const subtitleBtn = e.target.closest('[data-fv-playback-subtitle]');
      if (subtitleBtn) {
        e.stopPropagation();
        void _selectPlayback('subtitle', Number(subtitleBtn.dataset.fvPlaybackSubtitle));
        return;
      }
      const remoteBtn = e.target.closest('[data-fv-output-remote]');
      if (remoteBtn) {
        e.stopPropagation();
        void _startBrowserRemotePlayback();
        return;
      }
      const castBtn = e.target.closest('[data-fv-output-cast]');
      if (castBtn) {
        e.stopPropagation();
        void _startCastSender();
        return;
      }
      const managedCastBtn = e.target.closest('[data-fv-output-managed-cast]');
      if (managedCastBtn) {
        e.stopPropagation();
        void _openManagedCastPicker();
        return;
      }
      const providerBtn = e.target.closest('[data-fv-output-session]');
      if (providerBtn) {
        e.stopPropagation();
        void _startProviderRemoteSession(
          providerBtn.dataset.fvOutputSession || '',
          providerBtn.dataset.fvOutputLabel || '',
        );
        return;
      }
      const transportBtn = e.target.closest('[data-fv-output-transport]');
      if (transportBtn) {
        e.stopPropagation();
        void _startTransportReceiver({
          transport: transportBtn.dataset.fvOutputTransport || '',
          receiverId: transportBtn.dataset.fvOutputReceiverId || '',
          receiverProfile: transportBtn.dataset.fvOutputReceiverProfile || '',
          label: transportBtn.dataset.fvOutputLabel || '',
        });
      }
    });
    _buttonsWired = true;
  }
  _refreshViewport();
  return true;
}

function open({
  iframe,
  title,
  channel,
  thumbnail,
  videoId,
  api,
  mode,
  fileId,
  nextItem,
  onNext,
  playbackMenu,
  mount,
  audioSyncProfileKey,
}) {
  _syncProfileKey = String(audioSyncProfileKey || '').trim();
  // Callers may either hand us an existing iframe to reparent (HTML5
  // <video> in a Files row) or — for sources where reparenting reloads
  // the iframe and loses state (YouTube) — pass a ``mount(slot)``
  // callback that builds a fresh player inside our slot and returns
  // ``{iframe, api}``. ``mount`` wins if both are present.
  if (!_ensureDom()) return;
  if (typeof mount !== 'function' && !iframe) return;
  if (_remoteSession?.active) {
    _clearRemoteSession({ stopPlayback: false, hideRoot: false });
  }
  const shellMode = normalizeShellMode(mode || 'companion');
  const shouldResumeNativePiP = _pipActive;
  const nextFileId = fileId || '';
  if (_fileId !== nextFileId) {
    _resetOutputState(nextFileId);
  }

  if (_adoptedIframe && _adoptedIframe !== iframe) {
    void _pushProgress({ force: true });
    if (_pipActive) {
      _ignorePiPLeave = true;
      void _exitNativePiP();
      _pipActive = false;
    }
    _teardownPiP();
    _ignorePiPLeave = false;
    _teardownProgress();
    _stopApiPoll();
    _releaseAdopted();
  }

  let resolvedIframe = iframe || null;
  let resolvedApi = api || null;
  if (typeof mount === 'function') {
    // Mount owns slot population (typically by constructing a new
    // player whose iframe replaces a target div). It returns the
    // iframe element + a full api so we can drive controls without
    // touching the player directly.
    try {
      _iframeSlot.innerHTML = '';
      const mounted = mount(_iframeSlot);
      resolvedIframe = mounted?.iframe || _iframeSlot.querySelector('iframe') || null;
      resolvedApi = mounted?.api || resolvedApi;
    } catch (err) {
      console.warn('[fv] mount callback failed', err);
      return;
    }
  }

  _adoptedIframe = resolvedIframe;
  _metadata = {
    title: title || '',
    channel: channel || '',
    thumbnail: thumbnail || '',
    videoId: videoId || '',
  };
  _fileId = nextFileId;
  _nextItem = nextItem || null;
  _onNext = typeof onNext === 'function' ? onNext : null;
  _playbackMenu = playbackMenu || null;
  _playbackBusy = false;
  _api = resolvedApi || _elementApi(resolvedIframe);

  _syncUi();

  // When mount was used, the iframe is already a child of _iframeSlot
  // (the player constructor put it there). When we received an existing
  // iframe, reparent it now. appendChild on an already-child element
  // is a no-op move to the same position, so this is safe either way.
  if (resolvedIframe) {
    _iframeSlot.appendChild(resolvedIframe);
    resolvedIframe.classList.add('fv-adopted');
  }

  _bindProgress(resolvedIframe);
  _bindPiP(resolvedIframe);
  // api-driven sources (YouTube) get a polling loop in place of the
  // ``timeupdate`` events that _bindProgress wires for HTMLVideoElement.
  if (!_progressState) _startApiPoll();

  _root.hidden = false;
  _root.classList.add('open');

  _registerBus();
  _busHandle?.claim();

  openSession({
    shellMode: shouldResumeNativePiP ? 'native_pip' : shellMode,
    supportsNativePiP: _supportsNativePiP(),
    isNativePiPActive: shouldResumeNativePiP,
    fileId: _fileId || null,
    videoId: _metadata.videoId || null,
    title: _metadata.title || '',
    channel: _metadata.channel || '',
    thumbnail: _metadata.thumbnail || '',
    nextItem: _nextItem,
    hasPlaybackOptions: _hasPlaybackOptions(),
    remoteSessionActive: false,
    remoteSourceType: '',
    remoteTransportKind: '',
    remoteProvider: '',
    remoteServerId: '',
    remoteSessionId: '',
    remoteDeviceName: '',
    remoteSupportedCommands: [],
    isMuted: false,
    volumeLevel: null,
    canSeek: false,
  });
  if (shouldResumeNativePiP && _supportsNativePiP()) {
    void toggleNativePiP({ force: true });
  } else {
    setMode(shellMode);
  }
  _emitState();
}

function setMode(next) {
  if (!_root) return;
  _applyMode(normalizeShellMode(next));
}

function close() {
  if (!_root) return;
  const shouldStopRemote = !!_remoteSession?.active
    && _remoteSession?.stopOnClose !== false
    && _remoteSupports('Stop');
  void _pushProgress({ force: true });
  void _pushRemoteTransportProgress(_remoteSession, { force: true });
  if (shouldStopRemote) {
    void _sendRemotePlaystate('Stop');
  }
  if (_pipActive) {
    _ignorePiPLeave = true;
    void _exitNativePiP();
  }
  _teardownPiP();
  _teardownProgress();
  _stopApiPoll();
  _closeAudioContext();
  _stopRemotePolling();
  _busHandle?.release();
  _releaseAdopted();
  _root.hidden = true;
  _root.classList.remove('open');
  _root.dataset.mode = '';
  _root.dataset.device = inferDeviceKind();
  _root.dataset.remoteActive = '0';
  _root.dataset.remotePaused = '0';
  _mode = 'companion';
  _pipActive = false;
  _pipReturnMode = 'companion';
  _ignorePiPLeave = false;
  _duckBaseline = null;
  _containerPos = { x: null, y: null };
  _resizeStart = null;
  _root.style.top = '';
  _root.style.left = '';
  _root.style.right = '';
  _root.style.bottom = '';
  _root.style.width = '';
  _root.style.height = '';
  _metadata = {};
  _api = null;
  _fileId = '';
  _nextItem = null;
  _onNext = null;
  _playbackMenu = null;
  _playbackBusy = false;
  _remoteSession = null;
  _remoteSeekDragging = false;
  _remoteVolumeDragging = false;
  _localSeekDragging = false;
  _remoteProgressEvent = {
    fileId: '',
    currentTimeS: 0,
    durationS: 0,
  };
  _resetOutputState();
  _closePlaybackPopover();
  _syncUi();
  closeSession();
  _emitState();
  window.dispatchEvent(new CustomEvent('floating-video:closed'));
}

function isOpen() {
  return !!_root && !_root.hidden && (!!_adoptedIframe || !!_remoteSession?.active);
}

function getVideoId() {
  return _metadata.videoId || null;
}

function _videoNodeForNativePiP(node = _adoptedIframe) {
  if (typeof HTMLVideoElement === 'undefined' || !(node instanceof HTMLVideoElement)) return null;
  return node;
}

function _supportsNativePiP(node = _adoptedIframe) {
  const video = _videoNodeForNativePiP(node);
  return !!(
    video
    && typeof document !== 'undefined'
    && document.pictureInPictureEnabled
    && typeof video.requestPictureInPicture === 'function'
    && !video.disablePictureInPicture
  );
}

function _syncPiPUi() {
  if (!_pipBtn) return;
  const supported = !_remoteSession?.active && _supportsNativePiP();
  _pipBtn.hidden = !supported;
  _pipBtn.classList.toggle('is-active', _pipActive);
  const label = _pipActive ? 'Return from picture-in-picture' : 'Pop out to picture-in-picture';
  _pipBtn.title = label;
  _pipBtn.setAttribute('aria-label', label);
}

function _syncUi() {
  const remoteActive = !!_remoteSession?.active;
  if (_titleEl) _titleEl.textContent = _metadata.title || 'Video';
  if (_channelEl) {
    const parts = [];
    if (_pipActive) parts.push('Picture-in-picture');
    if (_metadata.channel) parts.push(_metadata.channel);
    _channelEl.textContent = parts.join(' | ');
  }
  if (_thumbEl) {
    if (_metadata.thumbnail) {
      _thumbEl.src = _metadata.thumbnail;
      _thumbEl.hidden = false;
    } else {
      _thumbEl.hidden = true;
      _thumbEl.removeAttribute('src');
    }
  }
  if (_detailsBtn) _detailsBtn.hidden = !_fileId;
  if (_playlistBtn) _playlistBtn.hidden = remoteActive || !(_metadata.videoId || _fileId);
  if (_nextBtn) _nextBtn.hidden = remoteActive || !_nextItem || !_onNext;
  if (_tracksBtn) _tracksBtn.hidden = remoteActive || !_hasPlaybackOptions();
  if (_outputsBtn) _outputsBtn.hidden = remoteActive || !_hasOutputCandidates();
  if (_fullscreenBtn) _fullscreenBtn.hidden = remoteActive || !_adoptedIframe;
  _tracksBtn?.classList.toggle('is-active', _popoverMode === 'playback' && !_popoverEl?.hidden);
  _outputsBtn?.classList.toggle('is-active', _popoverMode === 'outputs' && !_popoverEl?.hidden);
  _syncPiPUi();
  if (_root) {
    _root.dataset.device = inferDeviceKind();
    _root.dataset.remoteActive = remoteActive ? '1' : '0';
    _root.dataset.remotePaused = _remoteSession?.isPaused ? '1' : '0';
  }
  if (_popoverEl && _popoverMode === 'playback' && (!_playbackMenu || !_hasPlaybackOptions())) {
    _closePlaybackPopover();
  } else if (_popoverEl && !_popoverEl.hidden && _popoverMode === 'playback') {
    _renderPlaybackPopover();
  } else if (_popoverEl && !_popoverEl.hidden && _popoverMode === 'outputs') {
    _renderPreferredOutputPopover();
  }
  _renderRemoteUi();
  _renderLocalPlaybackUi();
  updateSession({
    title: _metadata.title || '',
    channel: _metadata.channel || '',
    thumbnail: _metadata.thumbnail || '',
    fileId: _fileId || null,
    videoId: _metadata.videoId || null,
    nextItem: _nextItem,
    hasPlaybackOptions: _hasPlaybackOptions(),
    supportsNativePiP: !remoteActive && _supportsNativePiP(),
    isNativePiPActive: _pipActive,
    remoteSessionActive: remoteActive,
    remoteSourceType: _remoteSourceType(_remoteSession),
    remoteTransportKind: _remoteTransportKind(_remoteSession),
    remoteProvider: _remoteSession?.provider || '',
    remoteServerId: _remoteSession?.serverId || '',
    remoteSessionId: _remoteSession?.sessionId || '',
    remoteDeviceName: _remoteSession?.deviceName || '',
    remoteSupportedCommands: Array.isArray(_remoteSession?.supportedCommands)
      ? _remoteSession.supportedCommands
      : [],
    isMuted: !!_remoteSession?.isMuted,
    volumeLevel: _remoteSession?.volumeLevel ?? null,
    canSeek: !!_remoteSession?.canSeek,
  });
}

function _fmtRemoteTime(totalSeconds) {
  const secs = Math.max(0, Math.floor(Number(totalSeconds || 0)));
  const hours = Math.floor(secs / 3600);
  const minutes = Math.floor((secs % 3600) / 60);
  const seconds = secs % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

function _currentPlaybackDurationS() {
  if (_api && typeof _api.getDuration === 'function') {
    return Math.max(0, Number(_api.getDuration() || 0));
  }
  const mediaEl = _progressState?.mediaEl;
  if (!mediaEl) return 0;
  return Number.isFinite(mediaEl.duration) ? Math.max(0, Number(mediaEl.duration || 0)) : 0;
}

function _currentPlaybackTimeS() {
  if (_api && typeof _api.getCurrentTime === 'function') {
    return Math.max(0, Number(_api.getCurrentTime() || 0));
  }
  const mediaEl = _progressState?.mediaEl;
  if (!mediaEl) return 0;
  return Math.max(0, Number(mediaEl.currentTime || 0));
}

function _currentPlaybackCanSeek() {
  if (_api && typeof _api.canSeek === 'function') {
    return !!_api.canSeek();
  }
  return _currentPlaybackDurationS() > 0;
}

function _usesManagedSeekUi() {
  return !!(_api && typeof _api.usesManagedSeek === 'function' && _api.usesManagedSeek());
}

function _renderLocalPlaybackUi() {
  if (!_localProgressEl || _remoteSession?.active) {
    if (_root) _root.dataset.localProgress = '0';
    if (_localProgressEl) _localProgressEl.hidden = true;
    return;
  }
  // Show the local progress bar for any source we can read time off —
  // either a real ``_fileId`` (HTMLVideoElement) or an api-driven
  // source that exposes ``getCurrentTime`` (YouTube / future iframe
  // players).
  const hasSeekableSource = !!(
    _fileId || (_api && typeof _api.getCurrentTime === 'function')
  );
  const shouldShow = !!(
    hasSeekableSource
    && _mode === 'companion'
    && _currentPlaybackCanSeek()
    && _currentPlaybackDurationS() > 0
  );
  if (_root) _root.dataset.localProgress = shouldShow ? '1' : '0';
  _localProgressEl.hidden = !shouldShow;
  if (!shouldShow) return;

  const durationS = _currentPlaybackDurationS();
  const currentTimeS = _currentPlaybackTimeS();
  // Post-seek hold: while the api catches up to a just-issued seek,
  // pin the displayed time to the seek target so the slider doesn't
  // snap back to the stale ``getCurrentTime``. Clear the hold once
  // the player is within tolerance OR the cooldown expires.
  let displayTimeS = currentTimeS;
  if (_localSeekCooldownUntil > 0) {
    const now = Date.now();
    const drift = Math.abs(currentTimeS - _localSeekTargetTimeS);
    if (now < _localSeekCooldownUntil && drift > 1.5) {
      displayTimeS = _localSeekTargetTimeS;
    } else {
      _localSeekCooldownUntil = 0;
    }
  }
  if (_localCurrentTimeEl && !_localSeekDragging) {
    _localCurrentTimeEl.textContent = _fmtRemoteTime(displayTimeS);
  }
  if (_localTotalTimeEl) {
    _localTotalTimeEl.textContent = _fmtRemoteTime(durationS);
  }
  if (_localSeekEl && !_localSeekDragging) {
    const frac = durationS > 0 ? Math.max(0, Math.min(1, displayTimeS / durationS)) : 0;
    _localSeekEl.value = String(Math.round(frac * 1000));
    _localSeekEl.disabled = !_currentPlaybackCanSeek();
    _localSeekEl.setAttribute(
      'aria-label',
      _usesManagedSeekUi() ? 'Seek video timeline' : 'Seek playback',
    );
  }

  // Transport buttons — read play state + volume directly from the
  // media element. Reading from the DOM each render keeps us honest
  // even when external code (autoplay, picture-in-picture, OS media
  // session) flips state without going through our handlers.
  const mediaEl = _progressState?.mediaEl;
  if (mediaEl) {
    const isPaused = !!mediaEl.paused;
    if (_localPlayIconEl)  _localPlayIconEl.hidden  = !isPaused;
    if (_localPauseIconEl) _localPauseIconEl.hidden =  isPaused;
    if (_localPlayBtn) {
      _localPlayBtn.setAttribute('aria-label', isPaused ? 'Play' : 'Pause');
      _localPlayBtn.title = isPaused ? 'Play' : 'Pause';
    }

    const isMuted = !!mediaEl.muted || mediaEl.volume === 0;
    if (_localVolumeOnIconEl)  _localVolumeOnIconEl.hidden  =  isMuted;
    if (_localVolumeOffIconEl) _localVolumeOffIconEl.hidden = !isMuted;
    if (_localMuteBtn) {
      _localMuteBtn.setAttribute('aria-label', isMuted ? 'Unmute' : 'Mute');
      _localMuteBtn.title = isMuted ? 'Unmute' : 'Mute';
    }
    if (_localVolumeEl && !_localVolumeDragging) {
      // mediaEl.volume is 0..1; slider is 0..100. Mute pulls slider
      // visibly to 0 so the user can see "muted = no volume."
      const v = isMuted ? 0 : Math.round(Math.max(0, Math.min(1, mediaEl.volume || 0)) * 100);
      _localVolumeEl.value = String(v);
    }
  } else if (_api && typeof _api.getCurrentTime === 'function') {
    // api-driven path (YouTube iframe). Mirror the mediaEl rendering
    // by reading state off ``_api`` so icons + slider stay truthful
    // even when YouTube's own controls (which still exist behind our
    // overlay) flip play-state out from under us.
    const isPlayingApi = typeof _api.isPlaying === 'function' ? !!_api.isPlaying() : false;
    const isPausedApi  = !isPlayingApi;
    if (_localPlayIconEl)  _localPlayIconEl.hidden  = !isPausedApi;
    if (_localPauseIconEl) _localPauseIconEl.hidden =  isPausedApi;
    if (_localPlayBtn) {
      _localPlayBtn.setAttribute('aria-label', isPausedApi ? 'Play' : 'Pause');
      _localPlayBtn.title = isPausedApi ? 'Play' : 'Pause';
    }

    const apiMuted = typeof _api.isMuted === 'function' ? !!_api.isMuted() : false;
    const apiVolume = typeof _api.getVolume === 'function' ? Number(_api.getVolume() || 0) : 1;
    const isMutedApi = apiMuted || apiVolume === 0;
    if (_localVolumeOnIconEl)  _localVolumeOnIconEl.hidden  =  isMutedApi;
    if (_localVolumeOffIconEl) _localVolumeOffIconEl.hidden = !isMutedApi;
    if (_localMuteBtn) {
      _localMuteBtn.setAttribute('aria-label', isMutedApi ? 'Unmute' : 'Mute');
      _localMuteBtn.title = isMutedApi ? 'Unmute' : 'Mute';
    }
    if (_localVolumeEl && !_localVolumeDragging) {
      const v = isMutedApi ? 0 : Math.round(Math.max(0, Math.min(1, apiVolume)) * 100);
      _localVolumeEl.value = String(v);
    }
  }

  // Skip buttons inherit seek capability from the seek bar — when
  // managed-seek is active and the duration is unknown, neither the
  // bar nor the skip buttons should fire.
  const seekable = _currentPlaybackCanSeek();
  if (_localSkipBackBtn)    _localSkipBackBtn.disabled    = !seekable;
  if (_localSkipForwardBtn) _localSkipForwardBtn.disabled = !seekable;
}

function _renderRemoteUi() {
  const remote = _remoteSession;
  if (!_remotePanelEl) return;
  const active = !!remote?.active;
  _remotePanelEl.hidden = !active;
  if (!active) return;
  const setDisabled = (selector, disabled) => {
    const el = _remotePanelEl.querySelector(selector);
    if (el) el.disabled = !!disabled;
  };

  if (_remoteThumbEl) {
    if (remote.thumbnail) {
      _remoteThumbEl.src = remote.thumbnail;
      _remoteThumbEl.hidden = false;
    } else {
      _remoteThumbEl.hidden = true;
      _remoteThumbEl.removeAttribute('src');
    }
  }
  if (_remoteTitleEl) {
    _remoteTitleEl.textContent = remote.nowPlayingTitle || _metadata.title || 'Remote playback';
  }
  if (_remoteSubtitleEl) {
    const transportKind = _remoteTransportKind(remote);
    const provider = remote.provider
      ? `${remote.provider[0].toUpperCase()}${remote.provider.slice(1)}`
      : 'Server';
    const subtitleSource = transportKind ? transportKind.toUpperCase() : provider;
    _remoteSubtitleEl.textContent = `${subtitleSource} | ${remote.deviceName || 'Remote device'}`;
    _remoteSubtitleEl.textContent = `${provider} • ${remote.deviceName || 'Remote device'}`;
  }
  if (_remoteNoteEl) {
    const note = _remoteCapabilityNote(remote);
    _remoteNoteEl.textContent = note;
    _remoteNoteEl.hidden = !note;
  }
  if (_remoteCurrentTimeEl) _remoteCurrentTimeEl.textContent = _fmtRemoteTime(remote.currentTimeS);
  if (_remoteTotalTimeEl) _remoteTotalTimeEl.textContent = _fmtRemoteTime(remote.durationS);
  if (_remoteSeekEl && !_remoteSeekDragging) {
    const frac = remote.durationS > 0 ? Math.max(0, Math.min(1, remote.currentTimeS / remote.durationS)) : 0;
    _remoteSeekEl.value = String(Math.round(frac * 1000));
    _remoteSeekEl.disabled = !remote.canSeek || remote.durationS <= 0;
  }
  if (_remoteVolumeEl && !_remoteVolumeDragging) {
    _remoteVolumeEl.value = String(
      Number.isFinite(Number(remote.volumeLevel)) ? Number(remote.volumeLevel) : 50,
    );
    _remoteVolumeEl.disabled = !remote.supportedCommands.includes('SetVolume');
  }
  setDisabled('[data-fv-remote-action="previous"]', !_remoteSupports('PreviousTrack'));
  setDisabled('[data-fv-remote-action="seek-back"]', !remote.canSeek);
  setDisabled('[data-fv-remote-action="toggle-play"]', !(
    _remoteSupports('PlayPause') || _remoteSupports('Pause') || _remoteSupports('Unpause')
  ));
  setDisabled('[data-fv-remote-action="seek-forward"]', !remote.canSeek);
  setDisabled('[data-fv-remote-action="next"]', !_remoteSupports('NextTrack'));
  setDisabled('[data-fv-remote-action="stop"]', !_remoteSupports('Stop'));
  setDisabled('[data-fv-remote-action="toggle-mute"]', !(
    _remoteSupports('ToggleMute') || _remoteSupports('Mute') || _remoteSupports('Unmute')
  ));
  setDisabled('[data-fv-remote-action="volume-down"]', !(
    _remoteSupports('VolumeDown') || _remoteSupports('SetVolume')
  ));
  setDisabled('[data-fv-remote-action="volume-up"]', !(
    _remoteSupports('VolumeUp') || _remoteSupports('SetVolume')
  ));
  _remotePanelEl.querySelector('[data-fv-remote-action="previous"]')?.toggleAttribute('hidden', !_remoteSupports('PreviousTrack'));
  _remotePanelEl.querySelector('[data-fv-remote-action="seek-back"]')?.toggleAttribute('hidden', !remote.canSeek);
  _remotePanelEl.querySelector('[data-fv-remote-action="toggle-play"]')?.toggleAttribute(
    'hidden',
    !(_remoteSupports('PlayPause') || _remoteSupports('Pause') || _remoteSupports('Unpause')),
  );
  _remotePanelEl.querySelector('[data-fv-remote-action="seek-forward"]')?.toggleAttribute('hidden', !remote.canSeek);
  _remotePanelEl.querySelector('[data-fv-remote-action="next"]')?.toggleAttribute('hidden', !_remoteSupports('NextTrack'));
  _remotePanelEl.querySelector('[data-fv-remote-action="stop"]')?.toggleAttribute('hidden', !_remoteSupports('Stop'));
}

function _applyMode(next) {
  if (_remoteSession?.active && next === 'fullscreen') {
    next = 'companion';
  }
  const prev = _mode;
  if (next === 'native_pip') {
    if (_remoteSession?.active) return;
    void toggleNativePiP({ force: true });
    return;
  }
  if (_pipActive) {
    _pipReturnMode = next;
    void _exitNativePiP();
    return;
  }
  _mode = next;
  _root.dataset.mode = next;
  setCompanionShellMode(next);
  _syncPiPUi();
  _applyShellLayout();
  _renderLocalPlaybackUi();
  _syncNativeVideoControls();

  if (next === 'fullscreen') {
    if (!document.fullscreenElement) {
      const target = _iframeSlot;
      if (target?.requestFullscreen) {
        target.requestFullscreen().catch(() => _applyMode('companion'));
      }
    }
  } else if (prev === 'fullscreen' && document.fullscreenElement) {
    if (typeof document.exitFullscreen === 'function') {
      document.exitFullscreen().catch(() => {});
    }
  }

  window.dispatchEvent(new CustomEvent('floating-video:mode-change', {
    detail: {
      mode: next,
      shellMode: next,
      videoId: _metadata.videoId,
      fileId: _fileId || null,
      deviceKind: inferDeviceKind(),
    },
  }));
  _closePlaybackPopover();
  _emitState();
}

function _bindPiP(node) {
  _teardownPiP();
  const video = _videoNodeForNativePiP(node);
  if (!video) {
    _syncPiPUi();
    return;
  }
  _pipBoundNode = video;
  _pipHandlers = {
    onEnter: () => {
      _pipActive = true;
      _mode = 'native_pip';
      _root.dataset.mode = 'native_pip';
      setCompanionShellMode('native_pip');
      _syncPiPUi();
      _applyShellLayout();
      _closePlaybackPopover();
      _emitState();
      window.dispatchEvent(new CustomEvent('floating-video:pip-change', {
        detail: {
          active: true,
          fileId: _fileId || null,
          videoId: _metadata.videoId || null,
        },
      }));
      window.dispatchEvent(new CustomEvent('floating-video:mode-change', {
        detail: {
          mode: 'native_pip',
          shellMode: 'native_pip',
          videoId: _metadata.videoId,
          fileId: _fileId || null,
          deviceKind: inferDeviceKind(),
        },
      }));
    },
    onLeave: () => {
      const restoreMode = _ignorePiPLeave ? _mode : (_pipReturnMode || 'companion');
      _pipActive = false;
      _ignorePiPLeave = false;
      _syncPiPUi();
      window.dispatchEvent(new CustomEvent('floating-video:pip-change', {
        detail: {
          active: false,
          fileId: _fileId || null,
          videoId: _metadata.videoId || null,
        },
      }));
      if (!isOpen()) {
        _emitState();
        return;
      }
      _applyMode(restoreMode === 'native_pip' ? 'companion' : restoreMode);
    },
  };
  video.addEventListener('enterpictureinpicture', _pipHandlers.onEnter);
  video.addEventListener('leavepictureinpicture', _pipHandlers.onLeave);
  _syncPiPUi();
}

function _teardownPiP() {
  if (_pipBoundNode && _pipHandlers) {
    _pipBoundNode.removeEventListener('enterpictureinpicture', _pipHandlers.onEnter);
    _pipBoundNode.removeEventListener('leavepictureinpicture', _pipHandlers.onLeave);
  }
  _pipBoundNode = null;
  _pipHandlers = null;
}

async function _exitNativePiP() {
  if (typeof document === 'undefined' || !document.pictureInPictureElement || typeof document.exitPictureInPicture !== 'function') {
    _pipActive = false;
    _syncPiPUi();
    return false;
  }
  try {
    await document.exitPictureInPicture();
    return true;
  } catch (err) {
    console.warn('[floating-video] failed to exit picture-in-picture:', err);
    return false;
  }
}

async function toggleNativePiP({ force = false } = {}) {
  if (_pipActive) {
    _pipReturnMode = 'companion';
    await _exitNativePiP();
    return;
  }
  if (!force && !_supportsNativePiP()) return;
  const video = _videoNodeForNativePiP();
  if (!_supportsNativePiP(video)) return;
  _pipReturnMode = _mode === 'native_pip' ? 'companion' : _mode;
  if (document.fullscreenElement) {
    if (typeof document.exitFullscreen === 'function') {
      await document.exitFullscreen().catch(() => {});
    }
  }
  try {
    await video.requestPictureInPicture();
  } catch (err) {
    console.warn('[floating-video] failed to enter picture-in-picture:', err);
    _pipActive = false;
    _syncPiPUi();
    if (_mode === 'native_pip') {
      _mode = 'companion';
      _root.dataset.mode = 'companion';
      setCompanionShellMode('companion');
      _applyShellLayout();
      _emitState();
    }
  }
}

function _currentDeviceKind() {
  return inferDeviceKind();
}

function _applyShellLayout() {
  if (!_root) return;
  const deviceKind = _currentDeviceKind();
  _root.dataset.device = deviceKind;
  setDeviceKind(deviceKind);
  if (deviceKind === 'mobile' || _mode === 'fullscreen') {
    _root.style.left = '';
    _root.style.top = '';
    _root.style.right = '';
    _root.style.bottom = '';
    _root.style.width = '';
    _root.style.height = '';
    _containerPos = { x: null, y: null };
    return;
  }

  const stored = getVideoCompanionState().layout || {};
  if (_mode === 'companion') {
    const width = Math.max(280, Math.min(window.innerWidth - 24, Number(stored.width) || 360));
    const height = Math.max(180, Math.min(window.innerHeight - 24, Number(stored.height) || Math.round(width * 9 / 16)));
    _root.style.width = `${width}px`;
    _root.style.height = `${height}px`;
  } else {
    _root.style.width = '';
    _root.style.height = '';
  }

  if (Number.isFinite(stored.x) && Number.isFinite(stored.y)) {
    const width = _root.offsetWidth || Number(stored.width) || 320;
    const height = _root.offsetHeight || Number(stored.height) || 180;
    const x = Math.max(4, Math.min(window.innerWidth - width - 4, Number(stored.x)));
    const y = Math.max(4, Math.min(window.innerHeight - height - 4, Number(stored.y)));
    _containerPos = { x, y };
    _root.style.left = `${x}px`;
    _root.style.top = `${y}px`;
    _root.style.right = 'auto';
    _root.style.bottom = 'auto';
  }
}

let _viewportRefreshPending = false;
function _refreshViewport() {
  if (!_root) return;
  if (_viewportRefreshPending) return;
  _viewportRefreshPending = true;
  requestAnimationFrame(() => {
    _viewportRefreshPending = false;
    if (!_root) return;
    _applyShellLayout();
    _emitState();
  });
}

function _onDragStart(e) {
  if (_mode !== 'companion' && _mode !== 'collapsed' && _mode !== 'native_pip') return;
  if (_currentDeviceKind() === 'mobile') return;
  e.preventDefault();
  const rect = _root.getBoundingClientRect();
  _dragStart = {
    pointerX: e.clientX,
    pointerY: e.clientY,
    containerX: rect.left,
    containerY: rect.top,
  };
  _dragHandleEl.setPointerCapture(e.pointerId);
  _dragHandleEl.addEventListener('pointermove', _onDragMove);
  _dragHandleEl.addEventListener('pointerup', _onDragEnd);
  _dragHandleEl.addEventListener('pointercancel', _onDragEnd);
}

function _onDragMove(e) {
  if (!_dragStart) return;
  const dx = e.clientX - _dragStart.pointerX;
  const dy = e.clientY - _dragStart.pointerY;
  let x = _dragStart.containerX + dx;
  let y = _dragStart.containerY + dy;
  const width = _root.offsetWidth;
  const height = _root.offsetHeight;
  x = Math.max(4, Math.min(window.innerWidth - width - 4, x));
  y = Math.max(4, Math.min(window.innerHeight - height - 4, y));
  _containerPos = { x, y };
  _root.style.left = `${x}px`;
  _root.style.top = `${y}px`;
  _root.style.right = 'auto';
  _root.style.bottom = 'auto';
}

function _onDragEnd(e) {
  if (!_dragStart) return;
  try {
    _dragHandleEl.releasePointerCapture(e.pointerId);
  } catch {
    // Ignore pointer-capture release failures.
  }
  _dragHandleEl.removeEventListener('pointermove', _onDragMove);
  _dragHandleEl.removeEventListener('pointerup', _onDragEnd);
  _dragHandleEl.removeEventListener('pointercancel', _onDragEnd);
  setLayout({ x: _containerPos.x, y: _containerPos.y });
  _dragStart = null;
}

function _onResizeStart(e) {
  if (_mode !== 'companion') return;
  if (_currentDeviceKind() === 'mobile') return;
  e.preventDefault();
  e.stopPropagation();
  const rect = _root.getBoundingClientRect();
  _resizeStart = {
    pointerX: e.clientX,
    pointerY: e.clientY,
    width: rect.width,
    height: rect.height,
  };
  _resizeHandleEl.setPointerCapture(e.pointerId);
  _resizeHandleEl.addEventListener('pointermove', _onResizeMove);
  _resizeHandleEl.addEventListener('pointerup', _onResizeEnd);
  _resizeHandleEl.addEventListener('pointercancel', _onResizeEnd);
}

function _onResizeMove(e) {
  if (!_resizeStart || !_root) return;
  const dx = e.clientX - _resizeStart.pointerX;
  const dy = e.clientY - _resizeStart.pointerY;
  const nextWidth = Math.max(280, Math.min(window.innerWidth - 24, Math.round(_resizeStart.width + dx)));
  const nextHeight = Math.max(180, Math.min(window.innerHeight - 24, Math.round(_resizeStart.height + dy)));
  _root.style.width = `${nextWidth}px`;
  _root.style.height = `${nextHeight}px`;
}

function _onResizeEnd(e) {
  if (!_resizeStart) return;
  try {
    _resizeHandleEl.releasePointerCapture(e.pointerId);
  } catch {
    // Ignore pointer-capture release failures.
  }
  _resizeHandleEl.removeEventListener('pointermove', _onResizeMove);
  _resizeHandleEl.removeEventListener('pointerup', _onResizeEnd);
  _resizeHandleEl.removeEventListener('pointercancel', _onResizeEnd);
  setLayout({
    width: _root.offsetWidth,
    height: _root.offsetHeight,
  });
  _resizeStart = null;
}

function _registerBus() {
  if (_busHandle) return;
  _busHandle = AudioBus.register({
    id: 'floating-video',
    tier: 'media',
    // Video content from Emby / Jellyfin / generic sources is usually
    // dialogue-heavy or mixed. 'mixed' as the safe default — widget
    // plays listening pose rather than dancing. A future enhancement
    // could read metadata (videoType=music_video, etc.) to override.
    kind: 'mixed',
    duck: (level) => {
      if (!_api || typeof _api.getVolume !== 'function' || typeof _api.setVolume !== 'function') return;
      if (_duckBaseline !== null) return;
      try {
        _duckBaseline = _api.getVolume();
        _api.setVolume(_duckBaseline * level);
      } catch {
        _duckBaseline = null;
      }
    },
    unduck: () => {
      if (!_api || typeof _api.setVolume !== 'function' || _duckBaseline === null) return;
      try {
        _api.setVolume(_duckBaseline);
      } catch {
        // Ignore restore failures.
      }
      _duckBaseline = null;
    },
  });
}

function _bindProgress(node) {
  _teardownProgress();
  if (!_fileId) return;
  if (typeof HTMLVideoElement === 'undefined' || !(node instanceof HTMLVideoElement)) return;

  const progress = {
    mediaEl: node,
    lastSentAt: 0,
    onTimeUpdate: () => {
      _renderLocalPlaybackUi();
      _emitState();
      void _pushProgress();
    },
    onPause: () => {
      void _pushProgress({ force: true });
      _renderLocalPlaybackUi();
      _emitState();
    },
    onPlay: () => {
      _renderLocalPlaybackUi();
      _emitState();
    },
    onEnded: () => {
      void _pushProgress({ force: true, isFinished: true });
      _renderLocalPlaybackUi();
      _emitState();
    },
    onPageHide: () => { void _pushProgress({ force: true }); },
    onVisibilityChange: () => {
      if (document.hidden) void _pushProgress({ force: true });
    },
  };

  node.addEventListener('timeupdate', progress.onTimeUpdate);
  node.addEventListener('pause', progress.onPause);
  node.addEventListener('play', progress.onPlay);
  node.addEventListener('ended', progress.onEnded);
  // Volume + mute changes can come from outside our buttons (OS
  // media keys, system tray volume, browser tab mute). Mirror them
  // back into the custom UI so the slider + icon don't lie about
  // current state.
  node.addEventListener('volumechange', () => _renderLocalPlaybackUi());
  window.addEventListener('pagehide', progress.onPageHide);
  document.addEventListener('visibilitychange', progress.onVisibilityChange);

  _progressState = progress;

  // Restore any persisted A/V offset for this file — but DEFER it
  // until ``loadeddata`` fires. Chrome (and other browsers) have a
  // race where calling createMediaElementSource() on a <video>
  // that's mid-Range-request can crash the GPU process, which
  // renders as colored static and tanks the whole tab. Waiting for
  // the first frame to decode before touching the audio graph
  // sidesteps the race entirely. This was observed firing
  // synchronously in _attachProgressNode (called as soon as the
  // <video> element mounts, before any frame has loaded), and
  // triggered the static + tab-crash incident on 2026-05-02.
  //
  // For files with no persisted offset (the common case), we don't
  // touch the audio graph at all — the slider stays at 0 and the
  // native audio path runs untouched.
  _audioOffsetMs = 0;
  _renderAudioSyncUi();
  const savedMs = _readPersistedOffsetMs(_fileId);
  if (savedMs > 0) {
    // Capture the element via the function parameter (`node`) — earlier
    // version referenced bare `mediaEl` which is undefined here, since
    // the local var lives inside `progress.mediaEl`. That ReferenceError
    // prevented the player from mounting at all.
    const onReady = () => {
      // Re-check: by the time loadeddata fires, the user might have
      // navigated to a different file. Verify _fileId still matches
      // before applying.
      const currentSaved = _readPersistedOffsetMs(_fileId);
      if (currentSaved > 0 && _progressState?.mediaEl === node) {
        try {
          _applyAudioOffsetMs(currentSaved);
        } catch (err) {
          console.warn('[fv] deferred offset restore failed:', err);
        }
      }
    };
    if (node.readyState >= 2 /* HAVE_CURRENT_DATA */) {
      // Already decoded a frame — safe to attach now.
      onReady();
    } else {
      node.addEventListener('loadeddata', onReady, { once: true });
    }
  }
}

function _startApiPoll() {
  if (_apiPoll) return;
  if (!_api || typeof _api.getCurrentTime !== 'function') return;
  _apiPoll = setInterval(() => {
    _renderLocalPlaybackUi();
    _emitState();
  }, 500);
}

function _stopApiPoll() {
  if (_apiPoll) {
    clearInterval(_apiPoll);
    _apiPoll = null;
  }
}

function _teardownProgress() {
  const progress = _progressState;
  if (!progress) return;
  const { mediaEl } = progress;
  mediaEl.removeEventListener('timeupdate', progress.onTimeUpdate);
  mediaEl.removeEventListener('pause', progress.onPause);
  mediaEl.removeEventListener('play', progress.onPlay);
  mediaEl.removeEventListener('ended', progress.onEnded);
  window.removeEventListener('pagehide', progress.onPageHide);
  document.removeEventListener('visibilitychange', progress.onVisibilityChange);
  // Audio graph is bound to the media element being torn down — no
  // way to reattach it, so disconnect the chain. Resets the offset
  // state for the next attach (different file may have a different
  // saved offset).
  _detachAudioGraph();
  _audioOffsetMs = 0;
  _closeAudioSyncPopover();
  _renderAudioSyncUi();
  _progressState = null;
}

async function _pushProgress({ force = false, isFinished = false } = {}) {
  const progress = _progressState;
  const fileId = _fileId;
  if (!progress?.mediaEl || !fileId) return null;

  // Suppress pushes while a resume seek is pending. Without this, an
  // early `timeupdate` (currentTime ≈ 0 before the loadedmetadata seek
  // lands) — or a pause/visibilitychange firing in that window —
  // writes 0 over the saved position both locally and upstream, so the
  // user's "continue watching" entry restarts from the beginning on the
  // next click. Marker is set in _primeFloatingVideoResume and cleared
  // on 'seeked'. Managed-seek videos never set the marker (their
  // baseOffsetS already encodes the resume), so this is a no-op there.
  if (Number(progress.mediaEl._augmentumPendingResume || 0) > 0) return null;

  const now = Date.now();
  if (!force && now - (progress.lastSentAt || 0) < 15000) return null;

  const duration = _currentPlaybackDurationS();
  const current = isFinished
    ? (duration || _currentPlaybackTimeS())
    : _currentPlaybackTimeS();
  if (!duration && current <= 0) return null;

  progress.lastSentAt = now;
  const result = await pushMediaProgress(fileId, {
    current_time_s: current,
    duration_s: duration,
    is_finished: isFinished,
  });
  if (result?.progress_pct == null) return result;

  window.dispatchEvent(new CustomEvent('media-player:progress', {
    detail: {
      kind: 'video',
      fileId,
      progressPct: result.progress_pct,
      currentTimeS: current,
      durationS: duration,
      isFinished,
    },
  }));
  window.dispatchEvent(new CustomEvent('floating-video:progress', {
    detail: {
      fileId,
      progressPct: result.progress_pct,
      currentTimeS: current,
      durationS: duration,
      isFinished,
    },
  }));
  return result;
}

function _openDetails() {
  if (!_fileId) return;
  window.dispatchEvent(new CustomEvent('video-player:expand', {
    detail: { fileId: _fileId },
  }));
}

async function _playNext() {
  if (!_nextItem || !_onNext) return;
  _closePlaybackPopover();
  void _pushProgress({ force: true });
  try {
    await _onNext();
  } catch (err) {
    console.warn('[floating-video] next failed:', err);
  }
}

function _providerLabel(provider) {
  const raw = String(provider || '').trim();
  if (!raw) return 'Server';
  return `${raw[0].toUpperCase()}${raw.slice(1)}`;
}

function _remoteSourceType(remote = _remoteSession) {
  return String(remote?.sourceType || 'provider').trim() || 'provider';
}

function _remoteTransportKind(remote = _remoteSession) {
  return String(remote?.transportKind || '').trim().toLowerCase();
}

function _isProviderRemote(remote = _remoteSession) {
  return !!remote?.active && _remoteSourceType(remote) === 'provider';
}

function _isTransportRemote(remote = _remoteSession) {
  return !!remote?.active && _remoteSourceType(remote) === 'transport';
}

function _isLocalTransportRemote(remote = _remoteSession) {
  return !!remote?.active && _remoteSourceType(remote) === 'local_transport';
}

function _remoteSupports(command) {
  const remote = _remoteSession;
  if (!remote?.active) return false;
  const supported = new Set(Array.isArray(remote.supportedCommands) ? remote.supportedCommands : []);
  if (supported.has(command)) return true;
  if (command === 'Seek') {
    return !!remote.canSeek;
  }
  return false;
}

function _remoteHasTransportControls() {
  return !!(
    _remoteSupports('PlayPause')
    || _remoteSupports('Pause')
    || _remoteSupports('Unpause')
    || _remoteSupports('Stop')
    || _remoteSupports('Seek')
    || _remoteSupports('NextTrack')
    || _remoteSupports('PreviousTrack')
  );
}

function _remoteCapabilityNote(remote = _remoteSession) {
  if (!remote?.active) return '';
  const client = String(remote.client || '').trim().toLowerCase();
  const transportKind = _remoteTransportKind(remote);
  if (transportKind === 'dlna' || transportKind === 'cast') {
    return '';
  }
  if (remote.canSeek && !_remoteHasTransportControls()) {
    return 'This device currently only exposes seeking for remote playback.';
  }
  if (!_remoteHasTransportControls()) {
    if (client.includes('dlna')) {
      return 'This DLNA target only exposes volume, mute, and stream selection through Emby.';
    }
    return 'This device does not expose transport controls to Emby.';
  }
  return '';
}

function _stopRemotePolling() {
  if (_remotePollTimer) {
    window.clearTimeout(_remotePollTimer);
    _remotePollTimer = 0;
  }
}

function _scheduleRemotePoll(delay = 5000) {
  if (!_remoteSession?.active) return;
  _stopRemotePolling();
  _remotePollTimer = window.setTimeout(() => {
    void _pollRemoteSession();
  }, Math.max(0, Number(delay) || 0));
}

function _setRemoteMetadata(remote) {
  const deviceName = remote.deviceName || 'Remote device';
  const transportKind = _remoteTransportKind(remote);
  const sourceLabel = transportKind
    ? `${transportKind.toUpperCase()} | ${deviceName}`
    : `${_providerLabel(remote.provider)} | ${deviceName}`;
  _metadata = {
    title: remote.nowPlayingTitle || _metadata.title || 'Remote playback',
    channel: `${_providerLabel(remote.provider)} • ${deviceName}`,
    thumbnail: remote.thumbnail || _metadata.thumbnail || '',
    videoId: '',
  };
  _metadata.channel = sourceLabel;
}

async function _pushRemoteTransportProgress(remote = _remoteSession, { force = false, isFinished = false } = {}) {
  if (!_isTransportRemote(remote) && !_isLocalTransportRemote(remote)) return null;
  if (!remote?.currentFileId) return null;
  const now = Date.now();
  if (!force && now - Number(remote.lastProgressSentAt || 0) < 15000) return null;
  const durationS = Math.max(0, Number(remote.durationS || 0));
  const currentTimeS = isFinished
    ? (durationS || Math.max(0, Number(remote.currentTimeS || 0)))
    : Math.max(0, Number(remote.currentTimeS || 0));
  if (!durationS && currentTimeS <= 0) return null;
  remote.lastProgressSentAt = now;
  return pushMediaProgress(remote.currentFileId, {
    current_time_s: currentTimeS,
    duration_s: durationS,
    is_finished: isFinished,
  });
}

function _emitRemoteProgressUpdate() {
  const remote = _remoteSession;
  if (!remote?.active || !remote.currentFileId) return;
  const currentTimeS = Math.max(0, Number(remote.currentTimeS || 0));
  const durationS = Math.max(0, Number(remote.durationS || 0));
  const sameFile = _remoteProgressEvent.fileId === remote.currentFileId;
  const sameTime = Math.abs((_remoteProgressEvent.currentTimeS || 0) - currentTimeS) < 0.5;
  const sameDuration = Math.abs((_remoteProgressEvent.durationS || 0) - durationS) < 0.5;
  if (sameFile && sameTime && sameDuration) return;

  _remoteProgressEvent = {
    fileId: remote.currentFileId,
    currentTimeS,
    durationS,
  };
  const progressPct = durationS > 0 ? Math.max(0, Math.min(100, (currentTimeS / durationS) * 100)) : 0;
  window.dispatchEvent(new CustomEvent('media-player:progress', {
    detail: {
      kind: 'video',
      fileId: remote.currentFileId,
      progressPct,
      currentTimeS,
      durationS,
      isFinished: false,
    },
  }));
  window.dispatchEvent(new CustomEvent('floating-video:progress', {
    detail: {
      fileId: remote.currentFileId,
      progressPct,
      currentTimeS,
      durationS,
      isFinished: false,
    },
  }));
  void _pushRemoteTransportProgress(remote);
}

function _applyRemoteSessionSnapshot(data = {}) {
  const session = data?.session && typeof data.session === 'object' ? data.session : {};
  const previous = _remoteSession || {};
  const nextExternalId = String(session.now_playing_item_id || previous.currentExternalId || '').trim();
  const nextFileId = String(data.current_file_id || '').trim();
  const remote = {
    active: true,
    sourceType: 'provider',
    transportKind: '',
    transportSessionId: '',
    stopOnClose: previous.stopOnClose !== false,
    provider: String(data.provider || previous.provider || '').trim(),
    serverId: String(data.server_id || previous.serverId || '').trim(),
    sessionId: String(session.session_id || previous.sessionId || '').trim(),
    deviceName: String(session.device_name || session.name || previous.deviceName || '').trim() || 'Remote device',
    client: String(session.client || previous.client || '').trim(),
    nowPlayingTitle: String(session.now_playing_title || data.current_file_name || previous.nowPlayingTitle || '').trim() || 'Remote playback',
    currentExternalId: nextExternalId,
    currentFileId: nextFileId || previous.currentFileId || '',
    thumbnail: String(data.current_cover_url || previous.thumbnail || '').trim(),
    currentTimeS: Math.max(0, Number(session.current_time_s || 0)),
    durationS: Math.max(0, Number(session.duration_s || 0)),
    isPaused: !!session.is_paused,
    isMuted: !!session.is_muted,
    canSeek: !!session.can_seek,
    volumeLevel: Number.isFinite(Number(session.volume_level))
      ? Number(session.volume_level)
      : (Number.isFinite(Number(previous.volumeLevel)) ? Number(previous.volumeLevel) : null),
    supportedCommands: Array.isArray(session.supported_commands)
      ? session.supported_commands.map((item) => String(item || '').trim()).filter(Boolean)
      : (Array.isArray(previous.supportedCommands) ? previous.supportedCommands : []),
    supportsMediaControl: !!session.supports_media_control,
    supportsRemoteControl: !!session.supports_remote_control,
    failures: 0,
    lastProgressSentAt: Number(previous.lastProgressSentAt || 0),
  };
  if (!nextFileId && nextExternalId && nextExternalId !== previous.currentExternalId) {
    remote.currentFileId = '';
  }
  _remoteSession = remote;
  _fileId = remote.currentFileId || _fileId || '';
  _nextItem = null;
  _onNext = null;
  _playbackMenu = null;
  _api = null;
  _setRemoteMetadata(remote);
  _syncUi();
  _emitState();
  _emitRemoteProgressUpdate();
}

function _applyTransportSessionSnapshot(data = {}, { sourceType = 'transport' } = {}) {
  const session = data?.session && typeof data.session === 'object' ? data.session : {};
  const previous = _remoteSession || {};
  const remote = {
    active: true,
    sourceType,
    transportKind: String(data.transport || session.transport_kind || previous.transportKind || '').trim().toLowerCase(),
    transportSessionId: String(session.session_id || previous.transportSessionId || '').trim(),
    stopOnClose: previous.stopOnClose !== false,
    provider: String(session.provider || previous.provider || '').trim(),
    serverId: String(session.server_id || previous.serverId || '').trim(),
    sessionId: String(session.session_id || previous.sessionId || '').trim(),
    deviceName: String(session.receiver_label || previous.deviceName || '').trim() || 'Remote device',
    client: String(session.transport_kind || previous.client || '').trim(),
    nowPlayingTitle: String(session.title || previous.nowPlayingTitle || '').trim() || 'Remote playback',
    currentExternalId: String(session.external_id || previous.currentExternalId || '').trim(),
    currentFileId: String(session.file_id || previous.currentFileId || '').trim(),
    thumbnail: String(session.thumbnail || previous.thumbnail || '').trim(),
    currentTimeS: Math.max(0, Number(session.current_time_s || 0)),
    durationS: Math.max(0, Number(session.duration_s || 0)),
    isPaused: !!session.is_paused,
    isMuted: !!session.is_muted,
    canSeek: !!session.can_seek,
    volumeLevel: Number.isFinite(Number(session.volume_level))
      ? Number(session.volume_level)
      : (Number.isFinite(Number(previous.volumeLevel)) ? Number(previous.volumeLevel) : null),
    supportedCommands: Array.isArray(session.supported_commands)
      ? session.supported_commands.map((item) => String(item || '').trim()).filter(Boolean)
      : (Array.isArray(previous.supportedCommands) ? previous.supportedCommands : []),
    supportsMediaControl: true,
    supportsRemoteControl: true,
    failures: 0,
    lastProgressSentAt: Number(previous.lastProgressSentAt || 0),
  };
  _remoteSession = remote;
  _fileId = remote.currentFileId || _fileId || '';
  _nextItem = null;
  _onNext = null;
  _playbackMenu = null;
  _api = null;
  _setRemoteMetadata(remote);
  _syncUi();
  _emitState();
  _emitRemoteProgressUpdate();
}

function _clearRemoteSession({ stopPlayback = false } = {}) {
  if (_remoteSession?.active && stopPlayback) {
    void _sendRemotePlaystate('Stop', { silent: true });
  }
  _stopRemotePolling();
  _remoteSession = null;
  _remoteSeekDragging = false;
  _remoteVolumeDragging = false;
  _remoteProgressEvent = {
    fileId: '',
    currentTimeS: 0,
    durationS: 0,
  };
}

function _startRemoteController({
  provider = '',
  serverId = '',
  sessionId = '',
  targetLabel = '',
  currentFileId = '',
  title = '',
  thumbnail = '',
  sessionSeed = null,
} = {}) {
  if (!_ensureDom()) return;
  if (document.fullscreenElement && typeof document.exitFullscreen === 'function') {
    document.exitFullscreen().catch(() => {});
  }
  if (_pipActive) {
    _ignorePiPLeave = true;
    void _exitNativePiP();
    _pipActive = false;
    _ignorePiPLeave = false;
  }
  _teardownPiP();
  _teardownProgress();
  _busHandle?.release();
  _releaseAdopted();
  const seed = sessionSeed && typeof sessionSeed === 'object' ? sessionSeed : {};
  _remoteSession = {
    active: true,
    sourceType: 'provider',
    transportKind: '',
    transportSessionId: '',
    stopOnClose: true,
    provider: String(provider || '').trim(),
    serverId: String(serverId || '').trim(),
    sessionId: String(sessionId || '').trim(),
    deviceName: String(targetLabel || seed.device_name || seed.name || '').trim() || 'Remote device',
    client: String(seed.client || '').trim(),
    nowPlayingTitle: String(seed.now_playing_title || title || _metadata.title || '').trim() || 'Remote playback',
    currentExternalId: String(seed.now_playing_item_id || '').trim(),
    currentFileId: String(currentFileId || _fileId || '').trim(),
    thumbnail: String(thumbnail || _metadata.thumbnail || '').trim(),
    currentTimeS: Math.max(0, Number(seed.current_time_s || 0)),
    durationS: Math.max(0, Number(seed.duration_s || 0)),
    isPaused: !!seed.is_paused,
    isMuted: !!seed.is_muted,
    canSeek: !!seed.can_seek,
    volumeLevel: Number.isFinite(Number(seed.volume_level)) ? Number(seed.volume_level) : null,
    supportedCommands: Array.isArray(seed.supported_commands)
      ? seed.supported_commands.map((item) => String(item || '').trim()).filter(Boolean)
      : [],
    supportsMediaControl: !!seed.supports_media_control,
    supportsRemoteControl: !!seed.supports_remote_control,
    failures: 0,
    lastProgressSentAt: 0,
  };
  _fileId = _remoteSession.currentFileId || '';
  _nextItem = null;
  _onNext = null;
  _playbackMenu = null;
  _playbackBusy = false;
  _api = null;
  _root.hidden = false;
  _root.classList.add('open');
  _setRemoteMetadata(_remoteSession);
  openSession({
    shellMode: 'collapsed',
    supportsNativePiP: false,
    isNativePiPActive: false,
    fileId: _fileId || null,
    videoId: null,
    title: _metadata.title || '',
    channel: _metadata.channel || '',
    thumbnail: _metadata.thumbnail || '',
    nextItem: null,
    hasPlaybackOptions: false,
    remoteSessionActive: true,
    remoteSourceType: _remoteSourceType(_remoteSession),
    remoteTransportKind: _remoteTransportKind(_remoteSession),
    remoteProvider: _remoteSession.provider || '',
    remoteServerId: _remoteSession.serverId || '',
    remoteSessionId: _remoteSession.sessionId || '',
    remoteDeviceName: _remoteSession.deviceName || '',
    remoteSupportedCommands: _remoteSession.supportedCommands || [],
    isMuted: !!_remoteSession.isMuted,
    volumeLevel: _remoteSession.volumeLevel ?? null,
    canSeek: !!_remoteSession.canSeek,
  });
  _syncUi();
  setMode('collapsed');
  _emitState();
  _scheduleRemotePoll(200);
}

function _startTransportController(data = {}) {
  if (!_ensureDom()) return;
  if (document.fullscreenElement && typeof document.exitFullscreen === 'function') {
    document.exitFullscreen().catch(() => {});
  }
  if (_pipActive) {
    _ignorePiPLeave = true;
    void _exitNativePiP();
    _pipActive = false;
    _ignorePiPLeave = false;
  }
  _teardownPiP();
  _teardownProgress();
  _busHandle?.release();
  _releaseAdopted();
  _root.hidden = false;
  _root.classList.add('open');
  _applyTransportSessionSnapshot(data);
  openSession({
    shellMode: 'collapsed',
    supportsNativePiP: false,
    isNativePiPActive: false,
    fileId: _fileId || null,
    videoId: null,
    title: _metadata.title || '',
    channel: _metadata.channel || '',
    thumbnail: _metadata.thumbnail || '',
    nextItem: null,
    hasPlaybackOptions: false,
    remoteSessionActive: true,
    remoteSourceType: _remoteSourceType(_remoteSession),
    remoteTransportKind: _remoteTransportKind(_remoteSession),
    remoteProvider: _remoteSession?.provider || '',
    remoteServerId: _remoteSession?.serverId || '',
    remoteSessionId: _remoteSession?.sessionId || '',
    remoteDeviceName: _remoteSession?.deviceName || '',
    remoteSupportedCommands: _remoteSession?.supportedCommands || [],
    isMuted: !!_remoteSession?.isMuted,
    volumeLevel: _remoteSession?.volumeLevel ?? null,
    canSeek: !!_remoteSession?.canSeek,
  });
  _syncUi();
  setMode('collapsed');
  _emitState();
  _scheduleRemotePoll(200);
}

async function _pollRemoteSession() {
  const remote = _remoteSession;
  if (!remote?.active || _remotePollInFlight) return;
  _remotePollInFlight = true;
  let nextDelay = 5000;
  try {
    let data = null;
    if (_isProviderRemote(remote)) {
      data = await fetchMediaRemoteSession(remote.serverId, remote.sessionId);
    } else if (_isTransportRemote(remote)) {
      data = await fetchMediaTransportSession(remote.transportSessionId || remote.sessionId);
    } else if (_isLocalTransportRemote(remote) && _remoteTransportKind(remote) === 'cast') {
      data = await _pollCastRemoteSession();
    }
    if (!data?.session) {
      remote.failures = Number(remote.failures || 0) + 1;
      if (remote.failures >= 3) {
        remote.stopOnClose = false;
        close();
        _showToast(`Lost connection to ${remote.deviceName || 'remote device'}`, 'warning');
        return;
      }
      nextDelay = 2500;
      return;
    }
    if (_isProviderRemote(remote)) _applyRemoteSessionSnapshot(data);
    else if (_isLocalTransportRemote(remote)) _applyTransportSessionSnapshot(data, { sourceType: 'local_transport' });
    else _applyTransportSessionSnapshot(data);
  } catch (err) {
    console.warn('[floating-video] remote session poll failed:', err);
    remote.failures = Number(remote.failures || 0) + 1;
    nextDelay = 2500;
  } finally {
    _remotePollInFlight = false;
    if (_remoteSession?.active) _scheduleRemotePoll(nextDelay);
  }
}

async function _sendRemotePlaystate(command, { seekPositionS = null, silent = false } = {}) {
  const remote = _remoteSession;
  if (!remote?.active) return false;
  let result = null;
  if (_isProviderRemote(remote)) {
    if (!remote.serverId || !remote.sessionId) return false;
    result = await sendMediaRemoteSessionPlaystate(remote.serverId, remote.sessionId, {
      command,
      seek_position_s: seekPositionS,
    });
  } else if (_isTransportRemote(remote)) {
    result = await sendMediaTransportSessionPlaystate(remote.transportSessionId || remote.sessionId, {
      command,
      seek_position_s: seekPositionS,
    });
  } else if (_isLocalTransportRemote(remote) && _remoteTransportKind(remote) === 'cast') {
    result = await _sendCastPlaystate(command, { seekPositionS });
  }
  if (!result?.status) {
    if (!silent) _showToast('Could not control remote playback', 'error');
    return false;
  }
  if (command === 'Pause') remote.isPaused = true;
  else if (command === 'Unpause') remote.isPaused = false;
  else if (command === 'PlayPause') remote.isPaused = !remote.isPaused;
  else if (command === 'Seek' && Number.isFinite(Number(seekPositionS))) {
    remote.currentTimeS = Math.max(0, Number(seekPositionS));
  } else if (command === 'Stop') {
    void _pushRemoteTransportProgress(remote, { force: true });
    remote.stopOnClose = false;
    close();
    return true;
  }
  _syncUi();
  _emitState();
  _scheduleRemotePoll(400);
  return true;
}

async function _sendRemoteGeneral(command, args = null, { silent = false } = {}) {
  const remote = _remoteSession;
  if (!remote?.active) return false;
  let result = null;
  if (_isProviderRemote(remote)) {
    if (!remote.serverId || !remote.sessionId) return false;
    result = await sendMediaRemoteSessionGeneral(remote.serverId, remote.sessionId, {
      command,
      arguments: args,
    });
  } else if (_isTransportRemote(remote)) {
    result = await sendMediaTransportSessionGeneral(remote.transportSessionId || remote.sessionId, {
      command,
      arguments: args,
    });
  } else if (_isLocalTransportRemote(remote) && _remoteTransportKind(remote) === 'cast') {
    result = await _sendCastGeneral(command, args);
  }
  if (!result?.status) {
    if (!silent) _showToast('Could not update the remote device', 'error');
    return false;
  }
  if (command === 'ToggleMute') remote.isMuted = !remote.isMuted;
  else if (command === 'Mute') remote.isMuted = true;
  else if (command === 'Unmute') remote.isMuted = false;
  else if (command === 'SetVolume' && Number.isFinite(Number(args?.Volume))) {
    remote.volumeLevel = Math.max(0, Math.min(100, Number(args.Volume)));
  }
  _syncUi();
  _emitState();
  _scheduleRemotePoll(400);
  return true;
}

async function _handleRemoteAction(action) {
  const remote = _remoteSession;
  if (!remote?.active) return;
  if (action === 'toggle-play') {
    if (_remoteSupports('PlayPause')) {
      await _sendRemotePlaystate('PlayPause');
    } else if (_remoteSupports('Pause') || _remoteSupports('Unpause')) {
      await _sendRemotePlaystate(remote.isPaused ? 'Unpause' : 'Pause');
    }
    return;
  }
  if (action === 'seek-back' && remote.canSeek) {
    await _sendRemotePlaystate('Seek', { seekPositionS: Math.max(0, Number(remote.currentTimeS || 0) - 15) });
    return;
  }
  if (action === 'seek-forward' && remote.canSeek) {
    const upper = remote.durationS > 0 ? remote.durationS : Number(remote.currentTimeS || 0) + 30;
    await _sendRemotePlaystate('Seek', { seekPositionS: Math.min(upper, Number(remote.currentTimeS || 0) + 30) });
    return;
  }
  if (action === 'previous' && _remoteSupports('PreviousTrack')) {
    await _sendRemotePlaystate('PreviousTrack');
    return;
  }
  if (action === 'next' && _remoteSupports('NextTrack')) {
    await _sendRemotePlaystate('NextTrack');
    return;
  }
  if (action === 'stop' && _remoteSupports('Stop')) {
    await _sendRemotePlaystate('Stop');
    return;
  }
  if (action === 'toggle-mute') {
    if (_remoteSupports('ToggleMute')) {
      await _sendRemoteGeneral('ToggleMute');
    } else if (_remoteSupports(remote.isMuted ? 'Unmute' : 'Mute')) {
      await _sendRemoteGeneral(remote.isMuted ? 'Unmute' : 'Mute');
    }
    return;
  }
  if (action === 'volume-down') {
    if (_remoteSupports('VolumeDown')) {
      await _sendRemoteGeneral('VolumeDown');
    } else if (_remoteSupports('SetVolume')) {
      await _sendRemoteGeneral('SetVolume', {
        Volume: Math.max(0, Number(remote.volumeLevel ?? 50) - 10),
      });
    }
    return;
  }
  if (action === 'volume-up') {
    if (_remoteSupports('VolumeUp')) {
      await _sendRemoteGeneral('VolumeUp');
    } else if (_remoteSupports('SetVolume')) {
      await _sendRemoteGeneral('SetVolume', {
        Volume: Math.min(100, Number(remote.volumeLevel ?? 50) + 10),
      });
    }
  }
}

async function _commitRemoteSeek() {
  const remote = _remoteSession;
  if (!remote?.active || !_remoteSeekEl || !remote.canSeek || remote.durationS <= 0) return;
  const frac = Math.max(0, Math.min(1, Number(_remoteSeekEl.value || 0) / 1000));
  await _sendRemotePlaystate('Seek', { seekPositionS: frac * remote.durationS });
}

async function _commitRemoteVolume() {
  const remote = _remoteSession;
  if (!remote?.active || !_remoteVolumeEl || !_remoteSupports('SetVolume')) return;
  await _sendRemoteGeneral('SetVolume', {
    Volume: Math.max(0, Math.min(100, Number(_remoteVolumeEl.value || 0))),
  });
}

async function _commitLocalSeek() {
  if (_remoteSession?.active || !_localSeekEl || !_currentPlaybackCanSeek()) return;
  const durationS = _currentPlaybackDurationS();
  if (durationS <= 0) return;
  const frac = Math.max(0, Math.min(1, Number(_localSeekEl.value || 0) / 1000));
  const targetTimeS = frac * durationS;
  // Set the hold BEFORE awaiting so the render at the end of this
  // function (and the 500ms poll that may fire during the await)
  // sees the cooldown active and pins to the target. 2000ms covers
  // even a sluggish YouTube buffer; the render will clear the cooldown
  // early once currentTime is within ~1.5s of target.
  _localSeekTargetTimeS = targetTimeS;
  _localSeekCooldownUntil = Date.now() + 2000;
  if (_api && typeof _api.seekTo === 'function') {
    await _api.seekTo(targetTimeS);
  } else {
    const mediaEl = _progressState?.mediaEl;
    if (mediaEl) mediaEl.currentTime = targetTimeS;
  }
  _renderLocalPlaybackUi();
  _emitState();
}

/**
 * Toggle play/pause on the local media element. Routes through the
 * playback API's `togglePlay` when available (preserves any
 * managed-seek semantics), falls back to direct mediaEl manipulation
 * for the common case. Re-renders immediately so the button icon
 * doesn't lag the next timeupdate tick.
 */
// Window-level keyboard shortcuts. Wired once in _ensureDom and gated
// inside the handler — keeps the API surface tight (no separate
// register/unregister at open/close) and matches the always-on nature
// of YouTube/Plex hotkeys. Each branch is a no-op when the player is
// hidden, a text field has focus, or a remote session owns playback
// (where the user expects the device's own remote semantics).
function _onWindowKeyDown(e) {
  if (!_root || _root.hidden) return;
  if (_remoteSession?.active) return;
  if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
  const target = e.target;
  if (target && typeof target.closest === 'function') {
    if (target.closest('input, textarea, select, [contenteditable=""], [contenteditable="true"]')) return;
  }
  // Don't poach key events that belong to focused buttons inside the
  // shell — e.g. Space on the play button fires the click handler, no
  // need for us to also intercept it.
  if (target && target instanceof HTMLElement && target.closest('.fv-root button, .fv-root [role="button"]')) return;

  switch (e.key) {
    case ' ':
    case 'k':
    case 'K':
      e.preventDefault();
      _toggleLocalPlayPause();
      return;
    case 'ArrowLeft':
    case 'j':
    case 'J':
      e.preventDefault();
      void _seekLocalBy(-10);
      return;
    case 'ArrowRight':
    case 'l':
    case 'L':
      e.preventDefault();
      void _seekLocalBy(10);
      return;
    case 'ArrowUp':
      e.preventDefault();
      _bumpLocalVolume(5);
      return;
    case 'ArrowDown':
      e.preventDefault();
      _bumpLocalVolume(-5);
      return;
    case 'm':
    case 'M':
      e.preventDefault();
      _toggleLocalMute();
      return;
    case 'f':
    case 'F':
      e.preventDefault();
      setMode(_mode === 'fullscreen' ? 'companion' : 'fullscreen');
      return;
    default:
  }
}

// In every mode except fullscreen, our custom overlay owns the chrome
// and native HTML5 controls would just duplicate the timeline. In
// fullscreen the overlay is unreachable (it lives on the root, not
// inside .fv-iframe-slot which is the fullscreened element), so the
// browser bar is what the user actually sees and gets to interact with.
// Toggle accordingly on every mode change.
function _syncNativeVideoControls() {
  const video = _videoNodeForNativePiP();
  if (!video) return;
  video.controls = _mode === 'fullscreen';
}

function _bumpLocalVolume(deltaPct) {
  if (!_localVolumeEl) return;
  const current = Number(_localVolumeEl.value || 0);
  const next = Math.max(0, Math.min(100, current + deltaPct));
  _localVolumeEl.value = String(next);
  _commitLocalVolume();
}

function _toggleLocalPlayPause() {
  if (_remoteSession?.active) return;
  // api-driven sources (YouTube) have no HTMLVideoElement to bind
  // .paused against — route through the api directly. Existing
  // HTML5 path retained below for media-server video.
  if (_api && typeof _api.togglePlay === 'function') {
    _api.togglePlay();
    _renderLocalPlaybackUi();
    _emitState();
    return;
  }
  const mediaEl = _progressState?.mediaEl;
  if (!mediaEl) return;
  if (mediaEl.paused) {
    mediaEl.play().catch(() => { /* autoplay-blocked; user can retry */ });
  } else {
    mediaEl.pause();
  }
  _renderLocalPlaybackUi();
}

/**
 * Skip the local timeline by ``deltaS`` seconds (negative for back,
 * positive for forward). Clamps to [0, duration] so the skip never
 * lands past the end. Uses the same seekTo path as the seek bar so
 * managed-seek transcoding restarts work correctly.
 */
async function _seekLocalBy(deltaS) {
  if (_remoteSession?.active || !_currentPlaybackCanSeek()) return;
  const durationS = _currentPlaybackDurationS();
  const currentS = _currentPlaybackTimeS();
  const target = Math.max(0, durationS > 0 ? Math.min(durationS, currentS + deltaS) : currentS + deltaS);
  // Same post-seek hold as _commitLocalSeek — prevents the slider
  // from jumping back to currentS while the api's getCurrentTime is
  // still catching up to the new position.
  _localSeekTargetTimeS = target;
  _localSeekCooldownUntil = Date.now() + 2000;
  if (_api && typeof _api.seekTo === 'function') {
    await _api.seekTo(target);
  } else {
    const mediaEl = _progressState?.mediaEl;
    if (mediaEl) mediaEl.currentTime = target;
  }
  _renderLocalPlaybackUi();
  _emitState();
}

/** Toggle ``mediaEl.muted``. The ``mute`` icon flips inside
 *  _renderLocalPlaybackUi so the click→render dance stays in one
 *  place; same reasoning as the play/pause icon update. */
function _toggleLocalMute() {
  if (_remoteSession?.active) return;
  if (_api && typeof _api.toggleMute === 'function') {
    _api.toggleMute();
    _renderLocalPlaybackUi();
    return;
  }
  const mediaEl = _progressState?.mediaEl;
  if (!mediaEl) return;
  mediaEl.muted = !mediaEl.muted;
  _renderLocalPlaybackUi();
}

/** Commit the volume slider value to the media element. The slider
 *  is 0..100 to match the existing remote volume slider's range; we
 *  divide by 100 to get the 0..1 ``volume`` property. Setting volume
 *  > 0 clears the mute flag for the obvious user-intent reason. */
function _commitLocalVolume() {
  if (_remoteSession?.active || !_localVolumeEl) return;
  const v = Math.max(0, Math.min(100, Number(_localVolumeEl.value || 0))) / 100;
  // api-driven sources (YouTube) route through the api setter. Unmute
  // on volume-up mirrors the HTML5 path's user-intent rule below.
  if (_api && typeof _api.setVolume === 'function' && !_progressState?.mediaEl) {
    _api.setVolume(v);
    if (v > 0 && typeof _api.isMuted === 'function' && _api.isMuted()
        && typeof _api.toggleMute === 'function') {
      _api.toggleMute();
    }
    _renderLocalPlaybackUi();
    return;
  }
  const mediaEl = _progressState?.mediaEl;
  if (!mediaEl) return;
  // When the audio graph is live (user has applied an offset), volume
  // routes through the GainNode — the videoEl's native audio output
  // is muted to avoid double playback. When the graph isn't live,
  // we keep using videoEl.volume directly so users on clean paths
  // don't pay any Web Audio overhead.
  if (_audioGainNode) {
    _audioGainNode.gain.value = v;
  } else {
    mediaEl.volume = v;
  }
  if (v > 0 && mediaEl.muted && !_audioGainNode) mediaEl.muted = false;
  _renderLocalPlaybackUi();
}

// --- A/V offset (lip-sync compensation) --------------------------------

// localStorage key prefixes. The profile prefix keys by source
// signature (provider|container|video_codec|audio_codec) so a fix
// tuned on one Jellyfin episode auto-applies to the next episode
// using the same transcode pipeline. The file-id prefix is the legacy
// per-file key — kept for read fallback so existing offsets survive
// the upgrade.
const _SYNC_OFFSET_STORAGE_PREFIX = 'augmentum.fv.audio-offset.';
const _SYNC_OFFSET_PROFILE_PREFIX = 'augmentum.fv.audio-offset.profile.';

function _clampOffsetMs(raw) {
  return Math.max(0, Math.min(2000, Number(raw) || 0));
}

function _readPersistedOffsetMs(fileId) {
  try {
    // Profile key wins when present — that's the cross-file memory.
    if (_syncProfileKey) {
      const raw = localStorage.getItem(_SYNC_OFFSET_PROFILE_PREFIX + _syncProfileKey);
      if (raw) return _clampOffsetMs(raw);
    }
    // Fallback to legacy per-file key so users don't lose offsets
    // they tuned before the profile keying landed.
    if (fileId) {
      const raw = localStorage.getItem(_SYNC_OFFSET_STORAGE_PREFIX + fileId);
      if (raw) return _clampOffsetMs(raw);
    }
  } catch { /* private mode / quota — best-effort */ }
  return 0;
}

function _persistOffsetMs(fileId, ms) {
  try {
    if (_syncProfileKey) {
      const k = _SYNC_OFFSET_PROFILE_PREFIX + _syncProfileKey;
      if (ms > 0) {
        localStorage.setItem(k, String(ms));
      } else {
        localStorage.removeItem(k);
      }
      // Once the profile key is in play, drop any legacy per-file
      // entry for this exact file so we don't double-bookkeep.
      if (fileId) localStorage.removeItem(_SYNC_OFFSET_STORAGE_PREFIX + fileId);
      return;
    }
    // No profile signature for this stream (local file, missing
    // playback metadata) — keep the legacy per-file behavior.
    if (!fileId) return;
    if (ms > 0) {
      localStorage.setItem(_SYNC_OFFSET_STORAGE_PREFIX + fileId, String(ms));
    } else {
      localStorage.removeItem(_SYNC_OFFSET_STORAGE_PREFIX + fileId);
    }
  } catch { /* private mode / quota — best-effort */ }
}

/**
 * Build the Web Audio graph on the current media element. Idempotent
 * for the same element; rebuilds if the element has changed (player
 * navigated to a different episode). Returns true if the graph is
 * live and ready to apply an offset.
 *
 * Browsers cap MediaElementAudioSourceNode to ONE per element — once
 * created, you can't recreate it. The router stays attached for the
 * life of the element. We tear down the AudioContext only when the
 * player closes (see _detachAudioGraph).
 */
// Per-session disable flag — flips true if Web Audio init crashes
// or refuses. Once set, _ensureAudioGraph short-circuits without
// retrying for the rest of the session, so one bad video can't keep
// triggering the same crash on every reattempt. Reset on full page
// reload (module unload).
let _audioGraphDisabledForSession = false;

function _ensureAudioGraph(mediaEl) {
  if (_audioGraphDisabledForSession) return false;
  if (!mediaEl) return false;
  if (_audioGraphMediaEl === mediaEl && _audioContext && _audioDelayNode) {
    return true;
  }
  // Pre-flight checks — bail BEFORE touching the element if the
  // browser is in a state where createMediaElementSource is risky.
  // readyState < HAVE_CURRENT_DATA means the decoder hasn't decoded
  // a frame yet; calling createMediaElementSource here was the
  // observed trigger for the GPU-process crash + RGB static incident
  // on 2026-05-02. Better to skip than to crash.
  if (mediaEl.readyState < 2 /* HAVE_CURRENT_DATA */) {
    return false;
  }
  // Different element than what we last attached — discard the old
  // graph. The old element's audio output won't fire again because
  // it's about to be removed from DOM, so leaking the disconnected
  // nodes is harmless.
  if (_audioGraphMediaEl && _audioGraphMediaEl !== mediaEl) {
    _detachAudioGraph();
  }
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return false;
    if (!_audioContext) {
      _audioContext = new Ctx();
    }
    // maxDelayTime is fixed at construction. 2.5 gives us headroom
    // above the 2000ms slider max so DelayNode never refuses to set
    // delayTime due to ceiling clipping.
    _audioDelayNode = _audioContext.createDelay(2.5);
    _audioDelayNode.delayTime.value = 0;
    _audioGainNode = _audioContext.createGain();
    // Seed the gain from the volume slider (or videoEl.volume if the
    // slider hasn't been touched yet) so we don't blast the user at
    // 1.0 right when the graph turns on.
    const initialVolume = _localVolumeEl
      ? Math.max(0, Math.min(100, Number(_localVolumeEl.value || 0))) / 100
      : Math.max(0, Math.min(1, Number(mediaEl.volume || 1)));
    _audioGainNode.gain.value = initialVolume;
    _audioSourceNode = _audioContext.createMediaElementSource(mediaEl);
    _audioSourceNode.connect(_audioDelayNode);
    _audioDelayNode.connect(_audioGainNode);
    _audioGainNode.connect(_audioContext.destination);
    // Mute the videoEl's native audio output to avoid double playback —
    // the audio now flows through the AudioContext only.
    mediaEl.muted = true;
    _audioGraphMediaEl = mediaEl;
    return true;
  } catch (err) {
    // Most likely cause: createMediaElementSource throws InvalidStateError
    // when the element is in an unsupported state (cross-origin tainted,
    // Range request mid-flight, etc.). Whatever the cause, mark Web
    // Audio routing dead for this session so we don't re-trigger and
    // nuke the GPU process with repeated attempts.
    console.warn('[fv] audio graph init failed; disabling A/V offset for session:', err);
    _audioGraphDisabledForSession = true;
    _detachAudioGraph();
    return false;
  }
}

function _detachAudioGraph() {
  // We can't unmake a MediaElementAudioSourceNode, but we can at
  // least disconnect the chain so audio routes back to the element.
  // In practice this only runs on player close, when the element is
  // being thrown away anyway.
  try { _audioSourceNode?.disconnect(); } catch { /* ignore */ }
  try { _audioDelayNode?.disconnect();  } catch { /* ignore */ }
  try { _audioGainNode?.disconnect();   } catch { /* ignore */ }
  _audioSourceNode = null;
  _audioDelayNode = null;
  _audioGainNode = null;
  _audioGraphMediaEl = null;
  // AudioContext is intentionally NOT closed here — closing is async
  // and a fresh open() of another video would race with it. The GC
  // will collect when the page tears down.
}

function _closeAudioContext() {
  const ctx = _audioContext;
  _audioContext = null;
  if (!ctx || ctx.state === 'closed') return;
  try {
    ctx.close().catch((err) => {
      console.warn('[fv] audio context close failed:', err);
    });
  } catch (err) {
    console.warn('[fv] audio context close failed:', err);
  }
}

/**
 * Apply ``ms`` as the audio offset. Lazy-initializes the graph on
 * first non-zero call; once initialized, subsequent calls (including
 * back to 0) just update DelayNode.delayTime. Persists the value
 * keyed by the current file_id so a per-file fix is sticky.
 */
function _applyAudioOffsetMs(ms) {
  const clamped = Math.max(0, Math.min(2000, Math.round(Number(ms) || 0)));
  _audioOffsetMs = clamped;

  const mediaEl = _progressState?.mediaEl;
  if (clamped === 0 && !_audioGraphMediaEl) {
    // No graph yet, no offset desired — nothing to do. Keep the
    // native-audio path clean. Render so the slider/value/dot reflect
    // the new (zeroed) state.
    _renderAudioSyncUi();
    _persistOffsetMs(_fileId, 0);
    return;
  }
  if (clamped > 0 && !_ensureAudioGraph(mediaEl)) {
    // Graph init failed (browser without Web Audio, autoplay policy).
    // Surface the failure via the dot/label so the user knows their
    // change didn't apply.
    _audioOffsetMs = 0;
    _renderAudioSyncUi();
    return;
  }
  if (_audioContext && _audioContext.state === 'suspended') {
    // Some browsers start the context suspended; resume on first use.
    _audioContext.resume().catch(() => { /* ignore */ });
  }
  if (_audioDelayNode) {
    // delayTime is in seconds, not ms. Use setTargetAtTime for a soft
    // 30ms ramp so dragging the slider doesn't audibly click. The
    // ramp converges to within ~3% of target after ~3 time-constants
    // (~90ms), which is well under the typical user adjustment cadence.
    const targetSec = clamped / 1000;
    try {
      _audioDelayNode.delayTime.setTargetAtTime(
        targetSec, _audioContext.currentTime, 0.03,
      );
    } catch {
      _audioDelayNode.delayTime.value = targetSec;
    }
  }
  _persistOffsetMs(_fileId, clamped);
  _renderAudioSyncUi();
}

function _renderAudioSyncUi() {
  if (_localSyncRangeEl) {
    _localSyncRangeEl.value = String(_audioOffsetMs);
  }
  if (_localSyncValueEl) {
    _localSyncValueEl.textContent = `+${_audioOffsetMs} ms`;
  }
  if (_localSyncBtn) {
    _localSyncBtn.classList.toggle('has-offset', _audioOffsetMs > 0);
  }
  if (_localSyncDotEl) {
    _localSyncDotEl.hidden = _audioOffsetMs === 0;
  }
}

function _openAudioSyncPopover() {
  if (!_localSyncPopoverEl || !_localSyncBtn) return;
  if (_audioGraphDisabledForSession) {
    // Don't pretend the slider works when Web Audio init failed for
    // this session — a previous attempt already errored. Surface
    // why through a toast import (lazy to keep the floating-video
    // bundle from depending on app.js at module init).
    import('./app.js').then(m => m.showToast?.(
      "Audio sync isn't available on this video. Try the next episode or refresh the page.",
      'warning', 3500,
    )).catch(() => { /* ignore */ });
    return;
  }
  _localSyncPopoverEl.hidden = false;
  _localSyncBtn.setAttribute('aria-expanded', 'true');
  // Refresh slider in case offset was changed elsewhere (file open,
  // direct call). The popover otherwise inherits whatever the last
  // _renderAudioSyncUi() set.
  _renderAudioSyncUi();
  // Outside-click + Escape dismiss. Capture-phase so a click on the
  // popover's own children stays inside.
  setTimeout(() => {
    document.addEventListener('click', _onAudioSyncOutside, true);
    document.addEventListener('keydown', _onAudioSyncKeydown, true);
  }, 0);
}

function _closeAudioSyncPopover() {
  if (!_localSyncPopoverEl) return;
  _localSyncPopoverEl.hidden = true;
  _localSyncBtn?.setAttribute('aria-expanded', 'false');
  document.removeEventListener('click', _onAudioSyncOutside, true);
  document.removeEventListener('keydown', _onAudioSyncKeydown, true);
}

function _onAudioSyncOutside(e) {
  if (!_localSyncPopoverEl || _localSyncPopoverEl.hidden) return;
  if (_localSyncPopoverEl.contains(e.target)) return;
  if (_localSyncBtn?.contains(e.target)) return;
  _closeAudioSyncPopover();
}

function _onAudioSyncKeydown(e) {
  if (e.key === 'Escape') _closeAudioSyncPopover();
}

function _emitState() {
  const mediaEl = _progressState?.mediaEl;
  const remote = _remoteSession;
  // api-driven sources (YouTube) have neither a mediaEl nor a remote
  // session — fall through to the api so the emitted state reflects
  // real playback instead of a stuck 0/0/false.
  const hasApiTime = !mediaEl && !remote?.active
    && _api && typeof _api.getCurrentTime === 'function';
  const currentTimeS = mediaEl || hasApiTime
    ? _currentPlaybackTimeS()
    : Math.max(0, Number(remote?.currentTimeS || 0));
  const durationS = mediaEl || hasApiTime
    ? _currentPlaybackDurationS()
    : Math.max(0, Number(remote?.durationS || 0));
  const isPlaying = mediaEl
    ? !!(!mediaEl.paused && !mediaEl.ended)
    : hasApiTime && typeof _api.isPlaying === 'function'
      ? !!_api.isPlaying()
      : !!(remote?.active && !remote?.isPaused);
  updateSession({
    isOpen: isOpen(),
    shellMode: _mode,
    deviceKind: _currentDeviceKind(),
    fileId: _fileId || null,
    videoId: _metadata.videoId || null,
    title: _metadata.title || '',
    channel: _metadata.channel || '',
    thumbnail: _metadata.thumbnail || '',
    nextItem: _nextItem,
    currentTimeS,
    durationS,
    isPlaying,
    hasPlaybackOptions: _hasPlaybackOptions(),
    supportsNativePiP: !remote?.active && _supportsNativePiP(),
    isNativePiPActive: _pipActive,
    remoteSessionActive: !!remote?.active,
    remoteSourceType: _remoteSourceType(remote),
    remoteTransportKind: _remoteTransportKind(remote),
    remoteProvider: remote?.provider || '',
    remoteServerId: remote?.serverId || '',
    remoteSessionId: remote?.sessionId || '',
    remoteDeviceName: remote?.deviceName || '',
    remoteSupportedCommands: Array.isArray(remote?.supportedCommands) ? remote.supportedCommands : [],
    isMuted: !!remote?.isMuted,
    volumeLevel: remote?.volumeLevel ?? null,
    canSeek: !!remote?.canSeek,
  });
  window.dispatchEvent(new CustomEvent('floating-video:state', {
    detail: {
      isOpen: isOpen(),
      mode: _mode,
      shellMode: _mode,
      deviceKind: _currentDeviceKind(),
      fileId: _fileId || null,
      videoId: _metadata.videoId || null,
      title: _metadata.title || '',
      channel: _metadata.channel || '',
      nextItem: _nextItem,
      currentTimeS,
      durationS,
      isPlaying,
      supportsNativePiP: !remote?.active && _supportsNativePiP(),
      isNativePiPActive: _pipActive,
      remoteSessionActive: !!remote?.active,
      remoteSourceType: _remoteSourceType(remote),
      remoteTransportKind: _remoteTransportKind(remote),
      remoteProvider: remote?.provider || '',
      remoteServerId: remote?.serverId || '',
      remoteSessionId: remote?.sessionId || '',
      remoteDeviceName: remote?.deviceName || '',
      isMuted: !!remote?.isMuted,
      volumeLevel: remote?.volumeLevel ?? null,
      canSeek: !!remote?.canSeek,
    },
  }));
}

function _playbackState() {
  if (!_playbackMenu || typeof _playbackMenu.getState !== 'function') return null;
  return _playbackMenu.getState() || null;
}

function _hasPlaybackOptions() {
  const playback = _playbackState();
  if (!playback?.media_sources?.length) return false;
  if (playback.media_sources.length > 1) return true;
  const selected = playback.media_sources.find((source) => source?.id === playback.selected_media_source_id)
    || playback.media_sources[0];
  return (selected?.audio_tracks?.length || 0) > 1 || (selected?.subtitle_tracks?.length || 0) >= 1;
}

function _currentMediaUrl(node = _adoptedIframe) {
  const video = _videoNodeForNativePiP(node);
  if (!video || typeof window === 'undefined') return null;
  const raw = String(video.currentSrc || video.src || '').trim();
  if (!raw) return null;
  try {
    return new URL(raw, window.location.href);
  } catch {
    return null;
  }
}

function _isLoopbackHost(hostname = '') {
  const host = String(hostname || '').trim().toLowerCase();
  return (
    !host
    || host === 'localhost'
    || host === '127.0.0.1'
    || host === '::1'
    || host === '0.0.0.0'
    || host.endsWith('.localhost')
  );
}

function _supportsBrowserRemotePlayback(node = _adoptedIframe) {
  const video = _videoNodeForNativePiP(node);
  if (!video || video.disableRemotePlayback) return false;
  return !!(
    (video.remote && typeof video.remote.prompt === 'function')
    || typeof video.webkitShowPlaybackTargetPicker === 'function'
  );
}

function _browserRemoteOutputLabel(node = _adoptedIframe) {
  const video = _videoNodeForNativePiP(node);
  if (video && typeof video.webkitShowPlaybackTargetPicker === 'function') return 'AirPlay';
  return 'Nearby device';
}

function _sessionSupportedCommands(session) {
  return (Array.isArray(session?.supported_commands) ? session.supported_commands : [])
    .map((command) => String(command || '').trim().toLowerCase())
    .filter(Boolean);
}

function _providerSessionControlScore(session) {
  const commands = new Set(_sessionSupportedCommands(session));
  const hasExplicitTransportControl = (
    commands.has('pause')
    || commands.has('unpause')
    || commands.has('playpause')
    || commands.has('stop')
    || commands.has('seek')
    || commands.has('nexttrack')
    || commands.has('previoustrack')
    || commands.has('fastforward')
    || commands.has('rewind')
  );
  if (hasExplicitTransportControl) return 3;
  if (!!session?.can_seek) return 2;
  const hasTransportControl = (
    commands.has('volumeup')
    || commands.has('volumedown')
    || commands.has('mute')
    || commands.has('unmute')
    || commands.has('togglemute')
    || commands.has('setvolume')
    || commands.has('setaudiostreamindex')
    || commands.has('setsubtitlestreamindex')
  );
  if (hasTransportControl) return 2;
  if (commands.has('playmediasource') || session?.supports_remote_control || session?.supports_media_control) return 1;
  return 0;
}

function _providerSessionCapabilityLabel(session) {
  const score = _providerSessionControlScore(session);
  if (score >= 3) return 'Full remote control';
  if (score === 2) {
    return session?.can_seek ? 'Seek-only remote control' : 'Limited remote control';
  }
  if (score === 1) return 'Launch only';
  return '';
}

function _outputPopoverGroup(label, innerHtml, note = '') {
  return `
    <div class="fv-popover-group">
      <div class="fv-popover-label">${_esc(label)}</div>
      ${innerHtml}
      ${note ? `<div class="fv-popover-note">${_esc(note)}</div>` : ''}
    </div>
  `;
}

function _isMediaProxyUrl(url) {
  return !!(
    url
    && typeof window !== 'undefined'
    && url.origin === window.location.origin
    && url.pathname.startsWith('/api/media/stream/')
  );
}

function _canCastFromDirectElementSource(url = _currentMediaUrl()) {
  return !!(
    url
    && typeof window !== 'undefined'
    && /^https?:$/i.test(url.protocol)
    && url.origin !== window.location.origin
    && !_isLoopbackHost(url.hostname)
  );
}

function _supportsCastOutput() {
  if (_remoteSession?.active) return false;
  if (typeof window === 'undefined' || !window.isSecureContext) return false;
  const video = _videoNodeForNativePiP();
  if (!video) return false;
  return _canCastFromDirectElementSource() || (_fileId && _isMediaProxyUrl(_currentMediaUrl()));
}

function _supportsManagedDeviceCast() {
  if (_remoteSession?.active) return false;
  const video = _videoNodeForNativePiP();
  const url = _currentMediaUrl();
  if (!video || !url) return false;
  return _canCastFromDirectElementSource(url) || !!(_fileId && _isMediaProxyUrl(url));
}

function _hasOutputCandidates() {
  if (_remoteSession?.active) return false;
  return !!(
    _supportsBrowserRemotePlayback()
    || _supportsCastOutput()
    || _supportsManagedDeviceCast()
    || (_fileId && _isMediaProxyUrl(_currentMediaUrl()))
  );
}

function _togglePlaybackPopover() {
  if (!_popoverEl || !_hasPlaybackOptions()) return;
  if (_popoverEl.hidden || _popoverMode !== 'playback') {
    _popoverMode = 'playback';
    _renderPlaybackPopover();
    _popoverEl.hidden = false;
  } else {
    _closePlaybackPopover();
  }
}

function _closePlaybackPopover() {
  if (_popoverEl) {
    _popoverEl.hidden = true;
    _popoverEl.innerHTML = '';
  }
  _popoverMode = '';
  _tracksBtn?.classList.remove('is-active');
  _outputsBtn?.classList.remove('is-active');
}

function _esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]
  ));
}

function _trackButtons(tracks, selectedIndex, attr) {
  return (tracks || []).map((track) => {
    const index = Number(track?.index);
    const active = Number.isFinite(index) && Number(selectedIndex) === index;
    return `<button type="button" class="fv-popover-btn${active ? ' active' : ''}" ${attr}="${_esc(String(index))}">${_esc(track?.label || 'Track')}</button>`;
  }).join('');
}

function _renderPlaybackPopover() {
  if (!_popoverEl) return;
  const playback = _playbackState();
  if (!playback?.media_sources?.length) {
    _closePlaybackPopover();
    return;
  }
  const selected = playback.media_sources.find((source) => source?.id === playback.selected_media_source_id)
    || playback.media_sources[0];
  const groups = [];
  if (playback.media_sources.length > 1) {
    groups.push(`
      <div class="fv-popover-group">
        <div class="fv-popover-label">Version</div>
        <div class="fv-popover-options">
          ${playback.media_sources.map((source) => `
            <button type="button" class="fv-popover-btn${source?.id === playback.selected_media_source_id ? ' active' : ''}" data-fv-playback-source="${_esc(source?.id || '')}">${_esc(source?.label || 'Version')}</button>
          `).join('')}
        </div>
      </div>
    `);
  }
  if ((selected?.audio_tracks?.length || 0) > 1) {
    groups.push(`
      <div class="fv-popover-group">
        <div class="fv-popover-label">Audio</div>
        <div class="fv-popover-options">
          ${_trackButtons(selected.audio_tracks || [], playback.selected_audio_stream_index, 'data-fv-playback-audio')}
        </div>
      </div>
    `);
  }
  if ((selected?.subtitle_tracks?.length || 0) >= 1) {
    // Always offer an explicit "Off" so a single embedded subtitle track is
    // still toggleable (index -1 = off; _chooseTrackIndex(allowNone) honors it).
    // Previously this gated on `> 1`, which hid the control entirely for the
    // common single-subtitle case — you could neither turn it on nor off.
    const subtitleOptions = [{ index: -1, label: 'Off' }, ...(selected.subtitle_tracks || [])];
    const subtitleSelected = Number.isFinite(Number(playback.selected_subtitle_stream_index))
      ? Number(playback.selected_subtitle_stream_index)
      : -1;
    groups.push(`
      <div class="fv-popover-group">
        <div class="fv-popover-label">Subtitles</div>
        <div class="fv-popover-options">
          ${_trackButtons(subtitleOptions, subtitleSelected, 'data-fv-playback-subtitle')}
        </div>
      </div>
    `);
  }
  _popoverEl.innerHTML = groups.join('') || '<div class="fv-popover-empty">No alternate tracks</div>';
  _tracksBtn?.classList.add('is-active');
  _outputsBtn?.classList.remove('is-active');
}

async function _toggleOutputPopover() {
  if (!_popoverEl || !_hasOutputCandidates()) return;
  if (_popoverEl.hidden || _popoverMode !== 'outputs') {
    _popoverMode = 'outputs';
    _renderPreferredOutputPopover();
    _popoverEl.hidden = false;
    _tracksBtn?.classList.remove('is-active');
    _outputsBtn?.classList.add('is-active');
    await _loadOutputOptions();
  } else {
    _closePlaybackPopover();
  }
}

async function _loadOutputOptions() {
  if (!_fileId || !_isMediaProxyUrl(_currentMediaUrl())) {
    _outputState = {
      ..._outputState,
      loading: false,
      loadedForFileId: '',
      serverId: '',
      provider: '',
      supportsProviderRemote: false,
      remoteSessions: [],
      transportReceivers: [],
      error: '',
    };
    if (_popoverMode === 'outputs' && _popoverEl && !_popoverEl.hidden) _renderPreferredOutputPopover();
    return;
  }

  const loadSeq = ++_outputLoadSeq;
  _outputState = {
    ..._outputState,
    loading: true,
    loadedForFileId: _fileId,
    error: '',
  };
  if (_popoverMode === 'outputs' && _popoverEl && !_popoverEl.hidden) _renderPreferredOutputPopover();

  const data = await fetchMediaOutputs(_fileId);
  if (loadSeq !== _outputLoadSeq) return;

  _outputState = {
    loading: false,
    loadedForFileId: _fileId,
    serverId: String(data?.server_id || '').trim(),
    provider: String(data?.provider || '').trim(),
    supportsProviderRemote: !!data?.supports_provider_remote,
    remoteSessions: Array.isArray(data?.remote_sessions) ? data.remote_sessions : [],
    transportReceivers: Array.isArray(data?.transport_receivers) ? data.transport_receivers : [],
    error: data ? '' : 'Could not load device targets',
  };
  if (_popoverMode === 'outputs' && _popoverEl && !_popoverEl.hidden) _renderPreferredOutputPopover();
}

function _renderOutputPopover() {
  if (!_popoverEl) return;

  const groups = [];
  if (_supportsManagedDeviceCast()) {
    groups.push(_outputPopoverGroup(
      'Connected Devices',
      `<div class="fv-popover-stack">
        <button type="button" class="fv-popover-btn fv-popover-btn--row" data-fv-output-managed-cast="1">
          <span class="fv-popover-text">Saved TVs and speakers</span>
          <span class="fv-popover-subtext">Augmentum-managed Cast and DLNA devices</span>
        </button>
      </div>`,
    ));
  }

  const browserButtons = [];
  const remoteLabel = _browserRemoteOutputLabel();
  if (_supportsBrowserRemotePlayback()) {
    browserButtons.push(`
      <button type="button" class="fv-popover-btn fv-popover-btn--row" data-fv-output-remote="1">
        <span class="fv-popover-text">${_esc(remoteLabel)}</span>
        <span class="fv-popover-subtext">Use the browser or OS device picker</span>
      </button>
    `);
  }
  if (_supportsCastOutput()) {
    browserButtons.push(`
      <button type="button" class="fv-popover-btn fv-popover-btn--row" data-fv-output-cast="1">
        <span class="fv-popover-text">Browser Cast</span>
        <span class="fv-popover-subtext">Browser-managed Cast fallback</span>
      </button>
    `);
  }
  if (browserButtons.length) {
    groups.push(`
      <div class="fv-popover-group">
        <div class="fv-popover-label">Nearby Playback</div>
        <div class="fv-popover-stack">
          ${browserButtons.join('')}
        </div>
      </div>
    `);
  }

  if (!_outputState.loading && (_outputState.transportReceivers || []).length) {
    const transportButtons = (_outputState.transportReceivers || []).map((receiver) => {
      const label = String(receiver?.label || receiver?.receiver_id || 'Receiver').trim() || 'Receiver';
      const description = [
        String(receiver?.manufacturer || '').trim(),
        String(receiver?.model_name || '').trim(),
      ].filter(Boolean).join(' • ') || 'Play through Augmentum DLNA control';
      return `
        <button
          type="button"
          class="fv-popover-btn fv-popover-btn--row"
          data-fv-output-transport="${_esc(receiver?.transport_kind || 'dlna')}"
          data-fv-output-receiver-id="${_esc(receiver?.receiver_id || '')}"
          data-fv-output-receiver-profile="${_esc(receiver?.receiver_profile || 'dlna_generic_video')}"
          data-fv-output-label="${_esc(label)}"
        >
          <span class="fv-popover-text">${_esc(label)}</span>
          <span class="fv-popover-subtext">${_esc(description)}</span>
        </button>
      `;
    }).join('');
    groups.push(`
      <div class="fv-popover-group">
        <div class="fv-popover-label">Augmentum Receivers</div>
        <div class="fv-popover-stack">
          ${transportButtons}
        </div>
      </div>
    `);
  }

  if (_outputState.loading) {
    groups.push(`
      <div class="fv-popover-group">
        <div class="fv-popover-label">${_esc(_outputState.provider || 'Server devices')}</div>
        <div class="fv-popover-empty">Loading remote devices…</div>
      </div>
    `);
  } else if (_outputState.supportsProviderRemote || _outputState.remoteSessions.length) {
    const providerLabel = _outputState.provider
      ? `${_outputState.provider[0].toUpperCase()}${_outputState.provider.slice(1)} devices`
      : 'Server devices';
    const sessionButtons = (_outputState.remoteSessions || []).map((session) => {
      const label = String(session?.name || session?.device_name || session?.client || 'Remote device').trim()
        || 'Remote device';
      const nowPlaying = String(session?.now_playing_title || '').trim();
      const description = nowPlaying
        ? `Now playing ${nowPlaying}`
        : (String(session?.client || session?.device_name || session?.user_name || '').trim() || 'Ready to receive playback');
      return `
        <button
          type="button"
          class="fv-popover-btn fv-popover-btn--row"
          data-fv-output-session="${_esc(session?.session_id || '')}"
          data-fv-output-label="${_esc(label)}"
        >
          <span class="fv-popover-text">${_esc(label)}</span>
          <span class="fv-popover-subtext">${_esc(description)}</span>
        </button>
      `;
    }).join('');
    groups.push(`
      <div class="fv-popover-group">
        <div class="fv-popover-label">${_esc(providerLabel)}</div>
        ${sessionButtons
          ? `<div class="fv-popover-stack">${sessionButtons}</div>`
          : '<div class="fv-popover-empty">No compatible server clients are active right now</div>'}
      </div>
    `);
  } else if (_outputState.error) {
    groups.push(`
      <div class="fv-popover-group">
        <div class="fv-popover-empty">${_esc(_outputState.error)}</div>
      </div>
    `);
  }

  _popoverEl.innerHTML = groups.join('') || '<div class="fv-popover-empty">No remote outputs available for this video</div>';
  _tracksBtn?.classList.remove('is-active');
  _outputsBtn?.classList.add('is-active');
}

function _renderPreferredOutputPopover() {
  if (!_popoverEl) return;

  const groups = [];
  const hasProviderTargets = !!(_outputState.supportsProviderRemote || _outputState.remoteSessions.length);
  const hasTransportTargets = !!((_outputState.transportReceivers || []).length);
  const hasManagedTargets = hasProviderTargets || hasTransportTargets || _supportsManagedDeviceCast();

  if (_outputState.loading) {
    groups.push(_outputPopoverGroup(
      _outputState.provider ? `${_providerLabel(_outputState.provider)} devices` : 'Server devices',
      '<div class="fv-popover-empty">Loading remote devices...</div>',
    ));
  } else if (hasProviderTargets) {
    const providerLabel = _outputState.provider
      ? `${_providerLabel(_outputState.provider)} devices`
      : 'Server devices';
    const sessionButtons = [...(_outputState.remoteSessions || [])]
      .sort((left, right) => {
        const scoreDiff = _providerSessionControlScore(right) - _providerSessionControlScore(left);
        if (scoreDiff) return scoreDiff;
        const activeDiff = Number(!!right?.now_playing_title) - Number(!!left?.now_playing_title);
        if (activeDiff) return activeDiff;
        return String(left?.name || '').localeCompare(String(right?.name || ''));
      })
      .map((session) => {
        const label = String(session?.name || session?.device_name || session?.client || 'Remote device').trim()
          || 'Remote device';
        const capability = _providerSessionCapabilityLabel(session);
        const nowPlaying = String(session?.now_playing_title || '').trim();
        const targetState = nowPlaying
          ? `Now playing ${nowPlaying}`
          : (String(session?.client || session?.device_name || session?.user_name || '').trim() || 'Ready to receive playback');
        const description = [capability, targetState].filter(Boolean).join(' | ');
        return `
          <button
            type="button"
            class="fv-popover-btn fv-popover-btn--row"
            data-fv-output-session="${_esc(session?.session_id || '')}"
            data-fv-output-label="${_esc(label)}"
          >
            <span class="fv-popover-text">${_esc(label)}</span>
            <span class="fv-popover-subtext">${_esc(description)}</span>
          </button>
        `;
      }).join('');
    groups.push(_outputPopoverGroup(
      providerLabel,
      sessionButtons
        ? `<div class="fv-popover-stack">${sessionButtons}</div>`
        : '<div class="fv-popover-empty">No compatible server clients are active right now</div>',
      'Best when the target is running a native Emby or Jellyfin client.',
    ));
  }

  if (!_outputState.loading && hasTransportTargets) {
    const transportButtons = [...(_outputState.transportReceivers || [])]
      .sort((left, right) => String(left?.label || '').localeCompare(String(right?.label || '')))
      .map((receiver) => {
        const label = String(receiver?.label || receiver?.receiver_id || 'Receiver').trim() || 'Receiver';
        const description = [
          String(receiver?.manufacturer || '').trim(),
          String(receiver?.model_name || '').trim(),
        ].filter(Boolean).join(' | ') || 'Augmentum-managed DLNA playback';
        return `
          <button
            type="button"
            class="fv-popover-btn fv-popover-btn--row"
            data-fv-output-transport="${_esc(receiver?.transport_kind || 'dlna')}"
            data-fv-output-receiver-id="${_esc(receiver?.receiver_id || '')}"
            data-fv-output-receiver-profile="${_esc(receiver?.receiver_profile || 'dlna_generic_video')}"
            data-fv-output-label="${_esc(label)}"
          >
            <span class="fv-popover-text">${_esc(label)}</span>
            <span class="fv-popover-subtext">${_esc(description)}</span>
          </button>
        `;
      }).join('');
    groups.push(_outputPopoverGroup(
      'Augmentum Receivers',
      `<div class="fv-popover-stack">${transportButtons}</div>`,
      'Augmentum launches and manages playback directly for these receivers.',
    ));
  } else if (!_outputState.loading && _outputState.error) {
    groups.push(_outputPopoverGroup(
      'Receiver status',
      `<div class="fv-popover-empty">${_esc(_outputState.error)}</div>`,
    ));
  }

  if (_supportsManagedDeviceCast()) {
    groups.push(_outputPopoverGroup(
      'Connected Devices',
      `<div class="fv-popover-stack">
        <button type="button" class="fv-popover-btn fv-popover-btn--row" data-fv-output-managed-cast="1">
          <span class="fv-popover-text">Saved TVs and speakers</span>
          <span class="fv-popover-subtext">Augmentum-managed Cast and DLNA devices</span>
        </button>
      </div>`,
    ));
  }

  const browserButtons = [];
  const remoteLabel = _browserRemoteOutputLabel();
  if (_supportsBrowserRemotePlayback()) {
    browserButtons.push(`
      <button type="button" class="fv-popover-btn fv-popover-btn--row" data-fv-output-remote="1">
        <span class="fv-popover-text">${_esc(remoteLabel)}</span>
        <span class="fv-popover-subtext">Browser or OS device picker | best-effort discovery</span>
      </button>
    `);
  }
  if (_supportsCastOutput()) {
    browserButtons.push(`
      <button type="button" class="fv-popover-btn fv-popover-btn--row" data-fv-output-cast="1">
        <span class="fv-popover-text">Browser Cast</span>
        <span class="fv-popover-subtext">Browser-managed Cast fallback</span>
      </button>
    `);
  }
  if (browserButtons.length) {
    groups.push(_outputPopoverGroup(
      'Browser / OS Outputs',
      `<div class="fv-popover-stack">${browserButtons.join('')}</div>`,
      hasManagedTargets
        ? 'These options use browser discovery, which can fail even when the groups above already see your TV.'
        : 'Only browser-managed output discovery is available right now.',
    ));
  }

  _popoverEl.innerHTML = groups.join('') || '<div class="fv-popover-empty">No remote outputs available for this video</div>';
  _tracksBtn?.classList.remove('is-active');
  _outputsBtn?.classList.add('is-active');
}

function _showToast(message, type = 'info', duration = 2800) {
  try {
    if (typeof window !== 'undefined' && typeof window.showToast === 'function') {
      window.showToast(message, type, duration);
      return;
    }
  } catch {
    // Ignore toast-layer failures.
  }
  console.debug(`[floating-video] ${message}`);
}

function _describeRemotePlaybackError(err) {
  const name = String(err?.name || '').trim();
  if (name === 'AbortError') return 'No nearby playback device was selected';
  if (name === 'NotFoundError') return 'No nearby playback device is available for this video';
  if (name === 'NotSupportedError') return 'This browser cannot send this video to a nearby device';
  if (name === 'InvalidAccessError') return 'Tap again after interacting with the player';
  if (name === 'InvalidStateError') return 'Nearby playback is disabled for this video';
  if (name === 'NotAllowedError') return 'Nearby playback permission was denied';
  if (name === 'OperationError') return 'A nearby-device request is already in progress';
  return '';
}

function _describeCastError(err) {
  const rawMessage = String(err?.message || err || '').trim();
  const lower = rawMessage.toLowerCase();
  if (!rawMessage) return 'Could not start Google Cast for this video';
  if (lower.includes('failed to load cast sdk')) {
    return 'Google Cast SDK could not load. Check HTTPS, certificate trust, and browser network filtering';
  }
  if (lower.includes('ssl certificate')) {
    return 'Google Cast SDK was blocked by an SSL certificate error';
  }
  if (lower.includes('session_error')) {
    return 'Google Cast session could not start. Check certificate trust and Cast availability in this browser';
  }
  if (lower.includes('cast sdk is unavailable')) {
    return 'Google Cast is unavailable in this browser session';
  }
  if (lower.includes('no cast session')) {
    return 'No Google Cast device session was established';
  }
  return `Google Cast failed: ${rawMessage}`;
}

function _describeBrowserRemotePlaybackError(err) {
  const name = String(err?.name || '').trim();
  if (name === 'AbortError') return 'No nearby playback device was selected';
  if (name === 'NotFoundError') return 'This browser did not find any nearby playback devices for this video';
  if (name === 'NotSupportedError') return 'This browser cannot send this video to a nearby device';
  if (name === 'InvalidAccessError') return 'Tap again after interacting with the player';
  if (name === 'InvalidStateError') return 'Nearby playback is disabled for this video';
  if (name === 'NotAllowedError') return 'Nearby playback permission was denied';
  if (name === 'OperationError') return 'A nearby-device request is already in progress';
  return '';
}

function _describeBrowserCastError(err) {
  const rawMessage = String(err?.message || err || '').trim();
  const lower = rawMessage.toLowerCase();
  if (!rawMessage) return 'Could not start Google Cast for this video';
  if (lower.includes('failed to load cast sdk')) {
    return 'Google Cast SDK could not load. Check HTTPS, certificate trust, and browser network filtering';
  }
  if (lower.includes('ssl certificate')) {
    return 'Google Cast SDK was blocked by an SSL certificate error';
  }
  if (lower.includes('session_error')) {
    return 'Browser Cast could not start. Use Connected Devices for app-managed TV control';
  }
  if (lower.includes('cast sdk is unavailable')) {
    return 'Google Cast is unavailable in this browser session';
  }
  if (lower.includes('no cast session')) {
    return 'No Google Cast device session was established';
  }
  return `Google Cast failed: ${rawMessage}`;
}

async function _startProviderRemoteSession(sessionId, label) {
  if (!_fileId || !sessionId) return;
  const targetLabel = String(label || 'device').trim() || 'device';
  const startedAt = _currentPlaybackTimeS();
  const targetSession = (_outputState.remoteSessions || []).find((session) => (
    String(session?.session_id || '') === String(sessionId)
  )) || null;
  try {
    const result = await playMediaOnRemoteSession(_fileId, {
      session_id: sessionId,
      start_time_s: startedAt,
      play_command: 'PlayNow',
    });
    if (!result?.status || result.status !== 'ok') {
      _showToast(`Could not start playback on ${targetLabel}`, 'error');
      return;
    }
    const serverId = String(result.server_id || _outputState.serverId || '').trim();
    const confirmed = await _waitForProviderRemoteStart({
      serverId,
      sessionId: String(result.session_id || sessionId).trim(),
      expectedFileId: _fileId,
      expectedExternalId: String(result.external_id || '').trim(),
    });
    const didStartExpectedItem = !!(
      confirmed?.session
      && (
        String(confirmed.current_file_id || '').trim() === _fileId
        || (
          String(result.external_id || '').trim()
          && String(confirmed.session?.now_playing_item_id || '').trim() === String(result.external_id || '').trim()
        )
      )
    );
    if (!didStartExpectedItem) {
      _showToast(`The server client stayed on its existing playback. ${targetLabel} did not accept this handoff.`, 'warning');
      return;
    }
    void _pushProgress({ force: true });
    _closePlaybackPopover();
    _startRemoteController({
      provider: String(result.provider || _outputState.provider || '').trim(),
      serverId,
      sessionId: String(result.session_id || sessionId).trim(),
      targetLabel,
      currentFileId: _fileId,
      title: _metadata.title || 'Remote playback',
      thumbnail: _metadata.thumbnail || '',
      sessionSeed: confirmed?.session || targetSession,
    });
    _showToast(`Playing on ${targetLabel}`, 'success');
  } catch (err) {
    console.warn('[floating-video] provider remote play failed:', err);
    _showToast(`Could not start playback on ${targetLabel}`, 'error');
  }
}

async function _startTransportReceiver({
  transport = '',
  receiverId = '',
  receiverProfile = '',
  label = '',
} = {}) {
  if (!_fileId || !transport || !receiverId) return;
  const targetLabel = String(label || 'receiver').trim() || 'receiver';
  try {
    const result = await playMediaOnTransportReceiver(_fileId, {
      transport,
      receiver_id: receiverId,
      receiver_profile: receiverProfile || 'dlna_generic_video',
    });
    if (!result?.status || result.status !== 'ok' || !result.session) {
      _showToast(`Could not start playback on ${targetLabel}`, 'error');
      return;
    }
    void _pushProgress({ force: true });
    _closePlaybackPopover();
    _startTransportController(result);
    _showToast(`Playing on ${targetLabel}`, 'success');
  } catch (err) {
    console.warn('[floating-video] transport play failed:', err);
    _showToast(`Could not start playback on ${targetLabel}`, 'error');
  }
}

async function _startBrowserRemotePlayback() {
  const video = _videoNodeForNativePiP();
  if (!video) return;
  try {
    if (video.remote && typeof video.remote.prompt === 'function') {
      await video.remote.prompt();
      _closePlaybackPopover();
      _showToast('Connected to nearby playback device', 'success');
      return;
    }
    if (typeof video.webkitShowPlaybackTargetPicker === 'function') {
      video.webkitShowPlaybackTargetPicker();
      _closePlaybackPopover();
      _showToast('AirPlay device picker opened', 'info');
      return;
    }
  } catch (err) {
    const message = _describeBrowserRemotePlaybackError(err);
    console.warn('[floating-video] remote playback prompt failed:', err);
    if (message) {
      _showToast(message, err?.name === 'AbortError' ? 'info' : 'warning');
      return;
    }
  }
  _showToast('No browser-managed playback target is available for this video', 'info');
}

function _guessContentType(url, node = _adoptedIframe) {
  const directType = String(
    node?.getAttribute?.('type')
    || node?.querySelector?.('source[type]')?.getAttribute?.('type')
    || '',
  ).trim();
  if (directType) return directType;
  const path = String(url?.pathname || '').toLowerCase();
  if (path.endsWith('.m3u8')) return 'application/vnd.apple.mpegurl';
  if (path.endsWith('.webm')) return 'video/webm';
  if (path.endsWith('.mkv')) return 'video/x-matroska';
  if (path.endsWith('.mov')) return 'video/quicktime';
  return 'video/mp4';
}

async function _waitForProviderRemoteStart({
  serverId = '',
  sessionId = '',
  expectedFileId = '',
  expectedExternalId = '',
  timeoutMs = 9000,
  intervalMs = 750,
} = {}) {
  const safeServerId = String(serverId || '').trim();
  const safeSessionId = String(sessionId || '').trim();
  if (!safeServerId || !safeSessionId) return null;
  const deadline = Date.now() + Math.max(1000, Number(timeoutMs) || 0);
  let lastSnapshot = null;
  while (Date.now() < deadline) {
    const snapshot = await fetchMediaRemoteSession(safeServerId, safeSessionId);
    if (snapshot?.session) {
      lastSnapshot = snapshot;
      const currentFileId = String(snapshot.current_file_id || '').trim();
      const currentExternalId = String(snapshot.session?.now_playing_item_id || '').trim();
      if (
        (expectedFileId && currentFileId === expectedFileId)
        || (expectedExternalId && currentExternalId === expectedExternalId)
      ) {
        return snapshot;
      }
    }
    await new Promise((resolve) => {
      window.setTimeout(resolve, Math.max(150, Number(intervalMs) || 0));
    });
  }
  return lastSnapshot;
}

function _buildDirectCastSpec() {
  const url = _currentMediaUrl();
  if (!_canCastFromDirectElementSource(url)) return null;
  return {
    supported: true,
    content_url: url.href,
    content_type: _guessContentType(url),
    title: _metadata.title || 'Video',
    poster_url: _metadata.thumbnail || '',
  };
}

function _managedCastContent() {
  const url = _currentMediaUrl();
  if (!_supportsManagedDeviceCast() || !url) return null;
  return {
    contentUrl: url.href,
    contentType: _guessContentType(url),
    title: _metadata.title || 'Video',
    posterUrl: _metadata.thumbnail || '',
    startTimeS: _currentPlaybackTimeS(),
    fileId: _fileId || '',
    contentKey: _fileId || url.href,
    metadata: {
      channel: _metadata.channel || '',
      provider: _outputState.provider || '',
      serverId: _outputState.serverId || '',
    },
  };
}

async function _openManagedCastPicker() {
  const content = _managedCastContent();
  if (!content) {
    _showToast('This video is not ready for Connected Devices casting', 'info');
    return;
  }
  _closePlaybackPopover();
  openCastPicker({
    anchor: _outputsBtn || _root,
    capability: 'media.video_play@1',
    content,
    onCast: () => {
      // Hand off cleanly: the TV is now the active player, so tear
      // down the local floating-video shell entirely. Leaving it open
      // (even paused) ends up as a frozen card the user has to dismiss
      // manually. The cast-shelf pill takes over as the controller —
      // ask it to open immediately so the user lands on the new
      // transport without a moment of "where did my player go?".
      try {
        close();
      } catch (err) {
        console.warn('[floating-video] handoff teardown failed', err);
        // Fall back to at least pausing if close() throws so we don't
        // leave a frozen-but-loud player behind.
        try { _progressState?.mediaEl?.pause?.(); } catch {}
      }
      import('./cast-shelf.js')
        .then(m => m.notifyCastStarted?.({ openShelf: true }))
        .catch(() => {});
    },
  });
}

async function _fetchCastSpec() {
  const directSpec = _buildDirectCastSpec();
  if (directSpec) return directSpec;
  if (!_fileId || !_isMediaProxyUrl(_currentMediaUrl())) {
    return {
      supported: false,
      reason: 'Cast requires a receiver-reachable stream for this video',
    };
  }
  const payload = await fetchMediaCastLoad(_fileId);
  if (!payload) {
    return {
      supported: false,
      reason: 'Could not prepare this video for Google Cast',
    };
  }
  return payload;
}

async function _loadCastSenderSdk() {
  if (typeof window === 'undefined') {
    throw new Error('Cast is only available in a browser');
  }
  if (window.cast?.framework && window.chrome?.cast) {
    _configureCastContext();
    return window.cast.framework;
  }
  if (_castSdkPromise) return _castSdkPromise;

  _castSdkPromise = new Promise((resolve, reject) => {
    let settled = false;
    const previous = window.__onGCastApiAvailable;

    const cleanup = () => {
      if (window.__onGCastApiAvailable === onAvailable) {
        window.__onGCastApiAvailable = previous;
      }
    };

    const finish = (err = null) => {
      if (settled) return;
      settled = true;
      cleanup();
      if (err) {
        _castSdkPromise = null;
        reject(err);
        return;
      }
      try {
        _configureCastContext();
        resolve(window.cast.framework);
      } catch (configErr) {
        _castSdkPromise = null;
        reject(configErr);
      }
    };

    function onAvailable(available, ...rest) {
      try {
        if (typeof previous === 'function') previous(available, ...rest);
      } catch {
        // Ignore prior callback failures.
      }
      if (available) finish();
      else finish(new Error('Cast SDK is unavailable'));
    }

    window.__onGCastApiAvailable = onAvailable;

    let script = document.querySelector('script[data-augmentum-cast-sdk="1"]');
    if (!script) {
      script = document.createElement('script');
      script.src = 'https://www.gstatic.com/cv/js/sender/v1/cast_sender.js?loadCastFramework=1';
      script.async = true;
      script.defer = true;
      script.dataset.augmentumCastSdk = '1';
      script.addEventListener('error', () => finish(new Error('Failed to load Cast SDK')));
      document.head.appendChild(script);
    }

    window.setTimeout(() => {
      if (window.cast?.framework && window.chrome?.cast) finish();
    }, 0);
  });

  return _castSdkPromise;
}

function _configureCastContext() {
  if (_castConfigured || !window.cast?.framework?.CastContext || !window.chrome?.cast) return;
  const context = window.cast.framework.CastContext.getInstance();
  context.setOptions({
    receiverApplicationId: window.chrome.cast.media.DEFAULT_MEDIA_RECEIVER_APP_ID || 'CC1AD845',
    autoJoinPolicy: window.chrome.cast.AutoJoinPolicy?.ORIGIN_SCOPED,
  });
  _castConfigured = true;
}

async function _getCastSession() {
  await _loadCastSenderSdk();
  const context = window.cast.framework.CastContext.getInstance();
  let session = context.getCurrentSession();
  if (session) return session;
  await context.requestSession();
  session = context.getCurrentSession();
  if (!session) throw new Error('No Cast session was started');
  return session;
}

function _castSessionObjects() {
  const context = window.cast?.framework?.CastContext?.getInstance?.();
  const session = context?.getCurrentSession?.() || null;
  const sessionObj = session?.getSessionObj?.() || null;
  const media = session?.getMediaSession?.() || (Array.isArray(sessionObj?.media) ? sessionObj.media[0] : null);
  return { session, sessionObj, media };
}

function _castSnapshotData(seed = {}) {
  const { session, sessionObj, media } = _castSessionObjects();
  if (!session) return null;
  const receiverName = String(
    seed.deviceName
    || sessionObj?.receiver?.friendlyName
    || sessionObj?.receiver?.label
    || 'Cast device',
  ).trim();
  const volume = sessionObj?.receiver?.volume || {};
  const durationS = Math.max(0, Number(
    media?.media?.duration
    || media?.mediaSession?.media?.duration
    || seed.durationS
    || 0,
  ));
  let currentTimeS = Number(
    media?.getEstimatedTime?.()
    || media?.currentTime
    || media?.media?.currentTime
    || seed.currentTimeS
    || 0,
  );
  if (!Number.isFinite(currentTimeS)) currentTimeS = 0;
  const playerState = String(
    media?.playerState
    || media?.mediaSession?.playerState
    || sessionObj?.media?.[0]?.playerState
    || '',
  ).trim().toUpperCase();
  const idleReason = String(
    media?.idleReason
    || media?.mediaSession?.idleReason
    || sessionObj?.media?.[0]?.idleReason
    || '',
  ).trim().toUpperCase();
  return {
    status: 'ok',
    transport: 'cast',
    session: {
      session_id: String(sessionObj?.sessionId || 'cast').trim() || 'cast',
      transport_kind: 'cast',
      receiver_label: receiverName,
      receiver_id: String(sessionObj?.receiver?.label || receiverName).trim(),
      receiver_profile: 'cast_video',
      provider: String(seed.provider || _remoteSession?.provider || '').trim(),
      server_id: String(seed.serverId || _remoteSession?.serverId || '').trim(),
      file_id: String(seed.fileId || _fileId || '').trim(),
      external_id: String(seed.externalId || _remoteSession?.currentExternalId || '').trim(),
      title: String(seed.title || _metadata.title || 'Video').trim() || 'Video',
      thumbnail: String(seed.thumbnail || _metadata.thumbnail || '').trim(),
      current_time_s: Math.max(0, currentTimeS),
      duration_s: durationS,
      is_paused: playerState === 'PAUSED',
      is_muted: !!volume.muted,
      can_seek: true,
      volume_level: Number.isFinite(Number(volume.level)) ? Math.round(Number(volume.level) * 100) : null,
      supported_commands: ['PlayPause', 'Pause', 'Unpause', 'Stop', 'Seek', 'SetVolume', 'Mute', 'Unmute', 'ToggleMute'],
      receiver_state: idleReason === 'FINISHED' ? 'FINISHED' : playerState,
    },
  };
}

async function _pollCastRemoteSession() {
  await _loadCastSenderSdk();
  const snapshot = _castSnapshotData();
  if (!snapshot?.session) return null;
  if (String(snapshot.session.receiver_state || '').toUpperCase() === 'FINISHED') {
    await _pushRemoteTransportProgress(_remoteSession, { force: true, isFinished: true });
    if (_remoteSession) _remoteSession.stopOnClose = false;
    close();
    return null;
  }
  return snapshot;
}

async function _sendCastPlaystate(command, { seekPositionS = null } = {}) {
  const { session, media } = _castSessionObjects();
  if (!session || !media) return null;
  const state = String(media?.playerState || '').trim().toUpperCase();
  if (command === 'PlayPause') {
    if (state === 'PAUSED' && typeof media.play === 'function') await media.play();
    else if (typeof media.pause === 'function') await media.pause();
    return { status: 'ok' };
  }
  if (command === 'Pause' && typeof media.pause === 'function') {
    await media.pause();
    return { status: 'ok' };
  }
  if (command === 'Unpause' && typeof media.play === 'function') {
    await media.play();
    return { status: 'ok' };
  }
  if (command === 'Stop' && typeof media.stop === 'function') {
    await media.stop();
    return { status: 'ok' };
  }
  if (command === 'Seek' && Number.isFinite(Number(seekPositionS)) && typeof media.seek === 'function') {
    const request = new window.chrome.cast.media.SeekRequest();
    request.currentTime = Math.max(0, Number(seekPositionS));
    await media.seek(request);
    return { status: 'ok' };
  }
  return null;
}

async function _sendCastGeneral(command, args = null) {
  const { session, sessionObj } = _castSessionObjects();
  if (!session) return null;
  const currentLevel = Number(sessionObj?.receiver?.volume?.level || 0);
  const currentMuted = !!sessionObj?.receiver?.volume?.muted;
  if (command === 'SetVolume') {
    const next = Math.max(0, Math.min(1, Number(args?.Volume || 0) / 100));
    if (typeof session.setVolume === 'function') await session.setVolume(next);
    else return null;
    return { status: 'ok' };
  }
  if (command === 'VolumeUp' || command === 'VolumeDown') {
    const delta = command === 'VolumeUp' ? 0.05 : -0.05;
    const next = Math.max(0, Math.min(1, currentLevel + delta));
    if (typeof session.setVolume === 'function') await session.setVolume(next);
    else return null;
    return { status: 'ok' };
  }
  if (command === 'Mute' || command === 'Unmute' || command === 'ToggleMute') {
    const nextMuted = command === 'ToggleMute' ? !currentMuted : command === 'Mute';
    if (typeof session.setMute === 'function') await session.setMute(nextMuted);
    else return null;
    return { status: 'ok' };
  }
  return null;
}

function _startCastRemoteController(castSpec = {}) {
  const snapshot = _castSnapshotData({
    fileId: _fileId,
    title: castSpec.title || _metadata.title || 'Video',
    thumbnail: castSpec.poster_url || _metadata.thumbnail || '',
    provider: _outputState.provider || _remoteSession?.provider || '',
    serverId: _outputState.serverId || _remoteSession?.serverId || '',
  });
  if (!snapshot?.session) return;
  if (!_ensureDom()) return;
  _teardownPiP();
  _teardownProgress();
  _busHandle?.release();
  _releaseAdopted();
  _root.hidden = false;
  _root.classList.add('open');
  _applyTransportSessionSnapshot(snapshot, { sourceType: 'local_transport' });
  openSession({
    shellMode: 'collapsed',
    supportsNativePiP: false,
    isNativePiPActive: false,
    fileId: _fileId || null,
    videoId: null,
    title: _metadata.title || '',
    channel: _metadata.channel || '',
    thumbnail: _metadata.thumbnail || '',
    nextItem: null,
    hasPlaybackOptions: false,
    remoteSessionActive: true,
    remoteSourceType: _remoteSourceType(_remoteSession),
    remoteTransportKind: _remoteTransportKind(_remoteSession),
    remoteProvider: _remoteSession?.provider || '',
    remoteServerId: _remoteSession?.serverId || '',
    remoteSessionId: _remoteSession?.sessionId || '',
    remoteDeviceName: _remoteSession?.deviceName || '',
    remoteSupportedCommands: _remoteSession?.supportedCommands || [],
    isMuted: !!_remoteSession?.isMuted,
    volumeLevel: _remoteSession?.volumeLevel ?? null,
    canSeek: !!_remoteSession?.canSeek,
  });
  _syncUi();
  setMode('collapsed');
  _emitState();
  _scheduleRemotePoll(750);
}

async function _startCastSender() {
  if (typeof window === 'undefined' || !window.isSecureContext) {
    _showToast('Google Cast needs a secure HTTPS session', 'warning');
    return;
  }
  const castSpec = await _fetchCastSpec();
  if (!castSpec?.supported) {
    _showToast(castSpec?.reason || 'Google Cast is unavailable for this video', 'info');
    return;
  }

  try {
    const session = await _getCastSession();
    const mediaInfo = new window.chrome.cast.media.MediaInfo(
      castSpec.content_url,
      castSpec.content_type || 'video/mp4',
    );
    mediaInfo.streamType = window.chrome.cast.media.StreamType.BUFFERED;

    const metadata = new window.chrome.cast.media.GenericMediaMetadata();
    metadata.title = castSpec.title || _metadata.title || 'Video';
    if (castSpec.poster_url || _metadata.thumbnail) {
      metadata.images = [{
        url: castSpec.poster_url || _metadata.thumbnail,
      }];
    }
    mediaInfo.metadata = metadata;

    const request = new window.chrome.cast.media.LoadRequest(mediaInfo);
    request.autoplay = true;
    request.currentTime = _currentPlaybackTimeS();
    await session.loadMedia(request);

    _closePlaybackPopover();
    try {
      _progressState?.mediaEl?.pause?.();
    } catch {
      // Ignore local pause failures.
    }
    _startCastRemoteController(castSpec);
    _showToast(`Casting ${castSpec.title || _metadata.title || 'video'}`, 'success');
  } catch (err) {
    console.warn('[floating-video] cast start failed:', err);
    _showToast(_describeBrowserCastError(err), 'error', 4200);
  }
}

async function _selectPlayback(kind, value) {
  if (_playbackBusy || !_playbackMenu) return;
  _playbackBusy = true;
  try {
    if (kind === 'mediaSource' && typeof _playbackMenu.selectMediaSource === 'function') {
      await _playbackMenu.selectMediaSource(value);
    } else if (kind === 'audio' && typeof _playbackMenu.selectAudioStream === 'function') {
      await _playbackMenu.selectAudioStream(value);
    } else if (kind === 'subtitle' && typeof _playbackMenu.selectSubtitleStream === 'function') {
      await _playbackMenu.selectSubtitleStream(value);
    }
  } catch (err) {
    console.warn('[floating-video] playback selection failed:', err);
  } finally {
    _playbackBusy = false;
    if (!_popoverEl?.hidden) _renderPlaybackPopover();
    _syncUi();
  }
}

function _releaseAdopted() {
  if (!_adoptedIframe) return;
  _adoptedIframe.classList.remove('fv-adopted');
  if (typeof HTMLMediaElement !== 'undefined' && _adoptedIframe instanceof HTMLMediaElement) {
    try {
      _adoptedIframe.pause();
    } catch {
      // Ignore pause failures.
    }
  }
  // api-driven sources (YouTube) get a destroy() hook so the player
  // tears its own iframe down cleanly instead of leaving an orphaned
  // <iframe> we yanked from under it.
  if (_api && typeof _api.destroy === 'function') {
    try { _api.destroy(); } catch { /* ignore destroy failures */ }
  } else {
    try {
      _adoptedIframe.remove();
    } catch {
      // Ignore DOM removal failures.
    }
  }
  _adoptedIframe = null;
  _api = null;
}

function _elementApi(node) {
  if (typeof HTMLMediaElement === 'undefined' || !(node instanceof HTMLMediaElement)) {
    return null;
  }
  return {
    setVolume: (value) => {
      node.volume = Math.max(0, Math.min(1, value));
    },
    getVolume: () => node.volume,
    pause: () => node.pause(),
    play: () => node.play(),
  };
}

export const FloatingVideo = { open, setMode, close, isOpen, getVideoId, toggleNativePiP };
export default FloatingVideo;

if (typeof window !== 'undefined') window.FloatingVideo = FloatingVideo;
