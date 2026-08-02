/**
 * Files panel — full-screen preview overlays, image gallery, and the
 * zip/html project handoff that lets files-mode open app-builder artifacts
 * in the library workspace.
 */

import { escapeHtml, showToast } from '../app.js';
import { state } from './state.js';
import {
  isImage, isAppProject, humanSize, formatDate, videoLikelyUnsupported, isMediaServerFile,
} from './helpers.js';
import {
  downloadUrl, renderUrl, deleteOne, mediaStreamUrl, mediaCoverUrl, mediaSubtitleUrl,
  pushMediaProgress, fetchFileEntry, fetchMediaDetails, updateMediaPlaybackSelection,
} from './api.js';

// Forward imports — these live in actions/render modules. ES module hoisting
// handles the circular edge because we only dereference at call time.
import {
  deleteOneUi, referenceInChat, inlineConfirm, downloadFile,
} from './actions.js';
import { renderGrid, updateDetail, updateSelectionUI } from './render.js';
import { effectiveDuration } from '../media-duration.js';

let _activeFloatingVideoSession = null;
let _floatingSelectionListenerWired = false;
const _BROWSER_UNSUPPORTED_AUDIO_CODECS = new Set([
  'ac3',
  'eac3',
  'dca',
  'dts',
  'dtshd_hra',
  'dtshd_ma',
  'truehd',
]);

// --- Zip/HTML project handoff ----------------------------------------
//
// Mirror library.js's activation path exactly so "open from Files" and
// "open from Library" are indistinguishable: PATCH /open for recency
// tracking, then delegate to workspace.openWorkspace() in play mode.
// openWorkspace auto-hydrates source_json from /api/artifacts/:id if we
// only pass an id, so no pre-fetch needed.

export async function openProject(file) {
  const artifactId = file.source_id;
  if (!artifactId) {
    console.warn('[files] openProject: missing source_id', file);
    return downloadFile(file.id);
  }
  // Fire-and-forget recency ping — same call library.js makes.
  fetch(`/api/artifacts/${encodeURIComponent(artifactId)}/open`, { method: 'PATCH' }).catch(() => {});
  try {
    // Workspace is a top-level overlay now (reparented to <body> on first
    // openWorkspace() since the library-shell refactor). Open it
    // immediately, then mount the Library underneath so the workspace's
    // Back button lands on a real Library surface rather than the chat
    // backdrop.
    const mod = await import('../workspace.js');
    await mod.openWorkspace({ id: artifactId }, 'play');
    import('../library.js').then(lib => lib.openLibrary()).catch(() => {});
  } catch (err) {
    console.warn('[files] project open failed:', err);
    showToast(`Failed to open project: ${err.message || 'unknown error'}`, 'error');
    downloadFile(file.id);
  }
}

// --- Full-screen media preview (PDF / video / HTML / rendered / audio) -

export function openMediaPreview(fileId, kind) {
  const file = _resolvePreviewFile(fileId);
  if (!file) return;
  const streamableMedia = (kind === 'video' || kind === 'audio') && isMediaServerFile(file);
  const url = streamableMedia
    ? (
      kind === 'video'
        ? _selectedVideoStreamUrl(file)
        : mediaStreamUrl(file.id, { episodeId: file.source_metadata?.selected_episode_id || '' })
    )
    : downloadUrl(file.id);
  const render = renderUrl(file.id);
  if (kind === 'video' && !videoLikelyUnsupported(file)) {
    const entityKind = String(file?.source_metadata?.entity_kind || '').toLowerCase();
    if (entityKind === 'music_video') {
      void _openMusicVideoInGrove(file);
      return;
    }
    _openFloatingVideoPlayer(file);
    return;
  }

  if (kind === 'audio' || kind === 'video') {
    // Companion presence: opened-for-playback counts as attention even
    // before the user hits play (no autoplay — gesture required).
    import('../architect-observer.js')
      .then(m => m.reportAttention('surface.media.playback_started', {
        label: file?.display_name || file?.name || '',
        kind,
        ref: String(file?.id || ''),
      }))
      .catch(() => {});
  }

  const overlay = document.createElement('div');
  overlay.className = 'files-preview-overlay';
  overlay.dataset.kind = kind;
  overlay.tabIndex = 0;
  document.body.appendChild(overlay);
  requestAnimationFrame(() => overlay.classList.add('visible'));
  overlay.focus();

  let stage = '';
  if (kind === 'pdf') {
    stage = `<iframe class="files-preview-iframe-full" src="${escapeHtml(render + '#view=FitH')}" title="${escapeHtml(file.name)}"></iframe>`;
  } else if (kind === 'html') {
    // Sandbox policy: scripts ON, same-origin OFF by default. An HTML
    // file uploaded by the user (or produced by a tool / downloaded
    // from chat) is served from Augmentum's origin, so pairing
    // allow-scripts with allow-same-origin would let the file's JS
    // call /api/auth/keys with the user's session cookies — a
    // classic confused-deputy attack on uploaded content.
    //
    // The trade-off: HTML exports that touch localStorage, module
    // imports, or location.origin on bootstrap will see opaque-origin
    // failures (the page still renders; just those specific APIs
    // throw). Per-file workaround: set the localStorage key
    // ``files.preview.trustSameOrigin.<file-id>`` to ``"1"`` to opt a
    // file the user explicitly trusts back into the loose sandbox,
    // and reload. Mirrors the coder preview pattern.
    let _trustSameOrigin = false;
    try {
      _trustSameOrigin = localStorage.getItem(`files.preview.trustSameOrigin.${file.id}`) === '1';
    } catch { /* unreadable storage — fall back to strict sandbox */ }
    const _sandbox = _trustSameOrigin
      ? 'allow-scripts allow-same-origin'
      : 'allow-scripts';
    stage = `<iframe class="files-preview-iframe-full" src="${escapeHtml(render)}" sandbox="${_sandbox}" title="${escapeHtml(file.name)}"></iframe>`;
  } else if (kind === 'video') {
    if (videoLikelyUnsupported(file)) {
      stage = `<div class="files-preview-unsupported">
        <p><strong>${escapeHtml(file.name)}</strong></p>
        <p style="opacity:0.7;margin:8px 0 16px">This video format isn't supported by your browser.</p>
        <a href="${escapeHtml(url)}" download class="btn btn-sm">Download to play</a>
      </div>`;
    } else {
      // No autoplay — browsers block it without a user gesture and show
      // a confusing flash; the controls are right there for the user.
      stage = `<video class="files-preview-video-full" src="${escapeHtml(url)}" controls playsinline></video>`;
    }
  } else if (kind === 'rendered') {
    stage = `<iframe class="files-preview-iframe-full" src="${escapeHtml(render)}" sandbox="allow-same-origin" title="${escapeHtml(file.name)}"></iframe>`;
  } else if (kind === 'audio') {
    // Drop autoplay — browser autoplay policies block it without a user
    // gesture, leaving an awkward muted state. User clicks play.
    stage = `<div class="files-preview-audio-full">
      <div class="files-preview-audio-badge">
        <svg width="72" height="72" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
      </div>
      <audio src="${escapeHtml(url)}" controls></audio>
    </div>`;
  } else {
    stage = `<div class="files-preview-unsupported">
      <p>Preview not available for this file type.</p>
      <a href="${escapeHtml(url)}" download class="btn btn-sm">Download</a>
    </div>`;
  }

  const playlistBtn = (kind === 'audio' || kind === 'video')
    ? `<button class="files-preview-chrome-btn" data-action="playlist" title="Add to Grove playlist" aria-label="Add to playlist">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="13" y2="18"/><line x1="18" y1="15" x2="18" y2="21"/><line x1="15" y1="18" x2="21" y2="18"/></svg>
        </button>`
    : '';

  overlay.innerHTML = `
    <div class="files-preview-chrome">
      <div class="files-preview-chrome-title">${escapeHtml(file.name)}</div>
      <div class="files-preview-chrome-actions">
        ${playlistBtn}
        <button class="files-preview-chrome-btn" data-action="download" title="Download">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        </button>
        <button class="files-preview-chrome-btn" data-action="reference" title="Reference in chat">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
        </button>
        <button class="files-preview-chrome-btn files-preview-close" data-action="close" title="Close (Esc)" aria-label="Close">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
        </button>
      </div>
    </div>
    <div class="files-preview-stage" data-kind="${escapeHtml(kind)}">${stage}</div>
  `;

  let _readerControls = null;
  const close = () => {
    if (_readerControls) { try { _readerControls.destroy(); } catch { /* noop */ } _readerControls = null; }
    _flushPreviewProgress(overlay, file, kind, { force: true });
    overlay.classList.remove('visible');
    setTimeout(() => overlay.remove(), 200);
  };
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay || e.target.classList.contains('files-preview-stage')) close();
  });
  overlay.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { e.preventDefault(); close(); }
    e.stopPropagation();
  });
  overlay.querySelectorAll('[data-action]').forEach(b => {
    b.addEventListener('click', (e) => {
      e.stopPropagation();
      const act = b.dataset.action;
      if (act === 'download') downloadFile(file.id);
      else if (act === 'reference') { close(); referenceInChat(file.id, file.name); }
      else if (act === 'playlist') {
        window.dispatchEvent(new CustomEvent('playlist:add-item', {
          detail: {
            type: 'file',
            fileId: file.id,
            name: file.name || '',
            kind: kind === 'video' ? 'video' : 'audio',
            thumbnail: '',
          },
        }));
      }
      else if (act === 'close') close();
    });
  });
  // EPUB read-aloud — add a Read-aloud button + voice/speed pickers to the
  // overlay chrome, just before Download. Lives in the parent DOM (the
  // preview iframe is sandboxed without scripts), driven by read-aloud.js.
  if (kind === 'rendered' && /\.epub$/i.test(file.name || '')) {
    const chromeActions = overlay.querySelector('.files-preview-chrome-actions');
    const dlBtn = chromeActions?.querySelector('[data-action="download"]');
    if (chromeActions) {
      const textUrl = `/api/files/epub-text/${encodeURIComponent(file.id)}`;
      const narrationUrl = `/api/files/narration/${encodeURIComponent(file.id)}`;
      import('../epub-reader-controls.js')
        .then(m => {
          if (!overlay.isConnected) return;
          const ctl = m.createReaderControls({ textUrl, narrationUrl });
          chromeActions.insertBefore(ctl.el, dlBtn || chromeActions.firstChild);
          _readerControls = ctl;
        })
        .catch(() => { /* TTS unavailable — no controls */ });
    }
  }
  if (streamableMedia) _wirePreviewProgress(overlay, file, kind);
}

export async function openVideoPreviewById(fileId) {
  const file = _resolvePreviewFile(fileId) || await fetchFileEntry(fileId);
  if (!file) return false;
  if (videoLikelyUnsupported(file)) return false;
  const entityKind = String(file?.source_metadata?.entity_kind || '').toLowerCase();
  if (entityKind === 'music_video') {
    await _openMusicVideoInGrove(file);
    return true;
  }
  await _openFloatingVideoPlayer(file);
  return true;
}

async function _openFloatingVideoPlayer(file) {
  // Cast intercept: if a device is armed, send the video to that device
  // instead of opening the floating local player. The cast-shelf row
  // for that receiver takes over as the playback surface.
  try {
    const { isArmed } = await import('../armed-device.js');
    if (isArmed()) {
      const { castOrPlay } = await import('../cast-or-play.js');
      const result = await castOrPlay({
        capability: 'media.video_play@1',
        surface: 'video-preview',
        args: {
          content_url: _selectedVideoStreamUrl(file, { startTimeS: _initialVideoStartTime(file) }),
          file_id: String(file?.id || file?.file_id || ''),
          requires_auth: true,
          title: file?.display_name || file?.name || file?.title || '',
          poster_url: file?.poster_url || file?.cover_url || '',
          author: file?.series_title || file?.author || '',
        },
        fallback: null,
      });
      if (result.cast) return;
    }
  } catch { /* armed-device or cast-or-play unavailable — fall through to local */ }

  // Companion presence: this video is now "what's playing".
  import('../architect-observer.js')
    .then(m => m.reportAttention('surface.media.playback_started', {
      label: file?.display_name || file?.name || file?.title || '',
      kind: 'video',
      ref: String(file?.id || ''),
    }))
    .catch(() => {});

  await _ensureVideoPlaybackState(file);
  _wireFloatingSelectionListener();
  // Stamp the last-played registry so the next page load can offer a
  // one-tap "Resume X?" toast (parity with audiobookshelf / LibriVox /
  // Grove). Lazy-imported so video paths that never touch the audio
  // surface don't pay the parse cost on cold start.
  try {
    const fileId = file?.id || file?.file_id;
    if (fileId) {
      const { recordLastPlayed } = await import('../media-resume.js');
      recordLastPlayed({
        kind: 'video',
        fileId: String(fileId),
        title: file?.display_name || file?.name || file?.title || 'Video',
        subtitle: file?.series_title || file?.author || '',
        coverUrl: file?.poster_url || file?.cover_url || '',
      });
    }
  } catch { /* non-fatal — resume toast is a polish feature */ }
  const videoEl = document.createElement('video');
  const initialStartTimeS = _initialVideoStartTime(file);
  videoEl.className = 'files-preview-video-full';
  videoEl.src = _selectedVideoStreamUrl(file, { startTimeS: initialStartTimeS });
  // Custom overlay (.fv-local-progress) owns the full playback chrome —
  // Plex/Jellyfin/Netflix pattern. Keyboard shortcuts (Space, ←/→, ↑/↓,
  // m, f), play/pause, seek, volume, mute, captions, A/V offset, PiP
  // and fullscreen all live in floating-video.js. Native HTML5 controls
  // are off so the timeline isn't duplicated — and so transcoded streams
  // (where the native bar would lie about duration) don't show a wrong
  // timeline next to the correct one.
  videoEl.controls = false;
  videoEl.playsInline = true;
  videoEl.preload = 'metadata';
  _applyVideoPlaybackContext(videoEl, file, { baseOffsetS: initialStartTimeS });
  _syncFloatingSubtitleTrack(videoEl, file);
  _primeFloatingVideoResume(videoEl, file);
  const nextItem = _nextVideoItemFor(file);
  const metadata = _videoPreviewMetadata(file);
  const playbackApi = _videoPlaybackApi(videoEl, file);
  try {
    const mod = await import('../floating-video.js');
    _activeFloatingVideoSession = {
      fileId: file.id,
      file,
      videoEl,
    };
    mod.FloatingVideo?.open({
      iframe: videoEl,
      title: metadata.title,
      channel: metadata.channel,
      thumbnail: metadata.thumbnail,
      mode: 'companion',
      fileId: file.id,
      nextItem,
      onNext: nextItem ? async () => {
        const nextEntry = await fetchFileEntry(nextItem.fileId);
        if (!nextEntry) return;
        if (file._videoNav?.siblings) {
          nextEntry._videoNav = {
            siblings: file._videoNav.siblings,
            index: nextItem.index,
          };
        }
        return _openFloatingVideoPlayer(nextEntry);
      } : null,
      playbackMenu: _buildFloatingPlaybackMenu(file, videoEl),
      api: playbackApi,
      audioSyncProfileKey: _audioSyncProfileKey(file),
    });
    videoEl.play().catch(() => {});
  } catch (err) {
    console.warn('[files] floating video open failed:', err);
  }
}

async function _openMusicVideoInGrove(file) {
  await _ensureVideoPlaybackState(file);
  _wireFloatingSelectionListener();

  const videoEl = document.createElement('video');
  const initialStartTimeS = _initialVideoStartTime(file);
  videoEl.className = 'files-preview-video-full';
  videoEl.src = _selectedVideoStreamUrl(file, { startTimeS: initialStartTimeS });
  videoEl.controls = false;
  videoEl.playsInline = true;
  videoEl.preload = 'metadata';
  _applyVideoPlaybackContext(videoEl, file, { baseOffsetS: initialStartTimeS });
  _syncFloatingSubtitleTrack(videoEl, file);
  _primeFloatingVideoResume(videoEl, file);

  const metadata = _videoPreviewMetadata(file);
  const playbackApi = _videoPlaybackApi(videoEl, file);
  const progressHost = {};
  _wirePreviewProgressHost(progressHost, videoEl, file, 'video');

  try {
    const [
      floatingMod,
      groveMod,
      ambientMod,
      orbDetachMod,
    ] = await Promise.all([
      import('../floating-video.js'),
      import('../grove.js'),
      import('../grove-ambient.js'),
      import('../grove-orb-detach.js'),
    ]);

    floatingMod.FloatingVideo?.close?.();

    if (!orbDetachMod.isDetached?.()) {
      groveMod.openGrove?.();
    }

    await ambientMod.loadMediaVideo?.({
      element: videoEl,
      video: metadata,
      api: playbackApi,
      onClose: () => {
        void _flushPreviewProgress(progressHost, file, 'video', { force: true });
      },
    });

    if (!orbDetachMod.isDetached?.()) {
      requestAnimationFrame(() => {
        try {
          orbDetachMod.detach?.();
        } catch (err) {
          console.warn('[files] grove orb detach failed:', err);
        }
      });
    }
  } catch (err) {
    console.warn('[files] grove music video open failed:', err);
    _openFloatingVideoPlayer(file);
  }
}

function _videoPlaybackState(file) {
  const playback = file?.source_metadata?.playback;
  return playback && Array.isArray(playback.media_sources) ? playback : null;
}

function _syncVideoPlaybackAcrossState(file) {
  const row = state.files.find((entry) => entry.id === file.id);
  if (row && row !== file) {
    row.source_metadata = {
      ...(row.source_metadata || {}),
      ...(file.source_metadata || {}),
    };
  }
  if (state.detailOverrideFile?.id === file.id && state.detailOverrideFile !== file && state.detailOverrideFile !== row) {
    state.detailOverrideFile.source_metadata = {
      ...(state.detailOverrideFile.source_metadata || {}),
      ...(file.source_metadata || {}),
    };
  }
  if (state.selection.size === 1 && state.selection.has(file.id)) {
    updateSelectionUI();
  }
}

function _currentDetailFileId() {
  if (state.detailOverrideFile?.id) return state.detailOverrideFile.id;
  return state.selection.size === 1 ? [...state.selection][0] : '';
}

function _videoPreviewMetadata(file) {
  return {
    title: file.name || 'Video',
    channel: file.source_metadata?.library_name || file.source_metadata?.series_name || file.source || '',
    thumbnail: file.source_metadata?.has_cover ? mediaCoverUrl(file.id) : (file.thumbnail || ''),
    duration: _videoPreviewDurationLabel(file),
    fileId: file.id,
    entityKind: String(file?.source_metadata?.entity_kind || '').toLowerCase() || 'video',
  };
}

function _videoPreviewDurationLabel(file) {
  const totalSeconds = Math.max(0, Math.round(Number(file?.source_metadata?.duration_s || 0)));
  if (!totalSeconds) return '';
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    return `${hours}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
  }
  return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

function _videoPlaybackApi(videoEl, file) {
  return {
    setVolume: (v) => { videoEl.volume = Math.max(0, Math.min(1, v)); },
    getVolume: () => videoEl.volume,
    pause: () => videoEl.pause(),
    play: () => videoEl.play(),
    getCurrentTime: () => _effectiveVideoCurrentTime(videoEl, file),
    getDuration: () => _effectiveVideoDuration(videoEl, file),
    canSeek: () => _effectiveVideoDuration(videoEl, file) > 0,
    usesManagedSeek: () => _videoNeedsManagedSeek(file),
    seekTo: (timeS) => _seekVideoTo(videoEl, file, timeS),
  };
}

function _wireFloatingSelectionListener() {
  if (_floatingSelectionListenerWired || typeof window === 'undefined') return;
  window.addEventListener('media-video-selection', (e) => {
    const detail = e.detail || {};
    const session = _activeFloatingVideoSession;
    const fileId = detail.fileId || '';
    if (!session || !fileId || session.fileId !== fileId) return;
    if ((detail.origin || '') === 'floating-player') return;
    const selection = detail.selection && typeof detail.selection === 'object'
      ? detail.selection
      : null;
    if (!selection) return;
    const patch = {};
    if (Object.prototype.hasOwnProperty.call(selection, 'mediaSourceId')) {
      patch.mediaSourceId = selection.mediaSourceId || '';
    }
    if (Object.prototype.hasOwnProperty.call(selection, 'audioStreamIndex')) {
      patch.audioStreamIndex = selection.audioStreamIndex;
    }
    if (Object.prototype.hasOwnProperty.call(selection, 'subtitleStreamIndex')) {
      patch.subtitleStreamIndex = selection.subtitleStreamIndex;
    }
    if (!Object.keys(patch).length) return;
    void _switchFloatingVideoSelection(session.videoEl, session.file, patch, {
      persist: false,
      refreshDetail: false,
    });
  });
  window.addEventListener('floating-video:closed', () => {
    _activeFloatingVideoSession = null;
  });
  _floatingSelectionListenerWired = true;
}

function _selectedVideoSource(playback) {
  if (!playback?.media_sources?.length) return null;
  return playback.media_sources.find((source) => source?.id === playback.selected_media_source_id)
    || playback.media_sources[0];
}

function _selectedVideoAudioTrack(file) {
  const playback = _videoPlaybackState(file);
  const source = _selectedVideoSource(playback);
  if (!playback || !source) return null;
  const selectedIndex = Number(playback.selected_audio_stream_index);
  if (!Number.isFinite(selectedIndex)) return null;
  return (source.audio_tracks || []).find((track) => Number(track?.index) === selectedIndex) || null;
}

function _videoNeedsManagedSeek(file) {
  const codec = String(_selectedVideoAudioTrack(file)?.codec || '').trim().toLowerCase();
  return _BROWSER_UNSUPPORTED_AUDIO_CODECS.has(codec);
}

function _currentVideoContext(videoEl, file) {
  const knownDurationS = Math.max(0, Number(file?.source_metadata?.duration_s || 0));
  const context = videoEl?._augmentumVideoContext;
  if (context && typeof context === 'object') {
    return {
      requiresRestartSeek: !!context.requiresRestartSeek,
      baseOffsetS: Math.max(0, Number(context.baseOffsetS || 0)),
      knownDurationS: Math.max(knownDurationS, Number(context.knownDurationS || 0)),
    };
  }
  return {
    requiresRestartSeek: _videoNeedsManagedSeek(file),
    baseOffsetS: 0,
    knownDurationS,
  };
}

function _applyVideoPlaybackContext(videoEl, file, { baseOffsetS = 0 } = {}) {
  if (!videoEl) return _currentVideoContext(videoEl, file);
  const context = {
    requiresRestartSeek: _videoNeedsManagedSeek(file),
    baseOffsetS: Math.max(0, Number(baseOffsetS || 0)),
    knownDurationS: Math.max(0, Number(file?.source_metadata?.duration_s || 0)),
  };
  videoEl._augmentumVideoContext = context;
  return context;
}

function _effectiveVideoCurrentTime(videoEl, file) {
  const rawCurrent = Math.max(0, Number(videoEl?.currentTime || 0));
  const context = _currentVideoContext(videoEl, file);
  if (!context.requiresRestartSeek) return rawCurrent;
  return Math.max(0, context.baseOffsetS + rawCurrent);
}

function _effectiveVideoDuration(videoEl, file) {
  const rawDuration = Number.isFinite(videoEl?.duration) && Number(videoEl.duration) > 0
    ? Number(videoEl.duration)
    : 0;
  const context = _currentVideoContext(videoEl, file);
  if (!context.requiresRestartSeek) {
    // Direct-play: max of (element-reported) and (upstream-known) —
    // identical to the generic media-duration helper used by the cast
    // surfaces, so all three player paths agree.
    return effectiveDuration(rawDuration, context.knownDurationS);
  }
  // Managed-seek (Emby/Jellyfin transcode restart from baseOffsetS):
  // raw duration is the length of the CURRENT transcode segment, so
  // the effective end is baseOffsetS + raw. Still floor at the
  // upstream-known runtime in case the segment hasn't grown yet.
  return Math.max(
    context.knownDurationS || 0,
    context.baseOffsetS + rawDuration,
  );
}

function _initialVideoStartTime(file) {
  if (!_videoNeedsManagedSeek(file)) return 0;
  if (file?.source_metadata?.is_finished) return 0;
  return Math.max(0, Number(file?.source_metadata?.current_time_s || 0));
}

function _chooseTrackIndex(tracks, preferred, allowNone = false) {
  const valid = new Set((tracks || []).map((track) => Number(track?.index)).filter(Number.isFinite));
  if (Number.isFinite(preferred) && (valid.has(Number(preferred)) || (allowNone && Number(preferred) === -1))) {
    return Number(preferred);
  }
  const defaultTrack = (tracks || []).find((track) => track?.is_default);
  if (defaultTrack && Number.isFinite(Number(defaultTrack.index))) {
    return Number(defaultTrack.index);
  }
  if (allowNone) return -1;
  const first = (tracks || []).find((track) => Number.isFinite(Number(track?.index)));
  return first ? Number(first.index) : null;
}

function _applyLocalVideoPlaybackSelection(file, patch = {}) {
  const playback = _videoPlaybackState(file);
  if (!playback?.media_sources?.length) return null;
  const hasMediaSource = Object.prototype.hasOwnProperty.call(patch, 'mediaSourceId');
  const hasAudio = Object.prototype.hasOwnProperty.call(patch, 'audioStreamIndex');
  const hasSubtitle = Object.prototype.hasOwnProperty.call(patch, 'subtitleStreamIndex');

  const nextSourceId = hasMediaSource
    ? (patch.mediaSourceId || playback.media_sources[0]?.id || '')
    : (playback.selected_media_source_id || playback.media_sources[0]?.id || '');
  const source = playback.media_sources.find((item) => item?.id === nextSourceId)
    || playback.media_sources[0];
  const nextAudio = _chooseTrackIndex(
    source?.audio_tracks || [],
    hasAudio ? patch.audioStreamIndex : playback.selected_audio_stream_index,
    false,
  );
  const nextSubtitle = _chooseTrackIndex(
    source?.subtitle_tracks || [],
    hasSubtitle ? patch.subtitleStreamIndex : playback.selected_subtitle_stream_index,
    true,
  );

  playback.selected_media_source_id = source?.id || '';
  playback.selected_audio_stream_index = nextAudio;
  playback.selected_subtitle_stream_index = nextSubtitle;

  playback.media_sources.forEach((item) => {
    const isSelectedSource = item?.id === playback.selected_media_source_id;
    item.is_selected = isSelectedSource;
    (item.audio_tracks || []).forEach((track) => {
      track.is_selected = !!(isSelectedSource && Number(track.index) === Number(nextAudio));
    });
    (item.subtitle_tracks || []).forEach((track) => {
      track.is_selected = !!(isSelectedSource && Number(track.index) === Number(nextSubtitle));
    });
  });

  file.source_metadata = {
    ...(file.source_metadata || {}),
    playback,
    preferred_media_source_id: playback.selected_media_source_id,
    preferred_audio_stream_index: nextAudio,
    preferred_subtitle_stream_index: nextSubtitle,
  };
  _syncVideoPlaybackAcrossState(file);
  return {
    mediaSourceId: playback.selected_media_source_id,
    audioStreamIndex: nextAudio,
    subtitleStreamIndex: nextSubtitle,
  };
}

/**
 * Source-signature key for cross-file A/V offset memory. Returns a
 * string like ``"jellyfin_main|mp4|h264|aac"`` so an offset tuned on
 * one Jellyfin episode auto-applies to the next episode using the
 * same transcode pipeline — drift signatures are remarkably consistent
 * within a single pipeline. Empty string when we can't fingerprint
 * the source (local file, missing playback metadata) — the floating
 * player falls back to legacy per-file_id storage in that case.
 */
function _audioSyncProfileKey(file) {
  const source = String(file?.source || '').trim();
  if (!source) return '';
  const playback = _videoPlaybackState(file);
  if (!playback?.media_sources?.length) return source;
  const selectedId = playback.selected_media_source_id
    || playback.media_sources[0]?.id || '';
  const src = playback.media_sources.find((s) => s?.id === selectedId)
    || playback.media_sources[0];
  const container = String(src?.container || '').trim().toLowerCase();
  const videoCodec = String(src?.video_codec || '').trim().toLowerCase();
  const audioIdx = playback.selected_audio_stream_index;
  const tracks = src?.audio_tracks || [];
  const track = tracks.find((t) => Number(t?.index) === Number(audioIdx)) || tracks[0];
  const audioCodec = String(track?.codec || '').trim().toLowerCase();
  return [source, container, videoCodec, audioCodec].filter(Boolean).join('|');
}

function _selectedVideoStreamUrl(file, { startTimeS = null } = {}) {
  const selection = _applyLocalVideoPlaybackSelection(file);
  return selection
    ? mediaStreamUrl(file.id, {
      mediaSourceId: selection.mediaSourceId,
      audioStreamIndex: selection.audioStreamIndex,
      startTimeS,
    })
    : mediaStreamUrl(file.id);
}

function _selectedSubtitleTrack(file) {
  const playback = _videoPlaybackState(file);
  const source = _selectedVideoSource(playback);
  if (!playback || !source) return { source: null, track: null };
  const subtitleIndex = Number(playback.selected_subtitle_stream_index);
  if (!Number.isFinite(subtitleIndex) || subtitleIndex < 0) {
    return { source, track: null };
  }
  const track = (source.subtitle_tracks || []).find((item) => Number(item?.index) === subtitleIndex) || null;
  return { source, track };
}

function _syncFloatingSubtitleTrack(videoEl, file) {
  if (!videoEl) return;
  videoEl.querySelectorAll('track[data-augmentum-subtitle="true"]').forEach((el) => el.remove());
  try {
    const tracks = videoEl.textTracks || [];
    for (let i = 0; i < tracks.length; i += 1) {
      tracks[i].mode = 'disabled';
    }
  } catch {
    // Ignore track reset failures.
  }

  const { source, track } = _selectedSubtitleTrack(file);
  const subtitleIndex = Number(track?.index);
  if (!source?.id || !Number.isFinite(subtitleIndex) || subtitleIndex < 0) return;

  const trackEl = document.createElement('track');
  trackEl.kind = 'subtitles';
  trackEl.label = track.label || 'Subtitles';
  trackEl.srclang = _subtitleLanguageCode(track);
  const context = _currentVideoContext(videoEl, file);
  trackEl.src = mediaSubtitleUrl(file.id, {
    mediaSourceId: source.id,
    subtitleStreamIndex: subtitleIndex,
    startTimeS: context.requiresRestartSeek ? context.baseOffsetS : null,
  });
  trackEl.default = true;
  trackEl.dataset.augmentumSubtitle = 'true';
  const showTrack = () => {
    try {
      const tracks = videoEl.textTracks || [];
      for (let i = 0; i < tracks.length; i += 1) {
        const textTrack = tracks[i];
        textTrack.mode = i === tracks.length - 1 ? 'showing' : 'disabled';
      }
    } catch {
      // Ignore text track activation failures.
    }
  };
  trackEl.addEventListener('load', () => {
    setTimeout(showTrack, 0);
  }, { once: true });
  videoEl.appendChild(trackEl);
}

function _subtitleLanguageCode(track) {
  const raw = String(track?.language_code || track?.language || '').trim().toLowerCase();
  if (/^[a-z]{2,3}([_-][a-z0-9]{2,8})?$/i.test(raw)) {
    return raw.replace('_', '-');
  }
  const label = String(track?.label || '').toLowerCase();
  if (raw.includes('english') || label.includes('english')) return 'en';
  if (raw.includes('japanese') || label.includes('japanese')) return 'ja';
  if (raw.includes('spanish') || label.includes('spanish')) return 'es';
  if (raw.includes('french') || label.includes('french')) return 'fr';
  if (raw.includes('german') || label.includes('german')) return 'de';
  return 'en';
}

async function _ensureVideoPlaybackState(file) {
  if (_videoPlaybackState(file)) return file;
  const details = await fetchMediaDetails(file.id);
  if (!details) return file;
  file.source_metadata = {
    ...(file.source_metadata || {}),
    current_time_s: details.current_time_s ?? file.source_metadata?.current_time_s ?? 0,
    duration_s: details.duration_s ?? file.source_metadata?.duration_s ?? 0,
    progress_pct: details.progress_pct ?? file.source_metadata?.progress_pct ?? 0,
    is_finished: details.is_finished ?? file.source_metadata?.is_finished ?? false,
    playback: details.playback || file.source_metadata?.playback || null,
  };
  _syncVideoPlaybackAcrossState(file);
  return file;
}

async function _persistVideoPlaybackSelection(file, selection) {
  await updateMediaPlaybackSelection(file.id, {
    media_source_id: selection.mediaSourceId || '',
    audio_stream_index: selection.audioStreamIndex,
    subtitle_stream_index: selection.subtitleStreamIndex,
  });
}

async function _seekVideoTo(videoEl, file, targetTimeS) {
  if (!videoEl) return false;
  const duration = _effectiveVideoDuration(videoEl, file);
  const clampedTarget = duration > 0
    ? Math.max(0, Math.min(duration, Number(targetTimeS || 0)))
    : Math.max(0, Number(targetTimeS || 0));
  const context = _currentVideoContext(videoEl, file);
  if (!context.requiresRestartSeek) {
    try {
      videoEl.currentTime = clampedTarget;
      return true;
    } catch {
      return false;
    }
  }
  const wasPlaying = !!(!videoEl.paused && !videoEl.ended);
  const nextUrl = _selectedVideoStreamUrl(file, { startTimeS: clampedTarget });
  _applyVideoPlaybackContext(videoEl, file, { baseOffsetS: clampedTarget });
  _syncFloatingSubtitleTrack(videoEl, file);
  videoEl.src = nextUrl;
  videoEl.load();
  videoEl.addEventListener('loadedmetadata', () => {
    if (wasPlaying) {
      videoEl.play().catch(() => {});
    }
  }, { once: true });
  return true;
}

async function _switchFloatingVideoSelection(videoEl, file, patch, opts = {}) {
  const { persist = true, refreshDetail = true } = opts;
  const selection = _applyLocalVideoPlaybackSelection(file, patch);
  if (!selection) return;
  const currentTime = _effectiveVideoCurrentTime(videoEl, file);
  if (persist) {
    void _persistVideoPlaybackSelection(file, selection);
  }
  _syncFloatingSubtitleTrack(videoEl, file);
  const nextNeedsManagedSeek = _videoNeedsManagedSeek(file);
  const nextUrl = _selectedVideoStreamUrl(file, {
    startTimeS: nextNeedsManagedSeek ? currentTime : null,
  });
  const resolvedNextUrl = typeof window !== 'undefined'
    ? new URL(nextUrl, window.location.href).href
    : nextUrl;
  const resolvedCurrentUrl = String(videoEl.currentSrc || videoEl.src || '');
  _applyVideoPlaybackContext(videoEl, file, {
    baseOffsetS: nextNeedsManagedSeek ? currentTime : 0,
  });
  if (resolvedCurrentUrl === resolvedNextUrl) {
    if (refreshDetail && _currentDetailFileId() === file.id) {
      updateDetail();
    }
    return;
  }
  videoEl.src = nextUrl;
  videoEl.load();
  videoEl.addEventListener('loadedmetadata', () => {
    try {
      const duration = Number.isFinite(videoEl.duration) ? videoEl.duration : 0;
      const maxResume = duration > 5 ? Math.max(0, duration - 5) : duration;
      if (!nextNeedsManagedSeek && currentTime > 0) {
        videoEl.currentTime = duration > 0 ? Math.min(currentTime, maxResume || currentTime) : currentTime;
      }
    } catch {
      // Ignore seek failures during stream swaps.
    }
    videoEl.play().catch(() => {});
  }, { once: true });
  if (refreshDetail && _currentDetailFileId() === file.id) {
    updateDetail();
  }
}

function _buildFloatingPlaybackMenu(file, videoEl) {
  return {
    getState: () => _videoPlaybackState(file),
    selectMediaSource: async (mediaSourceId) => {
      await _switchFloatingVideoSelection(videoEl, file, { mediaSourceId });
    },
    selectAudioStream: async (audioStreamIndex) => {
      await _switchFloatingVideoSelection(videoEl, file, { audioStreamIndex });
    },
    selectSubtitleStream: async (subtitleStreamIndex) => {
      await _switchFloatingVideoSelection(videoEl, file, { subtitleStreamIndex });
    },
  };
}

function _primeFloatingVideoResume(videoEl, file) {
  if (_videoNeedsManagedSeek(file)) return;
  const resumeTime = Number(file?.source_metadata?.current_time_s || 0);
  if (resumeTime <= 1 || file?.source_metadata?.is_finished) return;
  // Mark the element as awaiting a resume seek so floating-video's
  // progress push can suppress early timeupdates (currentTime ≈ 0
  // before the seek lands), which would otherwise clobber the saved
  // position on the backend. Cleared on 'seeked', or as a safety net
  // once we see currentTime cross the resume target.
  videoEl._augmentumPendingResume = resumeTime;
  const clearPending = () => { videoEl._augmentumPendingResume = 0; };
  videoEl.addEventListener('seeked', clearPending, { once: true });
  const applyResume = () => {
    try {
      const duration = Number.isFinite(videoEl.duration) ? videoEl.duration : 0;
      const maxResume = duration > 5 ? Math.max(0, duration - 5) : duration;
      videoEl.currentTime = duration > 0
        ? Math.min(resumeTime, maxResume || resumeTime)
        : resumeTime;
    } catch {
      // Seek refused — drop the gate so we don't suppress real pushes forever.
      clearPending();
    }
  };
  videoEl.addEventListener('loadedmetadata', applyResume, { once: true });
}

function _wirePreviewProgress(overlay, file, kind) {
  const mediaEl = overlay.querySelector(kind === 'video' ? 'video' : 'audio');
  if (!mediaEl) return;
  _wirePreviewProgressHost(overlay, mediaEl, file, kind);
}

function _wirePreviewProgressHost(host, mediaEl, file, kind) {
  host._progressState = { lastSentAt: 0, mediaEl };
  const send = (opts = {}) => _flushPreviewProgress(host, file, kind, opts);
  mediaEl.addEventListener('timeupdate', () => send());
  mediaEl.addEventListener('pause', () => send({ force: true }));
  mediaEl.addEventListener('ended', () => send({ force: true, isFinished: true }));
}

async function _flushPreviewProgress(
  overlay,
  file,
  kind,
  { force = false, isFinished = false } = {},
) {
  const state = overlay?._progressState;
  const mediaEl = state?.mediaEl;
  if (!mediaEl || !isMediaServerFile(file)) return;
  const now = Date.now();
  if (!force && now - (state.lastSentAt || 0) < 15000) return;
  const duration = Number.isFinite(mediaEl.duration) && mediaEl.duration > 0
    ? mediaEl.duration
    : Number(file.source_metadata?.duration_s || 0);
  const current = isFinished
    ? (duration || Number(mediaEl.currentTime || 0))
    : Number(mediaEl.currentTime || 0);
  if (!duration && current <= 0) return;
  state.lastSentAt = now;
  const resp = await pushMediaProgress(file.id, {
    current_time_s: current,
    duration_s: duration,
    is_finished: isFinished,
  });
  if (resp?.progress_pct == null) return;
  file.source_metadata = {
    ...(file.source_metadata || {}),
    current_time_s: current,
    duration_s: duration,
    progress_pct: resp.progress_pct,
    is_finished: isFinished,
  };
  window.dispatchEvent(new CustomEvent('media-player:progress', {
    detail: {
      fileId: file.id,
      progressPct: resp.progress_pct,
      currentTimeS: current,
    },
  }));
}

// --- Image gallery ----------------------------------------------------

function _getImageFiles() {
  return state.files.filter(isImage);
}

async function _openFullImageLightbox(file) {
  try {
    const mod = await import('../image.js');
    const url = downloadUrl(file.id);
    // Pass the real file_index id (file.id) as image_id, not a
    // filename-derived guess. The image-edit endpoints
    // (/api/image/{id}/remove-bg, /upscale, etc.) now resolve
    // file_index ids in addition to image_generations ids, so the
    // lightbox edit buttons work on uploaded phone photos, chat
    // images, and any other image-bearing source — not only AI
    // generations.
    //
    // Old behavior derived image_id from ``file.name.replace(...)``
    // which only happened to work for AI generations whose filename
    // was ``{image_id}.png``. For uploads ("IMG_1234.jpg") it
    // produced a bogus id that 404'd every edit call.
    const entry = { name: file.name, image_id: file.id };
    try {
      const resp = await fetch(`/api/image/${encodeURIComponent(file.id)}`);
      if (resp.ok) Object.assign(entry, await resp.json());
    } catch { /* metadata optional — non-AI images won't have any */ }
    closeGallery();
    mod.openLightbox(entry, url);
  } catch (err) {
    console.warn('[files] full lightbox handoff failed:', err);
  }
}

export function openGallery(fileId) {
  const images = _getImageFiles();
  const idx = images.findIndex(f => f.id === fileId);
  if (idx < 0) return;
  state.galleryIndex = idx;

  const overlay = document.createElement('div');
  overlay.className = 'files-gallery-overlay';
  overlay.tabIndex = 0;
  state.galleryOverlay = overlay;
  document.body.appendChild(overlay);
  requestAnimationFrame(() => overlay.classList.add('visible'));
  overlay.focus();

  overlay.addEventListener('click', (e) => { if (e.target === overlay) closeGallery(); });
  overlay.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { e.preventDefault(); closeGallery(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); galleryNav(-1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); galleryNav(1); }
    else if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      const f = _getImageFiles()[state.galleryIndex];
      if (f) _galleryDelete(f);
    }
    else if ((e.key === 'd' || e.key === 'D') && !e.ctrlKey && !e.metaKey) {
      e.preventDefault();
      const f = _getImageFiles()[state.galleryIndex];
      if (f) downloadFile(f.id);
    }
    e.stopPropagation();
  });

  renderGallery();
}

export function closeGallery() {
  if (!state.galleryOverlay) return;
  const overlay = state.galleryOverlay;
  overlay.classList.remove('visible');
  state.galleryOverlay = null;
  state.galleryIndex = -1;
  setTimeout(() => overlay.remove(), 200);
}

export function galleryNav(delta) {
  const images = _getImageFiles();
  const next = state.galleryIndex + delta;
  if (next < 0 || next >= images.length) return;
  state.galleryIndex = next;
  renderGallery();
}

async function _galleryDelete(file) {
  const ok = await inlineConfirm({
    message: `Move "${file.name}" to Trash?`,
    action: 'Move to Trash',
    danger: true,
  });
  if (!ok) return;
  try {
    const resp = await deleteOne(file.id);
    if (!resp.ok) return;
    state.files = state.files.filter(f => f.id !== file.id);
    state.selection.delete(file.id);
    const remaining = _getImageFiles();
    if (!remaining.length) { closeGallery(); renderGrid(); return; }
    if (state.galleryIndex >= remaining.length) state.galleryIndex = remaining.length - 1;
    renderGallery();
    renderGrid();
    updateSelectionUI();
  } catch (err) { console.warn('[files] gallery delete error:', err); }
}

export function renderGallery() {
  if (!state.galleryOverlay) return;
  const images = _getImageFiles();
  const file = images[state.galleryIndex];
  if (!file) return;
  const url = downloadUrl(file.id);

  const meta = [
    humanSize(file.size_bytes),
    formatDate(file.created_at),
    file.source ? file.source.charAt(0).toUpperCase() + file.source.slice(1) : '',
  ].filter(Boolean).map(s => `<span>${escapeHtml(s)}</span>`).join('');

  const isGenerated = file.source === 'images';

  state.galleryOverlay.innerHTML = `
    <span class="files-gallery-counter">${state.galleryIndex + 1} / ${images.length}</span>
    <button class="files-gallery-close" title="Close (Esc)" aria-label="Close">&times;</button>
    <button class="files-gallery-nav files-gallery-prev" title="Previous (\u2190)" aria-label="Previous" ${state.galleryIndex <= 0 ? 'disabled' : ''}>
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 18 9 12 15 6"/></svg>
    </button>
    <div class="files-gallery-stage">
      <img src="${escapeHtml(url)}" alt="${escapeHtml(file.name)}">
    </div>
    <button class="files-gallery-nav files-gallery-next" title="Next (\u2192)" aria-label="Next" ${state.galleryIndex >= images.length - 1 ? 'disabled' : ''}>
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>
    </button>
    <div class="files-gallery-meta">
      <div class="files-gallery-meta-name">${escapeHtml(file.name)}</div>
      <div class="files-gallery-meta-sub">${meta}</div>
    </div>
    <div class="files-gallery-actions">
      <button class="files-gallery-action" data-action="download" title="Download (D)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        Download
      </button>
      <button class="files-gallery-action" data-action="reference">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
        Reference in Chat
      </button>
      ${isGenerated ? `
      <button class="files-gallery-action" data-action="edit" title="Open in Image editor">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4 12.5-12.5z"/></svg>
        Edit
      </button>` : ''}
      <button class="files-gallery-action danger" data-action="delete" title="Delete (Del)">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg>
        Delete
      </button>
    </div>
  `;

  state.galleryOverlay.querySelector('.files-gallery-close').addEventListener('click', closeGallery);
  state.galleryOverlay.querySelector('.files-gallery-prev')?.addEventListener('click', () => galleryNav(-1));
  state.galleryOverlay.querySelector('.files-gallery-next')?.addEventListener('click', () => galleryNav(1));
  state.galleryOverlay.querySelectorAll('.files-gallery-action').forEach(btn => {
    btn.addEventListener('click', () => {
      const action = btn.dataset.action;
      if (action === 'download') downloadFile(file.id);
      else if (action === 'reference') referenceInChat(file.id, file.name);
      else if (action === 'edit') _openFullImageLightbox(file);
      else if (action === 'delete') _galleryDelete(file);
    });
  });
}

// Surfaces other than the grid (the continue rail today; future
// chrome like a recents strip / picker drawer) can register a small
// provider that supplies file entries which AREN'T in state.files.
// Resolution order: detailOverrideFile → grid (state.files) → any
// registered extra-source provider. Without this, the rail's
// episode entries (entity_kind=episode) wouldn't resolve on click
// because state.files for the shows chip holds SERIES rows, not
// episodes — and openMediaPreview would silently no-op.
const _extraFileProviders = new Set();
export function registerExtraFileSource(provider) {
  _extraFileProviders.add(provider);
  return () => _extraFileProviders.delete(provider);
}

function _resolvePreviewFile(fileId) {
  if (state.detailOverrideFile?.id === fileId) return state.detailOverrideFile;
  const fromGrid = state.files.find(f => f.id === fileId);
  if (fromGrid) return fromGrid;
  for (const provider of _extraFileProviders) {
    try {
      const entries = provider() || [];
      const hit = entries.find(f => f && f.id === fileId);
      if (hit) return hit;
    } catch { /* a flaky provider shouldn't break resolution */ }
  }
  return null;
}

function _nextVideoItemFor(file) {
  const nav = file?._videoNav;
  if (!nav || !Array.isArray(nav.siblings)) return null;
  const next = nav.siblings[Number(nav.index) + 1];
  if (!next?.file_id) return null;
  return {
    fileId: next.file_id,
    name: next.name || 'Next',
    index: Number(nav.index) + 1,
  };
}
