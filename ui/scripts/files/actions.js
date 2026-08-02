/**
 * Files panel — user action handlers: context menu, confirm prompts,
 * activation dispatcher, single-file mutations that touch server + local
 * state, and bulk actions.
 */

import { escapeHtml, showToast } from '../app.js';
import { state } from './state.js';
import {
  isImage, isPdf, isVideo, isAudio, isHtml, isEpub, isOffice, isMarkdown, isText,
  isAppProject, isArchive, isBookmark, isBuiltinLibrivox, isComic, isMediaServerFile, bookmarkUrl,
  humanSize, formatCount,
} from './helpers.js';
import {
  downloadUrl, fetchStats,
  toggleFavoriteApi, summarize, transformFile, zipDownload,
  deleteOne, bulkDeleteIds, restoreOne, bulkRestoreIds, purgeTrash, unpinLibrivox,
  pushMediaProgress,
} from './api.js';
// NOTE: `render` + `preview` are peers — import via module object to dodge
// the circular edge (ES module hoisting binds the exports at call time).
import * as R from './render.js';
import * as P from './preview.js';

// --- Download --------------------------------------------------------

export function downloadFile(id) {
  try {
    const a = document.createElement('a');
    a.href = downloadUrl(id);
    a.download = '';
    document.body.appendChild(a);
    a.click();
    a.remove();
  } catch (err) {
    console.warn('[files] download error:', err);
  }
}

// --- Activation dispatcher -------------------------------------------
// Ordered cascade: app project > image > pdf > video > audio > html >
// epub/office > markdown/text > download. First match wins.

// Best-effort: tell Discovery the user just opened this item so it
// shows up in History. Each kind maps to a content_type the history
// renderer understands (video → 'movie' or 'show' depending on entity
// metadata, comic → 'comic'). Audio is intentionally NOT logged here:
// library audiobooks/podcasts route through media-player.js, which
// emits its own media_play signal on play(). Logging both would
// double-stamp the same item.
//
// Dedupe: the upsert_history server-side keys on URL so re-emits
// don't create duplicate rows, BUT each click still costs a network
// round-trip + a DB upsert. The cache below skips the POST entirely
// when the same file was logged within the last 30 seconds. Long
// enough to absorb double-clicks and rapid drill-down/back patterns,
// short enough that a genuinely new session 30s+ later still records
// engagement so the recommender's frecency stays honest.
const _RECENTLY_LOGGED = new Map();  // file_id → timestamp
const _DEDUP_WINDOW_MS = 30 * 1000;
function _wasRecentlyLogged(fileId) {
  const now = Date.now();
  // Periodic prune so the map can't grow unbounded across a long
  // session of varied opens.
  if (_RECENTLY_LOGGED.size > 200) {
    for (const [k, t] of _RECENTLY_LOGGED) {
      if (now - t > _DEDUP_WINDOW_MS) _RECENTLY_LOGGED.delete(k);
    }
  }
  const last = _RECENTLY_LOGGED.get(fileId);
  if (last && now - last < _DEDUP_WINDOW_MS) return true;
  _RECENTLY_LOGGED.set(fileId, now);
  return false;
}

function _logHistoryFromFiles(file, contentType) {
  if (!file?.id) return;
  if (_wasRecentlyLogged(file.id)) return;
  const meta = file.source_metadata || {};
  fetch('/api/discovery/signal', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      signal_type: contentType === 'comic' ? 'comic_read' : 'media_play',
      source_url: `augm:media:${file.id}`,
      source_title: file.name || 'Untitled',
      content_type: contentType,
      metadata: {
        file_id: file.id,
        kind: file.kind || '',
        cover_url: `/api/media/cover/${file.id}`,
        author: meta.author || meta.director || meta.artist || '',
        entity_kind: String(meta.entity_kind || '').toLowerCase(),
      },
    }),
  }).catch(() => { /* fire-and-forget */ });
}

function _videoContentType(file) {
  // Series (TV shows) get 'show'; everything else under the video kind
  // is treated as 'movie' so the History row can label correctly.
  const ek = String(file?.source_metadata?.entity_kind || '').toLowerCase();
  return (ek === 'series' || ek === 'season' || ek === 'episode') ? 'show' : 'movie';
}

export function activateFile(file) {
  if (!file) return;
  // Bookmarks: external URL pointer, no on-disk file. Hand off to the
  // YouTube panel for YouTube URLs, otherwise open in a new tab.
  if (isBookmark(file)) return _openBookmark(file);
  // app-builder zip/html goes to the Library workspace; generic archives
  // (including user-uploaded zips) get the contents-listing preview.
  if (isAppProject(file)) return P.openProject(file);
  if (
    isMediaServerFile(file)
    && !file.source_metadata?.stream_path
    && !file.source_metadata?.selected_episode_id
  ) return;
  if (isImage(file))      return P.openGallery(file.id);
  if (isPdf(file))        return P.openMediaPreview(file.id, 'pdf');
  if (isVideo(file))      {
    _logHistoryFromFiles(file, _videoContentType(file));
    return P.openMediaPreview(file.id, 'video');
  }
  if (isAudio(file))      return P.openMediaPreview(file.id, 'audio');
  // Comics (CBZ/CBR/Komga/Suwayomi) go to the dedicated reader — must
  // come before isArchive() since CBZ is technically a zip but we want
  // the page viewer, not the contents-listing preview.
  if (isComic(file)) {
    _logHistoryFromFiles(file, 'comic');
    return import('../comic-reader/index.js?v=surface-handoff-20260512a').then(m => m.openComicReader(file));
  }
  if (isHtml(file))       return P.openMediaPreview(file.id, 'html');
  if (isEpub(file) || isOffice(file)) return P.openMediaPreview(file.id, 'rendered');
  if (isArchive(file))    return P.openMediaPreview(file.id, 'rendered');
  if (isMarkdown(file) || isText(file)) return P.openMediaPreview(file.id, 'rendered');
  downloadFile(file.id);
}

async function _openBookmark(file) {
  const url = bookmarkUrl(file);
  if (!url) {
    showToast('Bookmark URL missing', 'error');
    return;
  }
  const meta = file.source_metadata || {};
  // YouTube URLs go through the in-app player so transcripts + ask-bar
  // light up. Everything else opens in a new tab.
  const isYouTube = (meta.platform === 'youtube') ||
                    /youtube\.com|youtu\.be/i.test(url);
  if (isYouTube && meta.video_id) {
    try {
      const yt = await import('../youtube-panel.js');
      yt.openFromSearch(meta.video_id, file.name, meta.channel || '');
      return;
    } catch (err) {
      console.warn('[files] youtube panel open failed:', err);
    }
  }
  window.open(url, '_blank', 'noopener,noreferrer');
}

// --- Reference / Copy / Pack ----------------------------------------

export function referenceInChat(id, name) {
  const input = document.getElementById('chat-input');
  if (input) {
    const ref = `[file:${id}] ${name}`;
    input.value = input.value ? input.value + ' ' + ref : ref;
    input.focus();
    input.dispatchEvent(new Event('input', { bubbles: true }));
  }
  import('./index.js').then(m => m.closeFiles());
}

export function copyName(name) {
  navigator.clipboard?.writeText(name).catch(() => {});
}

// --- Inline confirm --------------------------------------------------

export function inlineConfirm({ message, action, danger = false }) {
  if (!state.el.bulkBar) return Promise.resolve(false);
  return new Promise((resolve) => {
    const prev = state.el.bulkBar.innerHTML;
    const cls = danger ? 'danger' : '';
    state.el.bulkBar.innerHTML = `
      <span class="files-bulk-count">${escapeHtml(message)}</span>
      <button class="btn btn-sm ${cls}" data-confirm-yes>${escapeHtml(action)}</button>
      <button class="btn btn-sm" data-confirm-no>Cancel</button>
    `;
    state.el.bulkBar.classList.add('visible', 'confirming');
    const cleanup = (ans) => {
      state.el.bulkBar.classList.remove('confirming');
      state.el.bulkBar.innerHTML = prev;
      updateBulkBar();
      resolve(ans);
    };
    state.el.bulkBar.querySelector('[data-confirm-yes]').addEventListener('click', () => cleanup(true));
    state.el.bulkBar.querySelector('[data-confirm-no]').addEventListener('click', () => cleanup(false));
  });
}

// --- Single-file mutations ------------------------------------------

async function _loadStats() {
  const data = await fetchStats();
  if (!data) return;
  if (state.el.stats) {
    state.el.stats.textContent = `${formatCount(data.total_count)} files \u00B7 ${humanSize(data.total_size)}`;
  }
  // Prefer scope-partitioned kind counts so the Audio chip under Local
  // scope doesn't count cloud audiobooks (which get filtered out of the
  // list query, producing the "chip says 120 but grid is empty" bug).
  // Falls back to the global by_kind if the server hasn't emitted the
  // scope breakdown yet.
  const kindByScope = data.by_kind_by_scope || {};
  const kindScoped = kindByScope[state.currentScope] || {};
  const kindGlobal = data.by_kind || {};
  const pickKind = (k) => (kindScoped[k] || kindGlobal[k] || {}).count || 0;

  const src = data.by_source || {};
  const scope = data.by_scope || {};
  // 'all' adapts to scope — Local/All shows local count, Cloud/All shows
  // cloud count — matching what the list query actually returns when
  // 'all' is selected under that scope. Falls back to total_count if
  // the backend hasn't returned scope totals yet (shouldn't happen, but
  // gracefully degrades instead of zeroing the chip).
  const localCount = (scope.local || {}).count;
  const cloudCount = (scope.cloud || {}).count;
  const scopeAwareAll = state.currentScope === 'cloud'
    ? (cloudCount ?? data.total_count ?? 0)
    : (localCount ?? data.total_count ?? 0);
  const counts = {
    all:        scopeAwareAll,
    images:     pickKind('image'),
    documents:  pickKind('document'),
    audio:      pickKind('audio'),
    video:      pickKind('video'),
    code:       pickKind('code'),
    archives:   pickKind('archive'),
    // Audiobooks/Comics are always cloud — use their source groups
    // directly; no scope partitioning needed for these.
    audiobooks: (src.audiobooks|| {}).count || 0,
    podcasts:   (src.podcasts || {}).count || 0,
    comics:     (src.comics    || {}).count || 0,
    shows:      (src.shows     || {}).count || 0,
    movies:     (src.movies    || {}).count || 0,
    music_videos: (src.music_videos || {}).count || 0,
    favorites:  data.favorites || 0,
    trash:      data.trash || 0,
  };
  document.querySelectorAll('[data-count-for]').forEach(el => {
    const slug = el.dataset.countFor;
    const v = counts[slug];
    el.textContent = v ? formatCount(v) : '';
  });
  // Scope toggle badges — one line per scope so users can see the
  // split at a glance. Empty strings hide via :empty CSS rule.
  document.querySelectorAll('[data-scope-count]').forEach(el => {
    const s = el.dataset.scopeCount;
    const v = (scope[s] || {}).count;
    el.textContent = v ? formatCount(v) : '';
  });
  // Podcasts are only meaningful when ABS has podcast rows. Keep the chip
  // discoverable once selected, but otherwise hide the empty affordance.
  const podcastsChip = document.querySelector('.files-chip[data-source="podcasts"]');
  if (podcastsChip) {
    podcastsChip.hidden = !counts.podcasts && state.currentSource !== 'podcasts';
  }
}
// Exported so index.js can trigger a refresh after openFiles etc.
export { _loadStats as loadStats };

export async function deleteFileAction(id) {
  const file = state.files.find(f => f.id === id);
  if (!file) return;
  // Built-in LibriVox rows belong to the Discover catalog, not the user's
  // storage. Route them through the unpin endpoint so the sentinel check
  // on the backend is satisfied and the Discover card flips back to
  // "+ Pin" instead of leaving a 400 from the generic file delete.
  if (isBuiltinLibrivox(file)) return unpinLibrivoxAction(id);
  const ok = await inlineConfirm({
    message: `Move "${file.name}" to Trash?`, action: 'Move to Trash', danger: true,
  });
  if (!ok) return;
  try {
    const resp = await deleteOne(id);
    if (!resp.ok) return;
    state.files = state.files.filter(f => f.id !== id);
    state.selection.delete(id);
    R.renderGrid();
    _loadStats();
  } catch (err) { console.warn('[files] delete error:', err); }
}

export async function unpinLibrivoxAction(id) {
  const file = state.files.find(f => f.id === id);
  if (!file) return;
  const ok = await inlineConfirm({
    message: `Unpin "${file.name}" from your library?`,
    action: 'Unpin',
    danger: true,
  });
  if (!ok) return;
  try {
    const resp = await unpinLibrivox(id);
    if (!resp.ok) {
      showToast('Failed to unpin from library', 'error', 4000);
      return;
    }
    state.files = state.files.filter(f => f.id !== id);
    state.selection.delete(id);
    R.renderGrid();
    _loadStats();
    // Notify the Discover overlay so a re-opened browse card shows "+ Pin".
    window.dispatchEvent(new CustomEvent('media-servers:changed'));
  } catch (err) { console.warn('[files] unpin error:', err); }
}
// Preview module imports this under the `deleteOneUi` alias it expects.
export { deleteFileAction as deleteOneUi };

export async function toggleFavoriteAction(id) {
  try {
    const data = await toggleFavoriteApi(id);
    if (!data) return;
    const file = state.files.find(f => f.id === id);
    if (file) file.is_favorite = data.is_favorite;
    const card = state.el.grid?.querySelector(`.files-card[data-id="${CSS.escape(id)}"]`);
    const star = card?.querySelector('.files-card-fav');
    if (star && file) {
      star.classList.toggle('active', !!file.is_favorite);
      const path = star.querySelector('path');
      if (path) path.setAttribute('fill', file.is_favorite ? 'currentColor' : 'none');
    }
  } catch (err) { console.warn('[files] favorite error:', err); }
}

export async function restoreFileAction(id) {
  try {
    const resp = await restoreOne(id);
    if (!resp.ok) return;
    state.files = state.files.filter(f => f.id !== id);
    state.selection.delete(id);
    R.renderGrid();
    _loadStats();
  } catch (err) { console.warn('[files] restore error:', err); }
}

// File extensions whose contents we can extract to plain text via the
// document_parse parsers on the server. `.txt`/`.md`/`.csv` are skipped
// because they're already readable text.
const _EXTRACTABLE_EXTS = new Set(['pdf', 'docx', 'pptx', 'xlsx']);

function _isExtractable(file) {
  const ext = (file.name.split('.').pop() || '').toLowerCase();
  return _EXTRACTABLE_EXTS.has(ext);
}

function _extractTextButton() {
  return `<button data-action="extract-text">`
    + `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">`
    + `<path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>`
    + `<polyline points="14 2 14 8 20 8"/>`
    + `<line x1="9" y1="13" x2="15" y2="13"/>`
    + `<line x1="9" y1="17" x2="15" y2="17"/>`
    + `</svg> Extract text</button>`;
}

// --- Media watch-state actions --------------------------------------
// Render the context-menu chunk for media-server rows. Only one of
// "Mark as watched" / "Mark as unwatched" shows at a time, picked by
// the row's current is_finished flag, so the menu reflects state.
// "Reset progress" appears whenever the row has any progress (a way to
// say "I never started this" without claiming you finished it; also
// removes it from the continue rail).
function _mediaWatchButtons(file) {
  const meta = file.source_metadata || {};
  const isFinished = !!meta.is_finished;
  const hasProgress = (Number(meta.progress_pct) || 0) > 0;
  const eyeSvg = (
    `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">`
    + `<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>`
    + `<circle cx="12" cy="12" r="3"/></svg>`
  );
  const undoSvg = (
    `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">`
    + `<polyline points="1 4 1 10 7 10"/>`
    + `<path d="M3.51 15a9 9 0 105.64-11.36L1 10"/></svg>`
  );
  const watchedBtn = isFinished
    ? `<button data-action="mark-unwatched">${undoSvg} Mark as unwatched</button>`
    : `<button data-action="mark-watched">${eyeSvg} Mark as watched</button>`;
  const resetBtn = (hasProgress && !isFinished)
    ? `<button data-action="reset-progress">${undoSvg} Remove from continue list</button>`
    : '';
  return `${watchedBtn}${resetBtn}<div class="files-context-divider"></div>`;
}

/**
 * Set ``is_finished`` for a media-server row. ``finished=true`` snaps
 * progress to 100% and pushes upstream so other clients (phone,
 * desktop ABS, etc.) reflect it. ``finished=false`` re-opens the row
 * for resume but preserves the existing position.
 *
 * Reuses ``pushMediaProgress`` (the same endpoint media-player.js
 * uses for time-update pushes) and dispatches the same
 * ``media-player:progress`` event so the grid + continue rail
 * update in place — same propagation path as live playback updates.
 */
export async function setMediaFinishedAction(file, finished) {
  if (!file) return;
  const meta = file.source_metadata || {};
  const durationS = Number(meta.duration_s) || 0;
  // When marking watched, snap to the end so the progress bar fills.
  // When un-marking, leave the existing position alone — the user
  // wants to RESUME, not restart.
  const currentTimeS = finished
    ? (durationS || Number(meta.current_time_s) || 0)
    : (Number(meta.current_time_s) || 0);
  try {
    const resp = await pushMediaProgress(file.id, {
      current_time_s: currentTimeS,
      duration_s: durationS,
      is_finished: finished,
    });
    if (!resp) {
      showToast('Could not update watch state', 'error');
      return;
    }
    showToast(finished ? 'Marked as watched' : 'Marked as unwatched', 'success', 2000);
    window.dispatchEvent(new CustomEvent('media-player:progress', {
      detail: {
        fileId: file.id,
        progressPct: finished ? 1 : (durationS > 0 ? currentTimeS / durationS : 0),
        currentTimeS,
        durationS,
        isFinished: finished,
      },
    }));
  } catch (err) {
    showToast('Could not update watch state: ' + err.message, 'error');
  }
}

/**
 * Snap progress to 0 and clear is_finished. Removes the row from the
 * continue rail (which requires progress > 0 AND not finished) and
 * leaves it in the not-started bucket for future resume from the top.
 * Symmetric to setMediaFinishedAction so the consumer paths are the
 * same — same endpoint, same event, same in-place UI update.
 */
export async function resetMediaProgressAction(file) {
  if (!file) return;
  const meta = file.source_metadata || {};
  const durationS = Number(meta.duration_s) || 0;
  try {
    const resp = await pushMediaProgress(file.id, {
      current_time_s: 0,
      duration_s: durationS,
      is_finished: false,
    });
    if (!resp) {
      showToast('Could not reset progress', 'error');
      return;
    }
    showToast('Removed from continue list', 'success', 2000);
    window.dispatchEvent(new CustomEvent('media-player:progress', {
      detail: {
        fileId: file.id,
        progressPct: 0,
        currentTimeS: 0,
        durationS,
        isFinished: false,
      },
    }));
  } catch (err) {
    showToast('Could not reset progress: ' + err.message, 'error');
  }
}

export async function extractTextAction(file) {
  if (!file) return;
  showToast(`Extracting text from "${file.name}"\u2026`, 'info', 2000);
  try {
    const data = await transformFile(file.id, 'extract_text', {}, 'new_file');
    if (data && data.file) {
      showToast(`Saved as "${data.file.name}"`, 'success', 3000);
      const { loadFiles } = await import('./index.js');
      await loadFiles({ reset: true });
      _loadStats();
    } else {
      showToast('Extracted, but no file row returned', 'warning');
    }
  } catch (err) {
    console.warn('[files] extract error:', err);
    showToast(err.message || 'Extract failed', 'error');
  }
}

// Document-conversion targets for markdown sources. Backend currently
// only accepts md → docx|pdf; widening here means widening the server's
// `_convert_document` dispatch first.
const _DOC_CONVERT_TARGETS = [
  { id: 'docx', label: 'DOCX' },
  { id: 'pdf',  label: 'PDF'  },
];

function _documentConvertButtons() {
  return _DOC_CONVERT_TARGETS
    .map(t => `<button data-action="convert-doc" data-target="${t.id}">`
      + `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">`
      + `<polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 014-4h14"/>`
      + `<polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 01-4 4H3"/>`
      + `</svg> Convert \u2192 ${t.label}</button>`)
    .join('');
}

export async function convertDocumentAction(file, target) {
  if (!file || !target) return;
  showToast(`Converting "${file.name}" to ${target.toUpperCase()}\u2026`, 'info', 2000);
  try {
    const data = await transformFile(file.id, 'convert_document', { target }, 'new_file');
    if (data && data.file) {
      showToast(`Saved as "${data.file.name}"`, 'success', 3000);
      const { loadFiles } = await import('./index.js');
      await loadFiles({ reset: true });
      _loadStats();
    } else {
      showToast('Converted, but no file row returned', 'warning');
    }
  } catch (err) {
    console.warn('[files] doc convert error:', err);
    showToast(err.message || 'Convert failed', 'error');
  }
}

// Format-conversion targets the user can pick from the context menu.
// `current` aliasing keeps `.jpeg` and `.jpg` from offering "convert to itself".
const _CONVERT_TARGETS = [
  { id: 'png',  label: 'PNG'  },
  { id: 'jpg',  label: 'JPG'  },
  { id: 'webp', label: 'WEBP' },
];

function _convertButtonsForImage(file) {
  const ext = (file.name.split('.').pop() || '').toLowerCase();
  const current = ext === 'jpeg' ? 'jpg' : ext;
  return _CONVERT_TARGETS
    .filter(t => t.id !== current)
    .map(t => `<button data-action="convert" data-target="${t.id}">`
      + `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">`
      + `<polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 014-4h14"/>`
      + `<polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 01-4 4H3"/>`
      + `</svg> Convert \u2192 ${t.label}</button>`)
    .join('');
}

export async function convertImageAction(file, target) {
  if (!file || !target) return;
  showToast(`Converting "${file.name}" to ${target.toUpperCase()}\u2026`, 'info', 2000);
  try {
    const data = await transformFile(file.id, 'convert_image', { target }, 'new_file');
    if (data && data.file) {
      showToast(`Saved as "${data.file.name}"`, 'success', 3000);
      // Refresh from server: adapter.save returns the upload-row id (ul_*),
      // not the file_index id (fi_*) that /download resolves against. Same
      // reload-after pattern that /upload uses.
      const { loadFiles } = await import('./index.js');
      await loadFiles({ reset: true });
      _loadStats();
    } else {
      showToast('Converted, but no file row returned', 'warning');
    }
  } catch (err) {
    console.warn('[files] convert error:', err);
    showToast(err.message || 'Convert failed', 'error');
  }
}

// TTS entry point for the Files detail panel. Pulls plain-text content
// from /api/files/text/{id} (which routes textual rows through directly
// and document rows through the parser registry) and hands it to the
// shared read-aloud helper. Toggling the same button stops playback.
export async function readAloudFile(id, btn) {
  const file = state.files.find(f => f.id === id);
  if (!file) return;
  const { readAloud, stopReadAloud, isReadAloudActive } = await import('../read-aloud.js');
  // If the button is already in playing state, treat the click as stop —
  // the shared helper also does this internally when the same button is
  // passed, but this early-return skips the redundant fetch.
  if (btn?.classList.contains('playing')) {
    stopReadAloud();
    return;
  }
  // Stop any in-flight session so the "Extracting..." toast doesn't race
  // with audio from a previous file.
  if (isReadAloudActive()) stopReadAloud();
  showToast(`Preparing "${file.name}" for playback\u2026`, 'info', 1500);
  try {
    const resp = await fetch(`/api/files/text/${encodeURIComponent(id)}`);
    if (resp.status === 415) {
      showToast('This file type can\'t be read aloud.', 'warning');
      return;
    }
    if (resp.status === 501) {
      showToast('Text extractor unavailable for this format.', 'warning');
      return;
    }
    if (resp.status === 422) {
      showToast('Couldn\'t extract readable text from this file.', 'warning');
      return;
    }
    if (!resp.ok) {
      showToast(`Couldn't load file text (${resp.status}).`, 'error');
      return;
    }
    const text = (await resp.text()).trim();
    if (!text) {
      showToast('No readable text in this file.', 'warning');
      return;
    }
    await readAloud(text, btn);
  } catch (err) {
    console.warn('[files] read-aloud error:', err);
    showToast('Read-aloud failed — network error.', 'error');
  }
}

// --- TTS narration ("audio partner" for EPUBs) -----------------------
// Tap the mic button to record (or play, if a narration already exists).
// Press and hold to pick a voice first. The synth runs as a background
// job server-side; the file's narration also gets built passively just by
// reading the book aloud in the viewer.

let _narrHoldTimer = null;
let _narrHeld = false;

export function narrationHoldStart(id, btn, name) {
  _narrHeld = false;
  clearTimeout(_narrHoldTimer);
  _narrHoldTimer = setTimeout(() => {
    _narrHeld = true;
    _openNarrationVoiceMenu(id, btn, name);
  }, 550);
}

export function narrationHoldCancel() {
  // Only stop the pending hold timer — leave `_narrHeld` alone so the
  // click that follows a completed hold can be swallowed by narrationTap().
  clearTimeout(_narrHoldTimer);
}

export function narrationTap(id, btn, name) {
  if (_narrHeld) { _narrHeld = false; return; }   // a hold just opened the menu
  _narrationActivate(id, btn, name, null);
}

async function _narrationStatus(id) {
  try {
    const r = await fetch(`/api/files/narration/${encodeURIComponent(id)}`);
    return r.ok ? await r.json() : null;
  } catch { return null; }
}

async function _narrationActivate(id, btn, name, voice) {
  const st = await _narrationStatus(id);
  if (st && st.status === 'done' && st.download_url) {
    try { const a = new Audio(st.download_url); a.play(); showToast(`Playing narration: ${name}`, 'info', 2000); }
    catch { window.open(st.download_url, '_blank'); }
    return;
  }
  if (st && (st.status === 'running' || st.status === 'pending')) {
    const pct = (typeof st.progress === 'number') ? ` (${Math.round(st.progress * 100)}%)` : '';
    showToast(`Narration is still being built${pct}${st.stage ? ' — ' + st.stage : ''}. Check back soon.`, 'info', 3500);
    return;
  }
  btn?.classList.add('busy');
  try {
    const r = await fetch(`/api/files/narration/${encodeURIComponent(id)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(voice ? { voice } : {}),
    });
    if (r.status === 422) {
      const d = await r.json().catch(() => ({}));
      showToast(d.detail || 'Narration needs the built-in Kokoro voice — set it in Settings.', 'warning', 6000);
      return;
    }
    if (!r.ok) { showToast(`Couldn't start narration (${r.status}).`, 'error'); return; }
    showToast(`Recording narration for "${name}" — building your audiobook in the background. Tap the mic again later to play it.`, 'info', 6000);
  } catch { showToast('Narration failed — network error.', 'error'); }
  finally { btn?.classList.remove('busy'); }
}

let _narrMenuEl = null;
function _closeNarrationMenu() {
  if (_narrMenuEl) { _narrMenuEl.remove(); _narrMenuEl = null; }
  document.removeEventListener('click', _onDocClickNarr, true);
}
function _onDocClickNarr(e) {
  if (_narrMenuEl && !_narrMenuEl.contains(e.target)) _closeNarrationMenu();
}

async function _openNarrationVoiceMenu(id, btn, name) {
  _closeNarrationMenu();
  const menu = document.createElement('div');
  menu.className = 'files-narration-menu';
  menu.style.cssText = 'position:fixed;z-index:9999;background:var(--bg-elevated,#161625);'
    + 'border:1px solid var(--border,#2d2d45);border-radius:8px;padding:4px;max-height:50vh;'
    + 'overflow:auto;min-width:190px;box-shadow:0 8px 24px rgba(0,0,0,.4);font-size:var(--text-sm,13px)';
  const r = btn.getBoundingClientRect();
  menu.style.left = `${Math.max(8, Math.min(r.left, window.innerWidth - 210))}px`;
  menu.style.top = `${Math.min(r.bottom + 4, window.innerHeight - 60)}px`;
  menu.innerHTML = '<div style="padding:6px 10px;color:var(--text-muted,#a1a1b5)">Record narration with…</div>'
    + '<div data-narr-loading style="padding:8px 10px;color:var(--text-muted,#a1a1b5)">Loading voices…</div>';
  document.body.appendChild(menu);
  _narrMenuEl = menu;
  setTimeout(() => document.addEventListener('click', _onDocClickNarr, true), 0);

  const mkItem = (label, voiceVal) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.style.cssText = 'display:block;width:100%;text-align:left;padding:6px 10px;background:none;'
      + 'border:none;color:var(--text,#ececf1);cursor:pointer;border-radius:6px;font:inherit;white-space:nowrap';
    b.textContent = label;
    b.addEventListener('mouseenter', () => { b.style.background = 'var(--accent-soft,rgba(108,138,255,.12))'; });
    b.addEventListener('mouseleave', () => { b.style.background = 'none'; });
    b.addEventListener('click', (e) => { e.stopPropagation(); _closeNarrationMenu(); _narrationActivate(id, btn, name, voiceVal); });
    return b;
  };

  let voices = [];
  try { const mc = await import('../model-cache.js'); voices = await mc.getVoices(); } catch { /* no TTS provider */ }
  if (_narrMenuEl !== menu) return;   // closed while loading
  menu.querySelector('[data-narr-loading]')?.remove();
  menu.appendChild(mkItem('Default voice', null));
  for (const v of (Array.isArray(voices) ? voices : [])) {
    const rawId = typeof v === 'string' ? v : (v.id || v.name || v.voice_id || '');
    if (!rawId) continue;
    const provId = (typeof v === 'object' && v.provider_id) ? v.provider_id : '';
    const label = typeof v === 'string' ? v : (v.name || v.id || rawId);
    menu.appendChild(mkItem(label, provId ? `${provId}::${rawId}` : rawId));
  }
}

export async function summarizeFile(id) {
  const file = state.files.find(f => f.id === id);
  showToast(`Summarizing "${file?.name || 'file'}"…`, 'info', 2000);
  try {
    const model = window.app?.state?.currentModel || '';
    const resp = await summarize(id, model);
    if (!resp.ok) {
      let msg = `Summarize failed (${resp.status})`;
      try { const err = await resp.json(); if (err.error) msg = err.error; } catch { /* ignore */ }
      showToast(msg, 'error');
      return;
    }
    const data = await resp.json();
    if (file && data.summary) {
      file.description = data.summary;
      R.updateDetail();
      showToast('Summary added', 'success', 2000);
    }
  } catch (err) {
    console.warn('[files] summarize error:', err);
    showToast('Summarize failed — network error', 'error');
  }
}

// --- Bulk bar --------------------------------------------------------

export function updateBulkBar() {
  if (!state.el.bulkBar) return;
  const inTrash = state.currentSource === 'trash';
  // Visible whenever there's anything to act on: selection, trash view,
  // or select mode is on (even with 0 selected, we want the "Done" exit
  // visible and the count chip telling the user to tap items).
  const shouldShow = state.selection.size > 0 || inTrash || state.selectMode;
  if (shouldShow) {
    state.el.bulkBar.classList.add('visible');
    if (state.el.bulkCount) {
      if (inTrash && state.selection.size === 0) {
        state.el.bulkCount.textContent = 'Trash';
      } else if (state.selectMode && state.selection.size === 0) {
        state.el.bulkCount.textContent = 'Tap to select';
      } else {
        state.el.bulkCount.textContent = `${state.selection.size} selected`;
      }
    }
    // In select mode the "Deselect" button doubles as the mode-exit, so
    // relabel it "Done" to match iOS/Android conventions. Reverting the
    // label when mode is off keeps desktop's terser semantic.
    const deselectBtn = state.el.bulkBar.querySelector('[data-action="bulk-deselect"]');
    if (deselectBtn) {
      deselectBtn.textContent = state.selectMode ? 'Done' : 'Deselect';
    }
    const downloadBtn = state.el.bulkBar.querySelector('[data-action="bulk-download"]');
    const deleteBtn = state.el.bulkBar.querySelector('[data-action="bulk-delete"]');
    let restoreBtn = state.el.bulkBar.querySelector('[data-action="bulk-restore"]');
    let emptyBtn = state.el.bulkBar.querySelector('[data-action="bulk-empty-trash"]');
    if (inTrash) {
      if (downloadBtn) downloadBtn.style.display = 'none';
      if (deleteBtn) deleteBtn.style.display = 'none';
      if (!restoreBtn) {
        restoreBtn = document.createElement('button');
        restoreBtn.className = 'btn btn-sm';
        restoreBtn.dataset.action = 'bulk-restore';
        restoreBtn.textContent = 'Restore All';
        state.el.bulkBar.insertBefore(restoreBtn, state.el.bulkBar.querySelector('[data-action="bulk-deselect"]'));
      }
      restoreBtn.style.display = '';
      if (!emptyBtn) {
        emptyBtn = document.createElement('button');
        emptyBtn.className = 'btn btn-sm';
        emptyBtn.style.color = 'var(--error, #ef4444)';
        emptyBtn.dataset.action = 'bulk-empty-trash';
        emptyBtn.textContent = 'Empty Trash';
        state.el.bulkBar.insertBefore(emptyBtn, state.el.bulkBar.querySelector('[data-action="bulk-deselect"]'));
      }
      emptyBtn.style.display = '';
    } else {
      if (downloadBtn) downloadBtn.style.display = '';
      if (deleteBtn) deleteBtn.style.display = '';
      if (restoreBtn) restoreBtn.style.display = 'none';
      if (emptyBtn) emptyBtn.style.display = 'none';
    }
  } else {
    state.el.bulkBar.classList.remove('visible');
  }
}

// --- Bulk operations ------------------------------------------------

export async function bulkDownload() {
  const selected = state.files.filter(f => state.selection.has(f.id));
  if (selected.length === 0) return;
  if (selected.length === 1) { downloadFile(selected[0].id); return; }
  try {
    const resp = await zipDownload(selected.map(f => f.id));
    if (!resp.ok) return;
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'files.zip';
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) { console.warn('[files] zip download error:', err); }
}

export async function bulkDelete() {
  const count = state.selection.size;
  if (!count) return;
  // Split: LibriVox pins go to DELETE /api/media/pin/{id} (sentinel check
  // rejects them from /api/files/bulk-delete). Regular files keep the
  // existing trash flow. Mixed selection fires both in parallel.
  const selected = state.files.filter(f => state.selection.has(f.id));
  const librivoxIds = selected.filter(isBuiltinLibrivox).map(f => f.id);
  const regularIds  = selected.filter(f => !isBuiltinLibrivox(f)).map(f => f.id);
  const allLibrivox = regularIds.length === 0 && librivoxIds.length > 0;
  const noun = count > 1 ? 'items' : 'item';
  const ok = await inlineConfirm({
    message: allLibrivox
      ? `Unpin ${count} ${noun} from your library?`
      : `Move ${count} ${noun} to Trash?`,
    action: allLibrivox ? 'Unpin' : 'Move to Trash',
    danger: true,
  });
  if (!ok) return;
  try {
    const tasks = [];
    if (regularIds.length) tasks.push(bulkDeleteIds(regularIds));
    for (const id of librivoxIds) tasks.push(unpinLibrivox(id));
    const results = await Promise.allSettled(tasks);
    const anyFailed = results.some(r =>
      r.status === 'rejected' || (r.value && !r.value.ok));
    state.files = state.files.filter(f => !state.selection.has(f.id));
    if (state.selectMode) R.exitSelectMode();
    else R.deselectAll();
    R.renderGrid();
    _loadStats();
    if (librivoxIds.length) {
      window.dispatchEvent(new CustomEvent('media-servers:changed'));
    }
    if (anyFailed) showToast('Some items failed to remove', 'warning', 3000);
  } catch (err) { console.warn('[files] bulk delete error:', err); }
}

export async function emptyTrash() {
  const ok = await inlineConfirm({
    message: 'Permanently delete all trashed files?',
    action: 'Empty Trash', danger: true,
  });
  if (!ok) return;
  try {
    const resp = await purgeTrash();
    if (!resp.ok) return;
    state.files = [];
    R.deselectAll();
    R.renderGrid();
    _loadStats();
  } catch (err) { console.warn('[files] purge trash error:', err); }
}

export async function restoreAll() {
  const ids = state.files.map(f => f.id);
  if (!ids.length) return;
  try {
    const resp = await bulkRestoreIds(ids);
    if (!resp.ok) return;
    state.files = [];
    R.deselectAll();
    R.renderGrid();
    _loadStats();
  } catch (err) { console.warn('[files] bulk restore error:', err); }
}

// --- Context menu ---------------------------------------------------

export function showContextMenu(x, y, file) {
  hideContextMenu();
  const menu = document.createElement('div');
  menu.className = 'files-context-menu';

  if (state.currentSource === 'trash') {
    menu.innerHTML = `
      <button data-action="restore"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 105.64-11.36L1 10"/></svg> Restore</button>
      <div class="files-context-divider"></div>
      <button data-action="delete" class="danger"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg> Delete Permanently</button>
    `;
  } else {
    const previewBtn = isImage(file)
      ? `<button data-action="preview"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg> Preview</button>`
      : '';
    const projectBtn = isAppProject(file)
      ? `<button data-action="project"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg> Open in Workspace</button>`
      : '';
    // Generic archive preview entry — shown only when the file is an archive
    // AND not an app-builder project (which already has its own entry above).
    const archiveBtn = (isArchive(file) && !isAppProject(file))
      ? `<button data-action="archive-preview"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 8v13H3V8"/><rect x="1" y="3" width="22" height="5"/><line x1="10" y1="12" x2="14" y2="12"/></svg> Show contents</button>`
      : '';
    const convertBtns = isImage(file) ? _convertButtonsForImage(file) : '';
    const extractBtn = _isExtractable(file) ? _extractTextButton() : '';
    const docConvertBtns = isMarkdown(file) ? _documentConvertButtons() : '';
    // Media-server rows (Emby/Jellyfin episodes & movies, audiobookshelf
    // books, podcasts) get watch-state actions. The endpoint pushes
    // upstream too so marking watched here syncs to your phone / TV
    // / other clients. Section is omitted entirely for non-media rows
    // — these actions are meaningless for an uploaded PDF.
    const mediaBtns = isMediaServerFile(file) ? _mediaWatchButtons(file) : '';
    // Built-in LibriVox pins show "Unpin from library" instead of "Delete"
    // — the destructive action is removing a Discover pin, not trashing a
    // user file, and the backend route is different (see unpinLibrivoxAction).
    const destructiveBtn = isBuiltinLibrivox(file)
      ? `<button data-action="unpin" class="danger"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/><line x1="3" y1="3" x2="21" y2="21"/></svg> Unpin from library</button>`
      : `<button data-action="delete" class="danger"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/></svg> Delete</button>`;
    menu.innerHTML = `
      ${projectBtn}
      ${archiveBtn}
      ${previewBtn}
      ${convertBtns}
      ${docConvertBtns}
      ${extractBtn}
      ${mediaBtns}
      <button data-action="download"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg> Download</button>
      <button data-action="reference"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg> Reference in Chat</button>
      <button data-action="summarize"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg> Summarize with AI</button>
      <div class="files-context-divider"></div>
      <button data-action="rename"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 013 3L12 15l-4 1 1-4 9.5-9.5z"/></svg> Rename</button>
      <button data-action="copyname"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg> Copy Name</button>
      <div class="files-context-divider"></div>
      ${destructiveBtn}
    `;
  }

  menu.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-action]');
    if (!btn) return;
    hideContextMenu();
    const action = btn.dataset.action;
    if (action === 'project') P.openProject(file);
    else if (action === 'archive-preview') P.openMediaPreview(file.id, 'rendered');
    else if (action === 'preview') P.openGallery(file.id);
    else if (action === 'download') downloadFile(file.id);
    else if (action === 'reference') referenceInChat(file.id, file.name);
    else if (action === 'summarize') summarizeFile(file.id);
    else if (action === 'rename') R.startRename(file.id);
    else if (action === 'copyname') copyName(file.name);
    else if (action === 'restore') restoreFileAction(file.id);
    else if (action === 'delete') deleteFileAction(file.id);
    else if (action === 'unpin') unpinLibrivoxAction(file.id);
    else if (action === 'convert') convertImageAction(file, btn.dataset.target);
    else if (action === 'convert-doc') convertDocumentAction(file, btn.dataset.target);
    else if (action === 'extract-text') extractTextAction(file);
    else if (action === 'mark-watched') setMediaFinishedAction(file, true);
    else if (action === 'mark-unwatched') setMediaFinishedAction(file, false);
    else if (action === 'reset-progress') resetMediaProgressAction(file);
  });

  document.body.appendChild(menu);
  const rect = menu.getBoundingClientRect();
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  if (x + rect.width > vw) x = vw - rect.width - 8;
  if (y + rect.height > vh) y = vh - rect.height - 8;
  menu.style.left = `${Math.max(4, x)}px`;
  menu.style.top = `${Math.max(4, y)}px`;
  requestAnimationFrame(() => menu.classList.add('visible'));
  state.contextMenu = menu;
}

export function hideContextMenu() {
  if (state.contextMenu) {
    state.contextMenu.remove();
    state.contextMenu = null;
  }
}
