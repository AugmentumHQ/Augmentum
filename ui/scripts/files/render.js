/**
 * Files panel — rendering surfaces: grid cards, selection, detail panel,
 * inline previews, rename. Kept in one module because selection changes
 * cascade through all three (bulk bar, detail panel, card highlight).
 */

import { escapeHtml } from '../app.js';
import { renderMarkdown, highlightCodeDeferred } from '../chat/markdown.js';
import {
  state, HLJS_LANG_MAP, GRID_ICON, LIST_ICON, GALLERY_ICON,
} from './state.js';
import {
  getExt, iconForFile, tintKey,
  isImage, isVideo, isAudio, isPdf, isMarkdown, isHtml, isEpub, isOffice, isText, isArchive, isAppProject,
  imageNeedsServerRender, videoLikelyUnsupported,
  isMediaServerFile, isBuiltinLibrivox, mediaProgress, hasMediaCover, supportsReadAloud,
  humanSize, formatDate, recencyClass, dateBucket, renderTagPills, fetchPeek,
  observeVisiblePeeks,
} from './helpers.js';
import {
  downloadUrl, renderUrl, thumbUrl, mediaStreamUrl, mediaCoverUrl,
  pushMediaProgress,
  patchName, patchTags, suggestTags, fetchFileEntry, fetchMediaDetails,
  updateMediaPlaybackSelection, mediaBackdropUrl, personImageUrl,
  fetchPersonProfile,
} from './api.js';
import { updateBulkBar } from './actions.js';
import { openGallery, openMediaPreview, openVideoPreviewById } from './preview.js';

// When an episode reports playback progress, any cached Series detail
// (which holds the "next up" CTA) is now stale. Invalidate the load
// flag on every series in state.files so the next series-detail open
// refetches — cheap and lazy; no extra work until the user navigates
// back. Also re-render if a series is currently on screen.
if (typeof window !== 'undefined') {
  window.addEventListener('media-player:progress', () => {
    for (const f of state.files || []) {
      const meta = f?.source_metadata;
      if (meta && String(meta.entity_kind || '').toLowerCase() === 'series') {
        meta._videoDetailsLoaded = false;
      }
    }
    const id = [...(state.selection || [])][0];
    const current = state.detailOverrideFile
      || (state.files || []).find((f) => f.id === id);
    if (current && String(current.source_metadata?.entity_kind || '').toLowerCase() === 'series') {
      updateDetail();
    }
  });
}

// --- Selection --------------------------------------------------------

export function selectOnly(id) {
  state.detailOverrideFile = null;
  state.detailNavStack = [];
  state.selection.clear();
  if (id) state.selection.add(id);
  updateSelectionUI();
}

export function toggleSelect(id) {
  state.detailOverrideFile = null;
  state.detailNavStack = [];
  if (state.selection.has(id)) state.selection.delete(id);
  else state.selection.add(id);
  updateSelectionUI();
}

export function selectRange(fromIndex, toIndex) {
  state.detailOverrideFile = null;
  state.detailNavStack = [];
  const lo = Math.min(fromIndex, toIndex);
  const hi = Math.max(fromIndex, toIndex);
  for (let i = lo; i <= hi; i++) {
    if (state.files[i]) state.selection.add(state.files[i].id);
  }
  updateSelectionUI();
}

export function selectAll() {
  state.detailOverrideFile = null;
  state.detailNavStack = [];
  state.files.forEach(f => state.selection.add(f.id));
  updateSelectionUI();
}

export function deselectAll() {
  state.detailOverrideFile = null;
  state.detailNavStack = [];
  state.selection.clear();
  state.lastClickedIndex = -1;
  updateSelectionUI();
}

// Explicit multi-select mode — `deselectAll` is a pure selection clear
// (some call sites want to drop the highlight without kicking the user
// out of curation). Mode transitions go through these helpers so the
// grid class, toolbar button state, and selection stay in lockstep.
export function setSelectMode(on) {
  state.selectMode = !!on;
  state.el.grid?.classList.toggle('select-mode', state.selectMode);
  state.el.selectBtn?.classList.toggle('active', state.selectMode);
  state.el.selectBtn?.setAttribute(
    'aria-pressed', state.selectMode ? 'true' : 'false');
  updateSelectionUI();
}

export function exitSelectMode() {
  state.detailOverrideFile = null;
  state.detailNavStack = [];
  state.selection.clear();
  state.lastClickedIndex = -1;
  setSelectMode(false);
}

export function getSelectedFiles() {
  return state.files.filter(f => state.selection.has(f.id));
}

export function updateSelectionUI() {
  state.el.grid?.querySelectorAll('.files-card').forEach(card => {
    card.classList.toggle('selected', state.selection.has(card.dataset.id));
  });
  updateBulkBar();
  updateDetail();
}

// --- Card HTML --------------------------------------------------------

function _mediaServerVideoMeta(f) {
  const meta = f?.source_metadata || {};
  const entityKind = String(meta.entity_kind || '').toLowerCase();
  const libraryName = String(meta.library_name || '').trim();
  const seriesName = String(meta.series_name || '').trim();
  const year = Number(meta.year) || 0;
  const seasonNumber = Number(meta.season_number) || 0;
  const episodeNumber = Number(meta.episode_number) || 0;
  const unplayedCount = Number(meta.unplayed_count) || 0;

  if (entityKind === 'series') {
    const parts = ['Show'];
    if (libraryName) parts.push(libraryName);
    if (unplayedCount > 0) parts.push(`${unplayedCount} new`);
    return parts.join(' \u00B7 ');
  }
  if (entityKind === 'season') {
    const label = seasonNumber > 0 ? `Season ${seasonNumber}` : 'Season';
    return [label, seriesName || libraryName].filter(Boolean).join(' \u00B7 ');
  }
  if (entityKind === 'episode') {
    let marker = 'Episode';
    if (seasonNumber > 0 && episodeNumber > 0) {
      marker = `S${seasonNumber}E${episodeNumber}`;
    } else if (episodeNumber > 0) {
      marker = `Episode ${episodeNumber}`;
    }
    return [marker, seriesName || libraryName].filter(Boolean).join(' \u00B7 ');
  }
  if (entityKind === 'movie') {
    const parts = ['Movie'];
    if (libraryName) parts.push(libraryName);
    if (year > 0) parts.push(String(year));
    return parts.join(' \u00B7 ');
  }
  if (entityKind === 'music_video') {
    const parts = ['Music Video'];
    if (libraryName) parts.push(libraryName);
    if (year > 0) parts.push(String(year));
    return parts.join(' \u00B7 ');
  }
  return '';
}

function _cardMetaLabel(f, size, source) {
  const videoMeta = isMediaServerFile(f) && isVideo(f) ? _mediaServerVideoMeta(f) : '';
  if (videoMeta) return videoMeta;
  return [source, size].filter(Boolean).join(' \u00B7 ');
}

function _cardBadgeLabel(f, ext) {
  if (isMediaServerFile(f) && isVideo(f)) {
    const entityKind = String(f?.source_metadata?.entity_kind || '').toLowerCase();
    if (entityKind === 'series') return 'SHOW';
    if (entityKind === 'season') return 'SEASON';
    if (entityKind === 'episode') return 'EP';
    if (entityKind === 'movie') return 'MOVIE';
    if (entityKind === 'music_video') return 'MUSIC';
  }
  return ext ? ext.toUpperCase() : '';
}

/**
 * Render the card DOM for one file — card only, no time divider. The
 * divider is the caller's responsibility because incremental grid diffing
 * needs to rebuild dividers independently from reused card nodes.
 *
 * The `cacheKey` we stamp on the card encodes every input that would
 * change its rendered output, so _renderCards can tell "reuse this node
 * verbatim" apart from "the file's name/favorite/tags changed; rebuild
 * this one card". Cheap to compute, scales O(1) per card.
 */
export function cardOnlyHtml(f, i) {
  const size = humanSize(f.size_bytes);
  const source = f.source ? f.source.charAt(0).toUpperCase() + f.source.slice(1) : '';
  const meta = _cardMetaLabel(f, size, source);
  const ext = getExt(f.name);
  const badge = _cardBadgeLabel(f, ext);
  const sel = state.selection.has(f.id) ? ' selected' : '';
  const focused = i === state.focusedIndex ? ' focused' : '';
  const rec = recencyClass(f.created_at);
  const recCls = rec ? ` ${rec}` : '';
  const peekable = isText(f);
  const iconSvg = iconForFile(f);
  const tint = tintKey(f);
  const imageLike = isImage(f);
  // Cover art for media-server rows: the server confirmed it has one,
  // so we don't waste a round-trip checking.
  //
  // For image rows we hit the cacheable thumbnail route first
  // (300px WebP, ~30KB, Cache-Control: max-age=1y) and fall back to
  // the full PNG via the <img onerror> chain only if no thumbnail
  // exists yet. This drops a 50-image grid page from ~100 MB of full
  // PNGs to ~1.5 MB of WebP thumbs. ``thumbFallback`` is empty when
  // the primary already IS the full asset (server-rendered or media
  // cover) — in that case the onerror just removes the broken img.
  let thumbSrc = '';
  let thumbFallback = '';
  if (f.thumbnail) {
    thumbSrc = f.thumbnail;
  } else if (imageLike) {
    thumbSrc = thumbUrl(f.id);
    thumbFallback = downloadUrl(f.id);
  } else if (hasMediaCover(f)) {
    thumbSrc = mediaCoverUrl(f.id);
  }
  const fallbackAttr = thumbFallback
    ? ` data-fallback="${escapeHtml(thumbFallback)}"` : '';
  // The onerror handler tries the fallback URL once (if present)
  // before giving up. Setting onerror=null first prevents an infinite
  // loop if the fallback also 404s.
  const onerror = thumbFallback
    ? `if(this.dataset.fallback&&this.src!==this.dataset.fallback){this.onerror=null;this.src=this.dataset.fallback;}else{this.remove();}`
    : `this.remove()`;
  const iconOrThumb = thumbSrc
    ? `<div class="files-card-thumb-img" data-tint="${escapeHtml(tint)}">
         <span class="files-card-thumb-fallback">${iconSvg}</span>
         <img src="${escapeHtml(thumbSrc)}" alt="" loading="lazy" decoding="async"${fallbackAttr} onerror="${onerror}">
       </div>`
    : `<div class="files-card-icon-badge" data-tint="${escapeHtml(tint)}">${iconSvg}</div>`;
  const peekBlock = (peekable && state.currentView === 'gallery')
    ? `<pre class="files-card-peek" data-ext="${escapeHtml(ext)}" aria-hidden="true"></pre>`
    : '';
  const tagBlock = renderTagPills(f.tags);
  // Media-server rows carry a progress %; thin bar along the card bottom
  // is the card-level affordance for "where you left off". Returns empty
  // string for non-media rows so the DOM stays clean.
  const progress = mediaProgress(f);
  const progressBlock = progress > 0
    ? `<div class="files-card-progress" aria-hidden="true"><span style="width:${(progress * 100).toFixed(1)}%"></span></div>`
    : '';

  // Cache key: everything that influences the rendered output except
  // position-only state (focused, index) which we can mutate on the
  // existing node after reuse. Tags list stringified so reordering shows
  // as a cache miss.
  const cacheKey = [
    f.id,
    f.name,
    f.is_favorite ? 1 : 0,
    f.size_bytes || 0,
    f.source || '',
    f.kind || '',
    f.thumbnail || '',
    (f.tags || []).join('|'),
    progress.toFixed(2),
    state.currentView,
    recCls,
  ].join('\x1f');

  return `
  <div class="files-card${sel}${focused}${recCls}"
       data-id="${escapeHtml(f.id)}"
       data-index="${i}"
       data-source="${escapeHtml(f.source || '')}"
       data-kind="${escapeHtml(f.kind || '')}"
       data-tint="${escapeHtml(tint)}"
       data-peekable="${peekable ? '1' : '0'}"
       data-cachekey="${escapeHtml(cacheKey)}"
       tabindex="0">
    ${peekBlock}
    ${iconOrThumb}
    <div class="files-card-info">
      <div class="files-card-name">${escapeHtml(f.name)}</div>
      <div class="files-card-meta">${escapeHtml(meta)}</div>
    </div>
    ${tagBlock}
    ${badge ? `<span class="files-card-ext">${escapeHtml(badge)}</span>` : ''}
    <span class="files-card-check" aria-hidden="true">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>
    </span>
    <button class="files-card-fav${f.is_favorite ? ' active' : ''}" data-fav-id="${escapeHtml(f.id)}" title="Toggle favorite">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="${f.is_favorite ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="2"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>
    </button>
    ${(isAudio(f) || isVideo(f)) ? `<button class="files-card-playlist" data-playlist-id="${escapeHtml(f.id)}" data-playlist-kind="${isVideo(f) ? 'video' : 'audio'}" data-playlist-name="${escapeHtml(f.name)}" title="Add to Grove playlist" aria-label="Add to playlist">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="13" y2="18"/><line x1="18" y1="15" x2="18" y2="21"/><line x1="15" y1="18" x2="21" y2="18"/></svg>
    </button>` : ''}
    ${progressBlock}
  </div>`;
}

function _cardHtml(f, i, showTimeSpine) {
  let html = '';
  if (showTimeSpine) {
    const bucket = dateBucket(f.created_at);
    if (bucket !== state.lastRenderedBucket) {
      state.lastRenderedBucket = bucket;
      html += `<div class="files-time-divider" aria-hidden="true"><span>${escapeHtml(bucket)}</span></div>`;
    }
  }
  html += cardOnlyHtml(f, i);
  return html;
}

// --- Full + incremental grid render ----------------------------------

export function renderGrid() {
  state.el.grid?.classList.remove('list-view', 'gallery-view');
  if (state.currentView === 'list') state.el.grid?.classList.add('list-view');
  else if (state.currentView === 'gallery') state.el.grid?.classList.add('gallery-view');
  _renderCards();
  updateViewToggle();
  updateBulkBar();
  updateSentinel();
  observeSentinel();
  observeVisiblePeeks();
}

/**
 * Render the files grid incrementally. Sort/filter/view-toggle changes
 * used to nuke the whole grid via innerHTML=, which dropped every card's
 * event listeners, hover state, focus, and pending intersection-observer
 * subscriptions — and on a power user's 500+ file collection, ate visible
 * frames. This version reuses existing card DOM nodes by id and
 * cacheKey: unchanged cards get moved into their new positions instead
 * of being rebuilt from scratch. Time dividers are rebuilt either way
 * since they're cheap and their placement depends on neighbour order.
 *
 * Net effect: refiltering a grid of 500 files goes from touching 500 DOM
 * subtrees to touching only the ones that entered/left/changed.
 */
function _renderCards() {
  if (!state.el.grid) return;
  if (!state.files.length) {
    // Scope-aware empty state. When the user lands on Cloud with no
    // connected media servers, show a direct CTA to the Media Servers
    // overlay — the blank grid with "no files found" was frustratingly
    // opaque for first-time cloud visits. Local scope keeps the generic
    // copy because local content can accumulate from many pathways
    // (artifacts, uploads, image gen) and a single CTA would be wrong.
    const isCloud = state.currentScope === 'cloud' &&
      (state.currentSource === 'all' ||
       state.currentSource === 'audiobooks' ||
       state.currentSource === 'podcasts' ||
       state.currentSource === 'comics' ||
       state.currentSource === 'shows' ||
       state.currentSource === 'movies' ||
       state.currentSource === 'music_videos');
    if (isCloud) {
      state.el.grid.innerHTML = `
        <div class="files-empty">
          <span class="files-empty-icon">\u2601\uFE0F</span>
          <span class="files-empty-text">No connected services yet</span>
          <span class="files-empty-hint">Connect Audiobookshelf, Emby, Jellyfin, Suwayomi, Komga, or another media server to browse your library here.</span>
          <button class="files-empty-cta" type="button" data-action="open-media-servers">
            Connect a media server
          </button>
        </div>
      `;
      // Wire the CTA — dynamic import so the media-servers module only
      // loads if the user actually needs it (idle cost stays zero).
      state.el.grid.querySelector('[data-action="open-media-servers"]')
        ?.addEventListener('click', async () => {
          const m = await import('../media-servers.js');
          m.openMediaServers?.();
        });
      return;
    }
    state.el.grid.innerHTML = `
      <div class="files-empty">
        <span class="files-empty-icon">\u{1F4C2}</span>
        <span class="files-empty-text">No files found</span>
        <span class="files-empty-hint">Files from artifacts, images, documents, and more will appear here</span>
      </div>
    `;
    return;
  }

  const grid = state.el.grid;
  const showTimeSpine = (state.currentSort === 'newest' || state.currentSort === 'oldest');
  state.lastRenderedBucket = '';

  // Snapshot existing card nodes by id. We detach (not remove) — reused
  // nodes get re-appended below, stale ones fall out of scope and GC.
  const existing = new Map();
  grid.querySelectorAll('.files-card').forEach(el => {
    const id = el.dataset.id;
    if (id) existing.set(id, el);
  });

  // Build the new list into a DocumentFragment so the browser only
  // reflows once when we swap contents.
  const frag = document.createDocumentFragment();
  let lastBucket = '';
  const scratch = document.createElement('div');

  for (let i = 0; i < state.files.length; i++) {
    const f = state.files[i];

    if (showTimeSpine) {
      const bucket = dateBucket(f.created_at);
      if (bucket !== lastBucket) {
        lastBucket = bucket;
        const divider = document.createElement('div');
        divider.className = 'files-time-divider';
        divider.setAttribute('aria-hidden', 'true');
        divider.innerHTML = `<span>${escapeHtml(bucket)}</span>`;
        frag.appendChild(divider);
      }
    }

    const prev = existing.get(f.id);
    const freshHtml = cardOnlyHtml(f, i);
    // Reuse only when the new card's cacheKey matches the old one —
    // otherwise something visible about the file changed (rename,
    // favorite, tags, progress) and we need a fresh DOM so the card
    // reflects the new state.
    if (prev) {
      existing.delete(f.id);
      const prevKey = prev.dataset.cachekey;
      scratch.innerHTML = freshHtml;
      const nextCard = scratch.firstElementChild;
      if (nextCard && prevKey === nextCard.dataset.cachekey) {
        // Cache hit — reuse the old node as-is, but sync the positional
        // attributes that aren't part of the cache key.
        if (prev.dataset.index !== String(i)) prev.dataset.index = String(i);
        prev.classList.toggle('focused', i === state.focusedIndex);
        frag.appendChild(prev);
      } else if (nextCard) {
        // Cache miss — insert the newly-built node instead.
        frag.appendChild(nextCard);
      }
    } else {
      scratch.innerHTML = freshHtml;
      const nextCard = scratch.firstElementChild;
      if (nextCard) frag.appendChild(nextCard);
    }
  }

  // replaceChildren detaches orphan nodes (cards no longer present in
  // state.files plus every old divider) in a single operation.
  grid.replaceChildren(frag);
}

export function appendCards(batch, startIndex) {
  if (!state.el.grid || !batch.length) return;
  const showTimeSpine = (state.currentSort === 'newest' || state.currentSort === 'oldest');
  const parts = batch.map((f, i) => _cardHtml(f, startIndex + i, showTimeSpine));
  const sentinel = state.el.grid.querySelector('.files-load-sentinel');
  if (sentinel) sentinel.remove();
  const frag = document.createElement('div');
  frag.innerHTML = parts.join('');
  while (frag.firstChild) state.el.grid.appendChild(frag.firstChild);
}

export function updateViewToggle() {
  if (!state.el.viewToggle) return;
  if (state.currentView === 'grid') {
    state.el.viewToggle.innerHTML = LIST_ICON;
    state.el.viewToggle.title = 'Switch to list view';
  } else if (state.currentView === 'list') {
    state.el.viewToggle.innerHTML = GALLERY_ICON;
    state.el.viewToggle.title = 'Switch to gallery view';
  } else {
    state.el.viewToggle.innerHTML = GRID_ICON;
    state.el.viewToggle.title = 'Switch to grid view';
  }
}

// --- Sentinel + scroll observer --------------------------------------

export function setupScrollObserver(onLoadMore) {
  if (state.scrollObserver || !state.el.grid) return;
  state.scrollObserver = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      if (entry.isIntersecting && state.hasMore && !state.loading) onLoadMore();
    }
  }, { root: state.el.grid, rootMargin: '300px 0px' });
}

export function observeSentinel() {
  if (!state.scrollObserver) return;
  const sentinel = state.el.grid?.querySelector('.files-load-sentinel');
  if (sentinel) state.scrollObserver.observe(sentinel);
}

export function updateSentinel() {
  if (!state.el.grid) return;
  const existing = state.el.grid.querySelector('.files-load-sentinel');
  if (state.hasMore) {
    if (!existing) {
      const el = document.createElement('div');
      el.className = 'files-load-sentinel';
      el.setAttribute('aria-hidden', 'true');
      state.el.grid.appendChild(el);
    } else {
      state.el.grid.appendChild(existing);
    }
  } else if (existing) {
    existing.remove();
  }
}

// --- Detail panel -----------------------------------------------------

export function updateDetail() {
  if (!state.el.detail) return;
  // In select mode the user is curating a batch, not inspecting one row.
  // Keep the detail panel out of the way regardless of selection size.
  if (state.selectMode || state.selection.size !== 1) {
    state.el.detail.classList.add('hidden');
    return;
  }
  const id = [...state.selection][0];
  const file = state.detailOverrideFile || state.files.find(f => f.id === id);
  if (!file) { state.el.detail.classList.add('hidden'); return; }

  const token = ++state.detailToken;

  // Media-server rows (audiobooks, and eventually any media library with
  // rich metadata) fan out to a richer render path that layers additional
  // sections — chapters, related strip, author/narrator pivot — on top of
  // the same hero/identity/primary/actions scaffolding every other kind uses.
  if (isMediaServerFile(file) && file.kind === 'audio') {
    _renderMediaServerDetail(file, token);
  } else {
    _renderGenericDetail(file, token);
  }

  _wireDetail(file);
}

function _closeButtonHtml() {
  return `<button class="files-detail-close icon-btn small" title="Close">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
  </button>`;
}

function _tagEditorHtml(file) {
  const tagPills = (file.tags || []).map(t =>
    `<span class="files-detail-tag">${escapeHtml(t)}<button class="files-tag-remove" data-remove-tag="${escapeHtml(t)}">&times;</button></span>`
  ).join('');
  return `
    <div class="files-detail-tags-editor">
      ${tagPills}
      <div class="files-tag-input-wrap">
        <input class="files-tag-input" type="text" placeholder="Add tag..." autocomplete="off"
               spellcheck="false" data-tag-file-id="${escapeHtml(file.id)}">
        <div class="files-tag-suggest" hidden></div>
      </div>
    </div>
  `;
}

/* Media-server detail — audiobooks (LibriVox / audiobookshelf) today,
 * broader media libraries later. Uses the SAME hero / identity / primary /
 * actions scaffolding as every other file kind; what's added is the
 * rich-metadata layer: author + narrator meta rows, chapters list,
 * related-author strip, genre chips, external links, progress visualization.
 * The wiring selectors (`[data-media-play]`, `.files-abs-chapter`,
 * `[data-related-*]`, `.files-abs-person`, `[data-chapter-list]`) are the
 * contract `_wireMediaServerPreview` binds to — if you rename one of these,
 * update that function too. */
function _renderMediaServerDetail(file, token) {
  const m = _mediaServerFields(file);
  const cta = _mediaServerCtaState(m);

  const coverHtml = m.coverSrc
    ? `<img src="${escapeHtml(m.coverSrc)}" alt="${escapeHtml(file.name)}" loading="lazy" decoding="async" onerror="this.closest('.files-abs-cover').classList.add('is-missing')">`
    : `<svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 4h11a4 4 0 0 1 4 4v12M4 4v16M4 20h11a4 4 0 0 0 4 0"/></svg>`;
  const coverCard = `<div class="files-abs-cover${m.coverSrc ? '' : ' is-missing'}">${coverHtml}</div>`;

  const progressBar = (!m.isFinished && m.progressPct > 0)
    ? `<div class="files-abs-progress" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${m.progressPct.toFixed(0)}">
         <div class="files-abs-progress-track"><div class="files-abs-progress-fill" style="width:${Math.min(100, Math.max(0, m.progressPct)).toFixed(1)}%"></div></div>
       </div>`
    : '';

  const metaRowsBlock = _mediaServerMetaRowsHtml(m);

  const relatedBlock = `
    <div class="files-abs-related" data-related-host hidden>
      <div class="files-abs-related-header">
        <span class="files-abs-related-title" data-related-label></span>
        <button type="button" class="files-abs-related-close" data-related-close aria-label="Close">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
        </button>
      </div>
      <div class="files-abs-related-strip" data-related-strip>
        <div class="files-abs-related-skeleton" aria-hidden="true">
          <span></span><span></span><span></span><span></span><span></span>
        </div>
      </div>
    </div>`;

  const chapterCount = m.chapters.length;
  const chaptersBlock = m.entityKind === 'podcast'
    ? _podcastEpisodesSectionHtml(m)
    : `
      <details class="files-detail-section files-abs-chapters" ${chapterCount ? 'open' : ''}>
        <summary>Chapters <span class="files-abs-chip" data-chapter-count>${chapterCount || '…'}</span></summary>
        <ol class="files-abs-chapter-list" role="list" data-chapter-list>
          ${m.chapters.map((c, idx) => _chapterRowHtml(c, idx)).join('')}
          ${!chapterCount ? `<li class="files-abs-chapter-empty">Loading chapters…</li>` : ''}
        </ol>
      </details>`;

  const genresBlock = m.genres.length
    ? `<div class="files-detail-inline-chips">${m.genres.map(g =>
         `<span class="files-chip files-chip-muted">${escapeHtml(String(g))}</span>`
       ).join('')}</div>`
    : '';

  const linksBlock = m.links.length
    ? `<div class="files-detail-inline-links">${m.links.map(l =>
         `<a href="${escapeHtml(l.href)}" target="_blank" rel="noopener noreferrer" class="files-abs-link">${escapeHtml(l.label)}</a>`
       ).join('')}</div>`
    : '';

  const descBlock = m.description
    ? `<details class="files-detail-section">
         <summary>${m.entityKind === 'podcast' ? 'About this podcast' : 'About this book'}</summary>
         <p class="files-detail-desc">${escapeHtml(m.description)}</p>
       </details>`
    : '';

  const ext = getExt(file.name);
  const sizeLabel = humanSize(file.size_bytes);

  state.el.detail.innerHTML = `
    <div class="files-detail-topbar">${_closeButtonHtml()}</div>
    <div class="files-detail-hero files-detail-hero-audiobook">${coverCard}</div>
    <div class="files-detail-identity">
      <h2 class="files-detail-title">${escapeHtml(file.name)}</h2>
      ${cta.statusChip ? `<div class="files-detail-chips">${cta.statusChip}</div>` : ''}
      ${progressBar}
    </div>
    ${metaRowsBlock}
    ${cta.hidden ? '' : `
      <div class="files-detail-primary">
        <button type="button" class="btn btn-primary files-primary-cta" data-media-play="${escapeHtml(file.id)}" data-cta-kind="${escapeHtml(cta.kind)}">
          ${cta.icon}
          <span>${escapeHtml(cta.label)}</span>
        </button>
        ${cta.secondary ? `
          <button type="button" class="files-primary-secondary" data-media-restart="${escapeHtml(file.id)}">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <polyline points="1 4 1 10 7 10"/>
              <path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/>
            </svg>
            <span>${escapeHtml(cta.secondary.label)}</span>
          </button>
        ` : ''}
      </div>
    `}
    <div class="files-detail-actions">
      <button class="btn btn-sm files-icon-btn" data-action="download" data-id="${escapeHtml(file.id)}" title="Download" aria-label="Download">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      </button>
      <button class="btn btn-sm files-icon-btn" data-action="reference" data-id="${escapeHtml(file.id)}" data-name="${escapeHtml(file.name)}" title="Reference in Chat" aria-label="Reference in Chat">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
      </button>
      <button class="btn btn-sm files-icon-btn" data-action="summarize" data-id="${escapeHtml(file.id)}" title="Summarize with AI" aria-label="Summarize">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
      </button>
      ${(isAudio(file) || isVideo(file)) ? `<button class="btn btn-sm files-icon-btn" data-action="add-to-playlist" data-id="${escapeHtml(file.id)}" data-name="${escapeHtml(file.name)}" data-kind="${isVideo(file) ? 'video' : 'audio'}" title="Add to Grove playlist" aria-label="Add to playlist">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="13" y2="18"/><line x1="18" y1="15" x2="18" y2="21"/><line x1="15" y1="18" x2="21" y2="18"/></svg>
      </button>` : ''}
      ${_readAlongControlHtml(file, m)}
      ${isBuiltinLibrivox(file) ? `<button class="btn btn-sm files-icon-btn" data-action="unpin" data-id="${escapeHtml(file.id)}" title="Unpin from library" aria-label="Unpin from library">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/><line x1="3" y1="3" x2="21" y2="21"/></svg>
      </button>` : ''}
    </div>
    ${relatedBlock}
    ${chaptersBlock}
    ${genresBlock}
    ${linksBlock}
    ${descBlock}
    <details class="files-detail-section">
      <summary>Details</summary>
      <div class="files-detail-props">
        ${ext ? `<div class="files-detail-row"><span class="files-detail-label">Type</span><span>${escapeHtml(ext.toUpperCase())}</span></div>` : ''}
        <div class="files-detail-row"><span class="files-detail-label">Size</span><span>${escapeHtml(sizeLabel)}</span></div>
        <div class="files-detail-row"><span class="files-detail-label">Source</span><span>${escapeHtml(file.source || '')}</span></div>
        <div class="files-detail-row"><span class="files-detail-label">Created</span><span>${escapeHtml(formatDate(file.created_at))}</span></div>
        ${file.mime_type ? `<div class="files-detail-row"><span class="files-detail-label">MIME</span><span>${escapeHtml(file.mime_type)}</span></div>` : ''}
      </div>
    </details>
    ${_tagEditorHtml(file)}
    ${cta.hidden ? '' : `
      <div class="files-detail-footer">
        <button type="button" class="btn btn-primary files-footer-cta" data-media-play="${escapeHtml(file.id)}" data-cta-kind="${escapeHtml(cta.kind)}">
          ${cta.icon}
          <span>${escapeHtml(cta.label)}</span>
        </button>
      </div>
    `}
  `;
  state.el.detail.classList.remove('hidden');
}

function _readAlongControlHtml(file, m) {
  // Three renderable states. 'unavailable' / 'missing' / empty (and no
  // Gutenberg source at all) all hide the control entirely — no silent
  // "button that does nothing" UI. The external `url_text_source` link
  // still surfaces under Links for readers who want to jump to the
  // Gutenberg page directly.
  if (m.gutenbergStatus === 'fetched') {
    const wc = m.gutenbergWordCount;
    const wcLabel = wc ? `<span class="files-readalong-wc">${wc.toLocaleString()} words</span>` : '';
    return `<button type="button" class="btn btn-sm files-readalong-btn" data-action="read-along" data-id="${escapeHtml(file.id)}" data-name="${escapeHtml(file.name)}" title="Open read-along text">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
      <span>Read along</span>${wcLabel}
    </button>`;
  }
  if (m.gutenbergStatus === 'fetching') {
    return `<span class="files-readalong-pending" data-readalong-pending data-id="${escapeHtml(file.id)}" title="Downloading Project Gutenberg text">
      <svg class="files-readalong-spin" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
      <span>Fetching text…</span>
    </span>`;
  }
  return '';
}

function _mediaServerFields(file) {
  const meta = file.source_metadata || {};
  const entityKind = String(meta.entity_kind || meta.library_kind || '').trim().toLowerCase();
  const translators = Array.isArray(meta.translators)
    ? meta.translators.filter(t => t && t.name).map(t => t.name) : [];
  const links = [
    meta.librivox_url    ? { label: 'LibriVox',       href: meta.librivox_url }    : null,
    meta.url_text_source ? { label: 'Read the text',  href: meta.url_text_source } : null,
    meta.url_project     ? { label: 'About',          href: meta.url_project }     : null,
    meta.url_zip_file    ? { label: 'Download MP3s',  href: meta.url_zip_file }    : null,
    meta.url_rss         ? { label: 'RSS',            href: meta.url_rss }         : null,
    meta.url_other       ? { label: 'More',           href: meta.url_other }       : null,
  ].filter(Boolean);
  // Read-along state — the backend populates these after the
  // gutenberg_fetch background job finishes for a pinned LibriVox book.
  // Values: 'fetched' | 'fetching' | 'unavailable' | 'missing' | ''.
  // Empty + a url_text_source pointing at gutenberg.org is treated as
  // "fetching" so the UI shows a pending indicator while the job runs.
  const hasGutenbergSource = /gutenberg\.org/i.test(String(meta.url_text_source || ''));
  let gutenbergStatus = String(meta.gutenberg_status || '');
  if (!gutenbergStatus && hasGutenbergSource) gutenbergStatus = 'fetching';

  const rawChapters = Array.isArray(meta.chapters) ? meta.chapters : [];
  const episodes = Array.isArray(meta.children)
    ? meta.children.filter((child) => child && child.episode_id)
    : [];
  const durationSec = Number(meta.duration_s) || (Number(meta.duration_ms) || 0) / 1000;
  const currentTimeSec = Number(meta.current_time_s) || 0;
  return {
    coverSrc:        hasMediaCover(file) ? mediaCoverUrl(file.id) : '',
    // Chapters are enriched with a derived state (played/in-progress/
    // unplayed) + progress fraction so the list can render a comics-style
    // state rail without a second pass. Backend only tracks book-level
    // current_time_s; per-chapter state is computed from the chapter
    // boundaries and the book's current position.
    chapters:        _deriveChapterStates(rawChapters, currentTimeSec, durationSec),
    durationSec,
    currentTimeSec,
    progressPct:     Number(meta.progress_pct) || 0,
    author:          String(meta.author || '').trim(),
    narrator:        String(meta.narrator || '').trim(),
    description:     String(meta.description || '').trim(),
    isFinished:      !!meta.is_finished,
    publishedYear:   String(meta.copyright_year || meta.published_year || '').trim(),
    language:        String(meta.language || '').trim(),
    genres:          Array.isArray(meta.genres) ? meta.genres.filter(Boolean) : [],
    translators,
    links,
    gutenbergStatus,
    gutenbergWordCount: Number(meta.gutenberg_word_count) || 0,
    hasGutenbergSource,
    entityKind,
    episodes,
    selectedEpisodeId: String(meta.selected_episode_id || '').trim(),
    selectedEpisodeTitle: String(meta.selected_episode_title || '').trim(),
    playable: entityKind === 'podcast'
      ? !!String(meta.selected_episode_id || '').trim()
      : !!meta.stream_path,
  };
}

function _mediaServerCtaState(m) {
  if (m.entityKind === 'podcast') {
    if (!m.playable) {
      return {
        label: '',
        kind: 'podcast-unselected',
        icon: _playIcon(),
        statusChip: '',
        secondary: null,
        hidden: true,
      };
    }
    const selectedChip = m.selectedEpisodeTitle
      ? `<span class="files-chip files-chip-muted">${escapeHtml(m.selectedEpisodeTitle)}</span>`
      : '';
    if (m.isFinished) {
      return {
        label: 'Play episode again',
        kind: 'finished',
        icon: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><polyline points="3 3 3 8 8 8"/></svg>',
        statusChip: selectedChip,
        secondary: null,
      };
    }
    if (m.currentTimeSec > 0) {
      const remaining = Math.max(0, (m.durationSec || 0) - m.currentTimeSec);
      const remainingLabel = remaining > 0 && m.durationSec > 0
        ? `${_fmtDurationLoose(remaining)} left`
        : _fmtTimecode(m.currentTimeSec);
      return {
        label: `Resume episode · ${remainingLabel}`,
        kind: 'resume',
        icon: _playIcon(),
        statusChip: selectedChip,
        secondary: { kind: 'restart', label: 'Restart episode' },
      };
    }
    return {
      label: 'Play selected episode',
      kind: 'play',
      icon: _playIcon(),
      statusChip: selectedChip || (
        m.durationSec > 0
          ? `<span class="files-chip files-chip-muted">${escapeHtml(_fmtDurationLoose(m.durationSec))}</span>`
          : ''
      ),
      secondary: null,
    };
  }
  // Warmer CTA voice — "Start listening" beats "Play", a human-readable
  // "Xm left" beats a full timecode, and the resume state exposes a
  // secondary "Start from the beginning" affordance that mirrors the
  // comics reader's Start-from-Ch.1 pattern.
  if (m.isFinished) {
    return {
      label:      'Listen again',
      kind:       'finished',
      icon:       '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><polyline points="3 3 3 8 8 8"/></svg>',
      statusChip: '<span class="files-chip files-chip-finished">Finished</span>',
      secondary:  null,
    };
  }
  if (m.currentTimeSec > 0) {
    // Compute "time left" from known duration. Falls back to a plain
    // timecode when we don't know the total (shouldn't happen in practice
    // but keeps the label sensible if duration arrives late).
    const remaining = Math.max(0, (m.durationSec || 0) - m.currentTimeSec);
    const remainingLabel = remaining > 0 && m.durationSec > 0
      ? `${_fmtDurationLoose(remaining)} left`
      : _fmtTimecode(m.currentTimeSec);
    return {
      label:      `Resume · ${remainingLabel}`,
      kind:       'resume',
      icon:       _playIcon(),
      statusChip: `<span class="files-chip files-chip-resume">${m.progressPct.toFixed(0)}% played</span>`,
      secondary:  { kind: 'restart', label: 'Start from the beginning' },
    };
  }
  return {
    label:      'Start listening',
    kind:       'play',
    icon:       _playIcon(),
    statusChip: m.durationSec > 0
      ? `<span class="files-chip files-chip-muted">${escapeHtml(_fmtDurationLoose(m.durationSec))}</span>`
      : '',
    secondary:  null,
  };
}

function _mediaServerMetaRowsHtml(m) {
  const authorEl = m.author
    ? `<button type="button" class="files-abs-person" data-related-by="author" title="Show other books by this author">${escapeHtml(m.author)}</button>`
    : '';
  const narratorEl = m.narrator
    ? `<button type="button" class="files-abs-person" data-related-by="narrator" title="Show other books narrated by ${escapeHtml(m.narrator)}">${escapeHtml(m.narrator)}</button>`
    : '';
  const translatorLabel = m.translators.length === 0 ? ''
    : m.translators.length === 1 ? m.translators[0]
    : m.translators.length <= 3 ? m.translators.join(', ')
    : `${m.translators[0]} + ${m.translators.length - 1} others`;
  const rows = [
    m.author   ? `<div class="files-detail-meta-row"><span class="files-detail-meta-key">By</span>${authorEl}</div>` : '',
    m.narrator ? `<div class="files-detail-meta-row"><span class="files-detail-meta-key">Narrated by</span>${narratorEl}</div>` : '',
    translatorLabel ? `<div class="files-detail-meta-row"><span class="files-detail-meta-key">Translated by</span><span>${escapeHtml(translatorLabel)}</span></div>` : '',
    m.durationSec > 0 && !m.isFinished ? `<div class="files-detail-meta-row"><span class="files-detail-meta-key">Length</span><span>${escapeHtml(_fmtDuration(m.durationSec))}</span></div>` : '',
    m.publishedYear ? `<div class="files-detail-meta-row"><span class="files-detail-meta-key">Published</span><span>${escapeHtml(m.publishedYear)}</span></div>` : '',
    m.language ? `<div class="files-detail-meta-row"><span class="files-detail-meta-key">Language</span><span>${escapeHtml(m.language)}</span></div>` : '',
  ].filter(Boolean).join('');
  return rows ? `<div class="files-detail-meta-rows">${rows}</div>` : '';
}

function _mediaServerVideoChildren(file) {
  const children = file?.source_metadata?.children;
  return Array.isArray(children) ? children.filter(c => c && c.file_id) : [];
}

function _attachVideoNavigation(file, siblings, index) {
  if (!file || !Array.isArray(siblings)) return file;
  file._videoNav = {
    siblings: siblings
      .filter((child) => child && child.file_id)
      .map((child) => ({
        file_id: child.file_id,
        name: child.name || 'Untitled',
      })),
    index: Number(index) || 0,
  };
  return file;
}

function _videoPlaybackState(file) {
  const playback = file?.source_metadata?.playback;
  return playback && Array.isArray(playback.media_sources) ? playback : null;
}

function _videoNeedsPlaybackDetails(file) {
  if (!isMediaServerFile(file) || file?.kind !== 'video') return false;
  const entityKind = String(file?.source_metadata?.entity_kind || '').toLowerCase();
  return ['movie', 'episode', 'music_video'].includes(entityKind);
}

function _videoDetailsLoadedEnough(file) {
  const meta = file?.source_metadata || {};
  if (!meta._videoDetailsLoaded) return false;
  if (!_videoNeedsPlaybackDetails(file)) return true;
  return !!_videoPlaybackState(file);
}

function _selectedPlaybackSource(playback) {
  if (!playback?.media_sources?.length) return null;
  return playback.media_sources.find((source) => source?.id === playback.selected_media_source_id)
    || playback.media_sources[0];
}

function _chooseVideoTrackIndex(tracks, preferred, allowNone = false) {
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

function _applyVideoPlaybackSelection(file, patch = {}) {
  const playback = _videoPlaybackState(file);
  if (!playback?.media_sources?.length) return null;
  const hasMediaSource = Object.prototype.hasOwnProperty.call(patch, 'mediaSourceId');
  const hasAudio = Object.prototype.hasOwnProperty.call(patch, 'audioStreamIndex');
  const hasSubtitle = Object.prototype.hasOwnProperty.call(patch, 'subtitleStreamIndex');

  const nextSourceId = hasMediaSource
    ? (patch.mediaSourceId || playback.media_sources[0]?.id || '')
    : (playback.selected_media_source_id || playback.media_sources[0]?.id || '');
  const selectedSource = playback.media_sources.find((source) => source?.id === nextSourceId)
    || playback.media_sources[0];
  const nextAudio = _chooseVideoTrackIndex(
    selectedSource?.audio_tracks || [],
    hasAudio ? patch.audioStreamIndex : playback.selected_audio_stream_index,
    false,
  );
  const nextSubtitle = _chooseVideoTrackIndex(
    selectedSource?.subtitle_tracks || [],
    hasSubtitle ? patch.subtitleStreamIndex : playback.selected_subtitle_stream_index,
    true,
  );

  playback.selected_media_source_id = selectedSource?.id || '';
  playback.selected_audio_stream_index = nextAudio;
  playback.selected_subtitle_stream_index = nextSubtitle;
  playback.media_sources.forEach((source) => {
    const isSelectedSource = source?.id === playback.selected_media_source_id;
    source.is_selected = isSelectedSource;
    (source.audio_tracks || []).forEach((track) => {
      track.is_selected = !!(isSelectedSource && Number(track.index) === Number(nextAudio));
    });
    (source.subtitle_tracks || []).forEach((track) => {
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
  return {
    mediaSourceId: playback.selected_media_source_id,
    audioStreamIndex: nextAudio,
    subtitleStreamIndex: nextSubtitle,
  };
}

function _syncVideoSelectionAcrossRows(fileId, sourceMeta) {
  const row = state.files.find((entry) => entry.id === fileId);
  if (row) {
    row.source_metadata = {
      ...(row.source_metadata || {}),
      ...sourceMeta,
    };
  }
  if (state.detailOverrideFile?.id === fileId && state.detailOverrideFile !== row) {
    state.detailOverrideFile.source_metadata = {
      ...(state.detailOverrideFile.source_metadata || {}),
      ...sourceMeta,
    };
  }
}

function _broadcastVideoPlaybackSelection(fileId, selection) {
  if (!fileId || !selection || typeof window === 'undefined') return;
  window.dispatchEvent(new CustomEvent('media-video-selection', {
    detail: {
      fileId,
      selection,
      origin: 'files-detail',
    },
  }));
}

function _videoChildMeta(child) {
  const entityKind = String(child?.entity_kind || '').toLowerCase();
  const seasonNumber = Number(child?.season_number) || 0;
  const episodeNumber = Number(child?.episode_number) || 0;
  if (entityKind === 'season') {
    return seasonNumber > 0 ? `Season ${seasonNumber}` : 'Season';
  }
  if (entityKind === 'episode') {
    if (seasonNumber > 0 && episodeNumber > 0) return `S${seasonNumber}E${episodeNumber}`;
    if (episodeNumber > 0) return `Episode ${episodeNumber}`;
    return 'Episode';
  }
  return child?.type || '';
}

function _nextUpCta(file) {
  const entityKind = String(file?.source_metadata?.entity_kind || '').toLowerCase();
  if (entityKind !== 'series') return null;
  const nextUp = file?.source_metadata?.next_up;
  if (!nextUp || !nextUp.file_id) return null;
  return nextUp;
}

function _nextUpCtaHtml(file) {
  const nextUp = _nextUpCta(file);
  if (!nextUp) return '';
  const play = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
  const label = String(nextUp.label || 'Continue').trim();
  const title = String(nextUp.name || '').trim();
  return `
    <div class="files-detail-primary">
      <button type="button" class="btn btn-primary files-primary-cta" data-next-up-file-id="${escapeHtml(nextUp.file_id)}">
        ${play}
        <span>${escapeHtml(label)}${title ? ` · ${escapeHtml(title)}` : ''}</span>
      </button>
    </div>
  `;
}

// --- Series/Season/Episode rich metadata ----------------------------
//
// Mirrors what Jellyfin/Emby surface on a series page: status chip
// (Continuing/Ended), year range, official + community ratings,
// network, episode/season counts, tagline, cast strip. Backend
// populates these onto source_metadata via _normalise_emby_compat_details
// + the writeback in /api/media/details.

function _seriesMetaChipsHtml(meta) {
  const startYear = Number(meta.published_year || meta.year) || 0;
  const endYear = Number(meta.end_year) || 0;
  const status = String(meta.status || '').trim();
  const officialRating = String(meta.official_rating || '').trim();
  const communityRating = Number(meta.community_rating) || 0;
  const network = String(meta.network || '').trim();
  const seasonCount = Number(meta.season_count) || 0;
  const episodeCount = Number(meta.episode_count) || 0;
  const runtimeS = Number(meta.duration_s) || 0;
  const runtimeMin = runtimeS > 0 ? Math.round(runtimeS / 60) : 0;

  const yearChip = startYear
    ? (status === 'Ended' && endYear && endYear !== startYear)
      ? `${startYear}–${endYear}`
      : status === 'Ended' && endYear === startYear
        ? `${startYear}`
        : (status === 'Continuing' || (!endYear && startYear))
          ? `${startYear}–`
          : `${startYear}`
    : '';

  const chips = [];
  if (yearChip) chips.push(`<span class="files-chip">${escapeHtml(yearChip)}</span>`);
  if (status) {
    const statusClass = status === 'Continuing' ? 'files-chip-active' : 'files-chip-muted';
    chips.push(`<span class="files-chip ${statusClass}">${escapeHtml(status)}</span>`);
  }
  if (officialRating) chips.push(`<span class="files-chip files-chip-rating">${escapeHtml(officialRating)}</span>`);
  if (communityRating > 0) chips.push(`<span class="files-chip files-chip-rating-community">★ ${communityRating.toFixed(1)}</span>`);
  if (seasonCount > 0) {
    const label = seasonCount === 1 ? '1 season' : `${seasonCount} seasons`;
    chips.push(`<span class="files-chip files-chip-muted">${escapeHtml(label)}</span>`);
  }
  if (episodeCount > 0) {
    const label = episodeCount === 1 ? '1 episode' : `${episodeCount} episodes`;
    chips.push(`<span class="files-chip files-chip-muted">${escapeHtml(label)}</span>`);
  }
  if (runtimeMin > 0) chips.push(`<span class="files-chip files-chip-muted">${runtimeMin} min</span>`);
  if (network) chips.push(`<span class="files-chip files-chip-network">${escapeHtml(network)}</span>`);
  return chips.length
    ? `<div class="files-detail-chips files-series-meta-chips">${chips.join('')}</div>`
    : '';
}

function _seriesHeroHtml(file) {
  const entityKind = String(file?.source_metadata?.entity_kind || '').toLowerCase();
  if (entityKind !== 'series') return '';
  const meta = file.source_metadata || {};
  const tagline = String(meta.tagline || '').trim();
  const chipsHtml = _seriesMetaChipsHtml(meta);
  if (!tagline && !chipsHtml) return '';
  return `
    <div class="files-series-hero">
      ${chipsHtml}
      ${tagline ? `<p class="files-series-tagline">${escapeHtml(tagline)}</p>` : ''}
    </div>
  `;
}

function _seasonOrEpisodeMetaHtml(file) {
  const entityKind = String(file?.source_metadata?.entity_kind || '').toLowerCase();
  if (!['season', 'episode'].includes(entityKind)) return '';
  const meta = file.source_metadata || {};
  const seasonNumber = Number(meta.season_number) || 0;
  const episodeNumber = Number(meta.episode_number) || 0;
  const episodeCount = Number(meta.episode_count) || 0;
  const premiereDate = String(meta.premiere_date || '').trim();
  const runtimeS = Number(meta.duration_s) || 0;
  const runtimeMin = runtimeS > 0 ? Math.round(runtimeS / 60) : 0;

  const chips = [];
  if (entityKind === 'episode' && seasonNumber > 0 && episodeNumber > 0) {
    chips.push(`<span class="files-chip">S${seasonNumber}E${episodeNumber}</span>`);
  } else if (entityKind === 'season' && seasonNumber > 0) {
    chips.push(`<span class="files-chip">Season ${seasonNumber}</span>`);
  }
  if (entityKind === 'season' && episodeCount > 0) {
    const label = episodeCount === 1 ? '1 episode' : `${episodeCount} episodes`;
    chips.push(`<span class="files-chip files-chip-muted">${escapeHtml(label)}</span>`);
  }
  if (premiereDate) chips.push(`<span class="files-chip files-chip-muted">${escapeHtml(premiereDate)}</span>`);
  if (entityKind === 'episode' && runtimeMin > 0) {
    chips.push(`<span class="files-chip files-chip-muted">${runtimeMin} min</span>`);
  }
  return chips.length
    ? `<div class="files-detail-chips files-series-meta-chips">${chips.join('')}</div>`
    : '';
}

function _castStripHtml(file) {
  const entityKind = String(file?.source_metadata?.entity_kind || '').toLowerCase();
  if (!['series', 'season', 'episode', 'movie'].includes(entityKind)) return '';
  const cast = Array.isArray(file?.source_metadata?.cast) ? file.source_metadata.cast : [];
  if (!cast.length) return '';
  const items = cast.slice(0, 18).map((member) => {
    const name = String(member?.name || '').trim();
    if (!name) return '';
    const role = String(member?.role || '').trim();
    const personId = String(member?.person_id || '').trim();
    const hasImage = !!String(member?.image_tag || '').trim();
    // Inline SVG silhouette fallback — avoids a network 404 when the
    // provider has no headshot for this actor. The fallback is a soft
    // gradient in the theme's muted tone so it reads as "unavailable"
    // rather than "broken".
    const imgHtml = hasImage && personId
      ? `<img class="files-cast-img" loading="lazy" src="${escapeHtml(personImageUrl(file.id, personId))}" alt="" onerror="this.classList.add('is-missing')"/>`
      : '<div class="files-cast-img files-cast-img-empty" aria-hidden="true"></div>';
    const clickable = !!personId;
    return `
      <button type="button" class="files-cast-item${clickable ? '' : ' is-static'}"
        ${clickable ? `data-person-id="${escapeHtml(personId)}" data-person-name="${escapeHtml(name)}"` : 'disabled'}>
        ${imgHtml}
        <div class="files-cast-text">
          <div class="files-cast-name">${escapeHtml(name)}</div>
          ${role ? `<div class="files-cast-role">${escapeHtml(role)}</div>` : ''}
        </div>
      </button>
    `;
  }).filter(Boolean).join('');
  return `
    <details class="files-detail-section" open>
      <summary>Cast</summary>
      <div class="files-cast-strip">${items}</div>
    </details>
  `;
}

// Person overlay — opened from a cast tile click. Renders profile info
// + filmography. Clicking a work that lives in the user's library
// closes the overlay and swaps the Files detail onto that work.
async function _openPersonModal(fileId, personId, presetName) {
  // Dedupe: clicking the same tile twice shouldn't stack overlays.
  document.querySelectorAll('.files-person-overlay').forEach((el) => el.remove());

  const overlay = document.createElement('div');
  overlay.className = 'files-person-overlay';
  overlay.tabIndex = 0;
  overlay.innerHTML = `
    <div class="files-person-backdrop" data-person-close></div>
    <div class="files-person-sheet" role="dialog" aria-label="Cast member details">
      <button type="button" class="files-person-close" data-person-close aria-label="Close">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
      </button>
      <div class="files-person-loading">Loading ${escapeHtml(presetName || 'cast member')}…</div>
    </div>
  `;
  document.body.appendChild(overlay);
  requestAnimationFrame(() => overlay.classList.add('visible'));
  overlay.focus();

  const close = () => {
    overlay.classList.remove('visible');
    setTimeout(() => overlay.remove(), 180);
    document.removeEventListener('keydown', onKey);
  };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  document.addEventListener('keydown', onKey);
  overlay.querySelectorAll('[data-person-close]').forEach((el) => {
    el.addEventListener('click', close);
  });

  const profile = await fetchPersonProfile(fileId, personId);
  if (!profile) {
    const sheet = overlay.querySelector('.files-person-sheet');
    if (sheet) {
      sheet.innerHTML = `
        <button type="button" class="files-person-close" data-person-close aria-label="Close">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
        </button>
        <div class="files-person-empty">Couldn't load profile.</div>
      `;
      sheet.querySelectorAll('[data-person-close]').forEach((el) => {
        el.addEventListener('click', close);
      });
    }
    return;
  }

  const imgHtml = profile.has_image
    ? `<img class="files-person-photo" src="${escapeHtml(personImageUrl(fileId, personId))}" alt=""/>`
    : '<div class="files-person-photo files-person-photo-empty" aria-hidden="true"></div>';
  const works = Array.isArray(profile.works) ? profile.works : [];
  const worksHtml = works.length ? works.map((w) => {
    const inLibrary = !!w.in_library && !!w.file_id;
    const year = w.year ? ` · ${w.year}` : '';
    const cover = inLibrary ? mediaCoverUrl(w.file_id) : '';
    const coverHtml = cover
      ? `<img class="files-person-work-cover" loading="lazy" src="${escapeHtml(cover)}" alt="" onerror="this.style.display='none'"/>`
      : '<div class="files-person-work-cover files-person-work-cover-empty" aria-hidden="true"></div>';
    return `
      <button type="button" class="files-person-work${inLibrary ? '' : ' is-external'}"
        ${inLibrary ? `data-work-file-id="${escapeHtml(w.file_id)}"` : 'disabled'}
        title="${escapeHtml(w.name || '')}">
        ${coverHtml}
        <div class="files-person-work-meta">
          <div class="files-person-work-title">${escapeHtml(w.name || 'Untitled')}</div>
          <div class="files-person-work-sub">${escapeHtml((w.entity_kind === 'movie' ? 'Movie' : 'Series') + year)}</div>
          ${inLibrary ? '' : '<div class="files-person-work-unavailable">Not in library</div>'}
        </div>
      </button>
    `;
  }).join('') : '<div class="files-person-empty">No works found.</div>';

  const lifespan = [profile.birth_date, profile.death_date].filter(Boolean).join(' – ');
  const subtitleParts = [lifespan, profile.birth_place].filter(Boolean);
  const sheet = overlay.querySelector('.files-person-sheet');
  if (!sheet) return;
  sheet.innerHTML = `
    <button type="button" class="files-person-close" data-person-close aria-label="Close">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
    </button>
    <div class="files-person-header">
      ${imgHtml}
      <div class="files-person-identity">
        <h2 class="files-person-name">${escapeHtml(profile.name || 'Unknown')}</h2>
        ${subtitleParts.length ? `<div class="files-person-sub">${escapeHtml(subtitleParts.join(' · '))}</div>` : ''}
        ${profile.overview ? `<p class="files-person-bio">${escapeHtml(profile.overview)}</p>` : ''}
      </div>
    </div>
    <div class="files-person-section-title">Appears in</div>
    <div class="files-person-works">${worksHtml}</div>
  `;
  sheet.querySelectorAll('[data-person-close]').forEach((el) => {
    el.addEventListener('click', close);
  });
  // Clicking a work routes the Files detail panel onto it. Force-add
  // to selection so updateDetail renders even when the work isn't in
  // the current grid filter (e.g., user is on Shows and clicked a
  // Movie). detailOverrideFile ensures the panel picks up the entry
  // regardless of whether the grid contains it.
  sheet.querySelectorAll('[data-work-file-id]').forEach((el) => {
    el.addEventListener('click', async () => {
      const workId = el.dataset.workFileId;
      if (!workId) return;
      close();
      const entry = await fetchFileEntry(workId);
      if (!entry) return;
      state.selection.clear();
      state.selection.add(entry.id);
      state.detailOverrideFile = entry;
      state.detailNavStack = [];
      updateSelectionUI();
    });
  });
}

function _useBackdropHero(file) {
  const entityKind = String(file?.source_metadata?.entity_kind || '').toLowerCase();
  if (!['series', 'movie'].includes(entityKind)) return false;
  return !!file?.source_metadata?.has_backdrop;
}

function _detailBreadcrumbHtml() {
  const stack = state.detailNavStack;
  if (!Array.isArray(stack) || !stack.length) return '';
  const parts = stack.map((entry, idx) => {
    const name = String(entry.name || 'Untitled').trim();
    return `<button type="button" class="files-detail-breadcrumb-link" data-breadcrumb-index="${idx}">${escapeHtml(name)}</button>`;
  });
  return `
    <nav class="files-detail-breadcrumb" aria-label="Detail navigation">
      ${parts.join('<span class="files-detail-breadcrumb-sep" aria-hidden="true">›</span>')}
      <span class="files-detail-breadcrumb-sep" aria-hidden="true">›</span>
    </nav>
  `;
}

function _videoChildrenSectionHtml(file) {
  const entityKind = String(file?.source_metadata?.entity_kind || '').toLowerCase();
  if (!isMediaServerFile(file) || !isVideo(file) || !['series', 'season'].includes(entityKind)) {
    return '';
  }
  const children = _mediaServerVideoChildren(file);
  const label = entityKind === 'series' ? 'Seasons' : 'Episodes';
  const body = children.length
    ? children.map((child) => `
        <button type="button" class="files-video-child" data-child-file-id="${escapeHtml(child.file_id)}">
          <span class="files-video-child-main">
            <span class="files-video-child-title">${escapeHtml(child.name || 'Untitled')}</span>
            <span class="files-video-child-meta">${escapeHtml(_videoChildMeta(child))}</span>
          </span>
          <span class="files-video-child-side">
            ${child.is_finished ? '<span class="files-chip files-chip-finished">Watched</span>' : ''}
            <span class="files-video-child-go" aria-hidden="true">&#8250;</span>
          </span>
        </button>
      `).join('')
    : `<div class="files-video-child-empty">Loading ${label.toLowerCase()}...</div>`;
  return `
    <details class="files-detail-section" open>
      <summary>${label}</summary>
      <div class="files-video-child-list">${body}</div>
    </details>
  `;
}

function _podcastEpisodeMeta(child) {
  const parts = [];
  const publishedAt = Number(child?.published_at) || 0;
  const pubDate = String(child?.pub_date || '').trim();
  if (publishedAt > 0) {
    parts.push(formatDate(new Date(publishedAt).toISOString()));
  } else if (pubDate) {
    parts.push(pubDate);
  }
  const duration = Number(child?.duration_s) || 0;
  if (duration > 0) {
    parts.push(_fmtDurationLoose(duration));
  }
  return parts.join(' · ');
}

function _podcastEpisodeRowsHtml(episodes, selectedEpisodeId = '') {
  if (!episodes.length) {
    return '<div class="files-video-child-empty">No downloaded episodes available yet.</div>';
  }
  return episodes.map((child, index) => {
    const meta = _podcastEpisodeMeta(child);
    const isActive = String(child?.episode_id || '') === String(selectedEpisodeId || '');
    let side = '<span class="files-video-child-go" aria-hidden="true">&#8250;</span>';
    if (child?.is_finished) {
      side = '<span class="files-chip files-chip-finished">Finished</span>';
    } else if ((Number(child?.progress_pct) || 0) > 0) {
      side = `<span class="files-chip files-chip-resume">${Math.round(Number(child.progress_pct) || 0)}% played</span>`;
    }
    return `
      <button
        type="button"
        class="files-video-child files-podcast-episode${isActive ? ' is-active' : ''}"
        data-podcast-episode-id="${escapeHtml(child.episode_id || '')}"
      >
        <span class="files-video-child-main">
          <span class="files-video-child-title">${escapeHtml(child.name || `Episode ${index + 1}`)}</span>
          <span class="files-video-child-meta">${escapeHtml(meta || 'Episode')}</span>
        </span>
        <span class="files-video-child-side">
          ${side}
        </span>
      </button>
    `;
  }).join('');
}

function _podcastEpisodesSectionHtml(m) {
  return `
    <details class="files-detail-section" open>
      <summary>Episodes <span class="files-abs-chip">${m.episodes.length || '…'}</span></summary>
      <div class="files-video-child-list" data-podcast-episode-list>
        ${m.episodes.length
          ? _podcastEpisodeRowsHtml(m.episodes, m.selectedEpisodeId)
          : '<div class="files-video-child-empty">Loading episodes…</div>'}
      </div>
    </details>
  `;
}

function _videoTrackButtons(tracks, selectedIndex, attr) {
  return (tracks || []).map((track) => {
    const index = Number(track?.index);
    const active = Number.isFinite(index) && Number(selectedIndex) === index;
    const isNone = !!track?.is_none;
    const badges = [];
    if (track?.is_default && !isNone) badges.push('Default');
    if (track?.is_forced) badges.push('Forced');
    if (track?.is_external) badges.push('External');
    return `
      <button type="button" class="files-video-track-btn${active ? ' active' : ''}"
              ${attr}="${escapeHtml(String(index))}">
        <span class="files-video-track-btn-main">
          <span class="files-video-track-btn-label">${escapeHtml(track?.label || 'Track')}</span>
          ${badges.length ? `<span class="files-video-track-btn-badges">${badges.map((badge) => `<span class="files-chip files-chip-muted">${escapeHtml(badge)}</span>`).join('')}</span>` : ''}
        </span>
      </button>
    `;
  }).join('');
}

function _videoPlaybackSectionHtml(file) {
  const playback = _videoPlaybackState(file);
  if (!playback?.media_sources?.length) return '';
  const selectedSource = _selectedPlaybackSource(playback);
  if (!selectedSource) return '';

  const showVersions = playback.media_sources.length > 1;
  const showAudio = (selectedSource.audio_tracks || []).length > 1;
  const showSubtitles = (selectedSource.subtitle_tracks || []).length > 1;
  if (!showVersions && !showAudio && !showSubtitles) return '';

  return `
    <details class="files-detail-section" open>
      <summary>Playback</summary>
      <div class="files-video-playback">
        ${showVersions ? `
          <div class="files-video-track-group">
            <div class="files-video-track-label">Version</div>
            <div class="files-video-track-options">
              ${playback.media_sources.map((source) => `
                <button type="button" class="files-video-track-btn${source?.id === playback.selected_media_source_id ? ' active' : ''}"
                        data-playback-media-source="${escapeHtml(source?.id || '')}">
                  <span class="files-video-track-btn-main">
                    <span class="files-video-track-btn-label">${escapeHtml(source?.label || 'Version')}</span>
                  </span>
                </button>
              `).join('')}
            </div>
          </div>
        ` : ''}
        ${showAudio ? `
          <div class="files-video-track-group">
            <div class="files-video-track-label">Audio</div>
            <div class="files-video-track-options">
              ${_videoTrackButtons(
                selectedSource.audio_tracks || [],
                playback.selected_audio_stream_index,
                'data-playback-audio',
              )}
            </div>
          </div>
        ` : ''}
        ${showSubtitles ? `
          <div class="files-video-track-group">
            <div class="files-video-track-label">Subtitles</div>
            <div class="files-video-track-options">
              ${_videoTrackButtons(
                selectedSource.subtitle_tracks || [],
                playback.selected_subtitle_stream_index,
                'data-playback-subtitle',
              )}
            </div>
          </div>
        ` : ''}
      </div>
    </details>
  `;
}

function _renderGenericDetail(file, token) {
  const ext = getExt(file.name);
  const contentPreview = _loadContentPreview(file, token);
  const kindLabel = _kindLabel(file, ext);
  const sizeLabel = humanSize(file.size_bytes);
  const showSize = !isMediaServerFile(file) || Number(file.size_bytes) > 0;
  const primary = _primaryAction(file);
  const nextUpBlock = _nextUpCtaHtml(file);
  const videoChildrenBlock = _videoChildrenSectionHtml(file);
  const videoPlaybackBlock = _videoPlaybackSectionHtml(file);
  const seriesHero = _seriesHeroHtml(file);
  const seasonEpisodeMeta = _seasonOrEpisodeMetaHtml(file);
  const castStrip = _castStripHtml(file);
  const breadcrumbBlock = _detailBreadcrumbHtml();
  const description = String(file.description || file.source_metadata?.description || '').trim();
  const typeLabel = kindLabel || (ext ? ext.toUpperCase() : '');

  state.el.detail.innerHTML = `
    <div class="files-detail-topbar">${_closeButtonHtml()}</div>
    <div class="files-detail-hero${_useBackdropHero(file) ? ' files-detail-hero--backdrop' : ''}">
      ${_useBackdropHero(file) ? `
        <img class="files-detail-backdrop" loading="lazy"
          src="${escapeHtml(mediaBackdropUrl(file.id))}" alt=""
          onerror="this.style.display='none';this.parentElement?.classList.remove('files-detail-hero--backdrop')"/>
        <div class="files-detail-backdrop-poster">${contentPreview}</div>
      ` : contentPreview}
    </div>
    <div class="files-detail-identity">
      ${breadcrumbBlock}
      <h2 class="files-detail-title">${escapeHtml(file.name)}</h2>
      <div class="files-detail-chips">
        ${kindLabel ? `<span class="files-chip">${escapeHtml(kindLabel)}</span>` : ''}
        ${showSize && sizeLabel ? `<span class="files-chip files-chip-muted">${escapeHtml(sizeLabel)}</span>` : ''}
        ${file.created_at ? `<span class="files-chip files-chip-muted">${escapeHtml(formatDate(file.created_at))}</span>` : ''}
      </div>
      ${seriesHero}
      ${seasonEpisodeMeta}
    </div>
    ${primary ? `
      <div class="files-detail-primary">
        <button class="btn btn-primary files-primary-cta" data-action="${escapeHtml(primary.action)}" data-id="${escapeHtml(file.id)}"${primary.kind ? ` data-kind="${escapeHtml(primary.kind)}"` : ''}>
          ${primary.icon}
          <span>${escapeHtml(primary.label)}</span>
        </button>
      </div>
    ` : nextUpBlock}
    <div class="files-detail-actions">
      <button class="btn btn-sm files-icon-btn" data-action="download" data-id="${escapeHtml(file.id)}" title="Download" aria-label="Download">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      </button>
      <button class="btn btn-sm files-icon-btn" data-action="reference" data-id="${escapeHtml(file.id)}" data-name="${escapeHtml(file.name)}" title="Reference in Chat" aria-label="Reference in Chat">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
      </button>
      <button class="btn btn-sm files-icon-btn" data-action="summarize" data-id="${escapeHtml(file.id)}" title="Summarize with AI" aria-label="Summarize">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
      </button>
      ${(isAudio(file) || isVideo(file)) ? `<button class="btn btn-sm files-icon-btn" data-action="add-to-playlist" data-id="${escapeHtml(file.id)}" data-name="${escapeHtml(file.name)}" data-kind="${isVideo(file) ? 'video' : 'audio'}" title="Add to Grove playlist" aria-label="Add to playlist">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="13" y2="18"/><line x1="18" y1="15" x2="18" y2="21"/><line x1="15" y1="18" x2="21" y2="18"/></svg>
      </button>` : ''}
      ${supportsReadAloud(file) ? `
        <button class="btn btn-sm files-icon-btn" data-action="read-aloud" data-id="${escapeHtml(file.id)}" title="Listen" aria-label="Listen">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 14h4l5-4v8l-5-4H3z"/><path d="M15 8a5 5 0 0 1 0 8"/></svg>
        </button>
      ` : ''}
      ${isEpub(file) ? `
        <button class="btn btn-sm files-icon-btn files-narration-btn" data-action="narration" data-id="${escapeHtml(file.id)}" data-name="${escapeHtml(file.name)}" title="Record / play TTS narration (press and hold to pick a voice)" aria-label="TTS narration">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="11" rx="3"/><path d="M5 10v1a7 7 0 0 0 14 0v-1"/><line x1="12" y1="19" x2="12" y2="22"/></svg>
        </button>
      ` : ''}
    </div>
    ${description ? `
      <details class="files-detail-section" open>
        <summary>Description</summary>
        <p class="files-detail-desc">${escapeHtml(description)}</p>
      </details>
    ` : ''}
    ${videoPlaybackBlock}
    ${videoChildrenBlock}
    ${castStrip}
    <details class="files-detail-section">
      <summary>Details</summary>
      <div class="files-detail-props">
        ${typeLabel ? `<div class="files-detail-row"><span class="files-detail-label">Type</span><span>${escapeHtml(typeLabel)}</span></div>` : ''}
        ${showSize && sizeLabel ? `<div class="files-detail-row"><span class="files-detail-label">Size</span><span>${escapeHtml(sizeLabel)}</span></div>` : ''}
        <div class="files-detail-row"><span class="files-detail-label">Source</span><span>${escapeHtml(file.source || '')}</span></div>
        <div class="files-detail-row"><span class="files-detail-label">Created</span><span>${escapeHtml(formatDate(file.created_at))}</span></div>
        ${file.mime_type ? `<div class="files-detail-row"><span class="files-detail-label">MIME</span><span>${escapeHtml(file.mime_type)}</span></div>` : ''}
      </div>
    </details>
    ${_tagEditorHtml(file)}
    ${primary ? _stickyFooterHtml({ ...primary, id: file.id }) : ''}
  `;
  state.el.detail.classList.remove('hidden');
}

function _playIcon() {
  return '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
}

/* Mobile overlay sticky bottom action bar. Hidden on desktop by CSS media
 * query (see `.files-detail-footer` rules in files.css). Rendered for every
 * file kind that has a primary action — ensures Play/Open/View is always
 * reachable on phones without scrolling past the description + tags. */
function _stickyFooterHtml({ action, label, icon, id, kind }) {
  const kindAttr = kind ? ` data-kind="${escapeHtml(kind)}"` : '';
  return `<div class="files-detail-footer">
    <button class="btn btn-primary files-footer-cta" data-action="${escapeHtml(action)}" data-id="${escapeHtml(id)}"${kindAttr}>
      ${icon}
      <span>${escapeHtml(label)}</span>
    </button>
  </div>`;
}

function _kindLabel(file, ext) {
  const entityKind = file?.source_metadata?.entity_kind || '';
  if (entityKind === 'series')  return 'Series';
  if (entityKind === 'season')  return 'Season';
  if (entityKind === 'episode') return 'Episode';
  if (entityKind === 'movie')   return 'Movie';
  if (entityKind === 'music_video') return 'Music Video';
  if (isImage(file))    return 'Photo';
  if (isVideo(file))    return 'Video';
  if (isAudio(file))    return 'Audio';
  if (isPdf(file))      return 'PDF';
  if (isEpub(file))     return 'EPUB';
  if (isOffice(file))   return 'Document';
  if (isArchive(file))  return 'Archive';
  if (isHtml(file))     return 'HTML';
  if (isMarkdown(file)) return 'Markdown';
  if (isText(file))     return ext ? ext.toUpperCase() : 'Text';
  return ext ? ext.toUpperCase() : '';
}

function _primaryAction(file) {
  const play  = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
  const eye   = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>';
  const open  = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>';
  if (isImage(file))    return { action: 'open-gallery',   label: 'View',  icon: eye };
  if (
    isMediaServerFile(file)
    && !file.source_metadata?.stream_path
    && !file.source_metadata?.selected_episode_id
  ) return null;
  if (isVideo(file))    return { action: 'expand-preview', label: 'Open player', icon: play, kind: 'video' };
  if (isAudio(file))    return { action: 'expand-preview', label: 'Play',  icon: play, kind: 'audio' };
  if (isPdf(file))      return { action: 'expand-preview', label: 'Open',  icon: open, kind: 'pdf' };
  // App-builder projects (zip/html with source_json) launch in the library
  // workspace — must come before isHtml/isArchive so the project path wins.
  if (isAppProject(file)) return { action: 'project',      label: 'Play',  icon: play };
  if (isHtml(file))     return { action: 'expand-preview', label: 'Open',  icon: open, kind: 'html' };
  if (isEpub(file) || isOffice(file) || isArchive(file) ||
      isMarkdown(file) || isText(file)) {
    return { action: 'expand-preview', label: 'Open', icon: open, kind: 'rendered' };
  }
  return null;
}

function _wireDetail(file) {
  state.el.detail.querySelector('.files-detail-close')?.addEventListener('click', () => deselectAll());

  state.el.detail.querySelectorAll('.files-tag-remove').forEach(btn => {
    btn.addEventListener('click', async () => {
      const tagToRemove = btn.dataset.removeTag;
      const newTags = (file.tags || []).filter(t => t !== tagToRemove);
      const data = await patchTags(file.id, newTags);
      if (data) file.tags = data.tags;
      updateDetail();
    });
  });

  const tagInput = state.el.detail.querySelector('.files-tag-input');
  if (tagInput) _wireTagAutocomplete(tagInput, file);

  state.el.detail.querySelector('.files-preview-img[data-gallery-id]')?.addEventListener('click', (e) => {
    openGallery(e.target.dataset.galleryId);
  });
  state.el.detail.querySelectorAll('[data-preview-expand]').forEach(btn => {
    btn.addEventListener('click', () => {
      openMediaPreview(btn.dataset.previewExpand, btn.dataset.kind);
    });
  });

  // Media-server preview: wire resume-on-load, chapter seek-on-click,
  // author/narrator pivot, and player-state subscription. The entire detail
  // panel is the wire scope now that the layout unified — selectors below
  // all live somewhere inside `.files-detail`.
  if (isMediaServerFile(file) && file.kind === 'audio') {
    _wireMediaServerPreview(state.el.detail, file);
  } else if (isMediaServerFile(file) && file.kind === 'video') {
    _wireMediaServerVideoDetail(state.el.detail, file);
  }
}

// --- Media-server preview wiring --------------------------------------
//
// The detail view is presentational: it renders cover + metadata + chapter
// list and delegates all actual playback to the global media-player
// singleton. That way playback keeps going after the panel closes, and a
// single audio element powers every surface that wants to play media.

function _wireMediaServerPreview(root, file) {
  const fileId = file.id;

  // 1) Kick an async details fetch so chapters / description / fresh
  //    progress land even if the library listing didn't include them.
  //    Non-blocking: current render shows whatever's cached; the enrich
  //    step patches the DOM in place when it returns.
  _fetchAndEnrichDetails(root, file);

  // 2) Wire the Play / Resume CTA: always delegates to the singleton.
  //    We import dynamically so file-browser users who never touch a
  //    media row don't pay the module cost. Use `querySelectorAll` —
  //    there may be two Play buttons (in-flow hero CTA + sticky mobile
  //    footer) and both need to fire the same action.
  root.querySelectorAll('[data-media-play]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const player = await import('../media-player.js');
      player.play(fileId);
    });
  });

  // Restart from beginning — shown only in the resume state. Start the
  // book via the same path the primary CTA uses, then seek to 0 so the
  // user's saved position is preserved upstream (the next time-update
  // will bump it back to the very start, which is the intent).
  root.querySelectorAll('[data-media-restart]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const player = await import('../media-player.js');
      await player.play(fileId);
      player.seek(0);
    });
  });

  // 3) Chapter buttons: if this file is currently playing in the
  //    singleton, just seek; otherwise load + seek in one go.
  root.addEventListener('click', async (e) => {
    const episodeBtn = e.target.closest('[data-podcast-episode-id]');
    if (episodeBtn) {
      const episodeId = String(episodeBtn.dataset.podcastEpisodeId || '').trim();
      if (!episodeId) return;
      const rich = await fetchMediaDetails(fileId, { episodeId });
      if (!rich) return;
      _applyAudioDetailsToRows(fileId, rich);
      if (_currentDetailFileId() === fileId) updateDetail();
      const player = await import('../media-player.js');
      await player.play(fileId, { details: rich });
      return;
    }
    const btn = e.target.closest('.files-abs-chapter');
    if (!btn) return;
    const start = parseFloat(btn.dataset.chapterStart) || 0;
    const player = await import('../media-player.js');
    if (player.getState().fileId === fileId) {
      player.seek(start);
      if (!player.getState().isPlaying) player.resume();
    } else {
      // Pre-set the current position so the fetch loads at the chapter.
      await player.play(fileId);
      player.seek(start);
    }
  });

  // 4) Subscribe to player state so we can highlight the active chapter
  //    in real time — but only when THIS file is the one playing.
  //    The unsubscribe is attached to the root so removing the DOM on
  //    selection-change releases the subscription cleanly.
  _subscribeMediaPlayerForPreview(root, fileId);

  // 5) If the panel rendered with a "Fetching text…" chip, keep polling
  //    the details endpoint until the Gutenberg job finishes (or gives
  //    up). The poll rewrites the chip in place — no full re-render.
  if (root.querySelector('[data-readalong-pending]')) {
    _pollReadAlongStatus(root, file);
  }

  // 5) Related-items strip — author/narrator links act as toggles.
  //    Click once to reveal "Also by X" with cover cards; click a card
  //    to swap the detail view to that book; click the close X or the
  //    author link again to collapse.
  _wireRelatedStrip(root, file);
}

function _currentDetailFileId() {
  if (state.detailOverrideFile?.id) return state.detailOverrideFile.id;
  return state.selection.size === 1 ? [...state.selection][0] : '';
}

function _applyAudioDetailsToRows(fileId, rich) {
  const row = state.files.find((entry) => entry.id === fileId);
  const baseMeta = row?.source_metadata
    || state.detailOverrideFile?.source_metadata
    || {};
  const patch = {
    chapters: rich.chapters || baseMeta.chapters || [],
    children: Array.isArray(rich.children) ? rich.children : [],
    entity_kind: rich.entity_kind || baseMeta.entity_kind || '',
    description: rich.description || baseMeta.description || '',
    duration_s: rich.duration_s ?? baseMeta.duration_s ?? 0,
    current_time_s: rich.current_time_s ?? baseMeta.current_time_s ?? 0,
    progress_pct: rich.progress_pct ?? baseMeta.progress_pct ?? 0,
    is_finished: rich.is_finished ?? baseMeta.is_finished ?? false,
    narrator: rich.narrator || baseMeta.narrator || '',
    selected_episode_id: rich.selected_episode_id || baseMeta.selected_episode_id || '',
    selected_episode_title: rich.selected_episode_title || baseMeta.selected_episode_title || '',
  };
  if (row) {
    row.source_metadata = {
      ...(row.source_metadata || {}),
      ...patch,
    };
  }
  if (state.detailOverrideFile?.id === fileId && state.detailOverrideFile !== row) {
    state.detailOverrideFile.source_metadata = {
      ...(state.detailOverrideFile.source_metadata || {}),
      ...patch,
    };
  }
}

function _wireMediaServerVideoDetail(root, file) {
  _fetchAndEnrichVideoDetails(file);
  const siblings = _mediaServerVideoChildren(file);
  root.querySelectorAll('[data-child-file-id]').forEach((btn, index) => {
    btn.addEventListener('click', async () => {
      const childId = btn.dataset.childFileId;
      if (!childId) return;
      const entry = await fetchFileEntry(childId);
      if (!entry) return;
      _attachVideoNavigation(entry, siblings, index);
      // Push the file we're leaving onto the breadcrumb stack so the
      // user can click back up the chain. We skip duplicates (defensive
      // — re-renders shouldn't fire this path twice with the same ID).
      const leaving = file;
      const top = state.detailNavStack[state.detailNavStack.length - 1];
      if (leaving && (!top || top.id !== leaving.id)) {
        state.detailNavStack.push({
          id: leaving.id,
          name: leaving.name,
          entity_kind: String(leaving.source_metadata?.entity_kind || '').toLowerCase(),
        });
      }
      state.detailOverrideFile = entry;
      updateDetail();
    });
  });
  // Continue Watching CTA on a Series: jump straight into the next-up
  // episode's player. Saves the Series → Season → Episode → Play drill.
  //
  // Uses openVideoPreviewById (not openMediaPreview) because the
  // episode's file_index row almost certainly is NOT in state.files —
  // state.files holds the series-level grid, while next_up.file_id
  // points at an individual episode. openMediaPreview's
  // _resolvePreviewFile only checks state.files + state.detailOverrideFile
  // and silently no-ops on a miss, which is exactly the "click does
  // nothing" symptom we were seeing on Start S1E1. openVideoPreviewById
  // falls back to fetchFileEntry by id, so the episode resolves
  // correctly regardless of what's currently in the grid.
  root.querySelectorAll('[data-next-up-file-id]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const nextId = btn.dataset.nextUpFileId;
      if (!nextId) return;
      const ok = await openVideoPreviewById(nextId);
      if (!ok) {
        // Either the episode isn't in the index (sync gap) or its
        // codec is flagged unsupported. Surface a soft toast rather
        // than a silent no-op so the user knows what happened.
        const { showToast } = await import('../app.js');
        showToast?.("Couldn't open this episode. Try refreshing the library.", 'warning', 3500);
      }
    });
  });
  // Cast tile click — open the person overlay (profile + filmography).
  // Clicking a listed work that's in our library closes the overlay and
  // navigates the Files grid onto that item, mirroring Jellyfin's Person
  // page → back-to-library flow.
  root.querySelectorAll('.files-cast-item[data-person-id]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const personId = btn.dataset.personId;
      const personName = btn.dataset.personName || '';
      if (!personId) return;
      _openPersonModal(file.id, personId, personName);
    });
  });
  // Breadcrumb segment click — pop the stack back to that level and
  // swap the detail to the chosen ancestor. Index 0 = root (grid file).
  root.querySelectorAll('[data-breadcrumb-index]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const idx = Number(btn.dataset.breadcrumbIndex);
      if (!Number.isFinite(idx) || idx < 0 || idx >= state.detailNavStack.length) return;
      const target = state.detailNavStack[idx];
      // Truncate everything below the clicked level — they're now
      // ahead of where the user is.
      state.detailNavStack = state.detailNavStack.slice(0, idx);
      const rootId = [...state.selection][0];
      if (target.id === rootId) {
        state.detailOverrideFile = null;
      } else {
        const entry = await fetchFileEntry(target.id);
        if (!entry) return;
        state.detailOverrideFile = entry;
      }
      updateDetail();
    });
  });
  root.querySelectorAll('[data-playback-media-source]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const selection = _applyVideoPlaybackSelection(file, {
        mediaSourceId: btn.dataset.playbackMediaSource || '',
      });
      if (!selection) return;
      _syncVideoSelectionAcrossRows(file.id, file.source_metadata || {});
      _broadcastVideoPlaybackSelection(file.id, selection);
      updateDetail();
      await updateMediaPlaybackSelection(file.id, {
        media_source_id: selection.mediaSourceId || '',
        audio_stream_index: selection.audioStreamIndex,
        subtitle_stream_index: selection.subtitleStreamIndex,
      });
    });
  });
  root.querySelectorAll('[data-playback-audio]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const selection = _applyVideoPlaybackSelection(file, {
        audioStreamIndex: Number(btn.dataset.playbackAudio),
      });
      if (!selection) return;
      _syncVideoSelectionAcrossRows(file.id, file.source_metadata || {});
      _broadcastVideoPlaybackSelection(file.id, selection);
      updateDetail();
      await updateMediaPlaybackSelection(file.id, {
        media_source_id: selection.mediaSourceId || '',
        audio_stream_index: selection.audioStreamIndex,
        subtitle_stream_index: selection.subtitleStreamIndex,
      });
    });
  });
  root.querySelectorAll('[data-playback-subtitle]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const selection = _applyVideoPlaybackSelection(file, {
        subtitleStreamIndex: Number(btn.dataset.playbackSubtitle),
      });
      if (!selection) return;
      _syncVideoSelectionAcrossRows(file.id, file.source_metadata || {});
      _broadcastVideoPlaybackSelection(file.id, selection);
      updateDetail();
      await updateMediaPlaybackSelection(file.id, {
        media_source_id: selection.mediaSourceId || '',
        audio_stream_index: selection.audioStreamIndex,
        subtitle_stream_index: selection.subtitleStreamIndex,
      });
    });
  });
}

async function _fetchAndEnrichVideoDetails(file) {
  if (!file || file.kind !== 'video') return;
  const meta = file.source_metadata || {};
  if (_videoDetailsLoadedEnough(file) || meta._videoDetailsLoading) return;
  meta._videoDetailsLoading = true;
  try {
    const rich = await fetchMediaDetails(file.id);
    if (!rich) return;
    const target = state.detailOverrideFile?.id === file.id ? state.detailOverrideFile : file;
    const playback = rich.playback || target.source_metadata?.playback || null;
    const loadedEnough = _videoNeedsPlaybackDetails(target)
      ? !!(playback && Array.isArray(playback.media_sources) && playback.media_sources.length)
      : true;
    target.source_metadata = {
      ...(target.source_metadata || {}),
      description: rich.description || target.source_metadata?.description || '',
      children: Array.isArray(rich.children) ? rich.children : (target.source_metadata?.children || []),
      next_up: rich.next_up || null,
      playback,
      current_time_s: rich.current_time_s ?? target.source_metadata?.current_time_s ?? 0,
      duration_s: rich.duration_s ?? target.source_metadata?.duration_s ?? 0,
      progress_pct: rich.progress_pct ?? target.source_metadata?.progress_pct ?? 0,
      is_finished: rich.is_finished ?? target.source_metadata?.is_finished ?? false,
      // Series/season/episode metadata — backend persists these on
      // every fetch so the next grid load also has them, but we still
      // copy onto the in-memory object so the current detail render
      // picks them up without waiting for the writeback to land.
      tagline: rich.tagline || '',
      status: rich.status || '',
      end_year: rich.end_year || 0,
      premiere_date: rich.premiere_date || '',
      official_rating: rich.official_rating || '',
      community_rating: rich.community_rating || 0,
      network: rich.network || '',
      season_count: rich.season_count || 0,
      episode_count: rich.episode_count || 0,
      has_backdrop: rich.has_backdrop ?? target.source_metadata?.has_backdrop ?? false,
      published_year: rich.published_year || target.source_metadata?.published_year || 0,
      cast: rich.people?.cast || [],
      _videoDetailsLoaded: loadedEnough,
    };
    if (rich.description && !target.description) {
      target.description = rich.description;
    }
    _syncVideoSelectionAcrossRows(file.id, target.source_metadata || {});
    if (_currentDetailFileId() === file.id) {
      updateDetail();
    }
  } catch {
    console.debug('[media] video details fetch failed');
  } finally {
    meta._videoDetailsLoading = false;
  }
}

function _wireRelatedStrip(root, file) {
  const host  = root.querySelector('[data-related-host]');
  const strip = root.querySelector('[data-related-strip]');
  const label = root.querySelector('[data-related-label]');
  if (!host || !strip || !label) return;

  let currentBy = '';
  const close = () => {
    host.setAttribute('hidden', '');
    currentBy = '';
    root.querySelectorAll('.files-abs-person').forEach(b => b.classList.remove('is-active'));
  };

  root.querySelector('[data-related-close]')?.addEventListener('click', close);

  root.querySelectorAll('.files-abs-person').forEach(btn => {
    btn.addEventListener('click', async () => {
      const by = btn.dataset.relatedBy;
      if (currentBy === by) { close(); return; }
      currentBy = by;
      root.querySelectorAll('.files-abs-person').forEach(b =>
        b.classList.toggle('is-active', b === btn));
      host.removeAttribute('hidden');

      // Reset to skeleton while fetching — avoids showing stale results
      // from a previous author click.
      strip.innerHTML = `<div class="files-abs-related-skeleton" aria-hidden="true">
        <span></span><span></span><span></span><span></span><span></span>
      </div>`;
      label.textContent = by === 'narrator'
        ? 'Also narrated by…'
        : 'Also by this author';

      try {
        const resp = await fetch(
          `/api/media/related/${encodeURIComponent(file.id)}?by=${encodeURIComponent(by)}&limit=30`
        );
        if (!root.isConnected) return;
        if (!resp.ok) { strip.innerHTML = _relatedErrorHtml(); return; }
        const data = await resp.json();
        const name = data.display_name || (by === 'narrator' ? 'this narrator' : 'this author');
        label.textContent = by === 'narrator' ? `Also narrated by ${name}` : `Also by ${name}`;
        if (!data.items?.length) {
          strip.innerHTML = _relatedEmptyHtml(name);
          return;
        }
        strip.innerHTML = data.items.map(_relatedCardHtml).join('');
        _wireRelatedCards(strip);
      } catch {
        if (root.isConnected) strip.innerHTML = _relatedErrorHtml();
      }
    });
  });
}

function _relatedCardHtml(item) {
  const progress = Math.max(0, Math.min(100, Number(item.progress_pct) || 0));
  return `
    <button class="files-abs-related-card" type="button" data-related-id="${escapeHtml(item.id)}"
            title="${escapeHtml(item.title)}">
      <div class="files-abs-related-cover">
        ${item.cover_url
          ? `<img src="${escapeHtml(item.cover_url)}" alt="" loading="lazy" decoding="async" onerror="this.remove()">`
          : `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 4h11a4 4 0 0 1 4 4v12M4 4v16M4 20h11a4 4 0 0 0 4 0"/></svg>`}
        ${progress > 0 ? `<div class="files-abs-related-progress"><span style="width:${progress.toFixed(1)}%"></span></div>` : ''}
      </div>
      <div class="files-abs-related-title">${escapeHtml(item.title)}</div>
    </button>`;
}

function _relatedEmptyHtml(name) {
  return `<div class="files-abs-related-empty">
    No other books found for ${escapeHtml(name)}.
  </div>`;
}

function _relatedErrorHtml() {
  return `<div class="files-abs-related-empty">
    Couldn't load related books just now. Try again in a moment.
  </div>`;
}

function _wireRelatedCards(strip) {
  strip.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-related-id]');
    if (!btn) return;
    const id = btn.dataset.relatedId;
    // Swap detail to the clicked book. If it's in the currently-loaded
    // grid we can reuse selectOnly; otherwise fetch the row and prepend
    // it so the detail panel can render immediately. Dynamic imports
    // break the circular files → render dependency at runtime.
    _navigateToFile(id);
  });
}

async function _navigateToFile(fileId) {
  // Already in the currently-loaded page? Fast path.
  const existing = state.files.find(f => f.id === fileId);
  if (existing) {
    selectOnly(fileId);
    const card = state.el.grid?.querySelector(`.files-card[data-id="${CSS.escape(fileId)}"]`);
    card?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    return;
  }
  // Not loaded: fetch the row and inject it at the top so the detail
  // panel has something to render. This is how deep-links from the
  // mini-player + related-strip stay consistent even with pagination.
  try {
    const resp = await fetch(`/api/files/entry/${encodeURIComponent(fileId)}`);
    if (!resp.ok) return;
    const row = await resp.json();
    if (!row?.id) return;
    state.files.unshift(row);
    selectOnly(fileId);
  } catch {
    // Silent — the user can still navigate via the Files list.
  }
}

// Reconcile the read-along control with the freshest server status.
// Called after every details-endpoint fetch and every poll tick. The
// element identity shifts between states (button ↔ pending chip ↔
// nothing) so we replace the whole control rather than mutating
// attributes on one static element — simpler and avoids stale
// event listeners.
function _applyReadAlongState(root, file, rich) {
  if (!root) return;
  const status = rich?.gutenberg_status ?? '';
  if (!status) return;  // still pending on the server side — keep the chip

  const actions = root.querySelector('.files-detail-actions');
  if (!actions) return;
  const existing = actions.querySelector(
    '[data-action="read-along"], [data-readalong-pending]',
  );

  if (status === 'fetched') {
    const m = _mediaServerFields(file);
    // Rebuild with fresh word count; the HTML helper already handles
    // this state, so parse it and swap in.
    const tmp = document.createElement('div');
    tmp.innerHTML = _readAlongControlHtml(file, m);
    const next = tmp.firstElementChild;
    if (!next) return;
    if (existing) existing.replaceWith(next);
    else actions.appendChild(next);
    return;
  }

  if (status === 'unavailable' || status === 'missing') {
    // Permanent failure — drop the control entirely. The external
    // url_text_source link under "Links" still gives the user a way
    // to read the Gutenberg page in a new tab.
    if (existing) existing.remove();
  }
}

function _pollReadAlongStatus(root, file) {
  const fileId = file.id;
  const startToken = state.detailToken;
  let tries = 0;
  const MAX_TRIES = 20;        // ~60s at 3s interval
  const INTERVAL_MS = 3000;

  const tick = async () => {
    // Selection changed (or panel closed) — stop. The token invariant
    // is the same one `_fetchAndEnrichDetails` uses to abandon stale
    // results; reusing it keeps cancellation semantics consistent.
    if (state.detailToken !== startToken) return;
    if (!root.isConnected) return;

    tries += 1;
    let rich = null;
    try {
      const resp = await fetch(`/api/media/details/${encodeURIComponent(fileId)}`);
      if (resp.ok) rich = await resp.json();
    } catch {
      // Network blip — burn a try, keep polling. A dead loop would
      // stop on the tries cap anyway.
    }

    if (state.detailToken !== startToken || !root.isConnected) return;

    const status = rich?.gutenberg_status ?? '';
    if (status && status !== 'fetching') {
      // Terminal state — merge onto the file row and apply once.
      file.source_metadata = {
        ...(file.source_metadata || {}),
        gutenberg_status:     status,
        gutenberg_word_count: rich.gutenberg_word_count || 0,
        gutenberg_byte_size:  rich.gutenberg_byte_size  || 0,
      };
      _applyReadAlongState(root, file, rich);
      return;
    }
    if (tries >= MAX_TRIES) {
      // Give up without mutating — the chip remains, the user can
      // close + reopen the panel later to re-check. We don't flip to
      // 'unavailable' because the job may still be running; it's just
      // slower than our poll window.
      return;
    }
    setTimeout(tick, INTERVAL_MS);
  };

  setTimeout(tick, INTERVAL_MS);
}

async function _fetchAndEnrichDetails(root, file) {
  try {
    const resp = await fetch(`/api/media/details/${encodeURIComponent(file.id)}`);
    if (!resp.ok) return;
    const rich = await resp.json();
    // Guard against the user switching selection mid-fetch.
    if (!root.isConnected) return;

    // Cache enriched data onto the file row so re-renders reuse it and
    // the Files grid progress bar refreshes without a full reload.
    _applyAudioDetailsToRows(file.id, rich);
    file.source_metadata = {
      ...(file.source_metadata || {}),
      // Read-along status lives on the row too — so a later re-render
      // (e.g. after tag edit) sees the latest state without a round trip.
      gutenberg_status:     rich.gutenberg_status ?? file.source_metadata?.gutenberg_status ?? '',
      gutenberg_word_count: rich.gutenberg_word_count ?? file.source_metadata?.gutenberg_word_count ?? 0,
      gutenberg_byte_size:  rich.gutenberg_byte_size  ?? file.source_metadata?.gutenberg_byte_size  ?? 0,
    };
    _applyReadAlongState(root, file, rich);

    if ((rich.entity_kind || file.source_metadata?.entity_kind || '').toLowerCase() === 'podcast') {
      const episodeList = root.querySelector('[data-podcast-episode-list]');
      if (episodeList) {
        const episodes = Array.isArray(rich.children) ? rich.children : [];
        episodeList.innerHTML = episodes.length
          ? _podcastEpisodeRowsHtml(episodes, rich.selected_episode_id || '')
          : '<div class="files-video-child-empty">No downloaded episodes available yet.</div>';
      }
      return;
    }

    // Patch chapter list + description in place rather than a full
    // re-render so the user's scroll / focus state stays intact.
    const chCount = root.querySelector('[data-chapter-count]');
    const chList  = root.querySelector('[data-chapter-list]');
    if (chCount && chList) {
      // Rehydrate per-chapter state from the freshest book position the
      // details endpoint returned, so the state rail + progress bars line
      // up with the user's actual playback state on open.
      const rawChapters = rich.chapters || [];
      const freshCurrent = rich.current_time_s ?? file.source_metadata?.current_time_s ?? 0;
      const freshDuration = rich.duration_s ?? file.source_metadata?.duration_s ?? 0;
      const chapters = _deriveChapterStates(rawChapters, freshCurrent, freshDuration);
      chCount.textContent = chapters.length || '0';
      chList.innerHTML = chapters.length
        ? chapters.map((c, i) => _chapterRowHtml(c, i)).join('')
        : `<li class="files-abs-chapter-empty">No chapter data available</li>`;
    }

    // If the fetch refreshed progress, nudge the CTA label(s) so Resume
    // reflects the freshest position. We have two Play CTAs today (in-flow
    // primary + sticky mobile footer), both keyed by `[data-cta-kind]` —
    // patch whichever are currently showing the initial "Play" state.
    if (rich.current_time_s > 1) {
      const label = `Resume · ${_fmtTimecode(rich.current_time_s)}`;
      root.querySelectorAll('[data-cta-kind="play"] span').forEach((span) => {
        span.textContent = label;
      });
    }
  } catch {
    // Details fetch is best-effort — the detail view is already usable
    // from the listing-cached metadata. Log only.
    console.debug('[media] details fetch failed');
  }
}

async function _subscribeMediaPlayerForPreview(root, fileId) {
  const player = await import('../media-player.js');
  const unsubscribe = player.subscribe((s) => {
    if (!root.isConnected) { unsubscribe(); return; }
    const isThis = s.fileId === fileId;
    root.classList.toggle('is-playing-this', isThis && s.isPlaying);
    const btns = root.querySelectorAll('.files-abs-chapter');
    btns.forEach((b, i) => {
      b.classList.toggle('is-active', isThis && i === s.currentChapterIdx);
    });
    root.querySelectorAll('[data-podcast-episode-id]').forEach((btn) => {
      btn.classList.toggle(
        'is-active',
        isThis && String(btn.dataset.podcastEpisodeId || '') === String(s.episodeId || ''),
      );
    });
  });
}

// --- Tag autocomplete ------------------------------------------------

function _wireTagAutocomplete(input, file) {
  const panel = input.parentElement?.querySelector('.files-tag-suggest');
  if (!panel) return;

  // Local state: current suggestion list + highlighted index. Kept on the
  // input element so a re-render flushes it cleanly without leaking.
  let items = [];
  let active = -1;
  let debounceId = 0;
  let reqSeq = 0;

  const close = () => { panel.hidden = true; panel.innerHTML = ''; items = []; active = -1; };

  const render = () => {
    if (!items.length) { panel.hidden = true; panel.innerHTML = ''; return; }
    panel.innerHTML = items.map((t, i) =>
      `<button type="button" class="files-tag-suggest-item${i === active ? ' active' : ''}"
               data-suggest-idx="${i}">${escapeHtml(t)}</button>`,
    ).join('');
    panel.hidden = false;
  };

  const commit = async (raw) => {
    const val = (raw ?? input.value).trim();
    if (!val) return;
    // Dedup against existing tags (case-insensitive, NFKC) — server does
    // the same but the UI shouldn't add a visible duplicate before save.
    const existing = (file.tags || []).map(t => t.normalize('NFKC').toLowerCase());
    if (existing.includes(val.normalize('NFKC').toLowerCase())) {
      input.value = '';
      close();
      return;
    }
    const newTags = [...(file.tags || []), val];
    input.value = '';
    close();
    const data = await patchTags(file.id, newTags);
    if (data) file.tags = data.tags;
    updateDetail();
  };

  const fetchSuggestions = async () => {
    const seq = ++reqSeq;
    const prefix = input.value.trim();
    const fetched = await suggestTags(prefix, 8);
    if (seq !== reqSeq) return;  // newer request already in flight
    const existing = new Set((file.tags || []).map(t => t.normalize('NFKC').toLowerCase()));
    items = fetched.filter(t => !existing.has(t.normalize('NFKC').toLowerCase()));
    active = items.length ? 0 : -1;
    render();
  };

  input.addEventListener('input', () => {
    clearTimeout(debounceId);
    debounceId = setTimeout(fetchSuggestions, 120);
  });

  input.addEventListener('focus', () => {
    // Show recent tags as soon as the user clicks in, even before typing.
    if (!items.length) fetchSuggestions();
    else render();
  });

  input.addEventListener('keydown', (e) => {
    e.stopPropagation();  // panel-level shortcuts shouldn't fire while typing
    if (e.key === 'ArrowDown' && items.length) {
      e.preventDefault();
      active = (active + 1) % items.length;
      render();
      return;
    }
    if (e.key === 'ArrowUp' && items.length) {
      e.preventDefault();
      active = (active - 1 + items.length) % items.length;
      render();
      return;
    }
    if (e.key === 'Escape') {
      if (!panel.hidden) { e.preventDefault(); close(); return; }
    }
    if (e.key === 'Tab' && !panel.hidden && active >= 0) {
      e.preventDefault();
      commit(items[active]);
      return;
    }
    if (e.key === 'Enter') {
      e.preventDefault();
      // Pick the highlighted suggestion if the user clearly intended it
      // (highlighted AND the input either matches it or is empty).
      // Otherwise commit the typed value verbatim.
      const typed = input.value.trim();
      const useSuggestion = active >= 0 && items[active] && (
        typed === '' || items[active].toLowerCase().startsWith(typed.toLowerCase())
      );
      commit(useSuggestion ? items[active] : typed);
    }
  });

  panel.addEventListener('mousedown', (e) => {
    // mousedown (not click) so the input doesn't blur first and trigger close
    const btn = e.target.closest('.files-tag-suggest-item');
    if (!btn) return;
    e.preventDefault();
    const idx = parseInt(btn.dataset.suggestIdx, 10);
    if (Number.isFinite(idx) && items[idx]) commit(items[idx]);
  });

  input.addEventListener('blur', () => {
    // Defer so a click on a suggestion has time to land.
    setTimeout(close, 150);
  });
}

// --- Inline preview dispatcher ---------------------------------------

function _loadContentPreview(file, token) {
  // Note: media-server rows don't flow through this dispatcher — they use
  // `_renderMediaServerDetail` which emits the full detail tree (not just
  // a hero card), because the hero is only one piece of that layout.
  if (
    isMediaServerFile(file)
    && !file.source_metadata?.stream_path
    && !file.source_metadata?.selected_episode_id
  ) return '';
  if (isImage(file))    return _previewImage(file);
  if (isVideo(file))    return _previewVideo(file);
  if (isAudio(file))    return _previewAudio(file);
  if (isPdf(file))      return _previewPdf(file);
  if (isEpub(file))     return _previewRendered(file);
  if (isOffice(file))   return _previewRendered(file);
  if (isArchive(file))  return _previewRendered(file);
  if (isHtml(file))     return _previewHtml(file);
  if (isMarkdown(file)) return _previewMarkdown(file, token);
  if (isText(file))     return _previewCode(file, token);
  return '';
}

// --- Media-server row detail view ------------------------------------

function _fmtDuration(totalSec) {
  // "4h 23m" / "42m 15s" — friendlier than seconds or "HH:MM:SS" for audiobooks.
  const s = Math.max(0, Math.floor(totalSec || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${r}s`;
  return `${r}s`;
}

function _fmtDurationLoose(totalSec) {
  // "Loose" variant — drops trailing seconds when minutes are dominant
  // so chrome like "Resume · 42m left" reads cleanly. Uses seconds only
  // for very short remainders (sub-minute). Preferred for CTA labels +
  // chapter duration chips where the precision isn't load-bearing.
  const s = Math.max(0, Math.floor(totalSec || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m`;
  return `${s}s`;
}

function _fmtTimecode(totalSec) {
  // Fixed-width for chapter starts — aligns cleanly in a list.
  const s = Math.max(0, Math.floor(totalSec || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = s % 60;
  return h
    ? `${h}:${String(m).padStart(2, '0')}:${String(r).padStart(2, '0')}`
    : `${m}:${String(r).padStart(2, '0')}`;
}

/** Enrich a chapter list with per-chapter state derived from the book's
 *  current position. Each returned row gains `state` ('played' | 'in-
 *  progress' | 'unplayed'), `progressPct` (0-100 within the chapter), and
 *  `durationSec` (chapter length, = next chapter's start or book end).
 *  Keeps the original fields (title, start) intact. */
function _deriveChapterStates(chapters, currentSec, bookDuration) {
  if (!Array.isArray(chapters) || !chapters.length) return [];
  return chapters.map((c, i) => {
    const start = Number(c.start) || 0;
    const nextStart = i + 1 < chapters.length
      ? (Number(chapters[i + 1].start) || 0)
      : (bookDuration || 0);
    const duration = Math.max(0, nextStart - start);
    let state = 'unplayed';
    let progressPct = 0;
    if (currentSec >= start + duration && duration > 0) {
      state = 'played';
      progressPct = 100;
    } else if (currentSec > start) {
      state = 'in-progress';
      progressPct = duration > 0
        ? Math.min(100, Math.max(0, ((currentSec - start) / duration) * 100))
        : 0;
    }
    return { ...c, start, duration, state, progressPct };
  });
}

function _chapterRowHtml(c, idx) {
  const state = c.state || 'unplayed';
  const pct = Number(c.progressPct) || 0;
  // State rail + progress bar echo the comic chapter list so both media
  // kinds feel catalogued by the same hand. The dot colour is painted by
  // CSS off `.state-*`; the progress bar only renders for in-progress
  // rows (zero-width bars add visual noise without carrying information).
  const progressBar = state === 'in-progress' && pct > 0
    ? `<span class="files-abs-chapter-progress" aria-hidden="true"><span style="width:${pct.toFixed(1)}%"></span></span>`
    : '';
  const durationLabel = c.duration > 0
    ? ` · ${_fmtDurationLoose(c.duration)}`
    : '';
  return `<li>
    <button class="files-abs-chapter state-${escapeHtml(state)}" type="button"
            data-chapter-start="${Number(c.start) || 0}"
            data-chapter-idx="${idx}"
            title="Jump to this chapter">
      <span class="files-abs-chapter-state" aria-hidden="true"></span>
      <span class="files-abs-chapter-idx">${idx + 1}</span>
      <span class="files-abs-chapter-body">
        <span class="files-abs-chapter-title">${escapeHtml(c.title || `Chapter ${idx + 1}`)}</span>
        ${progressBar}
      </span>
      <span class="files-abs-chapter-time">${escapeHtml(_fmtTimecode(c.start))}${escapeHtml(durationLabel)}</span>
    </button>
  </li>`;
}

// All preview renderers emit the inner hero visual. The outer
// `<div class="files-detail-hero">` wrapper is supplied by updateDetail.

function _previewImage(file) {
  const url = imageNeedsServerRender(file) ? renderUrl(file.id) : downloadUrl(file.id);
  return `<div class="files-preview-card files-preview-card-image">
    <img class="files-preview-img" src="${escapeHtml(url)}" alt="${escapeHtml(file.name)}" data-gallery-id="${escapeHtml(file.id)}">
  </div>`;
}

function _previewVideo(file) {
  if (isMediaServerFile(file) && !file.source_metadata?.stream_path) {
    return `<div class="files-preview-card files-preview-unsupported">
      <div class="files-preview-unsupported-msg">This item is a library container, not a directly playable file.</div>
    </div>`;
  }
  if (videoLikelyUnsupported(file)) {
    const ext = getExt(file.name).toUpperCase();
    return `<div class="files-preview-card files-preview-unsupported">
      <div class="files-preview-unsupported-icon">
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
      </div>
      <div class="files-preview-unsupported-msg">${escapeHtml(ext)} not playable in-browser</div>
    </div>`;
  }
  const poster = hasMediaCover(file)
    ? `<img class="files-preview-video-poster" src="${escapeHtml(mediaCoverUrl(file.id))}" alt="">`
    : `<div class="files-preview-video-fallback" aria-hidden="true">
        <svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>
      </div>`;
  return `<div class="files-preview-card files-preview-card-video files-preview-card-video-poster">
    ${poster}
    <div class="files-preview-video-overlay">
      <span class="files-preview-video-pill">Open player to watch</span>
    </div>
  </div>`;
}

function _previewAudio(file) {
  // Generic audio (not audiobookshelf/librivox — those render richly via
  // the audiobook path). Gradient tile + waveform icon; Play comes from
  // the primary CTA in the outer layout.
  const src = isMediaServerFile(file) ? mediaStreamUrl(file.id) : downloadUrl(file.id);
  return `<div class="files-preview-card files-preview-card-audio">
    <div class="files-preview-audio-art" aria-hidden="true">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h2l2-6 4 12 4-9 2 5h4"/></svg>
    </div>
    <audio class="files-preview-audio" controls preload="metadata" src="${escapeHtml(src)}"></audio>
  </div>`;
}

function _previewPdf(file) {
  const url = renderUrl(file.id) + '#view=FitH&toolbar=0';
  return `<div class="files-preview-card files-preview-card-doc files-preview-card-3-4">
    <iframe class="files-preview-iframe" src="${escapeHtml(url)}" title="${escapeHtml(file.name)}" loading="lazy"></iframe>
  </div>`;
}

function _previewHtml(file) {
  return `<div class="files-preview-card files-preview-card-doc">
    <iframe class="files-preview-iframe" src="${escapeHtml(renderUrl(file.id))}" sandbox="allow-scripts" title="${escapeHtml(file.name)}" loading="lazy"></iframe>
  </div>`;
}

function _previewRendered(file) {
  return `<div class="files-preview-card files-preview-card-doc files-preview-card-3-4">
    <iframe class="files-preview-iframe" src="${escapeHtml(renderUrl(file.id))}" sandbox="allow-same-origin" title="${escapeHtml(file.name)}" loading="lazy"></iframe>
  </div>`;
}

function _previewMarkdown(file, token) {
  const containerId = `preview-md-${file.id.replace(/[^a-zA-Z0-9]/g, '_')}`;
  fetchPeek(file.id).then(snippet => {
    if (token !== state.detailToken) return;
    const el = document.getElementById(containerId);
    if (!el) return;
    if (!snippet) { el.textContent = '(empty)'; return; }
    try {
      el.innerHTML = renderMarkdown(snippet, { mode: 'passthrough' });
      // Defer highlighting so a large markdown peek doesn't block the panel.
      highlightCodeDeferred(el);
    } catch { el.textContent = snippet; }
  }).catch(() => {});
  return `<div class="files-preview-card files-preview-card-doc files-preview-card-3-4">
    <div class="files-preview-markdown" id="${escapeHtml(containerId)}">Loading…</div>
  </div>`;
}

function _previewCode(file, token) {
  const containerId = `preview-code-${file.id.replace(/[^a-zA-Z0-9]/g, '_')}`;
  const ext = getExt(file.name);
  const lang = HLJS_LANG_MAP[ext] || ext || '';
  fetchPeek(file.id).then(snippet => {
    if (token !== state.detailToken) return;
    const el = document.getElementById(containerId);
    if (!el) return;
    const code = el.querySelector('code');
    if (!code) return;
    code.textContent = snippet || '(empty)';
    if (lang) code.className = `language-${lang}`;
    // Defer — 10k+ line files would otherwise block the detail panel.
    highlightCodeDeferred(el);
  }).catch(() => {});
  const langAttr = lang ? ` class="language-${escapeHtml(lang)}"` : '';
  return `<div class="files-preview-card files-preview-card-code files-preview-card-3-4">
    <pre class="files-preview-code" id="${escapeHtml(containerId)}"><code${langAttr}>Loading…</code></pre>
  </div>`;
}

// --- Rename -----------------------------------------------------------

export function startRename(id) {
  state.renamingId = id;
  const card = state.el.grid?.querySelector(`.files-card[data-id="${CSS.escape(id)}"]`);
  const nameEl = card?.querySelector('.files-card-name');
  if (!nameEl) return;

  const file = state.files.find(f => f.id === id);
  if (!file) return;

  const input = document.createElement('input');
  input.className = 'files-rename-input';
  input.type = 'text';
  input.value = file.name;
  input.dataset.originalName = file.name;

  nameEl.replaceWith(input);
  input.focus();
  const dot = file.name.lastIndexOf('.');
  input.setSelectionRange(0, dot > 0 ? dot : file.name.length);

  const commit = async () => {
    const newName = input.value.trim();
    if (newName && newName !== file.name) {
      try {
        const resp = await patchName(id, newName);
        if (resp.ok) file.name = newName;
      } catch (err) {
        console.warn('[files] rename error:', err);
      }
    }
    state.renamingId = null;
    renderGrid();
  };
  const cancel = () => { state.renamingId = null; renderGrid(); };

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); commit(); }
    else if (e.key === 'Escape') { e.preventDefault(); cancel(); }
    e.stopPropagation();
  });
  input.addEventListener('blur', commit);
}
