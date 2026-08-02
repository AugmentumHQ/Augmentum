/**
 * consumption/detail.js — shared item detail view (Media first; built
 * for cast-control/TV to adopt later, which is why playback is a
 * callback, not an import of the local player).
 *
 * Data comes from endpoints the Files surface already battle-tested:
 *   /api/media/details/{id}    — normalized rich detail (overview, cast,
 *                                ratings, tagline, next_up, children,
 *                                playback track inventory, gutenberg)
 *   /api/media/related/{id}    — author / narrator / genre peers
 *   /api/media/person/...      — cast profiles + filmography + headshots
 *   /api/media/progress/{id}   — mark watched / unwatched (is_finished)
 *   /api/media/selection/{id}  — persist version/audio/subtitle choice
 *
 * Playback-adjacent interactions delegate to the canonical players:
 * chapter taps and podcast episode switches drive the global audio
 * singleton (media-player.js) directly — same contract the Files
 * detail panel uses — and video plays through the onPlayFile callback.
 *
 * Degrades gracefully: when the details endpoint can't serve an entry
 * (local rows, provider offline), the view renders from the tile +
 * file_index source_metadata (overview/genres/year are synced locally).
 *
 * Comfort canon: soft backdrop wash (no autoplay, no parallax), quiet
 * idle state, theme tokens only.
 */

import { escapeHtml, showToast } from '../app.js';
import {
  fetchMediaDetails, pushMediaProgress, personImageUrl, mediaBackdropUrl,
  updateMediaPlaybackSelection, fetchPersonProfile, downloadUrl,
  fetchBookmarks, addBookmark, deleteBookmark,
} from '../files/api.js';
import { formatDuration } from './tile.js';

let _token = 0;

/**
 * Render the detail view for a tile into `container`.
 *
 * callbacks:
 *   onPlayFile(fileEntry, opts)  — REQUIRED. Hand off to the canonical
 *                                  player cascade (or cast dispatch on TV).
 *   onOpenItem(pseudoTile)       — open another item's detail (related
 *                                  strip, person filmography).
 *   onCast(tile, anchorEl)       — cast-picker hook; omit to hide the button.
 *   fetchFileEntry(fileId)       — REQUIRED. Resolve a full file_index row.
 */
export async function renderDetail(container, tile, callbacks = {}) {
  const token = ++_token;
  const { onPlayFile, onOpenItem, onCast, fetchFileEntry } = callbacks;

  container.innerHTML = _skeletonHtml(tile);

  let file = null;
  let rich = null;
  try {
    [file, rich] = await Promise.all([
      fetchFileEntry ? fetchFileEntry(tile.file_id) : null,
      fetchMediaDetails(tile.file_id),
    ]);
  } catch (err) {
    console.warn('[media-detail] load failed:', err);
  }
  if (token !== _token || !container.isConnected) return;
  if (!file && !rich) {
    container.innerHTML = `<div class="media-detail-error">Couldn't load details for this item.</div>`;
    return;
  }

  const meta = (file && typeof file.source_metadata === 'object' && file.source_metadata) || {};
  const d = _mergeDetail(tile, meta, rich);

  container.innerHTML = _detailHtml(d);
  _wireActions(container, { d, tile, file, onPlayFile, onCast, fetchFileEntry, token });
  _wireOverviewClamp(container);
  if (d.isSeries) _wireSeriesEpisodes(container, d, { onPlayFile, fetchFileEntry, token });
  if (d.isPodcast) _wirePodcastEpisodes(container, d, { token });
  if (d.kind === 'audio' && !d.isPodcast) {
    _wireChapterRows(container, d, { token });
    _wireBookmarks(container, d, { token });
  }
  if (d.kind === 'video') _wireTrackPickers(container, d);
  _wireCastStrip(container, d, { onOpenItem });
  _wireRelatedPivots(container, d, { onOpenItem, token });
}

/* ── Data merge ────────────────────────────────────────────────── */

function _mergeDetail(tile, meta, rich) {
  const r = rich || {};
  const kind = (tile.kind || '').toLowerCase();
  const entityKind = (r.entity_kind || tile.entity_kind || meta.entity_kind || '').toLowerCase();
  const isSeries = entityKind === 'series';
  const isPodcast = entityKind === 'podcast';
  const durationS = Number(r.duration_s ?? meta.duration_s ?? tile.duration_s) || 0;
  const progressPct = Number(r.progress_pct ?? meta.progress_pct ?? tile.progress_pct) || 0;
  const currentTimeS = Number(r.current_time_s ?? meta.current_time_s) || 0;
  const cast = Array.isArray(r.people?.cast) ? r.people.cast : [];
  return {
    fileId: tile.file_id,
    kind,
    entityKind,
    isSeries,
    isPodcast,
    title: r.title || tile.title || 'Untitled',
    tagline: r.tagline || '',
    overview: r.description || meta.overview || '',
    year: Number(r.published_year ?? meta.year ?? tile.year) || 0,
    endYear: Number(r.end_year) || 0,
    status: (r.status || '').toLowerCase(),
    durationS,
    progressPct,
    currentTimeS,
    isFinished: !!(r.is_finished ?? meta.is_finished ?? tile.is_finished),
    officialRating: r.official_rating || '',
    communityRating: Number(r.community_rating) || 0,
    network: r.network || '',
    seasonCount: Number(r.season_count) || 0,
    episodeCount: Number(r.episode_count) || 0,
    libraryName: r.library_name || meta.library_name || '',
    genres: (Array.isArray(r.genres) && r.genres.length ? r.genres : (meta.genres || []))
      .map((g) => String(g)).filter(Boolean),
    author: r.author || meta.author || '',
    narrator: r.narrator || meta.narrator || '',
    cast,
    chapters: Array.isArray(r.chapters) ? r.chapters : [],
    children: Array.isArray(r.children) ? r.children : [],
    nextUp: r.next_up || null,
    selectedEpisodeId: String(r.selected_episode_id || ''),
    playback: (r.playback && Array.isArray(r.playback.media_sources)) ? r.playback : null,
    gutenbergStatus: String(r.gutenberg_status || ''),
    gutenbergWordCount: Number(r.gutenberg_word_count) || 0,
    hasBackdrop: !!(r.has_backdrop ?? (tile.backdrop_url && tile.kind === 'video')),
    coverUrl: tile.cover_url || r.cover_url || '',
    playable: r.playable !== undefined ? !!r.playable : !!meta.stream_path,
    hasAuthorAxis: !!(meta.author_normalized || r.author),
    hasNarratorAxis: !!meta.narrator_normalized,
    hasSeriesAxis: !!(meta.series_normalized || meta.series_name || r.series_name),
    seriesName: r.series_name || meta.series_name || '',
  };
}

/* ── Markup ────────────────────────────────────────────────────── */

function _skeletonHtml(tile) {
  return `
    <div class="media-detail" data-loading>
      <div class="media-detail-hero"></div>
      <div class="media-detail-main">
        <div class="media-detail-poster">
          ${tile.cover_url ? `<img src="${escapeHtml(tile.cover_url)}" alt="">` : ''}
        </div>
        <div class="media-detail-info">
          <h2 class="media-detail-title">${escapeHtml(tile.title || '')}</h2>
          <div class="media-detail-skeleton" aria-hidden="true">
            <span></span><span></span><span></span>
          </div>
        </div>
      </div>
    </div>`;
}

function _metaRowHtml(d) {
  const bits = [];
  if (d.year) {
    let span = String(d.year);
    if (d.isSeries) {
      if (d.endYear && d.endYear !== d.year) span += `–${d.endYear}`;
      else if (d.status === 'continuing') span += '–';
    }
    bits.push(`<span>${escapeHtml(span)}</span>`);
  }
  if (!d.isSeries && !d.isPodcast && d.durationS) {
    bits.push(`<span>${escapeHtml(formatDuration(d.durationS))}</span>`);
  }
  if (d.isSeries && d.seasonCount) {
    bits.push(`<span>${d.seasonCount} season${d.seasonCount === 1 ? '' : 's'}</span>`);
  }
  if (d.isSeries && d.episodeCount) {
    bits.push(`<span>${d.episodeCount} episodes</span>`);
  }
  if (d.officialRating) {
    bits.push(`<span class="media-detail-cert">${escapeHtml(d.officialRating)}</span>`);
  }
  if (d.communityRating > 0) {
    bits.push(`<span class="media-detail-star">★ ${d.communityRating.toFixed(1)}</span>`);
  }
  if (d.network) bits.push(`<span>${escapeHtml(d.network)}</span>`);
  if (d.kind === 'audio' && d.author) bits.push(`<span>${escapeHtml(d.author)}</span>`);
  if (d.kind === 'audio' && d.narrator) {
    bits.push(`<span>read by ${escapeHtml(d.narrator)}</span>`);
  }
  if (d.libraryName) bits.push(`<span class="media-detail-lib">${escapeHtml(d.libraryName)}</span>`);
  return bits.join('<span class="media-detail-dot" aria-hidden="true">·</span>');
}

function _playLabel(d) {
  if (d.isSeries) return d.nextUp?.label || 'Play';
  if (d.isPodcast) return d.selectedEpisodeId ? 'Play episode' : 'Pick an episode';
  if (d.isFinished) return d.kind === 'audio' ? 'Listen again' : 'Watch again';
  if (d.progressPct > 0.5 && d.durationS) {
    const left = formatDuration(d.durationS * (1 - d.progressPct / 100));
    return left ? `Resume · ${left} left` : 'Resume';
  }
  return d.kind === 'audio' ? 'Listen' : 'Play';
}

function _actionsHtml(d, { canCast }) {
  const canPlay = d.isSeries ? !!d.nextUp : (d.playable && !(d.isPodcast && !d.selectedEpisodeId));
  // Watched toggle only on leaf items — provider semantics for marking
  // a whole series are inconsistent, so the series view does it per
  // episode instead.
  const canToggleWatched = !d.isSeries && !d.isPodcast
    && (d.kind === 'video' || d.kind === 'audio') && d.playable;
  // "Start over" mirrors the Files audiobook CTA's secondary — only
  // meaningful mid-listen (the global player owns the seek).
  const canStartOver = d.kind === 'audio' && !d.isPodcast && d.playable
    && !d.isFinished && d.progressPct > 0.5;
  const canQueue = !d.isSeries && d.playable && (d.kind === 'audio' || d.kind === 'video');
  // Bookmarks are an audiobook (and book-podcast) affordance — a saved
  // spot you can jump back to. Most useful mid-listen; we let the user
  // drop one any time and it lands at the current/resume position.
  const canBookmark = d.kind === 'audio' && !d.isPodcast && d.playable;
  const readAlong = d.kind === 'audio' && d.gutenbergStatus === 'fetched';
  const readAlongPending = d.kind === 'audio' && d.gutenbergStatus === 'fetching';
  return `
    <div class="media-detail-actions">
      ${canPlay ? `
        <button class="media-detail-play" type="button" data-detail-play>
          <svg viewBox="0 0 24 24" width="15" height="15" fill="currentColor" aria-hidden="true"><path d="M8 5.1v13.8L19 12z"/></svg>
          <span>${escapeHtml(_playLabel(d))}</span>
        </button>` : ''}
      ${canStartOver ? `
        <button class="media-detail-secondary" type="button" data-detail-restart>
          <span>Start over</span>
        </button>` : ''}
      ${canBookmark ? `
        <button class="media-detail-secondary" type="button" data-detail-bookmark title="Bookmark this spot">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/></svg>
          <span>Bookmark</span>
        </button>` : ''}
      ${canToggleWatched ? `
        <button class="media-detail-secondary" type="button" data-detail-watched
                aria-pressed="${d.isFinished}">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>
          <span>${d.isFinished ? 'Watched' : 'Mark watched'}</span>
        </button>` : ''}
      ${readAlong ? `
        <button class="media-detail-secondary" type="button" data-detail-readalong>
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>
          <span>Read along</span>
        </button>` : ''}
      ${readAlongPending ? `
        <span class="media-detail-pending">Fetching book text…</span>` : ''}
      ${canQueue ? `
        <button class="media-detail-secondary" type="button" data-detail-playlist title="Add to playlist">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="3" y1="6" x2="13" y2="6"/><line x1="3" y1="12" x2="13" y2="12"/><line x1="3" y1="18" x2="9" y2="18"/><line x1="18" y1="9" x2="18" y2="15"/><line x1="15" y1="12" x2="21" y2="12"/></svg>
          <span>Playlist</span>
        </button>` : ''}
      ${canCast ? `
        <button class="media-detail-secondary" type="button" data-detail-cast>
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="13" rx="2"/><line x1="8" y1="20.5" x2="16" y2="20.5"/></svg>
          <span>Cast</span>
        </button>` : ''}
      ${(!d.isSeries && d.playable) ? `
        <button class="media-detail-secondary media-detail-iconbtn" type="button" data-detail-download
                title="Download" aria-label="Download">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        </button>` : ''}
    </div>`;
}

function _castStripHtml(d) {
  if (!d.cast.length) return '';
  const cards = d.cast.map((p, i) => {
    const img = p.image_tag && p.person_id
      ? `<img src="${escapeHtml(personImageUrl(d.fileId, p.person_id))}" alt="" loading="lazy" decoding="async" onerror="this.closest('.media-detail-person-photo').classList.add('is-missing'); this.remove()">`
      : '';
    const clickable = !!p.person_id;
    return `
      <${clickable ? 'button type="button"' : 'div'} class="media-detail-person${clickable ? ' is-clickable' : ''}"
          ${clickable ? `data-person-idx="${i}"` : ''}
          title="${escapeHtml(p.name)}${p.role ? ` as ${escapeHtml(p.role)}` : ''}">
        <div class="media-detail-person-photo${img ? '' : ' is-missing'}">
          ${img || `<span>${escapeHtml((p.name || '?').slice(0, 1))}</span>`}
        </div>
        <div class="media-detail-person-name">${escapeHtml(p.name)}</div>
        ${p.role ? `<div class="media-detail-person-role">${escapeHtml(p.role)}</div>` : ''}
      </${clickable ? 'button' : 'div'}>`;
  }).join('');
  return `
    <section class="media-detail-section">
      <h3 class="media-detail-section-title">Cast</h3>
      <div class="media-detail-strip" data-cast-strip>${cards}</div>
    </section>`;
}

/* One episode card for the series picker. Episodes come from
 * /api/cast/library/episodes (full tiles: cover_url, duration_s,
 * progress_pct, is_finished, file_id), so each row gets a 16:9 still,
 * runtime, a resume bar, and a watched check — Plex/Plappa-level rows
 * rather than a bare title list. */
function _episodeCardHtml(ep) {
  const sn = Number(ep.season_number) || 0;
  const en = Number(ep.episode_number) || 0;
  const label = sn > 0 && en
    ? `S${sn}E${String(en).padStart(2, '0')}`
    : (en ? `Ep. ${en}` : '');
  const pct = Math.max(0, Math.min(100, Number(ep.progress_pct) || 0));
  const finished = !!ep.is_finished;
  const dur = Number(ep.duration_s) || 0;
  const durLabel = dur ? formatDuration(dur) : '';
  const cover = ep.cover_url || ep.backdrop_url || '';
  const fid = ep.file_id || '';
  return `
    <div class="media-ep-row${fid ? '' : ' is-disabled'}" data-episode-id="${escapeHtml(fid)}"
         role="button" tabindex="${fid ? '0' : '-1'}">
      <div class="media-ep-thumb${finished ? ' is-watched' : ''}">
        ${cover
          ? `<img src="${escapeHtml(cover)}" alt="" loading="lazy" decoding="async" onerror="this.style.display='none'">`
          : `<span class="media-ep-thumb-ph">${escapeHtml(label || '▶')}</span>`}
        ${finished ? '<span class="media-ep-check" aria-hidden="true">✓</span>' : ''}
        ${durLabel ? `<span class="media-ep-dur">${escapeHtml(durLabel)}</span>` : ''}
        ${(pct > 0.5 && !finished)
          ? `<span class="media-ep-prog"><span style="width:${pct.toFixed(1)}%"></span></span>`
          : ''}
      </div>
      <div class="media-ep-body">
        ${label ? `<div class="media-ep-num">${escapeHtml(label)}</div>` : ''}
        <div class="media-ep-title">${escapeHtml(ep.title || 'Untitled')}</div>
      </div>
    </div>`;
}

/* Podcast episode list — mirrors the Files panel's rows (episode_id /
 * name / is_finished / progress_pct on each child) with the same
 * selected-episode highlight. */
function _podcastEpisodesHtml(d) {
  if (!d.isPodcast || !d.children.length) return '';
  const rows = d.children.map((ep, i) => {
    const isActive = String(ep.episode_id || '') === d.selectedEpisodeId;
    let side = '';
    if (ep.is_finished) side = '<span class="media-detail-chip">Finished</span>';
    else if ((Number(ep.progress_pct) || 0) > 0.5) {
      side = `<span class="media-detail-chip is-resume">${Math.round(Number(ep.progress_pct))}%</span>`;
    }
    return `
      <button type="button" class="media-detail-episode-row${isActive ? ' is-active' : ''}"
              data-podcast-episode="${escapeHtml(ep.episode_id || '')}">
        <span class="media-detail-episode-title">${escapeHtml(ep.name || `Episode ${i + 1}`)}</span>
        ${side}
      </button>`;
  }).join('');
  return `
    <section class="media-detail-section">
      <h3 class="media-detail-section-title">Episodes <span class="media-detail-count">${d.children.length}</span></h3>
      <div class="media-detail-episode-list" data-podcast-list>${rows}</div>
    </section>`;
}

function _fmtTs(seconds) {
  const start = Number(seconds) || 0;
  const h = Math.floor(start / 3600);
  const m = Math.floor((start % 3600) / 60);
  const s = Math.floor(start % 60);
  return h > 0
    ? `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
    : `${m}:${String(s).padStart(2, '0')}`;
}

/* Audiobook chapters — clickable, with per-chapter listened state
 * derived from the book position (same derivation as the Files
 * panel's _deriveChapterStates). */
function _chaptersHtml(d) {
  if (d.kind !== 'audio' || d.isPodcast || !d.chapters.length) return '';
  const rows = d.chapters.map((c, i) => {
    const start = Number(c.start) || 0;
    const end = Number(d.chapters[i + 1]?.start) || d.durationS || Infinity;
    let state = 'unplayed';
    if (d.isFinished || (d.currentTimeS >= end && end !== Infinity)) state = 'played';
    else if (d.currentTimeS > start) state = 'playing';
    return `
      <li>
        <button type="button" class="media-detail-chapter is-${state}"
                data-chapter-start="${start}" title="Play from ${escapeHtml(_fmtTs(start))}">
          <span class="media-detail-chapter-ts">${escapeHtml(_fmtTs(start))}</span>
          <span class="media-detail-chapter-title">${escapeHtml(c.title || `Chapter ${i + 1}`)}</span>
          ${state === 'played' ? '<span class="media-detail-chapter-done" aria-hidden="true">✓</span>' : ''}
        </button>
      </li>`;
  }).join('');
  return `
    <section class="media-detail-section">
      <h3 class="media-detail-section-title">Chapters <span class="media-detail-count">${d.chapters.length}</span></h3>
      <ol class="media-detail-chapters" role="list">${rows}</ol>
    </section>`;
}

/* Version & track pickers — same inventory the floating player's
 * popover and the Files detail panel use (details.playback). */
function _trackPickersHtml(d) {
  const pb = d.playback;
  if (d.kind !== 'video' || !pb?.media_sources?.length || !d.playable) return '';
  const selected = pb.media_sources.find((s) => s.id === pb.selected_media_source_id)
    || pb.media_sources[0];
  const sourceOptions = pb.media_sources.map((s) =>
    `<option value="${escapeHtml(s.id)}"${s.id === selected.id ? ' selected' : ''}>${escapeHtml(s.label || s.id)}</option>`).join('');
  const audioOptions = (selected.audio_tracks || []).map((t) =>
    `<option value="${t.index}"${Number(pb.selected_audio_stream_index) === Number(t.index) ? ' selected' : ''}>${escapeHtml(t.label || `Audio ${t.index}`)}</option>`).join('');
  const subOptions = (selected.subtitle_tracks || []).map((t) =>
    `<option value="${t.index}"${Number(pb.selected_subtitle_stream_index) === Number(t.index) ? ' selected' : ''}>${escapeHtml(t.label || `Subtitle ${t.index}`)}</option>`).join('');
  return `
    <section class="media-detail-section media-detail-tracks">
      <h3 class="media-detail-section-title">Playback</h3>
      <div class="media-detail-track-row">
        ${pb.has_multiple_versions ? `
          <label class="media-detail-track">
            <span>Version</span>
            <select data-track-source>${sourceOptions}</select>
          </label>` : ''}
        ${(selected.audio_tracks || []).length > 1 ? `
          <label class="media-detail-track">
            <span>Audio</span>
            <select data-track-audio>${audioOptions}</select>
          </label>` : ''}
        ${(selected.subtitle_tracks || []).length > 1 ? `
          <label class="media-detail-track">
            <span>Subtitles</span>
            <select data-track-subtitle>${subOptions}</select>
          </label>` : ''}
      </div>
    </section>`;
}

/* Related pivots — audio gets explicit author/narrator chips (parity
 * with the Files panel's pivot buttons); video auto-loads the genre
 * axis with no chips. */
function _relatedSectionHtml(d) {
  const chips = [];
  // Series first — for a numbered series ("Vol. 10") it's the most
  // relevant pivot, so it leads and auto-loads when present.
  if (d.kind === 'audio' && d.hasSeriesAxis) {
    chips.push(`<button class="media-grid-chip" type="button" data-related-axis="series">More in ${escapeHtml(d.seriesName || 'this series')}</button>`);
  }
  if (d.kind === 'audio' && d.hasAuthorAxis) {
    chips.push(`<button class="media-grid-chip" type="button" data-related-axis="author">More by ${escapeHtml(d.author || 'author')}</button>`);
  }
  if (d.kind === 'audio' && d.hasNarratorAxis && d.narrator) {
    chips.push(`<button class="media-grid-chip" type="button" data-related-axis="narrator">Narrated by ${escapeHtml(d.narrator)}</button>`);
  }
  // "More like this" (shared genres) — now offered for audio too, not
  // just video. The genre axis already existed server-side; it was never
  // wired into the audio detail page.
  if (d.kind === 'audio' && d.genres.length) {
    chips.push(`<button class="media-grid-chip" type="button" data-related-axis="genre">More like this</button>`);
  }
  // First chip mirrors whichever axis auto-loads below.
  if (chips.length) chips[0] = chips[0].replace('media-grid-chip', 'media-grid-chip is-active');
  return `
    <section class="media-detail-section media-detail-related-host" data-related-section hidden>
      <h3 class="media-detail-section-title" data-related-title></h3>
      ${chips.length ? `<div class="media-detail-related-chips">${chips.join('')}</div>` : ''}
      <div class="media-detail-strip" data-related-strip></div>
    </section>`;
}

function _detailHtml(d) {
  const heroImg = d.hasBackdrop
    ? `<img src="${escapeHtml(mediaBackdropUrl(d.fileId))}" alt="" loading="lazy" decoding="async" onerror="this.remove()">`
    : (d.coverUrl ? `<img class="media-detail-hero-coverblur" src="${escapeHtml(d.coverUrl)}" alt="" aria-hidden="true">` : '');
  return `
    <div class="media-detail">
      <div class="media-detail-hero">${heroImg}<div class="media-detail-hero-wash" aria-hidden="true"></div></div>
      <div class="media-detail-main">
        <div class="media-detail-poster">
          ${d.coverUrl
            ? `<img src="${escapeHtml(d.coverUrl)}" alt="">`
            : `<div class="media-detail-poster-fallback">${escapeHtml(d.title.slice(0, 1))}</div>`}
          ${(!d.isFinished && d.progressPct > 0.5)
            ? `<div class="media-tile-progress" style="width:${Math.min(100, d.progressPct).toFixed(1)}%"></div>`
            : ''}
        </div>
        <div class="media-detail-info">
          <h2 class="media-detail-title">${escapeHtml(d.title)}</h2>
          ${d.tagline ? `<p class="media-detail-tagline">${escapeHtml(d.tagline)}</p>` : ''}
          <div class="media-detail-meta">${_metaRowHtml(d)}</div>
          ${_actionsHtml(d, { canCast: true })}
          ${d.overview ? `
            <div class="media-detail-overview" data-overview>
              <p>${escapeHtml(d.overview)}</p>
              <button class="media-detail-more" type="button" data-overview-toggle hidden>More</button>
            </div>` : ''}
          ${d.genres.length ? `
            <div class="media-detail-genres">${d.genres.map((g) =>
              `<span class="media-detail-genre">${escapeHtml(g)}</span>`).join('')}</div>` : ''}
        </div>
      </div>
      ${_trackPickersHtml(d)}
      ${_castStripHtml(d)}
      ${d.isSeries ? '<section class="media-detail-section media-detail-episodes-section" data-episodes-host hidden></section>' : ''}
      ${_podcastEpisodesHtml(d)}
      ${_chaptersHtml(d)}
      ${(d.kind === 'audio' && !d.isPodcast) ? '<section class="media-detail-section media-detail-bookmarks-section" data-bookmarks-host hidden></section>' : ''}
      ${_relatedSectionHtml(d)}
    </div>`;
}

/* ── Wiring ────────────────────────────────────────────────────── */

function _wireActions(container, { d, tile, file, onPlayFile, onCast, fetchFileEntry, token }) {
  const playBtn = container.querySelector('[data-detail-play]');
  if (playBtn && onPlayFile) {
    playBtn.addEventListener('click', async () => {
      if (d.isSeries && d.nextUp?.file_id) {
        const epFile = fetchFileEntry ? await fetchFileEntry(d.nextUp.file_id) : null;
        if (epFile) onPlayFile(epFile);
        return;
      }
      if (file) onPlayFile(file);
    });
  }

  container.querySelector('[data-detail-restart]')?.addEventListener('click', async () => {
    try {
      const player = await import('../media-player.js');
      await player.play(d.fileId);
      player.seek(0);
    } catch (err) {
      console.warn('[media-detail] restart failed:', err);
    }
  });

  container.querySelector('[data-detail-readalong]')?.addEventListener('click', async () => {
    try {
      const mod = await import('../files/read-along.js');
      mod.openReadAlong(d.fileId, d.title);
    } catch (err) {
      console.warn('[media-detail] read-along open failed:', err);
      showToast("Couldn't open the book text just now.", 'error', 2600);
    }
  });

  container.querySelector('[data-detail-playlist]')?.addEventListener('click', () => {
    // Carry both the playback `kind` (audio/video — the player needs it)
    // AND `entityKind` (the content category — the playlist boundary groups
    // by family: watch/music/spoken/comics). The chooser shows feedback,
    // so no premature "Added" toast here.
    window.dispatchEvent(new CustomEvent('playlist:add-item', {
      detail: {
        type: 'file',
        fileId: d.fileId,
        name: d.title,
        kind: d.kind === 'video' ? 'video' : 'audio',
        entityKind: d.entityKind || '',
        thumbnail: d.coverUrl || '',
      },
    }));
  });

  container.querySelector('[data-detail-download]')?.addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = downloadUrl(d.fileId);
    a.download = '';
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  const castBtn = container.querySelector('[data-detail-cast]');
  if (castBtn) {
    if (onCast) {
      castBtn.addEventListener('click', (ev) => onCast(tile, ev.currentTarget));
    } else {
      castBtn.remove();
    }
  }

  const watchedBtn = container.querySelector('[data-detail-watched]');
  if (watchedBtn) {
    watchedBtn.addEventListener('click', async () => {
      const target = !d.isFinished;
      watchedBtn.disabled = true;
      const resp = await pushMediaProgress(d.fileId, {
        current_time_s: target ? d.durationS : 0,
        duration_s: d.durationS,
        is_finished: target,
      });
      if (token !== _token || !container.isConnected) return;
      watchedBtn.disabled = false;
      if (!resp) {
        console.warn('[media-detail] watched toggle failed for', d.fileId);
        return;
      }
      d.isFinished = target;
      if (target) d.progressPct = 0;
      watchedBtn.setAttribute('aria-pressed', String(target));
      watchedBtn.querySelector('span').textContent = target ? 'Watched' : 'Mark watched';
      const play = container.querySelector('[data-detail-play] span');
      if (play) play.textContent = _playLabel(d);
      const poster = container.querySelector('.media-detail-poster .media-tile-progress');
      if (poster && target) poster.remove();
    });
  }
}

function _wireOverviewClamp(container) {
  const host = container.querySelector('[data-overview]');
  if (!host) return;
  const p = host.querySelector('p');
  const btn = host.querySelector('[data-overview-toggle]');
  requestAnimationFrame(() => {
    if (p.scrollHeight > p.clientHeight + 4) btn.hidden = false;
  });
  btn.addEventListener('click', () => {
    const expanded = host.classList.toggle('is-expanded');
    btn.textContent = expanded ? 'Less' : 'More';
  });
}

/* Series episode picker. The /details children for a Series are
 * SEASONS, not episodes (the old flat-list render found nothing and
 * showed only the Play button — the "it just plays ep 1, no picker"
 * bug). This fetches the fully-resolved, season-grouped episode list
 * and renders a season selector + rich episode rows, defaulting to the
 * season the "Continue" target lives in so the picker opens in context. */
async function _wireSeriesEpisodes(container, d, { onPlayFile, fetchFileEntry, token }) {
  const host = container.querySelector('[data-episodes-host]');
  if (!host) return;
  host.hidden = false;
  host.innerHTML = `
    <h3 class="media-detail-section-title">Episodes</h3>
    <div class="media-ep-loading"><div class="media-detail-skeleton"><span></span><span></span></div></div>`;

  let data = null;
  try {
    const r = await fetch(
      `/api/cast/library/episodes/${encodeURIComponent(d.fileId)}`,
      { credentials: 'same-origin', cache: 'no-store' },
    );
    if (r.ok) data = await r.json();
  } catch (err) {
    console.warn('[media-detail] episodes fetch failed:', err);
  }
  if (token !== _token || !container.isConnected) return;

  const seasons = (data && Array.isArray(data.seasons))
    ? data.seasons.filter((s) => Array.isArray(s.episodes) && s.episodes.length)
    : [];
  if (!seasons.length) { host.hidden = true; return; }

  // Open on the season that holds the Continue target; else the first
  // real season (skip Specials/season 0 when there's anything else).
  const nextSeason = Number(d.nextUp?.season_number) || 0;
  let activeSeason = seasons.find((s) => s.season_number === nextSeason)?.season_number;
  if (activeSeason == null) {
    activeSeason = (seasons.find((s) => s.season_number > 0) || seasons[0]).season_number;
  }

  const multi = seasons.length > 1;
  const selectorHtml = multi ? `
    <div class="media-ep-seasons" role="tablist" aria-label="Seasons">
      ${seasons.map((s) => `
        <button type="button" class="media-ep-season${s.season_number === activeSeason ? ' is-active' : ''}"
                role="tab" data-season="${s.season_number}"
                aria-selected="${s.season_number === activeSeason}">${escapeHtml(s.label || `Season ${s.season_number}`)}</button>`).join('')}
    </div>` : '';
  const lone = !multi ? seasons[0] : null;

  host.innerHTML = `
    <div class="media-detail-section-head">
      <h3 class="media-detail-section-title">Episodes</h3>
      ${lone ? `<span class="media-detail-count">${lone.episodes.length}</span>` : ''}
    </div>
    ${selectorHtml}
    <div class="media-ep-list" data-ep-list></div>`;

  const listEl = host.querySelector('[data-ep-list]');
  const renderSeason = (sn) => {
    const season = seasons.find((s) => s.season_number === sn) || seasons[0];
    listEl.innerHTML = (season.episodes || []).map(_episodeCardHtml).join('');
  };
  renderSeason(activeSeason);

  host.querySelectorAll('[data-season]').forEach((tab) => {
    tab.addEventListener('click', () => {
      host.querySelectorAll('[data-season]').forEach((t) => {
        const on = t === tab;
        t.classList.toggle('is-active', on);
        t.setAttribute('aria-selected', String(on));
      });
      renderSeason(Number(tab.dataset.season));
    });
  });

  const flat = seasons.flatMap((s) => s.episodes || []);
  const activate = async (row) => {
    const fid = row?.dataset?.episodeId;
    if (!fid || row.classList.contains('is-disabled')) return;
    const epFile = fetchFileEntry ? await fetchFileEntry(fid) : null;
    if (epFile && onPlayFile) onPlayFile(epFile);
  };
  listEl.addEventListener('click', (ev) => activate(ev.target.closest('[data-episode-id]')));
  listEl.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' || ev.key === ' ') {
      const row = ev.target.closest('[data-episode-id]');
      if (row) { ev.preventDefault(); activate(row); }
    }
  });
}

/* Podcast episode switch — same flow the Files panel uses: re-fetch
 * details scoped to the episode, then hand the enriched record to the
 * global audio player. */
function _wirePodcastEpisodes(container, d, { token }) {
  const list = container.querySelector('[data-podcast-list]');
  if (!list) return;
  list.addEventListener('click', async (ev) => {
    const row = ev.target.closest('[data-podcast-episode]');
    if (!row) return;
    const episodeId = row.dataset.podcastEpisode;
    if (!episodeId) return;
    row.classList.add('is-loading');
    try {
      const rich = await fetchMediaDetails(d.fileId, { episodeId });
      const player = await import('../media-player.js');
      await player.play(d.fileId, { details: rich });
      if (token !== _token || !container.isConnected) return;
      d.selectedEpisodeId = episodeId;
      list.querySelectorAll('[data-podcast-episode]').forEach((r) =>
        r.classList.toggle('is-active', r === row));
      const play = container.querySelector('[data-detail-play] span');
      if (play) play.textContent = 'Play episode';
    } catch (err) {
      console.warn('[media-detail] episode switch failed:', err);
      showToast("Couldn't start that episode.", 'error', 2600);
    } finally {
      row.classList.remove('is-loading');
    }
  });
}

/* Chapter click-to-seek — same contract as the Files panel: if this
 * book is already in the player, seek (and resume if paused);
 * otherwise start it and jump to the chapter. */
async function _wireChapterRows(container, d, { token } = {}) {
  const list = container.querySelector('.media-detail-chapters');
  container.querySelectorAll('[data-chapter-start]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const start = Number(btn.dataset.chapterStart) || 0;
      try {
        const player = await import('../media-player.js');
        const st = player.getState?.() || {};
        if (st.fileId === d.fileId) {
          player.seek(start);
          if (!st.isPlaying) player.resume();
        } else {
          await player.play(d.fileId);
          player.seek(start);
        }
      } catch (err) {
        console.warn('[media-detail] chapter seek failed:', err);
      }
    });
  });

  // Live-follow: while THIS book is playing, keep the chapter list's
  // played/playing state in sync with the player and auto-scroll the
  // active chapter into view as it advances (no scroll on first paint —
  // the user just opened the panel and expects to see the top).
  if (!list) return;
  try {
    const player = await import('../media-player.js');
    let lastIdx = -1;
    let primed = false;
    const unsub = player.subscribe((st) => {
      if ((token != null && token !== _token) || !container.isConnected) {
        unsub();
        return;
      }
      if (st.fileId !== d.fileId) return;
      const idx = st.currentChapterIdx;
      const btns = list.querySelectorAll('[data-chapter-start]');
      btns.forEach((b, i) => {
        b.classList.toggle('is-played', idx >= 0 && i < idx);
        b.classList.toggle('is-playing', i === idx);
        b.classList.toggle('is-unplayed', idx < 0 || i > idx);
      });
      if (idx !== lastIdx) {
        if (primed && idx >= 0 && btns[idx]) {
          btns[idx].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
        lastIdx = idx;
      }
      primed = true;
    });
  } catch (err) {
    console.warn('[media-detail] chapter follow wiring failed:', err);
  }
}

/* ── Bookmarks ─────────────────────────────────────────────────── */

function _bookmarkRowHtml(bm) {
  const pos = Number(bm.position_s) || 0;
  return `
    <li class="media-bm-row" data-bm-id="${escapeHtml(bm.id)}">
      <button type="button" class="media-bm-jump" data-bm-jump="${pos}" title="Jump to this spot">
        <span class="media-bm-ts">${escapeHtml(_fmtTs(pos))}</span>
        <span class="media-bm-body">
          <span class="media-bm-label">${escapeHtml(bm.label || 'Bookmark')}</span>
          ${bm.note ? `<span class="media-bm-note">${escapeHtml(bm.note)}</span>` : ''}
        </span>
      </button>
      <button type="button" class="media-bm-del" data-bm-del aria-label="Remove bookmark" title="Remove bookmark">
        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </li>`;
}

/* Bookmarks list for an audiobook. The host stays hidden until there's
 * at least one bookmark (comfort over an empty box); it reveals as soon
 * as one is added — from the detail "Bookmark" button, the mini-player,
 * or the companion ("mark my place"), all of which fire
 * `media-player:bookmark-added`. Jump seeks the global player; delete is
 * an inline X. */
async function _wireBookmarks(container, d, { token } = {}) {
  const host = container.querySelector('[data-bookmarks-host]');
  if (!host) return;

  const refresh = async () => {
    const items = await fetchBookmarks(d.fileId);
    if (token !== _token || !container.isConnected) return;
    if (!items.length) {
      host.hidden = true;
      host.innerHTML = '';
      return;
    }
    host.hidden = false;
    host.innerHTML = `
      <div class="media-detail-section-head">
        <h3 class="media-detail-section-title">Bookmarks</h3>
        <span class="media-detail-count">${items.length}</span>
      </div>
      <ul class="media-bm-list" role="list">${items.map(_bookmarkRowHtml).join('')}</ul>`;
  };

  // Delegated jump + delete — bound once to the persistent host element,
  // so it survives the innerHTML swaps in refresh().
  host.addEventListener('click', async (ev) => {
    const del = ev.target.closest('[data-bm-del]');
    if (del) {
      const row = del.closest('[data-bm-id]');
      const id = row?.dataset.bmId;
      if (!id) return;
      del.disabled = true;
      const ok = await deleteBookmark(id);
      if (!ok) { del.disabled = false; showToast("Couldn't remove that bookmark.", 'error', 1800); return; }
      row.remove();
      const remaining = host.querySelectorAll('[data-bm-id]').length;
      const countEl = host.querySelector('.media-detail-count');
      if (countEl) countEl.textContent = String(remaining);
      if (!remaining) { host.hidden = true; host.innerHTML = ''; }
      return;
    }
    const jump = ev.target.closest('[data-bm-jump]');
    if (jump) {
      const pos = Number(jump.dataset.bmJump) || 0;
      try {
        const player = await import('../media-player.js');
        const st = player.getState?.() || {};
        if (st.fileId === d.fileId) {
          player.seek(pos);
          if (!st.isPlaying) player.resume();
        } else {
          await player.play(d.fileId);
          player.seek(pos);
        }
      } catch (err) {
        console.warn('[media-detail] bookmark jump failed:', err);
      }
    }
  });

  // "Bookmark" action button: bookmark the live position when this book
  // is the one playing; otherwise drop one at its resume position.
  const addBtn = container.querySelector('[data-detail-bookmark]');
  if (addBtn) {
    addBtn.addEventListener('click', async () => {
      addBtn.disabled = true;
      let bm = null;
      try {
        const player = await import('../media-player.js');
        const st = player.getState?.() || {};
        if (st.fileId === d.fileId) {
          bm = await player.addBookmarkHere();   // fires bookmark-added → refresh
        } else {
          const pos = d.currentTimeS || 0;
          bm = await addBookmark(d.fileId, { position_s: pos, label: _fmtTs(pos) });
          if (bm) await refresh();
        }
      } catch (err) {
        console.warn('[media-detail] add bookmark failed:', err);
      } finally {
        addBtn.disabled = false;
      }
      showToast(bm ? 'Bookmarked.' : "Couldn't save the bookmark.", bm ? 'info' : 'error', 1600);
    });
  }

  // Refresh when a bookmark is added anywhere (player/mini-player/companion).
  const onAdded = (e) => {
    if (token !== _token || !container.isConnected) {
      window.removeEventListener('media-player:bookmark-added', onAdded);
      return;
    }
    if (e.detail?.fileId !== d.fileId) return;
    refresh();
  };
  window.addEventListener('media-player:bookmark-added', onAdded);

  refresh();
}

/* Version / audio / subtitle pickers. Persists via the same
 * /api/media/selection endpoint and fires the same
 * `media-video-selection` event the Files detail panel broadcasts, so
 * an open floating player on this file re-tunes live. */
function _wireTrackPickers(container, d) {
  const section = container.querySelector('.media-detail-tracks');
  if (!section || !d.playback) return;
  const pb = d.playback;

  const currentSource = () =>
    pb.media_sources.find((s) => s.id === pb.selected_media_source_id) || pb.media_sources[0];

  const commit = async () => {
    const selection = {
      media_source_id: pb.selected_media_source_id,
      audio_stream_index: pb.selected_audio_stream_index,
      subtitle_stream_index: pb.selected_subtitle_stream_index,
    };
    const resp = await updateMediaPlaybackSelection(d.fileId, selection);
    if (!resp) {
      showToast("Couldn't save the playback choice.", 'error', 2600);
      return;
    }
    window.dispatchEvent(new CustomEvent('media-video-selection', {
      detail: {
        fileId: d.fileId,
        selection: {
          mediaSourceId: pb.selected_media_source_id,
          audioStreamIndex: pb.selected_audio_stream_index,
          subtitleStreamIndex: pb.selected_subtitle_stream_index,
        },
        origin: 'media-detail',
      },
    }));
  };

  const rebuildTrackSelects = () => {
    const src = currentSource();
    const audioSel = section.querySelector('[data-track-audio]');
    const subSel = section.querySelector('[data-track-subtitle]');
    if (audioSel) {
      audioSel.innerHTML = (src.audio_tracks || []).map((t) =>
        `<option value="${t.index}"${Number(pb.selected_audio_stream_index) === Number(t.index) ? ' selected' : ''}>${escapeHtml(t.label || `Audio ${t.index}`)}</option>`).join('');
    }
    if (subSel) {
      subSel.innerHTML = (src.subtitle_tracks || []).map((t) =>
        `<option value="${t.index}"${Number(pb.selected_subtitle_stream_index) === Number(t.index) ? ' selected' : ''}>${escapeHtml(t.label || `Subtitle ${t.index}`)}</option>`).join('');
    }
  };

  section.querySelector('[data-track-source]')?.addEventListener('change', (ev) => {
    pb.selected_media_source_id = ev.target.value;
    const src = currentSource();
    // New source = new track inventory; re-derive defaults the same
    // way the Files panel does (is_default flag, Off for subs).
    const defAudio = (src.audio_tracks || []).find((t) => t.is_default) || (src.audio_tracks || [])[0];
    pb.selected_audio_stream_index = defAudio ? Number(defAudio.index) : null;
    const defSub = (src.subtitle_tracks || []).find((t) => t.is_default);
    pb.selected_subtitle_stream_index = defSub ? Number(defSub.index) : -1;
    rebuildTrackSelects();
    commit();
  });
  section.querySelector('[data-track-audio]')?.addEventListener('change', (ev) => {
    pb.selected_audio_stream_index = Number(ev.target.value);
    commit();
  });
  section.querySelector('[data-track-subtitle]')?.addEventListener('change', (ev) => {
    pb.selected_subtitle_stream_index = Number(ev.target.value);
    commit();
  });
}

/* Cast member click → profile popover with filmography; in-library
 * works open their own detail via onOpenItem. */
function _wireCastStrip(container, d, { onOpenItem }) {
  const strip = container.querySelector('[data-cast-strip]');
  if (!strip) return;
  strip.addEventListener('click', async (ev) => {
    const card = ev.target.closest('[data-person-idx]');
    if (!card) return;
    const person = d.cast[Number(card.dataset.personIdx)];
    if (!person?.person_id) return;
    _closePersonPopover();
    const pop = document.createElement('div');
    pop.className = 'media-person-popover';
    pop.setAttribute('role', 'dialog');
    pop.setAttribute('aria-label', person.name);
    pop.innerHTML = `<div class="media-person-popover-body"><div class="media-detail-skeleton"><span></span><span></span></div></div>`;
    document.body.appendChild(pop);
    const close = () => {
      pop.remove();
      document.removeEventListener('pointerdown', onAway, true);
      document.removeEventListener('keydown', onKey, true);
    };
    const onAway = (e) => { if (!pop.contains(e.target)) close(); };
    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        close();
      }
    };
    document.addEventListener('pointerdown', onAway, true);
    document.addEventListener('keydown', onKey, true);

    const profile = await fetchPersonProfile(d.fileId, person.person_id);
    if (!pop.isConnected) return;
    if (!profile) {
      pop.querySelector('.media-person-popover-body').innerHTML =
        `<p class="media-person-empty">No profile available for ${escapeHtml(person.name)}.</p>`;
      return;
    }
    const works = (profile.works || []).filter((w) => w.in_library && w.file_id);
    pop.querySelector('.media-person-popover-body').innerHTML = `
      <div class="media-person-head">
        <div class="media-detail-person-photo${profile.has_image ? '' : ' is-missing'}">
          ${profile.has_image
            ? `<img src="${escapeHtml(personImageUrl(d.fileId, person.person_id))}" alt="">`
            : `<span>${escapeHtml((profile.name || '?').slice(0, 1))}</span>`}
        </div>
        <div>
          <div class="media-person-name">${escapeHtml(profile.name)}</div>
          ${person.role ? `<div class="media-detail-person-role">as ${escapeHtml(person.role)}</div>` : ''}
          ${profile.birth_place ? `<div class="media-person-meta">${escapeHtml(profile.birth_place)}</div>` : ''}
        </div>
      </div>
      ${profile.overview ? `<p class="media-person-bio">${escapeHtml(profile.overview.slice(0, 360))}${profile.overview.length > 360 ? '…' : ''}</p>` : ''}
      ${works.length ? `
        <div class="media-person-works-title">In your library</div>
        <div class="media-person-works">
          ${works.slice(0, 12).map((w) => `
            <button type="button" class="media-person-work" data-work-id="${escapeHtml(w.file_id)}">
              <span>${escapeHtml(w.name)}</span>${w.year ? `<span class="media-person-meta">${w.year}</span>` : ''}
            </button>`).join('')}
        </div>` : '<p class="media-person-empty">Nothing else of theirs in your library yet.</p>'}
    `;
    pop.querySelectorAll('[data-work-id]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const work = works.find((w) => w.file_id === btn.dataset.workId);
        if (!work) return;
        close();
        if (onOpenItem) {
          onOpenItem({
            file_id: work.file_id,
            title: work.name,
            kind: 'video',
            entity_kind: work.entity_kind || '',
            cover_url: `/api/media/cover/${encodeURIComponent(work.file_id)}`,
            play: work.entity_kind === 'series' ? { action: 'browse_series' } : {},
          });
        }
      });
    });
  });
}

function _closePersonPopover() {
  document.querySelector('.media-person-popover')?.remove();
}

/* ── Related strips ────────────────────────────────────────────── */

async function _fetchRelated(d, by) {
  try {
    const resp = await fetch(
      `/api/media/related/${encodeURIComponent(d.fileId)}?by=${by}&limit=18`,
      { credentials: 'same-origin' },
    );
    if (resp.ok) return await resp.json();
  } catch (err) {
    console.warn('[media-detail] related fetch failed:', err);
  }
  return null;
}

function _relatedTitle(by, data) {
  if (by === 'author') return `More by ${data.display_name || 'this author'}`;
  if (by === 'narrator') return `Also narrated by ${data.display_name || 'this narrator'}`;
  if (by === 'series') return `More in ${data.display_name || 'this series'}`;
  return 'More like this';
}

function _wireRelatedPivots(container, d, { onOpenItem, token }) {
  const section = container.querySelector('[data-related-section]');
  const titleEl = container.querySelector('[data-related-title]');
  const strip = container.querySelector('[data-related-strip]');
  if (!section || !strip) return;

  const load = async (by) => {
    strip.innerHTML = '<div class="media-detail-skeleton"><span></span><span></span></div>';
    const data = await _fetchRelated(d, by);
    if (token !== _token || !container.isConnected) return;
    if (!data?.items?.length) {
      strip.innerHTML = `<p class="media-person-empty">Nothing related found.</p>`;
      section.hidden = false;
      titleEl.textContent = _relatedTitle(by, data || {});
      return;
    }
    titleEl.textContent = _relatedTitle(by, data);
    strip.innerHTML = data.items.map((item) => `
      <button class="media-detail-related-card" type="button" data-related-id="${escapeHtml(item.id)}" title="${escapeHtml(item.title)}">
        <div class="media-detail-related-cover">
          ${item.cover_url
            ? `<img src="${escapeHtml(item.cover_url)}" alt="" loading="lazy" decoding="async" onerror="this.remove()">`
            : `<span>${escapeHtml((item.title || '?').slice(0, 1))}</span>`}
        </div>
        <div class="media-detail-related-title">${escapeHtml(item.title)}</div>
      </button>`).join('');
    section.hidden = false;
    if (onOpenItem) {
      strip.querySelectorAll('[data-related-id]').forEach((btn) => {
        btn.addEventListener('click', () => {
          const item = data.items.find((x) => x.id === btn.dataset.relatedId);
          if (!item) return;
          onOpenItem({
            file_id: item.id,
            title: item.title,
            kind: d.kind,
            entity_kind: '',
            cover_url: item.cover_url || '',
            progress_pct: item.progress_pct || 0,
            play: {},
          });
        });
      });
    }
  };

  // Axis pick: audio defaults to author with optional narrator pivot;
  // video auto-loads the shared-genre axis. Comics never reach the
  // detail view (the series drill-in owns that flow).
  const chips = section.querySelectorAll('[data-related-axis]');
  chips.forEach((chip) => {
    chip.addEventListener('click', () => {
      chips.forEach((c) => c.classList.toggle('is-active', c === chip));
      load(chip.dataset.relatedAxis);
    });
  });

  if (d.kind === 'audio' && d.hasSeriesAxis) load('series');
  else if (d.kind === 'audio' && d.hasAuthorAxis) load('author');
  else if (d.kind === 'audio' && d.hasNarratorAxis) load('narrator');
  else if (d.kind === 'audio' && d.genres.length) load('genre');
  else if (d.kind === 'video' && d.genres.length) load('genre');
}
