/**
 * Pure helpers + filetype predicates + lightweight state-reading utilities
 * (peek cache, formatters). No DOM mutation here — that lives in render.js.
 */

import { state, KIND_ICONS, TEXT_EXTS, MEDIA_SOURCES } from './state.js';
import { escapeHtml } from '../app.js';

// --- Extension ---------------------------------------------------------

export function getExt(name) {
  if (!name) return '';
  const dot = name.lastIndexOf('.');
  if (dot < 1 || dot === name.length - 1) return '';
  return name.slice(dot + 1).toLowerCase();
}

// --- Filetype predicates ----------------------------------------------
// Each returns true for files that should be treated as that kind, using
// mime_type first and extension as fallback. Kept simple and explicit so
// the activation dispatcher in actions.js can cascade through them.

export function isImage(f) {
  const mime = (f.mime_type || '').toLowerCase();
  if (mime.startsWith('image/')) return true;
  const ext = getExt(f.name);
  if (['png','jpg','jpeg','gif','webp','svg','bmp','ico','heic','heif','tif','tiff','avif'].includes(ext)) return true;
  return f.source === 'images' || f.source === 'chat_images';
}

// Image formats no major browser renders natively — server-side transcode
// route (/api/files/render) hands back JPEG bytes for these.
export function imageNeedsServerRender(f) {
  const ext = getExt(f.name);
  if (['heic','heif','tif','tiff'].includes(ext)) return true;
  const mime = (f.mime_type || '').toLowerCase();
  return mime === 'image/heic' || mime === 'image/heif' || mime === 'image/tiff';
}

// External pointer rather than a real file on disk. The activate path
// short-circuits download/preview and opens the saved URL instead.
export function isBookmark(f) {
  return f.source === 'bookmarks';
}

// Pull the saved URL off a bookmark row. Lives in source_metadata
// (FileEntry's catch-all for source-specific extras).
export function bookmarkUrl(f) {
  return f?.source_metadata?.url || '';
}

// Rows that play through /api/media/stream/{id} (chapter-aware, Range-
// forwarding, auth-handling proxy) instead of /api/files/download. Covers
// user media servers (Audiobookshelf / Emby / Jellyfin / ...) and built-in
// browse-only libraries (LibriVox). MEDIA_SOURCES is the shared slug set.
export function isMediaServerFile(f) {
  return !!f && MEDIA_SOURCES.has(f.source);
}

// Pinned built-in LibriVox row. Matches the backend gate on
// `DELETE /api/media/pin/{id}` — generic file-delete won't unpin these
// cleanly, so the UI needs to route them through the unpin endpoint.
export function isBuiltinLibrivox(f) {
  if (!f || f.source !== 'librivox') return false;
  return f.source_metadata?.server_id === 'builtin-librivox';
}

// Normalised 0-1 playback progress for the card progress bar. Returns
// 0 when the server hasn't reported any position yet.
export function mediaProgress(f) {
  const raw = Number(f?.source_metadata?.progress_pct);
  if (!Number.isFinite(raw) || raw <= 0) return 0;
  // source_metadata.progress_pct is 0-100; clamp and convert to 0-1.
  return Math.min(1, Math.max(0, raw / 100));
}

// True when the backing server reports a cover — lets the UI choose a
// cover-backed thumb over the generic kind icon without an extra round
// trip to 404. Checks ``source_metadata`` directly: any row (audiobook,
// comic, future movie) whose provider populated a cover_url can ride
// the ``/api/media/cover/{file_id}`` proxy, regardless of whether the
// source is in MEDIA_SOURCES (which is specifically about the streaming
// path, not cover availability).
export function hasMediaCover(f) {
  if (!f) return false;
  const meta = f.source_metadata || {};
  return !!(meta.has_cover || meta.cover_url);
}

export function isVideo(f) {
  if ((f.mime_type || '').toLowerCase().startsWith('video/')) return true;
  return ['mp4','webm','mkv','mov','avi','m4v','ogv','wmv','flv'].includes(getExt(f.name));
}

// Containers/codecs the average browser can't play back. The detail panel
// surfaces a friendly "download to play" CTA instead of a silent broken
// <video>. mkv is technically a wrapper that some browsers handle; we
// flag it because the content is usually H.265 / Vorbis / etc. that
// don't load.
export function videoLikelyUnsupported(f) {
  const ext = getExt(f.name);
  if (['mkv','avi','wmv','flv','m4v','ogv','mov'].includes(ext)) {
    // canPlayType returns "" / "maybe" / "probably". Anything but ""
    // means the browser is willing to attempt it.
    try {
      const v = document.createElement('video');
      const guesses = {
        mkv: 'video/x-matroska', avi: 'video/x-msvideo',
        wmv: 'video/x-ms-wmv',   flv: 'video/x-flv',
        m4v: 'video/mp4',        ogv: 'video/ogg',
        mov: 'video/quicktime',
      };
      return v.canPlayType(guesses[ext] || '') === '';
    } catch { return true; }
  }
  return false;
}

export function isAudio(f) {
  const mime = (f.mime_type || '').toLowerCase();
  if (mime.startsWith('audio/')) return true;
  if (f.source === 'voices') return true;
  return ['mp3','wav','ogg','flac','webm','m4a','aac','opus'].includes(getExt(f.name));
}

// Comic / manga archive — CBZ, CBR, CBT, CB7, or anything Komga/Suwayomi
// classifies as ``application/vnd.comicbook+zip|-rar``. The server-side
// derive_kind() already stamps ``kind='comic'`` on these rows, so the
// fast path is just a kind check; mime + extension remain as fallbacks
// for any row that slipped through pre-101 migration.
export function isComic(f) {
  if (!f) return false;
  if (f.kind === 'comic') return true;
  const mime = (f.mime_type || '').toLowerCase();
  if (mime === 'application/vnd.comicbook+zip') return true;
  if (mime === 'application/vnd.comicbook-rar') return true;
  if (mime === 'application/x-cbz' || mime === 'application/x-cbr') return true;
  return ['cbz','cbr','cbt','cb7'].includes(getExt(f.name));
}

export function isPdf(f) {
  return (f.mime_type || '').toLowerCase() === 'application/pdf' || getExt(f.name) === 'pdf';
}

export function isMarkdown(f) {
  if ((f.mime_type || '').toLowerCase() === 'text/markdown') return true;
  return ['md','markdown','mdown','mkd'].includes(getExt(f.name));
}

export function isHtml(f) {
  if ((f.mime_type || '').toLowerCase() === 'text/html') return true;
  return ['html','htm'].includes(getExt(f.name));
}

export function isEpub(f) {
  return (f.mime_type || '').toLowerCase() === 'application/epub+zip' || getExt(f.name) === 'epub';
}

export function isOffice(f) {
  return ['docx','pptx','xlsx','csv'].includes(getExt(f.name));
}

// Archive file — server /render returns a listing of its contents.
// 7z deliberately excluded because stdlib can't read it.
export function isArchive(f) {
  return ['zip','tar','gz','tgz','bz2'].includes(getExt(f.name));
}

export function isText(f) {
  const mime = (f.mime_type || '').toLowerCase();
  if (mime.startsWith('text/')) return true;
  if (['application/json','application/xml','application/yaml'].includes(mime)) return true;
  return TEXT_EXTS.has(getExt(f.name));
}

// File types where `/api/files/text/{id}` can return usable prose for TTS.
// Textual rows route through directly, document rows go through a parser.
// Media (image/audio/video), archives, and bookmarks are excluded — the
// endpoint would 415 on them anyway.
const _READ_ALOUD_DOC_EXTS = new Set(['pdf','docx','pptx','xlsx','epub']);
export function supportsReadAloud(f) {
  if (!f) return false;
  if (isBookmark(f)) return false;
  if (isImage(f) || isVideo(f) || isAudio(f) || isArchive(f)) return false;
  if (_READ_ALOUD_DOC_EXTS.has(getExt(f.name))) return true;
  return isText(f) || isMarkdown(f) || isHtml(f);
}

// Zip + html artifacts that the app builder produced — same project shape
// the library uses, so we can hand them to workspace.openWorkspace().
// Source must be `artifacts` because that's where source_json lives.
export function isAppProject(f) {
  if (f.source !== 'artifacts') return false;
  const ext = getExt(f.name);
  return ext === 'zip' || ext === 'html' || ext === 'htm';
}

// --- Icon + tint ------------------------------------------------------

export function iconForFile(f) {
  return KIND_ICONS[f.kind] || KIND_ICONS.other;
}

export function tintKey(f) {
  if (isMediaServerFile(f)) {
    const entityKind = String(f?.source_metadata?.entity_kind || '').toLowerCase();
    if (entityKind === 'movie') return 'movie';
    if (entityKind === 'music_video') return 'music_video';
    if (entityKind === 'series' || entityKind === 'season' || entityKind === 'episode') {
      return 'show';
    }
  }
  return f.kind || f.source || '';
}

// --- DOM utility ------------------------------------------------------

export function isTextInput(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || el.isContentEditable;
}

// --- Formatters -------------------------------------------------------

export function humanSize(bytes) {
  if (bytes == null || bytes < 0) return '';
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const val = bytes / Math.pow(1024, i);
  return `${i === 0 ? val : val.toFixed(1)} ${units[i]}`;
}

export function formatCount(n) {
  if (n == null) return '';
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 10_000) return `${Math.round(n / 1_000)}K`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export function formatDate(iso) {
  if (!iso) return '';
  try {
    return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  } catch { return iso; }
}

export function recencyClass(iso) {
  if (!iso) return '';
  const t = Date.parse(iso);
  if (isNaN(t)) return '';
  const age = Date.now() - t;
  if (age < 3600_000)        return 'recency-fresh';
  if (age < 6 * 3600_000)    return 'recency-recent';
  if (age < 24 * 3600_000)   return 'recency-today';
  return '';
}

export function dateBucket(iso) {
  if (!iso) return 'Earlier';
  const t = Date.parse(iso);
  if (isNaN(t)) return 'Earlier';
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (t >= today)                  return 'Today';
  if (t >= today - 86_400_000)     return 'Yesterday';
  if (t >= today - 7 * 86_400_000) return 'This week';
  if (t >= today - 30 * 86_400_000) return 'This month';
  return 'Earlier';
}

export function renderTagPills(tags) {
  if (!tags || !tags.length) return '';
  const visible = tags.slice(0, 2);
  const overflow = tags.length - visible.length;
  const pills = visible.map(t =>
    `<span class="files-card-tag" title="${escapeHtml(t)}">${escapeHtml(t)}</span>`
  ).join('');
  const more = overflow > 0
    ? `<span class="files-card-tag files-card-tag-more" title="${escapeHtml(tags.slice(2).join(', '))}">+${overflow}</span>`
    : '';
  return `<div class="files-card-tags">${pills}${more}</div>`;
}

// --- Content peek cache ----------------------------------------------

export async function fetchPeek(id) {
  const cache = state.peekCache;
  if (cache.has(id)) return cache.get(id);
  if (state.peekInflight.has(id)) return null;
  state.peekInflight.add(id);
  try {
    const resp = await fetch(`/api/files/preview/${encodeURIComponent(id)}`);
    if (!resp.ok) { cache.set(id, ''); return ''; }
    const data = await resp.json();
    const snippet = (data.snippet || '').trim();
    cache.set(id, snippet);
    return snippet;
  } catch {
    cache.set(id, '');
    return '';
  } finally {
    state.peekInflight.delete(id);
  }
}

export function hydratePeek(card) {
  const id = card.dataset.id;
  const target = card.querySelector('.files-card-peek');
  if (!id || !target) return;
  if (target.dataset.hydrated) return;
  target.dataset.hydrated = '1';
  fetchPeek(id).then(snippet => {
    if (!snippet) {
      target.classList.add('empty');
      return;
    }
    const lines = snippet.split('\n').slice(0, 7)
      .map(l => l.length > 80 ? l.slice(0, 80) + '…' : l);
    target.textContent = lines.join('\n');
    target.classList.add('hydrated');
  });
}

export function setupPeekObserver() {
  if (state.peekObserver || !state.el.grid) return;
  state.peekObserver = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        hydratePeek(e.target);
        state.peekObserver.unobserve(e.target);
      }
    }
  }, { root: state.el.grid, rootMargin: '100px' });
}

export function observeVisiblePeeks() {
  if (!state.peekObserver || state.currentView !== 'gallery') return;
  state.el.grid?.querySelectorAll('.files-card[data-peekable="1"]').forEach(c => {
    state.peekObserver.observe(c);
  });
}
