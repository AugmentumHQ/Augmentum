/*
 * XR Media Library
 *
 * Data-only helpers for turning Augmentum's mixed media sources into headset
 * navigation rails. This file has no DOM, WebXR, or player side effects.
 */

export const XR_MEDIA_SECTIONS = Object.freeze([
  {
    id: 'continue',
    label: 'Continue',
    kinds: ['movie', 'show', 'episode', 'comic', 'audiobook', 'image', 'game'],
    surface: 'resume-strip',
    actions: ['resume', 'play', 'discuss'],
  },
  {
    id: 'shows_movies',
    label: 'Shows + Movies',
    kinds: ['movie', 'show', 'episode', 'video'],
    surface: 'theater',
    actions: ['play', 'queue', 'captions', 'summarize'],
  },
  {
    id: 'comics',
    label: 'Comics',
    kinds: ['comic', 'manga', 'book_page'],
    surface: 'reader',
    actions: ['open', 'next_page', 'zoom', 'discuss'],
  },
  {
    id: 'audiobooks',
    label: 'Audiobooks',
    kinds: ['audiobook', 'podcast', 'audio'],
    surface: 'listening-room',
    actions: ['play', 'chapter', 'sleep_timer', 'discuss'],
  },
  {
    id: 'images',
    label: 'Images',
    kinds: ['image', 'photo', 'gallery'],
    surface: 'gallery-wall',
    actions: ['open', 'slideshow', 'inspect', 'save'],
  },
  {
    id: 'local_files',
    label: 'Local Files',
    kinds: ['local_file', 'download', 'folder'],
    surface: 'file-shelf',
    actions: ['open', 'attach', 'sort', 'summarize'],
  },
  {
    id: 'games',
    label: 'Games',
    kinds: ['game', 'stream', 'rom'],
    surface: 'game-stage',
    actions: ['launch', 'resume', 'controller_mode', 'stop_stream'],
  },
]);

const KIND_ALIASES = Object.freeze({
  audiobook_chapter: 'audiobook',
  book: 'comic',
  cbz: 'comic',
  cbr: 'comic',
  folder: 'local_file',
  jpeg: 'image',
  jpg: 'image',
  manga_chapter: 'comic',
  mkv: 'movie',
  mp3: 'audio',
  mp4: 'movie',
  page: 'book_page',
  photo: 'image',
  png: 'image',
  series: 'show',
  tv: 'show',
});

function _clean(value) {
  return String(value || '').trim();
}

export function normalizeXrMediaKind(kind = '') {
  const key = _clean(kind).toLowerCase().replace(/[\s-]+/g, '_');
  return KIND_ALIASES[key] || key || 'unknown';
}

export function describeXrMediaSection(sectionId = '') {
  const normalized = normalizeXrMediaKind(sectionId);
  return XR_MEDIA_SECTIONS.find((section) => (
    section.id === normalized || section.kinds.includes(normalized)
  )) || null;
}

export function sectionForXrMediaKind(kind = '') {
  const normalized = normalizeXrMediaKind(kind);
  return XR_MEDIA_SECTIONS.find((section) => section.kinds.includes(normalized)) || null;
}

export function normalizeXrMediaItem(item = {}) {
  const kind = normalizeXrMediaKind(
    item.kind || item.mediaType || item.type || item.format || item.sourceType,
  );
  const section = sectionForXrMediaKind(kind);
  return {
    id: _clean(item.id || item.key || item.path || item.url || item.title),
    title: _clean(item.title || item.name || item.filename || 'Untitled'),
    subtitle: _clean(item.subtitle || item.series || item.author || item.album || ''),
    kind,
    sectionId: section?.id || 'local_files',
    source: _clean(item.source || item.provider || item.library || ''),
    url: _clean(item.url || item.href || item.src || ''),
    path: _clean(item.path || item.filePath || ''),
    thumbnail: _clean(item.thumbnail || item.poster || item.cover || item.image || ''),
    progress: Number.isFinite(Number(item.progress)) ? Number(item.progress) : 0,
    durationMs: Number.isFinite(Number(item.durationMs)) ? Number(item.durationMs) : 0,
    updatedAt: _clean(item.updatedAt || item.modifiedAt || item.createdAt || ''),
  };
}

export function buildXrMediaNavigationState({
  items = [],
  activeSection = '',
  nowPlaying = null,
  maxItemsPerSection = 12,
} = {}) {
  const normalizedItems = Array.isArray(items)
    ? items.map(normalizeXrMediaItem).filter((item) => item.id || item.title)
    : [];
  const active = describeXrMediaSection(activeSection)?.id || 'continue';
  const sections = XR_MEDIA_SECTIONS.map((section) => {
    const sectionItems = normalizedItems.filter((item) => {
      if (section.id === 'continue') return item.progress > 0 && item.progress < 0.98;
      return item.sectionId === section.id;
    });
    return {
      id: section.id,
      label: section.label,
      surface: section.surface,
      actions: section.actions,
      count: sectionItems.length,
      items: sectionItems.slice(0, Math.max(1, Number(maxItemsPerSection) || 12)),
    };
  });
  return {
    activeSection: active,
    sections,
    nowPlaying: nowPlaying ? normalizeXrMediaItem(nowPlaying) : null,
    displayModes: ['theater', 'reader', 'gallery-wall', 'listening-room', 'game-stage'],
  };
}
