/**
 * Files panel — shared state + constants.
 *
 * Single mutable `state` object imported by every sibling module; the object
 * identity never changes so live-binding works even across circular imports.
 */

// --- Preference persistence -----------------------------------------
// Every navigation pref — chip, scope, sort, view, sub-filters — lives
// in localStorage so the next Files open lands the user exactly where
// they left off, even across browser close or a server restart. The
// workspace is big enough that "return me to Audiobooks" / "return me
// to Comics series X" is the dominant UX expectation, and the earlier
// session-only behavior lost context any time the user reloaded the
// page (exactly when they needed continuity most).
//
// Per-user scoping isn't needed — this is a per-device preference and
// localStorage is already per-origin.

const PREF_KEYS = {
  view:            'augmentum.files.view',
  sort:            'augmentum.files.sort',
  source:          'augmentum.files.source',
  status:          'augmentum.files.mediaStatus',
  mediaView:       'augmentum.files.mediaView',
  scope:           'augmentum.files.scope',
  comicStatus:     'augmentum.files.comicStatus',
  comicCompletion: 'augmentum.files.comicCompletion',
  comicGenre:      'augmentum.files.comicGenre',
  videoStatus:     'augmentum.files.videoStatus',
  videoGenre:      'augmentum.files.videoGenre',
  videoYearFrom:   'augmentum.files.videoYearFrom',
  videoYearTo:     'augmentum.files.videoYearTo',
  // Continue-watching/listening rail open/closed state. Boolean — sticks
  // across reloads so a user who collapsed the rail to dig through the
  // full catalog stays in that mode until they explicitly re-open it.
  continueRailCollapsed: 'augmentum.files.continueRailCollapsed',
};
const VALID_VIEWS      = new Set(['grid', 'list', 'gallery']);
const VALID_SORTS      = new Set([
  'newest', 'oldest', 'name', 'size',
  'author', 'progress',                  // media-only
  'continue', 'updated', 'unread',       // comics-only
]);
const VALID_STATUS     = new Set(['all', 'in_progress', 'finished', 'not_started']);
const VALID_MEDIA_VIEW = new Set(['library', 'catalog']);
// Comics read-state rollup — mirrors the audiobook playback status but
// at the series level. Values match the backend /comics/series ?status=
// contract (server-side falls through on unknown values).
const VALID_COMIC_STATUS     = new Set(['all', 'reading', 'caught-up', 'unread']);
const VALID_COMIC_COMPLETION = new Set(['', 'ongoing', 'completed', 'hiatus']);
// Scope split: 'local' = files on this system (uploads, authored notes,
// AI-generated artifacts), 'cloud' = anything from a connected remote
// server (audiobooks, comics, eventually movies/TV). Default is 'local'
// so first-opens don't get buried under 20k manga chapters.
const VALID_SCOPES     = new Set(['local', 'cloud']);
const VALID_SOURCES    = new Set([
  'all',
  'images', 'documents', 'audio', 'video', 'code', 'archives',
  'audiobooks', 'podcasts', 'comics', 'shows', 'movies', 'music_videos',
  'live_tv',
  'favorites', 'trash',
]);

function _readPref(key, valid, fallback, store = localStorage) {
  try {
    const v = store.getItem(key);
    if (v && valid.has(v)) return v;
  } catch { /* private mode / disabled storage */ }
  return fallback;
}

export function savePref(name, value) {
  try {
    const key = PREF_KEYS[name];
    if (!key) return;
    localStorage.setItem(key, value);
  } catch { /* ignore */ }
}

// Boolean pref helper — readPref/savePref above are enum-only via the
// VALID_* sets. Booleans are stored as '1' / '0' so we can round-trip
// false (the empty-string fallback would treat false as missing).
export function readBoolPref(name, fallback = false) {
  try {
    const key = PREF_KEYS[name];
    if (!key) return fallback;
    const v = localStorage.getItem(key);
    if (v === '1') return true;
    if (v === '0') return false;
  } catch { /* private mode / disabled storage */ }
  return fallback;
}
export function saveBoolPref(name, value) {
  savePref(name, value ? '1' : '0');
}

export const state = {
  el: {},
  isOpen: false,
  currentSource: _readPref(PREF_KEYS.source, VALID_SOURCES, 'all'),
  currentView:   _readPref(PREF_KEYS.view,   VALID_VIEWS,      'grid'),
  currentSort:   _readPref(PREF_KEYS.sort,   VALID_SORTS,      'newest'),
  // Local/Cloud scope. Default 'local' so new users aren't flooded with
  // cloud catalog content on first open. See the Files panel toggle.
  currentScope:  _readPref(PREF_KEYS.scope,  VALID_SCOPES,     'local'),
  // Status filter applies only when a media-server source chip is
  // active. Restored eagerly so switching to Audiobookshelf lands on
  // the status the user last left it at (e.g. "In progress").
  currentMediaStatus: _readPref(PREF_KEYS.status, VALID_STATUS, 'all'),
  // Audiobooks view mode — library (local pins) vs catalog (live
  // LibriVox browse).
  currentMediaView: _readPref(PREF_KEYS.mediaView, VALID_MEDIA_VIEW, 'library'),
  // Comics chip filters. series-read-status mirrors the audiobook
  // media-status filter; completion + genre are comics-only narrowing.
  currentComicStatus:     _readPref(PREF_KEYS.comicStatus,     VALID_COMIC_STATUS,     'all'),
  currentComicCompletion: _readPref(PREF_KEYS.comicCompletion, VALID_COMIC_COMPLETION, ''),
  // Genre is free-form (populated at runtime from /comics/genres), so
  // we don't validate against a fixed set — just sanitise to a string.
  currentComicGenre:      (() => {
    try { return (localStorage.getItem(PREF_KEYS.comicGenre) || '').trim(); }
    catch { return ''; }
  })(),
  // Video chip filters (Shows / Movies / Music Videos). Watch state
  // shares the audiobook playback-status vocabulary; genre + year are
  // video-specific. Year bounds are 0 when unset (no bound).
  currentVideoStatus: _readPref(PREF_KEYS.videoStatus, VALID_STATUS, 'all'),
  currentVideoGenre: (() => {
    try { return (localStorage.getItem(PREF_KEYS.videoGenre) || '').trim(); }
    catch { return ''; }
  })(),
  currentVideoYearFrom: (() => {
    try { return parseInt(localStorage.getItem(PREF_KEYS.videoYearFrom) || '0', 10) || 0; }
    catch { return 0; }
  })(),
  currentVideoYearTo: (() => {
    try { return parseInt(localStorage.getItem(PREF_KEYS.videoYearTo) || '0', 10) || 0; }
    catch { return 0; }
  })(),
  files: [],
  selection: new Set(),
  lastClickedIndex: -1,
  focusedIndex: -1,
  // Explicit multi-select mode for touch. Entered via the toolbar Select
  // button or a long-press on a card; makes every tap toggle selection
  // and reveals a checkbox affordance on each card. Desktop users don't
  // need it — Ctrl/Shift-click still work independently of this flag.
  selectMode: false,
  searchDebounce: null,
  // When Files was opened by another overlay (Discovery card click, etc.)
  // we remember which one so a "back" button can return the user there.
  // Empty string = entered Files directly (no back affordance shown).
  // Cleared on explicit close (X button / Esc) so the next direct open
  // doesn't surface a stale back target.
  cameFromOverlay: '',
  renamingId: null,
  contextMenu: null,
  galleryOverlay: null,
  galleryIndex: -1,
  detailToken: 0,
  detailOverrideFile: null,
  // Breadcrumb stack for media-server video drill-down: each entry is
  // {id, name, entity_kind}. Empty when the detail panel shows the
  // grid-selected file. Pushed on Series→Season→Episode navigation.
  detailNavStack: [],
  offset: 0,
  hasMore: false,
  loading: false,
  scrollObserver: null,
  lastRenderedBucket: '',
  peekCache: new Map(),
  peekInflight: new Set(),
  peekObserver: null,
};

export const PAGE_SIZE = 60;

export const SORT_LABELS = {
  newest:   'Newest',
  oldest:   'Oldest',
  name:     'Name',
  size:     'Size',
  author:   'Author',
  progress: 'Recently played',
  continue: 'Continue reading',
  updated:  'Recently updated',
  unread:   'Most unread',
};

// Sort options that only make sense for media-server rows. Author and
// recently-played require values we only populate for that source.
export const MEDIA_SORTS = new Set(['author', 'progress']);

// Sort options specific to the Comics chip. The series-rollup endpoint
// knows how to interpret these against the chapter-aggregate view; the
// flat file search endpoint would fall back to `name` if these leaked in.
export const COMICS_SORTS = new Set(['continue', 'updated', 'unread']);

// Generic defaults that don't reflect a comics-aware choice. When the
// user enters the Comics chip with one of these active, we promote them
// to `continue` (the comics-default) since "newest by sync activity"
// rarely matches what someone reaching for the Comics chip wants. An
// explicit pick like `name` / `unread` / `updated` is preserved.
export const GENERIC_SORTS_FOR_COMICS_PROMOTION = new Set(['newest', 'oldest']);

// Status filter label lookup for the segmented control in the topbar.
export const STATUS_LABELS = {
  all:          'All',
  in_progress:  'In progress',
  finished:     'Finished',
  not_started:  'Not started',
};

// Tab slug routing — most filters map to `kind`, and media-server slugs
// (audiobookshelf, librivox, ...) are grouped under the "audiobooks"
// virtual chip so a user sees one tab for all their audiobook content
// regardless of provider.
export const TAB_KIND = new Set(['images', 'documents', 'audio', 'video', 'code', 'archives']);
// MEDIA_SOURCES = concrete row-level slugs that stream through
// /api/media/stream. Used for per-row checks (isMediaServerFile). Backend
// mirrors this in augmentum/vfs/index.py:_MEDIA_SOURCES.
export const MEDIA_SOURCES = new Set(['audiobookshelf', 'librivox', 'emby', 'jellyfin']);
// MEDIA_CHIP = the virtual chip slug the backend expands to
// MEDIA_SOURCES. Backend mirrors in files_routes.py:_SOURCE_GROUPS.
export const MEDIA_CHIP = 'audiobooks';
export const PODCASTS_CHIP = 'podcasts';
export const AUDIO_LIBRARY_CHIPS = new Set([MEDIA_CHIP, PODCASTS_CHIP]);
// Parallel chip for comic sources (Suwayomi/Komga/Kavita). Backend
// _SOURCE_GROUPS['comics'] expands this to concrete provider slugs.
export const COMICS_CHIP = 'comics';
export const SHOWS_CHIP = 'shows';
export const MOVIES_CHIP = 'movies';
export const MUSIC_VIDEOS_CHIP = 'music_videos';
export const VIDEO_CLOUD_CHIPS = new Set([SHOWS_CHIP, MOVIES_CHIP, MUSIC_VIDEOS_CHIP]);
// Live TV channels from Emby/Jellyfin. Distinct from SHOWS/MOVIES
// because the layout is fundamentally different (horizontal rails of
// channels keyed by EPG/network, not a flat grid of titles) and the
// per-row click handler resolves through a live-playback PlaybackInfo
// + HLS pipeline rather than the static-file /api/media/stream proxy.
export const LIVE_TV_CHIP = 'live_tv';
export const TAB_SOURCE = new Set([
  MEDIA_CHIP, PODCASTS_CHIP, COMICS_CHIP, SHOWS_CHIP, MOVIES_CHIP, MUSIC_VIDEOS_CHIP,
  LIVE_TV_CHIP,
]);
export const KIND_ALIAS = { images: 'image', documents: 'document', archives: 'archive' };

// Chip vocabulary per scope. Local scope = user's own content; Cloud
// scope = categorized remote-catalog content. The Files panel picks the
// right list when the scope pill toggles. Jellyfin/Plex/Emby land under
// cloud as Movies/TV Shows/Music when their providers ship; add slugs
// here and groups in files_routes.py:_SOURCE_GROUPS together.
export const LOCAL_CHIPS = [
  'all', 'images', 'documents', 'audio', 'video', 'code', 'archives',
];
export const CLOUD_CHIPS = [
  'all', MEDIA_CHIP, PODCASTS_CHIP, COMICS_CHIP, SHOWS_CHIP, MOVIES_CHIP, MUSIC_VIDEOS_CHIP,
  LIVE_TV_CHIP,
];

// Chip labels — human-readable names for the pill bar. Missing entries
// fall back to Title-cased slug, which matches current behavior.
export const CHIP_LABELS = {
  all:         'All',
  images:      'Images',
  documents:   'Documents',
  audio:       'Audio',
  video:       'Video',
  code:        'Code',
  archives:    'Archives',
  audiobooks:  'Audiobooks',
  podcasts:    'Podcasts',
  comics:      'Comics',
  shows:       'Shows',
  movies:      'Movies',
  music_videos:'Music Videos',
  live_tv:     'Live TV',
};

// SVG icons — shared by card rendering and view toggle.
const SVG_BASE = 'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"';

export const KIND_ICONS = {
  image:    `<svg ${SVG_BASE}><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="M21 15l-5-5L5 21"/></svg>`,
  document: `<svg ${SVG_BASE}><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="M9 13h6M9 17h6"/></svg>`,
  audio:    `<svg ${SVG_BASE}><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/></svg>`,
  video:    `<svg ${SVG_BASE}><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/></svg>`,
  archive:  `<svg ${SVG_BASE}><path d="M21 8v13H3V8"/><rect x="1" y="3" width="22" height="5"/><line x1="10" y1="12" x2="14" y2="12"/></svg>`,
  code:     `<svg ${SVG_BASE}><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>`,
  // Open-book glyph — reads as "comic / manga / graphic novel" across
  // cultures without committing to either Western (panel) or Japanese
  // (right-to-left manga) imagery. Without this, comic rows fall back
  // to the generic folder glyph and look unreadable / wrong.
  comic:    `<svg ${SVG_BASE}><path d="M2 4h7a3 3 0 0 1 3 3v13a2 2 0 0 0-2-2H2z"/><path d="M22 4h-7a3 3 0 0 0-3 3v13a2 2 0 0 1 2-2h8z"/></svg>`,
  other:    `<svg ${SVG_BASE}><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>`,
};
export const GRID_ICON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>';
export const LIST_ICON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>';
export const GALLERY_ICON = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><polyline points="21 15 16 10 5 21"/></svg>';

// Shared extension sets — text-like, per-category.
export const TEXT_EXTS = new Set([
  'js','ts','jsx','tsx','py','rb','go','rs','c','cpp','h','hpp','java','kt',
  'swift','cs','php','sh','bash','zsh','yml','yaml','toml','ini','cfg','conf',
  'json','xml','html','htm','css','scss','less','sql','md','txt','log','csv',
  'env','dockerfile','makefile','rst','vue','svelte',
]);

// highlight.js language hint by extension.
export const HLJS_LANG_MAP = {
  js: 'javascript', jsx: 'javascript', ts: 'typescript', tsx: 'typescript',
  py: 'python', rb: 'ruby', rs: 'rust', go: 'go', java: 'java',
  kt: 'kotlin', swift: 'swift', cs: 'csharp', cpp: 'cpp', cc: 'cpp',
  c: 'c', h: 'c', hpp: 'cpp', php: 'php', sh: 'bash', bash: 'bash',
  zsh: 'bash', yml: 'yaml', yaml: 'yaml', toml: 'toml', ini: 'ini',
  json: 'json', xml: 'xml', html: 'xml', htm: 'xml', css: 'css',
  scss: 'scss', less: 'less', sql: 'sql', md: 'markdown',
  dockerfile: 'dockerfile', makefile: 'makefile', vue: 'xml', svelte: 'xml',
};
