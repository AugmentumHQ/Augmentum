/**
 * library/types.js — shared format groupings + surface classifier.
 *
 * Multiple artifact ``format`` values collapse onto the same human
 * label. Both the sidebar (to roll up type counts) and the main pane
 * (to query /api/library/items with the right format list) need the
 * same view, so it lives here.
 *
 * The ``classifyItem`` helper mirrors the legacy ``_classifyArtifacts``
 * in ``ui/scripts/library.js`` so the library detail-pane open
 * dispatcher (and menu actions) can route items to the right surface
 * (emulator-stage / workspace / game-surface / studio). Keep
 * ``_KIND_TO_TYPE`` in sync with metadata.kind values stamped by
 * augmentum/titles/sources_*.py + augmentum/proxy/games_routes.py.
 */

// Groups map an item's *effective* format to a sidebar label. The effective
// format is what /api/library/items projects: an artifact's ``format`` OR,
// when that's empty (emulator ROMs), its ``metadata.kind``; and a
// publication's ``kind`` (game/app/doc/other). So the publication kinds
// (app/doc) and the ROM kind (emulator_rom) are enumerated here alongside
// the file extensions — without them, coder saves rolled into "Other" and
// ROMs (empty format) were counted nowhere and unreachable by type.
export const TYPE_GROUPS = {
  Apps:       ['html', 'htm', 'zip', 'app'],
  Games:      ['game', 'emulator_rom', 'streamed_game', 'js13k_game', 'web_app'],
  Documents:  ['pdf', 'docx', 'doc'],
  Books:      ['epub'],
  Notes:      ['md', 'txt', 'rst', 'log', 'json'],
  Slides:     ['pptx'],
  Sheets:     ['xlsx', 'csv'],
  Images:     ['png', 'jpg', 'jpeg', 'webp', 'svg', 'gif'],
};

// Reverse lookup: format -> label. Frozen so accidental mutation
// fails loudly in dev.
export const FORMAT_TO_LABEL = Object.freeze(
  Object.entries(TYPE_GROUPS).reduce((acc, [label, fmts]) => {
    for (const f of fmts) acc[f] = label;
    return acc;
  }, {}),
);

export const LABEL_FALLBACK = 'Other';

// metadata.kind → surface type. AXF titles + games routes stamp these.
// Anything not here falls through to TYPE_MAP[format].
const _KIND_TO_TYPE = Object.freeze({
  emulator_rom:   'game',
  game:           'game',
  js13k_game:     'game',
  streamed_game:  'game',
});

// format → surface type. Covers the extensions library actually
// surfaces in its sidebar virtual collections. 'app' / 'doc' / 'other'
// are publication-shape formats (kind values from
// library_publications.kind) — included so coder save publications
// classify consistently with their artifact cousins.
const TYPE_MAP = Object.freeze({
  zip: 'app', html: 'app', htm: 'app',
  pdf: 'doc', docx: 'doc', epub: 'doc',
  md: 'doc', txt: 'doc', rst: 'doc', log: 'doc', json: 'doc',
  pptx: 'pptx',
  xlsx: 'xlsx', csv: 'xlsx',
  png: 'chart', jpg: 'chart', jpeg: 'chart',
  svg: 'chart', gif: 'chart', webp: 'chart',
  game: 'game',
  app: 'app',
  doc: 'doc',
  other: 'doc',
});

// Catalog-friendly label for an item's format. The effective format the
// library projects can be a bare kind token (emulator_rom / game / app /
// doc / other) — surface something a person reads ("Game Boy", "Retro
// game", "App") rather than a raw "EMULATOR_ROM". Real file types fall
// through to the uppercased extension (PDF, PPTX, …).
const _FRIENDLY_FORMAT = {
  emulator_rom:  'Retro game',
  streamed_game: 'Cloud game',
  js13k_game:    'Web game',
  web_app:       'Web app',
  game:          'Game',
  app:           'App',
  zip:           'App',
  html:          'Web app',
  htm:           'Web app',
  doc:           'Document',
  other:         'File',
};

// Common libretro system ids → console names, so a ROM reads like a
// catalog entry ("Game Boy") when the source didn't stamp a system_label.
const _SYSTEM_LABELS = {
  gb: 'Game Boy', gbc: 'Game Boy Color', gba: 'Game Boy Advance',
  nes: 'NES', snes: 'SNES', n64: 'Nintendo 64', nds: 'Nintendo DS',
  gen: 'Genesis', md: 'Genesis', sms: 'Master System', gg: 'Game Gear',
  psx: 'PlayStation', ps1: 'PlayStation', pce: 'TurboGrafx-16',
  a2600: 'Atari 2600', arcade: 'Arcade',
};

export function friendlyFormat(item) {
  if (!item) return '';
  const meta = item.metadata || {};
  const kind = String(meta.kind || '').toLowerCase();
  // Retro ROMs read best as their console name.
  if (kind === 'emulator_rom') {
    const sid = String(meta.system_id || '').toLowerCase();
    return meta.system_label || _SYSTEM_LABELS[sid]
      || (sid ? sid.toUpperCase() : 'Retro game');
  }
  const fmt = String(item.format || '').toLowerCase();
  if (_FRIENDLY_FORMAT[fmt]) return _FRIENDLY_FORMAT[fmt];
  return fmt ? fmt.toUpperCase() : '';
}

export function formatsForLabel(label) {
  return TYPE_GROUPS[label] || [];
}

export function labelForFormat(fmt) {
  return FORMAT_TO_LABEL[fmt] || LABEL_FALLBACK;
}

/**
 * Stamp ``item._type`` ∈ {'app', 'game', 'doc', 'pptx', 'xlsx',
 * 'chart'} based on metadata.kind first (catches ROMs / streamed
 * games / agentic builds), then format. Falls back to 'doc' so the
 * detail-pane dispatcher can pick studio as a safe default.
 * Mutates in place AND returns the item so it composes with
 * .map() / single-item paths.
 */
export function classifyItem(item) {
  if (!item) return item;
  const kind = item.metadata?.kind || '';
  item._type = _KIND_TO_TYPE[kind] || TYPE_MAP[item.format] || 'doc';
  // Publication ids carry a distinct prefix; surface dispatchers
  // ignore _type for these and route to /api/library/play/{id}.
  if (typeof item.id === 'string' && item.id.startsWith('pub_')) {
    item._isPublication = true;
  }
  return item;
}

/**
 * Apply ``classifyItem`` to every item in an array, in-place.
 */
export function classifyItems(items) {
  if (Array.isArray(items)) {
    for (const it of items) classifyItem(it);
  }
  return items;
}
