/**
 * Orb Navigation — mobile bottom bar replacement.
 * Swipe arc (minimized) + press-to-expand bloom (expanded).
 * Surfaces: tap to switch, long-press to open alongside.
 */

import { SurfaceRegistry } from './surface-registry.js';
import { LayoutManager } from './layout-manager.js';
import { sessionStore } from './chat/sessions.js';
import { showToast, safeParseJSON } from './app.js';

// ── Orb Definitions ──────────────────────────────────────────────────
const ORB_ORDER = [
  { id: 'browse',      label: 'Browse',   desc: 'Web + reader',          type: 'feature', color: 'browse',      glyph: '\u2299' },  // ⊙ lens
  { id: 'narrative',   label: 'Story',    desc: 'Characters + roleplay', type: 'mode',    color: 'narrative',   glyph: '\u275D' },  // ❝ quote
  { id: 'coder',       label: 'Code',     desc: 'Terminal + editor',     type: 'mode',    color: 'coder',       glyph: '\u276F' },  // ❯ prompt
  { id: 'passthrough', label: 'Chat',     desc: 'Talk, search, tools',   type: 'mode',    color: 'passthrough', glyph: '\u22EF' },  // ⋯ speaking
  { id: 'analytical',  label: 'Analyze',  desc: 'Research + reason',     type: 'mode',    color: 'analytical',  glyph: '\u03BB' },  // λ reason
  { id: 'agentic',     label: 'Build',    desc: 'Create docs + apps',    type: 'mode',    color: 'agentic',     glyph: '\u2B21' },  // ⬡ block
  { id: 'notes',       label: 'Notes',    desc: 'Markdown editor',       type: 'feature', color: 'notes',       glyph: '\u00B6' },  // ¶ pilcrow
  { id: 'foryou',      label: 'For You',  desc: 'Discovery feed',        type: 'feature', color: 'foryou',      glyph: '\u273A' },  // ✺ discover
];

const DEFAULT_INDEX = 3; // Chat

// ── Tree Topology (bloom "tree" view) ────────────────────────────────
// Chat is the trunk; tier-1 modes branch up; tier-2 features are leaves.
//   passthrough
//     ├── analytical ── browse, foryou
//     ├── narrative  ── notes
//     └── agentic    ── coder
const TREE_TOPOLOGY = {
  root: 'passthrough',
  branches: [
    { id: 'analytical', leaves: ['browse', 'foryou'] },
    { id: 'narrative',  leaves: ['notes'] },
    { id: 'agentic',    leaves: ['coder'] },
  ],
};

// 'tree' (sprouting circuit, canonical) or 'ring' (original orbital ring).
// Persists across sessions via localStorage — existing users keep their pick.
let _bloomView = 'tree';

let _activeIndex = DEFAULT_INDEX;
let _modeIndex = DEFAULT_INDEX; // Tracks the real mode (features are temporary overlays)
let _featureOpen = null;        // Currently open feature id, or null
let _bloomOpen = false;

// Callbacks — set by init()
let _setMode = null;
// Optional "app boot complete" promise provider. Set by initOrbNav from
// app.js so _openAlongside can wait for the saved-workspace restore to
// finish before creating a singleton surface (see comment in
// _openAlongside for the race it guards against).
let _waitForBoot = null;
let _openBrowse = null;
let _openNotes = null;
let _openDiscovery = null;
let _closeBrowse = null;
let _closeNotes = null;
let _closeDiscovery = null;

// DOM refs
let _bar = null;
let _track = null;
let _label = null;
let _dots = null;
let _bloom = null;
let _bloomLayout = null;
let _bloomViewToggle = null;

// ── Helpers ──────────────────────────────────────────────────────────

function _wrap(index) {
  return ((index % ORB_ORDER.length) + ORB_ORDER.length) % ORB_ORDER.length;
}

/** Build the inline CSS variables for an orb's color triple */
function _colorVars(colorKey) {
  return `--orb-hi:var(--orb-${colorKey}-hi);--orb-base:var(--orb-${colorKey});--orb-lo:var(--orb-${colorKey}-lo)`;
}

/** Classify each orb's visual state relative to activeIndex */
function _stateFor(index) {
  const dist = Math.min(
    Math.abs(index - _activeIndex),
    ORB_ORDER.length - Math.abs(index - _activeIndex)
  );
  if (dist === 0) return 'active';
  if (dist === 1) return 'adjacent';
  if (dist === 2) return 'far';
  return 'hidden';
}

/** Create the DOM for a single orb (shared between bar and bloom).
 *  Returns a .orb-wrap div containing glow, sphere, and optional rings.
 *  The wrap div establishes the positioning context for absolute children. */
function _createOrbDOM(orb, includeRings) {
  const wrap = document.createElement('div');
  wrap.className = 'orb-wrap';

  const glow = document.createElement('div');
  glow.className = 'orb-glow';
  wrap.appendChild(glow);

  const sphere = document.createElement('div');
  sphere.className = 'orb-sphere';
  if (orb.glyph) {
    const glyph = document.createElement('span');
    glyph.className = 'orb-glyph';
    glyph.textContent = orb.glyph;
    glyph.setAttribute('aria-hidden', 'true');
    sphere.appendChild(glyph);
  }
  wrap.appendChild(sphere);

  if (includeRings) {
    const ring = document.createElement('div');
    ring.className = 'orb-ring';
    wrap.appendChild(ring);
    const ringOuter = document.createElement('div');
    ringOuter.className = 'orb-ring orb-ring-outer';
    wrap.appendChild(ringOuter);
  }

  return wrap;
}

// ── Bar Rendering ────────────────────────────────────────────────────

function _renderDots() {
  if (!_dots) return;
  _dots.innerHTML = '';
  for (let i = 0; i < ORB_ORDER.length; i++) {
    const dot = document.createElement('span');
    dot.className = 'orb-bar-dot';
    if (i === _activeIndex) {
      dot.classList.add('active');
      const c = `var(--orb-${ORB_ORDER[i].color})`;
      dot.style.background = c;
      dot.style.color = c;  // for box-shadow: currentColor
    }
    _dots.appendChild(dot);
  }
}

function renderBar() {
  if (!_track) return;
  _track.innerHTML = '';

  const active = ORB_ORDER[_activeIndex];

  // Update ambient haze color
  _bar.style.setProperty('--orb-active-color', `var(--orb-${active.color})`);

  _renderDots();

  for (let i = 0; i < ORB_ORDER.length; i++) {
    const orb = ORB_ORDER[i];
    const state = _stateFor(i);

    const slot = document.createElement('div');
    slot.className = 'orb-slot';
    slot.dataset.state = state;
    slot.dataset.index = i;
    slot.style.cssText = _colorVars(orb.color);
    slot.setAttribute('role', 'button');
    slot.setAttribute('aria-label', `Switch to ${orb.label}`);
    if (state === 'active') slot.setAttribute('aria-current', 'true');

    slot.appendChild(_createOrbDOM(orb, state === 'active'));

    // Tap adjacent orb to quick-switch
    if (state === 'adjacent' || state === 'far') {
      slot.addEventListener('click', () => _switchTo(i));
    }

    // Tap active orb to open bloom
    if (state === 'active') {
      slot.addEventListener('click', () => {
        if (!_swiping) _openBloom();
      });
    }

    _track.appendChild(slot);
  }

  // Update label
  _label.textContent = active.label;
}

// ── Bloom Rendering ──────────────────────────────────────────────────

function _loadBloomView() {
  try {
    const saved = localStorage.getItem('augmentum_bloom_view');
    if (saved === 'tree' || saved === 'ring') _bloomView = saved;
  } catch { /* ignore */ }
}

function _saveBloomView() {
  try { localStorage.setItem('augmentum_bloom_view', _bloomView); } catch {}
}

function _toggleBloomView() {
  _bloomView = (_bloomView === 'tree') ? 'ring' : 'tree';
  _saveBloomView();
  _syncViewToggleState();
  renderBloom();
}

function _syncViewToggleState() {
  if (!_bloomViewToggle) return;
  _bloomViewToggle.dataset.view = _bloomView;
  _bloomViewToggle.setAttribute(
    'aria-label',
    _bloomView === 'tree' ? 'Switch to ring view' : 'Switch to tree view'
  );
  _bloomViewToggle.title =
    _bloomView === 'tree' ? 'Ring view' : 'Tree view';
}

function renderBloom() {
  if (_bloomView === 'tree') _renderBloomTree();
  else _renderBloomRing();
}

// ── Bloom: Orbital Ring Layout ───────────────────────────────────────
// All 8 orbs on a circle. Active orb at top (12 o'clock).
// Constellation lines connect adjacent orbs in sequence order.
// User can swipe/drag around the ring to rotate selection.

const BLOOM_RING_RADIUS = 120; // px

function _bloomAngle(index, count, rotationOffset) {
  // Distribute evenly. Active at top (-90 deg / -PI/2).
  const step = (2 * Math.PI) / count;
  return (-Math.PI / 2) + (index * step) + rotationOffset;
}

let _bloomRotation = 0;      // current angular offset (radians)
let _bloomDragStart = null;   // {x, y, angle} on drag start
let _bloomDragging = false;

function _renderBloomRing() {
  if (!_bloomLayout) return;
  _bloomLayout.innerHTML = '';

  const active = ORB_ORDER[_activeIndex];
  const count = ORB_ORDER.length;
  const step = (2 * Math.PI) / count;
  const rotationOffset = -_activeIndex * step;

  // Reset rotation so active orb is at top
  _bloomRotation = rotationOffset;

  // Container
  const ring = document.createElement('div');
  ring.className = 'orb-bloom-ring';

  // SVG for ring track + constellation lines
  const svgNS = 'http://www.w3.org/2000/svg';
  const size = BLOOM_RING_RADIUS * 2 + 120;
  const cx = size / 2;
  const cy = size / 2;
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('class', 'orb-bloom-ring-svg');
  svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
  svg.setAttribute('width', size);
  svg.setAttribute('height', size);

  // Ring track circle
  const trackCircle = document.createElementNS(svgNS, 'circle');
  trackCircle.setAttribute('cx', cx);
  trackCircle.setAttribute('cy', cy);
  trackCircle.setAttribute('r', BLOOM_RING_RADIUS);
  trackCircle.setAttribute('class', 'orb-bloom-ring-track');
  svg.appendChild(trackCircle);

  // Constellation lines connecting adjacent orbs in sequence
  for (let i = 0; i < count; i++) {
    const j = (i + 1) % count;
    const a1 = _bloomAngle(i, count, rotationOffset);
    const a2 = _bloomAngle(j, count, rotationOffset);
    const line = document.createElementNS(svgNS, 'line');
    line.setAttribute('x1', cx + BLOOM_RING_RADIUS * Math.cos(a1));
    line.setAttribute('y1', cy + BLOOM_RING_RADIUS * Math.sin(a1));
    line.setAttribute('x2', cx + BLOOM_RING_RADIUS * Math.cos(a2));
    line.setAttribute('y2', cy + BLOOM_RING_RADIUS * Math.sin(a2));
    line.setAttribute('class', 'orb-bloom-ring-edge');
    line.style.stroke = `var(--orb-${ORB_ORDER[i].color})`;
    svg.appendChild(line);
  }

  ring.appendChild(svg);

  // Place orbs on the ring
  for (let i = 0; i < count; i++) {
    const orb = ORB_ORDER[i];
    const isActive = (i === _activeIndex);
    const role = isActive ? 'active' : orb.type;
    const angle = _bloomAngle(i, count, rotationOffset);
    const x = cx + BLOOM_RING_RADIUS * Math.cos(angle);
    const y = cy + BLOOM_RING_RADIUS * Math.sin(angle);

    const cell = _bloomCell(orb, role);
    cell.style.position = 'absolute';
    cell.style.left = x + 'px';
    cell.style.top = y + 'px';
    cell.style.transform = 'translate(-50%, -50%)';
    cell.dataset.bloomIndex = i;

    ring.appendChild(cell);
  }

  // Radial swipe — only activates on drag, not tap.
  // Pointer events go to cells first (for tap/long-press).
  // The ring only captures on actual drag movement.
  ring.addEventListener('pointerdown', _onBloomDragStart);
  ring.addEventListener('pointermove', _onBloomDragMove);
  ring.addEventListener('pointerup', _onBloomDragEnd);
  ring.addEventListener('pointercancel', _onBloomDragEnd);
  ring.style.touchAction = 'none';

  // Mark active orb with a crown indicator
  const activeMark = document.createElement('div');
  activeMark.className = 'orb-bloom-active-mark';
  activeMark.textContent = '\u25B2'; // small triangle pointing at active
  activeMark.style.left = (size / 2) + 'px';
  activeMark.style.top = (size / 2 - BLOOM_RING_RADIUS - 40) + 'px';
  ring.appendChild(activeMark);

  _bloomLayout.appendChild(ring);
}

/** Rotate orbs to their current positions (called during drag) */
function _updateBloomPositions() {
  const ring = _bloomLayout?.querySelector('.orb-bloom-ring');
  if (!ring) return;
  const count = ORB_ORDER.length;
  const size = BLOOM_RING_RADIUS * 2 + 120;
  const cx = size / 2;
  const cy = size / 2;

  const cells = ring.querySelectorAll('.orb-bloom-cell');
  cells.forEach(cell => {
    const i = parseInt(cell.dataset.bloomIndex, 10);
    const angle = _bloomAngle(i, count, _bloomRotation);
    const x = cx + BLOOM_RING_RADIUS * Math.cos(angle);
    const y = cy + BLOOM_RING_RADIUS * Math.sin(angle);
    cell.style.left = x + 'px';
    cell.style.top = y + 'px';
  });

  // Update constellation lines
  const svg = ring.querySelector('.orb-bloom-ring-svg');
  if (svg) {
    const lines = svg.querySelectorAll('.orb-bloom-ring-edge');
    lines.forEach((line, i) => {
      const j = (i + 1) % count;
      const a1 = _bloomAngle(i, count, _bloomRotation);
      const a2 = _bloomAngle(j, count, _bloomRotation);
      line.setAttribute('x1', cx + BLOOM_RING_RADIUS * Math.cos(a1));
      line.setAttribute('y1', cy + BLOOM_RING_RADIUS * Math.sin(a1));
      line.setAttribute('x2', cx + BLOOM_RING_RADIUS * Math.cos(a2));
      line.setAttribute('y2', cy + BLOOM_RING_RADIUS * Math.sin(a2));
    });
  }
}

// ── Bloom Ring Drag (radial swipe) ──────────────────────────────────

function _onBloomDragStart(e) {
  // Don't capture if the event originated on an orb cell (let cell handlers work)
  if (e.target.closest('.orb-bloom-cell')) {
    _bloomDragStart = null;
    return;
  }
  const ring = e.currentTarget;
  const rect = ring.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  _bloomDragStart = {
    x: e.clientX,
    y: e.clientY,
    angle: Math.atan2(e.clientY - cy, e.clientX - cx),
    rotation: _bloomRotation,
  };
  _bloomDragging = false;
  ring.setPointerCapture(e.pointerId);
}

function _onBloomDragMove(e) {
  if (!_bloomDragStart) return;
  const ring = e.currentTarget;
  const rect = ring.getBoundingClientRect();
  const cx = rect.left + rect.width / 2;
  const cy = rect.top + rect.height / 2;
  const currentAngle = Math.atan2(e.clientY - cy, e.clientX - cx);
  const delta = currentAngle - _bloomDragStart.angle;

  // Only start dragging after a small threshold
  if (!_bloomDragging && Math.abs(delta) > 0.05) {
    _bloomDragging = true;
    ring.classList.add('orb-bloom-dragging');
  }

  if (_bloomDragging) {
    _bloomRotation = _bloomDragStart.rotation + delta;
    _updateBloomPositions();
  }
}

function _onBloomDragEnd(e) {
  if (!_bloomDragStart) return;
  const ring = e.currentTarget;
  ring.classList.remove('orb-bloom-dragging');

  if (_bloomDragging) {
    // Snap to nearest orb position
    const count = ORB_ORDER.length;
    const step = (2 * Math.PI) / count;
    const snapped = Math.round(_bloomRotation / step) * step;
    _bloomRotation = snapped;

    // Determine which orb is now at the top (the new active)
    // The orb at top is the one whose angle is closest to -PI/2
    let closest = 0;
    let closestDist = Infinity;
    for (let i = 0; i < count; i++) {
      const angle = _bloomAngle(i, count, _bloomRotation);
      // Normalize to [-PI, PI]
      let diff = angle - (-Math.PI / 2);
      while (diff > Math.PI) diff -= 2 * Math.PI;
      while (diff < -Math.PI) diff += 2 * Math.PI;
      if (Math.abs(diff) < closestDist) {
        closestDist = Math.abs(diff);
        closest = i;
      }
    }

    if (closest !== _activeIndex) {
      _switchTo(closest);
    }

    _updateBloomPositions();
  }

  _bloomDragStart = null;
  _bloomDragging = false;
}

// ── Bloom: Tree Layout (sprouting circuit) ───────────────────────────
// Chat trunk at bottom; tier-1 modes branch up; tier-2 features are leaves.
// Coordinates are within a 360×420 canvas. The layout reads bottom-up.
const TREE_CANVAS = { w: 360, h: 420 };
const TREE_NODES = {
  passthrough: { cx: 180, cy: 375, tier: 0 },
  analytical:  { cx:  80, cy: 240, tier: 1, parent: 'passthrough' },
  narrative:   { cx: 180, cy: 205, tier: 1, parent: 'passthrough' },
  agentic:     { cx: 285, cy: 240, tier: 1, parent: 'passthrough' },
  browse:      { cx:  30, cy: 105, tier: 2, parent: 'analytical' },
  foryou:      { cx: 120,  cy: 65, tier: 2, parent: 'analytical' },
  notes:       { cx: 180,  cy: 80, tier: 2, parent: 'narrative' },
  coder:       { cx: 300, cy: 105, tier: 2, parent: 'agentic' },
};

function _treeBezier(from, to) {
  // S-curve that keeps tangents vertical at both endpoints — feels like
  // sap rising, not a wire diagonal.
  const midY = (from.cy + to.cy) / 2;
  return `M ${from.cx} ${from.cy} C ${from.cx} ${midY}, ${to.cx} ${midY}, ${to.cx} ${to.cy}`;
}

// Sample the same cubic Bezier used by _treeBezier at parameter t ∈ [0,1].
// Used to plant static checkpoint beads along each branch.
function _treePointAt(from, to, t) {
  const midY = (from.cy + to.cy) / 2;
  const mt = 1 - t;
  const b0 = mt * mt * mt;
  const b1 = 3 * mt * mt * t;
  const b2 = 3 * mt * t * t;
  const b3 = t * t * t;
  return {
    x: (b0 + b1) * from.cx + (b2 + b3) * to.cx,
    y: b0 * from.cy + (b1 + b2) * midY + b3 * to.cy,
  };
}

const TREE_CHECKPOINTS = [0.2, 0.5, 0.8];

function _treeActivePath() {
  const active = ORB_ORDER[_activeIndex];
  const path = new Set();
  if (!active) return path;
  let cur = TREE_NODES[active.id];
  if (!cur) return path;
  path.add(active.id);
  while (cur && cur.parent) {
    path.add(cur.parent);
    cur = TREE_NODES[cur.parent];
  }
  return path;
}

function _resolveOrbColor(colorKey) {
  // Resolve --orb-<colorKey> CSS variable at render time so SVG gradient
  // stops get a real rgb/hex value (safer than var() in stop-color across
  // older mobile browsers).
  const v = getComputedStyle(document.documentElement)
    .getPropertyValue(`--orb-${colorKey}`).trim();
  return v || '#ffffff';
}

function _renderBloomTree() {
  if (!_bloomLayout) return;
  _bloomLayout.innerHTML = '';

  const wrap = document.createElement('div');
  wrap.className = 'orb-bloom-tree';
  wrap.style.width = TREE_CANVAS.w + 'px';
  wrap.style.height = TREE_CANVAS.h + 'px';

  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('class', 'orb-bloom-tree-svg');
  svg.setAttribute('viewBox', `0 0 ${TREE_CANVAS.w} ${TREE_CANVAS.h}`);
  svg.setAttribute('width', TREE_CANVAS.w);
  svg.setAttribute('height', TREE_CANVAS.h);

  const defs = document.createElementNS(svgNS, 'defs');
  svg.appendChild(defs);

  const activeSet = _treeActivePath();
  let gradIdx = 0;

  for (const [id, node] of Object.entries(TREE_NODES)) {
    if (!node.parent) continue;
    const parent = TREE_NODES[node.parent];
    const parentOrb = ORB_ORDER.find(o => o.id === node.parent);
    const childOrb = ORB_ORDER.find(o => o.id === id);
    if (!parentOrb || !childOrb) continue;

    const gradId = `orb-tree-grad-${gradIdx++}`;
    const grad = document.createElementNS(svgNS, 'linearGradient');
    grad.setAttribute('id', gradId);
    grad.setAttribute('x1', parent.cx);
    grad.setAttribute('y1', parent.cy);
    grad.setAttribute('x2', node.cx);
    grad.setAttribute('y2', node.cy);
    grad.setAttribute('gradientUnits', 'userSpaceOnUse');

    const s1 = document.createElementNS(svgNS, 'stop');
    s1.setAttribute('offset', '0%');
    s1.setAttribute('stop-color', _resolveOrbColor(parentOrb.color));
    const s2 = document.createElementNS(svgNS, 'stop');
    s2.setAttribute('offset', '100%');
    s2.setAttribute('stop-color', _resolveOrbColor(childOrb.color));
    grad.append(s1, s2);
    defs.appendChild(grad);

    const pathId = `orb-tree-path-${gradIdx - 1}`;
    const path = document.createElementNS(svgNS, 'path');
    path.setAttribute('id', pathId);
    path.setAttribute('d', _treeBezier(parent, node));
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', `url(#${gradId})`);
    path.setAttribute('pathLength', '1');
    path.setAttribute('class', 'orb-bloom-tree-branch');
    path.dataset.from = node.parent;
    path.dataset.to = id;
    path.dataset.tier = String(node.tier);
    const isActiveBranch = activeSet.has(id);
    if (isActiveBranch) path.classList.add('active');
    // Stagger draw-on: trunk (tier-1 branches) first, then leaves.
    path.style.animationDelay = (node.tier === 1 ? 0 : 180) + 'ms';
    svg.appendChild(path);

    // Static checkpoint beads: visible nodes along every branch. Dim when
    // inactive, bright when on the active path. The traveling light (below)
    // sweeps over these and makes them flare as it passes.
    for (const t of TREE_CHECKPOINTS) {
      const pt = _treePointAt(parent, node, t);
      const checkpoint = document.createElementNS(svgNS, 'circle');
      checkpoint.setAttribute('cx', pt.x);
      checkpoint.setAttribute('cy', pt.y);
      checkpoint.setAttribute('r', '2.6');
      checkpoint.setAttribute(
        'class',
        'orb-tree-checkpoint' + (isActiveBranch ? ' active' : ''),
      );
      checkpoint.style.color = _resolveOrbColor(childOrb.color);
      checkpoint.dataset.branch = id;
      svg.appendChild(checkpoint);
    }

    // Traveling light: a luminous spark (+ child glyph) sweeps parent → child
    // on the active branch, passing through each checkpoint. Eased hops make
    // the light accelerate between checkpoints and settle briefly at each.
    // Fires after the branch has drawn on. Suppressed under
    // prefers-reduced-motion via CSS.
    if (isActiveBranch) {
      const sigil = document.createElementNS(svgNS, 'g');
      sigil.setAttribute('class', 'orb-tree-sigil');
      sigil.style.color = _resolveOrbColor(childOrb.color);

      const bead = document.createElementNS(svgNS, 'circle');
      bead.setAttribute('r', '4.5');
      bead.setAttribute('class', 'orb-tree-sigil-bead');
      sigil.appendChild(bead);

      if (childOrb.glyph) {
        const text = document.createElementNS(svgNS, 'text');
        text.setAttribute('class', 'orb-tree-sigil-glyph');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('dominant-baseline', 'central');
        text.textContent = childOrb.glyph;
        sigil.appendChild(text);
      }

      const motion = document.createElementNS(svgNS, 'animateMotion');
      motion.setAttribute('dur', '2.6s');
      motion.setAttribute('repeatCount', 'indefinite');
      motion.setAttribute('rotate', '0');
      // keyPoints snap the light onto each checkpoint (0.2/0.5/0.8) with
      // short dwells, then ease to the next — reads as "hopping between
      // checkpoints" rather than a smooth drift.
      motion.setAttribute('calcMode', 'spline');
      motion.setAttribute('keyTimes',
        '0;0.16;0.22;0.44;0.5;0.72;0.78;0.94;1');
      motion.setAttribute('keyPoints',
        '0;0.2;0.2;0.5;0.5;0.8;0.8;1;1');
      motion.setAttribute('keySplines',
        '0.4 0 0.2 1;0 0 1 1;0.4 0 0.2 1;0 0 1 1;0.4 0 0.2 1;0 0 1 1;0.4 0 0.2 1;0 0 1 1');
      motion.setAttribute('begin', node.tier === 1 ? '0.75s' : '0.95s');
      const mpath = document.createElementNS(svgNS, 'mpath');
      mpath.setAttribute('href', `#${pathId}`);
      mpath.setAttributeNS('http://www.w3.org/1999/xlink', 'xlink:href', `#${pathId}`);
      motion.appendChild(mpath);
      sigil.appendChild(motion);

      svg.appendChild(sigil);
    }
  }

  wrap.appendChild(svg);

  for (const [id, node] of Object.entries(TREE_NODES)) {
    const orb = ORB_ORDER.find(o => o.id === id);
    if (!orb) continue;
    const idx = ORB_ORDER.findIndex(o => o.id === id);
    const isActive = idx === _activeIndex;
    const role = isActive
      ? 'active'
      : (node.tier === 0 ? 'mode' : orb.type);

    const cell = _bloomCell(orb, role);
    cell.style.position = 'absolute';
    cell.style.left = node.cx + 'px';
    cell.style.top = node.cy + 'px';
    cell.style.transform = 'translate(-50%, -50%)';
    cell.dataset.treeTier = String(node.tier);
    if (activeSet.has(id)) cell.dataset.onActivePath = 'true';
    // Pop-in stagger to match branches
    cell.style.animationDelay =
      (node.tier === 0 ? 0 : node.tier === 1 ? 120 : 280) + 'ms';

    wrap.appendChild(cell);
  }

  _bloomLayout.appendChild(wrap);
}

function _bloomRow(extraClass) {
  const row = document.createElement('div');
  row.className = 'orb-bloom-row' + (extraClass ? ' ' + extraClass : '');
  return row;
}

function _bloomCell(orb, role) {
  const cell = document.createElement('div');
  cell.className = 'orb-bloom-cell';
  cell.dataset.bloomRole = role;
  cell.dataset.id = orb.id;
  cell.style.cssText = `color:var(--orb-${orb.color});${_colorVars(orb.color)}`;

  cell.appendChild(_createOrbDOM(orb, role === 'active'));

  const name = document.createElement('div');
  name.className = 'orb-bloom-name';
  name.textContent = orb.label;
  cell.appendChild(name);

  const desc = document.createElement('div');
  desc.className = 'orb-bloom-desc';
  desc.textContent = orb.desc;
  cell.appendChild(desc);

  // Color edit button — always visible, small paint icon under the orb
  const colorBtn = document.createElement('div');
  colorBtn.className = 'orb-bloom-color-btn';
  colorBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="12" r="10"/></svg>';
  colorBtn.style.color = `var(--orb-${orb.color})`;
  colorBtn.title = 'Change color';
  colorBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    _openColorPicker(cell, orb);
  });
  cell.appendChild(colorBtn);

  // Tap → switch mode, long-press → open alongside (or reorder if active orb)
  let _pressTimer = null;
  let _didLongPress = false;

  cell.addEventListener('pointerdown', (e) => {
    if (e.target.closest('.orb-bloom-color-btn') || e.target.closest('.orb-color-picker')) return;
    _didLongPress = false;
    _pressTimer = setTimeout(() => {
      _didLongPress = true;
      const idx = ORB_ORDER.findIndex(o => o.id === orb.id);
      if (idx === _activeIndex) {
        // Long-press on ACTIVE orb → reorder mode
        _startReorder(cell, orb);
      } else {
        // Long-press on ANOTHER orb → open alongside
        _openAlongside(orb);
        _closeBloom();
      }
    }, 500);
  });

  cell.addEventListener('pointerup', (e) => {
    if (e.target.closest('.orb-bloom-color-btn') || e.target.closest('.orb-color-picker')) return;
    clearTimeout(_pressTimer);
    if (_didLongPress) return;
    if (_reorderActive) {
      _finishReorder(cell, orb);
      return;
    }
    e.stopPropagation();
    const idx = ORB_ORDER.findIndex(o => o.id === orb.id);
    _switchTo(idx);
    _closeBloom();
  });

  cell.addEventListener('pointercancel', () => clearTimeout(_pressTimer));
  cell.addEventListener('pointerleave', () => clearTimeout(_pressTimer));

  return cell;
}

// ── Open Alongside (long-press non-active orb) ──────────────────────

/** Map orb IDs to surface types */
const _ORB_TO_SURFACE = {
  passthrough: 'chat',
  analytical: 'chat',
  narrative: 'narrative',
  coder: 'coder',
  agentic: 'chat',
  browse: 'browse',
  notes: 'browse',   // notes is a tab within browse surface
  foryou: 'browse',  // discovery is a tab within browse surface
};

/** Orb IDs that represent a chat/session-bearing mode */
const _ORB_SESSION_MODES = new Set(['passthrough', 'analytical', 'narrative', 'agentic']);

/**
 * Map "feature inside browse" orbs to the inner tab they should activate.
 * Notes and ForYou both fold into the browse-panel surface (the panel hosts
 * three subtrees switched by browse.js's `switchTab`); without this signal,
 * dropping the Notes orb on top of an existing Browse silently focused the
 * panel without changing tabs (audit §4.4).
 */
const _ORB_TO_BROWSE_TAB = {
  browse: 'browse',
  notes: 'notes',
  foryou: 'discovery',
};

function _dispatchBrowseTabFromOrb(orbId) {
  const tab = _ORB_TO_BROWSE_TAB[orbId];
  if (!tab) return;
  document.dispatchEvent(new CustomEvent('augmentum:switch-browse-tab', {
    detail: { tab },
  }));
}

/** Pretty label for an orb id; falls back to the id if not registered. */
function _orbLabel(orbId) {
  const orb = ORB_ORDER.find((o) => o.id === orbId);
  return orb?.label || orbId;
}

/**
 * Surface types that own singleton DOM elements (#browse-panel,
 * #coder-terminal-pane, #image-panel). appendChild moves nodes rather than
 * cloning them, so a second instance silently steals the DOM from the first.
 * Keep these capped at one instance until their DOM is refactored to
 * per-surface containers. Drag-open falls through to focusing the existing
 * instance instead of creating a broken second one.
 */
const _SINGLE_INSTANCE_SURFACES = new Set(['coder', 'browse', 'image']);

/**
 * Pick the best existing session to open inside a newly-created surface so
 * the user keeps their place instead of landing on an empty "new chat"
 * screen. Preference: the currently-active session (if its mode matches),
 * else the most recent session for that mode. Returns null when nothing
 * useful exists — the surface will then create a fresh session itself.
 */
export function resolveInheritedSession(orbId) {
  if (!_ORB_SESSION_MODES.has(orbId)) return null;
  const mode = orbId;
  // Sessions already displayed by a live tab (primary included) are off the
  // table: two surfaces sharing a sessionId both mutate the same tree
  // (last-writer-wins on stream) and share the same backend handler — the
  // corruption case the surface-architecture design forbids (audit §7.18,
  // design Q5 "same browser session = forbidden"). The new tab instead picks
  // up the most recent session of the mode that ISN'T already on screen,
  // else starts fresh — sensible, since the bound one is visible in its own
  // tab already.
  const bound = new Set();
  for (const s of SurfaceRegistry.all()) {
    if (s._sessionId) bound.add(s._sessionId);
  }
  const activeId = sessionStore.getActiveId();
  if (activeId && !bound.has(activeId)) {
    const active = sessionStore.get(activeId);
    if (active && (active.mode || 'passthrough') === mode) return activeId;
  }
  for (const m of sessionStore.forMode(mode)) {
    if (!bound.has(m.id)) return m.id;
  }
  return null;
}

/**
 * Canonical "open a surface alongside the current one" ladder.
 *
 * Every gesture and programmatic path that spawns a non-primary surface MUST
 * route through here — orb long-press/drag, sidebar ctrl+click, command
 * composer actions, voice commands. The audit's Phase 0 fixes (cap toast
 * §4.6, singleton-focus toast §4.7, notes/foryou inner-tab routing §4.4,
 * boot-restore race guard) were originally applied only to the orb path and
 * silently missing from every later caller; keeping one ladder is the
 * class fix.
 *
 * @param {object} opts
 * @param {string} [opts.orbId]   orb id (mode ids match orb ids); drives
 *                                browse inner-tab routing + label
 * @param {string} [opts.type]    explicit surface type — overrides orb map
 * @param {string} [opts.mode]    chat-family mode for the new surface
 * @param {object} [opts.config]  extra config forwarded to the surface
 * @param {boolean} [opts.inherit=true]  adopt the most recent un-displayed
 *                                session of the mode. Pass false for "new/
 *                                fresh" semantics (voice "new chat", command
 *                                chains) where resuming an old session would
 *                                contradict explicit user intent.
 * @returns {Promise<{ok: boolean, surface?: object, reason?: string}>}
 */
export async function openSurfaceAlongside({ orbId = null, type = null, mode = null, config = {}, inherit = true } = {}) {
  const surfaceType = type || _ORB_TO_SURFACE[orbId] || 'chat';

  // Check if we can create this surface type
  if (!SurfaceRegistry.hasType(surfaceType)) {
    console.warn('Surface type not registered:', surfaceType);
    return { ok: false, reason: 'unknown-type' };
  }

  // If app boot is still restoring surfaces, wait for it to finish before
  // creating anything. Without this a fast gesture during the ~100-500ms
  // boot window can produce a default-config surface that silently clobbers
  // the saved one boot was about to restore (losing workspaceId, sessionId,
  // etc), or push the restore over the 4-tab cap. Waiting takes ≤500ms in
  // practice and is harmless once boot has finished.
  if (_waitForBoot) {
    try { await _waitForBoot(); } catch { /* best-effort */ }
  }

  // Singleton-DOM surfaces can't safely multi-instance — focus the existing
  // one so the user's gesture lands somewhere useful instead of no-op-ing.
  if (_SINGLE_INSTANCE_SURFACES.has(surfaceType)) {
    const existing = SurfaceRegistry.ofType(surfaceType)[0];
    if (existing) {
      SurfaceRegistry.focus(existing.id);
      // Notes/ForYou land on the same browse surface — switch its inner
      // tab so the gesture visibly changes the view, not just the tab dot
      // (audit §4.4).
      if (orbId) _dispatchBrowseTabFromOrb(orbId);
      if (navigator.vibrate) navigator.vibrate(20);
      // Tell the user what happened. Without this, repeating the gesture
      // feels like a no-op (audit §4.7).
      showToast(`${_orbLabel(orbId || surfaceType)} already open — focused existing tab`, 'info');
      return { ok: false, reason: 'singleton-focused', surface: existing };
    }
  }

  // Cap at 4 surfaces. Surface the limit visibly — a silent drop feels
  // broken; users repeat the gesture thinking they missed (audit §4.6).
  if (SurfaceRegistry.all().length >= 4) {
    console.warn('Maximum surfaces reached (4)');
    if (navigator.vibrate) navigator.vibrate([20, 40, 20]);
    showToast('4-tab limit reached — close a tab first', 'warning');
    return { ok: false, reason: 'cap' };
  }

  const effectiveMode = mode || config.mode || (orbId && _ORB_SESSION_MODES.has(orbId) ? orbId : undefined);
  const cfg = {
    ...config,
    analytical: !!config.analytical || effectiveMode === 'analytical',
    mode: effectiveMode,
  };

  // Inherit a session so the new tab continues the user's work rather than
  // booting into an empty state. Narrative inherits character fields from
  // the session it adopts. Skipped when the caller pinned a sessionId or
  // asked for fresh-session semantics.
  if (inherit && !cfg.sessionId && effectiveMode && _ORB_SESSION_MODES.has(effectiveMode)) {
    const inheritedSessionId = resolveInheritedSession(effectiveMode);
    if (inheritedSessionId) {
      cfg.sessionId = inheritedSessionId;
      if (surfaceType === 'narrative') {
        const s = sessionStore.get(inheritedSessionId);
        if (s) {
          cfg.characterId = cfg.characterId || s.characterId || '';
          cfg.characterName = cfg.characterName || s.characterName || s.title || '';
        }
      }
    }
  }

  const surface = SurfaceRegistry.create(surfaceType, cfg);
  LayoutManager.mountSurface(surface);
  SurfaceRegistry.focus(surface.id);

  // For freshly-created browse surfaces, route Notes/ForYou orbs to the
  // matching inner tab so the new tab opens on the requested view.
  if (orbId) _dispatchBrowseTabFromOrb(orbId);

  // Save workspace state
  SurfaceRegistry.saveWorkspace();

  // Haptic feedback
  if (navigator.vibrate) navigator.vibrate(20);

  return { ok: true, surface };
}

async function _openAlongside(orb) {
  await openSurfaceAlongside({ orbId: orb.id });
}

/**
 * Find an existing surface that matches an orb. Prefer non-primary surfaces
 * (those have real pinned state); fall back to primary when no dedicated
 * surface exists yet. Returns null when nothing matches (caller should
 * fall through to the legacy mode-swap path).
 */
function _pickSurfaceForOrb(orbId) {
  const surfaceType = _ORB_TO_SURFACE[orbId] || 'chat';
  const all = SurfaceRegistry.ofType(surfaceType);
  if (all.length === 0) return null;

  if (surfaceType === 'chat') {
    // Chat-typed surfaces split by mode (passthrough/analytical/agentic)
    const nonPrimary = all.find(s => !s._isPrimary && s._mode === orbId);
    if (nonPrimary) return nonPrimary;
    const primary = all.find(s => s._isPrimary);
    return primary || null;
  }
  // narrative/coder/browse/image — single type
  const nonPrimary = all.find(s => !s._isPrimary);
  return nonPrimary || all.find(s => s._isPrimary) || null;
}

// ── Reorder (long-press in constellation) ────────────────────────────

let _reorderActive = false;
let _reorderSourceOrb = null;

function _startReorder(cell, orb) {
  _reorderActive = true;
  _reorderSourceOrb = orb;
  cell.classList.add('orb-reorder-source');

  // Pulse all other orbs to indicate drop targets
  const constellation = cell.closest('.orb-constellation');
  if (constellation) {
    constellation.querySelectorAll('.orb-bloom-cell').forEach(c => {
      if (c !== cell) c.classList.add('orb-reorder-target');
    });
  }
  // Haptic feedback if available
  if (navigator.vibrate) navigator.vibrate(30);
}

function _finishReorder(targetCell, targetOrb) {
  if (!_reorderSourceOrb || _reorderSourceOrb.id === targetOrb.id) {
    _cancelReorder();
    return;
  }
  // Swap positions in ORB_ORDER
  const srcIdx = ORB_ORDER.findIndex(o => o.id === _reorderSourceOrb.id);
  const dstIdx = ORB_ORDER.findIndex(o => o.id === targetOrb.id);
  if (srcIdx >= 0 && dstIdx >= 0) {
    [ORB_ORDER[srcIdx], ORB_ORDER[dstIdx]] = [ORB_ORDER[dstIdx], ORB_ORDER[srcIdx]];
    // Update active index to follow the moved orb if needed
    if (_activeIndex === srcIdx) _activeIndex = dstIdx;
    else if (_activeIndex === dstIdx) _activeIndex = srcIdx;
    _modeIndex = _activeIndex;
    _saveCustomOrder();
  }
  _cancelReorder();
  renderBloom();
  renderBar();
}

function _cancelReorder() {
  _reorderActive = false;
  _reorderSourceOrb = null;
  document.querySelectorAll('.orb-reorder-source, .orb-reorder-target').forEach(el => {
    el.classList.remove('orb-reorder-source', 'orb-reorder-target');
  });
}

function _saveCustomOrder() {
  const order = ORB_ORDER.map(o => o.id);
  const json = JSON.stringify(order);
  // Save to server (primary) + localStorage (fallback)
  localStorage.setItem('augmentum_orb_order', json);
  fetch('/api/config/ui', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ orbCustomOrder: json }),
  }).catch(() => {});
}

function _loadCustomOrder() {
  try {
    const saved = JSON.parse(localStorage.getItem('augmentum_orb_order'));
    if (!saved || !Array.isArray(saved)) return;
    _applyOrder(saved);
  } catch { /* ignore */ }
}

function _applyOrder(savedIds) {
  const byId = {};
  ORB_ORDER.forEach(o => byId[o.id] = o);
  const reordered = [];
  for (const id of savedIds) {
    if (byId[id]) { reordered.push(byId[id]); delete byId[id]; }
  }
  Object.values(byId).forEach(o => reordered.push(o));
  ORB_ORDER.length = 0;
  reordered.forEach(o => ORB_ORDER.push(o));
}

/** Load orb customizations from the server (called after settings load). */
async function _loadFromServer() {
  try {
    const resp = await fetch('/api/config/ui');
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.orbCustomOrder) {
      try {
        const order = JSON.parse(data.orbCustomOrder);
        if (Array.isArray(order) && order.length) {
          _applyOrder(order);
          localStorage.setItem('augmentum_orb_order', data.orbCustomOrder);
        }
      } catch { /* ignore */ }
    }
    if (data.orbCustomColors) {
      try {
        const colors = JSON.parse(data.orbCustomColors);
        if (colors && typeof colors === 'object') {
          _applyColors(colors);
          localStorage.setItem('augmentum_orb_colors', data.orbCustomColors);
        }
      } catch { /* ignore */ }
    }
    renderBar();
  } catch { /* ignore */ }
}

// ── Color Customization ─────────────────────────────────────────────

function _openColorPicker(cell, orb) {
  // Close any existing picker
  document.querySelectorAll('.orb-color-picker').forEach(p => p.remove());

  const picker = document.createElement('div');
  picker.className = 'orb-color-picker';

  // Position picker below if orb is near the top of the viewport
  const cellRect = cell.getBoundingClientRect();
  if (cellRect.top < 100) {
    picker.classList.add('orb-color-picker-below');
  }

  const PALETTE = [
    '#4caf50', '#2196f3', '#9c27b0', '#ff9800', '#06b6d4',
    '#e91e63', '#ffc107', '#8bc34a', '#f44336', '#00bcd4',
    '#7c4dff', '#ff5722', '#009688', '#cddc39', '#795548',
    '#607d8b',
  ];

  PALETTE.forEach(color => {
    const swatch = document.createElement('div');
    swatch.className = 'orb-color-swatch';
    swatch.style.background = color;
    // Check if this swatch matches the current color (compare via a temp element
    // since getComputedStyle returns rgb() not hex)
    const savedColors = safeParseJSON(localStorage.getItem('augmentum_orb_colors'), {});
    if (savedColors[orb.color] === color) {
      swatch.classList.add('active');
    }
    swatch.addEventListener('click', (e) => {
      e.stopPropagation();
      _applyCustomColor(orb, color);
      picker.remove();
      renderBloom();
      renderBar();
    });
    picker.appendChild(swatch);
  });

  cell.appendChild(picker);

  // Close picker on outside click
  setTimeout(() => {
    document.addEventListener('click', function _closePicker() {
      picker.remove();
      document.removeEventListener('click', _closePicker);
    }, { once: true });
  }, 0);
}

function _applyCustomColor(orb, hexColor) {
  // Generate hi/base/lo from hex by lightening/darkening
  const r = parseInt(hexColor.slice(1, 3), 16);
  const g = parseInt(hexColor.slice(3, 5), 16);
  const b = parseInt(hexColor.slice(5, 7), 16);

  const hi = `rgb(${Math.min(255, r + 80)}, ${Math.min(255, g + 80)}, ${Math.min(255, b + 80)})`;
  const lo = `rgb(${Math.max(0, r - 60)}, ${Math.max(0, g - 60)}, ${Math.max(0, b - 60)})`;

  document.documentElement.style.setProperty(`--orb-${orb.color}`, hexColor);
  document.documentElement.style.setProperty(`--orb-${orb.color}-hi`, hi);
  document.documentElement.style.setProperty(`--orb-${orb.color}-lo`, lo);

  // Persist to server + localStorage
  const saved = safeParseJSON(localStorage.getItem('augmentum_orb_colors'), {});
  saved[orb.color] = hexColor;
  const json = JSON.stringify(saved);
  localStorage.setItem('augmentum_orb_colors', json);
  fetch('/api/config/ui', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ orbCustomColors: json }),
  }).catch(() => {});
}

function _applyColors(colorMap) {
  for (const [colorKey, hex] of Object.entries(colorMap)) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    document.documentElement.style.setProperty(`--orb-${colorKey}`, hex);
    document.documentElement.style.setProperty(`--orb-${colorKey}-hi`,
      `rgb(${Math.min(255, r + 80)}, ${Math.min(255, g + 80)}, ${Math.min(255, b + 80)})`);
    document.documentElement.style.setProperty(`--orb-${colorKey}-lo`,
      `rgb(${Math.max(0, r - 60)}, ${Math.max(0, g - 60)}, ${Math.max(0, b - 60)})`);
  }
}

function _loadCustomColors() {
  try {
    const saved = JSON.parse(localStorage.getItem('augmentum_orb_colors'));
    if (!saved) return;
    _applyColors(saved);
  } catch { /* ignore */ }
}

// ── Bloom Open/Close ─────────────────────────────────────────────────

function _openBloom() {
  if (_bloomOpen) return;
  _bloomOpen = true;
  renderBloom();
  _bloom.classList.add('open');
  _bar.style.opacity = '0';
  _bar.style.pointerEvents = 'none';
}

function _closeBloom() {
  if (!_bloomOpen) return;
  _bloomOpen = false;
  _cancelReorder();
  _bloom.classList.remove('open');
  _bar.style.opacity = '';
  _bar.style.pointerEvents = '';
}

// ── Switch Logic ─────────────────────────────────────────────────────

/** Close whatever feature panel is currently open */
function _closeActiveFeature() {
  if (!_featureOpen) return;
  if (_featureOpen === 'browse' && _closeBrowse) _closeBrowse();
  else if (_featureOpen === 'notes' && _closeNotes) _closeNotes();
  else if (_featureOpen === 'foryou' && _closeDiscovery) _closeDiscovery();
  _featureOpen = null;
}

function _switchTo(index) {
  _activeIndex = _wrap(index);
  const orb = ORB_ORDER[_activeIndex];

  if (orb.type === 'mode') {
    // Switching to a mode — close any open feature panel first
    _closeActiveFeature();
    _modeIndex = _activeIndex;

    // Prefer focusing an already-open surface for this mode so the user's
    // pinned work isn't clobbered by a legacy mode-swap. Fall back to the
    // legacy path only when nothing matches (e.g., first time touching
    // coder without any coder surface yet).
    const target = _pickSurfaceForOrb(orb.id);
    if (target) {
      SurfaceRegistry.focus(target.id);
      // Realign the shared mode-DOM whenever the app's current mode diverges
      // from the orb we're switching to. Without this, switching out of a
      // mode that owns top-level DOM in main-area (coder's #coder-layout,
      // #coder-files-view, #coder-mobile-tabs — siblings of #surface-grid,
      // not adopted into any surface container) left that DOM on screen
      // because applyMode() never ran. Surface focus only covers DOM
      // actually reparented into a container.
      const currentMode = document.getElementById('app')?.getAttribute('data-mode');
      if (_setMode && currentMode !== orb.id) {
        _setMode(orb.id);
      }
    } else if (_setMode) {
      _setMode(orb.id);
    }
  } else {
    // Switching to a feature — it's a temporary overlay, not a mode change.
    // Close any previously open feature panel if different.
    if (_featureOpen && _featureOpen !== orb.id) _closeActiveFeature();

    _featureOpen = orb.id;
    if (orb.id === 'browse' && _openBrowse) _openBrowse();
    else if (orb.id === 'notes' && _openNotes) _openNotes();
    else if (orb.id === 'foryou' && _openDiscovery) _openDiscovery();
  }

  renderBar();
  if (_bloomOpen) renderBloom();
}

/** Called when a feature panel is closed externally (e.g. its X button).
 *  Snaps the orb bar back to the current mode. */
function returnToMode() {
  if (_featureOpen) {
    _featureOpen = null;
    _activeIndex = _modeIndex;
    renderBar();
  }
}

// ── Swipe Handling (bottom bar) ──────────────────────────────────────
// Continuous scrub: the track follows the finger in real time; on release
// we commit N orbs based on how far the user dragged (SCRUB_STEP per orb)
// and snap the transform back with the existing spring.

let _touchStartX = null;
let _touchStartY = 0;
let _scrubDeltaX = 0;
let _scrubbing = false;
let _swiping = false;  // kept for renderBar's click-vs-tap guard
const SCRUB_STEP = 50;              // px of drag per one-orb commit
const SCRUB_FLICK_THRESHOLD = 18;   // short fast flick still commits 1

function _haptic() {
  try {
    if (navigator.vibrate) navigator.vibrate(5);
  } catch (_) { /* ignore */ }
}

function _onTouchStart(e) {
  const t = e.touches[0];
  _touchStartX = t.clientX;
  _touchStartY = t.clientY;
  _scrubDeltaX = 0;
  _scrubbing = false;
  _swiping = false;
  if (_track) _track.style.transition = 'none';
}

function _onTouchMove(e) {
  if (_touchStartX == null) return;
  const t = e.touches[0];
  const dx = t.clientX - _touchStartX;
  const dy = t.clientY - _touchStartY;

  if (!_scrubbing && Math.abs(dx) > Math.abs(dy) && Math.abs(dx) > 6) {
    _scrubbing = true;
    _swiping = true;
  }
  if (_scrubbing) {
    e.preventDefault();
    _scrubDeltaX = dx;
    if (_track) _track.style.transform = `translateX(${dx * 0.7}px)`;
  }
}

function _onTouchEnd(e) {
  if (_touchStartX == null) return;

  // Restore spring + reset transform — the spring carries it back to 0
  // while the new layout (if we committed) renders into position.
  if (_track) {
    _track.style.transition = '';
    _track.style.transform = '';
  }

  if (_scrubbing) {
    const dx = _scrubDeltaX;
    let steps = Math.round(-dx / SCRUB_STEP);   // drag left → next orb
    if (steps === 0 && Math.abs(dx) >= SCRUB_FLICK_THRESHOLD) {
      steps = dx < 0 ? 1 : -1;
    }
    if (steps !== 0) {
      _switchTo(_activeIndex + steps);
      _haptic();
    }
  }

  _touchStartX = null;
  _scrubDeltaX = 0;
  _scrubbing = false;
  // Leave _swiping true briefly so the trailing click on active orb
  // doesn't open the bloom after a scrub that didn't preventDefault.
  setTimeout(() => { _swiping = false; }, 30);
}

// ── External Sync ────────────────────────────────────────────────────

function syncToMode(mode) {
  const idx = ORB_ORDER.findIndex(o => o.id === mode);
  if (idx >= 0) {
    _modeIndex = idx;
    // Only update visual if no feature overlay is active
    if (!_featureOpen) {
      _activeIndex = idx;
      renderBar();
    }
  }
}

// ── Init ─────────────────────────────────────────────────────────────

export function initOrbNav({ setMode, openBrowse, openNotes, openDiscovery, closeBrowse, closeNotes, closeDiscovery, initialMode, waitForBoot }) {
  _bar = document.getElementById('orb-nav-bar');
  _track = document.getElementById('orb-bar-track');
  _label = document.getElementById('orb-bar-label');
  _dots = document.getElementById('orb-bar-dots');
  _bloom = document.getElementById('orb-bloom');
  _bloomLayout = document.getElementById('orb-bloom-layout');
  _bloomViewToggle = document.getElementById('orb-bloom-view-toggle');

  if (!_bar || !_track) return;

  _setMode = setMode;
  _openBrowse = openBrowse;
  _openNotes = openNotes;
  _openDiscovery = openDiscovery;
  _closeBrowse = closeBrowse;
  _closeNotes = closeNotes;
  _closeDiscovery = closeDiscovery;
  _waitForBoot = waitForBoot || null;

  _bar.addEventListener('touchstart', _onTouchStart, { passive: true });
  _bar.addEventListener('touchmove', _onTouchMove, { passive: false });
  _bar.addEventListener('touchend', _onTouchEnd, { passive: true });

  _bloom.addEventListener('click', (e) => {
    if (e.target === _bloom) _closeBloom();
  });

  if (_bloomViewToggle) {
    _bloomViewToggle.addEventListener('click', (e) => {
      e.stopPropagation();
      _toggleBloomView();
    });
  }

  _loadBloomView();
  _syncViewToggleState();

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && _bloomOpen) _closeBloom();
  });

  document.addEventListener('augmentum:mode-changed', (e) => {
    syncToMode(e.detail.mode);
  });

  // When a feature panel is closed externally (its X button, Escape, etc.),
  // snap the orb bar back to the current mode
  document.addEventListener('augmentum:feature-closed', () => {
    returnToMode();
  });

  // Restore custom order and colors — localStorage first (instant), then server (async, overrides)
  _loadCustomOrder();
  _loadCustomColors();

  // Sync to actual mode before first render (default is Chat, but user may have saved a different mode)
  if (initialMode) {
    const idx = ORB_ORDER.findIndex(o => o.id === initialMode);
    if (idx >= 0) {
      _activeIndex = idx;
      _modeIndex = idx;
    }
  }

  renderBar();

  // Load server-side customizations (async — will re-render if different from localStorage)
  _loadFromServer();
}

export { syncToMode, returnToMode };
