/* ==========================================================================
   PDF Editor — in-browser PDF viewing + annotation via pdf.js + pdf-lib
   Lazy-loaded when a PDF artifact is opened in the Studio.

   Tools: Select, Text, Highlight, Draw, Shapes, Image, Signature
   Features: Undo/redo, text layer, search, annotation deletion,
             page manipulation, print, embedded annotations on save
   ========================================================================== */

import { escapeHtml } from './app.js';

// ---------------------------------------------------------------------------
// Lazy-loaded libraries
// ---------------------------------------------------------------------------
let _pdfjsLib = null;
let _PDFDocument = null;

async function _ensurePdfJs() {
  if (_pdfjsLib) return _pdfjsLib;
  _pdfjsLib = await import('/ui/lib/pdfjs/pdf.min.mjs');
  _pdfjsLib.GlobalWorkerOptions.workerSrc = '/ui/lib/pdfjs/pdf.worker.min.mjs';
  return _pdfjsLib;
}

async function _ensurePdfLib() {
  if (_PDFDocument) return _PDFDocument;
  if (!window.PDFLib) {
    await new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = '/ui/lib/pdf-lib/pdf-lib.min.js';
      script.onload = resolve;
      script.onerror = reject;
      document.head.appendChild(script);
    });
  }
  _PDFDocument = window.PDFLib.PDFDocument;
  return _PDFDocument;
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let _state = _freshState();

function _freshState() {
  return {
    pdfBytes: null,
    pdfDoc: null,         // pdf.js (render)
    pdfLibDoc: null,      // pdf-lib (edit)
    totalPages: 0,
    currentPage: 1,
    scale: 1.0,
    activeTool: 'select',
    activeShape: 'rect',  // rect | circle | arrow | line
    activeColor: '#000000',
    activeStrokeWidth: 2,
    activeFontSize: 14,
    annotations: [],      // [{page, type, ...data}]
    undoStack: [],
    redoStack: [],
    dirty: false,
    _onDirty: null,
    _drawing: false,
    _drawPoints: [],
    _highlightStart: null,
    _shapeStart: null,
    _selectedIdx: -1,     // index into annotations[]
    _searchOpen: false,
    _searchMatches: [],
    _searchIdx: -1,
    _sigPad: null,        // signature canvas context
    _sigPoints: [],
    _pageRotations: {},   // {pageNum: degrees} — client-side rotation tracking
  };
}

let _dom = {};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export async function initPdfEditor(container, pdfBytes, opts = {}) {
  _state = _freshState();
  _state.pdfBytes = pdfBytes instanceof Uint8Array ? pdfBytes : new Uint8Array(pdfBytes);
  _state._onDirty = opts.onDirty || (() => {});

  const [pdfjsLib, PDFDoc] = await Promise.all([_ensurePdfJs(), _ensurePdfLib()]);

  const loadingTask = pdfjsLib.getDocument({ data: _state.pdfBytes.slice() });
  _state.pdfDoc = await loadingTask.promise;
  _state.totalPages = _state.pdfDoc.numPages;
  _state.pdfLibDoc = await PDFDoc.load(_state.pdfBytes.slice());

  _buildUI(container);
  await _renderPage(_state.currentPage);
  _renderPageNav();
}

export async function getPdfBytes() {
  if (!_state.pdfLibDoc) return _state.pdfBytes;
  // Embed all overlay annotations into pdf-lib before saving
  _embedAnnotationsIntoPdf();
  return await _state.pdfLibDoc.save();
}

export function destroyPdfEditor() {
  if (_state.pdfDoc) { _state.pdfDoc.destroy(); _state.pdfDoc = null; }
  _state.pdfLibDoc = null;
  _state.pdfBytes = null;
  _state.annotations = [];
  _state.undoStack = [];
  _state.redoStack = [];
  _dom = {};
}

export function getCurrentPage() { return _state.currentPage; }
export function getTotalPages() { return _state.totalPages; }

// ---------------------------------------------------------------------------
// Embed overlay annotations into the actual PDF (called before save)
// ---------------------------------------------------------------------------
function _embedAnnotationsIntoPdf() {
  if (!_state.pdfLibDoc) return;
  const pages = _state.pdfLibDoc.getPages();
  const { rgb } = window.PDFLib;

  for (const ann of _state.annotations) {
    const page = pages[ann.page - 1];
    if (!page) continue;
    const { width: pw, height: ph } = page.getSize();

    // We need canvas dimensions at the time of annotation.
    // Use the annotation's stored canvasW/canvasH for accurate mapping.
    const cw = ann._canvasW || pw;
    const ch = ann._canvasH || ph;

    if (ann.type === 'highlight') {
      const { r, g, b: bl } = _hexToRgb(ann.color || '#ffff00');
      const pdfX = (ann.x / cw) * pw;
      const pdfY = ph - ((ann.y + ann.h) / ch) * ph;
      const pdfW = (ann.w / cw) * pw;
      const pdfH = (ann.h / ch) * ph;
      page.drawRectangle({
        x: pdfX, y: pdfY, width: pdfW, height: pdfH,
        color: rgb(r, g, bl),
        opacity: 0.25,
        borderWidth: 0,
      });
    }

    if (ann.type === 'draw' && ann.points?.length >= 2) {
      const { r, g, b: bl } = _hexToRgb(ann.color || '#000000');
      // Convert points to PDF path
      const pts = ann.points.map(p => ({
        x: (p.x / cw) * pw,
        y: ph - (p.y / ch) * ph,
      }));
      // Draw as series of line segments
      for (let i = 0; i < pts.length - 1; i++) {
        page.drawLine({
          start: { x: pts[i].x, y: pts[i].y },
          end: { x: pts[i + 1].x, y: pts[i + 1].y },
          thickness: ann.width || 2,
          color: rgb(r, g, bl),
        });
      }
    }

    if (ann.type === 'shape') {
      const { r, g, b: bl } = _hexToRgb(ann.color || '#000000');
      const x1 = (ann.x / cw) * pw;
      const y1 = ph - (ann.y / ch) * ph;
      const x2 = ((ann.x + ann.w) / cw) * pw;
      const y2 = ph - ((ann.y + ann.h) / ch) * ph;
      const sw = ann.strokeWidth || 2;

      if (ann.shape === 'rect') {
        page.drawRectangle({
          x: Math.min(x1, x2), y: Math.min(y1, y2),
          width: Math.abs(x2 - x1), height: Math.abs(y2 - y1),
          borderColor: rgb(r, g, bl), borderWidth: sw,
          color: undefined,
        });
      } else if (ann.shape === 'circle') {
        const cx = (x1 + x2) / 2;
        const cy = (y1 + y2) / 2;
        const rx = Math.abs(x2 - x1) / 2;
        const ry = Math.abs(y2 - y1) / 2;
        page.drawEllipse({
          x: cx, y: cy, xScale: rx, yScale: ry,
          borderColor: rgb(r, g, bl), borderWidth: sw,
          color: undefined,
        });
      } else if (ann.shape === 'line' || ann.shape === 'arrow') {
        page.drawLine({
          start: { x: x1, y: y1 }, end: { x: x2, y: y2 },
          thickness: sw, color: rgb(r, g, bl),
        });
        if (ann.shape === 'arrow') {
          // Draw arrowhead
          const angle = Math.atan2(y2 - y1, x2 - x1);
          const headLen = sw * 5;
          const a1 = angle + Math.PI / 6;
          const a2 = angle - Math.PI / 6;
          page.drawLine({
            start: { x: x2, y: y2 },
            end: { x: x2 - headLen * Math.cos(a1), y: y2 - headLen * Math.sin(a1) },
            thickness: sw, color: rgb(r, g, bl),
          });
          page.drawLine({
            start: { x: x2, y: y2 },
            end: { x: x2 - headLen * Math.cos(a2), y: y2 - headLen * Math.sin(a2) },
            thickness: sw, color: rgb(r, g, bl),
          });
        }
      }
    }

    if (ann.type === 'signature' && ann.dataUrl) {
      // Signatures are already embedded via _commitSignature → drawImage
      // (handled at creation time like images)
    }
  }
}

// ---------------------------------------------------------------------------
// Undo / Redo
// ---------------------------------------------------------------------------
function _pushUndo() {
  _state.undoStack.push(JSON.parse(JSON.stringify(_state.annotations)));
  _state.redoStack = [];
  if (_state.undoStack.length > 50) _state.undoStack.shift();
}

function _undo() {
  if (_state.undoStack.length === 0) return;
  _state.redoStack.push(JSON.parse(JSON.stringify(_state.annotations)));
  _state.annotations = _state.undoStack.pop();
  _state._selectedIdx = -1;
  _renderAnnotations(_state.currentPage);
  _renderDrawStrokes(_state.currentPage);
  _markDirty();
}

function _redo() {
  if (_state.redoStack.length === 0) return;
  _state.undoStack.push(JSON.parse(JSON.stringify(_state.annotations)));
  _state.annotations = _state.redoStack.pop();
  _state._selectedIdx = -1;
  _renderAnnotations(_state.currentPage);
  _renderDrawStrokes(_state.currentPage);
  _markDirty();
}

function _markDirty() {
  _state.dirty = true;
  _state._onDirty?.();
}

// ---------------------------------------------------------------------------
// UI Construction
// ---------------------------------------------------------------------------
function _buildUI(container) {
  _dom.container = container;
  container.innerHTML = '';
  container.className = (container.className || '').replace(/pdf-editor-host/g, '') + ' pdf-editor-host';

  container.innerHTML = `
    <div class="pdf-editor">
      <div class="pdf-editor__toolbar" id="pdf-toolbar">
        <div class="pdf-editor__tool-group">
          <button class="pdf-editor__tool active" data-tool="select" title="Select (V)">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3l7.07 16.97 2.51-7.39 7.39-2.51L3 3z"/></svg>
          </button>
          <button class="pdf-editor__tool" data-tool="text" title="Text (T)">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><polyline points="4 7 4 4 20 4 20 7"/><line x1="9" y1="20" x2="15" y2="20"/><line x1="12" y1="4" x2="12" y2="20"/></svg>
          </button>
          <button class="pdf-editor__tool" data-tool="highlight" title="Highlight (H)">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="10" width="18" height="6" rx="1" fill="rgba(255,255,0,0.3)"/><line x1="3" y1="20" x2="21" y2="20"/></svg>
          </button>
          <button class="pdf-editor__tool" data-tool="draw" title="Draw (D)">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 3a2.83 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/></svg>
          </button>
          <button class="pdf-editor__tool" data-tool="shape" title="Shapes (S)">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>
          </button>
          <button class="pdf-editor__tool" data-tool="image" title="Insert Image (I)">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
          </button>
          <button class="pdf-editor__tool" data-tool="signature" title="Signature (G)">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 19c3-2 5-6 7-6s2 4 4 4 3-5 5-5 2 3 4 3"/><line x1="2" y1="22" x2="22" y2="22"/></svg>
          </button>
        </div>

        <div class="pdf-editor__shape-picker hidden" id="pdf-shape-picker">
          <button class="pdf-editor__shape-btn active" data-shape="rect" title="Rectangle">
            <svg viewBox="0 0 20 20" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="4" width="16" height="12" rx="1"/></svg>
          </button>
          <button class="pdf-editor__shape-btn" data-shape="circle" title="Ellipse">
            <svg viewBox="0 0 20 20" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="10" cy="10" rx="8" ry="6"/></svg>
          </button>
          <button class="pdf-editor__shape-btn" data-shape="line" title="Line">
            <svg viewBox="0 0 20 20" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><line x1="2" y1="18" x2="18" y2="2"/></svg>
          </button>
          <button class="pdf-editor__shape-btn" data-shape="arrow" title="Arrow">
            <svg viewBox="0 0 20 20" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><line x1="2" y1="18" x2="18" y2="2"/><polyline points="10 2 18 2 18 10"/></svg>
          </button>
        </div>

        <div class="pdf-editor__tool-options">
          <input type="color" class="pdf-editor__color-picker" id="pdf-color" value="#000000" title="Color">
          <select class="pdf-editor__font-size" id="pdf-font-size" title="Font size">
            <option value="10">10</option>
            <option value="12">12</option>
            <option value="14" selected>14</option>
            <option value="18">18</option>
            <option value="24">24</option>
            <option value="36">36</option>
          </select>
          <select class="pdf-editor__stroke-width" id="pdf-stroke-width" title="Stroke width">
            <option value="1">1px</option>
            <option value="2" selected>2px</option>
            <option value="3">3px</option>
            <option value="5">5px</option>
            <option value="8">8px</option>
          </select>
        </div>

        <div class="pdf-editor__tool-group">
          <button class="pdf-editor__action-btn" id="pdf-undo" title="Undo (Ctrl+Z)">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>
          </button>
          <button class="pdf-editor__action-btn" id="pdf-redo" title="Redo (Ctrl+Shift+Z)">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
          </button>
        </div>

        <div class="pdf-editor__tool-group">
          <button class="pdf-editor__action-btn" id="pdf-search-btn" title="Find (Ctrl+F)">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          </button>
          <button class="pdf-editor__action-btn" id="pdf-print-btn" title="Print (Ctrl+P)">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 6 2 18 2 18 9"/><path d="M6 18H4a2 2 0 01-2-2v-5a2 2 0 012-2h16a2 2 0 012 2v5a2 2 0 01-2 2h-2"/><rect x="6" y="14" width="12" height="8"/></svg>
          </button>
          <button class="pdf-editor__action-btn" id="pdf-merge-btn" title="Merge another PDF">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="3" width="8" height="13" rx="1"/><rect x="14" y="8" width="8" height="13" rx="1"/><path d="M10 10h4"/><polyline points="12 8 14 10 12 12"/></svg>
          </button>
          <button class="pdf-editor__action-btn" id="pdf-bookmarks-btn" title="Bookmarks">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/></svg>
          </button>
        </div>

        <div class="pdf-editor__page-controls">
          <button class="pdf-editor__nav-btn pdf-editor__page-action" id="pdf-rotate-btn" title="Rotate page">
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>
          </button>
          <button class="pdf-editor__nav-btn pdf-editor__page-action" id="pdf-delete-page-btn" title="Delete page">
            <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 01-2 2H8a2 2 0 01-2-2L5 6m5 0V4a1 1 0 011-1h2a1 1 0 011 1v2"/></svg>
          </button>
          <span class="pdf-editor__page-sep"></span>
          <button class="pdf-editor__nav-btn" id="pdf-prev" title="Previous page">&lsaquo;</button>
          <span class="pdf-editor__page-label" id="pdf-page-label">1 / 1</span>
          <button class="pdf-editor__nav-btn" id="pdf-next" title="Next page">&rsaquo;</button>
        </div>

        <div class="pdf-editor__zoom-controls">
          <button class="pdf-editor__nav-btn" id="pdf-zoom-out">&minus;</button>
          <span class="pdf-editor__zoom-label" id="pdf-zoom-label">100%</span>
          <button class="pdf-editor__nav-btn" id="pdf-zoom-in">+</button>
        </div>
      </div>

      <div class="pdf-editor__search-bar hidden" id="pdf-search-bar">
        <input type="text" class="pdf-editor__search-input" id="pdf-search-input"
               placeholder="Find in document\u2026" autocomplete="off" spellcheck="false">
        <span class="pdf-editor__search-count" id="pdf-search-count"></span>
        <button class="pdf-editor__search-nav" id="pdf-search-prev" title="Previous">&lsaquo;</button>
        <button class="pdf-editor__search-nav" id="pdf-search-next" title="Next">&rsaquo;</button>
        <button class="pdf-editor__search-close" id="pdf-search-close">&times;</button>
      </div>

      <div class="pdf-editor__body">
        <div class="pdf-editor__pages" id="pdf-page-nav"></div>
        <div class="pdf-editor__bookmarks hidden" id="pdf-bookmarks">
          <div class="pdf-editor__bookmarks-header">Bookmarks</div>
          <div class="pdf-editor__bookmarks-list" id="pdf-bookmarks-list"></div>
        </div>
        <div class="pdf-editor__viewport" id="pdf-viewport">
          <div class="pdf-editor__canvas-wrap">
            <canvas id="pdf-canvas"></canvas>
            <div class="pdf-editor__text-layer" id="pdf-text-layer"></div>
            <canvas id="pdf-draw-canvas" class="pdf-editor__draw-canvas"></canvas>
            <div class="pdf-editor__overlay" id="pdf-overlay"></div>
          </div>
        </div>
      </div>

      <div class="pdf-editor__sig-modal hidden" id="pdf-sig-modal">
        <div class="pdf-editor__sig-dialog">
          <div class="pdf-editor__sig-header">
            <span>Draw your signature</span>
            <button class="pdf-editor__sig-close" id="pdf-sig-close">&times;</button>
          </div>
          <canvas class="pdf-editor__sig-canvas" id="pdf-sig-canvas" width="400" height="160"></canvas>
          <div class="pdf-editor__sig-actions">
            <button class="pdf-editor__sig-btn" id="pdf-sig-clear">Clear</button>
            <button class="pdf-editor__sig-btn pdf-editor__sig-btn--primary" id="pdf-sig-apply">Apply</button>
          </div>
        </div>
      </div>
    </div>
  `;

  // Cache DOM refs
  _dom.canvas = container.querySelector('#pdf-canvas');
  _dom.drawCanvas = container.querySelector('#pdf-draw-canvas');
  _dom.overlay = container.querySelector('#pdf-overlay');
  _dom.textLayer = container.querySelector('#pdf-text-layer');
  _dom.pageNav = container.querySelector('#pdf-page-nav');
  _dom.pageLabel = container.querySelector('#pdf-page-label');
  _dom.toolbar = container.querySelector('#pdf-toolbar');
  _dom.colorPicker = container.querySelector('#pdf-color');
  _dom.shapePicker = container.querySelector('#pdf-shape-picker');
  _dom.searchBar = container.querySelector('#pdf-search-bar');
  _dom.searchInput = container.querySelector('#pdf-search-input');
  _dom.searchCount = container.querySelector('#pdf-search-count');
  _dom.sigModal = container.querySelector('#pdf-sig-modal');
  _dom.sigCanvas = container.querySelector('#pdf-sig-canvas');

  // Tool selection
  _dom.toolbar.querySelectorAll('.pdf-editor__tool').forEach(btn => {
    btn.addEventListener('click', () => _setTool(btn.dataset.tool));
  });

  // Shape sub-picker
  _dom.shapePicker?.querySelectorAll('.pdf-editor__shape-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      _state.activeShape = btn.dataset.shape;
      _dom.shapePicker.querySelectorAll('.pdf-editor__shape-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  // Color + font size + stroke width
  _dom.colorPicker?.addEventListener('input', e => { _state.activeColor = e.target.value; });
  container.querySelector('#pdf-font-size')?.addEventListener('change', e => {
    _state.activeFontSize = parseInt(e.target.value, 10) || 14;
  });
  container.querySelector('#pdf-stroke-width')?.addEventListener('change', e => {
    _state.activeStrokeWidth = parseInt(e.target.value, 10) || 2;
  });

  // Undo/Redo
  container.querySelector('#pdf-undo')?.addEventListener('click', _undo);
  container.querySelector('#pdf-redo')?.addEventListener('click', _redo);

  // Search
  container.querySelector('#pdf-search-btn')?.addEventListener('click', _toggleSearch);
  _dom.searchInput?.addEventListener('input', _doSearch);
  container.querySelector('#pdf-search-prev')?.addEventListener('click', () => _navigateSearch(-1));
  container.querySelector('#pdf-search-next')?.addEventListener('click', () => _navigateSearch(1));
  container.querySelector('#pdf-search-close')?.addEventListener('click', _closeSearch);

  // Print
  container.querySelector('#pdf-print-btn')?.addEventListener('click', _printPdf);

  // Merge
  container.querySelector('#pdf-merge-btn')?.addEventListener('click', _openMergeFilePicker);
  const mergeInput = document.createElement('input');
  mergeInput.type = 'file';
  mergeInput.accept = 'application/pdf';
  mergeInput.multiple = true;
  mergeInput.className = 'hidden';
  mergeInput.id = 'pdf-merge-input';
  mergeInput.addEventListener('change', _handleMergeFiles);
  container.appendChild(mergeInput);

  // Bookmarks
  _dom.bookmarksPanel = container.querySelector('#pdf-bookmarks');
  _dom.bookmarksList = container.querySelector('#pdf-bookmarks-list');
  container.querySelector('#pdf-bookmarks-btn')?.addEventListener('click', _toggleBookmarks);

  // Page navigation
  container.querySelector('#pdf-prev')?.addEventListener('click', () => _goToPage(_state.currentPage - 1));
  container.querySelector('#pdf-next')?.addEventListener('click', () => _goToPage(_state.currentPage + 1));

  // Page manipulation
  container.querySelector('#pdf-rotate-btn')?.addEventListener('click', _rotatePage);
  container.querySelector('#pdf-delete-page-btn')?.addEventListener('click', _deletePage);

  // Zoom
  container.querySelector('#pdf-zoom-in')?.addEventListener('click', () => _setZoom(_state.scale + 0.25));
  container.querySelector('#pdf-zoom-out')?.addEventListener('click', () => _setZoom(_state.scale - 0.25));

  // Overlay interaction
  _dom.overlay.addEventListener('mousedown', _handleOverlayMouseDown);
  _dom.overlay.addEventListener('mousemove', _handleOverlayMouseMove);
  _dom.overlay.addEventListener('mouseup', _handleOverlayMouseUp);
  _dom.overlay.addEventListener('click', _handleOverlayClick);

  // Touch support for mobile
  _dom.overlay.addEventListener('touchstart', _handleTouchStart, { passive: false });
  _dom.overlay.addEventListener('touchmove', _handleTouchMove, { passive: false });
  _dom.overlay.addEventListener('touchend', _handleTouchEnd, { passive: false });

  // Signature modal
  _initSignaturePad();
  container.querySelector('#pdf-sig-close')?.addEventListener('click', _closeSignatureModal);
  container.querySelector('#pdf-sig-clear')?.addEventListener('click', _clearSignaturePad);
  container.querySelector('#pdf-sig-apply')?.addEventListener('click', _applySignature);

  // Hidden file input for image tool
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = 'image/*';
  fileInput.className = 'hidden';
  fileInput.id = 'pdf-image-input';
  fileInput.addEventListener('change', _handleImageUpload);
  container.appendChild(fileInput);

  // Keyboard shortcuts
  container.setAttribute('tabindex', '0');
  container.addEventListener('keydown', _handleKeydown);
}

function _setTool(tool) {
  _state.activeTool = tool;
  _state._selectedIdx = -1;
  _renderAnnotations(_state.currentPage);
  _dom.toolbar?.querySelectorAll('.pdf-editor__tool').forEach(b => b.classList.remove('active'));
  _dom.toolbar?.querySelector(`[data-tool="${tool}"]`)?.classList.add('active');
  if (_dom.overlay) _dom.overlay.dataset.tool = tool;

  // Show/hide shape picker
  _dom.shapePicker?.classList.toggle('hidden', tool !== 'shape');

  // Show/hide text layer based on tool (text layer enables selection in select mode)
  if (_dom.textLayer) {
    _dom.textLayer.classList.toggle('pdf-editor__text-layer--active', tool === 'select');
  }

  // Image tool triggers file picker immediately
  if (tool === 'image') {
    _dom.container?.querySelector('#pdf-image-input')?.click();
    setTimeout(() => _setTool('select'), 100);
  }

  // Signature tool opens modal
  if (tool === 'signature') {
    _openSignatureModal();
  }
}

// ---------------------------------------------------------------------------
// Page rendering
// ---------------------------------------------------------------------------
async function _renderPage(pageNum) {
  if (!_state.pdfDoc || pageNum < 1 || pageNum > _state.totalPages) return;
  _state.currentPage = pageNum;

  const page = await _state.pdfDoc.getPage(pageNum);
  const rotation = _state._pageRotations[pageNum] || 0;
  const viewport = page.getViewport({ scale: _state.scale * 2, rotation });
  const displayW = viewport.width / 2;
  const displayH = viewport.height / 2;

  // Main canvas (PDF render)
  _dom.canvas.width = viewport.width;
  _dom.canvas.height = viewport.height;
  _dom.canvas.style.width = `${displayW}px`;
  _dom.canvas.style.height = `${displayH}px`;

  // Draw canvas (freehand ink)
  _dom.drawCanvas.width = displayW;
  _dom.drawCanvas.height = displayH;
  _dom.drawCanvas.style.width = `${displayW}px`;
  _dom.drawCanvas.style.height = `${displayH}px`;

  // Overlay + text layer
  _dom.overlay.style.width = `${displayW}px`;
  _dom.overlay.style.height = `${displayH}px`;
  _dom.textLayer.style.width = `${displayW}px`;
  _dom.textLayer.style.height = `${displayH}px`;

  const ctx = _dom.canvas.getContext('2d');
  await page.render({ canvasContext: ctx, viewport }).promise;

  // Render text layer for selection/search
  await _renderTextLayer(page, viewport, displayW, displayH);

  if (_dom.pageLabel) _dom.pageLabel.textContent = `${pageNum} / ${_state.totalPages}`;

  _renderAnnotations(pageNum);
  _renderDrawStrokes(pageNum);

  _dom.pageNav?.querySelectorAll('.pdf-editor__thumb').forEach(t => {
    t.classList.toggle('active', parseInt(t.dataset.page, 10) === pageNum);
  });
}

async function _renderTextLayer(page, viewport, displayW, displayH) {
  if (!_dom.textLayer) return;
  _dom.textLayer.innerHTML = '';

  try {
    const textContent = await page.getTextContent();
    const textLayerViewport = page.getViewport({ scale: _state.scale, rotation: _state._pageRotations[_state.currentPage] || 0 });

    // pdf.js text layer API
    const pdfjsLib = await _ensurePdfJs();
    if (pdfjsLib.renderTextLayer) {
      const task = pdfjsLib.renderTextLayer({
        textContentSource: textContent,
        container: _dom.textLayer,
        viewport: textLayerViewport,
      });
      await task.promise;
    }
  } catch (err) {
    // Text layer is optional — degrade silently
    console.warn('[pdf-editor] Text layer render failed:', err);
  }
}

// ---------------------------------------------------------------------------
// Page thumbnails (with drag reorder)
// ---------------------------------------------------------------------------
async function _renderPageNav() {
  if (!_dom.pageNav || !_state.pdfDoc) return;
  _dom.pageNav.innerHTML = '';
  for (let i = 1; i <= _state.totalPages; i++) {
    const page = await _state.pdfDoc.getPage(i);
    const rotation = _state._pageRotations[i] || 0;
    const vp = page.getViewport({ scale: 0.2, rotation });
    const thumb = document.createElement('div');
    thumb.className = `pdf-editor__thumb${i === _state.currentPage ? ' active' : ''}`;
    thumb.dataset.page = i;
    thumb.draggable = true;

    const c = document.createElement('canvas');
    c.width = vp.width; c.height = vp.height;
    c.style.width = `${vp.width}px`; c.style.height = `${vp.height}px`;
    await page.render({ canvasContext: c.getContext('2d'), viewport: vp }).promise;

    const lbl = document.createElement('span');
    lbl.className = 'pdf-editor__thumb-label';
    lbl.textContent = i;
    thumb.appendChild(c); thumb.appendChild(lbl);
    thumb.addEventListener('click', () => _goToPage(i));

    // Drag reorder
    thumb.addEventListener('dragstart', e => {
      e.dataTransfer.setData('text/plain', String(i));
      thumb.classList.add('dragging');
    });
    thumb.addEventListener('dragend', () => thumb.classList.remove('dragging'));
    thumb.addEventListener('dragover', e => { e.preventDefault(); thumb.classList.add('drag-over'); });
    thumb.addEventListener('dragleave', () => thumb.classList.remove('drag-over'));
    thumb.addEventListener('drop', e => {
      e.preventDefault();
      thumb.classList.remove('drag-over');
      const fromPage = parseInt(e.dataTransfer.getData('text/plain'), 10);
      const toPage = i;
      if (fromPage !== toPage) _reorderPage(fromPage, toPage);
    });

    _dom.pageNav.appendChild(thumb);
  }
}

function _goToPage(n) { if (n >= 1 && n <= _state.totalPages) _renderPage(n); }

function _setZoom(s) {
  _state.scale = Math.max(0.5, Math.min(3.0, s));
  const lbl = _dom.container?.querySelector('#pdf-zoom-label');
  if (lbl) lbl.textContent = `${Math.round(_state.scale * 100)}%`;
  _renderPage(_state.currentPage);
}

// ---------------------------------------------------------------------------
// Page manipulation
// ---------------------------------------------------------------------------
async function _rotatePage() {
  const p = _state.currentPage;
  const current = _state._pageRotations[p] || 0;
  _state._pageRotations[p] = (current + 90) % 360;

  // Also rotate in pdf-lib
  if (_state.pdfLibDoc) {
    const page = _state.pdfLibDoc.getPages()[p - 1];
    if (page) page.setRotation(window.PDFLib.degrees((_state._pageRotations[p])));
  }

  _markDirty();
  await _renderPage(p);
  _renderPageNav();
}

async function _deletePage() {
  if (_state.totalPages <= 1) return; // Can't delete the only page
  if (!_state.pdfLibDoc) return;

  const p = _state.currentPage;
  _state.pdfLibDoc.removePage(p - 1);

  // Remove annotations for this page, shift page numbers for later pages
  _state.annotations = _state.annotations
    .filter(a => a.page !== p)
    .map(a => a.page > p ? { ...a, page: a.page - 1 } : a);

  // Remove rotation entry
  const newRotations = {};
  for (const [pg, rot] of Object.entries(_state._pageRotations)) {
    const n = parseInt(pg, 10);
    if (n < p) newRotations[n] = rot;
    else if (n > p) newRotations[n - 1] = rot;
  }
  _state._pageRotations = newRotations;

  _markDirty();
  await _refreshPdfJsDoc();
  if (_state.currentPage > _state.totalPages) _state.currentPage = _state.totalPages;
  await _renderPage(_state.currentPage);
  await _renderPageNav();
}

async function _reorderPage(fromPage, toPage) {
  if (!_state.pdfLibDoc || fromPage === toPage) return;
  if (fromPage < 1 || fromPage > _state.totalPages) return;
  if (toPage < 1 || toPage > _state.totalPages) return;

  // pdf-lib doesn't have a movePage method, so we copy pages into a new doc
  const PDFDoc = await _ensurePdfLib();
  const bytes = await _state.pdfLibDoc.save();
  const srcDoc = await PDFDoc.load(bytes);
  const newDoc = await PDFDoc.create();

  const pageOrder = [];
  for (let i = 1; i <= _state.totalPages; i++) pageOrder.push(i);
  // Remove from position, insert at new position
  pageOrder.splice(fromPage - 1, 1);
  pageOrder.splice(toPage - 1, 0, fromPage);

  for (const pg of pageOrder) {
    const [copied] = await newDoc.copyPages(srcDoc, [pg - 1]);
    newDoc.addPage(copied);
  }

  _state.pdfLibDoc = newDoc;

  // Remap annotations
  const pageMap = {};
  pageOrder.forEach((origPage, idx) => { pageMap[origPage] = idx + 1; });
  _state.annotations = _state.annotations.map(a => ({ ...a, page: pageMap[a.page] || a.page }));

  // Remap rotations
  const newRotations = {};
  for (const [pg, rot] of Object.entries(_state._pageRotations)) {
    const n = parseInt(pg, 10);
    if (pageMap[n]) newRotations[pageMap[n]] = rot;
  }
  _state._pageRotations = newRotations;

  _markDirty();
  await _refreshPdfJsDoc();
  _state.currentPage = toPage;
  await _renderPage(_state.currentPage);
  await _renderPageNav();
}

// ---------------------------------------------------------------------------
// Overlay interaction handlers
// ---------------------------------------------------------------------------
function _handleOverlayClick(e) {
  if (_state.activeTool === 'text') {
    _startTextInput(e);
  }
  if (_state.activeTool === 'select') {
    _trySelectAnnotation(e);
  }
}

function _handleOverlayMouseDown(e) {
  const rect = _dom.overlay.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  if (_state.activeTool === 'draw') {
    _state._drawing = true;
    _state._drawPoints = [{ x, y }];
    e.preventDefault();
  }
  if (_state.activeTool === 'highlight') {
    _state._highlightStart = { x, y };
    e.preventDefault();
  }
  if (_state.activeTool === 'shape') {
    _state._shapeStart = { x, y };
    e.preventDefault();
  }
}

function _handleOverlayMouseMove(e) {
  const rect = _dom.overlay.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  if (_state.activeTool === 'draw' && _state._drawing) {
    _state._drawPoints.push({ x, y });
    _drawLiveStroke();
  }
  if (_state.activeTool === 'highlight' && _state._highlightStart) {
    _drawLiveHighlight(_state._highlightStart.x, _state._highlightStart.y, x, y);
  }
  if (_state.activeTool === 'shape' && _state._shapeStart) {
    _drawLiveShape(_state._shapeStart.x, _state._shapeStart.y, x, y);
  }
}

function _handleOverlayMouseUp(e) {
  const rect = _dom.overlay.getBoundingClientRect();
  const endX = e.clientX - rect.left;
  const endY = e.clientY - rect.top;

  const canvasW = parseFloat(_dom.canvas.style.width);
  const canvasH = parseFloat(_dom.canvas.style.height);

  if (_state.activeTool === 'draw' && _state._drawing) {
    _state._drawing = false;
    if (_state._drawPoints.length > 2) {
      _pushUndo();
      _state.annotations.push({
        page: _state.currentPage, type: 'draw',
        points: [..._state._drawPoints],
        color: _state.activeColor,
        width: _state.activeStrokeWidth,
        _canvasW: canvasW, _canvasH: canvasH,
      });
      _markDirty();
    }
    _state._drawPoints = [];
    _renderDrawStrokes(_state.currentPage);
  }

  if (_state.activeTool === 'highlight' && _state._highlightStart) {
    const sx = _state._highlightStart.x;
    const sy = _state._highlightStart.y;
    const w = Math.abs(endX - sx);
    const h = Math.abs(endY - sy);
    if (w > 5 && h > 3) {
      _pushUndo();
      _state.annotations.push({
        page: _state.currentPage, type: 'highlight',
        x: Math.min(sx, endX), y: Math.min(sy, endY), w, h,
        color: _state.activeColor === '#000000' ? '#ffff00' : _state.activeColor,
        _canvasW: canvasW, _canvasH: canvasH,
      });
      _markDirty();
      _renderAnnotations(_state.currentPage);
    }
    _state._highlightStart = null;
    _dom.overlay.querySelector('.pdf-editor__live-highlight')?.remove();
  }

  if (_state.activeTool === 'shape' && _state._shapeStart) {
    const sx = _state._shapeStart.x;
    const sy = _state._shapeStart.y;
    const w = Math.abs(endX - sx);
    const h = Math.abs(endY - sy);
    if (w > 5 || h > 5) {
      _pushUndo();
      _state.annotations.push({
        page: _state.currentPage, type: 'shape',
        shape: _state.activeShape,
        x: Math.min(sx, endX), y: Math.min(sy, endY), w, h,
        // For line/arrow, store actual start/end (not min/max)
        x1: sx, y1: sy, x2: endX, y2: endY,
        color: _state.activeColor,
        strokeWidth: _state.activeStrokeWidth,
        _canvasW: canvasW, _canvasH: canvasH,
      });
      _markDirty();
      _renderAnnotations(_state.currentPage);
    }
    _state._shapeStart = null;
    _dom.overlay.querySelector('.pdf-editor__live-shape')?.remove();
  }
}

// Touch handlers — map to mouse events
function _handleTouchStart(e) {
  if (e.touches.length !== 1) return;
  e.preventDefault();
  const t = e.touches[0];
  _handleOverlayMouseDown({ clientX: t.clientX, clientY: t.clientY, preventDefault() {} });
}

function _handleTouchMove(e) {
  if (e.touches.length !== 1) return;
  e.preventDefault();
  const t = e.touches[0];
  _handleOverlayMouseMove({ clientX: t.clientX, clientY: t.clientY });
}

function _handleTouchEnd(e) {
  e.preventDefault();
  const t = e.changedTouches[0];
  _handleOverlayMouseUp({ clientX: t.clientX, clientY: t.clientY });
}

// ---------------------------------------------------------------------------
// Annotation selection + deletion
// ---------------------------------------------------------------------------
function _trySelectAnnotation(e) {
  const rect = _dom.overlay.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  const hitRadius = 10;

  // Check annotations in reverse order (topmost first)
  const pageAnns = _state.annotations
    .map((a, idx) => ({ ...a, _idx: idx }))
    .filter(a => a.page === _state.currentPage);

  let hitIdx = -1;
  for (let i = pageAnns.length - 1; i >= 0; i--) {
    const a = pageAnns[i];
    if (_hitTestAnnotation(a, x, y, hitRadius)) {
      hitIdx = a._idx;
      break;
    }
  }

  _state._selectedIdx = hitIdx;
  _renderAnnotations(_state.currentPage);
  _renderDrawStrokes(_state.currentPage);
}

function _hitTestAnnotation(ann, x, y, r) {
  if (ann.type === 'text') {
    return x >= ann.x - r && x <= ann.x + 200 && y >= ann.y - r && y <= ann.y + (ann.fontSize || 14) + r;
  }
  if (ann.type === 'highlight' || (ann.type === 'shape' && (ann.shape === 'rect' || ann.shape === 'circle'))) {
    return x >= ann.x - r && x <= ann.x + ann.w + r && y >= ann.y - r && y <= ann.y + ann.h + r;
  }
  if (ann.type === 'shape' && (ann.shape === 'line' || ann.shape === 'arrow')) {
    // Distance from point to line segment
    return _pointToSegmentDist(x, y, ann.x1, ann.y1, ann.x2, ann.y2) < r + 5;
  }
  if (ann.type === 'draw' && ann.points?.length >= 2) {
    for (let i = 0; i < ann.points.length - 1; i++) {
      if (_pointToSegmentDist(x, y, ann.points[i].x, ann.points[i].y, ann.points[i + 1].x, ann.points[i + 1].y) < r) {
        return true;
      }
    }
  }
  if (ann.type === 'signature') {
    return x >= ann.x - r && x <= ann.x + (ann.w || 100) + r && y >= ann.y - r && y <= ann.y + (ann.h || 40) + r;
  }
  return false;
}

function _pointToSegmentDist(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lenSq = dx * dx + dy * dy;
  if (lenSq === 0) return Math.hypot(px - x1, py - y1);
  let t = ((px - x1) * dx + (py - y1) * dy) / lenSq;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
}

function _deleteSelectedAnnotation() {
  if (_state._selectedIdx < 0 || _state._selectedIdx >= _state.annotations.length) return;
  _pushUndo();
  _state.annotations.splice(_state._selectedIdx, 1);
  _state._selectedIdx = -1;
  _markDirty();
  _renderAnnotations(_state.currentPage);
  _renderDrawStrokes(_state.currentPage);
}

// ---------------------------------------------------------------------------
// Text tool
// ---------------------------------------------------------------------------
function _startTextInput(e) {
  const rect = _dom.overlay.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  const input = document.createElement('textarea');
  input.className = 'pdf-editor__text-input';
  input.style.left = `${x}px`;
  input.style.top = `${y}px`;
  input.style.fontSize = `${_state.activeFontSize}px`;
  input.style.color = _state.activeColor;
  input.placeholder = 'Type here...';
  _dom.overlay.appendChild(input);
  input.focus();

  const commit = () => {
    const text = input.value.trim();
    if (text) {
      _pushUndo();
      _addTextAnnotation(_state.currentPage, x, y, text);
    }
    input.remove();
  };
  input.addEventListener('blur', commit);
  input.addEventListener('keydown', ke => {
    if (ke.key === 'Enter' && !ke.shiftKey) { ke.preventDefault(); commit(); }
    if (ke.key === 'Escape') input.remove();
  });
}

async function _addTextAnnotation(pageNum, x, y, text) {
  if (!_state.pdfLibDoc) return;

  const pages = _state.pdfLibDoc.getPages();
  const page = pages[pageNum - 1];
  if (!page) return;

  const { width, height } = page.getSize();
  const canvasW = parseFloat(_dom.canvas.style.width);
  const canvasH = parseFloat(_dom.canvas.style.height);
  const pdfX = (x / canvasW) * width;
  const pdfY = height - (y / canvasH) * height;

  const font = await _state.pdfLibDoc.embedFont(window.PDFLib.StandardFonts.Helvetica);
  const { r, g, b } = _hexToRgb(_state.activeColor);

  page.drawText(text, {
    x: pdfX, y: pdfY,
    size: _state.activeFontSize,
    font,
    color: window.PDFLib.rgb(r, g, b),
  });

  _state.annotations.push({
    page: pageNum, type: 'text', x, y, text,
    color: _state.activeColor, fontSize: _state.activeFontSize,
    _canvasW: canvasW, _canvasH: canvasH,
  });
  _markDirty();
  _renderAnnotations(pageNum);
}

// ---------------------------------------------------------------------------
// Draw tool — freehand ink
// ---------------------------------------------------------------------------
function _drawLiveStroke() {
  if (!_dom.drawCanvas) return;
  const ctx = _dom.drawCanvas.getContext('2d');
  const pts = _state._drawPoints;
  if (pts.length < 2) return;

  _renderDrawStrokes(_state.currentPage);

  ctx.strokeStyle = _state.activeColor;
  ctx.lineWidth = _state.activeStrokeWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.stroke();
}

function _renderDrawStrokes(pageNum) {
  if (!_dom.drawCanvas) return;
  const ctx = _dom.drawCanvas.getContext('2d');
  ctx.clearRect(0, 0, _dom.drawCanvas.width, _dom.drawCanvas.height);

  const strokes = _state.annotations.filter(a => a.page === pageNum && a.type === 'draw');
  for (const s of strokes) {
    if (s.points.length < 2) continue;
    const isSelected = _state.annotations.indexOf(s) === _state._selectedIdx;
    ctx.strokeStyle = s.color || '#000';
    ctx.lineWidth = s.width || 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (isSelected) {
      ctx.shadowColor = 'var(--accent, #4a9eff)';
      ctx.shadowBlur = 6;
    }
    ctx.beginPath();
    ctx.moveTo(s.points[0].x, s.points[0].y);
    for (let i = 1; i < s.points.length; i++) ctx.lineTo(s.points[i].x, s.points[i].y);
    ctx.stroke();
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;
  }
}

// ---------------------------------------------------------------------------
// Highlight tool — rectangular highlight
// ---------------------------------------------------------------------------
function _drawLiveHighlight(sx, sy, ex, ey) {
  let el = _dom.overlay.querySelector('.pdf-editor__live-highlight');
  if (!el) {
    el = document.createElement('div');
    el.className = 'pdf-editor__live-highlight';
    _dom.overlay.appendChild(el);
  }
  const x = Math.min(sx, ex);
  const y = Math.min(sy, ey);
  el.style.left = `${x}px`;
  el.style.top = `${y}px`;
  el.style.width = `${Math.abs(ex - sx)}px`;
  el.style.height = `${Math.abs(ey - sy)}px`;
  el.style.background = (_state.activeColor === '#000000' ? 'rgba(255,255,0,0.25)' : _hexToRgba(_state.activeColor, 0.25));
}

// ---------------------------------------------------------------------------
// Shape tool — rect, circle, line, arrow
// ---------------------------------------------------------------------------
function _drawLiveShape(sx, sy, ex, ey) {
  let svg = _dom.overlay.querySelector('.pdf-editor__live-shape');
  if (!svg) {
    svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.classList.add('pdf-editor__live-shape');
    svg.style.position = 'absolute';
    svg.style.top = '0';
    svg.style.left = '0';
    svg.style.width = '100%';
    svg.style.height = '100%';
    svg.style.pointerEvents = 'none';
    svg.style.zIndex = '5';
    _dom.overlay.appendChild(svg);
  }
  svg.innerHTML = '';

  const color = _state.activeColor;
  const sw = _state.activeStrokeWidth;
  const shape = _state.activeShape;

  if (shape === 'rect') {
    const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    r.setAttribute('x', Math.min(sx, ex));
    r.setAttribute('y', Math.min(sy, ey));
    r.setAttribute('width', Math.abs(ex - sx));
    r.setAttribute('height', Math.abs(ey - sy));
    r.setAttribute('stroke', color);
    r.setAttribute('stroke-width', sw);
    r.setAttribute('fill', 'none');
    r.setAttribute('rx', '2');
    svg.appendChild(r);
  } else if (shape === 'circle') {
    const el = document.createElementNS('http://www.w3.org/2000/svg', 'ellipse');
    el.setAttribute('cx', (sx + ex) / 2);
    el.setAttribute('cy', (sy + ey) / 2);
    el.setAttribute('rx', Math.abs(ex - sx) / 2);
    el.setAttribute('ry', Math.abs(ey - sy) / 2);
    el.setAttribute('stroke', color);
    el.setAttribute('stroke-width', sw);
    el.setAttribute('fill', 'none');
    svg.appendChild(el);
  } else if (shape === 'line') {
    const l = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    l.setAttribute('x1', sx); l.setAttribute('y1', sy);
    l.setAttribute('x2', ex); l.setAttribute('y2', ey);
    l.setAttribute('stroke', color);
    l.setAttribute('stroke-width', sw);
    svg.appendChild(l);
  } else if (shape === 'arrow') {
    const l = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    l.setAttribute('x1', sx); l.setAttribute('y1', sy);
    l.setAttribute('x2', ex); l.setAttribute('y2', ey);
    l.setAttribute('stroke', color);
    l.setAttribute('stroke-width', sw);
    svg.appendChild(l);
    // Arrowhead
    const angle = Math.atan2(ey - sy, ex - sx);
    const headLen = sw * 6;
    const a1x = ex - headLen * Math.cos(angle - Math.PI / 6);
    const a1y = ey - headLen * Math.sin(angle - Math.PI / 6);
    const a2x = ex - headLen * Math.cos(angle + Math.PI / 6);
    const a2y = ey - headLen * Math.sin(angle + Math.PI / 6);
    const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
    poly.setAttribute('points', `${ex},${ey} ${a1x},${a1y} ${a2x},${a2y}`);
    poly.setAttribute('fill', color);
    svg.appendChild(poly);
  }
}

// ---------------------------------------------------------------------------
// Image insert
// ---------------------------------------------------------------------------
async function _handleImageUpload(e) {
  const file = e.target.files?.[0];
  e.target.value = '';
  if (!file || !_state.pdfLibDoc) return;

  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    let img;
    if (file.type === 'image/png') {
      img = await _state.pdfLibDoc.embedPng(bytes);
    } else {
      img = await _state.pdfLibDoc.embedJpg(bytes);
    }

    const pages = _state.pdfLibDoc.getPages();
    const page = pages[_state.currentPage - 1];
    if (!page) return;

    const { width: pw, height: ph } = page.getSize();
    const maxW = pw * 0.4;
    const scale = Math.min(maxW / img.width, (ph * 0.4) / img.height, 1);
    const drawW = img.width * scale;
    const drawH = img.height * scale;

    page.drawImage(img, {
      x: (pw - drawW) / 2,
      y: (ph - drawH) / 2,
      width: drawW, height: drawH,
    });

    _pushUndo();
    _state.annotations.push({
      page: _state.currentPage, type: 'image',
      fileName: file.name,
    });
    _markDirty();

    await _refreshPdfJsDoc();
    await _renderPage(_state.currentPage);
  } catch (err) {
    console.warn('[pdf-editor] Image insert failed:', err);
  }
}

// ---------------------------------------------------------------------------
// Signature tool
// ---------------------------------------------------------------------------
function _initSignaturePad() {
  if (!_dom.sigCanvas) return;
  const ctx = _dom.sigCanvas.getContext('2d');
  _state._sigPad = ctx;
  _state._sigPoints = [];
  let drawing = false;

  const getPos = (e) => {
    const rect = _dom.sigCanvas.getBoundingClientRect();
    const scaleX = _dom.sigCanvas.width / rect.width;
    const scaleY = _dom.sigCanvas.height / rect.height;
    if (e.touches) {
      return { x: (e.touches[0].clientX - rect.left) * scaleX, y: (e.touches[0].clientY - rect.top) * scaleY };
    }
    return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY };
  };

  const onStart = (e) => {
    e.preventDefault();
    drawing = true;
    const p = getPos(e);
    _state._sigPoints.push([p]);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
  };

  const onMove = (e) => {
    if (!drawing) return;
    e.preventDefault();
    const p = getPos(e);
    const last = _state._sigPoints[_state._sigPoints.length - 1];
    if (last) last.push(p);
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#000';
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  };

  const onEnd = () => { drawing = false; };

  _dom.sigCanvas.addEventListener('mousedown', onStart);
  _dom.sigCanvas.addEventListener('mousemove', onMove);
  _dom.sigCanvas.addEventListener('mouseup', onEnd);
  _dom.sigCanvas.addEventListener('mouseleave', onEnd);
  _dom.sigCanvas.addEventListener('touchstart', onStart, { passive: false });
  _dom.sigCanvas.addEventListener('touchmove', onMove, { passive: false });
  _dom.sigCanvas.addEventListener('touchend', onEnd);
}

function _openSignatureModal() {
  _clearSignaturePad();
  _dom.sigModal?.classList.remove('hidden');
}

function _closeSignatureModal() {
  _dom.sigModal?.classList.add('hidden');
  _setTool('select');
}

function _clearSignaturePad() {
  if (!_dom.sigCanvas || !_state._sigPad) return;
  _state._sigPad.clearRect(0, 0, _dom.sigCanvas.width, _dom.sigCanvas.height);
  _state._sigPoints = [];
}

async function _applySignature() {
  if (!_state._sigPad || _state._sigPoints.length === 0 || !_state.pdfLibDoc) return;

  try {
    // Convert signature canvas to PNG
    const dataUrl = _dom.sigCanvas.toDataURL('image/png');
    const resp = await fetch(dataUrl);
    const blob = await resp.blob();
    const bytes = new Uint8Array(await blob.arrayBuffer());

    const img = await _state.pdfLibDoc.embedPng(bytes);
    const pages = _state.pdfLibDoc.getPages();
    const page = pages[_state.currentPage - 1];
    if (!page) return;

    const { width: pw, height: ph } = page.getSize();
    const maxW = pw * 0.3;
    const scale = Math.min(maxW / img.width, (ph * 0.15) / img.height, 1);
    const drawW = img.width * scale;
    const drawH = img.height * scale;

    // Place signature at bottom-center of page
    page.drawImage(img, {
      x: (pw - drawW) / 2,
      y: ph * 0.08,
      width: drawW, height: drawH,
    });

    _pushUndo();
    const canvasW = parseFloat(_dom.canvas.style.width);
    const canvasH = parseFloat(_dom.canvas.style.height);
    _state.annotations.push({
      page: _state.currentPage, type: 'signature',
      dataUrl,
      x: (canvasW - (drawW / pw) * canvasW) / 2,
      y: canvasH * 0.85,
      w: (drawW / pw) * canvasW,
      h: (drawH / ph) * canvasH,
      _canvasW: canvasW, _canvasH: canvasH,
    });
    _markDirty();

    await _refreshPdfJsDoc();
    await _renderPage(_state.currentPage);
    _closeSignatureModal();
  } catch (err) {
    console.warn('[pdf-editor] Signature apply failed:', err);
  }
}

// ---------------------------------------------------------------------------
// Search / Find
// ---------------------------------------------------------------------------
function _toggleSearch() {
  if (_state._searchOpen) {
    _closeSearch();
  } else {
    _state._searchOpen = true;
    _dom.searchBar?.classList.remove('hidden');
    _dom.searchInput?.focus();
  }
}

function _closeSearch() {
  _state._searchOpen = false;
  _state._searchMatches = [];
  _state._searchIdx = -1;
  _dom.searchBar?.classList.add('hidden');
  if (_dom.searchInput) _dom.searchInput.value = '';
  if (_dom.searchCount) _dom.searchCount.textContent = '';
  // Clear search highlights
  _dom.textLayer?.querySelectorAll('.pdf-editor__search-highlight').forEach(el => el.remove());
}

async function _doSearch() {
  const query = _dom.searchInput?.value?.trim();
  if (!query || !_state.pdfDoc) {
    _state._searchMatches = [];
    _state._searchIdx = -1;
    if (_dom.searchCount) _dom.searchCount.textContent = '';
    return;
  }

  const matches = [];
  const lowerQuery = query.toLowerCase();

  for (let p = 1; p <= _state.totalPages; p++) {
    const page = await _state.pdfDoc.getPage(p);
    const textContent = await page.getTextContent();
    const fullText = textContent.items.map(item => item.str).join(' ').toLowerCase();
    let idx = 0;
    while ((idx = fullText.indexOf(lowerQuery, idx)) !== -1) {
      matches.push({ page: p, offset: idx });
      idx += lowerQuery.length;
    }
  }

  _state._searchMatches = matches;
  _state._searchIdx = matches.length > 0 ? 0 : -1;
  _updateSearchDisplay();

  if (matches.length > 0) {
    await _goToPage(matches[0].page);
  }
}

function _navigateSearch(dir) {
  if (_state._searchMatches.length === 0) return;
  _state._searchIdx = (_state._searchIdx + dir + _state._searchMatches.length) % _state._searchMatches.length;
  _updateSearchDisplay();
  const match = _state._searchMatches[_state._searchIdx];
  if (match) _goToPage(match.page);
}

function _updateSearchDisplay() {
  if (!_dom.searchCount) return;
  const total = _state._searchMatches.length;
  if (total === 0) {
    _dom.searchCount.textContent = 'No results';
  } else {
    _dom.searchCount.textContent = `${_state._searchIdx + 1} / ${total}`;
  }
}

// ---------------------------------------------------------------------------
// Print
// ---------------------------------------------------------------------------
async function _printPdf() {
  if (!_state.pdfDoc) return;

  // Embed annotations before printing
  _embedAnnotationsIntoPdf();

  const bytes = await _state.pdfLibDoc.save();
  const blob = new Blob([bytes], { type: 'application/pdf' });
  const url = URL.createObjectURL(blob);

  const iframe = document.createElement('iframe');
  iframe.style.position = 'fixed';
  iframe.style.left = '-9999px';
  iframe.style.width = '1px';
  iframe.style.height = '1px';
  iframe.src = url;

  iframe.onload = () => {
    try {
      iframe.contentWindow.print();
    } catch {
      // Cross-origin fallback: open in new tab
      window.open(url, '_blank');
    }
    setTimeout(() => {
      iframe.remove();
      URL.revokeObjectURL(url);
    }, 5000);
  };

  document.body.appendChild(iframe);
}

// ---------------------------------------------------------------------------
// Annotation rendering (visual overlay)
// ---------------------------------------------------------------------------
function _renderAnnotations(pageNum) {
  if (!_dom.overlay) return;
  _dom.overlay.querySelectorAll('.pdf-editor__annotation').forEach(el => el.remove());

  const anns = _state.annotations.filter(a => a.page === pageNum);
  for (const ann of anns) {
    const globalIdx = _state.annotations.indexOf(ann);
    const isSelected = globalIdx === _state._selectedIdx;

    if (ann.type === 'text') {
      const el = document.createElement('div');
      el.className = `pdf-editor__annotation pdf-editor__annotation--text${isSelected ? ' selected' : ''}`;
      el.style.left = `${ann.x}px`;
      el.style.top = `${ann.y}px`;
      el.style.color = ann.color || '#000';
      el.style.fontSize = `${ann.fontSize || 14}px`;
      el.textContent = ann.text;
      _dom.overlay.appendChild(el);
    }

    if (ann.type === 'highlight') {
      const el = document.createElement('div');
      el.className = `pdf-editor__annotation pdf-editor__annotation--highlight${isSelected ? ' selected' : ''}`;
      el.style.left = `${ann.x}px`;
      el.style.top = `${ann.y}px`;
      el.style.width = `${ann.w}px`;
      el.style.height = `${ann.h}px`;
      el.style.background = _hexToRgba(ann.color || '#ffff00', 0.25);
      _dom.overlay.appendChild(el);
    }

    if (ann.type === 'shape') {
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.classList.add('pdf-editor__annotation', 'pdf-editor__annotation--shape');
      if (isSelected) svg.classList.add('selected');
      svg.style.position = 'absolute';
      svg.style.left = '0';
      svg.style.top = '0';
      svg.style.width = '100%';
      svg.style.height = '100%';
      svg.style.pointerEvents = 'none';

      const color = ann.color || '#000';
      const sw = ann.strokeWidth || 2;

      if (ann.shape === 'rect') {
        const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        r.setAttribute('x', ann.x); r.setAttribute('y', ann.y);
        r.setAttribute('width', ann.w); r.setAttribute('height', ann.h);
        r.setAttribute('stroke', color); r.setAttribute('stroke-width', sw);
        r.setAttribute('fill', 'none'); r.setAttribute('rx', '2');
        svg.appendChild(r);
      } else if (ann.shape === 'circle') {
        const el = document.createElementNS('http://www.w3.org/2000/svg', 'ellipse');
        el.setAttribute('cx', ann.x + ann.w / 2); el.setAttribute('cy', ann.y + ann.h / 2);
        el.setAttribute('rx', ann.w / 2); el.setAttribute('ry', ann.h / 2);
        el.setAttribute('stroke', color); el.setAttribute('stroke-width', sw);
        el.setAttribute('fill', 'none');
        svg.appendChild(el);
      } else if (ann.shape === 'line') {
        const l = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        l.setAttribute('x1', ann.x1); l.setAttribute('y1', ann.y1);
        l.setAttribute('x2', ann.x2); l.setAttribute('y2', ann.y2);
        l.setAttribute('stroke', color); l.setAttribute('stroke-width', sw);
        svg.appendChild(l);
      } else if (ann.shape === 'arrow') {
        const l = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        l.setAttribute('x1', ann.x1); l.setAttribute('y1', ann.y1);
        l.setAttribute('x2', ann.x2); l.setAttribute('y2', ann.y2);
        l.setAttribute('stroke', color); l.setAttribute('stroke-width', sw);
        svg.appendChild(l);
        const angle = Math.atan2(ann.y2 - ann.y1, ann.x2 - ann.x1);
        const headLen = sw * 6;
        const a1x = ann.x2 - headLen * Math.cos(angle - Math.PI / 6);
        const a1y = ann.y2 - headLen * Math.sin(angle - Math.PI / 6);
        const a2x = ann.x2 - headLen * Math.cos(angle + Math.PI / 6);
        const a2y = ann.y2 - headLen * Math.sin(angle + Math.PI / 6);
        const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        poly.setAttribute('points', `${ann.x2},${ann.y2} ${a1x},${a1y} ${a2x},${a2y}`);
        poly.setAttribute('fill', color);
        svg.appendChild(poly);
      }
      _dom.overlay.appendChild(svg);
    }

    if (ann.type === 'signature') {
      const el = document.createElement('div');
      el.className = `pdf-editor__annotation pdf-editor__annotation--signature${isSelected ? ' selected' : ''}`;
      el.style.left = `${ann.x}px`;
      el.style.top = `${ann.y}px`;
      el.style.width = `${ann.w}px`;
      el.style.height = `${ann.h}px`;
      el.innerHTML = `<span class="pdf-editor__sig-label">Signature</span>`;
      _dom.overlay.appendChild(el);
    }

    // Selection handles
    if (isSelected) {
      const del = document.createElement('button');
      del.className = 'pdf-editor__delete-btn';
      del.title = 'Delete (Del)';
      del.innerHTML = '&times;';
      del.style.position = 'absolute';
      // Position near the annotation
      if (ann.type === 'text') {
        del.style.left = `${ann.x - 8}px`;
        del.style.top = `${ann.y - 12}px`;
      } else if (ann.type === 'highlight' || ann.type === 'signature' || (ann.type === 'shape' && (ann.shape === 'rect' || ann.shape === 'circle'))) {
        del.style.left = `${ann.x + ann.w - 4}px`;
        del.style.top = `${ann.y - 12}px`;
      } else if (ann.type === 'shape' && (ann.shape === 'line' || ann.shape === 'arrow')) {
        del.style.left = `${Math.max(ann.x1, ann.x2)}px`;
        del.style.top = `${Math.min(ann.y1, ann.y2) - 12}px`;
      } else if (ann.type === 'draw' && ann.points?.length) {
        const maxX = Math.max(...ann.points.map(p => p.x));
        const minY = Math.min(...ann.points.map(p => p.y));
        del.style.left = `${maxX}px`;
        del.style.top = `${minY - 12}px`;
      }
      del.addEventListener('click', (ev) => { ev.stopPropagation(); _deleteSelectedAnnotation(); });
      _dom.overlay.appendChild(del);
    }
  }
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------
function _handleKeydown(e) {
  // Don't intercept when typing in textarea/input
  if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') {
    // But handle Escape in search input to close search
    if (e.key === 'Escape' && _state._searchOpen) {
      _closeSearch();
      e.preventDefault();
    }
    return;
  }

  // Navigation
  if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') { e.preventDefault(); _goToPage(_state.currentPage - 1); }
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') { e.preventDefault(); _goToPage(_state.currentPage + 1); }

  // Zoom
  if ((e.ctrlKey || e.metaKey) && e.key === '=') { e.preventDefault(); _setZoom(_state.scale + 0.25); }
  if ((e.ctrlKey || e.metaKey) && e.key === '-') { e.preventDefault(); _setZoom(_state.scale - 0.25); }

  // Undo/Redo
  if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) { e.preventDefault(); _undo(); }
  if ((e.ctrlKey || e.metaKey) && e.key === 'z' && e.shiftKey) { e.preventDefault(); _redo(); }
  if ((e.ctrlKey || e.metaKey) && e.key === 'y') { e.preventDefault(); _redo(); }

  // Find
  if ((e.ctrlKey || e.metaKey) && e.key === 'f') { e.preventDefault(); _toggleSearch(); }

  // Print
  if ((e.ctrlKey || e.metaKey) && e.key === 'p') { e.preventDefault(); _printPdf(); }

  // Delete selected annotation
  if ((e.key === 'Delete' || e.key === 'Backspace') && _state._selectedIdx >= 0) {
    e.preventDefault();
    _deleteSelectedAnnotation();
  }

  // Escape
  if (e.key === 'Escape') {
    if (_state._searchOpen) _closeSearch();
    else if (_state._selectedIdx >= 0) { _state._selectedIdx = -1; _renderAnnotations(_state.currentPage); }
    else _setTool('select');
  }

  // Tool shortcuts (only when no modifier)
  if (!e.ctrlKey && !e.metaKey && !e.altKey) {
    if (e.key === 'v') _setTool('select');
    if (e.key === 't') _setTool('text');
    if (e.key === 'h') _setTool('highlight');
    if (e.key === 'd') _setTool('draw');
    if (e.key === 's') _setTool('shape');
    if (e.key === 'i') _setTool('image');
    if (e.key === 'g') _setTool('signature');
  }
}

// ---------------------------------------------------------------------------
// Merge PDFs
// ---------------------------------------------------------------------------
function _openMergeFilePicker() {
  _dom.container?.querySelector('#pdf-merge-input')?.click();
}

async function _handleMergeFiles(e) {
  const files = Array.from(e.target.files || []);
  e.target.value = '';
  if (files.length === 0 || !_state.pdfLibDoc) return;

  try {
    const PDFDoc = await _ensurePdfLib();

    for (const file of files) {
      const bytes = new Uint8Array(await file.arrayBuffer());
      const srcDoc = await PDFDoc.load(bytes);
      const pageCount = srcDoc.getPageCount();
      const indices = Array.from({ length: pageCount }, (_, i) => i);
      const copiedPages = await _state.pdfLibDoc.copyPages(srcDoc, indices);
      for (const page of copiedPages) {
        _state.pdfLibDoc.addPage(page);
      }
    }

    _markDirty();
    await _refreshPdfJsDoc();
    await _renderPage(_state.currentPage);
    await _renderPageNav();
  } catch (err) {
    console.warn('[pdf-editor] Merge failed:', err);
  }
}

// ---------------------------------------------------------------------------
// Bookmarks panel
// ---------------------------------------------------------------------------
function _toggleBookmarks() {
  const panel = _dom.bookmarksPanel;
  if (!panel) return;
  const wasHidden = panel.classList.contains('hidden');
  panel.classList.toggle('hidden');
  if (wasHidden) _loadBookmarks();
}

async function _loadBookmarks() {
  if (!_state.pdfDoc || !_dom.bookmarksList) return;

  try {
    const outline = await _state.pdfDoc.getOutline();
    if (!outline || outline.length === 0) {
      _dom.bookmarksList.innerHTML = '<div class="pdf-editor__bookmarks-empty">No bookmarks in this PDF</div>';
      return;
    }
    _dom.bookmarksList.innerHTML = '';
    _renderBookmarkItems(outline, _dom.bookmarksList, 0);
  } catch {
    _dom.bookmarksList.innerHTML = '<div class="pdf-editor__bookmarks-empty">Could not read bookmarks</div>';
  }
}

function _renderBookmarkItems(items, container, depth) {
  for (const item of items) {
    const el = document.createElement('div');
    el.className = 'pdf-editor__bookmark-item';
    el.style.paddingLeft = `${8 + depth * 14}px`;
    el.textContent = item.title || 'Untitled';
    el.addEventListener('click', async () => {
      try {
        if (item.dest) {
          // Resolve destination to page number
          const dest = typeof item.dest === 'string'
            ? await _state.pdfDoc.getDestination(item.dest)
            : item.dest;
          if (dest) {
            const ref = dest[0];
            const pageIdx = await _state.pdfDoc.getPageIndex(ref);
            _goToPage(pageIdx + 1);
          }
        }
      } catch {
        // Fallback — some bookmark types can't be resolved
      }
    });
    container.appendChild(el);

    // Recurse into children
    if (item.items?.length) {
      _renderBookmarkItems(item.items, container, depth + 1);
    }
  }
}

// ---------------------------------------------------------------------------
// Re-load pdf.js from pdf-lib state
// ---------------------------------------------------------------------------
async function _refreshPdfJsDoc() {
  if (!_state.pdfLibDoc) return;
  const bytes = await _state.pdfLibDoc.save();
  if (_state.pdfDoc) _state.pdfDoc.destroy();
  const pdfjsLib = await _ensurePdfJs();
  _state.pdfDoc = await pdfjsLib.getDocument({ data: bytes }).promise;
  _state.totalPages = _state.pdfDoc.numPages;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function _hexToRgb(hex) {
  const h = hex.replace('#', '');
  return {
    r: parseInt(h.substring(0, 2), 16) / 255,
    g: parseInt(h.substring(2, 4), 16) / 255,
    b: parseInt(h.substring(4, 6), 16) / 255,
  };
}

function _hexToRgba(hex, alpha) {
  const { r, g, b } = _hexToRgb(hex);
  return `rgba(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)},${alpha})`;
}
