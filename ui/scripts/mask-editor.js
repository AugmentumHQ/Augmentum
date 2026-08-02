/**
 * Mask editor — portable inpaint mask painter.
 *
 * Extracted from the lightbox inpaint overlay (`image.js:_lb*` functions) so
 * both the image lightbox and Studio's image viewer can mount the same
 * editor. Instance-scoped — no singleton DOM ids — so multiple editors
 * (e.g., a lightbox instance and a Studio instance) can coexist.
 *
 * Usage:
 *   import { createMaskEditor } from './mask-editor.js';
 *   const ed = createMaskEditor({ container, sourceImg, variant, ... });
 *   // ed.destroy() tears down listeners + DOM.
 *
 * The editor itself is UI-only. Callers own the network call — the Generate
 * button fires `onGenerate({ maskBase64, prompt, negativePrompt, mode,
 * strength })` and the caller POSTs to whichever inpaint endpoint it
 * wants (image-keyed for the lightbox, artifact-keyed for Studio).
 */

import { escapeHtml } from './app.js';

// Black background + white painted strokes is the mask convention shared by
// every backend: white = "repaint this pixel", black = "preserve as-is".
const BG_COLOR = 'black';
const PAINT_COLOR = 'white';

// Undo stack cap matches the lightbox + sidebar editors — deep enough for
// real-use paint sessions, bounded so we don't keep N large ImageDatas
// referenced when someone cranks up the brush size.
const UNDO_MAX = 30;

/**
 * @typedef {'lightbox' | 'studio'} Variant
 * @typedef {{ maskBase64: string, prompt: string, negativePrompt: string, mode: string, strength: number }} GeneratePayload
 */

/**
 * @param {Object} cfg
 * @param {HTMLElement} cfg.container     — element to mount inside (overlay takes `inset:0` within).
 * @param {HTMLImageElement} cfg.sourceImg — already-loaded image whose natural dims seed the canvas.
 * @param {Variant} [cfg.variant='studio'] — theming variant.
 * @param {boolean} [cfg.showPromptStrip=true] — include prompt/negative/mode/strength row.
 * @param {string} [cfg.initialPrompt=''] — pre-fill the prompt textarea.
 * @param {(payload: GeneratePayload) => void} [cfg.onGenerate]
 * @param {() => void} [cfg.onCancel]     — called when the back/cancel button is hit.
 * @param {string} [cfg.generateLabel='Generate'] — button text.
 * @returns {{ destroy: () => void, getMaskBase64: () => string, setBusy: (busy: boolean, label?: string) => void }}
 */
export function createMaskEditor(cfg) {
  const {
    container,
    sourceImg,
    variant = 'studio',
    showPromptStrip = true,
    initialPrompt = '',
    onGenerate,
    onCancel,
    generateLabel = 'Generate',
  } = cfg;

  if (!container || !sourceImg) throw new Error('createMaskEditor: container and sourceImg required');
  if (!sourceImg.naturalWidth) throw new Error('createMaskEditor: sourceImg must be loaded (have naturalWidth)');

  // ---------- DOM ----------
  const root = document.createElement('div');
  root.className = `mask-editor mask-editor--${variant}`;
  root.innerHTML = _template({ variant, showPromptStrip, initialPrompt, generateLabel });
  container.appendChild(root);

  const els = {
    root,
    back:     root.querySelector('[data-act="back"]'),
    brush:    root.querySelector('[data-tool="brush"]'),
    eraser:   root.querySelector('[data-tool="eraser"]'),
    size:     root.querySelector('[data-inp="size"]'),
    opacity:  root.querySelector('[data-inp="opacity"]'),
    undo:     root.querySelector('[data-act="undo"]'),
    clear:    root.querySelector('[data-act="clear"]'),
    invert:   root.querySelector('[data-act="invert"]'),
    generate: root.querySelector('[data-act="generate"]'),
    canvasWrap: root.querySelector('[data-role="canvas-wrap"]'),
    bgCanvas:   root.querySelector('[data-canvas="bg"]'),
    overlayCanvas: root.querySelector('[data-canvas="overlay"]'),
    uiCanvas:   root.querySelector('[data-canvas="ui"]'),
    prompt:     root.querySelector('[data-inp="prompt"]'),
    negative:   root.querySelector('[data-inp="negative"]'),
    strength:   root.querySelector('[data-inp="strength"]'),
    strengthVal: root.querySelector('[data-inp="strength-val"]'),
    modePills:  root.querySelectorAll('[data-mode]'),
  };

  // ---------- State ----------
  // naturalW/H drive the data canvas + the rendered overlay. baseDisplayW/H
  // are layout-only — they resize on window-resize events.
  const state = {
    naturalW: sourceImg.naturalWidth,
    naturalH: sourceImg.naturalHeight,
    brushSize: 30,
    brushOpacity: 0.5,
    tool: 'brush',
    painting: false,
    lastPoint: null,
    undoStack: [],
    redoStack: [],
    mode: 'default',
    // Offscreen "data canvas" — the authoritative mask. Export to PNG at
    // send time. The on-screen overlayCanvas renders a tinted preview;
    // the uiCanvas draws the brush cursor only.
    dataCanvas: document.createElement('canvas'),
  };
  state.dataCanvas.width = state.naturalW;
  state.dataCanvas.height = state.naturalH;

  const ctx = {
    bg: els.bgCanvas.getContext('2d'),
    overlay: els.overlayCanvas.getContext('2d'),
    ui: els.uiCanvas.getContext('2d'),
    data: state.dataCanvas.getContext('2d'),
  };

  // Three-canvas stack sized to natural pixels so paint stays crisp; CSS
  // scales them down to fit the container (`_applyLayout` below).
  [els.bgCanvas, els.overlayCanvas, els.uiCanvas].forEach(c => {
    c.width = state.naturalW;
    c.height = state.naturalH;
  });
  ctx.data.fillStyle = BG_COLOR;
  ctx.data.fillRect(0, 0, state.naturalW, state.naturalH);
  ctx.bg.drawImage(sourceImg, 0, 0);

  // Seed the undo stack with the blank mask so the first undo returns to
  // "no paint" instead of a stack-underflow no-op.
  state.undoStack.push(ctx.data.getImageData(0, 0, state.naturalW, state.naturalH));

  // ---------- Layout ----------
  // Fit into the canvas-wrap, preserving aspect ratio. We listen for
  // ResizeObserver so Studio/orb-sidebar toggles reflow correctly.
  function _applyLayout() {
    const wrap = els.canvasWrap;
    if (!wrap.clientWidth || !wrap.clientHeight) return;
    const maxW = wrap.clientWidth * 0.95;
    const maxH = wrap.clientHeight * 0.95;
    const scale = Math.min(maxW / state.naturalW, maxH / state.naturalH, 1);
    const dispW = Math.round(state.naturalW * scale);
    const dispH = Math.round(state.naturalH * scale);
    [els.bgCanvas, els.overlayCanvas, els.uiCanvas].forEach(c => {
      c.style.width = `${dispW}px`;
      c.style.height = `${dispH}px`;
    });
  }
  _applyLayout();
  const ro = new ResizeObserver(_applyLayout);
  ro.observe(els.canvasWrap);

  // ---------- Paint ----------
  function _pointerToCanvas(e) {
    const rect = els.uiCanvas.getBoundingClientRect();
    const sx = state.naturalW / rect.width;
    const sy = state.naturalH / rect.height;
    return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy };
  }

  function _paint(e) {
    if (!state.painting) return;
    const pt = _pointerToCanvas(e);
    const c = ctx.data;
    c.globalCompositeOperation = state.tool === 'eraser' ? 'destination-out' : 'source-over';
    c.fillStyle = state.tool === 'eraser' ? 'rgba(0,0,0,1)' : PAINT_COLOR;
    c.strokeStyle = c.fillStyle;
    c.lineWidth = state.brushSize;
    c.lineCap = 'round';
    c.lineJoin = 'round';
    if (state.lastPoint) {
      c.beginPath();
      c.moveTo(state.lastPoint.x, state.lastPoint.y);
      c.lineTo(pt.x, pt.y);
      c.stroke();
    } else {
      c.beginPath();
      c.arc(pt.x, pt.y, state.brushSize / 2, 0, Math.PI * 2);
      c.fill();
    }
    c.globalCompositeOperation = 'source-over';
    state.lastPoint = pt;
    _renderMask();
    _renderCursor(e);
  }

  function _renderMask() {
    const w = state.naturalW, h = state.naturalH;
    ctx.overlay.clearRect(0, 0, w, h);
    const mask = ctx.data.getImageData(0, 0, w, h);
    const tint = ctx.overlay.createImageData(w, h);
    const src = mask.data;
    const dst = tint.data;
    // Iterate once — every painted pixel (R > 128 on a black-bg canvas)
    // gets the red tint. Faster than per-pixel globalCompositeOperation.
    for (let i = 0; i < src.length; i += 4) {
      if (src[i] > 128) {
        dst[i] = 255; dst[i + 1] = 40; dst[i + 2] = 80; dst[i + 3] = 200;
      }
    }
    ctx.overlay.globalAlpha = state.brushOpacity;
    ctx.overlay.putImageData(tint, 0, 0);
    ctx.overlay.globalAlpha = 1;
  }

  function _renderCursor(e) {
    const pt = _pointerToCanvas(e);
    ctx.ui.clearRect(0, 0, state.naturalW, state.naturalH);
    ctx.ui.beginPath();
    ctx.ui.arc(pt.x, pt.y, state.brushSize / 2, 0, Math.PI * 2);
    ctx.ui.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.ui.lineWidth = Math.max(1.5, state.naturalW / 1200);
    ctx.ui.stroke();
  }

  function _pushUndo() {
    state.undoStack.push(ctx.data.getImageData(0, 0, state.naturalW, state.naturalH));
    if (state.undoStack.length > UNDO_MAX) state.undoStack.shift();
    state.redoStack.length = 0;
  }

  // ---------- Pointer handlers (instance-captured so destroy can remove) ----------
  const onPointerDown = (e) => {
    e.preventDefault();
    els.uiCanvas.setPointerCapture?.(e.pointerId);
    state.painting = true;
    state.lastPoint = null;
    _paint(e);
  };
  const onPointerMove = (e) => {
    if (state.painting) _paint(e);
    else _renderCursor(e);
  };
  const onPointerUp = (e) => {
    if (!state.painting) return;
    state.painting = false;
    state.lastPoint = null;
    try { els.uiCanvas.releasePointerCapture?.(e.pointerId); } catch { /* cancelled */ }
    _pushUndo();
  };
  const onPointerLeave = () => {
    ctx.ui.clearRect(0, 0, state.naturalW, state.naturalH);
    if (state.painting) {
      state.painting = false;
      state.lastPoint = null;
      _pushUndo();
    }
  };

  els.uiCanvas.addEventListener('pointerdown', onPointerDown);
  els.uiCanvas.addEventListener('pointermove', onPointerMove);
  els.uiCanvas.addEventListener('pointerup', onPointerUp);
  els.uiCanvas.addEventListener('pointercancel', onPointerUp);
  els.uiCanvas.addEventListener('pointerleave', onPointerLeave);

  // ---------- Toolbar ----------
  els.back?.addEventListener('click', () => onCancel?.());

  els.brush.addEventListener('click', () => _setTool('brush'));
  els.eraser.addEventListener('click', () => _setTool('eraser'));
  function _setTool(t) {
    state.tool = t;
    els.brush.classList.toggle('active', t === 'brush');
    els.eraser.classList.toggle('active', t === 'eraser');
  }

  els.size.addEventListener('input', () => { state.brushSize = parseInt(els.size.value) || 30; });
  els.opacity.addEventListener('input', () => {
    state.brushOpacity = (parseInt(els.opacity.value) || 50) / 100;
    _renderMask();
  });

  els.undo.addEventListener('click', () => {
    if (state.undoStack.length <= 1) return;
    state.redoStack.push(state.undoStack.pop());
    ctx.data.putImageData(state.undoStack[state.undoStack.length - 1], 0, 0);
    _renderMask();
  });
  els.clear.addEventListener('click', () => {
    ctx.data.fillStyle = BG_COLOR;
    ctx.data.fillRect(0, 0, state.naturalW, state.naturalH);
    _pushUndo();
    _renderMask();
  });
  els.invert.addEventListener('click', () => {
    const img = ctx.data.getImageData(0, 0, state.naturalW, state.naturalH);
    const d = img.data;
    // RGB-only invert — alpha stays at 255 so the eraser's alpha work
    // doesn't get flipped into the painted area.
    for (let i = 0; i < d.length; i += 4) {
      d[i] = 255 - d[i]; d[i + 1] = 255 - d[i + 1]; d[i + 2] = 255 - d[i + 2];
    }
    ctx.data.putImageData(img, 0, 0);
    _pushUndo();
    _renderMask();
  });

  // ---------- Prompt + mode + strength ----------
  if (showPromptStrip) {
    els.modePills.forEach(pill => {
      pill.addEventListener('click', () => {
        state.mode = pill.dataset.mode || 'default';
        els.modePills.forEach(p => p.classList.toggle('active', p === pill));
      });
    });
    els.strength?.addEventListener('input', () => {
      const v = (parseInt(els.strength.value) || 100) / 100;
      if (els.strengthVal) els.strengthVal.textContent = v.toFixed(2);
    });
  }

  // ---------- Generate ----------
  els.generate.addEventListener('click', () => {
    if (!_hasPaint()) {
      // Flash the canvas as a cheap "please paint first" cue — avoids adding
      // a toast dependency at the module boundary.
      els.overlayCanvas.animate(
        [{ filter: 'drop-shadow(0 0 0 transparent)' }, { filter: 'drop-shadow(0 0 12px rgba(255,64,96,0.85))' }, { filter: 'drop-shadow(0 0 0 transparent)' }],
        { duration: 500 },
      );
      return;
    }
    onGenerate?.({
      maskBase64: _getMaskBase64(),
      prompt: (els.prompt?.value || '').trim(),
      negativePrompt: (els.negative?.value || '').trim(),
      mode: state.mode,
      strength: parseFloat(els.strength?.value ? els.strength.value / 100 : 1.0),
    });
  });

  function _hasPaint() {
    const d = ctx.data.getImageData(0, 0, state.naturalW, state.naturalH).data;
    for (let i = 0; i < d.length; i += 4) if (d[i] > 128) return true;
    return false;
  }
  function _getMaskBase64() {
    return state.dataCanvas.toDataURL('image/png').split(',')[1];
  }

  // ---------- Public API ----------
  function destroy() {
    ro.disconnect();
    els.uiCanvas.removeEventListener('pointerdown', onPointerDown);
    els.uiCanvas.removeEventListener('pointermove', onPointerMove);
    els.uiCanvas.removeEventListener('pointerup', onPointerUp);
    els.uiCanvas.removeEventListener('pointercancel', onPointerUp);
    els.uiCanvas.removeEventListener('pointerleave', onPointerLeave);
    root.remove();
  }

  function setBusy(busy, label) {
    els.generate.disabled = !!busy;
    els.generate.textContent = busy ? (label || 'Generating…') : generateLabel;
    if (busy) root.classList.add('is-busy');
    else root.classList.remove('is-busy');
  }

  // Initial render so the black-mask canvas paints at the right size and
  // the toolbar's active states reflect the defaults.
  _setTool('brush');
  _renderMask();

  return {
    destroy,
    getMaskBase64: _getMaskBase64,
    setBusy,
  };
}

// Separate template so the markup is scannable. Uses data-* attributes to
// let the module find elements without fighting DOM id uniqueness across
// multiple concurrent instances.
function _template({ variant, showPromptStrip, initialPrompt, generateLabel }) {
  const prompt = showPromptStrip ? `
    <div class="mask-editor-prompt">
      <textarea class="mask-editor-prompt-input" data-inp="prompt"
                placeholder="What should appear in the painted area?"
                spellcheck="false" rows="2">${escapeHtml(initialPrompt || '')}</textarea>
      <div class="mask-editor-prompt-controls">
        <input type="text" class="mask-editor-prompt-neg" data-inp="negative"
               placeholder="Negative (optional)" autocomplete="off" spellcheck="false">
        <div class="mask-editor-mode">
          <button class="mask-editor-mode-pill active" data-mode="default">Default</button>
          <button class="mask-editor-mode-pill" data-mode="improve">Improve</button>
          <button class="mask-editor-mode-pill" data-mode="modify">Modify</button>
        </div>
        <label class="mask-editor-strength">
          Strength
          <input type="range" data-inp="strength" min="5" max="100" value="100">
          <span data-inp="strength-val">1.00</span>
        </label>
      </div>
    </div>
  ` : '';

  return `
    <div class="mask-editor-toolbar">
      <button class="mask-editor-back" data-act="back" title="Cancel">&larr; Cancel</button>
      <div class="mask-editor-tools">
        <button class="mask-editor-tool active" data-tool="brush" title="Brush (B)">Brush</button>
        <button class="mask-editor-tool" data-tool="eraser" title="Eraser (E)">Eraser</button>
      </div>
      <label class="mask-editor-label">Size</label>
      <input type="range" data-inp="size" min="5" max="100" value="30" class="mask-editor-slider">
      <label class="mask-editor-label">Opacity</label>
      <input type="range" data-inp="opacity" min="10" max="100" value="50" class="mask-editor-slider">
      <div class="mask-editor-actions">
        <button class="mask-editor-tool" data-act="undo" title="Undo">Undo</button>
        <button class="mask-editor-tool" data-act="clear" title="Clear">Clear</button>
        <button class="mask-editor-tool" data-act="invert" title="Invert">Invert</button>
      </div>
      <button class="mask-editor-generate" data-act="generate">${escapeHtml(generateLabel)}</button>
    </div>
    <div class="mask-editor-canvas-wrap" data-role="canvas-wrap">
      <canvas class="mask-editor-canvas" data-canvas="bg"></canvas>
      <canvas class="mask-editor-canvas" data-canvas="overlay"></canvas>
      <canvas class="mask-editor-canvas" data-canvas="ui"></canvas>
    </div>
    ${prompt}
  `;
}
