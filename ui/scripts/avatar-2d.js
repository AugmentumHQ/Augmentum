/**
 * avatar-2d.js — 2D animated portrait renderer (DOM-based)
 *
 * Uses CSS transforms on DOM elements instead of Canvas 2D.
 * This approach survives mobile audio playback — DOM elements are
 * composited by the browser's own pipeline, immune to canvas context loss.
 *
 * Architecture: The portrait image is shown as a base <img>. Face regions
 * (mouth, eyes, brows) are rendered as positioned <div> elements that use
 * background-image + background-position to show the same region of the
 * portrait, then CSS transform to animate them (scale for mouth open,
 * scaleY for blink, translateY for brows). The base image underneath
 * provides the static fallback — warped overlays draw on top.
 */

// MediaPipe loaded on demand from CDN (~15MB, cached by browser)
let FaceLandmarkerClass = null;
let FilesetResolverClass = null;

const MEDIAPIPE_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MEDIAPIPE_BUNDLE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task';

async function ensureMediaPipe() {
  if (FaceLandmarkerClass) return;
  const vision = await import(MEDIAPIPE_BUNDLE);
  FaceLandmarkerClass = vision.FaceLandmarker;
  FilesetResolverClass = vision.FilesetResolver;
}

// ---- Face Region Landmark Indices ----
const MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78];
const MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
const LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246];
const RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398];
const LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46];
const RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276];
const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];

/**
 * Segment a portrait image into face regions.
 */
export async function segmentPortrait(imageOrUrl) {
  await ensureMediaPipe();

  const fileset = await FilesetResolverClass.forVisionTasks(MEDIAPIPE_CDN);
  const landmarker = await FaceLandmarkerClass.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL },
    outputFaceBlendshapes: true,
    runningMode: 'IMAGE',
    numFaces: 1,
  });

  let img = imageOrUrl;
  if (typeof imageOrUrl === 'string') {
    img = new Image();
    img.crossOrigin = 'anonymous';
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = imageOrUrl;
    });
  }

  const result = landmarker.detect(img);
  landmarker.close();

  if (!result.faceLandmarks?.length) {
    throw new Error('No face detected in portrait');
  }

  const lm = result.faceLandmarks[0];
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const pts = lm.map(p => ({ x: p.x * w, y: p.y * h, z: (p.z || 0) * w }));

  return {
    landmarks: pts,
    regions: {
      mouth: _regionBounds(pts, MOUTH_OUTER),
      leftEye: _regionBounds(pts, LEFT_EYE),
      rightEye: _regionBounds(pts, RIGHT_EYE),
      leftBrow: _regionBounds(pts, LEFT_BROW),
      rightBrow: _regionBounds(pts, RIGHT_BROW),
      faceOval: _regionBounds(pts, FACE_OVAL),
    },
    width: w,
    height: h,
  };
}

function _regionBounds(pts, indices) {
  const rpts = indices.map(i => pts[i]).filter(Boolean);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of rpts) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  return {
    indices,
    bounds: { x: minX, y: minY, w: maxX - minX, h: maxY - minY },
    center: { x: (minX + maxX) / 2, y: (minY + maxY) / 2 },
  };
}

// ---- DOM-Based 2D Portrait Renderer ----

export class Portrait2DRenderer {
  constructor(container, portraitUrl, segData) {
    this._container = container;
    this._segData = segData;
    this._portraitUrl = portraitUrl;

    // DOM elements for each animated region
    this._root = null;    // wrapper div
    this._baseImg = null; // full portrait <img>
    this._mouthEl = null; // mouth overlay div
    this._leftEyeEl = null;
    this._rightEyeEl = null;
    this._leftBrowEl = null;
    this._rightBrowEl = null;

    // Smoothed animation state
    this._mouth = { openY: 0, stretchX: 0, pucker: 0 };
    this._blinkL = 0;
    this._blinkR = 0;
    this._browOffset = 0;
    this._headRotY = 0;
    this._headRotX = 0;
    this._breathScale = 0;

    // Blink state machine
    this._blinkPhase = 0;
    this._blinkTimer = 0;
    this._nextBlink = 2 + Math.random() * 4;
  }

  async init() {
    // Load image to get natural dimensions
    const img = new Image();
    img.crossOrigin = 'anonymous';
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = this._portraitUrl;
    });
    this._naturalW = img.naturalWidth;
    this._naturalH = img.naturalHeight;

    // Build DOM structure
    this._root = document.createElement('div');
    this._root.className = 'avatar-2d-root';

    // Base portrait image — always visible, provides the static face
    this._baseImg = document.createElement('img');
    this._baseImg.src = this._portraitUrl;
    this._baseImg.crossOrigin = 'anonymous';
    this._baseImg.className = 'avatar-2d-base';
    this._baseImg.draggable = false;
    this._root.appendChild(this._baseImg);

    // Create region overlays for animatable face parts
    const seg = this._segData;
    const pad = 6; // extra pixels around each region for clean edges
    this._mouthEl = this._createRegionEl(seg.regions.mouth.bounds, pad + 2);
    this._leftEyeEl = this._createRegionEl(seg.regions.leftEye.bounds, pad);
    this._rightEyeEl = this._createRegionEl(seg.regions.rightEye.bounds, pad);
    this._leftBrowEl = this._createRegionEl(seg.regions.leftBrow.bounds, pad + 2);
    this._rightBrowEl = this._createRegionEl(seg.regions.rightBrow.bounds, pad + 2);

    this._root.appendChild(this._mouthEl);
    this._root.appendChild(this._leftEyeEl);
    this._root.appendChild(this._rightEyeEl);
    this._root.appendChild(this._leftBrowEl);
    this._root.appendChild(this._rightBrowEl);

    this._container.appendChild(this._root);

    // Observe container resizes to reposition region overlays
    // Safe — no canvas to clear, just recalculates CSS positions
    this._resizeObserver = new ResizeObserver(() => this._updateLayout());
    this._resizeObserver.observe(this._container);

    // Initial layout (may be zero if container not visible yet — resize() will fix)
    this._updateLayout();

    return this;
  }

  /**
   * Create a positioned div that shows a specific region of the portrait.
   * Uses background-image + background-position to clip to the region.
   */
  _createRegionEl(bounds, pad) {
    const el = document.createElement('div');
    el.className = 'avatar-2d-region';
    el.style.backgroundImage = `url(${CSS.escape(this._portraitUrl)})`;

    // Store region bounds (image pixel space) for layout calculations
    el._regionX = Math.max(0, bounds.x - pad);
    el._regionY = Math.max(0, bounds.y - pad);
    el._regionW = Math.min(this._naturalW - el._regionX, bounds.w + pad * 2);
    el._regionH = Math.min(this._naturalH - el._regionY, bounds.h + pad * 2);

    // Position will be set in _updateLayout
    el.style.willChange = 'transform';
    return el;
  }

  /**
   * Recalculate region overlay positions based on current container size.
   * Converts image-pixel coordinates to percentage-based positioning.
   */
  _updateLayout() {
    const iw = this._naturalW;
    const ih = this._naturalH;
    if (!iw || !ih) return;

    // The base image uses object-fit: contain, so we need to compute
    // the actual rendered image area within the container
    const containerRect = this._container.getBoundingClientRect();
    const cw = containerRect.width;
    const ch = containerRect.height;
    if (!cw || !ch) return;

    const imgAspect = iw / ih;
    const containerAspect = cw / ch;
    let renderW, renderH, offsetX, offsetY;
    if (containerAspect > imgAspect) {
      renderH = ch;
      renderW = ch * imgAspect;
    } else {
      renderW = cw;
      renderH = cw / imgAspect;
    }
    offsetX = (cw - renderW) / 2;
    offsetY = (ch - renderH) / 2;

    // Scale factor: image pixels → rendered pixels
    const sx = renderW / iw;
    const sy = renderH / ih;

    this._sx = sx;
    this._sy = sy;
    this._offsetX = offsetX;
    this._offsetY = offsetY;
    this._renderW = renderW;
    this._renderH = renderH;

    // Position each region overlay
    for (const el of [this._mouthEl, this._leftEyeEl, this._rightEyeEl, this._leftBrowEl, this._rightBrowEl]) {
      if (!el) continue;
      const rx = el._regionX;
      const ry = el._regionY;
      const rw = el._regionW;
      const rh = el._regionH;

      // Position in container space
      el.style.left = `${offsetX + rx * sx}px`;
      el.style.top = `${offsetY + ry * sy}px`;
      el.style.width = `${rw * sx}px`;
      el.style.height = `${rh * sy}px`;

      // Background: show the same region from the full image
      // background-size scales the full image to rendered dimensions
      el.style.backgroundSize = `${renderW}px ${renderH}px`;
      // background-position offsets so the correct region is visible
      el.style.backgroundPosition = `${-rx * sx}px ${-ry * sy}px`;
    }
  }

  /**
   * Update + render one frame. Called by animation loop.
   */
  update(delta, visemes, state) {
    const smooth = 1 - Math.pow(0.35, delta * 60);

    // Mouth: map visemes to transform params
    const targetOpenY = (visemes?.jaw || 0) * 0.5 + (visemes?.aa || 0) * 0.3;
    const targetStretchX = Math.max(visemes?.ih || 0, visemes?.ee || 0) * 0.35;
    const targetPucker = (visemes?.ou || 0) * 0.25;
    this._mouth.openY += (targetOpenY - this._mouth.openY) * smooth;
    this._mouth.stretchX += (targetStretchX - this._mouth.stretchX) * smooth;
    this._mouth.pucker += (targetPucker - this._mouth.pucker) * smooth;

    // Blinks
    this._updateBlink(delta);

    // Breathing
    this._breathScale = Math.sin(performance.now() / 1000 * Math.PI * 0.3) * 0.004;

    // Head sway — layered frequencies for organic drift
    const t = performance.now() / 1000;
    const headTargetY = Math.sin(t * 0.08) * 0.8 + Math.sin(t * 0.19) * 0.4;
    const headTargetX = Math.sin(t * 0.06 + 1.5) * 0.4 + Math.sin(t * 0.14 + 0.7) * 0.2;
    this._headRotY += (headTargetY - this._headRotY) * 0.012;
    this._headRotX += (headTargetX - this._headRotX) * 0.012;

    // Emotion-driven brow
    let browTarget = 0;
    if (state?.emotion === 'sad') browTarget = -0.2;
    else if (state?.emotion === 'happy') browTarget = 0.1;
    else if (state?.emotion === 'curious' || state?.emotion === 'surprised') browTarget = 0.2;
    else if (state?.emotion === 'angry') browTarget = -0.25;
    this._browOffset += (browTarget - this._browOffset) * 0.015;

    // Apply transforms
    this._applyTransforms();
  }

  _updateBlink(delta) {
    this._blinkTimer += delta;
    const CLOSE_SPEED = 12;
    const OPEN_SPEED = 8;

    if (this._blinkPhase === 0) {
      if (this._blinkTimer >= this._nextBlink) {
        this._blinkPhase = 1;
        this._blinkTimer = 0;
      }
    } else if (this._blinkPhase === 1) {
      this._blinkL = Math.min(1, this._blinkL + delta * CLOSE_SPEED);
      this._blinkR = this._blinkL;
      if (this._blinkL >= 1) this._blinkPhase = 2;
    } else if (this._blinkPhase === 2) {
      this._blinkL = Math.max(0, this._blinkL - delta * OPEN_SPEED);
      this._blinkR = this._blinkL;
      if (this._blinkL <= 0) {
        this._blinkPhase = 0;
        this._blinkTimer = 0;
        this._nextBlink = 2 + Math.random() * 4;
        if (Math.random() < 0.15) this._nextBlink = 0.2;
      }
    }
  }

  _applyTransforms() {
    // Root: head sway + breathing
    if (this._root) {
      const rotY = this._headRotY * 0.3;
      const rotX = this._headRotX * 0.2;
      const bs = 1 + this._breathScale;
      this._root.style.transform = `rotate(${rotY}deg) scale(${bs})`;
    }

    // Mouth: vertical scale (open) + horizontal scale (stretch/pucker)
    if (this._mouthEl) {
      const scaleY = 1 + this._mouth.openY * 0.8;
      const scaleX = 1 + (this._mouth.stretchX - this._mouth.pucker) * 0.3;
      this._mouthEl.style.transform = `scaleX(${scaleX}) scaleY(${scaleY})`;
      this._mouthEl.style.transformOrigin = 'center top';
      // Show mouth overlay only when animating (hides at rest so base image shows clean)
      this._mouthEl.style.opacity = (this._mouth.openY > 0.01 || this._mouth.stretchX > 0.01) ? '1' : '0';
    }

    // Eyes: vertical squeeze for blink
    if (this._leftEyeEl) {
      const scaleY = 1 - this._blinkL * 0.85;
      this._leftEyeEl.style.transform = `scaleY(${scaleY})`;
      this._leftEyeEl.style.transformOrigin = 'center center';
      this._leftEyeEl.style.opacity = this._blinkL > 0.01 ? '1' : '0';
    }
    if (this._rightEyeEl) {
      const scaleY = 1 - this._blinkR * 0.85;
      this._rightEyeEl.style.transform = `scaleY(${scaleY})`;
      this._rightEyeEl.style.transformOrigin = 'center center';
      this._rightEyeEl.style.opacity = this._blinkR > 0.01 ? '1' : '0';
    }

    // Brows: vertical shift
    if (this._leftBrowEl) {
      const shift = -this._browOffset * 4;
      this._leftBrowEl.style.transform = `translateY(${shift}px)`;
      this._leftBrowEl.style.opacity = Math.abs(this._browOffset) > 0.01 ? '1' : '0';
    }
    if (this._rightBrowEl) {
      const shift = -this._browOffset * 4;
      this._rightBrowEl.style.transform = `translateY(${shift}px)`;
      this._rightBrowEl.style.opacity = Math.abs(this._browOffset) > 0.01 ? '1' : '0';
    }
  }

  resize() {
    this._updateLayout();
  }

  dispose() {
    if (this._resizeObserver) {
      this._resizeObserver.disconnect();
      this._resizeObserver = null;
    }
    this._root?.remove();
    this._root = null;
    this._baseImg = null;
    this._mouthEl = null;
    this._leftEyeEl = null;
    this._rightEyeEl = null;
    this._leftBrowEl = null;
    this._rightBrowEl = null;
  }
}
