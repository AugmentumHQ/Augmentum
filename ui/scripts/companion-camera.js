/**
 * companion-camera.js — CompanionCameraView
 *
 * The shared "see-together" composite for the companion's live camera,
 * used identically by the presence widget (eye button) and the voice
 * call surface (camera button). It puts the user's live camera in the
 * SAME frame as the companion's VRM:
 *
 *   - The camera fills the view as a full-bleed backdrop.
 *   - The VRM render (already a transparent-background canvas in its host)
 *     is shrunk to an animated corner PIP by the host's CSS — so she's a
 *     reacting presence in the shot, not blocking it.
 *
 * Front camera (default) → you full-frame + her in the corner
 * ("what do you think of this?"). Back camera → the world full-frame + her
 * in the corner ("what is this?"). Same composite, only `facingMode`
 * differs. Front is mirrored (selfie convention); back is not.
 *
 * This module is transport-agnostic and render-agnostic: it owns ONLY the
 * camera `<video>` element + the host CSS state + the stream lifecycle.
 * The caller wires the stream into the backend (PTT session's
 * startLiveVision, or a call WS LiveVisionLoop) and owns the VRM. CSS in
 * the host's stylesheet (`becca-presence.css` / `voice.css`) positions the
 * PIP and reacts to the `companion-camera-on` / `data-cam-facing` flags.
 *
 * Usage:
 *   const cam = new CompanionCameraView({ host: stageEl, onStream, onError });
 *   await cam.start();                 // front by default
 *   await cam.flip();                  // front <-> back
 *   cam.stop();                        // idempotent; releases the camera
 */

import {
  openCameraStream,
  stopStream,
  listVideoDevices,
} from './camera.js';

const VIDEO_CLASS = 'companion-cam';

export class CompanionCameraView {
  constructor(opts = {}) {
    this.host = opts.host || null;
    // Called with the live MediaStream when it (re)opens — the caller wires
    // it into the frame loop. Called with null on stop.
    this._onStream = typeof opts.onStream === 'function' ? opts.onStream : () => {};
    this._onError = typeof opts.onError === 'function' ? opts.onError : () => {};
    this._onFacingChange = typeof opts.onFacingChange === 'function'
      ? opts.onFacingChange
      : () => {};

    this._stream = null;
    this._video = null;
    this._facing = 'user';        // 'user' (front) | 'environment' (back)
    this._active = false;
    this._busy = false;           // guards concurrent start/flip/stop
    this._multiCamera = null;     // lazily probed; gates the flip control
  }

  get active() { return this._active; }
  get stream() { return this._stream; }
  get facingMode() { return this._facing; }

  /** Open the camera and mount the backdrop. Idempotent-ish: a second
   *  call while active is a no-op. `facingMode` defaults to front. */
  async start({ facingMode = 'user' } = {}) {
    if (this._active || this._busy || !this.host) return false;
    this._busy = true;
    try {
      this._facing = facingMode === 'environment' ? 'environment' : 'user';
      const stream = await openCameraStream({
        facingMode: this._facing,
        video: true,
        audio: false,   // the voice pipeline owns the mic; camera is video-only
      });
      this._stream = stream;
      this._mountVideo(stream);
      this._active = true;
      this._applyHostState();
      this._onStream(stream);
      this._onFacingChange(this._facing);
      return true;
    } catch (err) {
      this._teardownVideo();
      this._onError(err);
      return false;
    } finally {
      this._busy = false;
    }
  }

  /** Swap front<->back. Reopens the stream with the other facing mode and
   *  re-emits it so the frame loop picks up the new track. No-op if only
   *  one camera exists or the loop isn't active. */
  async flip() {
    if (!this._active || this._busy || !this.host) return false;
    this._busy = true;
    const next = this._facing === 'user' ? 'environment' : 'user';
    const prev = this._stream;
    try {
      const stream = await openCameraStream({
        facingMode: next, video: true, audio: false,
      });
      // Swap in the new stream, then release the old one.
      this._stream = stream;
      this._facing = next;
      this._mountVideo(stream);
      this._applyHostState();
      stopStream(prev);
      this._onStream(stream);
      this._onFacingChange(this._facing);
      return true;
    } catch (err) {
      // Keep the existing stream on failure (e.g. back cam refused).
      this._onError(err);
      return false;
    } finally {
      this._busy = false;
    }
  }

  /** Tear down: remove the backdrop, stop the camera, clear host state.
   *  Idempotent. Does not touch the VRM. */
  stop() {
    this._active = false;
    this._teardownVideo();
    if (this._stream) { stopStream(this._stream); this._stream = null; }
    this._applyHostState();
    this._onStream(null);
  }

  /** True when the device has >1 video input (gates the flip control).
   *  Cached after first probe. Returns false on desktop / single camera. */
  async hasMultipleCameras() {
    if (this._multiCamera != null) return this._multiCamera;
    try {
      const devices = await listVideoDevices({ probeForLabels: false });
      this._multiCamera = Array.isArray(devices) && devices.length > 1;
    } catch (_) {
      this._multiCamera = false;
    }
    return this._multiCamera;
  }

  // ── internals ──────────────────────────────────────────────────────

  _mountVideo(stream) {
    let v = this._video;
    if (!v) {
      v = document.createElement('video');
      v.className = VIDEO_CLASS;
      v.muted = true;
      v.playsInline = true;
      v.setAttribute('playsinline', '');   // iOS/WebView attribute form
      v.setAttribute('webkit-playsinline', '');
      v.autoplay = true;
      // Critical layout as INLINE styles (not just a stylesheet class) so
      // the composite works on any surface, even ones that don't load
      // becca-presence.css. Fills the host as a backdrop behind the VRM.
      v.style.cssText = [
        'position:absolute',
        'inset:0',
        'width:100%',
        'height:100%',
        'object-fit:cover',
        'border-radius:inherit',
        'z-index:1',
        'background:#000',
      ].join(';');
      // Insert as the FIRST child so it paints behind the VRM canvas.
      this.host.insertBefore(v, this.host.firstChild);
      this._video = v;
    }
    v.srcObject = stream;
    try { v.play().catch(() => {}); } catch (_) {}
    try {
      const t = stream && stream.getVideoTracks ? stream.getVideoTracks()[0] : null;
      console.info('[companion-camera] video mounted',
        'host=', this.host && this.host.className,
        'track=', t && t.label, 'facing=', this._facing);
    } catch (_) { /* diagnostic only */ }
  }

  _teardownVideo() {
    const v = this._video;
    if (!v) return;
    try { v.pause(); } catch (_) {}
    try { v.srcObject = null; } catch (_) {}
    try { v.remove(); } catch (_) {}
    this._video = null;
  }

  _applyHostState() {
    if (!this.host) return;
    if (this._active) {
      this.host.classList.add('companion-camera-on');
      this.host.dataset.camFacing = this._facing;
      // Mirror the selfie view; the world view reads naturally.
      if (this._video) {
        this._video.style.transform =
          this._facing === 'user' ? 'scaleX(-1)' : 'none';
      }
      this._shrinkVrmToPip();
    } else {
      this.host.classList.remove('companion-camera-on');
      delete this.host.dataset.camFacing;
      this._restoreVrm();
    }
  }

  // Shrink the VRM canvas(es) to a corner PIP via INLINE styles with
  // !important — the base stylesheet pins canvas to width/height:100%
  // !important, which a plain inline style can't beat, so we must use
  // setProperty(..., 'important'). Originals are stashed for restore.
  _shrinkVrmToPip() {
    if (!this.host) return;
    const canvases = this.host.querySelectorAll('canvas');
    canvases.forEach((c) => {
      if (!c.dataset.camPipSaved) {
        c.dataset.camPipSaved = c.getAttribute('style') || '__none__';
      }
      const pip = {
        position: 'absolute',
        inset: 'auto 10px 10px auto',
        width: '36%',
        height: '36%',
        'border-radius': '14px',
        'z-index': '2',
        'box-shadow': '0 4px 18px rgba(0,0,0,0.45)',
        'pointer-events': 'none',
      };
      for (const [k, val] of Object.entries(pip)) c.style.setProperty(k, val, 'important');
    });
  }

  _restoreVrm() {
    if (!this.host) return;
    const canvases = this.host.querySelectorAll('canvas');
    canvases.forEach((c) => {
      const saved = c.dataset.camPipSaved;
      if (saved === undefined) return;
      if (saved === '__none__') c.removeAttribute('style');
      else c.setAttribute('style', saved);
      delete c.dataset.camPipSaved;
    });
  }
}
