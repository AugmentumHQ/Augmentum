/**
 * avatar-atmosphere.js - Reactive environmental stage for avatar calls.
 *
 * This canvas is avatar-mode only. The base voice starfield remains the global
 * call backdrop; this layer adds a higher-fidelity, state-aware call room
 * behind the avatar without adding visible UI chrome.
 */

const PHASE_THRESHOLDS = [0, 120, 300, 600];

const STATE_CONFIGS = {
  idle: {
    color: { r: 128, g: 142, b: 170 },
    accent: { r: 238, g: 205, b: 150 },
    drift: 0.16,
    focus: 0.28,
    threads: 0.10,
    constellation: 0.04,
  },
  listening: {
    color: { r: 98, g: 205, b: 212 },
    accent: { r: 235, g: 219, b: 164 },
    drift: 0.10,
    focus: 0.42,
    threads: 0.14,
    constellation: 0.10,
  },
  recording: {
    color: { r: 224, g: 144, b: 112 },
    accent: { r: 116, g: 210, b: 192 },
    drift: 0.22,
    focus: 0.54,
    threads: 0.18,
    constellation: 0.08,
  },
  processing: {
    color: { r: 142, g: 164, b: 210 },
    accent: { r: 184, g: 222, b: 186 },
    drift: 0.08,
    focus: 0.48,
    threads: 0.20,
    constellation: 0.34,
  },
  composing: {
    color: { r: 126, g: 154, b: 196 },
    accent: { r: 220, g: 205, b: 166 },
    drift: 0.06,
    focus: 0.36,
    threads: 0.12,
    constellation: 0.20,
  },
  speaking: {
    color: { r: 186, g: 148, b: 214 },
    accent: { r: 242, g: 203, b: 146 },
    drift: 0.24,
    focus: 0.62,
    threads: 0.28,
    constellation: 0.16,
  },
  disconnected: {
    color: { r: 190, g: 112, b: 100 },
    accent: { r: 235, g: 196, b: 130 },
    drift: 0.04,
    focus: 0.22,
    threads: 0.05,
    constellation: 0.02,
  },
};

const PHASE_CONFIGS = [
  { glowOpacity: 0.055, starScale: 0.74, glowScale: 1.0 },
  { glowOpacity: 0.080, starScale: 0.86, glowScale: 1.08 },
  { glowOpacity: 0.105, starScale: 1.0, glowScale: 1.18 },
  { glowOpacity: 0.130, starScale: 1.12, glowScale: 1.32 },
];

class StageStar {
  constructor(w, h, layer = 0) {
    this.layer = layer;
    this.reset(w, h, true);
  }

  reset(w, h, randomY = false) {
    this.x = Math.random() * w;
    this.y = randomY ? Math.random() * h : h + 12 + Math.random() * 80;
    this.depth = 0.35 + Math.random() * 0.65 + this.layer * 0.18;
    this.radius = (0.35 + Math.random() * 1.15) * this.depth;
    this.alpha = 0.08 + Math.random() * 0.36;
    this.phase = Math.random() * Math.PI * 2;
    this.twinkle = 0.45 + Math.random() * 1.2;
    this.drift = 0.25 + Math.random() * 0.9;
    this.tint = Math.random() < 0.22 ? 'accent' : 'main';
  }
}

export class AtmosphereEngine {
  constructor(canvas, overlay, savedState) {
    this._canvas = canvas;
    this._ctx = canvas.getContext('2d', { alpha: true });
    this._overlay = overlay;

    this._elapsed = savedState?.elapsed || 0;
    this._phase = savedState?.phase || 0;
    this._stars = [];
    this._starTarget = 0;

    this._glowOpacity = PHASE_CONFIGS[0].glowOpacity;
    this._glowColor = { ...STATE_CONFIGS.idle.color };
    this._accentColor = { ...STATE_CONFIGS.idle.accent };
    this._stateValues = {
      drift: STATE_CONFIGS.idle.drift,
      focus: STATE_CONFIGS.idle.focus,
      threads: STATE_CONFIGS.idle.threads,
      constellation: STATE_CONFIGS.idle.constellation,
    };

    this._spikeIntensity = 0;
    this._conversationState = 'idle';
    this._rms = 0;
    this._time = 0;
    this._disposed = false;

    this._resize();
    this._resizePending = false;
    this._scheduleResize = () => {
      if (this._resizePending || this._disposed) return;
      this._resizePending = true;
      requestAnimationFrame(() => {
        this._resizePending = false;
        if (!this._disposed) this._resize();
      });
    };
    this._onResize = this._scheduleResize;
    window.addEventListener('resize', this._onResize);
    const parent = canvas.parentElement;
    if (parent && typeof ResizeObserver !== 'undefined') {
      this._resizeObserver = new ResizeObserver(this._scheduleResize);
      this._resizeObserver.observe(parent);
    }
  }

  _resize() {
    const rect = this._canvas.parentElement?.getBoundingClientRect();
    if (!rect) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(1, rect.width);
    const h = Math.max(1, rect.height);
    this._canvas.width = Math.round(w * dpr);
    this._canvas.height = Math.round(h * dpr);
    this._canvas.style.width = `${w}px`;
    this._canvas.style.height = `${h}px`;
    this._ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this._ctx.imageSmoothingEnabled = true;
    this._ctx.imageSmoothingQuality = 'high';
    this._dpr = dpr;
    this._w = w;
    this._h = h;
    this._seedStars();
  }

  update(dt, state = {}) {
    if (this._disposed) return;
    this._elapsed += dt;
    this._time += dt;
    this._rms += ((state.rms || 0) - this._rms) * (1 - Math.exp(-12 * dt));
    this._conversationState = this._resolveConversationState(state);

    const newPhase = this._calcPhase();
    if (newPhase !== this._phase) this._phase = newPhase;
    const phaseCfg = PHASE_CONFIGS[this._phase];
    const targetCfg = STATE_CONFIGS[this._conversationState] || STATE_CONFIGS.idle;

    if (state.emotion && state.emotion !== 'neutral' && state.emotionIntensity > 0.5) {
      this._triggerSpike();
    }
    if (this._spikeIntensity > 0) {
      this._spikeIntensity *= Math.exp(-1.15 * dt);
      if (this._spikeIntensity < 0.01) this._spikeIntensity = 0;
    }

    const cLerp = 1 - Math.exp(-2.2 * dt);
    this._lerpColor(this._glowColor, targetCfg.color, cLerp);
    this._lerpColor(this._accentColor, targetCfg.accent, cLerp);
    for (const key of Object.keys(this._stateValues)) {
      this._stateValues[key] += (targetCfg[key] - this._stateValues[key]) * cLerp;
    }

    const targetOpacity = phaseCfg.glowOpacity + this._spikeIntensity * 0.12;
    const speechBoost = this._conversationState === 'speaking' ? this._rms * 0.10 : 0;
    this._glowOpacity += (targetOpacity + speechBoost - this._glowOpacity) * cLerp;

    const r = Math.round(this._glowColor.r);
    const g = Math.round(this._glowColor.g);
    const b = Math.round(this._glowColor.b);
    this._overlay.style.setProperty('--voice-glow-opacity', this._glowOpacity.toFixed(3));
    this._overlay.style.setProperty('--voice-glow-radius', `${Math.round(340 * phaseCfg.glowScale)}px`);
    this._overlay.style.setProperty('--voice-glow-color', `rgb(${r},${g},${b})`);

    this._manageStars(phaseCfg.starScale);
    this._updateStars(dt);
    this._render();
  }

  _resolveConversationState(state) {
    const voiceState = String(state.voiceState || '').toLowerCase();
    if (voiceState === 'speaking' || state.speaking) return 'speaking';
    if (voiceState === 'recording' || state.recording || state.userSpeaking) return 'recording';
    if (voiceState === 'composing' || state.composing) return 'composing';
    if (voiceState === 'processing' || state.processing) return 'processing';
    if (voiceState === 'disconnected' || voiceState === 'connecting') return 'disconnected';
    if (voiceState === 'listening' || state.listening) return 'listening';
    return 'idle';
  }

  _calcPhase() {
    for (let i = PHASE_THRESHOLDS.length - 1; i >= 0; i--) {
      if (this._elapsed >= PHASE_THRESHOLDS[i]) return i;
    }
    return 0;
  }

  _triggerSpike() {
    this._spikeIntensity = Math.min(1, this._spikeIntensity + 0.34);
  }

  _seedStars() {
    const target = this._targetStarCount(1);
    this._stars = [];
    for (let i = 0; i < target; i += 1) {
      this._stars.push(new StageStar(this._w, this._h, i % 3));
    }
    this._starTarget = target;
  }

  _targetStarCount(scale = 1) {
    const areaScale = Math.min(1.6, Math.max(0.52, (this._w * this._h) / (1280 * 900)));
    return Math.round(90 * areaScale * scale);
  }

  _manageStars(scale) {
    const target = this._targetStarCount(scale);
    this._starTarget = target;
    while (this._stars.length < target) {
      this._stars.push(new StageStar(this._w, this._h, this._stars.length % 3));
    }
    if (this._stars.length > target + 12) {
      this._stars.length = target;
    }
  }

  _updateStars(dt) {
    const state = this._stateValues;
    const speed = 1 + state.drift * 3.2 + this._rms * 1.8;
    const centerX = this._w * 0.5;
    const centerY = this._h * 0.45;

    for (const star of this._stars) {
      const dx = (star.x - centerX) / Math.max(1, this._w);
      const dy = (star.y - centerY) / Math.max(1, this._h);
      const inward = this._conversationState === 'listening' ? -0.28 : 0.08;
      star.x += ((star.drift * 7.5) + dx * state.focus * inward * 38) * dt * speed;
      star.y += ((star.depth * 4.5) + dy * state.focus * inward * 26) * dt * speed;

      if (star.y > this._h + 18 || star.x > this._w + 24 || star.x < -24) {
        star.reset(this._w, this._h, false);
      }
    }
  }

  _render() {
    const ctx = this._ctx;
    if (!ctx || !this._w || !this._h || this._canvas.offsetParent === null) return;

    ctx.clearRect(0, 0, this._w, this._h);
    this._renderVeil(ctx);
    this._renderThreads(ctx);
    this._renderConstellations(ctx);
    this._renderStars(ctx);
  }

  _renderVeil(ctx) {
    const main = this._glowColor;
    const accent = this._accentColor;
    const focus = this._stateValues.focus;
    const t = this._time;

    ctx.save();
    ctx.globalCompositeOperation = 'screen';

    const vertical = ctx.createLinearGradient(0, 0, 0, this._h);
    vertical.addColorStop(0, `rgba(${main.r},${main.g},${main.b},${0.025 + focus * 0.025})`);
    vertical.addColorStop(0.42, 'rgba(255,255,255,0)');
    vertical.addColorStop(1, `rgba(${accent.r},${accent.g},${accent.b},0.020)`);
    ctx.fillStyle = vertical;
    ctx.fillRect(0, 0, this._w, this._h);

    for (let i = 0; i < 3; i += 1) {
      const y = this._h * (0.22 + i * 0.24) + Math.sin(t * (0.14 + i * 0.03)) * 18;
      const band = ctx.createLinearGradient(0, y - 80, this._w, y + 80);
      band.addColorStop(0, 'rgba(255,255,255,0)');
      band.addColorStop(0.38, `rgba(${main.r},${main.g},${main.b},${0.020 + focus * 0.018})`);
      band.addColorStop(0.58, `rgba(${accent.r},${accent.g},${accent.b},0.016)`);
      band.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = band;
      ctx.fillRect(0, y - 90, this._w, 180);
    }

    ctx.restore();
  }

  _renderThreads(ctx) {
    const amount = this._stateValues.threads + this._rms * 0.16;
    if (amount < 0.02) return;

    const main = this._glowColor;
    const accent = this._accentColor;
    const t = this._time;
    const cx = this._w * 0.5;
    const top = this._h * 0.16;
    const bottom = this._h * 0.84;

    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    ctx.lineWidth = 0.8;

    for (let i = -3; i <= 3; i += 1) {
      const offset = i * this._w * 0.07;
      const wave = Math.sin(t * 0.42 + i) * this._w * 0.018;
      const alpha = amount * (0.035 + (3 - Math.abs(i)) * 0.010);
      ctx.strokeStyle = i % 2 === 0
        ? `rgba(${main.r},${main.g},${main.b},${alpha.toFixed(3)})`
        : `rgba(${accent.r},${accent.g},${accent.b},${(alpha * 0.75).toFixed(3)})`;
      ctx.beginPath();
      ctx.moveTo(cx + offset + wave, top);
      ctx.bezierCurveTo(
        cx + offset * 0.55 - wave, this._h * 0.34,
        cx + offset * 0.32 + wave, this._h * 0.58,
        cx + offset * 0.72 - wave, bottom,
      );
      ctx.stroke();
    }

    ctx.restore();
  }

  _renderConstellations(ctx) {
    const amount = this._stateValues.constellation;
    if (amount < 0.03 || this._stars.length < 8) return;

    const main = this._glowColor;
    const candidates = this._stars
      .filter((s) => s.alpha > 0.24 && s.y > this._h * 0.12 && s.y < this._h * 0.78)
      .slice(0, 34);

    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    ctx.lineWidth = 0.7;

    for (let i = 0; i < candidates.length; i += 1) {
      const a = candidates[i];
      for (let j = i + 1; j < Math.min(candidates.length, i + 7); j += 1) {
        const b = candidates[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.hypot(dx, dy);
        const maxDist = Math.min(190, Math.max(90, this._w * 0.14));
        if (dist > maxDist) continue;
        const alpha = amount * (1 - dist / maxDist) * 0.18;
        ctx.strokeStyle = `rgba(${main.r},${main.g},${main.b},${alpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }

    ctx.restore();
  }

  _renderStars(ctx) {
    const main = this._glowColor;
    const accent = this._accentColor;
    const t = this._time;

    ctx.save();
    ctx.globalCompositeOperation = 'screen';
    for (const star of this._stars) {
      const twinkle = 0.72 + Math.sin((t * star.twinkle) + star.phase) * 0.28;
      const alpha = Math.max(0, Math.min(1, star.alpha * twinkle * (0.78 + this._stateValues.focus * 0.35)));
      const color = star.tint === 'accent' ? accent : main;
      const radius = Math.max(0.35, star.radius);

      if (star.radius > 1.05) {
        const glow = ctx.createRadialGradient(star.x, star.y, 0, star.x, star.y, radius * 5);
        glow.addColorStop(0, `rgba(${color.r},${color.g},${color.b},${(alpha * 0.20).toFixed(3)})`);
        glow.addColorStop(1, `rgba(${color.r},${color.g},${color.b},0)`);
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(star.x, star.y, radius * 5, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.fillStyle = `rgba(${color.r},${color.g},${color.b},${alpha.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(star.x, star.y, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  _lerpColor(current, target, t) {
    current.r += (target.r - current.r) * t;
    current.g += (target.g - current.g) * t;
    current.b += (target.b - current.b) * t;
  }

  getState() {
    return {
      elapsed: this._elapsed,
      phase: this._phase,
    };
  }

  dispose() {
    this._disposed = true;
    window.removeEventListener('resize', this._onResize);
    this._resizeObserver?.disconnect();
    this._resizeObserver = null;
    this._stars = [];
    this._ctx?.clearRect(0, 0, this._w || 0, this._h || 0);
  }
}
