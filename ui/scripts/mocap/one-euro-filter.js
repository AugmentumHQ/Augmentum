// ui/scripts/mocap/one-euro-filter.js
// One-Euro Filter — adaptive low-pass filter for noisy input
// Reference: Casiez et al. 2012, CHI '12

export class OneEuroFilter {
  constructor(freq = 30, minCutoff = 1.0, beta = 0.0, dCutoff = 1.0) {
    this._freq = freq;
    this._minCutoff = minCutoff;
    this._beta = beta;
    this._dCutoff = dCutoff;
    this._x = null;
    this._dx = 0;
    this._lastTime = null;
  }

  _alpha(cutoff, dt) {
    const tau = 1.0 / (2 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  filter(value, timestamp) {
    if (this._lastTime === null) {
      this._x = value;
      this._dx = 0;
      this._lastTime = timestamp;
      return value;
    }

    const dt = timestamp - this._lastTime;
    if (dt <= 0) return this._x;
    this._lastTime = timestamp;

    const rawDx = (value - this._x) / dt;
    const alphaDx = this._alpha(this._dCutoff, dt);
    this._dx = alphaDx * rawDx + (1 - alphaDx) * this._dx;

    const cutoff = this._minCutoff + this._beta * Math.abs(this._dx);
    const alpha = this._alpha(cutoff, dt);

    this._x = alpha * value + (1 - alpha) * this._x;
    return this._x;
  }

  reset() {
    this._x = null;
    this._dx = 0;
    this._lastTime = null;
  }

  set minCutoff(v) { this._minCutoff = v; }
  set beta(v) { this._beta = v; }
}

export function createFilterBank(count, axes, freq = 30, minCutoff = 1.0, beta = 0.0) {
  const filters = [];
  for (let i = 0; i < count * axes; i++) {
    filters.push(new OneEuroFilter(freq, minCutoff, beta));
  }
  return {
    filter(values, timestamp) {
      const out = new Array(values.length);
      for (let i = 0; i < values.length; i++) {
        out[i] = filters[i] ? filters[i].filter(values[i], timestamp) : values[i];
      }
      return out;
    },
    reset() { for (const f of filters) f.reset(); },
    setParams(mc, b) { for (const f of filters) { f.minCutoff = mc; f.beta = b; } },
  };
}
