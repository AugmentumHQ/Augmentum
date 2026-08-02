/* ==========================================================================
   Chat Module — TTS (Text-to-Speech)
   Shared singleton TTS system: sentence buffering, audio pipeline,
   progressive streaming, and playback. Only one audio output plays at a time.
   ========================================================================== */

import { getSettings } from '../settings.js';
import { showToast } from '../app.js';
import { AudioBus } from '../audio-bus.js';
import { icons } from './constants.js';
import { unwrapWholeMessageMarkdownFence } from './markdown.js';

// ---------------------------------------------------------------------------
// Module-level state (singleton — only one audio plays at a time)
// ---------------------------------------------------------------------------

// iOS Safari (iPhone + iPadOS 13+) routes <audio> through AudioContext
// destination once createMediaElementSource() binds the element, then
// silences output via suspended-context / silent-switch quirks the
// element's native playback doesn't have. Skip the analyser bind there.
const IS_IOS = /iPad|iPhone|iPod/.test(navigator.userAgent)
            || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

let currentTtsAudio = null;
let currentTtsBtn = null;
let ttsQueue = [];        // Queue of { text, btn } for auto-read
let ttsQueuePlaying = false;
let ttsAborted = false;   // Flag to cancel in-flight fetch
// When true, audio.pause() came from external pause-resume control
// (read-aloud mini-player) rather than playback ending. The pause
// listener in ttsPlayBlob skips cleanup so the in-flight clip's
// promise stays unresolved and the queue advancement stalls until
// resumeTtsPlayback() calls audio.play() again.
let _externallyPaused = false;

// Progressive TTS state (set during streaming when auto-read is on)
let _activeTtsPipeline = null;
let _activeTtsBuffer = null;

let _ttsWarmupTs = 0;     // Last warmup timestamp (5-min cooldown)
const KOKORO_TTS_MAX_CHARS = 260;
const DEFAULT_TTS_MAX_CHARS = 800;

// ---------------------------------------------------------------------------
// Exclusive-playback epoch — the single source of truth for "who owns the
// speakers right now". There are three independent reader loops (the
// progressive TtsAudioPipeline, the one-shot ttsPlayMessage loop, and the
// legacy ttsProcessQueue), all coordinating through the mutable globals above.
// Taking over used to be non-atomic: _installActivePipeline cancelled prior
// PIPELINES but was blind to a running ttsPlayMessage loop, and ttsAborted got
// flipped back to false by whoever started next — reviving a loop that should
// have stayed dead. The result was two readers sounding at once (overlapping /
// out-of-order auto-read).
//
// The invariant we want for TTS is simple: exactly one voice — the most
// recent — owns playback. Each reader claims an epoch when it starts; the act
// of claiming bumps the counter AND hard-stops the previous clip. Every reader
// re-checks its epoch on each step and exits the moment a newer reader claims.
let _playbackEpoch = 0;

function _supersedePlayback() {
  _playbackEpoch += 1;
  // Hard-stop the in-flight clip so the outgoing reader can't keep sounding
  // while the new one plays. Its owning loop wakes on the resolved play
  // promise, sees the epoch moved, and exits.
  if (currentTtsAudio) {
    const audio = currentTtsAudio;
    currentTtsAudio = null;
    _externallyPaused = false;
    try {
      const wasPaused = audio.paused;
      audio.pause();
      // A paused <audio> fires no 'ended'/'pause' event, so its play promise
      // would hang — dispatch 'ended' explicitly to release it. The streaming
      // WAV handle has paused===false (static) so it never hits this; its
      // pause() resolves the stream's own cleanup instead.
      if (wasPaused) audio.dispatchEvent(new Event('ended'));
    } catch { /* stream handle or torn-down element — best effort */ }
  }
  // Reset the outgoing reader's button so a superseded one-shot's speaker
  // icon doesn't stay stuck in the "stop" state (the new reader sets its own).
  if (currentTtsBtn) {
    try {
      currentTtsBtn.innerHTML = icons.speaker;
      currentTtsBtn.classList.remove('active', 'tts-loading');
    } catch { /* button detached from DOM */ }
    currentTtsBtn = null;
  }
  return _playbackEpoch;
}

// ---------------------------------------------------------------------------
// Speech-bus duck — centralized at the playback PRIMITIVE so EVERY caller of
// ttsPlayBlob / ttsPlayWavStream ducks background music (Grove radio, the
// media player, the YouTube ambient orb), not just the high-level chat
// wrappers. The companion's voice path (becca-ptt.js::_flushTtsChunks) and
// voice-orb action acks call ttsPlayBlob DIRECTLY — before this they never
// claimed the bus, so Grove/music kept playing at full volume underneath her
// TTS. (The progressive reader and ttsPlayMessage hold their own longer-lived
// claims spanning fetch stalls; those stay and simply overlap this one — both
// are speech-tier, so the music ducks once either way.)
//
// Ref-counted with a short release grace so a multi-clip utterance (the
// companion's per-sentence flushes chained on _ttsPlayChain, progressive
// chunks) holds ONE continuous duck instead of audibly pumping the music
// down/up on every clip boundary.
let _speechBusClaim = null;
let _speechBusRefs = 0;
let _speechBusReleaseTimer = null;
const SPEECH_BUS_RELEASE_GRACE_MS = 450;

function _acquireSpeechBus() {
  _speechBusRefs += 1;
  if (_speechBusReleaseTimer) {
    clearTimeout(_speechBusReleaseTimer);
    _speechBusReleaseTimer = null;
  }
  if (_speechBusClaim) return;
  try {
    _speechBusClaim = AudioBus.claim({
      id: 'chat-tts-clip',
      tier: 'speech',
      kind: 'speech',
      duck: () => {},     // TTS is the dominator; not duckable itself
      unduck: () => {},
    });
  } catch (_) { _speechBusClaim = null; /* bus unavailable — playback continues */ }
}

function _releaseSpeechBus() {
  if (_speechBusRefs > 0) _speechBusRefs -= 1;
  if (_speechBusRefs > 0) return;
  // Brief grace: between clips of one utterance refs momentarily hit 0.
  // Holding the duck across that gap stops the background music from
  // audibly un-ducking and re-ducking on every sentence boundary.
  if (_speechBusReleaseTimer) clearTimeout(_speechBusReleaseTimer);
  _speechBusReleaseTimer = setTimeout(() => {
    _speechBusReleaseTimer = null;
    if (_speechBusRefs > 0) return;
    try { _speechBusClaim?.release(); } catch (_) { /* best-effort */ }
    _speechBusClaim = null;
  }, SPEECH_BUS_RELEASE_GRACE_MS);
}

// ---------------------------------------------------------------------------
// Analyser tap — exposes a Web Audio analyser node fed by every TTS clip
// played through ttsPlayBlob. The Becca presence widget (and any future
// audio-reactive surface) subscribes to drive lipsync + embodiment. The
// analyser is created lazily on first playback so we don't allocate an
// AudioContext when no one needs one. AudioContext spec only allows one
// MediaElementAudioSource per <audio> element, so each clip creates its
// own source that we connect to the shared analyser.
// ---------------------------------------------------------------------------

let _ttsAudioCtx = null;
let _ttsAnalyserNode = null;

// Companion voice volume — a user gain knob on the TTS output. Kokoro TTS
// runs soft and gets buried under Grove music / host-mode media, so the
// default is a boost (>1). A limiter sits after the gain to catch the peaks
// the boost introduces instead of hard-clipping at the destination. The knob
// lives in the companion widget's timeline panel and persists per-user via
// /api/config/ui (settings.companionVoiceVolume). Graph head is now:
//   per-clip source → gain → limiter → analyser → destination
const DEFAULT_VOICE_GAIN = 2.0;
const MIN_VOICE_GAIN = 0.2;
const MAX_VOICE_GAIN = 5.0;
let _voiceGainMult = DEFAULT_VOICE_GAIN;
let _ttsGainNode = null;
let _ttsLimiterNode = null;

function _clampVoiceGain(v) {
  const n = Number(v);
  if (isNaN(n)) return DEFAULT_VOICE_GAIN;
  return Math.max(MIN_VOICE_GAIN, Math.min(MAX_VOICE_GAIN, n));
}

/**
 * Live-update the companion voice output gain (called by the widget's
 * volume slider). Ramps smoothly so an adjustment mid-utterance doesn't
 * click. Safe to call before the audio graph exists — the value is
 * remembered and applied when the graph is first built.
 */
export function setCompanionVoiceGain(mult) {
  _voiceGainMult = _clampVoiceGain(mult);
  if (_ttsGainNode && _ttsAudioCtx) {
    try {
      _ttsGainNode.gain.setTargetAtTime(_voiceGainMult, _ttsAudioCtx.currentTime, 0.05);
    } catch (_) {
      try { _ttsGainNode.gain.value = _voiceGainMult; } catch (_) { /* node gone */ }
    }
  }
}

function _ensureTtsAudioGraph() {
  if (_ttsAudioCtx) return;
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    _ttsAudioCtx = new Ctx();
    _ttsAnalyserNode = _ttsAudioCtx.createAnalyser();
    _ttsAnalyserNode.fftSize = 2048;
    _ttsAnalyserNode.smoothingTimeConstant = 0.85;

    // Seed the gain from the user's saved setting (falls back to the boost
    // default if settings haven't loaded yet — first TTS play is well after
    // loadVoicePrefsFromServer, so this is normally the real value).
    try {
      const v = getSettings()?.companionVoiceVolume;
      if (v != null) _voiceGainMult = _clampVoiceGain(v);
    } catch (_) { /* defaults apply */ }

    _ttsGainNode = _ttsAudioCtx.createGain();
    _ttsGainNode.gain.value = _voiceGainMult;

    // Safety limiter — a near-brick-wall compressor so a hot boost (or a
    // loud syllable) is tamed instead of clipping the destination.
    _ttsLimiterNode = _ttsAudioCtx.createDynamicsCompressor();
    try {
      _ttsLimiterNode.threshold.value = -6;
      _ttsLimiterNode.knee.value = 0;
      _ttsLimiterNode.ratio.value = 20;
      _ttsLimiterNode.attack.value = 0.003;
      _ttsLimiterNode.release.value = 0.1;
    } catch (_) { /* older impls — node still limits with defaults */ }

    // Chain: gain → limiter → analyser → destination. Per-clip sources
    // connect to the gain head (see ttsPlayBlob / ttsPlayWavStream). Routing
    // analyser → destination keeps audio audible; the element's own native
    // stream is suppressed by the MediaElementSource bind.
    _ttsGainNode.connect(_ttsLimiterNode);
    _ttsLimiterNode.connect(_ttsAnalyserNode);
    _ttsAnalyserNode.connect(_ttsAudioCtx.destination);
  } catch (e) {
    // No AudioContext available (very old browser, or autoplay-blocked
    // before user gesture). Lipsync just won't drive — playback still works.
    console.warn('[tts] audio-graph init failed; lipsync disabled', e);
  }
}

// Auto-read can fire without a recent user gesture (server streams a
// response, TTS plays without the user tapping). On iOS that means the
// AudioContext stays suspended and any Web Audio path is silent. Hook the
// first user interaction with the page and prime the context — once
// primed, iOS treats subsequent audio as gesture-authorized even across
// brief suspensions.
function _primeAudioContextOnce() {
  if (!_ttsAudioCtx) return;
  try {
    if (_ttsAudioCtx.state === 'suspended') {
      _ttsAudioCtx.resume().catch(() => {});
    }
    const src = _ttsAudioCtx.createBufferSource();
    src.buffer = _ttsAudioCtx.createBuffer(1, 1, _ttsAudioCtx.sampleRate);
    src.connect(_ttsAudioCtx.destination);
    src.start(0);
  } catch { /* best-effort prime */ }
}

function _installAudioUnlockOnce() {
  const handler = () => {
    _ensureTtsAudioGraph();
    _primeAudioContextOnce();
  };
  document.addEventListener('pointerdown', handler, { once: true, capture: true });
  document.addEventListener('touchstart', handler, { once: true, capture: true });
  document.addEventListener('keydown', handler, { once: true, capture: true });
}
_installAudioUnlockOnce();

/** Returns the shared TTS analyser node, or null if not yet initialised. */
export function getTtsAnalyser() {
  return _ttsAnalyserNode;
}

/**
 * Wait until the analyser exists, then call cb(analyser). Resolves to the
 * analyser. For consumers that mount before any TTS has fired — they can
 * subscribe early and we'll deliver once the first clip plays.
 */
export function onTtsAnalyserReady(cb) {
  if (_ttsAnalyserNode) { cb(_ttsAnalyserNode); return Promise.resolve(_ttsAnalyserNode); }
  return new Promise((resolve) => {
    const t = setInterval(() => {
      if (_ttsAnalyserNode) {
        clearInterval(t);
        try { cb(_ttsAnalyserNode); } catch (_) {}
        resolve(_ttsAnalyserNode);
      }
    }, 200);
  });
}

// Injected dependency — set via setActiveSessionGetter()
let _getActiveSession = () => null;
// Injected dependency — set via setCharacterVoiceLookup().
// Takes a character name, returns their voice string (or '' if unknown).
let _getCharacterVoiceByName = () => '';

function _setAutoReadStreaming(active) {
  const btn = document.getElementById('auto-read-btn');
  if (btn) btn.classList.toggle('tts-streaming', !!active);
}

function _isLikelyKokoroVoice(voice = '') {
  const raw = String(voice || '').trim();
  if (!raw) return false;
  const [providerId, ...rest] = raw.includes('::') ? raw.split('::') : ['', raw];
  const bare = rest.join('::');
  if (providerId === 'kokoro-builtin' || providerId === 'kokoro-tts') return true;
  if (bare.startsWith('walk:')) return true;
  if (bare.includes('+') && /[a-z][fm]_[a-z0-9_]+/i.test(bare)) return true;
  return /^[abjzfhepi][fm]_[a-z0-9_]+$/i.test(bare);
}

/**
 * Engines whose synthesis path is "PCM-first" — they emit raw audio
 * samples and benefit from WAV response_format instead of MP3, both
 * because the MP3 encode round-trip on the server adds 30-50ms per
 * sentence and because the WAV streaming path on the server emits a
 * header + raw PCM frames as the model generates them (no waiting for
 * the full tensor before any byte hits the wire).
 *
 * Kokoro: covered by ``_isLikelyKokoroVoice`` above.
 * Pocket TTS: provider_id-prefixed only (voice names like ``alba`` are
 *   too generic to regex-match safely).
 */
function _prefersWavResponseFormat(voice = '') {
  if (_isLikelyKokoroVoice(voice)) return true;
  const raw = String(voice || '').trim();
  if (!raw) return false;
  const [providerId] = raw.includes('::') ? raw.split('::') : ['', raw];
  return providerId === 'pockettts-builtin';
}

function _sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function _chatTtsPauseMs(text) {
  const t = String(text || '').trim();
  if (!t) return 0;
  if (/[.]{2,}$/.test(t)) return 320;
  if (/[!?]$/.test(t)) return 180;
  if (/[.;]$/.test(t)) return 150;
  if (/[,;:]$/.test(t)) return 90;
  return 70;
}

function _finishSingleShotTts(btn = null) {
  const target = btn || currentTtsBtn;
  if (target) {
    target.innerHTML = icons.speaker;
    target.classList.remove('active', 'tts-loading');
  }
  if (!target || currentTtsBtn === target) currentTtsBtn = null;
  currentTtsAudio = null;
}

/**
 * Inject the getActiveSession function from the chat module.
 * Called once during init so TTS can resolve voice settings per session.
 */
export function setActiveSessionGetter(fn) {
  _getActiveSession = fn;
}

/**
 * Inject a lookup for a character's voice by display name.
 * Used by group chats to pick the speaker's own voice per turn.
 */
export function setCharacterVoiceLookup(fn) {
  _getCharacterVoiceByName = fn || (() => '');
}

// ---------------------------------------------------------------------------
// TtsSentenceBuffer
// ---------------------------------------------------------------------------

/**
 * Client-side sentence buffer mirroring Python SentenceBuffer.
 * Accumulates streaming tokens, emits at clause/sentence boundaries.
 */
export class TtsSentenceBuffer {
  // clauseTier = how many opener chunks may break on a clause boundary
  // (commas/semis/colons/em-dashes/closers) for fast time-to-first-audio.
  // After that, only sentence-end punctuation [.!?] counts.
  //
  // batchSentencesAfter = number of sentence-terminators to require per
  // chunk once we're past the opener. Pocket TTS keeps prosody continuous
  // across the input it sees in one call, so 2-sentence batches sound
  // markedly smoother than 1-sentence batches — the chunk seams stop
  // landing on prosodic reset points. Set 1 to keep the legacy single-
  // sentence behavior; set 0/falsey for opener-only behavior.
  //
  // schedule = punctuation-preferring fallback threshold per chunk index
  // (last value sticks). When the buffer hits this length without a real
  // sentence/clause emit, Priority 3 cuts at the LAST punctuation seen
  // (falling back to last space only when there's none). Opener chunks
  // sit at 200 so a full first sentence has room to arrive — better an
  // extra beat of silence than a mid-sentence chunk boundary.
  static MODE_PRESETS = {
    clause:   { schedule: [20, 30, 40, 50, 60],     clauseTier: Infinity, batchSentencesAfter: 1 },
    sentence: { schedule: [200, 200, 300, 400, 500], clauseTier: 2,        batchSentencesAfter: 2 },
    full:     { schedule: [],                        clauseTier: 0,        batchSentencesAfter: 1 },
  };

  // Abbreviations that end with period but aren't sentence endings
  static FALSE_ENDINGS = /(?:Mr|Mrs|Ms|Dr|Prof|Jr|Sr|vs|etc|approx|dept|est|vol)\.\s*$/i;

  constructor(mode = 'sentence') {
    const preset = TtsSentenceBuffer.MODE_PRESETS[mode] || TtsSentenceBuffer.MODE_PRESETS.sentence;
    this.schedule = preset.schedule;
    this.clauseTier = preset.clauseTier;
    this.batchSentencesAfter = preset.batchSentencesAfter || 1;
    this.buffer = '';
    this.chunkIndex = 0;
    this.minChars = 10;
    this.inCodeBlock = false;
    this.fenceCount = 0;
  }

  addToken(token) {
    this.buffer += token;

    // Track code fences
    const fences = (this.buffer.match(/```/g) || []).length;
    if (fences > this.fenceCount) {
      this.fenceCount = fences;
      if (fences % 2 === 1) {
        this.inCodeBlock = true;
      } else {
        this.inCodeBlock = false;
        this.buffer = this.buffer.replace(/```[\s\S]*?```/g, '');
        this.fenceCount = (this.buffer.match(/```/g) || []).length;
      }
    }

    if (this.inCodeBlock) return null;
    return this._tryExtract();
  }

  flush() {
    if (this.inCodeBlock) {
      const idx = this.buffer.lastIndexOf('```');
      if (idx >= 0) this.buffer = this.buffer.slice(0, idx);
      this.inCodeBlock = false;
    }
    const text = this.buffer.trim();
    this.buffer = '';
    if (text) {
      this.chunkIndex++;
      return text;
    }
    return null;
  }

  _tryExtract() {
    // Full mode: never emit intermediate chunks
    if (!this.schedule.length) return null;

    const threshold = this.schedule[Math.min(this.chunkIndex, this.schedule.length - 1)];
    const useClause = this.chunkIndex < this.clauseTier;
    // Opener chunks stay on 1-sentence emit for fast TTFA; post-opener
    // chunks aggregate so the TTS engine sees more context per call and
    // prosody flows across sentence boundaries instead of resetting on
    // every chunk seam.
    const sentencesNeeded = this.chunkIndex >= this.clauseTier
      ? this.batchSentencesAfter
      : 1;

    // Priority 1: Nth sentence-ending punctuation
    const sentenceRe = /[.!?]["')]*(?:\s|$)/g;
    let found = 0;
    let endIdx = -1;
    for (const m of this.buffer.matchAll(sentenceRe)) {
      const candidateEnd = m.index + m[0].length;
      const slice = this.buffer.slice(0, candidateEnd);
      // Abbreviations like "Dr." aren't real sentence ends — skip the
      // match without counting it so we keep walking.
      if (TtsSentenceBuffer.FALSE_ENDINGS.test(slice)) continue;
      found++;
      endIdx = candidateEnd;
      if (found >= sentencesNeeded) break;
    }
    if (found >= sentencesNeeded && endIdx > 0) {
      const candidate = this.buffer.slice(0, endIdx).trim();
      if (candidate.length >= this.minChars) {
        this.buffer = this.buffer.slice(endIdx);
        this.chunkIndex++;
        return candidate;
      }
    }

    // Priority 2: Clause breaks (tier 1 only)
    if (useClause && this.buffer.length >= this.minChars) {
      const clauseMatch = this.buffer.match(/[,;:\u2014)\]]+\s/);
      if (clauseMatch && (clauseMatch.index + clauseMatch[0].length) >= this.minChars) {
        const end = clauseMatch.index + clauseMatch[0].length;
        const candidate = this.buffer.slice(0, end).trim();
        if (candidate.length >= this.minChars) {
          this.buffer = this.buffer.slice(end);
          this.chunkIndex++;
          return candidate;
        }
      }
    }

    // Priority 3: Punctuation-preferring safety net at threshold.
    // When the buffer has accumulated past the schedule without a clean
    // sentence/clause emit, prefer the last punctuation+whitespace seen
    // in the buffer so the cut still lands on a natural prosodic edge.
    // Falling all the way back to word boundary is the last resort —
    // runs of text with zero punctuation are unusual in prose and the
    // alternative is letting the buffer grow without bound.
    if (this.buffer.length >= threshold) {
      let cutEnd = -1;
      const punctMatches = [...this.buffer.matchAll(/[.,!?;:—)\]]+\s/g)];
      if (punctMatches.length) {
        const last = punctMatches[punctMatches.length - 1];
        cutEnd = last.index + last[0].length;
      }
      if (cutEnd <= this.minChars) {
        const lastSpace = this.buffer.lastIndexOf(' ', this.buffer.length);
        if (lastSpace > this.minChars) cutEnd = lastSpace;
      }
      if (cutEnd > this.minChars) {
        const chunk = this.buffer.slice(0, cutEnd).trim();
        if (chunk.length >= this.minChars) {
          this.buffer = this.buffer.slice(cutEnd);
          this.chunkIndex++;
          return chunk;
        }
      }
    }

    return null;
  }
}

// ---------------------------------------------------------------------------
// TtsAudioPipeline
// ---------------------------------------------------------------------------

/**
 * Audio pipeline that pre-fetches TTS audio while previous chunks play.
 * Overlaps fetch N+1 with playback of N for minimal gaps.
 */
export class TtsAudioPipeline {
  constructor(voice, speed, isNarrative, btn) {
    this.voice = voice;
    this.speed = speed;
    this.isNarrative = isNarrative;
    this.btn = btn;
    this.queue = [];           // { text, blob?, fetching? }
    this.sealed = false;       // No more chunks will arrive
    this.cancelled = false;
    this.playing = false;
    this.started = false;      // First audio has started
    this._maxLookahead = 2;    // Pre-fetch at most 2 ahead
    this._busClaim = null;     // Holds Grove/media ducking for the whole streamed read
  }

  enqueue(rawText) {
    if (this.cancelled || this.sealed) return;
    if (!rawText || !rawText.trim()) return;
    // Send raw text — server clean_for_tts() handles all cleaning
    this.queue.push({ text: rawText.trim(), blob: null, fetching: false });
    this._prefetch();
    if (!this.playing) this._play();
  }

  seal() {
    this.sealed = true;
    // If nothing is playing and queue is empty, clean up
    if (!this.playing && this.queue.length === 0) this._finish();
  }

  cancel() {
    this.cancelled = true;
    this.queue = [];
    if (currentTtsAudio) {
      const audio = currentTtsAudio;
      // Clear the external-pause gate first so the pause listener
      // doesn't suppress cleanup. If the clip was already paused (user
      // hit pause and then stop), pause() is a no-op so no event
      // fires — dispatch 'ended' explicitly to resolve the in-flight
      // ttsPlayBlob promise and let the pipeline unwind.
      _externallyPaused = false;
      const wasPaused = audio.paused;
      audio.pause();
      currentTtsAudio = null;
      if (wasPaused) {
        try { audio.dispatchEvent(new Event('ended')); } catch { /* */ }
      }
    }
    this._finish();
  }

  _claimSpeechBus() {
    if (this._busClaim) return;
    this._busClaim = AudioBus.claim({
      id: 'chat-tts-progressive',
      tier: 'speech',
      kind: 'speech',
      duck: () => {},
      unduck: () => {},
    });
  }

  _releaseSpeechBus() {
    if (!this._busClaim) return;
    try { this._busClaim.release(); } catch (_) { /* audio bus release is best-effort */ }
    this._busClaim = null;
  }

  _prefetch() {
    let fetching = 0;
    for (const item of this.queue) {
      if (item.result || item.fetching) { fetching++; continue; }
      if (fetching >= this._maxLookahead) break;
      item.fetching = true;
      fetching++;
      ttsFetchAudio(item.text, this.voice, this.speed).then(result => {
        // ttsFetchAudio now returns {kind:'blob'|'stream', ...} or null.
        // Stash on item.result; _play branches via ttsPlayResult.
        item.result = result;
        // Backcompat — if any external code reads item.blob, also expose
        // the blob shape when applicable.
        if (result && result.kind === 'blob') item.blob = result.blob;
        // If we're waiting to play, kick it
        if (!this.playing && !this.cancelled) this._play();
      });
    }
  }

  async _play() {
    if (this.playing || this.cancelled) return;
    this.playing = true;
    // Claim exclusive ownership: stops any prior reader's in-flight clip and
    // marks every older loop stale so it exits instead of overlapping us.
    this._epoch = _supersedePlayback();

    while (this.queue.length > 0 && !this.cancelled && !ttsAborted
           && _playbackEpoch === this._epoch) {
      const item = this.queue[0];

      // Wait for fetch to complete (result arrives async, 30s timeout)
      if (!item.result && item.fetching) {
        if (!item._waitStart) item._waitStart = Date.now();
        if (Date.now() - item._waitStart > 30_000) {
          this.queue.shift();
          continue;
        }
        await new Promise(r => setTimeout(r, 50));
        continue;
      }
      if (!item.result) {
        // Fetch failed — skip this chunk
        this.queue.shift();
        continue;
      }

      // Show playing state on first chunk
      this._claimSpeechBus();
      if (!this.started && this.btn) {
        this.started = true;
        this.btn.classList.add('active');
        this.btn.classList.remove('tts-loading');
        this.btn.innerHTML = icons.speakerOff;
        currentTtsBtn = this.btn;
      }

      this.queue.shift();
      this._prefetch(); // Start fetching next chunks

      if (this.cancelled) break;
      // ttsPlayResult dispatches to ttsPlayBlob (legacy) or
      // ttsPlayWavStream (low-latency streaming) based on item.result.kind.
      await ttsPlayResult(item.result, { text: item.text, source: 'progressive' });
      if (!this.cancelled && !ttsAborted && this.queue.length > 0) {
        await _sleep(_chatTtsPauseMs(item.text));
      }
    }

    this.playing = false;
    // Superseded by a newer reader → this pipeline is done for good. Mark it
    // cancelled so a late enqueue() can't revive its loop, which would re-claim
    // the epoch and ping-pong with the winner.
    if (_playbackEpoch !== this._epoch) this.cancelled = true;
    if (this.sealed || this.cancelled || ttsAborted) this._finish();
  }

  _finish() {
    this._releaseSpeechBus();
    if (_activeTtsPipeline === this) _activeTtsPipeline = null;
    _setAutoReadStreaming(false);
    if (this.btn) {
      this.btn.innerHTML = icons.speaker;
      this.btn.classList.remove('active', 'tts-loading');
    }
    if (currentTtsBtn === this.btn) currentTtsBtn = null;
  }
}

// ---------------------------------------------------------------------------
// Progressive TTS functions
// ---------------------------------------------------------------------------

/**
 * Feed a streaming token into the progressive TTS pipeline.
 * Called from the NDJSON streaming loop in sendMessage().
 */
export function ttsProgressiveFeed(delta) {
  if (!_activeTtsBuffer || !_activeTtsPipeline) return;
  const chunk = _activeTtsBuffer.addToken(delta);
  if (chunk) _activeTtsPipeline.enqueue(chunk);
}

/**
 * Install the progressive pipeline from outside the module.
 * Callers build the buffer + pipeline, then hand ownership here so
 * ttsProgressiveFeed/Finish/Cancel can reach them.
 *
 * Defensive cancel: never leave a previous pipeline live. Two surfaces
 * can race to install (chat auto-read + a Listen button, or two Listen
 * buttons firing in the same tick before either claimed the singleton),
 * and an un-cancelled predecessor becomes an orphan that keeps playing
 * with no cancel path able to reach it — the overlapping-audio /
 * can't-stop bug. Cancelling the incumbent here guarantees the
 * "at most one pipeline" invariant regardless of caller ordering.
 */
export function _installActivePipeline(buffer, pipeline) {
  if (_activeTtsPipeline && _activeTtsPipeline !== pipeline) {
    try { _activeTtsPipeline.cancel(); } catch { /* already torn down */ }
  }
  ttsAborted = false;
  _activeTtsBuffer = buffer;
  _activeTtsPipeline = pipeline;
}

/**
 * Whether pause/resume actually works for a given voice. Pause only
 * works on the HTMLAudioElement (blob) path — the streaming-WAV path
 * schedules AudioBufferSourceNodes that can't be paused without
 * re-scheduling from an offset, so "pause" there is really stop. iOS
 * forces the blob path even for WAV-preferring voices, so it can pause.
 * The read-aloud mini-player uses this to hide a Pause button that
 * would otherwise lie (silently stop with no resume).
 */
export function ttsVoiceSupportsPause(voice = '') {
  return IS_IOS || !_prefersWavResponseFormat(voice);
}

/**
 * Flush remaining buffer and seal the pipeline (no more tokens coming).
 */
export function ttsProgressiveFinish() {
  if (_activeTtsBuffer) {
    const remainder = _activeTtsBuffer.flush();
    if (remainder && _activeTtsPipeline) _activeTtsPipeline.enqueue(remainder);
    _activeTtsBuffer = null;
  }
  if (_activeTtsPipeline) {
    _activeTtsPipeline.seal();
  }
}

/**
 * Cancel progressive TTS (user stopped generation or sent new message).
 */
export function ttsProgressiveCancel() {
  if (_activeTtsPipeline) {
    _activeTtsPipeline.cancel();
    _activeTtsPipeline = null;
  }
  _activeTtsBuffer = null;
  _externallyPaused = false;
  _setAutoReadStreaming(false);
}

/**
 * Hold the in-flight clip without tearing down the pipeline. The
 * audio element's pause event is gated on _externallyPaused so the
 * play promise doesn't resolve — TtsAudioPipeline._play stays awaiting
 * the current clip and won't advance the queue until resume is called.
 */
export function pauseTtsPlayback() {
  if (_externallyPaused) return;
  _externallyPaused = true;
  if (currentTtsAudio) {
    try { currentTtsAudio.pause(); } catch { /* element already gone */ }
  }
  // Tell audio-reactive surfaces the clip went quiet. Matters most for
  // dry clips (iOS) — their synthetic lipsync pulse has no audio to
  // follow, so without this the mouth keeps moving while paused.
  try {
    window.dispatchEvent(new CustomEvent('augmentum:tts-playback', {
      detail: { active: false, analyser: _ttsAnalyserNode, source: 'pause' },
    }));
  } catch (_) { /* non-fatal */ }
}

export function resumeTtsPlayback() {
  if (!_externallyPaused) return;
  _externallyPaused = false;
  if (currentTtsAudio) {
    currentTtsAudio.play().catch(() => { /* browser blocked autoplay; user re-clicks */ });
    try {
      window.dispatchEvent(new CustomEvent('augmentum:tts-playback', {
        detail: {
          active: true,
          analyser: _ttsAnalyserNode,
          // The resumed clip's analyser wiring is whatever it was at
          // play() time; on iOS that's always dry. Recompute cheaply:
          // an <audio> element handle means a ttsPlayBlob clip, and
          // those are dry exactly when IS_IOS (bind skipped) — the
          // desktop bind-failure case is rare enough that resuming it
          // as non-dry just reverts to the pre-pause behavior.
          analyserDry: IS_IOS && !(currentTtsAudio?._isStream),
          source: 'resume',
        },
      }));
    } catch (_) { /* non-fatal */ }
  }
}

export function isTtsPaused() {
  return _externallyPaused;
}

// ---------------------------------------------------------------------------
// Core TTS functions
// ---------------------------------------------------------------------------

/**
 * Warm the TTS connection by sending a tiny request.
 * 5-minute cooldown to avoid wasting API calls.
 */
export async function ttsChatWarmup() {
  const s = getSettings();
  if (!s.voiceAutoRead) return;
  if (Date.now() - _ttsWarmupTs < 300_000) return;
  _ttsWarmupTs = Date.now();
  try {
    await ttsFetchAudio('.', s.voiceDefaultVoice || '', s.voiceSpeed || 1.0);
  } catch { /* best-effort */ }
}

/**
 * Take exclusive ownership of TTS playback for a reader that drives the
 * playback primitives directly (e.g. the companion voice path in
 * becca-ptt.js::_flushTtsChunks) rather than going through the pipeline or
 * ttsPlayMessage. Stops the prior clip and bumps the epoch so any other
 * reader's loop exits — keeping the "exactly one voice" invariant across
 * every surface, not just the chat readers. Returns the claimed epoch.
 */
export function ttsBeginExclusivePlayback() {
  return _supersedePlayback();
}

/**
 * Stop all current TTS playback — progressive, queued, and single-shot.
 */
export function ttsStopCurrent() {
  ttsAborted = true;
  // Bump the epoch so any inline reader loop (ttsPlayMessage) that guards on
  // it exits — ttsAborted alone isn't enough because the next reader to start
  // flips it back to false.
  _playbackEpoch += 1;
  _setAutoReadStreaming(false);
  // The bus claim is released by the playback loop's finally clause once
  // ttsAborted=true propagates. No manual unduck needed here.
  // Cancel progressive pipeline if active
  if (_activeTtsPipeline) {
    _activeTtsPipeline.cancel();
    _activeTtsPipeline = null;
  }
  _activeTtsBuffer = null;
  _externallyPaused = false;
  if (currentTtsAudio) {
    const audio = currentTtsAudio;
    const wasPaused = audio.paused;
    audio.pause();
    currentTtsAudio = null;
    if (wasPaused) {
      try { audio.dispatchEvent(new Event('ended')); } catch { /* */ }
    }
  }
  if (currentTtsBtn) {
    currentTtsBtn.innerHTML = icons.speaker;
    currentTtsBtn.classList.remove('active');
    currentTtsBtn.classList.remove('tts-loading');
    currentTtsBtn = null;
  }
  ttsQueue = [];
  ttsQueuePlaying = false;
}

/**
 * Clean text for TTS — strip markdown, code, links, tables, HTML entities.
 * Narrative mode: handle *action text* and strip character name prefixes.
 */
export function ttsCleanText(text, isNarrative = false) {
  // Whole-message ```md fence: the LLM wrapped its entire reply. Unwrap
  // so the contents do get spoken — the visual renderer also unwraps this
  // case (see renderMarkdown). Smaller ```md / ```stats / ```scene panels
  // are left intact and stripped by the fence regex below — they're meta
  // bookkeeping, not narration, and TTS reading them sounds broken.
  if (isNarrative) {
    text = unwrapWholeMessageMarkdownFence(text);
  }
  let clean = text
    // Fenced code blocks → skip entirely
    .replace(/```[\s\S]*?```/g, '')
    // Indented code blocks (4+ spaces at line start)
    .replace(/^(?:    |\t).+$/gm, '')
    // HTML code/pre tags
    .replace(/<(?:code|pre)[^>]*>[\s\S]*?<\/(?:code|pre)>/gi, '')
    // Inline code → keep if natural language, skip if looks like code
    .replace(/`([^`]+)`/g, (_, c) => /[_./\\{}()<>=;]|^\d+$|^[A-Z_]+$/.test(c) ? '' : c)
    // Images → skip
    .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
    // Links → read link text only
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    // URLs → skip bare URLs
    .replace(/https?:\/\/\S+/g, '')
    // Tables → skip (pipes + dashes)
    .replace(/\|[^\n]+\|/g, '')
    .replace(/[-|:]{3,}/g, '')
    // Horizontal rules
    .replace(/^[-*_]{3,}\s*$/gm, '')
    // HTML entities
    .replace(/&amp;/g, 'and')
    .replace(/&lt;/g, 'less than')
    .replace(/&gt;/g, 'greater than')
    .replace(/&nbsp;/g, ' ')
    .replace(/&#?\w+;/g, '')
    // Blockquotes
    .replace(/^>\s*/gm, '')
    // Headers → treat as sentences
    .replace(/^#{1,6}\s+/gm, '')
    // Bold (**text**, ***text***) → always unwrap (emphasis, not action)
    .replace(/\*{2,3}([^*]+?)\*{2,3}/g, '$1')
    // Single-asterisk *text* can be emphasis or RP actions; speak it by default.
    .replace(/\*([^*\n]+?)\*/g, (_, content) => (
      getSettings().ttsIncludeActionText === false ? '' : content.trim()
    ))
    .replace(/_{1,3}([^_]+)_{1,3}/g, '$1')
    .replace(/~~([^~]+)~~/g, '$1')
    // List markers
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/^\s*\d+\.\s+/gm, '')
    // Remaining markdown punctuation
    .replace(/[[\]()]/g, '')
    // Collapse whitespace
    .replace(/\n{2,}/g, '. ')
    .replace(/\n/g, ' ')
    .replace(/\s{2,}/g, ' ')
    // Clean up double punctuation from stripped blocks
    .replace(/\.{2,}/g, '.')
    .replace(/\.\s*\./g, '.')
    .trim();

  if (isNarrative) {
    // Strip "CharacterName:" prefix at start of message
    clean = clean.replace(/^\w[\w\s]{0,30}:\s*/, '');
  }

  return clean;
}

/**
 * Split text into chunks at sentence boundaries for chunked TTS.
 * Each chunk targets roughly maxChars characters.
 */
export function ttsSplitChunks(text, maxChars = DEFAULT_TTS_MAX_CHARS) {
  const input = String(text || '').trim();
  if (!input) return [];
  if (input.length <= maxChars) return [input];

  const chunks = [];
  const sentences = input.match(/[^.!?]+[.!?]+["')\]]*\s*|[^.!?]+$/g) || [input];
  let current = '';

  const pushLong = (segment) => {
    let rest = segment.trim();
    while (rest.length > maxChars) {
      const windowText = rest.slice(0, maxChars + 1);
      const minBreak = Math.floor(maxChars * 0.45);
      let cut = -1;
      let match;

      const breakPatterns = [
        /[.!?]+["')\]]*\s/g,
        /[,;:]\s/g,
        /\s[-\u2013\u2014]\s/g,
      ];
      for (const pattern of breakPatterns) {
        pattern.lastIndex = 0;
        while ((match = pattern.exec(windowText)) !== null) {
          const end = match.index + match[0].length;
          if (end >= minBreak) cut = end;
        }
        if (cut >= minBreak) break;
      }

      if (cut < minBreak) {
        cut = windowText.lastIndexOf(' ');
      }
      if (cut < minBreak) {
        cut = maxChars;
      }

      chunks.push(rest.slice(0, cut).trim());
      rest = rest.slice(cut).trim();
    }
    if (rest) chunks.push(rest);
  };

  for (const sentence of sentences) {
    const trimmed = sentence.trim();
    if (!trimmed) continue;
    if (trimmed.length > maxChars) {
      if (current.trim()) {
        chunks.push(current.trim());
        current = '';
      }
      pushLong(trimmed);
    } else if (current.length + trimmed.length + 1 > maxChars && current.trim()) {
      chunks.push(current.trim());
      current = trimmed;
    } else {
      current = current ? `${current} ${trimmed}` : trimmed;
    }
  }
  if (current.trim()) chunks.push(current.trim());

  return chunks;
}

/**
 * Fetch TTS audio for a text chunk. Returns a Blob or null on failure.
 * Retries once on failure before giving up.
 */
export async function ttsFetchAudio(text, voice, speed) {
  const wantsWav = _prefersWavResponseFormat(voice);
  // We can consume a true byte stream when we have ReadableStream
  // getReader AND we're NOT on iOS — iOS Safari's AudioContext is too
  // unreliable for scheduled-buffer playback (ttsPlayWavStream forces
  // a blob fallback there anyway). Tell the server which header shape
  // to emit so its choice matches what we can decode:
  //   stream=1  → one sentinel-size WAV header + PCM as the model
  //               produces it (live streaming, first byte ~100 ms)
  //   stream=0  → one buffered WAV with real sizes at the end (waits
  //               for full synthesis but plays in every client path
  //               including iOS Audio elements)
  const canStream = wantsWav
    && !IS_IOS
    && typeof ReadableStream !== 'undefined'
    && typeof window !== 'undefined'
    && typeof window.fetch === 'function';

  const body = {
    input: text,
    response_format: wantsWav ? 'wav' : 'mp3',
    speed,
  };

  // Voice may be "provider_id::voice_id" — split and route to correct provider
  let endpoint = '/v1/audio/speech';
  if (voice && voice.includes('::')) {
    let [providerId, voiceId] = voice.split('::', 2);
    // Resolve provider aliases — kokoro-tts sidecar falls back to kokoro-builtin
    if (providerId === 'kokoro-tts') providerId = 'kokoro-builtin';
    body.voice = voiceId;
    endpoint = `/api/audio/speech?provider_id=${encodeURIComponent(providerId)}`;
  } else if (voice) {
    body.voice = voice;
  }
  if (canStream) {
    endpoint += (endpoint.includes('?') ? '&' : '?') + 'stream=1';
  }

  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const resp = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (resp.ok) {
        // Stream path is chosen by what we ASKED the server for, not
        // by sniffing Transfer-Encoding — HTTP/2 hides that header
        // from JS but the body still streams via getReader. Trusting
        // the request-side decision matches the server's emission.
        const ct = (resp.headers.get('content-type') || '').toLowerCase();
        if (canStream
            && ct.startsWith('audio/wav')
            && resp.body
            && typeof resp.body.getReader === 'function') {
          return { kind: 'stream', response: resp };
        }
        const blob = await resp.blob();
        return { kind: 'blob', blob };
      }
      // On last attempt, surface the failure with the HTTP status so an
      // empty body doesn't show as a bare "TTS failed: ". Body-text can
      // be empty when the upstream connection drops mid-response (proxy
      // returns 502 with no body) or when middleware bounces the request
      // before our handler can stamp a detail string.
      if (attempt === 1) {
        let detail = '';
        try { detail = (await resp.text()) || ''; } catch { /* body unreadable */ }
        detail = detail.trim() || resp.statusText || 'no detail';
        console.warn('[tts] fetch failed', resp.status, resp.statusText, detail.slice(0, 200));
        showToast(`TTS failed (${resp.status}): ${detail.slice(0, 100)}`, 'error');
      }
    } catch (err) {
      // AbortError is the normal path when the user starts a new
      // read-aloud or hits stop — don't yell at them about it.
      if (err?.name === 'AbortError') return null;
      if (attempt === 1) {
        console.warn('[tts] fetch threw', err);
        showToast(`TTS request failed: ${err?.message || err}`, 'error');
      }
    }
  }
  return null;
}

/**
 * Play whatever ``ttsFetchAudio`` returned. Routes streaming-WAV
 * responses through ``ttsPlayWavStream`` (low-latency Web Audio
 * scheduling) and blob responses through ``ttsPlayBlob`` (HTMLAudioElement).
 * Tolerant of the legacy ``Blob`` return shape so older call sites
 * still work without modification.
 */
export function ttsPlayResult(result, meta = {}) {
  if (!result) return Promise.resolve();
  // Legacy callers that still hand us a raw Blob.
  if (typeof Blob !== 'undefined' && result instanceof Blob) {
    return ttsPlayBlob(result, meta);
  }
  if (result.kind === 'stream' && result.response) {
    return ttsPlayWavStream(result.response, meta);
  }
  if (result.kind === 'blob' && result.blob) {
    return ttsPlayBlob(result.blob, meta);
  }
  return Promise.resolve();
}

/**
 * Play a single audio blob. Returns a promise that resolves when playback ends.
 *
 * Routes through the shared TTS analyser (lazily created on first call) so
 * audio-reactive surfaces (Becca presence widget, voice orb, future
 * embodiment) can drive lipsync + amplitude reactions off the live stream.
 * Playback fidelity is unchanged — the analyser node is upstream of the
 * AudioContext destination, so the user hears the same audio they would
 * have without it.
 */
export function ttsPlayBlob(blob, meta = {}) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    currentTtsAudio = audio;
    let finished = false;

    // Duck background music for the whole clip. Done here (the primitive)
    // rather than only in the chat wrappers so the companion / voice-orb
    // paths that call ttsPlayBlob directly also duck. Released in cleanup().
    _acquireSpeechBus();

    // Late-init the audio graph the first time we actually play something.
    // Has to happen here (not at module load) because some browsers gate
    // AudioContext construction on a user gesture.
    _ensureTtsAudioGraph();

    // Bind this element to the shared analyser. createMediaElementSource
    // can only be called once per element — since we create a fresh
    // <audio> per clip, that's always the first (and only) call. The
    // element's native output is suppressed by the bind, so the only
    // path to speakers is through analyser → destination.
    // Skip analyser bind on iOS — createMediaElementSource suppresses the
    // element's native output and routes everything through the
    // AudioContext destination, which is unreliable on iOS Safari
    // (suspended contexts + silent-switch quirks). Native <audio>
    // playback is the safe path. The avatar's amplitude-driven lipsync
    // loses input on iPad/iPhone; the phoneme-schedule path is unaffected.
    let source = null;
    if (!IS_IOS && _ttsAudioCtx && _ttsAnalyserNode) {
      try {
        // Some browsers suspend the context until a user gesture; resume
        // on first play so analyser data actually flows.
        if (_ttsAudioCtx.state === "suspended") {
          _ttsAudioCtx.resume().catch(() => {});
        }
        source = _ttsAudioCtx.createMediaElementSource(audio);
        // Connect to the gain head so the user's voice-volume knob applies.
        source.connect(_ttsGainNode || _ttsAnalyserNode);
      } catch (e) {
        // If the bind fails (audio element CORS / decoded format / repeat
        // bind on a recycled element), fall back to native playback so
        // the user still hears something — they just lose lipsync.
        console.warn('[tts] analyser bind failed — playback continues, lipsync disabled for this clip', e);
        source = null;
      }
    }

    // Notify audio-reactive surfaces (Becca presence widget, future
    // embodiment hosts) that a clip is now playing. Detail carries the
    // analyser ref so subscribers don't have to import this module.
    // ``analyserDry`` flags clips whose audio does NOT flow through the
    // analyser (iOS native-playback path, or a failed bind) — the
    // avatar's lipsync uses it to switch to a synthetic dry-pulse
    // instead of reading a silent analyser. Before this flag the start
    // event was skipped entirely for dry clips, so the widget never
    // learned TTS was playing and the mouth stayed closed for the
    // whole utterance (every utterance, on iPad).
    const _emitLifecycle = (active) => {
      try {
        console.info('[tts] lifecycle event',
          { active, hasSource: !!source, hasAnalyser: !!_ttsAnalyserNode });
        window.dispatchEvent(new CustomEvent('augmentum:tts-playback', {
          detail: {
            active,
            analyser: _ttsAnalyserNode,
            analyserDry: !source,
            text: String(meta.text || ''),
            source: String(meta.source || 'tts'),
          },
        }));
      } catch (_) { /* listener errors are non-fatal — playback continues */ }
    };

    const cleanup = () => {
      if (finished) return;
      finished = true;
      try { source?.disconnect(); } catch (_) {}
      URL.revokeObjectURL(url);
      if (currentTtsAudio === audio) currentTtsAudio = null;
      _releaseSpeechBus();
      _emitLifecycle(false);
      resolve();
    };

    audio.addEventListener('ended', cleanup);
    audio.addEventListener('pause', () => {
      // External pause-resume control holds the clip; skip cleanup so
      // the play promise stays unresolved and the queue stalls cleanly.
      if (_externallyPaused && currentTtsAudio === audio) return;
      if (ttsAborted || currentTtsAudio !== audio) cleanup();
    });
    audio.addEventListener('error', () => {
      // Only treat as error if we didn't already finish or get aborted
      if (!finished && currentTtsAudio === audio) {
        cleanup();
      }
    });

    _emitLifecycle(true);
    audio.play().catch(cleanup);
  });
}

/**
 * Stream a chunked WAV response into Web Audio as bytes arrive.
 *
 * The studio-professional path for PCM-first TTS engines (Pocket TTS,
 * any future Kyutai/Moshi streaming TTS): the server emits a 44-byte
 * WAV header followed by raw PCM int16 frames as the model produces
 * them. We parse the header from the first chunk, then schedule each
 * subsequent batch of samples as an AudioBufferSourceNode contiguous
 * with the previous one. First audio reaches the user's ear roughly
 * when the *first* PCM chunk arrives — typically ~100-200ms after the
 * request fires, instead of waiting for the full sentence (~500ms+).
 *
 * Routes through the shared analyser node so avatar lipsync /
 * amplitude reactions still work. Falls back to {@link ttsPlayBlob}
 * on any error (header parse failure, decode failure, audio graph
 * unavailable, browser missing ReadableStream support).
 *
 * Limitations vs ttsPlayBlob:
 *   * No pause/resume on the streaming path — once a buffer source is
 *     scheduled it can't be paused without re-scheduling from offset.
 *     Cancel (stop) DOES work and is the primary control users need.
 *   * No iOS support — iOS Safari's AudioContext suspends aggressively
 *     and won't reliably play scheduled buffers. iOS routes through the
 *     blob path automatically via the IS_IOS gate.
 */
export function ttsPlayWavStream(response, meta = {}) {
  return new Promise(async (resolve) => {
    // iOS / no-ReadableStream / no-AudioContext: fall back to blob.
    if (IS_IOS || !response.body || typeof response.body.getReader !== 'function') {
      try {
        const blob = await response.blob();
        return ttsPlayBlob(blob, meta).then(resolve);
      } catch (_) { return resolve(); }
    }

    _ensureTtsAudioGraph();
    const ctx = _ttsAudioCtx;
    const analyser = _ttsAnalyserNode;
    // Scheduled PCM buffers connect to the gain head (user voice-volume knob);
    // analyser stays the lipsync tap reference passed in the lifecycle event.
    const inputNode = _ttsGainNode || _ttsAnalyserNode;
    if (!ctx) {
      // No audio graph — fall back so the user still hears audio.
      try {
        const blob = await response.blob();
        return ttsPlayBlob(blob, meta).then(resolve);
      } catch (_) { return resolve(); }
    }

    if (ctx.state === 'suspended') {
      try { await ctx.resume(); } catch (_) {}
    }

    const reader = response.body.getReader();
    let pending = new Uint8Array(0);
    let sampleRate = 24000;
    let headerParsed = false;
    let scheduleAt = ctx.currentTime + 0.05; // ~50ms lead so first source can start cleanly
    const activeSources = new Set();
    let cancelled = false;
    let lifecycleStarted = false;

    // Stand-in for currentTtsAudio so existing cancel paths (pipeline
    // .cancel(), ttsStopCurrent, etc.) can stop us via the same idiom.
    const handle = {
      _isStream: true,
      paused: false,
      pause() {
        cancelled = true;
        for (const s of activeSources) { try { s.stop(); } catch (_) {} }
        activeSources.clear();
        try { reader.cancel(); } catch (_) {}
      },
    };
    currentTtsAudio = handle;

    // Duck background music for the streamed utterance. The early fallback
    // branches above route through ttsPlayBlob, which acquires on its own,
    // so this only covers the native-streaming path. Released in cleanup().
    _acquireSpeechBus();

    const emitLifecycle = (active) => {
      try {
        window.dispatchEvent(new CustomEvent('augmentum:tts-playback', {
          detail: {
            active,
            analyser,
            text: String(meta.text || ''),
            source: String(meta.source || 'tts'),
          },
        }));
      } catch (_) { /* listener errors are non-fatal — playback continues */ }
    };

    const cleanup = () => {
      if (currentTtsAudio === handle) currentTtsAudio = null;
      _releaseSpeechBus();
      if (lifecycleStarted) emitLifecycle(false);
      resolve();
    };

    try {
      while (!cancelled && !ttsAborted) {
        const { value, done } = await reader.read();
        if (done) break;
        if (!value || value.length === 0) continue;

        // Accumulate incoming bytes — chunks may arrive arbitrarily
        // small (one TCP packet) so we buffer until we can parse the
        // WAV header and schedule full int16 frames (need 2-byte
        // alignment).
        const merged = new Uint8Array(pending.length + value.length);
        merged.set(pending);
        merged.set(value, pending.length);
        pending = merged;

        if (!headerParsed) {
          if (pending.length < 44) continue;
          // Sample rate lives at offset 24 (little-endian uint32) in
          // the canonical RIFF/WAVE header. We don't validate magic
          // bytes — the server is trusted to emit a correct header.
          const dv = new DataView(pending.buffer, pending.byteOffset);
          sampleRate = dv.getUint32(24, true) || 24000;
          pending = pending.slice(44);
          headerParsed = true;
        }

        // Schedule whole-sample chunks; defer odd trailing byte for
        // the next read.
        if (pending.length >= 2) {
          const numSamples = Math.floor(pending.length / 2);
          const buf = ctx.createBuffer(1, numSamples, sampleRate);
          const channel = buf.getChannelData(0);
          // Build an Int16Array view over the aligned portion of pending.
          // pending.byteOffset may be nonzero; the view honours it.
          const pcm = new Int16Array(
            pending.buffer, pending.byteOffset, numSamples,
          );
          for (let i = 0; i < numSamples; i++) {
            channel[i] = pcm[i] / 32768;
          }
          const src = ctx.createBufferSource();
          src.buffer = buf;
          if (inputNode) {
            try { src.connect(inputNode); } catch (_) { src.connect(ctx.destination); }
          } else {
            src.connect(ctx.destination);
          }
          const startTime = Math.max(scheduleAt, ctx.currentTime + 0.005);
          src.start(startTime);
          scheduleAt = startTime + numSamples / sampleRate;
          activeSources.add(src);
          src.onended = () => activeSources.delete(src);

          if (!lifecycleStarted) {
            lifecycleStarted = true;
            emitLifecycle(true);
          }

          pending = pending.slice(numSamples * 2);
        }
      }
    } catch (err) {
      // Any error during streaming — cancel cleanly + resolve so the
      // pipeline can advance to the next sentence.
      console.warn('[tts] wav stream error', err);
    }

    // Wait until the final scheduled sample has played out.
    const playbackTail = Math.max(0, (scheduleAt - ctx.currentTime) * 1000);
    if (playbackTail > 0 && !cancelled) {
      await new Promise(r => setTimeout(r, playbackTail));
    }
    cleanup();
  });
}

/**
 * Main TTS entry point — clean text, chunk if needed, fetch and play sequentially.
 */
export async function ttsPlayMessage(text, btn, opts = {}) {
  // If already playing this message, stop it
  if (currentTtsBtn === btn && currentTtsAudio) {
    ttsStopCurrent();
    return;
  }

  // Stop any current playback
  ttsStopCurrent();
  ttsAborted = false;
  // Claim exclusive ownership for this one-shot read. If a newer reader (a
  // story-mode auto-read pipeline, another speak request) starts while we're
  // mid-read, the epoch moves and our loop below bails instead of overlapping.
  const _epoch = _supersedePlayback();

  const s = getSettings();
  const speed = s.voiceSpeed || 1.0;
  const session = _getActiveSession();
  const isNarrative = session && session.mode === 'narrative';
  // Prefer character voice for narrative sessions, fall back to default.
  // Read live from session (updated when character voice changes mid-session).
  // Voice resolution priority:
  //   1. Per-turn speaker voice (group chats — look up speaker's character card)
  //   2. Session's bound character voice (solo chats, narrative)
  //   3. Companion voice (opts.companion) — for the companion's own
  //      client-spoken output (action acks via _speakAck), so she sounds
  //      the same confirming an action as she does replying. Mirrors the
  //      server's _companion_voice_for_user priority (companionVoice →
  //      default). Provider-agnostic: companionVoice is whatever
  //      "<provider>::<voice>" the user picked; this code never assumes
  //      a specific engine. Unset → falls through to the app default
  //      below, i.e. exactly the prior behaviour (no regression).
  //   4. App-level default voice
  let voice = '';
  if (isNarrative && opts.speakerName) {
    voice = _getCharacterVoiceByName(opts.speakerName) || '';
  }
  if (!voice && isNarrative && session?.characterVoice) {
    voice = session.characterVoice;
  }
  if (!voice && opts.companion) {
    voice = s.companionVoice || '';
  }
  if (!voice) voice = s.voiceDefaultVoice || '';

  if (!text || !text.trim()) return;
  // Clean client-side BEFORE chunking so multi-word *actions* and other
  // span-based markdown are removed as whole units. Splitting first would
  // break asterisk/quote spans across chunk boundaries, causing the server's
  // span regexes to miss them — resulting in stage directions being spoken
  // interleaved with dialogue.
  const cleanText = ttsCleanText(text, isNarrative);

  // Set loading state
  if (btn) {
    btn.classList.add('active', 'tts-loading');
    btn.innerHTML = '<span class="tts-spinner"></span>';
    currentTtsBtn = btn;
  }

  const chunkMax = _isLikelyKokoroVoice(voice)
    ? KOKORO_TTS_MAX_CHARS
    : DEFAULT_TTS_MAX_CHARS;
  const chunks = ttsSplitChunks(cleanText, chunkMax);

  // Prefetch: start fetching next chunk while current one plays
  const blobCache = new Array(chunks.length).fill(null);
  const fetchChunk = (idx) => {
    if (idx >= chunks.length || blobCache[idx] || ttsAborted) return;
    blobCache[idx] = ttsFetchAudio(chunks[idx], voice, speed);
  };

  // Claim the bus at 'speech' tier — ducks Grove and any playing media
  // until we release. AudioBus is a noop if this TTS is the only source.
  const busClaim = AudioBus.claim({
    id: 'chat-tts',
    tier: 'speech',
    kind: 'speech',  // drives lipsync via the TTS analyser path
    duck: () => {},   // TTS is the dominator; not duckable itself for v1
    unduck: () => {},
  });

  // Kick off first two fetches immediately
  fetchChunk(0);
  fetchChunk(1);

  try {
    for (let i = 0; i < chunks.length; i++) {
      if (ttsAborted || _playbackEpoch !== _epoch) return;

      // ttsFetchAudio resolves to a wrapper — {kind:'blob',blob} for
      // buffered responses, {kind:'stream',response} for chunked WAV,
      // or a raw Blob for the legacy path. ttsPlayResult does the
      // dispatch. Calling ttsPlayBlob directly on the wrapper is the
      // bug that surfaced as "Failed to execute 'createObjectURL' on
      // 'URL': Overload resolution failed" — createObjectURL requires
      // a Blob and the wrapper object isn't one.
      const result = await blobCache[i];
      if (_playbackEpoch !== _epoch) return;  // a newer reader took over
      if (!result || ttsAborted) {
        ttsStopCurrent();
        return;
      }

      // Switch from loading to playing state on first chunk
      if (i === 0 && btn) {
        btn.classList.remove('tts-loading');
        btn.innerHTML = icons.speakerOff;
      }

      // Prefetch next chunk while this one plays
      fetchChunk(i + 2);

      await ttsPlayResult(result, { text: chunks[i], source: 'message' });

      if (ttsAborted || _playbackEpoch !== _epoch) return;
      if (i < chunks.length - 1) {
        await _sleep(_chatTtsPauseMs(chunks[i]));
      }
    }
  } finally {
    busClaim.release();
  }
  _finishSingleShotTts(btn);
}

/**
 * Queue a message for auto-read. Plays sequentially if multiple arrive.
 */
export function ttsQueueAutoRead(text, btn) {
  ttsQueue.push({ text, btn });
  if (!ttsQueuePlaying) ttsProcessQueue();
}

/**
 * Process queued TTS messages one at a time.
 */
export async function ttsProcessQueue() {
  ttsQueuePlaying = true;
  while (ttsQueue.length > 0) {
    const { text, btn } = ttsQueue.shift();
    if (ttsAborted) break;
    await ttsPlayMessage(text, btn);
  }
  ttsQueuePlaying = false;
}

/**
 * Update the character voice setting on a session object.
 */
export function updateCharacterVoice(sessionId, voice, sessionStore) {
  const session = sessionStore.get(sessionId);
  if (session) {
    session.characterVoice = voice;
    sessionStore.save(sessionId);
  }
}
