// ui/scripts/connect/voice-recorder.js
//
// Thin MediaRecorder wrapper for Connect voice messages. Captures mic
// audio into a Blob the caller uploads through the normal attachment
// pipeline (so it renders as a playable audio bubble). No UI here — the
// composer owns the recording bar; this just runs the capture + ticks
// elapsed seconds.

export function isVoiceRecordingSupported() {
  return !!(
    navigator.mediaDevices
    && typeof navigator.mediaDevices.getUserMedia === 'function'
    && typeof window.MediaRecorder !== 'undefined'
  );
}

// Most-preferred first. Opus-in-webm/ogg is small + widely decodable;
// mp4/aac is the Safari fallback.
const _MIME_CANDIDATES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/ogg;codecs=opus',
  'audio/mp4',
  'audio/aac',
];

/**
 * Begin recording. Resolves to a handle with `stop(cancel)`:
 *   stop(false) → { blob, durationSec, mime }   (send it)
 *   stop(true)  → null                          (discard)
 * Rejects if mic permission is denied / unavailable.
 */
export async function startVoiceRecording({ onTick } = {}) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

  let mime = '';
  const supports = window.MediaRecorder.isTypeSupported;
  if (typeof supports === 'function') {
    for (const t of _MIME_CANDIDATES) {
      if (supports(t)) { mime = t; break; }
    }
  }

  const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  const chunks = [];
  rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };

  const startedAt = performance.now();
  let tickTimer = null;
  if (typeof onTick === 'function') {
    onTick(0);
    tickTimer = setInterval(() => onTick((performance.now() - startedAt) / 1000), 200);
  }
  rec.start();

  function cleanup() {
    if (tickTimer) { clearInterval(tickTimer); tickTimer = null; }
    for (const tr of stream.getTracks()) { try { tr.stop(); } catch (_) {} }
  }

  function stop(cancel = false) {
    return new Promise((resolve) => {
      const finish = () => {
        cleanup();
        if (cancel) { resolve(null); return; }
        const fullType = rec.mimeType || mime || 'audio/webm';
        const blob = new Blob(chunks, { type: fullType });
        resolve({
          blob,
          durationSec: (performance.now() - startedAt) / 1000,
          mime: fullType.split(';')[0],
        });
      };
      if (rec.state === 'inactive') { finish(); return; }
      rec.onstop = finish;
      try { rec.stop(); } catch (_) { finish(); }
    });
  }

  return { stop };
}
