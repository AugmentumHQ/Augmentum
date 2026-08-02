/* ui/scripts/voice/batch-stt.js
 *
 * Hold-to-talk utterance capture via MediaRecorder → server batch STT.
 *
 * This is the SAME clean, on-device path the chat mic button's fallback
 * uses (ui/scripts/chat/stt.js): record the held utterance as a compressed
 * blob, POST it to /v1/audio/transcriptions, where the server transcodes
 * via ffmpeg to 16 kHz and runs Moonshine in BATCH. No streaming PCM, no
 * client/server VAD guessing the boundaries, no manual JS resampler — the
 * button press IS the boundary. Used by the PTT paths (companion widget +
 * voice-call modal) where the user defines the utterance by holding.
 *
 * Stays 100% local: the audio goes to our own server, not a vendor cloud
 * (unlike webkitSpeechRecognition, which streams to Google/Apple).
 *
 * Usage:
 *   const rec = createUtteranceRecorder(micStream);
 *   rec.start();                       // on press
 *   const text = await rec.stop();     // on release → transcript ('' if none)
 */

// Safari/iOS can't do webm/opus — it records audio/mp4. Pick what the
// platform supports; the server batch endpoint handles webm/ogg/m4a/mp3
// (it sniffs magic bytes + the filename hint, then ffmpeg-transcodes).
function _pickMime() {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/mp4',
    'audio/mpeg',
  ];
  for (const m of candidates) {
    try {
      if (typeof MediaRecorder !== 'undefined' && MediaRecorder.isTypeSupported(m)) return m;
    } catch (_) { /* isTypeSupported can throw on odd inputs */ }
  }
  return '';  // let the browser pick its default
}

function _extFor(mime) {
  if (mime.startsWith('audio/mp4')) return 'm4a';
  if (mime.startsWith('audio/mpeg')) return 'mp3';
  if (mime.startsWith('audio/ogg')) return 'ogg';
  return 'webm';
}

/**
 * @param {MediaStream} stream  An already-acquired mic stream to record from.
 * @returns {{ start: () => void, stop: () => Promise<string>, cancel: () => void }}
 */
export function createUtteranceRecorder(stream) {
  let recorder = null;
  let chunks = [];
  const mime = _pickMime();

  return {
    start() {
      chunks = [];
      try {
        recorder = mime
          ? new MediaRecorder(stream, { mimeType: mime })
          : new MediaRecorder(stream);
      } catch (err) {
        // Some browsers reject the explicit mimeType — retry with default.
        recorder = new MediaRecorder(stream);
      }
      recorder.addEventListener('dataavailable', (e) => {
        if (e.data && e.data.size > 0) chunks.push(e.data);
      });
      recorder.start();
    },

    /** Stop, upload, and resolve to the transcript ('' when nothing heard). */
    async stop() {
      if (!recorder) return '';
      const stopped = new Promise((resolve) => {
        recorder.addEventListener('stop', resolve, { once: true });
      });
      try { recorder.stop(); } catch (_) { /* already stopped */ }
      await stopped;
      const rec = recorder;
      recorder = null;
      const blobType = (rec && rec.mimeType) ? rec.mimeType.split(';')[0] : (mime.split(';')[0] || 'audio/webm');
      if (!chunks.length) return '';
      const blob = new Blob(chunks, { type: blobType });
      chunks = [];
      // Guard against empty / sub-frame captures (button tapped, not held).
      if (blob.size < 1200) return '';

      const fd = new FormData();
      fd.append('file', blob, `recording.${_extFor(blobType)}`);
      const resp = await fetch('/v1/audio/transcriptions', {
        method: 'POST',
        body: fd,
        credentials: 'same-origin',
      });
      if (!resp.ok) {
        throw new Error(`batch STT ${resp.status}: ${(await resp.text()).slice(0, 120)}`);
      }
      const data = await resp.json();
      return String(data.text || '').trim();
    },

    /** Abort without transcribing (e.g. silence/cancel). */
    cancel() {
      try { recorder?.stop(); } catch (_) {}
      recorder = null;
      chunks = [];
    },
  };
}
