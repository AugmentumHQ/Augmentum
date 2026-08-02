"""Diagnostic script to test Moonshine batch transcription step by step.

Run INSIDE the container:
    docker exec -it augmentum-augmentum-1 python tests/test_moonshine_batch.py

Tests each layer independently to find the exact failure point.
"""
from __future__ import annotations

import sys
import time
import numpy as np


def generate_test_tone(duration_s: float = 2.0, freq: float = 440.0, sr: int = 16000) -> np.ndarray:
    """Generate a sine wave tone as float32 [-1, 1]."""
    t = np.arange(int(sr * duration_s)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def generate_speech_like(duration_s: float = 3.0, sr: int = 16000) -> np.ndarray:
    """Generate speech-like audio (varying frequency, amplitude modulation)."""
    t = np.arange(int(sr * duration_s)) / sr
    # Varying frequency to simulate speech formants
    freq = 200 + 100 * np.sin(2 * np.pi * 3 * t)  # 3 Hz modulation
    phase = np.cumsum(freq / sr) * 2 * np.pi
    signal = 0.3 * np.sin(phase)
    # Add harmonics
    signal += 0.15 * np.sin(phase * 2)
    signal += 0.1 * np.sin(phase * 3)
    # Amplitude modulation (syllable-like)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)
    return (signal * envelope).astype(np.float32)


def test_step(name: str):
    print(f"\n{'='*60}")
    print(f"  TEST: {name}")
    print(f"{'='*60}")


def main():
    # ---------------------------------------------------------------
    # Step 1: Can we import moonshine_voice?
    # ---------------------------------------------------------------
    test_step("Import moonshine_voice")
    try:
        from moonshine_voice import Transcriber, get_model_for_language, TranscriptEventListener
        print("OK: moonshine_voice imported")
        print(f"   Transcriber: {Transcriber}")
        print(f"   get_model_for_language: {get_model_for_language}")
    except ImportError as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 2: Can we resolve the model?
    # ---------------------------------------------------------------
    test_step("Resolve model path")
    try:
        resolved_path, resolved_arch = get_model_for_language("en")
        print(f"OK: model_path = {resolved_path}")
        print(f"    model_arch = {resolved_arch}")
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 3: Can we create a Transcriber?
    # ---------------------------------------------------------------
    test_step("Create Transcriber")
    try:
        t = Transcriber(model_path=resolved_path, model_arch=resolved_arch)
        print(f"OK: Transcriber created: {t}")
    except Exception as e:
        print(f"FAIL: {e}")
        sys.exit(1)

    # ---------------------------------------------------------------
    # Step 4: Does transcribe_without_streaming exist?
    # ---------------------------------------------------------------
    test_step("Check transcribe_without_streaming method")
    if hasattr(t, 'transcribe_without_streaming'):
        print("OK: method exists")
        import inspect
        sig = inspect.signature(t.transcribe_without_streaming)
        print(f"    signature: {sig}")
    else:
        print("FAIL: transcribe_without_streaming NOT found")
        print(f"    Available methods: {[m for m in dir(t) if not m.startswith('_')]}")
        # Try alternative names
        for name in ['transcribe', 'run', 'infer', 'process']:
            if hasattr(t, name):
                print(f"    Found alternative: {name}")

    # ---------------------------------------------------------------
    # Step 5: Test with a sine tone (sanity check)
    # ---------------------------------------------------------------
    test_step("Transcribe sine tone (expect empty/noise)")
    try:
        tone = generate_test_tone(2.0)
        print(f"   Audio: {len(tone)} samples, {len(tone)/16000:.1f}s, peak={np.max(np.abs(tone)):.3f}")

        start = time.time()
        result = t.transcribe_without_streaming(tone.tolist(), 16000)
        elapsed = time.time() - start

        print(f"OK: Got result in {elapsed:.2f}s")
        print(f"    Type: {type(result)}")
        print(f"    Lines: {len(result.lines) if hasattr(result, 'lines') else 'N/A'}")
        if hasattr(result, 'lines'):
            for i, line in enumerate(result.lines):
                print(f"    Line {i}: text='{line.text}', duration={getattr(line, 'duration', '?')}s")
        else:
            print(f"    Result repr: {repr(result)[:200]}")
    except AttributeError as e:
        print(f"FAIL: Method error: {e}")
        print("   Trying streaming API as fallback test...")

        # Fallback: test the streaming API
        test_step("Fallback: Test streaming start/add_audio/stop")
        results = []

        class TestListener(TranscriptEventListener):
            def on_line_completed(self, event):
                text = (event.line.text or "").strip()
                print(f"    [LISTENER] on_line_completed: '{text}'")
                if text:
                    results.append(text)
            def on_line_text_changed(self, event):
                text = (event.line.text or "").strip()
                print(f"    [LISTENER] on_line_text_changed: '{text}'")
            def on_line_started(self, event):
                print(f"    [LISTENER] on_line_started")

        t.add_listener(TestListener())

        print("   Calling start()...")
        t.start()

        speech = generate_speech_like(3.0)
        print(f"   Feeding {len(speech)} samples ({len(speech)/16000:.1f}s)...")

        # Feed in chunks like real-time
        chunk_size = 16000  # 1 second
        for i in range(0, len(speech), chunk_size):
            chunk = speech[i:i+chunk_size]
            t.add_audio(chunk.tolist(), 16000)
            print(f"   Fed chunk {i//chunk_size + 1}, listeners fired: {len(results)} results so far")
            time.sleep(0.1)  # Simulate real-time pacing

        print("   Calling stop()...")
        t.stop()
        print(f"   Final results: {results}")

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()

    # ---------------------------------------------------------------
    # Step 6: Test with actual WAV file (if available)
    # ---------------------------------------------------------------
    test_step("Test with real WAV audio (if ffmpeg available)")
    try:
        import subprocess
        import tempfile
        import os

        # Generate a WAV with ffmpeg's built-in tone generator
        wav_path = tempfile.mktemp(suffix=".wav")
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            "sine=frequency=300:duration=2",
            "-ar", "16000", "-ac", "1", "-f", "wav", wav_path
        ], capture_output=True, timeout=10)

        if os.path.exists(wav_path):
            with open(wav_path, "rb") as f:
                wav_data = f.read()
            os.unlink(wav_path)

            # Strip WAV header
            pcm = wav_data[44:]
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            print(f"   WAV: {len(samples)} samples, {len(samples)/16000:.1f}s")

            # Create fresh transcriber
            t2 = Transcriber(model_path=resolved_path, model_arch=resolved_arch)
            try:
                result = t2.transcribe_without_streaming(samples.tolist(), 16000)
                print(f"OK: Lines={len(result.lines)}")
                for line in result.lines:
                    print(f"    '{line.text}'")
            except AttributeError:
                print("   transcribe_without_streaming not available, trying streaming...")
                results2 = []

                class L2(TranscriptEventListener):
                    def on_line_completed(self, event):
                        results2.append((event.line.text or "").strip())
                    def on_line_text_changed(self, event):
                        pass
                    def on_line_started(self, event):
                        pass

                t2.add_listener(L2())
                t2.start()
                t2.add_audio(samples.tolist(), 16000)
                time.sleep(1)  # Give it time to process
                t2.stop()
                print(f"   Streaming results: {results2}")
        else:
            print("   ffmpeg failed to generate test WAV")
    except Exception as e:
        print(f"FAIL: {e}")

    # ---------------------------------------------------------------
    # Step 7: Test the actual _moonshine_batch_transcribe path
    # ---------------------------------------------------------------
    test_step("Test _moonshine_batch_transcribe (the actual code path)")
    try:
        # Generate PCM16 bytes (what the actual function receives)
        speech = generate_speech_like(3.0)
        pcm16 = (speech * 32767).astype(np.int16).tobytes()
        print(f"   PCM16: {len(pcm16)} bytes, {len(pcm16)/(16000*2):.1f}s")

        import asyncio
        from augmentum.proxy.audio_routes import _moonshine_batch_transcribe

        async def _test():
            result = await _moonshine_batch_transcribe(pcm16, "test.wav")
            return result

        result = asyncio.run(_test())
        print(f"OK: transcript = '{result}'")
    except Exception as e:
        print(f"FAIL: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print("  DIAGNOSTIC COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
