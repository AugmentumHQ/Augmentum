# Voice Pipeline Research — March 2026

Comprehensive research into end-to-end voice interaction pipelines (STT -> LLM -> TTS) for AI assistants, with focus on open-source implementations, architecture patterns, and practical adoption strategies for an LLM proxy with existing OpenAI-compatible endpoints.

---

## 1. Architecture Patterns

### 1.1 Turn-Based Cascading Pipeline (STT -> LLM -> TTS)

The traditional approach chains three independent components sequentially:

```
Microphone -> VAD -> STT -> LLM -> TTS -> Speaker
```

**Characteristics:**
- Each stage waits for the previous to complete before starting
- Typical end-to-end latency: 1-3 seconds without optimization
- Maximum flexibility — swap any component independently
- Full tool calling, function execution, and reasoning capabilities preserved
- Better TTS quality (specialized models like ElevenLabs, Kokoro)
- Easier to build, debug, and maintain
- Consistent per-minute pricing regardless of conversation length

**Optimized cascading pipeline** with streaming overlap:
```
Microphone -> VAD -> Streaming STT
                         |
                    Streaming LLM (sentence boundary detection)
                         |
                    Chunked TTS (start on first sentence)
                         |
                    Streaming Audio Playback
```

With optimization, latency drops to **400-800ms** by overlapping stages.

### 1.2 Speech-to-Speech (End-to-End)

A single model processes audio-in -> audio-out directly:

```
Microphone -> Speech-to-Speech Model -> Speaker
```

**Characteristics:**
- Latency: 160-300ms (Moshi achieves 200ms in practice)
- Full-duplex: can listen and speak simultaneously
- Preserves tone, prosody, and emotional cues through audio processing
- ~10x higher cost than cascading pipelines
- Lower TTS quality compared to specialized TTS models
- Cannot do tool calling, function execution, or structured reasoning
- Hard to swap components — monolithic architecture
- Limited model options (Moshi, GPT-4o Realtime)

### 1.3 When to Use Which

| Criterion | Turn-Based Cascade | Speech-to-Speech |
|-----------|-------------------|-------------------|
| Latency target | 400-800ms (optimized) | 160-300ms |
| Cost | 1x | ~10x |
| Tool calling | Yes | No |
| TTS quality | Higher (specialized) | Lower (built-in) |
| Component flexibility | Full swap | Monolithic |
| Phone/PSTN deployments | Better (8kHz optimized) | S2S advantages lost |
| Interruption handling | Requires engineering | Native full-duplex |
| **Best for Augmentum** | **Yes — preserves proxy architecture** | Not suitable (no tool calling) |

**Verdict for Augmentum:** Turn-based cascading with streaming overlap. The proxy architecture demands component flexibility and tool calling support. Optimize latency through parallel pipeline stages and sentence-boundary TTS streaming.

---

## 2. Key Technologies

### 2.1 Speech-to-Text (STT) Models

#### Open-Source / Self-Hosted

| Model | Parameters | WER | Speed | Notes |
|-------|-----------|-----|-------|-------|
| **Whisper Large V3 Turbo** | 809M | ~5% | 5.4x faster than V3 | Reduced decoder layers (32->4), best balance |
| **Distil-Whisper** | Various | Within 1% of V3 | 6x faster than V3 | Distilled, great for real-time |
| **Faster-Whisper** | Uses V3 weights | Same as Whisper | 4-8x faster (CTranslate2) | Drop-in replacement, most popular self-hosted |
| **Canary Qwen 2.5B** | 2.5B | 5.63% (best open) | Moderate | Hybrid ASR+LLM architecture (SALM) |
| **whisper.cpp** | Uses V3 weights | Same as Whisper | Fast on CPU/Metal | C++ port, great for edge/embedded |
| **WhisperX** | Uses V3 weights | Same as Whisper | Batch optimized | Adds word timestamps + speaker diarization |

#### Cloud/API STT

| Provider | Latency | WER | Cost | Notes |
|----------|---------|-----|------|-------|
| **Groq Whisper** | <300ms | Same as Whisper V3 | $0.11/hr | 300x RTF on LPU hardware |
| **Deepgram Nova-3** | ~150ms TTFT | ~18% (real-world) | $4.30/1K min | Streaming native, cheapest |
| **AssemblyAI Universal-2** | 300-600ms | 14.5% (streaming) | $6.50/1K min | Best streaming accuracy |
| **OpenAI Whisper API** | ~500ms | ~5% | $6.00/1K min | Batch only, no streaming |

#### Recommendation for Augmentum
**Primary:** Proxy to any OpenAI-compatible STT endpoint (Speaches, Groq, Deepgram)
**Self-hosted option:** Faster-Whisper via Speaches (Docker, OpenAI-compatible API)

### 2.2 Text-to-Speech (TTS) Models

#### Compact / Low-Latency

| Model | Parameters | Latency | Voice Cloning | Emotion | License |
|-------|-----------|---------|---------------|---------|---------|
| **Kokoro** | 82M | Very fast | No | Limited | Apache 2.0 |
| **Piper** | Varies | Very fast (CPU) | No | No | MIT |
| **NeuTTS Air** | 500M | Real-time on-device | Yes (instant) | Limited | Proprietary? |

#### High-Quality / Expressive

| Model | Parameters | Latency | Voice Cloning | Emotion | License |
|-------|-----------|---------|---------------|---------|---------|
| **Orpheus** | 3B/1B/400M/150M | Streaming capable | Yes (zero-shot) | Yes (guided) | Apache 2.0 |
| **Dia (Nari Labs)** | 1.6B | ~40 tok/s on A4000 | Yes (zero-shot) | Yes + non-verbal | Apache 2.0 |
| **Sesame CSM** | 1B | Moderate | Yes | Yes (conversational) | Open |
| **Fish Audio S1** | 4B | Moderate | Yes (multilingual) | Yes | Open |
| **Chatterbox Turbo** | ~1B | Fast | Yes | Yes (exaggeration dial) | Open |
| **Qwen3-TTS** | 0.6B/1.7B | 97ms streaming | Yes (CustomVoice) | Yes (NL instruction) | Open |
| **CosyVoice 2** | ~1B | 150ms streaming | Yes | Yes (fine-grained) | Open |
| **Higgs Audio V2** | ~3B (Llama 3.2) | Moderate | Yes (multilingual) | Yes | Open |
| **EmotiVoice** | ~1B | Moderate | 2000+ preset voices | Yes | Open |
| **F5-TTS** | ~1B | Fast | Yes | Limited | Open |
| **XTTS v2 (Coqui)** | ~1B | Moderate | Yes | Limited | CPML |

#### OpenAI-Compatible TTS Servers (Self-Hosted)

| Project | Backend Models | API Compatibility | Notes |
|---------|---------------|-------------------|-------|
| **Kokoro-FastAPI** | Kokoro-82M | OpenAI `/v1/audio/speech` | Docker, CPU ONNX + GPU PyTorch, streaming |
| **Speaches** | Kokoro, Piper | OpenAI `/v1/audio/speech` + `/v1/audio/transcriptions` | "Ollama for TTS/STT", also proxies voice chat |
| **AllTalk TTS** | XTTS v2, custom | OpenAI `/v1/audio/speech` | Narrator mode, DeepSpeed, low VRAM |
| **openai-edge-tts** | Microsoft Edge TTS | OpenAI `/v1/audio/speech` | Free (uses Edge online service) |
| **openai-tts-server** | Kokoro, StyleTTS2, Piper, Faster-Whisper | OpenAI `/v1/audio/speech` + STT | Multi-engine support |
| **LocalAI** | Various (Piper, etc.) | OpenAI + ElevenLabs | General-purpose local AI server |
| **Sesame CSM OpenAI** | CSM 1B, Dia 1.6B | OpenAI `/v1/audio/speech` | Voice cloning from file/YouTube |

#### Recommendation for Augmentum
**Primary:** Proxy to any OpenAI-compatible TTS endpoint. Already have the endpoints.
**Self-hosted default:** Kokoro-FastAPI (82M, fast, Apache 2.0, Docker, OpenAI-compatible)
**Upgrade path:** Orpheus for emotion/cloning, Dia for dialogue, Qwen3-TTS for multilingual

### 2.3 Voice Activity Detection (VAD)

| VAD | Type | Accuracy (TPR@5%FPR) | Size | Latency | License |
|-----|------|----------------------|------|---------|---------|
| **Silero VAD** | DNN (ONNX/PyTorch) | 87.7% | 1.8MB | ~1ms per 30ms chunk | MIT |
| **WebRTC VAD** | GMM | 50% | Tiny | Sub-ms | BSD |
| **Cobra (Picovoice)** | Proprietary | 92.7% | Small | Sub-ms | Commercial |

**Silero VAD is the clear winner** for open-source: 4x fewer errors than WebRTC VAD, 100+ language training, MIT license, 1.8MB, runs everywhere. Used by Pipecat, RealtimeVoiceChat, and most modern voice pipelines.

### 2.4 Audio Processing

- **Web Audio API + AudioWorklets** — browser-side audio capture and processing
- **PyAudio / sounddevice** — Python audio I/O
- **FFmpeg** — audio format conversion, resampling
- **WebRTC** — real-time media transport (LiveKit, Daily.co)
- **WebSockets** — simpler alternative for audio chunk streaming
- **Opus codec** — low-latency audio compression for streaming

---

## 3. Open-Source Voice Pipeline Projects

### 3.1 Pipecat (by Daily.co)

**GitHub:** https://github.com/pipecat-ai/pipecat
**Stars:** 7k+ | **License:** BSD

The most mature open-source voice agent framework. Used by NVIDIA, Cresta, and others.

**Architecture:**
- Pipeline of Frame Processors handling real-time Frames (audio, text, video)
- Transport (WebRTC via Daily.co, WebSocket) -> STT -> LLM -> TTS -> Transport
- Built-in Silero VAD, interruption handling, turn detection
- 40+ AI service integrations as plugins
- SDKs: Python, JavaScript, React, iOS, Android, C++
- Round-trip latency: 500-800ms

**Key patterns to adopt:**
- Frame-based pipeline abstraction
- Modular processor chain with hot-swappable components
- Built-in interruption handling via VAD + transport coordination
- NVIDIA blueprint for enterprise deployment

### 3.2 LiveKit Agents

**GitHub:** https://github.com/livekit/agents
**Stars:** 5k+ | **License:** Apache 2.0

WebRTC-first infrastructure for real-time voice/video AI agents.

**Architecture:**
- SFU (Selective Forwarding Unit) media server written in Go (Pion WebRTC)
- Python/Node.js agent SDK joins LiveKit rooms as full participants
- STT-LLM-TTS pipeline with reliable turn detection and interruption handling
- WebRTC between frontend and agent, HTTP/WebSocket between agent and backend
- Used by OpenAI as a customer

**Key patterns:**
- WebRTC for client-server (better than raw WebSocket for media)
- Agent-as-participant model
- Horizontal scaling via SFU architecture

### 3.3 FastRTC (by Hugging Face/Gradio)

**GitHub:** https://github.com/gradio-app/fastrtc
**Stars:** 3k+ | **License:** Apache 2.0

Lightweight Python library for real-time audio/video.

**Architecture:**
- Wraps any Python function into a WebRTC or WebSocket stream
- Built-in VAD and turn detection (automatic)
- Built-in STT (Whisper) and TTS
- Auto-generates Gradio UI via `.ui.launch()`
- Mount on FastAPI via `.mount(app)` for WebRTC/WebSocket endpoints
- Telephone support via `fastphone()` (free temporary number)

**Key patterns for Augmentum:**
- `.mount(app)` pattern maps directly to FastAPI integration
- Automatic VAD + turn-taking removes complexity
- Simplest path to add voice to an existing FastAPI app

### 3.4 RealtimeVoiceChat (by KoljaB)

**GitHub:** https://github.com/KoljaB/RealtimeVoiceChat
**Stars:** 3k+ | **License:** MIT

Complete voice chat implementation optimized for local/self-hosted AI.

**Architecture:**
- Browser (Vanilla JS + Web Audio API + AudioWorklets) -> WebSocket -> FastAPI backend
- RealtimeSTT (Whisper-based) -> Ollama/OpenAI LLM -> RealtimeTTS -> WebSocket -> Browser
- Dynamic silence detection (turndetect.py) adapts to conversation pace
- Pluggable TTS: Kokoro, Coqui XTTSv2, Orpheus
- ~500ms latency with audio chunk streaming
- Docker Compose with NVIDIA GPU support

**Key patterns for Augmentum:**
- Most architecturally similar to what Augmentum would build
- FastAPI + WebSocket + vanilla JS frontend
- Pluggable LLM backend (Ollama default, OpenAI support)
- Dynamic turn detection with adaptive silence thresholds

**Related libraries by same author:**
- **RealtimeSTT** — robust STT with VAD, wake word, instant transcription
- **RealtimeTTS** — streaming TTS with sentence-boundary chunking
- **LocalAIVoiceChat** — earlier version using Zephyr 7B + XTTS

### 3.5 Home Assistant Voice Pipeline

**Docs:** https://www.home-assistant.io/integrations/wyoming/

**Architecture:**
- Wyoming Protocol: standardized TCP protocol for STT/TTS/Wake Word communication
- Piper TTS for on-device synthesis (runs on Raspberry Pi)
- Whisper for on-device STT
- Streaming TTS overhaul (2025): chunks of text -> chunks of audio, played immediately
- Satellite devices connect via Zeroconf discovery
- Multilingual support (2025.10+)

**Key patterns:**
- Wyoming Protocol as a model for standardized voice service communication
- Streaming TTS architecture (chunk text -> chunk audio -> play immediately)
- On-device / edge deployment focus

### 3.6 Moshi (by Kyutai)

**GitHub:** https://github.com/kyutai-labs/moshi
**License:** Apache 2.0 (code) / CC BY 4.0 (weights)

First open-source full-duplex speech-to-speech model.

**Architecture:**
- Single model: audio-in -> audio-out (no STT/TTS chain)
- Full-duplex: listens and speaks simultaneously
- "Inner Monologue": predicts time-aligned text tokens as prefix to audio tokens
- Uses Mimi neural audio codec for streaming
- 200ms practical latency, 160ms theoretical
- Multilingual checkpoint targeted Q1-2026

**Key pattern:**
- Inner Monologue technique for maintaining linguistic quality in S2S
- Kyutai Pocket TTS: 100M parameter model runs on CPU in real-time

### 3.7 Speaches (formerly faster-whisper-server)

**GitHub:** https://github.com/speaches-ai/speaches
**Docs:** https://speaches.ai/

"Ollama for TTS/STT models" — the most directly relevant project for Augmentum.

**Architecture:**
- OpenAI-compatible API server for both STT and TTS
- STT: Faster-Whisper (default: faster-distil-whisper-large-v3)
- TTS: Kokoro (default: Kokoro-82M ONNX), Piper
- Voice chat proxy: intercepts audio messages, transcribes, forwards to LLM, synthesizes response
- Docker deployment with model auto-download

**API Endpoints:**
- `POST /v1/audio/transcriptions` — STT (OpenAI-compatible)
- `POST /v1/audio/speech` — TTS (OpenAI-compatible)
- `POST /v1/chat/completions` — Voice chat proxy (audio in messages -> transcribe -> LLM -> TTS)

**Key patterns for Augmentum:**
- Exact same proxy pattern Augmentum uses
- OpenAI-compatible endpoints for both STT and TTS
- Voice chat as a transparent proxy layer
- Model-agnostic: swap STT/TTS models without changing API

---

## 4. LLM Frontend Voice Integrations

### 4.1 Open WebUI

**Voice Mode / Call Feature:**
- Browser MediaRecorder captures microphone audio
- Web Audio API analyzes audio for speech detection and silence
- Configurable STT: browser-native or OpenAI Whisper API
- Configurable TTS: browser-native TTS or OpenAI Speech API
- Custom TTS engine support (any OpenAI-compatible endpoint)
- Voice interruption support (configurable)
- Optional video/screen share during calls
- Playback rate adjustment

**Known issues (2025):** TTS response not streaming (waits for full generation), occasional voice mode failures.

**Integration pattern:** Uses OpenAI `/v1/audio/speech` and `/v1/audio/transcriptions` endpoints, configurable to point at any compatible server.

### 4.2 SillyTavern

**Speech Recognition:**
- Extension-based (Download Extensions & Assets menu)
- STT sources: OpenAI, MistralAI, Groq, browser-native Web Speech API
- Streaming transcription with partial results

**Text-to-Speech:**
- Multiple TTS providers: AllTalk, Kokoro-FastAPI, ElevenLabs, Azure, browser TTS
- Narrator mode: separate voices for character dialogue vs narration
- RVC (Retrieval-based Voice Conversion) extension for voice changing
- Sentence-by-sentence playback

**Integration with Kokoro-FastAPI:** Documented wiki guide for self-hosted TTS via OpenAI-compatible endpoint.

### 4.3 oobabooga / text-generation-webui

- AllTalk TTS extension (Coqui-based, XTTSv2)
- Direct integration: LLM response -> AllTalk TTS -> audio in chat
- Remote extension mode: run AllTalk on separate hardware
- Narrator feature: different voices for character vs narration
- DeepSpeed support for 3-4x TTS speedup
- Low VRAM mode for resource-constrained setups
- VRAM challenge: running LLM + TTS model simultaneously on 8GB

---

## 5. Latency Optimization Techniques

### 5.1 Sub-400ms Architecture (Production Target)

Based on Simplismart's production architecture:

| Stage | Technique | Latency |
|-------|-----------|---------|
| **STT** | Batch transcription (not token streaming) | ~120ms effective |
| **LLM** | Prefix/KV caching, concurrency caps | ~140ms TTFT (25 concurrent) |
| **TTS** | Sentence-aware streaming + chunked decode | ~100ms (overlapped) |
| **Total** | Parallel pipeline with overlap | **<400ms** |

### 5.2 Specific Optimization Techniques

1. **Sentence Boundary Detection for TTS Streaming**
   - Stream LLM tokens, detect sentence boundaries (`.`, `?`, `!`) in real-time
   - Dispatch each sentence to TTS immediately without waiting for full response
   - Use `PunctuatedBufferStreamer` pattern: regex-based punctuation detection, thread-safe queue
   - Augmentum already has streaming infrastructure — add sentence buffering layer

2. **Chunked Audio Decoding**
   - Decode audio in small slices (~3 frames) instead of full waveforms
   - Reduces audio TTFB from ~150ms to ~60ms
   - Send audio chunks to client as they're generated

3. **Parallel Pipeline Execution**
   - STT feeds partial transcripts to LLM
   - LLM streams tokens to sentence buffer
   - TTS synthesizes each sentence while LLM continues generating
   - Audio streams to client while TTS continues with next sentence

4. **Prefix/KV Caching for LLM**
   - Cache system context (2-4K tokens) across requests
   - Reduces TTFT from hundreds of ms to 30-40ms per single request
   - Augmentum already has prompt caching — leverage for voice

5. **STT: Batch Over Streaming**
   - For typical utterances (5-20s), batch transcription is faster than streaming
   - 10s audio transcribed in ~50ms with Faster-Whisper
   - Dynamic batching handles 25+ concurrent streams per GPU

6. **Client-Side Optimizations**
   - Use AudioWorklets for real-time audio processing (not deprecated ScriptProcessorNode)
   - Opus codec for low-latency compressed audio over WebSocket
   - Pre-buffer audio playback to avoid gaps between TTS chunks
   - WebRTC for production (better than WebSocket for media)

### 5.3 Latency Budget

For natural conversation (target <600ms total):

```
VAD detection:        ~30ms   (Silero VAD)
Audio transmission:   ~20ms   (WebSocket/WebRTC)
STT processing:       ~80ms   (Faster-Whisper, batched)
LLM TTFT:            ~100ms   (with KV cache)
Sentence detection:   ~10ms   (buffer until punctuation)
TTS first chunk:      ~80ms   (Kokoro streaming)
Audio transmission:   ~20ms   (WebSocket/WebRTC)
Playback start:       ~10ms
─────────────────────────────
Total:               ~350ms   (optimistic, single user)
                     ~600ms   (realistic, concurrent load)
```

---

## 6. Voice Chat UX Patterns

### 6.1 Input Modes

**Push-to-Talk (PTT):**
- User holds button while speaking, releases to send
- Simplest to implement, no VAD needed
- Best for noisy environments or shared spaces
- Implemented via: button press -> start recording, button release -> stop + send

**Auto-Detect (Voice Activity Detection):**
- Silero VAD detects speech onset and offset
- Dynamic silence threshold adapts to conversation pace
- Configurable silence duration before end-of-turn (typically 500-1500ms)
- More natural but prone to false triggers in noisy environments
- RealtimeVoiceChat's `turndetect.py` is a good reference implementation

**Hybrid:**
- Auto-detect with manual override (tap to force-send)
- Most production systems use this approach
- Open WebUI implements configurable interruption toggle

### 6.2 Turn-Taking Strategies

**Server-VAD (OpenAI Realtime API pattern):**
- Server runs VAD on incoming audio stream
- Server decides when user has finished speaking
- Server triggers STT -> LLM pipeline automatically
- Pro: consistent behavior, server controls conversation flow
- Con: server must process all audio (bandwidth + compute)

**Client-VAD:**
- Browser/app runs VAD locally (Silero VAD ONNX in browser)
- Client decides when to send completed utterance
- Client sends segmented audio chunks
- Pro: reduces server bandwidth, faster response
- Con: inconsistent across devices/browsers

**Hybrid VAD (recommended):**
- Client-side VAD for immediate feedback (recording indicator)
- Server-side VAD for authoritative end-of-turn detection
- Client streams audio chunks continuously
- Server uses VAD to detect speech boundaries in stream

### 6.3 Interruption Handling

**Basic (Open WebUI pattern):**
- If user starts speaking while AI audio is playing:
  - Stop audio playback immediately (`stopAllAudio()`)
  - Cancel any pending TTS generation
  - Start processing new user input
- Toggle: `voiceInterruption` setting

**Advanced (Pipecat/LiveKit pattern):**
- VAD detects user speech during AI output
- Classify: backchannel ("mm-hmm") vs true interruption
- If interruption: stop TTS, truncate LLM context to point of interruption
- If backchannel: continue playing, optionally acknowledge

**Full-Duplex (Moshi pattern):**
- System listens and speaks simultaneously
- No explicit interruption handling — model handles it natively
- Requires speech-to-speech model

### 6.4 UI Patterns for Voice Mode

**Call Mode (Open WebUI):**
- Full-screen overlay with waveform visualization
- Microphone button with recording indicator
- Optional video feed
- Settings: voice selection, playback rate, interruption toggle
- Emoji/avatar display during response

**Inline Voice (SillyTavern):**
- Microphone icon in chat input area
- Audio plays inline with chat messages
- STT result shown as user message
- TTS plays as character speaks

**Standalone Voice Chat (RealtimeVoiceChat):**
- Dedicated voice interface
- Waveform visualization
- Status indicators (listening, thinking, speaking)
- Minimal text display

---

## 7. Advanced Features

### 7.1 Voice Cloning

**Zero-Shot (reference audio):**
- Orpheus, Dia, CSM, Fish Audio, XTTS v2, Qwen3-TTS CustomVoice
- Provide 5-30s reference audio clip
- Model reproduces voice characteristics
- Quality varies: Dia and Orpheus lead in naturalness

**Instant Cloning:**
- NeuTTS Air: on-device instant cloning
- Chatterbox: real-time voice cloning
- CosyVoice 2: 150ms streaming with cloned voice

### 7.2 Emotion / Prosody Control

**Tag-based (Orpheus):**
- Insert emotion tags in text: `[laugh]`, `[sigh]`, `[excited]`
- Model generates appropriate prosody
- Guided emotion control during inference

**Natural Language Instruction (Qwen3-TTS):**
- Describe desired voice characteristics in natural language
- "Speak with a warm, cheerful tone" -> model adjusts prosody
- Multi-dimensional: timbre, emotion, speed, pitch

**Exaggeration Dial (Chatterbox):**
- Single parameter controls emotional intensity
- 0.0 = monotone, 1.0 = dramatically expressive
- Simple API integration

**Non-Verbal Cues (Dia):**
- Generates laughter, coughing, throat clearing, sighs
- Driven by transcript markers
- Most natural dialogue generation

### 7.3 Multi-Speaker

- **Dia:** Native multi-speaker dialogue in one pass (speaker tags in transcript)
- **CSM:** Conversational model handles speaker turns
- **SillyTavern:** Different voices per character via TTS extension
- **AllTalk:** Narrator mode (separate voice for narration vs dialogue)

### 7.4 Voice Changing / RVC

- **RVC (Retrieval-based Voice Conversion):** Post-processing filter
- SillyTavern has dedicated RVC extension
- Apply voice conversion after TTS generation
- Can transform any TTS output to target voice
- Lower quality than native voice cloning but more flexible

### 7.5 Real-Time Translation

- **SeamlessM4T (Meta):** Speech-to-speech translation
- **Whisper + translation flag:** Transcribe + translate in one step
- Cascading approach: STT(language A) -> translate -> TTS(language B)
- Qwen3-TTS: multilingual TTS supporting 10 languages

---

## 8. Implementation Strategy for Augmentum

### 8.1 Phase 1: OpenAI-Compatible Voice Proxy (Minimal)

Augmentum already has OpenAI-compatible proxy endpoints. Add:

1. **`POST /v1/audio/speech`** — Proxy to configured TTS backend
   - Accept: `model`, `input`, `voice`, `response_format`, `speed`
   - Route to: Kokoro-FastAPI, Speaches, AllTalk, or any OpenAI-compatible TTS
   - Support streaming response (chunked transfer encoding)

2. **`POST /v1/audio/transcriptions`** — Proxy to configured STT backend
   - Accept: audio file upload (`file`, `model`, `language`, `response_format`)
   - Route to: Speaches, Groq Whisper, or any OpenAI-compatible STT
   - Return transcription text

3. **Configuration:**
   ```
   AUGMENTUM_VOICE_ENABLED=true
   AUGMENTUM_TTS_BACKEND_URL=http://kokoro:8880/v1
   AUGMENTUM_STT_BACKEND_URL=http://speaches:8000/v1
   AUGMENTUM_TTS_DEFAULT_MODEL=kokoro
   AUGMENTUM_TTS_DEFAULT_VOICE=af_heart
   AUGMENTUM_STT_DEFAULT_MODEL=Systran/faster-whisper-large-v3
   ```

4. **Docker Compose addition:**
   ```yaml
   kokoro-tts:
     image: ghcr.io/remsky/kokoro-fastapi:latest
     ports: ["8880:8880"]
     profiles: ["voice"]
   speaches:
     image: ghcr.io/speaches-ai/speaches:latest
     ports: ["8000:8000"]
     profiles: ["voice"]
   ```

### 8.2 Phase 2: WebSocket Voice Chat

Add real-time voice conversation via WebSocket:

1. **WebSocket endpoint:** `ws://host:6000/ws/voice`
2. **Protocol:**
   ```
   Client -> Server: Binary audio chunks (PCM 16kHz 16-bit mono)
   Server -> Client: JSON control messages + binary audio chunks

   Control messages:
   {"type": "vad_start"}           — speech detected
   {"type": "vad_end"}             — silence detected, processing
   {"type": "transcription", "text": "..."}  — STT result
   {"type": "llm_start"}           — LLM processing started
   {"type": "llm_token", "token": "..."} — streaming LLM token (optional)
   {"type": "tts_start"}           — TTS synthesis started
   {"type": "tts_chunk"}           — followed by binary audio data
   {"type": "tts_end"}             — response complete
   {"type": "interrupted"}         — user interrupted, response cancelled
   ```

3. **Server-side pipeline:**
   ```python
   async def voice_pipeline(websocket, session_id):
       vad = SileroVAD()
       audio_buffer = AudioBuffer()

       async for message in websocket:
           if isinstance(message, bytes):
               audio_buffer.append(message)
               if vad.is_speech(message):
                   await websocket.send_json({"type": "vad_start"})
               elif vad.is_silence(message) and audio_buffer.has_speech:
                   # End of utterance
                   await websocket.send_json({"type": "vad_end"})

                   # STT
                   text = await stt_transcribe(audio_buffer.get_audio())
                   await websocket.send_json({"type": "transcription", "text": text})

                   # LLM (through existing Augmentum pipeline)
                   async for sentence in stream_llm_sentences(text, session_id):
                       # TTS per sentence
                       async for audio_chunk in tts_stream(sentence):
                           await websocket.send_bytes(audio_chunk)

                   await websocket.send_json({"type": "tts_end"})
                   audio_buffer.clear()
   ```

4. **Sentence boundary detection:**
   ```python
   class SentenceBuffer:
       """Buffer LLM tokens, emit on sentence boundaries."""
       def __init__(self):
           self.buffer = ""
           self.boundaries = re.compile(r'[.!?]\s+|[.!?]$|\n')

       def add_token(self, token: str) -> str | None:
           self.buffer += token
           match = self.boundaries.search(self.buffer)
           if match:
               sentence = self.buffer[:match.end()]
               self.buffer = self.buffer[match.end():]
               return sentence.strip()
           return None
   ```

### 8.3 Phase 3: UI Voice Mode

Add voice chat UI to the existing web interface:

1. **Call button** in chat header (phone icon)
2. **Voice overlay** with waveform visualization
3. **Push-to-talk** (spacebar) + **auto-detect** (toggle)
4. **Status indicators:** Listening / Thinking / Speaking
5. **Interruption:** Click or speak to interrupt AI response
6. **Settings:** Voice selection, auto-detect sensitivity, playback speed

### 8.4 Phase 4: Advanced Features (Optional)

- Voice selection per session (map to TTS voices)
- Narrative mode: different voices per character (leverage narrator mode from AllTalk/SillyTavern pattern)
- Emotion-aware TTS: extract emotion from LLM response, pass to TTS (Orpheus tags, Qwen3-TTS instructions)
- Voice cloning: upload reference audio, use for session
- WebRTC transport option (via FastRTC `.mount(app)` for production deployments)

---

## 9. Reference Projects Summary

| Project | GitHub | Architecture | Best For |
|---------|--------|-------------|----------|
| **Pipecat** | pipecat-ai/pipecat | Frame pipeline + Daily WebRTC | Production voice agents |
| **LiveKit Agents** | livekit/agents | SFU + WebRTC + agent SDK | Scalable multi-user voice |
| **FastRTC** | gradio-app/fastrtc | Python function -> WebRTC/WS | Fast prototyping, FastAPI integration |
| **RealtimeVoiceChat** | KoljaB/RealtimeVoiceChat | FastAPI + WebSocket + local AI | Self-hosted voice chat (closest to Augmentum) |
| **Speaches** | speaches-ai/speaches | OpenAI-compatible STT/TTS server | Drop-in TTS/STT backend |
| **Kokoro-FastAPI** | remsky/Kokoro-FastAPI | OpenAI-compatible TTS server | Lightweight self-hosted TTS |
| **Moshi** | kyutai-labs/moshi | End-to-end S2S model | Full-duplex research |
| **RealtimeSTT** | KoljaB/RealtimeSTT | Whisper + VAD + wake word | STT library |
| **RealtimeTTS** | KoljaB/RealtimeTTS | Multi-engine streaming TTS | TTS library |
| **Home Assistant Voice** | home-assistant (Wyoming) | Wyoming protocol + Piper + Whisper | Edge/IoT voice |
| **AllTalk TTS** | erew123/alltalk_tts | Coqui/XTTS + OpenAI API | Feature-rich self-hosted TTS |
| **voice-mode** | (PyPI) | OpenAI-compatible voice library | Python voice integration |

---

## 10. Key Takeaways

1. **The cascading pipeline (STT->LLM->TTS) is the right architecture for Augmentum.** It preserves tool calling, component flexibility, and the proxy pattern. Speech-to-speech models are not suitable for a proxy that needs to intercept and enhance LLM interactions.

2. **Speaches is the closest analog** — it's literally "Ollama for TTS/STT" with OpenAI-compatible endpoints and voice chat proxy. Could be used as a backend or as architectural inspiration.

3. **RealtimeVoiceChat is the closest implementation reference** — FastAPI + WebSocket + vanilla JS, pluggable backends, ~500ms latency. Almost exactly the stack Augmentum uses.

4. **Kokoro-82M is the default TTS choice** — 82M parameters, Apache 2.0, fast on CPU, OpenAI-compatible server exists (Kokoro-FastAPI). Upgrade to Orpheus/Dia for emotion and cloning.

5. **Faster-Whisper is the default STT choice** — CTranslate2 optimization, 4-8x faster than vanilla Whisper, available via Speaches server with OpenAI-compatible API.

6. **Silero VAD is the universal standard** — MIT license, 1.8MB, 87.7% accuracy, used by every major voice pipeline.

7. **Sentence-boundary TTS streaming is the key latency optimization** — don't wait for full LLM response, synthesize each sentence as it completes. This alone can cut perceived latency by 50-70%.

8. **Start with proxy endpoints, add WebSocket voice chat later.** Phase 1 (proxy to TTS/STT backends) is trivial given Augmentum's existing architecture. Phase 2 (WebSocket voice) is the real work but has clear reference implementations.

---

## Sources

- [Pipecat Framework](https://github.com/pipecat-ai/pipecat)
- [LiveKit Agents](https://github.com/livekit/agents)
- [FastRTC](https://github.com/gradio-app/fastrtc)
- [RealtimeVoiceChat](https://github.com/KoljaB/RealtimeVoiceChat)
- [Speaches](https://github.com/speaches-ai/speaches)
- [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI)
- [Moshi](https://github.com/kyutai-labs/moshi)
- [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT)
- [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS)
- [AllTalk TTS](https://github.com/erew123/alltalk_tts)
- [Dia TTS](https://github.com/nari-labs/dia)
- [Home Assistant Wyoming Protocol](https://www.home-assistant.io/integrations/wyoming/)
- [Silero VAD](https://pytorch.org/hub/snakers4_silero-vad_vad/)
- [Open WebUI Voice Discussion](https://github.com/open-webui/open-webui/discussions/4574)
- [SillyTavern TTS Docs](https://docs.sillytavern.app/extensions/tts/)
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)
- [Voice AI Stack 2026](https://www.assemblyai.com/blog/the-voice-ai-stack-for-building-agents)
- [Real-Time vs Turn-Based Architecture](https://softcery.com/lab/ai-voice-agents-real-time-vs-turn-based-tts-stt-architecture)
- [Sub-400ms Voice AI Architecture](https://simplismart.ai/blog/real-time-voice-ai-sub-400ms-latency)
- [Best Open-Source TTS 2026](https://www.bentoml.com/blog/exploring-the-world-of-open-source-text-to-speech-models)
- [Best Open-Source STT 2026](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
- [VAD Comparison 2026](https://picovoice.ai/blog/best-voice-activity-detection-vad/)
- [Latency-Aware TTS Pipeline](https://www.dupdub.com/blog/tts-latency-optimization)
- [Qwen3-TTS](https://stable-learn.com/en/qwen3-tts-0115-opensource/)
- [Orpheus TTS](https://www.blog.brightcoding.dev/2025/09/07/orpheus-tts-the-open-source-model-bringing-voice-cloning-and-emotion-control-to-the-masses/)
- [openai-tts-server](https://github.com/jakezp/openai-tts-server)
- [Sesame CSM OpenAI](https://github.com/phildougherty/sesame_csm_openai)
- [Full-Duplex Spoken Dialogue Survey](https://arxiv.org/html/2509.14515v1)
- [voice-mode PyPI](https://pypi.org/project/voice-mode/)
