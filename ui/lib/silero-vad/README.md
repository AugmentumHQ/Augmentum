# Silero VAD (browser)

Silero voice-activity-detection running entirely client-side via
`@ricky0123/vad-web` (ONNX Runtime Web + the Silero ONNX model).

Used by `ui/scripts/voice/vad-client.js` to detect speech boundaries
without sending audio frames to the server. When loaded, the client
advertises `vad: ['silero-wasm']` in the WebSocket capabilities frame
and the server's `pipeline_resolver` routes the VAD component to
`client:silero-wasm` (see `augmentum/voice/pipeline_resolver.py`).

## Version pins

| Asset | Version | Size | Source |
|---|---|---|---|
| `@ricky0123/vad-web` (loader + worklet) | **0.0.30** | ~50KB JS | https://www.jsdelivr.com/package/npm/@ricky0123/vad-web |
| `onnxruntime-web` (Wasm runtime) | **1.14.0** | ~10MB wasm (browser-cached) | https://www.jsdelivr.com/package/npm/onnxruntime-web |
| `silero_vad_legacy.onnx` (v4) | bundled with vad-web 0.0.30 | ~1.7MB | https://github.com/snakers4/silero-vad |

Both packages are CDN-loaded from jsdelivr at first use, mirroring the
tree-sitter loader in `ui/scripts/codemind.js`. The browser caches them
indefinitely after first fetch (typical first-session cost: ~12MB, all
zero on every subsequent session).

## Vendoring (offline operation)

If you need fully offline operation (no CDN reachable on first use):

1. From the repo root, download the assets into this folder:
   ```bash
   curl -L -o ui/lib/silero-vad/bundle.min.js          https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.30/dist/bundle.min.js
   curl -L -o ui/lib/silero-vad/vad.worklet.bundle.min.js https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.30/dist/vad.worklet.bundle.min.js
   curl -L -o ui/lib/silero-vad/silero_vad_legacy.onnx  https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.30/dist/silero_vad_legacy.onnx
   curl -L -o ui/lib/silero-vad/ort.min.js              https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js
   # ORT wasm side-files (largest — ~30MB total across variants):
   curl -L -o ui/lib/silero-vad/ort-wasm-simd.wasm          https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort-wasm-simd.wasm
   curl -L -o ui/lib/silero-vad/ort-wasm-simd-threaded.wasm https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort-wasm-simd-threaded.wasm
   curl -L -o ui/lib/silero-vad/ort-wasm.wasm              https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort-wasm.wasm
   ```
2. Edit `loader.js` and swap each `CDN.*` URL for the local
   `/ui/lib/silero-vad/<filename>` path. The loader's smoke test will
   detect both paths transparently.

Without these files the CDN path remains the default and is the
recommended config for new installs.

## Upgrade procedure

1. Check the latest version on jsdelivr (links above).
2. Update the `VERSIONS` constants in `ui/scripts/voice/vad-client.js`.
3. Update the version pins in this README.
4. Smoke test: open a voice call in dev — server logs should show
   `voice_client_capabilities_registered caps={'vad': ['silero-wasm']}`
   followed by `pipeline_targets_resolved ... targets={'vad': 'client:silero-wasm', ...}`.
