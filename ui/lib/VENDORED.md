# Vendored frontend libraries

Each entry below is a third-party library copied into the repo so the
shipped UI doesn't need a runtime CDN or a build step. We avoid CDNs
because (a) they're a privacy leak — every page load tells a third
party that an Augmentum user is here, and (b) they're a supply-chain
risk — the CDN can swap bytes under us.

The tradeoff is that we have to manually bump each library when
upstream ships a fix. This file records what we have, what version it
is, and where to get a fresh copy.

`sha256` is over the listed entry file. To verify locally:

```bash
( cd ui/lib && sha256sum -c VENDORED.sha256 )
```

When you bump any of these, regenerate the hash file:

```bash
./scripts/regen_vendored_hashes.sh
```

---

## DOMPurify

- **Purpose:** HTML sanitizer used throughout the chat / artifact /
  reader-mode surfaces. Together with `escapeHtml()` it's our second
  line of defense against XSS in rendered content.
- **Version:** 3.2.4
- **License:** Apache-2.0 OR MPL-2.0 (dual-licensed)
- **Source:** <https://github.com/cure53/DOMPurify>
- **Entry file:** `dompurify/purify.min.js`
- **sha256:** `8eb41b658831fab175fad9bcd00fcb2d84e0ed3a25a55053d4ecd4444b8b43a0`
- **Bump command:** `curl -fSL https://github.com/cure53/DOMPurify/raw/<version>/dist/purify.min.js -o ui/lib/dompurify/purify.min.js`

## mermaid

- **Purpose:** Diagram rendering for chat artifacts (sequence diagrams,
  flowcharts, ER diagrams). Used by the artifact studio + chat
  message renderer.
- **Version:** unknown (header is unminified — bump via release page)
- **License:** MIT
- **Source:** <https://github.com/mermaid-js/mermaid>
- **Entry file:** `mermaid/mermaid.min.js`
- **sha256:** `07e37dfa97b337ccc85365d57eddf99b9706f09db3b59b260d0333b23b343c4b`
- **Bump:** download `dist/mermaid.min.js` from the release artifact.

## hls.js

- **Purpose:** HLS streaming playback for the media surface (recorded
  video / live cast playback).
- **Version:** unknown (bump via release page)
- **License:** Apache-2.0
- **Source:** <https://github.com/video-dev/hls.js>
- **Entry file:** `hls.js/hls.min.js`
- **sha256:** `d016c1230496ee59f3f5b01c16cce4cc01b5a1d3d357adec200c908b131ebe49`

## highlight.js

- **Purpose:** Syntax highlighting for code blocks in chat + artifacts.
- **Version:** unknown (license header truncated in the .min.js; bump via release page)
- **License:** BSD-3-Clause
- **Source:** <https://github.com/highlightjs/highlight.js>
- **Entry file:** `highlight.js/highlight.min.js`
- **sha256:** `c4a399dd6f488bc97a3546e3476747b3e714c99c57b9473154c6fb8d259b9381`
- **Extras:** `github.min.css` + `github-dark.min.css` for the two
  theme variants.

## prism

- **Purpose:** Alternative syntax highlighter used in a few legacy
  surfaces (kept as a fallback while we consolidate on highlight.js).
- **Version:** unknown
- **License:** MIT
- **Source:** <https://github.com/PrismJS/prism>
- **Entry files:** `prism-{css,javascript,json,markup}.min.js`,
  `prism-tomorrow.min.css`

## PDF.js

- **Purpose:** PDF rendering inside the document/knowledge browse
  surface and the reader-mode preview.
- **Version:** unknown (Mozilla project, bump via release page)
- **License:** Apache-2.0
- **Source:** <https://github.com/mozilla/pdf.js>
- **Entry file:** `pdfjs/pdf.min.mjs`
- **sha256:** `b20bebbb3b5febecf6d574434bd90aceddeffc4ca4385aaf9917332061ec6a29`
- **Worker:** `pdfjs/pdf.worker.min.mjs` — required by pdf.min.mjs at
  runtime via `pdfjsLib.GlobalWorkerOptions.workerSrc`.

## pdf-lib

- **Purpose:** PDF creation (not just rendering) — used by the
  artifact studio for the `pdf` artifact kind.
- **Version:** unknown
- **License:** MIT
- **Source:** <https://github.com/Hopding/pdf-lib>
- **Entry file:** `pdf-lib/pdf-lib.min.js`
- **sha256:** `0f9a5cad07941f0826586c94e089d89b918c46e5c17cf2d5a3c6f666e3bc694f`

## three / three.js

- **Purpose:** WebGL renderer for the VRM avatar, scene props
  (phone / tablet / water bottle), and the room scene.
- **Version:** unknown (bump via the upstream r1XX release tag)
- **License:** MIT
- **Source:** <https://github.com/mrdoob/three.js>
- **Entry files:** `three/*.js` — each is a thin loader (BVHLoader,
  DRACOLoader, GLTFLoader, OrbitControls, plus core `three.module.min
  .js` and addons).

## @pixiv/three-vrm

- **Purpose:** VRM avatar loader and runtime (humanoid bones,
  expressions, look-at) layered on three.js.
- **Version:** 2023-2024 release (license header dates)
- **License:** MIT
- **Source:** <https://github.com/pixiv/three-vrm>
- **Entry file:** `three-vrm/three-vrm.module.min.js`
- **sha256:** `b61e825655c2889e9e7ef943085b80e0064dd6cc8fd33173cc615cf4cddf6200`
- **Animation runtime:** `three-vrm/three-vrm-animation.module.min.js`

## milkdown

- **Purpose:** WYSIWYG markdown editor used in note-taking surfaces
  (artifact studio, journal entries).
- **Version:** unknown — multiple CSS bundles, no version marker in
  the headers
- **License:** MIT
- **Source:** <https://github.com/Milkdown/milkdown>
- **Entry files:** various CSS bundles (`*.css`) — no JS shipped here
  because the JS is bundled into our own modules.

## silero-vad

- **Purpose:** Browser-side voice-activity detection for the voice
  PTT / ambient streaming flow (runs the ONNX model in-page so the
  server only ever sees voiced audio).
- **Version:** unknown — wraps the Silero ONNX model from
  <https://github.com/snakers4/silero-vad>
- **License:** MIT (the JS wrapper) + MIT (the model)
- **Source:** <https://github.com/ricky0123/vad-web>
- **Entry file:** `silero-vad/bundle.min.js`
- **sha256:** `e97b92ecc7f29cb75ca21c9b814c4cf289bed134021fdcd17183d54d181b657d`
- **Model + runtime:** ships an `.onnx` weight and ort-wasm runtime
  files alongside the JS bundle.

## EmulatorJS

- **Purpose:** In-browser game emulation for the game-stream / cast
  receiver surfaces (lets receiver TVs play retro ROMs without a
  per-platform native shell).
- **Version:** **4.2.3** (recorded in `emulator-js/VERSION`)
- **License:** see `emulator-js/LICENSE`
- **Source:** <https://github.com/EmulatorJS/EmulatorJS>
- **Entry file:** `emulator-js/data/loader.js`
- **sha256:** `69e0903bf1e2f62ced78895e7e511fa26e11316f7eb734925c35e919ba1287b2`
- **Note:** the `data/` subdirectory (~296 MB of WASM cores) is
  gitignored and fetched via `scripts/vendor_emulator_js.sh`. The
  hash above is over the loader, not the WASM payload.

---

## Asset directories (not libraries — listed for completeness)

- `animations/` — VRMA motion clips for the bundled avatars
- `bundled-avatars/` — VRM avatar files (10-bundle roster)
- `props/` — glTF prop models (phone, tablet, water bottle)
- `scenes/` — glTF room scenes (e.g. `modern-room.glb`)
- `utils/` — small in-house JS shims, not vendored

## @noble E2E crypto (vendored 2026-06-23 by scripts/vendor_noble.mjs)
Pinned @noble/{curves,ciphers,hashes}@2.2.0, flattened from esm.sh es2022 ESM.
- curves-ed25519.mjs  sha256:67d8ea8544d6e31b67cf68e53aa70a54
- ciphers-chacha.mjs  sha256:6bf2474cf4f872126dde58ef14ce0de0
- hashes-hkdf.mjs  sha256:984e4af98be6311c00079585164ffe5a
- hashes-sha2.mjs  sha256:37af844c4f78c0158bd33b7c4f6fa778
- dep_hashes_utils.mjs  sha256:b080a64690ba87d0a59cf9c29a3030b6
- dep_curves_abstract_edwards.mjs  sha256:e89a7ab53867ea912d5f369ce86b7971
- dep_curves_abstract_frost.mjs  sha256:7c41b0435193cbb464c939c3d2d4237c
- dep_curves_abstract_hash-to-curve.mjs  sha256:cc5f5dcc452231b1c2deef1c2d52aece
- dep_curves_abstract_modular.mjs  sha256:45e38233b2d2ff226114425f66fd3245
- dep_curves_abstract_montgomery.mjs  sha256:7bebef8e0e8ed6455f1c82beaa634b0d
- dep_curves_abstract_oprf.mjs  sha256:e104e5ce1944938e06fe2f35599d0780
- dep_curves_utils.mjs  sha256:91a49596911a1adfec47f700b19f4beb
- dep_ciphers__arx.mjs  sha256:3c2eb158ca86a95565e845b5d553b356
- dep_ciphers__poly1305.mjs  sha256:57ca7ac513ac21991ead2b6e8da21cad
- dep_ciphers_utils.mjs  sha256:e0c87055e498d1640480388fc0974caa
- dep_hashes_hmac.mjs  sha256:d984829b06646bd45d8da4f351d9c436
- dep_hashes__md.mjs  sha256:e243c7ab62f616117b15a25112fa315e
- dep_hashes__u64.mjs  sha256:bf8938289ab1dfd5e87f3a9db9823cfc
- dep_curves_abstract_curve.mjs  sha256:48e7f621254dafda09a3a8c574b82b94
- dep_curves_abstract_fft.mjs  sha256:7f2ee4262dfc4bd973e5ad82c6f56db2
