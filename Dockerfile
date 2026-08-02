# syntax=docker/dockerfile:1

# ---- llama-server: pre-built CPU image ----
# Default points at the locally-built tag for dev workflow.
# CI passes ``--build-arg LLAMA_SERVER_IMAGE=ghcr.io/augmentumhq/augmentum-llama-server-cpu:<version>``.
ARG LLAMA_SERVER_IMAGE=augmentum-llama-server-cpu:latest
FROM ${LLAMA_SERVER_IMAGE} AS llama-builder

# ---- Runtime stage ----
# Base image pinned to a specific digest so an upstream tag rotation
# (or a registry compromise that swaps the tag's target) can't slip
# new bytes into our build without a commit here. Rotate via
# ``scripts/upgrade_base_images.sh`` — that script pulls the latest
# manifest digest for each pinned tag and rewrites these lines so the
# diff is auditable.
FROM python:3.14-slim-bookworm@sha256:86f975aca15cf04a40b399eebede9aea7c82eae084d1f1a0a6ef6bcaae871a30

WORKDIR /app

# Install curl for healthcheck, gosu for entrypoint, ffmpeg for audio transcoding,
# sqlite3 for DB recovery (.recover on corruption),
# fonts-dejavu-core for Unicode PDF rendering (CO₂, em dashes, Greek letters, etc.),
# chromium for the App Builder's headless-browser verify step
# (see augmentum/tools/application_cdp.py).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gosu ffmpeg sqlite3 fonts-dejavu-core chromium bzip2 git && \
    rm -rf /var/lib/apt/lists/*

# cloudflared — the throwaway quick-tunnel binary the Connect invite "Anywhere"
# tier spawns to stand up an ephemeral, anonymous public door for an
# out-of-network guest (torn down on claim). Zero-config: reachability.py
# auto-detects it on PATH, so bundling it makes the public tier work out of the
# box. Pinned + verified — scripts/smoke_cloudflared.py runs a REAL quick tunnel
# against this exact version.
# TARGETARCH is auto-populated by Buildx per platform (amd64 / arm64) and maps
# directly to cloudflared's release asset names. Defaulted for plain `docker
# build` (no Buildx) so the amd64 path still works. This keeps the multi-arch
# CPU image from trying to run an amd64 binary on arm64 (which fails the build
# at the --version check below).
ARG CLOUDFLARED_VERSION=2026.7.2
ARG TARGETARCH=amd64
RUN curl -fsSL -o /usr/local/bin/cloudflared \
      "https://github.com/cloudflare/cloudflared/releases/download/${CLOUDFLARED_VERSION}/cloudflared-linux-${TARGETARCH}" && \
    chmod +x /usr/local/bin/cloudflared && \
    /usr/local/bin/cloudflared --version

# Create non-root user and data directories
RUN useradd -m -u 1000 augmentum && \
    mkdir -p /data/image_models /data/image_output && \
    chown -R augmentum:augmentum /data

# llama-server binary + shared libraries from the build stage (engine v2,
# CPU build). Lets the CPU variant default to AUGMENTUM_DEFAULT_BACKEND=engine
# and have a working LLM out of the box without an external Ollama/LM Studio.
COPY --from=llama-builder /usr/local/bin/llama-server /usr/local/bin/llama-server
COPY --from=llama-builder /usr/local/lib/lib*.so /usr/local/lib/
RUN ldconfig
# libgomp (OpenMP runtime) is required by the llama-server binary for
# threading. The slim base doesn't include it; libcurl4 is needed by the
# llama-server's HTTP layer when models reference remote URLs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libcurl4 && \
    rm -rf /var/lib/apt/lists/*

# Pre-download models that don't need pip packages — these rarely change
# and are expensive to re-download. Placing before pip install means a new
# dependency doesn't force re-downloading all models (~300MB saved).

# WeSpeaker speaker verification model (~28MB)
# HuggingFace mirror first (cos.ap-shanghai CDN unreliable outside China)
RUN mkdir -p /home/augmentum/.wespeaker/en && \
    curl -fsSL -o /home/augmentum/.wespeaker/en/voxceleb_resnet34_LM.onnx \
      "https://huggingface.co/openspeech/wespeaker-models/resolve/main/voxceleb_resnet34_LM.onnx" \
      --retry 3 --retry-delay 2 || \
    curl -fsSL -o /home/augmentum/.wespeaker/en/voxceleb_resnet34_LM.onnx \
      "https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx" \
      --retry 2 || true && \
    chown -R augmentum:augmentum /home/augmentum/.wespeaker

# DTLN speech enhancement models (~3.7MB)
RUN mkdir -p /home/augmentum/.dtln && \
    curl -fsSL -o /home/augmentum/.dtln/model_1.onnx \
      "https://raw.githubusercontent.com/breizhn/DTLN/master/pretrained_model/model_1.onnx" \
      --retry 3 --retry-delay 2 && \
    curl -fsSL -o /home/augmentum/.dtln/model_2.onnx \
      "https://raw.githubusercontent.com/breizhn/DTLN/master/pretrained_model/model_2.onnx" \
      --retry 3 --retry-delay 2 && \
    chown -R augmentum:augmentum /home/augmentum/.dtln

# SmartTurn v3.2 turn-completion model (~8MB ONNX)
# Lives outside /data so the volume mount can't shadow it.
RUN mkdir -p /home/augmentum/.smart-turn && \
    curl -fSL --connect-timeout 30 --max-time 60 \
      -o /home/augmentum/.smart-turn/smart-turn-v3.2-cpu.onnx \
      "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx" \
      --retry 3 --retry-delay 5 && \
    chown -R augmentum:augmentum /home/augmentum/.smart-turn || \
    echo "SmartTurn model will download on first voice call"

# Kokoro TTS models — each download is a separate layer so failures are isolated
# and individual models can be cached independently.

# INT8 model (88MB, required — CPU default)
RUN mkdir -p /home/augmentum/.kokoro && \
    curl -fSL --connect-timeout 30 --max-time 300 \
      -o /home/augmentum/.kokoro/kokoro-v1.0.int8.onnx \
      "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx" \
      --retry 3 --retry-delay 5

# Voices (27MB, required)
RUN curl -fSL --connect-timeout 30 --max-time 300 \
      -o /home/augmentum/.kokoro/voices-v1.0.bin \
      "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" \
      --retry 3 --retry-delay 5

# FP16 model (169MB, optional — GPU quality). Separate layer so a timeout
# here doesn't block INT8 + voices. Verify download size to catch truncation.
RUN curl -fSL --connect-timeout 30 --max-time 600 \
      -o /home/augmentum/.kokoro/kokoro-v1.0.fp16.onnx \
      "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx" \
      --retry 5 --retry-delay 10 && \
    FP16_SIZE=$(wc -c < /home/augmentum/.kokoro/kokoro-v1.0.fp16.onnx 2>/dev/null || echo 0) && \
    if [ "$FP16_SIZE" -lt 100000000 ]; then \
      echo "WARNING: FP16 model download appears truncated ($FP16_SIZE bytes), removing" && \
      rm -f /home/augmentum/.kokoro/kokoro-v1.0.fp16.onnx; \
    fi ; \
    chown -R augmentum:augmentum /home/augmentum/.kokoro

# Install Python dependencies (layer cached separately from code).
# Uses BuildKit cache mount so pip doesn't re-download on every rebuild.
#
# Reproducible-build defense: requirements.lock is a fully hash-locked
# manifest derived from uv.lock. ``--require-hashes`` makes pip refuse
# any wheel whose sha256 doesn't match — so a PyPI compromise that
# re-publishes a transitive dep under the same version can't slip
# untrusted bytes into the image. Regenerate via
# ``./scripts/regen_requirements_lock.sh`` after editing pyproject.toml.
#
# webrtcvad-wheels + resemblyzer install with --no-deps in a separate
# pass because resemblyzer's setup metadata pulls webrtcvad (no
# pre-built wheel; needs gcc) instead of webrtcvad-wheels. The bare
# requirements.lock entry is overridden by this earlier install — pip
# treats them as already-satisfied and skips on the locked pass.
COPY requirements.lock pyproject.toml README.md ./
COPY augmentum/__init__.py augmentum/__init__.py
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --retries 5 --timeout 60 webrtcvad-wheels resemblyzer --no-deps && \
    pip install --require-hashes --retries 5 --timeout 300 -r requirements.lock && \
    pip install --no-deps .

# Pocket TTS (Kyutai) — the ultra-light CPU TTS tier: ~100M params /
# ~236MB weights, 6 languages (english/french/german/italian/portuguese/
# spanish), named voices + voice cloning from short reference clips, Mimi
# codec. This is the low-resource engine, so it belongs on the CPU image
# most no-GPU hosts pull. Installed AFTER the hash-locked requirements pass
# so its unpinned deps can't interfere with the ``--require-hashes`` install
# above. Disabled by default; AUGMENTUM_TTS_POCKET_BUILTIN=true enables it at
# runtime. Both steps are non-fatal (|| echo) so a build without network
# access to the weights still produces a working image — Kokoro remains the
# CPU default either way. Pre-caching the default English weights saves the
# first runtime synth the ~236MB download.
RUN pip install --no-cache-dir pocket-tts || \
    echo "pocket-tts install failed — engine stays disabled"
RUN mkdir -p /home/augmentum/.cache/pocket_tts && \
    python -c "from pocket_tts import TTSModel; TTSModel.load_model()" && \
    chown -R augmentum:augmentum /home/augmentum/.cache/pocket_tts || \
    echo "Pocket TTS weight pre-cache failed — first runtime synth will download (~236MB)"

# Moonshine STT model (~1 GB) — needs moonshine-voice package from pip
RUN python -m moonshine_voice.download --language en

# Pre-download nltk data used by g2p_en (phoneme-driven lip sync).
# averaged_perceptron_tagger ~2 MB + cmudict ~3 MB. Avoids first-call
# download stall (g2p_en lazily fetches both on first invocation).
# Explicit download_dir is required — NLTK_DATA env var alone does not
# auto-create the target directory, which silently strands the download.
RUN mkdir -p /home/augmentum/nltk_data && \
    python -c "import nltk; nltk.download('averaged_perceptron_tagger', download_dir='/home/augmentum/nltk_data'); nltk.download('cmudict', download_dir='/home/augmentum/nltk_data')" && \
    chown -R augmentum:augmentum /home/augmentum/nltk_data

# Pre-download the nomic embedding model (~140MB) used by memory + RAG.
# fastembed caches under $HOME/.cache/fastembed; we override HOME so the
# cache lands in the augmentum user's home (outside /data volume).
RUN HOME=/home/augmentum python -c "from fastembed import TextEmbedding; TextEmbedding('nomic-ai/nomic-embed-text-v1.5-Q')" && \
    chown -R augmentum:augmentum /home/augmentum/.cache

# Pre-download the rembg background-removal model (~170MB).
# U2NET_HOME drives the rembg cache location; postprocess.py checks this
# path first at runtime and skips the on-demand download.
RUN U2NET_HOME=/home/augmentum/.u2net python -c "from rembg import new_session; new_session('isnet-general-use')" && \
    chown -R augmentum:augmentum /home/augmentum/.u2net

# Copy application code and UI
COPY augmentum/ augmentum/
COPY ui/ ui/
COPY config/ config/
COPY .augmentum/powers/ .augmentum/powers/
COPY .claude/skills/ .claude/skills/
COPY ui/lib/bundled-avatars/ /data/bundled-avatars/

# Entrypoint runs as root to fix perms once, then drops to augmentum
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 6100

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "augmentum.main"]
