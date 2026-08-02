# Sesame CSM-1B streaming TTS sidecar.
# Base matches CSM's pinned torch==2.4.0 + CUDA 12.4 (the fork's tested
# stack). Mirrors services/chatterbox-turbo/Dockerfile.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# build-essential is REQUIRED: the csm-streaming fork uses torch.compile
# (inductor), which JIT-compiles C++/CUDA kernels at load time and needs a
# C/C++ compiler. The pytorch -runtime base ships none, so load failed with
# "Failed to find C compiler".
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg git libsndfile1 build-essential && \
    rm -rf /var/lib/apt/lists/*

# Vendor the streaming fork (adds generate_stream() incremental decoding
# to stock CSM). We import its generator/models/watermarking modules;
# its Flask demo + whisper/vllm deps are NOT installed (see requirements).
RUN git clone --depth 1 https://github.com/davidbrowne17/csm-streaming /opt/csm
ENV PYTHONPATH=/opt/csm

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# gated weights (sesame/csm-1b + meta-llama/Llama-3.2-1B) need an HF token
# at runtime — passed in via the compose overlay (HF_TOKEN).
# CC/CXX make the compiler explicit for inductor. TORCHINDUCTOR_CACHE_DIR
# persists the compiled kernels onto the /cache volume so the (slow) first
# compile happens once, not on every container start / model reload.
ENV HOST=0.0.0.0 PORT=8920 DEVICE=cuda VOICE_LIBRARY_DIR=/voices \
    HF_HOME=/cache/hf CC=gcc CXX=g++ TORCHINDUCTOR_CACHE_DIR=/cache/inductor
EXPOSE 8920

# Long start period: first boot downloads ~3-4GB of gated weights + loads.
HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=600s \
    CMD curl --fail http://0.0.0.0:8920/health || exit 1

CMD ["python", "app.py"]
