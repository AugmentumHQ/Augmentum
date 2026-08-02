#!/bin/sh
# Fish Speech startup patches — fixes upstream bugs in the Docker image.
# Mounted as entrypoint wrapper; applies patches then starts the server.

set -e

# 1. Fix torchaudio scoping bug in reference_loader.py
#    Python sees `import torchaudio.io._load_audio_fileobj` in the except
#    block as a local assignment, making the module-level `torchaudio` import
#    unresolvable at line 39.
sed -i 's/import torchaudio.io._load_audio_fileobj/from torchaudio.io import _load_audio_fileobj/' \
    /app/fish_speech/inference_engine/reference_loader.py 2>/dev/null || true

# 2. Install blobfile for tiktoken local file loading
uv pip install -q blobfile 2>/dev/null || true

# 3. Tokenizer patch is bind-mounted directly over /app/fish_speech/tokenizer.py

echo "[patch] Fish Speech patches applied"

# Hand off to original entrypoint
exec /app/start_server.sh "$@"
