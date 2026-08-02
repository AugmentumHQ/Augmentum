#!/usr/bin/env bash
# Regenerate requirements.lock from uv.lock + pyproject.toml.
#
# requirements.lock is the hash-locked manifest the Dockerfiles install
# from with ``pip install --require-hashes``. It's derived from uv.lock
# (the canonical Python lockfile this project maintains) and includes the
# base + [image-cpu] + [file-extras] extras — matching what the GPU
# Dockerfile installs after the heavy-CUDA layer.
#
# Run this after any change to pyproject.toml's dependency list (or
# after running ``uv lock`` to bump transitive pins). Commit the
# resulting requirements.lock alongside the pyproject.toml change.
#
# The dev-only extras (pytest, ruff, etc.) are intentionally excluded —
# the container image only needs runtime deps. Local dev still installs
# them via ``pip install -e ".[dev]"``.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed — install with: pipx install uv" >&2
    exit 1
fi

HEADER="$(cat <<'EOF'
# Hash-locked dependency manifest for reproducible Docker builds.
#
# Generated from uv.lock + pyproject.toml. EVERY package carries a
# sha256 hash; pip refuses to install a wheel that doesn't match.
# This is the supply-chain defense against a transitive-dep replacement
# attack (someone yanks an old version and publishes a new one with
# the same name).
#
# Includes deps from base + [image-cpu] + [file-extras] extras.
# The Docker GPU variant additionally pip-installs CUDA torch/xformers
# from PyTorch's index BEFORE this lockfile pass — those are explicit
# at the Dockerfile layer because the CUDA index doesn't carry the
# same wheels uv resolves from PyPI.
#
# To regenerate after editing pyproject.toml:
#   uv lock                         # update uv.lock
#   ./scripts/regen_requirements_lock.sh
#
# Do NOT hand-edit. The hashes are load-bearing.

EOF
)"

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

uv export \
    --format requirements-txt \
    --no-dev \
    --extra image-cpu \
    --extra file-extras \
    --no-emit-project \
    --no-header > "$TMP"

{
    printf '%s\n' "$HEADER"
    cat "$TMP"
} > requirements.lock

echo "Wrote requirements.lock ($(wc -l < requirements.lock) lines, $(grep -c '^[a-zA-Z]' requirements.lock || true) packages)."
