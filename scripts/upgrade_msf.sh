#!/usr/bin/env bash
#
# Upgrade the pinned Metasploit Framework .deb used by the pentest
# workspace profile (augmentum-workspace:pentest).
#
# The pinned version and sha256 live at METASPLOIT_VERSION /
# METASPLOIT_SHA256 in the repo root so Dockerfile.workspace, the
# pentest stage, and this script share one source of truth — same
# pattern as upgrade_llama_server.sh + LLAMA_SERVER_VERSION.
#
# Usage:
#   ./scripts/upgrade_msf.sh                 # use currently-pinned version, recompute sha
#   ./scripts/upgrade_msf.sh 6.4.42          # pin to specific version
#
# Why no --latest: Rapid7's apt pool doesn't expose a stable
# "current version" API; bumping is a deliberate operator decision.
# Browse https://apt.metasploit.com/pool/main/m/metasploit-framework/
# for the available .deb names.
#
# After this finishes, rebuild the pentest image so it picks up the
# new .deb:
#
#   docker build -t augmentum-workspace:pentest --target pentest \
#     --build-arg METASPLOIT_VERSION="$(cat METASPLOIT_VERSION)" \
#     --build-arg METASPLOIT_SHA256="$(cat METASPLOIT_SHA256)" \
#     -f Dockerfile.workspace .
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_FILE="${REPO_ROOT}/METASPLOIT_VERSION"
SHA_FILE="${REPO_ROOT}/METASPLOIT_SHA256"

target="${1:-}"

if [[ -z "$target" ]]; then
    target="$(tr -d '\n' < "$VERSION_FILE")"
    echo "Using currently-pinned version: $target"
fi

# Validate version shape — Rapid7 uses semver.
if [[ ! "$target" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "ERROR: version '$target' does not look like a metasploit-framework release (X.Y.Z)" >&2
    exit 1
fi

deb_name="metasploit-framework_${target}-1rapid7-1_amd64.deb"
deb_url="https://apt.metasploit.com/pool/main/m/metasploit-framework/${deb_name}"
tmp_deb="$(mktemp --suffix=.deb)"
trap 'rm -f "$tmp_deb"' EXIT

echo ""
echo "Fetching ${deb_name}…"
if ! curl -fsSL -o "$tmp_deb" "$deb_url"; then
    echo "ERROR: download failed. Confirm the version exists at:" >&2
    echo "  ${deb_url}" >&2
    exit 1
fi

echo "Computing sha256…"
sha="$(sha256sum "$tmp_deb" | awk '{print $1}')"
size_mb="$(du -m "$tmp_deb" | awk '{print $1}')"
echo "  size:   ${size_mb} MB"
echo "  sha256: ${sha}"

# Persist pins atomically — write both or neither (avoids the failure
# mode where the version bumped but the sha lagged).
printf '%s\n' "$target" > "${VERSION_FILE}.tmp"
printf '%s\n' "$sha" > "${SHA_FILE}.tmp"
mv "${VERSION_FILE}.tmp" "$VERSION_FILE"
mv "${SHA_FILE}.tmp" "$SHA_FILE"

echo ""
echo "[OK] Pinned METASPLOIT_VERSION=${target}"
echo "[OK] Pinned METASPLOIT_SHA256=${sha}"
echo ""
echo "Next: rebuild the pentest image"
echo "  ./start.sh build           # rebuilds all profile tags"
echo "  -- or --"
echo "  docker build -t augmentum-workspace:pentest --target pentest \\"
echo "    --build-arg METASPLOIT_VERSION=\"${target}\" \\"
echo "    --build-arg METASPLOIT_SHA256=\"${sha}\" \\"
echo "    -f Dockerfile.workspace ."
