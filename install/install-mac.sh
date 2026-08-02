#!/usr/bin/env bash
set -euo pipefail

# Compatibility shim — macOS install is now handled by the universal installer
# (install.sh), which detects Linux vs macOS and Intel vs Apple Silicon on its
# own. This file stays so the older macOS URL keeps working; it just forwards
# to the one installer so there's a single source of truth and no drift.
#
#   curl -fsSL https://raw.githubusercontent.com/AugmentumHQ/Augmentum/main/install/install-mac.sh | bash
#
# New docs should point at install.sh directly.

RAW_BASE="https://raw.githubusercontent.com/AugmentumHQ/Augmentum/main"
echo "[i] macOS install is now the universal installer — forwarding to install.sh"
curl -fsSL "$RAW_BASE/install/install.sh" | bash
