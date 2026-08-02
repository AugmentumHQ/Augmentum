#!/usr/bin/env bash
# Self-healing doc-facts hook (wired as a Claude Code `Stop` hook).
#
# Keeps the <!--fact:NAME-->...<!--/--> blocks in CLAUDE.md / SKILL.md honest
# without anyone having to remember to run refresh_docs.py. The expensive part
# (model re-ingest inside refresh_docs.py) costs ~5s, so we GUARD on mtime:
# Python only runs when a fact-source file changed since the last heal.
#
#   - Common turn (no wiring-relevant edit) -> sub-100ms, no Python spawned.
#   - After a migration/route/config/settings/test edit -> heal (~5s, once).
#
# Exits 0 always (never blocks the agent from stopping). If it rewrote a doc,
# it prints a one-line notice so the agent knows to stage CLAUDE.md / SKILL.md
# alongside its other changes. It does NOT git-add — staging stays under the
# agent's control (parallel-session commit hygiene: `git commit --only <paths>`).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"  # scripts -> augmentum-dev -> skills -> .claude -> repo root
cd "${ROOT}"

CACHE_DIR=".augmentum-dev-cache"
STAMP="${CACHE_DIR}/.heal_stamp"
mkdir -p "${CACHE_DIR}"

# Fact-source paths: anything whose change can move a registered fact value.
# (migrations -> table/migration facts; config/routes/settings.js -> settings +
#  endpoint + registration facts; ui/scripts -> js_call facts; tests -> test facts)
SOURCES=(
  "augmentum/state/migrations"
  "augmentum/config.py"
  "augmentum/proxy"
  "ui/scripts"
  "tests"
)

# Decide whether to heal: stamp missing, or any source newer than the stamp.
needs_heal=0
if [[ ! -f "${STAMP}" ]]; then
  needs_heal=1
else
  # -newer is a fast filesystem walk; no Python interpreter spawned.
  if find "${SOURCES[@]}" -type f \
        \( -name '*.py' -o -name '*.js' -o -name '*.sql' \) \
        -newer "${STAMP}" -print -quit 2>/dev/null | grep -q .; then
    needs_heal=1
  fi
fi

if [[ "${needs_heal}" -eq 0 ]]; then
  exit 0
fi

# Heal. Capture output so we only speak up when something actually changed.
out="$(python "${SCRIPT_DIR}/refresh_docs.py" --apply 2>/dev/null || true)"
touch "${STAMP}"

if grep -q '^REWROTE' <<<"${out}"; then
  echo "augmentum-dev: refreshed stale doc-facts in CLAUDE.md / SKILL.md (review & stage with your other changes)."
  grep '^REWROTE' <<<"${out}" || true
fi

exit 0
