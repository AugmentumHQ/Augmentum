#!/usr/bin/env python3
"""Augmentum tooling-profile catalog validator.

Cross-checks the four sources of truth that have to stay aligned for a
workspace profile to actually work:

  1. ``augmentum/coder/profiles.py``     — the ``_PROFILES`` catalog
  2. ``Dockerfile.workspace``            — multi-stage build targets
  3. Repo-root pin files                 — e.g. METASPLOIT_VERSION /
                                            METASPLOIT_SHA256 for the
                                            pentest stage
  4. ``ui/scripts/coder.js``             — the dropdown must be
                                            server-fetched, not a
                                            hardcoded list

Catches drift that the wider audit doesn't:
  - A profile declared in code but missing a Dockerfile stage
    (and vice versa)
  - A vendored-binary stage that references a ``ARG _VERSION`` build
    arg without a matching pin file at repo root, or a pin file
    that's never read by any Dockerfile
  - Inheritance cycle in ``_PROFILES`` (would raise at resolve time —
    we want CI to catch it earlier)
  - UI dropdown that still hardcodes the catalog instead of fetching
    ``/api/coder/tooling-profiles``

Spec: docs/superpowers/specs/2026-06-02-tooling-profile-system-v2.md

Exit codes:
    0   clean
    1   findings present
    2   script error (couldn't find project root, parse failure, etc.)
"""

from __future__ import annotations

import re
import sys

import _common  # noqa: F401 — UTF-8-safe stdout/stderr

ROOT = _common.ROOT


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_profile_catalog() -> tuple[dict[str, dict], list[str]]:
    """Import profiles.py and return ``(catalog, errors)``.

    Reads ``_PROFILES`` directly rather than re-parsing the source —
    the dataclass is the source of truth, anything else would diverge.
    Returns an empty catalog + an error message if the import fails.
    """
    errors: list[str] = []
    sys.path.insert(0, str(ROOT))
    try:
        from augmentum.coder import profiles as _profiles
    except Exception as exc:  # noqa: BLE001 — surface the import error
        errors.append(f"failed to import augmentum.coder.profiles: {exc}")
        return {}, errors

    catalog: dict[str, dict] = {}
    for prof in _profiles.all_profiles():
        catalog[prof.id] = {
            "label": prof.label,
            "description": prof.description,
            "inherits": prof.inherits,
            "image_tag": prof.image_tag,
            "extra_caps": list(prof.extra_caps),
            "est_size_mb": prof.est_size_mb,
            "est_setup_sec": prof.est_setup_sec,
            "notice": prof.notice,
        }
    return catalog, errors


def _dockerfile_stages() -> set[str]:
    """Parse Dockerfile.workspace for ``FROM <base> AS <stage>`` targets."""
    path = ROOT / "Dockerfile.workspace"
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8", errors="replace")
    # Matches both ``FROM x AS y`` and ``FROM x:tag AS y``. Case-insensitive
    # — Docker treats AS / as / As identically.
    return {
        m.group(1).lower()
        for m in re.finditer(r"^FROM\s+\S+\s+AS\s+(\S+)", text, re.MULTILINE | re.IGNORECASE)
    }


def _dockerfile_args() -> set[str]:
    """Parse Dockerfile.workspace for ``ARG <name>`` declarations."""
    path = ROOT / "Dockerfile.workspace"
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        m.group(1)
        for m in re.finditer(r"^ARG\s+([A-Z_][A-Z0-9_]*)", text, re.MULTILINE)
    }


def _all_dockerfile_args() -> set[str]:
    """Parse every Dockerfile* at the repo root for ARG declarations.

    Used by the pin-file orphan check — a pin file may belong to any
    Dockerfile in the repo, not just Dockerfile.workspace. Without this
    the check flags LLAMA_SERVER_VERSION (read by Dockerfile.llama-server)
    as orphaned.
    """
    args: set[str] = set()
    for path in ROOT.glob("Dockerfile*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        args.update(
            m.group(1)
            for m in re.finditer(r"^ARG\s+([A-Z_][A-Z0-9_]*)", text, re.MULTILINE)
        )
    return args


def _coder_js_text() -> str:
    path = ROOT / "ui" / "scripts" / "coder.js"
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_required_fields(catalog: dict[str, dict]) -> list[str]:
    findings: list[str] = []
    for prof_id, prof in catalog.items():
        if not prof["label"]:
            findings.append(f"profile '{prof_id}': missing label")
        if not prof["description"]:
            findings.append(f"profile '{prof_id}': missing description")
        if not prof["image_tag"]:
            findings.append(f"profile '{prof_id}': missing image_tag")
        if prof["est_size_mb"] <= 0:
            findings.append(
                f"profile '{prof_id}': est_size_mb must be > 0 "
                f"(got {prof['est_size_mb']})"
            )
    return findings


def check_inheritance_resolves(catalog: dict[str, dict]) -> list[str]:
    """Walk each profile's inheritance chain. Catches cycles + unknown parents."""
    findings: list[str] = []
    for prof_id in catalog:
        seen: set[str] = set()
        cur: str | None = prof_id
        while cur is not None:
            if cur in seen:
                findings.append(
                    f"profile '{prof_id}': inheritance cycle through '{cur}'"
                )
                break
            seen.add(cur)
            if cur not in catalog:
                findings.append(
                    f"profile '{prof_id}': inherits from unknown profile '{cur}'"
                )
                break
            cur = catalog[cur]["inherits"]
    return findings


def check_dockerfile_alignment(
    catalog: dict[str, dict],
    stages: set[str],
) -> list[str]:
    """Each profile's image_tag :suffix should match a Dockerfile stage."""
    findings: list[str] = []
    for prof_id, prof in catalog.items():
        tag = prof["image_tag"]
        if ":" not in tag:
            findings.append(
                f"profile '{prof_id}': image_tag '{tag}' has no :stage suffix"
            )
            continue
        stage = tag.split(":", 1)[1].lower()
        if stage not in stages:
            findings.append(
                f"profile '{prof_id}': image_tag '{tag}' has no matching "
                f"'FROM ... AS {stage}' stage in Dockerfile.workspace"
            )
    # Stages declared in Dockerfile but unused by any profile are a softer
    # finding — they don't break correctness but suggest dead build paths.
    used_stages = {
        prof["image_tag"].split(":", 1)[1].lower()
        for prof in catalog.values()
        if ":" in prof["image_tag"]
    }
    for stage in stages - used_stages:
        findings.append(
            f"Dockerfile stage '{stage}' has no profile claiming "
            f"image_tag 'augmentum-workspace:{stage}'"
        )
    return findings


def check_vendored_pins(args: set[str]) -> list[str]:
    """Every ``ARG X_VERSION`` (or ``X_SHA256``) needs a matching pin file at repo root.

    Both directions: an ARG without a pin file means the build will silently
    use Docker's empty-default behaviour; a pin file without an ARG means
    nothing reads it (or someone renamed the build arg and orphaned the
    file).
    """
    findings: list[str] = []
    # Args whose name ends in _VERSION or _SHA256 are considered pinned-
    # artifact build args by convention.
    pin_args = {a for a in args if a.endswith(("_VERSION", "_SHA256"))}
    for arg in sorted(pin_args):
        pin_file = ROOT / arg
        if not pin_file.is_file():
            findings.append(
                f"Dockerfile.workspace ARG {arg} has no matching pin file "
                f"at repo root ({arg})"
            )

    # Reverse direction — pin files that nothing reads. A pin file is
    # considered "owned" if either (a) a Dockerfile has a same-named ARG,
    # or (b) an upgrade script under scripts/ mentions the filename. The
    # second clause covers cases where the pin filename and the build arg
    # diverge intentionally (LLAMA_SERVER_VERSION ↔ ARG LLAMA_CPP_VERSION,
    # bridged by scripts/upgrade_llama_server.sh).
    convention_re = re.compile(r"^(?:[A-Z][A-Z0-9_]*)_(?:VERSION|SHA256)$")
    all_args = _all_dockerfile_args()
    scripts_text = ""
    scripts_dir = ROOT / "scripts"
    if scripts_dir.is_dir():
        for sp in scripts_dir.iterdir():
            if sp.is_file() and sp.suffix in {".sh", ".bat", ".py"}:
                scripts_text += sp.read_text(encoding="utf-8", errors="replace")
    for child in ROOT.iterdir():
        if not child.is_file():
            continue
        name = child.name
        if not convention_re.match(name):
            continue
        if name in all_args:
            continue
        if name in scripts_text:
            continue
        findings.append(
            f"pin file '{name}' is not referenced by any Dockerfile ARG or "
            f"upgrade script — orphaned, or the build arg was renamed"
        )
    return findings


def check_ui_dropdown_server_fetched() -> list[str]:
    """coder.js must fetch the profile list from the API, not hardcode it.

    The legacy v1 ``TOOLING_PROFILES = [...]`` const drift was the whole
    reason for PR 1 — catch any regression that re-introduces a hardcoded
    catalog before it ships.
    """
    findings: list[str] = []
    text = _coder_js_text()
    if not text:
        # coder.js missing is its own audit elsewhere; not our job to flag
        return findings
    # Hardcoded array of >=3 profile-shaped object literals is the
    # regression we're catching — match the v1 shape:
    #   { id: "standard", label: "Standard", description: ... }
    suspicious = re.search(
        r"const\s+TOOLING_PROFILES\s*=\s*\[",
        text,
    )
    if suspicious:
        findings.append(
            "ui/scripts/coder.js declares a hardcoded TOOLING_PROFILES array "
            "— the dropdown should fetch /api/coder/tooling-profiles instead"
        )
    if "/api/coder/tooling-profiles" not in text:
        findings.append(
            "ui/scripts/coder.js never calls /api/coder/tooling-profiles "
            "— the dropdown can't be populated dynamically"
        )
    return findings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print(_common.bold("  Tooling-profile catalog validator"))
    print()

    catalog, load_errs = _load_profile_catalog()
    if load_errs:
        for err in load_errs:
            print(_common.red(f"  ERROR: {err}"))
        return 2

    stages = _dockerfile_stages()
    args = _dockerfile_args()

    findings: list[tuple[str, str]] = []  # (severity, message)

    print(_common.cyan(f"  [1/5] Required fields ({len(catalog)} profiles)..."))
    for f in check_required_fields(catalog):
        findings.append(("error", f))

    print(_common.cyan("  [2/5] Inheritance resolves..."))
    for f in check_inheritance_resolves(catalog):
        findings.append(("error", f))

    print(_common.cyan(f"  [3/5] Dockerfile alignment ({len(stages)} stages)..."))
    for f in check_dockerfile_alignment(catalog, stages):
        findings.append(("error", f))

    print(_common.cyan(f"  [4/5] Vendored-artifact pin files ({len(args)} build args)..."))
    for f in check_vendored_pins(args):
        findings.append(("warning", f))

    print(_common.cyan("  [5/5] UI dropdown server-fetched..."))
    for f in check_ui_dropdown_server_fetched():
        findings.append(("error", f))

    errors = [m for sev, m in findings if sev == "error"]
    warnings = [m for sev, m in findings if sev == "warning"]

    print()
    print(_common.bold("  Summary"))
    print(f"    Profiles checked:       {len(catalog)}")
    print(f"    Dockerfile stages:      {len(stages)}")
    print(f"    Build args:             {len(args)}")
    print()

    if errors:
        print(_common.red(f"  ERRORS ({len(errors)}):"))
        for e in errors:
            print(f"    - {e}")
        print()
    if warnings:
        print(_common.yellow(f"  WARNINGS ({len(warnings)}):"))
        for w in warnings:
            print(f"    - {w}")
        print()

    if not errors and not warnings:
        print(_common.green("  All clear — catalog, Dockerfile, pins, and UI are aligned."))
        print()

    if errors:
        print(_common.red(f"  {len(errors)} error(s), {len(warnings)} warning(s)"))
        return 1
    print(_common.green(f"  0 errors, {len(warnings)} warning(s)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
