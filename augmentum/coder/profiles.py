"""Workspace tooling profile catalog.

Single source of truth for the per-profile install/capability data that the
container manager consumes when creating workspaces. Each profile declares
package lists, post-install commands, capability adds, and image-tag
preferences; ``resolve()`` walks the inheritance chain to flatten them.

The v1 implementation hardcoded these as parallel ``if profile in {...}``
branches and a parallel UI list. This module replaces both — see
docs/superpowers/specs/2026-06-02-tooling-profile-system-v2.md.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolingProfile:
    """Declarative definition of a single profile.

    Inheritance: ``inherits`` references another profile id (or ``None`` for
    a root). At resolve time the chain is walked parent-first and the
    package/capability lists are concatenated. Image tags are NOT inherited
    — each profile declares its own prebake target.
    """
    id: str
    label: str
    description: str
    inherits: str | None
    apt_packages: tuple[str, ...] = ()
    pip_packages: tuple[str, ...] = ()
    npm_packages: tuple[str, ...] = ()
    post_install: tuple[str, ...] = ()
    extra_caps: tuple[str, ...] = ()
    est_size_mb: int = 0
    est_setup_sec: int = 0
    network_policy: str = "default"
    image_tag: str = "augmentum-workspace"
    notice: str = ""


@dataclass(frozen=True)
class ResolvedProfile:
    """Profile with inheritance chain flattened, in chain (parent-first) order."""
    id: str
    label: str
    description: str
    apt_packages: tuple[str, ...]
    pip_packages: tuple[str, ...]
    npm_packages: tuple[str, ...]
    post_install: tuple[str, ...]
    extra_caps: tuple[str, ...]
    est_size_mb: int
    est_setup_sec: int
    network_policy: str
    image_tag: str
    notice: str


# Power's package lists — preserved exactly from v1
# (augmentum/coder/containers.py:97-102) so resolve("power") emits an
# identical install line set. Tuple form because dataclasses are frozen.
_POWER_APT = (
    "procps", "iproute2", "lsof", "dnsutils", "netcat-openbsd", "psmisc",
    "net-tools", "cmake", "ninja-build", "gdb", "strace", "shellcheck",
    "shfmt", "zip", "rsync",
)
_POWER_PIP = ("uv", "pipx", "pre-commit", "pytest-cov", "ipython")
_POWER_NPM = ("pnpm", "yarn")

# Pentest profile package lists. APT covers the canonical Kali-on-Ubuntu
# subset; pip covers framework + protocol libs that don't have well-kept
# Ubuntu packages. post_install vendors gobuster + SecLists + Metasploit
# via pinned artifacts (see Dockerfile.workspace and the matching
# METASPLOIT_VERSION / METASPLOIT_SHA256 pins at repo root).
_PENTEST_APT = (
    "nmap", "masscan", "nikto", "sqlmap", "hydra", "hashcat", "john",
    "dirb", "wfuzz", "tshark", "whatweb", "dnsenum", "dnsrecon",
    "smbclient", "enum4linux", "libimage-exiftool-perl",
    "responder", "exploitdb", "zaproxy",
)
_PENTEST_PIP = ("theHarvester", "impacket", "scapy", "ffuf-py", "zap-cli")
# Idempotent shell lines — workspaces created on the :pentest prebake skip
# these entirely (handled by ``emit_install_lines`` + the prebaked check
# in ``containers.py``). They exist so an ubuntu:24.04 fallback workspace
# can still bring itself up to pentest parity at first boot.
_PENTEST_POST = (
    # Gobuster — single static binary, no apt package on Ubuntu 24.04.
    'if ! command -v gobuster >/dev/null 2>&1; then '
    'curl -fsSL -o /tmp/gobuster.tgz '
    '"https://github.com/OJ/gobuster/releases/download/v3.6.0/'
    'gobuster_Linux_x86_64.tar.gz" '
    '&& tar -xzf /tmp/gobuster.tgz -C /usr/local/bin gobuster '
    '&& rm /tmp/gobuster.tgz; fi',
    # SecLists wordlists — large (~700MB) but the canonical word list pack
    # for web/fuzzing work. Skip when already present.
    "[ -d /opt/seclists ] || git clone --depth=1 "
    "https://github.com/danielmiessler/SecLists.git /opt/seclists",
    # Metasploit is vendored at build time only (sha256-pinned .deb). The
    # fallback path leaves a one-liner the user can run manually rather
    # than re-downloading 600MB unverified at runtime — same threat-model
    # logic as the llama-server vendored binary.
    "command -v msfconsole >/dev/null 2>&1 || "
    "echo 'metasploit not installed on ubuntu:24.04 fallback — "
    "build the augmentum-workspace:pentest image to get it' "
    "> /workspace/.augmentum/pentest_msf_note.txt",
)

# Creative / 3D profile. Blender is the creation engine; the bpy Python API
# is the interface (the coder writes a build script and runs it headless).
# xvfb gives Blender a virtual framebuffer so Eevee / GL renders work with no
# display, on CPU — no GPU dependency for the MVP path. The libGL/libX* set is
# the runtime Blender links against even in --background mode. trimesh +
# pygltflib let pipeline steps inspect/patch glTF without launching Blender.
_CREATIVE_APT = (
    "blender", "xvfb",
    "libgl1", "libglu1-mesa", "libxi6", "libxrender1", "libxxf86vm1", "libsm6",
)
_CREATIVE_PIP = ("trimesh", "pygltflib")
# Idempotent fallback for an ubuntu:24.04 workspace not on the :creative
# prebake. Distro Blender lags and is large; rather than pull ~500MB unverified
# at first boot we leave a note (same threat-model reasoning as the pentest msf
# fallback). Workspaces created on the :creative image skip this entirely.
_CREATIVE_POST = (
    'command -v blender >/dev/null 2>&1 || '
    'echo "blender not installed on ubuntu:24.04 fallback — '
    'build the augmentum-workspace:creative image to get it" '
    "> /workspace/.augmentum/creative_blender_note.txt",
    # Drop the baked version-exact bpy API reference + cheatsheet into the
    # workspace so the coder model can grep it (raises the ceiling for every
    # model size — see scripts/gen_bpy_reference.py). Idempotent; skips if the
    # reference wasn't baked (ubuntu:24.04 fallback has no /opt copy).
    '[ -d /opt/blender-reference ] && mkdir -p /workspace/.reference/blender '
    '&& cp -n /opt/blender-reference/*.md /workspace/.reference/blender/ '
    "2>/dev/null || true",
)


# Profile catalog. Declaration order is preserved in ``all_profiles()`` and
# therefore in the UI dropdown order. PR 4 will append data.
_PROFILES: dict[str, ToolingProfile] = {
    "standard": ToolingProfile(
        id="standard",
        label="Standard",
        description="Fast baseline for most Python, JS, Go, and Rust work.",
        inherits=None,
        image_tag="augmentum-workspace:standard",
        est_size_mb=800,
        est_setup_sec=2,
    ),
    "power": ToolingProfile(
        id="power",
        label="Power",
        description="Adds process/network inspection, uv/pipx, pnpm/yarn, "
                    "and build/debug tools.",
        inherits="standard",
        apt_packages=_POWER_APT,
        pip_packages=_POWER_PIP,
        npm_packages=_POWER_NPM,
        image_tag="augmentum-workspace:power",
        est_size_mb=1000,
        est_setup_sec=5,
    ),
    # Since 2026-07-17 the browser profile no longer installs Playwright +
    # Chromium per workspace (~600MB + minutes of first setup) — browser
    # automation runs on the shared agent-browser sidecar service
    # (compose.browser.yaml). The profile id stays for compatibility
    # (it's the default tooling profile and existing workspaces reference
    # it); legacy workspaces that already have Playwright installed keep
    # working via the in-workspace ladder rung.
    "browser": ToolingProfile(
        id="browser",
        label="Browser/Test",
        description="Power profile; browser automation via the shared "
                    "browser sidecar service (no per-workspace Chromium).",
        inherits="power",
        image_tag="augmentum-workspace:browser",
        est_size_mb=1000,
        est_setup_sec=5,
    ),
    "pentest": ToolingProfile(
        id="pentest",
        label="Pentest",
        description="Power profile plus pentest CLI: nmap, ZAP, msf, "
                    "sqlmap, hashcat, SecLists, gobuster, and more. "
                    "Authorized testing only.",
        inherits="power",
        apt_packages=_PENTEST_APT,
        pip_packages=_PENTEST_PIP,
        post_install=_PENTEST_POST,
        # NET_RAW enables raw socket creation so nmap SYN scan / arp /
        # tcpdump-style capture works without root. It does NOT bypass
        # the ``coder_workspace_block_host_pivot`` iptables block (caps
        # only enable a syscall surface; they don't override netfilter).
        extra_caps=("NET_RAW",),
        image_tag="augmentum-workspace:pentest",
        est_size_mb=4000,
        est_setup_sec=15,
        network_policy="raw_sockets",
        notice="Pentest profile. Tools are for authorized testing only — "
               "the coder will ask for scope (target, contract reference) "
               "before scanning external hosts. Localhost and RFC1918 are "
               "exempt (CTF / local dev).",
    ),
    "creative": ToolingProfile(
        id="creative",
        label="Creative / 3D",
        description="Power profile plus Blender headless + glTF pipeline "
                    "(trimesh, pygltflib) for procedural 3D asset creation "
                    "and game-ready GLB export.",
        inherits="power",
        apt_packages=_CREATIVE_APT,
        pip_packages=_CREATIVE_PIP,
        post_install=_CREATIVE_POST,
        image_tag="augmentum-workspace:creative",
        est_size_mb=2500,
        est_setup_sec=12,
        notice="Creative profile: Blender runs headless via the bpy Python "
               "API. Renders are produced under xvfb on CPU (Eevee); GPU "
               "(Cycles/OptiX) is used only when the workspace or a fabric "
               "peer advertises a GPU.",
    ),
}


def all_profiles() -> list[ToolingProfile]:
    """Return profile catalog in declaration order."""
    return list(_PROFILES.values())


def has_profile(profile_id: str) -> bool:
    return profile_id in _PROFILES


def chain(profile_id: str) -> list[ToolingProfile]:
    """Walk inheritance chain parent-first. Raises on cycles or unknown ids."""
    if profile_id not in _PROFILES:
        raise ValueError(f"unknown tooling profile: {profile_id!r}")
    seen: set[str] = set()
    rev_chain: list[ToolingProfile] = []
    cur_id: str | None = profile_id
    while cur_id is not None:
        if cur_id in seen:
            raise ValueError(
                f"tooling profile inheritance cycle through {cur_id!r}"
            )
        if cur_id not in _PROFILES:
            raise ValueError(
                f"tooling profile {profile_id!r} inherits from unknown "
                f"profile {cur_id!r}"
            )
        seen.add(cur_id)
        prof = _PROFILES[cur_id]
        rev_chain.append(prof)
        cur_id = prof.inherits
    rev_chain.reverse()  # parent-first
    return rev_chain


def resolve(profile_id: str) -> ResolvedProfile:
    """Walk inheritance chain and accumulate package/capability lists.

    Parent packages come first in the flattened list, leaf last. Leaf
    profile owns the identity fields (id, label, image_tag, network_policy,
    notice). Image tag and capability additions accumulate from the chain;
    notice/network_policy/sizes are leaf-only.
    """
    walked = chain(profile_id)
    leaf = walked[-1]

    def accum(field_name: str) -> tuple[str, ...]:
        out: list[str] = []
        for p in walked:
            out.extend(getattr(p, field_name))
        return tuple(out)

    return ResolvedProfile(
        id=leaf.id,
        label=leaf.label,
        description=leaf.description,
        apt_packages=accum("apt_packages"),
        pip_packages=accum("pip_packages"),
        npm_packages=accum("npm_packages"),
        post_install=accum("post_install"),
        extra_caps=accum("extra_caps"),
        est_size_mb=leaf.est_size_mb,
        est_setup_sec=leaf.est_setup_sec,
        network_policy=leaf.network_policy,
        image_tag=leaf.image_tag,
        notice=leaf.notice,
    )


def metadata(profile_id: str) -> dict:
    """Drop-in replacement for the legacy ``_tooling_profile_metadata``.

    Returns the JSON written to ``/workspace/.augmentum/tooling-profile.json``
    so the agent's runtime introspection sees the same shape as v1.
    """
    resolved = resolve(profile_id)
    # The "includes" list is human-readable for the agent's environment
    # dump. Kept stable with v1 for standard/power/browser; new profiles
    # synthesize from their description.
    if resolved.id == "power":
        includes = [
            "process/network inspection",
            "modern Python package tooling",
            "modern JS package managers",
            "native build/debug helpers",
        ]
    elif resolved.id == "browser":
        includes = [
            "power profile",
            "browser automation via the shared browser sidecar "
            "(browser_* tools; no in-workspace Chromium)",
        ]
    elif resolved.id == "pentest":
        includes = [
            "power profile",
            "network scanning (nmap, masscan, tshark)",
            "web app testing (sqlmap, nikto, wfuzz, zaproxy, gobuster)",
            "credential / hash tooling (hydra, hashcat, john, responder)",
            "framework + protocol libs (Metasploit, impacket, scapy)",
            "SecLists wordlists at /opt/seclists",
            "raw-socket capability (NET_RAW)",
        ]
    elif resolved.id == "creative":
        includes = [
            "power profile",
            "Blender headless 3D creation via the bpy Python API "
            "(blender_run tool; xvfb/Eevee on CPU)",
            "glTF pipeline libs (trimesh, pygltflib)",
            "game-ready GLB export",
        ]
    elif resolved.id == "standard":
        includes = ["standard Augmentum workspace baseline"]
    else:
        includes = [resolved.description]
    return {"tooling_profile": resolved.id, "includes": includes}


def emit_install_lines(profile_id: str) -> list[str]:
    """Emit the setup-script lines for a profile's install chain.

    Walks the chain parent-first and emits one batched install per
    (level, manager) so v1's command-shape is preserved exactly (same
    number of apt-get / pip / npm invocations, same ordering). The
    DEBIAN_FRONTEND header is emitted once per build, the first time a
    level contributes anything; ``corepack enable`` is emitted once,
    immediately before the first npm install. ``post_install`` lines
    emit after the manager lines for the level they belong to.

    Standard contributes nothing — its tooling is baked into the
    prebaked image. The function is safe to call regardless of which
    image actually backs the workspace; the result is identical.
    """
    walked = chain(profile_id)

    lines: list[str] = []
    emitted_header = False
    emitted_corepack = False
    for level in walked:
        # Skip levels that contribute nothing (e.g. standard).
        if not (
            level.apt_packages
            or level.pip_packages
            or level.npm_packages
            or level.post_install
        ):
            continue
        if not emitted_header:
            lines.append("export DEBIAN_FRONTEND=noninteractive")
            emitted_header = True
        if level.apt_packages:
            lines.append(
                "apt-get update -qq 2>/dev/null; "
                "apt-get install -y -qq --no-install-recommends "
                + " ".join(level.apt_packages)
            )
        if level.pip_packages:
            lines.append(
                "python3 -m pip install --no-cache-dir --ignore-installed "
                + " ".join(level.pip_packages)
            )
        if level.npm_packages:
            if not emitted_corepack:
                lines.append("corepack enable 2>/dev/null || true")
                emitted_corepack = True
            lines.append("npm install -g " + " ".join(level.npm_packages))
        for cmd in level.post_install:
            lines.append(cmd)
    return lines
