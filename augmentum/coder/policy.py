"""Per-tool permission policy — workspace-scoped, file-driven.

Sits between the agent's tool dispatcher and the existing
:class:`augmentum.coder.permissions.PermissionRegistry` modal prompt.
For each tool call the resolver returns ``"allow"`` / ``"ask"`` /
``"deny"``; only ``"ask"`` actually hits the user-facing modal. This
is the layer that turns "approve every shell command" — the friction
that pushes users to disable approvals entirely — into the converged
shape: ``git *`` is auto-allowed, ``rm -rf *`` is auto-denied, the
remaining surface gets a modal.

Policy file: ``.augmentum/permissions.toml`` inside the workspace
container. Evaluated in order; first matching rule wins. When no
rule matches, the per-tool default applies; when no per-tool default
exists either, the top-level ``fallback`` applies (``ask`` if
absent).

Example::

    # Read-only tools never prompt
    [[rule]]
    tool = "file_read"
    action = "allow"

    [[rule]]
    tool = "file_list"
    action = "allow"

    [[rule]]
    tool = "code_grep"
    action = "allow"

    [[rule]]
    tool = "find_files"
    action = "allow"

    # Safe git ops auto-approve; destructive ones auto-deny
    [[rule]]
    tool = "shell_exec"
    arg_glob = { command = "git status*" }
    action = "allow"

    [[rule]]
    tool = "shell_exec"
    arg_glob = { command = "git log*" }
    action = "allow"

    [[rule]]
    tool = "shell_exec"
    arg_glob = { command = "rm -rf*" }
    action = "deny"

    # Anything not matched above: ask
    [defaults]
    fallback = "ask"

Implementation notes:

* Policy reads run on every check (no caching). The file is small,
  reads are infrequent (one per tool call), and a stale cache
  causing surprise allows/denies is much worse than the few ms cost.
* Missing file → builtin defaults: read-only tools allow, every
  other tool asks. Operators get a "safe by default" floor without
  any policy authoring.
* Glob matching uses :mod:`fnmatch` (shell-style wildcards). Case
  sensitivity matches the underlying argument (``command`` is
  case-sensitive; the ``rm`` shell is too).
* Unknown ``action`` values fall back to ``ask`` — the cautious
  default. A misconfigured policy never silently allows.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from typing import Any, Literal

try:
    import tomllib  # Python 3.11+ stdlib
except ImportError:  # pragma: no cover — only Python <3.11 paths
    tomllib = None  # type: ignore[assignment]

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


PolicyAction = Literal["allow", "ask", "deny"]


# Read-only tool names. These are the ones the default policy auto-
# allows when no .augmentum/permissions.toml exists. Keep narrow —
# any tool that mutates state or reads outside the workspace should
# default to ``ask``. Updated when new read-only tools land.
_BUILTIN_READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "file_read",
    "file_list",
    "dir_tree",
    "code_grep",
    "find_files",
    "doc_search",
    "pack_search",
    "code_search",
    "env_info",
    "task_list",
    "ask_question",
    "browser_screenshot",
    "browser_evaluate",
    "browser_wait",
    "browser_extract",
    # Sidecar-native observers (persistent browser): read page/console
    # state only. interact/navigate/tabs/find act on the page → ask.
    "browser_get",
    "browser_console",
    "db_inspect",
})


# Tools that ALWAYS ask, regardless of "fallback" setting. These are
# the highest-blast-radius surfaces; allowing them via a glob requires
# an explicit rule. The list is short on purpose — most write tools
# are fine to auto-allow by glob (e.g. file_write with path glob).
_BUILTIN_ALWAYS_ASK_TOOLS: frozenset[str] = frozenset({
    "http_request",
})


# Repo-relative path where the per-workspace policy file lives.
POLICY_PATH = "/workspace/.augmentum/permissions.toml"


@dataclass
class _Rule:
    """One compiled policy rule.

    ``arg_glob`` keys must all match (AND semantics) for the rule to
    fire. An empty ``arg_glob`` means "match any args for this tool".
    Tool name must match exactly (no globbing on tool names — the set
    is small and explicit > clever).
    """

    tool: str
    action: PolicyAction
    arg_glob: dict[str, str] = field(default_factory=dict)

    def matches(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
        if self.tool != tool_name:
            return False
        for arg_key, pattern in self.arg_glob.items():
            value = tool_input.get(arg_key)
            if value is None:
                # Scalar arg absent — check the batch (list) spelling of
                # the same arg so a path-scoped rule can't be bypassed by
                # calling the batch form (file_read paths=[...] vs
                # path=...). Deny/ask rules claim the call if ANY element
                # matches (conservative: one protected path taints the
                # batch); allow rules require ALL elements to match (an
                # allow-glob must cover everything the call reads).
                batch = tool_input.get(f"{arg_key}s")
                if (
                    isinstance(batch, list) and batch
                    and all(isinstance(v, str) for v in batch)
                ):
                    quantifier = all if self.action == "allow" else any
                    if quantifier(
                        fnmatch.fnmatchcase(v, pattern) for v in batch
                    ):
                        continue
                return False
            if not isinstance(value, str):
                # Non-string arg value can't match a string glob —
                # the rule's specificity wasn't met, so it doesn't
                # claim this call.
                return False
            if not fnmatch.fnmatchcase(value, pattern):
                return False
        return True


@dataclass
class Policy:
    """Compiled, in-memory permission policy.

    Constructed via :func:`load_policy` (reads the workspace's
    permissions.toml, applies defaults). The single entry point
    callers use is :meth:`decide` — given a tool call, return one
    of ``allow`` / ``ask`` / ``deny``.
    """

    rules: list[_Rule] = field(default_factory=list)
    tool_defaults: dict[str, PolicyAction] = field(default_factory=dict)
    fallback: PolicyAction = "ask"

    def decide(self, tool_name: str, tool_input: dict[str, Any]) -> PolicyAction:
        # Always-ask floor — destructive verbs whose blast radius
        # warrants a per-call confirmation even when the operator
        # has written a permissive policy.
        if tool_name in _BUILTIN_ALWAYS_ASK_TOOLS:
            # Operator can override by writing an explicit rule below
            # — we honour rules even for always-ask tools so an
            # http_request rule for `url = "https://localhost/*"` can
            # auto-allow loopback calls during testing.
            for rule in self.rules:
                if rule.matches(tool_name, tool_input):
                    return rule.action
            return "ask"

        for rule in self.rules:
            if rule.matches(tool_name, tool_input):
                return rule.action

        if tool_name in self.tool_defaults:
            return self.tool_defaults[tool_name]

        return self.fallback


def builtin_default_policy() -> Policy:
    """Sensible defaults shipped when no policy file exists.

    Read-only tools auto-allow. Everything else asks. Operators who
    want auto-approve for shell or write tools must author an
    explicit rule with the glob they're comfortable with.
    """
    rules: list[_Rule] = [
        _Rule(tool=name, action="allow")
        for name in sorted(_BUILTIN_READ_ONLY_TOOLS)
    ]
    return Policy(rules=rules, fallback="ask")


def _coerce_action(value: Any) -> PolicyAction:
    """Map raw TOML strings to a known action; unknown → ``ask``.

    The cautious default makes a misconfigured policy fail safe
    (asks instead of silently allowing).
    """
    text = str(value or "").strip().lower()
    if text in ("allow", "ask", "deny"):
        return text  # type: ignore[return-value]
    log.warning("coder.policy_unknown_action", value=value, fallback="ask")
    return "ask"


def parse_policy_text(text: str) -> Policy:
    """Parse a TOML policy string. Bad TOML → builtin defaults.

    Exposed for tests + the route layer that could accept a draft
    policy and surface a parse error to the operator.
    """
    if tomllib is None:
        log.warning("coder.policy_tomllib_missing")
        return builtin_default_policy()
    try:
        data = tomllib.loads(text)
    except Exception as exc:
        log.warning("coder.policy_parse_failed", error=str(exc)[:200])
        return builtin_default_policy()
    return _build_policy_from_dict(data)


def _build_policy_from_dict(data: dict[str, Any]) -> Policy:
    """Translate the TOML dict shape into a compiled Policy."""
    rules: list[_Rule] = []
    raw_rules = data.get("rule") or []
    if isinstance(raw_rules, dict):
        # Single inline-table rule rather than an array-of-tables.
        raw_rules = [raw_rules]
    for raw in raw_rules:
        if not isinstance(raw, dict):
            continue
        tool = str(raw.get("tool") or "").strip()
        if not tool:
            continue
        action = _coerce_action(raw.get("action"))
        arg_glob_raw = raw.get("arg_glob") or {}
        arg_glob: dict[str, str] = {}
        if isinstance(arg_glob_raw, dict):
            for k, v in arg_glob_raw.items():
                if isinstance(k, str) and isinstance(v, str):
                    arg_glob[k] = v
        rules.append(_Rule(tool=tool, action=action, arg_glob=arg_glob))

    tool_defaults: dict[str, PolicyAction] = {}
    raw_defaults = data.get("tool_defaults") or {}
    if isinstance(raw_defaults, dict):
        for k, v in raw_defaults.items():
            if isinstance(k, str):
                tool_defaults[k] = _coerce_action(v)

    fallback = _coerce_action(
        (data.get("defaults") or {}).get("fallback", "ask")
        if isinstance(data.get("defaults"), dict)
        else "ask"
    )
    return Policy(rules=rules, tool_defaults=tool_defaults, fallback=fallback)


async def load_policy(
    container_manager: Any, workspace_id: str,
) -> Policy:
    """Read the per-workspace policy file and compile it.

    Falls back to :func:`builtin_default_policy` on any error
    (missing file, bad TOML, container down). Never raises — a
    policy-load failure must not block tool dispatch.
    """
    if container_manager is None or not workspace_id:
        return builtin_default_policy()
    try:
        text = await container_manager.file_read(workspace_id, POLICY_PATH)
    except FileNotFoundError:
        return builtin_default_policy()
    except Exception as exc:  # noqa: BLE001 — degrade to safe defaults
        log.warning(
            "coder.policy_read_failed",
            workspace_id=workspace_id, error=str(exc)[:160],
        )
        return builtin_default_policy()
    if not text:
        return builtin_default_policy()
    return parse_policy_text(text)


def policy_as_dict(policy: Policy) -> dict[str, Any]:
    """Serializable representation, for /policy GET endpoints."""
    return {
        "rules": [
            {
                "tool": r.tool,
                "action": r.action,
                "arg_glob": dict(r.arg_glob),
            }
            for r in policy.rules
        ],
        "tool_defaults": dict(policy.tool_defaults),
        "fallback": policy.fallback,
    }


def policy_to_toml(policy: Policy) -> str:
    """Render a Policy back to TOML.

    Hand-rolled emitter (Python's stdlib has tomllib for reading but
    no tomli-w-equivalent for writing). The shape is small enough
    that the explicit format keeps the operator-edited file
    diff-friendly even after a round-trip.
    """
    lines: list[str] = []
    for r in policy.rules:
        lines.append("[[rule]]")
        lines.append(f'tool = {json.dumps(r.tool)}')
        if r.arg_glob:
            inline = ", ".join(
                f"{k} = {json.dumps(v)}" for k, v in r.arg_glob.items()
            )
            lines.append(f"arg_glob = {{ {inline} }}")
        lines.append(f'action = {json.dumps(r.action)}')
        lines.append("")
    if policy.tool_defaults:
        lines.append("[tool_defaults]")
        for k, v in policy.tool_defaults.items():
            lines.append(f"{k} = {json.dumps(v)}")
        lines.append("")
    lines.append("[defaults]")
    lines.append(f"fallback = {json.dumps(policy.fallback)}")
    return "\n".join(lines) + "\n"
