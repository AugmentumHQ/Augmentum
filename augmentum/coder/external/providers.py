"""External-agent provider descriptors — the provider-neutral source of truth
for the Coding Agents composer + history.

The composer offers *agents* (Claude Code, pi, Codex, …). Each differs in how
it's dispatched and whether/what models it can be pinned to. Rather than special-
casing one engine in the route + UI, every agent declares its capabilities here
ONCE; the composer builds its Model control from ``models``, the run route reads
``model_targetable`` to decide whether to forward a ``--model``, and the history
labels each run with the *real* model it reported.

Adding Codex is a data edit: flip ``enabled`` and fill ``models`` when the
driver lands — no route or UI change. pi is push-only (it runs on the user's
machine and self-reports), so it isn't model-targetable from here; its model is
still recorded + shown because the pushed run carries it.

Model ``value`` is what the engine's CLI wants after ``--model`` ("" = let the
engine/account pick its default — never force a choice on the user). ``label``
is what the user sees.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# "" as a model value = the engine's own default (account tier / CLI default).
# It is ALWAYS the first option so nothing is auto-picked on the user's behalf
# (never-auto-select): "Account default" is an explicit, visible choice.
_ACCOUNT_DEFAULT = {"value": "", "label": "Account default"}


@dataclass(frozen=True)
class ExternalProvider:
    """One external coding agent the composer can dispatch to."""

    id: str                    # UI agent key: "claude" | "pi" | "codex"
    label: str                 # human label shown on the badge + agent option
    dispatch: str              # "stream" (in-container SDK) | "assign" (bridge) | "push"
    model_targetable: bool     # can the user pin a model at dispatch?
    models: list[dict] = field(default_factory=list)  # [{value, label}]; [] = no picker
    enabled: bool = True       # False = shown-as-coming-soon / not offered yet


# The registry. Keyed by the UI agent key the composer uses.
_PROVIDERS: dict[str, ExternalProvider] = {
    "claude": ExternalProvider(
        id="claude",
        label="Claude Code",
        dispatch="stream",
        model_targetable=True,
        # Claude Code's `--model` accepts these aliases (plus full ids). We keep
        # the short, stable family aliases so the list survives point releases.
        models=[
            dict(_ACCOUNT_DEFAULT),
            {"value": "opus", "label": "Claude Opus"},
            {"value": "sonnet", "label": "Claude Sonnet"},
            {"value": "haiku", "label": "Claude Haiku"},
        ],
    ),
    "pi": ExternalProvider(
        id="pi",
        label="pi",
        dispatch="push",
        # pi runs on the user's own machine and pushes its runs back; the server
        # doesn't launch it, so there's no model to target from here. Its model
        # IS recorded + shown (the pushed run carries it).
        model_targetable=False,
        models=[],
    ),
    "codex": ExternalProvider(
        id="codex",
        label="Codex",
        dispatch="stream",
        model_targetable=True,
        # Filled when the Codex driver lands (registry.py::_candidates). The seam
        # is live now so the composer + route need no change then.
        models=[],
        enabled=False,
    ),
}


def get_provider(agent_id: str) -> ExternalProvider | None:
    return _PROVIDERS.get((agent_id or "").strip().lower())


def is_model_targetable(agent_id: str) -> bool:
    p = get_provider(agent_id)
    return bool(p and p.enabled and p.model_targetable)


def public_providers() -> list[dict]:
    """Serialized enabled providers for the composer (GET …/agents/providers)."""
    return [asdict(p) for p in _PROVIDERS.values() if p.enabled]
