"""The Setting dataclass — one declaration per user-tunable knob.

Designed so the existing 4 declaration sites (``config.py`` defaults,
``config_routes.py`` validation, ``server.py`` restore map,
``settings.js`` UI sync) can ALL be derived from this single shape
during Phase 1B/1C migration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

# Trust class for automated (Becca/agent/MCP) mutation. Read at
# dispatch time by the ``setting.set`` Tool (Phase 4).
#
# - local_reversible: just do it, announce it (theme, voice swap)
# - local_significant: voice/UI confirm before applying (clear memory,
#   swap default model, delete a session)
# - external:         visible preview + explicit confirm (clones a
#   repo, downloads a file, installs something, messages someone)
# - admin_only:       admins only, regardless of caller — pure
#   permission gate, not a consent class.
TrustTier = Literal[
    "local_reversible",
    "local_significant",
    "external",
    "admin_only",
]

# Ownership / lifetime of the persisted value. ``server`` is the
# existing default (a row in the ``settings`` table). ``user`` lives
# in ``user_settings``. ``session`` is per-session ephemeral (in
# ``ui_sessions`` JSON blob or equivalent). ``workspace`` is
# per-workspace (coder ``project_checkouts``).
SettingScope = Literal["server", "user", "session", "workspace"]

# The type discriminator. Used by the registry to validate ``default``
# and by the UI to pick a renderer. ``tristate`` matches the existing
# ``_TRI_STATE_BOOL_SETTINGS`` pattern in ``config_routes.py``: persisted
# as Optional[bool], runtime resolves to bool via ``tristate_resolver``.
SettingKind = Literal["bool", "int", "float", "str", "enum", "tristate"]


def _ascii_lower_kebab_or_snake(s: str) -> bool:
    # Strict ASCII check — ``str.isalnum`` accepts unicode letters,
    # which we want to reject so settings keys can never clash with
    # JS camelCase, JSON wire format, or env-var derivations.
    return (
        all(c in "abcdefghijklmnopqrstuvwxyz0123456789_." for c in s)
        and s == s.lower()
        and bool(s)
    )


@dataclass(frozen=True)
class Setting:
    """A single declarative knob.

    Construction validates the cross-field constraints (kind consistent
    with default, range/enum/maxlen only on the relevant kind). Throws
    ``ValueError`` at module import time if a declaration is malformed
    — by design, since that's how we mechanically prevent the 184/180
    wiring split from reappearing.
    """

    # --- Identity ---
    key: str
    """Canonical snake_case name. Must be the same key used by the
    existing ``settings`` table and any pre-existing JS DEFAULTS entry
    (post-migration). Dotted sections like ``engine.kv.ttl_days`` are
    discouraged — use a section field instead and keep the key flat."""

    kind: SettingKind
    """Type discriminator. Determines which optional fields are valid."""

    default: Any
    """The default value. Must be the appropriate Python type for
    ``kind`` (bool for "bool"/"tristate", int for "int", float for
    "float", str for "str"/"enum", None allowed for "tristate")."""

    # --- Documentation (REQUIRED — audit.py enforces non-empty) ---
    label: str
    """Short human title, recommended ≤60 chars. Shown as the UI
    field label and the natural name in search results."""

    description: str
    """One-line plain-English explanation, recommended 20-200 chars.
    Shown beneath the field in the UI and indexed by settings search."""

    section: str
    """Dotted path like ``companion.voice`` or ``engine.kv``. Used as
    the primary organizing axis in the UI. Multiple settings sharing
    a section render together; sub-sections become collapsible groups.
    Audit.py enforces that this is non-empty and lower-case."""

    # --- Organization (optional) ---
    tags: tuple[str, ...] = ()
    """Cross-cutting tags for filtering and search (``advanced``,
    ``experimental``, ``deprecated``, etc.). Settings can carry
    multiple. Search supports ``@tag:advanced`` operator."""

    # --- Validation (kind-dependent; ignored otherwise) ---
    min_value: float | int | None = None
    max_value: float | int | None = None
    max_length: int | None = None
    enum_values: tuple[str, ...] | None = None
    tristate_resolver: Callable[[bool | None], bool] | None = None

    # --- Lifecycle ---
    scope: SettingScope = "server"
    restart_required: bool = False
    """If True, the UI surfaces a "restart required" badge on this
    setting. Engine-v2 settings baked into the llama-server subprocess
    command line are the canonical example."""

    deprecated: str = ""
    """If non-empty, treated as a "use X instead" replacement message
    and the setting is rendered with a deprecation warning."""

    since_version: str = ""
    """When this setting was added — used for release-notes generation
    and the UI's "what's new" filter."""

    advanced: bool = False
    """Hide from the default UI surface; only shown when the user has
    opted into power-user mode. Equivalent to a ``tags=("advanced",)``
    shortcut but persists across tag schema changes."""

    # --- Agent surface (Phase 4 consumer) ---
    trust_tier: TrustTier = "local_reversible"
    """Consent class for automated (Becca / MCP / agent) mutation.
    See the TrustTier doc above. Default is local_reversible so
    automated callers get the cheapest path for the common case."""

    voice_aliases: tuple[str, ...] = ()
    """Natural-language aliases Becca matches against. Example for
    ``tts_default_voice``: ``("eva voice", "the warm voice")``. Empty
    means voice cannot directly address this setting (it still appears
    in the registry; only the direct voice routing is suppressed)."""

    companion_surfaceable: bool = True
    """Whether Becca can contextually offer to change this. Some
    settings (e.g. ``observation_primary_user_id``) should never
    appear as a Becca-suggested action."""

    # --- Live reload ---
    on_change: Callable[[Any, Any], None] | None = field(default=None, repr=False)
    """Optional callback ``(old, new) -> None`` invoked synchronously
    when the registry's typed setter mutates the value. Phase 1A
    declares the field; Phase 1B wires the first consumer."""

    def __post_init__(self) -> None:
        self._validate_key()
        self._validate_documentation()
        self._validate_kind_and_default()
        self._validate_kind_specific_fields()

    # ------------- validators -------------

    def _validate_key(self) -> None:
        if not self.key:
            raise ValueError("Setting.key must be non-empty")
        if not _ascii_lower_kebab_or_snake(self.key):
            raise ValueError(
                f"Setting.key {self.key!r} must be ascii lower-case "
                f"snake_case (a-z, 0-9, _, .)"
            )

    def _validate_documentation(self) -> None:
        if not self.label.strip():
            raise ValueError(f"Setting {self.key!r}: label must be non-empty")
        if not self.description.strip():
            raise ValueError(
                f"Setting {self.key!r}: description must be non-empty"
            )
        if not self.section.strip():
            raise ValueError(f"Setting {self.key!r}: section must be non-empty")
        if self.section != self.section.lower():
            raise ValueError(
                f"Setting {self.key!r}: section must be lowercase"
            )

    def _validate_kind_and_default(self) -> None:
        k, d = self.kind, self.default
        if k == "bool" and not isinstance(d, bool):
            raise ValueError(
                f"Setting {self.key!r}: kind=bool requires bool default, got {type(d).__name__}"
            )
        if k == "int":
            # bool is a subclass of int in Python — exclude it explicitly.
            if isinstance(d, bool) or not isinstance(d, int):
                raise ValueError(
                    f"Setting {self.key!r}: kind=int requires int default, got {type(d).__name__}"
                )
        if k == "float":
            if isinstance(d, bool) or not isinstance(d, (int, float)):
                raise ValueError(
                    f"Setting {self.key!r}: kind=float requires numeric default, got {type(d).__name__}"
                )
        if k == "str" and not isinstance(d, str):
            raise ValueError(
                f"Setting {self.key!r}: kind=str requires str default, got {type(d).__name__}"
            )
        if k == "enum" and not isinstance(d, str):
            raise ValueError(
                f"Setting {self.key!r}: kind=enum requires str default, got {type(d).__name__}"
            )
        if k == "tristate" and d is not None and not isinstance(d, bool):
            raise ValueError(
                f"Setting {self.key!r}: kind=tristate requires bool or None default, got {type(d).__name__}"
            )

    def _validate_kind_specific_fields(self) -> None:
        k = self.kind
        if k in ("int", "float"):
            if self.min_value is not None and self.max_value is not None and self.min_value > self.max_value:
                raise ValueError(
                    f"Setting {self.key!r}: min_value {self.min_value} > max_value {self.max_value}"
                )
        else:
            if self.min_value is not None or self.max_value is not None:
                raise ValueError(
                    f"Setting {self.key!r}: min/max_value only valid for int/float kinds"
                )
        if k in ("str", "enum"):
            if self.max_length is not None and self.max_length < 1:
                raise ValueError(
                    f"Setting {self.key!r}: max_length must be >= 1"
                )
        else:
            if self.max_length is not None:
                raise ValueError(
                    f"Setting {self.key!r}: max_length only valid for str/enum kinds"
                )
        if k == "enum":
            if not self.enum_values:
                raise ValueError(
                    f"Setting {self.key!r}: kind=enum requires non-empty enum_values"
                )
            if self.default not in self.enum_values:
                raise ValueError(
                    f"Setting {self.key!r}: default {self.default!r} not in enum_values {self.enum_values!r}"
                )
        else:
            if self.enum_values is not None:
                raise ValueError(
                    f"Setting {self.key!r}: enum_values only valid for enum kind"
                )
        if k == "tristate":
            if self.tristate_resolver is None:
                raise ValueError(
                    f"Setting {self.key!r}: kind=tristate requires tristate_resolver"
                )
        else:
            if self.tristate_resolver is not None:
                raise ValueError(
                    f"Setting {self.key!r}: tristate_resolver only valid for tristate kind"
                )

    # ------------- query helpers -------------

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags

    def is_advanced(self) -> bool:
        return self.advanced or "advanced" in self.tags

    def is_deprecated(self) -> bool:
        return bool(self.deprecated)

    def matches_search(self, query: str) -> bool:
        """Case-insensitive substring match against label, description,
        key, tags, voice_aliases. Used by ``SettingsRegistry.search``."""
        q = query.lower().strip()
        if not q:
            return False
        haystacks = [
            self.key,
            self.label,
            self.description,
            self.section,
            *self.tags,
            *self.voice_aliases,
        ]
        return any(q in h.lower() for h in haystacks)
