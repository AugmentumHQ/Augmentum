"""CompanionIdentity — persona kernel + drift detection + Aletheia overlay.

Sprint α — Aletheia × Augmentum arc, Piece 2 — adds the identity API
surface that the rest of the arc consumes:

- ``read_overlay`` / ``apply_overlay`` — human-readable freeform overlay
  text (Sprint 7 reflection populates this).
- ``read_trait_deltas`` / ``nudge_trait`` — bounded trait adjustments.
  Per-call cap ±0.01, cumulative cap ±0.05 per trait.
- ``read_traits_derived`` / ``write_traits_derived`` — the current
  effective trait values (Sprint 6 PAD/drives populate this).
- ``read_relationship_state`` / ``update_relationship_state`` —
  per-user relationship dynamics (trust_level, known_rhythms,
  nicknames_earned).

All overlay state lives in the ``kernel_overlay`` TEXT column added by
migration 179 as a single JSON document. Keeps the schema compact and
the parsing logic centralized in :func:`_parse_overlay`.


The identity object loads a row from ``companion_identities`` (migration
151) and exposes the persona kernel digest — the ~400-token compressed
identity prefix threaded into every dispatch via the kernel's
``DispatchContext``. The full personality document lives on disk at
``companions/<companion>/identity/personality.md`` and is digested
into the kernel on demand via :meth:`refresh_persona_kernel`.

Namespacing: this is ``CompanionIdentity``, not ``CompanionPersona``.
``augmentum/game_agent/companion.py::CompanionPersona`` exists for a
different concept (a game-agent chat character) and must not be
conflated.

Sprint 1 scope: this module supports load, persona kernel access, naive
section-based digestion of the on-disk doc, and the drift-score read
path. The full mechanical drift detector (periodic identity rehearsal,
anchor sampling, hard refresh) lands in Sprint 4a — this module exposes
the seams (`compute_drift`, `get_anchor_embedding`) but does not yet
schedule rehearsals.

Design spec: ``docs/superpowers/specs/2026-05-14-companion-runtime-design-v2.md``
sections 4 and 10.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING

from augmentum.memory.embeddings import EmbeddingService
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.state.backends.sqlite import SQLiteBackend

log = get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────

# Cosine distance hard cap between consecutive persona kernel versions.
# Enforced by ``check_drift_ceiling``; rewrites exceeding this are
# rejected so a refactor pass cannot accidentally reshape Becca.
DRIFT_CEILING = 0.15

# Target token budget for the digested kernel. Approximate — the naive
# Sprint 1 digester counts whitespace-separated words and stops near
# ``TARGET_TOKENS * 0.8`` to leave headroom for surrounding context.
TARGET_TOKENS = 400

# Words per section captured by the naive digester. Tuned to produce
# roughly TARGET_TOKENS across the 15 canonical sections of a Becca-shaped
# personality doc.
WORDS_PER_SECTION = 22

# Matches markdown section headings of the form "## N. Title" where N is
# 1-15 (the canonical Becca personality structure).
_SECTION_HEADING_RE = re.compile(r"^##\s+(\d+)\.\s+(.+?)$", re.MULTILINE)

# ── Sprint α — Aletheia overlay caps ──────────────────────────────────
# Per-call cap on a single trait nudge. Reflective growth is incremental;
# any single proposed nudge larger than this should require a
# higher-level deliberation, not slip through the standard nudge path.
PER_CALL_TRAIT_CAP: float = 0.01

# Cumulative cap on a trait's overlay delta (positive or negative
# magnitude). Caps the total drift any single trait can absorb between
# canonical-doc edits. When the canonical doc changes, the overlay
# resets to zero — that's the only path past this cap.
CUMULATIVE_TRAIT_CAP: float = 0.05


def _parse_overlay(text: str) -> dict:
    """Parse the kernel_overlay JSON document into a normalized shape.

    The overlay schema:

        {
          "trait_deltas": {trait_name: float, ...},
          "notes_text": str
        }

    Empty / malformed / pre-overlay-schema strings parse to the default
    shape so callers can always assume both keys exist with sensible
    defaults. This is the parse half of a tight serialize/parse pair —
    never raises.
    """
    if not text:
        return {"trait_deltas": {}, "notes_text": ""}
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            return {"trait_deltas": {}, "notes_text": ""}
        return {
            "trait_deltas": dict(data.get("trait_deltas") or {}),
            "notes_text": str(data.get("notes_text") or ""),
        }
    except (json.JSONDecodeError, AttributeError, TypeError):
        return {"trait_deltas": {}, "notes_text": ""}


def _serialize_overlay(trait_deltas: dict, notes_text: str) -> str:
    """Inverse of :func:`_parse_overlay`. Stable key order so DB diffs
    are minimal when only one half changes."""
    return json.dumps({
        "trait_deltas": dict(trait_deltas),
        "notes_text": notes_text,
    }, sort_keys=True)


class DriftCeilingExceeded(Exception):
    """Raised when a proposed persona kernel version exceeds the drift cap.

    The runtime catches this in the personality-doc write path and queues
    the proposed change for explicit operator review rather than committing
    silently.
    """


# ── BLOB encode/decode ────────────────────────────────────────────────

def _encode_embedding(vec: list[float]) -> bytes:
    """Pack a float vector as a sequence of float32 bytes."""
    return struct.pack(f"{len(vec)}f", *vec)


def _decode_embedding(blob: bytes | None) -> list[float] | None:
    """Inverse of :func:`_encode_embedding`. ``None``-safe."""
    if not blob:
        return None
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Cosine distance in ``[0.0, 2.0]``. Returns 2.0 (max) on zero norm."""
    if len(a) != len(b):
        raise ValueError(f"embedding dim mismatch: {len(a)} != {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 2.0
    return 1.0 - (dot / (na * nb))


# ── Naive Sprint 1 digester ────────────────────────────────────────────

def digest_personality_doc(doc_text: str, target_tokens: int = TARGET_TOKENS) -> str:
    """Compress a personality doc to a ~target_tokens canonical digest.

    Sprint 1 implementation: parses ``## N. Section`` headings and emits
    each heading plus its first ~WORDS_PER_SECTION words. Designed for
    the 15-section Becca-shaped doc.

    Sprint 4a will replace this with LLM-driven compression that preserves
    verbal tics, opinions, and the relationship-doc slice — that work
    lives in the drift detector's identity-rehearsal pass and is out of
    scope here.

    The output is deterministic given the input. The runtime stores it in
    ``companion_identities.persona_kernel_digest`` alongside its embedding.
    """
    sections: list[str] = []
    matches = list(_SECTION_HEADING_RE.finditer(doc_text))

    if not matches:
        # No structured sections — fall back to first ~target_tokens words
        words = doc_text.split()
        return " ".join(words[: int(target_tokens * 0.8)])

    # Compute per-section text slices between consecutive heading positions
    for i, m in enumerate(matches):
        heading_num = m.group(1)
        heading_title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc_text)
        body = doc_text[start:end].strip()
        words = body.split()
        excerpt = " ".join(words[:WORDS_PER_SECTION])
        sections.append(f"§{heading_num} {heading_title}: {excerpt}")

    return "\n".join(sections)


# ── CompanionIdentity ─────────────────────────────────────────────────

class CompanionIdentity:
    """Runtime identity object for a single companion.

    Use when:
    - The runtime needs the persona kernel digest for a dispatch.
    - A consolidation pass wants to refresh the digest from the on-disk
      personality doc.
    - The drift detector (Sprint 4a) needs the anchor embedding.

    Lifecycle: instantiate, ``await load()`` once, then read-only until
    ``refresh_persona_kernel()`` rewrites.
    """

    # ── Personality doc resolution (accumulation thesis Step 2) ──
    #
    # The canonical location for a companion's personality doc is now
    # ``companions/<companion_id>/identity/personality.md`` per the
    # directory-as-form architecture. The legacy location (date-stamped
    # under docs/superpowers/specs/) stays valid as a fallback for one
    # release so existing installs don't break — but new companions
    # and future migrations land in the new layout.
    #
    # Resolution order (first existing wins):
    #   1. ``companions/<companion_id>/identity/personality.md``
    #   2. ``docs/superpowers/specs/2026-05-14-<companion_id>-personality.md``
    #   3. Falls back to (1) even if missing — write paths target the
    #      canonical location, not the legacy one.
    _REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    _COMPANIONS_DIR = _REPO_ROOT / "companions"
    _LEGACY_DOCS_DIR = _REPO_ROOT / "docs" / "superpowers" / "specs"

    def __init__(
        self,
        backend: SQLiteBackend,
        companion_id: str,
        *,
        user_id: str = "",
    ) -> None:
        self._backend = backend
        self.companion_id = companion_id
        # Piece 1 — per-user invariant: each (user_id, companion_id) pair
        # is its own row. Empty user_id is the legacy/seed sentinel
        # (the pre-pivot Becca singleton lands here after migration 179
        # backfill when owner_user_id was NULL).
        self.user_id = user_id
        self._row: dict | None = None

    @property
    def display_name(self) -> str:
        return self._row["display_name"] if self._row else self.companion_id

    @property
    def persona_kernel_digest(self) -> str:
        """Current ~400-token digest. Empty string until first refresh."""
        return self._row["persona_kernel_digest"] if self._row else ""

    @property
    def drift_score(self) -> float:
        return float(self._row["drift_score"]) if self._row else 0.0

    @property
    def owner_user_id(self) -> str:
        """The user_id this companion belongs to in single-companion phase.

        Empty string when unowned (fresh install, test fixtures, or
        future household-phase deployments where ownership is recorded
        elsewhere). Callers that need user-scoped behavior (dreams,
        memory recall, etc.) must guard on this being non-empty.
        """
        return (self._row.get("owner_user_id") or "") if self._row else ""

    @property
    def personality_doc_path(self) -> Path:
        """Canonical on-disk personality doc for this companion.

        Returns the first existing path among (in priority order):
        1. ``companions/<companion_id>/identity/personality.md``
           — the post-migration canonical location.
        2. ``docs/superpowers/specs/2026-05-14-<companion_id>-personality.md``
           — the legacy location, kept for one release for compat.

        When neither exists, returns the canonical path (1) so write
        paths target the right place — callers that need to handle a
        missing doc check existence explicitly.
        """
        canonical = (
            self._COMPANIONS_DIR / self.companion_id
            / "identity" / "personality.md"
        )
        if canonical.exists():
            return canonical
        legacy = self._LEGACY_DOCS_DIR / f"2026-05-14-{self.companion_id}-personality.md"
        if legacy.exists():
            return legacy
        # Default to canonical even when missing — write paths point here.
        return canonical

    async def load(self) -> None:
        """Load the row from ``companion_identities``. Idempotent.

        Per-user resolution (Piece 1):

        * If ``self.user_id`` is set, loads the row for
          ``(user_id, companion_id)`` exactly. Raises if missing —
          callers should ``lazy_provision`` first.
        * If ``self.user_id`` is empty (legacy/seed path), loads the
          earliest-created row for ``companion_id`` (typically the
          migration-179-backfilled singleton with user_id=''). This
          preserves the pre-pivot semantics for code that hasn't
          adopted the per-user API yet.
        """
        if self.user_id:
            cursor = await self._backend.conn.execute(
                "SELECT companion_id, display_name, persona_kernel_digest, "
                "persona_kernel_embedding, personality_doc_version, drift_score, "
                "created_at, last_kernel_refresh_at, owner_user_id, "
                "kernel_overlay, traits_derived_json, relationship_state_json "
                "FROM companion_identities "
                "WHERE user_id = ? AND companion_id = ?",
                (self.user_id, self.companion_id),
            )
        else:
            cursor = await self._backend.conn.execute(
                "SELECT companion_id, display_name, persona_kernel_digest, "
                "persona_kernel_embedding, personality_doc_version, drift_score, "
                "created_at, last_kernel_refresh_at, owner_user_id, "
                "kernel_overlay, traits_derived_json, relationship_state_json "
                "FROM companion_identities "
                "WHERE companion_id = ? "
                "ORDER BY created_at ASC LIMIT 1",
                (self.companion_id,),
            )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            raise ValueError(
                f"no companion_identities row for "
                f"({self.user_id!r}, {self.companion_id!r}) — "
                "did migration 179 backfill or lazy_provision run?",
            )
        # Normalize aiosqlite Row -> dict for stable attribute-style access.
        # New Aletheia fields (kernel_overlay, traits_derived_json,
        # relationship_state_json) included; downstream readers can
        # always assume strings (NOT NULL DEFAULT '' / '{}' from mig 179).
        self._row = {
            "companion_id": row[0],
            "display_name": row[1],
            "persona_kernel_digest": row[2] or "",
            "persona_kernel_embedding": row[3],
            "personality_doc_version": int(row[4]) if row[4] is not None else 0,
            "drift_score": float(row[5]) if row[5] is not None else 0.0,
            "created_at": row[6],
            "last_kernel_refresh_at": row[7],
            "owner_user_id": row[8] or "",
            "kernel_overlay": row[9] or "",
            "traits_derived_json": row[10] or "{}",
            "relationship_state_json": row[11] or "{}",
        }

    async def set_owner_user_id(self, user_id: str) -> None:
        """Bind this companion row to ``user_id`` and persist.

        Idempotent — overwrites any prior owner. Empty string clears.
        Per-user-pivot (mig 179): scoped by both the row's user_id AND
        companion_id so we don't accidentally rewrite OTHER users'
        rows that happen to share the same companion_id ('becca').
        """
        user_id = (user_id or "").strip()
        await self._backend.conn.execute(
            "UPDATE companion_identities SET owner_user_id = ? "
            "WHERE user_id = ? AND companion_id = ?",
            (user_id or None, self.user_id, self.companion_id),
        )
        await self._backend.conn.commit()
        if self._row is not None:
            self._row["owner_user_id"] = user_id

    # ── Aletheia overlay API (Sprint α) ──────────────────────────────

    @property
    def kernel_overlay(self) -> str:
        """Raw kernel_overlay text as stored. Empty when none.

        Callers wanting parsed access should use :meth:`read_overlay`
        for the human-readable text or :meth:`read_trait_deltas` for
        the bounded trait adjustments.
        """
        return (self._row.get("kernel_overlay") or "") if self._row else ""

    async def read_overlay(self) -> str:
        """Return the human-readable freeform overlay text.

        This is the diary-flavored note Sprint 7's reflection populates
        from nightly synthesis. Persisted as part of the kernel_overlay
        JSON document; trait deltas live in the same column but are
        accessed via :meth:`read_trait_deltas`.
        """
        return _parse_overlay(self.kernel_overlay).get("notes_text", "")

    async def apply_overlay(self, *, text: str, force: bool = False) -> bool:
        """Replace the human-readable overlay text. Returns True on write.

        Trait deltas in the same JSON document are preserved unchanged.
        ``force`` is reserved for the Sprint 7 reflection→identity loop
        which will integrate the DRIFT_CEILING check against combined
        (canonical + overlay) embeddings; until then, this method just
        persists the new text.
        """
        parsed = _parse_overlay(self.kernel_overlay)
        new_overlay = _serialize_overlay(parsed["trait_deltas"], text)
        await self._write_overlay(new_overlay)
        return True

    async def read_trait_deltas(self) -> dict[str, float]:
        """Return the current accumulated trait deltas.

        Each entry is the *cumulative* overlay delta for that trait,
        capped at ±CUMULATIVE_TRAIT_CAP. The canonical trait value lives
        elsewhere (derived from the personality doc by Sprint 6's PAD
        layer); the effective trait is canonical + delta, with the
        canonical-doc edit being the only path past the cap.
        """
        return _parse_overlay(self.kernel_overlay).get("trait_deltas", {})

    async def nudge_trait(self, *, name: str, delta: float) -> bool:
        """Adjust a single trait's overlay delta by ``delta``.

        Returns True when applied; False when blocked by either cap:

        * **Per-call cap** — ``abs(delta) > PER_CALL_TRAIT_CAP``.
          Reflective growth is incremental; any single proposed nudge
          larger than this should go through a higher-level deliberation
          (Sprint 7 reflection cross-check), not this fast path.

        * **Cumulative cap** — ``abs(new_cumulative) > CUMULATIVE_TRAIT_CAP``.
          A trait can't drift further from its canonical value than the
          cumulative cap allows. The canonical doc edit is the only
          reset.

        Idempotent: re-running with the same (name, delta) re-applies
        the same nudge each time (delta-additive, not absolute-set).
        Callers wanting to know what the cumulative value is should
        :meth:`read_trait_deltas` afterward.
        """
        if abs(delta) > PER_CALL_TRAIT_CAP:
            log.debug(
                "nudge_trait_rejected_per_call_cap",
                trait=name, delta=delta, cap=PER_CALL_TRAIT_CAP,
            )
            return False
        parsed = _parse_overlay(self.kernel_overlay)
        deltas = parsed["trait_deltas"]
        current = float(deltas.get(name, 0.0))
        new_cumulative = current + delta
        if abs(new_cumulative) > CUMULATIVE_TRAIT_CAP:
            log.debug(
                "nudge_trait_rejected_cumulative_cap",
                trait=name, current=current, delta=delta,
                proposed=new_cumulative, cap=CUMULATIVE_TRAIT_CAP,
            )
            return False
        deltas[name] = new_cumulative
        new_overlay = _serialize_overlay(deltas, parsed["notes_text"])
        await self._write_overlay(new_overlay)
        log.info(
            "nudge_trait_applied",
            companion_id=self.companion_id,
            user_id=self.user_id,
            trait=name, delta=delta, new_cumulative=new_cumulative,
        )
        return True

    async def _write_overlay(self, overlay_text: str) -> None:
        """Persist a new kernel_overlay value, scoped to (user_id, companion_id).

        Legacy seed path (user_id='') is preserved for backward compat
        with code that still uses the pre-pivot singleton API.
        """
        if self.user_id:
            await self._backend.conn.execute(
                "UPDATE companion_identities SET kernel_overlay = ? "
                "WHERE user_id = ? AND companion_id = ?",
                (overlay_text, self.user_id, self.companion_id),
            )
        else:
            await self._backend.conn.execute(
                "UPDATE companion_identities SET kernel_overlay = ? "
                "WHERE user_id = '' AND companion_id = ?",
                (overlay_text, self.companion_id),
            )
        await self._backend.conn.commit()
        if self._row is not None:
            self._row["kernel_overlay"] = overlay_text

    async def read_traits_derived(self) -> dict[str, float]:
        """Return the currently-stored derived traits dict.

        Sprint α stub — returns the stored ``traits_derived_json`` as-is.
        Sprint 6's PAD/drives layer writes this from facet activations +
        baseline. Until that lands, callers should treat this as a
        forward-compat shape, often empty.
        """
        raw = (self._row.get("traits_derived_json") or "{}") if self._row else "{}"
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, AttributeError, TypeError):
            return {}

    async def write_traits_derived(self, traits: dict[str, float]) -> bool:
        """Persist a traits_derived dict. Sprint 6's PAD/drives layer
        owns this write; surfacing it here as a clean API so the layer
        doesn't bypass CompanionIdentity to talk to the DB directly.
        """
        # Coerce values to float so consumers can rely on the JSON
        # parse shape. Non-numeric values silently drop.
        clean: dict[str, float] = {}
        for k, v in traits.items():
            try:
                clean[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
        payload = json.dumps(clean, sort_keys=True)
        if self.user_id:
            await self._backend.conn.execute(
                "UPDATE companion_identities SET traits_derived_json = ? "
                "WHERE user_id = ? AND companion_id = ?",
                (payload, self.user_id, self.companion_id),
            )
        else:
            await self._backend.conn.execute(
                "UPDATE companion_identities SET traits_derived_json = ? "
                "WHERE user_id = '' AND companion_id = ?",
                (payload, self.companion_id),
            )
        await self._backend.conn.commit()
        if self._row is not None:
            self._row["traits_derived_json"] = payload
        return True

    async def read_relationship_state(self) -> dict:
        """Return the parsed relationship_state_json dict.

        Expected shape (per Aletheia design):
            {
              "trust_level": float in [0, 1],
              "known_rhythms": list[str],
              "nicknames_earned": list[str],
              ...
            }
        Sprint α returns whatever is stored; Sprint 7's reflection
        loop will populate it from observed activity.
        """
        raw = (self._row.get("relationship_state_json") or "{}") if self._row else "{}"
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, AttributeError, TypeError):
            return {}

    async def update_relationship_state(self, key: str, value) -> bool:
        """Merge a single key into relationship_state_json atomically.

        Uses SQLite ``json_patch`` so the merge is ONE statement — no
        read-modify-write window where two concurrent updates clobber
        each other's keys (audit 2026-06-17). The patch is serialized
        first so a non-serializable value returns False cleanly instead
        of corrupting the row.

        Note (RFC 7396 merge-patch semantics): setting ``value`` to
        ``None`` DELETES the key rather than storing JSON null. That's
        the desired behavior for this state doc.
        """
        try:
            patch = json.dumps({key: value})
        except TypeError:
            log.warning(
                "update_relationship_state_value_not_serializable",
                key=key, type=type(value).__name__,
            )
            return False

        uid = self.user_id  # '' for the legacy seed row
        cur = await self._backend.conn.execute(
            "UPDATE companion_identities "
            "SET relationship_state_json = "
            "    json_patch(COALESCE(relationship_state_json, '{}'), json(?)) "
            "WHERE user_id = ? AND companion_id = ? "
            "RETURNING relationship_state_json",
            (patch, uid, self.companion_id),
        )
        merged_row = await cur.fetchone()
        await cur.close()
        await self._backend.conn.commit()
        # Keep the in-memory cache in step with the merged result.
        if self._row is not None and merged_row is not None:
            self._row["relationship_state_json"] = merged_row[0]
        return True

    # ── Drift (unchanged) ────────────────────────────────────────────

    def get_anchor_embedding(self) -> list[float] | None:
        """The frozen baseline embedding for drift comparison.

        Sprint 1: this returns the *current* embedding (no frozen anchor
        yet). Sprint 4a's drift detector will load a separate anchor row
        — see the design spec §10.
        """
        return _decode_embedding(self._row["persona_kernel_embedding"]) if self._row else None

    async def compute_drift(self, new_digest: str) -> float:
        """Cosine distance between ``new_digest`` and the current anchor.

        Returns 0.0 when there is no anchor yet (first refresh). Async
        because the embed is offloaded off the event loop (audit
        2026-06-17) — the prior sync call blocked the loop on cold model
        load even though the only caller (refresh_persona_kernel) is async.
        """
        anchor = self.get_anchor_embedding()
        if anchor is None:
            return 0.0
        new_emb = await EmbeddingService.aembed_one(new_digest)
        return _cosine_distance(anchor, new_emb)

    async def check_drift_ceiling(self, new_digest: str) -> float:
        """Compute drift and raise :class:`DriftCeilingExceeded` if over cap.

        Returns the computed distance on success.
        """
        distance = await self.compute_drift(new_digest)
        if distance > DRIFT_CEILING:
            raise DriftCeilingExceeded(
                f"persona kernel drift {distance:.4f} exceeds cap "
                f"{DRIFT_CEILING:.4f} for companion {self.companion_id!r}",
            )
        return distance

    async def refresh_persona_kernel(self, *, force: bool = False) -> str:
        """Re-digest the on-disk personality doc and persist.

        Reads the canonical personality doc, runs the naive digester,
        checks drift against the current anchor (skipped on first refresh
        or when ``force=True``), embeds the new digest, and updates
        ``companion_identities``.

        Returns the new digest. Raises :class:`DriftCeilingExceeded` if
        the proposed digest exceeds the cap and ``force`` is False.
        """
        if self._row is None:
            await self.load()
        assert self._row is not None

        doc_path = self.personality_doc_path
        if not doc_path.is_file():
            raise FileNotFoundError(
                f"personality doc not found for {self.companion_id!r}: {doc_path}",
            )
        doc_text = doc_path.read_text(encoding="utf-8")
        new_digest = digest_personality_doc(doc_text)

        # Drift check (only if we have a prior anchor)
        if self._row["persona_kernel_embedding"] is not None and not force:
            distance = await self.check_drift_ceiling(new_digest)
        else:
            distance = 0.0

        new_embedding = await EmbeddingService.aembed_one(new_digest)
        new_blob = _encode_embedding(new_embedding)
        new_version = int(self._row["personality_doc_version"]) + 1

        # Per-user scoping (mig 179): if user_id is set, update only that
        # row. Empty user_id falls back to "the row this instance loaded"
        # — preserves the pre-pivot semantics for legacy seed-only calls.
        if self.user_id:
            await self._backend.conn.execute(
                "UPDATE companion_identities SET "
                "persona_kernel_digest = ?, "
                "persona_kernel_embedding = ?, "
                "personality_doc_version = ?, "
                "drift_score = ?, "
                "last_kernel_refresh_at = datetime('now') "
                "WHERE user_id = ? AND companion_id = ?",
                (new_digest, new_blob, new_version, distance,
                 self.user_id, self.companion_id),
            )
        else:
            await self._backend.conn.execute(
                "UPDATE companion_identities SET "
                "persona_kernel_digest = ?, "
                "persona_kernel_embedding = ?, "
                "personality_doc_version = ?, "
                "drift_score = ?, "
                "last_kernel_refresh_at = datetime('now') "
                "WHERE user_id = '' AND companion_id = ?",
                (new_digest, new_blob, new_version, distance, self.companion_id),
            )
        await self._backend.conn.commit()

        # Re-load to pick up new values
        await self.load()
        log.info(
            "companion_identity_refreshed",
            companion_id=self.companion_id,
            version=new_version,
            drift=distance,
            digest_chars=len(new_digest),
        )
        return new_digest

    def snapshot(self) -> dict:
        """Read-only view for telemetry / debug surfaces. Cheap."""
        if self._row is None:
            return {"companion_id": self.companion_id, "loaded": False}
        return {
            "companion_id": self.companion_id,
            "loaded": True,
            "display_name": self._row["display_name"],
            "personality_doc_version": self._row["personality_doc_version"],
            "drift_score": self._row["drift_score"],
            "digest_chars": len(self._row["persona_kernel_digest"] or ""),
            "has_anchor": self._row["persona_kernel_embedding"] is not None,
            "last_kernel_refresh_at": self._row["last_kernel_refresh_at"],
        }

    async def set_display_name(self, new_name: str) -> str:
        """Persist a new ``display_name`` for this (user, companion) row.

        Returns the stored name (trimmed). Empty / whitespace-only input
        is rejected (``ValueError``) so the chrome can't accidentally
        clear the name to "" — a renamed companion must always have
        SOMETHING to address them as.

        This is what powers the chrome's renameability path: settings
        UI POSTs the new name → this method writes it → subsequent
        prompts pick up the new ``{{char}}`` substitution because
        ``_resolve_user_display_name``'s sibling read (the companion
        side) consults ``runtime.identity.display_name`` directly.
        """
        trimmed = (new_name or "").strip()
        if not trimmed:
            raise ValueError("display_name cannot be empty")
        # Sane cap — companion_identities.display_name is a TEXT column
        # but very long names degrade every prompt + UI surface.
        if len(trimmed) > 64:
            trimmed = trimmed[:64].rstrip()
        if self._row is None:
            await self.load()
        if self.user_id:
            await self._backend.conn.execute(
                "UPDATE companion_identities SET display_name = ? "
                "WHERE user_id = ? AND companion_id = ?",
                (trimmed, self.user_id, self.companion_id),
            )
        else:
            await self._backend.conn.execute(
                "UPDATE companion_identities SET display_name = ? "
                "WHERE user_id = '' AND companion_id = ?",
                (trimmed, self.companion_id),
            )
        await self._backend.conn.commit()
        await self.load()
        log.info(
            "companion_identity_renamed",
            companion_id=self.companion_id,
            user_id=self.user_id or "",
            new_name=trimmed,
        )
        return trimmed


__all__ = [
    "CompanionIdentity",
    "DriftCeilingExceeded",
    "DRIFT_CEILING",
    "TARGET_TOKENS",
    "digest_personality_doc",
]
