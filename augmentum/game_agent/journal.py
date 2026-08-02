"""Persistent companion journal — survives across sessions.

Every production AI-game agent in the public ecosystem uses some
form of long-running knowledge store:

* Claude Plays Pokemon: an ``update_knowledge_base`` tool with four
  fixed sections (``current_status``, ``game_progress``,
  ``current_objectives``, ``inventory``) that survives context
  summarization within a session.
* VOYAGER (NVIDIA): an embedding-indexed skill library on disk.
* Cradle (BAAI): per-game memory directories keyed by game env.

Our journal is the lightest viable version of that pattern: a JSON
file per ``(user_id, title_id)`` pair, loaded at session start and
saved after every slow-path plan turn. It carries four fixed
sections (so the agent stays coherent across edits) plus a small
free-form notes list (so it can record one-off observations without
inventing keys).

Structural choices:

* **JSON file, not SQLite.** A 2-4 KB JSON document per (user, title)
  is below SQLite's overhead floor; the FS is the right tool. Easier
  to inspect, copy, version-control as a backup.
* **Pre-defined section names.** The agent's plan can only patch
  fields in the schema. Free-form key invention historically leads
  to "the model invents 27 sections, each one used once" — see the
  Cradle skill-library postmortem.
* **Fails soft.** Disk errors, corrupted JSON, missing user_id all
  degrade to "no journal this session"; the agent runs unaltered.
* **Atomic writes.** Write to a temp sibling, then rename. Killing
  the process mid-write can't produce a half-written file.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# Section caps so a runaway model can't write a 100 KB "notes" string
# that then dominates every subsequent prompt.
_SECTION_MAX_CHARS = 2000
_NOTES_MAX_ENTRIES = 20
_NOTE_MAX_CHARS = 300

# Path-safe ids: strip everything but [A-Za-z0-9._-]. Both user_id and
# title_id are externally controlled so the sanitizer is load-bearing.
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_id(raw: str) -> str:
    """Strip unsafe characters; cap length so absurd ids can't break paths."""

    cleaned = _SAFE_ID_RE.sub("_", (raw or "").strip())
    return cleaned[:96] or "_unknown_"


def _clip_string(value: Any, limit: int = _SECTION_MAX_CHARS) -> str:
    """Coerce-to-string and clip. Tolerates ints / None / nested types."""

    if value is None:
        return ""
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


@dataclass
class JournalSections:
    """The four fixed-name sections + a small notes list.

    Use when:
    - Constructing or updating an in-memory journal view. The class
      mirrors the on-disk shape so JSON round-trips through
      ``to_dict()`` / ``from_dict()`` are lossless.

    Field semantics:
    - ``status``: where the player currently is + what just happened.
      "On Route 102, after the wild Zigzagoon battle. HP 21/24."
    - ``progress``: persistent achievements / story beats. "Got
      Treecko. Beat Brendan's rival fight. Have not entered Petalburg."
    - ``objectives``: the goal stack. Top-level entry first.
      "1) Reach Petalburg Gym to meet Norman; 2) catch a Pokémon
      with HM Cut compatibility before Rustboro."
    - ``notes``: free-form observations the agent flags as worth
      remembering. Each entry is short; the list is bounded.
    """

    status: str = ""
    progress: str = ""
    objectives: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "progress": self.progress,
            "objectives": self.objectives,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, raw: Any) -> JournalSections:
        if not isinstance(raw, dict):
            return cls()
        notes_raw = raw.get("notes") or []
        notes: list[str] = []
        if isinstance(notes_raw, list):
            for n in notes_raw[:_NOTES_MAX_ENTRIES]:
                notes.append(_clip_string(n, _NOTE_MAX_CHARS))
        return cls(
            status=_clip_string(raw.get("status", "")),
            progress=_clip_string(raw.get("progress", "")),
            objectives=_clip_string(raw.get("objectives", "")),
            notes=notes,
        )

    def merge(self, patch: dict[str, Any]) -> bool:
        """Apply a partial update from a plan's ``journal_update`` field.

        Returns True iff anything actually changed (so the orchestrator
        can skip a disk write on no-ops). Notes are appended unless the
        patch supplies a full ``notes`` list, in which case it
        REPLACES the list (after clipping). This matches how the
        agent's prompt instructs it to use the section.
        """

        if not isinstance(patch, dict):
            return False
        changed = False
        for key in ("status", "progress", "objectives"):
            if key in patch:
                new_val = _clip_string(patch[key])
                if getattr(self, key) != new_val:
                    setattr(self, key, new_val)
                    changed = True
        if "notes_append" in patch:
            raw = patch["notes_append"]
            if isinstance(raw, list):
                additions = [_clip_string(n, _NOTE_MAX_CHARS) for n in raw]
            else:
                additions = [_clip_string(raw, _NOTE_MAX_CHARS)]
            for n in additions:
                if n and n not in self.notes:
                    self.notes.append(n)
                    changed = True
            # Cap from the front so latest survives.
            if len(self.notes) > _NOTES_MAX_ENTRIES:
                self.notes = self.notes[-_NOTES_MAX_ENTRIES:]
                changed = True
        if "notes" in patch and isinstance(patch["notes"], list):
            new_notes = [
                _clip_string(n, _NOTE_MAX_CHARS)
                for n in patch["notes"][:_NOTES_MAX_ENTRIES]
            ]
            if new_notes != self.notes:
                self.notes = new_notes
                changed = True
        return changed


class CompanionJournal:
    """A loaded, mutable journal backed by a JSON file on disk.

    Use when:
    - The orchestrator wants long-running memory the slow-path agent
      can read at every turn and update by emitting
      ``journal_update`` in its plan.

    Expects:
    - A writable directory passed in ``root_dir``. The journal lives
      at ``<root_dir>/<safe(user_id)>/<safe(title_id)>.json``.
    - ``user_id`` and ``title_id`` are caller-supplied; we sanitize
      them for path safety. Empty strings fall back to ``_unknown_``.

    Returns:
    - From :meth:`load_or_create`, a journal instance with either
      the persisted state or a fresh empty one. From :meth:`save`,
      ``True`` iff the write succeeded.
    """

    def __init__(
        self,
        *,
        root_dir: Path,
        user_id: str,
        title_id: str,
        sections: JournalSections | None = None,
    ) -> None:
        self._root = Path(root_dir)
        self._user_id = _safe_id(user_id)
        self._title_id = _safe_id(title_id)
        self.sections = sections or JournalSections()

    @property
    def path(self) -> Path:
        return self._root / self._user_id / f"{self._title_id}.json"

    @classmethod
    def load_or_create(
        cls,
        *,
        root_dir: Path,
        user_id: str,
        title_id: str,
        seed: JournalSections | None = None,
    ) -> CompanionJournal:
        """Load the on-disk journal if any, else return one seeded with defaults.

        Disk errors and malformed JSON degrade to ``seed`` (if provided)
        or an empty journal — the agent runs unaltered either way.

        @param seed:
            Optional :class:`JournalSections` to use when no on-disk
            journal exists yet (first session for this user+title). The
            seed pre-loads structural knowledge (intro sequence, key
            constraints) so the agent starts informed instead of blank.
            Ignored when the file already exists on disk.
        """

        inst = cls(root_dir=root_dir, user_id=user_id, title_id=title_id)
        try:
            with open(inst.path, encoding="utf-8") as f:
                raw = json.load(f)
            inst.sections = JournalSections.from_dict(raw.get("sections", raw))
        except FileNotFoundError:
            if seed is not None:
                inst.sections = seed
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(
                "companion_journal_load_failed",
                extra={"path": str(inst.path), "error": str(exc)[:200]},
            )
            if seed is not None:
                inst.sections = seed
        return inst

    def apply_update(self, patch: dict[str, Any]) -> bool:
        """Merge a plan-emitted partial update. Returns True if changed."""

        return self.sections.merge(patch)

    def save(self) -> bool:
        """Persist atomically to disk. Returns True iff the write succeeded.

        Writes to a sibling temp file, then renames -- so a process
        crash mid-write cannot produce a half-written journal file.
        """

        body = {
            "schema": "companion_journal.v1",
            "user_id": self._user_id,
            "title_id": self._title_id,
            "sections": self.sections.to_dict(),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # NamedTemporaryFile in the same dir so the rename is atomic
            # (cross-device rename would fall back to copy).
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.path.parent),
                prefix=".journal.",
                suffix=".tmp",
                delete=False,
            ) as tf:
                json.dump(body, tf, indent=2, ensure_ascii=False)
                tmp_path = tf.name
            os.replace(tmp_path, self.path)
            return True
        except OSError as exc:
            log.warning(
                "companion_journal_save_failed",
                extra={"path": str(self.path), "error": str(exc)[:200]},
            )
            return False

    def to_prompt_dict(self) -> dict[str, Any] | None:
        """Compact dict for prompt rendering. ``None`` when fully empty.

        Returning None on an empty journal lets the prompt builder skip
        the JOURNAL block entirely on turn 0 -- the model isn't told
        about a section that doesn't yet have content.
        """

        s = self.sections
        if not (s.status or s.progress or s.objectives or s.notes):
            return None
        return s.to_dict()


__all__ = ["CompanionJournal", "JournalSections"]
