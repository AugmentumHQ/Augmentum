"""Application builder tool — multi-pass pipeline for generating web applications."""

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.tools.application_scaffolds import (
    SCAFFOLDS,
    build_design_rules,
    build_fix_prompt,
    build_judge_prompt,
    build_plan_prompt,
)
from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from augmentum.tools.artifact_storage import ArtifactStore

log = get_logger(__name__)

_REQUEST_MODEL: ContextVar[str] = ContextVar("app_builder_request_model", default="")
_SESSION_ID: ContextVar[str] = ContextVar("app_builder_session_id", default="")
_USER_ID: ContextVar[str] = ContextVar("app_builder_user_id", default="")
_TASK_ID: ContextVar[str] = ContextVar("app_builder_task_id", default="")
_MAX_TOKENS_CTX: ContextVar[int] = ContextVar("app_builder_max_tokens", default=8192)


# --- Runtime contracts (toolkit spec §3) -----------------------------------
# Each planned file may declare a contract up front:
#   PROVIDES — global symbols it emits (``window.X``, ``ClassName``, ``fn()``)
#   DEPENDS  — symbols it consumes from other files
#   WIRES    — DOM selectors + events it attaches (e.g. ``#btn click``)
#
# Contracts are optional. When present they feed two things:
#   1. The generate prompt — the target file's contract is attached so the
#      LLM aims at specific names.
#   2. The validator — the actual code is parsed and compared against the
#      declaration; mismatches become errors in the fix loop.

# Columns on the plan line that introduce a contract field. Case-insensitive.
_CONTRACT_COLUMNS = ("PROVIDES", "DEPENDS", "DEPENDS_ON", "WIRES")
_CONTRACT_COL_ALIASES = {"DEPENDS_ON": "depends", "DEPENDS": "depends",
                         "PROVIDES": "provides", "WIRES": "wires"}


def _split_description_and_contract(raw_desc: str) -> tuple[str, dict]:
    """Separate the DESCRIPTION text from trailing contract columns.

    Plan lines carry contract fields appended with pipe separators —
    e.g. "Calculator logic | PROVIDES: window.calculate | DEPENDS: none".
    Returns ``(clean_description, {"provides": [...], "depends": [...],
    "wires": [...]})``. Missing contract fields produce empty lists.
    Values of "none" are treated as empty.
    """
    segments = [s.strip() for s in raw_desc.split("|") if s.strip()]
    if not segments:
        return raw_desc, {}
    description = segments[0]
    contract: dict[str, list[str]] = {"provides": [], "depends": [], "wires": []}
    for seg in segments[1:]:
        m = re.match(r"^(\w+)\s*:\s*(.*)$", seg)
        if not m:
            continue
        col = m.group(1).upper()
        if col not in _CONTRACT_COLUMNS:
            continue
        field_name = _CONTRACT_COL_ALIASES[col]
        value = m.group(2).strip()
        if value.lower() in ("", "none", "n/a", "-"):
            contract[field_name] = []
            continue
        contract[field_name] = [v.strip() for v in value.split(",") if v.strip()]
    return description, contract


# Regex patterns that extract the names a JS file actually exports.
# Matches ``window.X = ...``, top-level ``function X``, ``class X``,
# top-level ``const/let/var X =``. Used by analyze_file_contract.
_PROVIDES_PATTERNS = (
    re.compile(r"^\s*window\.(\w+)\s*=", re.MULTILINE),
    re.compile(r"^\s*(?:async\s+)?function\s+(\w+)\s*\(", re.MULTILINE),
    re.compile(r"^\s*class\s+(\w+)\b", re.MULTILINE),
    re.compile(r"^\s*(?:const|let|var)\s+(\w+)\s*=", re.MULTILINE),
)
# Regex that finds ``window.X`` references (i.e. cross-file dependencies).
_DEPENDS_PATTERN = re.compile(r"\bwindow\.(\w+)\b")


def _extract_provides(content: str) -> set[str]:
    names: set[str] = set()
    for pat in _PROVIDES_PATTERNS:
        for m in pat.finditer(content):
            names.add(m.group(1))
    return names


def _extract_depends(content: str) -> set[str]:
    names = {m.group(1) for m in _DEPENDS_PATTERN.finditer(content)}
    # Drop anything also declared as a provide — that's a self-reference,
    # not a cross-file dependency.
    return names - _extract_provides(content)


def _normalize_symbol(symbol: str) -> str:
    """Strip ``window.``, trailing ``()``, parenthetical free-text
    annotations, and whitespace so contract declarations compare equal
    to extracted names. LLMs love to annotate their PROVIDES entries
    ("window.Calculator (main class)"); the parenthetical is human
    prose the validator must ignore.

    Examples:
      ``window.calc()``                       → ``calc``
      ``window.PomodoroApp (init function)``  → ``PomodoroApp``
      ``Calculator (main class)``              → ``Calculator``
    """
    s = symbol.strip()
    # Drop any parenthetical annotation ("foo (description)") before
    # further processing — the parenthetical is free prose, not a type.
    s = re.sub(r"\s*\([^)]*\)\s*", "", s).strip()
    if s.startswith("window."):
        s = s[7:]
    if s.endswith("()"):
        s = s[:-2]
    return s


def _parse_wire_selectors(wires: list[str]) -> list[str]:
    """Pull CSS selectors out of contract WIRES entries. Each entry
    looks like ``#btn-go click`` or ``.item submit`` — we only need the
    selector part to check against the entry HTML."""
    selectors: list[str] = []
    for w in wires:
        token = w.strip().split()
        if token and (token[0].startswith("#") or token[0].startswith(".")):
            selectors.append(token[0])
    return selectors


def _format_contract_for_prompt(target: dict, planned: list[dict]) -> str:
    """Render the target file's contract (plus the project-wide symbol
    inventory it can depend on) as a prompt-ready block.

    Empty contracts produce an empty string so the caller doesn't pad
    the prompt with boilerplate. When the target has a contract we
    also surface what OTHER files are declared to provide — that
    guidance is what lets the generator target stable names instead
    of making up its own.
    """
    provides = target.get("provides") or []
    depends = target.get("depends") or []
    wires = target.get("wires") or []
    if not (provides or depends or wires):
        return ""

    parts = ["\n\nContract for this file:"]
    if provides:
        parts.append(f"  PROVIDES (you MUST define these): {', '.join(provides)}")
    if depends:
        parts.append(f"  DEPENDS on (available via window.X): {', '.join(depends)}")
    if wires:
        parts.append(f"  WIRES (must attach these handlers): {', '.join(wires)}")

    # Advertise project-wide PROVIDES so the target knows the allowed
    # dependency vocabulary — prevents "made up a window.X that nobody
    # defines" failures.
    other_provides: list[str] = []
    for p in planned:
        if p.get("path") == target.get("path"):
            continue
        for sym in p.get("provides") or []:
            if sym not in other_provides:
                other_provides.append(sym)
    if other_provides:
        parts.append(
            f"  Other files provide (safe to depend on): {', '.join(other_provides)}"
        )
    return "\n".join(parts)


def validate_contracts(files: list[dict], planned: list[dict], entry_html: str) -> list[str]:
    """Cross-check declared contracts against actual code / HTML.

    Returns a list of human-readable error strings suitable for the fix
    prompt. Empty list means the contracts line up. Rules:

    1. Every PROVIDES declared on a script file must appear in the code
       (LLM promised it but didn't deliver).
    2. Every DEPENDS declared on any file must be PROVIDED by some file
       (declared but nobody in the project produces it).
    3. Every actual ``window.X`` read in any script must be either
       declared in some file's PROVIDES or flagged as likely bug.
    4. Every WIRES selector must match a DOM id/class in the entry HTML
       (``#btn-x`` must appear as ``id="btn-x"`` somewhere).

    The list is deduplicated so multiple files failing the same symbol
    produce one message. Empty declarations / no entry HTML skip gracefully.
    """
    errors: list[str] = []
    planned_by_path = {p["path"]: p for p in planned}

    # Pull any inline <script> code out of the entry HTML so contracts
    # don't false-positive on globals defined there. Matches the
    # inline-style/js handling in _pass_validate for the same reason.
    inline_js_from_entry = ""
    if entry_html:
        for jm in re.finditer(
            r"<script(?![^>]*\ssrc=)[^>]*>([\s\S]*?)</script>",
            entry_html, re.IGNORECASE,
        ):
            inline_js_from_entry += "\n" + jm.group(1)

    # Union of all declared provides across the project.
    all_declared_provides: set[str] = set()
    for p in planned:
        for sym in p.get("provides") or []:
            all_declared_provides.add(_normalize_symbol(sym))

    # Union of all actual provides across script files AND any inline
    # <script> in the entry HTML.
    all_actual_provides: set[str] = set()
    for f in files:
        if f.get("role") not in ("script", "module"):
            continue
        all_actual_provides.update(_extract_provides(f.get("content", "")))
    if inline_js_from_entry:
        all_actual_provides.update(_extract_provides(inline_js_from_entry))

    # Rule 1: declared PROVIDES must show up in code.
    for f in files:
        if f.get("role") not in ("script", "module"):
            continue
        path = f["path"]
        plan = planned_by_path.get(path)
        if not plan or not plan.get("provides"):
            continue
        actual = _extract_provides(f.get("content", ""))
        for declared in plan["provides"]:
            sym = _normalize_symbol(declared)
            if sym and sym not in actual:
                errors.append(
                    f"CONTRACT: {path} declared PROVIDES '{declared}' but the "
                    f"symbol is not defined in the generated code."
                )

    # Rule 2: declared DEPENDS must be provided somewhere.
    for p in planned:
        for declared in p.get("depends") or []:
            sym = _normalize_symbol(declared)
            if sym and sym not in all_declared_provides and sym not in all_actual_provides:
                errors.append(
                    f"CONTRACT: {p['path']} declares DEPENDS '{declared}' but "
                    f"no file in the project PROVIDES it."
                )

    # Rule 3: actual window.X reads must be resolvable.
    for f in files:
        if f.get("role") not in ("script", "module"):
            continue
        actual_deps = _extract_depends(f.get("content", ""))
        for dep in actual_deps:
            if dep in all_actual_provides or dep in all_declared_provides:
                continue
            # Common benign globals — skip
            if dep in {"addEventListener", "removeEventListener", "location",
                       "document", "navigator", "history", "localStorage",
                       "sessionStorage", "fetch", "console"}:
                continue
            errors.append(
                f"CONTRACT: {f['path']} references window.{dep} but no file PROVIDES it."
            )

    # Rule 4: wires selectors must match entry HTML.
    if entry_html:
        ids_in_html = set(re.findall(r'id="([\w-]+)"', entry_html))
        classes_in_html = set(re.findall(r'class="([^"]+)"', entry_html))
        class_tokens: set[str] = set()
        for c in classes_in_html:
            class_tokens.update(c.split())
        for p in planned:
            for sel in _parse_wire_selectors(p.get("wires") or []):
                if sel.startswith("#") and sel[1:] not in ids_in_html:
                    errors.append(
                        f"CONTRACT: {p['path']} WIRES '{sel}' but the entry "
                        f"HTML has no element with id='{sel[1:]}'."
                    )
                elif sel.startswith(".") and sel[1:] not in class_tokens:
                    errors.append(
                        f"CONTRACT: {p['path']} WIRES '{sel}' but the entry "
                        f"HTML has no element with class '{sel[1:]}'."
                    )

    # Deduplicate while preserving order
    seen: set[str] = set()
    return [e for e in errors if not (e in seen or seen.add(e))]


# --- Self-doubt comment stripping (polish pass) ---------------------------
# LLMs often leak internal monologue into generated code comments:
# "Actually, App.js handles increment logic explicitly to be cleaner",
# "Let's assume this function reads current state...". These drop
# perceived code quality without adding information. The polish pass
# deletes individual comment lines that open with these markers.
# Conservative list — anything that could plausibly be a real comment
# stays. Observed patterns during the live pomodoro build drive this.
_SELF_DOUBT_OPENERS = (
    "actually, ", "actually ",
    "let's assume", "let's just", "let's say",
    "for simplicity", "for now,", "for now ",
    "however, the prompt", "however, we", "however, i",
    "i'll just", "we'll just", "i'm going to", "i will ",
    "but here we", "but for this", "but in this specific",
    "note: in this specific", "note: for now",
    "should be fine", "this is fine",
)


def _is_selfdoubt_comment_body(body: str) -> bool:
    """True if a // comment body is LLM internal-monologue rather than
    useful documentation. Matches case-insensitively against a curated
    opener list — other comment shapes pass through untouched."""
    lower = body.lower().strip()
    if not lower:
        return False
    return any(lower.startswith(opener) for opener in _SELF_DOUBT_OPENERS)


def strip_selfdoubt_comments(content: str) -> tuple[str, int]:
    """Remove LLM internal-monologue comment lines from ``content``.

    Returns ``(cleaned, count)``. ``count`` is the number of comment
    lines that were stripped. Only whole-line ``//`` comments are
    considered — trailing comments on code lines are preserved so we
    don't mangle ``foo(); // real note``.
    """
    lines = content.split("\n")
    out: list[str] = []
    removed = 0
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("//"):
            comment_body = stripped[2:]
            if _is_selfdoubt_comment_body(comment_body):
                removed += 1
                continue
        out.append(line)
    result = "\n".join(out)
    # Collapse runs of 3+ blank lines that the stripping may have
    # opened up — keep paragraphs readable.
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result, removed


# --- Stub markers (used by validate's structural checks) ------------------
# Substrings that identify placeholder / not-yet-implemented script bodies.
# Detected in validate now (previously in improve); any file whose content
# contains one of these is flagged so the fix loop fills it in.
_IMPROVE_STUB_MARKERS = (
    "todo", "implement", "placeholder", "your code here", "add code",
)


# --- Batch generation (toolkit spec §25) ----------------------------------
# Max number of planned files for which ``_pass_generate`` will try a
# single-call batched generation before falling through to sequential.
# Five covers micro + small scaffolds (entry + styles + app.js +
# optional state/config). Beyond that the prompt gets long and the
# LLM's adherence to one-file-per-fence drops sharply.
_BATCH_FILE_LIMIT = 5


# --- Improve-pass score gate ------------------------------------------------
# Judge score threshold below which the improve pass requests another
# iteration; ship with a user-visible quality warning once both attempts
# have run. Picked at 7.5 as the midpoint between "good enough" (≥8 in
# practice) and "clearly broken" (≤6) — tune via telemetry once we have
# score distributions across real builds.
_IMPROVE_GATE_THRESHOLD = 7.5
# Max improve iterations including the retry triggered by the gate. The
# outer loop's max_improve setting must allow at least this many.
_IMPROVE_GATE_MAX_RETRIES = 2
# Sentinel value returned by _parse_score when it can't find a SCORE:
# line in the judge response. Treated as "unknown" by the gate rather
# than acted on as a real low score.
_PARSE_SCORE_DEFAULT = 5.0


# Passes that ENHANCE an already-generated app rather than produce it.
# An exception inside one of these must never sink the whole build: the
# files already exist after generate, so a crash here (e.g. the polish
# pass's CSS regex choking on a malformed declaration) degrades to a
# user-visible quality warning and the pipeline proceeds to deliver, so
# the user still gets a usable, downloadable artifact instead of a
# dead-end error card. plan/generate/deliver stay fatal — without them
# there is genuinely no app to ship.
_NON_FATAL_PASSES = frozenset({"validate", "improve", "polish", "verify"})


# Library card thumbnails for app artifacts. Captured at build time and
# served by the preview-image route; the library renders <img> instead
# of a hover-driven live iframe so the grid doesn't spawn N sandboxed
# browsers as the user scans.
_PREVIEW_VIEWPORT_W = 800
_PREVIEW_VIEWPORT_H = 600


def _preview_image_path(file_path) -> Path:
    """Sibling PNG path for an artifact's zip on disk."""
    from pathlib import Path as _Path
    p = _Path(file_path)
    return p.with_name(p.stem + "-preview.png")


def _infer_file_role(path: str) -> str:
    """Infer an app-bundle file's role from its extension. Module-level so the
    ``_assemble`` backfill can call it even through the ``self=None`` shim
    (``assemble_application_html``)."""
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
    if ext in ("html", "htm"):
        return "entry"
    if ext == "css":
        return "style"
    if ext in ("js", "ts"):
        return "script"
    if ext == "json":
        return "data"
    if ext == "md":
        return "readme"
    return "script"


def assemble_application_html(files: list) -> str:
    """Module-level reusable wrapper around ``ApplicationBuilderTool._assemble``.

    The library backfill route doesn't have a live builder instance,
    so we expose the assembly logic through a thin shim. Re-uses the
    instance method to keep one source of truth for HTML assembly.
    """
    if not files:
        return ""
    return ApplicationBuilderTool._assemble(None, files)


async def capture_app_preview_screenshot(
    store: ArtifactStore,
    artifact_id: str,
    assembled_html: str,
    *,
    user_id: str,
) -> bool:
    """Capture a PNG thumbnail of an assembled app and persist it.

    Writes the PNG next to the artifact zip and flips
    ``metadata.preview_image`` to true so the library knows to render
    the static image instead of falling back to a placeholder.
    Returns True on success, False otherwise — caller never re-raises.
    """
    from augmentum.tools.application_cdp import capture_html_screenshot
    info = await store.get(artifact_id, user_id=user_id)
    if not info:
        return False
    file_path = store.get_file_path(info.get("path", ""))
    if not file_path:
        return False
    png = await capture_html_screenshot(
        assembled_html,
        viewport_w=_PREVIEW_VIEWPORT_W,
        viewport_h=_PREVIEW_VIEWPORT_H,
    )
    if not png:
        return False
    preview_path = _preview_image_path(file_path)
    try:
        preview_path.write_bytes(png)
    except OSError as exc:
        log.warning("artifact_preview_screenshot.write_failed",
                    artifact_id=artifact_id, error=str(exc))
        return False
    # Merge into metadata so the library card knows to render <img>.
    try:
        existing_meta = json.loads(info.get("metadata") or "{}") if isinstance(info.get("metadata"), str) else (info.get("metadata") or {})
    except (json.JSONDecodeError, TypeError):
        existing_meta = {}
    existing_meta["preview_image"] = True
    await store._db.execute(
        "UPDATE artifacts SET metadata = ? WHERE id = ? AND user_id = ?",
        [json.dumps(existing_meta), artifact_id, user_id],
    )
    await store._db.commit()
    return True


@dataclass
class PassResult:
    done: bool
    output: str = ""
    detail: str = ""
    error: str = ""
    recoverable: bool = True
    files_produced: list = field(default_factory=list)


@dataclass
class PipelineContext:
    description: str
    scaffold_id: str = "static"
    files: list = field(default_factory=list)
    planned_files: list = field(default_factory=list)
    generated_files: dict = field(default_factory=dict)
    is_iteration: bool = False
    project_name: str = ""
    score: float = 0.0
    preview_html: str = ""
    artifact_id: str = ""
    iterations: dict = field(default_factory=dict)
    # Max attempts per pass, mirrored from the pipeline loop so emit()
    # can surface "(attempt N/M)" in the build monitor. Without this the
    # user sees an iteration number with no sense of remaining budget.
    pass_budgets: dict = field(default_factory=dict)
    current_pass: str = ""
    # Most recent pass that finished cleanly. Captured at done=True so
    # the failure card can render "Failed at <current> · completed
    # <last>" instead of dropping the user into a passname-only tombstone.
    last_completed_pass: str = ""
    # Full traceback of the most recent pipeline exception, set by the
    # execute() except-block. Shown only when the user expands the
    # error card so the default UI stays compact.
    error_detail: str = ""
    errors: list = field(default_factory=list)
    quality_status: str = "clean"
    # User-facing quality flags surfaced in the final ToolResult.warnings.
    # Populated by passes that complete "successfully but degraded" — e.g.
    # the judge returned a score below the quality gate and we shipped
    # after exhausting improve attempts.
    quality_warnings: list = field(default_factory=list)
    blocking_errors: list = field(default_factory=list)
    working_doc: str = ""  # Manus-style persistent context across generation steps
    _validate_attempts: list = field(default_factory=list)  # Error preservation (Manus pattern)
    _verify_attempts: list = field(default_factory=list)
    _improve_attempts: list = field(default_factory=list)  # Same pattern, improve/judge loop.
    # Raw judge response from the most recent improve pass. The next
    # improve iteration parses IMPROVEMENTS: bullets out of this to
    # feed a targeted fix — without this the second attempt would just
    # ask the LLM the same thing and get the same output.
    _last_judge_response: str = ""
    _total_tokens: int = 0
    _total_llm_calls: int = 0
    # Model tier ("small" | "medium" | "large" | "frontier"), cached once
    # at execute() so plan/generate/fix can consistently size references,
    # cap toolkit sections, and adapt the system prompt per spec §5.
    model_tier: str = "medium"

    def flag_quality_issue(
        self,
        message: str,
        *,
        status: str = "needs_review",
        errors: list | None = None,
    ) -> None:
        """Record a user-visible degraded-success signal once."""
        if status and status != "clean":
            self.quality_status = status
        if message and message not in self.quality_warnings:
            self.quality_warnings.append(message)
        if errors:
            for error in errors:
                if error not in self.blocking_errors:
                    self.blocking_errors.append(error)

    def to_dict(self) -> dict:
        """Serialize for streaming metadata. Excludes previewHtml (large)
        — the frontend assembles it from files instead."""
        return {
            "name": self.project_name,
            "scaffold": self.scaffold_id,
            "files": self.files,
            "score": self.score,
            "artifactId": self.artifact_id,
            "status": "complete",
            "qualityStatus": self.quality_status,
            "quality_status": self.quality_status,
            "warnings": list(self.quality_warnings),
            "blockingErrors": list(self.blocking_errors),
            "blocking_errors": list(self.blocking_errors),
        }


# --- Project naming (deterministic, no LLM) --------------------------------
# The old extractor took the first 4 non-stopword tokens, which produced
# names like "Simple Calculator That Supports" for the prompt "make me a
# simple calculator that supports basic math". Instead, search for an
# anchor noun (the thing being built) and walk backwards to collect
# modifiers, stopping at connectors and verbs. Falls back to a filtered
# keyword grab when no anchor is present.

_NAME_ANCHOR_NOUNS = frozenset({
    "app", "application", "tool", "game", "dashboard", "tracker",
    "calculator", "generator", "editor", "player", "viewer", "manager",
    "checker", "visualizer", "timer", "clock", "planner", "organizer",
    "todo", "notes", "list", "board", "chart", "simulator", "browser",
    "explorer", "picker", "scheduler", "converter", "analyzer",
    "notepad", "journal", "gallery", "kanban", "pomodoro", "reader",
    "recorder", "finder", "search", "feed", "wiki", "page", "site",
    "form", "quiz", "survey", "poll", "map", "menu",
    "stopwatch", "chat", "spreadsheet",
})

_NAME_SKIP_VERBS = frozenset({
    "make", "build", "create", "generate", "want", "need", "like",
    "please", "could", "can", "would", "should", "let", "lets",
    "give", "show", "design", "draw", "write",
})

_NAME_SKIP_FILLER = frozenset({
    "a", "an", "the", "me", "i", "you", "some", "this", "that",
    "my", "for", "with", "of", "to", "in", "on", "by", "is", "be",
    "it", "us", "as",
})

_NAME_STOP_CONNECTORS = frozenset({
    "that", "which", "with", "to", "for", "and", "or", "but",
    "so", "while", "when", "where", "if", "from", "using", "via",
})

# Common acronyms — applied case-insensitively so "api explorer" renders
# as "API Explorer", not "Api Explorer". Add sparingly.
_NAME_ACRONYMS = {
    "api", "ai", "url", "css", "html", "js", "ui", "ux", "json",
    "yaml", "xml", "pdf", "sql", "csv", "id", "qr", "rss", "io",
    "os", "tv", "dvd", "cpu", "gpu", "ram", "hd", "ssd", "tcp",
    "udp", "http", "https",
}


def _name_titlecase(word: str) -> str:
    if not word:
        return ""
    low = word.lower()
    if low in _NAME_ACRONYMS:
        return low.upper()
    return word[:1].upper() + low[1:]


def derive_project_name(description: str) -> str:
    """Title a web-app build request without calling an LLM.

    Two-strategy ladder:
      1. Locate an anchor noun (``app``, ``tracker``, ``editor``…) and
         walk backwards collecting up to three modifier tokens, stopping
         at connectors (``that``, ``with``) or verbs (``make``, ``build``).
         "build me a simple todo app" → "Simple Todo App".
      2. If no anchor, fall back to the original "keep alphabetic
         non-stopword tokens" grab with a broader stopword set.
         "draw a sine wave on a canvas" → "Sine Wave Canvas".

    Caps at five tokens. Preserves common acronyms. Returns ``"Web App"``
    when every token is filler. Used by the App Builder runtime and by
    the persistence layer (``modes/passthrough/handler.py``) so the in-
    memory monitor and the database row agree.
    """
    if not description:
        return "Web App"

    raw_tokens = re.findall(r"[A-Za-z]+", description)
    if not raw_tokens:
        return "Web App"
    lower = [t.lower() for t in raw_tokens]

    # Strategy 1: anchor noun. Prefer the LAST anchor — phrasings like
    # "build a calorie tracker app" should yield "Calorie Tracker App",
    # not just "Calorie Tracker".
    anchor_idx = -1
    for idx, tok in enumerate(lower):
        if tok in _NAME_ANCHOR_NOUNS:
            anchor_idx = idx
    if anchor_idx >= 0:
        mods: list[str] = []
        j = anchor_idx - 1
        while j >= 0 and len(mods) < 3:
            w = lower[j]
            if w in _NAME_STOP_CONNECTORS or w in _NAME_SKIP_VERBS:
                break
            if w not in _NAME_SKIP_FILLER:
                mods.insert(0, raw_tokens[j])
            j -= 1
        name_tokens = mods + [raw_tokens[anchor_idx]]
        # If we ended up with just the bare anchor (anchor was first
        # meaningful token), sweep AHEAD for trailing context: "chat ui",
        # "chat interface", "page editor". Stops at the same connectors.
        if len(name_tokens) == 1:
            k = anchor_idx + 1
            while k < len(lower) and len(name_tokens) < 3:
                w = lower[k]
                if w in _NAME_STOP_CONNECTORS or w in _NAME_SKIP_VERBS:
                    break
                if w not in _NAME_SKIP_FILLER:
                    name_tokens.append(raw_tokens[k])
                k += 1
        joined = " ".join(_name_titlecase(w) for w in name_tokens[:5])
        return joined or "Web App"

    # Strategy 2: filtered keywords.
    skip = _NAME_SKIP_VERBS | _NAME_SKIP_FILLER | _NAME_STOP_CONNECTORS
    kept = [t for t in raw_tokens if t.lower() not in skip]
    if kept:
        return " ".join(_name_titlecase(w) for w in kept[:4])
    return "Web App"


class ApplicationBuilderTool(Tool):
    """Build complete web applications from a description via multi-pass pipeline."""

    def __init__(self, artifact_store: ArtifactStore, call_llm_fn: Callable, settings_fn: Callable | None = None, app_state: Any = None) -> None:
        self._store = artifact_store
        self._call_llm = call_llm_fn
        self._get_settings = settings_fn
        # When present, execute() routes to the coder-workspace builder
        # (run_build) instead of the legacy in-process pipeline. Set at
        # construction in server.py where app.state is available.
        self._app_state = app_state

    @property
    def _request_model(self) -> str:
        return _REQUEST_MODEL.get()

    @_request_model.setter
    def _request_model(self, value: str) -> None:
        _REQUEST_MODEL.set(value or "")

    @property
    def _session_id(self) -> str:
        return _SESSION_ID.get()

    @_session_id.setter
    def _session_id(self, value: str) -> None:
        _SESSION_ID.set(value or "")

    @property
    def _user_id(self) -> str:
        return _USER_ID.get()

    @_user_id.setter
    def _user_id(self, value: str) -> None:
        _USER_ID.set(value or "")

    @property
    def _task_id(self) -> str:
        return _TASK_ID.get()

    @_task_id.setter
    def _task_id(self, value: str) -> None:
        _TASK_ID.set(value or "")

    @property
    def _max_tokens(self) -> int:
        return _MAX_TOKENS_CTX.get()

    @_max_tokens.setter
    def _max_tokens(self, value: int) -> None:
        _MAX_TOKENS_CTX.set(int(value or 8192))

    @property
    def name(self) -> str:
        return "build_application"

    @property
    def description(self) -> str:
        return (
            "MULTI-FILE web application builder. DECISION RULE, checked before "
            "anything else: if the user asked for a single file ('one HTML file', "
            "'a single file', 'just the code') or the output naturally fits in ONE "
            "self-contained file, do NOT call this tool — write the complete code "
            "directly in your reply as a fenced code block, no matter how complex "
            "or polished the request is. A single-file request is a hard override; "
            "complexity is not a reason to escalate past it. Call this ONLY for "
            "projects that genuinely need several files working together "
            "(multi-screen dashboards, apps with separate HTML/CSS/JS modules) "
            "— OR whenever the user explicitly asks for the app builder / a "
            "multi-file app: their explicit request outranks the single-file "
            "default, even for small projects. "
            "It kicks off a multi-minute build pipeline behind a user confirmation "
            "and delivers a live preview + downloadable zip."
        )

    @property
    def model_hint(self) -> str:
        return (
            "Use ONLY for a genuine MULTI-FILE web app — a project that needs "
            "separate HTML + CSS + JS files (or several files) working together. "
            "If the answer is ONE self-contained file — a single HTML page, one "
            "script, one component, a snippet — or the user asked for 'a file', "
            "'a single file', or 'just the code', do NOT call this tool: write the "
            "file directly in your reply inside a fenced code block. This tool "
            "kicks off a multi-minute build pipeline and asks the user to confirm "
            "first, so it is the wrong choice for quick single-file output."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.ARTIFACT

    @property
    def produces(self) -> list[str]:
        return ["artifact_url"]

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "no_model_resolved": "No model is available. Make sure a model is selected in the chat before building.",
            "CancelledError": "The build was cancelled.",
            "timeout": "The build took too long. Try a simpler application description.",
        }

    @property
    def timeout(self) -> float:
        return 1080.0  # 18 minutes — large multi-file projects with retries

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def auto_invoke_when_enabled(self) -> bool:
        # The user turning on the builder IS the "please build" signal —
        # no need to parrot "build me a ..." in the prompt.
        return True

    @property
    def long_running(self) -> bool:
        # Runs in the background, pushes progress through ACTIVE_BUILDS
        # so the client-side monitor can render passes + files.
        return True

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What the application should do. Be specific about features, layout, and behavior.",
                },
                "scaffold": {
                    "type": "string",
                    "enum": list(SCAFFOLDS.keys()),
                    "default": "static",
                    "description": "Project template: static (general), dashboard (charts), game (canvas), form (interactive tool).",
                },
            },
            "required": ["description"],
        }

    async def execute(self, description: str, scaffold: str = "static", **kwargs: Any) -> ToolResult:
        existing_project = kwargs.get("existing_project")
        resume_from = kwargs.get("resume_from")  # Partial progress from a failed build
        progress_cb = kwargs.get("_progress_callback")
        build_id = kwargs.get("_build_id", "")
        request_context = kwargs.get("_request_context")
        self._request_model = kwargs.get("_request_model", "") or getattr(request_context, "model", "")
        self._session_id = kwargs.get("_session_id", "")
        self._user_id = Tool.extract_user_id(kwargs)
        self._task_id = kwargs.get("_task_id") or kwargs.get("task_id") or ""

        # Route to the coder-workspace builder (run_build) — the real
        # build-test-fix loop, same engine as the Library "Build an app"
        # button. execute() is the SINGLE chokepoint every caller hits
        # (agentic chain, /iterate, canvas, direct-invoke), so routing here
        # is what finally gives build mode a real workspace + browser
        # verification instead of the in-process quickjs pipeline. Falls back
        # to the legacy pipeline below only when the coder stack isn't wired
        # (headless / test environments).
        coder_result = await self._run_via_coder(description, build_id, progress_cb)
        if coder_result is not None:
            return coder_result

        from augmentum.tools.application_scaffolds import detect_model_tier

        ctx = PipelineContext(
            description=description,
            scaffold_id=scaffold if scaffold in SCAFFOLDS else "static",
            files=existing_project["files"] if existing_project else [],
            is_iteration=existing_project is not None and resume_from is None,
            model_tier=detect_model_tier(self._request_model),
        )

        # Resume from partial progress: pre-populate generated_files so the
        # generate pass skips already-completed files and picks up where it left off.
        if resume_from and resume_from.get("files"):
            ctx.files = resume_from["files"]
            ctx.generated_files = {f["path"]: f["content"] for f in resume_from["files"] if f.get("content")}
            if resume_from.get("planned_files"):
                ctx.planned_files = resume_from["planned_files"]
            log.info("app_builder.resuming", completed_files=list(ctx.generated_files.keys()),
                     total_planned=len(ctx.planned_files))

        # Extract project name from description
        ctx.project_name = self._extract_name(description)

        # Pre-warm: ensure model is loaded and hot before starting the pipeline
        try:
            await self._call_llm(
                [{"role": "user", "content": "ready"}],
                max_tokens=1,
                model=self._request_model,
            )
        except Exception as exc:
            # Best-effort prewarm — pipeline will retry on failure.
            log.debug("artifact_app_prewarm_failed", error=str(exc))

        try:
            # Select pipeline version from settings
            s = self._get_settings() if self._get_settings else {}
            use_v2 = s.get("app_builder_pipeline_v2", False)
            if use_v2:
                await self._run_pipeline_v2(ctx, progress_cb)
            else:
                await self._run_pipeline(ctx, progress_cb)
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            log.warning("app_builder.pipeline_failed", error=str(exc), traceback=tb)
            ctx.error_detail = tb
            return ToolResult(
                success=False,
                error=f"Build failed: {exc}",
                metadata={"project": {
                    "name": ctx.project_name, "status": "error",
                    "files": ctx.files, "error": str(exc),
                    "buildId": build_id,
                    # Resume context: enough state to pick up where we left off
                    "resumable": len(ctx.generated_files) > 0,
                    "planned_files": ctx.planned_files,
                    "completed_files": list(ctx.generated_files.keys()),
                    "current_pass": ctx.current_pass,
                    "failed_pass": ctx.current_pass,
                    "last_completed_pass": ctx.last_completed_pass,
                    "error_detail": tb,
                    "qualityStatus": ctx.quality_status,
                    "quality_status": ctx.quality_status,
                    "warnings": list(ctx.quality_warnings),
                    "blockingErrors": list(ctx.blocking_errors),
                }},
            )

        from augmentum.tools.base import format_output_with_warnings

        download_url = f"/api/artifacts/{ctx.artifact_id}/download"
        # File list for headless clients that want to inspect the project
        # without unzipping. Path + size only — full bytes go via download.
        file_index = [
            {"path": f.get("path", ""), "size": len(f.get("content", "") or "")}
            for f in ctx.files
            if f.get("path")
        ]
        base_output = (
            f"Application built: {ctx.project_name}\n"
            f"Files: {len(ctx.files)} ({', '.join(f['path'] for f in ctx.files[:5] if f.get('path'))}"
            f"{'…' if len(ctx.files) > 5 else ''})\n"
            f"Download (zip): {download_url}\n"
            f"Artifact ID: {ctx.artifact_id}"
        )
        project_meta = ctx.to_dict()
        if build_id:
            project_meta["buildId"] = build_id

        return ToolResult(
            success=True,
            output=format_output_with_warnings(base_output, ctx.quality_warnings),
            warnings=list(ctx.quality_warnings),
            # Explicit top-level fields so headless OpenAI/Ollama clients can
            # extract artifact info without parsing the human-facing output
            # text. The nested "project" key is kept for the bundled UI.
            metadata={
                "artifact_id": ctx.artifact_id,
                "download_url": download_url,
                "file_index": file_index,
                "project": project_meta,
            },
        )

    async def _run_via_coder(
        self, description: str, build_id: str, progress_cb: Callable | None,
    ) -> ToolResult | None:
        """Dispatch the build to the coder-workspace builder (run_build).

        Returns a ToolResult on success/failure, or None to fall through to the
        legacy in-process pipeline when the coder stack isn't available (so
        headless/test environments still work). Reuses the caller's ``build_id``
        when one was supplied (the agentic chain / iterate routes pre-create the
        build_run row) so we don't double-create or orphan rows.
        """
        app_state = self._app_state
        if app_state is None:
            return None
        cm = getattr(app_state, "container_manager", None)
        artifact_store = getattr(app_state, "artifact_store", None)
        build_store = getattr(app_state, "build_run_store", None)
        pr = getattr(app_state, "provider_registry", None)
        power_registry = getattr(app_state, "power_registry", None)
        if not (cm and artifact_store and build_store and pr):
            log.info("app_builder.coder_unavailable_fallback")
            return None

        try:
            backend, clean_model = await pr.resolve_backend_with_fabric(
                self._request_model, user_id=self._user_id,
            )
        except Exception as exc:  # noqa: BLE001 — fall back to legacy on resolve failure
            log.warning("app_builder.coder_backend_resolve_failed", error=str(exc))
            return None
        if backend is None:
            return None
        resolved_model = clean_model or self._request_model

        import asyncio
        import uuid

        from augmentum.builds.facade import run_build
        from augmentum.builds.runtime import _utc_now_iso
        from augmentum.modes.passthrough.handler import ACTIVE_BUILDS
        from augmentum.proxy.build_routes import _make_build_event_sink

        name = self._extract_name(description)
        created_here = not build_id
        if not build_id:
            build_id = f"build_{uuid.uuid4().hex[:16]}"

        build_state: dict = {
            "id": build_id, "kind": "application", "user_id": self._user_id,
            "session_id": self._session_id, "task_id": self._task_id,
            "started_at_iso": _utc_now_iso(), "model": resolved_model,
            "name": name, "status": "running", "passes": [], "error": None,
            "project": None, "workspace_id": "", "_checkpoint": {},
            "_change_event": asyncio.Event(),
        }
        ACTIVE_BUILDS[build_id] = build_state
        sink = _make_build_event_sink(build_state)

        # Mount the live card in the tab that started the build: the caller's
        # progress bridge re-emits this project_progress to the frontend
        # (chat/index.js → handleBuildStarted). The build_id it carries matches
        # ACTIVE_BUILDS, so /api/builds/{id}/stream renders it live.
        if progress_cb:
            try:
                await progress_cb({"project_progress": {
                    "build_id": build_id, "name": name,
                    "status": "running", "pass": "build",
                }})
            except Exception:  # noqa: BLE001 — a card-mount hiccup must not sink the build
                log.debug("app_builder.card_mount_emit_failed", exc_info=True)

        log.info("app_builder.coder_dispatch", build_id=build_id,
                 model=resolved_model, reuse_row=not created_here)
        result = await run_build(
            objective=description, user_id=self._user_id, backend=backend,
            model=resolved_model, container_manager=cm,
            artifact_store=artifact_store, build_run_store=build_store,
            power_registry=power_registry, session_id=self._session_id,
            build_id=build_id, event_sink=sink, create_row=created_here,
        )
        result = result if isinstance(result, dict) else {}

        artifact_id = result.get("artifact_id", "")
        status = (result.get("status", "") or "").lower()
        ws_id = result.get("workspace_id", "")
        rname = result.get("name") or name
        ok = status in ("completed", "paused")
        project = {
            "name": rname, "artifactId": artifact_id, "artifact_id": artifact_id,
            "workspaceId": ws_id, "workspace_id": ws_id,
            "status": status or "error",
            "qualityStatus": result.get("qualityStatus", "clean"),
            "quality_status": result.get("qualityStatus", "clean"),
            "behaviors": result.get("behaviors") or [],
        }
        if artifact_id:
            output = (
                f"Application built: {rname}\n"
                f"Download (zip): /api/artifacts/{artifact_id}/download\n"
                f"Artifact ID: {artifact_id}\n"
                f"Workspace: {ws_id or '(none)'}"
            )
        else:
            output = (
                f"Build stopped ({status or 'error'}): "
                f"{result.get('error') or 'no artifact was produced'}."
            )
        return ToolResult(
            success=ok,
            output=output,
            metadata={
                "artifact_id": artifact_id,
                "project": project,
                "build_started": {"build_id": build_id, "name": rname},
                "buildId": build_id,
                "workspace_id": ws_id,
            },
        )

    async def _run_pipeline(self, ctx: PipelineContext, progress_cb: Callable | None) -> None:
        # Read configurable budgets from settings
        s = self._get_settings() if self._get_settings else {}
        max_fix = s.get("app_builder_max_fix_iterations", 4)
        max_improve = s.get("app_builder_max_improve_iterations", 2)
        enable_improve = s.get("app_builder_improve_pass", True)
        self._max_tokens = s.get("app_builder_max_tokens", 8192)

        # If resuming from a failed build, skip the plan pass — we already have the plan
        is_resume = len(ctx.generated_files) > 0 and len(ctx.planned_files) > 0
        if is_resume:
            # Rebuild working document from existing files for resume context
            from augmentum.tools.application_references import select_references
            file_checklist = "\n".join(
                f"- [{'x' if f['path'] in ctx.generated_files else ' '}] {f['path']} ({f.get('role', 'script')})"
                for f in ctx.planned_files
            )
            design_rules = build_design_rules(ctx.description, ctx.scaffold_id)
            reference_section = select_references(ctx.description, ctx.scaffold_id)
            ctx.working_doc = (
                f"# Project: {ctx.project_name}\n\n"
                f"## Goal\n{ctx.description}\n\n"
                f"## Files to Generate (RESUMING)\n{file_checklist}\n\n"
                f"{reference_section}\n"
                f"{design_rules}\n\n"
            )
            ctx.working_doc = self._update_project_map(ctx)
            log.info("app_builder.resume_skipping_plan",
                     completed=len(ctx.generated_files), remaining=len(ctx.planned_files) - len(ctx.generated_files))

        passes = [
            ("plan", 2),  # 2 attempts — retry if first plan format unrecognized
            ("generate", 12),  # high ceiling — stops naturally when all files generated
            ("validate", max_fix),
        ]

        if is_resume:
            # Skip plan — jump straight to generate
            passes = [p for p in passes if p[0] != "plan"]
        if enable_improve:
            passes.append(("improve", max_improve))
        passes.append(("polish", 1))   # deterministic auto-enhancement (no LLM)
        passes.append(("verify", 2))   # assembled runtime check — simulates browser errors
        passes.append(("deliver", 1))

        for pass_name, max_iter in passes:
            ctx.current_pass = pass_name
            ctx.pass_budgets[pass_name] = max_iter

            for iteration in range(max_iter):
                ctx.iterations[pass_name] = iteration + 1

                # Emit "running" at the start of each iteration (not once per pass)
                # This ensures per-file announcements for generate iterations
                await self._emit(progress_cb, ctx, "running")

                try:
                    if pass_name == "plan":
                        result = await self._pass_plan(ctx, progress_cb)
                    elif pass_name == "generate":
                        result = await self._pass_generate(ctx, progress_cb)
                    elif pass_name == "validate":
                        result = await self._pass_validate(ctx)
                    elif pass_name == "improve":
                        result = await self._pass_improve(ctx)
                    elif pass_name == "polish":
                        result = await self._pass_polish(ctx)
                    elif pass_name == "verify":
                        result = await self._pass_verify(ctx)
                    elif pass_name == "deliver":
                        result = await self._pass_deliver(ctx)
                    else:
                        result = PassResult(done=True)
                except Exception as exc:
                    # Enhancer passes must not sink a build that already has
                    # files — degrade to a quality warning and move on to
                    # deliver. This is what turns "build failed for 2 CSS
                    # syntax errors" into "app delivered, polish skipped".
                    if pass_name not in _NON_FATAL_PASSES:
                        raise
                    import traceback as _tb
                    log.warning(
                        "app_builder.pass_exception_nonfatal",
                        pass_name=pass_name,
                        error=str(exc),
                        traceback=_tb.format_exc(),
                    )
                    ctx.flag_quality_issue(
                        f"The {pass_name} step hit an internal error and was "
                        f"skipped; the app was delivered without it.",
                    )
                    await self._emit(
                        progress_cb, ctx, "running",
                        f"{pass_name} skipped (internal error)",
                    )
                    ctx.last_completed_pass = ctx.last_completed_pass or pass_name
                    break  # stop retrying this pass; advance to the next

                await self._emit(progress_cb, ctx, "running", result.detail)

                if result.done:
                    await self._emit(progress_cb, ctx, "complete", result.detail)
                    ctx.last_completed_pass = pass_name
                    break

                if result.error and not result.recoverable:
                    await self._emit(progress_cb, ctx, "failed", result.error)
                    raise RuntimeError(f"Pass '{pass_name}' failed: {result.error}")
            else:
                if pass_name in ("validate", "verify") and ctx.errors:
                    label = "Validation" if pass_name == "validate" else "Runtime verification"
                    ctx.flag_quality_issue(
                        f"{label} reached the retry limit with {len(ctx.errors)} issue(s) still present.",
                        errors=ctx.errors[:10],
                    )
                await self._emit(progress_cb, ctx, "complete", f"max iterations ({max_iter})")

    # --- V2: Constraint-Driven Synthesis Pipeline ---

    async def _run_pipeline_v2(self, ctx: PipelineContext, progress_cb: Callable | None) -> None:
        """Constraint-driven synthesis pipeline (v2).

        Three phases:
        1. Comprehension -- LLM extracts structured spec
        2. Compilation -- deterministic: spec -> skeleton + tests
        3. Synthesis -- one behavior at a time, QuickJS verifies after each
        """
        from augmentum.tools.constraint_schema import parse_spec, validate_spec
        from augmentum.tools.synthesis_loop import SynthesisLoop

        # --- Phase 1: Comprehension ---
        ctx.current_pass = "comprehension"
        await self._emit(progress_cb, ctx, "running", "extracting spec")

        from augmentum.tools.application_scaffolds import build_comprehension_prompt
        messages = build_comprehension_prompt(ctx.description, ctx.scaffold_id)
        # Cap at 3072 tokens — enough for a full spec but prevents models
        # from generating implementation code inside the spec JSON.
        response = await self._call_llm(messages, max_tokens=3072, model=self._request_model)
        ctx._total_llm_calls += 1
        usage = getattr(self._call_llm, '_last_usage', None)
        if usage:
            ctx._total_tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

        try:
            spec = parse_spec(response)
        except ValueError:
            # Retry with a more explicit prompt emphasizing valid JSON
            log.warning("pipeline_v2.comprehension_retry", reason="JSON parse failed, retrying")
            retry_messages = [
                {"role": "system", "content": (
                    "Your previous response was not valid JSON. Output ONLY a valid JSON object.\n"
                    "Do NOT use trailing commas. Do NOT use single quotes. "
                    "Use null instead of leaving fields empty. "
                    "Do NOT wrap in markdown code fences.\n"
                    "Required structure: {\"name\": \"...\", \"state_schema\": {...}, "
                    "\"elements\": [...], \"constraints\": [...]}"
                )},
                {"role": "user", "content": messages[-1]["content"]},
            ]
            response2 = await self._call_llm(retry_messages, max_tokens=4096, model=self._request_model)
            ctx._total_llm_calls += 1
            try:
                spec = parse_spec(response2)
            except ValueError as exc:
                await self._emit(progress_cb, ctx, "failed", f"spec parse failed after retry: {exc}")
                raise RuntimeError(f"Comprehension failed: {exc}") from exc

        errors = validate_spec(spec)
        if errors:
            log.warning("pipeline_v2.spec_validation", errors=errors)

        ctx.project_name = spec.name or ctx.project_name
        await self._emit(progress_cb, ctx, "complete",
                         f"{len(spec.constraints)} constraints, {len(spec.elements)} elements")

        # --- Phase 2 + 3: Compilation + Synthesis ---
        ctx.current_pass = "synthesis"
        await self._emit(progress_cb, ctx, "running", "synthesizing behaviors")

        s = self._get_settings() if self._get_settings else {}
        max_attempts = s.get("app_builder_max_fix_iterations", 3)

        loop = SynthesisLoop(
            call_llm=self._call_llm,
            max_attempts=max_attempts,
            request_model=self._request_model,
        )
        result = await loop.run(spec, progress_cb=progress_cb)

        ctx._total_llm_calls += result.total_llm_calls
        ctx._total_tokens += result.total_tokens

        # Convert synthesis result to pipeline context files
        ctx.files = [
            {"path": "index.html", "role": "entry", "lang": "html", "content": result.skeleton},
            {"path": "styles.css", "role": "style", "lang": "css", "content": result.css},
            {"path": "app.js", "role": "script", "lang": "javascript", "content": result.js},
        ]

        # Score from constraint results
        passed = sum(1 for cr in result.constraint_results if cr["status"] == "passed")
        total = len(result.constraint_results)
        ctx.score = round((passed / total) * 10, 1) if total > 0 else 0.0

        await self._emit(progress_cb, ctx, "complete",
                         f"{passed}/{total} constraints satisfied")

        # --- Polish (same as v1, deterministic) ---
        # Wrapped: a polish crash (e.g. CSS regex) must not block deliver.
        ctx.current_pass = "polish"
        await self._emit(progress_cb, ctx, "running")
        try:
            polish_result = await self._pass_polish(ctx)
            await self._emit(progress_cb, ctx, "complete", polish_result.detail)
        except Exception as exc:
            import traceback as _tb
            log.warning("app_builder.pass_exception_nonfatal", pass_name="polish",
                        error=str(exc), traceback=_tb.format_exc())
            ctx.flag_quality_issue(
                "The polish step hit an internal error and was skipped; "
                "the app was delivered without it.",
            )
            await self._emit(progress_cb, ctx, "complete", "polish skipped (internal error)")

        # --- Deliver (same as v1) ---
        ctx.current_pass = "deliver"
        await self._emit(progress_cb, ctx, "running")
        deliver_result = await self._pass_deliver(ctx)
        await self._emit(progress_cb, ctx, "complete", deliver_result.detail)

    # --- Pass implementations ---

    async def _pass_plan(self, ctx: PipelineContext, progress_cb: Callable | None = None) -> PassResult:
        from augmentum.tools.application_scaffolds import GRAMMAR_PLAN, GRAMMAR_PLAN_ITERATE
        messages = build_plan_prompt(ctx.description, ctx.scaffold_id,
                                     ctx.files if ctx.is_iteration else None)
        # Iteration uses ACTION-prefixed lines, new builds use the
        # contract-required format — apply the matching grammar.
        plan_grammar = GRAMMAR_PLAN_ITERATE if ctx.is_iteration else GRAMMAR_PLAN
        response = await self._call_llm(messages, max_tokens=2048, model=self._request_model, grammar=plan_grammar)
        ctx._total_llm_calls += 1
        usage = getattr(self._call_llm, '_last_usage', None)
        if usage:
            ctx._total_tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        if not response.strip() and plan_grammar:
            log.warning(
                "app_builder.plan_empty_response_retry_without_grammar",
                model=self._request_model,
                iteration=ctx.iterations.get("plan", 0),
                is_iteration=ctx.is_iteration,
            )
            await self._emit(
                progress_cb,
                ctx,
                "running",
                "strict plan returned empty; retrying without grammar",
            )
            response = await self._call_llm(
                messages,
                max_tokens=2048,
                model=self._request_model,
            )
            ctx._total_llm_calls += 1
            usage = getattr(self._call_llm, '_last_usage', None)
            if usage:
                ctx._total_tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

        files = self._parse_file_plan(response)
        if not files:
            log.warning("app_builder.plan_parse_failed", response_len=len(response), response_head=response[:200])
            # On final retry (iteration 2), fall back to scaffold defaults rather than
            # delivering an empty project. This handles models that return empty/unparseable responses.
            if ctx.iterations.get("plan", 0) >= 2:
                scaffold = SCAFFOLDS.get(ctx.scaffold_id, SCAFFOLDS["static"])
                files = [
                    {"path": df["path"], "role": df["role"],
                     "lang": df.get("lang", self._infer_lang(df["path"])),
                     "action": "create",
                     "description": "(fallback: LLM plan failed, using scaffold defaults)"}
                    for df in scaffold["default_files"]
                ]
                log.warning("app_builder.plan_fallback_to_scaffold",
                            scaffold=ctx.scaffold_id, files=[f["path"] for f in files])
            else:
                return PassResult(done=False, error="Could not parse file plan — retrying", recoverable=True)

        # --- File count sanity check ---
        # If model plans 8+ files, ask it to reconsider. Most apps need 3-5 files.
        # Over-splitting is the #1 plan quality issue on smaller models.
        if len(files) > 7 and not ctx.is_iteration:
            file_list = "\n".join(f"FILE: {f['path']} | ROLE: {f.get('role', 'script')} | DESCRIPTION: {f.get('description', '')}" for f in files)
            sanity_messages = [
                {"role": "system", "content": (
                    "You are reviewing a file plan. The developer planned too many files for a simple app. "
                    "Most web apps need 3-5 files (HTML + CSS + JS + maybe 1-2 modules). "
                    "Only complex multi-view apps with distinct subsystems warrant 7+ files.\n\n"
                    "Review the plan below and output a trimmed version with only the essential files. "
                    "Merge files that can be combined. Use the same FILE: format.\n"
                    "End with __PASS_COMPLETE__"
                )},
                {"role": "user", "content": (
                    f"Project: {ctx.description}\n\n"
                    f"Original plan ({len(files)} files):\n{file_list}\n\n"
                    f"Is this many files necessary? Output the essential file list."
                )},
            ]
            log.info("app_builder.plan_sanity_check", original_count=len(files))
            sanity_response = await self._call_llm(sanity_messages, max_tokens=2048, model=self._request_model)
            ctx._total_llm_calls += 1
            usage = getattr(self._call_llm, '_last_usage', None)
            if usage:
                ctx._total_tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
            trimmed = self._parse_file_plan(sanity_response)
            if trimmed and len(trimmed) < len(files):
                log.info("app_builder.plan_trimmed", original=len(files), trimmed=len(trimmed))
                files = trimmed
            # If sanity check returned same or more files, keep original (model insists they're needed)

        # Sort files by dependency order: entry first, then styles, then scripts, then data
        role_order = {"entry": 0, "style": 1, "script": 2, "module": 3, "data": 4, "readme": 5}
        files.sort(key=lambda f: role_order.get(f.get("role", "script"), 3))

        # --- Scaffold minimum enforcement (new builds only) ---
        # In iteration mode, the project already has all required files —
        # don't add scaffold defaults or we'll regenerate files that don't need changes.
        if not ctx.is_iteration:
            scaffold = SCAFFOLDS.get(ctx.scaffold_id, SCAFFOLDS["static"])
            planned_roles = {f.get("role") for f in files}
            for default_file in scaffold["default_files"]:
                if default_file["role"] not in planned_roles:
                    files.append({
                        "path": default_file["path"],
                        "role": default_file["role"],
                        "lang": default_file.get("lang", self._infer_lang(default_file["path"])),
                        "action": "create",
                        "description": f"(auto-added: scaffold '{scaffold['name']}' requires a {default_file['role']} file)",
                    })
                    log.info("app_builder.plan_gap_filled", missing_role=default_file["role"],
                             injected=default_file["path"], scaffold=ctx.scaffold_id)

            # Re-sort after potential additions
            files.sort(key=lambda f: role_order.get(f.get("role", "script"), 3))

        ctx.planned_files = files
        ctx.project_name = ctx.project_name or self._extract_name(ctx.description)

        # Initialize working document (Manus-style persistent context)
        file_checklist = "\n".join(f"- [ ] {f['path']} ({f.get('role', 'script')}) — {f.get('description', '')}" for f in files)
        design_rules = build_design_rules(ctx.description, ctx.scaffold_id)

        # Reference implementations — dynamically selected based on description + scaffold.
        # Models pattern-match against working code examples far more reliably than
        # text instructions. select_references() picks the most relevant from
        # the library (keyword-scored, with always-include bases per scaffold).
        # Cap by tier (spec §2/§5): small models silently drop later instructions
        # when context grows past ~500 tokens — keep them on a single canonical
        # exemplar instead of drowning them in five.
        from augmentum.tools.application_references import select_references
        ref_caps = {"small": 1, "medium": 3, "large": 5, "frontier": 2}
        max_refs = ref_caps.get(ctx.model_tier, 3)
        reference_section = select_references(ctx.description, ctx.scaffold_id, max_refs=max_refs)

        ctx.working_doc = (
            f"# Project: {ctx.project_name}\n\n"
            f"## Goal\n{ctx.description}\n\n"
            f"## Files to Generate\n{file_checklist}\n\n"
            f"{reference_section}\n"
            f"{design_rules}\n\n"
            f"## Decisions\n(updated after each file)\n\n"
            f"## API Surface\n(tracks CSS classes, JS functions, DOM IDs across files)\n"
        )

        return PassResult(done=True, detail=f"{len(files)} files planned")

    def _build_generate_system_prompt(self, tier: str) -> str:
        """System prompt for the per-file generate pass, sized for ``tier``.

        Small models (≤8B) drop instructions silently when the system
        prompt exceeds ~500 tokens, so we strip the visual-polish block
        and keep only the correctness rules. Medium and larger tiers
        get the full quality bar.
        """
        from augmentum.tools.application_scaffolds import adapt_prompt_for_tier

        if tier == "small":
            base = (
                "You are generating one file of a multi-file web app.\n\n"
                "The working document below has the Project Map (defined IDs, "
                "globals, functions) and any Unresolved items this file should "
                "wire up.\n\n"
                "Output the file as a fenced code block whose label is the "
                "exact filename:\n"
                "  ```<filename>\n"
                "  file content\n"
                "  ```\n"
                "End with __PASS_COMPLETE__.\n\n"
                "Rules:\n"
                "- All JS shares ONE global scope. Use window.X for cross-file "
                "symbols. Never redeclare a name another file already defined.\n"
                "- Do NOT include <script src='…'> or <link href='…'> for "
                "project files — they are assembled automatically.\n"
                "- Use the design tokens from the working document — do not "
                "invent your own palette.\n"
                "- Honour the file's contract: define every PROVIDES and wire "
                "every WIRES selector.\n"
                "- Complete code only — no stubs, no TODO comments."
            )
            return adapt_prompt_for_tier(base, tier)

        full = (
            "You are a senior frontend developer generating production-quality code.\n"
            "Your output should be indistinguishable from a hand-crafted professional application.\n\n"
            "The working document below contains:\n"
            "- **Project Map** with all defined IDs, classes, globals, and functions\n"
            "- **⚠ Unresolved items** that THIS file should handle (wire event handlers, initialize canvas, etc.)\n"
            "- Reference implementations showing the level of quality expected\n\n"
            "IMPORTANT: Check the Unresolved section — if it lists buttons without handlers or canvas without init,\n"
            "this file MUST address those items.\n\n"
            "Output the file as a fenced code block with the filename:\n"
            "  ```<filename>\n"
            "  file content here\n"
            "  ```\n\n"
            "End with __PASS_COMPLETE__\n\n"
            "SCOPE RULES — All JS files share ONE global scope:\n"
            "- NEVER use const/let/class for a name that ANY previous file already declared\n"
            "- To share data between files, use window.X (e.g. window.CONFIG, window.gameState)\n"
            "- Only the FIRST file should define shared constants. Later files access them via window.X\n"
            "- Do NOT include <script src='...'> or <link href='...'> for project files — they are assembled automatically\n\n"
            "QUALITY STANDARD — go the extra mile:\n"
            "- Write complete, polished code — no stubs, no placeholders, no TODO comments\n"
            "- CSS: Use custom properties for ALL colors. Add hover states on every interactive element. "
            "Include transitions (150-300ms) on color/background/transform changes. "
            "Use a consistent spacing scale. Add focus-visible styles for keyboard users.\n"
            "- HTML: Use semantic elements (<nav>, <main>, <section>, <article>). "
            "Add aria-labels on icon-only buttons. Every form input needs a visible label.\n"
            "- JS: Escape ALL user-provided text before inserting into DOM (prevent XSS). "
            "Use addEventListener, never inline onclick. Wrap in IIFE with 'use strict'.\n"
            "- Visual: Make it beautiful. Thoughtful typography, cohesive palette, "
            "subtle shadows for depth, smooth animations that feel natural. "
            "The design should have personality — not generic bootstrap.\n"
            "- Responsive: Must look great on mobile (320px+). Use CSS Grid or Flexbox.\n"
            "- End with __PASS_COMPLETE__"
        )
        return adapt_prompt_for_tier(full, tier)

    async def _pass_generate(self, ctx: PipelineContext, progress_cb: Callable | None = None) -> PassResult:
        remaining = [f for f in ctx.planned_files if f["path"] not in ctx.generated_files]
        if not remaining:
            return PassResult(done=True, detail=f"{len(ctx.generated_files)} files")

        # --- Batch generation path (toolkit spec §25) ----------------------
        # For small apps (≤5 files) generating all files in a single LLM
        # call produces more coherent cross-file wiring than serialised
        # per-file generation against a growing working doc. Contracts
        # make the output shape constrained enough that the parser can
        # reliably recover all files from one response. We only attempt
        # batch on the FIRST iteration of the generate pass with an
        # empty generated_files — partial state from prior retries
        # stays in sequential mode where the existing intercept logic
        # can reason about it.
        settings = self._get_settings() if self._get_settings else {}
        batch_enabled = settings.get("app_builder_batch_small_apps", True)
        if (
            batch_enabled
            and not ctx.is_iteration
            and not ctx.generated_files
            and len(ctx.planned_files) <= _BATCH_FILE_LIMIT
            and len(ctx.planned_files) == len(remaining)
        ):
            done = await self._attempt_batch_generate(ctx, progress_cb)
            if done:
                return PassResult(
                    done=True,
                    detail=f"{len(ctx.generated_files)} files (batch)",
                )
            # Fall through to sequential — batch returned partial or
            # unparseable output; the sequential path will pick up
            # whichever files did land.
            remaining = [
                f for f in ctx.planned_files if f["path"] not in ctx.generated_files
            ]
            if not remaining:
                return PassResult(done=True, detail=f"{len(ctx.generated_files)} files (batch)")

        # Generate ONE file per iteration in dependency order.
        target = remaining[0]

        # --- Diff mode for MODIFY actions (iteration) ---
        # If the file exists and the plan says "modify", use SEARCH/REPLACE patches
        # instead of regenerating the entire file. Much cheaper and less destructive.
        if target.get("action") == "modify" and ctx.is_iteration:
            existing = next((f for f in ctx.files if f["path"] == target["path"]), None)
            if existing:
                return await self._generate_diff(ctx, target, existing, progress_cb)

        # Build context: working doc (includes project map) + previously generated files.
        # The project map gives the model a structured view of what exists and what's
        # missing — full file contents are included only for the MOST RECENT file
        # (the one the model needs to reference most closely). Earlier files get signatures.
        context_parts = [f"## Working Document\n{ctx.working_doc}"]

        gen_paths = list(ctx.generated_files.keys())
        for i, path in enumerate(gen_paths):
            content = ctx.generated_files[path]
            is_latest = (i == len(gen_paths) - 1)
            if is_latest or len(gen_paths) <= 3:
                # Show full content for the most recent file (or if few files)
                context_parts.append(f"=== {path} (full) ===\n{content}")
            else:
                # Show signature only for earlier files — map has the details
                exports = []
                for m in re.finditer(r'(?:window\.(\w+)\s*=|^(?:function|class|const)\s+(\w+))', content, re.MULTILINE):
                    exports.append(m.group(1) or m.group(2))
                sig = ", ".join(exports[:15]) if exports else "see Project Map"
                lines = content.count("\n") + 1
                context_parts.append(f"=== {path} ({lines} lines, exports: {sig}) ===")

        existing_context = "\n\n".join(context_parts)

        # Tier-adapted system prompt (spec §5). Small models choke on the
        # 15-bullet quality block, so they get a slimmer version that
        # focuses on correctness; medium+ tiers keep the full prompt.
        system_prompt = self._build_generate_system_prompt(ctx.model_tier)

        # Check if this file had a previous failed generation attempt
        retry_hint = ""
        gen_fail_key = f"_gen_fails_{target['path']}"
        fail_count = getattr(ctx, gen_fail_key, 0)
        if fail_count > 0:
            retry_hint = (
                f"\n\nWARNING: Previous attempt #{fail_count} for this file was rejected because "
                "it did not contain valid code. Output ONLY the file content as a fenced code block. "
                "Do NOT output descriptions, plans, or explanations — just the code.\n"
            )

        contract_block = _format_contract_for_prompt(target, ctx.planned_files)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"{existing_context}\n\n---\n\n"
                f"Project: {ctx.description}\n\n"
                f"Generate this file: ```{target['path']}```\n"
                f"- {target['path']} ({target.get('role', 'script')}): {target.get('description', '')}"
                f"{contract_block}"
                f"{retry_hint}"
            )},
        ]

        gen_tokens = getattr(self, '_max_tokens', 8192)

        # Run LLM call with heartbeat — sends keep-alive progress events every 30s
        # so the frontend doesn't think the connection died during long generations.
        import asyncio as _aio
        llm_task = _aio.create_task(self._call_llm(messages, max_tokens=gen_tokens, model=self._request_model))
        elapsed = 0
        while not llm_task.done():
            try:
                await _aio.wait_for(_aio.shield(llm_task), timeout=30.0)
            except TimeoutError:
                elapsed += 30
                # Heartbeat with token count
                tokens_info = ""
                usage = getattr(self._call_llm, '_last_usage', None)
                if usage and ctx._total_tokens > 0:
                    tokens_info = f" | {ctx._total_tokens:,} tokens used"
                await self._emit(progress_cb, ctx, "running",
                                 f"generating {target['path']}... ({elapsed}s{tokens_info})")
        response = llm_task.result()

        # Track cumulative token usage
        ctx._total_llm_calls += 1
        usage = getattr(self._call_llm, '_last_usage', None)
        if usage:
            ctx._total_tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)

        # Parse the generated file
        new_files = self._parse_generated_files(response)

        if not new_files:
            # Fallback 1: Look for language-tagged fenced blocks and map to target file
            # LLMs often output ```javascript instead of ```game.js
            lang_map = {"javascript": ".js", "js": ".js", "typescript": ".ts", "ts": ".ts",
                        "html": ".html", "htm": ".html", "css": ".css", "scss": ".css"}
            for m in re.finditer(r"```([a-zA-Z]+)\n([\s\S]*?)```", response):
                lang_tag = m.group(1).lower()
                content = m.group(2).rstrip()
                if lang_tag in lang_map and content.strip() and lang_tag not in ("context", "json", "text", "markdown"):
                    # Map to the target file if its extension matches
                    if target["path"].endswith(lang_map.get(lang_tag, "")):
                        new_files = [{"path": target["path"], "content": content.strip()}]
                        break

        if not new_files:
            # Fallback 2: treat the whole response as the file content (minus markers)
            content = re.sub(r"```context[\s\S]*?```", "", response)
            content = re.sub(r"__PASS_COMPLETE__.*", "", content).strip()
            if content.startswith("```"):
                content = re.sub(r"^```\S*\n?", "", content)
                content = re.sub(r"\n?```\s*$", "", content)
            if content.strip():
                # Validate the content looks like actual code, not echoed prompt text.
                # Models sometimes return the plan description or prompt instead of code.
                is_code = self._looks_like_code(content, target.get("role", "script"))
                if is_code:
                    new_files = [{"path": target["path"], "content": content.strip()}]
                else:
                    gen_fail_key = f"_gen_fails_{target['path']}"
                    setattr(ctx, gen_fail_key, getattr(ctx, gen_fail_key, 0) + 1)
                    fails = getattr(ctx, gen_fail_key)
                    log.warning("app_builder.generate_not_code",
                                file=target["path"], attempt=fails, head=content[:100])
                    if fails >= 3:
                        # After 3 failed attempts, skip this file
                        log.warning("app_builder.generate_skipped", file=target["path"])
                        ctx.generated_files[target["path"]] = ""
                        return PassResult(done=False, detail=f"skipped {target['path']} after {fails} failed attempts")
                    return PassResult(
                        done=False,
                        detail=f"model returned non-code for {target['path']} (attempt {fails}), retrying",
                    )

        # --- Truncation detection and continuation ---
        # If the response lacks __PASS_COMPLETE__ AND the code has unbalanced
        # braces/tags, the model likely hit max_tokens mid-generation.
        # Send the partial code back with "continue from here" instructions.
        has_pass_complete = "__PASS_COMPLETE__" in response
        for f in new_files:
            if not has_pass_complete and f.get("content"):
                content = f["content"]
                opens = content.count("{") - content.count("}")
                # Also check for truncated HTML (unclosed tags)
                html_opens = len(re.findall(r'<(?!br|hr|img|input|meta|link|!)[a-z][\w-]*', content, re.I))
                html_closes = len(re.findall(r'</[a-z][\w-]*>', content, re.I))
                is_truncated = opens > 2 or (target.get("role") == "entry" and html_opens > html_closes + 3)

                if is_truncated:
                    log.info("app_builder.truncated_file", file=f["path"],
                             open_braces=opens, lines=content.count("\n") + 1)
                    # Build context for continuation: project map + file structure + tail
                    lines = content.split("\n")
                    total_lines = len(lines)

                    # Extract what's already defined in this file (functions, classes, variables)
                    defined_in_file = []
                    for dm in re.finditer(r'(?:function\s+(\w+)|class\s+(\w+)|(?:const|let)\s+(\w+)\s*=)', content):
                        defined_in_file.append(dm.group(1) or dm.group(2) or dm.group(3))

                    # Show first 10 lines (structure/imports) + last 30 lines (where to continue)
                    head = "\n".join(lines[:10])
                    tail = "\n".join(lines[-30:])

                    continuation_msgs = [
                        {"role": "system", "content": (
                            "A file was truncated mid-generation due to token limits. "
                            "Your job: continue EXACTLY from where it stopped. "
                            "Do NOT repeat ANY code already written. "
                            "Do NOT add code fences or filenames. "
                            "Just output the remaining code to complete the file.\n\n"
                            "Close all unclosed braces, complete all unfinished functions, "
                            "and ensure the file is syntactically valid when your output "
                            "is appended to the existing code."
                        )},
                        {"role": "user", "content": (
                            f"## Project Context\n{ctx.working_doc}\n\n"
                            f"## File: {f['path']} ({target.get('role', 'script')})\n"
                            f"**{total_lines} lines generated, then truncated.**\n"
                            f"Already defined in this file: {', '.join(defined_in_file[:15]) if defined_in_file else 'see code below'}\n\n"
                            f"### Opening (lines 1-10):\n```\n{head}\n```\n\n"
                            f"### Truncation point (last 30 lines):\n```\n{tail}\n```\n\n"
                            f"Continue from the LAST LINE above. Complete the file. "
                            f"Do NOT repeat any of the code shown above."
                        )},
                    ]
                    gen_tokens = getattr(self, '_max_tokens', 8192)
                    try:
                        continuation = await self._call_llm(
                            continuation_msgs, max_tokens=gen_tokens,
                            model=self._request_model,
                        )
                        ctx._total_llm_calls += 1
                        c_usage = getattr(self._call_llm, '_last_usage', None)
                        if c_usage:
                            ctx._total_tokens += c_usage.get("prompt_tokens", 0) + c_usage.get("completion_tokens", 0)

                        # Strip any code fences or markers from continuation
                        cont_clean = continuation
                        cont_clean = re.sub(r"^```\S*\n?", "", cont_clean)
                        cont_clean = re.sub(r"\n?```\s*$", "", cont_clean)
                        cont_clean = re.sub(r"__PASS_COMPLETE__.*", "", cont_clean).strip()

                        if cont_clean:
                            f["content"] = content + "\n" + cont_clean
                            log.info("app_builder.continued_file", file=f["path"],
                                     added_lines=cont_clean.count("\n") + 1,
                                     total_lines=f["content"].count("\n") + 1)
                    except Exception as exc:
                        log.warning("app_builder.continuation_failed", file=f["path"], error=str(exc))

        # Per-file: intercept → store → refresh working doc. Refreshing
        # the project map AFTER each file (rather than once at the end
        # of the loop) means the NEXT file's intercept — and any retry
        # path that consults ctx.working_doc — sees the fresh API
        # surface. Previously the map only updated after the full batch,
        # so a multi-file LLM response processed the 2nd/3rd/... file
        # against a stale map.
        for f in new_files:
            # --- Layer 1: Streaming interception (v0 LLM Suspense pattern) ---
            # Fix predictable errors in the generated output BEFORE storing.
            # User never sees the broken intermediate state.
            f["content"] = self._intercept_generated_code(f, ctx)

            ctx.generated_files[f["path"]] = f["content"]
            existing = next((ef for ef in ctx.files if ef["path"] == f["path"]), None)
            if existing:
                existing["content"] = f["content"]
            else:
                planned = next((pf for pf in ctx.planned_files if pf["path"] == f["path"]), {})
                ctx.files.append({
                    "path": f["path"],
                    "lang": planned.get("lang", self._infer_lang(f["path"])),
                    "content": f["content"],
                    "role": planned.get("role", self._infer_role(f["path"])),
                })

            # Mark this file's checklist entry done and rebuild the
            # project map so later iterations of this loop see it.
            ctx.working_doc = ctx.working_doc.replace(
                f"- [ ] {f['path']}", f"- [x] {f['path']}"
            )
            ctx.working_doc = self._update_project_map(ctx)

        # Checkpoint: save partial progress to the background build state (not artifact store).
        # This keeps checkpoints out of the user-facing library while still enabling
        # resume-from-partial on retry. The build state is in app.state._active_builds.
        # No artifact store write — checkpoints are ephemeral server-side state.

        # Check if additional files were requested
        needs_another = "__NEEDS_ANOTHER_PASS__" in response
        still_remaining = [f for f in ctx.planned_files if f["path"] not in ctx.generated_files]

        if not still_remaining and not needs_another:
            # --- Post-generation gap analysis ---
            # Scan the entry HTML for <script src="X"> and <link href="X"> that
            # reference files not in the project. Instead of silently stripping
            # them later, add those files to the plan and keep generating.
            gap_files = self._detect_missing_references(ctx)
            if gap_files:
                for gf in gap_files:
                    ctx.planned_files.append(gf)
                    # Update the working doc checklist
                    ctx.working_doc += f"\n- [ ] {gf['path']} ({gf['role']}) — {gf.get('description', 'auto-detected from HTML references')}"
                log.info("app_builder.gap_analysis", added=[g["path"] for g in gap_files])
                return PassResult(
                    done=False,
                    detail=f"gap analysis added {len(gap_files)} missing files",
                    output=f"HTML references {len(gap_files)} files not yet generated",
                )
            return PassResult(done=True, detail=f"{len(ctx.generated_files)} files")

        return PassResult(
            done=False,
            detail=f"{len(ctx.generated_files)}/{len(ctx.planned_files)} files",
            output=f"Generated {target['path']}, {len(still_remaining)} remaining",
        )

    async def _generate_diff(self, ctx: PipelineContext, target: dict,
                             existing: dict, progress_cb: Callable | None) -> PassResult:
        """Generate SEARCH/REPLACE patches for an existing file instead of full regeneration.

        Used in iteration mode (ACTION: modify) to make targeted changes.
        Much cheaper (fewer tokens) and less destructive than full regeneration.
        """
        # Build context showing the file to modify + what other files export
        context_parts = [f"## Working Document\n{ctx.working_doc}"]
        # Show full content of the file to modify
        context_parts.append(f"=== {target['path']} (MODIFY THIS FILE) ===\n{existing['content']}")
        # Show signatures of other files for reference
        for f in ctx.files:
            if f["path"] == target["path"]:
                continue
            exports = []
            for m in re.finditer(
                r'(?:window\.(\w+)\s*=|^(?:function|class|const|var|let)\s+(\w+))',
                f["content"], re.MULTILINE,
            ):
                exports.append(m.group(1) or m.group(2))
            sig = ", ".join(exports[:20]) if exports else "none"
            context_parts.append(f"=== {f['path']} (exports: {sig}) ===")

        existing_context = "\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": (
                "You are modifying one file in an existing web application.\n"
                "Output ONLY SEARCH/REPLACE blocks — no full file regeneration.\n\n"
                "Format:\n"
                "<<<<<<< SEARCH\n"
                "exact lines to find in the current file\n"
                "=======\n"
                "replacement lines\n"
                ">>>>>>> REPLACE\n\n"
                "Rules:\n"
                "- Change ONLY what's needed for the requested modification\n"
                "- Keep all existing functionality intact\n"
                "- Match the existing code style (indentation, naming, patterns)\n"
                "- Multiple SEARCH/REPLACE blocks are fine for multi-part changes\n"
                "- End with __PASS_COMPLETE__"
            )},
            {"role": "user", "content": (
                f"{existing_context}\n\n---\n\n"
                f"Modification requested: {ctx.description}\n\n"
                f"File to modify: {target['path']}\n"
                f"Change: {target.get('description', ctx.description)}"
            )},
        ]

        gen_tokens = getattr(self, '_max_tokens', 8192)
        response = await self._call_llm(messages, max_tokens=gen_tokens, model=self._request_model)

        # Apply SEARCH/REPLACE patches to the existing file
        patches = self._apply_file_patches([existing], response)
        if patches == 0:
            # Fallback: if model returned full file instead of patches, use it
            new_files = self._parse_generated_files(response)
            if new_files:
                existing["content"] = new_files[0]["content"]
                patches = 1
                log.info("generate_diff.fallback_full_file", file=target["path"])

        if patches > 0:
            existing["content"] = self._intercept_generated_code(
                {"path": target["path"], "content": existing["content"]}, ctx
            )
            ctx.generated_files[target["path"]] = existing["content"]
            log.info("generate_diff.applied", file=target["path"], patches=patches)
        else:
            # No changes could be applied — mark as done anyway
            ctx.generated_files[target["path"]] = existing["content"]
            log.warning("generate_diff.no_patches", file=target["path"])

        # Update working doc
        ctx.working_doc = ctx.working_doc.replace(
            f"- [ ] {target['path']}", f"- [x] {target['path']}"
        )

        still_remaining = [f for f in ctx.planned_files if f["path"] not in ctx.generated_files]
        if not still_remaining:
            return PassResult(done=True, detail=f"{len(ctx.generated_files)} files")

        return PassResult(
            done=False,
            detail=f"{len(ctx.generated_files)}/{len(ctx.planned_files)} files",
            output=f"Modified {target['path']}, {len(still_remaining)} remaining",
        )

    async def _pass_validate(self, ctx: PipelineContext) -> PassResult:


        # ===================================================================
        # Layer 1: DETERMINISTIC AUTOFIX (v0 pattern — fix without LLM)
        # Fixes predictable structural errors instantly. No tokens burned.
        # ===================================================================
        auto_fixed = 0
        scripts = [f for f in ctx.files if f.get("role") in ("script", "module")]
        entry = next((f for f in ctx.files if f["role"] == "entry"), None)

        # --- Autofix 1: Duplicate const/let/class across files ---
        # When assembled, all scripts share global scope. Second declaration crashes.
        # Fix: rename second occurrence to use window.X assignment.
        # Only considers TOP-LEVEL declarations (brace depth 0) to avoid
        # false-positiving on `const i` inside a for-loop or function.
        if len(scripts) > 1:
            declarations: dict[str, str] = {}  # name → first file path
            for f in scripts:
                # Collect all replacements FIRST, then apply in reverse order
                # to avoid position corruption when string length changes.
                replacements: list[tuple[int, int, str]] = []  # (start, end, replacement)
                for m in re.finditer(r'^(const|let|class)\s+(\w+)', f["content"], re.MULTILINE):
                    if not self._is_top_level(f["content"], m.start()):
                        continue
                    keyword, name = m.group(1), m.group(2)
                    if name in declarations and declarations[name] != f["path"]:
                        if keyword == "class":
                            replacement = f"// class {name} already defined in {declarations[name]} — using window.{name}"
                        else:
                            replacement = f"window.{name}"
                        replacements.append((m.start(), m.end(), replacement))
                        auto_fixed += 1
                        log.info("autofix.duplicate_declaration", name=name, file=f["path"], first=declarations[name])
                    else:
                        declarations[name] = f["path"]
                # Apply replacements in REVERSE order so positions stay valid
                for start, end, replacement in reversed(replacements):
                    f["content"] = f["content"][:start] + replacement + f["content"][end:]

        # --- Autofix 2: Class used as function (ClassName.method() → window.ClassName.method()) ---
        # When a class is assigned to window.ClassName = new ClassName(), direct ClassName.method()
        # calls reference the class (no static method), not the instance.
        if len(scripts) > 1:
            # Find all class definitions and their window assignments
            class_instances: dict[str, str] = {}  # ClassName → window property name
            for f in scripts:
                for m in re.finditer(r'class\s+(\w+)', f["content"]):
                    cls = m.group(1)
                    # Check if there's a window.X = new ClassName() anywhere
                    for f2 in scripts:
                        wm = re.search(r'window\.(\w+)\s*=\s*new\s+' + cls + r'\b', f2["content"])
                        if wm:
                            class_instances[cls] = wm.group(1)
                            break

            # Fix bare ClassName.method() calls to window.instanceName.method()
            for cls, instance in class_instances.items():
                for f in scripts:
                    # Match ClassName.method( but not inside 'new ClassName' or 'class ClassName'
                    pattern = r'(?<!new\s)(?<!class\s)\b' + cls + r'\.(\w+)\s*\('
                    matches = list(re.finditer(pattern, f["content"]))
                    for m in reversed(matches):  # reverse to preserve positions
                        method = m.group(1)
                        # Don't fix if it's inside a string or comment
                        line_start = f["content"].rfind('\n', 0, m.start()) + 1
                        line = f["content"][line_start:m.start()]
                        if '//' in line or "'" in line or '"' in line:
                            continue
                        old = f"{cls}.{method}("
                        new = f"window.{instance}.{method}("
                        f["content"] = f["content"][:m.start()] + new + f["content"][m.end():]
                        auto_fixed += 1
                        log.info("autofix.class_as_function", cls=cls, method=method, file=f["path"], replacement=f"window.{instance}.{method}")

        # --- Autofix 3: Strip external file references from entry HTML ---
        # <script src="app.js"> and <link href="styles.css"> will 404 in assembled mode
        if entry:
            project_paths = {f["path"] for f in ctx.files if f["role"] != "entry"}
            for path in project_paths:
                before = entry["content"]
                entry["content"] = re.sub(
                    r'<link[^>]*href=["\']' + re.escape(path) + r'["\'][^>]*/?>(\s*\n)?',
                    '', entry["content"], flags=re.IGNORECASE,
                )
                entry["content"] = re.sub(
                    r'<script[^>]*src=["\']' + re.escape(path) + r'["\'][^>]*>\s*</script>(\s*\n)?',
                    '', entry["content"], flags=re.IGNORECASE,
                )
                if entry["content"] != before:
                    auto_fixed += 1

        if auto_fixed > 0:
            log.info("autofix.applied", count=auto_fixed)

        # ===================================================================
        # Layer 2: DETECTION (errors that can't be auto-fixed → LLM)
        # ===================================================================
        errors = []

        # Static lint
        for f in ctx.files:
            lang = f.get("lang", "")
            if lang in ("html", "htm"):
                issues = self._lint_html(f["content"])
                if issues:
                    errors.extend(f"{f['path']}: {issue}" for issue in issues)
            elif lang in ("javascript", "js"):
                issues = self._lint_js(f["content"])
                if issues:
                    errors.extend(f"{f['path']}: {issue}" for issue in issues)

        # Cross-file reference checks — the CSS/JS corpora must include
        # inline <style> and <script> blocks from entry HTML, otherwise
        # class/function references defined inline (e.g. a `.loading-fade`
        # keyframe in index.html's <style>) trip the "not defined" warning.
        # See the live pomodoro build: it false-positived on `loading-fade`
        # which was legitimately defined inside the entry's <style>.
        all_js = "\n".join(f["content"] for f in scripts)
        all_css = "\n".join(f["content"] for f in ctx.files if f.get("role") == "style")
        if entry:
            for sm in re.finditer(
                r"<style[^>]*>([\s\S]*?)</style>", entry["content"], re.IGNORECASE,
            ):
                all_css += "\n" + sm.group(1)
            # Only count inline scripts WITHOUT a src attribute — the others
            # reference project files already covered above.
            for jm in re.finditer(
                r"<script(?![^>]*\ssrc=)[^>]*>([\s\S]*?)</script>",
                entry["content"], re.IGNORECASE,
            ):
                all_js += "\n" + jm.group(1)
        if entry and all_js:
            # Onclick/event handlers referencing undefined functions
            for m in re.finditer(r'(?:onclick|onsubmit|onchange|oninput)\s*=\s*["\'](\w+)\s*\(', entry["content"]):
                fn_name = m.group(1)
                if fn_name not in all_js and fn_name not in ("alert", "confirm", "prompt", "console"):
                    errors.append(f"{entry['path']}: onclick references undefined function '{fn_name}'")

            # getElementById targeting non-existent IDs
            for f in scripts:
                for m in re.finditer(r'getElementById\s*\(\s*["\'](\w+)["\']', f["content"]):
                    elem_id = m.group(1)
                    if f'id="{elem_id}"' not in entry["content"] and f"id='{elem_id}'" not in entry["content"]:
                        errors.append(f"{f['path']}: getElementById('{elem_id}') but no element with id='{elem_id}' in HTML")

            # classList.add/remove/toggle on classes not defined in CSS
            if all_css:
                for f in scripts:
                    for m in re.finditer(r'classList\.(?:add|remove|toggle)\s*\(\s*["\'](\w[\w-]*)["\']', f["content"]):
                        cls = m.group(1)
                        if f".{cls}" not in all_css and cls not in ("hidden", "active", "visible", "open", "closed", "disabled", "selected"):
                            errors.append(f"{f['path']}: classList uses '{cls}' but class not defined in CSS (may be intentional)")

            # addEventListener on IDs that don't exist
            for f in scripts:
                for m in re.finditer(r'getElementById\s*\(\s*["\'](\w+)["\']\s*\)\.addEventListener', f["content"]):
                    elem_id = m.group(1)
                    if f'id="{elem_id}"' not in entry["content"]:
                        errors.append(f"{f['path']}: addEventListener on #{elem_id} but element doesn't exist — will throw TypeError")

        # Verify all window.X references point to something defined
        if len(scripts) > 1:
            window_assignments: set[str] = set()
            for f in scripts:
                for m in re.finditer(r'window\.(\w+)\s*=', f["content"]):
                    window_assignments.add(m.group(1))
            for f in scripts:
                for m in re.finditer(r'window\.(\w+)(?:\.\w+)*\s*[\(.]', f["content"]):
                    name = m.group(1)
                    if name not in window_assignments and name not in (
                        "window",  # window.window is valid (self-reference)
                        "addEventListener", "removeEventListener", "innerWidth", "innerHeight",
                        "location", "navigator", "document", "localStorage", "sessionStorage",
                        "setTimeout", "setInterval", "clearTimeout", "clearInterval",
                        "requestAnimationFrame", "cancelAnimationFrame", "getComputedStyle",
                        "open", "close", "alert", "confirm", "prompt", "scrollTo",
                        "matchMedia", "performance", "console", "history", "screen",
                        "devicePixelRatio", "scrollX", "scrollY", "onresize", "onload",
                        "onerror", "fetch", "dispatchEvent", "CustomEvent", "Image",
                        "Audio", "JSON", "Math", "Date", "Array", "Object", "Map", "Set",
                    ):
                        errors.append(f"{f['path']}: window.{name} used but never assigned — may be undefined at runtime")

        # --- Contract validation (toolkit spec §3) ---
        # If the plan declared PROVIDES/DEPENDS/WIRES contracts, cross-check
        # actual code + entry HTML against the declarations so contract
        # violations are caught here rather than at runtime.
        entry_html = entry["content"] if entry else ""
        contract_errors = validate_contracts(ctx.files, ctx.planned_files, entry_html)
        if contract_errors:
            errors.extend(contract_errors)
            log.info(
                "app_builder.contract_violations",
                count=len(contract_errors),
                sample=contract_errors[:3],
            )

        # --- Structural checks (toolkit §27 — moved from improve pass) ---
        # These were previously part of _pass_improve, but they're static
        # checks that belong here alongside lint/contracts. improve now
        # focuses solely on quality scoring.
        if entry:
            html_lc = entry["content"].lower()
            if "<html" not in html_lc:
                errors.append(f"{entry['path']}: missing <html> tag")
            if "<body" not in html_lc:
                errors.append(f"{entry['path']}: missing <body> tag")
            body_match = re.search(r"<body[^>]*>([\s\S]*?)</body>", entry["content"], re.IGNORECASE)
            if body_match and len(body_match.group(1).strip()) < 20:
                errors.append(f"{entry['path']}: <body> is essentially empty")
        for f in scripts:
            real_lines = [
                ln for ln in f["content"].split("\n")
                if ln.strip() and not ln.strip().startswith("//")
            ]
            content_lc = f["content"].lower()
            is_stub = (
                not real_lines
                or any(m in content_lc for m in _IMPROVE_STUB_MARKERS)
            )
            if is_stub:
                errors.append(f"{f['path']}: appears to be a stub/placeholder")

        if not errors:
            detail = "clean" + (f" ({auto_fixed} auto-fixed)" if auto_fixed else "")
            return PassResult(done=True, detail=detail)

        ctx.errors = errors
        log.info("app_builder.validate_errors", count=len(errors), errors=errors[:5])

        # --- Error preservation: include previous failed fix attempts ---
        from augmentum.tools.application_scaffolds import GRAMMAR_SEARCH_REPLACE
        messages = build_fix_prompt(ctx.files, errors,
                                    previous_attempts=ctx._validate_attempts or None,
                                    model_name=self._request_model)
        fix_tokens = max(getattr(self, '_max_tokens', 8192) // 2, 2048)
        response = await self._call_llm(messages, max_tokens=fix_tokens, model=self._request_model, grammar=GRAMMAR_SEARCH_REPLACE)

        patches_applied = self._apply_file_patches(ctx.files, response)

        if patches_applied > 0:
            return PassResult(done=False, detail=f"fixed {patches_applied} issues, re-validating")

        # Record failed attempt for next iteration
        ctx._validate_attempts.append(response)
        ctx.flag_quality_issue(
            f"Validation found {len(errors)} issue(s) that could not be auto-fixed.",
            errors=errors[:10],
        )
        return PassResult(done=True, detail=f"{len(errors)} validation issues need review")

    async def _pass_improve(self, ctx: PipelineContext) -> PassResult:
        # Structural checks (missing <html>, empty <body>, stub JS) moved
        # into _pass_validate as of toolkit §27 — validate is now the
        # home for all static checks. improve is pure quality scoring.

        # --- Judge call: score + actionable improvements ---
        # The score acts as a quality gate rather than passive metadata.
        # If the judge returns a low score, we feed its IMPROVEMENTS
        # bullets back through the fix loop for one more attempt. If the
        # score is still low after that, we ship anyway but flag the
        # artifact via ctx.quality_warnings so the user sees it.
        messages = build_judge_prompt(ctx.files, ctx.description)
        response = await self._call_llm(messages, max_tokens=2048, model=self._request_model)
        ctx._last_judge_response = response
        ctx.score = self._parse_score(response)

        # Iteration count the outer loop has recorded for this pass.
        # Starts at 1 during the first call (loop sets iterations[pass] = iteration + 1).
        iteration = ctx.iterations.get("improve", 1)

        if self._score_is_below_gate(ctx.score) and iteration < _IMPROVE_GATE_MAX_RETRIES:
            improvements = self._extract_judge_improvements(response)
            if improvements:
                from augmentum.tools.application_scaffolds import GRAMMAR_SEARCH_REPLACE
                # Error preservation: surface prior failed fix attempts so the
                # model doesn't re-issue the same patches. Matches the pattern
                # _pass_verify uses.
                messages = build_fix_prompt(
                    ctx.files, improvements,
                    previous_attempts=ctx._improve_attempts or None,
                    model_name=self._request_model,
                )
                fix_tokens = max(getattr(self, "_max_tokens", 8192) // 2, 2048)
                fix_response = await self._call_llm(
                    messages, max_tokens=fix_tokens,
                    model=self._request_model, grammar=GRAMMAR_SEARCH_REPLACE,
                )
                patches = self._apply_file_patches(ctx.files, fix_response)
                if patches > 0:
                    # Keep the last few attempts only — older ones stop
                    # informing the model and just eat context.
                    ctx._improve_attempts.append(fix_response)
                    if len(ctx._improve_attempts) > 3:
                        ctx._improve_attempts = ctx._improve_attempts[-3:]
                    log.info(
                        "app_builder.improve_low_score_retry",
                        score=ctx.score, patches=patches,
                        improvements=len(improvements),
                        prior_attempts=len(ctx._improve_attempts),
                    )
                    return PassResult(
                        done=False,
                        detail=f"Score {ctx.score}/10 below {_IMPROVE_GATE_THRESHOLD} — applied {patches} improvements",
                    )
                # Couldn't apply any patches — record the attempt (so the next
                # iteration sees it) and ship rather than loop forever.
                ctx._improve_attempts.append(fix_response)
                log.info(
                    "app_builder.improve_low_score_no_patches",
                    score=ctx.score, improvements=len(improvements),
                )

        # Shipping with a low score — record a warning so the user sees it.
        if self._score_is_below_gate(ctx.score):
            ctx.flag_quality_issue(
                f"Build quality score {ctx.score}/10 fell below the "
                f"{_IMPROVE_GATE_THRESHOLD}/10 target after improvement attempts. "
                "Consider regenerating or iterating on the app.",
                status="warning",
            )
            log.warning("app_builder.improve_low_score_ship", score=ctx.score)

        return PassResult(done=True, detail=f"Score: {ctx.score}/10")

    async def _pass_polish(self, ctx: PipelineContext) -> PassResult:
        """Deterministic auto-enhancement pass — no LLM cost, <10ms.

        Applies programmatic improvements that consistently raise output quality:
        1. CSS variable extraction (inline colors → :root custom properties)
        2. Accessibility enhancement (alt text, aria labels, lang, viewport)
        3. Semantic HTML upgrade (div.nav → <nav>, div.header → <header>)
        4. Security hardening (flag innerHTML with variables, eval usage)
        5. Dead CSS cleanup (classes defined but never used)
        """
        enhancements = 0

        entry = next((f for f in ctx.files if f["role"] == "entry"), None)
        styles = [f for f in ctx.files if f["role"] == "style"]
        scripts = [f for f in ctx.files if f.get("role") in ("script", "module")]
        all_html = entry["content"] if entry else ""
        all_js = "\n".join(f["content"] for f in scripts)
        all_css = "\n".join(f["content"] for f in styles)

        # =================================================================
        # 1. CSS VARIABLE EXTRACTION
        # =================================================================
        # Find inline hex colors in CSS that aren't already custom properties,
        # extract them to :root variables for a cleaner design system.
        if styles:
            for f in styles:
                content = f["content"]
                # Skip if already has a good :root block with 4+ custom properties
                existing_vars = re.findall(r'--[\w-]+\s*:', content)
                if len(existing_vars) >= 4:
                    continue

                # Extract all hex colors used more than once
                hex_colors = re.findall(r'#(?:[0-9a-fA-F]{3,4}){1,2}\b', content)
                color_counts: dict[str, int] = {}
                for c in hex_colors:
                    normalized = c.lower()
                    color_counts[normalized] = color_counts.get(normalized, 0) + 1

                # Only extract colors used 2+ times (worth making a variable)
                repeated = {c: n for c, n in color_counts.items() if n >= 2}
                if not repeated:
                    continue

                # Generate semantic variable names from ALL usage contexts
                var_map: dict[str, str] = {}
                used_names: set[str] = set()

                for color in repeated:
                    # Collect ALL contexts where this color appears
                    occurrences = [m.start() for m in re.finditer(re.escape(color), content, re.IGNORECASE)]
                    role_votes: dict[str, int] = {}
                    for pos in occurrences:
                        preceding = content[max(0, pos - 25):pos].lower()
                        if "background" in preceding or "bg" in preceding:
                            role_votes["surface"] = role_votes.get("surface", 0) + 1
                        elif "border" in preceding:
                            role_votes["border"] = role_votes.get("border", 0) + 1
                        elif "shadow" in preceding or "box-shadow" in preceding:
                            role_votes["shadow"] = role_votes.get("shadow", 0) + 1
                        elif "color" in preceding and "background" not in preceding:
                            role_votes["text"] = role_votes.get("text", 0) + 1
                        elif "hover" in preceding or ":hover" in preceding:
                            role_votes["hover"] = role_votes.get("hover", 0) + 1
                        elif "accent" in preceding or "primary" in preceding:
                            role_votes["accent"] = role_votes.get("accent", 0) + 1

                    # Pick the role — colors used in multiple contexts are likely accents
                    if role_votes:
                        if len(role_votes) >= 2 and ("surface" in role_votes or "hover" in role_votes):
                            role = "accent"  # Multi-context = accent/primary
                        else:
                            role = max(role_votes, key=role_votes.get)
                    else:
                        # Guess from the color value — dark colors are likely backgrounds
                        try:
                            hex_clean = color.lstrip('#')
                            if len(hex_clean) == 3:
                                hex_clean = ''.join(c*2 for c in hex_clean)
                            r_val = int(hex_clean[:2], 16)
                            g_val = int(hex_clean[2:4], 16)
                            b_val = int(hex_clean[4:6], 16)
                            brightness = (r_val * 299 + g_val * 587 + b_val * 114) / 1000
                            role = "surface" if brightness < 50 else "text" if brightness > 200 else "accent"
                        except (ValueError, IndexError):
                            role = "accent"

                    # Ensure unique name
                    base_name = f"--color-{role}"
                    name = base_name
                    suffix = 2
                    while name in used_names:
                        name = f"{base_name}-{suffix}"
                        suffix += 1
                    used_names.add(name)
                    var_map[color] = name

                if var_map:
                    # Build :root block
                    root_vars = "\n".join(f"  {name}: {color};" for color, name in var_map.items())
                    root_block = f":root {{\n{root_vars}\n}}\n"

                    # Replace inline colors with var() references
                    for color, name in var_map.items():
                        content = re.sub(re.escape(color), f"var({name})", content, flags=re.IGNORECASE)

                    # Prepend :root if not already at top
                    if ":root" not in content:
                        content = root_block + "\n" + content
                    else:
                        # Inject into existing :root
                        content = re.sub(
                            r'(:root\s*\{)',
                            r'\1\n' + root_vars + "\n",
                            content, count=1,
                        )

                    f["content"] = content
                    enhancements += len(var_map)
                    log.info("polish.css_variables", file=f["path"], extracted=len(var_map))

        # =================================================================
        # 2. ACCESSIBILITY ENHANCEMENT
        # =================================================================
        if entry:
            html = entry["content"]
            original = html

            # Add lang="en" if missing
            if "<html" in html and 'lang=' not in html.split(">")[0]:
                html = html.replace("<html", '<html lang="en"', 1)
                enhancements += 1

            # Images without alt text
            html = re.sub(
                r'<img(?![^>]*\balt=)([^>]*?)(/?>)',
                r'<img alt=""\1\2',
                html,
            )
            if html != original:
                enhancements += 1

            # Buttons with no accessible text (empty or icon-only)
            for m in re.finditer(r'<button([^>]*)>(.*?)</button>', html, re.DOTALL):
                attrs, inner = m.group(1), m.group(2).strip()
                # If button has no text and no aria-label
                text_content = re.sub(r'<[^>]+>', '', inner).strip()
                if not text_content and 'aria-label' not in attrs:
                    # Try to derive label from id or class
                    id_match = re.search(r'id=["\'](\w[\w-]*)["\']', attrs)
                    label = id_match.group(1).replace("-", " ").replace("_", " ") if id_match else "button"
                    old = f"<button{attrs}>"
                    new = f'<button{attrs} aria-label="{label}">'
                    html = html.replace(old, new, 1)
                    enhancements += 1

            # Ensure all form inputs have associated labels
            for m in re.finditer(r'<input([^>]*)>', html):
                attrs = m.group(1)
                input_id = re.search(r'id=["\'](\w[\w-]*)["\']', attrs)
                if input_id and f'for="{input_id.group(1)}"' not in html and f"for='{input_id.group(1)}'" not in html:
                    # Check if there's a label nearby (within 200 chars before)
                    before = html[max(0, m.start() - 200):m.start()]
                    if '<label' not in before:
                        # Add aria-label if no visible label exists
                        if 'aria-label' not in attrs and 'placeholder' not in attrs:
                            name = input_id.group(1).replace("-", " ").replace("_", " ").title()
                            old = m.group(0)
                            html = html.replace(old, old.replace("<input", f'<input aria-label="{name}"'), 1)
                            enhancements += 1

            if html != original:
                entry["content"] = html

        # =================================================================
        # 3. SEMANTIC HTML UPGRADE
        # =================================================================
        if entry:
            html = entry["content"]
            semantic_map = [
                (r'<div\s+class=["\'](?:[^"\']*\s)?nav(?:bar|igation)?(?:\s[^"\']*)?["\']', "<nav", "</div>", "</nav>"),
                (r'<div\s+class=["\'](?:[^"\']*\s)?header(?:\s[^"\']*)?["\']', "<header", "</div>", "</header>"),
                (r'<div\s+class=["\'](?:[^"\']*\s)?footer(?:\s[^"\']*)?["\']', "<footer", "</div>", "</footer>"),
            ]
            for pattern, new_open, old_close, new_close in semantic_map:
                m = re.search(pattern, html)
                if m:
                    # Simple replacement — only for top-level divs
                    old_tag = html[m.start():m.end()]
                    new_tag = old_tag.replace("<div", new_open, 1)
                    html = html.replace(old_tag, new_tag, 1)
                    enhancements += 1

            # Add <main> if missing and there's a primary content area.
            # The previous version did `html.replace("</div>", "</main>", 1)`
            # which swapped the FIRST close it saw — typically a header's
            # close, not the one matching <div id="app">. Do a proper
            # depth scan to find the correct matching close.
            if "<main" not in html and '<div id="app"' in html:
                open_match = re.search(r'<div\s+id="app"[^>]*>', html)
                if open_match:
                    start = open_match.end()
                    depth = 1
                    close_pos = -1
                    for m in re.finditer(r'<(/?)div\b[^>]*>', html[start:]):
                        depth += -1 if m.group(1) else 1
                        if depth == 0:
                            close_pos = start + m.start()
                            close_end = start + m.end()
                            break
                    if close_pos >= 0:
                        html = (
                            html[:open_match.start()]
                            + '<main id="app"' + open_match.group(0)[len('<div id="app"'):]
                            + html[open_match.end():close_pos]
                            + '</main>'
                            + html[close_end:]
                        )
                        enhancements += 1

            entry["content"] = html

        # =================================================================
        # 4. SECURITY HARDENING (detection + auto-fix where safe)
        # =================================================================
        for f in scripts:
            content = f["content"]
            original = content

            # Flag eval() usage
            if "eval(" in content and "// SAFE:" not in content:
                content = content.replace("eval(", "/* SECURITY: avoid eval */ eval(")
                enhancements += 1

            # Flag document.write()
            if "document.write(" in content:
                content = content.replace("document.write(", "/* SECURITY: avoid document.write */ document.write(")
                enhancements += 1

            if content != original:
                f["content"] = content

        # =================================================================
        # 5. DEAD CSS CLEANUP
        # =================================================================
        if styles and entry:
            all_content = entry["content"] + "\n" + all_js
            for f in styles:
                content = f["content"]
                # Find CSS class definitions
                css_classes = set(re.findall(r'\.(\w[\w-]*)\s*[{,:]', content))
                # Check which are used in HTML or JS
                unused = []
                for cls in css_classes:
                    if cls not in all_content and f".{cls}" not in all_content:
                        unused.append(cls)
                # Don't remove utility classes that might be toggled dynamically
                keep = {"hidden", "active", "visible", "open", "closed", "disabled", "selected",
                        "loading", "error", "success", "collapsed", "expanded", "dark", "light"}
                actually_unused = [c for c in unused if c not in keep]

                if actually_unused and len(actually_unused) < len(css_classes) * 0.3:
                    # Only clean up if <30% is unused (avoid nuking the whole file)
                    for cls in actually_unused:
                        # Remove simple single-selector rules: .className { ... }
                        content = re.sub(
                            r'\.' + re.escape(cls) + r'\s*\{[^}]*\}\s*\n?',
                            '', content,
                        )
                    f["content"] = content
                    enhancements += len(actually_unused)
                    log.info("polish.dead_css", file=f["path"], removed=len(actually_unused))

        # =================================================================
        # 6. MICRO-INTERACTIONS + VISUAL POLISH (CSS injection)
        # =================================================================
        # These small CSS additions make the difference between
        # "AI-generated" and "professionally built" — all deterministic.
        if styles:
            polish_css = []
            is_game = ctx.scaffold_id == "game"

            # Check what's already present to avoid duplicating
            combined_css = "\n".join(f["content"] for f in styles)

            # 6a. Button micro-interactions (ripple feel + active state)
            # Skip for games — game buttons are in overlay menus, handled by the game reference
            if not is_game and ":active" not in combined_css and "button" in all_html.lower():
                polish_css.append(
                    "/* Micro-interactions */\n"
                    "button, .btn { position: relative; overflow: hidden; }\n"
                    "button:active, .btn:active { transform: scale(0.97); }\n"
                    "button::after, .btn::after { content: ''; position: absolute; inset: 0; "
                    "background: radial-gradient(circle, rgba(255,255,255,0.25) 10%, transparent 70%); "
                    "opacity: 0; transition: opacity 0.3s; }\n"
                    "button:active::after, .btn:active::after { opacity: 1; transition: 0s; }"
                )
                enhancements += 1

            # 6b. Smooth input focus transitions (skip for games — no form inputs)
            if not is_game and "input:focus" not in combined_css and "input" in all_html.lower():
                polish_css.append(
                    "input, select, textarea { transition: border-color 0.2s, box-shadow 0.2s; }"
                )
                enhancements += 1

            # 6c. Staggered entrance animation for lists and grids (skip for games)
            if not is_game and ("@keyframes" not in combined_css or "fadeIn" not in combined_css):
                polish_css.append(
                    "/* Entrance animations */\n"
                    "@keyframes fadeSlideIn { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }\n"
                    ".card, .item, .stat-card, .chart-card, [class*='col'] > * { "
                    "animation: fadeSlideIn 0.4s ease-out both; }\n"
                    ".card:nth-child(2), .item:nth-child(2) { animation-delay: 0.08s; }\n"
                    ".card:nth-child(3), .item:nth-child(3) { animation-delay: 0.16s; }\n"
                    ".card:nth-child(4), .item:nth-child(4) { animation-delay: 0.24s; }\n"
                    ".card:nth-child(5), .item:nth-child(5) { animation-delay: 0.32s; }"
                )
                enhancements += 1

            # 6d. Smooth scroll + scroll margin for anchors
            if "scroll-behavior" not in combined_css:
                polish_css.append(
                    "html { scroll-behavior: smooth; }\n"
                    "[id] { scroll-margin-top: 2rem; }"
                )
                enhancements += 1

            # 6e. Custom scrollbar for dark themes
            is_dark = any(v in combined_css for v in ["#0a0", "#0c1", "#0f1", "#111", "#000", "#0d0", "#151", "dark"])
            if is_dark and "::-webkit-scrollbar" not in combined_css:
                polish_css.append(
                    "/* Dark scrollbar */\n"
                    "::-webkit-scrollbar { width: 8px; }\n"
                    "::-webkit-scrollbar-track { background: transparent; }\n"
                    "::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 4px; }\n"
                    "::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }"
                )
                enhancements += 1

            # 6f. Selection color matching theme
            if "::selection" not in combined_css:
                # Try to pick accent color from existing CSS vars
                accent_match = re.search(r'--accent\s*:\s*([^;]+)', combined_css)
                if accent_match:
                    accent = accent_match.group(1).strip()
                    polish_css.append(f"::selection {{ background: {accent}; color: white; }}")
                    enhancements += 1

            # 6g. Reduced motion respect (accessibility)
            if "prefers-reduced-motion" not in combined_css:
                polish_css.append(
                    "@media (prefers-reduced-motion: reduce) {\n"
                    "  *, *::before, *::after { animation-duration: 0.01ms !important; "
                    "animation-delay: 0ms !important; transition-duration: 0.01ms !important; }\n"
                    "}"
                )
                enhancements += 1

            # Inject all polish CSS at the end of the LAST style file
            if polish_css:
                styles[-1]["content"] += "\n\n" + "\n\n".join(polish_css) + "\n"

        # =================================================================
        # 7. FAVICON GENERATION (SVG from project name + accent color)
        # =================================================================
        # Skip when there's no entry HTML at all — the previous operator
        # precedence ("entry and X or Y") crashed on entry=None because
        # the right branch would subscript None["content"].
        if entry and ("<link" not in entry["content"] or "favicon" not in entry["content"]):
            # Generate a simple SVG favicon from the first letter of the project name
            first_letter = ctx.project_name[0].upper() if ctx.project_name else "A"
            # Find accent color from CSS
            accent_color = "#6366f1"  # default indigo
            if styles:
                combined_css = "\n".join(f["content"] for f in styles)
                accent_match = re.search(r'--accent\s*:\s*(#[0-9a-fA-F]{3,8})', combined_css)
                if accent_match:
                    accent_color = accent_match.group(1)

            svg_favicon = (
                f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
                f'<rect width="32" height="32" rx="8" fill="{accent_color}"/>'
                f'<text x="16" y="22" text-anchor="middle" fill="white" '
                f'font-family="system-ui,sans-serif" font-weight="700" font-size="18">{first_letter}</text>'
                f'</svg>'
            )
            import base64
            favicon_b64 = base64.b64encode(svg_favicon.encode()).decode()
            favicon_link = f'<link rel="icon" href="data:image/svg+xml;base64,{favicon_b64}">'

            html = entry["content"]
            if "</head>" in html:
                html = html.replace("</head>", f"  {favicon_link}\n</head>")
            elif "<head>" in html:
                html = html.replace("<head>", f"<head>\n  {favicon_link}")
            entry["content"] = html
            enhancements += 1

        # =================================================================
        # 8. STRIP LLM SELF-DOUBT COMMENTS
        # =================================================================
        # Scrub internal-monologue comments the LLM left in the code
        # ("Actually, we'll just...", "Let's assume..."). See
        # strip_selfdoubt_comments. Only touches whole-line // comments in
        # script files so trailing inline comments aren't disturbed.
        for f in scripts:
            cleaned, removed = strip_selfdoubt_comments(f["content"])
            if removed:
                f["content"] = cleaned
                enhancements += removed
                log.info(
                    "polish.selfdoubt_stripped",
                    file=f["path"],
                    comments=removed,
                )

        detail = f"{enhancements} enhancements" if enhancements else "no changes needed"
        if enhancements:
            log.info("polish.applied", total=enhancements)

        return PassResult(done=True, detail=detail)

    async def _pass_verify(self, ctx: PipelineContext) -> PassResult:
        """Runtime verification — execute assembled JS to catch real errors.

        Uses quickjs (embedded JS engine, <1ms execution) to actually run
        the assembled code against a DOM mock. Catches the same errors the
        browser would: SyntaxError, ReferenceError, TypeError, etc.

        Falls back to regex-based static analysis if quickjs is not installed.

        Pipeline: errors found → targeted LLM fix → re-verify (max 2 iterations).
        """


        assembled = self._assemble(ctx.files)

        # Extract JS blocks and HTML IDs
        js_blocks = re.findall(r"<script[^>]*>([\s\S]*?)</script>", assembled, re.IGNORECASE)
        all_js = "\n".join(js_blocks)

        if not all_js.strip():
            return PassResult(done=True, detail="no JS to verify")

        errors = self._execute_js_verify(all_js, assembled)

        # --- Browser verify (toolkit §26): augment quickjs with a real
        # headless chromium run when the setting is enabled. Catches
        # layout / CSS / real-browser-API bugs the mock DOM misses.
        # Errors from chromium are deduped against the quickjs output.
        settings = self._get_settings() if self._get_settings else {}
        if settings.get("app_builder_use_browser_verify", False):
            browser_errors = await self._run_browser_verify(assembled, ctx)
            if browser_errors:
                # Keep quickjs results first (they're usually clearer) and
                # append browser-only findings.
                existing = set(errors)
                for e in browser_errors:
                    if e not in existing:
                        errors.append(e)
                        existing.add(e)
                log.info(
                    "app_builder.browser_verify_errors",
                    count=len(browser_errors),
                    sample=browser_errors[:3],
                )

        # --- Intent verification: does the code implement what was asked for? ---
        intent_issues = self.verify_intent(ctx.description, ctx.files)
        if intent_issues:
            errors.extend(intent_issues)
            log.info("app_builder.intent_gaps", count=len(intent_issues), gaps=intent_issues[:3])

        if not errors:
            return PassResult(done=True, detail="runtime verified clean")

        # Deduplicate
        errors = list(dict.fromkeys(errors))
        ctx.errors = errors
        log.info("app_builder.verify_errors", count=len(errors), errors=errors[:5])

        # --- Snapshot before fix (rollback if fix makes things worse) ---
        snapshot = [{"path": f["path"], "content": f["content"]} for f in ctx.files]

        # --- Error preservation (Manus pattern): include previous failed attempts ---
        from augmentum.tools.application_scaffolds import GRAMMAR_SEARCH_REPLACE
        messages = build_fix_prompt(ctx.files, errors,
                                    previous_attempts=ctx._verify_attempts or None,
                                    model_name=self._request_model)
        fix_tokens = max(getattr(self, '_max_tokens', 8192) // 2, 2048)
        response = await self._call_llm(messages, max_tokens=fix_tokens, model=self._request_model, grammar=GRAMMAR_SEARCH_REPLACE)
        patches = self._apply_file_patches(ctx.files, response)

        if patches > 0:
            # Re-verify the fix before committing to it
            post_assembled = self._assemble(ctx.files)
            post_js = "\n".join(re.findall(r"<script[^>]*>([\s\S]*?)</script>", post_assembled, re.IGNORECASE))
            post_errors = self._execute_js_verify(post_js, post_assembled) if post_js.strip() else []

            # Check for regression: more errors, OR new error types that weren't
            # in the original set (the fix broke something different).
            orig_set = set(errors)
            new_errors = [e for e in post_errors if e not in orig_set]
            regressed = len(post_errors) > len(errors) or len(new_errors) > 0

            if regressed:
                # Fix made things worse or broke something new — rollback
                for orig in snapshot:
                    target = next((f for f in ctx.files if f["path"] == orig["path"]), None)
                    if target:
                        target["content"] = orig["content"]
                # Record this failed attempt so the next iteration avoids it
                ctx._verify_attempts.append(response)
                log.warning("app_builder.verify_rollback",
                            before=len(errors), after=len(post_errors),
                            new_errors=new_errors[:3],
                            reason="fix introduced new errors")
                ctx.flag_quality_issue(
                    f"Runtime verification found {len(errors)} issue(s); an attempted fix regressed and was rolled back.",
                    errors=errors[:10],
                )
                return PassResult(done=True, detail=f"{len(errors)} runtime issues need review (fix rolled back)")

            if not post_errors:
                return PassResult(done=False, detail=f"fixed {patches} runtime issues, re-verifying")

            # Fix improved but didn't fully resolve — record attempt, continue loop
            ctx._verify_attempts.append(response)
            ctx.errors = post_errors
            return PassResult(done=False, detail=f"fixed {patches} issues, {len(post_errors)} remain")

        # No patches applied — record failed attempt, deliver anyway
        ctx._verify_attempts.append(response)
        ctx.flag_quality_issue(
            f"Runtime verification found {len(errors)} issue(s) that could not be auto-fixed.",
            errors=errors[:10],
        )
        return PassResult(done=True, detail=f"{len(errors)} runtime issues need review")

    @staticmethod
    def _parse_html_dom(assembled_html: str) -> dict:
        """Parse assembled HTML into a DOM descriptor for the quickjs mock.

        Uses stdlib html.parser — no dependencies. Extracts:
        - ids: set of all element IDs
        - classes: set of all CSS class names
        - tags: set of all tag names (lowercased)
        - elements: list of {tag, id, classes} for querySelector matching
        - form_fields: list of {id, name, type} for form validation

        This gives the quickjs DOM mock enough information to return null
        for querySelector('.nonexistent') while returning a mock element for
        querySelector('.real-class').
        """
        from html.parser import HTMLParser


        # Strip scripts so we only parse HTML structure
        html_only = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", assembled_html, flags=re.IGNORECASE)

        dom = {
            "ids": set(),
            "classes": set(),
            "tags": set(),
            "elements": [],
            "form_fields": [],
        }

        class _DOMParser(HTMLParser):
            def handle_starttag(self_, tag: str, attrs: list) -> None:
                tag_lower = tag.lower()
                dom["tags"].add(tag_lower)
                attr_dict = dict(attrs)

                elem_id = attr_dict.get("id", "")
                elem_classes = attr_dict.get("class", "").split()

                if elem_id:
                    dom["ids"].add(elem_id)
                for cls in elem_classes:
                    dom["classes"].add(cls)

                dom["elements"].append({
                    "tag": tag_lower,
                    "id": elem_id,
                    "classes": elem_classes,
                })

                # Track form fields
                if tag_lower in ("input", "select", "textarea"):
                    dom["form_fields"].append({
                        "id": elem_id,
                        "name": attr_dict.get("name", ""),
                        "type": attr_dict.get("type", "text"),
                    })

        parser = _DOMParser()
        try:
            parser.feed(html_only)
        except Exception as exc:
            # Malformed HTML — leave dom partially populated rather than
            # abort the pipeline.
            log.debug("artifact_app_dom_parse_failed", error=str(exc))

        return dom

    async def _attempt_batch_generate(
        self,
        ctx: PipelineContext,
        progress_cb: Callable | None,
    ) -> bool:
        """Try to generate every planned file in a single LLM call.

        Returns True iff ALL planned files came back parseable and got
        stored — caller can then short-circuit the sequential path.
        Returns False on partial output, parse failure, or LLM error;
        any files that DID parse are still stored so the sequential
        fallback has less work to do.

        The batch prompt lists every file with its contract so the
        model holds the full mental model in one forward pass — this
        is where the cross-file wiring benefits come from.
        """
        # Build a bulleted list of files + contracts so the model can
        # address each one in its response.
        file_blocks: list[str] = []
        for f in ctx.planned_files:
            contract_lines: list[str] = []
            if f.get("provides"):
                contract_lines.append(f"PROVIDES: {', '.join(f['provides'])}")
            if f.get("depends"):
                contract_lines.append(f"DEPENDS: {', '.join(f['depends'])}")
            if f.get("wires"):
                contract_lines.append(f"WIRES: {', '.join(f['wires'])}")
            contract_block = (
                "\n    " + "\n    ".join(contract_lines) if contract_lines else ""
            )
            file_blocks.append(
                f"- {f['path']} ({f.get('role', 'script')}): "
                f"{f.get('description', '')}{contract_block}"
            )
        files_listing = "\n".join(file_blocks)

        from augmentum.tools.application_scaffolds import adapt_prompt_for_tier
        base_system = (
            "You are generating a complete small web application in ONE response.\n"
            "Produce every file below as a separate fenced code block whose "
            "label is the exact filename:\n\n"
            "  ```index.html\n"
            "  <full file contents>\n"
            "  ```\n\n"
            "Rules:\n"
            "- Output every listed file — do not skip any.\n"
            "- Order: entry HTML first, then CSS, then JS modules in dependency order.\n"
            "- All JS files share ONE global scope when assembled; cross-file symbols "
            "  go on window.X. Never redeclare a window.X another file already provides.\n"
            "- Honour each file's contract exactly: PROVIDES must be defined, DEPENDS "
            "  must be consumed only via the project's exports, WIRES must attach to "
            "  elements that exist in the entry HTML.\n"
            "- Do NOT emit <script src='…'> or <link href='…'> referencing other project "
            "  files — they're assembled automatically.\n"
            "End with __PASS_COMPLETE__.\n\n"
            f"{ctx.working_doc}"
        )
        system_prompt = adapt_prompt_for_tier(base_system, ctx.model_tier)
        user_prompt = (
            f"Project: {ctx.description}\n\n"
            f"Generate every file below in a single response:\n\n"
            f"{files_listing}"
        )

        gen_tokens = getattr(self, "_max_tokens", 8192)
        try:
            if progress_cb:
                await self._emit(progress_cb, ctx, "running",
                                 f"batch-generating {len(ctx.planned_files)} files")
            response = await self._call_llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=gen_tokens,
                model=self._request_model,
            )
            ctx._total_llm_calls += 1
            usage = getattr(self._call_llm, '_last_usage', None)
            if usage:
                ctx._total_tokens += usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        except Exception as exc:
            log.warning("app_builder.batch_generate_llm_failed", error=str(exc))
            return False

        parsed = self._parse_generated_files(response)
        parsed_by_path = {p["path"]: p["content"] for p in parsed if p.get("path")}

        # Success = every planned file parsed back with non-empty content.
        planned_paths = [f["path"] for f in ctx.planned_files]
        missing = [p for p in planned_paths if not parsed_by_path.get(p)]
        if missing:
            log.info(
                "app_builder.batch_generate_incomplete",
                planned=len(planned_paths),
                parsed=len(planned_paths) - len(missing),
                missing=missing,
            )
            # Store whatever DID parse so the sequential path has less to do.
            self._store_batch_files(ctx, parsed_by_path, planned_paths)
            if progress_cb:
                await self._emit(
                    progress_cb,
                    ctx,
                    "running",
                    f"batch output missed {len(missing)} file(s); continuing one file at a time",
                )
            return False

        self._store_batch_files(ctx, parsed_by_path, planned_paths)
        log.info("app_builder.batch_generate_ok", count=len(planned_paths))
        return True

    def _store_batch_files(
        self,
        ctx: PipelineContext,
        parsed_by_path: dict[str, str],
        planned_paths: list[str],
    ) -> None:
        """Intercept → store the files that came back from a batch call,
        refreshing the project map after each one so later entries see
        earlier entries' exports (mirrors the per-file loop in the
        sequential generate path)."""
        for path in planned_paths:
            content = parsed_by_path.get(path)
            if not content:
                continue
            planned = next(
                (pf for pf in ctx.planned_files if pf["path"] == path), {},
            )
            file_info = {"path": path, "content": content}
            file_info["content"] = self._intercept_generated_code(file_info, ctx)
            ctx.generated_files[path] = file_info["content"]
            existing = next((ef for ef in ctx.files if ef["path"] == path), None)
            if existing:
                existing["content"] = file_info["content"]
            else:
                ctx.files.append({
                    "path": path,
                    "lang": planned.get("lang", self._infer_lang(path)),
                    "content": file_info["content"],
                    "role": planned.get("role", self._infer_role(path)),
                })
            ctx.working_doc = ctx.working_doc.replace(
                f"- [ ] {path}", f"- [x] {path}",
            )
            ctx.working_doc = self._update_project_map(ctx)

    async def _run_browser_verify(
        self,
        assembled_html: str,
        ctx: PipelineContext | None = None,
    ) -> list[str]:
        """Run ``assembled_html`` through a real headless chromium via CDP
        and return runtime errors.

        When ``ctx`` is supplied, derives a smoke-click sequence from
        the plan's WIRES + the description's intent features so the
        browser exercises real interaction paths (Start → Pause →
        Start → Reset and similar). Errors raised during clicks land
        as ``Runtime.exceptionThrown`` events the fix loop can see.

        Failures inside the CDP client (chromium not installed, browser
        didn't start, websocket hiccup) are logged and absorbed — the
        App Builder must continue to work on hosts without a browser,
        falling back to the quickjs signal alone.
        """
        try:
            from augmentum.tools.application_cdp import (
                BrowserVerifier,
                ChromiumNotAvailable,
                derive_smoke_sequence,
                filter_browser_errors,
            )
        except ImportError as exc:  # pragma: no cover — module is in-tree
            log.warning("app_builder.browser_verify_import_failed", error=str(exc))
            return []

        smoke: list[str] = []
        if ctx is not None:
            try:
                smoke = derive_smoke_sequence(ctx.planned_files, ctx.description)
            except Exception as exc:
                log.info("app_builder.smoke_derive_failed", error=str(exc))

        try:
            async with BrowserVerifier() as bv:
                result = await bv.verify_html(
                    assembled_html,
                    click_sequence=smoke if smoke else None,
                )
        except ChromiumNotAvailable as exc:
            # Benign: the image doesn't have chromium installed yet.
            log.info("app_builder.browser_verify_skipped", reason=str(exc))
            return []
        except Exception as exc:
            log.warning("app_builder.browser_verify_failed", error=str(exc), exc_info=True)
            return []

        if smoke:
            log.info(
                "app_builder.browser_smoke_sequence",
                clicks=len(smoke), load_ms=result.load_ms,
                errors=len(result.errors),
            )
        return filter_browser_errors(result.errors)

    def _execute_js_verify(self, all_js: str, assembled_html: str) -> list[str]:
        """Execute assembled JS in quickjs with a DOM mock to catch real runtime errors.

        Falls back to regex-based static analysis if quickjs is unavailable.
        Returns a list of error strings suitable for the LLM fix prompt.
        """
        dom = self._parse_html_dom(assembled_html)

        try:
            return self._quickjs_verify(all_js, dom)
        except ImportError:
            log.info("app_builder.verify_fallback", reason="quickjs not installed, using regex")
            return self._regex_verify(all_js, dom["ids"])

    def _quickjs_verify(self, all_js: str, dom: dict) -> list[str]:
        """Run JS in quickjs engine with DOM mock. Catches real runtime errors."""
        import quickjs

        errors: list[str] = []
        ctx = quickjs.Context()

        # Sandbox: 32MB memory, 5 second timeout
        ctx.set_memory_limit(32 * 1024 * 1024)
        ctx.set_time_limit(5)

        # Build DOM mock from parsed HTML structure
        html_ids = dom.get("ids", set())
        html_classes = dom.get("classes", set())
        html_tags = dom.get("tags", set())
        id_list = ", ".join(f'"{eid}"' for eid in html_ids)
        class_list = ", ".join(f'"{cls}"' for cls in html_classes)
        tag_list = ", ".join(f'"{tag}"' for tag in html_tags)
        dom_mock = rf"""
        // --- DOM Mock (generated from assembled HTML) ---
        var _knownIds = new Set([{id_list}]);
        var _knownClasses = new Set([{class_list}]);
        var _knownTags = new Set([{tag_list}]);

        // --- Element registry: tracks state across the application ---
        var _elementRegistry = {{}};  // id → element (for state tracking)
        var _eventLog = [];           // tracks all registered listeners
        var _domMutations = [];       // tracks all DOM changes

        var _mockElement = function(tag, id) {{
            // Return existing element if already created (preserves state)
            if (id && _elementRegistry[id]) return _elementRegistry[id];

            var el = {{
                tagName: tag || 'DIV',
                id: id || '',
                textContent: '',
                innerHTML: '',
                innerText: '',
                value: '',
                checked: false,
                disabled: false,
                hidden: false,
                _listeners: {{}},
                style: new Proxy({{}}, {{ get: function() {{ return ''; }}, set: function() {{ return true; }} }}),
                classList: {{
                    _classes: new Set(),
                    add: function() {{ for(var i=0;i<arguments.length;i++) this._classes.add(arguments[i]); }},
                    remove: function() {{ for(var i=0;i<arguments.length;i++) this._classes.delete(arguments[i]); }},
                    toggle: function(c) {{ if(this._classes.has(c)) this._classes.delete(c); else this._classes.add(c); return this._classes.has(c); }},
                    contains: function(c) {{ return this._classes.has(c); }}
                }},
                dataset: {{}},
                children: [],
                parentNode: null,
                appendChild: function(c) {{ this.children.push(c); c.parentNode = this; return c; }},
                removeChild: function(c) {{ var i=this.children.indexOf(c); if(i>=0) this.children.splice(i,1); return c; }},
                insertBefore: function(c) {{ this.children.push(c); return c; }},
                replaceChild: function(n, o) {{ return o; }},
                addEventListener: function(type, handler) {{
                    if (!this._listeners[type]) this._listeners[type] = [];
                    this._listeners[type].push(handler);
                    _eventLog.push({{id: this.id, type: type}});
                    if (this.id) this._hasSubmitListener = this._hasSubmitListener || type === 'submit';
                }},
                removeEventListener: function() {{}},
                setAttribute: function(k, v) {{ this[k] = v; }},
                getAttribute: function(k) {{ return this[k] || null; }},
                querySelector: function() {{ return null; }},
                querySelectorAll: function() {{ return []; }},
                getBoundingClientRect: function() {{ return {{x:0,y:0,width:100,height:100,top:0,left:0,right:100,bottom:100}}; }},
                // Simulate click — fires registered click handlers
                click: function() {{ if(this._listeners['click']) this._listeners['click'].forEach(function(fn){{ try{{fn({{preventDefault:function(){{}},target:el}})}}catch(e){{}} }}); }},
                focus: function(){{}}, blur: function(){{}},
                remove: function(){{}},
                cloneNode: function() {{ return _mockElement(); }},
                closest: function() {{ return null; }},
                matches: function() {{ return false; }},
                before: function(){{}}, after: function(){{}},
                replaceWith: function(){{}},
                offsetWidth: 100, offsetHeight: 100,
                clientWidth: 100, clientHeight: 100,
                scrollWidth: 100, scrollHeight: 100,
                scrollTop: 0, scrollHeight: 100,
                width: 800, height: 600,
                getContext: function(type) {{
                    if (typeof _canvasInitialized !== 'undefined') _canvasInitialized = true;
                    if (type === '2d') return {{
                        fillStyle: '', strokeStyle: '', lineWidth: 1, font: '',
                        globalAlpha: 1, shadowBlur: 0, shadowColor: '',
                        save: function(){{}}, restore: function(){{}},
                        fillRect: function(){{}}, strokeRect: function(){{}}, clearRect: function(){{}},
                        beginPath: function(){{}}, closePath: function(){{}}, moveTo: function(){{}},
                        lineTo: function(){{}}, arc: function(){{}}, fill: function(){{}}, stroke: function(){{}},
                        fillText: function(){{}}, measureText: function() {{ return {{width:0}}; }},
                        drawImage: function(){{}}, createLinearGradient: function() {{
                            return {{ addColorStop: function(){{}} }};
                        }},
                        createRadialGradient: function() {{ return {{ addColorStop: function(){{}} }}; }},
                        translate: function(){{}}, rotate: function(){{}}, scale: function(){{}},
                        setTransform: function(){{}}, resetTransform: function(){{}},
                        clip: function(){{}},
                    }};
                    return {{}};
                }},
            }};
            // Register element by ID for state tracking
            if (id) _elementRegistry[id] = el;
            return el;
        }};

        // querySelector that validates selectors against parsed HTML.
        // For compound selectors ("a b"), ALL parts must match.
        function _matchSelector(sel) {{
            if (!sel || typeof sel !== 'string') return false;
            sel = sel.trim();
            // Multi-part (descendant) selectors — ALL parts must match
            var parts = sel.split(/\s+/).filter(function(p) {{ return p && p !== '>' && p !== '+' && p !== '~'; }});
            if (parts.length > 1) {{
                for (var i = 0; i < parts.length; i++) {{
                    if (!_matchSingle(parts[i])) return false;
                }}
                return true;
            }}
            return _matchSingle(sel);
        }}

        function _matchSingle(sel) {{
            // #id
            if (sel.startsWith('#')) {{
                var id = sel.slice(1).split(/[.:[>+~]/, 1)[0];
                return _knownIds.has(id);
            }}
            // .class
            if (sel.startsWith('.')) {{
                var cls = sel.slice(1).split(/[.:[>+~#]/, 1)[0];
                return _knownClasses.has(cls);
            }}
            // tag (optionally with .class or #id suffix)
            var tag = sel.split(/[.:[>+~#]/, 1)[0].toLowerCase();
            if (!tag) return false;
            var tagExists = _knownTags.has(tag);
            // If selector is just a tag name, that's enough
            if (sel === tag) return tagExists;
            // tag.class — both must match
            if (sel.indexOf('.') > 0) {{
                var cls2 = sel.split('.')[1].split(/[:[>+~#]/, 1)[0];
                return tagExists && _knownClasses.has(cls2);
            }}
            // tag#id — both must match
            if (sel.indexOf('#') > 0) {{
                var id2 = sel.split('#')[1].split(/[.:[>+~]/, 1)[0];
                return tagExists && _knownIds.has(id2);
            }}
            return tagExists;
        }}

        var document = {{
            getElementById: function(id) {{
                if (_knownIds.has(id)) return _mockElement('DIV', id);
                return null;
            }},
            querySelector: function(sel) {{
                if (_matchSelector(sel)) return _mockElement();
                return null;
            }},
            querySelectorAll: function(sel) {{
                if (_matchSelector(sel)) return [_mockElement()];
                return [];
            }},
            createElement: function(tag) {{ return _mockElement(tag); }},
            createDocumentFragment: function() {{ return _mockElement('FRAGMENT'); }},
            createTextNode: function() {{ return {{ textContent: '' }}; }},
            addEventListener: function() {{}},
            removeEventListener: function() {{}},
            body: _mockElement('BODY'),
            head: _mockElement('HEAD'),
            documentElement: _mockElement('HTML'),
            cookie: '',
            title: '',
            readyState: 'complete',
        }};

        // Trigger DOMContentLoaded listeners immediately
        var _dcl_listeners = [];
        var _orig_addEventListener = document.addEventListener;
        document.addEventListener = function(type, fn) {{
            if (type === 'DOMContentLoaded') _dcl_listeners.push(fn);
        }};

        var window = globalThis;
        window.document = document;
        window.innerWidth = 1024;
        window.innerHeight = 768;
        window.devicePixelRatio = 1;
        window.scrollX = 0;
        window.scrollY = 0;
        window.addEventListener = function(type, fn) {{
            if (type === 'DOMContentLoaded') _dcl_listeners.push(fn);
        }};
        window.removeEventListener = function() {{}};
        // setTimeout/setInterval: collect callbacks, execute AFTER main script (not synchronously)
        var _deferredTimers = [];
        window.setTimeout = function(fn, ms) {{ if (typeof fn === 'function') _deferredTimers.push(fn); return _deferredTimers.length; }};
        window.setInterval = function() {{ return 1; }};
        window.clearTimeout = function() {{}};
        window.clearInterval = function() {{}};
        window.requestAnimationFrame = function(fn) {{ return 1; }};
        window.cancelAnimationFrame = function() {{}};
        window.getComputedStyle = function() {{ return {{}}; }};
        window.matchMedia = function() {{ return {{ matches: false, addEventListener: function(){{}} }}; }};
        window.scrollTo = function() {{}};
        window.open = function() {{ return null; }};
        window.close = function() {{}};
        window.alert = function() {{}};
        window.confirm = function() {{ return false; }};
        window.prompt = function() {{ return null; }};
        window.fetch = function() {{ return new Promise(function(r){{ r({{ ok: true, json: function(){{ return new Promise(function(r2){{ r2({{}}); }}); }} }}); }}); }};
        window.navigator = {{ userAgent: 'quickjs-verify', language: 'en', clipboard: {{ writeText: function(){{}} }} }};
        window.location = {{ href: 'about:blank', origin: '', pathname: '/', search: '', hash: '' }};
        window.history = {{ pushState: function(){{}}, replaceState: function(){{}}, back: function(){{}}, forward: function(){{}} }};
        window.screen = {{ width: 1024, height: 768 }};
        window.performance = {{ now: function() {{ return 0; }} }};

        var console = {{ log: function(){{}}, error: function(){{}}, warn: function(){{}}, info: function(){{}}, debug: function(){{}} }};
        var localStorage = {{ _d: {{}}, getItem: function(k){{ return this._d[k] || null; }}, setItem: function(k,v){{ this._d[k]=String(v); }}, removeItem: function(k){{ delete this._d[k]; }}, clear: function(){{ this._d={{}}; }} }};
        var sessionStorage = {{ _d: {{}}, getItem: function(k){{ return this._d[k] || null; }}, setItem: function(k,v){{ this._d[k]=String(v); }}, removeItem: function(k){{ delete this._d[k]; }}, clear: function(){{ this._d={{}}; }} }};

        // Canvas/Audio constructors
        var AudioContext = function() {{ this.createGain = function(){{ return {{gain:{{value:1}}, connect:function(){{}}}}; }}; this.createOscillator = function(){{ return {{connect:function(){{}},start:function(){{}},stop:function(){{}},frequency:{{value:440}}}}; }}; this.destination = {{}}; }};
        var OfflineAudioContext = AudioContext;
        """

        try:
            ctx.eval(dom_mock)
        except quickjs.JSException as e:
            log.warning("app_builder.verify_mock_error", error=str(e))
            return [f"VERIFY: DOM mock setup failed — {e}"]

        # --- Execute each script block individually with error collection ---
        # Instead of running all JS as one blob (where first error kills execution),
        # inject a try/catch error collector so ALL errors are captured in one pass.
        # This means the LLM sees every error at once and can fix them all together.
        ctx.eval("var _verifyErrors = [];")

        # Split assembled JS back into per-file blocks (separated by /* filename */ comments)

        script_blocks = re.split(r'/\*\s*[\w./]+\s*\*/', all_js)
        script_blocks = [b.strip() for b in script_blocks if b.strip()]

        if not script_blocks:
            script_blocks = [all_js]

        for i, block in enumerate(script_blocks):
            # Wrap each block in try/catch to collect errors without stopping
            wrapped = (
                "try {\n"
                + block + "\n"
                + "} catch(_e) { _verifyErrors.push(_e.constructor.name + ': ' + _e.message); }\n"
            )
            try:
                ctx.eval(wrapped)
            except quickjs.JSException as e:
                # Catch errors that even try/catch can't handle (SyntaxError in eval)
                error_str = str(e).split("\n")[0]
                errors.append(f"RUNTIME: {error_str}")

        # Fire DOMContentLoaded listeners — each in its own try/catch
        try:
            ctx.eval("""
            for (var i = 0; i < _dcl_listeners.length; i++) {
                try {
                    _dcl_listeners[i]();
                } catch(_e) {
                    _verifyErrors.push('DOMContentLoaded: ' + _e.constructor.name + ': ' + _e.message);
                }
            }
            """)
        except quickjs.JSException as e:
            errors.append(f"RUNTIME (DOMContentLoaded): {str(e).split(chr(10))[0]}")

        # Fire deferred setTimeout callbacks (collected, not executed synchronously)
        try:
            ctx.eval("""
            for (var i = 0; i < _deferredTimers.length; i++) {
                try {
                    _deferredTimers[i]();
                } catch(_e) {
                    _verifyErrors.push('setTimeout: ' + _e.constructor.name + ': ' + _e.message);
                }
            }
            """)
        except quickjs.JSException as e:
            errors.append(f"RUNTIME (setTimeout): {str(e).split(chr(10))[0]}")

        # =================================================================
        # AUTO-GENERATED SMOKE TESTS
        # =================================================================
        # Generated deterministically from the project analysis — no LLM.
        # Tests that every button has a listener, every output container
        # got content, every form has a submit handler, etc.
        test_script = self._generate_smoke_tests(dom, all_js)
        if test_script:
            try:
                ctx.eval(test_script)
            except quickjs.JSException as e:
                errors.append(f"SMOKE TEST: {str(e).split(chr(10))[0]}")

        # Collect all errors from the JS-side error array (includes smoke test failures)
        try:
            collected = ctx.eval("JSON.stringify(_verifyErrors)")
            if collected:
                import json as _json
                for err in _json.loads(collected):
                    errors.append(f"RUNTIME: {err}")
        except Exception as exc:
            # quickjs eval / JSON parse can fail if _verifyErrors is
            # undefined or corrupt — surface the existing errors list
            # without losing what was already collected.
            log.debug("artifact_app_quickjs_verify_collect_failed", error=str(exc))

        return errors

    @staticmethod
    def _generate_smoke_tests(dom: dict, all_js: str) -> str:
        """Generate deterministic smoke tests from project analysis.

        Each test produces an actionable error message telling the LLM
        EXACTLY what file, element, and fix is needed.

        Tests run AFTER all code + DOMContentLoaded + setTimeout — they
        check the FINAL state of the application.

        Levels:
        1. WIRING: Do buttons have listeners? Do forms have submit handlers?
        2. STATE: Did the app initialize state? Is localStorage used?
        3. OUTPUT: Did containers get content? Are outputs populated?
        4. INTERACTION: Simulate a click — does it change anything?
        """
        tests = []
        ids = dom.get("ids", set())
        tags = dom.get("tags", set())

        # Track canvas initialization
        preamble = "var _canvasInitialized = false;\n" if "canvas" in tags else ""

        # ===== LEVEL 1: WIRING TESTS =====

        # Every button-like element should have at least one event listener
        # Check both direct listeners AND event delegation (handler on parent that
        # references the button's ID via closest/matches/dataset).
        button_ids = [eid for eid in ids if any(
            kw in eid.lower() for kw in ("btn", "button", "submit", "start", "restart", "add", "save", "delete", "close", "toggle", "reset", "cancel")
        )]
        # Pre-check: which button IDs are referenced in addEventListener calls in the JS?
        # This catches event delegation patterns like: container.addEventListener('click', (e) => { if (e.target.id === 'btn-start') ... })
        delegated_ids = set()
        for eid in button_ids:
            # Check if the ID appears near an addEventListener in the source
            if re.search(r"addEventListener\s*\(\s*['\"]click['\"][\s\S]{0,500}" + re.escape(eid), all_js):
                delegated_ids.add(eid)
            # Also check for onclick assignment: document.getElementById('btn-x').onclick
            if re.search(re.escape(eid) + r"['\"][\s\S]{0,50}\.onclick\s*=", all_js):
                delegated_ids.add(eid)
            # Check for querySelector with the ID + click
            if re.search(r"querySelector\s*\(\s*['\"]#" + re.escape(eid) + r"['\"][\s\S]{0,100}click", all_js):
                delegated_ids.add(eid)

        for eid in button_ids:
            if eid in delegated_ids:
                continue  # Handler exists via delegation or direct reference in JS
            tests.append(
                f'(function() {{ var el = _elementRegistry["{eid}"]; '
                f'if (!el) _verifyErrors.push("SMOKE-WIRING: #{eid} button not found in DOM. '
                f'Ensure getElementById(\'{eid}\') is called during initialization."); '
                f'else if (!el._listeners || !el._listeners.click) '
                f'_verifyErrors.push("SMOKE-WIRING: #{eid} button has no click handler. '
                f'Add: document.getElementById(\'{eid}\').addEventListener(\'click\', handler); '
                f'The handler should perform the button\'s action (start game, submit form, etc.).'
                f'"); }})();'
            )

        # Forms should have submit listeners
        if "form" in tags:
            form_ids = [eid for eid in ids if "form" in eid.lower()]
            for eid in form_ids:
                tests.append(
                    f'(function() {{ var el = _elementRegistry["{eid}"]; '
                    f'if (el && (!el._listeners || !el._listeners.submit)) '
                    f'_verifyErrors.push("SMOKE-WIRING: #{eid} form has no submit handler. '
                    f'Add: document.getElementById(\'{eid}\').addEventListener(\'submit\', function(e) {{ '
                    f'e.preventDefault(); /* process form */ }}); '
                    f'The handler MUST call e.preventDefault() to prevent page reload.'
                    f'"); }})();'
                )

        # Canvas should have getContext called
        if "canvas" in tags:
            tests.append(
                'if (typeof _canvasInitialized === "undefined" || !_canvasInitialized) '
                '_verifyErrors.push("SMOKE-WIRING: <canvas> exists but getContext() was never called. '
                'The JS must call canvas.getContext(\'2d\') to initialize drawing. '
                'Ensure this happens during DOMContentLoaded or at script load time.");'
            )

        # ===== LEVEL 2: STATE TESTS =====

        # Check window.X exports are defined (not undefined from failed IIFEs)
        seen_exports = set()
        for m in re.finditer(r'window\.(\w+)\s*=', all_js):
            name = m.group(1)
            if name in ('addEventListener', 'removeEventListener') or name in seen_exports:
                continue
            seen_exports.add(name)
            tests.append(
                f'if (typeof window.{name} === "undefined") '
                f'_verifyErrors.push("SMOKE-STATE: window.{name} was assigned in code but is undefined at runtime. '
                f'The IIFE that defines it likely threw an error during initialization. '
                f'Check the file that assigns window.{name} for syntax errors or missing dependencies.");'
            )

        # Game state should have expected properties
        if "window.gameState" in all_js:
            tests.append(
                'if (window.gameState && typeof window.gameState.phase === "undefined") '
                '_verifyErrors.push("SMOKE-STATE: window.gameState exists but has no .phase property. '
                'Game state should be initialized with at least { phase: \'menu\', score: 0 }. '
                'Set this in the game initialization code.");'
            )

        # ===== LEVEL 3: OUTPUT TESTS =====

        # Check that output/result containers got content after initialization
        output_ids = [eid for eid in ids if any(
            kw in eid.lower() for kw in ("result", "output", "display", "list", "container", "content", "preview")
        )]
        for eid in output_ids:
            tests.append(
                f'(function() {{ var el = _elementRegistry["{eid}"]; '
                f'if (el && el.innerHTML === "" && el.textContent === "" && el.children.length === 0) '
                f'_verifyErrors.push("SMOKE-OUTPUT: #{eid} is empty after initialization. '
                f'If this container should show default content, a placeholder, or an empty state message, '
                f'add initial content during DOMContentLoaded. '
                f'Empty containers make the app look broken on first load.'
                f'"); }})();'
            )

        # ===== LEVEL 4: INTERACTION TESTS =====

        # Simulate clicking start/restart buttons and check if state changes
        for eid in button_ids:
            if any(kw in eid.lower() for kw in ("start", "restart", "play", "begin")):
                tests.append(
                    f'(function() {{ var el = _elementRegistry["{eid}"]; '
                    f'if (el && el._listeners && el._listeners.click) {{ '
                    f'try {{ '
                    f'var beforePhase = window.gameState ? window.gameState.phase : null; '
                    f'el.click(); '
                    f'var afterPhase = window.gameState ? window.gameState.phase : null; '
                    f'if (beforePhase && afterPhase && beforePhase === afterPhase) '
                    f'_verifyErrors.push("SMOKE-INTERACTION: Clicking #{eid} did not change gameState.phase. '
                    f'The start/restart button handler should transition from menu/gameover to playing state. '
                    f'Check the click handler — it should set gameState.phase = \'playing\' and initialize the game loop.'
                    f'"); '
                    f'}} catch(e) {{ _verifyErrors.push("SMOKE-INTERACTION: Clicking #{eid} threw: " + e.constructor.name + ": " + e.message + ". '
                    f'The button click handler has a runtime error. Fix the handler function."); }} '
                    f'}} }})();'
                )

        # Simulate form submit and check for errors
        if "form" in tags:
            for eid in [eid for eid in ids if "form" in eid.lower()]:
                tests.append(
                    f'(function() {{ var el = _elementRegistry["{eid}"]; '
                    f'if (el && el._listeners && el._listeners.submit) {{ '
                    f'try {{ '
                    f'el._listeners.submit.forEach(function(fn) {{ fn({{preventDefault: function(){{}}, target: el}}); }}); '
                    f'}} catch(e) {{ _verifyErrors.push("SMOKE-INTERACTION: Submitting #{eid} threw: " + e.constructor.name + ": " + e.message + ". '
                    f'The form submit handler has a runtime error. Check that all referenced elements exist and functions are defined.'
                    f'"); }} '
                    f'}} }})();'
                )

        # ===== LEVEL 5: PERSISTENCE TEST =====

        # Check if localStorage was written to (if app implies persistence)
        if "localStorage" in all_js:
            tests.append(
                '(function() { var keys = Object.keys(localStorage._d || {}); '
                'if (keys.length === 0) '
                '_verifyErrors.push("SMOKE-PERSIST: Code references localStorage but nothing was saved during initialization. '
                'If the app should persist data, call localStorage.setItem() during init or on first user action. '
                'If data is only saved on user action, this is expected — but consider saving defaults on first load.'
                '"); })();'
            )

        if not tests:
            return ""

        return preamble + "\n".join(tests)

    @staticmethod
    def verify_intent(description: str, files: list[dict]) -> list[str]:
        """Verify that generated code implements the features described in the prompt.

        Delegates to :func:`application_intent.verify_intent_gaps`, which
        consumes the unified :data:`_INTENT_RULES` table. Both plan-time
        hints and verify-time patterns live on the same rule, so adding
        a new feature means editing one dict in one file.
        """
        all_js = "\n".join(f["content"] for f in files if f.get("role") in ("script", "module"))
        all_html = "\n".join(f["content"] for f in files if f.get("role") == "entry")
        all_css = "\n".join(f["content"] for f in files if f.get("role") == "style")

        # Strip comments from JS before checking patterns — prevents
        # "// TODO: add multiply" from passing the multiply check.
        js_no_comments = re.sub(r'//.*$', '', all_js, flags=re.MULTILINE)  # single-line
        js_no_comments = re.sub(r'/\*[\s\S]*?\*/', '', js_no_comments)     # multi-line
        all_code = all_html + "\n" + all_css + "\n" + js_no_comments

        from augmentum.tools.application_intent import verify_intent_gaps
        return verify_intent_gaps(description, all_code)

    def _regex_verify(self, all_js: str, html_ids: set[str]) -> list[str]:
        """Regex-based static analysis fallback when quickjs is unavailable."""

        errors: list[str] = []

        # Duplicate class definitions
        class_defs: dict[str, int] = {}
        for m in re.finditer(r"\bclass\s+(\w+)\s*(?:extends\s+\w+\s*)?\{", all_js):
            name = m.group(1)
            class_defs[name] = class_defs.get(name, 0) + 1
        for name, count in class_defs.items():
            if count > 1:
                errors.append(f"RUNTIME: class '{name}' defined {count} times — SyntaxError in browser")

        # Brace mismatch
        opens = all_js.count("{")
        closes = all_js.count("}")
        if abs(opens - closes) > 2:
            errors.append(f"RUNTIME: Brace mismatch — {opens} open, {closes} close")

        # Missing DOM IDs
        for m in re.finditer(r'getElementById\s*\(\s*["\'](\w[\w-]*)["\']', all_js):
            if m.group(1) not in html_ids:
                errors.append(f"RUNTIME: getElementById('{m.group(1)}') — element not in HTML")

        return errors

    async def _pass_deliver(self, ctx: PipelineContext) -> PassResult:
        # Sanitize file paths — strip backticks, quotes, whitespace from LLM output artifacts
        for f in ctx.files:
            f["path"] = f["path"].strip('`"\' \t')

        # Deduplicate files by path (LLM may output same file with slightly different names)
        seen_paths: set[str] = set()
        deduped: list[dict] = []
        for f in ctx.files:
            if f["path"] not in seen_paths:
                seen_paths.add(f["path"])
                deduped.append(f)
        ctx.files = deduped

        # Assemble preview HTML
        ctx.preview_html = self._assemble(ctx.files)

        # Create zip
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in ctx.files:
                zf.writestr(f["path"], f["content"])
            zf.writestr("README.md", f"# {ctx.project_name}\n\nGenerated by Augmentum.\n\nOpen `index.html` in your browser to run.\n")

        # Save source_json with files array so the workspace can load and edit the project
        import json as _json
        source = _json.dumps({
            "type": "application",
            "name": ctx.project_name,
            "scaffold": ctx.scaffold_id,
            "files": [{"path": f["path"], "role": f.get("role", ""), "content": f["content"]} for f in ctx.files],
            "qualityStatus": ctx.quality_status,
            "warnings": list(ctx.quality_warnings),
            "blockingErrors": list(ctx.blocking_errors),
        })

        artifact = await self._store.save(
            data=zip_buf.getvalue(),
            filename=f"{ctx.project_name.lower().replace(' ', '-')}.zip",
            fmt="zip",
            task_id=self._task_id,
            session_id=self._session_id,
            display_name=ctx.project_name,
            metadata={
                "file_count": len(ctx.files),
                "score": ctx.score,
                "description": ctx.description[:200],
                "quality_status": ctx.quality_status,
                "warnings": list(ctx.quality_warnings),
                "blocking_errors": list(ctx.blocking_errors),
            },
            source_json=source,
            user_id=self._user_id,
        )

        ctx.artifact_id = artifact["id"]

        # Snapshot a version row so the workspace history list and the
        # revert button have something to read. Best-effort — a failed
        # snapshot must not poison the build, the artifact is already
        # safely persisted above.
        user_id = self._user_id
        if user_id and ctx.files:
            label = "Initial build" if not ctx.is_iteration else (ctx.description[:120] or "Iteration")
            try:
                await self._store.save_version(
                    ctx.artifact_id,
                    ctx.files,
                    user_id=user_id,
                    label=label,
                    score=ctx.score or None,
                )
            except Exception as exc:
                log.warning("artifact_versions.snapshot_failed",
                            artifact_id=ctx.artifact_id, error=str(exc))

        # Library thumbnail. Best-effort — if chromium isn't installed or
        # the capture errors out, the library falls back to a placeholder
        # and the lazy-backfill route can retry on first card render.
        if user_id and ctx.preview_html:
            try:
                await capture_app_preview_screenshot(
                    self._store,
                    ctx.artifact_id,
                    ctx.preview_html,
                    user_id=user_id,
                )
            except Exception as exc:
                log.warning("artifact_preview_screenshot.failed",
                            artifact_id=ctx.artifact_id, error=str(exc)[:200])

        return PassResult(done=True, detail="delivered")

    # --- Helpers ---

    _PASS_ICONS = {"plan": "\U0001F4CB", "generate": "\U0001F4C4", "validate": "\U0001F50D", "improve": "\u2B50", "polish": "\u2728", "verify": "\U0001F9EA", "deliver": "\U0001F4E6"}

    async def _emit(self, cb: Callable | None, ctx: PipelineContext, status: str, detail: str = "") -> None:
        if not cb:
            return
        import asyncio

        # Build visible progress text for the chat message.
        # Only the deliver completion produces chat-visible text — all other
        # progress is shown exclusively in the build monitor popup.
        text = ""
        if status == "complete" and ctx.current_pass == "deliver":
            file_count = len(ctx.files)
            score_text = f" \u00B7 Score {ctx.score}/10" if ctx.score else ""
            token_text = f" \u00B7 {ctx._total_tokens:,} tokens" if ctx._total_tokens else ""
            calls_text = f" \u00B7 {ctx._total_llm_calls} LLM calls" if ctx._total_llm_calls else ""
            if ctx.quality_status != "clean" or ctx.quality_warnings:
                text = f"\u26A0\uFE0F **Build complete - review recommended.** {file_count} files{score_text}{token_text}{calls_text}\n\n"
                for warning in ctx.quality_warnings[:3]:
                    text += f"- {warning}\n"
                text += "\n"
            else:
                text = f"\u2705 **Build complete!** {file_count} files{score_text}{token_text}{calls_text}\n\n"
            for f in ctx.files:
                lines = f["content"].count("\n") + 1
                text += f"`{f['path']}` ({lines} lines)\n"
            text += "\n"

        remaining_files = [f["path"] for f in ctx.planned_files if f["path"] not in ctx.generated_files]
        progress_data = {
            "name": ctx.project_name,
            "pass": ctx.current_pass,
            "status": status,
            "iteration": ctx.iterations.get(ctx.current_pass, 0),
            "max_iterations": ctx.pass_budgets.get(ctx.current_pass, 0),
            "detail": detail,
            "filesComplete": list(ctx.generated_files.keys()),
            "filesRemaining": remaining_files,
            "currentFile": remaining_files[0] if remaining_files else "",
            "score": ctx.score,
            "totalTokens": ctx._total_tokens,
            "llmCalls": ctx._total_llm_calls,
            "qualityStatus": ctx.quality_status,
            "quality_status": ctx.quality_status,
            "warnings": list(ctx.quality_warnings),
            "blockingErrors": list(ctx.blocking_errors),
            "blocking_errors": list(ctx.blocking_errors),
        }
        # Checkpoint current files on every emit once files exist so
        # background builds can recover partial progress if the pipeline crashes.
        if ctx.files:
            progress_data["files"] = ctx.files
            progress_data["planned_files"] = ctx.planned_files
            progress_data["completed_files"] = list(ctx.generated_files.keys())
        data = {
            "project_progress": progress_data,
            "_content_delta": text,  # visible text for streaming message
        }
        result = cb(data)
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            await result

    def _intercept_generated_code(self, file_info: dict, ctx: PipelineContext) -> str:
        """Layer 1: Streaming interception — fix errors in generated code before storing.

        Runs deterministic find-and-replace on the generated output BEFORE storing.
        The user never sees the broken intermediate state.
        Modeled after v0's LLM Suspense architecture.
        """
        content = file_info["content"]
        path = file_info["path"]

        # Strip pipeline sentinels. Historically only the fallback-2 parser stripped
        # these, so when the model embedded the sentinel INSIDE its code fence
        # (common) the token shipped to the browser and crashed the preview with
        # "__PASS_COMPLETE__ is not defined". Do it once, here, for every path.
        for _sentinel in ("__PASS_COMPLETE__", "__NEEDS_ANOTHER_PASS__"):
            content = re.sub(re.escape(_sentinel) + r"[^\n]*", "", content)
        content = content.rstrip() + "\n" if content.strip() else content

        is_js = path.endswith(".js") or path.endswith(".ts")
        is_css = path.endswith(".css") or path.endswith(".scss")
        is_html = path.endswith(".html") or path.endswith(".htm")
        project_paths = {f["path"] for f in ctx.planned_files}
        intercepted = 0

        # --- Build context from previously generated files ---
        declared_names: set[str] = set()
        declared_functions: set[str] = set()
        class_to_instance: dict[str, str] = {}
        all_prev_js = ""

        for prev_path, prev_content in ctx.generated_files.items():
            if prev_path == path:
                continue
            for m in re.finditer(r'^(?:const|let|class)\s+(\w+)', prev_content, re.MULTILINE):
                declared_names.add(m.group(1))
            for m in re.finditer(r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:function|\(.*?\)\s*=>))', prev_content):
                declared_functions.add(m.group(1) or m.group(2))
            for m in re.finditer(r'window\.(\w+)\s*=\s*new\s+(\w+)', prev_content):
                class_to_instance[m.group(2)] = m.group(1)
            if prev_path.endswith((".js", ".ts")):
                all_prev_js += prev_content + "\n"

        # =================================================================
        # JS INTERCEPTORS
        # =================================================================
        if is_js:
            # 1. Duplicate const/let declarations → window.X
            #    Only fix TOP-LEVEL declarations (brace depth 0). A `const i`
            #    inside a function is a local variable and won't collide.
            for name in declared_names:
                pattern = re.compile(r'^(const|let)\s+' + re.escape(name) + r'\b', re.MULTILINE)
                match = pattern.search(content)
                if match and self._is_top_level(content, match.start()):
                    content = content[:match.start()] + f'window.{name}' + content[match.end():]
                    intercepted += 1

            # 2. Class-as-function: ClassName.method() → window.instance.method()
            for cls, instance in class_to_instance.items():
                pattern = re.compile(r'(?<!new\s)(?<!class\s)\b' + re.escape(cls) + r'\.(\w+)\s*\(')
                if pattern.search(content):
                    content = pattern.sub(f'window.{instance}.\\1(', content)
                    intercepted += 1

            # 3. ES module syntax in non-module context → strip or convert
            #    LLMs often generate import/export in regular scripts
            if re.search(r'^import\s+', content, re.MULTILINE):
                content = re.sub(r'^import\s+.*?;\s*\n?', '', content, flags=re.MULTILINE)
                intercepted += 1
            if re.search(r'^export\s+(default\s+)?', content, re.MULTILINE):
                content = re.sub(r'^export\s+default\s+', '', content, flags=re.MULTILINE)
                content = re.sub(r'^export\s+', '', content, flags=re.MULTILINE)
                intercepted += 1

            # 4. Cross-file function access without window prefix
            #    If function foo() is defined in a previous file and this file calls foo(),
            #    it works in assembled mode. But if it's assigned to const foo = ... in the
            #    previous file, we already moved it to window.foo — so bare foo() calls break.
            #    Fix: if name was declared in previous file AND we renamed it to window.X,
            #    update bare references in this file too.
            for name in declared_names:
                # Only fix if this file uses the name as a function call
                call_pattern = re.compile(r'(?<![.\w])' + re.escape(name) + r'\s*\(')
                if call_pattern.search(content):
                    # Check it's not a local declaration in THIS file
                    local_decl = re.search(r'^(?:const|let|var|function)\s+' + re.escape(name) + r'\b', content, re.MULTILINE)
                    if not local_decl:
                        content = call_pattern.sub(f'window.{name}(', content)
                        intercepted += 1

            # 5. querySelector/getElementById on elements likely not yet in DOM
            #    If code runs at top level (not in a function/event listener) and accesses DOM,
            #    wrap in DOMContentLoaded. Only for top-level getElementById, not inside functions.
            top_level_dom = re.search(
                r'^(?:const|let|var)\s+\w+\s*=\s*document\.(?:getElementById|querySelector)\s*\(',
                content, re.MULTILINE,
            )
            if top_level_dom:
                # Check if there's already a DOMContentLoaded wrapper
                if 'DOMContentLoaded' not in content and 'window.addEventListener' not in content:
                    # Don't auto-wrap — too risky. But add a comment warning.
                    content = (
                        "// NOTE: DOM access at top level — ensure this script loads after the HTML body.\n"
                        + content
                    )

            # 6. Bare 'this' in arrow functions passed as event handlers
            #    Common LLM mistake: class methods using arrow functions lose 'this' context
            #    Detection only — too complex to auto-fix reliably
            pass

        # =================================================================
        # HTML INTERCEPTORS
        # =================================================================
        if is_html:
            # 7. Strip ALL local script/link references (they'll be inlined during assembly)
            # Match any src/href that isn't a CDN URL (http/https/protocol-relative)
            _CDN_PREFIXES = r'(?:https?://|//)'
            before = content
            # Strip <script src="local-file.js"> (not CDN)
            content = re.sub(
                r'<script[^>]*src=["\'](?!' + _CDN_PREFIXES + r')[^"\']+\.js["\'][^>]*>\s*</script>\s*\n?',
                '', content, flags=re.IGNORECASE,
            )
            # Strip <link href="local-file.css"> (not CDN)
            content = re.sub(
                r'<link[^>]*href=["\'](?!' + _CDN_PREFIXES + r')[^"\']+\.css["\'][^>]*/?\s*>\s*\n?',
                '', content, flags=re.IGNORECASE,
            )
            if content != before:
                intercepted += 1

            # 8. Duplicate HTML IDs — second occurrence gets suffixed
            seen_ids: set[str] = set()
            def _dedup_id(m: re.Match) -> str:
                id_val = m.group(1)
                if id_val in seen_ids:
                    return f'id="{id_val}-2"'
                seen_ids.add(id_val)
                return m.group(0)
            content = re.sub(r'id="(\w[\w-]*)"', _dedup_id, content)

            # 9. Missing viewport meta tag (causes mobile rendering issues)
            if '<meta name="viewport"' not in content and '<head>' in content:
                content = content.replace(
                    '<head>',
                    '<head>\n    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
                )
                intercepted += 1

            # 10. Missing charset meta tag
            if '<meta charset' not in content and '<head>' in content:
                content = content.replace(
                    '<head>',
                    '<head>\n    <meta charset="UTF-8">',
                )
                intercepted += 1

        # =================================================================
        # CSS INTERCEPTORS
        # =================================================================
        if is_css:
            # 11. Common CSS property typos (LLM spelling mistakes)
            _CSS_TYPOS = {
                'backgroud': 'background', 'backgorund': 'background',
                'heigth': 'height', 'widht': 'width',
                'colro': 'color', 'colur': 'color',
                'maring': 'margin', 'marign': 'margin',
                'pading': 'padding', 'paddig': 'padding',
                'boarder': 'border', 'bordr': 'border',
                'disply': 'display', 'dispaly': 'display',
                'positon': 'position', 'postion': 'position',
                'trasition': 'transition', 'tranistion': 'transition',
                'oveflow': 'overflow', 'overlfow': 'overflow',
                'font-weitght': 'font-weight', 'font-weigth': 'font-weight',
                'visibilty': 'visibility', 'visiblity': 'visibility',
                'opacty': 'opacity', 'opcaity': 'opacity',
                'z-idnex': 'z-index', 'z-indx': 'z-index',
                'trasform': 'transform', 'transfrom': 'transform',
                'animaton': 'animation', 'aniamtion': 'animation',
                'flex-dierction': 'flex-direction', 'flex-direciton': 'flex-direction',
                'justify-contet': 'justify-content', 'justfy-content': 'justify-content',
                'align-itmes': 'align-items', 'align-tems': 'align-items',
            }
            for typo, correct in _CSS_TYPOS.items():
                if typo in content:
                    content = content.replace(typo, correct)
                    intercepted += 1

            # 12. Missing semicolons at end of CSS declarations
            content = re.sub(
                r'([a-zA-Z0-9%"\')])(\s*\n\s*[a-zA-Z-]+\s*:)',
                r'\1;\2', content,
            )

            # 13. Unclosed braces (add closing brace if opens > closes)
            opens = content.count('{')
            closes = content.count('}')
            if opens > closes:
                content += '\n}' * (opens - closes)
                intercepted += 1

        # =================================================================
        # UNIVERSAL INTERCEPTORS (all file types)
        # =================================================================

        # 14. SEARCH/REPLACE artifact contamination
        #     When the LLM or a fix pass leaks diff markers into generated code.
        #     Pattern: bare ======= or <<<<<<< SEARCH or >>>>>>> REPLACE on a line
        #     These are NEVER valid in HTML/CSS/JS. Safe to strip unconditionally.
        sr_markers = re.compile(r'^(?:<{3,}\.?\s*SEARCH|={3,}|>{3,}\.?\s*REPLACE)\s*$', re.MULTILINE)
        if sr_markers.search(content):
            content = sr_markers.sub('', content)
            # Clean up resulting double blank lines
            content = re.sub(r'\n{3,}', '\n\n', content)
            intercepted += 1
            log.info("intercept.sr_artifact_stripped", file=path)

        # 15. Intra-file duplicate block detection
        #     LLMs lose track during long generations and repeat entire sections.
        #     Strategy: find duplicate top-level class/function definitions, then
        #     detect large verbatim block repetition as a safety net.
        if is_js or is_css:
            content, dedup_count = self._dedup_intra_file(content, is_js)
            intercepted += dedup_count

        if intercepted > 0:
            log.info("intercept.fixed", file=path, count=intercepted)

        return content

    @staticmethod
    def _dedup_intra_file(content: str, is_js: bool) -> tuple[str, int]:
        """Remove duplicated code blocks within a single file.

        Handles three cases:
        1. Duplicate class definitions: `class Foo { ... }` appears twice
        2. Duplicate function definitions: `function bar() { ... }` appears twice
        3. Large verbatim block repetition: 8+ consecutive lines that match
           an earlier section of the file

        Returns (cleaned_content, fix_count).
        """

        fixes = 0

        if is_js:
            # --- Case 1 & 2: Duplicate class/function definitions ---
            # Find all top-level class and function definitions with their content
            # by tracking brace depth from the opening declaration.
            seen_defs: dict[str, int] = {}  # name → first occurrence line
            lines = content.split("\n")
            skip_until = -1
            cleaned_lines: list[str] = []

            for i, line in enumerate(lines):
                if i < skip_until:
                    continue

                # Check for class or function definition
                m = re.match(r'^(\s*)(?:class\s+(\w+)|function\s+(\w+))\s*(?:\(|extends|\{)', line)
                if m:
                    indent = len(m.group(1))
                    name = m.group(2) or m.group(3)

                    if name in seen_defs:
                        # This is a duplicate — find the end of this block and skip it.
                        # Track brace depth from this line forward.
                        depth = 0
                        block_end = i
                        for j in range(i, len(lines)):
                            depth += lines[j].count('{') - lines[j].count('}')
                            if depth <= 0 and j > i:
                                block_end = j
                                break
                        else:
                            block_end = len(lines) - 1

                        skip_until = block_end + 1
                        fixes += 1
                        log.info("intercept.intra_dedup",
                                 name=name, type="class" if m.group(2) else "function",
                                 first_line=seen_defs[name] + 1,
                                 dup_line=i + 1, dup_end=block_end + 1)
                        continue
                    else:
                        seen_defs[name] = i

                cleaned_lines.append(line)

            if fixes > 0:
                content = "\n".join(cleaned_lines)

        # --- Case 3: Large verbatim block repetition (all file types) ---
        # Sliding window: if 8+ consecutive lines match an earlier section,
        # remove the duplicate. This catches the entities.js pattern where
        # the IIFE closes then the same code restarts without the wrapper.
        lines = content.split("\n")
        min_block = 8  # minimum lines to consider a duplicate
        trimmed = [l.strip() for l in lines]
        result_lines: list[str] = []
        i = 0
        while i < len(lines):
            # Check if lines[i:i+min_block] matches any earlier section
            if i >= min_block and i + min_block <= len(lines):
                # Build the candidate block (trimmed for comparison)
                candidate = trimmed[i:i + min_block]
                # Skip if the block is mostly empty lines
                if sum(1 for l in candidate if l) >= min_block // 2:
                    # Search for this exact sequence in earlier content
                    found_dup = False
                    for start in range(0, i - min_block + 1):
                        if trimmed[start:start + min_block] == candidate:
                            # Found a match — now extend to find the full duplicated range
                            end = i + min_block
                            while (end < len(lines)
                                   and start + (end - i) < i
                                   and trimmed[end] == trimmed[start + (end - i)]):
                                end += 1
                            dup_len = end - i
                            if dup_len >= min_block:
                                log.info("intercept.block_dedup",
                                         start_line=i + 1, end_line=end,
                                         dup_lines=dup_len,
                                         matches_line=start + 1)
                                i = end  # skip the entire duplicate block
                                fixes += 1
                                found_dup = True
                                break
                    if found_dup:
                        continue

            result_lines.append(lines[i])
            i += 1

        if len(result_lines) < len(lines):
            content = "\n".join(result_lines)

        return content, fixes

    @staticmethod
    def _build_depth_map(content: str) -> list[int]:
        """Precompute brace depth at every position in JS source.

        Skips braces inside string literals (including template literal
        interpolations), single-line comments, and multi-line comments.
        Called once per file, then depth at any position is O(1) lookup.
        """
        n = len(content)
        depths = [0] * n
        depth = 0
        i = 0
        while i < n:
            ch = content[i]
            # Skip single-line comments
            if ch == '/' and i + 1 < n and content[i + 1] == '/':
                while i < n and content[i] != '\n':
                    depths[i] = depth
                    i += 1
                continue
            # Skip multi-line comments
            if ch == '/' and i + 1 < n and content[i + 1] == '*':
                end = content.find('*/', i + 2)
                end = end + 2 if end != -1 else n
                while i < end:
                    depths[i] = depth
                    i += 1
                continue
            # Skip string literals (handles template literal ${} interpolation)
            if ch in ('"', "'", '`'):
                quote = ch
                depths[i] = depth
                i += 1
                while i < n:
                    if content[i] == '\\':
                        depths[i] = depth
                        i += 1
                        if i < n:
                            depths[i] = depth
                            i += 1
                        continue
                    if quote == '`' and content[i] == '$' and i + 1 < n and content[i + 1] == '{':
                        # Template literal interpolation — braces inside ARE real scope
                        depths[i] = depth
                        i += 1
                        depths[i] = depth
                        i += 1
                        depth += 1
                        # The matching } will be handled by the normal brace logic
                        break
                    if content[i] == quote:
                        depths[i] = depth
                        i += 1
                        break
                    depths[i] = depth
                    i += 1
                continue
            if ch == '{':
                depths[i] = depth
                depth += 1
            elif ch == '}':
                depth = max(0, depth - 1)
                depths[i] = depth
            else:
                depths[i] = depth
            i += 1
        return depths

    @staticmethod
    def _is_top_level(content: str, pos: int) -> bool:
        """Check if a position in JS source is at top-level scope (brace depth 0).

        Uses precomputed depth map when available (cached on the content string
        via a module-level dict to avoid recomputation). Falls back to building
        the map on demand.
        """
        # Cache depth maps per content id to avoid recomputing for each declaration
        cache = ApplicationBuilderTool._depth_cache
        content_id = id(content)
        if content_id not in cache:
            cache.clear()  # Only cache one file at a time to limit memory
            cache[content_id] = ApplicationBuilderTool._build_depth_map(content)
        depths = cache[content_id]
        if pos < len(depths):
            return depths[pos] <= 0
        return True  # Past end = top level

    _depth_cache: dict[int, list[int]] = {}

    @staticmethod
    def _normalize_patch_filename(filename: str) -> str:
        """Normalize common LLM FILE header decorations to a project path."""
        name = (filename or "").strip().strip("`'\"")
        name = re.sub(r"\s*\((?:FULL|full|signature|REFERENCE ONLY:[^)]+)\)\s*$", "", name).strip()
        if name.lower().startswith("file:"):
            name = name[5:].strip()
        return name.lstrip("./")

    @staticmethod
    def _find_patch_target(files: list, filename: str) -> dict | None:
        """Find a patch target by exact path, normalized path, or unique basename."""
        normalized = ApplicationBuilderTool._normalize_patch_filename(filename)
        for f in files:
            if f.get("path") == normalized:
                return f
        for f in files:
            if ApplicationBuilderTool._normalize_patch_filename(f.get("path", "")) == normalized:
                return f
        if "/" not in normalized and "\\" not in normalized:
            matches = [
                f for f in files
                if re.split(r"[\\/]", f.get("path", ""))[-1] == normalized
            ]
            if len(matches) == 1:
                return matches[0]
        return None

    @staticmethod
    def analyze_project(files: list[dict]) -> dict:
        """Analyze project files and return structured dependency data.

        This is the single source of truth for project health. Used by:
        - Working document project map (during generation)
        - Verify pass (post-generation structural checks)
        - Fix endpoint (detect issues after modifications)
        - Iterate endpoint (pre/post modification comparison)

        Returns a dict with: defined_ids, defined_classes, defined_globals,
        defined_functions, defined_classes_js, used_ids, used_globals,
        unresolved, per_file summaries, and structural_issues.
        """
        _WINDOW_BUILTINS = frozenset({
            'addEventListener', 'removeEventListener', 'innerWidth', 'innerHeight',
            'setTimeout', 'setInterval', 'requestAnimationFrame', 'cancelAnimationFrame',
            'document', 'location', 'navigator', 'console', 'localStorage', 'sessionStorage',
            'performance', 'matchMedia', 'getComputedStyle', 'scrollTo', 'scrollX', 'scrollY',
            'devicePixelRatio', 'fetch', 'alert', 'confirm', 'prompt', 'open', 'close',
            'onresize', 'onload', 'onerror', 'dispatchEvent', 'history', 'screen',
            'JSON', 'Math', 'Date', 'Array', 'Object', 'Map', 'Set', 'window',
        })

        result = {
            "defined_ids": {},        # id → file
            "defined_classes": {},    # class → file
            "defined_globals": {},    # window.X → file
            "defined_functions": {},  # funcName → file
            "defined_classes_js": {}, # ClassName → file
            "used_ids": {},           # file → set of IDs
            "used_globals": {},       # file → set of window.X
            "unresolved": [],         # issues the project has
            "structural_issues": [],  # auto-fixable or LLM-fixable problems
            "per_file": [],           # [{path, role, lines, defines, uses}]
        }

        d = result  # shorthand

        for f in files:
            path, content, role = f["path"], f["content"], f.get("role", "")
            file_info = {"path": path, "role": role, "lines": content.count("\n") + 1, "defines": [], "uses": []}

            if role == "entry":
                for m in re.finditer(r'id=["\'](\w[\w-]*)["\']', content):
                    d["defined_ids"][m.group(1)] = path
                for m in re.finditer(r'class=["\']([^"\']+)["\']', content):
                    for cls in m.group(1).split():
                        d["defined_classes"][cls] = path
                # Buttons without inline handlers
                for m in re.finditer(r'<button[^>]*id=["\'](\w[\w-]*)["\']', content):
                    snippet = content[m.start():m.start() + 200]
                    if 'onclick' not in snippet:
                        d["unresolved"].append(f"#{m.group(1)} button needs a click handler")
                # Canvas needing init
                for m in re.finditer(r'<canvas[^>]*id=["\'](\w[\w-]*)["\']', content):
                    d["unresolved"].append(f"#{m.group(1)} canvas needs getContext + drawing logic")
                # Forms needing submit handler
                for m in re.finditer(r'<form[^>]*id=["\'](\w[\w-]*)["\']', content):
                    snippet = content[m.start():m.start() + 200]
                    if 'onsubmit' not in snippet:
                        d["unresolved"].append(f"#{m.group(1)} form needs submit handler")

                file_info["defines"] = list(d["defined_ids"].keys())[-8:]

            elif role == "style":
                for m in re.finditer(r'\.(\w[\w-]*)\s*[{,:]', content):
                    d["defined_classes"][m.group(1)] = path
                file_info["defines"] = [k for k, v in d["defined_classes"].items() if v == path][-8:]

            elif role in ("script", "module"):
                for m in re.finditer(r'window\.(\w+)\s*=', content):
                    d["defined_globals"][m.group(1)] = path
                for m in re.finditer(r'(?:function\s+(\w+)|(?:const|let)\s+(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>))', content):
                    d["defined_functions"][m.group(1) or m.group(2)] = path
                for m in re.finditer(r'class\s+(\w+)', content):
                    d["defined_classes_js"][m.group(1)] = path

                # What this file uses
                file_used_ids = set()
                for m in re.finditer(r'getElementById\s*\(\s*["\'](\w[\w-]*)["\']', content):
                    file_used_ids.add(m.group(1))
                d["used_ids"][path] = file_used_ids

                file_used_globals = set()
                for m in re.finditer(r'window\.(\w+)(?:\.\w+)*\s*[\(.]', content):
                    if m.group(1) not in _WINDOW_BUILTINS:
                        file_used_globals.add(m.group(1))
                d["used_globals"][path] = file_used_globals

                # Check for addEventListener wiring buttons
                for m in re.finditer(r"getElementById\s*\(\s*['\"](\w[\w-]*)['\"].*addEventListener", content):
                    # This button IS handled — remove from unresolved
                    btn_id = m.group(1)
                    d["unresolved"] = [u for u in d["unresolved"] if f"#{btn_id}" not in u]

                file_info["defines"] = ([k for k, v in d["defined_functions"].items() if v == path][:6] +
                                        [f"window.{k}" for k, v in d["defined_globals"].items() if v == path][:4])
                file_info["uses"] = list(file_used_ids)[:6]

            d["per_file"].append(file_info)

        # Cross-file checks
        for path, ids in d["used_ids"].items():
            for eid in ids:
                if eid not in d["defined_ids"]:
                    d["structural_issues"].append(f"{path}: getElementById('{eid}') but #{eid} not in HTML")

        all_globals = set(d["defined_globals"].keys())
        for path, globals_used in d["used_globals"].items():
            for g in globals_used:
                if g not in all_globals:
                    d["structural_issues"].append(f"{path}: window.{g} used but never assigned")

        # CSS classes used in JS but not defined in CSS
        all_css_classes = set(d["defined_classes"].keys())
        for f in files:
            if f.get("role") in ("script", "module"):
                for m in re.finditer(r"classList\.(?:add|remove|toggle)\s*\(\s*['\"](\w[\w-]*)['\"]", f["content"]):
                    cls = m.group(1)
                    if cls not in all_css_classes and cls not in ("hidden", "active", "visible", "open", "closed", "disabled", "selected", "loading", "error"):
                        d["structural_issues"].append(f"{f['path']}: classList uses '{cls}' but not in CSS")

        # --- Performance checks ---
        if not d.get("performance_issues"):
            d["performance_issues"] = []

        for f in files:
            content = f["content"]
            path = f["path"]
            role = f.get("role", "")

            if role in ("script", "module"):
                # querySelector/getElementById inside a loop (layout thrashing risk)
                for m in re.finditer(r'for\s*\([^)]*\)\s*\{[^}]*(?:getElementById|querySelector)\s*\(', content, re.DOTALL):
                    d["performance_issues"].append(
                        f"{path}: DOM query inside a for-loop — cache the element before the loop. "
                        f"Repeated DOM lookups cause layout thrashing and slow performance."
                    )

                # scroll/resize handler without debounce/throttle
                for m in re.finditer(r"addEventListener\s*\(\s*['\"](?:scroll|resize)['\"]", content):
                    # Check if debounce/throttle is nearby (within 200 chars)
                    context = content[max(0, m.start()-100):m.end()+200]
                    if 'debounce' not in context and 'throttle' not in context and 'setTimeout' not in context:
                        d["performance_issues"].append(
                            f"{path}: scroll/resize event handler without debounce/throttle. "
                            f"Wrap the handler: addEventListener('scroll', debounce(handler, 100))"
                        )

                # setInterval without clearInterval reference
                if 'setInterval' in content and 'clearInterval' not in content:
                    d["performance_issues"].append(
                        f"{path}: setInterval() used but clearInterval() never called. "
                        f"This causes a memory leak — store the interval ID and clear it when done."
                    )

            if role == "style":
                # Inline base64 images > 50KB
                for m in re.finditer(r'url\s*\(\s*["\']?data:image/[^)]{50000,}', content):
                    size_kb = len(m.group(0)) // 1024
                    d["performance_issues"].append(
                        f"{path}: Inline base64 image (~{size_kb}KB) is too large. "
                        f"Use an external image file instead — inline base64 over 10KB blocks rendering."
                    )

            if role == "entry":
                # Synchronous <script> in <head> without defer/async
                for m in re.finditer(r'<head>[\s\S]*?<script(?![^>]*(?:defer|async|src="https?))[^>]*>[\s\S]*?</script>', content):
                    if 'charset' not in m.group(0) and 'application/ld' not in m.group(0):
                        d["performance_issues"].append(
                            f"{path}: Synchronous <script> in <head> blocks page rendering. "
                            f"Add 'defer' attribute or move the script to end of <body>."
                        )

        return result

    def _update_project_map(self, ctx: PipelineContext) -> str:
        """Rebuild the Project Map section of the working document using analyze_project."""
        # Strip old map section
        base = ctx.working_doc
        for marker in ("## API Surface", "## Project Map"):
            if marker in base:
                base = base[:base.index(marker)]
                break

        analysis = self.analyze_project(ctx.files)
        remaining = [f["path"] for f in ctx.planned_files if f["path"] not in ctx.generated_files]

        lines = ["## Project Map (auto-analyzed)\n"]

        if analysis["defined_ids"]:
            lines.append(f"**DOM IDs:** {', '.join(f'#{k}' for k in sorted(analysis['defined_ids']))}")
        if analysis["defined_classes"]:
            lines.append(f"**CSS Classes:** {', '.join(f'.{c}' for c in sorted(analysis['defined_classes'])[:15])}")
        if analysis["defined_globals"]:
            lines.append(f"**Globals:** {', '.join(f'window.{k} ({v})' for k, v in sorted(analysis['defined_globals'].items()))}")
        if analysis["defined_functions"]:
            lines.append(f"**Functions:** {', '.join(f'{k}()' for k in sorted(analysis['defined_functions'])[:12])}")
        if analysis["defined_classes_js"]:
            lines.append(f"**Classes:** {', '.join(f'{k} ({v})' for k, v in sorted(analysis['defined_classes_js'].items()))}")

        lines.append("")
        for fi in analysis["per_file"]:
            defines_str = " | ".join(fi["defines"][:6]) if fi["defines"] else "generated"
            lines.append(f"- **{fi['path']}** ({fi['lines']}L): {defines_str}")

        # Unresolved items (only shown if there are files left to generate)
        if analysis["unresolved"] and remaining:
            lines.append("")
            lines.append("**\u26a0 Unresolved (next file should handle):**")
            for u in analysis["unresolved"][:8]:
                lines.append(f"- {u}")

        # Structural issues (always shown — these are bugs)
        if analysis["structural_issues"]:
            lines.append("")
            lines.append("**\u274c Structural issues detected:**")
            for issue in analysis["structural_issues"][:6]:
                lines.append(f"- {issue}")

        return base.rstrip() + "\n\n" + "\n".join(lines) + "\n"

    @staticmethod
    def _looks_like_code(content: str, role: str) -> bool:
        """Check if content looks like actual code vs. echoed prompt/plan text.

        Models sometimes return plan descriptions, prompt text, or
        markdown explanations instead of code. This catches obvious
        non-code before it gets stored as a file.
        """
        content_lower = content.lower().strip()

        # Dead giveaways that it's NOT code:
        non_code_signals = [
            "generate this file",
            "file content here",
            "your code here",
            "implement this",
            "todo: implement",
            "| action:",
            "| role:",
            "| description:",
            "file: <path>",
        ]
        for signal in non_code_signals:
            if signal in content_lower:
                return False

        # Check for code-like content based on role
        if role == "entry":
            # HTML should have tags
            return "<" in content and ">" in content
        elif role == "style":
            # CSS should have braces and colons
            return "{" in content and ":" in content
        elif role in ("script", "module"):
            # JS should have function-like constructs
            return any(kw in content for kw in ("function", "const ", "let ", "var ", "=>", "class "))
        elif role == "data":
            # JSON should start with { or [
            stripped = content.strip()
            return stripped.startswith(("{", "["))

        # Default: if it has some code-like characters, accept it
        return "{" in content or "(" in content or "<" in content

    def _extract_name(self, description: str) -> str:
        return derive_project_name(description)

    def _detect_missing_references(self, ctx: PipelineContext) -> list[dict]:
        """Scan entry HTML for file references that weren't generated.

        Catches the common case where the LLM's HTML includes <script src="app.js">
        or <link href="styles.css"> but those files weren't in the plan.
        Also detects inline event handlers (onclick) calling functions that don't
        exist in any generated JS — signals a missing script file.
        Returns a list of file descriptors to add to the plan.
        """
        entry = next((f for f in ctx.files if f.get("role") == "entry"), None)
        if not entry:
            return []

        html = entry["content"]
        generated = {f["path"] for f in ctx.files}
        # CDN URLs should not be treated as missing project files
        _CDN_PREFIXES = ("http://", "https://", "//", "data:")
        gap_files: list[dict] = []
        seen: set[str] = set()

        # 1. <script src="filename.js"> references
        for m in re.finditer(r'<script[^>]*src=["\']([^"\']+)["\']', html, re.IGNORECASE):
            src = m.group(1).strip()
            if any(src.startswith(p) for p in _CDN_PREFIXES):
                continue
            if src not in generated and src not in seen:
                seen.add(src)
                gap_files.append({
                    "path": src,
                    "role": self._infer_role(src),
                    "lang": self._infer_lang(src),
                    "action": "create",
                    "description": f"Referenced by {entry['path']} via <script src>",
                })

        # 2. <link href="filename.css"> references
        for m in re.finditer(r'<link[^>]*href=["\']([^"\']+)["\']', html, re.IGNORECASE):
            href = m.group(1).strip()
            if any(href.startswith(p) for p in _CDN_PREFIXES):
                continue
            # Only treat as project file if it has a known extension
            if not href.endswith((".css", ".scss")):
                continue
            if href not in generated and href not in seen:
                seen.add(href)
                gap_files.append({
                    "path": href,
                    "role": "style",
                    "lang": self._infer_lang(href),
                    "action": "create",
                    "description": f"Referenced by {entry['path']} via <link href>",
                })

        # 3. Inline event handlers calling functions not defined in any generated JS
        #    e.g., onclick="calculate()" but no JS file defines calculate()
        all_js = "\n".join(f["content"] for f in ctx.files if f.get("role") in ("script", "module"))
        if not all_js and not gap_files:
            # No JS files at all — check if the HTML has event handlers or interactive elements
            has_handlers = re.search(
                r'(?:onclick|onsubmit|onchange|oninput|onkeyup|onkeydown)\s*=',
                html, re.IGNORECASE,
            )
            has_interactive = re.search(
                r'<(?:button|input|select|form|canvas)\b', html, re.IGNORECASE,
            )
            if has_handlers or has_interactive:
                # The HTML expects JS but none was generated — add a default script
                scaffold = SCAFFOLDS.get(ctx.scaffold_id, SCAFFOLDS["static"])
                default_js = next(
                    (df for df in scaffold["default_files"] if df["role"] == "script"),
                    {"path": "app.js"},
                )
                if default_js["path"] not in generated and default_js["path"] not in seen:
                    seen.add(default_js["path"])
                    gap_files.append({
                        "path": default_js["path"],
                        "role": "script",
                        "lang": "javascript",
                        "action": "create",
                        "description": "HTML has interactive elements/handlers but no JS file was generated",
                    })

        return gap_files

    def _parse_file_plan(self, response: str) -> list:
        files = []
        # Primary: structured FILE: format
        for m in re.finditer(r"FILE:\s*(\S+)\s*\|\s*(?:ROLE:\s*(\w+)\s*\|)?\s*(?:LANG:\s*(\w+)\s*\|)?\s*(?:ACTION:\s*(\w+)\s*\|)?\s*DESCRIPTION:\s*(.+)", response):
            raw_desc = m.group(5).strip()
            # The description line may carry optional contract columns
            # (PROVIDES / DEPENDS / WIRES) appended via pipe separators.
            # Split them off so they don't pollute the description text
            # shown in prompts/logs.
            description, contract = _split_description_and_contract(raw_desc)
            entry = {
                "path": m.group(1),
                "role": m.group(2) or self._infer_role(m.group(1)),
                "lang": m.group(3) or self._infer_lang(m.group(1)),
                "action": m.group(4) or "create",
                "description": description,
            }
            if contract:
                entry.update(contract)
            files.append(entry)

        if files:
            return files

        # Fallback 1: numbered list with filenames (e.g., "1. index.html - main page")
        for m in re.finditer(r"(?:^|\n)\s*\d+\.\s*[`*]*(\w+\.\w+)[`*]*\s*[-:\u2013\u2014]\s*(.+)", response):
            path = m.group(1)
            desc = m.group(2).strip()
            files.append({
                "path": path,
                "role": self._infer_role(path),
                "lang": self._infer_lang(path),
                "action": "create",
                "description": desc,
            })

        if files:
            return files

        # Fallback 2: detect any filenames with extensions mentioned in the text
        seen = set()
        for m in re.finditer(r"\b(\w+\.(?:html|css|js|json|md|ts|svg))\b", response):
            path = m.group(1)
            if path not in seen:
                seen.add(path)
                files.append({
                    "path": path,
                    "role": self._infer_role(path),
                    "lang": self._infer_lang(path),
                    "action": "create",
                    "description": "",
                })

        return files

    def _parse_generated_files(self, response: str) -> list:
        """Parse fenced code blocks with filenames from LLM response.

        Handles various LLM output formats:
        - ```filename.ext  (clean)
        - ```filename.ext`  (trailing backtick — common LLM mistake)
        - ```language filename.ext  (language then filename)
        - ```language  (language only — matched to target file if available)
        """
        files = []
        # Match fenced blocks: ``` followed by a label (may include trailing backticks),
        # then content, then closing ```. The label regex allows backticks so we can strip them.
        for m in re.finditer(r"```([^\n]+?)\s*\n([\s\S]*?)```", response):
            raw_label = m.group(1).strip().strip('`').strip()
            content = m.group(2).rstrip()

            # Skip empty content
            if not content.strip():
                continue

            # Skip known non-file labels (context blocks, etc.)
            if raw_label.lower() in ('context', 'json', 'text', 'markdown', 'md', 'bash', 'sh', 'shell', 'diff'):
                continue

            filename = None

            # Case 1: Label contains a dot — it's a filename (index.html, app.js, styles.css)
            if '.' in raw_label:
                # Handle "language filename.ext" format (e.g., "html index.html")
                parts = raw_label.split()
                filename = next((p for p in parts if '.' in p), raw_label)
                # Clean any remaining backticks or quotes
                filename = filename.strip('`"\' ')

            # Case 2: Label is a language tag (javascript, css, html, python)
            #         Map to the most likely target file based on language
            elif raw_label.lower() in ('javascript', 'js', 'typescript', 'ts'):
                filename = None  # Will be handled by fallback in _pass_generate
            elif raw_label.lower() in ('html', 'htm') or raw_label.lower() in ('css', 'scss'):
                filename = None

            if filename:
                files.append({"path": filename, "content": content})

        return files

    def _parse_score(self, response: str) -> float:
        m = re.search(r"SCORE:\s*(\d+(?:\.\d+)?)\s*/\s*10", response)
        return float(m.group(1)) if m else 5.0

    @staticmethod
    def _score_is_below_gate(score: float) -> bool:
        """True if the judge returned a confident low score.

        Filters out the _parse_score default (5.0) which signals "no
        parseable score" — we don't want to trigger a retry loop on
        parse failures, only on genuine low scores from the model.
        Scores of exactly 5.0 that the model actually meant will be
        under-gated here; that's an acceptable false-negative tradeoff
        against the alternative of burning LLM calls on every parse
        failure.
        """
        return 0.0 < score < _IMPROVE_GATE_THRESHOLD and score != _PARSE_SCORE_DEFAULT

    @staticmethod
    def _extract_judge_improvements(response: str) -> list[str]:
        """Pull the bullet list under ``IMPROVEMENTS:`` out of a judge
        response. These bullets drive the targeted fix attempt when the
        gate triggers. Returns ``[]`` when the section is missing or
        empty — caller should skip the retry in that case.
        """
        m = re.search(
            r"IMPROVEMENTS:\s*\n(.*?)(?:\n\s*(?:__PASS_COMPLETE__|__NEEDS_ANOTHER_PASS__|SCORE:|STRENGTHS:)|\Z)",
            response,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if not m:
            return []
        block = m.group(1)
        bullets: list[str] = []
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # Strip common bullet markers (-, *, •, numbered)
            stripped = re.sub(r"^[-*•]\s+", "", stripped)
            stripped = re.sub(r"^\d+[.)]\s+", "", stripped)
            if stripped and stripped != "None needed":
                bullets.append(stripped)
        return bullets

    @staticmethod
    def _fuzzy_apply(content: str, search: str, replace: str) -> tuple[str, bool]:
        """Apply a SEARCH/REPLACE patch with 3-tier matching.

        Tier 1: Exact substring match
        Tier 2: Trimmed-line matching (ignores indentation differences),
                rebases replacement to the target's indentation
        Tier 3: Whitespace-normalized matching (collapses all runs of
                whitespace to single spaces for comparison)

        Returns (new_content, applied).
        """
        # Tier 1: exact match
        if search in content:
            return content.replace(search, replace, 1), True

        s_lines = search.split("\n")
        c_lines = content.split("\n")
        s_trimmed = [l.strip() for l in s_lines]

        # Tier 2: trimmed-line matching with indent rebase
        if len(s_lines) <= len(c_lines):
            for i in range(len(c_lines) - len(s_lines) + 1):
                if all(s_trimmed[j] == c_lines[i + j].strip() for j in range(len(s_lines))):
                    # Matched — rebase replacement indentation to target's level
                    base_indent = ""
                    stripped = c_lines[i].lstrip()
                    if stripped:
                        base_indent = c_lines[i][: len(c_lines[i]) - len(stripped)]
                    r_lines = replace.split("\n")
                    rebased = []
                    for k, rl in enumerate(r_lines):
                        if not rl.strip():
                            rebased.append(rl)
                        else:
                            rebased.append(base_indent + rl.lstrip())
                    new_lines = c_lines[:i] + rebased + c_lines[i + len(s_lines):]
                    return "\n".join(new_lines), True

        # Tier 3: whitespace-normalized (collapse all whitespace runs)

        s_norm = re.sub(r"\s+", " ", search.strip())
        for i in range(len(c_lines)):
            for span in range(1, min(len(s_lines) + 3, len(c_lines) - i + 1)):
                chunk = "\n".join(c_lines[i:i + span])
                c_norm = re.sub(r"\s+", " ", chunk.strip())
                if s_norm == c_norm:
                    base_indent = ""
                    stripped = c_lines[i].lstrip()
                    if stripped:
                        base_indent = c_lines[i][: len(c_lines[i]) - len(stripped)]
                    r_lines = replace.split("\n")
                    rebased = [base_indent + rl.lstrip() if rl.strip() else rl for rl in r_lines]
                    new_lines = c_lines[:i] + rebased + c_lines[i + span:]
                    return "\n".join(new_lines), True

        return content, False

    def _apply_file_patches(self, files: list, response: str) -> int:
        """Apply SEARCH/REPLACE patches from LLM response to project files."""
        applied = 0
        section_re = re.compile(r"===\s*FILE:\s*(.+?)\s*===\s*\n([\s\S]*?)(?=\n===\s*FILE:|$)", re.IGNORECASE)
        sections = list(section_re.finditer(response))

        for m in sections:
            filename = self._normalize_patch_filename(m.group(1))
            section = m.group(2).strip()
            target = self._find_patch_target(files, filename)
            if not target:
                continue

            sr_re = re.compile(r"<<<<<<<?\.?\s*SEARCH\n([\s\S]*?)\n?={3,}\n([\s\S]*?)\n?>>>>>>>?\.?\s*REPLACE", re.IGNORECASE)
            content = target["content"]
            for patch in sr_re.finditer(section):
                search = patch.group(1).rstrip()
                replace = patch.group(2).rstrip()
                content, ok = self._fuzzy_apply(content, search, replace)
                if ok:
                    applied += 1
            target["content"] = content

        # Also try without FILE wrappers (simpler models)
        if applied == 0 and not sections:
            sr_re = re.compile(r"<<<<<<<?\.?\s*SEARCH\n([\s\S]*?)\n?={3,}\n([\s\S]*?)\n?>>>>>>>?\.?\s*REPLACE", re.IGNORECASE)
            for patch in sr_re.finditer(response):
                search = patch.group(1).rstrip()
                replace = patch.group(2).rstrip()
                candidates = []
                for f in files:
                    new_content, ok = self._fuzzy_apply(f["content"], search, replace)
                    if ok:
                        candidates.append((f, new_content))
                if len(candidates) == 1:
                    f, new_content = candidates[0]
                    f["content"] = new_content
                    applied += 1
                elif len(candidates) > 1:
                    log.warning("apply_patches.ambiguous_bare_patch", matches=len(candidates))

        # Fallback: model returned full file content instead of patches.
        # Common with smaller models — they regenerate rather than patch.
        # Detect: FILE section exists but has no SEARCH/REPLACE blocks AND
        # contains code-like content (not just descriptions).
        if applied == 0:
            for m in sections:
                raw_filename = m.group(1).strip()
                section = m.group(2).strip()
                filename = self._normalize_patch_filename(raw_filename)
                target = self._find_patch_target(files, filename)
                if not target:
                    continue
                # Check if section looks like complete file content (not instructions)
                if not sr_re.search(section) and len(section) > 50:
                    # Strip any fenced code block wrapper
                    code = section
                    if code.startswith("```"):
                        code = re.sub(r"^```\S*\n?", "", code)
                        code = re.sub(r"\n?```\s*$", "", code)
                    code = re.sub(r"\n?__PASS_COMPLETE__\s*$", "", code).strip()
                    # Validate it's actual code
                    if self._looks_like_code(code, target.get("role", "script")):
                        target["content"] = code.strip()
                        applied += 1
                        log.info("apply_patches.full_file_replacement",
                                 file=filename, lines=code.count("\n") + 1)

        return applied

    def _assemble(self, files: list) -> str:
        """Assemble project files into a single runnable HTML page."""
        # Defensive: some artifact bundles (older saves, partial/failed builds,
        # externally-imported apps) carry file entries with no "role" key.
        # Indexing f["role"] below then raised KeyError, 500-ing the whole
        # /capture-preview. Backfill the role from the file extension so
        # assembly is robust to that schema drift instead of crashing.
        for f in files:
            if isinstance(f, dict) and not f.get("role"):
                f["role"] = _infer_file_role(f.get("path", ""))
        entry = next((f for f in files if f["role"] == "entry"), None)
        if not entry:
            # No entry file — wrap everything
            styles = "\n".join(f["content"] for f in files if f["role"] == "style")
            scripts = "\n".join(f["content"] for f in files if f["role"] == "script")
            return f"<!DOCTYPE html><html><head><style>{styles}</style></head><body><script>{scripts}</script></body></html>"

        html = entry["content"]

        # Strip ALL local file references (they'll 404 in assembled single-page mode)
        # Keep CDN URLs (http/https/protocol-relative), strip everything else
        _CDN = r'(?:https?://|//)'
        html = re.sub(
            r'<script[^>]*src=["\'](?!' + _CDN + r')[^"\']+\.js["\'][^>]*>\s*</script>(\s*\n)?',
            '', html, flags=re.IGNORECASE,
        )
        html = re.sub(
            r'<link[^>]*href=["\'](?!' + _CDN + r')[^"\']+\.css["\'][^>]*/?>(\s*\n)?',
            '', html, flags=re.IGNORECASE,
        )

        # Inject CSS
        styles = [f for f in files if f["role"] == "style"]
        if styles:
            style_block = "\n".join(f"/* {f['path']} */\n{f['content']}" for f in styles)
            if "</head>" in html:
                html = html.replace("</head>", f"<style>\n{style_block}\n</style>\n</head>")
            else:
                html = f"<style>\n{style_block}\n</style>\n" + html

        # Inject JS
        scripts = [f for f in files if f["role"] == "script"]
        modules = [f for f in files if f["role"] == "module"]
        if scripts or modules:
            script_block = ""
            for f in scripts:
                safe = re.sub(r'</script\s*>', '<\\/script>', f["content"], flags=re.IGNORECASE)
                script_block += f"<script>\n/* {f['path']} */\n{safe}\n</script>\n"
            for f in modules:
                safe = re.sub(r'</script\s*>', '<\\/script>', f["content"], flags=re.IGNORECASE)
                script_block += f'<script type="module">\n/* {f["path"]} */\n{safe}\n</script>\n'
            if "</body>" in html:
                html = html.replace("</body>", f"{script_block}</body>")
            else:
                html += f"\n{script_block}"

        # Inject data files (validated as JSON to prevent script injection)
        data_files = [f for f in files if f["role"] == "data"]
        if data_files:
            data_block = "<script>\n"
            for f in data_files:
                var_name = re.sub(r"[^a-zA-Z0-9]", "_", f["path"].rsplit(".", 1)[0])
                # Validate content is safe JSON before injecting into script context
                try:
                    json.loads(f["content"])
                    data_block += f"const {var_name} = {f['content']};\n"
                except (json.JSONDecodeError, ValueError):
                    # Not valid JSON — wrap in JSON.parse with escaped string to prevent injection
                    safe = f["content"].replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n")
                    safe = re.sub(r'</script\s*>', '<\\/script>', safe, flags=re.IGNORECASE)
                    data_block += f"const {var_name} = JSON.parse('{safe}');\n"
            data_block += "</script>\n"
            if "</head>" in html:
                html = html.replace("</head>", f"{data_block}</head>")
            else:
                html = data_block + html

        return html

    def _lint_html(self, content: str) -> list:
        issues = []
        opens = len(re.findall(r"<(?!br|hr|img|input|meta|link|!)[a-z][\w-]*(?:\s[^>]*)?>", content, re.I))
        closes = len(re.findall(r"</[a-z][\w-]*>", content, re.I))
        if opens > closes + 2:
            issues.append(f"Possibly {opens - closes} unclosed HTML tags")
        return issues

    def _lint_js(self, content: str) -> list:
        issues = []
        opens = content.count("{")
        closes = content.count("}")
        if opens != closes:
            issues.append(f"Mismatched braces: {opens} open, {closes} close")
        return issues

    def _infer_lang(self, path: str) -> str:
        ext_map = {"html": "html", "htm": "html", "css": "css", "js": "javascript",
                    "json": "json", "md": "markdown", "svg": "svg", "ts": "typescript"}
        ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        return ext_map.get(ext, ext or "text")

    def _infer_role(self, path: str) -> str:
        return _infer_file_role(path)
